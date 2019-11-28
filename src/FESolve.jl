module FESolve

export solvePoissonProblem!,computeBestApproximation!,computeFEInterpolation!,eval_interpolation_error!,eval_interpolation_error2!

using SparseArrays
using LinearAlgebra
using BenchmarkTools
using FiniteElements

using Grid
using Quadrature

function accumarray!(A,subs, val, sz=(maximum(subs),))
      for i = 1:length(val)
          @inbounds A[subs[i]] += val[i]
      end
  end

# matrix for L2 bestapproximation
function global_mass_matrix!(aa,ii,jj,grid::Grid.Mesh)
    ncells::Int = size(grid.nodes4cells,1);
    celldim::Int = size(grid.nodes4cells,2);
    
    # local mass matrix (the same on every triangle)
    local_mass_matrix = (ones(Int64,celldim,celldim) + LinearAlgebra.I(celldim)) * 1 // ((celldim)*(celldim+1));
    
    # do the 'integration'
    index = 0;
    for i = 1:celldim, j = 1:celldim
        for cell = 1 : ncells
            @inbounds begin
                ii[index+cell] = grid.nodes4cells[cell,i];
                jj[index+cell] = grid.nodes4cells[cell,j];
                aa[index+cell] = local_mass_matrix[i,j] * grid.volume4cells[cell];
            end
        end    
        index += ncells;
    end
end


# matrix for L2 bestapproximation
function global_mass_matrix4FE!(aa,ii,jj,grid::Grid.Mesh,FE::FiniteElements.FiniteElement)
    ncells::Int = size(grid.nodes4cells,1);
    ndofs4cell::Int = size(FE.dofs4cells,2);
    dim::Int = size(grid.nodes4cells,2);
    
    # local mass matrix (the same on every triangle)
    @assert length(aa) == ncells*ndofs4cell^2;
    @assert length(ii) == ncells*ndofs4cell^2;
    @assert length(jj) == ncells*ndofs4cell^2;
    
    index = 0;
    for i = 1:dim, j = 1:dim
        for cell = 1 : ncells
            @inbounds begin
                ii[index+cell] = grid.nodes4cells[cell,i];
                jj[index+cell] = grid.nodes4cells[cell,j];
                aa[index+cell] = FE.local_mass_matrix[i,j] * grid.volume4cells[cell];
            end
        end    
        index += ncells;
    end
end

# old version
function global_mass_matrix_old!(aa,ii,jj,grid::Grid.Mesh)
    ncells::Int = size(grid.nodes4cells,1);
    ii[:] = repeat(grid.nodes4cells',3)[:];
    jj[:] = repeat(grid.nodes4cells'[:]',3)[:];
    aa[:] = repeat([2 1 1 1 2 1 1 1 2]'[:] * 1 // 12,ncells)[:].*repeat(grid.volume4cells',9)[:];
end


# matrix for H1 bestapproximation and gradients on each cell
# version inspired by Matlab-AFEM group of C. Carstensen
# based on the formula
#
# gradients = [1 1 1; coords]^{-1} [0 0; 1 0; 0 1];
#
function global_stiffness_matrix_with_gradients!(aa,ii,jj,gradients4cells,grid::Grid.Mesh)
    ncells::Int = size(grid.nodes4cells,1);
    dim::Int = size(grid.nodes4cells,2)-1;
    
    # compute local stiffness matrices
    Aloc = zeros(typeof(grid.coords4nodes[1]),dim+1,dim+1,ncells);
    for cell = 1 : ncells
        if dim == 1
            @views gradients4cells[:,:,cell] = [1,-1]' / grid.volume4cells[cell]
        elseif dim == 2
            @views gradients4cells[:,:,cell] = [1 1 1; grid.coords4nodes[grid.nodes4cells[cell,:],:]'] \ [0 0; 1 0;0 1];
        end    
        @views Aloc[:,:,cell] = grid.volume4cells[cell] .* (gradients4cells[:,:,cell] * gradients4cells[:,:,cell]');
    end
    
    ii[:] = repeat(grid.nodes4cells',dim+1)[:];
    jj[:] = repeat(grid.nodes4cells'[:]',dim+1)[:];
    aa[:] = Aloc[:];
end

function global_stiffness_matrix4FE!(aa,ii,jj,gradients4cells,grid,FE::FiniteElements.FiniteElement)
    ncells::Int = size(grid.nodes4cells,1);
    ndofs4cell::Int = size(FE.dofs4cells,2);
    xdim::Int = size(grid.coords4nodes,2);
    celldim::Int = size(grid.nodes4cells,2);
    midpoint = zeros(eltype(grid.coords4nodes),xdim);
    
    
    @assert length(aa) == ncells*ndofs4cell^2;
    @assert length(ii) == ncells*ndofs4cell^2;
    @assert length(jj) == ncells*ndofs4cell^2;
    
    FE2 = FiniteElements.get_P1FiniteElementFD(grid);
    
    # compute local stiffness matrices
    index::Int = 0;
    curindex::Int = 0;
    for cell = 1 : ncells
        # compute cell midpoint
        fill!(midpoint,0);
        for j = 1 : xdim
            for i = 1 : celldim
                midpoint[j] += grid.coords4nodes[grid.nodes4cells[cell,i],j]
            end
            midpoint[j] /= celldim
        end
        
        # evaluate gradients
        for i = 1 : ndofs4cell
            FE.bfun_grad![i](view(gradients4cells,i,:,cell),midpoint,grid,cell);
        end    
        
        # fill fields aa,ii,jj
        for i = 1 : ndofs4cell, j = 1 : ndofs4cell
            curindex = index+(i-1)*ndofs4cell+j;
            aa[curindex] = grid.volume4cells[cell] * dot(gradients4cells[i,:,cell],gradients4cells[j,:,cell]);
            ii[curindex] = FE.dofs4cells[cell,i];
            jj[curindex] = FE.dofs4cells[cell,j];
        end
        index += ndofs4cell^2;
    end
end

#
# matrix for H1 bestapproximation
# this version is inspired by Julia iFEM (for dim=2)
# (http://www.stochasticlifestyle.com/julia-ifem2)
#
# Explanations:
# it uses that the gradient of a nodal basis functions
# is constant and equal to the normal vector / height
# of the opposite edge, this leads to
#
# int_T grad_j grad_k = |T| dot(n_j/h_j,n_k/h_k) = |T| dot(t_j/h_j,t_k/h_k)
#
# where t are the tangents (rotations do not change the integral)
# moreover, the t_j/h_k can be expressed as differences of coordinates d_j = x_j+1 - x_j-1
# leading to t_j = d_j/|E_j| which togehter with |E_j| h_j = 2 |T| leads to the simple formula
#
# int_T grad_j grad_k = |T| dot(n_j/h_j,n_k/h_k) = dot(d_j,df_k)/(4|T|)
#
function global_stiffness_matrix!(aa,ii,jj,grid::Grid.Mesh)
    ncells::Int = size(grid.nodes4cells,1);
    dim::Int = size(grid.nodes4cells,2)-1;
    
    if dim == 1
        local_matrix = -ones(Int64,dim+1,dim+1) + 2 // 1 * LinearAlgebra.I(dim+1);
        
        # do the 'integration'
        index = 0;
        for i = 1:dim+1, j = 1:dim+1
            @inbounds begin
                ii[index+1:index+ncells] = view(grid.nodes4cells,:,i);
                jj[index+1:index+ncells] = view(grid.nodes4cells,:,j);
                aa[index+1:index+ncells] = local_matrix[i,j] / grid.volume4cells;
            end    
            index += ncells;
        end
    elseif dim == 2
        ve = Array{typeof(grid.coords4nodes[1])}(undef, ncells,2,3);
        # compute coordinate differences (= weighted tangents)
        @views ve[:,:,3] = grid.coords4nodes[vec(grid.nodes4cells[:,2]),:]-grid.coords4nodes[vec(grid.nodes4cells[:,1]),:];
        @views ve[:,:,1] = grid.coords4nodes[vec(grid.nodes4cells[:,3]),:]-grid.coords4nodes[vec(grid.nodes4cells[:,2]),:];
        @views ve[:,:,2] = grid.coords4nodes[vec(grid.nodes4cells[:,1]),:]-grid.coords4nodes[vec(grid.nodes4cells[:,3]),:];
    
        # do the 'integration'
        index = 0;
        for i = 1:3, j = 1:3
            @inbounds begin
                ii[index+1:index+ncells] = view(grid.nodes4cells,:,i);
                jj[index+1:index+ncells] = view(grid.nodes4cells,:,j);
                aa[index+1:index+ncells] = sum(ve[:,:,i].* ve[:,:,j], dims=2) ./ (4 * grid.volume4cells);
            end    
            index += ncells;
        end
    end    
end



# scalar functions times P1 basis functions
function rhs_integrandL2!(result,x,xref,cellIndex::Int,f!::Function)
    f!(view(result, 1), x);
    for j=length(xref):-1:1
        result[j] = view(result,1) .* xref[j];
    end
end


function assembleSystem(norm_lhs::String,norm_rhs::String,volume_data!::Function,grid::Grid.Mesh,FE::FiniteElements.FiniteElement,quadrature_order::Int)

    ncells::Int = size(grid.nodes4cells,1);
    nnodes::Int = size(grid.coords4nodes,1);
    dim::Int = size(grid.nodes4cells,2)-1;
    
    Grid.ensure_volume4cells!(grid);
    
    
    aa = Vector{typeof(grid.coords4nodes[1])}(undef, (dim+1)^2*ncells);
    ii = Vector{Int64}(undef, (dim+1)^2*ncells);
    jj = Vector{Int64}(undef, (dim+1)^2*ncells);
    
    if norm_lhs == "L2"
        println("mass matrix")
        @time A = global_mass_matrix4FE!(aa,ii,jj,grid,FE);
    elseif norm_lhs == "H1"
        println("stiffness matrix")
        gradients4cells = zeros(typeof(grid.coords4nodes[1]),dim+1,dim,ncells);
        global_stiffness_matrix4FE!(aa,ii,jj,gradients4cells,grid,FE);
    end 
    
    A = sparse(ii,jj,aa,nnodes,nnodes);
    
    # compute right-hand side vector
    rhsintegral4cells = zeros(Base.eltype(grid.coords4nodes),ncells,dim+1); # f x P1basis (dim+1 many)
    if norm_rhs == "L2"
        println("integrate rhs");
        wrapped_integrand_L2!(result,x,xref,cellIndex) = rhs_integrandL2!(result,x,xref,cellIndex,volume_data!);
        @time integrate!(rhsintegral4cells,wrapped_integrand_L2!,grid,quadrature_order,dim+1);
    elseif norm_rhs == "H1"
        @assert norm_lhs == "H1"
        # compute cell-wise integrals for right-hand side vector (f expected to be dim-dimensional)
        println("integrate rhs");
        fintegral4cells = zeros(eltype(grid.coords4nodes),ncells,dim);
        wrapped_integrand_f!(result,x,xref,cellIndex) = volume_data!(result,x);
        @time integrate!(fintegral4cells,wrapped_integrand_f!,grid,quadrature_order,dim);
        
        # multiply with gradients
        for j = 1 : dim + 1
            for k = 1 : dim
                rhsintegral4cells[:,j] += (fintegral4cells[:,k] .* gradients4cells[j,k,:]);
            end
        end                            
    end
    
    # accumulate right-hand side vector
    println("accumarray");
    b = zeros(eltype(grid.coords4nodes),nnodes);
    @time accumarray!(b,grid.nodes4cells,rhsintegral4cells,nnodes)
    
    return A,b
end

# computes Bestapproximation in norm="L2" or "H1"
# volume_data! for norm="H1" is expected to be the gradient of the function that is bestapproximated
function computeBestApproximation!(val4coords::Array,norm::String ,volume_data!::Function,boundary_data!,grid::Grid.Mesh,FE::FiniteElements.FiniteElement,quadrature_order::Int)
    # assemble system 
    A, b = assembleSystem(norm,norm,volume_data!,grid,FE,quadrature_order);
    
    # find boundary dofs
    if boundary_data! == Nothing
        bdofs = [];
    else
        Grid.ensure_bfaces!(grid);
        bdofs = unique(FE.dofs4faces[grid.bfaces,:]);
    end    
    dofs = setdiff(unique(FE.dofs4cells[:]),bdofs);
    
    # solve
    println("solve");
    fill!(val4coords,0)
    if length(bdofs) > 0
        boundary_data!(view(val4coords,bdofs),view(FE.coords4dofs,bdofs,:),0);
        b = b - A*val4coords;
    end    
    
    try
        @time val4coords[dofs] = A[dofs,dofs]\b[dofs];
    catch    
        println("Unsupported Number type for sparse lu detected: trying again with dense matrix");
        try
            @time val4coords[dofs] = Array{typeof(grid.coords4nodes[1]),2}(A[dofs,dofs])\b[dofs];
        catch OverflowError
            println("OverflowError (Rationals?): trying again as Float64 sparse matrix");
            @time val4coords[dofs] = Array{Float64,2}(A[dofs,dofs])\b[dofs];
        end
    end
end

# computes solution of Poisson problem
function solvePoissonProblem!(val4coords::Array,volume_data!::Function,boundary_data!,grid::Grid.Mesh,FE::FiniteElements.FiniteElement,quadrature_order::Int)
    # assemble system 
    A, b = assembleSystem("H1","L2",volume_data!,grid,FE,quadrature_order);
    
    # find boundary dofs
    if boundary_data! == Nothing
        bdofs = [];
    else
        Grid.ensure_bfaces!(grid);
        bdofs = unique(FE.dofs4faces[grid.bfaces,:]);
    end    
    dofs = setdiff(unique(FE.dofs4cells[:]),bdofs);
    
    # solve
    println("solve");
    fill!(val4coords,0)
    if length(bdofs) > 0
        boundary_data!(view(val4coords,bdofs),view(FE.coords4dofs,bdofs,:),0);
        b = b - A*val4coords;
    end    
    
    try
        @time val4coords[dofs] = A[dofs,dofs]\b[dofs];
    catch    
        println("Unsupported Number type for sparse lu detected: trying again with dense matrix");
        try
            @time val4coords[dofs] = Array{typeof(grid.coords4nodes[1]),2}(A[dofs,dofs])\b[dofs];
        catch OverflowError
            println("OverflowError (Rationals?): trying again as Float64 sparse matrix");
            @time val4coords[dofs] = Array{Float64,2}(A[dofs,dofs])\b[dofs];
        end
    end
end



function computeFEInterpolation!(val4dofs::Array,source_function!::Function,grid::Grid.Mesh,FE::FiniteElements.FiniteElement)
    source_function!(val4dofs,FE.coords4dofs);
end


function eval_interpolation_error!(result, x, xref, exact_function!, coeffs_interpolation, dofs_interpolation)
    # evaluate exact function
    exact_function!(result, x);
    # subtract nodal interpolation
    for cellIndex =1 : size(dofs_interpolation, 1)
        @inbounds result[1] -= sum(coeffs_interpolation[dofs_interpolation[cellIndex, :]] .* xref)
    end
end

function eval_interpolation_error2!(result, x, xref, cellIndex, exact_function!, coeffs_interpolation, dofs_interpolation)
    # evaluate exact function
    exact_function!(result, x);
    # subtract nodal interpolation
    @inbounds result[1] -= sum(coeffs_interpolation[dofs_interpolation[cellIndex, :]] .* xref)
end

end
