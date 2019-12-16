module FESolveCommon

export accumarray, computeBestApproximation!, computeFEInterpolation!, eval_interpolation_error!, eval_L2_interpolation_error!

using SparseArrays
using LinearAlgebra
using BenchmarkTools
using FiniteElements
using Grid
using Quadrature

function accumarray!(A,subs, val, sz=(maximum(subs),))
    for i = 1:length(val)
        A[subs[i]] += val[i]
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
    
    # local mass matrix (the same on every triangle)
    @assert length(aa) == ncells*ndofs4cell^2;
    @assert length(ii) == ncells*ndofs4cell^2;
    @assert length(jj) == ncells*ndofs4cell^2;

    if size(FE.local_mass_matrix,2) == ndofs4cell # if local mass matrix is the same on all cells and defined in FE
        index = 0;
        for i = 1:ndofs4cell, j = 1:ndofs4cell
            for cell = 1 : ncells
                @inbounds begin
                    ii[index+cell] = FE.dofs4cells[cell,i];
                    jj[index+cell] = FE.dofs4cells[cell,j];
                    aa[index+cell] = FE.local_mass_matrix[i,j] * grid.volume4cells[cell];
                end
            end    
            index += ncells;
        end
    else # do it the hard way
        T = eltype(grid.coords4nodes);
        qf = QuadratureFormula{T}(2*(FE.polynomial_order), FE.ncomponents);
        
        # pre-allocate memory for gradients
        evals4cell = Array{Array{T,1}}(undef,ndofs4cell);
        for j = 1: ndofs4cell
            evals4cell[j] = zeros(T,FE.ncomponents);
        end
    
        fill!(aa,0.0);
        curindex::Int = 0;
        x = zeros(T,FE.ncomponents);
        celldim::Int = size(grid.nodes4cells,2);
    
        # quadrature loop
        @time for i in eachindex(qf.w)
            curindex = 0
            for cell = 1 : ncells
                fill!(x, 0)
                for j = 1 : FE.ncomponents
                    for k = 1 : celldim
                        x[j] += grid.coords4nodes[grid.nodes4cells[cell, k], j] * qf.xref[i][k]
                    end
                end
                # evaluate basis functions at quadrature point
                for dof_i = 1 : ndofs4cell
                    evals4cell[dof_i] = FE.bfun[dof_i](x,grid,cell);
                end    
        
                # fill fields aa,ii,jj
                for dof_i = 1 : ndofs4cell, dof_j = dof_i : ndofs4cell
                    curindex += 1;
                    # fill upper right part and diagonal of matrix
                    @inbounds begin
                    for k = 1 : FE.ncomponents
                        aa[curindex] += (evals4cell[dof_i][k]*evals4cell[dof_j][k] * qf.w[i] * grid.volume4cells[cell]);
                    end
                    if (i == 1)
                        ii[curindex] = FE.dofs4cells[cell,dof_i];
                        jj[curindex] = FE.dofs4cells[cell,dof_j];
                    end    
                    # fill lower left part of matrix
                    if dof_j > dof_i
                        curindex += 1;
                        if (i == length(qf.w))
                            aa[curindex] = aa[curindex-1];
                            ii[curindex] = FE.dofs4cells[cell,dof_j];
                            jj[curindex] = FE.dofs4cells[cell,dof_i];
                        end    
                    end    
                    end
                end
            end
        end
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

function global_stiffness_matrix4FE!(aa,ii,jj,grid,FE::FiniteElements.FiniteElement)
    ncells::Int = size(grid.nodes4cells,1);
    ndofs4cell::Int = size(FE.dofs4cells,2);
    xdim::Int = size(grid.coords4nodes,2);
    celldim::Int = size(grid.nodes4cells,2);
    
    
    @assert length(aa) == ncells*ndofs4cell^2;
    @assert length(ii) == length(aa);
    @assert length(jj) == length(aa);
    
    T = eltype(grid.coords4nodes);
    qf = QuadratureFormula{T}(2*(FE.polynomial_order-1), xdim);
    
    fill!(aa,0.0);
    # compute local stiffness matrices
    curindex::Int = 0;
    x = zeros(T,xdim);
    
    # pre-allocate memory for gradients
    gradients4cell = Array{Array{T,1}}(undef,ndofs4cell);
    for j = 1: ndofs4cell
        gradients4cell[j] = zeros(T,xdim);
    end
    
    # quadrature loop
    @time for i in eachindex(qf.w)
      curindex = 0
      for cell = 1 : ncells
        # compute global quadrature point in cell
        fill!(x, 0)
        for j = 1 : xdim
          for k = 1 : celldim
            x[j] += grid.coords4nodes[grid.nodes4cells[cell, k], j] * qf.xref[i][k]
          end
        end
        
        # evaluate gradients at quadrature point
        for dof_i = 1 : ndofs4cell
           FE.bfun_grad![dof_i](gradients4cell[dof_i],x,qf.xref[i],grid,cell);
        end    
        
        # fill fields aa,ii,jj
        for dof_i = 1 : ndofs4cell, dof_j = dof_i : ndofs4cell
            curindex += 1;
            # fill upper right part and diagonal of matrix
            @inbounds begin
            for k = 1 : xdim
                aa[curindex] += (gradients4cell[dof_i][k]*gradients4cell[dof_j][k] * qf.w[i] * grid.volume4cells[cell]);
            end
            if (i == 1)
                ii[curindex] = FE.dofs4cells[cell,dof_i];
                jj[curindex] = FE.dofs4cells[cell,dof_j];
            end    
            # fill lower left part of matrix
            if dof_j > dof_i
                curindex += 1;
                if (i == length(qf.w))
                    aa[curindex] = aa[curindex-1];
                    ii[curindex] = FE.dofs4cells[cell,dof_j];
                    jj[curindex] = FE.dofs4cells[cell,dof_i];
                end    
            end    
            end
        end
      end  
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
function rhs_integrandL2!(f!::Function,FE::FiniteElements.FiniteElement,dim)
    cache = zeros(eltype(FE.grid.coords4nodes),dim)
    basisval = zeros(eltype(FE.grid.coords4nodes),dim)
    ndofcell::Int = size(FE.dofs4cells,2)
    function closure(result,x,xref,cellIndex::Int)
        f!(cache, x);
        for j=1:ndofcell
            basisval = FE.bfun_ref[j](xref,FE.grid,cellIndex);
            result[j] = 0.0;
            for d=1:dim
                result[j] += cache[d] * basisval[d];
            end    
        end
    end
end


function assembleSystem(norm_lhs::String,norm_rhs::String,volume_data!::Function,grid::Grid.Mesh,FE::FiniteElements.FiniteElement,quadrature_order::Int)

    ncells::Int = size(grid.nodes4cells,1);
    ndofscell::Int = size(FE.dofs4cells,2);
    nnodes::Int = size(grid.coords4nodes,1);
    ndofs::Int = size(FE.coords4dofs,1);
    celldim::Int = size(grid.nodes4cells,2);
    xdim::Int = size(grid.coords4nodes,2);
    
    Grid.ensure_volume4cells!(grid);
    
    
    aa = Vector{typeof(grid.coords4nodes[1])}(undef, ndofscell^2*ncells);
    ii = Vector{Int64}(undef, ndofscell^2*ncells);
    jj = Vector{Int64}(undef, ndofscell^2*ncells);
    
    if norm_lhs == "L2"
        println("mass matrix")
        @time A = global_mass_matrix4FE!(aa,ii,jj,grid,FE);
    elseif norm_lhs == "H1"
        println("stiffness matrix")
        @time global_stiffness_matrix4FE!(aa,ii,jj,grid,FE);
    end 
    A = sparse(ii,jj,aa,ndofs,ndofs);
    
    # compute right-hand side vector
    rhsintegral4cells = zeros(Base.eltype(grid.coords4nodes),ncells,ndofscell); # f x FEbasis
    if norm_rhs == "L2"
        println("integrate rhs");
        @time integrate!(rhsintegral4cells,rhs_integrandL2!(volume_data!,FE,FE.ncomponents),grid,quadrature_order,ndofscell);
    elseif norm_rhs == "H1"
        @assert norm_lhs == "H1"
        # compute cell-wise integrals for right-hand side vector (f expected to be dim-dimensional)
        println("integrate rhs");
        fintegral4cells = zeros(eltype(grid.coords4nodes),ncells,xdim);
        wrapped_integrand_f!(result,x,xref,cellIndex) = volume_data!(result,x);
        @time integrate!(fintegral4cells,wrapped_integrand_f!,grid,quadrature_order,xdim);
        
        # multiply with gradients
        gradient4cell = zeros(eltype(grid.coords4nodes),xdim);
        midpoint = zeros(eltype(grid.coords4nodes),xdim);
        for cell = 1 : ncells
            fill!(midpoint,0);
            for j = 1 : xdim
                for i = 1 : celldim
                    midpoint[j] += grid.coords4nodes[grid.nodes4cells[cell,i],j]
                end
                midpoint[j] /= celldim
            end
            for j = 1 : ndofscell
                FE.bfun_grad![j](gradient4cell,midpoint,[1//3 1//3 1//3],grid,cell);
                rhsintegral4cells[cell,j] += dot(fintegral4cells[cell,:], gradient4cell);
            end                  
        end
    end
    
    # accumulate right-hand side vector
    println("accumarray");
    b = zeros(eltype(grid.coords4nodes),ndofs);
    @time accumarray!(b,FE.dofs4cells,rhsintegral4cells)
    
    return A,b
end

# computes Bestapproximation in approx_norm="L2" or "H1"
# volume_data! for norm="H1" is expected to be the gradient of the function that is bestapproximated
function computeBestApproximation!(val4coords::Array,approx_norm::String ,volume_data!::Function,boundary_data!,grid::Grid.Mesh,FE::FiniteElements.FiniteElement,quadrature_order::Int, dirichlet_penalty = 1e60)
    # assemble system 
    A, b = assembleSystem(approx_norm,approx_norm,volume_data!,grid,FE,quadrature_order);
    
    # find boundary dofs
    xdim = FE.ncomponents;
    ndofs::Int = size(FE.coords4dofs,1);
    dofs = 1:ndofs;
    
    bdofs = [];
    if ((boundary_data! == Nothing) || (size(FE.dofs4faces,1) == 0))
    else
        Grid.ensure_bfaces!(grid);
        Grid.ensure_cells4faces!(grid);
        temp = zeros(eltype(grid.coords4nodes),xdim);
        cell::Int = 0;
        j::Int = 1;
        ndofs4bfaces = size(FE.dofs4faces,2);
        A4bface = Matrix{Float64}(undef,ndofs4bfaces,ndofs4bfaces)
        b4bface = Vector{Float64}(undef,ndofs4bfaces)
        bdofs4bface = Vector{Int}(undef,ndofs4bfaces)
        celldof2facedof = zeros(Int,ndofs4bfaces)
        for i in eachindex(grid.bfaces)
            cell = grid.cells4faces[grid.bfaces[i],1];
            # setup local system of equations to determine piecewise interpolation of boundary data
            bdofs4bface = FE.dofs4faces[grid.bfaces[i],:]
            append!(bdofs,bdofs4bface);
            # find position of face dofs in cell dofs
            for j=1:size(FE.dofs4cells,2), k = 1 : ndofs4bfaces
                if FE.dofs4cells[cell,j] == bdofs4bface[k]
                    celldof2facedof[k] = j;
                end    
            end
            # assemble matrix    
            for k = 1:ndofs4bfaces
                for l = 1:ndofs4bfaces
                    A4bface[k,l] = dot(FE.bfun[celldof2facedof[k]](view(FE.coords4dofs,bdofs4bface[k],:),grid,cell),FE.bfun[celldof2facedof[l]](view(FE.coords4dofs,bdofs4bface[k],:),grid,cell));
                end
                boundary_data!(temp,view(FE.coords4dofs,bdofs4bface[k],:));
                b4bface[k] = dot(temp,FE.bfun[celldof2facedof[k]](view(FE.coords4dofs,bdofs4bface[k],:),grid,cell));
            end
            val4coords[bdofs4bface] = A4bface\b4bface;
            if norm(A4bface*val4coords[bdofs4bface]-b4bface) > eps(1e4)
                println("WARNING: large residual, boundary data may be inexact");
            end
        end    
        # b = b - A*val4coords;
        unique!(bdofs)
        for i = 1 : length(bdofs)
            A[bdofs[i],bdofs[i]] = dirichlet_penalty;
            b[bdofs[i]] = val4coords[bdofs[i]]*dirichlet_penalty;
        end
    end   
    
    
    # solve
    println("solve");
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
    return norm(A[dofs,dofs]*val4coords[dofs] - b[dofs])
end


function computeFEInterpolation!(val4dofs::Array,source_function!::Function,grid::Grid.Mesh,FE::FiniteElements.FiniteElement)
    source_function!(val4dofs,FE.coords4dofs);
end


function eval_interpolation_error!(exact_function!, coeffs_interpolation, FE::FiniteElements.FiniteElement)
    temp = zeros(Float64,FE.ncomponents);
    function closure(result, x, xref, cellIndex)
        # evaluate exact function
        exact_function!(result, x);
        # subtract nodal interpolation
        ndofcell = size(FE.dofs4cells,2);
        for j = 1 : ndofcell
            temp = FE.bfun_ref[j](xref, FE.grid, cellIndex);
            temp *= coeffs_interpolation[FE.dofs4cells[cellIndex, j]];
            for k = 1 : length(temp);
                result[k] -= temp[k] 
            end    
        end    
    end
end


function eval_L2_interpolation_error!(exact_function!, coeffs_interpolation, FE::FiniteElements.FiniteElement)
    temp = zeros(Float64,FE.ncomponents);
    function closure(result, x, xref, cellIndex)
        # evaluate exact function
        exact_function!(result, x);
        # subtract nodal interpolation
        ndofcell = size(FE.dofs4cells,2);
        for j = 1 : ndofcell
            temp = FE.bfun_ref[j](xref, FE.grid, cellIndex);
            temp *= coeffs_interpolation[FE.dofs4cells[cellIndex, j]];
            for k = 1 : length(temp);
                result[k] -= temp[k] 
            end    
        end   
        # square for L2 norm
        for j = 1 : length(result)
            result[j] = result[j]^2
        end    
    end
end

end
