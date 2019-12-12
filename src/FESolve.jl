module FESolve

export solvePoissonProblem!, solveStokesProblem!, computeBestApproximation!, computeFEInterpolation!, eval_interpolation_error!, eval_interpolation_error2!

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

function StokesOperator4FE!(aa,ii,jj,grid,FE_velocity::FiniteElements.FiniteElement,FE_pressure::FiniteElements.FiniteElement)
    ncells::Int = size(grid.nodes4cells,1);
    xdim::Int = size(grid.coords4nodes,2);
    ndofs4cell_velocity::Int = size(FE_velocity.dofs4cells,2);
    ndofs4cell_pressure::Int = size(FE_pressure.dofs4cells,2);
    ndofs_velocity = size(FE_velocity.coords4dofs,1);
    ndofs4cell::Int = ndofs4cell_velocity+ndofs4cell_pressure;
    celldim::Int = size(grid.nodes4cells,2);
    
    @assert length(aa) == ncells*(ndofs4cell^2);
    @assert length(ii) == length(aa);
    @assert length(jj) == length(aa);
    
    # first assemble stiffness matrix for one velocity component
    
    
    T = eltype(grid.coords4nodes);
    quadorder = minimum([FE_pressure.polynomial_order + FE_velocity.polynomial_order-1, 2*FE_velocity.polynomial_order]);
    qf = QuadratureFormula{T}(quadorder, xdim);
    
    # compute local stiffness matrices
    curindex::Int = 0;
    x = zeros(T,xdim);
    
    # pre-allocate memory for gradients
    velogradients4cell = Array{Array{T,1}}(undef,ndofs4cell_velocity);
    pressure4cell = Array{T,1}(undef,ndofs4cell_pressure);
    for j = 1 : ndofs4cell_velocity
        velogradients4cell[j] = zeros(T,xdim*xdim);
    end
    for j = 1 : ndofs4cell_pressure
        pressure4cell[j] = 0.0;
    end
    
    # quadrature loop
    fill!(aa, 0.0);
    trace_indices = 1:(xdim+1):xdim^2
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
        for dof_i = 1 : ndofs4cell_velocity
            FE_velocity.bfun_grad![dof_i](velogradients4cell[dof_i],x,qf.xref[i],grid,cell);
        end    
        # evaluate pressures at quadrature point
        for dof_i = 1 : ndofs4cell_pressure
            pressure4cell[dof_i] = FE_pressure.bfun_ref[dof_i](qf.xref[i],grid,cell);
        end
        
        # fill fields aa,ii,jj
        for dof_i = 1 : ndofs4cell_velocity
            # stiffness matrix for velocity
            for dof_j = 1 : ndofs4cell_velocity
                curindex += 1;
                for k = 1 : xdim*xdim
                    aa[curindex] += (velogradients4cell[dof_i][k] * velogradients4cell[dof_j][k] * qf.w[i] * grid.volume4cells[cell]);
                end
                if (i == 1)
                    ii[curindex] = FE_velocity.dofs4cells[cell,dof_i];
                    jj[curindex] = FE_velocity.dofs4cells[cell,dof_j];
                end    
            end
        end    
        # divvelo x pressure matrix
        for dof_i = 1 : ndofs4cell_velocity
            for dof_j = 1 : ndofs4cell_pressure
                curindex += 1;
                for k = 1 : length(trace_indices)
                    aa[curindex] -= (velogradients4cell[dof_i][trace_indices[k]] * pressure4cell[dof_j] * qf.w[i] * grid.volume4cells[cell]);
                end
                if (i == 1)
                    ii[curindex] = FE_velocity.dofs4cells[cell,dof_i];
                    jj[curindex] = ndofs_velocity + FE_pressure.dofs4cells[cell,dof_j];
                end  
                #copy transpose
                curindex += 1;
                aa[curindex] = aa[curindex-1]
                if (i == 1)
                    ii[curindex] = ndofs_velocity + FE_pressure.dofs4cells[cell,dof_j];
                    jj[curindex] = FE_velocity.dofs4cells[cell,dof_i];
                end 
            end
        end  
        # pressure x pressure block (empty)
        for dof_i = 1 : ndofs4cell_pressure, dof_j = 1 : ndofs4cell_pressure
            curindex +=1
            aa[curindex] = 0;
            if (i == 1)
                ii[curindex] = ndofs_velocity + FE_pressure.dofs4cells[cell,dof_i];
                jj[curindex] = ndofs_velocity + FE_pressure.dofs4cells[cell,dof_j];
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
function rhs_integrandL2!(result,x,xref,cellIndex::Int,f!::Function,FE::FiniteElements.FiniteElement)
    f!(view(result, 1), x);
    ndofcell::Int = size(FE.dofs4cells,2);
    for j=ndofcell:-1:1
        result[j] = view(result,1) .* FE.bfun_ref[j](xref,FE.grid,cellIndex);
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
        wrapped_integrand_L2!(result,x,xref,cellIndex) = rhs_integrandL2!(result,x,xref,cellIndex,volume_data!,FE);
        @time integrate!(rhsintegral4cells,wrapped_integrand_L2!,grid,quadrature_order,ndofscell);
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
    dofs = setdiff(1:size(FE.coords4dofs,1),bdofs);
    # solve
    println("solve");
    fill!(val4coords,0)
    if length(bdofs) > 0
        boundary_data!(view(val4coords,bdofs),view(FE.coords4dofs,bdofs,:),0);
        b = b - A*val4coords;
    end    
    #show(Array{typeof(grid.coords4nodes[1]),2}(A))
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



# scalar functions times P1 basis functions
function rhs_integrand4Stokes!(f!::Function,FE::FiniteElements.FiniteElement,dim)
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

function assembleStokesSystem(volume_data!::Function,grid::Grid.Mesh,FE_velocity::FiniteElements.FiniteElement,FE_pressure::FiniteElements.FiniteElement,quadrature_order::Int)

    ncells::Int = size(grid.nodes4cells,1);
    nnodes::Int = size(grid.coords4nodes,1);
    celldim::Int = size(grid.nodes4cells,2);
    xdim::Int = size(grid.coords4nodes,2);
    ndofs_velocity::Int = size(FE_velocity.coords4dofs,1)
    ndofs::Int = ndofs_velocity + size(FE_pressure.coords4dofs,1);
    
    Grid.ensure_volume4cells!(grid);
    
    ndofs4cell_velocity = size(FE_velocity.dofs4cells,2)
    ndofs4cell = ndofs4cell_velocity + size(FE_pressure.dofs4cells,2)
    aa = Vector{typeof(grid.coords4nodes[1])}(undef, ndofs4cell^2*ncells);
    ii = Vector{Int64}(undef, ndofs4cell^2*ncells);
    jj = Vector{Int64}(undef, ndofs4cell^2*ncells);
    
    println("assembling Stokes matrix...")
    @time StokesOperator4FE!(aa,ii,jj,grid,FE_velocity,FE_pressure);
    A = sparse(ii,jj,aa,ndofs,ndofs);
    
    # compute right-hand side vector
    rhsintegral4cells = zeros(Base.eltype(grid.coords4nodes),ncells,ndofs4cell_velocity); # f x FEbasis
    println("integrate rhs");
    @time integrate!(rhsintegral4cells, rhs_integrand4Stokes!(volume_data!, FE_velocity,xdim), grid, quadrature_order, ndofs4cell_velocity);
         
    # accumulate right-hand side vector
    println("accumarray");
    b = zeros(eltype(grid.coords4nodes),ndofs);
    @time accumarray!(b,FE_velocity.dofs4cells,rhsintegral4cells);
    
    return A,b
end


function solveStokesProblem!(val4coords::Array,volume_data!::Function,boundary_data!,grid::Grid.Mesh,FE_velocity::FiniteElements.FiniteElement,FE_pressure::FiniteElements.FiniteElement,quadrature_order::Int)
    # assemble system 
    A, b = assembleStokesSystem(volume_data!,grid,FE_velocity,FE_pressure,quadrature_order);
    
    
    # find boundary dofs
    xdim = size(grid.coords4nodes,2);
    ndofs_velocity::Int = size(FE_velocity.coords4dofs,1);
    ndofs::Int = ndofs_velocity + size(FE_pressure.coords4dofs,1);
    dofs = 1:ndofs;
    
    if boundary_data! == Nothing
        bdofs = [];
    else
        Grid.ensure_bfaces!(grid);
        bdofs = FE_velocity.dofs4faces[grid.bfaces,:]
        dofs = setdiff(dofs,bdofs);
        boundary_data!(view(val4coords, bdofs),view(FE_velocity.coords4dofs,bdofs,:),0); 
        b = b - A*val4coords;
    end    
    
    # remove one pressure dof
    #fixed_pressure_dof = dofs[end];
    #println("fixing one pressure dof with dofnr=",fixed_pressure_dof);
    #dofs = setdiff(dofs, fixed_pressure_dof);
    #A[fixed_pressure_dof,fixed_pressure_dof] = 1e30;
    
    #dofs = setdiff(dofs, ndofs_velocity+1:ndofs);
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
    
    # move integral mean to zero
    integral_mean = 0.0;
    ncells = size(grid.nodes4cells,1);
    ndofs4cell_pressure = size(FE_pressure.dofs4cells,2)
    for j = 1 : ncells
        integral_mean += sum(FE_pressure.local_mass_matrix * val4coords[ndofs_velocity .+ FE_pressure.dofs4cells[j,:]]) * grid.volume4cells[j]
    end
    
    integral_mean /= sum(grid.volume4cells);
    val4coords[ndofs_velocity .+ FE_pressure.dofs4cells[:]] .-= integral_mean;
end


function computeFEInterpolation!(val4dofs::Array,source_function!::Function,grid::Grid.Mesh,FE::FiniteElements.FiniteElement)
    source_function!(val4dofs,FE.coords4dofs);
end


function eval_interpolation_error!(result, x, xref, cellIndex, exact_function!, coeffs_interpolation, FE::FiniteElements.FiniteElement)
    # evaluate exact function
    exact_function!(result, x);
    # subtract nodal interpolation
    ndofcell = size(FE.dofs4cells,2);
    for j = 1 : ndofcell
        result[1] -= coeffs_interpolation[FE.dofs4cells[cellIndex, j]] .* FE.bfun_ref[j](xref, FE.grid, cellIndex)
    end    
end


end
