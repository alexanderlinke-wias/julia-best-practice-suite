module FESolveStokes

export solveStokesProblem!

using SparseArrays
using LinearAlgebra
using BenchmarkTools
using FiniteElements
using FESolveCommon
using Grid
using Quadrature


function StokesOperator4FE!(aa, ii, jj, grid, FE_velocity::FiniteElements.FiniteElement, FE_pressure::FiniteElements.FiniteElement, pressure_diagonal = 1e-12)
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
    
    T = eltype(grid.coords4nodes);
    quadorder = maximum([FE_pressure.polynomial_order + FE_velocity.polynomial_order-1, 2*(FE_velocity.polynomial_order-1)]);
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
    for i in eachindex(qf.w)
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
            if (dof_i == dof_j)
                aa[curindex] = pressure_diagonal;
            end    
            if (i == 1)
                ii[curindex] = ndofs_velocity + FE_pressure.dofs4cells[cell,dof_i];
                jj[curindex] = ndofs_velocity + FE_pressure.dofs4cells[cell,dof_j];
            end 
        end
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
    
    println("assembling Stokes matrix for FE pair " * FE_velocity.name * " x " * FE_pressure.name * "...");
    @time StokesOperator4FE!(aa,ii,jj,grid,FE_velocity,FE_pressure);
    A = sparse(ii,jj,aa,ndofs,ndofs);
    
    # compute right-hand side vector
    rhsintegral4cells = zeros(Base.eltype(grid.coords4nodes),ncells,ndofs4cell_velocity); # f x FEbasis
    println("integrate rhs for velocity");
    @time integrate!(rhsintegral4cells, rhs_integrand4Stokes!(volume_data!, FE_velocity,xdim), grid, quadrature_order, ndofs4cell_velocity);
         
    # accumulate right-hand side vector
    b = zeros(eltype(grid.coords4nodes),ndofs);
    FESolveCommon.accumarray!(b,FE_velocity.dofs4cells,rhsintegral4cells);
    
    return A,b
end


function solveStokesProblem!(val4coords::Array,volume_data!::Function,boundary_data!,grid::Grid.Mesh,FE_velocity::FiniteElements.FiniteElement,FE_pressure::FiniteElements.FiniteElement,quadrature_order::Int, dirichlet_penalty = 1e60)
    # assemble system 
    A, b = assembleStokesSystem(volume_data!,grid,FE_velocity,FE_pressure,quadrature_order);
    
    
    # find boundary dofs
    xdim = size(grid.coords4nodes,2);
    ndofs_velocity::Int = size(FE_velocity.coords4dofs,1);
    ndofs::Int = ndofs_velocity + size(FE_pressure.coords4dofs,1);
    dofs = 1:ndofs;
    
    bdofs = [];
    if boundary_data! == Nothing
    else
        Grid.ensure_bfaces!(grid);
        Grid.ensure_cells4faces!(grid);
        temp = zeros(eltype(grid.coords4nodes),xdim);
        cell::Int = 0;
        j::Int = 1;
        ndofs4bfaces = size(FE_velocity.dofs4faces,2);
        A4bface = Matrix{Float64}(undef,ndofs4bfaces,ndofs4bfaces)
        b4bface = Vector{Float64}(undef,ndofs4bfaces)
        bdofs4bface = Vector{Int}(undef,ndofs4bfaces)
        celldof2facedof = zeros(Int,ndofs4bfaces)
        for i in eachindex(grid.bfaces)
            cell = grid.cells4faces[grid.bfaces[i],1];
            # setup local system of equations to determine piecewise interpolation of boundary data
            bdofs4bface = FE_velocity.dofs4faces[grid.bfaces[i],:]
            append!(bdofs,bdofs4bface);
            # find position of face dofs in cell dofs
            for j=1:size(FE_velocity.dofs4cells,2), k = 1 : ndofs4bfaces
                if FE_velocity.dofs4cells[cell,j] == bdofs4bface[k]
                    celldof2facedof[k] = j;
                end    
            end
            # assemble matrix    
            for k = 1:ndofs4bfaces
                for l = 1:ndofs4bfaces
                    A4bface[k,l] = dot(FE_velocity.bfun[celldof2facedof[k]](view(FE_velocity.coords4dofs,bdofs4bface[k],:),grid,cell),FE_velocity.bfun[celldof2facedof[l]](view(FE_velocity.coords4dofs,bdofs4bface[k],:),grid,cell));
                end
                boundary_data!(temp,view(FE_velocity.coords4dofs,bdofs4bface[k],:));
                b4bface[k] = dot(temp,FE_velocity.bfun[celldof2facedof[k]](view(FE_velocity.coords4dofs,bdofs4bface[k],:),grid,cell));
            end
            val4coords[bdofs4bface] = A4bface\b4bface;
            if norm(A4bface*val4coords[bdofs4bface]-b4bface) > eps(1e4)
                println("WARNING: large residual, boundary data may be inexact");
            end
        end    
        #b = b - A*val4coords;
    end    
    
    #setdiff(dofs,bdofs)
    unique!(bdofs)
    for i = 1 : length(bdofs)
        A[bdofs[i],bdofs[i]] = dirichlet_penalty;
        b[bdofs[i]] = val4coords[bdofs[i]]*dirichlet_penalty;
    end
    
    # remove one pressure dof
    #fixed_pressure_dof = dofs[end];
    #println("fixing one pressure dof with dofnr=",fixed_pressure_dof);
    #A[fixed_pressure_dof,fixed_pressure_dof] = 1e30;
    #b[fixed_pressure_dof] = 0;
    #val4coords[fixed_pressure_dof] = 0;
    
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
    
    # compute residual
    #dofs = setdiff(dofs,bdofs);
    residual = norm(A[dofs,dofs]*val4coords[dofs] - b[dofs])
    
    # move integral mean to zero
    integral_mean = 0;
    ncells = size(grid.nodes4cells,1);
    ndofs4cell_pressure = size(FE_pressure.dofs4cells,2)
    for j = 1 : ncells
        integral_mean += sum(FE_pressure.local_mass_matrix * val4coords[ndofs_velocity .+ FE_pressure.dofs4cells[j,:]]) * grid.volume4cells[j]
    end
    
    integral_mean /= sum(grid.volume4cells);
    val4coords[ndofs_velocity .+ FE_pressure.dofs4cells[:]] .-= integral_mean;
    
    return residual
end

end
