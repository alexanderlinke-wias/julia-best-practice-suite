module FESolvePoisson

export solvePoissonProblem!

using SparseArrays
using LinearAlgebra
using BenchmarkTools
using FiniteElements
using FESolveCommon
using Grid


# computes solution of Poisson problem
function solvePoissonProblem!(val4coords::Array,volume_data!::Function,boundary_data!,grid::Grid.Mesh,FE::FiniteElements.FiniteElement,quadrature_order::Int, dirichlet_penalty = 1e60)
    # assemble system 
    A, b = FESolveCommon.assembleSystem("H1","L2",volume_data!,grid,FE,quadrature_order);
    
    # find boundary dofs
    xdim = FE.ncomponents;
    ndofs::Int = FE.ndofs;
    dofs = 1:ndofs;
    bdofs = [];
    if ((boundary_data! == Nothing) || (size(FE.dofs4faces,1) == 0))
    else
        Grid.ensure_bfaces!(grid);
        Grid.ensure_cells4faces!(grid);
        xref = zeros(eltype(FE.xref4dofs4cell),size(FE.xref4dofs4cell,2));
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
                for l = 1 : length(xref)
                    xref[l] = FE.xref4dofs4cell[celldof2facedof[k],l];
                end    
                for l = 1:ndofs4bfaces
                    A4bface[k,l] = dot(FE.bfun_ref[celldof2facedof[k]](xref,grid,cell),FE.bfun_ref[celldof2facedof[l]](xref,grid,cell));
                end
                
                boundary_data!(temp,FE.loc2glob_trafo(grid,cell)(xref));
                b4bface[k] = dot(temp,FE.bfun_ref[celldof2facedof[k]](xref,grid,cell));
            end
            val4coords[bdofs4bface] = A4bface\b4bface;
            if norm(A4bface*val4coords[bdofs4bface]-b4bface) > eps(1e3)
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

end
