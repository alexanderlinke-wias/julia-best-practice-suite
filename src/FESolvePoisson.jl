module FESolvePoisson

export solvePoissonProblem!

using SparseArrays
using LinearAlgebra
using BenchmarkTools
using FiniteElements
using FESolveCommon
using Grid


# computes solution of Poisson problem
function solvePoissonProblem!(val4coords::Array,volume_data!::Function,boundary_data!,grid::Grid.Mesh,FE::FiniteElements.FiniteElement,quadrature_order::Int)
    # assemble system 
    A, b = FESolveCommon.assembleSystem("H1","L2",volume_data!,grid,FE,quadrature_order);
    
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

end
