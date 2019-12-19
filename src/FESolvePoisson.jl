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
    
    # apply boundary data
    bdofs = FESolveCommon.computeDirichletBoundaryData!(val4coords,FE,boundary_data!);
    for i = 1 : length(bdofs)
       A[bdofs[i],bdofs[i]] = dirichlet_penalty;
       b[bdofs[i]] = val4coords[bdofs[i]]*dirichlet_penalty;
    end
    
    try
        @time val4coords[:] = A\b;
    catch    
        println("Unsupported Number type for sparse lu detected: trying again with dense matrix");
        try
            @time val4coords[dofs] = Array{typeof(grid.coords4nodes[1]),2}(A[dofs,dofs])\b[dofs];
        catch OverflowError
            println("OverflowError (Rationals?): trying again as Float64 sparse matrix");
            @time val4coords[dofs] = Array{Float64,2}(A[dofs,dofs])\b[dofs];
        end
    end
    
    # compute residual (exclude bdofs)
    residual = A*val4coords - b
    residual[bdofs] .= 0
    
    return norm(residual)
end

end
