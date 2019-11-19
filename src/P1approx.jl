module P1approx

export solvePoissonProblem!,computeP1BestApproximation!,computeP1Interpolation!,eval_interpolation_error!,eval_interpolation_error2!

using SparseArrays
using LinearAlgebra
using BenchmarkTools

using Grid
using Quadrature

function accumarray!(A,subs, val, sz=(maximum(subs),))
      for i = 1:length(val)
          @inbounds A[subs[i]] += val[i]
      end
  end

# matrix for L2 bestapproximation
function global_mass_matrix(grid::Grid.Mesh)
    ncells::Int = size(grid.nodes4cells,1);
    nnodes::Int = size(grid.coords4nodes,1);
    dim::Int = size(grid.nodes4cells,2)-1;
    aa = Vector{eltype(grid.coords4nodes)}(undef, (dim+1)^2*ncells);
    
    # local mass matrix (the same on every triangle)
    local_mass_matrix = ones(Int64,dim+1,dim+1) + LinearAlgebra.I(dim+1);
    local_mass_matrix *= 1 // ((dim+1)*(dim+2));
    
    # do the 'integration'
    index = 0;
    for i = 1:dim+1, j = 1:dim+1
       @inbounds aa[index+1:index+ncells] = local_mass_matrix[i,j] * grid.volume4cells;
       index += ncells;
    end
    
    # setup sparse matrix
    ii = repeat(grid.nodes4cells,dim+1)[:];
    jj = repeat(grid.nodes4cells',dim+1)'[:];
    return sparse(ii,jj,aa,nnodes,nnodes);
end


# matrix for H1 bestapproximation and gradients on each cell
# version inspired by Matlab-AFEM group of C. Carstensen
# based on the formula
#
# gradients = [1 1 1; coords]^{-1} [0 0; 1 0; 0 1];
#
function global_stiffness_matrix_with_gradients(grid::Grid.Mesh)
    ncells::Int = size(grid.nodes4cells,1);
    nnodes::Int = size(grid.coords4nodes,1);
    dim::Int = size(grid.nodes4cells,2)-1;
    
    # compute local stiffness matrices
    Aloc = zeros(typeof(grid.coords4nodes[1]),dim+1,dim+1,ncells);
    gradients4cells = zeros(typeof(grid.coords4nodes[1]),dim+1,dim,ncells);
    for cell = 1 : ncells
        if dim == 1
            @views gradients4cells[:,:,cell] = [1,-1]' / grid.volume4cells[cell]
        elseif dim == 2
            @views gradients4cells[:,:,cell] = [1 1 1; grid.coords4nodes[grid.nodes4cells[cell,:],:]'] \ [0 0; 1 0;0 1];
        end    
        @views Aloc[:,:,cell] = grid.volume4cells[cell] .* (gradients4cells[:,:,cell] * gradients4cells[:,:,cell]');
    end
    
    ii = repeat(grid.nodes4cells',dim+1)[:];
    jj = repeat(grid.nodes4cells'[:]',dim+1)[:];
    A = sparse(ii,jj,Aloc[:],nnodes,nnodes);
    return A, gradients4cells
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
function global_stiffness_matrix(grid::Grid.Mesh)
    ncells::Int = size(grid.nodes4cells,1);
    nnodes::Int = size(grid.coords4nodes,1);
    dim::Int = size(grid.nodes4cells,2)-1;
    sA = Vector{typeof(grid.coords4nodes[1])}(undef, (dim+1)^2*ncells);
    
    if dim == 1
        local_matrix = -ones(Int64,dim+1,dim+1) + 2*LinearAlgebra.I(dim+1);
        
        # do the 'integration'
        index = 0;
        for i = 1:dim+1, j = 1:dim+1
            @inbounds sA[index+1:index+ncells] = local_matrix[i,j] / grid.volume4cells;
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
            @inbounds sA[index+1:index+ncells] = sum(ve[:,:,i].* ve[:,:,j], dims=2) ./ (4 * grid.volume4cells);
            index += ncells;
        end
    end    
    
    # setup sparse matrix
    ii = repeat(grid.nodes4cells,dim+1)[:];
    jj = repeat(grid.nodes4cells',dim+1)'[:];
    return sparse(ii,jj,sA,nnodes,nnodes);
end



# scalar functions times P1 basis functions
function rhs_integrandL2!(result,x,xref,cellIndex::Int,f!::Function)
    f!(view(result, 1), x);
    for j=length(xref):-1:1
        result[j] = view(result,1) .* xref[j];
    end
end


function assembleSystem(norm_lhs::String,norm_rhs::String,volume_data!::Function,grid::Grid.Mesh,quadrature_order::Int)

    ncells::Int = size(grid.nodes4cells,1);
    nnodes::Int = size(grid.coords4nodes,1);
    dim::Int = size(grid.nodes4cells,2)-1;
    
    Grid.ensure_volume4cells!(grid);
    
    if norm_lhs == "L2"
        println("mass matrix")
        @time A = global_mass_matrix(grid);
    elseif norm_lhs == "H1"
        println("stiffness matrix")
        if norm_rhs == "H1"
            A, gradients4cells = global_stiffness_matrix_with_gradients(grid);
        else
            @time A = global_stiffness_matrix(grid);
        end    
    end 
    
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
        fintegral4cells = zeros(typeof(grid.coords4nodes[1]),ncells,dim);
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
    b = zeros(typeof(grid.coords4nodes[1]),nnodes);
    @time accumarray!(b,grid.nodes4cells,rhsintegral4cells,nnodes)
    
    return A,b
end

# computes Bestapproximation in norm="L2" or "H1"
# volume_data! for norm="H1" is expected to be the gradient of the function that is bestapproximated
function computeP1BestApproximation!(val4coords::Array,norm::String ,volume_data!::Function,boundary_data!,grid::Grid.Mesh,quadrature_order::Int)
    # assemble system 
    A, b = assembleSystem(norm,norm,volume_data!,grid,quadrature_order);
    
    # find boundary nodes
    if size(grid.nodes4cells,2) == 2
        if boundary_data! == Nothing
            bnodes = [];
        else
            bnodes = [1 size(grid.coords4nodes,1)];
        end    
    elseif size(grid.nodes4cells,2) == 3
        Grid.ensure_bfaces!(grid);
        bnodes = unique(grid.nodes4faces[grid.bfaces,:]);
    end    
    dofs = setdiff(unique(grid.nodes4cells[:]),bnodes);
    
    # solve
    println("solve");
    fill!(val4coords,0)
    if length(bnodes) > 0
        boundary_data!(view(val4coords,bnodes),view(grid.coords4nodes,bnodes,:),0);
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
function solvePoissonProblem!(val4coords::Array,volume_data!::Function,boundary_data!,grid::Grid.Mesh,quadrature_order::Int)
    # assemble system 
    A, b = assembleSystem("H1","L2",volume_data!,grid,quadrature_order);
    
    # find boundary nodes
    if size(grid.nodes4cells,2) == 2
        if boundary_data! == Nothing
            bnodes = [];
        else
            bnodes = [1 size(grid.coords4nodes,1)];
        end    
    elseif size(grid.nodes4cells,2) == 3
        Grid.ensure_bfaces!(grid);
        bnodes = unique(grid.nodes4faces[grid.bfaces,:]);
    end    
    dofs = setdiff(unique(grid.nodes4cells[:]),bnodes);
    
    # solve
    println("solve");
    fill!(val4coords,0)
    if length(bnodes) > 0
        boundary_data!(view(val4coords,bnodes),view(grid.coords4nodes,bnodes,:),0);
        b = b - A*val4coords;
    end    
    
    try
        @time val4coords[dofs] = A[dofs,dofs]\b[dofs];
    catch   
        println("Unsupported Number type for sparse lu detected: trying again with dense matrix");
        try
            @time val4coords[dofs] = Array{typeof(grid.coords4nodes[1]),2}(A[dofs,dofs])\b[dofs];
        catch e
            if isa(e,OverflowError)
                println("OverflowError (Rationals?): trying again as Float64 sparse matrix");
                @time val4coords[dofs] = Array{Float64,2}(A[dofs,dofs])\b[dofs];
            end    
        end
    end
end



function computeP1Interpolation!(val4coords::Array,source_function!::Function,grid::Grid.Mesh)
    source_function!(val4coords,grid.coords4nodes);
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
