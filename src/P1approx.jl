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
function global_mass_matrix(T::Grid.Triangulation)
    ncells::Int = size(T.nodes4cells,1);
    nnodes::Int = size(T.coords4nodes,1);
    sA = Vector{Float64}(undef, 9*ncells);
    
    # local mass matrix (the same on every triangle)
    local_mass_matrix = [2 1 1; 1 2 1; 1 1 2] ./ 12;
    
    # do the 'integration'
    index = 0;
    for i = 1:3, j = 1:3
       @inbounds sA[index+1:index+ncells] = local_mass_matrix[i,j] * T.area4cells;
       index += ncells;
    end
    
    # setup sparse matrix
    I = repeat(T.nodes4cells,3)[:];
    J = repeat(T.nodes4cells',3)'[:];
    return sparse(I,J,sA);
end


# matrix for H1 bestapproximation and gradients on each cell
# version inspired by Matlab-AFEM group of C. Carstensen
# based on the formula
#
# gradients = [1 1 1; coords]^{-1} [0 0; 1 0; 0 1];
#
function global_stiffness_matrix_with_gradients(T::Grid.Triangulation)
    ncells::Int = size(T.nodes4cells,1);
    nnodes::Int = size(T.coords4nodes,1);
    
    # compute local stiffness matrices
    Aloc = zeros(Float64,3,3,ncells);
    gradients4cells = zeros(Float64,3,2,ncells);
    for cell = 1 : ncells
        @views gradients4cells[:,:,cell] = [1.0 1.0 1.0; T.coords4nodes[T.nodes4cells[cell,:],:]'] \ [0.0 0.0; 1.0 0.0;0.0 1.0];
        @views Aloc[:,:,cell] = T.area4cells[cell] .* (gradients4cells[:,:,cell] * gradients4cells[:,:,cell]');
    end
    
    I = repeat(T.nodes4cells',3)[:];
    J = repeat(T.nodes4cells'[:]',3)[:];
    A = sparse(I,J,Aloc[:],nnodes,nnodes);
    return A, gradients4cells
end

#
# matrix for H1 bestapproximation
# this version is inspired by Julia iFEM
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
function global_stiffness_matrix(T::Grid.Triangulation)
    ncells::Int = size(T.nodes4cells,1);
    nnodes::Int = size(T.coords4nodes,1);
    sA = Vector{Float64}(undef, 9*ncells);
    ve = Array{Float64}(undef, ncells,2,3);
    
    # compute coordinate differences (= weighted tangents)
    @views ve[:,:,3] = T.coords4nodes[vec(T.nodes4cells[:,2]),:]-T.coords4nodes[vec(T.nodes4cells[:,1]),:];
    @views ve[:,:,1] = T.coords4nodes[vec(T.nodes4cells[:,3]),:]-T.coords4nodes[vec(T.nodes4cells[:,2]),:];
    @views ve[:,:,2] = T.coords4nodes[vec(T.nodes4cells[:,1]),:]-T.coords4nodes[vec(T.nodes4cells[:,3]),:];
    
    # do the 'integration'
    index = 0;
    for i = 1:3, j = 1:3
       @inbounds sA[index+1:index+ncells] = sum(ve[:,:,i].* ve[:,:,j], dims=2) ./ (4 * T.area4cells);
       index += ncells;
    end
    
    # setup sparse matrix
    I = repeat(T.nodes4cells,3)[:];
    J = repeat(T.nodes4cells',3)'[:];
    return sparse(I,J,sA);
end



# scalar functions times P1 basis functions
function rhs_integrandL2!(result::Array,x::Array,xref::Array,f!::Function)
    f!(view(result, :, 1), x)
    for j=3:-1:1
        result[:,j] = view(result,:,1) .* xref[j];
    end
end


function assembleSystem(norm_lhs::String,norm_rhs::String,volume_data!::Function,T::Grid.Triangulation,quadrature_order::Int)

    ncells::Int = size(T.nodes4cells,1);
    nnodes::Int = size(T.coords4nodes,1);
    dim::Int = size(T.coords4nodes,2);
    
    Grid.ensure_area4cells!(T);
    
    if norm_lhs == "L2"
        println("mass matrix")
        @time A = global_mass_matrix(T);
    elseif norm_lhs == "H1"
        println("stiffness matrix")
        if norm_rhs == "H1"
            A, gradients4cells = global_stiffness_matrix_with_gradients(T);
        else
            @time A = global_stiffness_matrix(T);
        end    
    end 
    
    # compute right-hand side vector
    rhsintegral4cells = zeros(Float64,ncells,dim+1); # f x P1basis (dim+1 many)
    if norm_rhs == "L2"
        println("integrate rhs");
        wrapped_integrand_L2!(result,x,xref) = rhs_integrandL2!(result,x,xref,volume_data!);
        @time integrate!(rhsintegral4cells,wrapped_integrand_L2!,T,quadrature_order,dim+1);
    elseif norm_rhs == "H1"
        @assert norm_lhs == "H1"
        # compute cell-wise integrals for right-hand side vector (f expected to be dim-dimensional)
        println("integrate rhs");
        fintegral4cells = zeros(Float64,ncells,dim);
        wrapped_integrand_f!(result,x,xref) = volume_data!(result,x);
        @time integrate!(fintegral4cells,wrapped_integrand_f!,T,quadrature_order,dim);
        
        # multiply with gradients
        for j = 1 : dim + 1
            for k = 1 : dim
                rhsintegral4cells[:,j] += (fintegral4cells[:,k] .* gradients4cells[j,k,:]);
            end
        end                            
    end
    
    # accumulate right-hand side vector
    println("accumarray");
    b = zeros(Float64,nnodes);
    @time accumarray!(b,T.nodes4cells,rhsintegral4cells,nnodes)
    
    return A,b
end

# computes Bestapproximation in norm="L2" or "H1"
# volume_data! for norm="H1" is expected to be the gradient of the function that is bestapproximated
function computeP1BestApproximation!(val4coords::Array,norm::String ,volume_data!::Function,boundary_data!::Function,T::Grid.Triangulation,quadrature_order::Int)
    # assemble system 
    A, b = assembleSystem(norm,norm,volume_data!,T,quadrature_order);
    
    # find boundary nodes
    Grid.ensure_nodes4faces!(T);
    Grid.ensure_bfaces!(T);
    bnodes = unique(T.nodes4faces[T.bfaces,:]);
    dofs = setdiff(1:size(T.coords4nodes,1),bnodes);
    
    # solve
    println("solve");
    fill!(val4coords,0.0)
    boundary_data!(view(val4coords,bnodes),view(T.coords4nodes,bnodes,:),0);
    b = b - A*val4coords;
    
    @time val4coords[dofs] = A[dofs,dofs]\b[dofs];
end

# computes solution of Poisson problem
function solvePoissonProblem!(val4coords::Array,volume_data!::Function,boundary_data!::Function,T::Grid.Triangulation,quadrature_order::Int)
    # assemble system 
    A, b = assembleSystem("H1","L2",volume_data!,T,quadrature_order);
    
    # find boundary nodes
    Grid.ensure_nodes4faces!(T);
    Grid.ensure_bfaces!(T);
    bnodes = unique(T.nodes4faces[T.bfaces,:]);
    dofs = setdiff(1:size(T.coords4nodes,1),bnodes);
    
    # solve
    println("solve");
    fill!(val4coords,0.0)
    boundary_data!(view(val4coords,bnodes),view(T.coords4nodes,bnodes,:),0);
    b = b - A*val4coords;
    
    @time val4coords[dofs] = A[dofs,dofs]\b[dofs];
end



function computeP1Interpolation!(val4coords::Array,source_function!::Function,T::Grid.Triangulation)
    source_function!(val4coords,T.coords4nodes);
end


function eval_interpolation_error!(result, x, xref, exact_function!, coeffs_interpolation, dofs_interpolation)
    # evaluate exact function
    exact_function!(result, x);
    # subtract nodal interpolation
    for i=1 : size(dofs_interpolation, 1)
      for j=1:3
        @inbounds result[i] -= coeffs_interpolation[dofs_interpolation[i, j]] * xref[j]
      end
    end
end

function eval_interpolation_error2!(result, x, xref, cellIndex, exact_function!, coeffs_interpolation, dofs_interpolation)
    # evaluate exact function
    exact_function!(result, x);
    # subtract nodal interpolation
    for j=1:3
      @inbounds result[1] -= coeffs_interpolation[dofs_interpolation[cellIndex, j]] * xref[j]
    end
end

end
