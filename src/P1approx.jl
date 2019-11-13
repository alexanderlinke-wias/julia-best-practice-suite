module P1approx

export solvePoissonProblem!,computeP1BestApproximation!,computeP1Interpolation!,eval_interpolation_error!

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
    I = repeat(T.nodes4cells',3)[:];
    J = repeat(T.nodes4cells'[:]',3)[:];
    V = repeat([2 1 1 1 2 1 1 1 2]'[:] ./ 12,ncells)[:].*repeat(T.area4cells',9)[:];
    A = sparse(I,J,V,nnodes,nnodes);
end

# matrix for H1 bestapproximation and gradients
function global_stiffness_matrix(T::Grid.Triangulation)
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

# matrix for H1 bestapproximation
function global_stiffness_matrix(T::Grid.Triangulation)
    ncells::Int = size(T.nodes4cells,1);
    nnodes::Int = size(T.coords4nodes,1);
    
    # compute local stiffness matrices
    Aloc = zeros(Float64,3,3,ncells);
    grads = zeros(Float64,3,2);
    for cell = 1 : ncells
        @inbounds grads = [1.0 1.0 1.0; view(T.coords4nodes,view(T.nodes4cells,cell,:),:)'] \ [0.0 0.0; 1.0 0.0;0.0 1.0];
        Aloc[:,:,cell] = T.area4cells[cell] .* grads * grads';
    end
    
    I = repeat(T.nodes4cells',3)[:];
    J = repeat(T.nodes4cells'[:]',3)[:];
    return sparse(I,J,Aloc[:],nnodes,nnodes);
end


# scalar functions times P1 basis functions
function rhs_integrandL2!(result::Array,x::Array,xref::Array,f!::Function)
    f!(view(result,:,1),x)
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
            A, gradients4cells = global_stiffness_matrix(T);
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


function eval_interpolation_error!(result,x,xref,exact_function!,coeffs_interpolation,dofs_interpolation)
    exact_function!(view(result,:,1),x);
    result[:] -= sum(coeffs_interpolation[dofs_interpolation] .* repeat(xref[:]',size(dofs_interpolation,1)),dims=2);
end

end
