module P1approx

export computeP1BestApproximation!,computeP1Interpolation!,eval_interpolation_error!

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

# matrix for H1 bestapproximation
function global_stiffness_matrix(T::Grid.Triangulation)
    ncells::Int = size(T.nodes4cells,1);
    nnodes::Int = size(T.coords4nodes,1);
    
    # compute local stiffness matrices
    Aloc = zeros(Float64,3,3,ncells);
    grads = zeros(Float64,3,2);
    for cell = 1 : ncells
        grads = [1.0 1.0 1.0; T.coords4nodes[T.nodes4cells[cell,:],:]'] \ [0.0 0.0; 1.0 0.0;0.0 1.0];
        Aloc[:,:,cell] = T.area4cells[cell] .* (grads * grads');
    end
    
    I = repeat(T.nodes4cells',3)[:];
    J = repeat(T.nodes4cells'[:]',3)[:];
    A = sparse(I,J,Aloc[:],nnodes,nnodes);
end


function rhs_integrand!(result::Array,x::Array,xref::Array,f!::Function)
    f!(view(result,:,1),x)
    for j=3:-1:1
        result[:,j] = view(result,:,1) .* xref[j];
    end
end


# computes Bestapproximation in L2 or H1 which_norm
# volume_data! in H1 norm is expected to be the Laplacian of the function that is bestapproximated
function computeP1BestApproximation!(val4coords::Array,which_norm::String ,volume_data!::Function,boundary_data!::Function,T::Grid.Triangulation,quadrature_order::Int)
    ncells::Int = size(T.nodes4cells,1);
    nnodes::Int = size(T.coords4nodes,1);
    
    Grid.ensure_area4cells!(T);
    Grid.ensure_nodes4faces!(T);
    Grid.ensure_faces4cells!(T);
    Grid.ensure_bfaces!(T);
    
    # compute global mass matrix
    println("mass matrix")
    if which_norm == "L2"
        @time A = global_mass_matrix(T);
    elseif which_norm == "H1"
        @time A = global_stiffness_matrix(T);
    end    
    
    # compute right-hand side vector
    println("integrate rhs");
    wrapped_integrand!(result,x,xref) = rhs_integrand!(result,x,xref,volume_data!);
    integral4cells = zeros(Float64,ncells,3);
    @time integrate!(integral4cells,wrapped_integrand!,T,quadrature_order,3);
    
    println("accumarray");
    b = zeros(Float64,nnodes);
    @time accumarray!(b,T.nodes4cells,integral4cells,nnodes)
    
    
    # find boundary nodes
    bnodes = unique(T.nodes4faces[T.bfaces,:]);
    dofs = setdiff(1:nnodes,bnodes);
    
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
