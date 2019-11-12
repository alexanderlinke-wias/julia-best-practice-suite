module P1approx

export computeP1BestApproximation!,computeP1Interpolation!,eval_interpolation_error!,P1Test1

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



function global_mass_matrix(T::Grid.Triangulation)
    ncells::Int = size(T.nodes4cells,1);
    nnodes::Int = size(T.coords4nodes,1);
    I = repeat(T.nodes4cells',3)[:];
    J = repeat(T.nodes4cells'[:]',3)[:];
    V = repeat([2 1 1 1 2 1 1 1 2]'[:] ./ 12,ncells)[:].*repeat(T.area4cells',9)[:];
    A = sparse(I,J,V,nnodes,nnodes);
end

function rhs_integrand!(result::Array,x::Array,xref::Array,f!::Function)
    f!(view(result,:,1),x)
    for j=3:-1:1
        result[:,j] = view(result,:,1) .* xref[j];
    end
end



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
    b -= A*val4coords;
    
    @time val4coords[dofs] = A[dofs,dofs]\b[dofs];
end


function computeP1Interpolation!(val4coords::Array,source_function!::Function,T::Grid.Triangulation)
    source_function!(val4coords,T.coords4nodes);
end


function eval_interpolation_error!(result,x,xref,exact_function!,coeffs_interpolation,dofs_interpolation)
    exact_function!(view(result,:,1),x);
    result[:] -= sum(coeffs_interpolation[dofs_interpolation] .* repeat(xref[:]',size(dofs_interpolation,1)),dims=2);
end



### TESTS ###
function P1Test1()
  # define problem data
  function volume_data!(result,x)
    result[:] = @views x[:,1] + x[:,2];
  end

  # define grid
  coords4nodes_init = [0.0 0.0;
                       1.0 0.0;
                       1.0 1.0;
                       0.1 1.0;
                       0.5 0.6];
                       
  nodes4cells_init = [1 2 5;
                      2 3 5;
                      3 4 5;
                      4 1 5];
               
  println("Loading grid...");
  T = Grid.Triangulation(coords4nodes_init, nodes4cells_init, 1);
  println("Computing P1 Interpolation...");
  val4coords = zeros(size(T.coords4nodes, 1));
  computeP1Interpolation!(val4coords,volume_data!,T);
  wrapped_interpolation_error_integrand!(result,x,xref) = eval_interpolation_error!(result,x,xref,volume_data!,val4coords,T.nodes4cells);
  println("Computing errors by quadrature...")
  integral4cells = zeros(size(T.nodes4cells,1),1);
  integrate!(integral4cells,wrapped_interpolation_error_integrand!,T,1);
  println("interpolation_error(integrate(order=1)) = " * string(sum(integral4cells)));
  println(abs(sum(integral4cells[:])))
  
  return abs(sum(integral4cells[:])) < eps(1.0)
end

end
