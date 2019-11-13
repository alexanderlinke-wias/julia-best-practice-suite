module P1approxTests

export TestInterpolation,TestL2BestApproximation,TestH1BestApproximation,TestPoissonSolver

using SparseArrays
using LinearAlgebra
using P1approx
using Grid
using Quadrature


function load_test_grid(nrRefinements::Int = 1)
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
               
    return Grid.Triangulation(coords4nodes_init,nodes4cells_init,nrRefinements);
end



  # define problem data
  # = linear function f(x,y) = x + y and its derivatives
  function volume_data!(result,x)
    result[:] = @views x[:,1] + x[:,2];
  end
  function volume_data_gradient!(result,x)
    result = ones(Float64,size(x));
  end
  function volume_data_laplacian!(result,x)
    result[:] = zeros(Float64,size(result));
  end
  boundary_data!(result,x,xref) = volume_data!(result,x);


function TestInterpolation()
  T = load_test_grid();
  println("Testing P1 Interpolation...");
  val4coords = zeros(size(T.coords4nodes, 1));
  computeP1Interpolation!(val4coords,volume_data!,T);
  wrapped_interpolation_error_integrand!(result,x,xref) = eval_interpolation_error!(result,x,xref,volume_data!,val4coords,T.nodes4cells);
  integral4cells = zeros(size(T.nodes4cells,1),1);
  integrate!(integral4cells,wrapped_interpolation_error_integrand!,T,1);
  integral = sum(integral4cells);
  println("interpolation_error(integrate(order=1)) = " * string(integral));
  return abs(integral) < eps(1.0)
end


function TestL2BestApproximation()
  T = load_test_grid();
  println("Testing L2-Bestapproximation...");
  val4coords = zeros(size(T.coords4nodes,1));
  computeP1BestApproximation!(val4coords,"L2",volume_data!,boundary_data!,T,2);
  wrapped_interpolation_error_integrand!(result,x,xref) = eval_interpolation_error!(result,x,xref,volume_data!,val4coords,T.nodes4cells);
  integral4cells = zeros(size(T.nodes4cells,1),1);
  integrate!(integral4cells,wrapped_interpolation_error_integrand!,T,1);
  integral = sum(integral4cells);
  println("interpolation_error(integrate(order=1)) = " * string(integral));
  return abs(integral) < eps(1.0)
end


function TestH1BestApproximation()
  T = load_test_grid();
  println("Testing H1-Bestapproximation...");
  val4coords = zeros(size(T.coords4nodes,1));
  computeP1BestApproximation!(val4coords,"H1",volume_data_gradient!,boundary_data!,T,2);
  wrapped_interpolation_error_integrand!(result,x,xref) = eval_interpolation_error!(result,x,xref,volume_data!,val4coords,T.nodes4cells);
  integral4cells = zeros(size(T.nodes4cells,1),1);
  integrate!(integral4cells,wrapped_interpolation_error_integrand!,T,1);
  integral = sum(integral4cells);
  println("interpolation_error(integrate(order=1)) = " * string(integral));
  return abs(integral) < eps(1.0)
end


function TestPoissonSolver()
  T = load_test_grid();
  println("Testing H1-Bestapproximation via Poisson solver...");
  val4coords = zeros(size(T.coords4nodes,1));
  solvePoissonProblem!(val4coords,volume_data_laplacian!,boundary_data!,T,1);
  wrapped_interpolation_error_integrand!(result,x,xref) = eval_interpolation_error!(result,x,xref,volume_data!,val4coords,T.nodes4cells);
  integral4cells = zeros(size(T.nodes4cells,1),1);
  integrate!(integral4cells,wrapped_interpolation_error_integrand!,T,1);
  integral = sum(integral4cells);
  println("interpolation_error(integrate(order=1)) = " * string(integral));
  return abs(integral) < eps(1.0)
end

function TimeStiffnessMatrix()
  T = load_test_grid(7);
  println("nnodes=",size(T.coords4nodes,1));
  println("ncells=",size(T.nodes4cells,1));
  Grid.ensure_area4cells!(T);
  @time A1 = P1approx.global_stiffness_matrix(T);
  @time A2,blah = P1approx.global_stiffness_matrix_with_gradients(T);
end

end
