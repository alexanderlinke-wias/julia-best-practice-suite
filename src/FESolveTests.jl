module FESolveTests

export TestInterpolation1D,TestL2BestApproximation1D,TestH1BestApproximation1D,TestPoissonSolver1D,TestL2BestApproximation1DBoundaryGrid,TestInterpolation2D,TestL2BestApproximation2D,TestH1BestApproximation2D,TestPoissonSolver2D

using SparseArrays
using LinearAlgebra
using FESolve
using Grid
using Quadrature
using FiniteElements


function load_test_grid(nrefinements::Int = 1)
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
               
    return Grid.Mesh(coords4nodes_init,nodes4cells_init,nrefinements);
end


function load_test_grid1D(nrefinements::Int = 0)
    # define grid
    coords4nodes_init = Array{Float64,2}([0.0 0.5 1.0]');
    nodes4cells_init = [1 2; 2 3];
    return Grid.Mesh(coords4nodes_init,nodes4cells_init,nrefinements);
end



  # define problem data
  # = linear function f(x,y) = x + y and its derivatives
  function volume_data1D!(result, x)
    for i in eachindex(result)
      @inbounds result[i] = x[i] + 1
    end
  end
  
  function volume_data!(result, x)
    for i in eachindex(result)
      @inbounds result[i] = x[i, 1] + x[i, 2]
    end
  end
  
  function volume_data_gradient!(result,x)
    result = ones(Float64,size(x));
  end
  
  function volume_data_laplacian!(result,x)
    result[:] = zeros(Float64,size(result));
  end
  boundary_data!(result,x,xref) = volume_data!(result,x);
  boundary_data1D!(result,x,xref) = volume_data1D!(result,x);

  

function TestInterpolation1D()
  grid = load_test_grid1D();
  
  # compute volume4cells
  Grid.ensure_volume4cells!(grid);  
  println("Testing P1 Interpolation in 1D...");
  FE = FiniteElements.get_P1FiniteElement(grid);
  val4dofs = zeros(size(FE.coords4dofs, 1));
  computeFEInterpolation!(val4dofs, volume_data1D!, grid, FE);
  wrapped_interpolation_error_integrand!(result, x, xref, cellIndex) = eval_interpolation_error!(result, x, xref, cellIndex, volume_data1D!, val4dofs, FE);
  
  integral4cells = zeros(size(grid.nodes4cells, 1), 1);
  integrate!(integral4cells, wrapped_interpolation_error_integrand!, grid, 1);
  integral = sum(integral4cells);
  println("interpolation_error = " * string(integral));

  return abs(integral) < eps(1.0)
end


function TestL2BestApproximation1D()
  grid = load_test_grid1D();
  println("Testing L2-Bestapproximation in 1D...");
  FE = FiniteElements.get_P1FiniteElement(grid);
  val4coords = zeros(size(FE.coords4dofs,1));
  computeBestApproximation!(val4coords,"L2",volume_data1D!,boundary_data1D!,grid,FE,2);
  wrapped_interpolation_error_integrand!(result, x, xref, cellIndex) = eval_interpolation_error!(result, x, xref, cellIndex, volume_data1D!, val4coords, FE);
  integral4cells = zeros(size(grid.nodes4cells,1),1);
  integrate!(integral4cells,wrapped_interpolation_error_integrand!,grid,1);
  integral = sum(integral4cells);
  println("interpolation_error = " * string(integral));
  return abs(integral) < eps(1.0)
end

function TestL2BestApproximation1DBoundaryGrid()
  grid = get_boundary_grid(load_test_grid(2););
  println("Testing L2-Bestapproximation on boundary grid of 2D triangulation...");
  FE = FiniteElements.get_P1FiniteElement(grid,true);
  val4coords = zeros(size(FE.coords4dofs,1));
  computeBestApproximation!(val4coords,"L2",volume_data!,Nothing,grid,FE,2);
  wrapped_interpolation_error_integrand!(result, x, xref, cellIndex) = eval_interpolation_error!(result, x, xref, cellIndex, volume_data!, val4coords, FE);
  integral4cells = zeros(size(grid.nodes4cells,1),1);
  integrate!(integral4cells,wrapped_interpolation_error_integrand!,grid,1);
  integral = sum(integral4cells);
  println("interpolation_error = " * string(integral));
  return abs(integral) < eps(1.0)
end



function TestH1BestApproximation1D()
  grid = load_test_grid1D();
  println("Testing H1-Bestapproximation in 1D...");
  FE = FiniteElements.get_P1FiniteElement(grid);
  val4coords = zeros(size(FE.coords4dofs,1));
  computeBestApproximation!(val4coords,"H1",volume_data_gradient!,boundary_data1D!,grid,FE,2);
  wrapped_interpolation_error_integrand!(result, x, xref, cellIndex) = eval_interpolation_error!(result, x, xref, cellIndex, volume_data1D!, val4coords, FE);
  integral4cells = zeros(size(grid.nodes4cells,1),1);
  integrate!(integral4cells,wrapped_interpolation_error_integrand!,grid,1);
  integral = sum(integral4cells);
  println("interpolation_error = " * string(integral));
  return abs(integral) < eps(1.0)
end


function TestPoissonSolver1D()
  grid = load_test_grid1D();
  println("Testing H1-Bestapproximation via Poisson solver in 1D...");
  FE = FiniteElements.get_P1FiniteElement(grid);
  val4coords = zeros(size(FE.coords4dofs,1));
  solvePoissonProblem!(val4coords,volume_data_laplacian!,boundary_data1D!,grid,FE,1);
  wrapped_interpolation_error_integrand!(result, x, xref, cellIndex) = eval_interpolation_error!(result, x, xref, cellIndex, volume_data1D!, val4coords, FE);
  integral4cells = zeros(size(grid.nodes4cells,1),1);
  integrate!(integral4cells,wrapped_interpolation_error_integrand!,grid,1);
  integral = sum(integral4cells);
  println("interpolation_error = " * string(integral));
  return abs(integral) < eps(1.0)
end

function TestInterpolation2D()
  grid = load_test_grid();
  
  # compute volume4cells
  Grid.ensure_volume4cells!(grid);  
  println("Testing P1 Interpolation in 2D...");
  
  FE = FiniteElements.get_P1FiniteElement(grid);
  val4dofs = zeros(size(FE.coords4dofs, 1));
  
  computeFEInterpolation!(val4dofs, volume_data!, grid, FE);
  
  wrapped_interpolation_error_integrand!(result, x, xref, cellIndex) = eval_interpolation_error!(result, x, xref, cellIndex, volume_data!, val4dofs, FE);
  
  integral4cells = zeros(size(grid.nodes4cells, 1), 1);
  integrate!(integral4cells, wrapped_interpolation_error_integrand!, grid, 1);
  integral = sum(integral4cells);
  println("interpolation_error = " * string(integral));

  return abs(integral) < eps(1.0)
end

function TestL2BestApproximation2DP1()
  grid = load_test_grid();
  println("Testing L2-Bestapproximation in 2D for P1-FEM...");
  FE = FiniteElements.get_P1FiniteElement(grid);
  val4coords = zeros(size(FE.coords4dofs,1));
  computeBestApproximation!(val4coords,"L2",volume_data!,boundary_data!,grid,FE,2);
  wrapped_interpolation_error_integrand!(result, x, xref, cellIndex) = eval_interpolation_error!(result, x, xref, cellIndex, volume_data!, val4coords, FE);
  integral4cells = zeros(size(grid.nodes4cells,1),1);
  integrate!(integral4cells,wrapped_interpolation_error_integrand!,grid,1);
  integral = sum(integral4cells);
  println("interpolation_error = " * string(integral));
  return abs(integral) < eps(1.0)
end

function TestL2BestApproximation2DP2()
  grid = load_test_grid();
  println("Testing L2-Bestapproximation in 2D for P2-FEM...");
  FE = FiniteElements.get_P2FiniteElement(grid);
  val4coords = zeros(size(FE.coords4dofs,1));
  computeBestApproximation!(val4coords,"L2",volume_data!,boundary_data!,grid,FE,4);
  wrapped_interpolation_error_integrand!(result, x, xref, cellIndex) = eval_interpolation_error!(result, x, xref, cellIndex, volume_data!, val4coords, FE);
  integral4cells = zeros(size(grid.nodes4cells,1),1);
  integrate!(integral4cells,wrapped_interpolation_error_integrand!,grid,2);
  integral = sum(integral4cells);
  println("interpolation_error = " * string(integral));
  return abs(integral) < eps(10.0)
end


function TestL2BestApproximation2DCR()
  grid = load_test_grid();
  println("Testing L2-Bestapproximation in 2D for CR-FEM...");
  FE = FiniteElements.get_CRFiniteElement(grid);
  val4coords = zeros(size(FE.coords4dofs,1));
  computeBestApproximation!(val4coords,"L2",volume_data!,boundary_data!,grid,FE,2);
  wrapped_interpolation_error_integrand!(result, x, xref, cellIndex) = eval_interpolation_error!(result, x, xref, cellIndex, volume_data!, val4coords, FE);
  integral4cells = zeros(size(grid.nodes4cells,1),1);
  integrate!(integral4cells,wrapped_interpolation_error_integrand!,grid,1);
  integral = sum(integral4cells);
  println("interpolation_error = " * string(integral));
  return abs(integral) < eps(1.0)
end


function TestH1BestApproximation2D()
  grid = load_test_grid();
  println("Testing H1-Bestapproximation in 2D...");
  FE = FiniteElements.get_P1FiniteElement(grid);
  val4coords = zeros(size(FE.coords4dofs,1));
  computeBestApproximation!(val4coords,"H1",volume_data_gradient!,boundary_data!,grid,FE,2);
  wrapped_interpolation_error_integrand!(result, x, xref, cellIndex) = eval_interpolation_error!(result, x, xref, cellIndex, volume_data!, val4coords, FE);
  integral4cells = zeros(size(grid.nodes4cells,1),1);
  integrate!(integral4cells,wrapped_interpolation_error_integrand!,grid,1);
  integral = sum(integral4cells);
  println("interpolation_error = " * string(integral));
  return abs(integral) < eps(1.0)
end


function TestPoissonSolver2DP1()
  grid = load_test_grid();
  println("Testing H1-Bestapproximation via Poisson solver in 2D for P1-FEM...");
  FE = FiniteElements.get_P1FiniteElement(grid);
  val4coords = zeros(size(FE.coords4dofs,1));
  solvePoissonProblem!(val4coords,volume_data_laplacian!,boundary_data!,grid,FE,1);
  wrapped_interpolation_error_integrand!(result, x, xref, cellIndex) = eval_interpolation_error!(result, x, xref, cellIndex, volume_data!, val4coords, FE);
  integral4cells = zeros(size(grid.nodes4cells,1),1);
  integrate!(integral4cells,wrapped_interpolation_error_integrand!,grid,1);
  integral = sum(integral4cells);
  println("interpolation_error = " * string(integral));
  return abs(integral) < eps(1.0)
end


function TestPoissonSolver2DCR()
  grid = load_test_grid();
  println("Testing H1-Bestapproximation via Poisson solver in 2D for CR-FEM...");
  FE = FiniteElements.get_CRFiniteElement(grid);
  val4coords = zeros(size(FE.coords4dofs,1));
  solvePoissonProblem!(val4coords,volume_data_laplacian!,boundary_data!,grid,FE,1);
  wrapped_interpolation_error_integrand!(result, x, xref, cellIndex) = eval_interpolation_error!(result, x, xref, cellIndex, volume_data!, val4coords, FE);
  integral4cells = zeros(size(grid.nodes4cells,1),1);
  integrate!(integral4cells,wrapped_interpolation_error_integrand!,grid,1);
  integral = sum(integral4cells);
  println("interpolation_error = " * string(integral));
  return abs(integral) < eps(1.0)
end


function TestPoissonSolver2DP2()
  grid = load_test_grid(2);
  println("Testing H1-Bestapproximation via Poisson solver in 2D for P2-FEM...");
  FE = FiniteElements.get_P2FiniteElement(grid);
  val4coords = zeros(size(FE.coords4dofs,1));
  solvePoissonProblem!(val4coords,volume_data_laplacian!,boundary_data!,grid,FE,3);
  wrapped_interpolation_error_integrand!(result, x, xref, cellIndex) = eval_interpolation_error!(result, x, xref, cellIndex, volume_data!, val4coords, FE);
  integral4cells = zeros(size(grid.nodes4cells,1),1);
  integrate!(integral4cells,wrapped_interpolation_error_integrand!,grid,3);
  integral = sum(integral4cells);
  println("interpolation_error = " * string(integral));
  return abs(integral) < eps(10.0)
end


function TimeStiffnessMatrixP1()
  grid = load_test_grid(7);
  ncells::Int = size(grid.nodes4cells,1);
  println("ncells=",ncells);
  Grid.ensure_volume4cells!(grid);
  dim=2
  
  aa = Vector{typeof(grid.coords4nodes[1])}(undef, (dim+1)^2*ncells);
  ii = Vector{Int64}(undef, (dim+1)^2*ncells);
  jj = Vector{Int64}(undef, (dim+1)^2*ncells);
  
  println("\n Stiffness-Matrix with exact gradients (fast version)");
  @time FESolve.global_stiffness_matrix!(aa,ii,jj,grid);
  M1 = sparse(ii,jj,aa);
  show(size(M1))
  println("\n Stiffness-Matrix with exact gradients");
  FE = FiniteElements.get_P1FiniteElement(grid,false);
  @time FESolve.global_stiffness_matrix4FE!(aa,ii,jj,grid,FE);
  M2 = sparse(ii,jj,aa);
  show(norm(M1-M2))
  println("\n Stiffness-Matrix with ForwardDiff gradients");
  FE = FiniteElements.get_P1FiniteElement(grid,true);
  @time FESolve.global_stiffness_matrix4FE!(aa,ii,jj,grid,FE);
  M3 = sparse(ii,jj,aa);
  show(norm(M1-M3))
end


function TimeStiffnessMatrixP2()
  grid = load_test_grid(6);
  ncells::Int = size(grid.nodes4cells,1);
  println("ncells=",ncells);
  Grid.ensure_volume4cells!(grid);
  dim=2
  
  aa = Vector{typeof(grid.coords4nodes[1])}(undef, (2*(dim+1))^2*ncells);
  ii = Vector{Int64}(undef, (2*(dim+1))^2*ncells);
  jj = Vector{Int64}(undef, (2*(dim+1))^2*ncells);
  
  println("\n Stiffness-Matrix with exact gradients");
  FE = FiniteElements.get_P2FiniteElement(grid, false);
  @time FESolve.global_stiffness_matrix4FE!(aa,ii,jj,grid,FE);
  M1 = sparse(ii,jj,aa);
  
  println("\n Stiffness-Matrix with ForwardDiff gradients");
  FE = FiniteElements.get_P2FiniteElement(grid, true);
  @time FESolve.global_stiffness_matrix4FE!(aa,ii,jj,grid,FE);
  M2 = sparse(ii,jj,aa);
  show(norm(M1-M2))
end


function TimeStiffnessMatrixCR()
  grid = load_test_grid(6);
  ncells::Int = size(grid.nodes4cells,1);
  println("ncells=",ncells);
  Grid.ensure_volume4cells!(grid);
  dim=2
  
  aa = Vector{typeof(grid.coords4nodes[1])}(undef, 9*ncells);
  ii = Vector{Int64}(undef, 9*ncells);
  jj = Vector{Int64}(undef, 9*ncells);
  
  println("\n Stiffness-Matrix with exact gradients");
  FE = FiniteElements.get_CRFiniteElement(grid, false);
  @time FESolve.global_stiffness_matrix4FE!(aa,ii,jj,grid,FE);
  M1 = sparse(ii,jj,aa);
  
  println("\n Stiffness-Matrix with ForwardDiff gradients");
  FE = FiniteElements.get_CRFiniteElement(grid, true);
  @time FESolve.global_stiffness_matrix4FE!(aa,ii,jj,grid,FE);
  M2 = sparse(ii,jj,aa);
  show(norm(M1-M2))
end

function TimeMassMatrix()
  grid = load_test_grid(5);
  ncells::Int = size(grid.nodes4cells,1);
  println("ncells=",ncells);
  Grid.ensure_volume4cells!(grid);
  dim=2
  
  aa = Vector{typeof(grid.coords4nodes[1])}(undef, (dim+1)^2*ncells);
  ii = Vector{Int64}(undef, (dim+1)^2*ncells);
  jj = Vector{Int64}(undef, (dim+1)^2*ncells);
  gradients4cells = zeros(typeof(grid.coords4nodes[1]),dim+1,dim,ncells);
  
  # old mass matrix
  println("\nold mass matrix routine...")
  @time FESolve.global_mass_matrix_old!(aa,ii,jj,grid);
  @time FESolve.global_mass_matrix_old!(aa,ii,jj,grid);
  M = sparse(ii,jj,aa);
  
  # new mass matrix
  println("\nnew mass matrix routine...")
  @time FESolve.global_mass_matrix!(aa,ii,jj,grid);
  @time FESolve.global_mass_matrix!(aa,ii,jj,grid);
  M2 = sparse(ii,jj,aa);
  
  show(norm(M-M2))
  
  # new mass matrix
  println("\nnew mass matrix routine with FE...")
  FE = FiniteElements.get_P1FiniteElement(grid);
  @time FESolve.global_mass_matrix4FE!(aa,ii,jj,grid,FE);
  @time FESolve.global_mass_matrix4FE!(aa,ii,jj,grid,FE);
  M2 = sparse(ii,jj,aa);
  show(norm(M-M2))
end

end
