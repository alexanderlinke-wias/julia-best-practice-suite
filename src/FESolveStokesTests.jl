module FESolveStokesTests

using SparseArrays
using LinearAlgebra
using FESolveStokes
using Grid
using Quadrature
using FiniteElements


function load_test_grid(nrefinements::Int = 1)
    # define grid
    coords4nodes_init = [0.0 0.0;
                        1.0 0.0;
                        1.0 1.0;
                        0.0 1.0;
                        0.5 0.5];
    nodes4cells_init = [1 2 5;
                        2 3 5;
                        3 4 5;
                        4 1 5];
               
    return Grid.Mesh(coords4nodes_init,nodes4cells_init,nrefinements);
end

  
function TimeStokesOperatorTH()
  grid = load_test_grid(5);
  ncells::Int = size(grid.nodes4cells,1);
  println("ncells=",ncells);
  Grid.ensure_volume4cells!(grid);
  dim=2
  
  
  FE_velocity = FiniteElements.get_P2VectorFiniteElement(grid,false);
  FE_pressure = FiniteElements.get_P1FiniteElement(grid,false);
  ndofs4cell = size(FE_velocity.dofs4cells,2) + size(FE_pressure.dofs4cells,2)
  aa = Vector{typeof(grid.coords4nodes[1])}(undef, ndofs4cell^2*ncells);
  ii = Vector{Int64}(undef, ndofs4cell^2*ncells);
  jj = Vector{Int64}(undef, ndofs4cell^2*ncells);
  
  println("\n Stokes-Matrix with exact gradients");
  @time FESolveStokes.StokesOperator4FE!(aa,ii,jj,grid,FE_velocity,FE_pressure);
  
  M1 = sparse(ii,jj,aa);
  
  println("\n Stokes-Matrix with ForwardDiff gradients");
  FE_velocity = FiniteElements.get_P2VectorFiniteElement(grid,true);
  FE_pressure = FiniteElements.get_P1FiniteElement(grid,true);
  @time FESolveStokes.StokesOperator4FE!(aa,ii,jj,grid,FE_velocity,FE_pressure);
  M2 = sparse(ii,jj,aa);
  show(norm(M1-M2))
end

end
