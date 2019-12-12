module FESolveStokesTests

using SparseArrays
using LinearAlgebra
using FESolveCommon
using FESolveStokes
using Grid
using Quadrature
using FiniteElements
ENV["MPLBACKEND"]="tkagg"
using PyPlot


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

  
function TestStokesTH(show_plot::Bool = false)

    # data for test problem u = (y^2,x^2), p = x
    function volume_data!(result, x) 
        fill!(result,-2.0);
        result[1] += 1.0;
    end

    function exact_velocity!(result,x)
        result[1] = x[2]^2;
        result[2] = x[1]^2;
    end
    
    function exact_pressure!(result,x)
        result[1] = x[1] - 2*x[2] + 1 // 2;
    end

    # define grid
    coords4nodes_init = [0 0;
                        1 0;
                        1 1;
                        0 1];
    nodes4cells_init = [1 2 3;
                        1 3 4];
               
    println("Loading grid...");
    @time grid = Grid.Mesh{Float64}(coords4nodes_init,nodes4cells_init,2);
    println("nnodes=",size(grid.coords4nodes,1));
    println("ncells=",size(grid.nodes4cells,1));

    println("Solving Stokes problem...");
    FE_velocity = FiniteElements.get_P2VectorFiniteElement(grid,false);
    FE_pressure = FiniteElements.get_P1FiniteElement(grid,false);
    ndofs_velocity = size(FE_velocity.coords4dofs,1);
    ndofs_pressure = size(FE_pressure.coords4dofs,1);
    ndofs_total = ndofs_velocity + ndofs_pressure;
    println("ndofs_velocity=",ndofs_velocity);
    println("ndofs_pressure=",ndofs_pressure);
    println("ndofs_total=",ndofs_total);
    val4coords = zeros(Base.eltype(grid.coords4nodes),ndofs_total);
    residual = solveStokesProblem!(val4coords,volume_data!,exact_velocity!,grid,FE_velocity,FE_pressure,4);
    println("residual = " * string(residual));
    integral4cells = zeros(size(grid.nodes4cells,1),1);
    integrate!(integral4cells,eval_interpolation_error!(exact_pressure!, val4coords[ndofs_velocity+1:end], FE_pressure), grid, 2);
    integral_pressure = abs(sum(integral4cells));
    println("pressure_error = " * string(integral_pressure));
    integral4cells = zeros(size(grid.nodes4cells,1),2);
    integrate!(integral4cells,eval_interpolation_error!(exact_velocity!, val4coords[1:ndofs_velocity], FE_velocity), grid, 4, 2);
    integral_velocity = abs(sum(integral4cells[:]));
    println("velocity_error = " * string(integral_velocity));
    
    # plot
    if show_plot
        pygui(true)
        offset_1 = Int(ndofs_velocity / 2);
        offset_2 = ndofs_velocity;
        PyPlot.figure(1)
        PyPlot.plot_trisurf(view(FE_velocity.coords4dofs,1:offset_1,1),view(FE_velocity.coords4dofs,1:offset_1,2),val4coords[1:offset_1],cmap=get_cmap("ocean"))
        PyPlot.title("Stokes Problem Solution - velocity component 1")
        PyPlot.figure(2)
        PyPlot.plot_trisurf(view(FE_velocity.coords4dofs,offset_1+1:offset_2,1),view(FE_velocity.coords4dofs,offset_1+1:offset_2,2),val4coords[offset_1+1:offset_2],cmap=get_cmap("ocean"))
        PyPlot.title("Stokes Problem Solution - velocity component 2")
        PyPlot.figure(3)
        PyPlot.plot_trisurf(view(FE_pressure.coords4dofs,:,1),view(FE_pressure.coords4dofs,:,2),val4coords[offset_2+1:end],cmap=get_cmap("ocean"))
        PyPlot.title("Stokes Problem Solution - pressure")
    #show()
    end    
    return integral_velocity + integral_pressure <= eps(1e5)
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
