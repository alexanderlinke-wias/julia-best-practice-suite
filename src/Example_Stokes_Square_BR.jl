using Grid
using Quadrature
using SparseArrays
using FiniteElements
using FESolve
ENV["MPLBACKEND"]="tkagg"
using PyPlot


function main()
# define problem data

function volume_data!(result, x)
    fill!(result, 1.0)
end

function boundary_data!(result,x,xref)
    fill!(result, 0.0);
end

# define grid
coords4nodes_init = [0 0;
                     1 0;
                     1 1;
                     0 1];
nodes4cells_init = [1 2 3;
                    1 3 4];
               
println("Loading grid...");
@time grid = Grid.Mesh{Float64}(coords4nodes_init,nodes4cells_init,3);
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
solveStokesProblem!(val4coords,volume_data!,boundary_data!,grid,FE_velocity,FE_pressure,3);

# plot
if size(grid.coords4nodes,1) < 5000
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
end


main()
