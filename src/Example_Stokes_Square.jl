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
    fill!(result, 1.0);
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
FE = FiniteElements.get_TaylorHoodCompositeFE(grid,false);
val4coords = zeros(Base.eltype(grid.coords4nodes),FE.dofoffset4component[4]);
solveStokesProblem!(val4coords,volume_data!,boundary_data!,grid,FE,3);

# plot
if size(grid.coords4nodes,1) < 5000
    pygui(true)
    PyPlot.figure(1)
    PyPlot.plot_trisurf(view(FE.FE4component[1].coords4dofs,:,1),view(FE.FE4component[1].coords4dofs,:,2),val4coords[1:FE.dofoffset4component[2]],cmap=get_cmap("ocean"))
    PyPlot.title("Stokes Problem Solution - velocity component 1")
    PyPlot.figure(2)
    PyPlot.plot_trisurf(view(FE.FE4component[2].coords4dofs,:,1),view(FE.FE4component[2].coords4dofs,:,2),val4coords[FE.dofoffset4component[2]+1:FE.dofoffset4component[3]],cmap=get_cmap("ocean"))
    PyPlot.title("Stokes Problem Solution - velocity component 2")
    PyPlot.figure(3)
    PyPlot.plot_trisurf(view(FE.FE4component[3].coords4dofs,:,1),view(FE.FE4component[3].coords4dofs,:,2),val4coords[FE.dofoffset4component[3]+1:FE.dofoffset4component[4]],cmap=get_cmap("ocean"))
    PyPlot.title("Stokes Problem Solution - pressure")
    #show()
end    
end


main()
