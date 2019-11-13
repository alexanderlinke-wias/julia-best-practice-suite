using Grid
using Quadrature
using P1approx
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
coords4nodes_init = [-1.0 -1.0;
                     0.0 -1.0;
                     0.0 0.0;
                     1.0 0.0;
                     1.0 1.0;
                     0.0 1.0;
                     -1.0 1.0;
                     -1.0 0.0];
nodes4cells_init = [1 2 3;
                    1 3 8;
                    3 4 5;
                    5 6 3;
                    8 3 6;
                    8 6 7];
               
println("Loading grid...");
@time T = Grid.Triangulation(coords4nodes_init,nodes4cells_init,6);
println("nnodes=",size(T.coords4nodes,1));
println("ncells=",size(T.nodes4cells,1));

println("Solving Poisson problem...");
val4coords = zeros(size(T.coords4nodes,1));

@time solvePoissonProblem!(val4coords,volume_data!,boundary_data!,T,4);
@time solvePoissonProblem!(val4coords,volume_data!,boundary_data!,T,4);

# plot
pygui(true)
PyPlot.figure(1)
PyPlot.plot_trisurf(view(T.coords4nodes,:,1),view(T.coords4nodes,:,2),val4coords,cmap=get_cmap("ocean"))
PyPlot.title("Poisson Problem Solution")
show()
end


main()
