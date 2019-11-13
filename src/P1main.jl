using Grid
using Quadrature
using P1approx
ENV["MPLBACKEND"]="tkagg"
using PyPlot


function main()
# define problem data

function volume_data!(result,x)
    result[:] = @views x[:,1] .* (1 .- x[:,1]) .* x[:,2];
end
function Laplacian!(result,x)
    result[:] = @views 2.0 .* x[:,2];
end
boundary_data!(result,x,xref) = volume_data!(result,x);

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
               
println("Loading grid...");
@time T = Grid.Triangulation(coords4nodes_init,nodes4cells_init,4);
println("nnodes=",size(T.coords4nodes,1));
println("ncells=",size(T.nodes4cells,1));
println("Computing grid stuff...");
@time Grid.ensure_area4cells!(T)
@time Grid.ensure_nodes4faces!(T)
@time Grid.ensure_faces4cells!(T)
@time Grid.ensure_bfaces!(T)


# interpolate
println("Computing P1 Interpolation...");
val4coords = zeros(size(T.coords4nodes,1));

@time computeP1Interpolation!(val4coords,volume_data!,T);

# bestapproximate
println("Computing P1 L2-Bestapproximation...");
val4coords2 = zeros(size(T.coords4nodes,1));
@time computeP1BestApproximation!(val4coords2,"L2",volume_data!,boundary_data!,T,4);

# bestapproximate
println("Computing P1 H1-Bestapproximation via Poisson solver...");
val4coords3 = zeros(size(T.coords4nodes,1));
@time solvePoissonProblem!(val4coords3,Laplacian!,boundary_data!,T,4);


# compute interpolation error and bestapproximation error
wrapped_interpolation_error_integrand!(result,x,xref) = eval_interpolation_error!(result,x,xref,volume_data!,val4coords,T.nodes4cells);
wrapped_L2bestapproximation_error_integrand!(result,x,xref) = eval_interpolation_error!(result,x,xref,volume_data!,val4coords2,T.nodes4cells);
wrapped_H1bestapproximation_error_integrand!(result,x,xref) = eval_interpolation_error!(result,x,xref,volume_data!,val4coords3,T.nodes4cells);

println("Computing errors by quadrature...")
integral4cells = zeros(size(T.nodes4cells,1),1);
@time integrate!(integral4cells,wrapped_interpolation_error_integrand!,T,1);
println("interpolation_error(integrate(order=1)) = " * string(sum(integral4cells)));
@time integrate!(integral4cells,wrapped_interpolation_error_integrand!,T,2);
println("interpolation_error(integrate(order=2)) = " * string(sum(integral4cells)));
@time integrate!(integral4cells,wrapped_interpolation_error_integrand!,T,3);
println("interpolation_error(integrate(order=3)) = " * string(sum(integral4cells)));
@time integrate!(integral4cells,wrapped_interpolation_error_integrand!,T,4);
println("interpolation_error(integrate(order=4)) = " * string(sum(integral4cells)));
@time integrate!(integral4cells,wrapped_L2bestapproximation_error_integrand!,T,4);
println("L2bestapprox_error(integrate(order=4)) = " * string(sum(integral4cells)));
@time integrate!(integral4cells,wrapped_H1bestapproximation_error_integrand!,T,4);
println("H1bestapprox_error(integrate(order=4)) = " * string(sum(integral4cells)));


# plot interpolation and bestapproximation
pygui(true)
PyPlot.figure(1)
PyPlot.plot_trisurf(view(T.coords4nodes,:,1),view(T.coords4nodes,:,2),val4coords,cmap=get_cmap("ocean"))
PyPlot.title("Interpolation")
PyPlot.figure(2)
PyPlot.plot_trisurf(view(T.coords4nodes,:,1),view(T.coords4nodes,:,2),val4coords2,cmap=get_cmap("ocean"))
PyPlot.title("L2-Bestapproximation")
PyPlot.figure(3)
PyPlot.plot_trisurf(view(T.coords4nodes,:,1),view(T.coords4nodes,:,2),val4coords3,cmap=get_cmap("ocean"))
PyPlot.title("H1-Bestapproximation")
show()
end


main()
