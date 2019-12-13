using Grid
using Quadrature
using SparseArrays
using FiniteElements
using FESolveCommon
using FESolveStokes
ENV["MPLBACKEND"]="tkagg"
using PyPlot


function main()
# define problem data

function volume_data!(result, x)
    result[1] = 0.0;
    result[2] = 10.0;
end


function exact_pressure!(result,x)
    result[1] = 10*x[2] - 5.0;
end

function exact_velocity!(result,x)
    result[1] = x[1]
    result[2] = 0.0
end

# define grid
coords4nodes_init = [0 0;
                     1 0;
                     1 1;
                     0 1];
nodes4cells_init = [1 2 3;
                    1 3 4];
               
println("Loading grid...");
@time grid = Grid.Mesh{Float64}(coords4nodes_init,nodes4cells_init,4);
println("nnodes=",size(grid.coords4nodes,1));
println("ncells=",size(grid.nodes4cells,1));

println("Solving Stokes problem...");
fem = "BR"

if fem == "BR"
    # Bernardi-Raugel
    FE_velocity = FiniteElements.get_BRFiniteElement(grid,true);
    FE_pressure = FiniteElements.get_P0FiniteElement(grid);
elseif fem == "TH"
    # Taylor--Hood
    FE_velocity = FiniteElements.get_P2VectorFiniteElement(grid,false);
    FE_pressure = FiniteElements.get_P1FiniteElement(grid,false);
end    
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
integrate!(integral4cells,eval_L2_interpolation_error!(exact_pressure!, val4coords[ndofs_velocity+1:end], FE_pressure), grid, 2);
integral = sqrt(abs(sum(integral4cells)));
println("L2_pressure_error = " * string(integral));
integral4cells = zeros(size(grid.nodes4cells,1),2);
integrate!(integral4cells,eval_L2_interpolation_error!(exact_velocity!, val4coords[1:ndofs_velocity], FE_velocity), grid, 4, 2);
integral = sqrt(abs(sum(integral4cells[:])));
println("L2_velocity_error = " * string(integral));

# plot
if size(grid.coords4nodes,1) < 5000
    pygui(true)
    if fem == "BR"
        nnodes = size(grid.coords4nodes,1)
        nfaces = size(grid.nodes4faces,1)
        offset_1 = nnodes;
        offset_2 = 2*nnodes;
        offset_3 = 2*nnodes+nfaces;
    elseif fem == "TH"
        offset_1 = Int(ndofs_velocity / 2);
        offset_2 = ndofs_velocity;
        offset_3 = ndofs_velocity;
    end    
    
    PyPlot.figure(1)
    PyPlot.plot_trisurf(view(FE_velocity.coords4dofs,1:offset_1,1),view(FE_velocity.coords4dofs,1:offset_1,2),val4coords[1:offset_1],cmap=get_cmap("ocean"))
    PyPlot.title("Stokes Problem Solution - velocity component 1")
    PyPlot.figure(2)
    PyPlot.plot_trisurf(view(FE_velocity.coords4dofs,offset_1+1:offset_2,1),view(FE_velocity.coords4dofs,offset_1+1:offset_2,2),val4coords[offset_1+1:offset_2],cmap=get_cmap("ocean"))
    PyPlot.title("Stokes Problem Solution - velocity component 2")
    PyPlot.figure(3)
    PyPlot.plot_trisurf(view(FE_pressure.coords4dofs,:,1),view(FE_pressure.coords4dofs,:,2),val4coords[offset_3+1:end],cmap=get_cmap("ocean"))
    PyPlot.title("Stokes Problem Solution - pressure")
    #show()
end    
end


main()
