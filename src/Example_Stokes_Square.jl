using Grid
using Quadrature
using SparseArrays
using FiniteElements
using FESolveCommon
using FESolveStokes
using ForwardDiff
ENV["MPLBACKEND"]="tkagg"
using PyPlot


function main()
# define problem data

use_problem = "P7vortex"
maxlevel = 5;
fem = "BR"
show_plots = true
show_convergence_history = true

function theta(problem)
    function closure(x)
        if problem == "linear"
            return - x[2]^2 * 1//2;
        
        elseif problem == "P7vortex"
            return x[1]^2 * (x[1] - 1)^2 * x[2]^2 * (x[2] - 1)^2
        end
    end    
end    


function exact_pressure(problem)
    function closure(x)
        if problem == "linear"
            return 10*x[2] - 5;
        elseif problem == "P7vortex"
            return x[1]^3 + x[2]^3 - 1//2    
        end
    end    
end


function volume_data!(problem)
    gradp = [0.0,0.0]
        gradtheta = [0.0, 0.0]
        hessian = [0.0 0.0;0.0 0.0]
        function velo1(x)
            
            return -gradtheta[2]
            return 0.0;
        end 
        function velo2(x)
            #ForwardDiff.gradient!(gradtheta,theta(problem),x);
            #return gradtheta[1]
            return 0.0;
        end 
    return function closure(result, x)  
        # compute gradient of pressure
        p(x) = exact_pressure(problem)(x)
        ForwardDiff.gradient!(gradp,p,x);
        result[1] = gradp[1]
        result[2] = gradp[2]
        # add Laplacian of velocity
        velo_rotated(a) = ForwardDiff.gradient(theta(problem),a);
        velo1 = x -> -velo_rotated(x)[2]
        velo2 = x -> velo_rotated(x)[1]
        hessian = ForwardDiff.hessian(velo1,x)
        result[1] -= hessian[1] + hessian[3]
        hessian = ForwardDiff.hessian(velo2,x)
        result[2] -= hessian[1] + hessian[3]
    end    
end


function exact_velocity!(problem)
    gradtheta = [0.0, 0.0]
    return function closure(result, x)
        ForwardDiff.gradient!(gradtheta,theta(problem),x);
        result[1] = -gradtheta[2]
        result[2] = gradtheta[1]
    end    
    
end

# define grid
coords4nodes_init = [0 0;
                     1 0;
                     1 1;
                     0 1];
nodes4cells_init = [1 2 3;
                    1 3 4];
               
println("Loading grid...");


L2error_velocity = zeros(Float64,maxlevel)
L2error_pressure = zeros(Float64,maxlevel)
ndofs = zeros(Int,maxlevel)

for level = 1 : maxlevel
println("Solving Stokes problem on refinement level...", level);
@time grid = Grid.Mesh{Float64}(coords4nodes_init,nodes4cells_init,level);
println("nnodes=",size(grid.coords4nodes,1));
println("ncells=",size(grid.nodes4cells,1));

if fem == "BR"
    # Bernardi-Raugel
    FE_velocity = FiniteElements.get_BRFiniteElement(grid,false);
    FE_pressure = FiniteElements.get_P0FiniteElement(grid);
elseif fem == "TH"
    # Taylor--Hood
    FE_velocity = FiniteElements.get_P2VectorFiniteElement(grid,false);
    FE_pressure = FiniteElements.get_P1FiniteElement(grid,false);
end    
ndofs_velocity = size(FE_velocity.coords4dofs,1);
ndofs_pressure = size(FE_pressure.coords4dofs,1);
ndofs[level] = ndofs_velocity + ndofs_pressure;
println("ndofs_velocity=",ndofs_velocity);
println("ndofs_pressure=",ndofs_pressure);
println("ndofs_total=",ndofs[level]);
val4coords = zeros(Base.eltype(grid.coords4nodes),ndofs[level]);
residual = solveStokesProblem!(val4coords,volume_data!(use_problem),exact_velocity!(use_problem),grid,FE_velocity,FE_pressure,FE_velocity.polynomial_order);
println("residual = " * string(residual));
integral4cells = zeros(size(grid.nodes4cells,1),1);
function wrap_pressure(result,x)
    result[1] = exact_pressure(use_problem)(x)
end    
integrate!(integral4cells,eval_L2_interpolation_error!(wrap_pressure, val4coords[ndofs_velocity+1:end], FE_pressure), grid, maximum([2,2*FE_pressure.polynomial_order]), 1);
L2error_pressure[level] = sqrt(abs(sum(integral4cells)));
println("L2_pressure_error = " * string(L2error_pressure[level]));
integral4cells = zeros(size(grid.nodes4cells,1),2);
integrate!(integral4cells,eval_L2_interpolation_error!(exact_velocity!(use_problem), val4coords[1:ndofs_velocity], FE_velocity), grid, 2*FE_velocity.polynomial_order, 2);
L2error_velocity[level] = sqrt(abs(sum(integral4cells[:])));
println("L2_velocity_error = " * string(L2error_velocity[level]));

# plot
if (show_plots) && (level == maxlevel)
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
end # loop over levels

println("\n L2 pressure error");
show(L2error_pressure)
println("\n L2 velocity error");
show(L2error_velocity)

if (show_convergence_history)
    PyPlot.figure(4)
    PyPlot.loglog(ndofs,L2error_velocity)
    PyPlot.loglog(ndofs,L2error_pressure)
    PyPlot.legend(("L2 error velocity","L2 error pressure"))
    PyPlot.title("Convergence history")
end    

    
end


main()
