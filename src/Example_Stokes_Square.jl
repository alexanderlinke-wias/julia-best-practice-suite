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

#fem = "MINI"
#fem = "BR"
fem = "TH"
use_problem = "P7vortex"; f_order = 6; error_order = 6;
#use_problem = "linear"; f_order = 1; error_order = 3;
#use_problem = "quadratic"; f_order = 0; error_order = 4;
maxlevel = 6;
use_FDgradients = false
show_plots = false
show_convergence_history = true


function theta(problem)
    function closure(x)
        if problem == "linear"
            return - x[2]^2 * 1//2;
        elseif problem == "P7vortex"
            return x[1]^2 * (x[1] - 1)^2 * x[2]^2 * (x[2] - 1)^2
        elseif problem == "quadratic"
            return x[1]^3+x[2]^3;
        end
    end    
end    


function exact_pressure(problem)
    function closure(x)
        if problem == "linear"
            return 10*x[2] - 5;
        elseif problem == "P7vortex"
            return x[1]^3 + x[2]^3 - 1//2 
        elseif problem == "quadratic"
            return 0.0;    
        end
    end    
end


function volume_data!(problem)
    gradp = [0.0,0.0]
    gradtheta = [0.0, 0.0]
    hessian = [0.0 0.0;0.0 0.0]
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
        result[1] -= hessian[1] + hessian[4]
        hessian = ForwardDiff.hessian(velo2,x)
        result[2] -= hessian[1] + hessian[4]
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

function wrap_pressure(result,x)
    result[1] = exact_pressure(use_problem)(x)
end    

# define grid
coords4nodes_init = [0 0;
                     1 0;
                     1 1;
                     0 1;
                     0.5 0.5;];
nodes4cells_init = [1 2 5;
                    2 3 5;
                    3 4 5;
                    4 1 5];
               
println("Loading grid...");


L2error_velocity = zeros(Float64,maxlevel)
L2error_pressure = zeros(Float64,maxlevel)
L2error_velocityBA = zeros(Float64,maxlevel)
L2error_pressureBA = zeros(Float64,maxlevel)
ndofs = zeros(Int,maxlevel)

for level = 1 : maxlevel

# geenerate grid
println("Solving Stokes problem on refinement level...", level);
@time grid = Grid.Mesh{Float64}(coords4nodes_init,nodes4cells_init,level-1);
println("nnodes=",size(grid.coords4nodes,1));
println("ncells=",size(grid.nodes4cells,1));

# load finite element
if fem == "BR"
    # Bernardi-Raugel
    FE_velocity = FiniteElements.get_BRFiniteElement(grid,use_FDgradients);
    FE_pressure = FiniteElements.get_P0FiniteElement(grid);
elseif fem == "TH"
    # Taylor--Hood
    FE_velocity = FiniteElements.get_P2VectorFiniteElement(grid,use_FDgradients);
    FE_pressure = FiniteElements.get_P1FiniteElement(grid,use_FDgradients);
elseif fem == "MINI"
    # Taylor--Hood
    FE_velocity = FiniteElements.get_MINIFiniteElement(grid,use_FDgradients);
    FE_pressure = FiniteElements.get_P1FiniteElement(grid,use_FDgradients);
end    
ndofs_velocity = FE_velocity.ndofs;
ndofs_pressure = FE_pressure.ndofs;
ndofs[level] = ndofs_velocity + ndofs_pressure;
println("ndofs_velocity=",ndofs_velocity);
println("ndofs_pressure=",ndofs_pressure);
println("ndofs_total=",ndofs[level]);

# solve Stokes problem
val4coords = zeros(Base.eltype(grid.coords4nodes),ndofs[level]);
residual = solveStokesProblem!(val4coords,volume_data!(use_problem),exact_velocity!(use_problem),grid,FE_velocity,FE_pressure,FE_velocity.polynomial_order+f_order);
println("residual = " * string(residual));

# compute pressure best approximation
val4coords_pressureBA = zeros(Base.eltype(grid.coords4nodes),ndofs_pressure);
residual = computeBestApproximation!(val4coords_pressureBA,"L2",wrap_pressure,wrap_pressure,grid,FE_pressure,2)
println("residual = " * string(residual));

# compute velocity best approximation
val4coords_velocityBA = zeros(Base.eltype(grid.coords4nodes),ndofs_velocity);
residual = computeBestApproximation!(val4coords_velocityBA,"L2",exact_velocity!(use_problem),exact_velocity!(use_problem),grid,FE_velocity,4)
println("residual = " * string(residual));

# compute errors
integral4cells = zeros(size(grid.nodes4cells,1),1);
integrate!(integral4cells,eval_L2_interpolation_error!(wrap_pressure, val4coords[ndofs_velocity+1:end], FE_pressure), grid, error_order, 1);
L2error_pressure[level] = sqrt(abs(sum(integral4cells)));
println("L2_pressure_error_STOKES = " * string(L2error_pressure[level]));
integrate!(integral4cells,eval_L2_interpolation_error!(wrap_pressure, val4coords_pressureBA, FE_pressure), grid, error_order, 1);
L2error_pressureBA[level] = sqrt(abs(sum(integral4cells)));
println("L2_pressure_error_BA = " * string(L2error_pressureBA[level]));
integral4cells = zeros(size(grid.nodes4cells,1),2);
integrate!(integral4cells,eval_L2_interpolation_error!(exact_velocity!(use_problem), val4coords[1:ndofs_velocity], FE_velocity), grid, error_order, 2);
L2error_velocity[level] = sqrt(abs(sum(integral4cells[:])));
println("L2_velocity_error_STOKES = " * string(L2error_velocity[level]));
integrate!(integral4cells,eval_L2_interpolation_error!(exact_velocity!(use_problem), val4coords_velocityBA, FE_velocity), grid, error_order, 2);
L2error_velocityBA[level] = sqrt(abs(sum(integral4cells[:])));
println("L2_velocity_error_BA = " * string(L2error_velocityBA[level]));

# plot
# if (show_plots) && (level == maxlevel)
#     pygui(true)
#     if fem == "BR"
#         nnodes = size(grid.coords4nodes,1)
#         nfaces = size(grid.nodes4faces,1)
#         velo1_dofs = 1:nnodes;
#         velo2_dofs = nnodes+1:2*nnodes;
#     elseif fem == "TH"
#         velo1_dofs = 1:Int(ndofs_velocity / 2);
#         velo2_dofs = Int(ndofs_velocity / 2)+1:ndofs_velocity;
#     elseif fem == "MINI"
#         nnodes = size(grid.coords4nodes,1)
#         ncells = size(grid.nodes4cells,1);
#         velo1_dofs = 1:nnodes;
#         velo2_dofs = nnodes+ncells+1:2*nnodes+ncells;
#     end    
#     pressure_dofs = ndofs_velocity+1:ndofs[level];
#     
#     PyPlot.figure(1)
#     PyPlot.plot_trisurf(view(FE_velocity.coords4dofs,velo1_dofs,1),view(FE_velocity.coords4dofs,velo1_dofs,2),val4coords[velo1_dofs],cmap=get_cmap("ocean"))
#     PyPlot.title("Stokes Problem Solution - velocity component 1")
#     PyPlot.figure(2)
#     PyPlot.plot_trisurf(view(FE_velocity.coords4dofs,velo2_dofs,1),view(FE_velocity.coords4dofs,velo2_dofs,2),val4coords[velo2_dofs],cmap=get_cmap("ocean"))
#     PyPlot.title("Stokes Problem Solution - velocity component 2")
#     PyPlot.figure(3)
#     PyPlot.plot_trisurf(view(FE_pressure.coords4dofs,:,1),view(FE_pressure.coords4dofs,:,2),val4coords[pressure_dofs],cmap=get_cmap("ocean"))
#     PyPlot.title("Stokes Problem Solution - pressure")
#     PyPlot.figure(4)
#     PyPlot.plot_trisurf(view(FE_pressure.coords4dofs,:,1),view(FE_pressure.coords4dofs,:,2),val4coords_pressureBA,cmap=get_cmap("ocean"))
#     PyPlot.title("Stokes Problem Solution - pressure BA")
#     PyPlot.figure(5)
#     PyPlot.plot_trisurf(view(FE_velocity.coords4dofs,velo1_dofs,1),view(FE_velocity.coords4dofs,velo1_dofs,2),val4coords_velocityBA[velo1_dofs],cmap=get_cmap("ocean"))
#     PyPlot.title("Stokes Problem Solution - velocity BA component 1")
#     PyPlot.figure(6)
#     PyPlot.plot_trisurf(view(FE_velocity.coords4dofs,velo2_dofs,1),view(FE_velocity.coords4dofs,velo2_dofs,2),val4coords_velocityBA[velo2_dofs],cmap=get_cmap("ocean"))
#     PyPlot.title("Stokes Problem Solution - velocity BA component 2")
#     show()
# end
end # loop over levels

println("\n L2 pressure error");
show(L2error_pressure)
println("\n L2 pressure BA error");
show(L2error_pressureBA)
println("\n L2 velocity error");
show(L2error_velocity)
println("\n L2 velocity BA error");
show(L2error_velocityBA)

if (show_convergence_history)
    PyPlot.figure(7)
    PyPlot.loglog(ndofs,L2error_velocity,"-o")
    PyPlot.loglog(ndofs,L2error_pressure,"-o")
    PyPlot.loglog(ndofs,L2error_velocityBA,"-o")
    PyPlot.loglog(ndofs,L2error_pressureBA,"-o")
    PyPlot.loglog(ndofs,ndofs.^(-1/2),"--",color = "gray")
    PyPlot.loglog(ndofs,ndofs.^(-1),"--",color = "gray")
    PyPlot.loglog(ndofs,ndofs.^(-3/2),"--",color = "gray")
    PyPlot.legend(("L2 error velocity","L2 error pressure","L2 error velocity BA","L2 error pressure BA","O(h)","O(h^2)","O(h^3)"))
    PyPlot.title("Convergence history (fem=" * fem * " problem=" * use_problem * ")")
    ax = PyPlot.gca()
    ax.grid(true)
end    

    
end


main()
