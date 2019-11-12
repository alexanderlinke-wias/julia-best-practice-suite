
using SparseArrays
using LinearAlgebra
ENV["MPLBACKEND"]="tkagg"
using PyPlot
using BenchmarkTools
include("grid.jl")
#using .Grid

function accumarray!(A,subs, val, sz=(maximum(subs),))
      for i = 1:length(val)
          @inbounds A[subs[i]] += val[i]
      end
  end

# computes quadrature points and weights by Stroud Conical Product rule
function get_generic_quadrature_Stroud(order)
    ngpts::Int64 = ceil((order+1)/2);
    
    # compute 1D Gauss points on interval [-1,1] and weights
    gamma = (1 : ngpts-1) ./ sqrt.(4 .* (1 : ngpts-1).^2 .- ones(ngpts-1,1) );
    F = eigen(diagm(1 => gamma[:], -1 => gamma[:]));
    r = F.values;
    a = 2*F.vectors[1,:].^2;
    
    # compute 1D Gauss-Jacobi Points for Intervall [-1,1] and weights
    delta = -1 ./ (4 .* (1 : ngpts).^2 .- ones(ngpts,1));
    gamma = sqrt.((2 : ngpts) .* (1 : ngpts-1)) ./ (2 .* (2 : ngpts) .- ones(ngpts-1,1));
    F = eigen(diagm(0 => delta[:], 1 => gamma[:], -1 => gamma[:]));
    s = F.values;
    b = 2*F.vectors[1,:].^2;
    
    # transform to interval [0,1]
    r = .5 .* r .+ .5;
    s = .5 .* s .+ .5;
    a = .5 .* a';
    b = .5 .* b';
    
    # apply conical product rule
    # xref[:,[1 2]] = [ s_j , r_i(1-s_j) ] 
    # xref[:,3] = 1 - xref[:,1] - xref[:,2]
    # w = a_i*b_j
    s = repeat(s',ngpts,1)[:];
    r = repeat(r,ngpts,1);
    xref = s*[1 0 -1] - (r.*(s.-1))*[0 1 -1] + ones(length(s))*[0 0 1];
    w = a'*b;
    
    return xref,w
end

# integrate a smooth function over the triangulation with arbitrary order
function integrate!(integral4cells,integrand!::Function,T::Grid.Triangulation,order::Signed,resultdim = 1)
    ncells::Int64 = size(T.nodes4cells,1);
    
    # get quadrature point and weights
    if order <= 1 # cell midpoint rule
        xref = [1/3 1/3 1/3];
        w = [1.0];
    elseif order == 2 # face midpoint rule
        xref = [0.5 0.5 0.0;
                0.0 0.5 0.5;
                0.5 0.0 0.5];
        w = [1/3 1/3 1/3];        
    else
        xref, w = get_generic_quadrature_Stroud(order)
    end    
    nqp::Int64 = size(xref,1);
    
    # compute area4cells
    Grid.ensure_area4cells!(T);
    
    # loop over quadrature points
    fill!(integral4cells,0.0);
    x = zeros(Float64,ncells,2);
    result = zeros(Float64,ncells,resultdim);
    for qp= 1 : nqp
        # map xref to x in each triangle
        x = ( xref[qp,1] .* view(T.coords4nodes,view(T.nodes4cells,:,1),:)
            + xref[qp,2] .* view(T.coords4nodes,view(T.nodes4cells,:,2),:)
            + xref[qp,3] .* view(T.coords4nodes,view(T.nodes4cells,:,3),:));
    
        # evaluate integrand multiply with quadrature weights
        integrand!(result,x,xref[qp,:])
        integral4cells .+= result .* repeat(T.area4cells,1,resultdim) .* w[qp];
    end
end


function global_mass_matrix(T::Grid.Triangulation)
    ncells::Int64 = size(T.nodes4cells,1);
    nnodes::Int64 = size(T.coords4nodes,1);
    I = repeat(T.nodes4cells',3)[:];
    J = repeat(T.nodes4cells'[:]',3)[:];
    V = repeat([2 1 1 1 2 1 1 1 2]'[:] ./ 12,ncells)[:].*repeat(T.area4cells',9)[:];
    A = sparse(I,J,V,nnodes,nnodes);
end

function rhs_integrand!(result::Array,x::Array,xref::Array,f!::Function)
    f!(view(result,:,1),x)
    for j=3:-1:1
        result[:,j] = view(result,:,1) .* xref[j];
    end
end



function computeP1BestApproximation!(val4coords::Array,which_norm::String ,volume_data!::Function,boundary_data::Function,T::Grid.Triangulation,quadrature_order::Signed)
    ncells::Int64 = size(T.nodes4cells,1);
    nnodes::Int64 = size(T.coords4nodes,1);
    
    Grid.ensure_area4cells!(T);
    Grid.ensure_nodes4faces!(T);
    Grid.ensure_faces4cells!(T);
    Grid.ensure_bfaces!(T);
    
    # compute global mass matrix
    println("mass matrix")
    if which_norm == "L2"
        @time A = global_mass_matrix(T);
    end    
    
    # compute right-hand side vector
    println("integrate rhs");
    wrapped_integrand!(result,x,xref) = rhs_integrand!(result,x,xref,volume_data!);
    integral4cells = zeros(Float64,ncells,3);
    @time integrate!(integral4cells,wrapped_integrand!,T,quadrature_order,3);
    
    println("accumarray");
    b = zeros(Float64,nnodes);
    @time accumarray!(b,T.nodes4cells,integral4cells,nnodes)
    
    # find boundary nodes
    bnodes = unique(T.nodes4faces[T.bfaces,:]);
    dofs = setdiff(1:nnodes,bnodes);
    
    # solve
    println("solve");
    fill!(val4coords,0.0)
    @inbounds val4coords[bnodes] = boundary_data(view(T.coords4nodes,bnodes,:),0);
    b -= A*val4coords;
    
    @time val4coords[dofs] = A[dofs,dofs]\b[dofs];
end




function computeP1Interpolation!(val4coords::Array,source_function!::Function,T::Grid.Triangulation)
    source_function!(val4coords,T.coords4nodes);
end



function volume_data!(result,x)
    result[:] = @views x[:,1] .* (1 .- x[:,1]) .* x[:,2];
end

function eval_interpolation_error!(result,x,xref,exact_function!,coeffs_interpolation,dofs_interpolation)
    exact_function!(view(result,:,1),x);
    result[:] -= sum(coeffs_interpolation[dofs_interpolation] .* repeat(xref[:]',size(dofs_interpolation,1)),dims=2);
end




function main()

# define problem data
volume_data(x,xref) = x[:,1] .* (1 .- x[:,1]) .* x[:,2];
volume_data_views(x,xref) = view(x,:,1).* (1 .- view(x,:,1)) .* view(x,:,2);
boundary_data(x,xref) = volume_data(x,xref);

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
               
println("Loading grid...");
@time T = Grid.Triangulation(coords4nodes_init,nodes4cells_init,5);
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
println("Computing P1 Bestapproximation...");
val4coords2 = zeros(size(T.coords4nodes,1));
@time computeP1BestApproximation!(val4coords2,"L2",volume_data!,boundary_data,T,4);
@time computeP1BestApproximation!(val4coords2,"L2",volume_data!,boundary_data,T,4);


# compute interpolation error and bestapproximation error
wrapped_interpolation_error_integrand!(result,x,xref) = eval_interpolation_error!(result,x,xref,volume_data!,val4coords,T.nodes4cells);
wrapped_bestapproximation_error_integrand!(result,x,xref) = eval_interpolation_error!(result,x,xref,volume_data!,val4coords2,T.nodes4cells);

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
@time integrate!(integral4cells,wrapped_bestapproximation_error_integrand!,T,2);
println("bestapprox_error(integrate(order=4)) = " * string(sum(integral4cells)));


# plot interpolation and bestapproximation
pygui(true)
PyPlot.figure(1)
PyPlot.plot_trisurf(view(T.coords4nodes,:,1),view(T.coords4nodes,:,2),val4coords,cmap=get_cmap("ocean"))
PyPlot.title("Interpolation")
PyPlot.figure(2)
PyPlot.plot_trisurf(view(T.coords4nodes,:,1),view(T.coords4nodes,:,2),val4coords2,cmap=get_cmap("ocean"))
PyPlot.title("Bestapproximation")
show()
end




main()
