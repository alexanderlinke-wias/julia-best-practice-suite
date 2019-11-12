module Quadrature

using LinearAlgebra
using Grid

export integrate!

# computes quadrature points and weights by Stroud Conical Product rule
function get_generic_quadrature_Stroud(order::Int)
    ngpts::Int = ceil((order+1)/2);
    
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
function integrate!(integral4cells::Array, integrand!::Function, T::Grid.Triangulation, order::Int, resultdim = 1)
    ncells::Int = size(T.nodes4cells,1);
    
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
    nqp::Int = size(xref,1);
    
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




### TESTS ###
function load_test_grid()
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
    return Grid.Triangulation(coords4nodes_init,nodes4cells_init);
end

function QuadratureTest(order::Int)
    function test_function!(result,x,xref)
        result[:] = @views x[:,1].^order + 2 .* x[:,2].^(order-1);
    end
    exact_integral = 1.0/(order+1.0) + 2.0/order;

    T = load_test_grid();
    integral4cells = zeros(size(T.nodes4cells,1),1);
    integrate!(integral4cells,test_function!,T,order);
    integral = abs(sum(integral4cells));
    println("Testing integration of x^" * string(order) * " +2y^"* string(order-1));
    println("expected integral = " * string(exact_integral));
    println("computed integral = " * string(integral));
    println("error = " * string(integral-exact_integral));
    return isapprox(integral,exact_integral);
end


end
