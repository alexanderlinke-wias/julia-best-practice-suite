module QuadratureTests

using LinearAlgebra
using Quadrature
using Grid

export TestExactness

function load_test_grid(nrefinements::Int = 0)
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
    return Grid.Triangulation(coords4nodes_init,nodes4cells_init,nrefinements);
end

function get_exact_function!(result,x,order)
    for i in eachindex(result)
        @inbounds result[i] = x[i,1].^order + 2 .* x[i,2].^(order-1);
    end
end

function get_exact_integral(order::Int)
    return 1.0/(order+1.0) + 2.0/order;
end    
    


function TestExactness(order::Int)
    # load unit square grid
    T = load_test_grid();
    # load polynomial of given order and its exact integral
    test_function!(result,x,xref = Nothing,cellIndex = Nothing) = get_exact_function!(result,x,order);
    exact_integral = get_exact_integral(order)
    
    # integrate
    integral4cells = zeros(size(T.nodes4cells,1),1);
    integrate2!(integral4cells,test_function!,T,order);
    integral = sum(integral4cells);
    println("Testing integration of x^" * string(order) * " +2y^"* string(order-1));
    println("expected integral = " * string(exact_integral));
    println("computed integral = " * string(integral));
    println("error = " * string(integral-exact_integral));
    return isapprox(integral,exact_integral);
end

function TimeIntegrations(order::Int)
    # load unit square grid
    T = load_test_grid(4);
    # load polynomial of given order and its exact integral
    test_function!(result,x,xref = Nothing,cellIndex = Nothing) = get_exact_function!(result,x,order);
    exact_integral = get_exact_integral(order)
    integral4cells = zeros(size(T.nodes4cells,1),1);
    @time integrate!(integral4cells,test_function!,T,order);
    @time integrate!(integral4cells,test_function!,T,order);
    println("integrate1 = ",sum(integral4cells));
    @time integrate2!(integral4cells,test_function!,T,order);
    @time integrate2!(integral4cells,test_function!,T,order);
    println("integrate2 = ",sum(integral4cells));
end


end
