module QuadratureTests

using LinearAlgebra
using Quadrature
using Grid

export TestExactness

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


function TestExactness(order::Int)
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
