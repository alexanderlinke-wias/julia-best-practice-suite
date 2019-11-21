module FiniteElementsTests

using Grid
using FiniteElements


function TestP1()
    coords4nodes_init = [0.0 0.0;
                        1.0 0.0;
                        1.0 1.0;
                        0.1 1.0;
                        0.5 0.6];
    nodes4cells_init = [1 2 5;
                        2 3 5;
                        3 4 5;
                        4 1 5];
               
    grid = Grid.Mesh(coords4nodes_init,nodes4cells_init,0);
    ensure_volume4cells!(grid);
    
    FE = FiniteElements.get_P1FiniteElement(grid);
    
    x = [0.25 0.25]
    println("\nbasis functions 1...");
    show(FE.bfun[1](x,grid,1));
    println("\nbasis functions 2...");
    show(FE.bfun[2](x,grid,1));
    println("\nbasis functions 3...");
    show(FE.bfun[3](x,grid,1));
    println("\npartition of unity test...");
    sum = FE.bfun[1](x,grid,1) + FE.bfun[2](x,grid,1) + FE.bfun[3](x,grid,1);
    show(sum);
    
    result1 = [0.0 0.0];
    result2 = [0.0 0.0];
    result3 = [0.0 0.0];
    println("\ngradient of basis function 1...");
    FE.bfun_grad![1](result1,x,grid,1);
    show(result1);
    println("\ngradient of basis function 2...");
    FE.bfun_grad![2](result2,x,grid,1);
    show(result2);
    println("\ngradient of basis function 3...");
    FE.bfun_grad![3](result3,x,grid,1);
    show(result3);
    println("\npartition of unity test...");
    sum2 = result1 + result2 + result3;
    show(sum2);
    return (sum2[1] <= eps(1.0)) && (sum2[2] <= eps(1.0)) && (sum - 1 <= eps(1.0))
end


end # module
