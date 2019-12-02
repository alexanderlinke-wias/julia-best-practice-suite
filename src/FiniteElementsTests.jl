module FiniteElementsTests

using Grid
using FiniteElements

export FiniteElement

function TestP1()
    coords4nodes_init = [0.0 0.0;
                        1.0 0.0;
                        0.0 1.0];
    nodes4cells_init = zeros(Int64,1,3);
    nodes4cells_init[1,:] = [1 2 3];
               
    grid = Grid.Mesh{Float64}(coords4nodes_init,nodes4cells_init);
    
    FE = FiniteElements.get_P1FiniteElement(grid,false);
    
    x = [0 1//2]
    xref = [0 1//2 1//2]
    cellnr = 1
    println("\nbasis functions 1...");
    show(FE.bfun[1](x,grid,cellnr));
    println("\nbasis functions 2...");
    show(FE.bfun[2](x,grid,cellnr));
    println("\nbasis functions 3...");
    show(FE.bfun[3](x,grid,cellnr));
    println("\npartition of unity test...");
    sum = FE.bfun[1](x,grid,cellnr) + FE.bfun[2](x,grid,cellnr) + FE.bfun[3](x,grid,cellnr);
    show(sum);
    
    result1 = [0.0 0.0];
    result2 = [0.0 0.0];
    result3 = [0.0 0.0];
    println("\ngradient of basis function 1...");
    FE.bfun_grad![1](result1,x,xref,grid,1);
    show(result1);
    println("\ngradient of basis function 2...");
    FE.bfun_grad![2](result2,x,xref,grid,1);
    show(result2);
    println("\ngradient of basis function 3...");
    FE.bfun_grad![3](result3,x,xref,grid,1);
    show(result3);
    println("\npartition of unity test...");
    sum2 = result1 + result2 + result3;
    show(sum2);
    return (sum2[1] <= eps(1.0)) && (sum2[2] <= eps(1.0)) && (sum - 1 <= eps(1.0))
end


function TestP2()
    coords4nodes_init = [0.0 0.0;
                        2.0 0.1;
                        0.0 1.0];
    nodes4cells_init = zeros(Int64,1,3);
    nodes4cells_init[1,:] = [1 2 3];
               
    grid = Grid.Mesh{Float64}(coords4nodes_init,nodes4cells_init);
    
    FE = FiniteElements.get_P2FiniteElement(grid,false);
    FE2 = FiniteElements.get_P2FiniteElement(grid,true);
    
    x = [0.0 0.0]
    xref = [1.0 0.0 0]
    println("\nbasis functions 1...");
    show(FE.bfun[1](x,grid,1));
    println("\nbasis functions 2...");
    show(FE.bfun[2](x,grid,1));
    println("\nbasis functions 3...");
    show(FE.bfun[3](x,grid,1));
    println("\nbasis functions 4...");
    show(FE.bfun[4](x,grid,1));
    println("\nbasis functions 5...");
    show(FE.bfun[5](x,grid,1));
    println("\nbasis functions 6...");
    show(FE.bfun[6](x,grid,1));
    println("\npartition of unity test...");
    sum = FE.bfun[1](x,grid,1) + FE.bfun[2](x,grid,1) + FE.bfun[3](x,grid,1) + FE.bfun[4](x,grid,1) + FE.bfun[5](x,grid,1) + FE.bfun[6](x,grid,1);
    show(sum);
    
    result1 = [0.0 0.0];
    result2 = [0.0 0.0];
    result3 = [0.0 0.0];
    result4 = [0.0 0.0];
    result5 = [0.0 0.0];
    result6 = [0.0 0.0];
    result12 = [0.0 0.0];
    result22 = [0.0 0.0];
    result32 = [0.0 0.0];
    result42 = [0.0 0.0];
    result52 = [0.0 0.0];
    result62 = [0.0 0.0];
    println("\ngradient of basis function 1...");
    FE.bfun_grad![1](result1,x,xref,grid,1);
    FE2.bfun_grad![1](result12,x,xref,grid,1);
    show([result1; result12]);
    println("\ngradient of basis function 2...");
    FE.bfun_grad![2](result2,x,xref,grid,1);
    FE2.bfun_grad![2](result22,x,xref,grid,1);
    show([result2; result22]);
    println("\ngradient of basis function 3...");
    FE.bfun_grad![3](result3,x,xref,grid,1);
    FE2.bfun_grad![3](result32,x,xref,grid,1);
    show([result3; result32]);
    println("\ngradient of basis function 4...");
    FE.bfun_grad![4](result4,x,xref,grid,1);
    FE2.bfun_grad![4](result42,x,xref,grid,1);
    show([result4; result42]);
    println("\ngradient of basis function 5...");
    FE.bfun_grad![5](result5,x,xref,grid,1);
    FE2.bfun_grad![5](result52,x,xref,grid,1);
    show([result5; result52]);
    println("\ngradient of basis function 6...");
    FE.bfun_grad![6](result6,x,xref,grid,1);
    FE2.bfun_grad![6](result62,x,xref,grid,1);
    show([result6; result62]);
    println("\npartition of unity test...");
    sum2 = result1 + result2 + result3 + result4 + result5 + result6;
    show(sum2);
    return (sum2[1] <= eps(1.0)) && (sum2[2] <= eps(1.0)) && (sum - 1 <= eps(1.0))
end


end # module
