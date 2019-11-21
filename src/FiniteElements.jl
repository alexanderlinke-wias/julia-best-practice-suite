module FiniteElements

using Grid
using LinearAlgebra
using ForwardDiff

function triangle_bary1_ref(xref)
    return xref[1];
end    


function triangle_bary2_ref(xref)
    return xref[2];
end    


function triangle_bary3_ref(xref)
    return xref[3];
end  

function triangle_bary1(x,grid::Grid.Mesh,cell)
    return (grid.coords4nodes[grid.nodes4cells[cell,2],1]*grid.coords4nodes[grid.nodes4cells[cell,3],2]
    - grid.coords4nodes[grid.nodes4cells[cell,3],1]*grid.coords4nodes[grid.nodes4cells[cell,2],2]
    + x[1]*(grid.coords4nodes[grid.nodes4cells[cell,2],2]- grid.coords4nodes[grid.nodes4cells[cell,3],2])
    + x[2]*(grid.coords4nodes[grid.nodes4cells[cell,3],1] - grid.coords4nodes[grid.nodes4cells[cell,2],1]))/(2*grid.volume4cells[cell])
end    


function triangle_bary2(x,grid::Grid.Mesh,cell)
    return (-grid.coords4nodes[grid.nodes4cells[cell,1],1]*grid.coords4nodes[grid.nodes4cells[cell,3],2]
    - grid.coords4nodes[grid.nodes4cells[cell,3],1]*grid.coords4nodes[grid.nodes4cells[cell,1],2]
    + x[1]*(grid.coords4nodes[grid.nodes4cells[cell,3],2]- grid.coords4nodes[grid.nodes4cells[cell,1],2])
    + x[2]*(grid.coords4nodes[grid.nodes4cells[cell,1],1] - grid.coords4nodes[grid.nodes4cells[cell,3],1]))/(2*grid.volume4cells[cell])
end    


function triangle_bary3(x,grid::Grid.Mesh,cell)
    return (grid.coords4nodes[grid.nodes4cells[cell,1],1]*grid.coords4nodes[grid.nodes4cells[cell,2],2]
    - grid.coords4nodes[grid.nodes4cells[cell,1],1]*grid.coords4nodes[grid.nodes4cells[cell,2],2]
    + x[1]*(grid.coords4nodes[grid.nodes4cells[cell,1],2]- grid.coords4nodes[grid.nodes4cells[cell,2],2])
    + x[2]*(grid.coords4nodes[grid.nodes4cells[cell,2],1] - grid.coords4nodes[grid.nodes4cells[cell,1],1]))/(2*grid.volume4cells[cell])
end    

function FDgradient!(result,x,grid,cell,bfun::Function)
    f(x) = bfun(x,grid,cell);
    g = x -> ForwardDiff.gradient(f,x);
    result[:] = g(x);
end    

function triangle_bary1_grad!(result,x,grid,cell)
    FDgradient!(result,x,grid,cell,triangle_bary1);
end

function triangle_bary2_grad!(result,x,grid,cell)
    FDgradient!(result,x,grid,cell,triangle_bary2);
end

function triangle_bary3_grad!(result,x,grid,cell)
    FDgradient!(result,x,grid,cell,triangle_bary3);
end


struct FiniteElement
    grid::Grid.Mesh;
    dofs4e::Array{Int64,2};
    bfun_ref::Array{Function,1};
    bfun::Array{Function,1};
    bfun_grad!::Array{Function,1};
end

function get_P1FiniteElement(grid::Grid.Mesh)
    dofs4e = grid.nodes4cells;
    bfun_ref = [triangle_bary1_ref,triangle_bary2_ref,triangle_bary3_ref];
    bfun = [triangle_bary1,triangle_bary2,triangle_bary3];
    bfun_grad! = [triangle_bary1_grad!,triangle_bary2_grad!,triangle_bary3_grad!];
    return FiniteElement(grid,dofs4e,bfun_ref,bfun,bfun_grad!);
end



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
    
    FE = get_P1FiniteElement(grid);
    
    x = [0.25 0.25]
    println("\nbasis functions...");
    show(FE.bfun[1](x,grid,1));
    show(FE.bfun[2](x,grid,1));
    show(FE.bfun[3](x,grid,1));
    println("\npartition of unity test...");
    show(FE.bfun[1](x,grid,1) + FE.bfun[2](x,grid,1) + FE.bfun[3](x,grid,1) );
    
    result1 = [0.0 0.0];
    result2 = [0.0 0.0];
    result3 = [0.0 0.0];
    println("\ngradients...");
    FE.bfun_grad![1](result1,x,grid,1);
    show(result1);
    FE.bfun_grad![2](result2,x,grid,1);
    show(result2);
    FE.bfun_grad![3](result3,x,grid,1);
    show(result3);
    println("\npartition of unity test...");
    show(result1 + result2 + result3);
    
end


end # module
