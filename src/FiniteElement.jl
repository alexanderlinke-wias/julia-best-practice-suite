module FiniteElement

using Grid
using LinearAlgebra
using ForwardDiff

function triangle_bary1(x,grid,cell)
    result = 0.0;
    A = zeros(eltype(x),3,3);
    A[1,:] = [1 1 1];
    A[2,1] = x[1];
    A[2,2] = grid.coords4nodes[grid.nodes4cells[cell,2],1];
    A[2,3] = grid.coords4nodes[grid.nodes4cells[cell,3],1];
    A[3,1] = x[2];
    A[3,2] = grid.coords4nodes[grid.nodes4cells[cell,2],2];
    A[3,3] = grid.coords4nodes[grid.nodes4cells[cell,3],2];
    return det(A)/(2*grid.volume4cells[cell])
end    


function triangle_bary2(x,grid,cell)
    result = 0.0;
    A = zeros(eltype(x),3,3);
    A[1,:] = [1 1 1];
    A[2,1] = grid.coords4nodes[grid.nodes4cells[cell,1],1];
    A[2,2] = x[1];
    A[2,3] = grid.coords4nodes[grid.nodes4cells[cell,3],1];
    A[3,1] = grid.coords4nodes[grid.nodes4cells[cell,1],2];
    A[3,2] = x[2];
    A[3,3] = grid.coords4nodes[grid.nodes4cells[cell,3],2];
    return det(A)/(2*grid.volume4cells[cell])
end    


function triangle_bary3(x,grid,cell)
    result = 0.0;
    A = zeros(eltype(x),3,3);
    A[1,:] = [1 1 1];
    A[2,1] = grid.coords4nodes[grid.nodes4cells[cell,1],1];
    A[2,2] = grid.coords4nodes[grid.nodes4cells[cell,2],1];
    A[2,3] = x[1];
    A[3,1] = grid.coords4nodes[grid.nodes4cells[cell,1],2];
    A[3,2] = grid.coords4nodes[grid.nodes4cells[cell,2],2];
    A[3,3] = x[2];
    return det(A)/(2*grid.volume4cells[cell])
end    

function gradient!(result,x,f::Function)
    g = x -> ForwardDiff.gradient(f,x);
    result[:] = g(x);
end    


function Test()
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
    
    
    result = [0.0 0.0];
    f1(x) = triangle_bary1!(result,x,grid,1);
    gradient!(result,[0.25 0.25],f1);
    show(result);
    f2(x) = triangle_bary2!(result,x,grid,1);
    gradient!(result,[0.25 0.25],f2);
    show(result);
    f3(x) = triangle_bary3!(result,x,grid,1);
    gradient!(result,[0.25 0.25],f3);
    show(result);
    
end


end # module
