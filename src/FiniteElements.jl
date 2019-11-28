module FiniteElements

using Grid
using LinearAlgebra
using ForwardDiff



struct FiniteElement{T <: Real}
    grid::Grid.Mesh;
    dofs4cells::Array{Int64,2};
    dofs4faces::Array{Int64,2};
    coords4dofs::Array{T,2};
    bfun_ref::Array{Function,1};
    bfun::Array{Function,1};
    bfun_grad!::Array{Function,1};
    local_mass_matrix::Array{T,2};
end

function bary(j::Int)
    function closure(xref)
        return xref[j];
    end    
end

function P1FEFunctions1D(j)
    function closure(x,grid::Grid.Mesh,cell)
        return sqrt(sum((grid.coords4nodes[grid.nodes4cells[cell,j],:] - x).^2)) / grid.volume4cells[cell];        
    end
end

function P1FEFunctions2D(j)
    if (j == 3)
        index1 = 1;
    else
        index1 = j + 1;
    end    
    if (j == 1)
        index2 = 3;
    else
        index2 = j - 1;
    end   
    function closure(x,grid::Grid.Mesh,cell)
        return (grid.coords4nodes[grid.nodes4cells[cell,index1],1]*grid.coords4nodes[grid.nodes4cells[cell,index2],2]
    - grid.coords4nodes[grid.nodes4cells[cell,index2],1]*grid.coords4nodes[grid.nodes4cells[cell,index2],2]
    + x[1]*(grid.coords4nodes[grid.nodes4cells[cell,index1],2] - grid.coords4nodes[grid.nodes4cells[cell,index2],2])
    + x[2]*(grid.coords4nodes[grid.nodes4cells[cell,index2],1] - grid.coords4nodes[grid.nodes4cells[cell,index1],1]))/(2*grid.volume4cells[cell])
    end
end

function line_bary1_grad!(result,x,grid,cell)
    @assert size(grid.coords4nodes,2) == 1 # todo: implement exact gradient for higher dimension line segments
    result[1] = -1 / grid.volume4cells[cell];
end   

function line_bary2_grad!(result,x,grid,cell)
    @assert size(grid.coords4nodes,2) == 1 # todo: implement exact gradient for higher dimension line segments
    result[1] = 1 / grid.volume4cells[cell];
end   

function triangle_bary1_grad!(result,x,grid,cell)
    result[1] = grid.coords4nodes[grid.nodes4cells[cell,2],2] - grid.coords4nodes[grid.nodes4cells[cell,3],2];
    result[2] = grid.coords4nodes[grid.nodes4cells[cell,3],1] - grid.coords4nodes[grid.nodes4cells[cell,2],1];
    result[:] /= (2*grid.volume4cells[cell]);
end

function triangle_bary2_grad!(result,x,grid,cell)
    result[1] = grid.coords4nodes[grid.nodes4cells[cell,3],2] - grid.coords4nodes[grid.nodes4cells[cell,1],2];
    result[2] = grid.coords4nodes[grid.nodes4cells[cell,1],1] - grid.coords4nodes[grid.nodes4cells[cell,3],1];
    result[:] /= (2*grid.volume4cells[cell]);
end

function triangle_bary3_grad!(result,x,grid,cell)
    result[1] = grid.coords4nodes[grid.nodes4cells[cell,1],2] - grid.coords4nodes[grid.nodes4cells[cell,2],2];
    result[2] = grid.coords4nodes[grid.nodes4cells[cell,2],1] - grid.coords4nodes[grid.nodes4cells[cell,1],1];
    result[:] /= (2*grid.volume4cells[cell]);
end


function FDgradient!(bfun::Function)
    function closure(result,x,grid,cell)
        f(x) = bfun(x,grid,cell);
        g = x -> ForwardDiff.gradient(f,x);
        result[:] = g(x);
    end    
end    


function get_P1FiniteElementFD(grid::Grid.Mesh)
    T = eltype(grid.coords4nodes)
    dofs4cells = grid.nodes4cells;
    ensure_nodes4faces!(grid);
    ensure_volume4cells!(grid);
    dofs4faces = grid.nodes4faces;
    coords4dof = grid.coords4nodes;
    println("Initialising FiniteElement with ForwardDiff gradients...");
    
    
    # group basis functions
    celldim = size(grid.nodes4cells,2);
    if celldim == 3 # triangles
        bfun_ref = [bary(1),
                    bary(2),
                    bary(3)];
        bfun = [P1FEFunctions2D(1),
                P1FEFunctions2D(2),
                P1FEFunctions2D(3)];
        bfun_grad! = [FDgradient!(P1FEFunctions2D(1)),
                      FDgradient!(P1FEFunctions2D(2)),
                      FDgradient!(P1FEFunctions2D(3))];
    elseif celldim == 2 # line segments
        bfun_ref = [bary(1),
                    bary(2)];
        bfun = [P1FEFunctions1D(1),
                P1FEFunctions1D(2)];
        bfun_grad! = [FDgradient!(P1FEFunctions1D(1)),
                      FDgradient!(P1FEFunctions1D(2))];
    end    
    
    local_mass_matrix = (ones(T,celldim,celldim) + LinearAlgebra.I(celldim)) * 1 // ((celldim)*(celldim+1));
    return FiniteElement{T}(grid,dofs4cells,dofs4faces,coords4dof,bfun_ref,bfun,bfun_grad!,local_mass_matrix);
end

function get_P1FiniteElement(grid::Grid.Mesh)
    T = eltype(grid.coords4nodes)
    dofs4cells = grid.nodes4cells;
    ensure_nodes4faces!(grid);
    dofs4faces = grid.nodes4faces;
    coords4dof = grid.coords4nodes;
    ensure_volume4cells!(grid);
    println("Initialising FiniteElement with exact gradients...");
    
    
    # group basis functions
    celldim = size(grid.nodes4cells,2);
    if celldim == 3 # triangles
        bfun_ref = [bary(1),
                    bary(2),
                    bary(3)];
        bfun = [P1FEFunctions2D(1),
                P1FEFunctions2D(2),
                P1FEFunctions2D(3)];
        bfun_grad! = [triangle_bary1_grad!,
                      triangle_bary2_grad!,
                      triangle_bary3_grad!];
    elseif celldim == 2 # line segments
        bfun_ref = [bary(1),
                    bary(2)];
        bfun = [P1FEFunctions1D(1),
                P1FEFunctions1D(2)];
        bfun_grad! = [line_bary1_grad!,
                      line_bary2_grad!];
    end    
    
    local_mass_matrix = (ones(T,celldim,celldim) + LinearAlgebra.I(celldim)) * 1 // ((celldim)*(celldim+1));
    return FiniteElement{T}(grid,dofs4cells,dofs4faces,coords4dof,bfun_ref,bfun,bfun_grad!,local_mass_matrix);
end


end # module
