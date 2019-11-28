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

function bary1_ref(xref)
    return xref[1];
end    

function bary2_ref(xref)
    return xref[2];
end    

function bary3_ref(xref)
    return xref[3];
end  

function line_bary1(x,grid::Grid.Mesh,cell)
    return sqrt(sum((grid.coords4nodes[grid.nodes4cells[cell,2],:] - x).^2)) / grid.volume4cells[cell];
end   

function line_bary2(x,grid::Grid.Mesh,cell)
    return sqrt(sum((grid.coords4nodes[grid.nodes4cells[cell,1],:] - x).^2)) / grid.volume4cells[cell];
end   

function line_bary1_grad!(result,x,grid,cell)
    @assert size(grid.coords4nodes,2) == 1 # todo: implement exact gradient for higher dimension line segments
    result[1] = -1 / grid.volume4cells[cell];
end   

function line_bary2_grad!(result,x,grid,cell)
    @assert size(grid.coords4nodes,2) == 1 # todo: implement exact gradient for higher dimension line segments
    result[1] = 1 / grid.volume4cells[cell];
end   

function triangle_bary1(x,grid::Grid.Mesh,cell)
    return (grid.coords4nodes[grid.nodes4cells[cell,2],1]*grid.coords4nodes[grid.nodes4cells[cell,3],2]
    - grid.coords4nodes[grid.nodes4cells[cell,3],1]*grid.coords4nodes[grid.nodes4cells[cell,2],2]
    + x[1]*(grid.coords4nodes[grid.nodes4cells[cell,2],2] - grid.coords4nodes[grid.nodes4cells[cell,3],2])
    + x[2]*(grid.coords4nodes[grid.nodes4cells[cell,3],1] - grid.coords4nodes[grid.nodes4cells[cell,2],1]))/(2*grid.volume4cells[cell])
end    


function triangle_bary2(x,grid::Grid.Mesh,cell)
    return (-grid.coords4nodes[grid.nodes4cells[cell,1],1]*grid.coords4nodes[grid.nodes4cells[cell,3],2]
    - grid.coords4nodes[grid.nodes4cells[cell,3],1]*grid.coords4nodes[grid.nodes4cells[cell,1],2]
    + x[1]*(grid.coords4nodes[grid.nodes4cells[cell,3],2] - grid.coords4nodes[grid.nodes4cells[cell,1],2])
    + x[2]*(grid.coords4nodes[grid.nodes4cells[cell,1],1] - grid.coords4nodes[grid.nodes4cells[cell,3],1]))/(2*grid.volume4cells[cell])
end    


function triangle_bary3(x,grid::Grid.Mesh,cell)
    return (grid.coords4nodes[grid.nodes4cells[cell,1],1]*grid.coords4nodes[grid.nodes4cells[cell,2],2]
    - grid.coords4nodes[grid.nodes4cells[cell,1],1]*grid.coords4nodes[grid.nodes4cells[cell,2],2]
    + x[1]*(grid.coords4nodes[grid.nodes4cells[cell,1],2] - grid.coords4nodes[grid.nodes4cells[cell,2],2])
    + x[2]*(grid.coords4nodes[grid.nodes4cells[cell,2],1] - grid.coords4nodes[grid.nodes4cells[cell,1],1]))/(2*grid.volume4cells[cell])
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


function FDgradient!(result,x,grid,cell,bfun::Function)
    f(x) = bfun(x,grid,cell);
    g = x -> ForwardDiff.gradient(f,x);
    result[:] = g(x);
end    

function triangle_bary1_FDgrad!(result,x,grid,cell)
    FDgradient!(result,x,grid,cell,triangle_bary1);
end

function triangle_bary2_FDgrad!(result,x,grid,cell)
    FDgradient!(result,x,grid,cell,triangle_bary2);
end

function triangle_bary3_FDgrad!(result,x,grid,cell)
    FDgradient!(result,x,grid,cell,triangle_bary3);
end

function line_bary1_FDgrad!(result,x,grid,cell)
    FDgradient!(result,x,grid,cell,line_bary1);
end

function line_bary2_FDgrad!(result,x,grid,cell)
    FDgradient!(result,x,grid,cell,line_bary2);
end

function get_P1FiniteElementFD(grid::Grid.Mesh)
    T = eltype(grid.coords4nodes)
    dofs4cells = grid.nodes4cells;
    ensure_nodes4faces!(grid);
    ensure_volume4cells!(grid);
    dofs4faces = grid.nodes4faces;
    coords4dof = grid.coords4nodes;
    
    # group basis functions
    celldim = size(grid.nodes4cells,2);
    if celldim == 3 # triangles
        bfun_ref = [bary1_ref,bary2_ref,bary3_ref];
        bfun = [triangle_bary1,triangle_bary2,triangle_bary3];
        bfun_grad! = [triangle_bary1_FDgrad!,triangle_bary2_FDgrad!,triangle_bary3_FDgrad!];
    elseif celldim == 2 # line segments
        bfun_ref = [bary1_ref,bary2_ref];
        bfun = [line_bary1,line_bary2];
        bfun_grad! = [line_bary1_FDgrad!,line_bary2_FDgrad!];
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
    
    
    # group basis functions
    celldim = size(grid.nodes4cells,2);
    if celldim == 3 # triangles
        bfun_ref = [bary1_ref,bary2_ref,bary3_ref];
        bfun = [triangle_bary1,triangle_bary2,triangle_bary3];
        bfun_grad! = [triangle_bary1_grad!,triangle_bary2_grad!,triangle_bary3_grad!];
    elseif celldim == 2 # line segments
        bfun_ref = [bary1_ref,bary2_ref];
        bfun = [line_bary1,line_bary2];
        bfun_grad! = [line_bary1_grad!,line_bary2_grad!];
    end    
    
    local_mass_matrix = (ones(T,celldim,celldim) + LinearAlgebra.I(celldim)) * 1 // ((celldim)*(celldim+1));
    return FiniteElement{T}(grid,dofs4cells,dofs4faces,coords4dof,bfun_ref,bfun,bfun_grad!,local_mass_matrix);
end


end # module
