module FiniteElements

using Grid
using LinearAlgebra
using ForwardDiff



struct FiniteElement{T <: Real}
    grid::Grid.Mesh;
    polynomial_order::Int;
    dofs4cells::Array{Int64,2};
    dofs4faces::Array{Int64,2};
    coords4dofs::Array{T,2};
    bfun_ref::Array{Function,1};
    bfun::Array{Function,1};
    bfun_grad!::Array{Function,1};
    local_mass_matrix::Array{T,2};
end

# wrapper for P1 partition of unity
function bary(j::Int)
    function closure(xref)
        return xref[j];
    end    
end

# wrapper for CR partition of unity
function CRbary(j::Int)
    if j == 1
        index = 3
    else
        index = j - 1
    end    
    function closure(xref)
        return 1-2*xref[index];
    end    
end

# wrapper for P2 partition of unity
function P2bary(j::Int)
    if j <= 3 # quadratic nodal basis functions
        return function closure1(xref)
                  return 2*xref[j]*(xref[j] - 1 // 2);
               end
    else # face bubbles
        if (j == 6)
            index1 = 1;
        else
            index1 = j - 3 + 1;
        end    
        index2 = j-3
        return function closure2(xref)
                  return 4*xref[index1]*xref[index2];
               end
    end    
end

# wrapper for P1 basis functions on a line
function P1FEFunctions1D(j)
    function closure(x,grid::Grid.Mesh,cell)
        return sqrt(sum((grid.coords4nodes[grid.nodes4cells[cell,j],:] - x).^2)) / grid.volume4cells[cell];        
    end
end

# wrapper for P2 basis functions on a line
function P2FEFunctions1D(j)
    if j <= 2 # quadratic nodal basis functions
        return function closure1(x,grid::Grid.Mesh,cell)
                  return 2*P1FEFunctions1D(j)(x,grid,cell).^2 - P1FEFunctions1D(j)(x,grid,cell);
               end
    else # interval cell bubble
        return function closure2(x,grid::Grid.Mesh,cell)
                  return 4*P1FEFunctions1D(1)(x,grid,cell)*P1FEFunctions1D(2)(x,grid,cell);
               end
    end           
end

# wrapper for P2 basis functions on a triangle
function P2FEFunctions2D(j)
    if j <= 3 # quadratic nodal basis functions
        return function closure1(x,grid::Grid.Mesh,cell)
                  return 2*P1FEFunctions2D(j)(x,grid,cell).^2 - P1FEFunctions2D(j)(x,grid,cell);
               end
    else # face bubbles
        if (j == 6)
            index1 = 1;
        else
            index1 = j - 3 + 1;
        end    
        index2 = j-3
        return function closure2(x,grid::Grid.Mesh,cell)
                  return 4*P1FEFunctions2D(index1)(x,grid,cell)*P1FEFunctions2D(index2)(x,grid,cell);
               end
    end           
end

# wrapper for P1 basis functions on a triangle
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
    - grid.coords4nodes[grid.nodes4cells[cell,index2],1]*grid.coords4nodes[grid.nodes4cells[cell,index1],2]
    + x[1]*(grid.coords4nodes[grid.nodes4cells[cell,index1],2] - grid.coords4nodes[grid.nodes4cells[cell,index2],2])
    + x[2]*(grid.coords4nodes[grid.nodes4cells[cell,index2],1] - grid.coords4nodes[grid.nodes4cells[cell,index1],1]))/(2*grid.volume4cells[cell])
    end
end


# wrapper for P1 basis functions on a triangle
function CRFEFunctions2D(j)
    if (j == 1)
        index = 3;
    else
        index = j - 1;
    end   
    function closure(x,grid::Grid.Mesh,cell)
        return 1 - 2*P1FEFunctions2D(index)(x,grid,cell)
    end
end


# the two exact gradients of the P1 basis functions on a line
function line_bary1_grad!(result,x,xref,grid,cell)
    @assert size(grid.coords4nodes,2) == 1 # todo: implement exact gradient for higher dimension line segments
    result[1] = -1 / grid.volume4cells[cell];
end   
function line_bary2_grad!(result,x,xref,grid,cell)
    @assert size(grid.coords4nodes,2) == 1 # todo: implement exact gradient for higher dimension line segments
    result[1] = 1 / grid.volume4cells[cell];
end   


# the three exact gradients of the P2 basis functions on a line
function line_P2_1_grad!(result,x,xref,grid,cell)
    line_bary1_grad!(result,x,xref,grid,cell);
    result[:] .*= (4*bary(1)(xref)-1);
end   
function line_P2_2_grad!(result,x,xref,grid,cell)
    line_bary2_grad!(result,x,xref,grid,cell);
    result[:] .*= (4*bary(1)(xref)-1);
end   
function line_P2_3_grad!(result,x,xref,grid,cell)
    temp = zeros(eltype(result),length(result));
    line_bary1_grad!(temp,x,xref,grid,cell)
    line_bary2_grad!(result,x,xref,grid,cell)
    result[:] = 4*(temp[:] .* bary(2)(xref) + result[:] .* bary(1)(xref));
end


# the three exact gradients of the P1 basis functions on a triangle
function triangle_bary1_grad!(result,x,xref,grid,cell)
    result[1] = grid.coords4nodes[grid.nodes4cells[cell,2],2] - grid.coords4nodes[grid.nodes4cells[cell,3],2];
    result[2] = grid.coords4nodes[grid.nodes4cells[cell,3],1] - grid.coords4nodes[grid.nodes4cells[cell,2],1];
    result[:] /= (2*grid.volume4cells[cell]);
end
function triangle_bary2_grad!(result,x,xref,grid,cell)
    result[1] = grid.coords4nodes[grid.nodes4cells[cell,3],2] - grid.coords4nodes[grid.nodes4cells[cell,1],2];
    result[2] = grid.coords4nodes[grid.nodes4cells[cell,1],1] - grid.coords4nodes[grid.nodes4cells[cell,3],1];
    result[:] /= (2*grid.volume4cells[cell]);
end
function triangle_bary3_grad!(result,x,xref,grid,cell)
    result[1] = grid.coords4nodes[grid.nodes4cells[cell,1],2] - grid.coords4nodes[grid.nodes4cells[cell,2],2];
    result[2] = grid.coords4nodes[grid.nodes4cells[cell,2],1] - grid.coords4nodes[grid.nodes4cells[cell,1],1];
    result[:] /= (2*grid.volume4cells[cell]);
end


# the six exact gradients of the CR basis functions on a triangle
function triangle_CR_1_grad!(result,x,xref,grid,cell)
    triangle_bary3_grad!(result,x,xref,grid,cell);
    result[:] .*= -2;
end
function triangle_CR_2_grad!(result,x,xref,grid,cell)
    triangle_bary1_grad!(result,x,xref,grid,cell);
    result[:] .*= -2;
end
function triangle_CR_3_grad!(result,x,xref,grid,cell)
    triangle_bary2_grad!(result,x,xref,grid,cell);
    result[:] .*= -2;
end


# the six exact gradients of the P2 basis functions on a triangle
function triangle_P2_1_grad!(result,x,xref,grid,cell)
    triangle_bary1_grad!(result,x,xref,grid,cell);
    result[:] .*= (4*bary(1)(xref)-1);
end
function triangle_P2_2_grad!(result,x,xref,grid,cell)
    triangle_bary2_grad!(result,x,xref,grid,cell)
    result[:] .*= (4*bary(2)(xref)-1);
end
function triangle_P2_3_grad!(result,x,xref,grid,cell)
    triangle_bary3_grad!(result,x,xref,grid,cell)
    result[:] .*= (4*bary(3)(xref)-1);
end
function triangle_P2_4_grad!(result,x,xref,grid,cell)
    temp = zeros(eltype(result),length(result));
    triangle_bary1_grad!(temp,x,xref,grid,cell)
    triangle_bary2_grad!(result,x,xref,grid,cell)
    result[:] = 4*(temp[:] .* bary(2)(xref) + result[:] .* bary(1)(xref));
end
function triangle_P2_5_grad!(result,x,xref,grid,cell)
    temp = zeros(eltype(result),length(result));
    triangle_bary2_grad!(temp,x,xref,grid,cell)
    triangle_bary3_grad!(result,x,xref,grid,cell)
    result[:] = 4*(temp[:] .* bary(3)(xref) + result[:] .* bary(2)(xref));
end
function triangle_P2_6_grad!(result,x,xref,grid,cell)
    temp = zeros(eltype(result),length(result));
    triangle_bary3_grad!(temp,x,xref,grid,cell)
    triangle_bary1_grad!(result,x,xref,grid,cell)
    result[:] = 4*(temp[:] .* bary(1)(xref) + result[:] .* bary(3)(xref));
end



# wrapper for FowardDiff
function FDgradient!(bfun::Function)
    function closure(result,x,xref,grid,cell)
        f(a) = bfun(a,grid,cell);
        #cfg = GradientConfig(f, x, Chunk{2}());
        ForwardDiff.gradient!(result,f,x);
    end    
end    



##################################
### FINITE ELEMENT DEFINITIONS ###
##################################


function get_CRFiniteElement(grid::Grid.Mesh, FDgradients::Bool = false)
    T = eltype(grid.coords4nodes)
    ensure_nodes4faces!(grid);
    ensure_volume4cells!(grid);
    ensure_faces4cells!(grid);
    ncells::Int = size(grid.nodes4cells,1);
    nfaces::Int = size(grid.nodes4faces,1);
    dofs4cells = grid.faces4cells;
    dofs4faces = zeros(Int64,nfaces,1);
    dofs4faces[:,1] = 1:nfaces;
    coords4dof = 1 // 2 * (grid.coords4nodes[grid.nodes4faces[:,1],:] + grid.coords4nodes[grid.nodes4faces[:,2],:]);
    
    
    # group basis functions
    celldim = size(grid.nodes4cells,2);
    if celldim == 3 # triangles
        bfun_ref = [CRbary(1),
                    CRbary(2),
                    CRbary(3)];
        bfun = [CRFEFunctions2D(1),
                CRFEFunctions2D(2),
                CRFEFunctions2D(3)];
        if FDgradients
            println("Initialising 2D CR-FiniteElement with ForwardDiff gradients...");
            bfun_grad! = [FDgradient!(CRFEFunctions2D(1)),
                          FDgradient!(CRFEFunctions2D(2)),
                          FDgradient!(CRFEFunctions2D(3))];
        else
            println("Initialising 2D CR-FiniteElement with exact gradients...");
            bfun_grad! = [triangle_CR1_grad!,
                          triangle_CR2_grad!,
                          triangle_CR3_grad!];
        end
    end    
    
    local_mass_matrix = LinearAlgebra.I(celldim) * 1 // 3;
    return FiniteElement{T}(grid,1,dofs4cells,dofs4faces,coords4dof,bfun_ref,bfun,bfun_grad!,local_mass_matrix);
end


function get_P1FiniteElement(grid::Grid.Mesh, FDgradients::Bool = false)
    T = eltype(grid.coords4nodes)
    dofs4cells = grid.nodes4cells;
    ensure_nodes4faces!(grid);
    ensure_volume4cells!(grid);
    dofs4faces = grid.nodes4faces;
    coords4dof = grid.coords4nodes;
    
    
    # group basis functions
    celldim = size(grid.nodes4cells,2);
    if celldim == 3 # triangles
        bfun_ref = [bary(1),
                    bary(2),
                    bary(3)];
        bfun = [P1FEFunctions2D(1),
                P1FEFunctions2D(2),
                P1FEFunctions2D(3)];
        if FDgradients
            println("Initialising 2D P1-FiniteElement with ForwardDiff gradients...");
            bfun_grad! = [FDgradient!(P1FEFunctions2D(1)),
                          FDgradient!(P1FEFunctions2D(2)),
                          FDgradient!(P1FEFunctions2D(3))];
        else
            println("Initialising 2D P1-FiniteElement with exact gradients...");
            bfun_grad! = [triangle_bary1_grad!,
                          triangle_bary2_grad!,
                          triangle_bary3_grad!];
        end
    elseif celldim == 2 # line segments
        bfun_ref = [bary(1),
                    bary(2)];
        bfun = [P1FEFunctions1D(1),
                P1FEFunctions1D(2)];
        if FDgradients
            println("Initialising 1D P1-FiniteElement with ForwardDiff gradients...");
            bfun_grad! = [FDgradient!(P1FEFunctions1D(1)),
                          FDgradient!(P1FEFunctions1D(2))];
        else
            println("Initialising 1D P1-FiniteElement with exact gradients...");
            bfun_grad! = [line_bary1_grad!,
                          line_bary2_grad!];
        end
    end    
    
    local_mass_matrix = (ones(T,celldim,celldim) + LinearAlgebra.I(celldim)) * 1 // ((celldim)*(celldim+1));
    return FiniteElement{T}(grid,1,dofs4cells,dofs4faces,coords4dof,bfun_ref,bfun,bfun_grad!,local_mass_matrix);
end


function get_P2FiniteElement(grid::Grid.Mesh, FDgradients::Bool = false)
    T = eltype(grid.coords4nodes)
    ensure_nodes4faces!(grid);
    ensure_faces4cells!(grid);
    ensure_volume4cells!(grid);
    ncells::Int = size(grid.nodes4cells,1);
    nnodes::Int = size(grid.coords4nodes,1);
    
    
    # group basis functions
    celldim = size(grid.nodes4cells,2);
    @assert celldim == 3
    if celldim == 3 # triangles
        dofs4cells = [grid.nodes4cells (nnodes .+ grid.faces4cells)];
        dofs4faces = [grid.nodes4faces[:,1] 1:size(grid.nodes4faces,1) grid.nodes4faces[:,2]];
        dofs4faces[:,2] .+= nnodes;
        coords4dof = [grid.coords4nodes;
            1 // 2 * (grid.coords4nodes[grid.nodes4faces[:,1],:] + grid.coords4nodes[grid.nodes4faces[:,2],:])]
        
        bfun_ref = [P2bary(1),
                    P2bary(2),
                    P2bary(3),
                    P2bary(4),
                    P2bary(5),
                    P2bary(6)];
        bfun = [P2FEFunctions2D(1),
                P2FEFunctions2D(2),
                P2FEFunctions2D(3),
                P2FEFunctions2D(4),
                P2FEFunctions2D(5),
                P2FEFunctions2D(6)];
        if FDgradients
            println("Initialising 2D P2-FiniteElement with ForwardDiff gradients...");
            test(x,grid::Grid.Mesh,cell) = 2*P1FEFunctions2D(1)(x,grid,cell)*(P1FEFunctions2D(1)(x,grid,cell) - 1//2)
            bfun_grad! = [FDgradient!(P2FEFunctions2D(1)),
                          FDgradient!(P2FEFunctions2D(2)),
                          FDgradient!(P2FEFunctions2D(3)),
                          FDgradient!(P2FEFunctions2D(4)),
                          FDgradient!(P2FEFunctions2D(5)),
                          FDgradient!(P2FEFunctions2D(6))];
        else                  
            println("Initialising 2D P2-FiniteElement with exact gradients...");
            bfun_grad! = [triangle_P2_1_grad!,
                          triangle_P2_2_grad!,
                          triangle_P2_3_grad!,
                          triangle_P2_4_grad!,
                          triangle_P2_5_grad!,
                          triangle_P2_6_grad!];
                      
        end   
    elseif celldim == 2 # line segments
        dofs4cells = [grid.nodes4cells (nnodes .+ 1:ncells)];
        dofs4faces = grid.nodes4faces;
        coords4dof = [grid.coords4nodes;
            1 // 2 * (grid.coords4nodes[grid.nodes4cells[:,1],:] + grid.coords4nodes[grid.nodes4cells[:,2],:])]
        
        bfun_ref = [P2bary(1),
                    P2bary(2),
                    P2bary(4)];
        bfun = [P2FEFunctions1D(1),
                P2FEFunctions1D(2),
                P3FEFunctions1D(3)];
        if FDgradients
            println("Initialising 1D P2-FiniteElement with ForwardDiff gradients...");
            bfun_grad! = [FDgradient!(P2FEFunctions1D(1)),
                          FDgradient!(P2FEFunctions1D(2)),
                          FDgradient!(P2FEFunctions1D(3))];
        else
            println("Initialising 1D P2-FiniteElement with exact gradients...");
            bfun_grad! = [line_P2_1_grad!,
                          line_P2_2_grad!,
                          line_P2_3_grad!];
        end
    end    
    
    # todo: update mass matrix
    local_mass_matrix = (ones(T,celldim,celldim) + LinearAlgebra.I(celldim)) * 1 // ((celldim)*(celldim+1));
    return FiniteElement{T}(grid,2,dofs4cells,dofs4faces,coords4dof,bfun_ref,bfun,bfun_grad!,local_mass_matrix);
end


end # module
