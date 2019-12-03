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
    bfun_ref::Vector{Function};
    bfun::Vector{Function};
    bfun_grad!::Vector{Function};
    local_mass_matrix::Array{T,2};
end

# wrapper for P1 partition of unity
function bary(j::Int)
    function closure(xref)
        return xref[j];
    end    
end


P1basis_ref = [xref -> xref[1],  # 1st node
               xref -> xref[2],  # 2nd node
               xref -> xref[3],  # 3rd node (only 2D, 3D)  
               xref -> xref[4]]; # 4th node (3D)

CRbasis_ref = [xref -> 1 - 2*xref[3],  # 1st side
               xref -> 1 - 2*xref[1],  # 2nd side
               xref -> 1 - 2*xref[2]]; # 3rd side
               
P2basis_ref = [xref -> 2*xref[1]*(xref[1] - 1//2), # 1st node
               xref -> 2*xref[2]*(xref[2] - 1//2), # 2nd node
               xref -> 2*xref[3]*(xref[3] - 1//2), # 3rd node 
               xref -> 4*xref[1]*xref[2],  # 1st side
               xref -> 4*xref[2]*xref[3],  # 2nd side
               xref -> 4*xref[3]*xref[1]]; # 3rd side


function get_P1function_1D(index)
  return (x,grid,cell) ->
    sqrt(sum((grid.coords4nodes[grid.nodes4cells[cell,index],:] - x).^2)) / grid.volume4cells[cell];        
end               
           
P1basis_1D = [get_P1function_1D(2),  # 1st node
              get_P1function_1D(1)]; # 2nd node
              

function get_P1function_2D(index1,index2)
  return (x,grid,cell) -> 
    (grid.coords4nodes[grid.nodes4cells[cell,index1],1]*grid.coords4nodes[grid.nodes4cells[cell,index2],2]
    - grid.coords4nodes[grid.nodes4cells[cell,index2],1]*grid.coords4nodes[grid.nodes4cells[cell,index1],2]
    + x[1]*(grid.coords4nodes[grid.nodes4cells[cell,index1],2]
    - grid.coords4nodes[grid.nodes4cells[cell,index2],2])
    + x[2]*(grid.coords4nodes[grid.nodes4cells[cell,index2],1]
    - grid.coords4nodes[grid.nodes4cells[cell,index1],1]))/(2*grid.volume4cells[cell])
end               
               
               
P1basis_2D = [get_P1function_2D(2,3), # 1st node
              get_P1function_2D(3,1), # 2nd node
              get_P1function_2D(1,2)] # 3rd node
           


CRbasis = [(x,grid,cell) -> 1 - 2*P1basis_2D[3](x,grid,cell),  # 1st side in 2D/4th side in 3D
           (x,grid,cell) -> 1 - 2*P1basis_2D[1](x,grid,cell),  # 2nd side
           (x,grid,cell) -> 1 - 2*P1basis_2D[2](x,grid,cell),  # 3rd side
           (x,grid,cell) -> 1 - 2*P1basis_2D[4](x,grid,cell)]; # 1st side in 3D
           

P2_mask_node(a) = 2*a*(a - 1//2)           
P2_mask_face(a,b) = 4*a*b
           
P2basis_2D = [(x,grid,cell) -> P2_mask_node(P1basis_2D[1](x,grid,cell)), # 1st node              
              (x,grid,cell) -> P2_mask_node(P1basis_2D[2](x,grid,cell)), # 2nd node
              (x,grid,cell) -> P2_mask_node(P1basis_2D[3](x,grid,cell)), # 3rd node 
              (x,grid,cell) -> P2_mask_face(P1basis_2D[1](x,grid,cell),P1basis_2D[2](x,grid,cell)),  # 1st face
              (x,grid,cell) -> P2_mask_face(P1basis_2D[2](x,grid,cell),P1basis_2D[3](x,grid,cell)),  # 2nd face
              (x,grid,cell) -> P2_mask_face(P1basis_2D[3](x,grid,cell),P1basis_2D[1](x,grid,cell))]; # 3rd face


P2basis_1D = [(x,grid,cell) -> P2_mask_node(P1basis_1D[1](x,grid,cell)), # 1st node              
              (x,grid,cell) -> P2_mask_node(P1basis_1D[2](x,grid,cell)), # 2nd node
              (x,grid,cell) -> P2_mask_face(P1basis_1D[1](x,grid,cell),P1basis_1D[2](x,grid,cell))];  # 1st face



# wrapper for ForwardDiff & DiffResults
function FDgradient(bfun::Function, dim::Int)
    DRresult = DiffResults.GradientResult(Vector{Float64}(undef, dim));
    function closure(result,x,xref,grid,cell)
        f(a) = bfun(a,grid,cell);
        ForwardDiff.gradient!(DRresult,f,x);
        result[:] = DiffResults.gradient(DRresult);
    end    
end    






 #######################################################################################################
 #######################################################################################################
 ### FFFFF II NN    N II TTTTTT EEEEEE     EEEEEE LL     EEEEEE M     M EEEEEE NN    N TTTTTT SSSSSS ###
 ### FF    II N N   N II   TT   EE         EE     LL     EE     MM   MM EE     N N   N   TT   SS     ###
 ### FFFF  II N  N  N II   TT   EEEEE      EEEEE  LL     EEEEE  M M M M EEEEE  N  N  N   TT    SSSS  ###
 ### FF    II N   N N II   TT   EE         EE     LL     EE     M  M  M EE     N   N N   TT       SS ###
 ### FF    II N    NN II   TT   EEEEEE     EEEEEE LLLLLL EEEEEE M     M EEEEEE N    NN   TT   SSSSSS ###
 #######################################################################################################
 #######################################################################################################


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
    
    
    # group basis functions
    celldim = size(grid.nodes4cells,2);
    xdim = size(grid.coords4nodes,2);
    @assert celldim >= 3
    if celldim == 3 # triangles
        coords4dof = 1 // 2 * (grid.coords4nodes[grid.nodes4faces[:,1],:] +                        
                               grid.coords4nodes[grid.nodes4faces[:,2],:]);
        bfun_ref = CRbasis_ref[1:3];
        bfun = CRbasis;
        if FDgradients
            println("Initialising 2D CR-FiniteElement with ForwardDiff gradients...");
            bfun_grad! = Vector{Function}(undef,length(bfun));
            for k = 1:length(bfun)
                bfun_grad![k] = FDgradient(bfun[k],xdim);
            end
        else
            println("Initialising 2D CR-FiniteElement with exact gradients...");
            bfun_grad! = [triangle_CR_1_grad!,
                          triangle_CR_2_grad!,
                          triangle_CR_3_grad!];
        end
    elseif celldim == 4 # tetrahedra
        coords4dof = 1 // 3 * (grid.coords4nodes[grid.nodes4faces[:,1],:] +                        
                               grid.coords4nodes[grid.nodes4faces[:,2],:] +
                               grid.coords4nodes[grid.nodes4faces[:,3],:]);
        bfun_ref = CRbasis_ref[[4,1,2,3]];
        bfun = CRbasis;
        if FDgradients
            println("Initialising 2D CR-FiniteElement with ForwardDiff gradients...");
            bfun_grad! = Vector{Function}(undef,length(bfun));
            for k = 1:length(bfun)
                bfun_grad![k] = FDgradient(bfun[k],xdim);
            end
        else
            println("Initialising 2D CR-FiniteElement with exact gradients...");
            bfun_grad! = [triangle_CR_1_grad!,
                          triangle_CR_2_grad!,
                          triangle_CR_3_grad!];
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
    xdim = size(grid.coords4nodes,2);
    if celldim == 3 # triangles
        bfun_ref = P1basis_ref[1:3];
        bfun = P1basis_2D;
        if FDgradients
            println("Initialising 2D P1-FiniteElement with ForwardDiff gradients...");
            bfun_grad! = Vector{Function}(undef,length(bfun));
            for k = 1:length(bfun)
                bfun_grad![k] = FDgradient(bfun[k],xdim);
            end
        else
            println("Initialising 2D P1-FiniteElement with exact gradients...");
            bfun_grad! = [triangle_bary1_grad!,
                          triangle_bary2_grad!,
                          triangle_bary3_grad!];
        end
    elseif celldim == 2 # line segments
        bfun_ref = P1basis_ref[1:2];
        bfun = P1basis_1D;
        if FDgradients
            println("Initialising 1D P1-FiniteElement with ForwardDiff gradients...");
            bfun_grad! = Vector{Function}(undef,length(bfun));
            for k = 1:length(bfun)
                bfun_grad![k] = FDgradient(bfun[k],xdim);
            end
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
    xdim = size(grid.coords4nodes,2);
    celldim = size(grid.nodes4cells,2);
    if celldim == 3 # triangles
        dofs4cells = [grid.nodes4cells (nnodes .+ grid.faces4cells)];
        dofs4faces = [grid.nodes4faces[:,1] 1:size(grid.nodes4faces,1) grid.nodes4faces[:,2]];
        dofs4faces[:,2] .+= nnodes;
        coords4dof = [grid.coords4nodes;
            1 // 2 * (grid.coords4nodes[grid.nodes4faces[:,1],:] + grid.coords4nodes[grid.nodes4faces[:,2],:])]
        
        bfun_ref = P2basis_ref[1:6];
        bfun = P2basis_2D;
        if FDgradients
            println("Initialising 2D P2-FiniteElement with ForwardDiff gradients...");
            bfun_grad! = Vector{Function}(undef,length(bfun));
            for k = 1:length(bfun)
                bfun_grad![k] = FDgradient(bfun[k],xdim);
            end
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
        dofs4cells = [grid.nodes4cells 1:ncells];
        dofs4cells[:,3] .+= nnodes;
        dofs4faces = grid.nodes4faces;
        coords4dof = [grid.coords4nodes;
            1 // 2 * (grid.coords4nodes[grid.nodes4cells[:,1],:] + grid.coords4nodes[grid.nodes4cells[:,2],:])]
        
        bfun_ref = P2basis_ref[[1,2,4]];
        bfun = P2basis_1D;
        if FDgradients
            println("Initialising 1D P2-FiniteElement with ForwardDiff gradients...");
            bfun_grad! = Vector{Function}(undef,length(bfun));
            for k = 1:length(bfun)
                bfun_grad![k] = FDgradient(bfun[k],xdim);
            end
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







#### exact gradients


              
              

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
    result[:] .*= (4*bary(2)(xref)-1);
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











end # module
