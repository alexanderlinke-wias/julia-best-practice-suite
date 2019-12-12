module FiniteElements

using Grid
using LinearAlgebra
using ForwardDiff


# Finite Element structure
# = container for set of basis functions and their gradients
#   local dof numbers, coordinates for dofs (=where one basis functions equals one)
#   local_mass_matrix 
#
# todo/ideas: dual functionals for dofs?, vector-valued finite elements (Hdiv, Hcurl)
struct FiniteElement{T <: Real}
    name::String;
    grid::Grid.Mesh;
    polynomial_order::Int;
    ncomponents::Int; # 1 = scalar
    dofs4cells::Array{Int64,2};
    dofs4faces::Array{Int64,2};
    coords4dofs::Array{T,2};
    bfun_ref::Vector{Function};
    bfun::Vector{Function};
    bfun_grad!::Vector{Function};
    local_mass_matrix::Array{T,2};
end


P1basis_ref = [(xref,grid,cell) -> xref[1],  # 1st node
               (xref,grid,cell) -> xref[2],  # 2nd node
               (xref,grid,cell) -> xref[3],  # 3rd node (only 2D, 3D)  
               (xref,grid,cell) -> xref[4]]; # 4th node (only 3D)
               

P1basis2DV_ref = [(xref,grid,cell) -> [xref[1], 0.0],  # 1st node 1st component
                  (xref,grid,cell) -> [xref[2], 0.0],  # 2nd node 1st component
                  (xref,grid,cell) -> [xref[3], 0.0],  # 3rd node 1st component
                  (xref,grid,cell) -> [0.0, xref[1]],  # 1st node 2nd component
                  (xref,grid,cell) -> [0.0, xref[2]],  # 2nd node 2nd component
                  (xref,grid,cell) -> [0.0, xref[3]]]; # 3rd node 2nd component 
                  
                  
BRbasis2DV_ref = [(xref,grid,cell) -> [xref[1], 0.0],  # 1st node 1st component
                  (xref,grid,cell) -> [xref[2], 0.0],  # 2nd node 1st component
                  (xref,grid,cell) -> [xref[3], 0.0],  # 3rd node 1st component
                  (xref,grid,cell) -> [0.0, xref[1]],  # 1st node 2nd component
                  (xref,grid,cell) -> [0.0, xref[2]],  # 2nd node 2nd component
                  (xref,grid,cell) -> [0.0, xref[3]],  # 3rd node 2nd component 
                  (xref,grid,cell) -> 4*xref[1]*xref[2] .* grid.normals4face[faces4cells[cell,1],:],  # 1st side
                  (xref,grid,cell) -> 4*xref[2]*xref[3] .* grid.normals4face[faces4cells[cell,2],:],  # 2nd side
                  (xref,grid,cell) -> 4*xref[3]*xref[1] .* grid.normals4face[faces4cells[cell,3],:]]; # 3rd side
               

CRbasis_ref = [(xref,grid,cell) -> 1 - 2*xref[3],  # 1st side in 2D / 4th side in 3D
               (xref,grid,cell) -> 1 - 2*xref[1],  # 2nd side
               (xref,grid,cell) -> 1 - 2*xref[2],  # 3rd side
               (xref,grid,cell) -> 1 - 2*xref[4]]; # 1st side in 3D
               
P2basis_ref = [(xref,grid,cell) -> 2*xref[1]*(xref[1] - 1//2), # 1st node
               (xref,grid,cell) -> 2*xref[2]*(xref[2] - 1//2), # 2nd node
               (xref,grid,cell) -> 2*xref[3]*(xref[3] - 1//2), # 3rd node 
               (xref,grid,cell) -> 4*xref[1]*xref[2],  # 1st side
               (xref,grid,cell) -> 4*xref[2]*xref[3],  # 2nd side
               (xref,grid,cell) -> 4*xref[3]*xref[1]]; # 3rd side
               

P2basis2DV_ref = [(xref,grid,cell) -> [2*xref[1]*(xref[1] - 1//2),0.0], # 1st node
                  (xref,grid,cell) -> [2*xref[2]*(xref[2] - 1//2),0.0], # 2nd node
                  (xref,grid,cell) -> [2*xref[3]*(xref[3] - 1//2),0.0], # 3rd node 
                  (xref,grid,cell) -> [4*xref[1]*xref[2],0.0],  # 1st side
                  (xref,grid,cell) -> [4*xref[2]*xref[3],0.0],  # 2nd side
                  (xref,grid,cell) -> [4*xref[3]*xref[1],0.0],
                  (xref,grid,cell) -> [0.0, 2*xref[1]*(xref[1] - 1//2)], # 1st node
                  (xref,grid,cell) -> [0.0, 2*xref[2]*(xref[2] - 1//2)], # 2nd node
                  (xref,grid,cell) -> [0.0, 2*xref[3]*(xref[3] - 1//2)], # 3rd node 
                  (xref,grid,cell) -> [0.0, 4*xref[1]*xref[2]],  # 1st side
                  (xref,grid,cell) -> [0.0, 4*xref[2]*xref[3]],  # 2nd side
                  (xref,grid,cell) -> [0.0, 4*xref[3]*xref[1]]]; # 3rd side


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
           
P1basis_2DV = [(x,grid,cell) -> [get_P1function_2D(2,3)(x,grid,cell), 0.0], # 1st node
               (x,grid,cell) -> [get_P1function_2D(3,1)(x,grid,cell), 0.0], # 2nd node
               (x,grid,cell) -> [get_P1function_2D(1,2)(x,grid,cell), 0.0], # 3rd node
               (x,grid,cell) -> [0.0, get_P1function_2D(2,3)(x,grid,cell)], # 1st node
               (x,grid,cell) -> [0.0, get_P1function_2D(3,1)(x,grid,cell)], # 2nd node
               (x,grid,cell) -> [0.0, get_P1function_2D(1,2)(x,grid,cell)]] # 3rd node     


CRbasis = [(x,grid,cell) -> 1 - 2*P1basis_2D[3](x,grid,cell),  # 1st side in 2D / 4th side in 3D
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
              
              
P2basis_2DV = [(x,grid,cell) -> [P2_mask_node(P1basis_2D[1](x,grid,cell)), 0.0], # 1st node 1st component             
               (x,grid,cell) -> [P2_mask_node(P1basis_2D[2](x,grid,cell)), 0.0], # 2nd node
               (x,grid,cell) -> [P2_mask_node(P1basis_2D[3](x,grid,cell)), 0.0], # 3rd node 
               (x,grid,cell) -> [P2_mask_face(P1basis_2D[1](x,grid,cell),P1basis_2D[2](x,grid,cell)), 0.0],  # 1st face
               (x,grid,cell) -> [P2_mask_face(P1basis_2D[2](x,grid,cell),P1basis_2D[3](x,grid,cell)), 0.0],  # 2nd face
               (x,grid,cell) -> [P2_mask_face(P1basis_2D[3](x,grid,cell),P1basis_2D[1](x,grid,cell)), 0.0],
               (x,grid,cell) -> [0.0, P2_mask_node(P1basis_2D[1](x,grid,cell)) ], # 1st node 2nd component             
               (x,grid,cell) -> [0.0, P2_mask_node(P1basis_2D[2](x,grid,cell))], # 2nd node
               (x,grid,cell) -> [0.0, P2_mask_node(P1basis_2D[3](x,grid,cell))], # 3rd node 
               (x,grid,cell) -> [0.0, P2_mask_face(P1basis_2D[1](x,grid,cell),P1basis_2D[2](x,grid,cell))],  # 1st face
               (x,grid,cell) -> [0.0, P2_mask_face(P1basis_2D[2](x,grid,cell),P1basis_2D[3](x,grid,cell))],  # 2nd face
               (x,grid,cell) -> [0.0, P2_mask_face(P1basis_2D[3](x,grid,cell),P1basis_2D[1](x,grid,cell))]]; # 3rd face


P2basis_1D = [(x,grid,cell) -> P2_mask_node(P1basis_1D[1](x,grid,cell)), # 1st node              
              (x,grid,cell) -> P2_mask_node(P1basis_1D[2](x,grid,cell)), # 2nd node
              (x,grid,cell) -> P2_mask_face(P1basis_1D[1](x,grid,cell),P1basis_1D[2](x,grid,cell))];  # 1st face


              
BRbasis_2DV = [(x,grid,cell) -> [get_P1function_2D(2,3), 0.0], # 1st node
               (x,grid,cell) -> [get_P1function_2D(3,1), 0.0], # 2nd node
               (x,grid,cell) -> [get_P1function_2D(1,2), 0.0], # 3rd node
               (x,grid,cell) -> [0.0, get_P1function_2D(2,3)], # 1st node
               (x,grid,cell) -> [0.0, get_P1function_2D(3,1)], # 2nd node
               (x,grid,cell) -> [0.0, get_P1function_2D(1,2)], # 3rd node   
               (x,grid,cell) -> P2_basis_2D[4](x,grid,cell).* grid.normals4face[faces4cells[cell,1],:],
               (x,grid,cell) -> P2_basis_2D[5](x,grid,cell).* grid.normals4face[faces4cells[cell,2],:],
               (x,grid,cell) -> P2_basis_2D[6](x,grid,cell).* grid.normals4face[faces4cells[cell,3],:]] 
              

# wrapper for ForwardDiff & DiffResults
function FDgradient(bfun::Function, x) where T <: Real
    DRresult = DiffResults.GradientResult(x);
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

 
 ##########################################################
 ### CROUZEIX-RAVIART FINITE ELEMENT (H1-nonconforming) ###
 ##########################################################
 
 
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
        bfun = CRbasis[1:3];
        if FDgradients
            println("Initialising 2D CR-FiniteElement with ForwardDiff gradients...");
            bfun_grad! = Vector{Function}(undef,length(bfun));
            for k = 1:length(bfun)
                bfun_grad![k] = FDgradient(bfun[k],coords4dof[1,:]);
            end
        else
            println("Initialising 2D CR-FiniteElement with exact gradients...");
            bfun_grad! = [triangle_CR_1_grad!(0),
                          triangle_CR_2_grad!(0),
                          triangle_CR_3_grad!(0)];
        end
        local_mass_matrix = LinearAlgebra.I(celldim) * 1 // 3;
    elseif celldim == 4 # tetrahedra
        coords4dof = 1 // 3 * (grid.coords4nodes[grid.nodes4faces[:,1],:] +                        
                               grid.coords4nodes[grid.nodes4faces[:,2],:] +
                               grid.coords4nodes[grid.nodes4faces[:,3],:]);
        bfun_ref = CRbasis_ref[[4,1,2,3]];
        bfun = CRbasis[[4,1,2,3]];
        if FDgradients
            println("Initialising 3D CR-FiniteElement with ForwardDiff gradients...");
            bfun_grad! = Vector{Function}(undef,length(bfun));
            for k = 1:length(bfun)
                bfun_grad![k] = FDgradient(bfun[k],coords4dof[1,:]);
            end
        else
            println("Initialising 3D CR-FiniteElement with exact gradients...");
            bfun_grad! = [triangle_CR_1_grad!(0),
                          triangle_CR_2_grad!(0),
                          triangle_CR_3_grad!(0)];
        end   
        local_mass_matrix = zeros(T,celldim,celldim);
    end    
    
    return FiniteElement{T}("CR", grid,1, 1, dofs4cells, dofs4faces, coords4dof, bfun_ref, bfun, bfun_grad!, local_mass_matrix);
end


 #########################
 ### P0 FINITE ELEMENT ###
 #########################

function get_P0FiniteElement(grid::Grid.Mesh)
    T = eltype(grid.coords4nodes)
    ensure_nodes4faces!(grid);
    ensure_volume4cells!(grid);
    ncells::Int = size(grid.nodes4cells,1);
    dofs4cells = zeros(Int64,ncells,1);
    dofs4cells[:,1] = 1:ncells;
    dofs4faces = [[] []];
    
    celldim = size(grid.nodes4cells,2);
    xdim = size(grid.coords4nodes,2);
    coords4dof = zeros(eltype(grid.coords4nodes),ncells,xdim);
    for cell = 1 : ncells
        for j = 1 : xdim
            for i = 1 : celldim
                coords4dof[cell, j] += grid.coords4nodes[grid.nodes4cells[cell,i],j]
            end
            coords4dof[cell, j] /= celldim
        end
    end    
    
    # group basis functions
    bfun_ref = Vector{Function}(undef,1)
    bfun = Vector{Function}(undef,1)
    bfun_grad! = Vector{Function}(undef,1)
    bfun_ref[1] = (xref,grid,cell) -> 1;
    bfun[1] = (x,grid,cell) -> 1;
    bfun_grad![1] = function P0gradient(result,x,xref,grid,cell) 
                      result[:] .= 0
                    end  
    
    local_mass_matrix = LinearAlgebra.I(1);
    return FiniteElement{T}("P0", grid, 0, 1, dofs4cells, dofs4faces, coords4dof, bfun_ref, bfun, bfun_grad!, local_mass_matrix);
end 


 #################################################
 ### COURANT P1 FINITE ELEMENT (H1-conforming) ###
 #################################################
  
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
                bfun_grad![k] = FDgradient(bfun[k],coords4dof[1,:]);
            end
        else
            println("Initialising 2D P1-FiniteElement with exact gradients...");
            bfun_grad! = [triangle_bary1_grad!(0),
                          triangle_bary2_grad!(0),
                          triangle_bary3_grad!(0)];
        end
    elseif celldim == 2 # line segments
        bfun_ref = P1basis_ref[1:2];
        bfun = P1basis_1D;
        if FDgradients
            println("Initialising 1D P1-FiniteElement with ForwardDiff gradients...");
            bfun_grad! = Vector{Function}(undef,length(bfun));
            for k = 1:length(bfun)
                bfun_grad![k] = FDgradient(bfun[k],coords4dof[1,:]);
            end
        else
            println("Initialising 1D P1-FiniteElement with exact gradients...");
            bfun_grad! = [line_bary1_grad!,
                          line_bary2_grad!];
        end
    end    
    
    local_mass_matrix = (ones(T,celldim,celldim) + LinearAlgebra.I(celldim)) * 1 // ((celldim)*(celldim+1));
    return FiniteElement{T}("P1", grid, 1, 1, dofs4cells, dofs4faces, coords4dof, bfun_ref, bfun, bfun_grad!, local_mass_matrix);
end


function get_P1VectorFiniteElement(grid::Grid.Mesh, FDgradients::Bool = false)
    T = eltype(grid.coords4nodes)
    celldim = size(grid.nodes4cells,2);
    xdim = size(grid.coords4nodes,2);
    nnodes::Int = size(grid.coords4nodes,1);
    ncells::Int = size(grid.nodes4cells,1);
    dofs4cells = zeros(Int64,ncells,(celldim-1)*celldim);
    dofs4cells[:,1:celldim] = grid.nodes4cells
    ensure_nodes4faces!(grid);
    ensure_volume4cells!(grid);
    nfaces = size(grid.nodes4faces,1);
    dofs4faces = zeros(Int64,nfaces,(celldim-1)*(celldim-1));
    dofs4faces[:,1:celldim-1] = grid.nodes4faces
    coords4dof = zeros(T,nnodes*(celldim-1),xdim);
    coords4dof[1:nnodes,:] = grid.coords4nodes;
    
    
    # group basis functions
    if celldim == 3 # triangles
        dofs4cells[:,celldim+1:2*celldim] = nnodes.+grid.nodes4cells;
        dofs4faces[:,celldim:2*(celldim-1)] = nnodes.+grid.nodes4faces;
        coords4dof[nnodes+1:2*nnodes,:] = grid.coords4nodes;
        bfun_ref = P1basis2DV_ref;
        bfun = P1basis_2DV;
        if FDgradients
            println("Initialising 2D Vector P1-FiniteElement with ForwardDiff gradients...");
            bfun_grad! = Vector{Function}(undef,length(bfun));
            for k = 1:length(bfun)
                bfun_grad![k] = FDgradient(bfun[k],coords4dof[1,:]);
            end
        else
            println("Initialising 2D Vector P1-FiniteElement with exact gradients...");
            bfun_grad! = [triangle_bary1_grad!(0),
                          triangle_bary2_grad!(0),
                          triangle_bary3_grad!(0),
                          triangle_bary1_grad!(2),
                          triangle_bary2_grad!(2),
                          triangle_bary3_grad!(2)];
        end
    elseif celldim == 2 # line segments
        bfun_ref = P1basis_ref[1:2];
        bfun = P1basis_1D;
        if FDgradients
            println("Initialising 1D P1-FiniteElement with ForwardDiff gradients...");
            bfun_grad! = Vector{Function}(undef,length(bfun));
            for k = 1:length(bfun)
                bfun_grad![k] = FDgradient(bfun[k],coords4dof[1,:]);
            end
        else
            println("Initialising 1D P1-FiniteElement with exact gradients...");
            bfun_grad! = [line_bary1_grad!,
                          line_bary2_grad!];
        end
    end    
    
    local_mass_matrix = (ones(T,celldim,celldim) + LinearAlgebra.I(celldim)) * 1 // ((celldim)*(celldim+1));
    return FiniteElement{T}("P1", grid, 1, celldim-1, dofs4cells, dofs4faces, coords4dof, bfun_ref, bfun, bfun_grad!, local_mass_matrix);
end


 #################################################
 ### COURANT P2 FINITE ELEMENT (H1-conforming) ###
 #################################################

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
                bfun_grad![k] = FDgradient(bfun[k],coords4dof[1,:]);
            end
        else                  
            println("Initialising 2D P2-FiniteElement with exact gradients...");
            bfun_grad! = [triangle_P2_1_grad!(0),
                          triangle_P2_2_grad!(0),
                          triangle_P2_3_grad!(0),
                          triangle_P2_4_grad!(coords4dof[1,:],0),
                          triangle_P2_5_grad!(coords4dof[1,:],0),
                          triangle_P2_6_grad!(coords4dof[1,:],0)];
                      
        end   
        local_mass_matrix = [ 6 -1 -1  0 -4  0;
                             -1  6 -1  0  0 -4;
                             -1 -1  6 -4  0  0;
                              0  0 -4 32 16 16;
                             -4  0  0 16 32 16;
                              0 -4  0 16 16 32] * 1//180;
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
                bfun_grad![k] = FDgradient(bfun[k],coords4dof[1,:]);
            end
        else
            println("Initialising 1D P2-FiniteElement with exact gradients...");
            bfun_grad! = [line_P2_1_grad!,
                          line_P2_2_grad!,
                          line_P2_3_grad!(coords4dof[1,:])];
        end
        local_mass_matrix = [ 6 -1  0;
                             -1  6  0;
                              0  0 32] * 1//180;
    end    
    
    return FiniteElement{T}("P2", grid, 2, 1, dofs4cells, dofs4faces, coords4dof, bfun_ref, bfun, bfun_grad!, local_mass_matrix);
end


function get_P2VectorFiniteElement(grid::Grid.Mesh, FDgradients::Bool = false)
    T = eltype(grid.coords4nodes)
    ensure_nodes4faces!(grid);
    ensure_faces4cells!(grid);
    ensure_volume4cells!(grid);
    ncells::Int = size(grid.nodes4cells,1);
    nnodes::Int = size(grid.coords4nodes,1);
    nfaces::Int = size(grid.nodes4faces,1);
    
    # group basis functions
    xdim = size(grid.coords4nodes,2);
    celldim = size(grid.nodes4cells,2);
    if celldim == 3 # triangles
        dofs4cells = zeros(Int64,ncells,12);
        dofs4cells[:,1:3] = grid.nodes4cells;
        dofs4cells[:,4:6] = nnodes .+ grid.faces4cells;
        dofs4cells[:,7:9] = (nnodes + nfaces) .+ grid.nodes4cells;
        dofs4cells[:,10:12] = (2*nnodes + nfaces) .+ grid.faces4cells;
        dofs4faces = zeros(Int64,nfaces,6);
        dofs4faces[:,[1,3]] = grid.nodes4faces;
        dofs4faces[:,2] = nnodes .+ Array(1:nfaces);
        dofs4faces[:,[4,6]] = (nnodes + nfaces) .+ grid.nodes4faces;
        dofs4faces[:,5] = (2*nnodes + nfaces) .+ Array(1:nfaces);
        coords4dof = zeros(T,2*(nnodes+nfaces),xdim);
        coords4dof[1:nnodes,:] = grid.coords4nodes
        coords4dof[nnodes+nfaces+1:2*nnodes + nfaces,:] = grid.coords4nodes
        coords4dof[nnodes+1:nnodes+nfaces,:] = 1 // 2 * (grid.coords4nodes[grid.nodes4faces[:,1],:] + grid.coords4nodes[grid.nodes4faces[:,2],:])
        coords4dof[2*nnodes+nfaces+1:2*(nnodes+nfaces),:] = coords4dof[nnodes+1:nnodes+nfaces,:]
        
        bfun_ref = P2basis2DV_ref;
        bfun = P2basis_2DV;
        if FDgradients
            println("Initialising 2D Vector P2-FiniteElement with ForwardDiff gradients...");
            bfun_grad! = Vector{Function}(undef,length(bfun));
            for k = 1:length(bfun)
                bfun_grad![k] = FDgradient(bfun[k],[coords4dof[1,:];coords4dof[1,:]]);
            end
        else                  
            println("Initialising 2D Vector P2-FiniteElement with exact gradients...");
            bfun_grad! = [triangle_P2_1_grad!(0),
                          triangle_P2_2_grad!(0),
                          triangle_P2_3_grad!(0),
                          triangle_P2_4_grad!(coords4dof[1,:],0),
                          triangle_P2_5_grad!(coords4dof[1,:],0),
                          triangle_P2_6_grad!(coords4dof[1,:],0),
                          triangle_P2_1_grad!(2),
                          triangle_P2_2_grad!(2),
                          triangle_P2_3_grad!(2),
                          triangle_P2_4_grad!(coords4dof[1,:],2),
                          triangle_P2_5_grad!(coords4dof[1,:],2),
                          triangle_P2_6_grad!(coords4dof[1,:],2)];
                      
        end   
        local_mass_matrix = [ 6 -1 -1  0 -4  0;
                             -1  6 -1  0  0 -4;
                             -1 -1  6 -4  0  0;
                              0  0 -4 32 16 16;
                             -4  0  0 16 32 16;
                              0 -4  0 16 16 32] * 1//180;
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
                bfun_grad![k] = FDgradient(bfun[k],coords4dof[1,:]);
            end
        else
            println("Initialising 1D P2-FiniteElement with exact gradients...");
            bfun_grad! = [line_P2_1_grad!,
                          line_P2_2_grad!,
                          line_P2_3_grad!(coords4dof[1,:])];
        end
        local_mass_matrix = [ 6 -1  0;
                             -1  6  0;
                              0  0 32] * 1//180;
    end    
    
    return FiniteElement{T}("P2", grid, 2, celldim - 1, dofs4cells, dofs4faces, coords4dof, bfun_ref, bfun, bfun_grad!, local_mass_matrix);
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
    result .*= (4*xref[1]-1);
end   
function line_P2_2_grad!(result,x,xref,grid,cell)
    line_bary2_grad!(result,x,xref,grid,cell);
    result .*= (4*xref[2]-1);
end   
function line_P2_3_grad!(x)
    temp = zeros(eltype(x),length(x));
    function closure(result,x,xref,grid,cell)
        line_bary1_grad!(temp,x,xref,grid,cell)
        line_bary2_grad!(result,x,xref,grid,cell)
        for j = 1 : length(x)
            result[j] = 4*(temp[j] .* xref[2] + result[j] .* xref[1]);
        end
    end
end


# the three exact gradients of the P1 basis functions on a triangle
function triangle_bary1_grad!(offset)
    function closure(result,x,xref,grid,cell)
        result[1+offset] = grid.coords4nodes[grid.nodes4cells[cell,2],2] - grid.coords4nodes[grid.nodes4cells[cell,3],2];
        result[2+offset] = grid.coords4nodes[grid.nodes4cells[cell,3],1] - grid.coords4nodes[grid.nodes4cells[cell,2],1];
        result ./= (2*grid.volume4cells[cell]);
    end    
end
function triangle_bary2_grad!(offset)
    function closure(result,x,xref,grid,cell)
        result[1+offset] = grid.coords4nodes[grid.nodes4cells[cell,3],2] - grid.coords4nodes[grid.nodes4cells[cell,1],2];
        result[2+offset] = grid.coords4nodes[grid.nodes4cells[cell,1],1] - grid.coords4nodes[grid.nodes4cells[cell,3],1];
        result ./= (2*grid.volume4cells[cell]);
    end    
end
function triangle_bary3_grad!(offset)
    function closure(result,x,xref,grid,cell)
        result[1+offset] = grid.coords4nodes[grid.nodes4cells[cell,1],2] - grid.coords4nodes[grid.nodes4cells[cell,2],2];
        result[2+offset] = grid.coords4nodes[grid.nodes4cells[cell,2],1] - grid.coords4nodes[grid.nodes4cells[cell,1],1];
        result ./= (2*grid.volume4cells[cell]);
    end
end


# the six exact gradients of the CR basis functions on a triangle
function triangle_CR_1_grad!(offset)
    function closure(result,x,xref,grid,cell)
        triangle_bary3_grad!(offset)(result,x,xref,grid,cell);
        result[1+offset] *= -2;
        result[2+offset] *= -2;
    end    
end
function triangle_CR_2_grad!(offset)
    function closure(result,x,xref,grid,cell)
        triangle_bary1_grad!(offset)(result,x,xref,grid,cell);
        result[1+offset] *= -2;
        result[2+offset] *= -2;
    end    
end
function triangle_CR_3_grad!(offset)
    function closure(result,x,xref,grid,cell)
        triangle_bary2_grad!(offset)(result,x,xref,grid,cell);
        result[1+offset] *= -2;
        result[2+offset] *= -2;
    end    
end


# the six exact gradients of the P2 basis functions on a triangle
function triangle_P2_1_grad!(offset)
    function closure(result,x,xref,grid,cell)
        triangle_bary1_grad!(offset)(result,x,xref,grid,cell);
        result[offset+1] *= (4*xref[1]-1);
        result[offset+2] *= (4*xref[1]-1);
    end
end
function triangle_P2_2_grad!(offset)
    function closure(result,x,xref,grid,cell)
        triangle_bary2_grad!(offset)(result,x,xref,grid,cell)
        result[offset+1] *= (4*xref[2]-1);
        result[offset+2] *= (4*xref[2]-1);
    end    
end
function triangle_P2_3_grad!(offset)
    function closure(result,x,xref,grid,cell)
        triangle_bary3_grad!(offset)(result,x,xref,grid,cell)
        result[offset+1] *= (4*xref[3]-1);
        result[offset+2] *= (4*xref[3]-1);
    end    
end
function triangle_P2_4_grad!(x, offset = 0)
    temp = zeros(eltype(x),length(x));
    function closure(result,x,xref,grid,cell)
        triangle_bary1_grad!(0)(temp,x,xref,grid,cell)
        triangle_bary2_grad!(offset)(result,x,xref,grid,cell)
        for j = 1 : length(x)
            result[offset+j] = 4*(temp[j] .* xref[2] + result[offset+j] .* xref[1]);
        end
    end    
end
function triangle_P2_5_grad!(x, offset = 0)
    temp = zeros(eltype(x),length(x));
    function closure(result,x,xref,grid,cell)
        triangle_bary2_grad!(0)(temp,x,xref,grid,cell)
        triangle_bary3_grad!(offset)(result,x,xref,grid,cell)
        for j = 1 : length(x)
            result[offset+j] = 4*(temp[j] .* xref[3] + result[offset+j] .* xref[2]);
        end
    end
end
function triangle_P2_6_grad!(x, offset = 0)
    temp = zeros(eltype(x),length(x));
    function closure(result,x,xref,grid,cell)
        triangle_bary3_grad!(0)(temp,x,xref,grid,cell)
        triangle_bary1_grad!(offset)(result,x,xref,grid,cell)
        for j = 1 : length(x)
            result[offset+j] = 4*(temp[j] .* xref[1] + result[offset+j] .* xref[3]);
        end
    end
end

function triangle_BR_1_grad!(x, offset = 0)
    temp = zeros(eltype(x),length(x));
    function closure(result,x,xref,grid,cell)
        triangle_bary1_grad!(0)(temp,x,xref,grid,cell)
        triangle_bary2_grad!(offset)(result,x,xref,grid,cell)
        for j = 1 : length(x)
            result[offset+j] = 4*(temp[j] .* xref[2] + result[offset+j] .* xref[1]);
        end
    end    
end











end # module
