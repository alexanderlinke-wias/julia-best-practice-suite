CRbasis_ref = [(xref,grid,cell) -> 1 - 2*xref[3],  # 1st side in 2D / 4th side in 3D
               (xref,grid,cell) -> 1 - 2*xref[1],  # 2nd side
               (xref,grid,cell) -> 1 - 2*xref[2],  # 3rd side
               (xref,grid,cell) -> 1 - 2*xref[4]]; # 1st side in 3D
               

CRbasis = [(x,grid,cell) -> 1 - 2*P1basis_2D[3](x,grid,cell),  # 1st side in 2D / 4th side in 3D
           (x,grid,cell) -> 1 - 2*P1basis_2D[1](x,grid,cell),  # 2nd side
           (x,grid,cell) -> 1 - 2*P1basis_2D[2](x,grid,cell),  # 3rd side
           (x,grid,cell) -> 1 - 2*P1basis_2D[4](x,grid,cell)]; # 1st side in 3D

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
        local_mass_matrix = LinearAlgebra.I(celldim) * 1 // 3; # diagonal in 2D
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
        local_mass_matrix = [[] []]; # todo: compute this
    end    
    
    return FiniteElement{T}("CR", grid,1, 1, dofs4cells, dofs4faces, coords4dof, bfun_ref, bfun, bfun_grad!, local_mass_matrix);
end


##################################################################
#### exact gradients for finite element basis functions above ####
##################################################################

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
