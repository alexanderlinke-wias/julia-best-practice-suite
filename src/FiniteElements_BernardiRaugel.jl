BRbasis2DV_ref = [(xref,grid,cell) -> [xref[1], 0.0],  # 1st node 1st component
                  (xref,grid,cell) -> [xref[2], 0.0],  # 2nd node 1st component
                  (xref,grid,cell) -> [xref[3], 0.0],  # 3rd node 1st component
                  (xref,grid,cell) -> [0.0, xref[1]],  # 1st node 2nd component
                  (xref,grid,cell) -> [0.0, xref[2]],  # 2nd node 2nd component
                  (xref,grid,cell) -> [0.0, xref[3]],  # 3rd node 2nd component 
                  (xref,grid,cell) -> 4*xref[1]*xref[2] .* grid.normal4faces[grid.faces4cells[cell,1],:],  # 1st side
                  (xref,grid,cell) -> 4*xref[2]*xref[3] .* grid.normal4faces[grid.faces4cells[cell,2],:],  # 2nd side
                  (xref,grid,cell) -> 4*xref[3]*xref[1] .* grid.normal4faces[grid.faces4cells[cell,3],:]]; # 3rd side
               

BRbasis_2DV = [(x,grid,cell) -> [get_P1function_2D(2,3)(x,grid,cell), 0.0], # 1st node
               (x,grid,cell) -> [get_P1function_2D(3,1)(x,grid,cell), 0.0], # 2nd node
               (x,grid,cell) -> [get_P1function_2D(1,2)(x,grid,cell), 0.0], # 3rd node
               (x,grid,cell) -> [0.0, get_P1function_2D(2,3)(x,grid,cell)], # 1st node
               (x,grid,cell) -> [0.0, get_P1function_2D(3,1)(x,grid,cell)], # 2nd node
               (x,grid,cell) -> [0.0, get_P1function_2D(1,2)(x,grid,cell)], # 3rd node   
               (x,grid,cell) -> P2basis_2D[4](x,grid,cell).* grid.normal4faces[grid.faces4cells[cell,1],:],
               (x,grid,cell) -> P2basis_2D[5](x,grid,cell).* grid.normal4faces[grid.faces4cells[cell,2],:],
               (x,grid,cell) -> P2basis_2D[6](x,grid,cell).* grid.normal4faces[grid.faces4cells[cell,3],:]] 
           


function get_BRFiniteElement(grid::Grid.Mesh, FDgradients::Bool = false)
    T = eltype(grid.coords4nodes)
    ensure_nodes4faces!(grid);
    ensure_faces4cells!(grid);
    ensure_volume4cells!(grid);
    ensure_normal4faces!(grid);
    ncells::Int = size(grid.nodes4cells,1);
    nnodes::Int = size(grid.coords4nodes,1);
    nfaces::Int = size(grid.nodes4faces,1);
    
    # group basis functions
    xdim = size(grid.coords4nodes,2);
    celldim = size(grid.nodes4cells,2);
    if celldim == 3 # triangles
        dofs4cells = zeros(Int64,ncells,9);
        dofs4cells[:,1:3] = grid.nodes4cells;
        dofs4cells[:,4:6] = nnodes .+ grid.nodes4cells;
        dofs4cells[:,7:9] = 2*nnodes .+ grid.faces4cells;
        dofs4faces = zeros(Int64,nfaces,5);
        dofs4faces[:,[1,3]] = grid.nodes4faces;
        dofs4faces[:,[2,4]] = nnodes .+ grid.nodes4faces;
        dofs4faces[:,5] = 2*nnodes .+ Array(1:nfaces);
        coords4dof = zeros(T,2*nnodes+nfaces,xdim);
        coords4dof[1:nnodes,:] = grid.coords4nodes
        coords4dof[nnodes+1:2*nnodes,:] = grid.coords4nodes
        coords4dof[2*nnodes+1:2*nnodes+nfaces,:] = 1 // 2 * (grid.coords4nodes[grid.nodes4faces[:,1],:] + grid.coords4nodes[grid.nodes4faces[:,2],:])
        
        bfun_ref = BRbasis2DV_ref;
        bfun = BRbasis_2DV;
        if FDgradients
            println("Initialising 2D Bernardi-Raugel-FiniteElement with ForwardDiff gradients...");
            bfun_grad! = Vector{Function}(undef,length(bfun));
            for k = 1:length(bfun)
                bfun_grad![k] = FDgradient(bfun[k],coords4dof[1,:],xdim);
            end
        else                  
            println("Initialising 2D Bernardi-Raugel-FiniteElement with exact gradients...");
            bfun_grad! = [triangle_bary1_grad!(0),
                          triangle_bary2_grad!(0),
                          triangle_bary3_grad!(0),
                          triangle_bary1_grad!(2),
                          triangle_bary2_grad!(2),
                          triangle_bary3_grad!(2),
                          triangle_BR_1_grad!(coords4dof[1,:]),
                          triangle_BR_2_grad!(coords4dof[1,:]),
                          triangle_BR_3_grad!(coords4dof[1,:])];
                      
        end   
    end    
    
    return FiniteElement{T}("BR=(P1V+nFB)", grid, 2, celldim - 1, dofs4cells, dofs4faces, coords4dof, bfun_ref, bfun, bfun_grad!, [[] []]);
end


##################################################################
#### exact gradients for finite element basis functions above ####
##################################################################

function triangle_BR_1_grad!(x)
    temp = zeros(eltype(x),length(x));
    function closure(result,x,xref,grid,cell)
        triangle_bary1_grad!(0)(temp,x,xref,grid,cell)
        triangle_bary2_grad!(0)(result,x,xref,grid,cell)
        for j = 1 : length(x)
            result[2+j] = result[j]
            result[j] = 4*(temp[j] .* xref[2] + result[j] .* xref[1]) * grid.normal4faces[grid.faces4cells[cell,1],1];
            result[2+j] = 4*(temp[j] .* xref[2] + result[2+j] .* xref[1]) * grid.normal4faces[grid.faces4cells[cell,1],2];
        end
    end    
end


function triangle_BR_2_grad!(x)
    temp = zeros(eltype(x),length(x));
    function closure(result,x,xref,grid,cell)
        triangle_bary2_grad!(0)(temp,x,xref,grid,cell)
        triangle_bary3_grad!(0)(result,x,xref,grid,cell)
        for j = 1 : length(x)
            result[2+j] = result[j]
            result[j] = 4*(temp[j] .* xref[3] + result[j] .* xref[2]) * grid.normal4faces[grid.faces4cells[cell,2],1];
            result[2+j] = 4*(temp[j] .* xref[3] + result[2+j] .* xref[2]) * grid.normal4faces[grid.faces4cells[cell,2],2];
        end
    end    
end

function triangle_BR_3_grad!(x)
    temp = zeros(eltype(x),length(x));
    function closure(result,x,xref,grid,cell)
        triangle_bary3_grad!(0)(temp,x,xref,grid,cell)
        triangle_bary1_grad!(0)(result,x,xref,grid,cell)
        for j = 1 : length(x)
            result[2+j] = result[j]
            result[j] = 4*(temp[j] .* xref[1] + result[j] .* xref[3]) * grid.normal4faces[grid.faces4cells[cell,3],1];
            result[2+j] = 4*(temp[j] .* xref[1] + result[2+j] .* xref[3]) * grid.normal4faces[grid.faces4cells[cell,3],2];
        end
    end    
end
