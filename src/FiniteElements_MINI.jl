MINIbasis2DV_ref = [(xref,grid,cell) -> [xref[1], 0.0],  # 1st node 1st component
                    (xref,grid,cell) -> [xref[2], 0.0],  # 2nd node 1st component
                    (xref,grid,cell) -> [xref[3], 0.0],  # 3rd node 1st component
                    (xref,grid,cell) -> [27*xref[1]*xref[2]*xref[3], 0.0],  # cell bubble 1st component
                    (xref,grid,cell) -> [0.0, xref[1]],  # 1st node 2nd component
                    (xref,grid,cell) -> [0.0, xref[2]],  # 2nd node 2nd component
                    (xref,grid,cell) -> [0.0, xref[3]],  # 3rd node 2nd component 
                    (xref,grid,cell) -> [0.0, 27*xref[1]*xref[2]*xref[3]]]  # cell bubble 2nd component

function get_P3bubble_2D(x,grid,cell)
    return 27*get_P1function_2D(2,3)(x,grid,cell) * get_P1function_2D(3,1)(x,grid,cell) * get_P1function_2D(1,2)(x,grid,cell)
end



MINIbasis_2DV = [(x,grid,cell) -> [get_P1function_2D(2,3)(x,grid,cell), 0.0], # 1st node
                 (x,grid,cell) -> [get_P1function_2D(3,1)(x,grid,cell), 0.0], # 2nd node
                 (x,grid,cell) -> [get_P1function_2D(1,2)(x,grid,cell), 0.0], # 3rd node
                 (x,grid,cell) -> [get_P3bubble_2D(x,grid,cell), 0.0],  # cell bubble 1st component
                 (x,grid,cell) -> [0.0, get_P1function_2D(2,3)(x,grid,cell)], # 1st node
                 (x,grid,cell) -> [0.0, get_P1function_2D(3,1)(x,grid,cell)], # 2nd node
                 (x,grid,cell) -> [0.0, get_P1function_2D(1,2)(x,grid,cell)], # 3rd node   
                 (x,grid,cell) -> [0.0, get_P3bubble_2D(x,grid,cell)]]  # cell bubble 2nd component
           


function get_MINIFiniteElement(grid::Grid.Mesh, FDgradients::Bool = false)
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
        dofs4cells = zeros(Int64,ncells,8);
        dofs4cells[:,1:3] = grid.nodes4cells;
        dofs4cells[:,4] = nnodes .+ Array(1:ncells);
        dofs4cells[:,5:7] = (ncells + nnodes) .+ grid.nodes4cells;
        dofs4cells[:,8] = (ncells + 2*nnodes) .+ Array(1:ncells);
        dofs4faces = zeros(Int64,nfaces,4);
        dofs4faces[:,[1,2]] = grid.nodes4faces;
        dofs4faces[:,[3,4]] = (ncells + nnodes) .+ grid.nodes4faces;
        coords4dof = zeros(T,2*nnodes+2*ncells,xdim);
        coords4dof[1:nnodes,:] = grid.coords4nodes
        coords4dof[ncells + nnodes+1:ncells + 2*nnodes,:] = grid.coords4nodes
        coords4dof[nnodes+1:nnodes+ncells,:] = 1 // 3 * (grid.coords4nodes[grid.nodes4cells[:,1],:] + grid.coords4nodes[grid.nodes4cells[:,2],:] + grid.coords4nodes[grid.nodes4cells[:,3],:])
        coords4dof[ncells + 2*nnodes+1:2*nnodes+2*ncells,:] = coords4dof[nnodes+1:nnodes+ncells,:]
        
        bfun_ref = MINIbasis2DV_ref;
        bfun = MINIbasis_2DV;
        if FDgradients
            println("Initialising 2D MINI-FiniteElement with ForwardDiff gradients...");
            bfun_grad! = Vector{Function}(undef,length(bfun));
            for k = 1:length(bfun)
                bfun_grad![k] = FDgradient(bfun[k],coords4dof[1,:],xdim);
            end
        else                  
            println("Initialising 2D MINI-FiniteElement with exact gradients...");
            bfun_grad! = [triangle_bary1_grad!(0),
                          triangle_bary2_grad!(0),
                          triangle_bary3_grad!(0),
                          triangle_cellbubble_grad!(coords4dof[1,:],0),
                          triangle_bary1_grad!(2),
                          triangle_bary2_grad!(2),
                          triangle_bary3_grad!(2),
                          triangle_cellbubble_grad!(coords4dof[1,:],2)];
                      
        end   
    end    
    
    return FiniteElement{T}("MINI=(P1V+CBV)", grid, 3, celldim - 1, dofs4cells, dofs4faces, coords4dof, bfun_ref, bfun, bfun_grad!, [[] []]);
end


##################################################################
#### exact gradients for finite element basis functions above ####
##################################################################

function triangle_cellbubble_grad!(x, offset)
    temp = zeros(eltype(x),length(x));
    temp2 = zeros(eltype(x),length(x));
    function closure(result,x,xref,grid,cell)
        triangle_bary1_grad!(0)(temp,x,xref,grid,cell)
        triangle_bary2_grad!(0)(temp2,x,xref,grid,cell)
        triangle_bary3_grad!(offset)(result,x,xref,grid,cell)
        for j = 1 : length(x)
            result[offset+j] = 27*(temp[j]*xref[2]*xref[3] + temp2[j]*xref[1]*xref[3] + result[offset+j]*xref[1]*xref[2])
        end
    end    
end
