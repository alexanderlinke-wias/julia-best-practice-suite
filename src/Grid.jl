module Grid

using SparseArrays
using LinearAlgebra

export Triangulation,ensure_area4cells!,ensure_bfaces!,ensure_faces4cells!,ensure_nodes4faces!

mutable struct Triangulation
    coords4nodes::Array{Float64,2}
    nodes4cells::Array{Int,2}
    
    area4cells::Array{Float64,1}
    nodes4faces::Array{Int,2}
    faces4cells::Array{Int,2}
    bfaces::Array{Int,1}
    
    function Triangulation(coords,nodes)
        # only 2d triangulations allowed yet
        @assert size(coords,2) == 2
        @assert size(nodes,2) == 3
        new(coords,nodes,[],[[] []],[[] []],[]);
    end
    
end


function Triangulation(coords,nodes,nrefinements)
    for j=1:nrefinements
        coords, nodes = uniform_refinement(coords,nodes)
    end
    return Triangulation(coords,nodes);
end


# perform a uniform (red) refinement of the triangulation
function uniform_refinement(coords4nodes::Array,nodes4cells::Array)
    # uniform_refinement only works for 
    @assert size(coords4nodes,2) == 2
    @assert size(nodes4cells,2) == 3
    
    nnodes = size(coords4nodes,1);
    ncells = size(nodes4cells,1);
    
    # compute nodes4faces
    nodes4faces = [nodes4cells[:,1] nodes4cells[:,2]; nodes4cells[:,2] nodes4cells[:,3]; nodes4cells[:,3] nodes4cells[:,1]];
    # find unique rows -> this fixes the enumeration of the faces!
    sort!(nodes4faces, dims = 2); # sort each row
    nodes4faces = unique(nodes4faces, dims = 1);
    nfaces = size(nodes4faces,1);
    
    # compute and append face midpoints
    coords4nodes = [coords4nodes; 0.5*(coords4nodes[nodes4faces[:,1],:] + coords4nodes[nodes4faces[:,2],:])];
    
    # mapping to get number of new mipoint between two old nodes
    newnode4nodes = sparse(nodes4faces[:,1],nodes4faces[:,2],(1:nfaces) .+       nnodes,nnodes,nnodes);
    newnode4nodes = newnode4nodes + newnode4nodes';
    
    # build up new nodes4cells of uniform refinements
    nodes4cells_new = zeros(Int,4*ncells,3);
    newnodes = zeros(Int,3);
    for cell = 1 : ncells
        newnodes = map(j->(newnode4nodes[nodes4cells[cell,j],nodes4cells[cell,mod(j,3)+1]]),1:3);
        nodes4cells_new[(1:4) .+ (cell-1)*4,:] = 
            [nodes4cells[cell,1] newnodes[1] newnodes[3];
             newnodes[1] nodes4cells[cell,2] newnodes[2];
             newnodes[2] newnodes[3] newnodes[1];
             newnodes[3] newnodes[2] nodes4cells[cell,3]];
    end
    return coords4nodes, nodes4cells_new;
end

function ensure_area4cells!(Grid::Triangulation)
    if size(Grid.area4cells,1) != size(Grid.nodes4cells,1)
        Grid.area4cells = @views 0.5*(
               Grid.coords4nodes[Grid.nodes4cells[:,1],1] .* (Grid.coords4nodes[Grid.nodes4cells[:,2],2] - Grid.coords4nodes[Grid.nodes4cells[:,3],2])
            .+ Grid.coords4nodes[Grid.nodes4cells[:,2],1] .* (Grid.coords4nodes[Grid.nodes4cells[:,3],2] - Grid.coords4nodes[Grid.nodes4cells[:,1],2])
            .+ Grid.coords4nodes[Grid.nodes4cells[:,3],1] .* (Grid.coords4nodes[Grid.nodes4cells[:,1],2] - Grid.coords4nodes[Grid.nodes4cells[:,2],2]));
    end        
end    

# determine the face numbers of the boundary faces
function ensure_bfaces!(Grid::Triangulation)
    if size(Grid.bfaces,1) <= 0
        ensure_faces4cells!(Grid::Triangulation)
        ncells = size(Grid.faces4cells,1);    
        nfaces = size(Grid.nodes4faces,1);
        takeface = BitArray(zeros(Bool,nfaces));
        faces = [1 2 3];
        for cell = 1 : ncells
            faces = view(Grid.faces4cells,cell,:);
            takeface[faces] = .!takeface[faces];
        end
        Grid.bfaces = findall(takeface);
    end
end

# compute nodes4faces (implicating an enumeration of the faces)
function ensure_nodes4faces!(Grid::Triangulation)
    if (size(Grid.nodes4faces,1) <= 0)
        # compute nodes4faces with duplicates
        Grid.nodes4faces = [Grid.nodes4cells[:,1] Grid.nodes4cells[:,2]; Grid.nodes4cells[:,2] Grid.nodes4cells[:,3]; Grid.nodes4cells[:,3] Grid.nodes4cells[:,1]];
    
        # find unique rows -> this fixes the enumeration of the faces!
        sort!(Grid.nodes4faces, dims = 2); # sort each row
        Grid.nodes4faces = unique(Grid.nodes4faces, dims = 1);
    end    
end

# compute faces4cells
function ensure_faces4cells!(Grid::Triangulation)
    if size(Grid.faces4cells,1) != size(Grid.nodes4cells,1)
        ensure_nodes4faces!(Grid)

        nnodes = size(Grid.coords4nodes,1);
        nfaces = size(Grid.nodes4faces,1);
        ncells = size(Grid.nodes4cells,1);
    
        face4nodes = sparse(view(Grid.nodes4faces,:,1),view(Grid.nodes4faces,:,2),1:nfaces,nnodes,nnodes);
        face4nodes = face4nodes + face4nodes';
    
        Grid.faces4cells = zeros(Int,size(Grid.nodes4cells,1),3);
        for cell = 1 : ncells
            Grid.faces4cells[cell,1] = face4nodes[Grid.nodes4cells[cell,1],Grid.nodes4cells[cell,2]];
            Grid.faces4cells[cell,2] = face4nodes[Grid.nodes4cells[cell,2],Grid.nodes4cells[cell,3]];
            Grid.faces4cells[cell,3] = face4nodes[Grid.nodes4cells[cell,3],Grid.nodes4cells[cell,1]];
        end
    end    
end



end # module
