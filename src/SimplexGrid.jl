module SimplexGrid

  export Grid, getGridType, getGridDim, getSimplexNum
  
  struct Grid{T <: Real, gridDim}
    nodes:: Array{T, 1}
    elemNodes:: Array{Int, 1}
    areas:: Array{T, 1}
  end

  # constructor for structured grid
  function Grid(a::T, b::T, n::Int) where T <: Real
    nodes::Array{T, 1} = [ a + (i-1) * (b - a) / n for i=1:n+1]
    elemNodes::Array{Int, 1} = zeros(Int, 2n)
    areas::Array{T, 1} = zeros(T, n)
    
    for i=1:n
      elemNodes[2i-1] = i
      elemNodes[2i] = i+1
    end
  
    for i=1:n
      areas[i] = nodes[i+1] - nodes[i]
    end
    
    Grid{T, 1}(nodes, elemNodes, areas)
  end

  function getGridType(grid::Grid{T, gridDim}) where T <: Real where gridDim
    return T
  end
  
  function getGridDim(grid::Grid{T, gridDim}) where T <: Real where gridDim
    return gridDim
  end
  
  function getSimplexNum(grid::Grid{T, gridDim}) where T <: Real where gridDim
    return div(size(grid.elemNodes, 1), gridDim + 1)
  end
  
end
