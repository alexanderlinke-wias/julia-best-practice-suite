module IntegrationTest

  using SimplexQuadrature
  using SimplexGrid

  export main, integrate

  function integrate(g::Function, grid::Grid, qf::QuadratureFormula)
    T = getGridType(grid)
    dim = getGridDim(grid)
    simplexNum = getSimplexNum(grid)
    x = zeros(T, dim)
    
    weightsNum::Int = size(qf.w, 1)

    s::T = 0
    simplexS::T = 0

    for i=1:simplexNum
      simplexS = 0
    
      index = grid.elemNodes[2i - 1]
      
      for w=1:weightsNum
        for d=1:dim
          x[d] = qf.xref[2w-1] * grid.nodes[index] + qf.xref[2w] * grid.nodes[index + 1]
        end
        simplexS += qf.w[w] * g(x[1])
      end
      s += simplexS * grid.areas[i]
    end
    
    return s
  end
 
  function f(x)
    return 1 / (1 + x^2)
  end
  
  function main(n::Int)
    qf = QuadratureFormula{Float64, 1}(2)
    intGrid = Grid(0.0, 1.0, n)
    
    @time s = integrate(f, intGrid, qf)
    @time s= integrate(f, intGrid, qf)
    println("4s = ", 4s)
  end
end
