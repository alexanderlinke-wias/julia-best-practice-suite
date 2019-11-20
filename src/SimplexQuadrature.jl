module SimplexQuadrature

  export QuadratureFormula

  struct QuadratureFormula{T <: Real, integrationDim}
    xref::Array{T, 2}
    w::Array{T, 1}
  end

  function QuadratureFormula{T, 1}(order::Int) where {T<:Real}
    if order <= 1
      xref = [1 // 2 1 // 2]'
      w = [1]
    else
      xref = [1 1 // 2 0; 0 1 // 2 1]
      w = [1 // 6; 2 // 3; 1 // 6]
    end
    
    return QuadratureFormula{T, 1}(xref, w)
  end

  function QuadratureFormula{T, 2}(order::Int) where {T<:Real}
    if order <= 1
      xref = [1// 3 1//3 1//3]'
      w = [1]
    else
        xref = [1//2  1//2 0//1;
                0//1 1//2 1//2;
                1//2 0//1 1/2]'
        w = [1//3; 1//3; 1//3]
    end
    
    return QuadratureFormula{T, 2}(xref, w)
  end
end
