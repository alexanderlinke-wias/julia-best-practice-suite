module Quadrature

using LinearAlgebra
using Grid

export QuadratureFormula, integrate!, integrate2!

# struct BarycentricCoordinates{T <: Real, intDim} where intDim
# struct BarycentricCoordinates{T <: Real} where intDim
#  coords::NTuple{intDim + 1, T}
# end

mutable struct QuadratureFormula{T <: Real}
  xref::Array{T, 2}
  w::Array{T, 1}
end

function QuadratureFormula{T}(order::Int, dim::Int = 2) where {T<:Real}
    # xref = Array{T}(undef,2)
    # w = Array{T}(undef, 1)
    
    if order <= 1 # cell midpoint rule
        xref = ones(T,1,dim+1) * 1 // (dim+1)
        w = [1]
    elseif order == 2 # face midpoint rule
        if dim == 2
            xref = [1//2  1//2 0//1;
                    0//1 1//2 1//2;
                    1/2 0//1 1/2]
            w = [1//3; 1//3; 1//3]     
        elseif dim == 1
            xref = [0  1;
                    1//2 1//2;
                    1 0]
            w = [1//6; 2//3; 1//6]     
        end
    else
      xref, w = get_generic_quadrature_Stroud(order)
    end
    return QuadratureFormula{T}(xref, w)
end


  
# computes quadrature points and weights by Stroud Conical Product rule
function get_generic_quadrature_Stroud(order::Int)
    ngpts::Int = div(order, 2) + 1
    
    # compute 1D Gauss points on interval [-1,1] and weights
    gamma = (1 : ngpts-1) ./ sqrt.(4 .* (1 : ngpts-1).^2 .- ones(ngpts-1,1) );
    F = eigen(diagm(1 => gamma[:], -1 => gamma[:]));
    r = F.values;
    a = 2*F.vectors[1,:].^2;
    
    # compute 1D Gauss-Jacobi Points for Intervall [-1,1] and weights
    delta = -1 ./ (4 .* (1 : ngpts).^2 .- ones(ngpts,1));
    gamma = sqrt.((2 : ngpts) .* (1 : ngpts-1)) ./ (2 .* (2 : ngpts) .- ones(ngpts-1,1));
    F = eigen(diagm(0 => delta[:], 1 => gamma[:], -1 => gamma[:]));
    s = F.values;
    b = 2*F.vectors[1,:].^2;
    
    # transform to interval [0,1]
    r = .5 .* r .+ .5;
    s = .5 .* s .+ .5;
    a = .5 .* a';
    b = .5 .* b';
    
    # apply conical product rule
    # xref[:,[1 2]] = [ s_j , r_i(1-s_j) ] 
    # xref[:,3] = 1 - xref[:,1] - xref[:,2]
    # w = a_i*b_j
    s = repeat(s',ngpts,1)[:];
    r = repeat(r,ngpts,1);
    xref = s*[1 0 -1] - (r.*(s.-1))*[0 1 -1] + ones(length(s))*[0 0 1];
    w = a'*b;
    
    return xref, w[:]
end

function cell_integrate(integrand!::Function, grid::Grid.Mesh, cellIndex::Int, qf::QuadratureFormula{T}, resultdim::Int=1) where {T<:Real}
   x::Array{T, 2} = zeros(T, 1, 2)
   
   dim = size(grid.coords4nodes,2)
   sum = zeros(T, resultdim)
   result = zeros(T, resultdim)
   
   for i in eachindex(qf.w)
     fill!(x, 0)
     for j = 1 : dim
        for k = 1 : dim+1
          x[1,j] += grid.coords4nodes[grid.nodes4cells[cellIndex, k], j] * qf.xref[i, k]
        end
     end
     integrand!(result, x, qf.xref[i,:], cellIndex)
     sum += result * qf.w[i] * grid.volume4cells[cellIndex]
   end
   return sum
end

function integrate2!(integral4cells::Array, integrand!::Function, grid::Grid.Mesh, order::Int, resultdim = 1)
    ncells::Int = size(grid.nodes4cells, 1);
    dim::Int = size(grid.coords4nodes,2);
    
    qf = QuadratureFormula{Base.eltype(grid.coords4nodes)}(order, dim);
    
    # compute volume4cells
    Grid.ensure_volume4cells!(grid);
    
    # loop over cells
    fill!(integral4cells, 0.0)
    for cell = 1 : ncells
        try
            integral4cells[cell, :] = cell_integrate(integrand!, grid, cell, qf, resultdim);
        catch OverflowError
            println("OverflowError (due to Rationals?): trying again with Float64");
            qf = QuadratureFormula{Float64}(qf.xref,qf.w);
            integral4cells[cell, :] = cell_integrate(integrand!, grid, cell, qf, resultdim);
        end 
    end
end


# integrate a smooth function over the triangulation with arbitrary order
function integrate!(integral4cells::Array, integrand!::Function, grid::Grid.Mesh, order::Int, resultdim = 1)
    ncells::Int = size(grid.nodes4cells, 1);
    dim::Int = size(grid.coords4nodes,2);
    
    # get quadrature point and weights
    T = Base.eltype(grid.coords4nodes);
    qf = QuadratureFormula{T}(order, dim);
    nqp::Int = size(qf.xref, 1);
    
    # compute volume4cells
    Grid.ensure_volume4cells!(grid);
    
    # loop over quadrature points
    fill!(integral4cells, 0);
    x = zeros(T, ncells, 2);
    result = zeros(T, ncells, resultdim);
    for qp = 1 : nqp
        # map xref to x in each triangle
        fill!(x, 0.0);
        for d = 1 : dim+1
          x += qf.xref[qp,d] .* view(grid.coords4nodes, view(grid.nodes4cells, :, d), :);
        end
        
        # evaluate integrand multiply with quadrature weights
        integrand!(result, x, qf.xref[qp, :]) # this routine must be improved!
        integral4cells .+= result .* repeat(grid.volume4cells,1,resultdim) .* qf.w[qp];
    end
end

end
