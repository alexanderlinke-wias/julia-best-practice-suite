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
    println("Loading quadrature formula of order ",order);
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

function integrate!(integral4cells::Array, integrand!::Function, grid::Grid.Mesh, order::Int, resultdim = 1)
    ncells::Int = size(grid.nodes4cells, 1);
    celldim::Int = size(grid.nodes4cells,2)-1;
    xdim::Int = size(grid.coords4nodes,2);
    
    T = Base.eltype(grid.coords4nodes);
    qf = QuadratureFormula{T}(order, celldim);
    
    # compute volume4cells
    Grid.ensure_volume4cells!(grid);
    
    # loop over cells
    fill!(integral4cells, 0.0)
    x::Array{T, 2} = zeros(T, 1, xdim)
    result = zeros(T, resultdim)
    for cell = 1 : ncells
      for i in eachindex(qf.w)
        fill!(x, 0)
        for j = 1 : xdim
          for k = 1 : celldim + 1
            x[1,j] += grid.coords4nodes[grid.nodes4cells[cell, k], j] * qf.xref[i, k]
          end
        end
        integrand!(result, x, view(qf.xref,i,:), cell)
        for j = 1 : resultdim
          integral4cells[cell, j] += result[j] * qf.w[i] * grid.volume4cells[cell];
        end
      end  
    end
end


# integrate a smooth function over the triangulation with arbitrary order
function integrate_old!(integral4cells::Array, integrand!::Function, grid::Grid.Mesh, order::Int, resultdim = 1)
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
    x = zeros(T, ncells, dim);
    result = zeros(T, ncells, resultdim);
    for qp = 1 : nqp
        # map xref to x in each triangle
        fill!(x, 0.0);
        for d = 1 : dim+1
          x += qf.xref[qp,d] .* view(grid.coords4nodes, view(grid.nodes4cells, :, d), :);
        end
        
        # evaluate integrand multiply with quadrature weights
        integrand!(result, x, qf.xref[qp, :])
        for j = 1 : resultdim
            @inbounds integral4cells[:,j] += result[:,j] .* grid.volume4cells * qf.w[qp];
        end    
    end
end

end
