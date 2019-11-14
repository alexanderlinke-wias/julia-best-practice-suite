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

function QuadratureFormula{T}(order::Int) where {T<:Real}
    # xref = Array{T}(undef,2)
    # w = Array{T}(undef, 1)
    
    if order <= 1 # cell midpoint rule
        xref = [1// 3 1//3 1//3]
        w = [1]
    elseif order == 2 # face midpoint rule
        xref = [1//2  1//2 0//1;
                0//1 1//2 1//2;
                1/2 0//1 1/2]
        w = [1//3; 1//3; 1//3]     
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

function cell_integrate(integrand!::Function, mesh::Grid.Triangulation, cellIndex::Int, qf::QuadratureFormula{T}, resultdim::Int=1) where {T<:Real}
   x::Array{T, 2} = zeros(T, 1, 2)
   
   dim = size(mesh.coords4nodes,2)
   sum = zeros(T, resultdim)
   result = zeros(T, resultdim)
   
   for i in eachindex(qf.w)
     fill!(x, 0)
     for j = 1 : dim
        for k = 1 : dim+1
          x[1,j] += mesh.coords4nodes[mesh.nodes4cells[cellIndex, k], j] * qf.xref[i, k]
        end
     end
     integrand!(result, x, qf.xref[i,:], cellIndex)
     sum += result * qf.w[i] * mesh.area4cells[cellIndex]
   end
   return sum
end

function integrate2!(integral4cells::Array, integrand!::Function, mesh::Grid.Triangulation, order::Int, resultdim = 1)
    ncells::Int = size(mesh.nodes4cells, 1);
    
    qf = QuadratureFormula{Float64}(order);
    
    # compute area4cells
    Grid.ensure_area4cells!(mesh);
    
    # loop over cells
    fill!(integral4cells, 0.0)
    for cell = 1 : ncells
        integral4cells[cell, :] = cell_integrate(integrand!, mesh, cell, qf, resultdim);
    end
end


# integrate a smooth function over the triangulation with arbitrary order
function integrate!(integral4cells::Array, integrand!::Function, T::Grid.Triangulation, order::Int, resultdim = 1)
    ncells::Int = size(T.nodes4cells, 1);
    
    # get quadrature point and weights
    qf = QuadratureFormula{Float64}(order);
    nqp::Int = size(qf.xref, 1);
    
    # compute area4cells
    Grid.ensure_area4cells!(T);
    
    # loop over quadrature points
    fill!(integral4cells, 0.0);
    x = zeros(Float64, ncells, 2);
    result = zeros(Float64, ncells, resultdim);
    for qp = 1 : nqp
        # map xref to x in each triangle
         x = ( qf.xref[qp,1] .* view(T.coords4nodes, view(T.nodes4cells, :, 1), :)
            + qf.xref[qp,2] .* view(T.coords4nodes, view(T.nodes4cells, :, 2), :)
            + qf.xref[qp,3] .* view(T.coords4nodes, view(T.nodes4cells, :, 3), :));
    
        # evaluate integrand multiply with quadrature weights
        integrand!(result, x, qf.xref[qp, :]) # this routine must be improved!
        integral4cells .+= result .* repeat(T.area4cells,1,resultdim) .* qf.w[qp];
    end
end

end
