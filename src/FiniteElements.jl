module FiniteElements

using Grid
using LinearAlgebra
using ForwardDiff


# Finite Element structure
# = container for set of basis functions and their gradients
#   local dof numbers, coordinates for dofs etc.
#
# todo/ideas:
# - define dual functionals for dofs?
# - Hdiv spaces (RT, BDM)
# - Hcurl spaces (Nedelec)
# - elasticity (Kouhia-Stenberg)
# - Stokes (MINI, P2B)
# - reconstruction (2nd set of bfun?)
struct FiniteElement{T <: Real}
    name::String; # name of finite element (used in messages)
    grid::Grid.Mesh; # link to grid
    polynomial_order::Int; # polyonomial degree of basis functions (used for quadrature)
    ncomponents::Int; # length of return value ofr basis functions, 1 = scalar, >1 vector-valued
    dofs4cells::Array{Int64,2}; # dof numbers for each cell
    dofs4faces::Array{Int64,2}; # dof numbers for each face
    coords4dofs::Array{T,2}; # coordinates for degrees of freedom
    bfun_ref::Vector{Function}; # basis functions evaluated in local coordinates
    bfun::Vector{Function}; # basis functions evaluated in global coordinates
    bfun_grad!::Vector{Function}; # gradients of basis functions (either exactly given, or ForwardDiff of bfun)
    local_mass_matrix::Array{T,2}; # some elements have same local MAMA on each cell, if defined used for faster mass matrix calculation
end

   
# wrapper for ForwardDiff & DiffResults
function FDgradient(bfun::Function, x::Vector{T}, xdim = 1) where T <: Real
    if xdim == 1
        DRresult = DiffResults.GradientResult(Vector{T}(undef, length(x)));
    else
        DRresult = DiffResults.DiffResult(Vector{T}(undef, length(x)),Matrix{T}(undef,length(x),xdim));
    end
    function closure(result,x,xref,grid,cell)
        f(a) = bfun(a,grid,cell);
        if xdim == 1
            ForwardDiff.gradient!(DRresult,f,x);
        else
            ForwardDiff.jacobian!(DRresult,f,x);
        end    
        result[:] = DiffResults.gradient(DRresult);
    end    
end    
                  
   
 #######################################################################################################
 #######################################################################################################
 ### FFFFF II NN    N II TTTTTT EEEEEE     EEEEEE LL     EEEEEE M     M EEEEEE NN    N TTTTTT SSSSSS ###
 ### FF    II N N   N II   TT   EE         EE     LL     EE     MM   MM EE     N N   N   TT   SS     ###
 ### FFFF  II N  N  N II   TT   EEEEE      EEEEE  LL     EEEEE  M M M M EEEEE  N  N  N   TT    SSSS  ###
 ### FF    II N   N N II   TT   EE         EE     LL     EE     M  M  M EE     N   N N   TT       SS ###
 ### FF    II N    NN II   TT   EEEEEE     EEEEEE LLLLLL EEEEEE M     M EEEEEE N    NN   TT   SSSSSS ###
 #######################################################################################################
 #######################################################################################################

 include("FiniteElements_Lagrange.jl")
 include("FiniteElements_CrouzeixRaviart.jl")
 include("FiniteElements_BernardiRaugel.jl")
 include("FiniteElements_MINI.jl")
 


end # module
