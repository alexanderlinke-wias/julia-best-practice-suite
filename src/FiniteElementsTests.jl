module FiniteElementsTests

using Grid
using FiniteElements
using LinearAlgebra
using DiffResults
using Quadrature

export FiniteElement



function compute_local_mass_matrix(bfun_ref::Vector{Function}, xdim, poly_order, grid, cellnr)
    qf = QuadratureFormula{Float64}(2*poly_order, xdim);
    
    ndof::Int = length(bfun_ref);
    local_mass_matrix = zeros(Float64,ndof,ndof);
    basis_eval = zeros(Float64,ndof);
    
    for i in eachindex(qf.w)
        for k = 1 : ndof
            basis_eval[k] = bfun_ref[k](qf.xref[i],grid,cellnr);
        end    
        for k = 1 : ndof
            for j = 1 : ndof
                local_mass_matrix[j,k] += basis_eval[j] * basis_eval[k] * qf.w[i];
            end
        end
    end
    
    return local_mass_matrix;
end


function TestFEConsistency(FE::FiniteElements.FiniteElement, cellnr, check_gradients::Bool = true)
    ndof4cell = size(FE.dofs4cells,2);
    celldim = size(FE.grid.nodes4cells,2);
    xdim = size(FE.coords4dofs,2);
    T = eltype(FE.grid.coords4nodes);
    basiseval_ref = zeros(T,ndof4cell,ndof4cell);
    basiseval = zeros(T,ndof4cell,ndof4cell);
    allok = true;
    gradient_exact = zeros(T,ndof4cell,ndof4cell,xdim*FE.ncomponents);
    gradient_FD = zeros(T,ndof4cell,ndof4cell,xdim*FE.ncomponents);
    x = zeros(T,xdim);
    xref = zeros(T,xdim+1);
    if check_gradients
        FDgradients = Vector{Function}(undef,ndof4cell);
        for j = 1 : ndof4cell
            FDgradients[j] = FiniteElements.FDgradient(FE.bfun[j],x,FE.ncomponents)
        end    
    end
    for j = 1 : ndof4cell
        x = FE.coords4dofs[FE.dofs4cells[cellnr,j],:];
        for k = 1 : celldim
            if celldim == 3
                xref[k] = FiniteElements.P1basis_2D[k](x,FE.grid,cellnr);
            elseif celldim == 2
                xref[k] = FiniteElements.P1basis_1D[k](x,FE.grid,cellnr);
            end
        end    
        println("\ncoordinate of dof nr ",j);
        print("x = "); show(x); println("");
        print("xref = "); show(xref); println("");
        for k = 1 : ndof4cell
            basiseval[j,k] = maximum(FE.bfun[k](x,FE.grid,cellnr))
            basiseval_ref[j,k] = maximum(FE.bfun_ref[k](xref,FE.grid,cellnr))
            FE.bfun_grad![k](view(gradient_exact,j,k,:),x,xref,FE.grid,cellnr);
            if check_gradients
                FDgradients[k](view(gradient_FD,j,k,:),x,xref,FE.grid,cellnr);
            end    
        end
        println("\neval of basis functions at dof nr ",j);
        show(basiseval[j,:]);
        println("\neval of basis functions in xref at dof nr ",j);
        show(basiseval_ref[j,:]);
        println("\neval of active gradients of basis functions at dof nr ",j);
        for k = 1 : ndof4cell
            show(gradient_exact[j,k,:]); println("");
        end    
        if check_gradients
            println("eval of ForwardDiff gradients of basis functions at dof nr ",j);
            for k = 1 : ndof4cell
                show(gradient_FD[j,k,:]); println("");
            end
        end
    end
    #A = compute_local_mass_matrix(FE.bfun_ref,xdim,FE.polynomial_order,FE.grid,cellnr);
    #println("\n local mass matrix from quadrature:");
    #show(A)
    #println("\n local mass matrix from finite element:");
    #show(Array{Float64,2}(FE.local_mass_matrix));
    #println("");
    
    println("\nVERDICT:");
    if norm(basiseval - LinearAlgebra.I(ndof4cell)) > eps(1.0)
        allok = false
        println("basis functions seem wrong");
    else
        println("basis functions seem ok");
    end
    
    if norm(basiseval_ref - LinearAlgebra.I(ndof4cell)) > eps(1.0)
        allok = false
        println("basis functions in xref seem wrong");
    else
        println("basis functions in xref seem ok");
    end
    
    if check_gradients
        if norm(gradient_exact - gradient_FD) > eps(1.0)
            allok = false
            println("gradients of basis functions seem wrong");
        else
            println("gradients of basis functions seem ok");
        end
    end    
    
    #if norm(A - FE.local_mass_matrix) > eps(10.0)
    #    allok = false
    #    println("local mass matrix seems wrong");
    #else
    #    println("local mass matrix seems ok");
    #end
    return allok;
end


function TestP0()
    # generate reference domain
    coords4nodes_init = [0.0 0.0;
                        1.0 0.0;
                        0.0 1.0];
    nodes4cells_init = zeros(Int64,1,3);
    nodes4cells_init[1,:] = [1 2 3];
               
    grid = Grid.Mesh{Rational{Int64}}(coords4nodes_init,nodes4cells_init);
    FE = FiniteElements.get_P0FiniteElement(grid);
    TestFEConsistency(FE,1,false);
end

function TestP1_1D()
    # generate reference domain
    coords4nodes_init = zeros(Rational,2,1);
    coords4nodes_init[:] = [0 1];
    nodes4cells_init = zeros(Int64,1,2);
    nodes4cells_init[1,:] = [1 2];
               
    grid = Grid.Mesh{Rational{Int64}}(coords4nodes_init,nodes4cells_init);
    FE = FiniteElements.get_P1FiniteElement(grid,false);
    TestFEConsistency(FE,1,false);
end

function TestP2_1D()
    # generate reference domain
    coords4nodes_init = zeros(Rational,2,1);
    coords4nodes_init[:] = [0 1];
    nodes4cells_init = zeros(Int64,1,2);
    nodes4cells_init[1,:] = [1 2];
               
    grid = Grid.Mesh{Rational{Int64}}(coords4nodes_init,nodes4cells_init);
    FE = FiniteElements.get_P2FiniteElement(grid,false);
    TestFEConsistency(FE,1,false);
end

function TestP1()
    # generate reference domain
    coords4nodes_init = [0.0 0.0;
                        1.0 0.0;
                        0.0 1.0];
    nodes4cells_init = zeros(Int64,1,3);
    nodes4cells_init[1,:] = [1 2 3];
               
    grid = Grid.Mesh{Rational{Int64}}(coords4nodes_init,nodes4cells_init);
    FE = FiniteElements.get_P1FiniteElement(grid,false);
    TestFEConsistency(FE,1);
end


function TestP1V()
    # generate reference domain
    coords4nodes_init = [0.0 0.0;
                        1.0 0.0;
                        0.0 1.0];
    nodes4cells_init = zeros(Int64,1,3);
    nodes4cells_init[1,:] = [1 2 3];
               
    grid = Grid.Mesh{Rational{Int64}}(coords4nodes_init,nodes4cells_init);
    FE = FiniteElements.get_P1VectorFiniteElement(grid,false);
    TestFEConsistency(FE,1,false);
end


function TestBR()
    # generate reference domain
    coords4nodes_init = [0.0 0.0;
                        1.0 0.0;
                        0.0 1.0];
    nodes4cells_init = zeros(Int64,1,3);
    nodes4cells_init[1,:] = [1 2 3];
               
    grid = Grid.Mesh{Rational{Int64}}(coords4nodes_init,nodes4cells_init);
    FE = FiniteElements.get_BRFiniteElement(grid,false);
    TestFEConsistency(FE,1,false);
end

function TestP2()
    # generate reference domain
    coords4nodes_init = [0.0 0.0;
                        1.0 0.0;
                        0.0 1.0];
    nodes4cells_init = zeros(Int64,1,3);
    nodes4cells_init[1,:] = [1 2 3];
               
    grid = Grid.Mesh{Rational{Int64}}(coords4nodes_init,nodes4cells_init);
    FE = FiniteElements.get_P2FiniteElement(grid,false);
    @time TestFEConsistency(FE,1);
end

function TestP2V()
    # generate reference domain
    coords4nodes_init = [0.0 0.0;
                        1.0 0.0;
                        0.0 1.0];
    nodes4cells_init = zeros(Int64,1,3);
    nodes4cells_init[1,:] = [1 2 3];
               
    grid = Grid.Mesh{Rational{Int64}}(coords4nodes_init,nodes4cells_init);
    FE = FiniteElements.get_P2VectorFiniteElement(grid,false);
    @time TestFEConsistency(FE,1,false);
end

function TestCR()
    # generate reference domain
    coords4nodes_init = [0.0 0.0;
                        1.0 0.0;
                        0.0 1.0];
    nodes4cells_init = zeros(Int64,1,3);
    nodes4cells_init[1,:] = [1 2 3];
               
    grid = Grid.Mesh{Rational{Int64}}(coords4nodes_init,nodes4cells_init);
    FE = FiniteElements.get_CRFiniteElement(grid,false);
    TestFEConsistency(FE,1);
end


function TestCR_3D()
    # generate reference domain
    coords4nodes_init = [0.0 0.0 0.0;
                        1.0 0.0 0.0;
                        0.0 1.0 0.0;
                        0.0 0.0 1.0];
    nodes4cells_init = zeros(Int64,1,4);
    nodes4cells_init[1,:] = [1 2 3 4];
               
    grid = Grid.Mesh{Rational{Int64}}(coords4nodes_init,nodes4cells_init);
    FE = FiniteElements.get_CRFiniteElement(grid,false);
    TestFEConsistency(FE,1);
end


end # module
