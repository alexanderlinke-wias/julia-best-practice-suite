module FiniteElementsTests

using Grid
using FiniteElements
using LinearAlgebra

export FiniteElement



function TestFEConsistency(FE::FiniteElements.FiniteElement, cellnr, check_gradients::Bool = true)
    ndof4cell = size(FE.dofs4cells,2);
    celldim = size(FE.grid.nodes4cells,2);
    xdim = size(FE.coords4dofs,2);
    basiseval = zeros(Rational,ndof4cell,ndof4cell);
    allok = true;
    gradient_exact = zeros(Rational,ndof4cell,ndof4cell,xdim);
    gradient_FD = zeros(Rational,ndof4cell,ndof4cell,xdim);
    x = zeros(Rational,xdim+1);
    xref = zeros(Rational,xdim+1);
    for j = 1 : ndof4cell
        x = FE.coords4dofs[FE.dofs4cells[cellnr,j],:];
        for k = 1 : celldim
            if celldim == 3
                xref[k] = FiniteElements.P1FEFunctions2D(k)(x,FE.grid,cellnr);
            elseif celldim == 2
                xref[k] = FiniteElements.P1FEFunctions1D(k)(x,FE.grid,cellnr);
            end
        end    
        println("\ncoordinate of dof nr ",j);
        print("x = "); show(x); println("");
        print("xref = "); show(xref); println("");
        for k = 1 : ndof4cell
            basiseval[j,k] = FE.bfun[k](x,FE.grid,cellnr)
            FE.bfun_grad![k](view(gradient_exact,j,k,:),x,xref,FE.grid,cellnr);
            if check_gradients
                FiniteElements.FDgradient!(FE.bfun[k])(view(gradient_FD,j,k,:),x,xref,FE.grid,cellnr);
            end    
        end
        println("\neval of basis functions at dof nr ",j);
        show(basiseval[j,:]);
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
    
    println("\nVERDICT:");
    if norm(basiseval - LinearAlgebra.I(ndof4cell)) > eps(1.0)
        allok = false
        println("basis functions seem wrong");
    else
        println("basis functions seem ok");
    end
    
    if check_gradients
        if norm(gradient_exact - gradient_FD) > eps(1.0)
            allok = false
            println("gradients of basis functions seem wrong");
        else
            println("gradients of basis functions seem ok");
        end
    end    
    return allok;
end


function TestP1_1D()
    # generate reference domain
    coords4nodes_init = zeros(Rational,2,1);
    coords4nodes_init[:] = [0 1];
    nodes4cells_init = zeros(Int64,1,2);
    nodes4cells_init[1,:] = [1 2];
               
    grid = Grid.Mesh{Rational}(coords4nodes_init,nodes4cells_init);
    FE = FiniteElements.get_P1FiniteElement(grid,false);
    TestFEConsistency(FE,1,false);
end

function TestP2_1D()
    # generate reference domain
    coords4nodes_init = zeros(Rational,2,1);
    coords4nodes_init[:] = [0 1];
    nodes4cells_init = zeros(Int64,1,2);
    nodes4cells_init[1,:] = [1 2];
               
    grid = Grid.Mesh{Rational}(coords4nodes_init,nodes4cells_init);
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
               
    grid = Grid.Mesh{Rational}(coords4nodes_init,nodes4cells_init);
    FE = FiniteElements.get_P1FiniteElement(grid,false);
    TestFEConsistency(FE,1);
end

function TestP2()
    # generate reference domain
    coords4nodes_init = [0.0 0.0;
                        1.0 0.0;
                        0.0 1.0];
    nodes4cells_init = zeros(Int64,1,3);
    nodes4cells_init[1,:] = [1 2 3];
               
    grid = Grid.Mesh{Rational}(coords4nodes_init,nodes4cells_init);
    FE = FiniteElements.get_P2FiniteElement(grid,false);
    TestFEConsistency(FE,1);
end

function TestCR()
    # generate reference domain
    coords4nodes_init = [0.0 0.0;
                        1.0 0.0;
                        0.0 1.0];
    nodes4cells_init = zeros(Int64,1,3);
    nodes4cells_init[1,:] = [1 2 3];
               
    grid = Grid.Mesh{Rational}(coords4nodes_init,nodes4cells_init);
    FE = FiniteElements.get_CRFiniteElement(grid,false);
    TestFEConsistency(FE,1);
end


end # module
