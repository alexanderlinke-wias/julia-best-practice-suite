using Test
push!(LOAD_PATH, "../src")


using QuadratureTests
println("\nStarting tests for 1D Quadrature")
for order = 1 : 2
    println("Testing quadrature formula for order=", order);
    @test QuadratureTests.TestExactness(order,1)
end
println("\nStarting tests for 2D Quadrature")
for order = 1 : 20
    println("Testing quadrature formula for order=", order);
    @test QuadratureTests.TestExactness(order,2)
end


using FiniteElementsTests
println("\nStarting tests for FiniteElements")
@test FiniteElementsTests.TestP1()
@test FiniteElementsTests.TestP2()
@test FiniteElementsTests.TestCR()


using FESolveTests
println("\nStarting tests for FESolve")
@test FESolveTests.TestInterpolation1D()
@test FESolveTests.TestL2BestApproximation1D()
@test FESolveTests.TestL2BestApproximation1DBoundaryGrid()
@test FESolveTests.TestH1BestApproximation1D()
@test FESolveTests.TestPoissonSolver1D()

@test FESolveTests.TestInterpolation2D()
@test FESolveTests.TestL2BestApproximation2D()
@test FESolveTests.TestH1BestApproximation2D()
@test FESolveTests.TestPoissonSolver2D()


