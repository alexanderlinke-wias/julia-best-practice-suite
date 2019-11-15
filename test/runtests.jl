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


using P1approxTests
println("Starting tests for P1approx")
@test P1approxTests.TestInterpolation1D()
@test P1approxTests.TestL2BestApproximation1D()

@test P1approxTests.TestInterpolation2D()
@test P1approxTests.TestL2BestApproximation2D()
@test P1approxTests.TestH1BestApproximation()
@test P1approxTests.TestPoissonSolver()


