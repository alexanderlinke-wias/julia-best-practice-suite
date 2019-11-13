using Test
push!(LOAD_PATH, "../src")

using QuadratureTests
println("\nStarting tests for Quadrature")
for order = 1 : 20
    println("Testing quadrature formula for order=", order);
    @test QuadratureTests.TestExactness(order)
end


using P1approxTests
println("Starting tests for P1approx")
@test P1approxTests.TestInterpolation()
@test P1approxTests.TestL2BestApproximation()
@test P1approxTests.TestH1BestApproximation()
@test P1approxTests.TestPoissonSolver()

