using Test
push!(LOAD_PATH, "../src")

using Quadrature
println("\nStarting tests for Quadrature")
for order = 1 : 20
    println("Testing quadrature formula for order=", order);
    @test Quadrature.QuadratureTest(order)
end


using P1approx
println("Starting tests for P1approx")
@test P1approx.TestInterpolation()
@test P1approx.TestL2BestApproximation()

