using Test
push!(LOAD_PATH, "../src")

using Quadrature
println("\nStarting tests for Quadrature")
for order = 1 : 20
    println("Testing order ", order);
    @test Quadrature.QuadratureTest(order)
end


using P1approx
println("Starting tests for P1approx")
@test P1approx.P1Test1()
@test P1approx.P1Test2()

