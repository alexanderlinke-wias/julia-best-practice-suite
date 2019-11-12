using Test

push!(LOAD_PATH, "../src")
using P1approx

println("Starting tests for P1approx")
@test P1approx.P1Test1()
@test P1approx.P1Test2()
