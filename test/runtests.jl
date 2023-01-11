using .Magma
using .Magma: gesv!,gels!
using Test
using LinearAlgebra

@testset "Magma.jl" begin
    # Write your tests here.
    @testset "dimension mistach" begin
        for elty in (Float32, Float64, ComplexF32, ComplexF64)
            A8x8, B9x9 = Matrix{elty}.(undef, ((8,8), (9,9)))
            @test_throws DimensionMismatch gels!('N',A8x8,B9x9)
            @test_throws DimensionMismatch gels!('T',A8x8,B9x9)
            @test_throws DimensionMismatch gesv!(A8x8,B9x9)
        end
    end
end
