using Main.Magma
using Main.Magma: gesv!,gels!
using Test
using Main.LibMagma
using LinearAlgebra,Random

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
    @testset "gels" begin
        @testset for elty in (Float32, Float64, ComplexF32, ComplexF64)
            Random.seed!(913)
            A = rand(elty,10,10)
            X = rand(elty,10)
            magma_init()
            Y= gels!('N',copy(A),copy(X))
            magma_finalize()
            @test A\X ≈ Y
            
        end
    end
end
