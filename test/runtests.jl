using Main.Magma
using Main.Magma: gesv!,gels!,magma_init,magma_finalize
using Test
using Random
#using Main.LibMagma
#import LinearAlgebra.LAPACK: gels! as lgels!,gesv! as lgesv!

@testset "Magma.jl" begin
    # Write your tests here.
    @testset "dimension mistach" begin
        for elty in (Float32, Float64, ComplexF32, ComplexF64)
            A8x8, B9x9 = Matrix{elty}.(undef, ((8,8), (9,9)))
            @test_throws DimensionMismatch gels!('N',A8x8,B9x9)
            @test_throws DimensionMismatch gels!('T',A8x8,B9x9)
            #@test_throws DimensionMismatch gesv!(A8x8,B9x9)
        end
    end
    @testset "gels" begin
        @testset for elty in (Float32, Float64, ComplexF32, ComplexF64)
            Random.seed!(913)
            A = rand(elty,10,10)
            X = rand(elty,10)
            A_cop=copy(A)
            X_cop=copy(X)
            #Y_e=lgels!('N',A,X)
            magma_init()
            Y= gels!('N',A_cop,X_cop)
            magma_finalize()
            #println(Y)
            #println(Y_e)
            #@test Array(Y) ≈ Array(Y_e[2])
            
        end
    end
end
