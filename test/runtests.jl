using Magma
using Magma: gesv!,gels!,magma_init,magma_finalize
using Test
using Random
#using libstramopil
#using Main.LibMagma
import LinearAlgebra.LAPACK: gels! as lgels!,gesv! as lgesv!
#using libblastrampoline_jll

@testset "Magma.jl" begin
    # Write your tests here.
    @testset "gesv" begin
        @testset for elty in (Float32,)
            #Random.seed!(913)
            A = rand(elty,10,10)
            X = rand(elty,10,10)
            A_cop=copy(A)
            X_cop=copy(X)
            expected_res=lgesv!(A,X)
            magma_init()
            println("initiated magma")
            # we seg fault in gesv! call
            actual_res= gesv!(A_cop,X_cop)
            println("after calling the function gesv")
            magma_finalize()
            @test Array(actual_res[1]) ≈ Array(expected_res[1])
            
        end
    end
end
