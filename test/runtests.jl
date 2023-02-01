using Magma
using Magma: gesv!,gels!,posv!,hesv!,sysv!,geev!,gesvd!,gesdd!,getrf!,geqrf!,gebrd!,geqlf!,gelqf!,getri!,getrs!,magma_init,magma_finalize
using Test
using Random
using CUDA
using LinearAlgebra
import LinearAlgebra.LAPACK: gels! as lgels!,gesv! as lgesv!, posv! as lposv!, hesv! as lhesv!, geev! as lgeev!,gesvd! as lgesvd!,gesdd! as lgesdd!, 
getrf! as lgetrf!, geqrf! as lgeqrf!, gebrd! as lgebrd!,getri! as lgetri!,geqlf! as lgeqlf!, gelqf! as lgelqf!,getrs! as lgetrs!
#using libblastrampoline_jll

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
    @testset "gesv" begin
        @testset for elty in (Float32,Float64,ComplexF32,ComplexF64)
            #Random.seed!(913)
            A = rand(elty,10,10)
            X = rand(elty,10,10)
            A_cop=copy(A)
            X_cop=copy(X)
            expected_res=lgesv!(A,X)
            magma_init()
            #println("initiated magma")
            # we seg fault in gesv! call
            actual_res= gesv!(A_cop,X_cop)
            #println("after caling the function gesv")
            magma_finalize()
            for i in 1:3
                #println("inside the for loop")
                @test Array(actual_res[i]) ≈ Array(expected_res[i])
            end
            
            
        end
    end

    @testset "gesv_gpu" begin
        @testset for elty in (Float32,Float64,ComplexF32,ComplexF64)
            #Random.seed!(913)
            A = rand(elty,10,10)
            X = rand(elty,10,10)
            A_cu=cu(A)
            X_cu=cu(X)
            expected_res=lgesv!(A,X)
            magma_init()
            #println("initiated magma")
            # we seg fault in gesv! call
            actual_res= gesv!(A_cu,X_cu)
            #println("after caling the function gesv")
            magma_finalize()
            for i in 1:3
                #println("inside the for loop")
                @test Array(actual_res[i]) ≈ Array(expected_res[i])
            end
            
            
        end
    end

    @testset "gels" begin
        @testset for elty in (Float32,Float64,ComplexF32,ComplexF64)
            A = rand(elty,10,10)
            X = rand(elty,10,10)
            A_cop=copy(A)
            X_cop=copy(X)
            expected_res=lgels!('N',A,X)
            magma_init()
            #println("initiated magma")
            # we seg fault in gesv! call
            actual_res= gels!('N',A_cop,X_cop)
            #println("after caling the function gesv")
            magma_finalize()
            for i in 1:3
                #println("inside the for loop")
                @test Array(actual_res[i]) ≈ Array(expected_res[i])
            end 
        end
    end

    @testset "gels_gpu" begin
        @testset for elty in (Float32,Float64,ComplexF32,ComplexF64)
            A = rand(elty,10,10)
            X = rand(elty,10,10)
            A_cu=cu(A)
            X_cu=cu(X)
            expected_res=lgels!('N',A,X)
            magma_init()
            actual_res= gels!('N',A_cu,X_cu)
            magma_finalize()
            for i in 1:3
                @test Array(actual_res[i]) ≈ Array(expected_res[i])
            end 
        end
    end

    @testset "posv" begin
        @testset for elty in (Float32,Float64,ComplexF32,ComplexF64)
            A = rand(elty,10,10)/100
            A += real(diagm(0 => 10*real(rand(elty,10))))
            if elty <: Complex
                A = A + A'
            else
                A = A + transpose(A)
            end
            X = rand(elty,10,10)
            A_cop = copy(A)
            X_cop = copy(X)
            expected_res=lposv!('U',A,X)
            magma_init()
            actual_res= posv!('U',A_cop,X_cop)
            magma_finalize()
            for i in 1:2
                @test Array(actual_res[i]) ≈ Array(expected_res[i])
            end 
        end
    end

    @testset "posv_gpu" begin
        @testset for elty in (Float32,Float64,ComplexF32,ComplexF64)
            A = rand(elty,10,10)/100
            A += real(diagm(0 => 10*real(rand(elty,10))))
            if elty <: Complex
                A = A + A'
            else
                A = A + transpose(A)
            end
            X = rand(elty,10,10)
            A_cu= cu(A)
            X_cu = cu(X)
            expected_res=lposv!('U',A,X)
            magma_init()
            actual_res= posv!('U',A_cu,X_cu)
            magma_finalize()
            for i in 1:2
                @test Array(actual_res[i]) ≈ Array(expected_res[i])
            end 
        end
    end
    @testset "hesv" begin
        @testset for elty in (ComplexF32,ComplexF64)
            A = rand(elty,10,10)
            A = A + A'
            X = rand(elty,10)
            A_cop = copy(A)
            X_cop = copy(X)
            expected_res=lhesv!('U',A,X)
            magma_init()
            #println("initiated magma")
            # we seg fault in gesv! call
            actual_res= hesv!('U',A_cop,X_cop)
            #println("after caling the function gesv")
            magma_finalize()
            for i in 1:3
                #println("inside the for loop")
                @test Array(actual_res[i]) ≈ Array(expected_res[i])
            end 
        end
    end

    @testset "sysv" begin
        @testset for elty in (Float32,Float64)
            A = rand(elty,10,10)
            A = A + transpose(A)
            X = rand(elty,10)
            A_cop = copy(A)
            X_cop = copy(X)
            expected_res= A \ X
            magma_init()
            #println("initiated magma")
            # we seg fault in gesv! call
            actual_res= sysv!('U',A_cop,X_cop)
            #println("after caling the function gesv")
            magma_finalize()
            @test actual_res[1] ≈ expected_res
        end
    end

    @testset "geev!" begin
        @testset for elty in (Float32,Float64,ComplexF32,ComplexF64)
            A = rand(elty,10,10)
            A_cop=copy(A)
            expect_res= lgeev!('V','V',A)
            
            magma_init()
            actual_res=geev!('V','V',A_cop)
            magma_finalize()
            #println("yoni")
            for i in 1:length(actual_res)
                @test Array(actual_res[i]) ≈ Array(expect_res[i])
            end
        end
    end
    @testset "gesvd" begin
        @testset for elty in (Float32,Float64,ComplexF32,ComplexF64)
            A = rand(elty,10,5)
            A_cop=copy(A)
            expect_res= lgesvd!('A','A',A)
            
            magma_init()
            actual_res=gesvd!('A','A',A_cop)
            magma_finalize()
            for i in 1:length(actual_res)
                @test Array(actual_res[i]) ≈ Array(expect_res[i])
            end
        end
    end
    @testset "gesdd" begin
        @testset for elty in (Float32,Float64,ComplexF32,ComplexF64)
            A = rand(elty,10,5)
            A_cop=copy(A)
            expect_res= lgesdd!('A',A)
            
            magma_init()
            actual_res=gesdd!('A',A_cop)
            magma_finalize()
            for i in 1:length(actual_res)
                @test Array(actual_res[i]) ≈ Array(expect_res[i])
            end
        end
    end

    @testset "getrf" begin
        @testset for elty in (Float32,Float64,ComplexF32,ComplexF64)
            A = rand(elty,10,10)
            iA = inv(A)
            magma_init()
            A, ipiv = getrf!(A)
            magma_finalize()
            A = lgetri!(Array(A), ipiv)
            @test A ≈ iA
        end
    end

    @testset "getrf_gpu" begin
        @testset for elty in (Float32,Float64,ComplexF32,ComplexF64)
            A = rand(elty,10,10)
            iA = inv(A)
            A_cu=cu(A)
            magma_init()
            A, ipiv = getrf!(A_cu)
            magma_finalize()
            A = lgetri!(Array(A), Array(ipiv))
            @test A ≈ iA
        end
    end
    @testset "getri_gpu" begin
        @testset for elty in (Float32,Float64,ComplexF32,ComplexF64)
            A = rand(elty,10,10)
            A_l=copy(A)


            A,ipiv,info= lgetrf!(A)
            A=lgetri!(A,ipiv)

            A_l,ipiv_l,info_l=lgetrf!(A_l)

            A_l=cu(A_l)
            magma_init()
            A_l=getri!(A_l,ipiv_l)
            magma_finalize()
            
            @test Array(A_l) ≈ Array(A)

        end
    end

    @testset "getrs_gpu" begin
        @testset for elty in (Float32,Float64,ComplexF32,ComplexF64)
            A = rand(elty,10,10)
            B = rand(elty,10,10)
            A_l=copy(A)
            B_l=copy(B)


            A,ipiv,info= lgetrf!(A)
            B=lgetrs!('N',A,ipiv,B)

            A_l,ipiv_l,info_l=lgetrf!(A_l)

            A_l=cu(A_l)
            B_l=cu(B_l)
            magma_init()
            B_l=getrs!('N',A_l,ipiv_l,B_l)
            magma_finalize()
            
            @test Array(B_l) ≈ Array(B)

        end
    end

    @testset "geqrf" begin
        @testset for elty in (Float32,Float64,ComplexF32,ComplexF64)
            A = rand(elty,10,10)
            A_cop=copy(A)
            expect_res= lgeqrf!(A)
            magma_init()
            actual_res=geqrf!(A_cop)
            magma_finalize()
            for i in 1:length(actual_res)
                @test Array(actual_res[i]) ≈ Array(expect_res[i])
            end

            
        end
    end

    @testset "geqrf_gpu" begin
        @testset for elty in (Float32,Float64,ComplexF32,ComplexF64)
            A = rand(elty,10,10)
            A_cu=cu(A)
            expect_res= lgeqrf!(A)
            magma_init()
            actual_res=geqrf!(A_cu)
            magma_finalize()
            for i in 1:length(actual_res)
                @test Array(actual_res[i]) ≈ Array(expect_res[i])
            end

            
        end
    end

    #=@testset "gebrd" begin
        @testset for elty in (Float32,Float64,ComplexF32,ComplexF64) 
            A = rand(elty,5,5)
            A_cop=copy(A)
            expect_res= lgebrd!(A)
            #idx=[1,4,5]
            magma_init()
            actual_res=gebrd!(A_cop)
            magma_finalize()
            for i in 1:length(actual_res)
                @test Array(actual_res[i]) ≈ Array(expect_res[i])
            end
        end
    end=#

    @testset "geqlf" begin
        @testset for elty in (Float32,Float64,ComplexF32,ComplexF64)
            A = rand(elty,10,10)
            A_cop=copy(A)
            expect_res= lgeqlf!(A)
            magma_init()
            actual_res=geqlf!(A_cop)
            magma_finalize()
            for i in 1:length(actual_res)
                @test Array(actual_res[i]) ≈ Array(expect_res[i])
            end    
        end
    end

    @testset "gelqf" begin
        @testset for elty in (Float32,Float64,ComplexF32,ComplexF64)
            A = rand(elty,10,10)
            A_cop=copy(A)
            expect_res= lgelqf!(A)
            magma_init()
            actual_res=gelqf!(A_cop)
            magma_finalize()
            for i in 1:length(actual_res)
                @test Array(actual_res[i]) ≈ Array(expect_res[i])
            end    
        end
    end



end
