module MagmaBenchmarks

using CUDA
using LinearAlgebra
using BenchmarkTools
using Magma: magma_init,magma_finalize,gesvd!,geqrf!


const SIZES =(2^4,2^8)
const SUITE = BenchmarkGroup(["Matrixes"])
function magma_svd(A)
    magma_init()
    r=gesvd!('A','A',A)
    magma_finalize()
    return r
end

function magma_qr(A)
    magma_init()
    r=geqrf!(A)
    magma_finalize()
    return r
end

g=addgroup!(SUITE,"magma")
for s in SIZES,elty in (Float32,Float64,ComplexF32,ComplexF64)
    A=rand(elty,s,s)
    A_qr=copy(A)
    g["svd",elty,s] = @benchmarkable magma_svd($A)
    g["qr",elty,s] =  @benchmarkable magma_qr($A_qr)
end

g=addgroup!(SUITE,"CUDA")
for s in SIZES,elty in (Float32,Float64,ComplexF32,ComplexF64)
    A=cu(rand(elty,s,s))
    A_qr=cu(copy(A))

    CUDA.@sync begin
        g["svd",elty,s] = @benchmarkable CUDA.svd($A)
        g["qr",elty,s] =  @benchmarkable CUDA.qr($A_qr) 
    end
end

g=addgroup!(SUITE,"LinearAlg")

for s in SIZES,elty in (Float32,Float64,ComplexF32,ComplexF64)
    A=rand(elty,s,s)
    A_qr=copy(A)
    g["svd",elty,s] = @benchmarkable LinearAlgebra.svd($A)
    g["qr",elty,s] =  @benchmarkable LinearAlgebra.qr($A_qr)
end


for s in SIZES, elty in (Float32,Float64,ComplexF32,ComplexF64)

    time = @elapsed run(SUITE["magma"][("svd",elty,s)])
    println("done (magma_svd took $time seconds")
    time_ = @elapsed run(SUITE["magma"][("qr",elty,s)])
    println("done (magma_qr took $time_ seconds")
end

for s in SIZES, elty in (Float32,Float64,ComplexF32,ComplexF64)

    time = @elapsed run(SUITE["CUDA"][("svd",elty,s)])
    println("done (cuda_svd took $time seconds")
    time_ = @elapsed run(SUITE["CUDA"][("qr",elty,s)])
    println("done (cuda_qr took $time_ seconds")
end





end # module