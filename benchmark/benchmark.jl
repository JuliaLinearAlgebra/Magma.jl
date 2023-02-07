module MagmaBenchmarks

using CUDA
using LinearAlgebra
using BenchmarkTools
using Magma: magma_init,magma_finalize,gesvd!,geqrf!

using Plots


const SIZES =(2^6,2^8,2^10)
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

magma_svd_t=[]
magma_qr_t= []
for s in SIZES, elty in (Float32,Float64,ComplexF32,ComplexF64)

    time = @elapsed run(SUITE["magma"][("svd",elty,s)])
    push!(magma_svd_t,time)
    println("done (magma_svd took $time seconds")
    time_ = @elapsed run(SUITE["magma"][("qr",elty,s)])
    push!(magma_qr_t,time_)
    println("done (magma_qr took $time_ seconds")
end
cuda_svd_t=[]
cuda_qr_t= []
for s in SIZES, elty in (Float32,Float64,ComplexF32,ComplexF64)

    time = @elapsed run(SUITE["CUDA"][("svd",elty,s)])
    push!(cuda_svd_t,time)
    println("done (cuda_svd took $time seconds")
    time_ = @elapsed run(SUITE["CUDA"][("qr",elty,s)])
    push!(cuda_qr_t,time_)
    println("done (cuda_qr took $time_ seconds")
end

display(plot!(magma_svd_t,label="magma_svd"))
display(plot!(magma_qr_t,label="magma_qr"))
display(plot!(cuda_svd_t,label="cuda_svd"))
display(plot!(cuda_qr_t,label="cuda_qr"))



end # module