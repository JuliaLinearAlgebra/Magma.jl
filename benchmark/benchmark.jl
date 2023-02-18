module MagmaBenchmarks

using CUDA
using LinearAlgebra
using BenchmarkTools
using Magma: magma_init,magma_finalize,gesvd!,geqrf!

using Plots


const SIZES =1024:256:4096
const SUITE = BenchmarkGroup(["Matrixes"])
#magma_init()
#A=cu(rand(Float32,256,256))
#r=geqrf!(A)
#magma_finalize()
function magma_svd(A)
    #magma_init()
    r=gesvd!('A','A',A)
    #magma_finalize()
    return r
end

function magma_qr(A)
    #magma_init()
    r=geqrf!(A)
    #magma_finalize()
    return r
end

g=addgroup!(SUITE,"magma")
magma_init()
for s in SIZES,elty in (Float32,)
    A=rand(elty,s,s)
    g["svd",elty,s] = @benchmarkable magma_svd($A)
    
end

for s in SIZES,elty in (Float32,)
    A=cu(rand(elty,s,s))
    CUDA.@sync begin
        g["qr",elty,s] = @benchmarkable magma_qr($A)
        
    end
      
end
magma_finalize()

g=addgroup!(SUITE,"CUDA")
for s in SIZES,elty in (Float32,)
    A=cu(rand(elty,s,s))
    A_qr=cu(copy(A))

    CUDA.@sync begin
        g["svd",elty,s] = @benchmarkable CUDA.svd($A)
        g["qr",elty,s] =  @benchmarkable CUDA.qr($A_qr) 
    end
end

#=g=addgroup!(SUITE,"LinearAlg")

for s in SIZES,elty in (Float32,)
    A=rand(elty,s,s)
    A_qr=copy(A)
    g["svd",elty,s] = @benchmarkable LinearAlgebra.svd($A)
    g["qr",elty,s] =  @benchmarkable LinearAlgebra.qr($A_qr)
end=#

magma_svd_t=[]
magma_qr_t= []
magma_init()
for s in SIZES, elty in (Float32,)
    

    time = @elapsed run(SUITE["magma"][("svd",elty,s)])
    push!(magma_svd_t,time)
    #println("done (magma_svd took $time seconds")
    #time_ = @elapsed run(SUITE["magma"][("qr",elty,s)])
    #push!(magma_qr_t,time_)
    #println("done (magma_qr took $time_ seconds")
end
magma_finalize()
cuda_svd_t=[]
cuda_qr_t= []
for s in SIZES, elty in (Float32,)

    time = @elapsed run(SUITE["CUDA"][("svd",elty,s)])
    push!(cuda_svd_t,time)
    #println("done (cuda_svd took $time seconds")
    #time_ = @elapsed run(SUITE["CUDA"][("qr",elty,s)])
    #push!(cuda_qr_t,time_)
    #println("done (cuda_qr took $time_ seconds")
end

plt = plot(SIZES,magma_svd_t,label="magma_svd",margin=5Plots.mm,marker=:auto)
#plot!(plt,SIZES,magma_qr_t,label="magma_qr",marker=:auto)
#plot!(plt,SIZES,cuda_svd_t,label="cuda_svd",marker=:auto)
plot!(plt,SIZES,cuda_svd_t,label="cuda_svd",margin=5Plots.mm,marker=:auto)
xaxis!(plt, "size (N x N)")
yaxis!(plt, "time in seconds")
savefig("magma_benchmark.png")
savefig("magma_benchmark.pdf")


end # module