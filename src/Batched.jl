for(gesv_batched,elty) in (
    (:magma_dgesv_batched,:Float64),
    (:magma_sgesv_batched,:Float32),
    (:magma_zgesv_batched,:ComplexF64),
    (:magma_cgesv_batched,:ComplexF32)
)
@eval begin
     function gesv_batched!(A::Vector{<:StridedCuMatrix{$elty}},B::Vector{<:StridedCuMatrix{$elty}})

        if length(A) != length(B)
            throw(DimensionMismatch(""))
        end
        batch_count=length(A)
        n=size(A[1],2)
        ipiv=similar(A,BlasInt,(n,batch_count))
        info =similar(A,BlasInt,batch_count)
        queue= Ref{LibMagma.magma_queue_t}()
        LibMagma.magma_queue_create_internal(CUDA.device().handle, queue, "", "", 0)
        nrhs=size(B[1],2)
        ida = max(1,stride(A[1],2))
        idb = max(1,stride(B[1],2))
        Aptrs = CUDA.CUBLAS.unsafe_batch(A)
        Bptrs = CUDA.CUBLAS.unsafe_batch(B)
        LibMagma.$gesv_batched(n,nrhs,Aptrs,ida,ipiv,Bptrs,idb,info,batch_count,queue)

        info=Array(info)
        for i in 1:batch_count
            checkmagmaerror(info[i])


        end
        
        CUDA.unsafe_free!(Bptrs)
        CUDA.unsafe_free!(Aptrs)
        return B,A,ipiv


     end #function

end #eval


end#for loop