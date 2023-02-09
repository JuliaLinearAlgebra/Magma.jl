for(gesv_batched,set_device,get_device,queue_create,elty) in (
    (:magma_dgesv_batched,:magma_setdevice,:magma_getdevice,:magma_queue_create_internal,:Float64),
    (:magma_sgesv_batched,:magma_setdevice,:magma_getdevice,:magma_queue_create_internal,:Float32),
    (:magma_zgesv_batched,:magma_setdevice,:magma_getdevice,:magma_queue_create_internal,:ComplexF64),
    (:magma_cgesv_batched,:magma_setdevice,:magma_getdevice,:magma_queue_create_internal,:ComplexF32)
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
        queue= Ref{magma_queue_t}()

        device=Ref{BlasInt}()
        f=Ref{Char}()
        line=Ref{Char}()
        func_devset=eval(@funcexpr($set_device))
        func_devget=eval(@funcexpr($get_device))
        creat_q=eval(@funcexpr($queue_create))
        func_devget(device)
        func_devset(device[])
        creat_q(device[],queue,"","",Cint(1))


        nrhs=size(B[1],2)
        ida = max(1,stride(A[1],2))
        idb = max(1,stride(B[1],2))
        Aptrs = CUDA.CUBLAS.unsafe_batch(A)
        Bptrs = CUDA.CUBLAS.unsafe_batch(B)
        func=eval(@funcexpr($gesv_batched))
        func(n,nrhs,Aptrs,ida,ipiv,Bptrs,idb,info,batch_count,queue[])
        info=Array(info)
        for i in 1:batch_count
            checkmagmaerror(info[i])

        end
        
        CUDA.unsafe_free!(Bptrs)
        CUDA.unsafe_free!(Aptrs)
        magma
        return B,A,ipiv


     end #function

end #eval


end#for loop