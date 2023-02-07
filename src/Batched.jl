for(gesv_batched,elty) in (
    (:magma_dgesv_batched,:Float64),
    (:magma_sgesv_batched,:Float32),
    (magma_zgesv_batched,:ComplexF64),
    (:magma_cgesv_batched,:ComplexF32)
)
@eval begin
     function gesv_batched(A::CuMatrix{$elty},B::CuMatrix{$elty})
        m,n,b=size(A)
        if n != size(B,1)
            throw(DimensionMismatch("B has a leading dimension $(size(B,1)), but needs $n"))
        end
        ipiv=similar(A,BlasInt,(n,b))
        info =Ref{BlasInt}()
        nrhs=size(B,2)
        ida=max(1,stride(A,2))
        idb=max(1,stride(B,2))
        func=eval(@funcexpr($gesv_batched))
        func(n,nrhs,A,ida,ipiv,B,idb,info,b,)
        checkmagmaerror(info[])
        return B,A,ipiv


     end
end
end