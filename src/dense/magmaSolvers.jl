for(gels,gesv,elty) in (
    (:magma_dgels,:magma_dgesv,:Float64),
    (:magma_sgels,:magma_sgesv,:Float32),
    (:magma_zgels,:magma_zgesv,:ComplexF64),
    (:magma_cgels,:magma_cgesv,:ComplexF32)
)

@eval begin
    function gels!(trans::AbstractChar,A::AbstractMatrix{$elty},B::AbstractVecOrMat{$elty})
        checktranspose(trans)
        m,n =size(A)
        btrn= trans == 'N'
        if size(B,1) != (btrn ? n : m)
            throw(DimensionMismatch("matrix has dimensions ($m,$n), transposed: $btrn but the leading dimension of B is $(size(B,1))"))
        end
        info =Ref{BlasInt}()
        nrhs=size(B,2)
        ida=max(1,stride(A,2))
        idb=max(1,stride(B,2))
        work=Vector{$elty}(undef,1)
        lwork=BlasInt(-1)
        #println(gels)
        for i = 1:2
            func=eval(@funcexpr($gels))
            func(MagmaNoTrans,m,n,nrhs,A,ida,B,idb,work,lwork,info)
            checkmagmaerror(info[])
            if i==1
               lwork=ceil(BlasInt,real(work[1]))
               resize!(work,lwork)
            end

        end
        k   = min(m, n)
        F   = m < n ? tril(A[1:k, 1:k]) : triu(A[1:k, 1:k])
        ssr = Vector{$elty}(undef, size(B, 2))
        for i = 1:size(B,2)
            x = zero($elty)
            for j = k+1:size(B,1)
                x += abs2(B[j,i])
            end
            ssr[i] = x
        end
        return F, subsetrows(B, B, k), ssr
    end
    function gesv!(A::AbstractMatrix{$elty},B::AbstractVecOrMat{$elty})
        n=checksquare(A)
        if n != size(B,1)
            throw(DimensionMismatch("B has a leading dimension $(size(B,1)), but nees $n"))
        end
        ipiv=similar(A,BlasInt,n)
        info =Ref{BlasInt}()
        nrhs=size(B,2)
        ida=max(1,stride(A,2))
        idb=max(1,stride(B,2))
        func=eval(@funcexpr($gesv))
        func(n,nrhs,A,ida,ipiv,B,idb,info)
        checkmagmaerror(info[])
        return B,A,ipiv
    end
end 

end


for(gesv,gels,elty) in (
    (:magma_dgesv_gpu,:magma_dgels_gpu,:Float64),
    (:magma_sgesv_gpu,:magma_sgels_gpu,:Float32),
    (:magma_zgesv_gpu,:magma_zgels_gpu,:ComplexF64),
    (:magma_cgesv_gpu,:magma_cgels_gpu,:ComplexF32)
)
@eval begin
    function gesv!(A::CuArray{$elty},B::CuArray{$elty})
        n=checksquare(A)
        if n != size(B,1)
            throw(DimensionMismatch("B has a leading dimension $(size(B,1)), but needs $n"))
        end
        ipiv=similar(Matrix(A),BlasInt,n)
        info =Ref{BlasInt}()
        nrhs=size(B,2)
        ida=max(1,stride(A,2))
        idb=max(1,stride(B,2))
        func=eval(@funcexpr($gesv))
        func(n,nrhs,A,ida,ipiv,B,idb,info)
        checkmagmaerror(info[])
        return B,A,ipiv
    end

    function gels!(trans::AbstractChar,A::CuArray{$elty},B::CuArray{$elty})
        checktranspose(trans)
        m,n =size(A)
        btrn= trans == 'N'
        if size(B,1) != (btrn ? n : m)
            throw(DimensionMismatch("matrix has dimensions ($m,$n), transposed: $btrn but the leading dimension of B is $(size(B,1))"))
        end
        info =Ref{BlasInt}()
        nrhs=size(B,2)
        ida=max(1,stride(A,2))
        idb=max(1,stride(B,2))
        work=Vector{$elty}(undef,1)
        lwork=BlasInt(-1)
        for i = 1:2
            func=eval(@funcexpr($gels))
            func(MagmaNoTrans,m,n,nrhs,A,ida,B,idb,work,lwork,info)
            checkmagmaerror(info[])
            if i==1
               lwork=ceil(BlasInt,real(work[1]))
               resize!(work,lwork)
            end

        end
        k   = min(m, n)
        F   = m < n ? tril(A[1:k, 1:k]) : triu(A[1:k, 1:k])
        ssr = Vector{$elty}(undef, size(B, 2))
        for i = 1:size(B,2)
            x = zero($elty)
            for j = k+1:size(B,1)
                x += abs2(B[j,i])
            end
            ssr[i] = x
        end
        return F, subsetrows(B, B, k), ssr
    end

    
    
end 


end 



for(posv,elty) in (
    (:magma_dposv,:Float64),
    (:magma_sposv,:Float32),
    (:magma_zposv,:ComplexF64),
    (:magma_cposv,:ComplexF32)

)
@eval begin
    function posv!(uplo::AbstractChar,A::AbstractMatrix{$elty},B::AbstractVecOrMat{$elty})
        n=checksquare(A)
        checkuplo(uplo)
        uplo_magma= uplo == 'U' ? MagmaUpper : MagmaLower
        if n !=size(B,1)
            throw(DimensionMismatch("first dimension of B, $(size(B,1)) and size of A,($n,$n), must be the same!"))
        end
        info =Ref{BlasInt}()
        nrhs=size(B,2)
        ida=max(1,stride(A,2))
        idb=max(1,stride(B,2))
        func =eval(@funcexpr($posv))
        func(uplo_magma,n,nrhs,A,ida,B,idb,info)
        checkmagmaerror(info[])
        return A,B

    end

end

end


for(posv,elty) in (
    (:magma_dposv_gpu,:Float64),
    (:magma_sposv_gpu,:Float32),
    (:magma_zposv_gpu,:ComplexF64),
    (:magma_cposv_gpu,:ComplexF32)

)
   @eval begin
    function posv!(uplo::AbstractChar,A::CuArray{$elty},B::CuArray{$elty})
        n=checksquare(A)
        checkuplo(uplo)
        uplo_magma= uplo == 'U' ? MagmaUpper : MagmaLower
        if n !=size(B,1)
            throw(DimensionMismatch("first dimension of B, $(size(B,1)) and size of A,($n,$n), must be the same!"))
        end
        info =Ref{BlasInt}()
        nrhs=size(B,2)
        ida=max(1,stride(A,2))
        idb=max(1,stride(B,2))
        func =eval(@funcexpr($posv))
        func(uplo_magma,n,nrhs,A,ida,B,idb,info)
        checkmagmaerror(info[])
        return A,B

    end

   end
end




for (hesv,elty) in (
    (:magma_chesv,ComplexF32),
    (:magma_zhesv,ComplexF64)
)

@eval begin
    function hesv!(uplo::AbstractChar,A::AbstractMatrix{$elty},B::AbstractVecOrMat{$elty})
        n=checksquare(A)
        checkuplo(uplo)
        uplo_magma= uplo == 'U' ? MagmaUpper : MagmaLower
        if n !=size(B,1)
            throw(DimensionMismatch("first dimension of B, $(size(B,1)) and size of A,($n,$n), must be the same!"))
        end
        ipiv=similar(A,BlasInt,n)
        info =Ref{BlasInt}()
        nrhs=size(B,2)
        ida=max(1,stride(A,2))
        idb=max(1,stride(B,2))
        func = eval(@funcexpr($hesv))
        func(uplo_magma,n,nrhs,A,ida,ipiv,B,idb,info)
        checkmagmaerror(info[])
        return B,A,ipiv
    end

end

end



for (sysv,elty) in (
    (:magma_dsysv,:Float64),
    (:magma_ssysv,:Float32)
)

@eval begin
    function sysv!(uplo::AbstractChar,A::AbstractMatrix{$elty},B::AbstractVecOrMat{$elty})
        n=checksquare(A)
        checkuplo(uplo)
        uplo_magma= uplo == 'U' ? MagmaUpper : MagmaLower
        if n !=size(B,1)
            throw(DimensionMismatch("first dimension of B, $(size(B,1)) and size of A,($n,$n), must be the same!"))
        end
        ipiv=similar(A,BlasInt,n)
        info =Ref{BlasInt}()
        nrhs=size(B,2)
        ida=max(1,stride(A,2))
        idb=max(1,stride(B,2))
        func = eval(@funcexpr($sysv))
        func(uplo_magma,n,nrhs,A,ida,ipiv,B,idb,info)
        checkmagmaerror(info[])
        return B,A,ipiv
    end

end

end