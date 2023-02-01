module Magma
using CUDA
using LinearAlgebra: triu, tril
import LinearAlgebra: BlasInt

include("linearsolvers.jl")
using .LibMagma


# Write your package code here.

struct MAGMAException <:Exception
    info::BlasInt
end
macro funcexpr(funcname)
    return Expr(:quote,Symbol(funcname))
end

function checksquare(A) 
    m,n =size(A)
    m==n || throw(DimensionMismatch("matrix is not square: dimensions are $((size(A)))"))
    return m
end
#checking if any magma error is generated

function checkmagmaerror(ret::BlasInt)
    if ret==0
        return 
   elseif ret <0
    throw(ArgumentError("invalid argument $(-ret) to magma call"))
   else
    throw(MAGMAException(ret)) 
   end
end

function checktranspose(trans::AbstractChar) 
    if(!(trans == 'T'  || trans =='N' || trans =='T'))
        throw(ArgumentError("trans argument must be 'T' (transpose) , 'N' (no transpose), 'C' (conjugate transpose), but got $trans"))
    end
    return trans
end
#copied from julia lpack.jl
function checkuplo(uplo::AbstractChar)
    if !(uplo == 'U' || uplo =='L')
        throw(ArgumentError("uplo argmument has to be U(upper) or L(lower), got $uplo"))
    end
    return uplo
end

subsetrows(X::AbstractVector, Y::AbstractArray, k) = Y[1:k]
subsetrows(X::AbstractMatrix, Y::AbstractArray, k) = Y[1:k, :]

for(gels,gesv,elty) in (
    (:magma_dgels,:magma_dgesv,:Float64),
    (:magma_sgels,:magma_sgesv,:Float32),
    (:magma_zgels,:magma_zgesv,:ComplexF64),
    (:magma_cgels,:magma_cgesv,:ComplexF32)
)
#println(gels,"first")
@eval begin
    function gels!(trans::AbstractChar,A::AbstractMatrix{$elty},B::AbstractVecOrMat{$elty})
        #println(gels,"second")
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
        #println(func)
        #segfault happens in the following func call. func evaluates to magma_sgesv
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
        #println(gels,"second")
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

for (geev,gesvd,gesdd,elty,relty) in (
    (:magma_dgeev,:magma_dgesvd,:magma_dgesdd,:Float64,:Float64),
    (:magma_sgeev,:magma_sgesvd,:magma_sgesdd,:Float32,:Float32),
    (:magma_cgeev,:magma_cgesvd,:magma_cgesdd,:ComplexF32,:Float32),
    (:magma_zgeev,:magma_zgesvd,:magma_zgesdd,:ComplexF64,:Float64)
)

@eval begin
    function geev!(jobvl::AbstractChar,jobvr::AbstractChar,A::AbstractMatrix{$elty})
        n=checksquare(A)
        lvecs = jobvl=='V'
        rvecs = jobvr =='V'
        jobvl_int = jobvl == 'V' ? MagmaVec : MagmaNoVec
        jobvr_int = jobvr == 'V' ? MagmaVec : MagmaNoVec
        VL= similar(A,$elty,(n,lvecs ? n : 0))
        VR=similar(A,$elty,(n,rvecs ? n : 0))
        is_complex = eltype(A) <:Complex

        if is_complex
            W=similar(A,$elty,n)
            rwork=similar(A,$relty,2n)
        else
            WR=similar(A,$elty,n)
            WI=similar(A,$elty,n)
        end
        work=Vector{$elty}(undef,1)
        lwork=BlasInt(-1)
        info=Ref{BlasInt}()
        ida=max(1,stride(A,2))
        idvl=n
        idvr=n
        for i = 1:2
            if is_complex
                func =eval(@funcexpr($geev))
                #println("I am here")
                func(jobvl_int,jobvr_int,n,A,ida,W,VL,idvl,VR,idvr,work,lwork,rwork,info)
            else
                func= eval(@funcexpr($geev))
                func(jobvl_int,jobvr_int,n,A,ida,WR,WI,VL,idvl,VR,idvr,work,lwork,info)
            end
            checkmagmaerror(info[])
            if i==1
                lwork=ceil(BlasInt,real(work[1]))
                resize!(work,lwork)
            end
        end

        is_complex ? (W,VL,VR) : (WR,WI,VL,VR)
    end

    function gesvd!(jobu::AbstractChar,jobvt::AbstractChar,A::AbstractMatrix{$elty})
        m,n = size(A)
        minmn=min(m,n)
        S= similar(A,$relty,minmn)
        U=similar(A,$elty,jobu=='A' ? (m,m) : (jobu =='S' ? (m,minmn) : (m,0)))
        VT=similar(A,$elty,jobvt == 'A' ? (n,n) : (jobvt =='S' ? (minmn,n) : (n,0)))
        jobu_c= jobu=='A' ? MagmaAllVec : (jobu=='S' ? MagmaSomeVec : (jobu=='O' ?  MagmaOverwriteVec : MagmaNoVec ))
        jobvt_c= jobvt=='A' ? MagmaAllVec : (jobvt=='S' ? MagmaSomeVec : (jobvt=='O' ?  MagmaOverwriteVec : MagmaNoVec ))
        work=Vector{$elty}(undef,1)
        is_complex = eltype(A) <:Complex
        if is_complex
            rwork=Vector{$relty}(undef,5minmn)
        end
        lwork=BlasInt(-1)
        info=Ref{BlasInt}()
        ida=max(1,stride(A,2))
        idu=max(1,stride(U,2))
        idv=max(1,stride(VT,2))
        for i in 1:2
            if is_complex
                func =eval(@funcexpr($gesvd))
                func(jobu_c,jobvt_c,m,n,A,ida,S,U,idu,VT,idv,work,lwork,rwork,info)
            else

                func=eval(@funcexpr($gesvd))
                func(jobu_c,jobvt_c,m,n,A,ida,S,U,idu,VT,idv,work,lwork,info)
            end
            checkmagmaerror(info[])
            if i==1
                lwork=ceil(BlasInt,real(work[1]))
                resize!(work,lwork)
            end

        end

        if jobu =='O'
            return (A,S,VT)
        elseif jobvt =='O'
            return (U,S,A)
        else
            return (U,S,VT)
        end


    end

    function gesdd!(jobz::AbstractChar,A::AbstractMatrix{$elty})
        m, n   = size(A)
        minmn  = min(m, n)
        S= similar(A,$relty,minmn)
        jobz_m = jobz =='A' ? MagmaAllVec : (jobz=='S' ? MagmaSomeVec : (jobz=='O' ?  MagmaOverwriteVec : MagmaNoVec ))
        if jobz == 'A'
            U  = similar(A, $elty, (m, m))
            VT = similar(A, $elty, (n, n))
        elseif jobz == 'S'
            U  = similar(A, $elty, (m, minmn))
            VT = similar(A, $elty, (minmn, n))
        elseif jobz == 'O'
            U  = similar(A, $elty, (m, m >= n ? 0 : m))
            VT = similar(A, $elty, (n, m >= n ? n : 0))
        else
            U  = similar(A, $elty, (m, 0))
            VT = similar(A, $elty, (n, 0))
        end
        work=Vector{$elty}(undef,1)
        is_complex = eltype(A) <:Complex
        if is_complex
            rwork = Vector{$relty}(undef, jobz == 'N' ? 7*minmn : minmn*max(5*minmn+7, 2*max(m,n)+2*minmn+1))
        end
        lwork=BlasInt(-1)
        iwork  = Vector{BlasInt}(undef, 8*minmn)
        info=Ref{BlasInt}()
        ida=max(1,stride(A,2))
        idu=max(1,stride(U,2))
        idv=max(1,stride(VT,2))
        for i in 1:2
            if is_complex
                func = eval(@funcexpr($gesdd))
                func(jobz_m,m,n,A,ida,S,U,idu,VT,idv,work,lwork,rwork,iwork,info)
            else
                func=eval(@funcexpr($gesdd))
                func(jobz_m,m,n,A,ida,S,U,idu,VT,idv,work,lwork,iwork,info)
            end
            checkmagmaerror(info[])
            if i==1
                #workaround truncating doubles
                lwork=round(BlasInt,nextfloat(real(work[1])))
                resize!(work,lwork)
            end

        end

        if jobz =='O'
            if m>=n
                return (A,S,VT)
            else
                return (U,S,A)
            end
        else
            return (U,S,VT)
        end


    end
    
end

end

for(gebrd,getrf,gelqf,geqlf,geqrf,elty,relty) in (
    (:magma_dgebrd,:magma_dgetrf,:magma_dgelqf,:magma_dgeqlf,:magma_dgeqrf,:Float64,:Float64),
    (:magma_sgebrd,:magma_sgetrf,:magma_sgelqf,:magma_sgeqlf,:magma_sgeqrf,:Float32,:Float32),
    (:magma_cgebrd,:magma_cgetrf,:magma_cgelqf,:magma_cgeqlf,:magma_cgeqrf,:ComplexF32,:Float32),
    (:magma_zgebrd,:magma_zgetrf,:magma_zgelqf,:magma_zgeqlf,:magma_zgeqrf,:ComplexF64,:Float64)
)
 @eval begin
    function gebrd!(A::AbstractMatrix{$elty})
        m,n=size(A)
        minmn=min(m,n)
        d=similar(A,$relty,minmn)
        e=similar(A,$relty,minmn)
        tauq=similar(A,$elty,minmn)
        taup=similar(A,$elty,minmn)
        work=Vector{$elty}(undef,1)
        lwork=BlasInt(-1)
        info  = Ref{BlasInt}()
        ida=max(1,stride(A,2))

        for i= 1:2
            func=eval(@funcexpr($gebrd))
            func=(m,n,A,ida,d,e,tauq,taup,work,lwork,info)
            checkmagmaerror(info[])
            if i==1
                lwork=ceil(BlasInt,real(work[1]))
                resize!(work,lwork)
            end

        end
       return A,d,e,tauq,taup

    end
    function getrf!(A::AbstractMatrix{$elty})
        m,n=size(A)
        minmn=min(m,n)
        ipiv=similar(A,BlasInt,minmn)
        info  = Ref{BlasInt}()
        ida=max(1,stride(A,2))
        func=eval(@funcexpr($getrf))
        func(m,n,A,ida,ipiv,info)
        checkmagmaerror(info[])
        return A,ipiv,info[]
    end

    function gelqf!(A::AbstractMatrix{$elty})
        m,n=size(A)
        minmn=min(m,n)
        tau=similar(A,$elty,minmn)
        ida=max(1,stride(A,2))
        work=Vector{$elty}(undef,1)
        lwork=BlasInt(-1)
        info = Ref{BlasInt}()
        for i= 1:2
            func=eval(@funcexpr($gelqf))
            func(m,n,A,ida,tau,work,lwork,info)
            checkmagmaerror(info[])
            if i==1
                lwork=ceil(BlasInt,real(work[1]))
                resize!(work,lwork)
            end

        end
        return A,tau
    end

    function geqlf!(A::AbstractMatrix{$elty})
        m,n=size(A)
        minmn=min(m,n)
        tau=similar(A,$elty,minmn)
        ida=max(1,stride(A,2))
        work=Vector{$elty}(undef,1)
        lwork=BlasInt(-1)
        info = Ref{BlasInt}()
        for i= 1:2
            func=eval(@funcexpr($geqlf))
            func(m,n,A,ida,tau,work,lwork,info)
            checkmagmaerror(info[])
            if i==1
                lwork=ceil(BlasInt,real(work[1]))
                resize!(work,lwork)
            end

        end
        return A,tau

    end
    function geqrf!(A::AbstractMatrix{$elty})
        m,n=size(A)
        minmn=min(m,n)
        tau=similar(A,$elty,minmn)
        ida=max(1,stride(A,2))
        work=Vector{$elty}(undef,1)
        lwork=BlasInt(-1)
        info = Ref{BlasInt}()
        for i= 1:2
            func=eval(@funcexpr($geqrf))
            func(m,n,A,ida,tau,work,lwork,info)
            checkmagmaerror(info[])
            if i==1
                lwork = ceil(BlasInt,real(work[1]))
                resize!(work,lwork)
            end

        end
        return A,tau

    end

 end


end

for (getrf,geqrf,geqrfnb,getri,getrinb,getrs,elty,relty) in
    (
        (:magma_dgetrf_gpu,:magma_dgeqrf_gpu,:magma_get_dgeqrf_nb,:magma_dgetri_gpu,:magma_get_dgetri_nb,:magma_dgetrs_gpu,:Float64,:Float64),
        (:magma_sgetrf_gpu,:magma_sgeqrf_gpu,:magma_get_sgeqrf_nb,:magma_sgetri_gpu,:magma_get_sgetri_nb,:magma_sgetrs_gpu,:Float32,:Float32),
        (:magma_zgetrf_gpu,:magma_zgeqrf_gpu,:magma_get_zgeqrf_nb,:magma_zgetri_gpu,:magma_get_zgetri_nb,:magma_zgetrs_gpu,:ComplexF64,:Float64),
        (:magma_cgetrf_gpu,:magma_cgeqrf_gpu,:magma_get_cgeqrf_nb,:magma_cgetri_gpu,:magma_get_cgetri_nb,:magma_cgetrs_gpu,:ComplexF32,:Float32)
    )
   
    @eval begin
        
        function getrf!(A::CuArray{$elty})
            m,n=size(A)
            minmn=min(m,n)
            ipiv=similar(Matrix(A),BlasInt,minmn)
            info  = Ref{BlasInt}()
            ida=max(1,stride(A,2))
            func=eval(@funcexpr($getrf))
            func(m,n,A,ida,ipiv,info)
            checkmagmaerror(info[])
            return A,ipiv,info[]
        end

        function geqrf!(A::CuArray{$elty})
            m,n=size(A)
            minmn=min(m,n)
            func_nb=eval(@funcexpr($geqrfnb))
            nb=func_nb(m,n)
            tau=similar(Matrix(A),$elty,minmn)
            ida=max(1,stride(A,2))
            dT=similar(Matrix(A),$elty,(2minmn + ceil(BlasInt,n/32)*32)*nb)
            info = Ref{BlasInt}()
            func=eval(@funcexpr($geqrf))
            func(m,n,A,ida,tau,dT,info)
            return A,tau
    
        end

        function getri!(A::CuMatrix{$elty},ipiv::Array{BlasInt})
            n=checksquare(A)
            if n != length(ipiv)
                throw(DimensionMismatch("ipiv has length $(length(ipiv)), but needs $n"))
            end
            ida=max(1,stride(A,2))
            func_nb=eval(@funcexpr($getrinb))
            lwork=ceil(BlasInt,real(n*func_nb(n)))
            work=cu(Vector{$elty}(undef,max(1,lwork)))
            info = Ref{BlasInt}()
            func=eval(@funcexpr($getri))
            func(n,A,ida,ipiv,work,lwork,info)
            return A
        end

        function getrs!(trans::AbstractChar,A::CuMatrix{$elty},ipiv::Array{BlasInt},B::CuArray{$elty},)
            n=checksquare(A)
            trans_m = trans == 'T' ? MagmaTrans : ( trans == 'C' ? MagmaConjTrans : MagmaNoTrans)
            if n != size(B, 1)
                throw(DimensionMismatch("B has leading dimension $(size(B,1)), but needs $n"))
            end

            nrhs = size(B, 2)
            ida=max(1,stride(A,2))
            idb=max(1,stride(B,2))
            info = Ref{BlasInt}()
            func= eval(@funcexpr($getrs))
            func(trans_m,n,nrhs,A,ida,ipiv,B,idb,info)
            return B

        end


    end



end



end
