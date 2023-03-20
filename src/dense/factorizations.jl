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
                LibMagma.$geev(jobvl_int,jobvr_int,n,A,ida,W,VL,idvl,VR,idvr,work,lwork,rwork,info)
            else
                LibMagma.$geev(jobvl_int,jobvr_int,n,A,ida,WR,WI,VL,idvl,VR,idvr,work,lwork,info)
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
                LibMagma.$gesvd(jobu_c,jobvt_c,m,n,A,ida,S,U,idu,VT,idv,work,lwork,rwork,info)
            else

                LibMagma.$gesvd(jobu_c,jobvt_c,m,n,A,ida,S,U,idu,VT,idv,work,lwork,info)
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
                LibMagma.$gesdd(jobz_m,m,n,A,ida,S,U,idu,VT,idv,work,lwork,rwork,iwork,info)
            else
                LibMagma.$gesdd(jobz_m,m,n,A,ida,S,U,idu,VT,idv,work,lwork,iwork,info)
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
            LibMagma.$gebrd(m,n,A,ida,d,e,tauq,taup,work,lwork,info)
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
        LibMagma.$getrf(m,n,A,ida,ipiv,info)
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
            LibMagma.$gelqf(m,n,A,ida,tau,work,lwork,info)
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
            LibMagma.$geqlf(m,n,A,ida,tau,work,lwork,info)
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
            LibMagma.$geqrf(m,n,A,ida,tau,work,lwork,info)
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

for (getrf,geqrf,geqrf_m,geqrfnb,getri,getrinb,getrs,elty,relty) in
    (
        (:magma_dgetrf_gpu,:magma_dgeqrf_gpu,:magma_dgeqrf_m,:magma_get_dgeqrf_nb,:magma_dgetri_gpu,:magma_get_dgetri_nb,:magma_dgetrs_gpu,:Float64,:Float64),
        (:magma_sgetrf_gpu,:magma_sgeqrf_gpu,:magma_sgeqrf_m,:magma_get_sgeqrf_nb,:magma_sgetri_gpu,:magma_get_sgetri_nb,:magma_sgetrs_gpu,:Float32,:Float32),
        (:magma_zgetrf_gpu,:magma_zgeqrf_gpu,:magma_zgeqrf_m,:magma_get_zgeqrf_nb,:magma_zgetri_gpu,:magma_get_zgetri_nb,:magma_zgetrs_gpu,:ComplexF64,:Float64),
        (:magma_cgetrf_gpu,:magma_cgeqrf_gpu,:magma_cgeqrf_m,:magma_get_cgeqrf_nb,:magma_cgetri_gpu,:magma_get_cgetri_nb,:magma_cgetrs_gpu,:ComplexF32,:Float32)
    )
   
    @eval begin
        
        function getrf!(A::StridedCuMatrix{$elty})
            m,n=size(A)
            minmn=min(m,n)
            ipiv=similar(Matrix(A),BlasInt,minmn)
            info  = Ref{BlasInt}()
            ida=max(1,stride(A,2))
            LibMagma.$getrf(m,n,A,ida,ipiv,info)
            checkmagmaerror(info[])
            return A,ipiv,info[]
        end

        function geqrf!(A::StridedCuMatrix{$elty})
            m,n=size(A)
            minmn=min(m,n)
            nb=LibMagma.$geqrfnb(m,n)
            #println(nb)
            tau=similar(Matrix(A),$elty,minmn)
            ida=max(1,stride(A,2))
            dT=cu(similar(Matrix(A),$elty,(2minmn + ceil(BlasInt,n/32)*32)*nb))
            info = Ref{BlasInt}()
            LibMagma.$geqrf(m,n,A,ida,tau,dT,info)
            return A,tau
    
        end
        function geqrf!(A::AbstractMatrix{$elty},ngpus::BlasInt)
            m,n=size(A)
            minmn=min(m,n)
            tau=similar(Matrix(A),$elty,minmn)
            ida=max(1,stride(A,2))
            work=Vector{$elty}(undef,1)
            lwork=BlasInt(-1)
            info = Ref{BlasInt}()
            for i= 1:2
                LibMagma.$geqrf_m(ngpus,m,n,A,ida,tau,work,lwork,info)
                checkmagmaerror(info[])
                if i==1
                    lwork = ceil(BlasInt,real(work[1]))
                    resize!(work,lwork)
                end

            end
    
        end

        function getri!(A::StridedCuMatrix{$elty},ipiv::Array{BlasInt})
            n=checksquare(A)
            if n != length(ipiv)
                throw(DimensionMismatch("ipiv has length $(length(ipiv)), but needs $n"))
            end
            ida=max(1,stride(A,2))
            lwork=ceil(BlasInt,real(n*LibMagma.$getrinb(n)))
            work=cu(Vector{$elty}(undef,max(1,lwork)))
            info = Ref{BlasInt}()
            LibMagma.$getri(n,A,ida,ipiv,work,lwork,info)
            return A
        end

        function getrs!(trans::AbstractChar,A::StridedCuMatrix{$elty},ipiv::Array{BlasInt},B::StridedCuMatrix{$elty})
            n=checksquare(A)
            trans_m = trans == 'T' ? MagmaTrans : ( trans == 'C' ? MagmaConjTrans : MagmaNoTrans)
            if n != size(B, 1)
                throw(DimensionMismatch("B has leading dimension $(size(B,1)), but needs $n"))
            end

            nrhs = size(B, 2)
            ida=max(1,stride(A,2))
            idb=max(1,stride(B,2))
            info = Ref{BlasInt}()
            LibMagma.$getrs(trans_m,n,nrhs,A,ida,ipiv,B,idb,info)
            return B

        end


    end



end