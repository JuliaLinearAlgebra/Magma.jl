struct MAGMAException <:Exception
    info::BlasInt
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