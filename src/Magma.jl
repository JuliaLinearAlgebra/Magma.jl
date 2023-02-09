module Magma
using CUDA
using LinearAlgebra: triu, tril
import LinearAlgebra: BlasInt

include("linearsolvers.jl")
using .LibMagma


# Write your package code here.




include("utils.jl")
include("dense/magmaSolvers.jl")
include("dense/factorizations.jl")




end
