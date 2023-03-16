module LibMagma
# Write your package code here.
using MAGMA_jll
using CUDA
const libmagma = MAGMA_jll.libmagma_path

export magma_init
export magma_finalize
include("../lib/LibMagma.jl")

#exports
const PREFIXES =["magma_s","magma_d","magma_c","magma_z","Magma","magma_get"]
for name in names(@__MODULE__;all=true), prefix in PREFIXES
    if startswith(string(name),prefix)
        @eval export $name
    end
end
end