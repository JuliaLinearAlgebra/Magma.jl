module LibMagmaWrap
# Write your package code here.
include("../lib/LibMagma.jl")

#exports
const PREFIXES =["magma_s","magma_d","magma_c","magma_z"]
for name in names(@__MODULE__;all=true), prefix in PREFIXES
    if startswith(name,prefix)
        @eval export $name
    end
end

end