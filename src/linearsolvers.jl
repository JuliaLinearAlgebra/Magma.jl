module LibMagma
# Write your package code here.
using MAGMA_jll
using CUDA
using Preferences

const libmagma = begin
    libmagma_pref = @load_preference("libmagma", nothing)
    if !isnothing(libmagma_pref)
        libmagma_pref
    else
        if MAGMA_jll.is_available()
            MAGMA_jll.libmagma_path
        else
            @warn("""
            MAGMA_jll isn't working. Will assume that a system libmagma is dynamically loadable.
            To hide this warning, set the libmagma preference to a specific path
            (e.g. with Magma.LibMagma.set_libmagma_path).
            """)
            "libmagma"
        end
    end
end

function set_libmagma_path(path::AbstractString)
    @set_preferences!("libmagma" => path)
    @info("New libmagma path set; please restart Julia to see this take effect", path)
end

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
