using Clang.Generators

cd(@__DIR__)

const HEADER_BASE = joinpath(MAGMA_jll.artifact_dir, "include")
const MAGMA = joinpath(HEADER_BASE, "magma.h")

headers= [MAGMA]

options = load_options(joinpath(@__DIR__, "generator.toml"))
args = get_default_args()
push!(args, "-I$HEADER_BASE")
push!(args, "-fparse-all-comments")

ctx = create_context(headers, args, options)

build!(ctx)
