using Clang.Generators

cd(@__DIR__)

const HEADER_BASE = "/tmp/yonatandelelegn/spack-stage/spack-stage-magma-2.7.0-3pi6vr3whu4n6f3lhqb2crr52fqty3j5/spack-src/include"
const MAGMA = joinpath(HEADER_BASE, "magma.h")

headers= [MAGMA]

options = load_options(joinpath(@__DIR__, "generator.toml"))
args = get_default_args()
push!(args, "-I$HEADER_BASE")
push!(args, "-fparse-all-comments")

ctx = create_context(headers, args, options)

build!(ctx)