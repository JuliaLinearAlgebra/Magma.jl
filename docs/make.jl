using Magma
using Documenter

DocMeta.setdocmeta!(Magma, :DocTestSetup, :(using Magma); recursive=true)

makedocs(;
    modules=[Magma],
    authors="Yonatan Delelegn",
    repo="https://github.com/yonatanwesen/Magma.jl/blob/{commit}{path}#{line}",
    sitename="Magma.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)
