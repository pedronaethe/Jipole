using HDF5
using CairoMakie

if length(ARGS) != 1
    error("Usage: julia --project=. --threads=12 plot.jl path/to/dumps")
end

dumps_dir = expanduser(ARGS[1])

files = filter(f -> endswith(f, ".h5"), readdir(dumps_dir; join=true))
isempty(files) && error("No .h5 files found in $dumps_dir")

println("Found $(length(files)) .h5 files in $dumps_dir")
println("Creating /figs folder in $dumps_dir")
mkpath("$dumps_dir/figs")

for file in files
    Image, fov_μas, t = h5open(file, "r") do f
        read(f["unpol"]), read(f["header/camera/fovx_dsource"]), read(f["header/t"])
    end   
    half = fov_μas / 2
    xlims = (-half, half)
    ylims = (-half, half)

    Ny, Nx = size(Image)
    x = range(xlims[1], xlims[2], length=Nx + 1)
    y = range(ylims[1], ylims[2], length=Ny + 1)

    fig = Figure(size = (600, 500))

    ax = Axis(fig[1, 1],
        xlabel = "Relative R.A [μas]",
        ylabel = "Relative Dec [μas]",
        xlabelsize = 24,
        ylabelsize = 24,
        xticklabelsize = 20,
        yticklabelsize = 20,
        limits = (xlims, ylims),
        aspect = DataAspect()
    )

    crange = extrema(Image)

    hm = heatmap!(ax, x, y, Image;
        colormap = :hot,
        colorrange = crange
    )

    Colorbar(fig[1, 2], hm;
        label = "Intensity",
        labelsize = 20,
        ticklabelsize = 16,
        width = 15
    )
    ax.title = "t = $(round(t, digits=1)) GM/c³"
    out = joinpath(dumps_dir, "figs", replace(basename(file), ".h5" => ".png"))
    save(out, fig)
    println("Saved $out")
end