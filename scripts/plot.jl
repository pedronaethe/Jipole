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
mkpath(joinpath(dumps_dir, "figs"))

const NIMG = 5
const POL_NAMES = ["I", "Q", "U", "V"]   # pol slots 1..4 (slot 5 is Faraday depth)

"""
Return the pol array as (nx, ny, NIMG), or `nothing` if polarization wasn't
computed (dataset missing, or all zeros as written by `generate_output_ipole`).
Handles both (NIMG, nx, ny) and (nx, ny, NIMG) layouts.
"""
function read_pol(f)
    haskey(f, "pol") || return nothing
    pol = read(f["pol"])
    ndims(pol) == 3 || return nothing
    any(!iszero, pol) || return nothing
    if size(pol, 1) == NIMG
        return permutedims(pol, (2, 3, 1))
    elseif size(pol, 3) == NIMG
        return pol
    else
        @warn "Unexpected pol shape $(size(pol)); skipping polarization"
        return nothing
    end
end

function plot_map(img, fov_μas, t, label, out; diverging=false)
    half = fov_μas / 2
    xlims = (-half, half)
    ylims = (-half, half)

    Nx, Ny = size(img)                 # img is indexed [x, y]
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
        aspect = DataAspect(),
        title = "$label   t = $(round(t, digits=1)) GM/c³",
    )

    if diverging
        # Q, U, V can be negative: symmetric range around zero
        m = maximum(abs, img)./10
        m = m == 0 ? 1.0 : m
        cmap, crange = :RdBu, (-m, m)
    else
        lo, hi = extrema(img)./10
        hi == lo && (hi = lo + 1)
        cmap, crange = :hot, (lo, hi)
    end

    hm = heatmap!(ax, x, y, img; colormap = cmap, colorrange = crange)
    Colorbar(fig[1, 2], hm;
        label = "Intensity",
        labelsize = 20,
        ticklabelsize = 16,
        width = 15,
    )

    save(out, fig)
    println("Saved $out")
end

for file in files
    unpol, pol, fov_μas, t = h5open(file, "r") do f
        read(f["unpol"]), read_pol(f),
        read(f["header/camera/fovx_dsource"]), read(f["header/t"])
    end

    stem = replace(basename(file), ".h5" => "")
    outname(tag) = joinpath(dumps_dir, "figs", "$(stem)_$(tag).png")

    plot_map(unpol, fov_μas, t, "unpol", outname("unpol"))

    if pol === nothing
        println("  $(basename(file)): no polarization data, plotted unpol only")
        continue
    end

    for (k, name) in enumerate(POL_NAMES)
        plot_map(pol[:, :, k], fov_μas, t, name, outname(name);
                 diverging = name != "I")
    end
end