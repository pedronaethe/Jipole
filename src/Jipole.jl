"""
Jipole: a Julia-based radiative transfer code for curved spacetimes, with
automatic differentiation capabilities.

Every submodule is reachable as `Jipole.<ModuleName>`, e.g.
`Jipole.Camera.camera_position`. Three interchangeable emission models are
provided — [`Jipole.Analytic`](@ref), [`Jipole.ThinDisk`](@ref), and
[`Jipole.Iharm`](@ref) — each defining its own parameters type
(`<: Jipole.AbstractModels.AbstractModel`); which model runs is decided by
which parameters object is passed in, via multiple dispatch, rather than
by a global model switch.
"""
module Jipole

include("model_base.jl")
include("constants.jl")
include("geo_types.jl")
include("debug_functions.jl")
include("coords.jl")
include("metrics.jl")
include("camera.jl")
include("utils.jl")
include("tetrads.jl")
include("maxwell_juettner.jl")
include("grid.jl")
include("radiation.jl")
include("geodesics.jl")
include("models/analytic.jl")
include("models/thin_disk.jl")
include("models/iharm.jl")

include("output.jl")
include("imaging.jl")
include("utils_gpu.jl")
include("slowlight.jl")
end
