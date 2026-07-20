"""
Geodesic-trajectory record types shared across the engine (`Geodesics`,
`Radiation`, `Autodiff`, `GradientDescent`).
"""
module GeoTypes

using StaticArrays

export OfTrajM, OfTrajS, OfTraj

"""
Mutable record of a single geodesic step, used while a trajectory is being
integrated forward/backward before being frozen into an `OfTrajS`.
"""
mutable struct OfTrajM
    dl::Float64
    X::MVector{4,Float64}
    Kcon::MVector{4,Float64}
    Xhalf::MVector{4,Float64}
    Kconhalf::MVector{4,Float64}
end

"""Immutable, frozen record of a single geodesic step."""
struct OfTrajS
    dl::Float64
    X::SVector{4,Float64}
    Kcon::SVector{4,Float64}
    Xhalf::SVector{4,Float64}
    Kconhalf::SVector{4,Float64}
end

"""
Immutable record of a single geodesic step, extended with the derivatives
of position/momentum with respect to observer inclination (`θo`) and the
autodiff parameter (spin `a` or `Rhigh`), used by the `Autodiff` module.
"""
struct OfTraj
    dl::Float64
    X::SVector{4,Float64}
    Kcon::SVector{4,Float64}
    Xhalf::SVector{4,Float64}
    Kconhalf::SVector{4,Float64}
    dX_dθo::SVector{4,Float64}
    dK_dθo::SVector{4,Float64}
    dX_da::SVector{4,Float64}
    dK_da::SVector{4,Float64}
end

end
