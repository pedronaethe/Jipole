"""
Geodesic-trajectory record types structures shared across the code (`Geodesics`,
`Radiation`, `Autodiff`, `GradientDescent`).
"""
module GeoTypes

using StaticArrays

export OfTrajM, OfTrajS, OfTraj, OfTrajGRMHD

"""
Mutable structure of a single geodesic step, used while a trajectory is being
integrated forward/backward before being frozen into an `OfTrajS`.
"""
mutable struct OfTrajM
    dl::Float64
    X::MVector{4,Float64}
    Kcon::MVector{4,Float64}
    Xhalf::MVector{4,Float64}
    Kconhalf::MVector{4,Float64}
end

"""Immutable, frozen structure of a single geodesic step."""
struct OfTrajS
    dl::Float64
    X::SVector{4,Float64}
    Kcon::SVector{4,Float64}
    Xhalf::SVector{4,Float64}
    Kconhalf::SVector{4,Float64}
end

"""
Immutable structure of a single geodesic step, extended with the derivatives
of position/momentum with respect to observer inclination (`θo`) and the
autodiff parameter (spin `a`), used by the `Autodiff` module.
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


"""
Immutable structure of a single geodesic step, extended with the derivatives
of position/momentum with respect to observer inclination (`θo`).
Primarily used for GPU version.
"""
struct OfTrajGRMHD
    dl::Float64
    X::SVector{4, Float64}
    Kcon::SVector{4, Float64}
    Xhalf::SVector{4, Float64}
    Kconhalf::SVector{4, Float64}
    dX_dθo::SVector{4, Float64}
    dK_dθo::SVector{4, Float64}
end

"""
Immutable structure of a single geodesic step, generic over element type `T`
so it can carry `ForwardDiff.Dual` (used when differentiating the full
geodesic + radiative-transfer computation directly, as opposed to the
separate approach of integrating the sensitivities ODE, which is the case for
`OfTrajGRMHD`).
"""
struct OfTrajDual{T}
    dl::T
    X::SVector{4,T}
    Kcon::SVector{4,T}
end

end
