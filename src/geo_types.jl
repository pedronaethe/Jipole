"""
Geodesic-trajectory record types structures shared across the code (`Geodesics`,
`Radiation`, `Autodiff`, `GradientDescent`).
"""
module GeoTypes

using StaticArrays

export OfTrajGeneric
"""
Immutable structure of a single geodesic step, generic over element type `T`
so it can carry `ForwardDiff.Dual` (used when differentiating the full
geodesic + radiative-transfer computation directly, as opposed to the
separate approach of integrating the sensitivities ODE, which is the case for
`OfTrajGRMHD`).
"""
struct OfTrajGeneric{T}
    dl::T
    X::SVector{4,T}
    Kcon::SVector{4,T}
    Xhalf::SVector{4,T}
    Kconhalf::SVector{4,T}
end

end
