"""
Shared supertype for every model's parameter object.

Concrete models (`Analytic.AnalyticParams`, `ThinDisk.ThinDiskParams`,
`Iharm.IharmParams`) subtype `AbstractModel`. Engine modules (`Coordinates`,
`Metrics`, `Camera`, `Radiation`, `Geodesics`) dispatch on this common
supertype to provide a default implementation, which a specific model can
override with a more specific method for its own params type. This is what
replaces the old `MODEL::String` global and its `if MODEL == "..."` branches
throughout the codebase.
"""
module AbstractModels

export AbstractModel

abstract type AbstractModel end

end
