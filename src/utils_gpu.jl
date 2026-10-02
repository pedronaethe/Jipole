module Utils_GPU
using ..Iharm
export copy_iharm_to_gpu

"""
    copy_iharm_to_gpu(cpu_data)

Copy a CPU-resident [`Iharm.IharmData`](@ref) snapshot to the GPU,
returning an equivalent `IharmData` whose array fields are `CuArray`s.

# Arguments
- `cpu_data`: `IharmData` with `Array`-backed fields.

# Returns
- An `IharmData` with `CuArray`-backed fields.
"""
function copy_iharm_to_gpu end

end