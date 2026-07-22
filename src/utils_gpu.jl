module Utils_GPU
using ..Iharm
using CUDA
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
function copy_iharm_to_gpu(cpu_data)
    return Iharm.IharmData(
        Float64(cpu_data.t),
        CuArray(cpu_data.RHO),
        CuArray(cpu_data.UU),
        CuArray(cpu_data.U1),
        CuArray(cpu_data.U2),
        CuArray(cpu_data.U3),
        CuArray(cpu_data.B1),
        CuArray(cpu_data.B2),
        CuArray(cpu_data.B3),
        CuArray(cpu_data.ne),
        CuArray(cpu_data.b),
        CuArray(cpu_data.θe),
        CuArray(cpu_data.sigma),
        CuArray(cpu_data.beta),
        CuArray(cpu_data.dθedRhi)
    )
end

end