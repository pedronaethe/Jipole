"""
Snapshot containers and time maps shared by every light mode.

Fast, brisk and slow light evaluate the *same* geodesics, redshift factors and
source positions. They differ in exactly one thing: how a geodesic step's
coordinate time becomes an absolute fluid time. That one difference is captured
by [`TimeMap`](@ref), so a single radiative-transfer loop serves all three:

| mode  | `t_abs(step)`                                                   |
|:------|:----------------------------------------------------------------|
| fast  | `t_frame` — constant, no light-travel delay across the image    |
| brisk | the slow-light time, clipped into the band's retained interval  |
| slow  | `t_geo + t_frame` — the full delay map                          |

The snapshots a frame may interpolate between live in a [`SnapshotSet`](@ref):
a compile-time-sized, time-ordered tuple. Looking a time up in it picks the
**two nearest** snapshots and the linear weight between them, which is the only
temporal interpolation any mode performs.

Both types are `isbits` given `isbits` snapshots and carry no heap-allocated
members, so `Adapt` can move them to a device as-is and the lookup compiles
inside a GPU kernel.
"""
module TimeInterp

using StaticArrays
using Adapt

export SnapshotSet, TimeMap, nearest_pair, map_time, n_snapshots,
    fastlight_timemap, slowlight_timemap, snapshot_times

# ---------------------------------------------------------------------------
# Snapshot sets
# ---------------------------------------------------------------------------

"""
    SnapshotSet(snaps::NTuple{N,D}, times::SVector{N,T})

`N` fluid snapshots in ascending time order, together with their times.

`N` is part of the type, which is what keeps [`nearest_pair`](@ref) type stable
and lets the bracket search unroll — a `Vector` of snapshots cannot do either,
and indexing a tuple with a runtime index is not GPU-compilable at all (see
[`select_snapshot`](@ref) for the way around that).

Build one with [`SnapshotSet`](@ref)'s vector constructor, which sorts for you.
"""
struct SnapshotSet{N,D,T}
    snaps::NTuple{N,D}
    times::SVector{N,T}
end

Adapt.adapt_structure(to, s::SnapshotSet) =
    SnapshotSet(map(x -> Adapt.adapt(to, x), s.snaps), s.times)

"""
    SnapshotSet(snaps::AbstractVector, times::AbstractVector)

Build a `SnapshotSet` from vectors, sorting both by time. `N` is taken from the
length, so this is the (type-unstable) boundary where a runtime-sized list of
loaded dumps becomes a compile-time-sized set; call it once per frame, outside
any hot loop.
"""
function SnapshotSet(snaps::AbstractVector, times::AbstractVector)
    length(snaps) == length(times) ||
        throw(ArgumentError("SnapshotSet: $(length(snaps)) snapshots but $(length(times)) times"))
    isempty(snaps) && throw(ArgumentError("SnapshotSet: need at least one snapshot"))
    perm = sortperm(times)
    N = length(snaps)
    T = float(eltype(times))
    return SnapshotSet(ntuple(i -> snaps[perm[i]], N), SVector{N,T}(times[perm]))
end

"""
    SnapshotSet(snaps::AbstractVector)

Build a `SnapshotSet`, taking each snapshot's time from its own `.t` field.
"""
SnapshotSet(snaps::AbstractVector) = SnapshotSet(snaps, [s.t for s in snaps])

"""
    n_snapshots(s)

Number of snapshots in the set, known at compile time.
"""
@inline n_snapshots(::SnapshotSet{N}) where {N} = N

"""
    snapshot_times(s)

The set's times, ascending.
"""
@inline snapshot_times(s::SnapshotSet) = s.times

# Make a `SnapshotSet` usable anywhere a vector or tuple of snapshots was
# expected. The model getters index `data[1]` with a literal to promote element
# types, so this has to exist for the set to be a drop-in replacement; routing
# it through `select_snapshot` keeps even a runtime index GPU-compilable, and a
# literal one constant-folds away entirely.
@inline Base.getindex(s::SnapshotSet, k::Integer) = select_snapshot(s.snaps, k)
@inline Base.length(::SnapshotSet{N}) where {N} = N
@inline Base.firstindex(::SnapshotSet) = 1
@inline Base.lastindex(::SnapshotSet{N}) where {N} = N
@inline Base.eltype(::SnapshotSet{N,D}) where {N,D} = D
@inline Base.iterate(s::SnapshotSet, k::Int = 1) = k > length(s) ? nothing : (s[k], k + 1)

"""
    select_snapshot(snaps::Tuple, k)

`snaps[k]` for a runtime `k`, without indexing a tuple by a runtime value.

Tuple fields live in registers, so a runtime index has nowhere to point and
`getindex` on a tuple is not GPU-compilable. Recursing with `Base.tail` turns
the lookup into a chain of comparisons against compile-time positions instead;
every level has a concrete tuple type, so the whole chain inlines and the
element type never changes.
"""
@inline select_snapshot(snaps::Tuple{D}, k::Integer) where {D} = snaps[1]
@inline select_snapshot(snaps::Tuple, k::Integer) = _select(snaps, k, 1)
# Past ~16 snapshots the recursion defeats inference (a 40-snapshot set is ~200x slower per lookup and
# allocates), so a homogeneous tuple that long is indexed directly: type stable on the CPU, which is the only
# engine brisk light has. A GPU kernel would need another route for sets this large.
@inline select_snapshot(snaps::NTuple{N,D}, k::Integer) where {N,D} =
    N <= 16 ? _select(snaps, k, 1) : (@inbounds snaps[k])

@inline _select(snaps::Tuple{D}, k::Integer, i::Int) where {D} = snaps[1]
@inline _select(snaps::Tuple, k::Integer, i::Int) =
    k == i ? snaps[1] : _select(Base.tail(snaps), k, i + 1)

"""
    bracket_index(times::SVector{N,T}, t) -> Int

Index `iA` of the lower member of the pair bracketing `t`, in `1:N-1`.

A branchless sweep rather than a binary search: `N` is a compile-time constant
and small (it is the number of dumps a frame needs, not the length of the run),
so the loop unrolls into a handful of `ifelse`s with no divergence — which is
what a GPU wants. Times at or beyond either end clamp to the terminal pair, so
the weight returned by [`nearest_pair`](@ref) always lies in `[0, 1]` and the
plasma is frozen rather than extrapolated.
"""
@inline function bracket_index(times::SVector{N,T}, t) where {N,T}
    # Long sets (brisk p near 1 holds hundreds of dumps): binary search, same result as the sweep below.
    N > 16 && return clamp(searchsortedlast(times, t), 1, N - 1)
    iA = 1
    @inbounds for i in 1:(N - 1)
        iA = ifelse(times[i + 1] <= t, i + 1, iA)
    end
    return min(iA, N - 1)
end

@inline bracket_index(times::SVector{1,T}, t) where {T} = 1

"""
    nearest_pair(s::SnapshotSet, t) -> (snapA, snapB, w)

The two snapshots of `s` nearest `t` and the weight `w` on `snapA`, such that a
field interpolates as `w * fieldA + (1 - w) * fieldB`. `w == 1` reproduces
`snapA` exactly.

A one-snapshot set returns that snapshot twice with `w = 1`, so the same blend
expression is exact with no branch — that is how a `p = 0` band sitting on a
dump time, and fast light, reduce to a single snapshot without a special case.
"""
@inline function nearest_pair(s::SnapshotSet{1,D,T}, t) where {D,T}
    snap = s.snaps[1]
    return snap, snap, one(T)
end

@inline function nearest_pair(s::SnapshotSet{N,D,T}, t) where {N,D,T}
    iA = bracket_index(s.times, t)
    @inbounds tA = s.times[iA]
    @inbounds tB = s.times[iA + 1]
    snapA = select_snapshot(s.snaps, iA)
    snapB = select_snapshot(s.snaps, iA + 1)
    dt = tB - tA
    w = dt > zero(dt) ? one(T) - (t - tA) / dt : one(T)
    # Clamp rather than extrapolate: a query outside the set's span freezes the
    # plasma at the nearest end. Callers normally clamp the time first (see
    # `TimeMap`'s availability clamp), but a weight outside [0, 1] would silently
    # produce negative densities, so the guarantee is enforced here as well.
    return snapA, snapB, clamp(w, zero(T), one(T))
end

# ---------------------------------------------------------------------------
# Time maps
# ---------------------------------------------------------------------------

"""
    TimeMap(t_frame, t_ref, lo, hi, tA, tB)

Maps a geodesic step's coordinate time to an absolute fluid time, for one frame.

Geodesics are traced backwards from the camera with `t = 0` there, so a step's
`X[1]` is negative and is (minus) the light-travel time back from the camera.

# Fields
- `t_frame`: absolute fluid time this frame is labelled with — by convention the
  time of its `n = 0` image, i.e. the time fast light would use. This is the
  synchronization anchor: at `p = 0` band 0 is evaluated at exactly `t_frame`.
- `t_ref`: the geometric time that `t_frame` corresponds to, `t̄_0` (the modal
  emission time of band 0). Subtracting it is what makes `t_frame` mean "the
  `n = 0` image time" rather than "the camera time"; the two clocks differ by
  this constant, which matters when pairing frames against slow light.
- `lo`, `hi`: per-band clip bounds in *geometric* time, band `n` at index
  `n + 1`. The retained interval `T_{n,p}`; `p = 0` collapses it to a point and
  `p = 1` opens it to `(-Inf, Inf)`.
- `tA`, `tB`: times of the first and last available snapshot. A step clipped to
  this range sees a locally frozen plasma instead of an extrapolated one; the
  fraction of steps that hit it is a correctness diagnostic, not a free knob.
"""
struct TimeMap{NB,T}
    t_frame::T
    t_ref::T
    lo::SVector{NB,T}
    hi::SVector{NB,T}
    tA::T
    tB::T
end

Adapt.adapt_structure(to, tm::TimeMap) = tm

"""
    n_bands(tm::TimeMap)

Number of bands the map carries, known at compile time.
"""
@inline n_bands(::TimeMap{NB}) where {NB} = NB

"""
    map_time(tm::TimeMap, t_geo, band) -> T

Absolute fluid time for a step at geometric time `t_geo` belonging to `band`
(0-based lensing-band index).

Two clamps, in order: the brisk-light clip into the band's retained interval
(Eq. V.2), then the snapshot-availability clamp. With `lo == hi` the first
clamp collapses the band onto a single time; with `lo = -Inf, hi = Inf` it is
the identity and what remains is slow light's map.
"""
@inline function map_time(tm::TimeMap{NB,T}, t_geo, band::Integer) where {NB,T}
    b = clamp(Int(band) + 1, 1, NB)
    shift = tm.t_frame - tm.t_ref
    @inbounds lo = tm.lo[b] + shift
    @inbounds hi = tm.hi[b] + shift
    return clamp(clamp(t_geo + shift, lo, hi), tm.tA, tm.tB)
end

"""
    fastlight_timemap(t_frame, tA, tB)

A map that sends every step to `t_frame`: fast light, which ignores
light-travel delay across the image. Combined with a two-snapshot
[`SnapshotSet`](@ref) this is fast light interpolated to an arbitrary time
between dumps, which is what lets every mode share one observer-time grid.
"""
function fastlight_timemap(t_frame::T, tA::T, tB::T) where {T}
    z = zero(T)
    return TimeMap{1,T}(t_frame, z, SVector{1,T}(z), SVector{1,T}(z), tA, tB)
end

"""
    slowlight_timemap(t_frame, tA, tB)

A map with no clipping: `t_abs = t_geo + t_frame`, clamped only by snapshot
availability. Note `t_ref = 0`, so `t_frame` here is the *camera* clock, which
is slow light's own convention and differs from brisk light's `n = 0` clock by
`t̄_0`.
"""
function slowlight_timemap(t_frame::T, tA::T, tB::T) where {T}
    return TimeMap{1,T}(t_frame, zero(T),
                        SVector{1,T}(T(-Inf)), SVector{1,T}(T(Inf)), tA, tB)
end

end # module TimeInterp
