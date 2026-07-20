"""
Diagnostic printing helpers used throughout the engine to report vectors
and matrices when something numerically invalid is encountered.
"""
module DebugFunctions

using Printf
using StaticArrays

export print_vector, print_matrix

"""
    print_vector(name, vec)

Print a labeled 4-vector, one component per line.

# Arguments
- `name`: Label to print above the vector.
- `vec`: The 4-vector to print.
"""
function print_vector(name::String, vec::MVector{4,Float64})
    println("Vector: $name")
    for i in eachindex(vec)
        print("$(vec[i]) ")
    end
    println()
end

"""
    print_matrix(name, mat)

Print a labeled matrix in scientific notation.

# Arguments
- `name`: Label to print above the matrix.
- `mat`: The matrix to print.
"""
function print_matrix(name::String, mat)
    println("Matrix: $name")
    for i in axes(mat, 1)
        for j in axes(mat, 2)
            @printf("%.15e ", mat[i, j])
        end
        println()
    end
end

end
