using Printf
using Base.Threads
using LinearAlgebra
using StaticArrays
using ForwardDiff

#Mutable 4-dimensional vector allocated on the stack
const MVec4  = MVector{4,Float64}

#
const TMVec4{T} = MVector{4,T}

#Mutable 4x4 matrix allocated on the stack
const MMat4 = MMatrix{4,4,Float64}

const TMMat4{T} = MMatrix{4,4,T,16}

#Mutable 3-dimensional vector allocated on the stack
const Tensor3D = MArray{Tuple{4,4,4}, Float64, 3, 64}  # 4×4×4 mutable tensor
const TTensor3D{T} = MArray{Tuple{4,4,4}, T, 3, 64}  # 4×4×4 mutable tensor



mutable struct OfTraj
    dl::Float64
    X::MVec4
    Kcon::MVec4
    Xhalf::MVec4
    Kconhalf::MVec4
    dX_dθo::MVec4
    dK_dθo::MVec4
    dX_da::MVec4
    dK_da::MVec4
end

include("constants.jl")
include("set_globals.jl")
include("camera.jl")
include("debug_functions.jl")
include("metrics.jl")
include("coords.jl")
include("tetrads.jl")
include("utils.jl")
include("radiation.jl")

##MODELS
include("maxwell_juettner.jl")
include("grid.jl")
include("./models/$(MODEL).jl")

include("geodesics.jl")
include("autodiff.jl")
include("gradientdescent.jl")



function IpoleGeoIntensityIntegration(traj, freq_cgs::Float64, nx::Int,ny::Int, scalefactor::Float64, bhspin::Float64, data = nothing)
    """
    Once the trajectories are calculated, integrate the intensity for each pixel.
    Parameters:
    @traj: Matrix of geodesic trajectories for each pixel.
    @freq_cgs: Frequency in cgs units.
    @res: Resolution of the image (number of pixels).
    @scalefactor: Scale factor for the image intensity.

    Returns:
    A matrix representing the integrated intensity for each pixel in the image.
    """
    Image = zeros(Float64, nx, ny)
    Threads.@threads for i in 0:(nx - 1)
        println("Processing row $i out of $(nx)")
        for j in 0:(ny - 1)
            integrate_emission!(traj[i+1, j+1], length(traj[i+1, j+1]), Image, i + 1, j + 1, freq_cgs, bhspin, data)
        end
    end

    return (Image * freq_cgs^3)
end

function OutputStokesParameters(Image, freq_cgs, scale_factor, res, Dsource)
    println("Image processing complete. Calculating total flux and averages...")
    Ftot::Float64 = 0.0
    Iavg::Float64 = 0.0
    Imax::Float64 = 0.0
    imax::Int = 0
    jmax::Int = 0
    for i in 1:res
        for j in 1:res
            Ftot += Image[i, j] * scale_factor
            Iavg += Image[i, j]
            if (Image[i,j]) > Imax
                imax = i
                jmax = j
                Imax = Image[i, j]
            end
        end
    end
    Iavg *= 1.0/ (res * res)
    @printf("Scale = %.15e\n", scale_factor)
    println("imax = $imax, jmax = $jmax, Imax = $Imax, Iavg = $Iavg")
    println("Using freq_cgs = $freq_cgs, Ftot = $Ftot")
    println("nuLnu = $(Ftot * Dsource * Dsource * JY * freq_cgs * 4.0 * π)")
end

function CalculateScaleFactor(sizex, sizey, pixelsx, pixelsy, SourceD, LengthUnit)
    """
    Calculate the scale factor for the image based on the camera parameters and the source distance.

    Parameters:
    @sizex: Size of the image in the x direction in Rg.
    @sizey: Size of the image in the y direction in Rg.
    @pixelsx: Number of pixels in the x direction.
    @pixelsy: Number of pixels in the y direction.
    @SourceD: Distance to the source in cm.
    @LengthUnit: Length unit in cm (e.g., Rg).
    
    Returns:
    A scalar scale factor for the image intensity.
    """
    return (sizex * LengthUnit / pixelsx) * (sizey * LengthUnit / pixelsy) / (SourceD * SourceD) / JY

end


function main()
    @time begin
        check_parameters()
        println("Generating an image with size $(nx) x $(ny) pixels")
        println("Model Parameters: A = $A, α = $α_analytic, height = $height, l0 = $l0, a = $a")
        println("MBH = $MBH, L_unit = $L_unit")
        println("Dsource = $Dsource")

        Xcam = camera_position(rcam, thcam, phicam, Rout)

        fovx = DX/(rcam)
        fovy = DY/(rcam) 

        scale_factor = CalculateScaleFactor(DXsize, DYsize, nx, ny, SourceD, L_unit)
        println("Running on ", Threads.nthreads(), " threads")

        
        trajectory = CalculateGeodesics(Xcam, fovx, fovy, freqcgs, maxnstep, nx, ny)
        Image =  zeros(Float64, nx, ny)
        Image = IpoleGeoIntensityIntegration(trajectory, freqcgs, nx,ny, scale_factor)
    end
end


#main()
#GC.gc()

