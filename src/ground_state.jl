function mask_center(pos, σ_c; pos_c=[0, 0], dims=[2, 3])
    mask = sqrt.(sum(abs2.(pos[dims, :] .- pos_c), dims=1))
    mask = vec(mask .< 3 * σ_c)

    pos[:, mask]
end


function get_sig_fourier(pulse_shape, t_exc)
    K = Dict(
    :rect => 0.443,
    :gauss => 0.441,
    :sech2 => 0.315,
    :lorentz => 0.142
    )

    K[pulse_shape] / t_exc
end


function get_Ω(pos, Ω_c, σ_c; pos_c=[0, 0],  dims=[2, 3])
    dist = sum(abs2.(pos[dims, :] .- pos_c), dims=1)
    vec(Ω_c * exp.(-dist / (2 * σ_c.^2)))
end


function get_Rc(pos, Ω_2γ, σ_fourier, laser_width, C6)
    N = size(pos)[2]
    Nb = ones(N)
    Rc = similar(Nb)

    convergence = false
    while !convergence
        Nb_old = Nb
        Rc .= get_Rc_from_Nb(Nb, Ω_2γ, σ_fourier, laser_width, C6)
        Nb .= get_Nb_from_Rc(pos, Rc)
        convergence = sum(Nb_old - Nb) <= 0
    end
    return Rc
end


function get_Rc_from_Nb(Nb, Ω_2γ, σ_fourier, laser_width, C6)
    Ω_eff = sqrt.(Nb) .* Ω_2γ
    σ_EIT = sqrt.(Ω_eff.^2 .+ laser_width^2 .+ σ_fourier^2)
    (C6 ./ σ_EIT) .^ (1 / 6)
end


function inrange(tree::NNTree{V},
                 point::AbstractMatrix{T},
                 radii::Vector{T},
                 sortres=false) where {V, T <: Number}
    dim = size(point, 1)
    npoints = size(point, 2)
    if isbitstype(T)
        new_data = copy_svec(T, point, Val(dim))
    else
        new_data = SVector{dim,T}[SVector{dim,T}(point[:, i]) for i in 1:npoints]
    end

    idxs = [Vector{Int}() for _ in 1:length(new_data)]
    for i in 1:length(new_data)
        inrange_point!(tree, new_data[i], radii[i], sortres, idxs[i])
    end
    return idxs
end


function get_Nb_from_Rc(pos, Rc)
    tree = KDTree(pos)
    idxs = inrange(tree, pos, Rc, false)
    length.(idxs)
end

function d_bloch(dρ, ρ, p, t)
    Ω, Γ, γ, δ = p
    ee, gg, ge, eg = ρ
    dρ[1] = 1im * Ω/2 * (eg - ge) - Γ*ee
    dρ[2] = -1im * Ω/2 * (eg - ge) + Γ * ee
    dρ[3] = - (γ + 1im*δ) * ge - 1im * Ω/2 * (ee - gg)
    dρ[4] = - (γ - 1im*δ) * eg + 1im * Ω/2 * (ee - gg)
    nothing
end

function get_ρ(Ω::T, Δ::T, Γ::T, tspan::Tuple{T, T}) where T
    ρ = ComplexF64[0., 1., 0., 0.]
    p = 2π*Float64[Ω, 0, Γ, Δ]
    prob = ODEProblem(d_bloch, ρ, tspan,p)
    sol = solve(prob, Tsit5(), save_idxs=1)
    sol
end

get_ρ(Ω::T, Δ::T, Γ::T, t::T) where T = real(get_ρ(Ω, Δ, Γ, (0., t))(t))

function get_ρ(Ω::AbstractVector{T}, Δ::T, Γ::T, t::T) where T
    Ω_max = maximum(Ω)
    Ω_min = minimum(Ω)
    # time_scale = 1.0 / maximum([Ω_max, Δ, Γ])
    # Ω_list = Ω_min:(time_scale/100):Ω_max
    Ω_list = LinRange(Ω_min, Ω_max, 100)
    #println(Ω_list)

    ρ_list = [get_ρ(Ω_i, Δ, Γ, t) for Ω_i in Ω_list]
    #println(ρ_list)
    itp = CubicSplineInterpolation(Ω_list, ρ_list)
    return real(itp(Ω))
end

get_ρ(Ω::T, Δ::T, t::T) where T = Ω^2 / (Δ^2 + Ω^2) * sin(2π * Ω * t / 2)^2
