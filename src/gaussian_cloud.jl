"Criteria to determine whether to excite gs atom to ryd atom."
abstract type ExciteMethod end

###############################################################################
#### Rydberg Excitation #######################################################
###############################################################################

@with_kw mutable struct RydbergParams{F, T} <: ExciteMethod where {F<:VariateForm, T<:Number}
    distribution::Distribution{F, Continuous} = MvNormal([235.0; 35.0; 35.0])
    n_gs::Int64 = 130000
    rot::AbstractMatrix{T} = RotZ(2π/360 * -45.0)
    t_exc::T = 1.0
    Ω_c_0::T = 13.0
    σ_c::T = 70.0
    pos_c::Vector{T} = [0.0; 0.0]
    Ω_p_0::T = 1.3
    σ_p::T = 100.0
    pos_p::Vector{T} = [0.0; 0.0]
    Γ_p::T = 6.0
    Δ_1γ::T = 97.0
    Δ_2γ::T = 0.0
    γ::T = 0.0
    C6::T = 9300.0
    pulse_shape::Union{Symbol, Float64} = :rect  # 'Shape of the excitation pulse': objects=['rect', 'gauss', 'sech2', 'lorentz']
    laser_width::T = 0.14
end

function update!(params; kwargs...)
    kwargs = Dict(kwargs)
    for pair in kwargs
        setfield!(params, pair[1], pair[2])
    end
end

###############################################################################
#### Ground Excitable #########################################################
###############################################################################

mutable struct GroundExcitable{T<:Real}
    gs_pos::Matrix{T}
    ge_pos::Matrix{T}
    Ω_2γ::Vector{T}
    Rc::Vector{T}
    Nb::Vector{Int64}
    Ω_eff::Vector{T}
    ρ::Vector{T}
end

function _get_ge(p::RydbergParams)
    gs_pos = rand(p.distribution, p.n_gs)
    gs_pos = p.rot * gs_pos

    ge_pos = mask_center(gs_pos, p.σ_p, pos_c=p.pos_p, dims=[1, 2])
    ge_pos = mask_center(gs_pos, p.σ_c, pos_c=p.pos_c, dims=[2, 3])

    Ω_p = get_Ω(ge_pos, p.Ω_p_0, p.σ_p, pos_c=p.pos_p, dims=[1, 2])
    Ω_c = get_Ω(ge_pos, p.Ω_c_0, p.σ_c, pos_c=p.pos_c, dims=[2, 3])
    Ω_2γ =  Ω_c .* Ω_p ./ (2 * p.Δ_1γ)
    σ_fourier = get_sig_fourier(p.pulse_shape, p.t_exc)

    Rc = get_Rc(ge_pos, Ω_2γ, σ_fourier, p.laser_width, p.C6)
    println(minimum(Rc))
    Nb = get_Nb_from_Rc(ge_pos, Rc)
    println(minimum(Nb))

    Ω_eff = sqrt.(Nb) .* Ω_2γ
    if p.γ == 0
        ρ = get_ρ.(Ω_eff, p.Δ_2γ, p.t_exc)
    else
        ρ = get_ρ(Ω_eff, p.Δ_2γ, p.γ, p.t_exc)
    end
    GroundExcitable(
        gs_pos,
        ge_pos,
        Ω_2γ,
        Rc,
        Nb,
        Ω_eff,
        ρ
    )
end

GroundExcitable(params::RydbergParams) = _get_ge(params)

###############################################################################
#### excite_rydbergs ##########################################################
###############################################################################

excite_rydbergs(ge::GroundExcitable) = pick_hard_spheres(
    ge.ge_pos,
    ge.ρ, ge.Rc
)
excite_rydbergs(params::RydbergParams) = excite_rydbergs(GroundExcitable(params))

function excite_rydbergs(params; kwargs...)
    update!(params; kwargs...)
    excite_rydbergs(params)
end
