using Base: length
using Random: AbstractRNG, MersenneTwister
import Distributions: _rand!, Continuous, Sampleable, Multivariate
using LinearAlgebra

###############################################################################
#### Sphere ###################################################################
###############################################################################

volume_from_radius(r::Real) = 4/3 *π * r^3
radius_from_volume(V::Real) = (3/(4π) * V)^(1/3)
radius_from_density(n::Real, N::Int64) = (3/(4π) * N/n)^(1/3)
radius_from_wigner(r_s::Real, N::Int64) = N^(1/3) * r_s

wigner_from_volume(V::Real, N::Int64) = (3.0/(4.0 * π * N/V))^(1/3)

# Sampler for Uniform Spherical

@with_kw struct UniformSphericalSampler{T} <: Sampleable{Multivariate, Continuous} where {T <: Real}
    dim::Int64 = 3
    r::T = 1.0
end

Base.length(s::UniformSphericalSampler) = s.dim

function _rand!(rng::AbstractRNG, spl::UniformSphericalSampler, x::AbstractVector{T}) where T<:Real
    dim = spl.dim
    r = spl.r
    s = 0.0
    x .= randn(rng, dim)
    normalize!(x)
    @. x *= r
end


# Sampler for Uniform Ball

@with_kw struct UniformBallSampler{T} <: Sampleable{Multivariate, Continuous} where {T <: Real}
    dim::Int64 = 3
    r::T
end

Base.length(s::UniformBallSampler) = s.dim

function _rand!(rng::AbstractRNG, spl::UniformBallSampler, x::AbstractVector{T}) where T<:Real
    dim = spl.dim
    r = spl.r
    # defer to UniformSphericalSampler for calculation of unit-vector
    _rand!(rng, UniformSphericalSampler(dim=dim, r=r), x)

    # re-scale x
    u = rand(rng)
    r = (u^inv(dim))
    x .*= r
    return x
end


###############################################################################
#### Blockaded particles in Sphere ############################################
###############################################################################

# Blockaded particles in Ball

@with_kw struct UniformBlockadedBallSampler{T} <: Sampleable{Multivariate, Continuous} where {T <: Real}
    dim::Int64 = 3
    n_init::Int64 = 1e3
    r::T
    r_bl::T = 5
end

Base.length(s::UniformBlockadedBallSampler) = s.dim

function _rand!(rng::AbstractRNG, spl::UniformBlockadedBallSampler, x::AbstractVector{T}) where T<:Real
    x = _rand!(rng, UniformBallSampler(dim=spl.dim, r=spl.r), x)
    return x
end

function _rand!(rng::AbstractRNG, spl::UniformBlockadedBallSampler, x::DenseMatrix{T}) where T<:Real
    N = size(x)[2]
    ball_sampler = UniformBallSampler(spl.dim, spl.r)
    pos_init = rand(rng, ball_sampler, spl.n_init)
    x .= pick_hard_spheres(pos_init, N, spl.r_bl)
    return x
end