using Revise

using RandomPositions
using Distances
using LinearAlgebra

package_density = 3.5
N_spins = 100
radius_ws = 1  # wigner seitz radius
total_volume = RandomPositions.volume_from_radius(radius_ws) * N_spins
filled_volume = package_density * total_volume
r_bl = RandomPositions.radius_from_volume(filled_volume / N_spins)
total_radius = RandomPositions.radius_from_wigner(radius_ws, N_spins)

ball_sampler = UniformBlockadedBallSampler(r=total_radius, r_bl=r_bl, n_init=1e5)
pos = rand(ball_sampler, N_spins)

dist = pairwise(Euclidean(), pos, dims = 2)
dist .= dist + Diagonal(ones(N_spins))* 1e8
minimum(dist)