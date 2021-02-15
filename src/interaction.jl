abstract type InteractionRange end

@with_kw struct NearestNeighbor <: InteractionRange
    J_nn::Float64 = 1.0
    N::Int64 = 10
end

function get_J(int_range::NearestNeighbor)
    N = int_range.N
    int_range.J_nn * SymTridiagonal(zeros(N), ones(N - 1))
end

"Power law interaction with distance r as C/r^α	"
@with_kw struct PowerLaw <: InteractionRange
    α::Float64 = 6.0
    C::Float64 = 1.0
end

function get_J(pos, power_law::PowerLaw)
    N = size(pos)[2]
    dist = pairwise(Euclidean(), pos, dims = 2)
    dist .= dist + Diagonal(ones(N))
    power_law.C .* (1.0 ./ dist .^ power_law.α - Diagonal(ones(N)))
end

function dipolar_angular_dependence(pos)
    angles = zeros((size(pos, 2), size(pos, 2)))
    for i in 1:size(pos, 2)
        for j in 1:i-1
            d = pos[:, i] - pos[:, j]
            normalize!(d)
            angles[i, j] = 1 - 3*d[3]^2
        end
    end
    angles + angles'
end

function get_J_dipolar(t, rydberg_params)
    pos = excite_rydbergs(rydberg_params, t_exc=t)
    N = size(pos, 2)
    dist = pairwise(Euclidean(), pos)

    J = get_J(pos, PowerLaw(3, 2*2π*1150)) # C3 for mj1/2 to mj1/2 between 48S and 48P state

    return J .* dipolar_angular_dependence(pos)
end


function get_J_dipolar1D(p)
    pos = getPosition1D(p)
    N = size(pos, 2)
    dist = pairwise(Euclidean(), pos)

    J = get_J(pos, PowerLaw(3, 2*2π*1150)) # C3 for mj1/2 to mj1/2 between 48S and 48P state

    return J .* dipolar_angular_dependence(pos),pos
end
