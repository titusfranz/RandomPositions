using RandomPositions
using Distributions
using Rotations
using CSV
using DataFrames

k_b = 1.3806e-23
m_Rb = 1.6605e-27 * 87
T = 60e-6

w_x = 198. * sqrt(2)
w_yz = 110.
n_gs = round(374000)

params = RydbergParams(
    distribution = MvNormal([w_x; w_yz; w_yz]),
    n_gs = n_gs,
    rot = RotZ(2π/360 * -45.0),
    t_exc = 10.0,
    Ω_c_0 = 0.6,
    σ_c = 70.0,
    pos_c = [0.0; 0.0],
    Ω_p_0 = 0.83,
    σ_p = 1500.0,
    pos_p = [0.0; 0.0],
    Γ_p = 6.0,
    Δ_1γ = 97.0,
    Δ_2γ = 0.0,
    γ = 0.0,
    C6 = 60000.0,
    pulse_shape = :rect,  # 'Shape of the excitation pulse': objects=['rect', 'gauss', 'sech2', 'lorentz']
    laser_width = 0.2
)


pos = excite_rydbergs(params)
pos_df = DataFrame(pos', [:x, :y, :z]);
CSV.write("pos.csv", pos_df)
pos