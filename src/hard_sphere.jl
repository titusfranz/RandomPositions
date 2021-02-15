"Criterion to determine whether to pick an atom. Needs to implement 'test_pick_condition'."
abstract type PickingCondition end

"Criterion to determine when to stop the picking process. Needs to implement 'test_pick_criterion'.'"
abstract type StoppingCondition end

"Criterion to determine which atoms to block. Needs to implement 'is_blocked'."
abstract type BlockingCondition end

"""
    excite_rydbergs(pos, excitation_criteria, stopping_criteria)

Chooses rydberg atoms from pos according to excitation and stopping_criteria.
"""
function pick_hard_spheres(
    pos,
    picking_condition::PickingCondition,
    stopping_condition::StoppingCondition,
    blocking_condition::BlockingCondition,
)
    tree = KDTree(pos)
    gs_length = size(pos, 2)
    inital_indices = IntSet(1:gs_length)
    picked_indices = Array{Int64,1}()
    i = 0
    while true
        if test_stopping_condition(picked_indices, inital_indices, stopping_condition)
            break
        end
        i = pop!(inital_indices)

        if test_picking_condition(i, picking_condition)
            blocked = is_blocked(tree, pos, i, blocking_condition)
            setdiff!(inital_indices, blocked)
            append!(picked_indices, i)
        end

    end

    pos[:, picked_indices]
end

###############################################################################
#### Stopping the iteration ###################################################
###############################################################################

"Iterate only once. Iterate one by one until end of pos."
struct NoStop <: StoppingCondition end

function test_stopping_condition(
    picked_indices,
    initial_indices,
    stopping_condition::NoStop,
)
    return isempty(initial_indices)
end

"Iterate until fixed number N of Rydberg is reached. Chooses index randomly"
struct FixedNStop <: StoppingCondition
    N::Int64
end

function test_stopping_condition(
    picked_indices,
    initial_indices,
    stopping_criterion::FixedNStop,
)
    if isempty(initial_indices)
        @warn "Already fully blockaded. Desired number of rydbergs was not reached"
        return true
    end
    length(picked_indices) >= stopping_criterion.N
end

###############################################################################
#### Excitation criteria ######################################################
###############################################################################

"Excite each atom"
struct PickAlways <: PickingCondition end

function test_picking_condition(i, excite_method::PickAlways)
    return true
end

"Excite atom with probability 'p' and blockade radius 'R_b'."
struct UniformPicking <: PickingCondition
    exc_prob::Float64
end

function test_picking_condition(i, excite_method::UniformPicking)
    return excite_method.exc_prob > rand()
end

"Excite eachatom with independant probability 'p' and blockade radius 'R_b'."
struct NonuniformPicking <: PickingCondition
    exc_prob::Vector{Float64}
end

function test_picking_condition(i, excite_method::NonuniformPicking)
    excite_method.exc_prob[i] > rand()
end

###############################################################################
#### Blocking condition #######################################################
###############################################################################

"Same blockade for each atom"
struct UniformBlockade <: BlockingCondition
    r_bl::Float64
end

function is_blocked(tree, pos, i, blocking_condition::UniformBlockade)
    new_pos = pos[:, i]
    inrange(tree, new_pos, blocking_condition.r_bl, false)
end

"Position dependent blocking"
struct NonuniformBlockade <: BlockingCondition
    r_bl::Vector{Float64}
end

function is_blocked(tree, pos, i, blocking_condition::NonuniformBlockade)
    new_pos = pos[:, i]
    inrange(tree, new_pos, blocking_condition.r_bl[i], false)
end

###############################################################################
#### Helper Methods ###########################################################
###############################################################################

pick_hard_spheres(pos, p::Float64, r_bl::Float64) =
    pick_hard_spheres(pos, UniformPicking(p), NoStop(), UniformBlockade(r_bl))

pick_hard_spheres(pos, N::Int64, r_bl::Float64) =
    pick_hard_spheres(pos, PickAlways(), FixedNStop(N), UniformBlockade(r_bl))

pick_hard_spheres(pos, p::Vector{Float64}, r_bl::Vector{Float64}) =
    pick_hard_spheres(pos, NonuniformPicking(p), NoStop(), NonuniformBlockade(r_bl))





    