# compare with the other gibbs
function gibbs{T}(init::T, update::Function, B::Int, burn::Int, thin::Int, printFreq::Int)
  const out = Array{T}(B)
  const state = deepcopy(init)
  out[1] = deepcopy(init)

  for i in 2:(B+burn)
    if i <= burn
      update(state)
    else
      for t in 1:thin
        update(state)
      end
      out[i-burn] = deepcopy(state)
    end

    if printFreq > 0 && i % printFreq == 0
      print("\rProgress: ",i,"/",B+burn)
    end
  end

  return out
end

"""
metropolis step with normal proposal
"""
function metropolis(curr::Float64, ll::Function, lp::Function, cs::Float64)

  const cand = rand(Normal(curr,cs))

  if ll(cand) + lp(cand) - ll(curr) - lp(curr) > log(rand())
    new_state = cand
  else
    new_state = curr
  end

  return new_state
end


"""
multivariate metropolis step with diagonal covariance proposal
"""
function metropolis(curr::Vector{Float64}, ll::Function, lp::Function, cs::Float64)

  const cand = rand(MvNormal(curr,cs))

  if ll(cand) + lp(cand) - ll(curr) - lp(curr) > log(rand())
    new_state = cand
  else
    new_state = curr
  end

  return new_state
end

"""
Langevin Monte Carlo (multivariate)
"""
function lmc(curr::Vector{Float64}, gradLogFC::Function, eps::Float64)
  rand(MvNormal(curr + eps * gradLogFC(curr), sqrt(2*eps)))
end

"""
Hamiltonian Monte Carlo (multivariate)
"""
function hmc(curr::Vector{Float64}, U::Function, grad_U::Function,
             eps::Float64, L::Int)::Vector{Float64}
  const M = length(curr)
  q = curr
  p = randn(M)

  # Make half a step for momentum at beginning 
  p -= eps * grad_U(q) / 2
  
  # Alternate full steps for position and momentum
  for i in 1:L
    # Make a full step for the position
    q += eps * p
    # Make a full step for the momentum, except at end of trajectory
    if i <L
      p -= eps * grad_U(q)
    else
      # Make a half step for momentum at the end.
      p -= eps * grad_U(q) / 2
    end
  end

  # Negate momentum at end of trajectory to make the proposal symmetric
  p = -p

  # Evaluate potential and kinetic energies at start and end of trajectory
  const current_U = U(current_q)
  const current_K = sum(current_p .^ 2) / 2
  const proposed_U = U(q)
  const proposed_K = sum(p .^ 2) / 2

  # Accept or reject the state at end of trajectory, returning either
  # the position at the end of the trajectory or the initial position
  if current_U - proposed_U + current_K - proposed_K > log(rand())
    return q # accept
  else
    return current_q # reject
  end
end
