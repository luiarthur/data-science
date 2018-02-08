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
