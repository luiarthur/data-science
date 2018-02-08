function updateTheta(state::State, data::Data, prior::Prior)::Void
  const N = size(data.Y,1)
  (Theta_vec, dims) = toVec(state.Theta)
  P1 = size(state.Theta[1], 1)
  P2 = size(state.Theta[2], 1)
  CAP = prod(dims[1])

  function ll(v::Vector{Float64})
    const Theta = unvec(v, dims)
    A = sigmoid(data.X * Theta[1])
    A[:,1] = 1
    const P = sigmoid(A * Theta[2])
    return sum(data.Y .* log.(P) + (1-data.Y) .* log.(1-P))
  end

  function lp(v::Vector{Float64})
    const v2 = v.^2
    out = sum(v2[1:P1:CAP] / (2*prior.lambda_0))
    out += sum(v2[(CAP+1):P2:end] / (2*state.lambda))
    return -out
  end


  #Theta_vec = metropolis(Theta_vec, ll, lp, prior.cs)
  #state.Theta .= unvec(Theta_vec, dims)
  #state.loglike = ll(Theta_vec) / N
  ### Speed up ###
  const cand = rand(MvNormal(Theta_vec, prior.cs))
  const log_fc_curr = ll(cand) + lp(cand)
  if log_fc_curr - state.log_fc * N > log(rand())
    state.Theta .= unvec(cand, dims)
    state.log_fc = log_fc_curr / N
  end
  ### End of Speed up ###


  return Void()
end
