function updateTheta(state::State, data::Data, prior::Prior)::Void
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

  Theta_vec = metropolis(Theta_vec, ll, lp, prior.cs)

  state.Theta .= unvec(Theta_vec, dims)

  state.loglike = ll(Theta_vec) / size(data.Y, 1)

  return Void()
end
