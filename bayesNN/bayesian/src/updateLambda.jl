function updateLambda(state::State, data::Data, prior::Prior)::Void
  const t2 = [t .^ 2 for t in state.Theta]
  const a0_new = prior.a0 + sum( [size(t,2) for t in state.Theta] ) / 2
  const b0_new = prior.b0 + sum([ sum(tt[1,:]) for tt in t2 ]) / 2

  state.lambda_0 = rand(InverseGamma(a0_new, b0_new))

  const a1_new = prior.a1 + sum([(size(t,1)-1) * size(t,2) for t in state.Theta]) / 2
  const b1_new = prior.b1 + sum([ sum(tt[2:end,:]) for tt in t2 ]) / 2

  state.lambda_1 = rand(InverseGamma(a1_new, b1_new))

  return Void()
end
