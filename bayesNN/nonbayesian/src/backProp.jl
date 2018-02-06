function backProp!(X::Matrix{Float64}, Y::Matrix{Int32}, state::State;
                   activationFn::Function=sigmoid, lambda::Float64=0.0)::Void

  """
  Backpropogation is an algorithm to compute
  the gradient of the cost function with respect to the
  parameters θ.
  """

  const K = size(Y,2)
  const L = getL(state)
  const N = size(X,1)

  #const Theta_grad = [zeros(theta) for theta in Theta]

  for n in 1:N
    dl = reshape(state.A[end][n,:] - Y[n,:], K, 1)
    sl = size(state.A[end-1], 2)
    state.ThetaGrad[end] .+= reshape(state.A[end-1][n,:], sl, 1) * dl' / N

    for l in (L-2):-1:1
      dl = (vec(state.Theta[l+1] * dl)) .* #[2:end] .* 
           activationFn(vec(state.Z[l+1][n,:]), gradient=true)
      dl = reshape(dl, length(dl), 1)
      sl = size(state.A[l], 2)
      #state.ThetaGrad[l][:,2:end] .+= (reshape(state.A[l][n,:], sl, 1) * dl'/ N)[:,2:end]
      state.ThetaGrad[l][2:end,:] .+= (reshape(state.A[l][n,:], sl, 1) * dl'/ N)[2:end,:]
      #state.ThetaGrad[l] .+= (reshape(state.A[l][n,:], sl, 1) * dl'/ N)
    end
  end

  for l in 1:(L-1)
    #state.ThetaGrad[l][:,2:end] .+= lambda * state.Theta[l][:,2:end] / N
    state.ThetaGrad[l][2:end,:] .+= lambda * state.Theta[l][2:end,:] / N
  end

  return Void()
end
