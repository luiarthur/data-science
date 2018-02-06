function backProp!(X::Matrix{Float64}, Y::Matrix{Int32}, state::State;
                   activationFn::Function=sigmoid)::Void

  """
  Backpropogation is an algorithm to compute
  the gradient of the cost function with respect to the
  parameters θ.
  """

  const L = getL(state)
  const N = size(X,1)

  #const Theta_grad = [zeros(theta) for theta in Theta]
  dl_plus1 = 0.0

  for n in 1:N
    for l in (L-1):-1:1
      if l == (L-1)
        dl = (state.A[end][n,:] - Y[n,:])'
        state.Theta_grad[l] .+= dl * state.A[l] / N
      else
        dl = (state.Theta[l+1] * dl)[2:end] .* activationFn(Z[l], gradient=true)
        state.Theta_grad[l] .+= dl * state.A[l][n,:] / N
      end
    end
  end

  return Void()
end
