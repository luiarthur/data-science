function forwardProp!(X::Matrix{Float64}, Y::Matrix{Int32}, state::State;
                      activationFn::Function=sigmoid)::Void

  const L = getL(state)
  const N = size(X,1)

  for l in 2:L
    state.Z[l] .= state.A[l-1] * state.Theta[l-1]
    state.A[l] .= activationFn(state.Z[l])
    if l < L
      state.A[l][:,1] = 1
    end
  end

  return Void()
end
