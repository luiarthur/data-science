function cost(state::State, X::Matrix{Float64}, Y::Matrix{Int32};
              activationFn::Function=sigmoid, lambda::Float64=0)::Float64

  const N = size(X, 1)

  J::Float64 = -sum(Y .* log.(state.A[end]) + (1-Y) .* log.(1-state.A[end])) / N
  if lambda > 0
    for theta in state.Theta
      J += sum(theta[2:end,:] .^ 2) * lambda / (2*N)
    end
  end

  return J
end


#function costM(Theta::Array{Matrix{Float64}}, X::Matrix{Float64}, Y::Matrix{Int32};
#               activationFn::Function=sigmoid, lambda::Float64=0)::Float64
#
#  const N = size(X, 1)
#  const (A,dummy) = predict(X, Theta, activationFn=activationFn)
#
#  J = -sum(Y .* log.(A) + (1-Y) .* log.(1-A)) / N
#  if lambda > 0
#    for theta in Theta
#      J += sum(theta[:,2:end] .^ 2) * lambda / (2*N)
#    end
#  end
#
#  return J
#end

