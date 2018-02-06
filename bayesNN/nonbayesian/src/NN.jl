module NN

function sigmoid(x; gradient::Bool=false)
  ### Vectorized implementation
  if gradient
    return sigmoid(x) .* sigmoid(-x)
  else
    return 1 ./ (1 + exp.(-x))
  end
end

function optim(Theta::Array{Matrix{Float64}}, Grad::Array{Matrix{Float64}},
               cost::Function, maxIters::Int64=100, eps::Float64=1E-3)::Theta
  iter = 1

  J_prev = Inf
  J_curr = 0.0

  while iter < maxIters || abs(J_prev - J_prev) > eps
    J_prev = J_curr
    J_curr = cost(Theta, Grad)
  end

  if (iter >= maxIters)
    println("Not converged at iteration ", iter, ".")
  else
    println("Converged at iteration ", iter, ".")
  end

  return state
end

function fit(X::Matrix{Float64}, Y::Matrix{Int32},
             numUnitsInHiddenLayer::Array{Int64}, alpha::Float64;
             lambda::Float64=0.0, activationFn::Function=sigmoid,
             maxIters::Int64=100, eps::Float64=1E-3)

end

end
