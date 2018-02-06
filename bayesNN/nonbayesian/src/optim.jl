function optim(state::State, update::Function;
               maxIters::Int64=100, eps::Float64=1E-3)::State
  iter = 1

  J_prev = Inf
  J_curr = 0.0

  while iter < maxIters && abs(J_prev - J_curr) > eps
    J_prev = J_curr
    J_curr = update(state)
    println("iter: ", iter, " | J: ", J_curr)
    iter += 1
  end

  if (iter >= maxIters)
    println("Not converged at iteration ", iter, ".")
  else
    println("Converged at iteration ", iter, ".")
  end

  return state
end
