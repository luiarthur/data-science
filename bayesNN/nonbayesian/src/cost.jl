function cost(Theta::Array{Matrix{Float64}},
              X::Matrix{Float64}, Y::Matrix{Int32}; 
              activationFn::Function=sigmoid,
              activationGradient::Function=sigmoidGradient,
              lambda::Float64=0)

  ### Compute Cost (J)
  const L = length(Theta) + 1 # number of layers
  const A = Array{Matrix{Float64}}(L)
  const N = size(X, 1)
  const numClasses = size(Y,2)

  const Z = Array{Matrix{Float64}}(L)

  A[1] = [1 X]
  for l in 2:L
    Z[l] = A[l-1] * Theta[l-1]
    A[l] = activationFn.(Z[l])
  end

  J = -sum(Y .* log(A[end]) + (1-Y) .* log(1-A[end])) / N
  if lambda > 0
    for theta in Theta
      J += sum(theta[:,2:end] .^ 2) * lambda / (2*N)
    end
  end
  ### End of Computing Cost (J)

  ### Compute Gradient: TODO. Refer to ML coursera hw.
  const Theta_grad = [zeros(theta) for theta in Theta]
  dl_plus1 = 0.0

  for n in 1:N
    for l in (L-1):-1:1
      if l == (L-1)
        dl = (A[end][n,:] - Y[n,:])'
        Theta_grad[l] .+= dl * A[l] / m
      else
        dl = (Theta[l+1] * dl)[2:end] .* activationGradient(Z[l])
        Theta_grad[l] .+= dl * A[l][n,:] / m
      end
    end
  end
  ### End of Compute Gradient

  return (J, Theta_grad)
end
