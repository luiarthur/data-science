function cost(Theta::Array{Matrix{Float64}},
              X::Matrix{Float64}, Y::Matrix{Int32}; lambda::Float64=0)

  ### Compute Cost (J)
  const L = length(Theta) # num Layers
  const A = Array{Matrix{Float64}}(L + 1)
  const N = size(X, 1)

  A[1] = [1 X]
  for l in 2:L
    A[l] = sigmoid.(A[l-1] * Theta[l-1])
  end

  J = -sum(Y .* log(A[end]) + (1-Y) .* log(1-A[end])) / N
  if lambda > 0
    for theta in Theta
      J += sum(theta[:,2:end] .^ 2) * lambda / (2*N)
    end
  end
  ### End of Computing Cost (J)

  ### Compute Gradient: TODO. Refer to ML coursera hw.
  for n in 1:N
  end
  ### End of Compute Gradient

  return (J, grad)
end
