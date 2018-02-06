function predict(X::Matrix{Float64}, Theta::Array{Matrix{Float64}};
                 activationFn::Function=sigmoid)
  const N = size(X,1)
  const L = length(Theta)
  A = X

  for l in 1:L
    Z = A * Theta[l]
    A = activationFn(Z)
  end

  predY = [indmax(A[i,:]) for i in 1:N]

  return (A,predY)
end
