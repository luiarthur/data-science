function predict(X::Matrix{Float64}, Theta::Array{Matrix{Float64}};
                 activationFn::Function=sigmoid)
  const N = size(X,1)
  const L = length(Theta)
  h = X

  for l in 1:L
    h = activationFn(h * Theta[l])
  end

  predY = [indmax(h[i,:]) for i in 1:N]

  return (h,predY)
end
