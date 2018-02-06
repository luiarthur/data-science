function sigmoid(x::Float64; gradient::Bool=false)
  if gradient
    return sigmoid(x) * sigmoid(-x)
  else
    return 1 / (1 + exp(-x))
  end
end

function sigmoid(x::Matrix{Float64}; gradient::Bool=false)
  ### Vectorized implementation
  if gradient
    return sigmoid(x) .* sigmoid(-x)
  else
    return 1 ./ (1 + exp.(-x))
  end
end

function sigmoid(x::Array{Float64}; gradient::Bool=false)
  ### Vectorized implementation
  if gradient
    return sigmoid(x) .* sigmoid(-x)
  else
    return 1 ./ (1 + exp.(-x))
  end
end

#= Test. When possible, use the vectorized implementation
@time sigmoid(randn(100,50));
@time sigmoid(randn(100,50), gradient=true);
@time [sigmoid(3.0) for i in 1:5000];
@time [sigmoid(3.0, gradient=true) for i in 1:5000];
=# 
