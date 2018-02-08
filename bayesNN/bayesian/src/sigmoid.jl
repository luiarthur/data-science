function sigmoid(x::Float64)
  return 1 / (1 + exp(-x))
end

function sigmoid(x::Matrix{Float64})
  return 1 ./ (1 + exp.(-x))
end

function sigmoid(x::Array{Float64})
  return 1 ./ (1 + exp.(-x))
end
