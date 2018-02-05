function sigmoid(x::Float64)
  1 ./ (1 + exp(-x))
end
