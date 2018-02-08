immutable Prior 
  a::Float64
  b::Float64
  lambda_0::Float64
  cs::Float64
end

const defaultPrior = Prior(10, 2, 10, .01)
