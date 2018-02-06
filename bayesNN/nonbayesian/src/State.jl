immutable State 
  Theta::Array{Matrix{Float64}}
  ThetaGrad::Array{Matrix{Float64}}
  A::Array{Matrix{Float64}}
  Z::Array{Matrix{Float64}}
end

function getL(state::State)::Int32
  """
  Get number of layers (L), including the response layer.
  """
  return length(state.Theta) + 1
end
