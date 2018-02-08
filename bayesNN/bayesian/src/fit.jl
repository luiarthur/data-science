function fit(y::Vector{Int64}, X::Matrix{Float64}, H::Int64;
             B::Int64=100, burn::Int64=100, numClasses::Int=0, prior=defaultPrior,
             thin::Int=1, printEvery::Int64=100)::Array{State}

  """
  H: number of hidden units including bias
  """
  if numClasses == 0
    numClasses = length(unique(y))
  end

  ### Create Data Class ###
  const N = size(X, 1)
  const Y = begin
    YY = zeros(Int64, N, numClasses)
    for i in 1:N
      c = (y[i] == 0) ? numClasses : y[i]
      YY[i, c] = 1
    end
    YY
  end
  const data = Data(Y, [ones(N) X])
  ### Create Data Class ###


  function update(state::State)::Void
    updateLambda(state, data, prior)
    updateTheta(state, data, prior)
    return Void()
  end

  const init_Theta = [randn(size(data.X,2), H), randn(H, numClasses)]
  const init = State(init_Theta, 1.0, 1.0)

  return gibbs(init, update, B, burn, thin, printEvery)
end
