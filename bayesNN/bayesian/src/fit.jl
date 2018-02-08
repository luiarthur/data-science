function fit(y::Vector{Int64}, X::Matrix{Float64}, H::Int64;
             B::Int64=100, burn::Int64=100, numClasses::Int=0, 
             prior::Prior=defaultPrior, printLoglike::Bool=false,
             thin::Int=1, printEvery::Int64=100, init=nothing, eps::Float64=.001)

  """
  H: number of hidden units including bias
  """
  if numClasses == 0
    numClasses = length(unique(y))
  end

  ### Create Data Class ###
  const N = size(X, 1)
  const Y = y_vec2mat(y)
  const data = Data(Y, [ones(N) X])
  ### Create Data Class ###


  function update(state::State)::Void
    updateLambda(state, data, prior)
    updateTheta(state, data, prior)
    if printLoglike
      println("log full conditional: ", state.log_fc)
    end
    return Void()
  end

  const init_Theta = [rand(size(data.X,2), H) * 2eps - eps, randn(H, numClasses)]
  const init_lambda = 1.0
  const init_loglike = -Inf
  if init === nothing
    init = State(init_Theta, init_lambda, init_loglike)
  end

  return gibbs(init, update, B, burn, thin, printEvery)
end
