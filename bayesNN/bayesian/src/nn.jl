include("MCMC.jl")

immutable Prior 
  a0::Float64
  b0::Float64
  a1::Float64
  b1::Float64
end

State = Array{Matrix{Float64}}

function logistic(x::Float64)
  1 / (1 + exp(-x))
end

#logistic.(randn(3,5))

function fit_nn(y::Matrix{Int32}, X::Matrix{Float64}, num_units::Array{Int32},
                prior::Prior, B::Int64, burn::Int64;
                thin::Int=1, addIntercept=true, printFreq::Int64=100)::Array{State}
  """
  num_units: number of units (only including layers with weights)
  """

  num_units

  const N = size(X)
  const X1 = addIntercept ? [ones(N) X] : X
  prepend!(num_units, size(X1,2))
  const L = length(num_units) # number of layers with Parameters

  function update(state::State)::Float64
    function ll(state::State)::Float64
      A = X1
      for l in 2:L
        A = logistic.(A * state[l])
        A[:,1] = ones(N)
      end
      return y .* log(A) + (1-y) .* log(1-A)
    end

    function lp(state::State)::Float64
      #= TODO
      log prior
      =#
    end

    #= TODO
    update
    =#
    return state
  end


  const init = [ randn(N, num_units) for nu in num_units ]
  return gibbs(init, update, B, burn, printFreq)
end


