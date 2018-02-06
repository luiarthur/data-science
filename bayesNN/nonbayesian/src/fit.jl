function fit(X::Matrix{Float64}, Y::Matrix{Int32},
             numUnitsInHiddenLayer::Array{Int64}, alpha::Float64;
             lambda::Float64=0.0, activationFn::Function=sigmoid,
             maxIters::Int64=100, eps::Float64=1E-3, eps_init::Float64=0.1)

  """
  Assumes X has a intercept column.
  numUnitsInHiddenLayer: including bias unit
  """
  const (N,P) = size(X)
  const numClasses = size(Y,2)
  const numUnitsInLayer = [P; numUnitsInHiddenLayer; numClasses]
  const L = length(numUnitsInLayer)

  const state = begin
    local A = Array{Matrix{Float64}}(L)
    A[1] = X

    local Theta = [ rand(numUnitsInLayer[l], numUnitsInLayer[l+1]) * eps_init * 2 - eps_init for l in 1:(L-1)]
    local ThetaGrad = deepcopy(Theta)

    local Z = Array{Matrix{Float64}}(L)
    for l in 2:L
      Z[l] = zeros(N, numUnitsInLayer[l])
      A[l] = zeros(N, numUnitsInLayer[l])
    end

    State(Theta,ThetaGrad,A,Z)
  end

  ### Check gradient######
  #const (dummy, vDimGlobal) = toVec(state.Theta)
  #i = 1
  #function f(v::Vector{Float64})
  #  i += 1
  #  println(i)
  #  const Theta = unvec(v, vDimGlobal)
  #  return costM(Theta, X, Y, activationFn=activationFn, lambda=lambda)
  #end
  ########################

  function update(state::State)
    forwardProp!(X, Y, state, activationFn=activationFn)
    backProp!(X, Y, state, activationFn=activationFn, lambda=lambda)

    (v, vDim) = toVec(state.Theta)
    (g, gDim) = toVec(state.ThetaGrad)

    v .-= alpha * g

    const newTheta = unvec(v, vDim)
    for l in 1:length(state.Theta)
      state.Theta[l] .= newTheta[l]
    end

    return cost(state, X, Y, activationFn=activationFn, lambda=lambda)
  end 

  return optim(state, update, maxIters=maxIters, eps=eps) # TODO:TURN ON

  #### TODO: TURN OFF THE FOLLOWING WHEN DONE CHECKING
  #(init_v, dummy) = toVec(state.Theta)
  #init_v = rand(length(init_v)) - 0.5
  ##println(length(init_v))
  #return optimize(f, init_v, iterations=20)
end
