function fit(X::Matrix{Float64}, Y::Matrix{Int32},
             numUnitsInHiddenLayer::Array{Int32},
             activationFn::Function, activationGradient::Function)
  """
  Assumes X has a intercept column.
  numUnitsInHiddenLayer: including bias unit
  """
  const (N,P) = size(X)
  const numClasses = size(Y,2)
  const numUnitsInLayer = [P, numUnitsInHiddenLayer, numClasses]
  const L = length(numUnitsInLayer)

  const state = begin
    local A = Array{Matrix{Float64}}(L)
    A[1] = randn(N, size(X,2))

    local Theta = [randn(numUnitsInLayer[l], numUnitsInLayer[l+1]) for l in 1:(L-1)]
    local ThetaGrad = deepcopy(Theta)

    local Z = Array{Matrix{Float64}}(L)
    for l in 2:L
      Z[l] = zeros(N, numUnitsInLayer[l])
    end

    State(Theta,ThetaGrad,A,Z)
  end
  
  #=TODO:
    call algorithm
  =#
end
