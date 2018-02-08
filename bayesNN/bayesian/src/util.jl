function toVec(xs::Array{Matrix{Float64}})
  const L = length(xs)
  out = Array{Float64,1}()

  for l in 1:L
    append!(out, vec(xs[l]))
  end

  const dims = [size(x) for x in xs]

  return (out, dims)
end

function unvec(v, dims)
  const L = length(dims)
  out = Array{Matrix{Float64}}(L)
  idxUpper = cumsum([prod(d) for d in dims])
  idxLower = [1; idxUpper[1:end-1]+1]
  for l in 1:L
    il = idxLower[l]
    iu = idxUpper[l]
    out[l] = reshape(v[il:iu], dims[l])
  end
  return out
end

function getNumClasses(y)
  return length(unique(y))
end

function format_y(y)
  yy = copy(y)
  yy[yy .== 0] = getNumClasses(y)
  return yy
end

#function computeError(X, y, posterior; numClasses=0)::Float64
#  yy = format_y(y)
#  (dummy, pred) = NN.predict(X, Theta)
#  return mean(yy .!= pred)
#end

function scale(X::Matrix{Float64}, xbar=nothing, xmax=nothing)
  if xbar === nothing || xmax === nothing
    xbar = mean(X, 1)
    xmax = maximum(X)
    out = (X .- xbar) / xmax
    return (out, xbar, xmax)
  else
    return (X .- xbar) / xmax
  end
end

function format_X(X)
  const N = size(X,1)
  const P = prod(size(X, 2, 3))
  newX = zeros(N, P)
  for i in 1:N
    newX[i, :] = vec(X[i,:,:])
  end
  return newX
end

function y_vec2mat(y::Vector{Int64})
  const N = length(y)
  const numClasses = getNumClasses(y)
  Y = zeros(Int64, N, numClasses)
  for i in 1:N
    c = (y[i] == 0) ? numClasses : y[i]
    Y[i, c] = 1
  end
  Y
end
