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
