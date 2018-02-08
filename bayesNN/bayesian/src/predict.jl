function mode(y::Vector{Int})
  m = sort(countmap(y))
  return m.keys[indmax(m.vals)]
end  

function predict(X, posterior; activationFn::Function=sigmoid)
  const N = size(X,1)
  const X1 = [ones(N) X]
  const B = length(posterior)
  const L = length(posterior[1].Theta)
  const numClasses = size(posterior[1].Theta[end], 2)

  pred_Y = zeros(Int, B, N)
  for i in 1:length(posterior)
    h = X1
    for l in 1:L
      h = activationFn(h * posterior[i].Theta[l])
    end
    pred_Y[i,:] = [indmax(h[n,:]) for n in 1:N]
  end

  #pred_y = mean(pred_Y, 1)
  pred_y = [mode(pred_Y[:,n]) for n in 1:N]

  return (pred_y, pred_Y)
end
