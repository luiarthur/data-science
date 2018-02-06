function stupidTest!(X::Matrix{Float64})::Void
  X .= X + 1
  X[end,end] = 9
  return Void()
end

function stupidTest3(X::Matrix{Float64})::Void
  X .= X + 1
  X[end,end] = 9
  return Void()
end


function stupidTest2!(X::Matrix{Float64})::Void
  X = X + 1
  return Void()
end

#=
@time const x = zeros(1000,5000);
@time stupidTest!(x)
@time stupidTest2!(x)
@time stupidTest3(x)
=#

type Bla
  x::Array{Float64}
  y::Array{Float64}
end

const bla = Bla(zeros(5,3), zeros(3,5))
bla.x += 1
bla.y += 1

bla2  = bla
bla3 = deepcopy(bla)
bla2.x += 1

arr = [Bla(zeros(2,3), zeros(1,1)) for i in 1:3]

for a in arr
  a.x += 2
end
