module MyNN

using Optim ### TODO: Turn off after debugging

include("sigmoid.jl")
include("State.jl")
include("cost.jl")
include("forwardProp.jl")
include("backProp.jl")
include("optim.jl")
include("fit.jl")
include("predict.jl")

end
