module BayesNN

import Distributions.Normal
import Distributions.InverseGamma # shape and rate. mean = b / (a-1)
import Distributions.MvNormal

include("util.jl")
include("MCMC.jl")
include("Data.jl")
include("Prior.jl")
include("sigmoid.jl")
include("State.jl")
include("updateLambda.jl")
include("updateTheta.jl")
include("fit.jl")

end
