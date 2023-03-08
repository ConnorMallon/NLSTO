using Gridap
using Gridap.FESpaces
using Gridap.ReferenceFEs
using Gridap.Arrays
using Gridap.MultiField
using Gridap.MultiField
using Gridap.Algebra
using Gridap.Geometry
import Gridap: ∇
import Gridap.Algebra: NewtonRaphsonSolver
using Gridap
using GridapEmbedded
using Gridap
using GridapEmbedded.LevelSetCutters
using Flux
using Random, Distributions
using ChainRulesCore
using Nonconvex
using Optim
using Test
using NLsolve
using Roots
using Statistics
using Zygote
using LinearAlgebra
using Plots
using ImageFiltering
using ForwardDiff
using ReverseDiff
using ForwardDiff
using LinearAlgebra
using Test
using LineSearches
using AbstractDifferentiation

struct PriorType{Kind} end
struct MethodType{Kind} end
struct OptimiserType{Kind} end
struct ProblemType{Kind} end
struct ImageSizeType{Kind} end
struct TOmethodType{Kind} end

#include("LSR.jl")
include("FEProblem.jl")
include(srcdir("ChainRules.jl"))
include(srcdir("Optimisation.jl"))
include(srcdir("Network.jl"))
include(srcdir("Utilities.jl"))
include(srcdir("Routine.jl"))

start_time=time()
js=[]
ts=[]
i = 0

Nonconvex.@load NLopt
Nonconvex.@load Ipopt
Nonconvex.@load MMA