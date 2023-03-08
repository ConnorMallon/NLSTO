module OptimizationDriver
using ChainRules
using DrWatson; quickactivate("art-inverse-dnn-fem")
include(srcdir("Preamble.jl"))
n_cells = 47 # 95
nf = 16
nd = 16 
prior = "neural" # use "pixel" for a pixel parameterisation and "neural" for a neural parameterisation
optimiser = "ADAM" #LBFGS #MMA #ADAM #IPOPT
problem = "heat" #"heat_simp" (for the SIMP solver) or "heat" (for the LS solver)
Vₘₐₓ = 0.4 
α₂ = 1e-2 

n_holes = 4 
image_size = "scale_with_mesh" # this will inrease the no. of parameters in the NN if the mesh resolution increases.
method = "constrained" # use "constrained" for use with an optimiser that already enforces the volume constraint (MMA, IPOPT) and "unconstrained" for one that doesnt (ADAM, LBFGS)

var_params = n_cells,α₂,nf,nd,image_size,n_holes,prior,method,optimiser,problem,Vₘₐₓ
fcalls,gcalls,iters,J,p0,pf,Ω0,Ωf,js,ts = RunDriver(var_params)
end # module