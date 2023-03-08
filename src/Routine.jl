function RunDriver(var_params)

  n_cells,α₂,nf,nd,imagesizeName,n_holes,priorName,methodName,optimiserName,problemName,Vₘₐₓ= var_params
  image_size = ImageSizeType{Symbol(imagesizeName)}()
  prior = PriorType{Symbol(priorName)}() 
  method = MethodType{Symbol(methodName)}()
  optimiser = OptimiserType{Symbol(optimiserName)}()
  problem = ProblemType{Symbol(problemName)}()
  
  # Background model 
  bgmodel = build_bgmodel(problem,n_cells)

  # Spaces
  Vbgϕ,Ubgϕ,Vbgu,Ubgu = get_FE_Spaces(problem,bgmodel)

  # Initialise parameters
  println("Prior Initialising")
  n_holes= n_holes#2#4#2#3#4
  Vi=0.6
  pP   = nucleate_holes(n_holes,bgmodel,Vi,Vbgϕ,problem) 
  pN0,N = NetworkInit(bgmodel,nf,nd,pP,image_size)
  pN = pN0
  p0 = initialise_p(pP,pN,prior)

  # Defining fixed parameters
  # p_to_ϕₙ₁
  # ϕₙ₁_to_ϕᵧ₂
  filter_weights = [0 0 0.4 0 0; 0 2.4-2^0.5 1.4 2.4-2^0.5 0 ; 0.4 1.4 2.4 1.4 0.4 ; 0 2.4-2^0.5 1.4 2.4-2^0.5 0 ; 0 0 0.4 0 0]./13.543145750507621
  ab_r = AD.ReverseDiffBackend()
  pb_f = AD.pullback_function(ab_r, ϕₙ₁ -> apply_filter(filter_weights,bgmodel,ϕₙ₁) , zeros(num_free_dofs(Vbgϕ)))
  # ϕᵧ₂_to_ϕₛ₃
  res_LSR, jac_LSR = get_LSR_residual_and_jacobian(Vbgϕ)
  #ϕₛ₃_to_ϕ
  Vₘᵢₙ = 0.0#Vₘₐₓ#0.5#0.5
  # ϕ_to_u 
  a_FE, b_FE, res_FE = get_FE_forms(problem,Vbgϕ,α₂)
  # ϕ_to_j
  j = get_objective_function(problem,Vbgϕ,α₂)

  # building maps
  p_to_ϕₙ₁ = NeuralGeometry(N,pP) # already built
  ϕₙ₁_to_ϕᵧ₂ = LinearFilter(bgmodel,filter_weights,pb_f)
  ϕᵧ₂_to_ϕₛ₃ = InitialisableFEStateMap(res_LSR,jac_LSR,get_geo_params,Vbgϕ,Ubgϕ,Vbgϕ)
  ϕₛ₃_to_ϕ = VolumeConstraintMap(ϕ_b_to_ϕc,b,ϕ_b_to_Vol,Vbgϕ,Vₘₐₓ,problem)
  ϕ_to_u = AffineFEStateMap(a_FE,b_FE,res_FE,Vbgϕ,Ubgu,Vbgu)
  u_to_j = LossFunction(j,Vbgϕ,Ubgu)

  ϕ_to_Vol = VolumeMap(ϕc_to_Vol,Vbgϕ,problem)

  function p_to_j(p,prior,method,problem)
    ϕₙ₁  = p_to_ϕₙ₁(p,prior)      # parameters (p) to unfiltered unconstrained nodal values (s)
    ϕᵧ₂ = ϕₙ₁_to_ϕᵧ₂(ϕₙ₁)     # unfiltered unconstrained nodal values (s) to filtered unconstrained values (ϕᵤ)     
    ϕₛ₃ = ϕᵧ₂_to_ϕₛ₃(ϕᵧ₂,MethodType{Symbol("unconstrained")}(),problem)       
    ϕ  = ϕₛ₃_to_ϕ(ϕₛ₃,method)    # filtered unconstrained values (ϕᵤ) to filtered constrained values (ϕ)
    u  = ϕ_to_u(ϕ)#,bg_params)      # filtered constrained values (ϕ) to solution values (u)
    j  = u_to_j(u,ϕ)#,bg_params)    # solution values (u) to objective values (j)
  end
  p_to_j(prior,method,problem) = p -> p_to_j(p,prior,method,problem)
  
  function p_to_Vol(p,prior,method,problem)
    ϕₙ₁  = p_to_ϕₙ₁(p,prior)      # parameters (p) to unfiltered unconstrained nodal values (s)
    ϕᵧ₂ = ϕₙ₁_to_ϕᵧ₂(ϕₙ₁)     # unfiltered unconstrained nodal values (s) to filtered unconstrained values (ϕᵤ)      
    ϕₛ₃ = ϕᵧ₂_to_ϕₛ₃(ϕᵧ₂,MethodType{Symbol("unconstrained")}(),problem)       
    ϕ  = ϕₛ₃_to_ϕ(ϕₛ₃,method)    # filtered unconstrained values (ϕᵤ) to filtered constrained values (ϕ)
    Vol = ϕ_to_Vol(ϕ)
  end

  function Vol_constr_ub(p,prior,method,problem,Vₘₐₓ) 
    ub = p_to_Vol(p,prior,method,problem) - Vₘₐₓ
  end
  Vol_constr_ub(prior,method,problem,Vₘₐₓ) = p -> Vol_constr_ub(p,prior,method,problem,Vₘₐₓ) 

  function Vol_constr_lb(p,prior,method,problem,Vₘᵢₙ)
    lb = -p_to_Vol(p,prior,method,problem) + Vₘᵢₙ 
  end
  Vol_constr_lb(prior,method,problem,Vₘᵢₙ) = p -> Vol_constr_lb(p,prior,method,problem,Vₘᵢₙ) 

  # Optimise Topology 
  println("Optimising Topology")
  iterations = 3000
  optimisation_results = execute_optimisation(p_to_j(prior,method,problem),Vol_constr_ub(prior,method,problem,Vₘₐₓ),Vol_constr_lb(prior,method,problem,Vₘᵢₙ) ,copy(p0),iterations,optimiser,problem)
  fcalls,gcalls,iters,jf,pf,js,ts = optimisation_results
  Ω0=""
  Ωf=""
  fcalls,gcalls,iters,jf,p0,pf,Ω0,Ωf,js,ts

end # function RunDriver

initialise_p(pP,pN,prior::PriorType{:pixel}) = pP
initialise_p(pP,pN,prior::PriorType{:neural}) = pN