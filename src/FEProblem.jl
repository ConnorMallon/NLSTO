# =============
# Heat Problem
# =============

function build_bgmodel(problem::ProblemType{:heat},n_cells)
	Lx=1
	Ly=1
	n_cellsx=n_cells
	n_cellsy=n_cells
	domain = (0,Lx,0,Ly); 
	cells=(n_cellsx,n_cellsy)
	bgmodel = CartesianDiscreteModel(domain,cells)
end

function get_FE_Spaces(problem::ProblemType{:heat},bgmodel)
	order_u = 1
	degree_u = 2 * order_u
	reffe_u = ReferenceFE(lagrangian,Float64,order_u)
	ud(x) = 0.0
	Ωbg = Triangulation(bgmodel)
	dΩ_bg = Measure(Ωbg,degree_u)
	Vbgϕ = TestFESpace(bgmodel,FiniteElements(PhysicalDomain(),bgmodel,lagrangian,Float64,order_u))
	Ubgϕ = TrialFESpace(Vbgϕ)
	Vbgu = TestFESpace(bgmodel,FiniteElements(PhysicalDomain(),bgmodel,lagrangian,Float64,order_u),dirichlet_tags=[1,3,4,6,7])
	Ubgu = TrialFESpace(Vbgu,ud)
	(Vbgϕ,Ubgϕ,Vbgu,Ubgu)
end

build_bgmodel(problem::ProblemType{:heat_simp},n_cells) = build_bgmodel(ProblemType{Symbol("heat")}(),n_cells)

get_FE_Spaces(problem::ProblemType{:heat_simp},bgmodel) = get_FE_Spaces(ProblemType{Symbol("heat")}(),bgmodel)

function get_FE_forms(problem::ProblemType{:heat},Q,α₂)

	α1 = 1
	α2 = α₂ #1e-2
	f1(x) = 1e-2
	f2(x) = 1e-2

	function a( u,v,geo_params)
		dΩ1 = geo_params[:dΩ1]
		dΩ2 = geo_params[:dΩ2]	
		∫( α1 * ∇(v)⋅∇(u) ) * dΩ1 + ∫( α2 * ∇(v)⋅∇(u) ) * dΩ2  
	end

	function l( v, geo_params )
		dΩ1 = geo_params[:dΩ1]
		dΩ2 = geo_params[:dΩ2]	
		∫( v*f1 )dΩ1 + 
		∫( v*f2 )dΩ2 
	end	

	a(ϕ) = (u,v) -> a(u,v,get_geo_params(ϕ,Q)[1])
	l(ϕ) = v -> l(v,get_geo_params(ϕ,Q)[1])

	function res(u,v,ϕ,Q) 
		fϕ,_ = get_geo_params(ϕ,Q)
		a(u,v,fϕ) - l(v,fϕ) 
	end

	a,l,res

end

# #=
function get_FE_forms(problem::ProblemType{:heat_simp},Q,α₂)

	α1 = 1
	α2 = α₂ #1e-2
	f1(x) = 1e-2
	#f2(x) = 1e-2

	Ω_bg = get_triangulation(Q)
	order = 1
	degree = 2*order
	dΩ_bg = Measure(Ω_bg,degree)

	power(ϕ) = ϕ^3

	function a( u,v,ϕ)
		ϕₕ = ϕ_to_ϕₕ(ϕ)
		∫( α₂ +  (1-α₂) * ( power ∘ (ϕₕ) ) * ∇(v)⋅∇(u) )dΩ_bg
	end

	function l( v, ϕ )	
		∫( v*f1 )dΩ_bg
	end	

	function ϕ_to_ϕₕ(ϕ::AbstractArray)
		ϕ = FEFunction(Q,ϕ)
	end

	function ϕ_to_ϕₕ(ϕ::CellField)
		ϕ
	end

	function ϕ_to_ϕₕ(ϕ::FEFunction)
		ϕ
	end

	res(u,v,ϕ,Q) = a(u,v,ϕ) - l(v,ϕ) 

	a(ϕ) = (u,v) -> a(u,v,ϕ)
	l(ϕ) = v -> l(v,ϕ)

	a,l,res
end

function get_objective_function(problem::ProblemType{:heat},P,α₂) 
	function j(uₕ,ϕₕ)
		geo_params,_=get_geo_params(ϕₕ,P) 
		dΩ1 = geo_params[:dΩ1]
		dΩ2 = geo_params[:dΩ2]
		∫(uₕ)dΩ1 + ∫(uₕ)dΩ2
	end
	j
end

function get_objective_function(problem::ProblemType{:heat_simp},Q,α₂) 
	Ω_bg = get_triangulation(Q)
	order = 1
	degree = 2*order
	dΩ_bg = Measure(Ω_bg,degree)
	function j(uₕ,ϕₕ)
		∫(uₕ)dΩ_bg
	end
	j
end

# ============
# LSR problem
# ============
function get_LSR_residual_and_jacobian(Q)
	Ωbg = get_triangulation(Q)
	bgmodel = get_background_model(Ωbg)
	cd = Gridap.Geometry.get_cartesian_descriptor(bgmodel)
	h = maximum(cd.sizes)
	γd = 20
	γg = 0.1
	dt = h * 5
	ν  = 1 
	Δt = dt
	αₜ = 1.0
	s(ϕ₀) = sign∘ϕ₀
	ϵ = 1e-20
	d1(∇u) = 1 / ( ϵ + norm(∇u) ) 
	a(u) =  s(u) * ∇(u) * ( d1 ∘ ( ∇(u) ) )
	cₐ = 3
	νₐ(w) = cₐ*h * ( sqrt∘( w⋅w ) ) 
	function a_ν(w,u,v,geo_params) 
		dΓ = geo_params[:dΓ]
		dΩ_bg = geo_params[:dΩ_bg]
		∫( (γd/h)*v*u ) * dΓ +
		∫( νₐ(a(w))*∇(u)⋅∇(v) +  v*a(w)⋅∇(u)  )dΩ_bg #+
	end
	function b_ν(w,v,geo_params) 
		dΩ_bg = geo_params[:dΩ_bg]
		∫( s(w)*v )dΩ_bg 
	end
	res(u,v,ϕ,Q )    = a_ν(u,u,v,get_geo_params(ϕ,Q)[1])  - b_ν(u,v,get_geo_params(ϕ,Q)[1]) 
	jac(u,du,v,ϕ,Q)  = a_ν(u,du,v,get_geo_params(ϕ,Q)[1])  
	res(fϕ) = (u,   v) -> res(u,   v,fϕ,Q)
	jac(fϕ) = (u,du,v) -> jac(u,du,v,fϕ,Q)
	res,jac	
end
	

	
