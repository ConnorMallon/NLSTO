ϕ_b_to_ϕc(ϕₛ₃,b,problem) = ϕₛ₃ .+ b
ϕ_b_to_ϕc(ϕₛ₃,b,problem::ProblemType{:heat_simp}) = 1 ./ ( 1 .+ exp.( 1*( .- ϕₛ₃ .- b )))
ϕ_b_to_ϕc(ϕₛ₃,b,problem::ProblemType{:cantilever_simp}) = ϕ_b_to_ϕc(ϕₛ₃,b,ProblemType{Symbol("heat_simp")}())
ϕ_b_to_ϕc(ϕₛ₃,b,problem::ProblemType{:MBB_simp}) = ϕ_b_to_ϕc(ϕₛ₃,b,ProblemType{Symbol("heat_simp")}())

function ϕc_to_Vol(lvl_set_m,Vbg,problem)
	ϕ = collect1d(lvl_set_m) 
	dΩ1 =  get_dΩ1_only(ϕ,Vbg)
	Vol = sum(∫(1)dΩ1)
	# convert volume fraction to actual volume 
	bgmodel = Vbg.fe_basis.trian.model
	cd = get_cartesian_descriptor(bgmodel)
	Lx = cd.sizes[1]*cd.partition[1]
	Ly = cd.sizes[2]*cd.partition[2]
	Vol / ( Lx * Ly )
end

function ϕ_b_to_Vol(lvl_set_m,b,Vbg,problem)
	ϕc = collect1d(lvl_set_m) .+ b
	ϕc_to_Vol(ϕc,Vbg,problem)
end

function ϕc_to_Vol(lvl_set_m,Vbg,problem::ProblemType{:heat_simp})
	ϕ = collect1d(lvl_set_m) # works for simp aswell
	Vol = Statistics.mean(ϕ)
	# convert volume fraction to actual volume 
	bgmodel = Vbg.fe_basis.trian.model
	cd = get_cartesian_descriptor(bgmodel)
	Lx = cd.sizes[1]*cd.partition[1]
	Ly = cd.sizes[2]*cd.partition[2]     
	Vol 
end

function ϕ_b_to_Vol(lvl_set_m,b,Vbg,problem::ProblemType{:heat_simp})
	ϕc =  ϕ_b_to_ϕc(collect1d(lvl_set_m),b,problem)
	ϕc_to_Vol(ϕc,Vbg,problem)
end

ϕ_b_to_Vol(lvl_set_m,b,Vbg,problem::ProblemType{:cantilever_simp}) =  ϕ_b_to_Vol(lvl_set_m,b,Vbg,ProblemType{Symbol("heat_simp")}())
ϕc_to_Vol(lvl_set_m,Vbg,problem::ProblemType{:cantilever_simp}) =  ϕc_to_Vol(lvl_set_m,Vbg,ProblemType{Symbol("heat_simp")}())

ϕ_b_to_Vol(lvl_set_m,b,Vbg,problem::ProblemType{:MBB_simp}) =  ϕ_b_to_Vol(lvl_set_m,b,Vbg,ProblemType{Symbol("heat_simp")}())
ϕc_to_Vol(lvl_set_m,Vbg,problem::ProblemType{:MBB_simp}) =  ϕc_to_Vol(lvl_set_m,Vbg,ProblemType{Symbol("heat_simp")}())

function b(lvl_set,Vbg,Vₘₐₓ,problem)
	f(b)=ϕ_b_to_Vol(lvl_set,b,Vbg,problem) - Vₘₐₓ#actual
    ϵ=1e-5
	bmax = maximum(lvl_set)-ϵ
	bmin = minimum(lvl_set)+ϵ
	b = find_zero(f,(-(bmax),-(bmin)),Bisection(),atol=1e-10)#atol=1e-2)
end 

function b(lvl_set,Vbg,Vₘₐₓ,problem::ProblemType{:heat_simp})
	f(b) = ϕ_b_to_Vol(lvl_set,b,Vbg,problem) - Vₘₐₓ
	ϵ=1e-5
	b = find_zero(f,(-1e2,1e2),Bisection(),atol=1e-10)#atol=1e-2)
end 

b(lvl_set,Vbg,Vₘₐₓ,problem::ProblemType{:cantilever_simp}) = b(lvl_set,Vbg,Vₘₐₓ,ProblemType{Symbol("heat_simp")}())
b(lvl_set,Vbg,Vₘₐₓ,problem::ProblemType{:MBB_simp}) = b(lvl_set,Vbg,Vₘₐₓ,ProblemType{Symbol("heat_simp")}())


function nucleate_holes(n_holes,bgmodel,Vi,Vbg,problem)
  cd = Gridap.Geometry.get_cartesian_descriptor(bgmodel)
  Lx = cd.sizes[1]*cd.partition[1]
  Ly = cd.sizes[2]*cd.partition[2]
  n_holes_y=n_holes
  n_holes_x = Int(round((Lx/Ly)*n_holes; digits = 5))# length rations must be an Int
  geoc = disk(Lx,x0=Point(Lx/2,Ly/2)) # contains whole domain
  geoc = discretize(geoc,bgmodel)
  ls0 = geoc.tree.data[1]
  for i in 1:n_holes_x
    for j in 1:n_holes_y
      xh = Lx/n_holes_x
    yh = Ly/n_holes_y
      dhy = yh*0.3
      R = (yh-dhy)/4
      geoa = disk(R*0.5,x0=Point(xh*i-xh/2,yh*j-yh/2))
      geob = discretize(geoa,bgmodel)
      ls0 = max.(geob.tree.data[1]*-1,ls0)
    end
  end
  Np = ls0
  bp = b(Np,Vbg,Vi,problem)+0.003
  qp = Np.+bp
  p00 = qp
end

function nucleate_holes(n_holes,bgmodel,Vi,Vbg,problem::ProblemType{:heat_simp})
	p0 = 0.95*Vi * ones(num_free_dofs(Vbg))	
end

function cv_to_dof(cv,V)
	fv=zeros(eltype(eltype(cv)),num_free_dofs(V))
	gather_free_values!(fv,V,cv)
end

function field_to_cv(uh::FEFunction)
	get_cell_dof_values(uh)
end

function field_to_cv(cf::CellField)
	cv=cf.cell_field.args[1]
end

function get_geo_params(ϕₕ::FEFunction,Vbg)
  Ωbg = get_triangulation(Vbg)
  bgmodel = get_background_model(Ωbg)
  point_to_coords = collect1d(get_node_coordinates(bgmodel))
  ls_to_point_to_value_unmasked = field_to_cv(ϕₕ)
  p0 = cv_to_dof(ls_to_point_to_value_unmasked,Vbg)
  geo1 = DiscreteGeometry(p0,point_to_coords)
  geo2 = DiscreteGeometry(-1*p0,point_to_coords,name="")
  get_geo_params(ls_to_point_to_value_unmasked,geo1,geo2,bgmodel)
end

function get_geo_params(ϕₕ::CellField,Vbg)
  Ωbg = get_triangulation(Vbg)
  bgmodel = get_background_model(Ωbg)
  point_to_coords = collect1d(get_node_coordinates(bgmodel))
  ls_to_point_to_value_unmasked = field_to_cv(ϕₕ)
  p0 = cv_to_dof(ls_to_point_to_value_unmasked,Vbg)
  geo1 = DiscreteGeometry(p0,point_to_coords)
  geo2 = DiscreteGeometry(-1*p0,point_to_coords,name="")
  get_geo_params(ls_to_point_to_value_unmasked,geo1,geo2,bgmodel)
end

function get_geo_params(ϕ::AbstractVector,Vbg)
	Ωbg = get_triangulation(Vbg)
	bgmodel = get_background_model(Ωbg)
	point_to_coords = collect1d(get_node_coordinates(bgmodel))
	geo1 = DiscreteGeometry(ϕ,point_to_coords,name="")
	geo2 = DiscreteGeometry(-ϕ,point_to_coords,name="")
	ϕh = 	FEFunction(Vbg,ϕ)
	ls_to_point_to_value_unmasked = get_cell_dof_values(ϕh)
	get_geo_params(ls_to_point_to_value_unmasked,geo1,geo2,bgmodel)
end

function get_geo_params(ls_to_point_to_value_unmasked,geo1,geo2,bgmodel)#(ϕ,bg_params)
	cutgeo1= cut(bgmodel,geo1,ls_to_point_to_value_unmasked)
	cutgeo2= cut(bgmodel,geo2,-ls_to_point_to_value_unmasked)
	# Setup interpolation meshes
	Ω1_act = Triangulation(cutgeo1,ACTIVE)#,cutgeo1)
	Ω2_act = Triangulation(cutgeo2,ACTIVE)#,cutgeo2)
	# Setup integration meshes
	Ω1 = Triangulation(cutgeo1,PHYSICAL)#,cutgeo1)
	Ω2 = Triangulation(cutgeo2,PHYSICAL)#,cutgeo2)
	Ω_bg = Triangulation(bgmodel)
  Γ = EmbeddedBoundary(cutgeo1)
	# Setup Lebesgue measures
	order = 1
	degree = 2*order
	dΩ1 = Measure(Ω1,degree)
	dΩ2 = Measure(Ω2,degree)
	dΩ_bg = Measure(Ω_bg,degree)
	dΓ = Measure(Γ,degree)
	geo_params  = Dict{Symbol,Any}(
		(
		 :dΩ1 => dΩ1,
		 :dΩ2 =>dΩ2,
		 :dΓ => dΓ,
		 :dΩ_bg => dΩ_bg,
		) 
	)
	( geo_params, (cutgeo1,cutgeo2,order, Ω1,Ω2,Ω1_act,Ω2_act) )
end

function get_dΩ1_only(ϕ::AbstractVector,Vbg)
	Ωbg = get_triangulation(Vbg)
	bgmodel = get_background_model(Ωbg)
	point_to_coords = collect1d(get_node_coordinates(bgmodel))
	geo1 = DiscreteGeometry(ϕ,point_to_coords,name="")
	ϕh = 	FEFunction(Vbg,ϕ)
	ls_to_point_to_value_unmasked = get_cell_dof_values(ϕh)	
	cutgeo1= cut(bgmodel,geo1,ls_to_point_to_value_unmasked)
	Ω1_act = Triangulation(cutgeo1,ACTIVE)#,cutgeo1)
	Ω1 = Triangulation(cutgeo1,PHYSICAL)#,cutgeo1)
	order = 1
	degree = 2*order
	dΩ1 = Measure(Ω1,degree)
end
