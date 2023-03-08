using Base: entry_point_and_project_file_inside

function NetworkInit(bgmodel,nf,nd,pP,image_size)
	function normalisation(x)
		mean = Statistics.mean([1])
		variance = Statistics.var([1,2])
		epsilon=1e-6
		x.-=mean
		x*=(1/sqrt(variance+epsilon))
		x
	end
	cd = Gridap.Geometry.get_cartesian_descriptor(bgmodel)
	cellsx,cellsy = cd.partition
	n_cells_x = cellsx+1
	n_cells_y = cellsy+1
	filters=(nf*8,nf*4,nf*2,nf,1)
	k=5
	kernel_size = (k,k)
	image_size_value = get_images_size(n_cells_y,image_size)
	h = image_size_value
	w = image_size_value * Int(n_cells_x/n_cells_y)
	r_0 = 1 
	r_1 = r_2 = r_3 = 2
	r_4x = Int( n_cells_x / ( w * prod([r_0,r_1,r_2,r_3]) ) )  # resize to fit mesh 
	r_4y = Int( n_cells_y / ( h * prod([r_0,r_1,r_2,r_3]) ) )  # resize to fit mesh 
	resizes_x = ( r_0 ,r_1, r_2, r_3, r_4x)  
	resizes_y = ( r_0 ,r_1, r_2, r_3, r_4y)  
	@assert prod([r_0,r_1,r_2,r_3,r_4x])*w == n_cells_x
	@assert prod([r_0,r_1,r_2,r_3,r_4y])*h == n_cells_y
	trainable_init = nd*2*2 
	dense_channels = nd
	init=Flux.glorot_uniform 

	m =  Chain(
		x->[1], # ignore input
		
		Dense(1, trainable_init; bias=false), # trainaible initialisation
		Dense(trainable_init, dense_channels*w*h),    

		x -> reshape(x,(w,h,dense_channels)),
		x->tanh.(x),
		Flux.unsqueeze(4),
		x->upsample_bilinear(x, (resizes_x[1],resizes_y[1])),
		x->Flux.normalise(x;dims=ndims(x)-1), 
		Conv(kernel_size, dense_channels => filters[1], pad=SamePad(), ),
		
		x->tanh.(x),
		x->upsample_bilinear(x, (resizes_x[2],resizes_y[2])),
		x->Flux.normalise(x;dims=ndims(x)-1), 
		Conv( kernel_size, filters[1] => filters[2], pad=SamePad(), ),
		
		x->tanh.(x),
		x->upsample_bilinear(x, (resizes_x[3],resizes_y[3])),
		x->Flux.normalise(x;dims=ndims(x)-1), 
		Conv( kernel_size, filters[2] => filters[3], pad=SamePad(), ), 

		x->tanh.(x),
		x->upsample_bilinear(x, (resizes_x[4],resizes_y[4])),
		x->Flux.normalise(x;dims=ndims(x)-1),
		Conv( kernel_size, filters[3] => filters[4], pad=SamePad(), ),

		x->tanh.(x),
		x->upsample_bilinear(x, (resizes_x[5],resizes_y[5])),
		x->Flux.normalise(x;dims=ndims(x)-1),
		Conv( kernel_size, filters[4] => filters[5], pad=SamePad(), ),

		x->Flux.flatten(x),
		)

	p00,re =  Flux.destructure(m)
	N(p) = (re(p)(""))
	(p00,N)
end

get_images_size(n_cells,image_size::ImageSizeType{:scale_with_mesh}) = Int((n_cells)/8)
get_images_size(n_cells,image_size::ImageSizeType{:dont_scale_with_mesh}) = 6 