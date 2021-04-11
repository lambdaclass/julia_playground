### A Pluto.jl notebook ###
# v0.11.12

using Markdown
using InteractiveUtils

# ╔═╡ 5d5d520e-f6b1-11ea-12d5-296b9951b41a
begin 
	using Turing
	using Plots
	using Optim
	using Distributions
	using DynamicHMC
end

# ╔═╡ 54530af4-f6b2-11ea-32e2-371c31557914
md"## Supply chain Optimization problem using Turing.jl"

# ╔═╡ a945cf66-f6b1-11ea-2df7-e1ea51d50985
md"The unknown variables will be used to simulate data"

# ╔═╡ e9618838-f698-11ea-1ced-a1b5d442c46b
begin
	supplier_yield = [.9, .5, .8] # unknown
	supplier_yield_sd = [.1, .2, .2] # unknown
	prices = [220.0, 100.0, 120.0] # known
	max_order_size = [100, 80, 100] # known
end

# ╔═╡ b4e368a6-f6b1-11ea-1557-0d6687f3dd3e
md"We have bought 30, 20 and 2 times repectively from each supplier"

# ╔═╡ f5d879be-f698-11ea-396f-3950f33f6d57
n_obs = [30, 20, 2];

# ╔═╡ 0b913188-f699-11ea-0946-bb1f4e89ac54
begin
σ = supplier_yield_sd
μ = supplier_yield
end

# ╔═╡ 61fdc356-f699-11ea-3a6c-53eb4541dabc
k = μ .* (1 .- μ) ./ (σ .^ 2) .- 1

# ╔═╡ 65ffe114-f699-11ea-2ab3-4105e458ee20
begin
	α = μ .* k
	β = (1 .- μ ) .* k
end

# ╔═╡ 6f05fef6-f699-11ea-1459-7dacc1d0d528
begin
	data = []
	for i in  1:length(β)
	push!(data,rand(Beta(α[i],β[i]),n_obs[i]))
	end
end

# ╔═╡ 766874ec-f699-11ea-0ac8-0b5018161c25
begin
	h1 = histogram(data[1], xlim=(0,1), nbins=5, label="prov1", normed=true)
	h2 = histogram(data[2], xlim=(0,1), nbins=5, label="prov2", normed=true)
	h3 = histogram(data[3], xlim=(0,1), label="prov3", normed=true, nbins=3)
	plot(h1, h2, h3, layout=(1,3), size=(650,200))
end

# ╔═╡ d5b42ff2-f6b1-11ea-37ed-fb69823134b3
md"We have the sale price and the holding cost of each engine."

# ╔═╡ 7cac440c-f699-11ea-30ac-2b91dff0d5a9
begin
	sales_price = 500
	holding_cost = 100
end;

# ╔═╡ 856d4d46-f699-11ea-1d46-a91087a7812b
function loss(in_stock, demand, buy_price, sales_price=sales_price, holding_cost=holding_cost)
	margin = sales_price - buy_price
    # Do we have more in stock than demanded?
	reward = zeros(size(in_stock))
	for i in 1:length(in_stock)
		if in_stock[i] > demand
			total_profit = demand * margin
			# everything left over after demand was met goes into holding
			total_holding_cost = (in_stock[i] - demand) * holding_cost
			reward[i] = total_profit - total_holding_cost
		else
			# Can only sell what we have in stock, no storage required
			reward[i] = in_stock[i]* margin
		# Usually we minimize, so invert
		end
	end
    return -reward
end

# ╔═╡ 8ae9807a-f699-11ea-2474-09f25d3943aa
demand_samples = rand(Poisson(100),1000);

# ╔═╡ f35b35dc-f6b1-11ea-1f1b-6d7c569b5af5
md"This is the data we have for past demands"

# ╔═╡ fa9a6f54-f6b1-11ea-3888-55a6fef99d02
begin
histogram(demand_samples, nbins=25, legend=false)
xlabel!("Demand")
ylabel!("occurrences")
title!("Historic demand")
end

# ╔═╡ 08e87fc2-f6b2-11ea-2ce5-b5c80a436e9d
md"Now we define our model, we suppose each yield is sampled from a beta distribution"

# ╔═╡ a221d422-f699-11ea-1b34-a7b62b285ef3
@model function m_yield(x)
	α_yield ~ TruncatedNormal(0, 1, 0, Inf)
	β_yield ~ TruncatedNormal(0, 1, 0, Inf)
	for i in eachindex(x)
		x[i] ~ Beta(α_yield, β_yield)
	end
end

# ╔═╡ 10b91248-f6b2-11ea-38ac-db06e155bfb5
md"Here we compute the posterior distributions for our parameters"

# ╔═╡ e6f39cde-f699-11ea-0ebb-77e42f1147e2
post_yield1 = sample(m_yield(data[1]), DynamicNUTS(), 1000)

# ╔═╡ ca11ade8-f69d-11ea-3400-dbc03a23ecd5
post_yield1_values = collect(get(post_yield1,[:α_yield, :β_yield]));

# ╔═╡ 059d594a-f69a-11ea-0d13-336a05106f48
post_yield2 = sample(m_yield(data[2]), DynamicNUTS(), 1000)

# ╔═╡ 7537692a-f6a0-11ea-0974-adc11695bc00
post_yield2_values = collect(get(post_yield2,[:α_yield, :β_yield]));

# ╔═╡ 384bbf0a-f69d-11ea-300a-4d4c623abd8a
post_yield3 = sample(m_yield(data[3]), DynamicNUTS(), 1000)

# ╔═╡ 79e71e5c-f6a0-11ea-1a47-1b66eab1eb69
post_yield3_values = collect(get(post_yield3,[:α_yield, :β_yield]));

# ╔═╡ 44221ba8-f69d-11ea-2a70-d54d5c785002
begin
	post_β = [collect(post_yield1_values[1]),collect(post_yield2_values[1]), collect(post_yield3_values[1])];
	
	post_α = [collect(post_yield1_values[2]),collect(post_yield2_values[2]), collect(post_yield3_values[2])];
end;

# ╔═╡ fa90e110-f6a9-11ea-08f2-7709b07cb513
begin
	alph = 0.4
	h_α = histogram(post_α[1], normed=true, alpha=alph)
	histogram!(post_α[2], normed=true, alpha=alph)
	histogram!(post_α[3], normed=true, alpha=alph)
	title!("Posterior distributions of α")
	
	h_β = histogram(post_β[1], normed=true, alpha=alph)
	histogram!(post_β[2], normed=true, alpha=alph)
	histogram!(post_β[3], normed=true, alpha=alph)
	title!("Posterior distributions of β")
	
	plot(h_α, h_β, layout=(1,2), legend=false)
end

# ╔═╡ eb060c92-f6a0-11ea-02bb-312df3ef3177
yield1_sample_post = [rand(Beta(α, β),1)[1] for (α, β) in zip(collect(post_yield1_values[2]), collect(post_yield1_values[1]))];

# ╔═╡ db24efd2-f6aa-11ea-0556-2bd46cc30846
begin
histogram(yield1_sample_post, legend=false, nbins=20)
title!("Histogram Yield Prov 1")
xlabel!("yield Prov 1")
ylabel!("counts")
end

# ╔═╡ 6c470b30-f6ab-11ea-306c-2792a5e58acd
yield2_sample_post = [rand(Beta(α, β),1)[1] for (α, β) in zip(collect(post_yield2_values[2]), collect(post_yield2_values[1]))];

# ╔═╡ 80f17c0a-f6ab-11ea-2f26-91e20393d57c
begin
histogram(yield2_sample_post, legend=false, nbins=20)
title!("Histogram Yield Prov 2")
xlabel!("yield Prov 2")
ylabel!("counts")
end

# ╔═╡ 8768d916-f6ab-11ea-21ae-f760431aa6f8
yield3_sample_post = [rand(Beta(α, β),1)[1] for (α, β) in zip(collect(post_yield3_values[2]), collect(post_yield3_values[1]))];

# ╔═╡ 9d68f912-f6ab-11ea-154d-7f1547acfd43
begin
histogram(yield3_sample_post, bins=15, legend=false, nbins=20)
title!("Histogram Yield Prov 3")
xlabel!("yield Prov 3")
ylabel!("counts")
end

# ╔═╡ a7f99904-f6ab-11ea-20d3-8d971479d080
function calc_yield_and_price(orders, supplier_yield_, prices=prices)
    full_yield = sum(supplier_yield_ .* orders)
    price_per_item = sum(orders .* prices) / sum(orders)
    
    return (full_yield, price_per_item)
end 

# ╔═╡ b3c7d020-f6ab-11ea-2ac2-1f078191c3a7
supplier_yield_post_predict = [[i,j,k] for (i,j,k) in zip(yield1_sample_post,yield2_sample_post,yield3_sample_post)];

# ╔═╡ 2689b55a-f6b2-11ea-3d7a-5fe33045ccc9
md"Now we define a objective function (loss) to minimize"

# ╔═╡ bc67b2cc-f6ab-11ea-0bc3-bdf25f6bc929
function objective(orders, supplier_yield=supplier_yield_post_predict,
              demand_samples=demand_samples, max_order_size=max_order_size)
    losses = []
    
    # Negative orders are impossible, indicated by np.inf
    if any(orders .< 0)
        return Inf
	end
    # Ordering more than the supplier can ship is also impossible
    if any(orders .> max_order_size)
        return Inf
	end
    
    # Iterate over post pred samples provided in supplier_yield
    for i in 1:length(supplier_yield)
        full_yield, price_per_item = calc_yield_and_price(
            orders,
            supplier_yield[i]
        )
        # evaluate loss over each sample with one sample from the demand distribution
        loss_i = loss(full_yield,  demand_samples[i], price_per_item)[1] 
        push!(losses,loss_i)
		end 
    return mean(losses)
end

# ╔═╡ c1e03742-f6ab-11ea-115c-9d11de3eb428
begin 
	lower = [0., 0., 0.]
	upper = [100., 80., 100.]
	initial_x = [50., 50., 50.]
end

# ╔═╡ d2412ee8-f6ab-11ea-34eb-276e18742c11
res = optimize(objective, lower, upper, initial_x);

# ╔═╡ d6babfb6-f6ab-11ea-1ffa-bb5382f04644
optim = Optim.minimizer(res)

# ╔═╡ dd4b63c4-f6ab-11ea-10af-c52a136706fa
supplier_yield_mean = mean.(data)

# ╔═╡ e27b7c1e-f6ab-11ea-3031-596fed10849c
function objective_(orders, supplier_yield=supplier_yield_mean,
              demand_samples=100, max_order_size=max_order_size)
    losses = []
    
    # Negative orders are impossible, indicated by np.inf
    if any(orders .< 0)
        return Inf
	end
    # Ordering more than the supplier can ship is also impossible
    if any(orders .> max_order_size)
        return Inf
	end
    
    # Iterate over post pred samples provided in supplier_yield

        full_yield, price_per_item = calc_yield_and_price(
            orders,
            supplier_yield
        )
        # evaluate loss over each sample with one sample from the demand distribution
        loss_i = loss(full_yield, demand_samples, price_per_item)[1] 
        
        push!(losses,loss_i)
    return mean(losses)
end

# ╔═╡ e8c71f7c-f6ab-11ea-0f59-dfa089343423
res_2 = optimize(objective_, lower, upper, initial_x)

# ╔═╡ eed30b76-f6ab-11ea-2c7a-4d542ad6825f
optim_2 = Optim.minimizer(res_2)

# ╔═╡ f07b1cfa-f6ab-11ea-39e6-f7c358a7c160
begin
data_new = []
for (supplier_yield_i, supplier_yield_sd_i, n_obs_i) in zip(supplier_yield, supplier_yield_sd, n_obs)
	σ = supplier_yield_sd_i
	μ = supplier_yield_i
	k = μ .* (1 .- μ) ./ (σ .^ 2) .- 1
	α = μ .* k
	β = (1 .- μ ) .* k
	push!(data_new,rand(Beta(α,β),1000))
end
end


# ╔═╡ f7d759de-f6ab-11ea-3883-1d5e7b92ad33
begin
	new_data = []
	for i in  1:length(β)
	push!(new_data,permutedims(rand(Beta(α[i],β[i]),10000)))
	end
end

# ╔═╡ fd70c04c-f6ab-11ea-3942-2dec9c0ac41e
new_data_rs = [[new_data[1][i],new_data[1][i], new_data[1][i]] for i in 1:10000]

# ╔═╡ 022b98a2-f6ac-11ea-221d-1595ef038cb4
new_demand = rand(Poisson(100),10000);

# ╔═╡ 4fd2e7b6-f6ae-11ea-2261-b93ff86718b4
function objective_func(orders, supplier_yield=supplier_yield_post_predict,
              demand_samples=demand_samples, max_order_size=max_order_size)
    losses = []
    
    # Negative orders are impossible, indicated by np.inf
    if any(orders .< 0)
        return Inf
	end
    # Ordering more than the supplier can ship is also impossible
    if any(orders .> max_order_size)
        return Inf
	end
    
    # Iterate over post pred samples provided in supplier_yield
    for i in 1:length(supplier_yield)
        full_yield, price_per_item = calc_yield_and_price(
            orders,
            supplier_yield[i]
        )
        println(i)
        # evaluate loss over each sample with one sample from the demand distribution
        loss_i = loss(full_yield, demand_samples[i], price_per_item)[1] 
        
        push!(losses,loss_i)
		end 
    return losses
end

# ╔═╡ 0cd1d24a-f6ac-11ea-00b5-dbc0f8020071
begin
histogram(-objective_func(optim, new_data_rs, new_demand) ./ new_demand, normed=true, label="Bayesian model")
histogram!(-objective_func(optim_2, new_data_rs, new_demand) ./ new_demand, normed=true, label="Frequentist")
xlabel!("Profit")
title!("Histogram of profit with ''future'' data")
end

# ╔═╡ 133fc44a-f6ac-11ea-1793-bb084faf0633
median(-objective_func(optim, new_data_rs, new_demand) ./ new_demand)

# ╔═╡ 17646648-f6ac-11ea-121d-410b6a813578
median(-objective_func(optim_2, new_data_rs, new_demand) ./ new_demand)

# ╔═╡ 1d9b8cc6-f6ac-11ea-38fa-316253e1090c
sum(optim)

# ╔═╡ 2208844e-f6ac-11ea-3f41-0d3b39407804
sum(optim_2)

# ╔═╡ 26e4d90e-f6ac-11ea-04d6-85c834d4a789
optim

# ╔═╡ 2b079fc6-f6ac-11ea-1e13-85a3a51f84cd
optim_2

# ╔═╡ Cell order:
# ╟─54530af4-f6b2-11ea-32e2-371c31557914
# ╠═5d5d520e-f6b1-11ea-12d5-296b9951b41a
# ╟─a945cf66-f6b1-11ea-2df7-e1ea51d50985
# ╠═e9618838-f698-11ea-1ced-a1b5d442c46b
# ╟─b4e368a6-f6b1-11ea-1557-0d6687f3dd3e
# ╠═f5d879be-f698-11ea-396f-3950f33f6d57
# ╠═0b913188-f699-11ea-0946-bb1f4e89ac54
# ╠═61fdc356-f699-11ea-3a6c-53eb4541dabc
# ╠═65ffe114-f699-11ea-2ab3-4105e458ee20
# ╠═6f05fef6-f699-11ea-1459-7dacc1d0d528
# ╠═766874ec-f699-11ea-0ac8-0b5018161c25
# ╟─d5b42ff2-f6b1-11ea-37ed-fb69823134b3
# ╠═7cac440c-f699-11ea-30ac-2b91dff0d5a9
# ╠═856d4d46-f699-11ea-1d46-a91087a7812b
# ╠═8ae9807a-f699-11ea-2474-09f25d3943aa
# ╟─f35b35dc-f6b1-11ea-1f1b-6d7c569b5af5
# ╠═fa9a6f54-f6b1-11ea-3888-55a6fef99d02
# ╟─08e87fc2-f6b2-11ea-2ce5-b5c80a436e9d
# ╠═a221d422-f699-11ea-1b34-a7b62b285ef3
# ╟─10b91248-f6b2-11ea-38ac-db06e155bfb5
# ╠═e6f39cde-f699-11ea-0ebb-77e42f1147e2
# ╠═ca11ade8-f69d-11ea-3400-dbc03a23ecd5
# ╠═059d594a-f69a-11ea-0d13-336a05106f48
# ╠═7537692a-f6a0-11ea-0974-adc11695bc00
# ╠═384bbf0a-f69d-11ea-300a-4d4c623abd8a
# ╠═79e71e5c-f6a0-11ea-1a47-1b66eab1eb69
# ╠═44221ba8-f69d-11ea-2a70-d54d5c785002
# ╠═fa90e110-f6a9-11ea-08f2-7709b07cb513
# ╠═eb060c92-f6a0-11ea-02bb-312df3ef3177
# ╠═db24efd2-f6aa-11ea-0556-2bd46cc30846
# ╠═6c470b30-f6ab-11ea-306c-2792a5e58acd
# ╠═80f17c0a-f6ab-11ea-2f26-91e20393d57c
# ╠═8768d916-f6ab-11ea-21ae-f760431aa6f8
# ╠═9d68f912-f6ab-11ea-154d-7f1547acfd43
# ╠═a7f99904-f6ab-11ea-20d3-8d971479d080
# ╠═b3c7d020-f6ab-11ea-2ac2-1f078191c3a7
# ╟─2689b55a-f6b2-11ea-3d7a-5fe33045ccc9
# ╠═bc67b2cc-f6ab-11ea-0bc3-bdf25f6bc929
# ╠═c1e03742-f6ab-11ea-115c-9d11de3eb428
# ╠═d2412ee8-f6ab-11ea-34eb-276e18742c11
# ╠═d6babfb6-f6ab-11ea-1ffa-bb5382f04644
# ╠═dd4b63c4-f6ab-11ea-10af-c52a136706fa
# ╠═e27b7c1e-f6ab-11ea-3031-596fed10849c
# ╠═e8c71f7c-f6ab-11ea-0f59-dfa089343423
# ╠═eed30b76-f6ab-11ea-2c7a-4d542ad6825f
# ╠═f07b1cfa-f6ab-11ea-39e6-f7c358a7c160
# ╠═f7d759de-f6ab-11ea-3883-1d5e7b92ad33
# ╠═fd70c04c-f6ab-11ea-3942-2dec9c0ac41e
# ╠═022b98a2-f6ac-11ea-221d-1595ef038cb4
# ╠═4fd2e7b6-f6ae-11ea-2261-b93ff86718b4
# ╠═0cd1d24a-f6ac-11ea-00b5-dbc0f8020071
# ╠═133fc44a-f6ac-11ea-1793-bb084faf0633
# ╠═17646648-f6ac-11ea-121d-410b6a813578
# ╠═1d9b8cc6-f6ac-11ea-38fa-316253e1090c
# ╠═2208844e-f6ac-11ea-3f41-0d3b39407804
# ╠═26e4d90e-f6ac-11ea-04d6-85c834d4a789
# ╠═2b079fc6-f6ac-11ea-1e13-85a3a51f84cd
