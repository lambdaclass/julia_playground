### A Pluto.jl notebook ###
# v0.11.12

using Markdown
using InteractiveUtils

# ╔═╡ 393d15fc-f2a2-11ea-2741-d5ec0d980f8b
begin 
	using Soss
	using Plots
	using Optim
end

# ╔═╡ e1831534-f380-11ea-17c2-27bba65b030f
md"## Supply Chain Optimization problem using Soss.jl"

# ╔═╡ 4673ac54-f2a2-11ea-2e2a-6dab49e6c60c
begin
	supplier_yield = [.9, .5, .8] # unknown
	supplier_yield_sd = [.1, .2, .2] # unknown
	prices = [220.0, 100.0, 120.0] # known
	max_order_size = [100, 80, 100] # known
end

# ╔═╡ 8a93e464-f381-11ea-13ec-ed405a6d90f2
md"The unknown variables will be used to simulate data"

# ╔═╡ 1e39b9f0-f382-11ea-3341-93d7f592a259
md"We have bought 30, 20 and 2 times repectively from each supplier"

# ╔═╡ 749ad2fe-f2a2-11ea-08ea-83e9aeee8249
n_obs = [30, 20, 2];

# ╔═╡ 52511d6e-f382-11ea-1879-550600c70795
md"We suppose the data follow a beta distribution, so we simulate the yield we have in our records"

# ╔═╡ 9e7bcc6a-f2a2-11ea-315a-29e7318fcee6
begin
σ = supplier_yield_sd
μ = supplier_yield
end

# ╔═╡ a56e952c-f2a2-11ea-29fc-03edb66ff20a
k = μ .* (1 .- μ) ./ (σ .^ 2) .- 1

# ╔═╡ ac5a59f0-f2a2-11ea-39a0-9f7955066c0a
begin
	α = μ .* k
	β = (1 .- μ ) .* k
end

# ╔═╡ b468f926-f2a2-11ea-015c-afcadd9c7641
begin
	data = []
	for i in  1:length(β)
	push!(data,rand(Beta(α[i],β[i]),n_obs[i]))
	end
end

# ╔═╡ 835038d0-f3a2-11ea-17d8-6d41ddbbf3c2
data

# ╔═╡ c938223c-f2a2-11ea-217a-39ede7cf7485

begin
	h1 = histogram(data[1], xlim=(0,1), nbins=5, label="prov1", normed=true)
	h2 = histogram(data[2], xlim=(0,1), nbins=5, label="prov2", normed=true)
	h3 = histogram(data[3], xlim=(0,1), label="prov3", normed=true, nbins=3)
	plot(h1, h2, h3, layout=(1,3), size=(650,200))
end

# ╔═╡ efb7c558-f43b-11ea-0acb-21c38311d03b
max_order_size

# ╔═╡ feebde6c-f43b-11ea-20eb-e5259acc8b91
prices

# ╔═╡ d6c81b82-f383-11ea-0ed1-e3624923523d
md"We have the sale price and the holding cost of each engine."

# ╔═╡ 6ef4eede-f2a4-11ea-34fd-893dd9478d23
begin
	sales_price = 500
	holding_cost = 100
end;

# ╔═╡ 7ad4631a-f2a4-11ea-17dc-cd8eb1abfd3e
in_stock = collect(1:100)

# ╔═╡ 8d47ebcc-f2a2-11ea-341f-7b53ddd18589
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

# ╔═╡ ba95f27c-f2a2-11ea-20b6-bb3a6e5a6b51
begin
scatter(in_stock, -loss(in_stock, 50, 50), legend=false)
vline!([50],[50], color="black", lw=3, style=:dash)
xlabel!("In stock")
ylabel!("Profit")
title!("Profit vs In stock (demand = 50 units)")
end

# ╔═╡ abdd684e-f2a4-11ea-1d31-d3cc96b75c0e
demand_samples = rand(Poisson(100),1000);

# ╔═╡ d0b53166-f384-11ea-3903-719d5ff4fb77
md"This is the data we have for past demands"

# ╔═╡ ca8bfbc4-f2d2-11ea-0de1-5df4918768cf
begin
histogram(demand_samples, nbins=25, legend=false)
xlabel!("Demand")
ylabel!("occurrences")
title!("Historic demand")
end

# ╔═╡ 126a9226-f2a5-11ea-10d4-958a9fbef365
profit = [-loss(100, demand_samples[i],10)[1] for i in 1:length(demand_samples)];

# ╔═╡ ff8c9fe6-f2a4-11ea-2fc4-d1e79e9614d1
begin
scatter(demand_samples, profit, xlim=(50,150), legend=false)
title!("Profit vs demand (in_stock=100)")
xlabel!("Demand")
ylabel!("Profit")
end

# ╔═╡ fa005a34-f387-11ea-1ee6-73d0ebfaa696
md"Now we define our model, we suppose each yield is sampled from a beta distribution"

# ╔═╡ 38af90bc-f2a5-11ea-14d6-695be5b45978
m_yield = @model begin
	α_yield ~ HalfNormal()
	β_yield ~ HalfNormal()
	yield ~ For(eachindex(yield)) do j
		Beta(α_yield, β_yield)
	end
end

# ╔═╡ 5b08b504-f39e-11ea-390c-91fc7ce54a9c
md"Here we compute the posterior distributions for our parameters"

# ╔═╡ efb78ae4-f2a5-11ea-1405-43564470c43d
post_yield1 = dynamicHMC(m_yield(), (yield=data[1],));

# ╔═╡ 11a05ac8-f2a6-11ea-0ecc-1de43adf87c4
post_yield2 = dynamicHMC(m_yield(), (yield=data[2],));

# ╔═╡ 18f9b79c-f2a6-11ea-3d07-336748d8509e
post_yield3 = dynamicHMC(m_yield(), (yield=data[3],));

# ╔═╡ e51e3392-f3a4-11ea-1d2b-81127dd5f5ce
begin
	post_α = hcat([[post_yield1.α_yield, post_yield2.α_yield, post_yield3.α_yield] for (post_yield1, post_yield2, post_yield3) in zip(post_yield1, post_yield2, post_yield3)]...)';
	post_β = hcat([[post_yield1.β_yield, post_yield2.β_yield, post_yield3.β_yield] for (post_yield1, post_yield2, post_yield3) in zip(post_yield1, post_yield2, post_yield3)]...)';
end;

# ╔═╡ 2424c526-f3a5-11ea-05d7-13d4e3acb877
begin
	alph = 0.4
	h_α = histogram(post_α[:,1], normed=true, alpha=alph)
	histogram!(post_α[:,2], normed=true, alpha=alph)
	histogram!(post_α[:,3], normed=true, alpha=alph)
	title!("Posterior distributions of α")

	h_β = histogram(post_β[:,1], normed=true, alpha=alph)
	histogram!(post_β[:,2], normed=true, alpha=alph)
	histogram!(post_β[:,3], normed=true, alpha=alph)
	title!("Posterior distributions of β")
	plot(h_α, h_β, layout=(1,2), legend=false)
end

# ╔═╡ 284fa7e4-f2a6-11ea-3331-e3a5c8de4148
yield1_sample_post = [rand(Beta(vec.α_yield, vec.β_yield),1)[1] for vec in post_yield1] 

# ╔═╡ 57f7ddd6-f39e-11ea-243c-09db4bdf5c17


# ╔═╡ 61ba12f8-f2a6-11ea-103b-950471b0ecc8
begin
histogram(yield1_sample_post, legend=false, nbins=20)
title!("Histogram Yield Prov 1")
xlabel!("yield Prov 1")
ylabel!("counts")
end

# ╔═╡ 6e5f8132-f2a6-11ea-30f5-ad77eb6ce0bb
yield2_sample_post = [rand(Beta(vec.α_yield, vec.β_yield),1)[1] for vec in post_yield2] 

# ╔═╡ 7aa90cb8-f2a6-11ea-23fa-75359600afd9
begin
histogram(yield2_sample_post, legend=false, nbins=20)
title!("Histogram Yield Prov 2")
xlabel!("yield Prov 2")
ylabel!("counts")
end

# ╔═╡ 7fbcce4c-f2a6-11ea-01bb-b3bb1aeef63e
yield3_sample_post = [rand(Beta(vec.α_yield, vec.β_yield),1)[1] for vec in post_yield3]

# ╔═╡ 86142808-f2a6-11ea-3cbb-63e2ed0649f4
begin
histogram(yield3_sample_post, bins=15, legend=false, nbins=20)
title!("Histogram Yield Prov 3")
xlabel!("yield Prov 3")
ylabel!("counts")
end

# ╔═╡ 34ee16c0-f39e-11ea-14bb-3d5b8c69ebeb
md"We define a helper function to calculate engines and price for each order and yield"

# ╔═╡ 2f57ea80-f2a7-11ea-1583-dd24fdcec992
function calc_yield_and_price(orders, supplier_yield_, prices=prices)
    full_yield = sum(supplier_yield_ .* orders)
    price_per_item = sum(orders .* prices) / sum(orders)
    
    return (full_yield, price_per_item)
end 

# ╔═╡ f1071d8e-f39e-11ea-2b2e-2fcbe3980817
md"Here we sample from our posterior distribution." 

# ╔═╡ 8a4221b2-f2bf-11ea-31b4-4bd2ed7c5bed
supplier_yield_post_predict = [[i,j,k] for (i,j,k) in zip(yield1_sample_post,yield2_sample_post,yield3_sample_post)]

# ╔═╡ f10bcd38-f39f-11ea-319f-5f14e68b90fe
md"Now we define a objective function (loss) to minimize"

# ╔═╡ dc6737d2-f2aa-11ea-0226-650d491228d1
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
    for i in 1:length(supplier_yield_post_predict)
        full_yield, price_per_item = calc_yield_and_price(
            orders,
            supplier_yield_post_predict[i]
        )
        # evaluate loss over each sample with one sample from the demand distribution
        loss_i = loss(full_yield,  demand_samples[i], price_per_item)[1] 
        push!(losses,loss_i)
		end 
    return mean(losses)
end

# ╔═╡ 507a1d66-f2af-11ea-03fd-95d1ff4258ec
lower = [0., 0., 0.]

# ╔═╡ 2eecdb5a-f2b1-11ea-2d81-9fadfe230c20
upper = [100., 80., 100.]

# ╔═╡ 38f47d60-f2b1-11ea-13c5-c9fb5f0485ee
initial_x = [50., 50., 50.]

# ╔═╡ 112c9790-f2b1-11ea-1eea-830e235afbb1
res = optimize(objective, lower, upper, initial_x);

# ╔═╡ 50b1539c-f2b1-11ea-2bd2-636a14a93d77
optim = Optim.minimizer(res)

# ╔═╡ 1cedff70-f2b8-11ea-1237-a90505df8ef1
supplier_yield_mean = mean.(data)

# ╔═╡ 8c601dc8-f38e-11ea-1803-7d17a1afa373
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
            supplier_yield_mean
        )
        # evaluate loss over each sample with one sample from the demand distribution
        loss_i = loss(full_yield, demand_samples, price_per_item)[1] 
        
        push!(losses,loss_i)
    return mean(losses)
end

# ╔═╡ 7a4e04ae-f2b9-11ea-35fd-abc3d07e102e
res_2 = optimize(objective_, lower, upper, initial_x)

# ╔═╡ dab7a364-f2b8-11ea-14b2-459e039cd9a8
optim_2 = Optim.minimizer(res_2)

# ╔═╡ c190c436-f2bf-11ea-0244-b1ef12ecef6c
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


# ╔═╡ 4cf67c30-f2c7-11ea-0b31-6582661e2c4a
begin
	new_data = []
	for i in  1:length(β)
	push!(new_data,permutedims(rand(Beta(α[i],β[i]),10000)))
	end
end

# ╔═╡ 7ec0a4c8-f2ce-11ea-010b-9f19af6bb5cc
demand_samples

# ╔═╡ 97a90278-f2ce-11ea-1bc0-c1e7995bbf5b
new_data_rs = [[new_data[1][i],new_data[1][i], new_data[1][i]] for i in 1:10000]

# ╔═╡ d6229dd6-f392-11ea-2c49-2b375fff1d98
new_demand = rand(Poisson(100),10000);

# ╔═╡ ccbd0a08-f2cf-11ea-2218-8bf1eae2a93b
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

# ╔═╡ d1041836-f2cf-11ea-0b1c-551b547693bd
begin
histogram(-objective_func(optim, new_data_rs,new_demand) ./ new_demand, normed=true, label="Bayesian model")
histogram!(-objective_func(optim_2, new_data_rs,new_demand) ./ new_demand, normed=true, label="Frequentist")
xlabel!("Profit")
title!("Histogram of profit with ''future'' data")
end

# ╔═╡ ce1a456a-f39b-11ea-0d6c-5f311e09eadc
md"Median profit, bayesian model:"

# ╔═╡ 84989bca-f2d4-11ea-16cd-ebd4bb0f8098
median(-objective_func(optim, new_data_rs, new_demand) ./ new_demand)

# ╔═╡ e9d6cf76-f39b-11ea-3902-8f0643ed99ca
md"Median profit, frequentist model:"

# ╔═╡ 8c24dcfa-f2d4-11ea-1885-d38117360220
median(-objective_func(optim_2, new_data_rs, new_demand) ./ new_demand)

# ╔═╡ 403607b0-f39c-11ea-0ca8-2f2e54557bcd
md"Engines bought with bayesian model"

# ╔═╡ 01085b32-f2d5-11ea-0baf-c17d25006169
sum(optim)

# ╔═╡ b5de9806-f39c-11ea-3f2f-bdfaa65f5de9
md"Engines bought with frequentist "

# ╔═╡ 11602ffa-f2d5-11ea-1a3c-cb5d1ec11662
sum(optim_2)

# ╔═╡ c311daa6-f39c-11ea-2f4b-e38b426953dc
optim

# ╔═╡ dafd7530-f39c-11ea-1c68-efc6b57fc81e
optim_2

# ╔═╡ Cell order:
# ╟─e1831534-f380-11ea-17c2-27bba65b030f
# ╠═393d15fc-f2a2-11ea-2741-d5ec0d980f8b
# ╟─4673ac54-f2a2-11ea-2e2a-6dab49e6c60c
# ╟─8a93e464-f381-11ea-13ec-ed405a6d90f2
# ╟─1e39b9f0-f382-11ea-3341-93d7f592a259
# ╠═749ad2fe-f2a2-11ea-08ea-83e9aeee8249
# ╟─52511d6e-f382-11ea-1879-550600c70795
# ╠═9e7bcc6a-f2a2-11ea-315a-29e7318fcee6
# ╠═a56e952c-f2a2-11ea-29fc-03edb66ff20a
# ╠═ac5a59f0-f2a2-11ea-39a0-9f7955066c0a
# ╠═b468f926-f2a2-11ea-015c-afcadd9c7641
# ╠═835038d0-f3a2-11ea-17d8-6d41ddbbf3c2
# ╠═c938223c-f2a2-11ea-217a-39ede7cf7485
# ╠═efb7c558-f43b-11ea-0acb-21c38311d03b
# ╠═feebde6c-f43b-11ea-20eb-e5259acc8b91
# ╟─d6c81b82-f383-11ea-0ed1-e3624923523d
# ╠═6ef4eede-f2a4-11ea-34fd-893dd9478d23
# ╟─7ad4631a-f2a4-11ea-17dc-cd8eb1abfd3e
# ╠═8d47ebcc-f2a2-11ea-341f-7b53ddd18589
# ╠═ba95f27c-f2a2-11ea-20b6-bb3a6e5a6b51
# ╠═abdd684e-f2a4-11ea-1d31-d3cc96b75c0e
# ╟─d0b53166-f384-11ea-3903-719d5ff4fb77
# ╠═ca8bfbc4-f2d2-11ea-0de1-5df4918768cf
# ╠═126a9226-f2a5-11ea-10d4-958a9fbef365
# ╟─ff8c9fe6-f2a4-11ea-2fc4-d1e79e9614d1
# ╟─fa005a34-f387-11ea-1ee6-73d0ebfaa696
# ╠═38af90bc-f2a5-11ea-14d6-695be5b45978
# ╟─5b08b504-f39e-11ea-390c-91fc7ce54a9c
# ╠═efb78ae4-f2a5-11ea-1405-43564470c43d
# ╠═11a05ac8-f2a6-11ea-0ecc-1de43adf87c4
# ╠═18f9b79c-f2a6-11ea-3d07-336748d8509e
# ╠═e51e3392-f3a4-11ea-1d2b-81127dd5f5ce
# ╠═2424c526-f3a5-11ea-05d7-13d4e3acb877
# ╠═284fa7e4-f2a6-11ea-3331-e3a5c8de4148
# ╠═57f7ddd6-f39e-11ea-243c-09db4bdf5c17
# ╠═61ba12f8-f2a6-11ea-103b-950471b0ecc8
# ╠═6e5f8132-f2a6-11ea-30f5-ad77eb6ce0bb
# ╠═7aa90cb8-f2a6-11ea-23fa-75359600afd9
# ╠═7fbcce4c-f2a6-11ea-01bb-b3bb1aeef63e
# ╠═86142808-f2a6-11ea-3cbb-63e2ed0649f4
# ╟─34ee16c0-f39e-11ea-14bb-3d5b8c69ebeb
# ╠═2f57ea80-f2a7-11ea-1583-dd24fdcec992
# ╟─f1071d8e-f39e-11ea-2b2e-2fcbe3980817
# ╠═8a4221b2-f2bf-11ea-31b4-4bd2ed7c5bed
# ╟─f10bcd38-f39f-11ea-319f-5f14e68b90fe
# ╠═dc6737d2-f2aa-11ea-0226-650d491228d1
# ╠═507a1d66-f2af-11ea-03fd-95d1ff4258ec
# ╠═2eecdb5a-f2b1-11ea-2d81-9fadfe230c20
# ╠═38f47d60-f2b1-11ea-13c5-c9fb5f0485ee
# ╠═112c9790-f2b1-11ea-1eea-830e235afbb1
# ╠═50b1539c-f2b1-11ea-2bd2-636a14a93d77
# ╠═1cedff70-f2b8-11ea-1237-a90505df8ef1
# ╠═8c601dc8-f38e-11ea-1803-7d17a1afa373
# ╠═7a4e04ae-f2b9-11ea-35fd-abc3d07e102e
# ╠═dab7a364-f2b8-11ea-14b2-459e039cd9a8
# ╠═c190c436-f2bf-11ea-0244-b1ef12ecef6c
# ╠═4cf67c30-f2c7-11ea-0b31-6582661e2c4a
# ╠═7ec0a4c8-f2ce-11ea-010b-9f19af6bb5cc
# ╠═97a90278-f2ce-11ea-1bc0-c1e7995bbf5b
# ╠═d6229dd6-f392-11ea-2c49-2b375fff1d98
# ╠═ccbd0a08-f2cf-11ea-2218-8bf1eae2a93b
# ╠═d1041836-f2cf-11ea-0b1c-551b547693bd
# ╟─ce1a456a-f39b-11ea-0d6c-5f311e09eadc
# ╠═84989bca-f2d4-11ea-16cd-ebd4bb0f8098
# ╟─e9d6cf76-f39b-11ea-3902-8f0643ed99ca
# ╠═8c24dcfa-f2d4-11ea-1885-d38117360220
# ╟─403607b0-f39c-11ea-0ca8-2f2e54557bcd
# ╠═01085b32-f2d5-11ea-0baf-c17d25006169
# ╟─b5de9806-f39c-11ea-3f2f-bdfaa65f5de9
# ╠═11602ffa-f2d5-11ea-1a3c-cb5d1ec11662
# ╠═c311daa6-f39c-11ea-2f4b-e38b426953dc
# ╠═dafd7530-f39c-11ea-1c68-efc6b57fc81e
