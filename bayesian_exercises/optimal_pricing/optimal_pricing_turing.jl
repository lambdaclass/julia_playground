### A Pluto.jl notebook ###
# v0.11.12

using Markdown
using InteractiveUtils

# ╔═╡ de1833c6-f695-11ea-292a-0b4edf4a288e
begin
	using Turing
	using Plots
end

# ╔═╡ e7c2debc-f695-11ea-341a-6fd18cac6451
md"### Using Turing.jl to find the optimal price
We are goint to model the relationship between quantity and price using the equation:
Q = a$$P^c$$.
The priors of **a** and **c** are modeled as a *cauchy* distribution and Q is modeled as a *poisson* distribution."

# ╔═╡ 220f1bc6-f696-11ea-0898-4de9426d5ca3
md"In order to linearize the model, we apply Log function in both sides of the equation:

Log(E(Q|P)) = Log(a) + c * Log(P)
"

# ╔═╡ 2b85f76a-f696-11ea-28fd-fb83f7fc1ede
@model function quantity(qval,p0)
	loga ~ Cauchy()
	c ~ Cauchy()
	logμ0 = loga .+ c*log.(p0)
	μ0 = exp.(logμ0)
	for i in eachindex(µ0)	
		qval[i] ~ Poisson(μ0[i])
	end
end

# ╔═╡ 1b8532d0-f697-11ea-0206-ef5f6891880c
md"Now we plot the data:"

# ╔═╡ 259b4e58-f697-11ea-3412-3debf3847330
begin
	Quantity = [53, 45, 25, 26, 25]
	Price = [30, 35, 40, 45, 50]
end;

# ╔═╡ 470eef36-f697-11ea-2147-d324510b2f72
begin
scatter(Price, Quantity, markersize=6, color="orange", ylim=(0,60), legend=false)
xlabel!("Price")
ylabel!("Quantity")
end

# ╔═╡ b437c448-f697-11ea-2ac4-dd0af4dfbf03
md"Now, we estimate the posterior and sample from it."

# ╔═╡ 058cd946-f698-11ea-049f-ab7144c6dc4e
model = quantity(Quantity,Price)

# ╔═╡ b68a1a5c-f697-11ea-276b-1391a6f5a809
post = sample(model, NUTS(),1000);

# ╔═╡ bdc1d068-f69b-11ea-2fb6-7b249b7044db
post_c = collect(get(post, :c));

# ╔═╡ f67e55c0-f69b-11ea-2d98-b3e0835b7846
begin
histogram(post_c, normed=true, bins=15, label = false)
title!("Posterior distribution of c")
xlabel!("c")
end

# ╔═╡ 4740786c-f69c-11ea-12a6-7be5f68e1ea7
post_loga = collect(get(post, :loga));

# ╔═╡ 531b7920-f69c-11ea-15a5-419cb5240585
begin
histogram(post_loga, normed=true, legend=false, bins=15)
xlabel!("log(a)")
title!("Posterior distribution of log(a)")
end

# ╔═╡ 02304f00-f6a0-11ea-15d0-fbfa262c3890
md"If we plot c vs log(a), we notice that the two variables have multicollinearity."

# ╔═╡ 5972fc04-f6a0-11ea-2113-fd9317a357a4
begin
scatter(post_c, post_loga, legend=false)
title!("log(a) vs c")
xlabel!("c")
ylabel!("log(a)")
end

# ╔═╡ a7dbb796-f6a0-11ea-25ad-091eec0bdad5
md" Now we reparametrize the model to fix the multicollinearity problem, subtracting the mean of *log(p)*. We rename our model variables as β and α."

# ╔═╡ b006c1ea-f6a0-11ea-28d0-f95e21f89e2b
@model function quantity_(qval,p0)
	α ~ Cauchy()
	β ~ Cauchy()
	logμ0_ = α .+ β*(log.(p0) .- mean(log.(p0)))
	μ0_ = exp.(logμ0_)
	for i in eachindex(µ0_)	
		qval[i] ~ Poisson(μ0_[i])
	end
end

# ╔═╡ 26e982d4-f6a1-11ea-1af6-a7753c3eb35d
model_2 = quantity_(Quantity, Price)

# ╔═╡ 3f0556c2-f6a1-11ea-0672-cbf35b8721da
post_2 = sample(model_2, NUTS(),1000);

# ╔═╡ 65a17042-f6a1-11ea-3bb9-f19b5412b967
post_α = collect(get(post_2, :α));

# ╔═╡ 7c9d5692-f6a1-11ea-25ad-251c2669c6ff
post_β = collect(get(post_2, :β));

# ╔═╡ 91cf3152-f6a1-11ea-2f6b-15247730bdff
begin
histogram(post_α, normed=true, bins=15, label = false)
title!("Posterior distribution of α")
xlabel!("α")
end

# ╔═╡ ab879968-f6a1-11ea-275e-f5d61d92ac40
begin
	histogram(post_β, normed=true, legend=false, bins=20)
	xlabel!("β")
	title!("Posterior distribution of β")
end

# ╔═╡ c8583d9a-f6a1-11ea-1104-6f1809309cf6
md"If we plot β vs α, we see that they are not correlated."

# ╔═╡ 5da9ad5c-f6a2-11ea-33fe-bfce123feece
begin
scatter(post_α, post_β, legend=false)
xlabel!("α")
ylabel!("β")
end

# ╔═╡ 8fbb7d16-f6a2-11ea-30e7-9112e29683dc
md"Now we want to plot different samples of α and β"

# ╔═╡ 9d1b2e34-f6a2-11ea-13b7-31ab92662ffe
p = range(25,65,step = 1);

# ╔═╡ a6f5c8d0-f6a2-11ea-2f09-155dc09c4a08
t = sample(post_2,1000);

# ╔═╡ c7dd3d2e-f6a2-11ea-3149-e155f82ecd51
sample_α = collect(get(t, :α));

# ╔═╡ 4344f4ca-f6a3-11ea-3ddf-af8c863f8123
sample_β = collect(get(t, :β));

# ╔═╡ 5949bb84-f6a3-11ea-0e72-c382e0ff61ae
begin
	μ = zeros(length(p),length(sample_β[1]))
	for i in collect(1:length(sample_β[1]))
		μ[:,i] = exp.(sample_α[1][i] .+ sample_β[1][i] .* (log.(p) .- mean(log.(p))))
	end
end

# ╔═╡ 41579d58-f6ac-11ea-0111-65cddb5bcb25
begin
	plot(p,μ[:,1])
	for i in collect(1:length(sample_β))
	end
end

# ╔═╡ 5b5a78e4-f6ac-11ea-12ac-cd0ad37a309f
begin
gr()
plot(p,μ[:,1])
	for i in collect(1:length(sample_β[1]))
			plot!(p,μ[:,i], color="blue", legend=false, alpha = 0.1)
	end
plot!(p, mean(μ, dims=2), color="red", lw=4)
scatter!(Price, Quantity, color="orange", markersize=7)
title!("E[Q∣P] samplig from the posterior distribution")
ylabel!("E[Q∣ P]")
xlabel!("Price")
current()
end

# ╔═╡ 84a7d050-f6ac-11ea-2987-9bb962663bf9
md"Taking into account the unit cost of k=\$20."

# ╔═╡ 877cde4c-f6ac-11ea-1921-b95af7982ffd
k = 20

# ╔═╡ 8e7ed27c-f6ac-11ea-1e8e-5f2579ce2f54
md"We compute now the profit π:"

# ╔═╡ 9526ca62-f6ac-11ea-2341-b128785b09f9
π = (p .- k).*μ;

# ╔═╡ 9d3b47e6-f6ac-11ea-2159-6d2cf2c9c622
md"Now we find the maximum value and plot:"

# ╔═╡ a60e33e2-f6ac-11ea-1fc5-274d32193e03
mxval, mxindx = findmax(mean(π, dims=2); dims=1);

# ╔═╡ acfbf7fc-f6ac-11ea-31bf-45157e8c9838
mxval

# ╔═╡ b29d3540-f6ac-11ea-1f32-3f577ce0dd2c
p[mxindx]

# ╔═╡ cbc982e4-f6ac-11ea-2763-474a62fd8381
mxval[1]/(p[mxindx][1] - k)

# ╔═╡ d623ccc0-f6ac-11ea-237e-63ad8d22f8ca
begin
plot(p,mean(π, dims=2), color = "red", lw=4, label="")
for i in collect(1:length(sample_β[1]))
			plot!(p,π[:,i], color="blue", label=false, alpha = 0.1)
	end
plot!(p,mean(π, dims=2), color = "red", lw=4, label="E[π|P]")
plot!(p,mean(π, dims=2) + std(π, dims=2),  color = "orange", lw=2, label ="E[π|P]±σ")
plot!(p,mean(π, dims=2) - std(π, dims=2),  color = "orange", lw=2, label="")
vline!(p[mxindx], p[mxindx], line = (:black, 3), label="ArgMax(E[π|P])")
xlabel!("Price")
ylabel!("Profit")
title!("Profit vs Price")
plot!(legend=true)
current()
end

# ╔═╡ Cell order:
# ╠═de1833c6-f695-11ea-292a-0b4edf4a288e
# ╟─e7c2debc-f695-11ea-341a-6fd18cac6451
# ╟─220f1bc6-f696-11ea-0898-4de9426d5ca3
# ╠═2b85f76a-f696-11ea-28fd-fb83f7fc1ede
# ╟─1b8532d0-f697-11ea-0206-ef5f6891880c
# ╠═259b4e58-f697-11ea-3412-3debf3847330
# ╠═470eef36-f697-11ea-2147-d324510b2f72
# ╟─b437c448-f697-11ea-2ac4-dd0af4dfbf03
# ╠═058cd946-f698-11ea-049f-ab7144c6dc4e
# ╠═b68a1a5c-f697-11ea-276b-1391a6f5a809
# ╠═bdc1d068-f69b-11ea-2fb6-7b249b7044db
# ╠═f67e55c0-f69b-11ea-2d98-b3e0835b7846
# ╠═4740786c-f69c-11ea-12a6-7be5f68e1ea7
# ╠═531b7920-f69c-11ea-15a5-419cb5240585
# ╟─02304f00-f6a0-11ea-15d0-fbfa262c3890
# ╠═5972fc04-f6a0-11ea-2113-fd9317a357a4
# ╟─a7dbb796-f6a0-11ea-25ad-091eec0bdad5
# ╠═b006c1ea-f6a0-11ea-28d0-f95e21f89e2b
# ╠═26e982d4-f6a1-11ea-1af6-a7753c3eb35d
# ╠═3f0556c2-f6a1-11ea-0672-cbf35b8721da
# ╠═65a17042-f6a1-11ea-3bb9-f19b5412b967
# ╠═7c9d5692-f6a1-11ea-25ad-251c2669c6ff
# ╠═91cf3152-f6a1-11ea-2f6b-15247730bdff
# ╠═ab879968-f6a1-11ea-275e-f5d61d92ac40
# ╟─c8583d9a-f6a1-11ea-1104-6f1809309cf6
# ╠═5da9ad5c-f6a2-11ea-33fe-bfce123feece
# ╟─8fbb7d16-f6a2-11ea-30e7-9112e29683dc
# ╠═9d1b2e34-f6a2-11ea-13b7-31ab92662ffe
# ╠═a6f5c8d0-f6a2-11ea-2f09-155dc09c4a08
# ╠═c7dd3d2e-f6a2-11ea-3149-e155f82ecd51
# ╠═4344f4ca-f6a3-11ea-3ddf-af8c863f8123
# ╠═5949bb84-f6a3-11ea-0e72-c382e0ff61ae
# ╠═41579d58-f6ac-11ea-0111-65cddb5bcb25
# ╠═5b5a78e4-f6ac-11ea-12ac-cd0ad37a309f
# ╟─84a7d050-f6ac-11ea-2987-9bb962663bf9
# ╟─877cde4c-f6ac-11ea-1921-b95af7982ffd
# ╟─8e7ed27c-f6ac-11ea-1e8e-5f2579ce2f54
# ╠═9526ca62-f6ac-11ea-2341-b128785b09f9
# ╟─9d3b47e6-f6ac-11ea-2159-6d2cf2c9c622
# ╠═a60e33e2-f6ac-11ea-1fc5-274d32193e03
# ╠═acfbf7fc-f6ac-11ea-31bf-45157e8c9838
# ╠═b29d3540-f6ac-11ea-1f32-3f577ce0dd2c
# ╠═cbc982e4-f6ac-11ea-2763-474a62fd8381
# ╠═d623ccc0-f6ac-11ea-237e-63ad8d22f8ca
