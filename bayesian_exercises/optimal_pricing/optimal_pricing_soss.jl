### A Pluto.jl notebook ###
# v0.11.12

using Markdown
using InteractiveUtils

# ╔═╡ b8565198-f118-11ea-0d40-4d7c36e9b4a5
using Pkg

# ╔═╡ d95b935a-f120-11ea-26d5-6d9ebf2d1ead
using Plots

# ╔═╡ cf53a8e8-f118-11ea-3152-816608e54eab
using Soss

# ╔═╡ 2dc2ccd8-f1e1-11ea-0db9-29ef463871ba
md"### Using Soss.jl to find the optimal price
We are goint to model the relationship between quantity and price using the equation:
Q = a$$P^c$$.
The priors of **a** and **c** are modeled as a *cauchy* distribution and Q is modeled as a *poisson* distribution."

# ╔═╡ 43092bdc-f439-11ea-180b-c9d426019dab
md"In order to linearize the model, we apply Log function in both sides of the equation:

Log(E(Q|P)) = Log(a) + c * Log(P)
"

# ╔═╡ d22fd35c-f118-11ea-3edb-7718bdde9af2
m = @model begin
	loga ~ Cauchy()
	c ~ Cauchy()
	logμ0 = loga .+ c*log.(p0)
	μ0 = exp.(logμ0)
	qval ~ For(eachindex(μ0)) do j
		Poisson(μ0[j])
	end
end

# ╔═╡ 5d43885c-f1e4-11ea-2434-35eae85b3864
md"Now we plot the data"

# ╔═╡ 47ecc5be-f119-11ea-389a-bb40e3785bba
begin
	Quantity = [53, 45, 25, 26, 25]
	Price = [30, 35, 40, 45, 50]
end

# ╔═╡ 76b36d78-f1e4-11ea-292b-21ce994361df
begin
scatter(Price, Quantity, markersize=6, color="orange", ylim=(0,60), legend=false)
xlabel!("Price")
ylabel!("Quantity")
end

# ╔═╡ dc299f60-f1e4-11ea-3e96-cd52c2ade4f9
md"Now, we estimate the posterior and sample from it."

# ╔═╡ d17d2580-f119-11ea-2c04-13423d23d98e
post = dynamicHMC(m(), (p0=Price,qval=Quantity,))

# ╔═╡ a511c95e-f120-11ea-3907-55d9769531af
begin
	loga_ = [vec.loga for vec in post]
	c_ = [vec.c for vec in post]
end

# ╔═╡ 73240334-f121-11ea-36d0-67b999d54bba
begin
histogram(c_, normed=true, bins=15)
title!("Posterior distribution of c")
xlabel!("c")
end

# ╔═╡ 7e45f0ba-f121-11ea-252f-8d9a95ba84bd
begin
histogram(loga_, normed=true, legend=false, bins=15)
xlabel!("log(a)")
title!("Posterior distribution of log(a)")
end

# ╔═╡ 03da3eca-f1e5-11ea-0394-01c1b2fb0aa5
md"If we plot c vs log(a), we notice that the two variables have multicollinearity."

# ╔═╡ b0497fbe-f121-11ea-2cb4-cf75cb2c595a
begin
scatter(c_, loga_, legend=false)
title!("log(a) vs c")
xlabel!("c")
ylabel!("log(a)")
end

# ╔═╡ 2d3b4132-f1e6-11ea-2ee1-fb48595b88db
md" Now we reparametrize the model to fix the multicollinearity problem, subtracting the mean of *log(p)*. We rename our model variables as β and α."

# ╔═╡ 1cee0a76-f123-11ea-0aab-3b184f23f7c0
m2 = @model begin
	α ~ Cauchy()
	β ~ Cauchy()
	logμ0_ = α .+ β*(log.(p0) .- mean(log.(p0)))
	μ0_ = exp.(logμ0_)
	qval ~ For(eachindex(μ0_)) do j
			Poisson(μ0_[j])
	end
end

# ╔═╡ 72b5aacc-f123-11ea-1712-396ad0fa25c6
post_m2 = dynamicHMC(m2(), (p0=Price,qval=Quantity,))

# ╔═╡ e5618412-f123-11ea-1a18-dd6c99b360a3
begin
	α_m2 = [vec.α for vec in post_m2]
	β_m2 = [vec.β for vec in post_m2]
end

# ╔═╡ 457c1410-f443-11ea-221d-898e52d5abd7
begin
	histogram(α_m2, normed=true, legend=false, bins=15)
	xlabel!("α")
	title!("Posterior distribution of α")
end

# ╔═╡ 76454b0c-f443-11ea-26b9-2f64cc290f4d
begin
	histogram(β_m2, normed=true, legend=false, bins=15)
	xlabel!("β")
	title!("Posterior distribution of β")
end

# ╔═╡ 14a4fd6a-f1e7-11ea-2fbb-1ff61d059ee1
md"If we plot β vs α, we see that they are not correlated."

# ╔═╡ 1f5e634a-f124-11ea-0c82-ff8d18b93857
begin
scatter(α_m2, β_m2, legend=false)
xlabel!("α")
ylabel!("β")
end

# ╔═╡ b14639c8-f443-11ea-2728-61b17d518771
md"Log(E(Q|P)) = α + β * ( Log(P) - mean(Log(P) ) "

# ╔═╡ 6986958c-f1e7-11ea-1d44-a13e86669b79
md"Now we want to plot different samples of α and β"

# ╔═╡ ea7579a6-f142-11ea-3baf-677c16d16983
p = range(25,65,step = 1);

# ╔═╡ 62759f08-f146-11ea-3e16-615c2d65589c
t = sample(post_m2,1000);

# ╔═╡ f9ecbae4-f146-11ea-03e8-4fd1154ef76d
sample_α = [i.α for i in t];

# ╔═╡ 17c5dfa0-f147-11ea-2c23-03a4b9a0015c
sample_β = [i.β for i in t];

# ╔═╡ 1863575c-f143-11ea-0919-83bcd3990507
begin
	μ = zeros(length(p),length(sample_β))
	for i in collect(1:length(sample_β))
		μ[:,i] = exp.(sample_α[i] .+ sample_β[i] .* (log.(p) .- mean(log.(p))))
	end
	µ
end;

# ╔═╡ 37788a2a-f147-11ea-1197-2bd5de8d32c5

begin
	plot(p,μ[:,1])
	for i in collect(1:length(sample_β))
	end
end

# ╔═╡ 99c24b46-f148-11ea-0d98-0356af21d751
begin
gr()
plot(p,μ[:,1])
	for i in collect(1:length(sample_β))
			plot!(p,μ[:,i], color="blue", legend=false, alpha = 0.1)
	end
plot!(p, mean(μ, dims=2), color="red", lw=4)
scatter!(Price, Quantity, color="orange", markersize=7)
title!("E[Q∣P] samplig from the posterior distribution")
ylabel!("E[Q∣ P]")
xlabel!("Price")
current()
end

# ╔═╡ b394c1fc-f1e8-11ea-3e21-0977e52cf6bf
md"Taking into account the unit cost of k=\$20."

# ╔═╡ 655ccbde-f14c-11ea-2939-1593d3c184b1
k = 20

# ╔═╡ f5681fac-f1e8-11ea-0a96-dfb103f91817
md"We compute now the profit π:"

# ╔═╡ 60d8e250-f14c-11ea-2939-bf82f0d72b48
π = (p .- k).*μ;

# ╔═╡ 1ff198c0-f1e9-11ea-10ee-9ba9f735f5f7
md"Now we find the maximum value and plot:"

# ╔═╡ 10e1f5e6-f1da-11ea-065e-c5b81776c798
mxval, mxindx = findmax(mean(π, dims=2); dims=1);

# ╔═╡ 46e85cca-f3a1-11ea-0489-7113e055c9a2
mxval

# ╔═╡ 4f489452-f3a1-11ea-2c0b-395d7e590f9c
p[mxindx]

# ╔═╡ 96e0edf8-f43e-11ea-1b7d-a99827c2a548
mxval[1]/(p[mxindx][1] - k)

# ╔═╡ 795baea4-f14c-11ea-3305-67250f6846c6
begin
plot(p,mean(π, dims=2), color = "red", lw=4, label="")
for i in collect(1:length(sample_β))
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
# ╠═b8565198-f118-11ea-0d40-4d7c36e9b4a5
# ╠═d95b935a-f120-11ea-26d5-6d9ebf2d1ead
# ╠═cf53a8e8-f118-11ea-3152-816608e54eab
# ╟─2dc2ccd8-f1e1-11ea-0db9-29ef463871ba
# ╟─43092bdc-f439-11ea-180b-c9d426019dab
# ╠═d22fd35c-f118-11ea-3edb-7718bdde9af2
# ╠═5d43885c-f1e4-11ea-2434-35eae85b3864
# ╠═47ecc5be-f119-11ea-389a-bb40e3785bba
# ╠═76b36d78-f1e4-11ea-292b-21ce994361df
# ╟─dc299f60-f1e4-11ea-3e96-cd52c2ade4f9
# ╠═d17d2580-f119-11ea-2c04-13423d23d98e
# ╠═a511c95e-f120-11ea-3907-55d9769531af
# ╠═73240334-f121-11ea-36d0-67b999d54bba
# ╠═7e45f0ba-f121-11ea-252f-8d9a95ba84bd
# ╟─03da3eca-f1e5-11ea-0394-01c1b2fb0aa5
# ╠═b0497fbe-f121-11ea-2cb4-cf75cb2c595a
# ╟─2d3b4132-f1e6-11ea-2ee1-fb48595b88db
# ╠═1cee0a76-f123-11ea-0aab-3b184f23f7c0
# ╠═72b5aacc-f123-11ea-1712-396ad0fa25c6
# ╠═e5618412-f123-11ea-1a18-dd6c99b360a3
# ╠═457c1410-f443-11ea-221d-898e52d5abd7
# ╠═76454b0c-f443-11ea-26b9-2f64cc290f4d
# ╟─14a4fd6a-f1e7-11ea-2fbb-1ff61d059ee1
# ╠═1f5e634a-f124-11ea-0c82-ff8d18b93857
# ╟─b14639c8-f443-11ea-2728-61b17d518771
# ╟─6986958c-f1e7-11ea-1d44-a13e86669b79
# ╠═ea7579a6-f142-11ea-3baf-677c16d16983
# ╠═62759f08-f146-11ea-3e16-615c2d65589c
# ╠═f9ecbae4-f146-11ea-03e8-4fd1154ef76d
# ╠═17c5dfa0-f147-11ea-2c23-03a4b9a0015c
# ╠═1863575c-f143-11ea-0919-83bcd3990507
# ╠═37788a2a-f147-11ea-1197-2bd5de8d32c5
# ╠═99c24b46-f148-11ea-0d98-0356af21d751
# ╟─b394c1fc-f1e8-11ea-3e21-0977e52cf6bf
# ╟─655ccbde-f14c-11ea-2939-1593d3c184b1
# ╟─f5681fac-f1e8-11ea-0a96-dfb103f91817
# ╠═60d8e250-f14c-11ea-2939-bf82f0d72b48
# ╟─1ff198c0-f1e9-11ea-10ee-9ba9f735f5f7
# ╠═10e1f5e6-f1da-11ea-065e-c5b81776c798
# ╠═46e85cca-f3a1-11ea-0489-7113e055c9a2
# ╠═4f489452-f3a1-11ea-2c0b-395d7e590f9c
# ╠═96e0edf8-f43e-11ea-1b7d-a99827c2a548
# ╠═795baea4-f14c-11ea-3305-67250f6846c6
