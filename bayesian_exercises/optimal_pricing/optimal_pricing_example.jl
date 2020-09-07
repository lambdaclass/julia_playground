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

# ╔═╡ 47ecc5be-f119-11ea-389a-bb40e3785bba
begin
	Quantity = [53, 45, 25, 26, 25]
	Price = [30, 35, 40, 45, 50]
end

# ╔═╡ d17d2580-f119-11ea-2c04-13423d23d98e
post = dynamicHMC(m(), (p0=Price,qval=Quantity,))

# ╔═╡ a511c95e-f120-11ea-3907-55d9769531af
begin
	loga_ = [vec.loga for vec in post]
	c_ = [vec.c for vec in post]
end

# ╔═╡ 73240334-f121-11ea-36d0-67b999d54bba
histogram(c_)

# ╔═╡ 7e45f0ba-f121-11ea-252f-8d9a95ba84bd
histogram(loga_)

# ╔═╡ b0497fbe-f121-11ea-2cb4-cf75cb2c595a
scatter(c_, loga_)

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

# ╔═╡ 1f5e634a-f124-11ea-0c82-ff8d18b93857
scatter(α_m2, β_m2)

# ╔═╡ ea7579a6-f142-11ea-3baf-677c16d16983
p = range(25,55,step = 1)

# ╔═╡ 62759f08-f146-11ea-3e16-615c2d65589c
t = sample(post_m2,1000)


# ╔═╡ f9ecbae4-f146-11ea-03e8-4fd1154ef76d
sample_α = [i.α for i in t]

# ╔═╡ 17c5dfa0-f147-11ea-2c23-03a4b9a0015c
sample_β = [i.β for i in t]

# ╔═╡ 1863575c-f143-11ea-0919-83bcd3990507
begin
	μ = zeros(length(p),length(sample_β))
	for i in collect(1:length(sample_β))
		μ[:,i] = exp.(sample_α[i] .+ sample_β[i] .* (log.(p) .- mean(log.(Price))))
	end
end

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
scatter!(Price, Quantity, color="red", markersize=7)
title!("E[Q∣P] samplig from the posterior distribution")
ylabel!("E[Q∣ P]")
xlabel!("P")
current()
end

# ╔═╡ 655ccbde-f14c-11ea-2939-1593d3c184b1
k = 20

# ╔═╡ 60d8e250-f14c-11ea-2939-bf82f0d72b48
π = (p .- k).*μ;

# ╔═╡ 795baea4-f14c-11ea-3305-67250f6846c6
begin
	plot(p,mean(π, dims=2), color = "red", lw=4)
	for i in collect(1:length(sample_β))
			plot!(p,π[:,i], color="blue", legend=false, alpha = 0.1)
	end
plot!(p,mean(π, dims=2), color = "red", lw=4)
plot!(p,mean(π, dims=2) + std(π, dims=2),  color = "red", lw=2)
plot!(p,mean(π, dims=2) - std(π, dims=2),  color = "red", lw=2)
current()
end

# ╔═╡ Cell order:
# ╠═b8565198-f118-11ea-0d40-4d7c36e9b4a5
# ╠═d95b935a-f120-11ea-26d5-6d9ebf2d1ead
# ╠═cf53a8e8-f118-11ea-3152-816608e54eab
# ╠═d22fd35c-f118-11ea-3edb-7718bdde9af2
# ╠═47ecc5be-f119-11ea-389a-bb40e3785bba
# ╠═d17d2580-f119-11ea-2c04-13423d23d98e
# ╠═a511c95e-f120-11ea-3907-55d9769531af
# ╠═73240334-f121-11ea-36d0-67b999d54bba
# ╠═7e45f0ba-f121-11ea-252f-8d9a95ba84bd
# ╠═b0497fbe-f121-11ea-2cb4-cf75cb2c595a
# ╠═1cee0a76-f123-11ea-0aab-3b184f23f7c0
# ╠═72b5aacc-f123-11ea-1712-396ad0fa25c6
# ╠═e5618412-f123-11ea-1a18-dd6c99b360a3
# ╠═1f5e634a-f124-11ea-0c82-ff8d18b93857
# ╠═ea7579a6-f142-11ea-3baf-677c16d16983
# ╠═62759f08-f146-11ea-3e16-615c2d65589c
# ╠═f9ecbae4-f146-11ea-03e8-4fd1154ef76d
# ╠═17c5dfa0-f147-11ea-2c23-03a4b9a0015c
# ╠═1863575c-f143-11ea-0919-83bcd3990507
# ╠═37788a2a-f147-11ea-1197-2bd5de8d32c5
# ╠═99c24b46-f148-11ea-0d98-0356af21d751
# ╠═655ccbde-f14c-11ea-2939-1593d3c184b1
# ╠═60d8e250-f14c-11ea-2939-bf82f0d72b48
# ╠═795baea4-f14c-11ea-3305-67250f6846c6
