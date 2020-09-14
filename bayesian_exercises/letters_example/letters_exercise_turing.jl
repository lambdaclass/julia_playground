### A Pluto.jl notebook ###
# v0.11.12

using Markdown
using InteractiveUtils

# ╔═╡ 2f3f2746-eec1-11ea-1946-0d71d014f2fe
begin 
	using Turing
	using DynamicHMC
	using Plots
	import StatsPlots
end

# ╔═╡ 27cbfc10-f6b3-11ea-24a1-fb107aac7d80
md"# Letters exercise using turing.jl"

# ╔═╡ 73e552fe-f695-11ea-2927-ebd758ca6850
data = vcat([i <= 36 ? rand(Poisson(10), 1)[1] : rand(Poisson(15), 1)[1] for i in 1:74])

# ╔═╡ 347e9520-f6b4-11ea-1350-f11281937e94
md"We plot the data"

# ╔═╡ 7cc960fe-f695-11ea-1ad6-65eff4c17f3a
begin 
	days = collect(1:74)
	bar(days, data, legend=false, xlabel="Día", ylabel="Cartas enviadas", alpha=0.8)
end

# ╔═╡ 54e15236-f695-11ea-0ee2-cdc480f6acb9
md"We define the model"

# ╔═╡ 3f9044ae-eec1-11ea-3f74-61f83363d09f
@model function m(x)
   N = length(x)
  μ1 ~ Exponential()
  μ2 ~ Exponential()
  τ ~ Uniform(0, N)
	for j in eachindex(x)
		if τ<j 
        	x[j] ~ Poisson(μ1)
		else 
        	x[j] ~ Poisson(μ2)
		end
    end
end

# ╔═╡ 5cf9447c-eef4-11ea-2b7e-c54c1ccc200d
chn = sample(m(data), DynamicNUTS(), 1000)

# ╔═╡ b6127366-f693-11ea-0eab-693c877b2901
post_values = get(chn, [:μ1, :μ2, :τ]);

# ╔═╡ ae0e8e44-f694-11ea-1f8b-f3b0c658e7fc
begin
	mu1 = histogram(post_values.μ1, bins=10, normed=true, color="green", xlabel="μ1", ylabel="Probabilidad", legend=false, xlim=(5,18), title="Posterior de μ1", alpha=0.8)
	mu2 = histogram(post_values.μ2 ,bins=10, normed=true, color="red", xlabel="μ2", ylabel="Probabilidad", legend=false, xlim=(5,18), title="Posterior de µ2", alpha=0.7)
	tau = histogram(post_values.τ, bins=25, normed=true, xlabel="Día", ylabel="Probabilidad", legend=false, title="Posterior de τ",alpha=0.7)
	plot(mu1, mu2, tau, layout=(3,1))
end

# ╔═╡ Cell order:
# ╟─27cbfc10-f6b3-11ea-24a1-fb107aac7d80
# ╠═2f3f2746-eec1-11ea-1946-0d71d014f2fe
# ╠═73e552fe-f695-11ea-2927-ebd758ca6850
# ╟─347e9520-f6b4-11ea-1350-f11281937e94
# ╠═7cc960fe-f695-11ea-1ad6-65eff4c17f3a
# ╟─54e15236-f695-11ea-0ee2-cdc480f6acb9
# ╠═3f9044ae-eec1-11ea-3f74-61f83363d09f
# ╠═5cf9447c-eef4-11ea-2b7e-c54c1ccc200d
# ╠═b6127366-f693-11ea-0eab-693c877b2901
# ╠═ae0e8e44-f694-11ea-1f8b-f3b0c658e7fc
