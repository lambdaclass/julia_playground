### A Pluto.jl notebook ###
# v0.11.12

using Markdown
using InteractiveUtils

# ╔═╡ 2e596292-f380-11ea-3bd1-f5144495e8b7
begin
	using Distributions
	using StatsPlots
	using Soss
	using Plots
end

# ╔═╡ 7c796a76-f380-11ea-201f-dbd03c41e18d
data = vcat([i <= 36 ? rand(Poisson(10), 1)[1] : rand(Poisson(15), 1)[1] for i in 1:74])

# ╔═╡ 423d32dc-f384-11ea-37b6-27c576e7b3cf
md"
Podemos ver los datos de cartas diarias enviadas durante aproximadamente dos meses y medio en el siguiente gráfico:
"

# ╔═╡ 39cfde02-f381-11ea-21a0-575ce6a2003d
begin
	days = collect(1:74)
	bar(days, data, legend=false, xlabel="Día", ylabel="Cartas enviadas", alpha=0.8)
end

# ╔═╡ 44dbee78-f386-11ea-364d-736089beb2a0
md"
En la siguiente celda, definimos nuestro modelo.
"

# ╔═╡ 20723620-f382-11ea-33ba-ed02683fd548
begin
	m = @model begin
		N = length(x)
		μ1 ~ Exponential(5)
		μ2 ~ Exponential(5)
		τ ~ Uniform(0, N)
		x ~ For(eachindex(x)) do j
			if j < τ
				Poisson(μ1)
			else
				Poisson(μ2)			
			end
		end
	end
end

# ╔═╡ 8ab9ffb2-f42d-11ea-0701-690fa85b0de5
md" Los priors que proponemos para el modelo son:"

# ╔═╡ f7809d40-f42d-11ea-3675-23430cbc56c3
begin
	pri_mus = rand(Exponential(), 10000)
	pri_taus = rand(Uniform(0, 74), 10000)
	hist_primu = histogram(pri_mus, legend=false, xlabel="μ", ylabel="Probabilidad", normed=true, title="Priors asignadas a las μ")
	hist_pritaus = histogram(pri_taus, legend=false, xlabel="τ", ylabel="Probabilidad", normed=true, title="Priors asignadas a τ")
	plot(hist_primu, hist_pritaus, layout=(1,2))
end

# ╔═╡ 37de6bb2-f382-11ea-2dec-05f0f92df578
begin
	n_samp = length(data)
	total_count = sum(data)
end;

# ╔═╡ 55bab7b2-f382-11ea-0614-7b028363725d
post = dynamicHMC(m(N=n_samp), (x=data,), 10000)

# ╔═╡ 1b8e4fc6-f383-11ea-3bea-d70cac623e23
total_post = length(post)

# ╔═╡ 3418864c-f383-11ea-3d57-cd49b43acb52
begin
	post_μ1 = [i.μ1 for i in post]
	post_μ2 = [i.μ2 for i in post]
	post_τ = [i.τ for i in post]
end

# ╔═╡ 4269d192-f383-11ea-0539-bf8dd431da7a
begin
	mu1 = histogram(post_μ1, bins=10, normed=true, color="green", xlabel="μ1", ylabel="Probabilidad", legend=false, xlim=(5,18), title="Posterior de μ1", alpha=0.8)
	mu2 = histogram(post_μ2, bins=10, normed=true, color="red", xlabel="μ2", ylabel="Probabilidad", legend=false, xlim=(5,18), title="Posterior de µ2", alpha=0.7)
	tau =histogram(post_τ, bins=25, normed=true, xlabel="Día", ylabel="Probabilidad", legend=false, title="Posterior de τ",alpha=0.7)
	plot(mu1, mu2, tau, layout=(3,1))
end

# ╔═╡ Cell order:
# ╠═2e596292-f380-11ea-3bd1-f5144495e8b7
# ╠═7c796a76-f380-11ea-201f-dbd03c41e18d
# ╟─423d32dc-f384-11ea-37b6-27c576e7b3cf
# ╟─39cfde02-f381-11ea-21a0-575ce6a2003d
# ╟─44dbee78-f386-11ea-364d-736089beb2a0
# ╠═20723620-f382-11ea-33ba-ed02683fd548
# ╟─8ab9ffb2-f42d-11ea-0701-690fa85b0de5
# ╟─f7809d40-f42d-11ea-3675-23430cbc56c3
# ╠═37de6bb2-f382-11ea-2dec-05f0f92df578
# ╠═55bab7b2-f382-11ea-0614-7b028363725d
# ╠═1b8e4fc6-f383-11ea-3bea-d70cac623e23
# ╠═3418864c-f383-11ea-3d57-cd49b43acb52
# ╠═4269d192-f383-11ea-0539-bf8dd431da7a
