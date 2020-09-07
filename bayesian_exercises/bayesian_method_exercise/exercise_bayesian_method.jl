### A Pluto.jl notebook ###
# v0.11.12

using Markdown
using InteractiveUtils

# ╔═╡ 094787f2-eef0-11ea-3226-23e7a473f829
begin
	using Distributions
	using StatsPlots
	using Soss
end

# ╔═╡ 084d8074-f10f-11ea-069f-59cdcd73d204
md"
An example from the first chapter of the book [Probabilistic Programming and Bayesian Methods for Hackers](https://nbviewer.jupyter.org/github/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/blob/master/Chapter1_Introduction/Ch1_Introduction_PyMC3.ipynb). 
We have some text message counts for each day of a two month period. We suspect there is a change in the behaviour of incoming messeges and the data is modeled as two Poisson distributions. 
We are going to estimate the parameters μ1 and μ2 from the two Poisson distributions and the day of the change in the behavior.
" 

# ╔═╡ 55321fda-eef0-11ea-00eb-0d5146f22494
data = [1.300000000000000000e+01
2.400000000000000000e+01
8.000000000000000000e+00
2.400000000000000000e+01
7.000000000000000000e+00
3.500000000000000000e+01
1.400000000000000000e+01
1.100000000000000000e+01
1.500000000000000000e+01
1.100000000000000000e+01
2.200000000000000000e+01
2.200000000000000000e+01
1.100000000000000000e+01
5.700000000000000000e+01
1.100000000000000000e+01
1.900000000000000000e+01
2.900000000000000000e+01
6.000000000000000000e+00
1.900000000000000000e+01
1.200000000000000000e+01
2.200000000000000000e+01
1.200000000000000000e+01
1.800000000000000000e+01
7.200000000000000000e+01
3.200000000000000000e+01
9.000000000000000000e+00
7.000000000000000000e+00
1.300000000000000000e+01
1.900000000000000000e+01
2.300000000000000000e+01
2.700000000000000000e+01
2.000000000000000000e+01
6.000000000000000000e+00
1.700000000000000000e+01
1.300000000000000000e+01
1.000000000000000000e+01
1.400000000000000000e+01
6.000000000000000000e+00
1.600000000000000000e+01
1.500000000000000000e+01
7.000000000000000000e+00
2.000000000000000000e+00
1.500000000000000000e+01
1.500000000000000000e+01
1.900000000000000000e+01
7.000000000000000000e+01
4.900000000000000000e+01
7.000000000000000000e+00
5.300000000000000000e+01
2.200000000000000000e+01
2.100000000000000000e+01
3.100000000000000000e+01
1.900000000000000000e+01
1.100000000000000000e+01
1.800000000000000000e+01
2.000000000000000000e+01
1.200000000000000000e+01
3.500000000000000000e+01
1.700000000000000000e+01
2.300000000000000000e+01
1.700000000000000000e+01
4.000000000000000000e+00
2.000000000000000000e+00
3.100000000000000000e+01
3.000000000000000000e+01
1.300000000000000000e+01
2.700000000000000000e+01
0.000000000000000000e+00
3.900000000000000000e+01
3.700000000000000000e+01
5.000000000000000000e+00
1.400000000000000000e+01
1.300000000000000000e+01
2.200000000000000000e+01]


# ╔═╡ 5e6a2320-f111-11ea-2fbe-ab531fde7cf0
md" 
First, we define the model
"

# ╔═╡ 7206bd9e-eef0-11ea-0759-e51b8e43c23f
begin
	m = @model begin
		N = length(x)
		μ1 ~ Exponential()
		μ2 ~ Exponential()
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

# ╔═╡ aabbb566-eef0-11ea-1cf3-e37dd7e7b3a6
begin
	n_samp = length(data)
	total_count = sum(data)
end

# ╔═╡ 727c1286-f111-11ea-2a2c-0f4ddcf2a44c
md" 
Using Hamiltonian Monte-Carlo, we sample from the posterior distribution.
"

# ╔═╡ a02a7f82-eef0-11ea-0fcc-ab831132902a
post = dynamicHMC(m(N=n_samp), (x=data,))
# We have to play with the parameters from the initialization
# https://tamaspapp.eu/DynamicHMC.jl/stable/interface/#DynamicHMC.mcmc_with_warmup

# ╔═╡ de18e226-f113-11ea-11b9-69236e7272b5
total_post = length(post)

# ╔═╡ b41f46fa-eef1-11ea-0ac8-71196e5b37df
begin
	post_μ1 = [i.μ1 for i in post]
	post_μ2 = [i.μ2 for i in post]
	post_τ = [i.τ for i in post]
end

# ╔═╡ 8ba0e658-f111-11ea-3385-6b5384fbb67a
md" 
An histogram of the results in shown
"

# ╔═╡ 8bddbfa4-eef2-11ea-3b8d-936703a4b439
histogram(post_μ1, bins=10, normed=true, color="green", ylabel="Probability", legend=false, title="Posterior probability of μ1")

# ╔═╡ bce91834-f111-11ea-1c44-175578606104
histogram(post_μ2, bins=10, normed=true, ylabel="Probability", legend=false, title="Posterior probability of µ2")

# ╔═╡ 292b1290-eef3-11ea-1a1e-29e0826fe38c
histogram(post_τ, bins=10, normed=true, ylabel="Probability", legend=false, title="Posterior probability of τ")

# ╔═╡ Cell order:
# ╠═094787f2-eef0-11ea-3226-23e7a473f829
# ╟─084d8074-f10f-11ea-069f-59cdcd73d204
# ╟─55321fda-eef0-11ea-00eb-0d5146f22494
# ╟─5e6a2320-f111-11ea-2fbe-ab531fde7cf0
# ╠═7206bd9e-eef0-11ea-0759-e51b8e43c23f
# ╠═aabbb566-eef0-11ea-1cf3-e37dd7e7b3a6
# ╟─727c1286-f111-11ea-2a2c-0f4ddcf2a44c
# ╠═a02a7f82-eef0-11ea-0fcc-ab831132902a
# ╠═de18e226-f113-11ea-11b9-69236e7272b5
# ╠═b41f46fa-eef1-11ea-0ac8-71196e5b37df
# ╠═8ba0e658-f111-11ea-3385-6b5384fbb67a
# ╠═8bddbfa4-eef2-11ea-3b8d-936703a4b439
# ╠═bce91834-f111-11ea-1c44-175578606104
# ╠═292b1290-eef3-11ea-1a1e-29e0826fe38c
