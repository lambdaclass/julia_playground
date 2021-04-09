### A Pluto.jl notebook ###
# v0.14.1

using Markdown
using InteractiveUtils

# ╔═╡ 0b8375d7-72c3-4ab2-aa98-bacdd2e86ee6
begin
	using Soss
	using Random
end

# ╔═╡ b6901132-8d91-11eb-2ef4-b706c66db8c1
model1 = @model begin
	α ~ Normal(1,1)
	β ~ Normal(α^2,1)
end

# ╔═╡ e7b437b6-8d91-11eb-36b0-d32076b77c8f
model2 = @model μα, σα, μβ, σβ begin
	α ~ Normal(μα,σα)
	β ~ Normal(μβ,σβ)
end

# ╔═╡ 1c3d9162-8d92-11eb-3166-bfc8cc21356b
sourceParticleImportance(model1, model2)

# ╔═╡ 14a065e6-8d89-11eb-2762-cba71306448f
md"### Example linear regression"

# ╔═╡ 23d4468a-8d8a-11eb-0306-d552739d78db
X = randn(6,2);

# ╔═╡ c8c0a9dc-8d89-11eb-1c08-07bb424519e8
	model = @model X begin
    p = size(X, 2) # number of features
    α ~ Normal(0, 1) # intercept
    β ~ Normal(0, 1) |> iid(p) # coefficients
    σ ~ HalfNormal(1) # dispersion
    η = α .+ X * β # linear predictor
    μ = η # `μ = g⁻¹(η) = η`
    y ~ For(eachindex(μ)) do j
        Normal(μ[j], σ) # `Yᵢ ~ Normal(mean=μᵢ, variance=σ²)`
	end
end

# ╔═╡ fce45a28-399f-4d29-a3d7-b90e4a9e652a
begin
m = @model x begin
	μ ~ Normal(0,5)
	σ ~ HalfCauchy(3)
	N ~ length(x)
	x ~ Normal(μ, σ) |> iid(N)
end
end

# ╔═╡ 18484a9a-8d81-11eb-3145-eb46bc3f52a8
mPrior = prior(m)

# ╔═╡ 30742968-8d81-11eb-2fc2-03c3a6990b8d
sourceRand(mPrior)

# ╔═╡ 9453d084-8d84-11eb-0e65-7d404de7934c
m(N=3)

# ╔═╡ ed696728-8d85-11eb-0ecf-09049d86f1b2
m(μ = :(σ^2))

# ╔═╡ 9943b7c0-8d8a-11eb-3533-97053ed2eb7e
m

# ╔═╡ edf850ea-8d86-11eb-37a0-9116f6c40bca
dynamicHMC(m;x=randn(100))

# ╔═╡ ac3525ca-8d90-11eb-332a-b39ed34a4e78
f = codegen(m(N=100))

# ╔═╡ 2640e9a0-8d8a-11eb-2450-a1e122c21afd
forward_sample = rand(model(X=X))

# ╔═╡ 5ff17a0c-8d8a-11eb-0add-bd828d2c566d
begin
num_rows = 1_000
num_features = 2
X_ = randn(num_rows, num_features)
end;

# ╔═╡ 72ac2168-8d8a-11eb-06df-7b1c92788e4e
begin
β_true = [2.0, -1.0]
α_true = 1.0
σ_true = 0.5
end

# ╔═╡ b7222ff4-8d8c-11eb-268a-27386df714b0
begin
η_true = α_true .+ X_ * β_true
μ_true = η_true
noise = randn(num_rows) .* σ_true
y_true = μ_true .+ noise
end

# ╔═╡ f18d1f00-8d8c-11eb-2729-d79b217952e7
posterior = dynamicHMC(model(X=X), (y=y_true,))

# ╔═╡ 15586818-8d8d-11eb-05f9-0fbc2bf4a852
particles(posterior)

# ╔═╡ 33a07ff6-8d8d-11eb-3d07-5d3ff3032c30
pairs(particles(posterior))

# ╔═╡ 5552feba-8d8d-11eb-3308-0b4b6ca1194b
posterior_predictive = predictive(model, :β)

# ╔═╡ a3eb2728-8d8d-11eb-3b1b-a381ff50f748
y_ppc = [rand(posterior_predictive(;X=X, p...)).y for p in posterior]

# ╔═╡ 0a4709e7-e041-4728-8d82-6ce7f84a47bb
md"## Using Stable documentation"

# ╔═╡ 4c4f8ab3-5b31-4c9d-a893-292676120ecc
md"#### Model transform"

# ╔═╡ 12f310f8-eace-4756-8d57-c806bc70af84
md"##### Do"

# ╔═╡ 41f1521f-4754-48c9-a4f9-ef7c0069cd15
m_ = @model (n, k) begin
    β ~ Gamma()
    α ~ Gamma()
    θ ~ Beta(α, β)
    x ~ Binomial(n, θ)
    z ~ Binomial(k, α / (α + β))
end;


# ╔═╡ 856692eb-c1dd-4d63-a595-3a29f06bc244
Do(m_, :θ)

# ╔═╡ 1d50e9d8-ed43-4d42-ab53-91682a806208
md"""Returns a model transformed by adding xs... to arguments. The remainder of the body remains the same, consistent with Judea Pearl's "Do" operator. Unneeded arguments are trimmed."""

# ╔═╡ 0045a8af-0e85-484c-bc6d-e97baa5abdba
md"##### After"

# ╔═╡ 5836ec6a-9f41-4368-ae3b-7c9d092551e1
m_

# ╔═╡ 69d07595-644a-4db0-bc6d-988694505173
Soss.after(m_, :α)

# ╔═╡ 6a76e11f-1558-42db-9070-425fe3123066
Soss.after(m_, :α, strict = true) #What is a decendant in this context? No differences

# ╔═╡ f0ebb74e-888a-4cd4-949f-8acb8d34d89e
md"Transforms m by moving xs to arguments. If strict=true, only descendants of xs are retained in the body. Otherwise, the remaining variables in the body are unmodified. Unused arguments are trimmed.
"

# ╔═╡ 3a5278de-de6b-4386-9f4b-d82b81fef23e
md"##### Before"

# ╔═╡ 09fa9812-18d8-4f9d-95d8-722fbb59b644
m_

# ╔═╡ 5e692336-8fe1-40ae-a920-f9fbe76a3772
Soss.before(m_, :θ, inclusive = true, strict = false)

# ╔═╡ 6634820a-e78c-4187-8127-5b388747b195
Soss.before(m_, :θ, inclusive = true, strict = true)

# ╔═╡ 185c4634-cb7c-4ea7-aab7-922020a07ad9
Soss.before(m_, :θ, inclusive = false, strict = false)

# ╔═╡ d3a1334a-57f9-42a0-879d-98dd88a3e462
Soss.before(m_, :θ, inclusive = false, strict = true)

# ╔═╡ db9602a7-52ef-410e-a62c-1a682835315e
md"Transforms m by retaining all ancestors of any of xs if strict=true; if strict=false, retains all variables that are not descendants of any xs. Note that adding more variables to xs cannot result in a larger model. If inclusive=true, xs is considered to be an ancestor of itself and is always included in the returned Model. Unneeded arguments are trimmed."

# ╔═╡ 909db8f9-e1f4-43e3-8dc7-d6228e48a246
md"#### prior"

# ╔═╡ c138d9d6-3544-4259-80bd-1e7cb4e696c9
md"Returns the minimal model required to sample random variables xs.... Useful for extracting a prior distribution from a joint model m by designating xs... and the variables they depend on as the prior and hyperpriors."

# ╔═╡ 4fe4bc08-004d-4c89-871b-d433f02d111e
m_

# ╔═╡ 3e7b1ca4-ecfd-445d-9b02-537ebbe4d84e
Soss.prior(m_, :x)

# ╔═╡ ca99eca7-e979-42aa-847c-2533954c7467
Soss.prior(m_, :z)

# ╔═╡ 78696204-1aed-4645-a787-65d0696e8194
md"#### likelihood"

# ╔═╡ 7a73ef60-7806-4779-b670-16c5fc8eb014
m_

# ╔═╡ f05df2f8-1826-4b19-bcc2-cf7385ce5f36
Soss.likelihood(m_, :x)

# ╔═╡ fbbe1a61-a953-4f30-8f9f-6e6fb66b6be5
Soss.likelihood(m_, :x, :θ)

# ╔═╡ 3fa5763b-8a0a-4e5c-ac47-86594d9cc5f9
Soss.likelihood(m_, :x, :θ, :β)

# ╔═╡ Cell order:
# ╠═0b8375d7-72c3-4ab2-aa98-bacdd2e86ee6
# ╠═fce45a28-399f-4d29-a3d7-b90e4a9e652a
# ╠═18484a9a-8d81-11eb-3145-eb46bc3f52a8
# ╠═30742968-8d81-11eb-2fc2-03c3a6990b8d
# ╠═9453d084-8d84-11eb-0e65-7d404de7934c
# ╠═ed696728-8d85-11eb-0ecf-09049d86f1b2
# ╠═9943b7c0-8d8a-11eb-3533-97053ed2eb7e
# ╠═edf850ea-8d86-11eb-37a0-9116f6c40bca
# ╠═ac3525ca-8d90-11eb-332a-b39ed34a4e78
# ╠═b6901132-8d91-11eb-2ef4-b706c66db8c1
# ╠═e7b437b6-8d91-11eb-36b0-d32076b77c8f
# ╠═1c3d9162-8d92-11eb-3166-bfc8cc21356b
# ╟─14a065e6-8d89-11eb-2762-cba71306448f
# ╠═c8c0a9dc-8d89-11eb-1c08-07bb424519e8
# ╠═23d4468a-8d8a-11eb-0306-d552739d78db
# ╠═2640e9a0-8d8a-11eb-2450-a1e122c21afd
# ╠═5ff17a0c-8d8a-11eb-0add-bd828d2c566d
# ╠═72ac2168-8d8a-11eb-06df-7b1c92788e4e
# ╠═b7222ff4-8d8c-11eb-268a-27386df714b0
# ╠═f18d1f00-8d8c-11eb-2729-d79b217952e7
# ╠═15586818-8d8d-11eb-05f9-0fbc2bf4a852
# ╠═33a07ff6-8d8d-11eb-3d07-5d3ff3032c30
# ╠═5552feba-8d8d-11eb-3308-0b4b6ca1194b
# ╠═a3eb2728-8d8d-11eb-3b1b-a381ff50f748
# ╟─0a4709e7-e041-4728-8d82-6ce7f84a47bb
# ╟─4c4f8ab3-5b31-4c9d-a893-292676120ecc
# ╟─12f310f8-eace-4756-8d57-c806bc70af84
# ╠═41f1521f-4754-48c9-a4f9-ef7c0069cd15
# ╠═856692eb-c1dd-4d63-a595-3a29f06bc244
# ╟─1d50e9d8-ed43-4d42-ab53-91682a806208
# ╟─0045a8af-0e85-484c-bc6d-e97baa5abdba
# ╠═5836ec6a-9f41-4368-ae3b-7c9d092551e1
# ╠═69d07595-644a-4db0-bc6d-988694505173
# ╠═6a76e11f-1558-42db-9070-425fe3123066
# ╟─f0ebb74e-888a-4cd4-949f-8acb8d34d89e
# ╟─3a5278de-de6b-4386-9f4b-d82b81fef23e
# ╠═09fa9812-18d8-4f9d-95d8-722fbb59b644
# ╠═5e692336-8fe1-40ae-a920-f9fbe76a3772
# ╠═6634820a-e78c-4187-8127-5b388747b195
# ╠═185c4634-cb7c-4ea7-aab7-922020a07ad9
# ╠═d3a1334a-57f9-42a0-879d-98dd88a3e462
# ╟─db9602a7-52ef-410e-a62c-1a682835315e
# ╟─909db8f9-e1f4-43e3-8dc7-d6228e48a246
# ╟─c138d9d6-3544-4259-80bd-1e7cb4e696c9
# ╠═4fe4bc08-004d-4c89-871b-d433f02d111e
# ╠═3e7b1ca4-ecfd-445d-9b02-537ebbe4d84e
# ╠═ca99eca7-e979-42aa-847c-2533954c7467
# ╟─78696204-1aed-4645-a787-65d0696e8194
# ╠═7a73ef60-7806-4779-b670-16c5fc8eb014
# ╠═f05df2f8-1826-4b19-bcc2-cf7385ce5f36
# ╠═fbbe1a61-a953-4f30-8f9f-6e6fb66b6be5
# ╠═3fa5763b-8a0a-4e5c-ac47-86594d9cc5f9
