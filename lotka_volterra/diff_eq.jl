### A Pluto.jl notebook ###
# v0.11.14

using Markdown
using InteractiveUtils

# ╔═╡ 1e2d8a4c-24f9-11eb-3af1-793bebfa34e0
begin
	using Plots
	using CSV
	using DataFrames
	using Turing
	using Distributions
	using DifferentialEquations
	using StatsPlots
	using MCMCChains
	using DiffEqSensitivity
end

# ╔═╡ 28c72564-24fc-11eb-104a-233da7d239f4
lynxhare_df = CSV.read("lynxhare.csv")

# ╔═╡ 37ac6bac-24fc-11eb-0154-49cb77bf512c
lynxhare = Matrix(hcat(lynxhare_df[2][50: end], lynxhare_df[3][50: end])')

# ╔═╡ 32e1106c-24f9-11eb-3c81-bd54dec1d6be
function lotka_volterra1(du,u,p,t)
  x, y = u
  α, β, δ, γ = p
  du[1] = dx = (α - β*y)x
  du[2] = dy = (δ*x - γ)y
end

# ╔═╡ 9a592f72-24f9-11eb-2497-c31d671d4395
begin
	p = [1.1, 0.4, 0.1, 0.4]
	u0 = [70,46]
end;

# ╔═╡ 952eb490-24f9-11eb-1dcb-2bc0d4814a2d
prob = ODEProblem(lotka_volterra1,u0,(0.0,60.0),p);

# ╔═╡ af19ff86-24f9-11eb-14eb-dd27201b932e
sol = solve(prob,Tsit5(),saveat= 0.1);

# ╔═╡ c27762cc-25b6-11eb-0097-395576ee5821
plot(sol)

# ╔═╡ bc3bcd5c-24f9-11eb-23de-f719ec9ee69e
begin
	p1 = plot(1:42, lynxhare[1, :],legend=false)
	plot!(1:42, lynxhare[2, :],legend=false)
	p3 = plot(sol)
	plot(p1, p3, layout=(1,2))
end

# ╔═╡ c392184c-251d-11eb-3fdc-8b0ff54e3266
function lotka_volterra(du,u,p,t)
  x, y = u
  α, β, δ, γ = p
  du[1] = dx = (α - β*y)x
  du[2] = dy = (δ*x - γ)y
end

# ╔═╡ 30329ec2-2516-11eb-3635-21c7c94a2c56
u_init = [70,46]

# ╔═╡ 41de0b16-24fb-11eb-2e20-ed283467e581
@model function fitlv(data)
    σ ~ InverseGamma(2, 3)
    α ~ truncated(Normal(2,1),0,4)
    β ~ truncated(Normal(2,1),0,4)
    γ ~ truncated(Normal(2,1),0,4)
    δ ~ truncated(Normal(2,1),0,4)

    k = [α,β,γ,δ]
    prob = ODEProblem(lotka_volterra,u_init,(0.0,41.0),k)
    predicted = solve(prob,Tsit5(),saveat=1)

    for i = 1:length(predicted)
        data[:,i] ~ MvNormal(predicted[i], σ)
    end
end


# ╔═╡ f1a7f93e-24fe-11eb-25b4-338b37d5bc02
model = fitlv(lynxhare)

# ╔═╡ 02683d9c-24ff-11eb-0f02-1f6b676f8840
chain = sample(model, NUTS(.6),1000);

# ╔═╡ 223b3786-2500-11eb-2a60-2b10bc47ecdd
describe(chain);

# ╔═╡ 487fdcd0-2500-11eb-38d7-15c3930eae6e
begin
	plot(chain)
#savefig("plots")
end

# ╔═╡ Cell order:
# ╠═1e2d8a4c-24f9-11eb-3af1-793bebfa34e0
# ╠═28c72564-24fc-11eb-104a-233da7d239f4
# ╠═37ac6bac-24fc-11eb-0154-49cb77bf512c
# ╠═32e1106c-24f9-11eb-3c81-bd54dec1d6be
# ╠═9a592f72-24f9-11eb-2497-c31d671d4395
# ╠═952eb490-24f9-11eb-1dcb-2bc0d4814a2d
# ╠═af19ff86-24f9-11eb-14eb-dd27201b932e
# ╠═c27762cc-25b6-11eb-0097-395576ee5821
# ╠═bc3bcd5c-24f9-11eb-23de-f719ec9ee69e
# ╠═c392184c-251d-11eb-3fdc-8b0ff54e3266
# ╠═30329ec2-2516-11eb-3635-21c7c94a2c56
# ╠═41de0b16-24fb-11eb-2e20-ed283467e581
# ╠═f1a7f93e-24fe-11eb-25b4-338b37d5bc02
# ╠═02683d9c-24ff-11eb-0f02-1f6b676f8840
# ╠═223b3786-2500-11eb-2a60-2b10bc47ecdd
# ╠═487fdcd0-2500-11eb-38d7-15c3930eae6e
