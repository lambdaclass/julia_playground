### A Pluto.jl notebook ###
# v0.11.14

using Markdown
using InteractiveUtils

# ╔═╡ 7c215ff6-2455-11eb-2c12-5feb0735f26a
begin
	using Plots
	using CSV
	using DataFrames
end

# ╔═╡ 7518fc20-2458-11eb-0974-ef3fb083b840
begin
	using Turing
	using Distributions
	using DifferentialEquations
	using StatsPlots
	using MCMCChains
	using DiffEqSensitivity
end

# ╔═╡ fa3464ba-2455-11eb-139a-69576736e9d1
lynxhare_df = CSV.read("lynxhare.csv")

# ╔═╡ e7659d28-252b-11eb-14be-2995e68f383e
md"hola"

# ╔═╡ 2d4f16a0-24f2-11eb-3c62-45f473f31f52
lynxhare = [lynxhare_df[2][50: end], lynxhare_df[3][50: end]]

# ╔═╡ 4a9fce20-24f7-11eb-087e-8f52116653de
lynxhare[1]

# ╔═╡ b5773a7e-2456-11eb-2735-e1f2185f7aff
begin
	plot(lynxhare_df[1],label="Hares Population" ,lynxhare_df[2])
	plot!(lynxhare_df[1], label="Lynx Population",lynxhare_df[3])
end

# ╔═╡ 2ddc13ce-245d-11eb-18b5-99a7bec856f1
function lotka_volterra(du,u,p,t)
  H, L = u
  birth_h, mortality_h, birth_l, mortality_l = p
  du[1] = dH = (birth_h - mortality_h * L)*H
  du[2] = dL = (birth_l * H - mortality_l)*L
end

# ╔═╡ 92dcda56-2462-11eb-3051-7da5e8ea6a07
@model lynx_hares_intaraction(h, l)
	
	#Lotka Volterra´s ODE parameters priors
	birth_h ~ truncated(Normal(1,0.5),0,Inf)
    mortality_h ~ truncated(Normal(0.05,00.5),0,Inf)
    birth_l ~ truncated(Normal(0.5,0.5),0,Inf)
    mortality_l ~ truncated(Normal(1,0.5),0,Inf)

	#Sample priors
	σ_h ~ Exponential(1)
	σ_l ~ Exponential(1)
	p_h ~ Beta(5,60)
	p_l ~ Beta(5,80)

	h ~ filldist(LogNormal(Log(p_h*H), σ_h), length(h))
	l ~ filldist(LogNormal(Log(p_h*L), σ_l), length(l))

	
	
	

	


# ╔═╡ 9c5f6a3e-24ef-11eb-1fde-59d28be4b67c
u0 = (20, 30)

# ╔═╡ c94cd8d4-24fd-11eb-3fa3-87902d2b0a47
Turing.setadbackend(:forwarddiff)

# ╔═╡ dbf0a948-24ee-11eb-0bf3-3997573f58ae
@model function fitlv(hares, lynxs)
    σ ~ InverseGamma(2, 3)
    birth_h ~ truncated(Normal(1,0.5),0.0,Inf)
    mortality_h ~ truncated(Normal(0.05,00.5),0.0,Inf)
    birth_l ~ truncated(Normal(0.5,0.5),0.0,Inf)
    mortality_l ~ truncated(Normal(1,0.5),0.0,Inf)
	
    p = [birth_h,mortality_h,birth_l,mortality_l]
    prob = ODEProblem(lotka_volterra,u0,(0.0,10.0),p)
    
	predicted = solve(prob,saveat=0.1,sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))
    
	for i = 1:length(predicted)
        hares[i] ~ Normal(predicted[i][1], σ)
		lynxs[i] ~ Normal(predicted[i][2], σ)
    end
end

# ╔═╡ dff837f8-24ef-11eb-3d0a-67066d695d17
model = fitlv(lynxhare[1], lynxhare[2])

# ╔═╡ e6519eb4-24ef-11eb-1298-638fb969b65f
chain = sample(model, NUTS(.65),1000)

# ╔═╡ Cell order:
# ╠═7c215ff6-2455-11eb-2c12-5feb0735f26a
# ╠═fa3464ba-2455-11eb-139a-69576736e9d1
# ╠═e7659d28-252b-11eb-14be-2995e68f383e
# ╠═2d4f16a0-24f2-11eb-3c62-45f473f31f52
# ╠═4a9fce20-24f7-11eb-087e-8f52116653de
# ╠═b5773a7e-2456-11eb-2735-e1f2185f7aff
# ╠═7518fc20-2458-11eb-0974-ef3fb083b840
# ╠═2ddc13ce-245d-11eb-18b5-99a7bec856f1
# ╠═92dcda56-2462-11eb-3051-7da5e8ea6a07
# ╠═9c5f6a3e-24ef-11eb-1fde-59d28be4b67c
# ╠═c94cd8d4-24fd-11eb-3fa3-87902d2b0a47
# ╠═dbf0a948-24ee-11eb-0bf3-3997573f58ae
# ╠═dff837f8-24ef-11eb-3d0a-67066d695d17
# ╠═e6519eb4-24ef-11eb-1298-638fb969b65f
