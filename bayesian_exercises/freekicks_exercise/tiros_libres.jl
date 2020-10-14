### A Pluto.jl notebook ###
# v0.12.3

using Markdown
using InteractiveUtils

# ╔═╡ cbc8c7bc-0a75-11eb-2f91-e15252a0b636
begin
	using CSV
	using JSON
	using DataFrames
	using Turing
	using Plots
	using StatsPlots
	using StatsBase
end

# ╔═╡ 41773b52-0d79-11eb-1aa8-4172f65bebff
begin
	using StatsFuns: logistic

	@model fk_logistic(x,y,n,J) = begin
	  # parameters
	  a ~ Normal(0, 1)
	  b ~ Normal(0, 1)

	  # model
	  for i in 1:J
		p = logistic(a + b * x[i])
		y[i] ~ Binomial(n[i], p)
	  end
	end
end

# ╔═╡ 21b25d60-0a77-11eb-39f9-93e480bf4a72
md" recordar agradecer a **Luca Pappalardo** por los datos. Referenciar al paper: [A public data set of spatio-temporal match events in soccer competitions](https://www.nature.com/articles/s41597-019-0247-7)
"

# ╔═╡ e40df4b4-0a75-11eb-0d67-8b7c1c67b19a
begin
	events_wc = JSON.parsefile("events/events_World_Cup.json")
	events_england = JSON.parsefile("events/events_England.json")
	events_euro_champ = JSON.parsefile("events/events_European_Championship.json")
	events_france = JSON.parsefile("events/events_France.json")
	events_germany = JSON.parsefile("events/events_Germany.json")
	events_italy = JSON.parsefile("events/events_Italy.json")
	events_spain = JSON.parsefile("events/events_Spain.json")
end

# ╔═╡ 258373da-0a76-11eb-01af-15d333bf9828
all_events = cat(events_wc, events_england, events_euro_champ, events_germany, events_france, events_spain, events_italy, dims=1);

# ╔═╡ 628eddb4-0a76-11eb-1dbb-7d51e21647d6
begin
	freekick_shots = []
	for event in all_events
		if event["eventId"] == 3 && event["subEventId"] == 33
			push!(freekick_shots, event)
		end
	end
end

# ╔═╡ 7001f05a-0a76-11eb-003b-593eaf66bef7
begin
	freekick_goals = []
	for fk in freekick_shots
		if Dict{String,Any}("id" => 101) in fk["tags"]
			push!(freekick_goals, fk)
		end
	end
end

# ╔═╡ 9875f0fc-0a76-11eb-15e5-d5d0c91afc05
md" **positions**: the origin and destination positions associated with the event. Each position is a pair of coordinates (x, y). Te x and y coordinates are always in the range [0, 100] and indicate the percentage of the feld
from the perspective of the attacking team. In particular, the value of the x coordinate indicates the event’s nearness (in percentage) to the opponent’s goal, while the value of the y coordinates indicates the event’s nearness (in percentage) to the right side of the feld.
"

# ╔═╡ 76e8ef98-0a76-11eb-119d-61bd78310c6c
begin
	number_freekicks = length(freekick_shots)
	x_field = 115 # en yardas
	y_field = 75 # en yardas

	freekick_ids = Array{Int64,1}(undef, number_freekicks)
	freekick_positions = Array{Float64,2}(undef, (number_freekicks,2))
	freekick_dist = Array{Float64,1}(undef, number_freekicks)
	freekick_scored = Array{Int64,1}(undef, number_freekicks)

	for i in 1:number_freekicks
		freekick_ids[i] = freekick_shots[i]["id"]
		freekick_positions[i,1] = freekick_shots[i]["positions"][1]["x"] * x_field/100
		freekick_positions[i,2] = freekick_shots[i]["positions"][1]["y"] * y_field/100
		freekick_dist[i] = sqrt((115/100*x_field)^2 + (75/200*y_field)^2) - 								   sqrt(freekick_positions[i,1]^2 + freekick_positions[i,2]^2)
		if Dict{String,Any}("id" => 101) in freekick_shots[i]["tags"]
			freekick_scored[i] = 1
		else
			freekick_scored[i] = 0
		end
	end
end

# ╔═╡ 93aecd80-0a76-11eb-0fcf-099e83780e36
begin
	pre_df = hcat(freekick_ids, freekick_positions, freekick_dist, freekick_scored)
	df_shots = DataFrame(ID = pre_df[:,1], x = pre_df[:,2], y = pre_df[:,3], distance = pre_df[:,4], scored = pre_df[:,5])
end

# ╔═╡ f75ef050-0d69-11eb-0823-1f38562db4aa
distancias = collect(6:2:90)

# ╔═╡ 3417a0f8-0a79-11eb-3b00-15ede2700de4
histogram(freekick_dist, xlabel="Distance to marker (yr)", bins=distancias, ylabel="Frequency", legend=false)

# ╔═╡ c4b94818-0d68-11eb-3428-fb17c7565111
freekick_dist_scored = [freekick_dist[i] for i in 1:number_freekicks if freekick_scored[i] == 1]

# ╔═╡ 2c25ae10-0d69-11eb-1e65-9bd728903b30
histogram(freekick_dist_scored, xlabel="Distance to marker (yr)", nbins=distancias, ylabel="Frequency", legend=false)

# ╔═╡ f55eec32-0d6a-11eb-0391-a3a5f3fdda4b
h_fk = fit(Histogram, freekick_dist, distancias)

# ╔═╡ 580f0dbc-0d6b-11eb-2460-e5108a22f14b
h_fk_sc = fit(Histogram, freekick_dist_scored, distancias)

# ╔═╡ 71ad15f4-0d78-11eb-28df-fbd666dc4497
begin
	accuracy = [i/j for (i,j) in zip(h_fk_sc.weights, h_fk.weights)]
	#error = @. sqrt((accuracy * (1 - accuracy) / h_fk.weights));
end

# ╔═╡ 0b9010ae-0d79-11eb-27f3-a5fa056e0c77
begin
	dist_eff = distancias[8:27]
	acc_eff = accuracy[7:27]
	fk_eff = h_fk.weights[8:27]
	fk_sc_eff = h_fk_sc.weights[8:27]
	scatter(distancias[8:27], accuracy[7:27]) #, yerror=error)
end

# ╔═╡ b0f05e64-0d7c-11eb-0c5c-13f9525f1421
chn = sample(fk_logistic(dist_eff, fk_sc_eff, fk_eff, length(dist_eff)), NUTS(), MCMCThreads(), 4000, 4);

# ╔═╡ b5ed9818-0d7f-11eb-1700-9f49c25a1208
begin
    post_a = collect(reshape(chn[:a], size(chn[:a],1)*size(chn[:a],2), 1))
    post_b = collect(reshape(chn[:b], size(chn[:b],1)*size(chn[:b],2), 1))
end

# ╔═╡ bb067d44-0d7d-11eb-14da-71cda3a95219
begin
    a_med = median(post_a)
    b_med = median(post_b)
    xrng = 1:1:21
    post_log_med = [logistic(a_med + b_med * x) for x in dist_eff]
    a_samp = StatsBase.sample(post_a, 100)
    b_samp = StatsBase.sample(post_b, 100)
    post_samp = [logistic(a_samp[i] + b_samp[i] * x) for x in dist_eff, i = 1:100]
    scatter(dist_eff, acc_eff, ylim=(0,1), xlabel="Distance from hole (ft)", ylabel="Probability of success", legend=false)
    plot!(dist_eff, post_log_med, color=:green)
    plot!(post_samp, alpha=0.2)
end

# ╔═╡ Cell order:
# ╟─21b25d60-0a77-11eb-39f9-93e480bf4a72
# ╠═cbc8c7bc-0a75-11eb-2f91-e15252a0b636
# ╠═e40df4b4-0a75-11eb-0d67-8b7c1c67b19a
# ╠═258373da-0a76-11eb-01af-15d333bf9828
# ╠═628eddb4-0a76-11eb-1dbb-7d51e21647d6
# ╠═7001f05a-0a76-11eb-003b-593eaf66bef7
# ╟─9875f0fc-0a76-11eb-15e5-d5d0c91afc05
# ╠═76e8ef98-0a76-11eb-119d-61bd78310c6c
# ╠═93aecd80-0a76-11eb-0fcf-099e83780e36
# ╠═f75ef050-0d69-11eb-0823-1f38562db4aa
# ╠═3417a0f8-0a79-11eb-3b00-15ede2700de4
# ╠═c4b94818-0d68-11eb-3428-fb17c7565111
# ╠═2c25ae10-0d69-11eb-1e65-9bd728903b30
# ╠═f55eec32-0d6a-11eb-0391-a3a5f3fdda4b
# ╠═580f0dbc-0d6b-11eb-2460-e5108a22f14b
# ╠═71ad15f4-0d78-11eb-28df-fbd666dc4497
# ╠═0b9010ae-0d79-11eb-27f3-a5fa056e0c77
# ╠═41773b52-0d79-11eb-1aa8-4172f65bebff
# ╠═b0f05e64-0d7c-11eb-0c5c-13f9525f1421
# ╠═b5ed9818-0d7f-11eb-1700-9f49c25a1208
# ╠═bb067d44-0d7d-11eb-14da-71cda3a95219
