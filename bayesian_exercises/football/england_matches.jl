### A Pluto.jl notebook ###
# v0.11.14

using Markdown
using InteractiveUtils

# ╔═╡ 2642cf5c-1a1e-11eb-0c0a-93565781a529
begin
	using CSV
	using JSON
	using DataFrames
	using Plots
end

# ╔═╡ 84fe1936-1a1f-11eb-30be-4572afbfb8f6
using Turing

# ╔═╡ 4f00d9a2-1a1e-11eb-1fc1-5df751317628
england_league = JSON.parsefile("matches_England.json");

# ╔═╡ 6266d28a-1a1e-11eb-3e05-b36d4198645a
england_league[1]

# ╔═╡ 3a0d1564-1a1f-11eb-3601-893be9e277d5
matches_df = DataFrame(home = [], away = [], score_home = [], score_away = []);

# ╔═╡ 3d2809a2-1a1f-11eb-389a-2dc4e0b0b27e
begin
	matches = []
	for match in england_league
		push!(matches, split(match["label"], ","))
end
end

# ╔═╡ 4e35d1c0-1a1f-11eb-2594-efdc8c03ab14
begin
for match in matches
	home, away = split(match[1], " - ")
	score_home, score_away = split(match[2], " - ")
	
	push!(matches_df,[home, away, parse(Int,score_home), parse(Int,score_away)])
end
end	

# ╔═╡ 56b29036-1a1f-11eb-1370-f32f504a1a3b
matches_df

# ╔═╡ 6443c7a4-1a1f-11eb-3765-6d09bc0e6f0d
teams = unique(collect(matches_df[1]))

# ╔═╡ 7011e504-1a1f-11eb-0202-4bdcfd1fa9bf
team_index = collect(1:length(teams))

# ╔═╡ 7639dc84-1a1f-11eb-10e2-b39f98dca4ed
teams_df = DataFrame(team = teams, idex = team_index);

# ╔═╡ 8b6be714-1a1f-11eb-216b-83462872ee6f
begin
	@model function football_matches(home_teams, away_teams, score_home, score_away, teams)
	#hiper priors
	μatt ~ Normal(0, 0.0001)
	σatt ~ Gamma(0.1, 0.1)
	μdeff ~ Normal(0, 0.0001)
	σdeff ~ Gamma(0.1,0.1)
		
	#Team-specific effects
	home ~ Normal(0,0.0001)
		
	att_ = Vector{Real}(undef, length(teams))
	deff_ = Vector{Real}(undef, length(teams))
		
	for i in eachindex(teams)
		att_[i] ~ Normal(µatt, σatt)
		deff_[i] ~ Normal(μdeff, σdeff)
	end
	
	dict = Dict{String, Int64}()
	for (i, team) in enumerate(teams)
		dict[team] = i
	end
		
	#Zero-sum constrains
	mean_att = mean(att_)
	mean_deff = mean(deff_)
	
	for i in eachindex(teams)
		att_[i] = att_[i] - mean_att
		deff_[i] = deff_[i] - mean_deff
		end
		
	θ_home = Vector{Real}(undef, length(home_teams))
	θ_away = Vector{Real}(undef, length(home_teams))
		
	#Modeling score-rate and scores
	for i in 1:length(home_teams)
		#score-rate
		θ_home[i] = exp(home + att_[dict[home_teams[i]]] + deff_[dict[away_teams[i]]])
		θ_away[i] = exp(att_[dict[away_teams[i]]] + deff_[dict[home_teams[i]]])
		#scores
		score_home[i] ~ Poisson(θ_home[i])
		score_away[i] ~ Poisson(θ_away[i])
	end
	
	end
end

# ╔═╡ def7c618-1a4d-11eb-160c-0d4a28e15da1
model = football_matches(matches_df[1], matches_df[2], matches_df[3], matches_df[4], teams_df[1])

# ╔═╡ ad743226-1a1f-11eb-0821-ff775a89a4ef
posterior = sample(model, NUTS(),1000)

# ╔═╡ 6f93f71e-1a26-11eb-3f18-55268a95c5fd
post_home = collect(get(posterior, :home))

# ╔═╡ 81b593ce-1a26-11eb-352d-7120a509bd15
mean(post_home[1])

# ╔═╡ 5b50707e-1a27-11eb-155b-5fead73b049a
histogram(post_home[1])

# ╔═╡ Cell order:
# ╠═2642cf5c-1a1e-11eb-0c0a-93565781a529
# ╠═4f00d9a2-1a1e-11eb-1fc1-5df751317628
# ╠═6266d28a-1a1e-11eb-3e05-b36d4198645a
# ╠═3a0d1564-1a1f-11eb-3601-893be9e277d5
# ╠═3d2809a2-1a1f-11eb-389a-2dc4e0b0b27e
# ╠═4e35d1c0-1a1f-11eb-2594-efdc8c03ab14
# ╠═56b29036-1a1f-11eb-1370-f32f504a1a3b
# ╠═6443c7a4-1a1f-11eb-3765-6d09bc0e6f0d
# ╠═7011e504-1a1f-11eb-0202-4bdcfd1fa9bf
# ╠═7639dc84-1a1f-11eb-10e2-b39f98dca4ed
# ╠═84fe1936-1a1f-11eb-30be-4572afbfb8f6
# ╠═8b6be714-1a1f-11eb-216b-83462872ee6f
# ╠═def7c618-1a4d-11eb-160c-0d4a28e15da1
# ╠═ad743226-1a1f-11eb-0821-ff775a89a4ef
# ╠═6f93f71e-1a26-11eb-3f18-55268a95c5fd
# ╠═81b593ce-1a26-11eb-352d-7120a509bd15
# ╠═5b50707e-1a27-11eb-155b-5fead73b049a
