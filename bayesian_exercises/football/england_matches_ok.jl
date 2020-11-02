### A Pluto.jl notebook ###
# v0.11.14

using Markdown
using InteractiveUtils

# ╔═╡ ded336b0-1d34-11eb-2784-015cbf2b8bdb
begin
	using CSV
	using JSON
	using DataFrames
	using Plots
end

# ╔═╡ 56de188c-1d35-11eb-0064-c191886a8931
using Turing

# ╔═╡ 2153f38a-1d35-11eb-189c-53219c315b3b
england_league = JSON.parsefile("matches_England.json");

# ╔═╡ 258cae36-1d35-11eb-0397-15d7bb86c6ac
matches_df = DataFrame(home = [], away = [], score_home = [], score_away = []);

# ╔═╡ 2f13e372-1d35-11eb-08c0-fd641c41126f
begin
	matches = []
	for match in england_league
		push!(matches, split(match["label"], ","))
end
end

# ╔═╡ 308c4c58-1d35-11eb-3ec5-03440844fde5
begin
for match in matches
	home, away = split(match[1], " - ")
	score_home, score_away = split(match[2], " - ")
	
	push!(matches_df,[home, away, parse(Int,score_home), parse(Int,score_away)])
end
end	

# ╔═╡ 3520fa48-1d35-11eb-2e56-adf72156db57
matches_df

# ╔═╡ 398c0230-1d35-11eb-0c30-43da7081d468
teams = unique(collect(matches_df[1]))

# ╔═╡ 40487464-1d35-11eb-1c16-0950d5422227
team_index = collect(1:length(teams))

# ╔═╡ 45a5fd66-1d35-11eb-3aa6-030a988e6935
teams_df = DataFrame(team = teams, idex = team_index)

# ╔═╡ 5d879190-1d35-11eb-3411-e3c6012f8026
begin
	@model function football_matches(home_teams, away_teams, score_home, score_away, teams)
	#hiper priors
	σatt ~ Gamma(0.1, 0.1)
	σdeff ~ Gamma(0.1,0.1)
		
	#Team-specific effects
	home ~ Normal(0,100)
		
	att_ ~ filldist(Normal(0, σatt), length(teams))
	deff_ ~ filldist(Normal(0, σdeff), length(teams))
	
	dict = Dict{String, Int64}()
	for (i, team) in enumerate(teams)
		dict[team] = i
	end
		
	#Zero-sum constrains
	offset = mean(att_) + mean(deff_)
	
	log_θ_home = Vector{Real}(undef, length(home_teams))
	log_θ_away = Vector{Real}(undef, length(home_teams))
		
	#Modeling score-rate and scores
	for i in 1:length(home_teams)
		#score-rate
		log_θ_home[i] = home + att_[dict[home_teams[i]]] + deff_[dict[away_teams[i]]] - offset
		log_θ_away[i] = att_[dict[away_teams[i]]] + deff_[dict[home_teams[i]]] - offset
		#scores
		score_home[i] ~ LogPoisson(log_θ_home[i])
		score_away[i] ~ LogPoisson(log_θ_away[i])
	end
	
	end
end

# ╔═╡ 668b86d4-1d35-11eb-3ce0-f5fff3291ee1
model = football_matches(matches_df[1], matches_df[2], matches_df[3], matches_df[4], teams_df[1])

# ╔═╡ 67f1fcbc-1d35-11eb-2fa7-d16c2288c8dc
posterior = sample(model, NUTS(),1000)

# ╔═╡ 3bac52b6-1d37-11eb-0676-7f2c0aa20bd0
begin
	post_att = collect(get(posterior, :att_)[1])
	post_def = collect(get(posterior, :deff_)[1])
	post_def = collect(get(posterior, :home)[1])
end

# ╔═╡ 44223104-1d37-11eb-073b-41e8a684d252
mean(post_att[4])

# ╔═╡ 30586da6-1d3b-11eb-1372-4173d4f3f890
mean(post_def[4])

# ╔═╡ 0613a2a4-1d3b-11eb-1a8e-fb750a9d1021
mean(post_att[11])

# ╔═╡ 0feb853c-1d3b-11eb-243f-abeaba311466
mean(post_def[11])

# ╔═╡ 1790782c-1d3b-11eb-2825-8d9d3335f8ee


# ╔═╡ Cell order:
# ╠═ded336b0-1d34-11eb-2784-015cbf2b8bdb
# ╠═2153f38a-1d35-11eb-189c-53219c315b3b
# ╠═258cae36-1d35-11eb-0397-15d7bb86c6ac
# ╠═2f13e372-1d35-11eb-08c0-fd641c41126f
# ╠═308c4c58-1d35-11eb-3ec5-03440844fde5
# ╠═3520fa48-1d35-11eb-2e56-adf72156db57
# ╠═398c0230-1d35-11eb-0c30-43da7081d468
# ╠═40487464-1d35-11eb-1c16-0950d5422227
# ╠═45a5fd66-1d35-11eb-3aa6-030a988e6935
# ╠═56de188c-1d35-11eb-0064-c191886a8931
# ╠═5d879190-1d35-11eb-3411-e3c6012f8026
# ╠═668b86d4-1d35-11eb-3ce0-f5fff3291ee1
# ╠═67f1fcbc-1d35-11eb-2fa7-d16c2288c8dc
# ╠═3bac52b6-1d37-11eb-0676-7f2c0aa20bd0
# ╠═44223104-1d37-11eb-073b-41e8a684d252
# ╠═30586da6-1d3b-11eb-1372-4173d4f3f890
# ╠═0613a2a4-1d3b-11eb-1a8e-fb750a9d1021
# ╠═0feb853c-1d3b-11eb-243f-abeaba311466
# ╠═1790782c-1d3b-11eb-2825-8d9d3335f8ee
