### A Pluto.jl notebook ###
# v0.11.14

using Markdown
using InteractiveUtils

# ╔═╡ ded336b0-1d34-11eb-2784-015cbf2b8bdb
begin
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
	#post_def = collect(get(posterior, :home)[1])
end;

# ╔═╡ de185204-1d3f-11eb-2d00-39a5dd7785d4
begin
	teams_att = []
	teams_def = []
	for i in 1:length(post_att)
		push!(teams_att, post_att[i])
		push!(teams_def, post_def[i])
	end
end

# ╔═╡ 6418d3ce-1d40-11eb-0523-3914c40b4149
begin
teams_att_μ = mean.(teams_att)
teams_def_μ = mean.(teams_def)
teams_att_σ = std.(teams_att)
end

# ╔═╡ 8b91e1a6-1d41-11eb-3248-614810563a36
abbr_names = [t[1:3] for t in teams]

# ╔═╡ 277795ac-1d42-11eb-166c-79c565d6bf2a
sorted_att = sortperm(teams_att_μ)

# ╔═╡ 72bdf09e-1d45-11eb-0085-5ba014db07f0
sorted_names = abbr_names[sorted_att]

# ╔═╡ e5706920-1d43-11eb-25fe-593fcbf6d932
begin
	scatter(1:20, teams_att_μ[sorted_att], grid=false, legend=false, yerror=teams_att_σ[sorted_att], color=:blue)
	annotate!(collect(1:20), teams_att_μ[sorted_att] .+ 0.2, text.(sorted_names, :black, :center, 8))
	ylabel!("Mean team attack")
end

# ╔═╡ 86e9ce8c-1d48-11eb-0354-1f1c27728e0e
table_position = [12, 5, 9, 4, 13, 14, 1, 15, 12, 6, 2, 16, 10, 17, 20, 3, 7, 8, 19, 18]

# ╔═╡ 4fa2b18e-1d50-11eb-169e-4777411537c1
position = sortperm(table_position) #Indice de los valores ordenados

# ╔═╡ 7dfd3942-1d40-11eb-1a1e-250774644dc2
begin
scatter(teams_att_μ, teams_def_μ, legend=false)
annotate!(teams_att_μ, teams_def_μ.+ 0.015, text.(abbr_names, :black, :center, 8))
annotate!(teams_att_μ, teams_def_μ.- 0.015, text.(position, :left, :center, 8))

xlabel!("Mean team attack")
ylabel!("Mean team defense")
end

# ╔═╡ bc1b6958-1d4f-11eb-2f77-0d992418be42
teams_df[1]

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
# ╠═de185204-1d3f-11eb-2d00-39a5dd7785d4
# ╠═6418d3ce-1d40-11eb-0523-3914c40b4149
# ╠═7dfd3942-1d40-11eb-1a1e-250774644dc2
# ╠═8b91e1a6-1d41-11eb-3248-614810563a36
# ╠═277795ac-1d42-11eb-166c-79c565d6bf2a
# ╠═e5706920-1d43-11eb-25fe-593fcbf6d932
# ╠═72bdf09e-1d45-11eb-0085-5ba014db07f0
# ╠═86e9ce8c-1d48-11eb-0354-1f1c27728e0e
# ╠═4fa2b18e-1d50-11eb-169e-4777411537c1
# ╠═bc1b6958-1d4f-11eb-2f77-0d992418be42
