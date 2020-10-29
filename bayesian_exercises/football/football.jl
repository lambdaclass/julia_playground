### A Pluto.jl notebook ###
# v0.11.14

using Markdown
using InteractiveUtils

# ╔═╡ 9608129c-1894-11eb-2bac-add686fd5bd0
begin
	using CSV
	using JSON
	using DataFrames
end

# ╔═╡ ca1d5908-194e-11eb-3bfb-8b46d16c8d77
using Turing

# ╔═╡ 05066a88-1a1e-11eb-1cfa-5bd84f5e42b4
using Plots

# ╔═╡ 3b991b52-1895-11eb-32b5-a700f9f21232
liga_espaniola = JSON.parsefile("matches_Spain.json")

# ╔═╡ 15838eec-1931-11eb-199b-0f91d2b40179
liga_espaniola[1]

# ╔═╡ 107709f6-189b-11eb-3be8-1db3ac974d3a
liga_espaniola[1]["label"]

# ╔═╡ 72a3b0a8-1a0c-11eb-1bf7-5fbbd6e67d6b
liga_espaniola[150]

# ╔═╡ d44fb2a2-189f-11eb-22af-093de9c8011a
md"Team data tiene como key el id de cada equipo"

# ╔═╡ 1e4e952a-1934-11eb-3e1f-ed18cc19d0c1
matches_df = DataFrame(home = [], away = [], score_home = [], score_away = []);

# ╔═╡ c0bb92d4-1934-11eb-3fad-3da61a28bc85
begin
	matches = []
	for match in liga_espaniola
		push!(matches, split(match["label"], ","))
end
end

# ╔═╡ 05b35962-1944-11eb-3484-fb50d0cce9a6
begin
for match in matches
	home, away = split(match[1], " - ")
	score_home, score_away = split(match[2], " - ")
	
	push!(matches_df,[home, away, parse(Int,score_home), parse(Int,score_away)])
end
end	

# ╔═╡ 29eb0736-1948-11eb-183f-23f196302a2b
matches_df

# ╔═╡ b8a2c19a-1a14-11eb-175e-79fba41adc8a
sum(matches_df[3])

# ╔═╡ c247a972-1a14-11eb-2043-19276903ba70
sum(matches_df[4][1])

# ╔═╡ 6cfc102e-1a15-11eb-0d69-a9bdc1749976

function sum_(n, data)
	a = 0
	h = 0
	e = 0
	for x in 1:n
		if data[4][x] > data[3][x]
			a = a + 1
		elseif data[4][x] == data[3][x]
			e = e + 1
		else
			h = h + 1
		end
	end
	return h, e, a
end

	

# ╔═╡ 07889450-1a18-11eb-39e7-c7d8904abdd7
suma = sum_(length(matches_df[4]), matches_df)

# ╔═╡ a4145910-1a18-11eb-082a-cd25b214bdad
sum(suma)

# ╔═╡ a40c3942-194b-11eb-10b1-25f8a49a93ef
teams = unique(collect(matches_df[1]))

# ╔═╡ a307000e-194b-11eb-092b-279f780709b8
team_index = collect(1:length(teams))

# ╔═╡ 66a6465a-194c-11eb-1e45-b3fe03f40171
teams_df = DataFrame(team = teams, idex = team_index)

# ╔═╡ 0672caec-1967-11eb-1677-b38e4c8ee5ac
teams_df[1]

# ╔═╡ 3a5e25a2-194e-11eb-2c3a-9923f9a2b1c2
md"### The model"

# ╔═╡ 0ee26c98-194f-11eb-0501-a71c88843202
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

# ╔═╡ dd00824c-1966-11eb-1402-ff7a30eff65f
model = football_matches(matches_df[1], matches_df[2], matches_df[3], matches_df[4], teams_df[1])

# ╔═╡ 1b09a46a-1967-11eb-0a2d-93dd939189c7
posterior = sample(model, NUTS(),1000)

# ╔═╡ 902001e6-1975-11eb-0e1e-c1360515d706
post_att = collect(get(posterior, :att_)[1]);

# ╔═╡ 5f75456c-1a25-11eb-228c-1fade2114a56
post_def = collect(get(posterior, :deff_)[1]);

# ╔═╡ b0f4347c-1a11-11eb-063c-016fb54dde48
barca_μ = mean(post_att[1])

# ╔═╡ 4480f3f6-1a21-11eb-263b-293e50f26ca6
barca_σ = std(post_att[1])

# ╔═╡ 28f9250a-1a25-11eb-2dd5-7313ae3e9724
atletico_μ = mean(post_att[2])

# ╔═╡ 416a2616-1a25-11eb-1a6a-955848a0d08e
atletico_σ = std(post_att[2])

# ╔═╡ 6e920f6a-1a0b-11eb-24f2-a7df6b1d66c8
post_home = collect(get(posterior, :home));

# ╔═╡ 844291d6-1a0b-11eb-03a1-df08448409c4
median(post_home[1])

# ╔═╡ b1142e36-1a0b-11eb-2942-cf8c6eefdfe7
post_μatt = collect(get(posterior, :μatt));

# ╔═╡ 57cc73b4-1a0c-11eb-1a35-25ee25f24b71
mean(post_μatt[1])

# ╔═╡ d87c3544-1a1d-11eb-2075-41b0fbd1a7f0
histogram(post_μatt[1], nbins=20, legend=false)

# ╔═╡ 8eb9d722-1a11-11eb-2815-fd2c2b01e89e
post_σatt = collect(get(posterior, :σatt))

# ╔═╡ 2c21e722-1a26-11eb-2a78-17912a04deb0
begin
a = 0.0
b = randn(10)
med_b = median(b)
max_b = maximum(b)
min_b = minimum(b)
scatter((a, med_b),
		yerror=[(med_b-min_b,
				max_b-med_b)])
end

# ╔═╡ Cell order:
# ╠═9608129c-1894-11eb-2bac-add686fd5bd0
# ╠═3b991b52-1895-11eb-32b5-a700f9f21232
# ╠═15838eec-1931-11eb-199b-0f91d2b40179
# ╠═107709f6-189b-11eb-3be8-1db3ac974d3a
# ╠═72a3b0a8-1a0c-11eb-1bf7-5fbbd6e67d6b
# ╟─d44fb2a2-189f-11eb-22af-093de9c8011a
# ╠═1e4e952a-1934-11eb-3e1f-ed18cc19d0c1
# ╠═c0bb92d4-1934-11eb-3fad-3da61a28bc85
# ╠═05b35962-1944-11eb-3484-fb50d0cce9a6
# ╠═29eb0736-1948-11eb-183f-23f196302a2b
# ╠═b8a2c19a-1a14-11eb-175e-79fba41adc8a
# ╠═c247a972-1a14-11eb-2043-19276903ba70
# ╠═6cfc102e-1a15-11eb-0d69-a9bdc1749976
# ╠═07889450-1a18-11eb-39e7-c7d8904abdd7
# ╠═a4145910-1a18-11eb-082a-cd25b214bdad
# ╠═a40c3942-194b-11eb-10b1-25f8a49a93ef
# ╠═a307000e-194b-11eb-092b-279f780709b8
# ╠═66a6465a-194c-11eb-1e45-b3fe03f40171
# ╠═0672caec-1967-11eb-1677-b38e4c8ee5ac
# ╟─3a5e25a2-194e-11eb-2c3a-9923f9a2b1c2
# ╠═ca1d5908-194e-11eb-3bfb-8b46d16c8d77
# ╠═0ee26c98-194f-11eb-0501-a71c88843202
# ╠═dd00824c-1966-11eb-1402-ff7a30eff65f
# ╠═1b09a46a-1967-11eb-0a2d-93dd939189c7
# ╠═902001e6-1975-11eb-0e1e-c1360515d706
# ╠═5f75456c-1a25-11eb-228c-1fade2114a56
# ╠═b0f4347c-1a11-11eb-063c-016fb54dde48
# ╠═4480f3f6-1a21-11eb-263b-293e50f26ca6
# ╠═28f9250a-1a25-11eb-2dd5-7313ae3e9724
# ╠═416a2616-1a25-11eb-1a6a-955848a0d08e
# ╠═6e920f6a-1a0b-11eb-24f2-a7df6b1d66c8
# ╠═844291d6-1a0b-11eb-03a1-df08448409c4
# ╠═b1142e36-1a0b-11eb-2942-cf8c6eefdfe7
# ╠═57cc73b4-1a0c-11eb-1a35-25ee25f24b71
# ╠═05066a88-1a1e-11eb-1cfa-5bd84f5e42b4
# ╠═d87c3544-1a1d-11eb-2075-41b0fbd1a7f0
# ╠═8eb9d722-1a11-11eb-2815-fd2c2b01e89e
# ╠═2c21e722-1a26-11eb-2a78-17912a04deb0
