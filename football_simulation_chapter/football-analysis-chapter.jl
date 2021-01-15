### A Pluto.jl notebook ###
# v0.12.18

using Markdown
using InteractiveUtils

# ╔═╡ a4f0a914-1ec9-11eb-3c22-db87a9776aaa
begin
	using JSON
	using DataFrames
end

# ╔═╡ 5f83510e-1fc5-11eb-3c40-83a844bf2907
using Turing

# ╔═╡ 2414104e-1d41-11eb-37cb-9b5cee78a98b
md"## Creating our conjectures"

# ╔═╡ 3b7f8a1e-1d42-11eb-1ab1-c11fda416e80
md"""If there is one thing we can all agree on, it is that the reality in which we live is complex. The explanation for the things we usually see, and have naturalized in our daily lives, is usually quite complex and requires abstraction from what "is simply seen".

In order to give an explanation and gain a deeper understanding of the things we see, we tend to generate models that seek to explain them in a simple and generalized way. In this way we can reduce the noise of our observations to general rules that "govern" them.

For example, it is obvious to everyone that if we push a glass it will move in the same direction as we did. We also know that if we keep pushing it and it goes beyond the limits of the table, it will fall to the floor. But one thing is to have the intuition of what's going to happen, and another is to have an understanding of the laws that govern that movement. In this case, they are the Newton´s Law´s of motion:

"""

# ╔═╡ 4e4385b4-203b-11eb-0c5b-cbe41156401b
md"$ \vec{F}^{\} = m*\vec{a}^{\} $"

# ╔═╡ 2dca2f42-203b-11eb-2cb3-0d6a3823cdcd
md"""In this way, and with only one formula, it is possible to gain an understanding that is generalizable to many aspects of reality. 

#### Observable variables vs Latent variables

Now, it is worth noting that in this case all the variables that make up our model are observable. This means that they can be measured directly.

In the case of a glass, we could weigh it with a scale. Then, by pushing it, we could measure the acceleration it acquired and from these two measurements we could obtain the force we applied to it. So, every parameter of the model is fully defined.

However, as we try to advance in our understanding of reality, we arrive at more and more complex models and many times we are not so lucky to be able to define them with simple observable variables.

For example, this is very common in the economic sciences, where models are created with variables such as "quality of life". Economists will try to measure this latent variable with other variables that can be observed (such as quality of life, schooling rate, number of hospitals for a certain number of inhabitants, etc), but that do not have an obvious and direct relationship as if they had newton's equations.

This type of latent variables are used in different models to gain greater abstraction and to be able to obtain information that a priori is not found at first sight in the data. For example, in the case of economics, from concrete measures of a country's economy it is possible to generalize knowledge and be able to infer an abstract variable such as quality of life.

### Bayesian hierarchical models

The Bayesian framework allows us to build statistical models that can generalize the information obtained from the data and make inferences from latent variables. 

A nice way to think about this kind of models is that they allow us to build our "story" about which are the variables that generate the data we are observing. Basically, they allow us to increase the "depth" of our model by indicating that the parameters of our prior distributions also follow other probability distributions.

This sure is sounding very strange. Don't worry, let's move on to an example to clarify it.

#### Football analysis

Let's imagine for a moment that we are brilliant statisticians. We find ourselves looking for new interesting challenges to solve and we come across a sports bookmaker. They tell us that they want to expand into football betting and that they would like us to be able to build a model that allows them to analyze the strengths and weaknesses of English Premier League teams. They are interested because they want to be able to predict possible outcomes and thus be able to price the bets. 

The problem is that, as they have never worked in this sector before, they only have the results of the league matches. So what can we do? """

# ╔═╡ 0c0268dc-1ed3-11eb-2f73-b3706305b298
md"We have the data stored in a specific format called JSON, so the first thing to do is to parse and visualize it"

# ╔═╡ cbaaf72c-1eb4-11eb-3509-932b337f270b
begin
england_league = JSON.parsefile("matches_England.json")
matches_df = DataFrame(home = [], away = [], score_home = [], score_away = [])
end;

# ╔═╡ ab6fcb22-1ed2-11eb-2749-3bef16911972
begin
	matches = []
	for match in england_league
		push!(matches, split(match["label"], ","))
end
end

# ╔═╡ d2a81a14-1ed2-11eb-1307-bfb169aebbb4
begin
for match in matches
	home, away = split(match[1], " - ")
	score_home, score_away = split(match[2], " - ")
	
	push!(matches_df,[home, away, parse(Int,score_home), parse(Int,score_away)])
end
end	

# ╔═╡ dd80fc38-1ed2-11eb-0e8c-49859d27c72c
matches_df

# ╔═╡ 7f0c72d4-1f9e-11eb-06f1-cb587d7e5436
teams = unique(collect(matches_df[1]))

# ╔═╡ 868e5c8c-1ed3-11eb-291c-fb26d512c103
md"""So, we have the data of the 380 matches that were played in the Premier League 2017/2018 and our challenge is to be able to analyze the characteristics of these teams. 

A priori it may seem that we are missing data, that with the data we have we cannot infer "characteristics" specific to each team. At most, it might be possible to see who the teams that scored the most goals, the averages of goals per game or how the positions were after the tournament, but to obtain characteristics of the teams? how could we face this problem?

#### Creating our stories

Okay, let's see what information we have from our data: 
On one hand we have specified the names of each team and which one is local. On the other hand, we have the number of goals scored.

A possible approach to this data is to realize that the goals scored by each team can be modeled with a poisson distribution. 

Why? You have to remember that this distribution describes "arrivals" - discrete events - in a continuum. For example, it is widely used to describe customer arrivals to a location as time passes or failures in continuous industrial processes (e.g. failure in the production of a pipe). 

In this particular case, we could propose that the goals scored by a team are the discrete events that occur in the time continuum that the game last:

"""

# ╔═╡ 04a8f16c-1ed5-11eb-3979-af35c6d9dea1
md"$Score \sim Poisson(θ)$"

# ╔═╡ 9f1e1618-1edb-11eb-02af-9fa5c8904163
md"""Well, we have an improvement. We've already told our model how to think about goals scored. 

Now we can use the flexibility of Bayesianism to indicate what the "goal rate" of our Poisson depends on.  You can think of it literally as the number of goals a team scores per unit of time. And this is where we have to take advantage of all the information provided by the data set.

As expected, this rate has to be particular to each match the team plays and take into account the opponent. We can therefore propose that the scoring rate of each team (in each particular match) depends on the "attacking power" of the team on the one hand, and the "defensive power" of the opponent on the other:

"""

# ╔═╡ 42104900-1f80-11eb-019f-a7c561ca3e4b
md"$θ_{team1} \sim att_{team1} + def_{team2}$" 

# ╔═╡ 9d0f3a1e-1f80-11eb-3422-fb7b4696cc13
md"""In this way we could be capturing, from the results of each game, the attack and defence strengths of each team. 

Another latent variable that we could obtain, given the data, is if there is an effect that increases (or decreases) the goal rate related to whether the team is local or not. This would also help - in case there is indeed an effect - in not the attack and defence parameters be disrupted by having to "contain" that information."""

# ╔═╡ 0b1f9c68-1f83-11eb-11e7-e7614c021c05
md"$θ_{home} \sim home + att_{home} + def_{away}$
$θ_{away} \sim att_{away} + def_{home}$"

# ╔═╡ 043c438c-1f84-11eb-2cd7-6dd07a125910
md"""This leaves one attack and one defense parameter for each team, and a global league parameter that indicates the effect of being local on the scoring rate.

#### Letting the information flow 

Okay, we are already getting much closer to the initial goal we set. As a last step, we must be able to make the information flow between the two independent poissons that we proposed to model the score of each of the two teams that are playing. We need to do that precisely because we have proposed that the poissons are independent, but we need that when making the inference of the parameters the model can access the information from both scores so it can catch the correlation between them. In other words, we have to find a way to interconnect our model.

And that is exactly what hierarchical Bayesian models allow us to do. How? By letting us choose probability distributions for the parameters that represent the characteristics of both equipment. With the addition that these parameters will share the same prior distributions. Let's see how:

The first thing to do, as we already know, is to assign the prior distributions of our attack and defense parameters. A reasonable idea would be to propose that they follow a normal distribution since it is consistent that there are some teams that have a very good defense, so the parameter would take negative values; or there may be others that have a very bad one, taking positive values (since they would "add up" to the goal rate of the opposing team). The normal distribution allows us to contemplate both cases.

Now, when choosing the parameters we are not going to stop and assign fixed numbers, but we will continue to deepen the model and add another layer of distributions:

"""

# ╔═╡ 4a31541e-1f97-11eb-038e-736446224c21
md"$att_{t} \sim Normal(μ_{att}, σ_{att})$
$def_{t} \sim Normal(μ_{def}, σ_{def})$"

# ╔═╡ e3a99868-1f97-11eb-1872-79c8f6abe1be
md"Where the t sub-index is indicating us that there are a couple of these parameters for each team.

Then, as a last step to have our model defined, we have to assign the priority distributions that follow the parameters of each normal distribution. We have to define our hyper priors.

"

# ╔═╡ c3ca1cc4-1f98-11eb-3c23-33acb95fbc03
md"$μ_{att}, μ_{def} \sim Normal(0, 0.1)$
$σ_{att}, σ_{def} \sim Exponential(1)$"

# ╔═╡ 4106e55a-1f99-11eb-110e-b79221774f12
md"We must not forget the parameter that represents the advantage of being local"

# ╔═╡ bff4fdae-1f99-11eb-2d94-aba7bf9d097f
md"$home \sim Normal(0,1)$"

# ╔═╡ 2b135202-1fa7-11eb-324d-93a5eda41e3e
md"""Now that our model is fully define, let's add one last restriction to the characteristics of the teams to make it easier to compare them: subtract the average of all the attack and defence powers from each one. In this way we will have the features centred on zero, with negative values for the teams that have less attacking power than the average and positive values for those that have more. As we already said, the opposite analysis applies to the defence, negative values are the ones that will indicate that a team has a strong defence as they will be "subtracting" from the scoring rate of the opponent. This is equivalent to introducing the restriction:
"""

# ╔═╡ ddc7b166-203a-11eb-0ac2-056c309bb590
md"$\sum att_{t} = 0$
$\sum def_{t} = 0$"

# ╔═╡ 8e164be0-203b-11eb-293a-6b32dba96133
md"Let's translate all this into Turing code:"

# ╔═╡ 51a309bc-2033-11eb-10c0-ed17545df33d
begin
	@model function football_matches(home_teams, away_teams, score_home, score_away, teams)
	#hyper priors
	σatt ~ Exponential(1)
	σdeff ~ Exponential(1)
	μatt ~ Normal(0,0.1)
	μdef ~ Normal(0,0.1)
	
	home ~ Normal(0,1)
		
	#Team-specific effects	
	att ~ filldist(Normal(μatt, σatt), length(teams))
	def ~ filldist(Normal(μatt, σdeff), length(teams))
	
	dict = Dict{String, Int64}()
	for (i, team) in enumerate(teams)
		dict[team] = i
	end
		
	#Zero-sum constrains
	offset = mean(att) + mean(def)
	
	log_θ_home = Vector{Real}(undef, length(home_teams))
	log_θ_away = Vector{Real}(undef, length(home_teams))
		
	#Modeling score-rate and scores (as many as there were games in the league) 
	for i in 1:length(home_teams)
		#score-rate
		log_θ_home[i] = home + att[dict[home_teams[i]]] + def[dict[away_teams[i]]] - offset
		log_θ_away[i] = att[dict[away_teams[i]]] + def[dict[home_teams[i]]] - offset
		#scores
		score_home[i] ~ LogPoisson(log_θ_home[i])
		score_away[i] ~ LogPoisson(log_θ_away[i])
	end
	
	end
end

# ╔═╡ f3f0b6f8-2033-11eb-0f9e-d951d791001d
md"""As you can see, the turing code is very clear and direct. In the first block we define our hyperpriors for the distributions of the characteristics of the equipment.

In the second one, we define the priors distributions that will encapsulate the information about the attack and defense powers of the teams. With the *filldist* function we are telling Turing that we need as many of these parameters as there are teams in the league *length(teams)*

Then, we calculate the average of the defense and attack parameters that we are going to use to centralize those variables, and we use the LogPoisson distribution to allow the theta to take some negative value in the inference process and give more sensitivity to the parameters that make it up.

As we said before, we will model the thetas for each game played in the league, that's why the *for* of the last block goes from 1 to *length(home_teams)*, which is the list that contains who was the local team of each game played.

So let´s run it and see if all of this effort was worth it:
"""

# ╔═╡ f3f4bfba-203f-11eb-142c-a187112744d2
model = football_matches(matches_df[1], matches_df[2], matches_df[3], matches_df[4], teams)

# ╔═╡ 2ce4435c-2040-11eb-1670-e39ad4cc690c
posterior = sample(model, NUTS(),3000);

# ╔═╡ 9e50e56e-5674-11eb-2919-4bde8964bc59
rand(LogPoisson(0.2), 100)

# ╔═╡ 8c1dd9de-2040-11eb-35a3-39e8196577a1
md"#### Analyzing the results
In order to compare and corroborate that the inference of our model makes sense, it is key to have the ranking table of how the teams actually performed in the 2017/2018 Premier League.
"

# ╔═╡ 0f9406ae-2054-11eb-3f0c-1d03ba779f18
begin
	table_positions = 
	[11, 5, 9, 4, 13, 14, 1, 15, 12, 6, 2, 16, 10, 17, 20, 3, 7, 8, 19, 18]
	
	games_won = 
	[32, 25, 23, 21, 21, 19, 14, 13, 12, 12, 11, 11, 10, 11, 9, 9, 7, 8, 7, 6]
	
	teams_ = []
	for i in table_positions
		push!(teams_, teams[i])
	end
	
	table_position_df = DataFrame(Table_of_positions = teams_, Wins = games_won)

end

# ╔═╡ 099e4188-2054-11eb-1e7e-69bdb2b0202c
md"Let's now explore a little bit the a posteriori values we obtained."

# ╔═╡ 16572944-2045-11eb-38d0-fb20d981e3e9
begin
	post_att = collect(get(posterior, :att)[1])
	post_def = collect(get(posterior, :def)[1])
	post_home = collect(get(posterior, :home)[1])
end;

# ╔═╡ 4ae7d968-206f-11eb-1925-c5b5a31e6940
begin
using Plots
histogram(post_home, legend=false, title="Posterior distribution of home parameter")
end

# ╔═╡ 93d3bf98-5684-11eb-361e-c7b0f1a3bb0e
get(posterior, :att)[:att]

# ╔═╡ 18f36e1a-5682-11eb-2826-ab4d4c457af3
posterior.value.data

# ╔═╡ 6f096138-5682-11eb-179f-f973803b9268
collect(get(posterior, :att)[1])

# ╔═╡ 8cec9318-206e-11eb-3caa-3bad9d788d3c
md"As a first measure to analyze, it is interesting to see and quantify (if any) the effect that being local has on the score rate:"

# ╔═╡ 8b77d12c-206f-11eb-19a5-f557157ac05e
mean(post_home)

# ╔═╡ b56df7ea-206f-11eb-18b9-0544cc596ca5
md"So, to include in the model the parameter home was a good idea. indeed being local provides a very big advantage. 

Beyond the fact that it is interesting to be able to quantify how much the location influences the scoring rate of the teams, including it in the analysis allow us to have better estimates of the defense and attack parameters of the teams. This is true because if it had not been included, this positive effect would have manifested itself in the only parameters it would have found, the attack and defense parameters, deforming the real measure of these.

So, being confident that we are on the right track, let´s find the attack and defence parameters of each team."

# ╔═╡ 554b4ef0-2045-11eb-3218-050ec936f1aa
begin
	teams_att = []
	teams_def = []
	for i in 1:length(post_att)
		push!(teams_att, post_att[i])
		push!(teams_def, post_def[i])
	end
end

# ╔═╡ 8a4df438-2045-11eb-2798-af592dbbffb5
md"This way we obtain all the samples of the posterior distributions for each one of the parameters of each equipment. Scroll right to explore the entire array."

# ╔═╡ 6b1a1520-2045-11eb-1302-0778b4e7a836
teams_att

# ╔═╡ da93bf52-2045-11eb-2b6f-3f31f57d8653
md"For example, if we would like to se the posterior distribution of the attack parameter for Burnley:"

# ╔═╡ 1fa92570-2046-11eb-3541-e5fccf7afabd
teams[1]

# ╔═╡ 255c4d08-2046-11eb-1cf0-b32d2cd711bf
histogram(teams_att[1], legend=false, title="Posterior distribution of Burnley´s attack power")

# ╔═╡ 473e6a46-205b-11eb-2731-7b47b99d86a5
mean(teams_att[1])

# ╔═╡ 917d4550-2058-11eb-174b-65436a4e6cc8
md"Comparing it to the attacking power of Manchester City, champion of the Premier league:"

# ╔═╡ c2de4714-2058-11eb-1a0d-f1bf96649844
teams[11]

# ╔═╡ 0103406e-2059-11eb-333f-7b5814f21a54
begin
	histogram(teams_att[11], legend=false, title="Posterior distribution of Manchester City´s attack power")
end

# ╔═╡ 579a4d9e-205b-11eb-2352-83beca145211
mean(teams_att[11])

# ╔═╡ 6924f372-2059-11eb-2b98-6f4b65b777f4
md"Okay, when comparing the league champion against a mid-table team, we can clearly see the superiority in attack. For now, it seems that the inference comes in handy. 

Let's try now to have an overview of the attacking powers of each team. To do this, just take the average of each and plot it next to the standard deviation 
"

# ╔═╡ ece1d052-2060-11eb-2b50-ef46663ed88f
begin
	teams_att_μ = mean.(teams_att)
	teams_def_μ = mean.(teams_def)
	teams_att_σ = std.(teams_att)
	teams_def_σ = std.(teams_def)
end;

# ╔═╡ 43a39b30-2061-11eb-28c5-37cbc4aed84b
md"""Remember that the "." operator is used for broadcasting. This means that it will apply the function to each component of the array"""

# ╔═╡ 41ca2316-2062-11eb-3fe6-f33a789fd578
begin
	teams_att_μ
	sorted_att = sortperm(teams_att_μ)
	abbr_names = [t[1:3] for t in teams]
end;

# ╔═╡ 0ef8a16e-2063-11eb-1391-6b3b959de541
begin
	abbr_names[5] = "Mun"
	abbr_names[10] = "Whu"
	abbr_names[11] = "Mci"
	abbr_names[16] = "Bou"
	abbr_names[18] = "Wba"
	abbr_names[19] = "Stk"
	abbr_names[20] = "Bha"
end;

# ╔═╡ 1eb73246-2063-11eb-0f20-416e433b5144
sorted_names = abbr_names[sorted_att]

# ╔═╡ abace764-2062-11eb-2949-fd9e86b50f17
begin
	scatter(1:20, teams_att_μ[sorted_att], grid=false, legend=false, yerror=teams_att_σ[sorted_att], color=:blue, title="Premier league 17/18 teams attack power")
	annotate!(collect(1:20), teams_att_μ[sorted_att] .+ 0.238, text.(sorted_names, :black, :center, 8))
	ylabel!("Mean team attack")
end

# ╔═╡ ebd267a4-2064-11eb-02a7-b93465de6319
md"""Although there is a high correlation between the attacking power of each team and its position on the table after the league ends, it is clear that this is not enough to explain the results. For example, Manchester City was the league's sub-champion, but only appeared in fifth place.

Let's explore what happens to the defence power: """

# ╔═╡ f6ad5f52-2065-11eb-0be7-f159941e0d89
begin
	sorted_def = sortperm(teams_def_μ)
	sorted_names_def = abbr_names[sorted_def]
end

# ╔═╡ 539e68b4-2066-11eb-09c2-b337241c36bc
begin
	scatter(1:20, teams_def_μ[sorted_def], grid=false, legend=false, yerror=teams_def_σ[sorted_def], color=:blue, title="Premier league 17/18 teams defence power")
	annotate!(collect(1:20), teams_def_μ[sorted_def] .+ 0.2, text.(sorted_names_def, :black, :center, 8))
	ylabel!("Mean team defence")
end

# ╔═╡ 69e7e54c-2067-11eb-21a1-cb72f4b423cd
md"To read this graph we have to remember that the defense effect is better the more negative it is, since it is representing the scoring rate that takes away from the opponent team. As we already said:"

# ╔═╡ f59d488a-2067-11eb-1d4b-1d9dd97ea39e
md"$θ_{team1} \sim att_{team1} + def_{team2}$"

# ╔═╡ 27d5a290-2068-11eb-06f5-0b16f6ec6cc5
md"As the $def_{team2}$ is adding up in the equation, if it take negative values, it is going to start substracting the scoring rate of the oponent.

Things, then, begin to make a little more sense. Now we can see that Manchester United is the team with the strongest defence, so being second in the overall is not extrange.

To gain a deeper understanding of what´s going on here, let's chart both characteristics together. This is going to let us see the combined effect they have. Also i´m going to add the final position of each team to improve the interpretability."

# ╔═╡ b8f9bae8-206a-11eb-1426-031f6fd05fd6
begin
	table_position = 
	[11, 5, 9, 4, 13, 14, 1, 15, 12, 6, 2, 16, 10, 17, 20, 3, 7, 8, 19, 18]
	position = sortperm(table_position)
end

# ╔═╡ ee45d48e-206a-11eb-0edf-2b8b893bb583
begin
scatter(teams_att_μ, teams_def_μ, legend=false)
annotate!(teams_att_μ, teams_def_μ.+ 0.016, text.(abbr_names, :black, :center, 6))
annotate!(teams_att_μ, teams_def_μ.- 0.016, text.(position, :left, :center, 5))

xlabel!("Mean team attack")
ylabel!("Mean team defence")
end

# ╔═╡ c794f45e-206b-11eb-33c4-05bc429ba846
md"Well, great! Now we have some interesting information to analyze the teams and the league in general. It´s easier now to perceive how the two features interact with each other, comparing between teams and being able to see how that affects the final position. 

For example, looking at the cases of Liverpool and Tottenham, or Leicester City and Everton; one could say (against general common sense) that the power of defense has a greater effect on the performance of each team than the attack. But we leave you to do those analysis for the betting house.

Well, we went from having a problem that seemed almost impossible to have a solid solution, with a quantitative analysis of the characteristics of each team. We even know how much the localization of the teams increases the scoring rate. We were able to achieve this thanks to the hierarchical framework that Bayesianism provides us. Using this tool allows us to create models proposing latent variables that cannot be observed, to infer them and to gain a much deeper and more generalized knowledge than we had at first. You just have to imagine a good story."

# ╔═╡ fcb7e8d2-2071-11eb-020d-7bb1d53f8a6d
md"### Bibliography 

- [Paper of Gianluca Baio and Marta A. Blangiardo](https://discovery.ucl.ac.uk/id/eprint/16040/1/16040.pdf)
- [Post of Daniel Weitzenfeld](http://danielweitzenfeld.github.io/passtheroc/blog/2014/10/28/bayes-premier-league/)

"

# ╔═╡ fac413ec-5684-11eb-3167-d56d967247fc
mun_att_post = collect(get(posterior, :att)[:att])[5][:,1];

# ╔═╡ cc6681a2-5688-11eb-1d8e-e1edcb80a20d
mun_def_post = collect(get(posterior, :def)[:def])[5][:,1];

# ╔═╡ f01e7438-5688-11eb-2900-8df62e9feb18
liv_att_post = collect(get(posterior, :att)[:att])[4][:,1];

# ╔═╡ fae4e938-5688-11eb-3ab8-e3a8a4a61692
liv_def_post = collect(get(posterior, :def)[:def])[4][:,1];

# ╔═╡ 84d5e7a2-56dc-11eb-0d54-b790c7c714fb


# ╔═╡ 7704b9e6-56a0-11eb-0ade-2194a0845f61
post_home;

# ╔═╡ 03f063cc-5689-11eb-3d90-b7a5991332ba
# This function simulates matches given the attach, defense and home parameters.
# The first pair of parameters alwas correspond to the home team.
function simulate_matches_(att₁, def₁, att₂, def₂, home, n_matches, home_team=1)
	if home_team == 1
		logθ₁ = home + att₁ + def₂
		logθ₂ = att₂ + def₁
				
	elseif home_team == 2
		logθ₁ = att₁ + def₂
		logθ₂ = home + att₂ + def₁
	else
		return DomainError(home_team, "Invalid home_team value")
	end
	
	scores₁ = rand(LogPoisson(logθ₁), n_matches)
	scores₂ = rand(LogPoisson(logθ₂), n_matches)
	
	results = [(s₁, s₂) for (s₁, s₂) in zip(scores₁, scores₂)]
	
	return results
end

# ╔═╡ a8a9179c-5693-11eb-1758-e7d34722c7fa
mat = simulate_matches_(0.2, -0.1, 0.4, -0.3, 0.2, 1000, 2)

# ╔═╡ 7e82f15c-5695-11eb-20f1-85763cbfa572
max_h = maximum(map(x -> x[1], mat))

# ╔═╡ bd11306c-5697-11eb-22ed-99ae7703be59
max_a = maximum(map(x -> x[2], mat))

# ╔═╡ d99e52fa-5697-11eb-12fe-ab4ae36ab362
function simulate_matches(team1_att_post, team1_def_post, team2_att_post,				team2_def_post, home_post, n_matches)
	
	team1_as_home_results = Tuple{Int64,Int64}[]
	team2_as_home_results = Tuple{Int64,Int64}[]
	
	for (t1_att, t1_def, t2_att, t2_def, home) in zip(team1_att_post, team1_def_post, team2_att_post, team2_def_post, home_post)
		
		team1_as_home_results = vcat(team1_as_home_results, simulate_matches_(t1_att, t1_def, t2_att, t2_def, home, n_matches, 1))
		
		team2_as_home_results = vcat(team2_as_home_results, simulate_matches_(t1_att, t1_def, t2_att, t2_def, home, n_matches, 2))
	end
	
	return team1_as_home_results, team2_as_home_results
end

# ╔═╡ 5a3d6966-56a0-11eb-0db1-c5a06ea2dd83
kjj, kjjj = simulate_matches(mun_att_post, mun_def_post, liv_att_post, liv_def_post, post_home, 1000)

# ╔═╡ 9d1f1e7e-56a1-11eb-2539-f5da33710a32
function match_heatmaps(team1_as_home_results, team2_as_home_results)
	
	max_t1_as_home = maximum(map(x -> x[1], team1_as_home_results))
	max_t2_as_away = maximum(map(x -> x[2], team1_as_home_results))
	
	max_t1_as_away = maximum(map(x -> x[1], team2_as_home_results))
	max_t2_as_home = maximum(map(x -> x[2], team2_as_home_results))
	
	matrix_t1_as_home = zeros(Int64, (max_t1_as_home + 1, max_t2_as_away + 1))
	matrix_t2_as_home = zeros(Int64, (max_t1_as_away + 1, max_t2_as_home + 1))
	
	for match in team1_as_home_results
		matrix_t1_as_home[match[1] + 1, match[2] + 1] += 1
	end
	
	for match in team2_as_home_results
		matrix_t2_as_home[match[1] + 1, match[2] + 1] += 1
	end
	
	gr()
	#heat_t1_home = heatmap(0:max_t1_as_home,
	heatmap(0:max_t1_as_home,
								0:max_t2_as_away,
								matrix_t1_as_home,
								c=cgrad([:blue, :white,:red, :yellow]),
								xlabel="Team1 score", ylabel="Team2 score",
								title="Team1 as home")
	savefig("./t1_as_home")
	
	#heat_t2_home = heatmap(0:max_t1_as_away,
	heatmap(0:max_t1_as_away,
								0:max_t2_as_home,
								matrix_t2_as_home,
								c=cgrad([:blue, :white,:red, :yellow]),
								xlabel="Team1 Score", ylabel="Team2 Score",
								title="Team2 as home")
	savefig("./t2_as_home")
	#plot(heat_t1_home, heat_t2_home, layout=(2,1), size=(900, 400))
	#current()
	
	return matrix_t1_as_home, matrix_t2_as_home
end

# ╔═╡ d7bfc106-56d0-11eb-21b3-3b9036905041
m_1_home, m_t2_home = match_heatmaps(kjj, kjjj)

# ╔═╡ 9ceba0fa-56d4-11eb-2696-21f4babf699c
begin
	gr()
	heatmap(collect(0:(size(m_t2_home, 1)-1)),		
			collect(0:(size(m_t2_home, 2)-1)),	
			m_t2_home,
			c=cgrad([:blue, :white,:red, :yellow]),
			xlabel="Team1 score", ylabel="Team2 score",
			title="Team1 as home")
end

# ╔═╡ 0da9d3a0-56d5-11eb-08a9-c9e83106dfc3
m_t1_home

# ╔═╡ db8ed140-5735-11eb-2078-2f90d15d6325
"hola"

# ╔═╡ Cell order:
# ╟─2414104e-1d41-11eb-37cb-9b5cee78a98b
# ╟─3b7f8a1e-1d42-11eb-1ab1-c11fda416e80
# ╟─4e4385b4-203b-11eb-0c5b-cbe41156401b
# ╟─2dca2f42-203b-11eb-2cb3-0d6a3823cdcd
# ╠═a4f0a914-1ec9-11eb-3c22-db87a9776aaa
# ╠═0c0268dc-1ed3-11eb-2f73-b3706305b298
# ╠═cbaaf72c-1eb4-11eb-3509-932b337f270b
# ╠═ab6fcb22-1ed2-11eb-2749-3bef16911972
# ╠═d2a81a14-1ed2-11eb-1307-bfb169aebbb4
# ╠═dd80fc38-1ed2-11eb-0e8c-49859d27c72c
# ╠═7f0c72d4-1f9e-11eb-06f1-cb587d7e5436
# ╟─868e5c8c-1ed3-11eb-291c-fb26d512c103
# ╟─04a8f16c-1ed5-11eb-3979-af35c6d9dea1
# ╟─9f1e1618-1edb-11eb-02af-9fa5c8904163
# ╟─42104900-1f80-11eb-019f-a7c561ca3e4b
# ╟─9d0f3a1e-1f80-11eb-3422-fb7b4696cc13
# ╟─0b1f9c68-1f83-11eb-11e7-e7614c021c05
# ╟─043c438c-1f84-11eb-2cd7-6dd07a125910
# ╟─4a31541e-1f97-11eb-038e-736446224c21
# ╟─e3a99868-1f97-11eb-1872-79c8f6abe1be
# ╟─c3ca1cc4-1f98-11eb-3c23-33acb95fbc03
# ╟─4106e55a-1f99-11eb-110e-b79221774f12
# ╟─bff4fdae-1f99-11eb-2d94-aba7bf9d097f
# ╟─2b135202-1fa7-11eb-324d-93a5eda41e3e
# ╟─ddc7b166-203a-11eb-0ac2-056c309bb590
# ╟─8e164be0-203b-11eb-293a-6b32dba96133
# ╠═5f83510e-1fc5-11eb-3c40-83a844bf2907
# ╠═51a309bc-2033-11eb-10c0-ed17545df33d
# ╟─f3f0b6f8-2033-11eb-0f9e-d951d791001d
# ╠═f3f4bfba-203f-11eb-142c-a187112744d2
# ╠═2ce4435c-2040-11eb-1670-e39ad4cc690c
# ╠═9e50e56e-5674-11eb-2919-4bde8964bc59
# ╟─8c1dd9de-2040-11eb-35a3-39e8196577a1
# ╟─0f9406ae-2054-11eb-3f0c-1d03ba779f18
# ╟─099e4188-2054-11eb-1e7e-69bdb2b0202c
# ╠═16572944-2045-11eb-38d0-fb20d981e3e9
# ╠═93d3bf98-5684-11eb-361e-c7b0f1a3bb0e
# ╠═18f36e1a-5682-11eb-2826-ab4d4c457af3
# ╠═6f096138-5682-11eb-179f-f973803b9268
# ╟─8cec9318-206e-11eb-3caa-3bad9d788d3c
# ╠═4ae7d968-206f-11eb-1925-c5b5a31e6940
# ╠═8b77d12c-206f-11eb-19a5-f557157ac05e
# ╟─b56df7ea-206f-11eb-18b9-0544cc596ca5
# ╠═554b4ef0-2045-11eb-3218-050ec936f1aa
# ╟─8a4df438-2045-11eb-2798-af592dbbffb5
# ╠═6b1a1520-2045-11eb-1302-0778b4e7a836
# ╟─da93bf52-2045-11eb-2b6f-3f31f57d8653
# ╠═1fa92570-2046-11eb-3541-e5fccf7afabd
# ╠═255c4d08-2046-11eb-1cf0-b32d2cd711bf
# ╠═473e6a46-205b-11eb-2731-7b47b99d86a5
# ╟─917d4550-2058-11eb-174b-65436a4e6cc8
# ╠═c2de4714-2058-11eb-1a0d-f1bf96649844
# ╠═0103406e-2059-11eb-333f-7b5814f21a54
# ╠═579a4d9e-205b-11eb-2352-83beca145211
# ╟─6924f372-2059-11eb-2b98-6f4b65b777f4
# ╠═ece1d052-2060-11eb-2b50-ef46663ed88f
# ╟─43a39b30-2061-11eb-28c5-37cbc4aed84b
# ╠═41ca2316-2062-11eb-3fe6-f33a789fd578
# ╟─0ef8a16e-2063-11eb-1391-6b3b959de541
# ╠═1eb73246-2063-11eb-0f20-416e433b5144
# ╠═abace764-2062-11eb-2949-fd9e86b50f17
# ╟─ebd267a4-2064-11eb-02a7-b93465de6319
# ╠═f6ad5f52-2065-11eb-0be7-f159941e0d89
# ╠═539e68b4-2066-11eb-09c2-b337241c36bc
# ╟─69e7e54c-2067-11eb-21a1-cb72f4b423cd
# ╟─f59d488a-2067-11eb-1d4b-1d9dd97ea39e
# ╟─27d5a290-2068-11eb-06f5-0b16f6ec6cc5
# ╠═b8f9bae8-206a-11eb-1426-031f6fd05fd6
# ╠═ee45d48e-206a-11eb-0edf-2b8b893bb583
# ╟─c794f45e-206b-11eb-33c4-05bc429ba846
# ╟─fcb7e8d2-2071-11eb-020d-7bb1d53f8a6d
# ╠═fac413ec-5684-11eb-3167-d56d967247fc
# ╠═cc6681a2-5688-11eb-1d8e-e1edcb80a20d
# ╠═f01e7438-5688-11eb-2900-8df62e9feb18
# ╠═fae4e938-5688-11eb-3ab8-e3a8a4a61692
# ╠═84d5e7a2-56dc-11eb-0d54-b790c7c714fb
# ╠═7704b9e6-56a0-11eb-0ade-2194a0845f61
# ╠═03f063cc-5689-11eb-3d90-b7a5991332ba
# ╠═a8a9179c-5693-11eb-1758-e7d34722c7fa
# ╠═7e82f15c-5695-11eb-20f1-85763cbfa572
# ╠═bd11306c-5697-11eb-22ed-99ae7703be59
# ╠═d99e52fa-5697-11eb-12fe-ab4ae36ab362
# ╠═5a3d6966-56a0-11eb-0db1-c5a06ea2dd83
# ╠═9d1f1e7e-56a1-11eb-2539-f5da33710a32
# ╠═d7bfc106-56d0-11eb-21b3-3b9036905041
# ╠═9ceba0fa-56d4-11eb-2696-21f4babf699c
# ╠═0da9d3a0-56d5-11eb-08a9-c9e83106dfc3
# ╠═db8ed140-5735-11eb-2078-2f90d15d6325
