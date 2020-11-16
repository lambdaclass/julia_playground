### A Pluto.jl notebook ###
# v0.11.14

using Markdown
using InteractiveUtils

# ╔═╡ 4828c18e-2848-11eb-1ca1-37b2524cba9f
using DifferentialEquations, Plots, DiffEqSensitivity, StatsPlots

# ╔═╡ d7a65e08-2851-11eb-0b16-5135864960e4
using CSV

# ╔═╡ 5e4becd6-2855-11eb-3c87-cdcf7b5a849d
using Turing, MCMCChains

# ╔═╡ 74cc60be-2821-11eb-3ad7-e581052437bd
md"""# The Ultima Online Catastrophe

[Ultima online](https://uo.com/) is a fantasy massively multiplayer online role-playing game (MMORPG) created by [Richard Garriott](https://en.wikipedia.org/wiki/Richard_Garriott) between 1995 and 1997, when it was realesed.

The game consisted of a medieval fantasy world in which each player could build their own character. What was interesting and disruptive was that all players interacted with each other, and what one did had repercussions on the general map. So, the game was "live" in the sense that if two people were fighting in one area and another one came in, the latter could see that fight. Also, the warriors had to hunt and look for resources to get points and improve their skills and again, if a treasure was discovered, or an animal was hunted, it would no longer be available for the rest.

During the development process of the game, Garriott and his team realized that, due to the massiveness of the game, there was going to be a moment when they would not be able to create content at the same speed as the players were consuming it. So they decided to automate the process.

After a lot of work, one of the ideas they came up with was to generate a "Virtual Ecosystem". This was a really incredible idea in which a whole ecosystem in harmony was simulated inside the game. For example, if an area started to grow grass, the herbivorous animals would come and start eating it. If many animals arrived, they would surely end up eating all the food and would have to go look for other places, it could even happen that some of them were not lucky and died on the way, starving.  In the same way, the carnivorous animals (that is, the predators of the herbivores) would strive to hunt as many animals as they could, but if in doing so they killed a significant number of them, food would become scarce causing them to die as well. In this way, as in "real" nature, a beautiful balance was generated.

But how this was even possible?

### The Lotka-Volterra model for population dynamics

To begin to understand how this complex ecosystem was created, it makes sense to go to the roots and study a dynamic system in which the interaction between a prey and a predatory population is modeled.

The idea is the same as we mentioned above. In this case, we will have a "prey" population that will have such a reproduction rate that:
"""

# ╔═╡ 39dd435a-2840-11eb-02af-cd234775c966
md"$Pray Population_{t+1} \sim PrayPopulation_{t} * BirthRate$"

# ╔═╡ 88ea0418-2840-11eb-21c7-99f98c018463
md"And a mortality rate that will be also affected by the prey population"

# ╔═╡ 23d5b9a6-2841-11eb-107f-8f9b1a0a24a4
md"$Pray Population_{t+1} \sim PrayPopulation_{t}* MortalityRate$"

# ╔═╡ 3dc55be0-2842-11eb-0b8d-6ba99bfe4d4e
md"So, calling PrayPopulation as $Pray$, Birthrate as $b_{pray}$ and Mortality rate as $m_{pray}$ for simplicity, we can write this as:"

# ╔═╡ 488ff80a-2842-11eb-216a-71ec0e4c81c6
md"$\frac{dPray}{dt} = Pray*(b_{pray} - m_{pray})$"

# ╔═╡ dbf274d6-2844-11eb-2cb3-f78a78e14d92
md"The population at time *t* multiplies at both rates because if the population is zero there can be no births or deaths. This leads us to the simplest ecological model, in which per capita growth is the difference between the birth rate and the mortality rate. 

But the model we are looking for have to explain the *interaction* between the two species. To do so, we must include the Pradator Population in order to modify the mortality rate of the Pray, leaving us with:
"

# ╔═╡ e0f34b58-2845-11eb-0480-9d200fc79403
md"$\frac{dPray}{dt} = Pray*(b_{pray} - m_{pray}*Pred)$"

# ╔═╡ 0da79014-2846-11eb-22b1-ed2430951f98
md"In a similar way we can think the interaction on the side of the predator, were the mortality rate would be constant and the birth rate will depend upon the Pray population:"

# ╔═╡ cd219e94-2846-11eb-294b-53ed40d7f36d
md"$\frac{dPred}{dt} = Pred*(b_{pred}*Pray - m_{pred})$"

# ╔═╡ 30e52eac-2847-11eb-3f7c-3b48e2baf012
md"In this way we obtain the Lotka-Volterra model in which the population dynamics is determined by a system of coupled ordinal differential equations (ODEs). This is a very powerful model which tells us that two hunter-prey populations in equilibrium will oscillate without finding a stable value. In order to see it, we will have to use some [SciML](https://sciml.ai/) libraries"

# ╔═╡ 2afd1060-2848-11eb-067a-1f378247d404
md"#### SciML to simulate population dynamics"

# ╔═╡ 5da9eb00-2848-11eb-0dfa-17ca0490bd44
#Let´s define our Lotka-Volterra model

function lotka_volterra(du,u,p,t)
  pray, pred  = u
  birth_pray, mort_pray, birth_pred, mort_pred = p
  du[1] = dpray = (birth_pray - mort_pray * pred)*pray
  du[2] = dpred = (birth_pred * pray - mort_pred)*pred
end

# ╔═╡ 4fee10b2-2849-11eb-1d6e-e7dfd2125e64
begin
	#And make explicit our example parameters, initial value and define our problem
	p = [1.1, 0.5, 0.1, 0.2]
	u0 = [1,1]
	prob = ODEProblem(lotka_volterra,u0,(0.0,40.0),p)
end;

# ╔═╡ 7efce850-284a-11eb-144a-cdf4794703bd
sol = solve(prob,Tsit5());

# ╔═╡ a259fb62-284a-11eb-132c-9f101f8e2f2a
plot(sol)

# ╔═╡ 4aff3714-284b-11eb-2047-814ca175e07b
md"""#### Obtaining the model from the data

Back to the terrible case of Ultima Online. Suppose we had data on the population of predators and prey that were in harmony during the game at a given time. If we wanted to venture out and analyze what parameters Garriot and his team used to model their great ecosystem, would it be possible? Of course it is, we just need to add a little Bayesianism.
"""

# ╔═╡ e3da565c-2851-11eb-0fa7-a3cd80bb9d63
data = CSV.read("ultima_online_data.csv")

# ╔═╡ 7c5509d4-2852-11eb-1888-5b9f04ff83a5
ultima_online = Array(data)'

# ╔═╡ 9b4ba8f6-2852-11eb-0cbd-19d36ac63a62
md"Probably seeing this data in a table is not being very enlightening, let's plot it and see if it makes sense with what we have been discussing:"

# ╔═╡ 012c3bc0-2853-11eb-0b54-e728d12af1a0
begin
	time = collect(1:16)
	scatter(time, ultima_online[1, :], label="Pray")
	scatter!(time, ultima_online[2, :], label="Pred")
end

# ╔═╡ 6585ddce-2853-11eb-1c57-99c9afbdb616
md"Can you spot the pattern? Let's connect the dots to make our work easier"

# ╔═╡ 99dc5b7a-2853-11eb-2fbe-03a4ccd858fa
begin
	plot(time, ultima_online[1,:], label="Pray")
	plot!(time, ultima_online[2, :], label="Pred")
end

# ╔═╡ 067e3d8e-2854-11eb-0bbe-21361e908a3a
md"Well, this already looks much nicer, but could you venture to say what are the parameters that govern it? 
This task that seems impossible a priori, is easily achievable with the SciML engine:"

# ╔═╡ 6b783782-2857-11eb-2f38-371de1067304
u_init = [1,1]

# ╔═╡ d96267ee-2856-11eb-0cc3-4bdba4e5e8e4
@model function fitlv(data)
    σ ~ InverseGamma(2, 3)
    birth_pray ~ truncated(Normal(1,0.5),0,2)
    mort_pray ~ truncated(Normal(1,0.5),0,2)
    birth_pred ~ truncated(Normal(1,0.5),0,2)
    mort_pred ~ truncated(Normal(1,0.5),0,2)

    k = [birth_pray, mort_pray, birth_pred, mort_pred]
    prob = ODEProblem(lotka_volterra,u_init,(0.0,30),k)
    predicted = solve(prob,Tsit5(),saveat=2)

    for i = 1:length(predicted)
        data[:,i] ~ MvNormal(predicted[i], σ)
    end
end

# ╔═╡ 30c1adee-2857-11eb-04de-6fab46e4754c
model = fitlv(ultima_online)

# ╔═╡ 46f9b992-2857-11eb-2682-6d5ad3200adc
chain = sample(model, NUTS(.6),10000); #p = [0.8, 0.4, 0.2, 0.4]

# ╔═╡ 3c97037a-2858-11eb-206a-57dc08cde44f
plot(chain)

# ╔═╡ Cell order:
# ╟─74cc60be-2821-11eb-3ad7-e581052437bd
# ╟─39dd435a-2840-11eb-02af-cd234775c966
# ╟─88ea0418-2840-11eb-21c7-99f98c018463
# ╟─23d5b9a6-2841-11eb-107f-8f9b1a0a24a4
# ╟─3dc55be0-2842-11eb-0b8d-6ba99bfe4d4e
# ╟─488ff80a-2842-11eb-216a-71ec0e4c81c6
# ╟─dbf274d6-2844-11eb-2cb3-f78a78e14d92
# ╟─e0f34b58-2845-11eb-0480-9d200fc79403
# ╟─0da79014-2846-11eb-22b1-ed2430951f98
# ╟─cd219e94-2846-11eb-294b-53ed40d7f36d
# ╟─30e52eac-2847-11eb-3f7c-3b48e2baf012
# ╟─2afd1060-2848-11eb-067a-1f378247d404
# ╠═4828c18e-2848-11eb-1ca1-37b2524cba9f
# ╠═5da9eb00-2848-11eb-0dfa-17ca0490bd44
# ╠═4fee10b2-2849-11eb-1d6e-e7dfd2125e64
# ╠═7efce850-284a-11eb-144a-cdf4794703bd
# ╠═a259fb62-284a-11eb-132c-9f101f8e2f2a
# ╟─4aff3714-284b-11eb-2047-814ca175e07b
# ╠═d7a65e08-2851-11eb-0b16-5135864960e4
# ╠═e3da565c-2851-11eb-0fa7-a3cd80bb9d63
# ╠═7c5509d4-2852-11eb-1888-5b9f04ff83a5
# ╟─9b4ba8f6-2852-11eb-0cbd-19d36ac63a62
# ╠═012c3bc0-2853-11eb-0b54-e728d12af1a0
# ╟─6585ddce-2853-11eb-1c57-99c9afbdb616
# ╠═99dc5b7a-2853-11eb-2fbe-03a4ccd858fa
# ╟─067e3d8e-2854-11eb-0bbe-21361e908a3a
# ╠═5e4becd6-2855-11eb-3c87-cdcf7b5a849d
# ╠═6b783782-2857-11eb-2f38-371de1067304
# ╠═d96267ee-2856-11eb-0cc3-4bdba4e5e8e4
# ╠═30c1adee-2857-11eb-04de-6fab46e4754c
# ╠═46f9b992-2857-11eb-2682-6d5ad3200adc
# ╠═3c97037a-2858-11eb-206a-57dc08cde44f
