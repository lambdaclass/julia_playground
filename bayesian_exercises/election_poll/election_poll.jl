### A Pluto.jl notebook ###
# v0.11.12

using Markdown
using InteractiveUtils

# ╔═╡ 036183d2-f775-11ea-0976-eb09b0c00f66
begin
	using Soss
	using Distributions
	using Plots
end

# ╔═╡ 5e231d96-f778-11ea-3c30-f134d4b4b537
md"## Using Soss.jl to analize an election poll"

# ╔═╡ 87ccbd50-f778-11ea-3451-97f4184db99d
md"We have data about the preferences of 1,447 likely voters:

Bush: 727
Dukakis: 583
Other: 137
"

# ╔═╡ f1fcfc00-f778-11ea-322a-95b2444823ff
begin
	data = [727,583,137]
	n = sum(data)
	k = length(data)
	p = data ./n
end;

# ╔═╡ 02721cc8-f781-11ea-0894-67ee2caeea30
n

# ╔═╡ 25630eb6-f779-11ea-0779-1f36f318865f
md"Modelating this data as a Multinomial Random Variable, with each probability of success as a Dirichlet distribution.

The dirichlet distribution can be seen as a generalization of the beta distribution. Initializing it as a non-informative uniform, we have:" 

# ╔═╡ d45aeb32-f779-11ea-39cd-dd159722ab98
m = @model begin
	a = ones(k)
	theta ~ Dirichlet(a) 
	diff_bush_Dukakis = theta[1] - theta[2]
	
	x ~ Multinomial(n, theta) 
end;

# ╔═╡ e6185a68-f77d-11ea-39e3-87e57ebf2022
post_prob = dynamicHMC(m(k=k,n=n), (x=data,))

# ╔═╡ bdb22550-f781-11ea-1623-437b271a730c
begin
	post_bush = [i.theta[1] for i in post_prob]
	post_dukakis = [i.theta[2] for i in post_prob]
end;

# ╔═╡ e27b4696-f783-11ea-26ac-010e5dd402a7
histogram(post_bush, title = "Posterior probability of Bush wining", label = false)

# ╔═╡ 272e63ca-f784-11ea-069a-3fb3c8fe96a9
histogram(post_dukakis, title = "Posterior probability of Dukakis wining", label = false)

# ╔═╡ 4ff6cfea-f784-11ea-09cd-bfe2241233ee
post_diff_bush_dukakis = [i.theta[1] - i.theta[2] for i in post_prob]

# ╔═╡ 6c1a7960-f784-11ea-296a-41c248a97b37
histogram(post_diff_bush_dukakis, title = "Posterior difference between Bush and Dukakis", label = false)

# ╔═╡ 83a261c2-f786-11ea-0b18-e5cf47acb83f
md"As we can see, Bush is likely to win"

# ╔═╡ 0f0a3e34-f786-11ea-0af4-1389804e8d9f
md"### Comparison of two multinomial observations

Now we are have the the data of two surveys, one before and one after a debate. We want to analize if was a shift towards Bush"

# ╔═╡ 45ba50a4-f786-11ea-3724-2b3d194b0845
data_2 = [[294, 307,  38],[288, 332,  10]]

# ╔═╡ 98b61c42-f787-11ea-2746-b1d9cb6beaae
begin
	n_1 = sum(data_2[1])
	m_1 = sum(data_2[1][1:2])
end

# ╔═╡ c745129a-f787-11ea-21ab-8911b378225e
begin
	n_2 = sum(data_2[2])
	m_2 = sum(data_2[2][1:2])
end

# ╔═╡ d9ecadc0-f787-11ea-3bb4-9592606e20e0
md"""As we can see, we have different number of samples.
We are going to use "m" to normalize across supporters for the 2 major candidates"""

# ╔═╡ e7dfaaea-f787-11ea-14a7-fb6004bf3aca
m1 = @model begin
	b = ones(k)
	theta_1 ~ Dirichlet(b)
	theta_2 ~ Dirichlet(b)
	
	x1 ~ Multinomial(n_1, theta_1)
	x2 ~ Multinomial(n_2, theta_2)
	
end;

# ╔═╡ ea021da8-f787-11ea-3969-ebcf6a76fbfd
post_prob_1 = dynamicHMC(m1(k=k,n_1=n_1, n_2=n_2), (x1=data_2[1],x2=data_2[2]));

# ╔═╡ e9e8ce70-f787-11ea-17f2-211cd57e10b0
md"As we say, we are interested in know if was an shift towars Bush"

# ╔═╡ e9d16ed0-f787-11ea-3e06-7994f118c89a
begin
	supp_bush_1 = [i.theta_1[1] * (n_1/m_1) for i in post_prob_1]
	supp_bush_2 = [i.theta_2[1] * (n_2/m_2) for i in post_prob_1]
end;

# ╔═╡ e9338f24-f787-11ea-1869-3bed81d9eea8
begin
	alph = 0.4
	p1 = histogram(supp_bush_1, title = "Bush support before and after the debate" ,label = "Before debate", alpha = alph, nbins = 25)
	histogram!(supp_bush_2, label = "After debate", alpha = alph, nbins = 25)
	
end

# ╔═╡ e91a04d2-f787-11ea-3a2d-336665d988db
shift = [supp_bush_2[i] - supp_bush_1[i] for i in 1:length(supp_bush_1)]

# ╔═╡ 67f098e8-f78d-11ea-28de-cd1f18782775
begin
	histogram(shift, label = false, title = "Shift in Bush supporters after debate")
	vline!((0,0), line = (:orange, 3), label="Cero shift")
end

# ╔═╡ 55dea8b0-f78e-11ea-0c1c-bbfbdc07bb95
mean(supp_bush_1)

# ╔═╡ 80868a64-f78f-11ea-3814-2b7c232a553c
mean(supp_bush_2)

# ╔═╡ ca0b6664-f78f-11ea-16fc-7796fccae5a9


# ╔═╡ Cell order:
# ╠═036183d2-f775-11ea-0976-eb09b0c00f66
# ╟─5e231d96-f778-11ea-3c30-f134d4b4b537
# ╟─87ccbd50-f778-11ea-3451-97f4184db99d
# ╠═f1fcfc00-f778-11ea-322a-95b2444823ff
# ╠═02721cc8-f781-11ea-0894-67ee2caeea30
# ╟─25630eb6-f779-11ea-0779-1f36f318865f
# ╠═d45aeb32-f779-11ea-39cd-dd159722ab98
# ╠═e6185a68-f77d-11ea-39e3-87e57ebf2022
# ╠═bdb22550-f781-11ea-1623-437b271a730c
# ╠═e27b4696-f783-11ea-26ac-010e5dd402a7
# ╠═272e63ca-f784-11ea-069a-3fb3c8fe96a9
# ╠═4ff6cfea-f784-11ea-09cd-bfe2241233ee
# ╠═6c1a7960-f784-11ea-296a-41c248a97b37
# ╟─83a261c2-f786-11ea-0b18-e5cf47acb83f
# ╟─0f0a3e34-f786-11ea-0af4-1389804e8d9f
# ╠═45ba50a4-f786-11ea-3724-2b3d194b0845
# ╠═98b61c42-f787-11ea-2746-b1d9cb6beaae
# ╠═c745129a-f787-11ea-21ab-8911b378225e
# ╟─d9ecadc0-f787-11ea-3bb4-9592606e20e0
# ╠═e7dfaaea-f787-11ea-14a7-fb6004bf3aca
# ╠═ea021da8-f787-11ea-3969-ebcf6a76fbfd
# ╟─e9e8ce70-f787-11ea-17f2-211cd57e10b0
# ╠═e9d16ed0-f787-11ea-3e06-7994f118c89a
# ╠═e9338f24-f787-11ea-1869-3bed81d9eea8
# ╠═e91a04d2-f787-11ea-3a2d-336665d988db
# ╠═67f098e8-f78d-11ea-28de-cd1f18782775
# ╠═55dea8b0-f78e-11ea-0c1c-bbfbdc07bb95
# ╠═80868a64-f78f-11ea-3814-2b7c232a553c
# ╠═ca0b6664-f78f-11ea-16fc-7796fccae5a9
