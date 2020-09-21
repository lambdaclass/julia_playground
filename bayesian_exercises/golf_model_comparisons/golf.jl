### A Pluto.jl notebook ###
# v0.11.14

using Markdown
using InteractiveUtils

# ╔═╡ 732bdf94-f6c8-11ea-35ae-490c5536996d
begin
	using Turing
	using HTTP
	using CSV
	using Plots
	using StatsFuns: logistic
	using StatsPlots
	using StatsBase
end

# ╔═╡ ce2eb27c-f6c8-11ea-02c8-d1dcefce145d
begin
	resp = HTTP.get("https://raw.githubusercontent.com/stan-dev/example-models/master/knitr/golf/golf_data.txt");

	data = CSV.read(IOBuffer(resp.body), normalizenames = true, header = 3)
	x, n, y = (data[:,1], data[:,2], data[:,3]);
end

# ╔═╡ 46428662-f6c9-11ea-3835-2dde7344ca40
begin
	pj = y ./ n
	error = @. sqrt(pj * (1 - pj) / n)
end

# ╔═╡ 925fb2d6-f6c9-11ea-245f-f33635531f50
scatter(x, pj, yerror=error, ylim=(0,1), xlabel="Distance from hole (ft)", ylabel="Probability of success", legend=false)

# ╔═╡ 205dc84c-f6cb-11ea-3f31-c18a34f651b1
begin
	@model golf_logistic(x, y, n, J) = begin
		# parameters
		a ~ Normal(0, 1)
		b ~ Normal(0, 1)
		
		for i in 1:J
			p = logistic(a + b * x[i])
			y[i] ~ Binomial(n[i], p)
		end
	end
end

# ╔═╡ a588cb4c-f6cc-11ea-31af-11c4e68303b1
chn = sample(golf_logistic(x, y, n, length(x)), NUTS(), MCMCThreads(), 4000, 4);

# ╔═╡ 538f0188-f6ce-11ea-04e9-ed8cb67b6910
begin
	post_a = collect(reshape(chn[:a], size(chn[:a],1)*size(chn[:a],2), 1))
	post_b = collect(reshape(chn[:b], size(chn[:b],1)*size(chn[:b],2), 1))
end

# ╔═╡ a832d73c-f6ce-11ea-18c4-676db09efc60
begin
	h1 = histogram(post_a, xlabel="a", ylabel="Probability", normed=true)
	h2 = histogram(post_b, xlabel="b", ylabel="Probability", normed=true)
	plot(h1, h2, layout=(1,2), legend=false)
end

# ╔═╡ 704996e8-f6cf-11ea-26db-716076e7e963
begin
	a_med = median(post_a)
	b_med = median(post_b)
	xrng = 1:1:21
	post_log_med = [logistic(a_med + b_med * x) for x in xrng]
	a_samp = StatsBase.sample(post_a, 50)
	b_samp = StatsBase.sample(post_b, 50)
	
	post_samp = [logistic(a_samp[i] + b_samp[i] * x) for x in xrng, i = 1:50]
	
	scatter(x, pj, yerror=error, ylim=(0,1), xlabel="Distance from hole (ft)", ylabel="Probability of success", legend=false)
	plot!(xrng, post_log_med, color=:green)
	plot!(post_samp, alpha=0.2)
end

# ╔═╡ f8f70b00-f757-11ea-0f80-2bed60c21af2
md"Now we use a model from first principles, where we consider the human error when hitting the ball"

# ╔═╡ ec1b5af4-f75b-11ea-3892-7595cafe914c
md"$y_{j} \sim Binomial(n_{j}, p_{j})$
$p_{j} = 2\Phi{\huge \left( {\normalsize \frac{sin^{-1}((R-r)/x_{j})}{\sigma}}\right)} - 1$"

# ╔═╡ a0cc6b5e-f75d-11ea-3c62-0d1af769df41
Phi(x) = cdf.(Normal(0, 1), x);

# ╔═╡ e41a932c-f75d-11ea-37a2-b1fa17307660
begin
	@model golf_angle(x, y, n, J, r, R) = begin
		
		th_angle = asin.((R - r)./x)
		
		σ ~ Truncated(Normal(0,1), 0, Inf)
		
		p = 2 * Phi(th_angle/σ) .- 1
		
		for i in 1:J
			y[i] ~ Binomial(n[i], p[i])
		end
	end

	r = (1.68 / 2) / 12;
	R = (4.25 / 2) / 12;
end

# ╔═╡ 07019622-f762-11ea-0dc6-6d2c035f232c
md"Here we perform a prior predictive check for the new model"

# ╔═╡ 92e68c3c-f761-11ea-2caf-49f701ab16ca
prior_angle = sample(golf_angle(x, y, n, length(x), r, R), Prior(), 4000);

# ╔═╡ aa00f690-f765-11ea-09ca-073178a57b97
angle_prior = StatsBase.sample(prior_angle[:σ], 500)

# ╔═╡ 0b8c1d1a-f766-11ea-2484-81672e6f2021
begin
	angle_of_shot = rand.(Normal.(0, angle_prior), 1)
	angle_of_shot = getindex.(angle_of_shot)
end

# ╔═╡ 2c61eeb0-f767-11ea-2b54-03eeea073bed
begin
	distance = 20
	final_positions = [
		distance * cos.(angle_of_shot),
		distance * sin.(angle_of_shot)
		]
end

# ╔═╡ 0be5c868-f768-11ea-3491-e1418b4e3210
begin
	 plot(
   [[0, i] for i in final_positions[1]],
   [[0, i] for i in final_positions[2]],
   labels = false,
   legend = :topleft,
   color = :black,
   alpha = 0.3,
   title = "Prior distribution of putts from 20 feet away"
   )
	scatter!(final_positions[1], final_positions[2], color = :black, labels = false)
	scatter!((0,0), color = :green, label = "start", markersize = 6)
	scatter!((20, 0), color = :red, label = "goal", markersize = 6)
end

# ╔═╡ 4a4af7e0-f768-11ea-07ac-1ffb6aba0a09
begin
	chn_angle = sample(golf_angle(x, y, n, length(x), r, R), NUTS(), MCMCThreads(), 4000, 4)
end

# ╔═╡ 9529514e-f768-11ea-17cd-7d1f95ea3a96
prob_angle(threshold, sigma) = 2 * Phi(threshold / sigma) .- 1

# ╔═╡ 5646dd6a-f769-11ea-3bb0-7d62f4b8b54a
begin
	post_sigma_med = median(chn_angle[:σ])
	th_angle = [asin((R - r) / x) for x = xrng]
	geom_lines = prob_angle(th_angle, post_sigma_med)
end

# ╔═╡ 8cca381e-f769-11ea-1c7d-29c181a89174
begin
	scatter(
	  x, pj,
	  yerror= error,
	  label = "",
	  ylim = (0, 1),
	  ylab = "Probability of Success",
	  xlab = "Distance from hole (ft)")
	plot!(post_log_med, color = :black, label = "Logistic regression")
	plot!(geom_lines, color = 1, label = "Geometry-based model")
end

# ╔═╡ 01cd5152-f76d-11ea-1f3b-611a907b2144
begin
	angle_post = StatsBase.sample(chn_angle[:σ], 500)
	angle_of_shot_post = rand.(Normal.(0, angle_post), 1)
	angle_of_shot_post = getindex.(angle_of_shot_post)
	
	final_positions_post = [
		distance * cos.(angle_of_shot_post),
		distance * sin.(angle_of_shot_post)
		]
end

# ╔═╡ f989ecb2-f76c-11ea-0ee5-eff04df348f3
begin
	 plot(
   [[0, i] for i in final_positions_post[1]],
   [[0, i] for i in final_positions_post[2]],
   labels = false,
	xlim = (-21, 21),
	ylim = (-21, 21),
   legend = :topleft,
   color = :black,
   alpha = 0.3,
   title = "Posterior distribution of putts from 20 feet away"
   )
	scatter!(final_positions_post[1], final_positions_post[2], color = :black, labels = false)
	scatter!((0,0), color = :green, label = "start", markersize = 6)
	scatter!((20, 0), color = :red, label = "goal", markersize = 6)
end

# ╔═╡ e97ed2e0-f76e-11ea-0872-59a9c501c9c1
md"New golf data"

# ╔═╡ f72d0f68-f76e-11ea-2a38-7984d2033ed5
begin
	resp_new = HTTP.get("https://raw.githubusercontent.com/stan-dev/example-models/master/knitr/golf/golf_data_new.txt");
	data_new = CSV.read(IOBuffer(resp_new.body), normalizenames = true, header = 3);
end

# ╔═╡ 6d03ceea-f76f-11ea-348d-6184b9b5b569
begin
	x_new, n_new, y_new = (data_new[:,1], data_new[:,2], data_new[:,3])
	p_new = y_new ./ n_new
	xrng_new = 1:80
end

# ╔═╡ e418d980-f76f-11ea-06c1-ef6a06b87118
begin
	th_angle2 = [asin((R - r) / x) for x = xrng_new]
	geom_lines2 = prob_angle(th_angle2, post_sigma_med)
end

# ╔═╡ ca43e738-f770-11ea-0912-b99ba175a7df
begin
	scatter(
	  x, pj,
	  label = "Old data",
	  ylab = "Probability of Success",
	  xlab = "Distance from hole (ft)",
	  color = 1)
	
	scatter!(x_new, p_new, color = 2, label = "New data")
	plot!(geom_lines2, label = "", color = 1)
end

# ╔═╡ 44b76546-f77e-11ea-2015-434b69672f7b
md"Now we use a slightly more accurate model, where we take into account a dependence with the distance"

# ╔═╡ 36afc8c2-f782-11ea-0993-a9c0a7a74799
prob_distance(distance, tol, overshot, sigma) =
  Phi((tol - overshot) ./ ((distance .+ overshot) * sigma)) -
    Phi(-overshot ./ ((distance .+ overshot) * sigma));

# ╔═╡ 85cab520-f782-11ea-36eb-bb56201b6951
begin
	@model golf_angle_dist(x, y, n, J, r, R, overshot, dist_tolerance) = begin

		th_angle = asin.((R - r) ./ x)

		σ_angle ~ truncated(Normal(0, 1), 0, Inf)
		σ_distance ~ truncated(Normal(0, 1), 0, Inf)
		
		p_angle = prob_angle(th_angle, σ_angle)
		p_distance = prob_distance(x, dist_tolerance, overshot, σ_distance)
		p = p_angle .* p_distance
		
		for i in 1:J
			y[i] ~ Binomial(n[i], p[i])
		end
	end
		
	overshot = 1
	dist_tolerance = 3.
end

# ╔═╡ 2e731a16-f784-11ea-2b12-55f36ea52242
chn_angle_dist = sample(golf_angle_dist(x_new, y_new, n_new, length(x_new), r, R, overshot, dist_tolerance), NUTS(), MCMCThreads(), 6000, 4)

# ╔═╡ d40d6c26-f784-11ea-08be-4ffb8c2d1dbe
begin
	post_siga = median(chn_angle_dist[:σ_angle])
	post_sigd = median(chn_angle_dist[:σ_distance])
	
	p_angle = prob_angle(th_angle2, post_siga)
	p_distance = prob_distance(xrng_new, dist_tolerance, overshot, post_sigd)
	
	geom2_lines = p_angle .* p_distance
end

# ╔═╡ 694c1b38-f786-11ea-008f-e9f803ff2109
begin
	scatter(
	  x_new, p_new,
	  legend = false,
	  color = 2,
	  ylab = "Probability of Success",
	  xlab = "Distance from hole (ft)")
	plot!(geom2_lines, color = 2)
end

# ╔═╡ e8a7daa6-f789-11ea-04ec-f7d7b907551f
md"This new model fits better the data, but we can see it underestimates the probability in middle distances"

# ╔═╡ 36352b98-f788-11ea-140c-d5f0a95ee2a4
begin
	@model golf_angle_dist_resid(x, y, n, J, r, R, overshot, dist_tolerance, raw) = begin

		# transformed data
		th_angle = asin.((R - r) ./ x)

 	  # parameters
		σ_angle ~ truncated(Normal(0, 1), 0, Inf)
		σ_distance ~ truncated(Normal(0, 1), 0, Inf)
		σ_y ~ truncated(Normal(0, 1), 0, Inf)

		# model
		p_angle = prob_angle(th_angle, σ_angle)
		p_distance = prob_distance(x, dist_tolerance, overshot, σ_distance)
		p = p_angle .* p_distance

		for i in 1:J
			raw[i] ~ Normal(p[i], sqrt(p[i] * (1-p[i]) / n[i] + σ_y^2))
		end
	end
end

# ╔═╡ b43a2b54-f78a-11ea-29a6-a9ca1c666457
chn_resid = sample(golf_angle_dist_resid(x_new, y_new, n_new, length(x_new), r, R, overshot, dist_tolerance, y_new ./ n_new), NUTS(), MCMCThreads(), 4000, 4)

# ╔═╡ 5516002a-f78b-11ea-3e78-69aa81f4f270
begin
	post_siga2 = median(chn_resid[:σ_angle])
	post_sigd2 = median(chn_resid[:σ_distance])

	p_angle2 = prob_angle(th_angle2, post_siga2)
	p_distance2 = prob_distance(xrng_new, dist_tolerance, overshot, post_sigd2)

	geom_lines_resid = p_angle2 .* p_distance2

	scatter(
	  x_new, p_new,
	  legend = false,
	  color = 2,
	  ylab = "Probability of Success",
	  xlab = "Distance from hole (ft)")
	plot!(geom_lines_resid, color = 2)
end

# ╔═╡ Cell order:
# ╠═732bdf94-f6c8-11ea-35ae-490c5536996d
# ╠═ce2eb27c-f6c8-11ea-02c8-d1dcefce145d
# ╠═46428662-f6c9-11ea-3835-2dde7344ca40
# ╠═925fb2d6-f6c9-11ea-245f-f33635531f50
# ╠═205dc84c-f6cb-11ea-3f31-c18a34f651b1
# ╠═a588cb4c-f6cc-11ea-31af-11c4e68303b1
# ╠═538f0188-f6ce-11ea-04e9-ed8cb67b6910
# ╠═a832d73c-f6ce-11ea-18c4-676db09efc60
# ╠═704996e8-f6cf-11ea-26db-716076e7e963
# ╠═f8f70b00-f757-11ea-0f80-2bed60c21af2
# ╠═ec1b5af4-f75b-11ea-3892-7595cafe914c
# ╠═a0cc6b5e-f75d-11ea-3c62-0d1af769df41
# ╠═e41a932c-f75d-11ea-37a2-b1fa17307660
# ╠═07019622-f762-11ea-0dc6-6d2c035f232c
# ╠═92e68c3c-f761-11ea-2caf-49f701ab16ca
# ╠═aa00f690-f765-11ea-09ca-073178a57b97
# ╠═0b8c1d1a-f766-11ea-2484-81672e6f2021
# ╠═2c61eeb0-f767-11ea-2b54-03eeea073bed
# ╠═0be5c868-f768-11ea-3491-e1418b4e3210
# ╠═4a4af7e0-f768-11ea-07ac-1ffb6aba0a09
# ╠═9529514e-f768-11ea-17cd-7d1f95ea3a96
# ╠═5646dd6a-f769-11ea-3bb0-7d62f4b8b54a
# ╠═8cca381e-f769-11ea-1c7d-29c181a89174
# ╠═01cd5152-f76d-11ea-1f3b-611a907b2144
# ╠═f989ecb2-f76c-11ea-0ee5-eff04df348f3
# ╠═e97ed2e0-f76e-11ea-0872-59a9c501c9c1
# ╠═f72d0f68-f76e-11ea-2a38-7984d2033ed5
# ╠═6d03ceea-f76f-11ea-348d-6184b9b5b569
# ╠═e418d980-f76f-11ea-06c1-ef6a06b87118
# ╠═ca43e738-f770-11ea-0912-b99ba175a7df
# ╠═44b76546-f77e-11ea-2015-434b69672f7b
# ╠═36afc8c2-f782-11ea-0993-a9c0a7a74799
# ╠═85cab520-f782-11ea-36eb-bb56201b6951
# ╠═2e731a16-f784-11ea-2b12-55f36ea52242
# ╠═d40d6c26-f784-11ea-08be-4ffb8c2d1dbe
# ╠═694c1b38-f786-11ea-008f-e9f803ff2109
# ╟─e8a7daa6-f789-11ea-04ec-f7d7b907551f
# ╠═36352b98-f788-11ea-140c-d5f0a95ee2a4
# ╠═b43a2b54-f78a-11ea-29a6-a9ca1c666457
# ╠═5516002a-f78b-11ea-3e78-69aa81f4f270
