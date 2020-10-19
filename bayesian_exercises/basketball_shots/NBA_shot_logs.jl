### A Pluto.jl notebook ###
# v0.12.3

using Markdown
using InteractiveUtils

# ╔═╡ cf3bd2c2-0e5a-11eb-0044-436d9e311673
begin
	using CSV
	using Turing
	using StatsPlots
	using StatsBase
	using Distributions
	using DataFrames
	using Plots
end

# ╔═╡ bee71b32-0efc-11eb-1726-73be2549a827
begin
	using StatsFuns: logistic

	@model basketball_logistic(x, y, n, J) = begin
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

# ╔═╡ 15c3803c-0e5b-11eb-365b-a512b3638daf
shots_data = CSV.read("shot_logs.csv");

# ╔═╡ c8e91178-0e5f-11eb-0dfe-09d78eead41f
begin
	shots_dist = shots_data["SHOT_DIST"]
	shots_result = shots_data["SHOT_RESULT"]
	h_bins = 1:0.2:maximum(shots_dist)
	shots_dist_scored = [shots_dist[i] for i in 1:length(shots_dist) if shots_result[i] == "made"]
end

# ╔═╡ 38c36394-0e61-11eb-048d-c14ce6dfe6bf
begin
	h_total_shots = histogram(shots_dist, bins=h_bins, label=false, xlabel="Distance (ft)", ylabel="Frequency")
	h_scored_shots = histogram(shots_dist_scored, bins=h_bins, label=false, xlabel="Distance (ft)", ylabel="Frequency")
	plot(h_total_shots, h_scored_shots, layout=(1,2))
end

# ╔═╡ 4592af06-0e65-11eb-0e2f-ed31b11e5ad1
begin
	histo_total_shots = fit(Histogram, shots_dist, h_bins)
	total_shots = histo_total_shots.weights
	histo_scored_shots = fit(Histogram, shots_dist_scored, h_bins)
	scored_shots = histo_scored_shots.weights
end

# ╔═╡ 151c6ca6-0ef9-11eb-16e3-67dd05cee664
begin
	# We are going to consider distances up to 30 ft, data is too noisy after that.
	idx_distmax = Int64(25/0.2)
	shot_accurracy = [i/j for (i,j) in zip(scored_shots, total_shots)][1:idx_distmax]
	distances = h_bins[2:end][1:idx_distmax]
end

# ╔═╡ 5cc181e6-0f00-11eb-1fcc-273d7a0cabe6
error = @. sqrt(shot_accurracy * (1 - shot_accurracy) / total_shots[1:idx_distmax])

# ╔═╡ cde73cfa-0ef9-11eb-15b8-ef65484c9037
scatter(distances, shot_accurracy, label=false, xlabel="Distance (ft)", ylabel="Accurracy", yerror=error, markersize=4, color="green", alpha=0.7)

# ╔═╡ 0ac52920-0f13-11eb-169c-f798c482f550
md" ### Logistic vanilla model"

# ╔═╡ 605beae2-0efd-11eb-24ed-45621cf45f9b
chn = sample(basketball_logistic(distances, scored_shots, total_shots, length(distances)), NUTS(), MCMCThreads(), 4000, 4);

# ╔═╡ 341b5cca-0efe-11eb-0cb1-fbeb5c5c5a33
begin
    post_a = collect(reshape(chn[:a], size(chn[:a],1)*size(chn[:a],2), 1))
    post_b = collect(reshape(chn[:b], size(chn[:b],1)*size(chn[:b],2), 1))
end

# ╔═╡ e06dcafa-0eff-11eb-0250-0d9b77c271ed
begin
	h_a = histogram(post_a, xlabel="a", ylabel="Probability", normed=true, color="red", alpha=0.7, label=false)
	h_b = histogram(post_b, xlabel="b", ylabel="Probability", normed=true, color="green", alpha=0.7, label=false)
	plot(h_a, h_b)
end

# ╔═╡ 45d5364c-0f02-11eb-362b-1d0f41b4d2e7
begin
	n_samp = 100
	prior_a = rand(Normal(0, 1), n_samp)
	prior_b = rand(Normal(0, 1), n_samp)
end

# ╔═╡ 3fc06d94-0f02-11eb-0b1b-cb1a4028004b
begin
	# the prior predictive distribution
	a_pri_med = median(prior_a)
	b_pri_med = median(prior_b)
	prior_log_med = [logistic(a_pri_med + b_pri_med * x) for x in distances]
	
	prior_pred_samp = [logistic(prior_a[i] + prior_b[i] * x) for x in distances, i = 1:n_samp]
	
	scatter(distances, shot_accurracy, yerror=error, ylim=(0,1),
		xlabel="Shot distance (ft)", ylabel="Probability of success", legend=false)
	plot!(distances, prior_log_med, color="green")
	plot!(distances, prior_pred_samp, alpha=0.2)
end

# ╔═╡ 418c3076-0f00-11eb-144f-a52ff608ccf3
begin
	# the fit to the model with the data
	a_med = median(post_a)
	b_med = median(post_b)
	post_log_med = [logistic(a_med + b_med * x) for x in distances]
	a_samp = StatsBase.sample(post_a, 100)
	b_samp = StatsBase.sample(post_b, 100)
	
	post_pred_samp = [logistic(a_samp[i] + b_samp[i] * x) for x in distances, i = 1:100]
	
	scatter(distances, shot_accurracy, yerror=error, ylim=(0,1),
		xlabel="Shot distance (ft)", ylabel="Probability of success", color="green", legend=false, alpha=0.7)
	plot!(distances, post_log_med, color="green")
	plot!(distances, post_pred_samp, alpha=0.2)
end

# ╔═╡ e7be54a2-0f16-11eb-121c-e51c937a9f81
idx_sort_shot_dist = sortperm(shots_dist)

# ╔═╡ 8b540d3a-0f1b-11eb-3f1d-99708962145c
sorted_shots_dist = shots_dist[idx_sort_shot_dist]

# ╔═╡ bd10d2ee-0f1a-11eb-0cf1-ed11dffcd2dd
begin
	closest_def_dist = shots_data["CLOSE_DEF_DIST"]
	closest_def_dist_sorted = closest_def_dist[idx_sort_shot_dist]
end

# ╔═╡ 028b757c-0f1b-11eb-38ad-7b9b63d5e39a
begin
	i0 = 1
	dist_dist_dict = Dict{Float64, Array{Float64,1}}()
	for d in distances
		dist_dist_dict[d] = Float64[]
		for i in i0:length(sorted_shots_dist)
			if sorted_shots_dist[i] < d
				push!(dist_dist_dict[d], closest_def_dist_sorted[i])
			else
				i0 = i
				break
			end
		end
	end
end	

# ╔═╡ df103ebe-0f1c-11eb-2942-eb32200d9078
begin
	closest_avg_dist = Array{Float64,1}(undef, length(distances))
	for i in 1:length(distances)
		closest_avg_dist[i] = mean(dist_dist_dict[distances[i]])
	end
end

# ╔═╡ 016f20f6-0f22-11eb-3a7e-9b4818727f68
scatter(distances, closest_avg_dist, xlabel="Shot distance (ft)", ylabel="Average closest player distance", label=false, color="purple", alpha=0.7)

# ╔═╡ 20e6e234-0fc2-11eb-2fbf-bf718b0a50da
begin
	dist_standard = collect(distances) ./ maximum(collect(distances))
	closest_standard = closest_avg_dist ./ maximum(closest_avg_dist)
end

# ╔═╡ 01916de6-0fc2-11eb-1e8b-83ab1eb56260
begin
	@model basketball_logistic2(x, y, z, n, J) = begin
	  # parameters
	  a ~ Normal(0, 1)
	  b ~ Normal(-1, 1)
	  c ~ Normal(1, 1)
		
	  # model
	  for i in 1:J
		p = logistic(a + b * x[i] + c * z[i])
		y[i] ~ Binomial(n[i], p)
	  end
	end
end

# ╔═╡ 59aa17a0-0fc5-11eb-2b05-4bcc9b2acf93
chn2 = sample(basketball_logistic2(distances, scored_shots, closest_avg_dist, total_shots, length(distances)), NUTS(), MCMCThreads(), 4000, 2);

# ╔═╡ 8d090750-0fc5-11eb-3bb5-873b215f8ef4
begin
    post_a2 = collect(reshape(chn2[:a], size(chn2[:a],1)*size(chn2[:a],2), 1))
    post_b2 = collect(reshape(chn2[:b], size(chn2[:b],1)*size(chn2[:b],2), 1))
	post_c2 = collect(reshape(chn2[:c], size(chn2[:c],1)*size(chn2[:c],2), 1))
end

# ╔═╡ f8845156-0fc5-11eb-3427-4df2e3a77152
begin
	# the fit to the model with the data
	a_med2 = median(post_a2)
	b_med2 = median(post_b2)
	c_med2 = median(post_c2)
	post_log_med2 = [logistic(a_med2 + b_med2 * x + c_med2 * z) for (x,z) in zip(distances, closest_avg_dist)]
	a_samp2 = StatsBase.sample(post_a2, 100)
	b_samp2 = StatsBase.sample(post_b2, 100)
	c_samp2 = StatsBase.sample(post_c2, 100)
	
	post_pred_samp2 = [logistic(a_samp2[i] + b_samp2[i] * x + c_samp2[i] * z) for (x,z) in zip(distances,closest_avg_dist), i = 1:100]
	
	scatter(distances, shot_accurracy, yerror=error, ylim=(0,1),
		xlabel="Shot distance (ft)", ylabel="Probability of success", color="green", legend=false, alpha=0.7)
	plot!(distances, post_log_med2, color="green")
	plot!(distances, post_pred_samp2, alpha=0.2)
end

# ╔═╡ 7dfc7476-0fc6-11eb-2080-f10ec344aaa8
scatter(post_b2, post_a2, xlabel="b", ylabel="a", alpha=0.7, label=false)

# ╔═╡ 82ce1bc4-0fc8-11eb-0d37-9143b5aaccc2
md" ### Model with standardized variables"

# ╔═╡ 0e4c7d82-0fc7-11eb-09ee-9529c2f0bf3f
chn3 = sample(basketball_logistic2(dist_standard, scored_shots, closest_standard, total_shots, length(distances)), NUTS(), MCMCThreads(), 4000, 2);

# ╔═╡ 5ede0d7e-0fc7-11eb-3ec5-7721c1a320de
begin
    post_a3 = collect(reshape(chn3[:a], size(chn3[:a],1)*size(chn3[:a],2), 1))
    post_b3 = collect(reshape(chn3[:b], size(chn3[:b],1)*size(chn3[:b],2), 1))
	post_c3 = collect(reshape(chn3[:c], size(chn3[:c],1)*size(chn3[:c],2), 1))
end

# ╔═╡ 759a290c-0fcb-11eb-33d7-cf44c1a335ff
histogram(post_b3)

# ╔═╡ 4ecf7512-0fc7-11eb-3b85-2d3f9174156d
begin
	# the fit to the model with the data
	a_med3 = median(post_a3)
	b_med3 = median(post_b3)
	c_med3 = median(post_c3)
	post_log_med3 = [logistic(a_med3 + b_med3 * x + c_med3 * z) for (x,z) in zip(dist_standard, closest_standard)]
	a_samp3 = StatsBase.sample(post_a3, 100)
	b_samp3 = StatsBase.sample(post_b3, 100)
	c_samp3 = StatsBase.sample(post_c3, 100)
	
	post_pred_samp3 = [logistic(a_samp3[i] + b_samp3[i] * x + c_samp3[i] * z) for (x,z) in zip(dist_standard, closest_standard), i = 1:100]
	
	scatter(distances, shot_accurracy, yerror=error, ylim=(0,1),
		xlabel="Shot distance (ft)", ylabel="Probability of success", color="green", legend=false, alpha=0.7)
	plot!(distances, post_log_med3, color="green")
	plot!(distances, post_pred_samp3, alpha=0.2)
end

# ╔═╡ a04640aa-0fc8-11eb-07e5-67e821c01960
md" ### Model update"

# ╔═╡ bb70a24e-0fc8-11eb-3f68-d9981b131963
begin
	@model basketball_logistic3(x, y, z, n, J) = begin
	  # parameters
	  a ~ Normal(0, 1)
	  b ~ Normal(0, 1)
	  c ~ Normal(0, 1)
	  d ~ Normal(0, 1)
	  e ~ Normal(0, 1)
		
	  # model
	  for i in 1:J
		p = logistic(a + b * x[i] + c * log(z[i])) * logistic(d + e * log(z[i]))
		#p = logistic(a + b * x[i] + c * z[i])
		y[i] ~ Binomial(n[i], p)
	  end
	end
end

# ╔═╡ f8abc832-0fc8-11eb-14ef-91bc6847e8bb
chn4 = sample(basketball_logistic3(dist_standard, scored_shots, closest_standard, total_shots, length(distances)), NUTS(), MCMCThreads(), 4000, 1);

# ╔═╡ 15d99a42-0fc9-11eb-15b2-27a68cf3708b
begin
    post_a4 = collect(reshape(chn4[:a], size(chn4[:a],1)*size(chn4[:a],2), 1))
    post_b4 = collect(reshape(chn4[:b], size(chn4[:b],1)*size(chn4[:b],2), 1))
	post_c4 = collect(reshape(chn4[:c], size(chn4[:c],1)*size(chn4[:c],2), 1))
	post_d4 = collect(reshape(chn4[:d], size(chn4[:d],1)*size(chn4[:d],2), 1))
	post_e4 = collect(reshape(chn4[:e], size(chn4[:e],1)*size(chn4[:e],2), 1))
end

# ╔═╡ 2514c856-0fc9-11eb-26da-dd06cd7068c8
begin
	# the fit to the model with the data
	a_med4 = median(post_a4)
	b_med4 = median(post_b4)
	c_med4 = median(post_c4)
	d_med4 = median(post_d4)
	e_med4 = median(post_e4)
	
	post_log_med4 = [logistic(a_med4 + b_med4 * x + c_med4 * log(z)) * logistic(d_med4 + e_med4 * log(z)) for (x,z) in zip(dist_standard, closest_standard)]
	
	a_samp4 = StatsBase.sample(post_a4, 1000)
	b_samp4 = StatsBase.sample(post_b4, 1000)
	c_samp4 = StatsBase.sample(post_c4, 1000)
	d_samp4 = StatsBase.sample(post_d4, 1000)
	e_samp4 = StatsBase.sample(post_e4, 1000)
	
	post_pred_samp4 = [logistic(a_samp4[i] + b_samp4[i] * x + c_samp4[i] * log(z)) * logistic(d_samp4[i] + e_samp4[i] * log(z)) for (x,z) in zip(dist_standard, closest_standard), i = 1:200]
	
	scatter(distances, shot_accurracy, yerror=error, ylim=(0,1),
		xlabel="Shot distance (ft)", ylabel="Probability of success", color="green", legend=false, alpha=1)
	plot!(distances, post_log_med4, color="green", linewidth=5)
	plot!(distances, post_pred_samp4, alpha=0.2)
end

# ╔═╡ 8ba4d36e-1222-11eb-32e3-f3146cb81191
begin
	@model basketball_logistic4(x, y, z, n, J) = begin
	  # parameters
	  a ~ Normal(0, 1)
	  b ~ Normal(0, 1)
	  c ~ Normal(0, 1)
	  d ~ Normal(0, 1)
		
	  # model
	  for i in 1:J
		p = logistic(a + b * x[i]) * logistic(c + d * z[i] * log(z[i]))
		y[i] ~ Binomial(n[i], p)
	  end
	end
end

# ╔═╡ 87f3881a-1224-11eb-1e41-e9ed9254e66e
chn5 = sample(basketball_logistic4(dist_standard, scored_shots, closest_standard, total_shots, length(distances)), NUTS(), MCMCThreads(), 4000, 1);

# ╔═╡ a7d5b46e-1224-11eb-0c7c-25829dff39e0
begin
    post_a5 = collect(reshape(chn5[:a], size(chn5[:a],1)*size(chn5[:a],2), 1))
    post_b5 = collect(reshape(chn5[:b], size(chn5[:b],1)*size(chn5[:b],2), 1))
	post_c5 = collect(reshape(chn5[:c], size(chn5[:c],1)*size(chn5[:c],2), 1))
	post_d5 = collect(reshape(chn5[:d], size(chn5[:d],1)*size(chn5[:d],2), 1))
end

# ╔═╡ cace4346-1224-11eb-33ff-670520902437
begin
	# the fit to the model with the data
	a_med5 = median(post_a5)
	b_med5 = median(post_b5)
	c_med5 = median(post_c5)
	d_med5 = median(post_d5)
	
	post_log_med5 = [logistic(a_med5 + b_med5 * x) * logistic(c_med5 + d_med5 * log(z)) for (x,z) in zip(dist_standard, closest_standard)]
	
	a_samp5 = StatsBase.sample(post_a5, 1000)
	b_samp5 = StatsBase.sample(post_b5, 1000)
	c_samp5 = StatsBase.sample(post_c5, 1000)
	d_samp5 = StatsBase.sample(post_d5, 1000)
	
	post_pred_samp5 = [logistic(a_samp5[i] + b_samp5[i] * x) * logistic(c_samp5[i] + d_samp5[i] * log(z)) for (x,z) in zip(dist_standard, closest_standard), i = 1:500]
	
	scatter(distances, shot_accurracy, yerror=error, ylim=(0,1),
		xlabel="Shot distance (ft)", ylabel="Probability of success", color="green", legend=false, alpha=1)
	plot!(distances, post_log_med4, color="purple", linewidth=5)
	plot!(distances, post_pred_samp4, alpha=0.2)
end

# ╔═╡ Cell order:
# ╠═cf3bd2c2-0e5a-11eb-0044-436d9e311673
# ╠═15c3803c-0e5b-11eb-365b-a512b3638daf
# ╠═c8e91178-0e5f-11eb-0dfe-09d78eead41f
# ╠═38c36394-0e61-11eb-048d-c14ce6dfe6bf
# ╠═4592af06-0e65-11eb-0e2f-ed31b11e5ad1
# ╠═151c6ca6-0ef9-11eb-16e3-67dd05cee664
# ╠═5cc181e6-0f00-11eb-1fcc-273d7a0cabe6
# ╠═cde73cfa-0ef9-11eb-15b8-ef65484c9037
# ╠═0ac52920-0f13-11eb-169c-f798c482f550
# ╠═bee71b32-0efc-11eb-1726-73be2549a827
# ╠═605beae2-0efd-11eb-24ed-45621cf45f9b
# ╠═341b5cca-0efe-11eb-0cb1-fbeb5c5c5a33
# ╠═e06dcafa-0eff-11eb-0250-0d9b77c271ed
# ╠═45d5364c-0f02-11eb-362b-1d0f41b4d2e7
# ╠═3fc06d94-0f02-11eb-0b1b-cb1a4028004b
# ╠═418c3076-0f00-11eb-144f-a52ff608ccf3
# ╠═e7be54a2-0f16-11eb-121c-e51c937a9f81
# ╠═8b540d3a-0f1b-11eb-3f1d-99708962145c
# ╠═bd10d2ee-0f1a-11eb-0cf1-ed11dffcd2dd
# ╠═028b757c-0f1b-11eb-38ad-7b9b63d5e39a
# ╠═df103ebe-0f1c-11eb-2942-eb32200d9078
# ╠═016f20f6-0f22-11eb-3a7e-9b4818727f68
# ╠═20e6e234-0fc2-11eb-2fbf-bf718b0a50da
# ╠═01916de6-0fc2-11eb-1e8b-83ab1eb56260
# ╠═59aa17a0-0fc5-11eb-2b05-4bcc9b2acf93
# ╠═8d090750-0fc5-11eb-3bb5-873b215f8ef4
# ╠═f8845156-0fc5-11eb-3427-4df2e3a77152
# ╠═7dfc7476-0fc6-11eb-2080-f10ec344aaa8
# ╠═82ce1bc4-0fc8-11eb-0d37-9143b5aaccc2
# ╠═0e4c7d82-0fc7-11eb-09ee-9529c2f0bf3f
# ╠═5ede0d7e-0fc7-11eb-3ec5-7721c1a320de
# ╠═759a290c-0fcb-11eb-33d7-cf44c1a335ff
# ╠═4ecf7512-0fc7-11eb-3b85-2d3f9174156d
# ╟─a04640aa-0fc8-11eb-07e5-67e821c01960
# ╠═bb70a24e-0fc8-11eb-3f68-d9981b131963
# ╠═f8abc832-0fc8-11eb-14ef-91bc6847e8bb
# ╠═15d99a42-0fc9-11eb-15b2-27a68cf3708b
# ╠═2514c856-0fc9-11eb-26da-dd06cd7068c8
# ╠═8ba4d36e-1222-11eb-32e3-f3146cb81191
# ╠═87f3881a-1224-11eb-1e41-e9ed9254e66e
# ╠═a7d5b46e-1224-11eb-0c7c-25829dff39e0
# ╠═cace4346-1224-11eb-33ff-670520902437
