### A Pluto.jl notebook ###
# v0.11.14

using Markdown
using InteractiveUtils

# ╔═╡ 01de92b6-2a9c-11eb-179a-553dd5875e07
begin
	using CSV
	using Plots
	using DataFrames
	using Dates
end

# ╔═╡ be4b4484-2b6d-11eb-229d-29f422c2100a
begin
	using TimeseriesPrediction 
	using DynamicalSystemsBase 
	using TimeSeries
end

# ╔═╡ 1ca34e3e-2a9c-11eb-29d1-7d2c75e46122
passengers = CSV.read("AirPassengers.csv")

# ╔═╡ 48bcce70-2b6c-11eb-142c-e38d4f6ff989
plot(passengers[1], passengers[2], label=false, title="Number of monthly passengers")

# ╔═╡ b66ee498-2b72-11eb-1104-2990401f2cf3
N_train_ = 100

# ╔═╡ c0521afc-2b72-11eb-047b-61a67363bc59
begin
	dates_train = passengers[1:N_train_, 1]
	pass_train = passengers[1:N_train_, 2]
	dates_test  = passengers[N_train_:end,1]
	pass_test  = passengers[N_train_:end,2]
end

# ╔═╡ 8ff7a3ac-2b74-11eb-00a6-411d1aa10ef3
train = Vector(dates_train, pass_train)

# ╔═╡ 6f7855aa-2b73-11eb-28d8-8f246ae292af


# ╔═╡ 83ed18b8-2b73-11eb-20cc-c1fff1563e72


# ╔═╡ cde32c12-2b75-11eb-1201-35b2e673d0f6
begin
ds = Systems.roessler(0.1ones(3))
dt = 0.1
data = trajectory(ds, 1000; dt=dt)
N_train = 6001
s_train = data[1:N_train, 1]
s_test  = data[N_train:end,1]

ntype = FixedMassNeighborhood(3)

p = 500
s_pred = localmodel_tsp(s_train, 4, 15, p; ntype=ntype)
end

# ╔═╡ da38fff2-2b78-11eb-0259-9f27223ecd4e
begin
plot(550:dt:600, s_train[5501:end], label = "training (trunc.)", color = "black")
plot!(600:dt:(600+p*dt), s_test[1:p+1], color = "Orange", label = "actual signal")
plot!(600:dt:(600+p*dt), s_pred, color = "Blue", label="predicted",legend=:bottomleft)
end

# ╔═╡ Cell order:
# ╠═01de92b6-2a9c-11eb-179a-553dd5875e07
# ╠═1ca34e3e-2a9c-11eb-29d1-7d2c75e46122
# ╠═48bcce70-2b6c-11eb-142c-e38d4f6ff989
# ╠═be4b4484-2b6d-11eb-229d-29f422c2100a
# ╠═b66ee498-2b72-11eb-1104-2990401f2cf3
# ╠═c0521afc-2b72-11eb-047b-61a67363bc59
# ╠═8ff7a3ac-2b74-11eb-00a6-411d1aa10ef3
# ╟─6f7855aa-2b73-11eb-28d8-8f246ae292af
# ╠═83ed18b8-2b73-11eb-20cc-c1fff1563e72
# ╠═cde32c12-2b75-11eb-1201-35b2e673d0f6
# ╠═da38fff2-2b78-11eb-0259-9f27223ecd4e
