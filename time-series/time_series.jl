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

# ╔═╡ 5e2e1eb4-2e5b-11eb-13c3-61416943306d
using SingularSpectrumAnalysis

# ╔═╡ be2b670a-2e7d-11eb-06b1-fd53fec88d85
using LinearAlgebra, FredData, Optim, Measures

# ╔═╡ 72e4393e-2e7f-11eb-195d-e7ab707e4b73
using TSAnalysis

# ╔═╡ 1ca34e3e-2a9c-11eb-29d1-7d2c75e46122
passengers = CSV.read("AirPassengers.csv")

# ╔═╡ f3be6804-2e86-11eb-2d9d-f3118b11fc68
x = DataFrame(x=[], y=[])

# ╔═╡ d7c9a2e4-2e84-11eb-1bb9-f35a19dc5881
DataFrames.rename!(x, [:x, :y])

# ╔═╡ 48bcce70-2b6c-11eb-142c-e38d4f6ff989
plot(passengers[1], passengers[2], label=false, title="Number of monthly passengers")

# ╔═╡ b66ee498-2b72-11eb-1104-2990401f2cf3
N_train_ = 100

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

# ╔═╡ 9b7528fe-2e6b-11eb-3727-ad7b9caa38ca
begin
	n_train = 100
	pass_train = passengers[2][1:n_train]
	pass_test = passengers[2][n_train:end]
	p_ = 44
end

# ╔═╡ 8ff7a3ac-2b74-11eb-00a6-411d1aa10ef3
train = Vector(dates_train, pass_train)

# ╔═╡ cec173ec-2e6c-11eb-079d-fda960a2ba25
pass_train1 = convert(Array{Float64}, pass_train)

# ╔═╡ 9b474cea-2e6b-11eb-331e-338b8529895a
pass_pred = localmodel_tsp(pass_train1, p_; ntype=ntype, stepsize=1)

# ╔═╡ 9b2fd588-2e6b-11eb-2d4b-bb636aefe7e1
begin
	plot(1:100, pass_train, label="training")
	plot!(100:144,pass_test, label="test")
	plot!(100:144, pass_pred, label = "pred")
end

# ╔═╡ 9b19af86-2e6b-11eb-2347-7366ee1f9b73
md"## SingularSpectrumAnalysis.jl"

# ╔═╡ 391a0eac-2e68-11eb-2027-418dfcb2e410
begin
	L   = 20                      # Window length
	K   = 100
	N   = K*L;                    # number of datapoints
	t   = 1:N;                    # Time vector
	T   = 20;                     # period of main oscillation
	y_  = sin.(2pi/T*t);          # Signal
	y   = sin.(2pi/T*t);
end


# ╔═╡ 0d1de8d4-2e69-11eb-3f21-65e9e02d11f8
plot1 = plot(t, y_, xlim=(0,100))

# ╔═╡ 967b042e-2e69-11eb-0791-b92b93176d78
y__ = (0.5sin.(2pi/T*4*t)).^2

# ╔═╡ 9d2439c6-2e69-11eb-00a1-ef81e2a1b682
plot2 = plot(t, y__, xlim=(0,100))

# ╔═╡ edefeb80-2e68-11eb-3856-3fd234caa396
begin
	y .+= (0.5sin.(2pi/T*4*t)).^2 # Add another frequency
	e   = 0.1randn(N);            # Add some noise
	yn = y+e;
end

# ╔═╡ 27325a22-2e69-11eb-0e93-19da461aa4b3
plot3 = plot(t, y, xlim=(0,100))

# ╔═╡ ccb96956-2e69-11eb-07d8-0341e3ea2fc5
plot(plot1, plot2, plot3)

# ╔═╡ 00d8fa66-2e69-11eb-2d1c-8d320acf8854
begin
	yt, ys = analyze(yn, L, robust=true) # trend and seasonal components
	plot(yt, lab="Trend")
	plot!(ys, lab="Season")
end

# ╔═╡ 7296650e-2e6a-11eb-23fc-c77c3c23ad45
plot(yt, lab="Trend", xlim=(0,100))

# ╔═╡ 842ef68a-2e6a-11eb-0da0-7bb97713d15b
plot(ys, lab="Season", xlim=(0,1000))

# ╔═╡ c06188de-2e6a-11eb-3acd-e546adda8558
pass_trend, pass_season = analyze(passengers[2],10,robust=true)
#Doesnt work with no continues values

# ╔═╡ 8d11108a-2e9c-11eb-2c4a-19b90a07fa56
md"## TSAnalysis.jl"

# ╔═╡ 7a8121e8-2e7f-11eb-11ba-9fc1ef32bd53
begin
	plotlyjs();
	f = Fred();
end

# ╔═╡ d7ccc484-2e83-11eb-2e07-1b5999262ac9
function download_fred_vintage(tickers::Array{String,1}, transformations::Array{String,1})

    # Initialise output
    output_data = DataFrame();

    # Loop over tickers
    for i=1:length(tickers)

        # Download from FRED2
        fred_data = get_data(f, tickers[i], observation_start="1984-01-01", units=transformations[i]).data[:, [:date, :value]];
        
		DataFrames.rename!(fred_data, Symbol.(["date", tickers[i]]));

        # Store current vintage
        if i == 1
            output_data = copy(fred_data);
        else
            output_data = join(output_data, fred_data, on=:date, kind = :outer);
        end
    end

    # Return output
    return output_data;
end

# ╔═╡ 09b0f8d0-2e84-11eb-0f5b-476086417910
# Download data of interest
Y_df = download_fred_vintage(["INDPRO"], ["log"])

# ╔═╡ 57a93e26-2e95-11eb-10e9-879a427b2c9a
DataFrames.tail(Y_df)

# ╔═╡ 0e83a412-2e9c-11eb-3522-9301ebfd427b
Y_df[:,2:end]

# ╔═╡ 18e273c4-2e84-11eb-0efa-ed054d85eaae
# Convert to JArray{Float64}
Y = Y_df[:,2:end] |> JArray{Float64}

# ╔═╡ 8e9e37d8-2e89-11eb-2964-ffafec0ad61b
Y_ = permutedims(Y)

# ╔═╡ cb0d338a-2e99-11eb-3c78-9d622b16de2d
plot(1:442,vec(Y_))

# ╔═╡ 59b14490-2e96-11eb-1651-85d3d753b68b
length(Y_)

# ╔═╡ 9b41cf4c-2e89-11eb-1b5b-9170281aea5e
begin
	# Estimation settings for an ARIMA(1,1,1)
d = 1;
p__ = 1;
q = 1;
arima_settings = ARIMASettings(Y_, d, p__, q);
end

# ╔═╡ c638e4b8-2e89-11eb-203f-ab1100d49990
arima_out = arima(arima_settings, NelderMead(), Optim.Options(iterations=10000, f_tol=1e-2, x_tol=1e-2, g_tol=1e-2, show_trace=true, show_every=500))

# ╔═╡ e4a939c4-2e8a-11eb-03cb-df285f70acd5
begin
max_hz = 12;
fc = forecast(arima_out, max_hz, arima_settings)
end

# ╔═╡ c51b6bfa-2e98-11eb-1b65-fba2ab0f9dd4
date_ext_ = Y_df[!,:date]

# ╔═╡ cf800cfe-2e98-11eb-0991-c539f276c52a
length(date_ext_)

# ╔═╡ f049347e-2e93-11eb-2b66-c3a6349558f0
begin
	# Extend date vector
date_ext = Y_df[!,:date] |> Array{Date,1};

for hz=1:max_hz
    last_month = month(date_ext[end]);
    last_year = year(date_ext[end]);

    if last_month == 12
        last_month = 1;
        last_year += 1;
    else
        last_month += 1;
    end

    push!(date_ext, Date("01/$(last_month)/$(last_year)", "dd/mm/yyyy"))
end
date_ext
end



# ╔═╡ 27820466-2e99-11eb-3166-bd056727a04a
length(date_ext)

# ╔═╡ 17a54936-2e99-11eb-2086-a310558dde13
# Generate plot
begin
p_arima = plot(date_ext, [Y_[1,:]; NaN*ones(max_hz)], label="Data", color=RGB(0,0,200/255),
               xtickfont=font(8, "Helvetica Neue"), ytickfont=font(8, "Helvetica Neue"),
               title="INDPRO", titlefont=font(10, "Helvetica Neue"), framestyle=:box,
               legend=:right, size=(800,250), margin = 5mm);

plot!(date_ext, [NaN*ones(length(date_ext)-size(fc,2)); fc[1,:]], label="Forecast", color=RGB(0,0,200/255), line=:dot)
end

# ╔═╡ 642234f4-2e9b-11eb-0b30-13b630a02f6c
md"### Passengers data"

# ╔═╡ 2a99cfb0-2e9b-11eb-3d10-9d527922db30
pass = passengers[:,2:end] |> JArray{Float64}

# ╔═╡ 47e94400-2e9c-11eb-14e4-85edaa8297c9
pass_ = permutedims(pass)

# ╔═╡ af587d14-2e9b-11eb-3a80-ede63e9b83d3
arima_settings_pass = ARIMASettings(pass_, d, p__, q)

# ╔═╡ 5c174972-2e9c-11eb-3b79-09529685749c
arima_out_pass = arima(arima_settings_pass, NelderMead(), Optim.Options(iterations=10000, f_tol=1e-2, x_tol=1e-2, g_tol=1e-2, show_trace=true, show_every=500))

# ╔═╡ 86e8b0c8-2e9c-11eb-210e-d5c1a77479c5
begin
fc_pass = forecast(arima_out_pass, max_hz, arima_settings_pass)
end

# ╔═╡ ff624f0a-2e9c-11eb-3673-9b4441f9887b
passengers[!,1]

# ╔═╡ e3d0ef44-2e9c-11eb-2968-b595402771ea
begin
	# Extend date vector
date_ext_pass = passengers[!,1] |> Array{Date,1};

for hz=1:max_hz
    last_month = month(date_ext_pass[end]);
    last_year = year(date_ext_pass[end]);

    if last_month == 12
        last_month = 1;
        last_year += 1;
    else
        last_month += 1;
    end

    push!(date_ext_pass, Date("01/$(last_month)/$(last_year)", "dd/mm/yyyy"))
end
date_ext_pass
end

# ╔═╡ d1b48ee6-2e9d-11eb-38e4-0b259b7e904c
length(date_ext_pass)

# ╔═╡ 28fa5010-2e9d-11eb-0e03-531f345d5186
begin
p_arima_pass = plot(date_ext_pass, [pass_[1,:]; NaN*ones(max_hz)], label="Data", color=RGB(0,0,200/255),
               xtickfont=font(8, "Helvetica Neue"), ytickfont=font(8, "Helvetica Neue"),
               title="Passengers", titlefont=font(10, "Helvetica Neue"), framestyle=:box,
               legend=:right, size=(800,250), margin = 5mm);

plot!(date_ext_pass, [NaN*ones(length(date_ext_pass)-size(fc_pass,2)); fc_pass[1,:]], label="Forecast", color=RGB(0,0,200/255), line=:dot,legend=:bottomleft)
end

# ╔═╡ Cell order:
# ╠═01de92b6-2a9c-11eb-179a-553dd5875e07
# ╠═1ca34e3e-2a9c-11eb-29d1-7d2c75e46122
# ╠═f3be6804-2e86-11eb-2d9d-f3118b11fc68
# ╠═d7c9a2e4-2e84-11eb-1bb9-f35a19dc5881
# ╠═48bcce70-2b6c-11eb-142c-e38d4f6ff989
# ╠═be4b4484-2b6d-11eb-229d-29f422c2100a
# ╠═b66ee498-2b72-11eb-1104-2990401f2cf3
# ╠═8ff7a3ac-2b74-11eb-00a6-411d1aa10ef3
# ╟─6f7855aa-2b73-11eb-28d8-8f246ae292af
# ╠═83ed18b8-2b73-11eb-20cc-c1fff1563e72
# ╠═cde32c12-2b75-11eb-1201-35b2e673d0f6
# ╠═da38fff2-2b78-11eb-0259-9f27223ecd4e
# ╠═9b7528fe-2e6b-11eb-3727-ad7b9caa38ca
# ╠═cec173ec-2e6c-11eb-079d-fda960a2ba25
# ╠═9b474cea-2e6b-11eb-331e-338b8529895a
# ╠═9b2fd588-2e6b-11eb-2d4b-bb636aefe7e1
# ╟─9b19af86-2e6b-11eb-2347-7366ee1f9b73
# ╠═5e2e1eb4-2e5b-11eb-13c3-61416943306d
# ╠═391a0eac-2e68-11eb-2027-418dfcb2e410
# ╠═0d1de8d4-2e69-11eb-3f21-65e9e02d11f8
# ╠═967b042e-2e69-11eb-0791-b92b93176d78
# ╠═9d2439c6-2e69-11eb-00a1-ef81e2a1b682
# ╠═edefeb80-2e68-11eb-3856-3fd234caa396
# ╠═27325a22-2e69-11eb-0e93-19da461aa4b3
# ╠═ccb96956-2e69-11eb-07d8-0341e3ea2fc5
# ╠═00d8fa66-2e69-11eb-2d1c-8d320acf8854
# ╠═7296650e-2e6a-11eb-23fc-c77c3c23ad45
# ╠═842ef68a-2e6a-11eb-0da0-7bb97713d15b
# ╠═c06188de-2e6a-11eb-3acd-e546adda8558
# ╠═8d11108a-2e9c-11eb-2c4a-19b90a07fa56
# ╠═be2b670a-2e7d-11eb-06b1-fd53fec88d85
# ╠═72e4393e-2e7f-11eb-195d-e7ab707e4b73
# ╠═7a8121e8-2e7f-11eb-11ba-9fc1ef32bd53
# ╠═d7ccc484-2e83-11eb-2e07-1b5999262ac9
# ╠═09b0f8d0-2e84-11eb-0f5b-476086417910
# ╠═57a93e26-2e95-11eb-10e9-879a427b2c9a
# ╠═0e83a412-2e9c-11eb-3522-9301ebfd427b
# ╠═18e273c4-2e84-11eb-0efa-ed054d85eaae
# ╠═8e9e37d8-2e89-11eb-2964-ffafec0ad61b
# ╠═cb0d338a-2e99-11eb-3c78-9d622b16de2d
# ╠═59b14490-2e96-11eb-1651-85d3d753b68b
# ╠═9b41cf4c-2e89-11eb-1b5b-9170281aea5e
# ╠═c638e4b8-2e89-11eb-203f-ab1100d49990
# ╠═e4a939c4-2e8a-11eb-03cb-df285f70acd5
# ╠═c51b6bfa-2e98-11eb-1b65-fba2ab0f9dd4
# ╠═cf800cfe-2e98-11eb-0991-c539f276c52a
# ╠═f049347e-2e93-11eb-2b66-c3a6349558f0
# ╠═27820466-2e99-11eb-3166-bd056727a04a
# ╠═17a54936-2e99-11eb-2086-a310558dde13
# ╠═642234f4-2e9b-11eb-0b30-13b630a02f6c
# ╠═2a99cfb0-2e9b-11eb-3d10-9d527922db30
# ╠═47e94400-2e9c-11eb-14e4-85edaa8297c9
# ╠═af587d14-2e9b-11eb-3a80-ede63e9b83d3
# ╠═5c174972-2e9c-11eb-3b79-09529685749c
# ╠═86e8b0c8-2e9c-11eb-210e-d5c1a77479c5
# ╠═ff624f0a-2e9c-11eb-3673-9b4441f9887b
# ╠═e3d0ef44-2e9c-11eb-2968-b595402771ea
# ╠═d1b48ee6-2e9d-11eb-38e4-0b259b7e904c
# ╠═28fa5010-2e9d-11eb-0e03-531f345d5186
