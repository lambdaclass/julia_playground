### A Pluto.jl notebook ###
# v0.12.7

using Markdown
using InteractiveUtils

# ╔═╡ a8985a28-301a-11eb-13fb-3540d83641bb
using Plots

# ╔═╡ 97ceb2ca-2f5a-11eb-2a7b-cd339b881ea1
md"## Exponential Smoothing"

# ╔═╡ f03467ca-2f5a-11eb-1d53-f3d95c4267db
function SES(time_serie, α)
	y_pred = 0
	N = length(time_serie)
	
	for i in 1:N
		y_pred += time_serie[N - (i - 1)] * (α * ((1-α)^(i-1)))
	end
	
	return y_pred
end	

# ╔═╡ bc7a1146-3010-11eb-1b31-d940642ff8aa
y_ =[445.36,453.20,454.41,422.38,456.04,440.39,425.19,486.21,500.43,521.28,508.95,488.89,509.87,456.72,473.82,525.95,549.83,542.34]

# ╔═╡ 2cc65384-3009-11eb-26f9-9f7c099e6114
SES(y_, 0.83)

# ╔═╡ 340dc8bc-3017-11eb-04ac-19fc0e45d6af
function SES_loss(time_serie, α, l0)
	loss = 0
	#time_serie_l0 = vcat([l0], time_serie)
	N = length(time_serie)
	y_pred = 0
	
	for i in 1:(N)
		if i == 1
			y_pred = l0
		else
			y_pred = time_serie[i - 1] * α + y_pred * (1 - α)
		end
	
		loss += (time_serie[i] - y_pred)^2
		
	end
	
	return loss
end
		

# ╔═╡ afda21c6-30c5-11eb-141a-bbc93204323b


# ╔═╡ 1ade257e-3019-11eb-1629-cb9a519f86fb
los= SES_loss([445.36,453.20,454.41,422.38,456.04,440.39,425.19,486.21,500.43,521.28,508.95,488.89,509.87,456.72,473.82,525.95,549.83,542.34], 0.833, 446.59)

# ╔═╡ 98ef988a-3019-11eb-1461-f38c5714521f
aplhas = collect(0:0.01:1)

# ╔═╡ acd8fb36-3019-11eb-2a51-d34ed19b2646
begin
	a = Array{Float64}(undef, length(aplhas))
for j in 1:length(aplhas)
	a[j] = SES_loss(y_, aplhas[j], 446.6)
end
end


# ╔═╡ a131db1a-301a-11eb-293b-e128292362f2
plot(0:0.01:1, a)

# ╔═╡ 118c8542-30b9-11eb-0e73-0f90ef1da61e
begin
	plot(y_)
	plot!(pred, legend=false)
end

# ╔═╡ 7f2faf2a-30b9-11eb-28fe-8bef7e092da7
pred

# ╔═╡ Cell order:
# ╟─97ceb2ca-2f5a-11eb-2a7b-cd339b881ea1
# ╠═f03467ca-2f5a-11eb-1d53-f3d95c4267db
# ╠═2cc65384-3009-11eb-26f9-9f7c099e6114
# ╠═bc7a1146-3010-11eb-1b31-d940642ff8aa
# ╠═340dc8bc-3017-11eb-04ac-19fc0e45d6af
# ╠═afda21c6-30c5-11eb-141a-bbc93204323b
# ╠═1ade257e-3019-11eb-1629-cb9a519f86fb
# ╠═98ef988a-3019-11eb-1461-f38c5714521f
# ╠═acd8fb36-3019-11eb-2a51-d34ed19b2646
# ╠═a8985a28-301a-11eb-13fb-3540d83641bb
# ╠═a131db1a-301a-11eb-293b-e128292362f2
# ╠═118c8542-30b9-11eb-0e73-0f90ef1da61e
# ╠═7f2faf2a-30b9-11eb-28fe-8bef7e092da7
