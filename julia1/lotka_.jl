### A Pluto.jl notebook ###
# v0.12.18

using Markdown
using InteractiveUtils

# ╔═╡ 0b8b6fb6-4ea1-11eb-2186-5519aa14f523
using DataDrivenDiffEq: SINDy

# ╔═╡ 48b803c4-4942-11eb-3d1d-ab098962b332
begin
using OrdinaryDiffEq
using ModelingToolkit
using DataDrivenDiffEq
using LinearAlgebra, DiffEqSensitivity, Optim
using DiffEqFlux, Flux
using Plots
gr()
end


# ╔═╡ 0af98dbc-4a02-11eb-1190-0d158a97b116


# ╔═╡ aa6ec3d4-4e98-11eb-0a3f-c151fb4813f8
pwd()

# ╔═╡ 622a856c-4944-11eb-0051-c171af3b75cd
function lotka(du, u, p, t)
    α, β, γ, δ = p
    du[1] = α*u[1] - β*u[2]*u[1]
    du[2] = γ*u[1]*u[2]  - δ*u[2]
end

# ╔═╡ 081badd0-49e6-11eb-0141-a3fa1a577937
# Define the experimental parameter
begin
tspan = (0.0f0,3.0f0)
u0 = Float32[0.44249296,4.6280594]
p_ = Float32[1.3, 0.9, 0.8, 1.8]
prob = ODEProblem(lotka, u0,tspan, p_)
solution = solve(prob, Vern7(), abstol=1e-12, reltol=1e-12, saveat = 0.1)
end;

# ╔═╡ 44a1b18c-49e6-11eb-1f11-e1ab13151aa0
begin
scatter(solution, alpha = 0.25)
plot!(solution, alpha = 0.5)
end

# ╔═╡ 4df77b4c-49e6-11eb-08dc-4dc2c44c4329
# Ideal data
begin
X = Array(solution)
# Add noise to the data
println("Generate noisy data")
Xₙ = X + Float32(1e-3)*randn(eltype(X), size(X))
end

# ╔═╡ 8080849e-49e6-11eb-2fe1-6d0571a00e4a
# Define the neueral network which learns L(x, y, y(t-τ))
# Actually, we do not care about overfitting right now, since we want to
# extract the derivative information without numerical differentiation.
begin
L = FastChain(FastDense(2, 32, tanh),FastDense(32, 32, tanh), FastDense(32, 2))
p = initial_params(L)
end

# ╔═╡ f9078e58-49e6-11eb-3887-1ba2d3f1ccf3
# Define our incomplete knowledge of the system.
function dudt_(u, p,t)
    x, y = u
    z = L(u,p)
    [p_[1]*x + z[1],
    -p_[4]*y + z[2]]
end

# ╔═╡ 1f5fbf6c-49e7-11eb-1b30-b1a70085d340
prob_nn = ODEProblem(dudt_,u0, tspan, p)

# ╔═╡ b6148488-49e7-11eb-2642-c3074db57c1e
sol_nn = solve(prob_nn, Tsit5(), u0 = u0, p = p, saveat = solution.t)

# ╔═╡ 1e85a476-49e9-11eb-3310-2169955bbb7c
begin
plot(solution)
plot!(sol_nn)
end

# ╔═╡ 5c3dc632-49e9-11eb-0607-d737e659d427
function predict(θ)
    Array(solve(prob_nn, Vern7(), u0 = u0, p=θ, saveat = solution.t,
                         abstol=1e-6, reltol=1e-6,
                         sensealg =
						 InterpolatingAdjoint(autojacvec=ReverseDiffVJP())))
end

# ╔═╡ ffbcf9c2-49e9-11eb-233d-53bd9fed0308
# No regularisation right now
function loss(θ)
    pred = predict(θ)
    sum(abs2, Xₙ .- pred), pred 
end

# ╔═╡ 8916a742-49ea-11eb-3225-af755add4700
#test
loss(p)

# ╔═╡ 070d33ea-49ec-11eb-21ee-c5aadf4628fc
begin
	
const losses = []

callback(θ,l,pred) = begin
    push!(losses, l)
    if length(losses)%50==0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    false
end

end

# ╔═╡ 35948d4e-49ec-11eb-3455-51f718c1e57c
# First train with ADAM for better convergence
res1 = DiffEqFlux.sciml_train(loss, p, ADAM(0.01), cb=callback, maxiters = 200)

# ╔═╡ e8920da4-49ec-11eb-2a85-0305d716ce77
res2 = DiffEqFlux.sciml_train(loss, res1.minimizer, BFGS(initial_stepnorm=0.01), cb=callback, maxiters = 10000)

# ╔═╡ e8268b74-49ec-11eb-3b89-c9e0acdc6072
md"Final training loss after $(length(losses)) iterations: $(losses[end])"

# ╔═╡ 124c10ce-49ee-11eb-0236-1d4d68ca1fa0
plot(losses,yaxis = :log, xaxis = :log, xlabel = "Iterations", ylabel = "Loss")

# ╔═╡ 7f681cf4-49ee-11eb-2706-71c96f649880
begin
# Plot the data and the approximation
NNsolution = predict(res2.minimizer)
# Trained on noisy data vs real solution
plot(solution.t, NNsolution')
plot!(solution.t, X')
end

# ╔═╡ a7b0abac-49ee-11eb-256b-8f03b8b0b3ec
# Ideal derivatives
DX = Array(solution(solution.t, Val{1}))

# ╔═╡ a34cf972-49ef-11eb-3ed6-3779167ea626
begin
prob_nn2 = ODEProblem(dudt_,u0, tspan, res2.minimizer)
_sol = solve(prob_nn2, Tsit5())
DX_ = Array(_sol(solution.t, Val{1}))
end

# ╔═╡ e6cfc4da-49f0-11eb-150d-65d2fbd099af
# The learned derivatives
begin
plot(DX')
plot!(DX_')
end

# ╔═╡ ff77dff4-49f0-11eb-1b85-ffef73bff82f
begin
# Ideal data
L̄ = [-p_[2]*(X[1,:].*X[2,:])';p_[3]*(X[1,:].*X[2,:])']
# Neural network guess
L̂ = L(Xₙ,res2.minimizer)
end

# ╔═╡ 2b2c6412-49f1-11eb-0d00-69555af0aab3
begin
scatter(L̄')
plot!(L̂')
end

# ╔═╡ 6f8d8fc0-49f1-11eb-11c4-19ac6f8c9573
scatter(abs.(L̄-L̂)', yaxis = :log)

# ╔═╡ 1ddc64a0-49f2-11eb-3f11-a5bde1a4b536
## Sparse Identification 

begin
# Create a Basis
@variables u[1:2]
# Lots of polynomials
polys = ModelingToolkit.Operation[1]

for i ∈ 1:5
    push!(polys, u[1]^i)
    push!(polys, u[2]^i)
    for j ∈ i:5
        if i != j
            push!(polys, (u[1]^i)*(u[2]^j))
            push!(polys, u[2]^i*u[1]^i)
        end
    end
end
	
end

# ╔═╡ 84e37dee-49f2-11eb-3563-3f32b6205a9e
begin
# And some other stuff
h = [cos.(u)...; sin.(u)...; polys...]
basis = Basis(h, u)
end

# ╔═╡ 03db67e0-49fc-11eb-2ca0-89973d1336b6
begin
# Create an optimizer for the SINDy problem
opt = SR3()
# Create the thresholds which should be used in the search process
λ = exp10.(-7:0.1:3)
# Target function to choose the results from; x = L0 of coefficients and L2-Error of the model
g(x) = x[1] < 1 ? Inf : norm(x, 2)
end

# ╔═╡ acb91b82-49fc-11eb-3d3e-f982ffcb22f6
Ψ = Ψ = SINDy(Xₙ[:, :], DX[:, :], basis, λ, opt, g = g, maxiter = 10000) # Fail

# ╔═╡ aea1208e-4ea9-11eb-30f8-3f7879098b17
md"$println(Ψ)"

# ╔═╡ ced705e2-4ea9-11eb-32a9-bd2e163b5bfb
print_equations(Ψ)

# ╔═╡ ea3b3380-4eb8-11eb-0d88-31d2eb8152af
Ψ1 = SINDy(Xₙ[:, 1:end], L̄[:, 1:end], basis, λ,opt, g = g, maxiter = 10000) # Succeed

# ╔═╡ 2620712c-4ec0-11eb-3bbb-f7f88b2f5a72


# ╔═╡ Cell order:
# ╟─0af98dbc-4a02-11eb-1190-0d158a97b116
# ╠═aa6ec3d4-4e98-11eb-0a3f-c151fb4813f8
# ╠═0b8b6fb6-4ea1-11eb-2186-5519aa14f523
# ╠═48b803c4-4942-11eb-3d1d-ab098962b332
# ╠═622a856c-4944-11eb-0051-c171af3b75cd
# ╠═081badd0-49e6-11eb-0141-a3fa1a577937
# ╠═44a1b18c-49e6-11eb-1f11-e1ab13151aa0
# ╠═4df77b4c-49e6-11eb-08dc-4dc2c44c4329
# ╠═8080849e-49e6-11eb-2fe1-6d0571a00e4a
# ╠═f9078e58-49e6-11eb-3887-1ba2d3f1ccf3
# ╠═1f5fbf6c-49e7-11eb-1b30-b1a70085d340
# ╠═b6148488-49e7-11eb-2642-c3074db57c1e
# ╠═1e85a476-49e9-11eb-3310-2169955bbb7c
# ╠═5c3dc632-49e9-11eb-0607-d737e659d427
# ╠═ffbcf9c2-49e9-11eb-233d-53bd9fed0308
# ╠═8916a742-49ea-11eb-3225-af755add4700
# ╠═070d33ea-49ec-11eb-21ee-c5aadf4628fc
# ╠═35948d4e-49ec-11eb-3455-51f718c1e57c
# ╠═e8920da4-49ec-11eb-2a85-0305d716ce77
# ╟─e8268b74-49ec-11eb-3b89-c9e0acdc6072
# ╠═124c10ce-49ee-11eb-0236-1d4d68ca1fa0
# ╠═7f681cf4-49ee-11eb-2706-71c96f649880
# ╠═a7b0abac-49ee-11eb-256b-8f03b8b0b3ec
# ╠═a34cf972-49ef-11eb-3ed6-3779167ea626
# ╠═e6cfc4da-49f0-11eb-150d-65d2fbd099af
# ╠═ff77dff4-49f0-11eb-1b85-ffef73bff82f
# ╠═2b2c6412-49f1-11eb-0d00-69555af0aab3
# ╠═6f8d8fc0-49f1-11eb-11c4-19ac6f8c9573
# ╠═1ddc64a0-49f2-11eb-3f11-a5bde1a4b536
# ╠═84e37dee-49f2-11eb-3563-3f32b6205a9e
# ╠═03db67e0-49fc-11eb-2ca0-89973d1336b6
# ╠═acb91b82-49fc-11eb-3d3e-f982ffcb22f6
# ╠═aea1208e-4ea9-11eb-30f8-3f7879098b17
# ╠═ced705e2-4ea9-11eb-32a9-bd2e163b5bfb
# ╠═ea3b3380-4eb8-11eb-0d88-31d2eb8152af
# ╠═2620712c-4ec0-11eb-3bbb-f7f88b2f5a72
