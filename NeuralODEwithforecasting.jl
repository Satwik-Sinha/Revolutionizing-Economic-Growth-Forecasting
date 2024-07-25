using OrdinaryDiffEq, ModelingToolkit, DataDrivenDiffEq, SciMLSensitivity, LinearAlgebra, Statistics
using DataDrivenSparse, Optimization, OptimizationOptimisers, OptimizationOptimJL
using ComponentArrays, Lux, Zygote, Plots, StableRNGs

# Define the Solow-Swan model parameters
s = 0.3
n = 0.02
δ = 0.1
α = 0.3
g = 0.09

# Define the production function
f(k) = k .^ α

# Set a random seed for reproducibility
rng = StableRNG(1111)

# Define the ODE problem
function capital_ode!(dk, k, p, t)
    s, n, δ, α, g = p
    dk[1] = s * f(k[1]) - (n + g + δ) * k[1]
end

# Define initial condition and time span
u0 = [1.0f0]
tspan = (0.0f0, 10.0f0)
p_ = [s, n, δ, α, g]


# Function to add noise to the solution
function add_noise(X, rng, noise_magnitude=5e-3)
    x̄ = mean(X, dims=2)
    return X .+ (noise_magnitude * x̄) .* randn(rng, eltype(X), size(X))
end

# Create and solve the ODE problem
prob = ODEProblem(capital_ode!, u0, tspan, p_)
solution = solve(prob, Tsit5(), abstol=1e-12, reltol=1e-12, saveat=0.25)
X = Array(solution)
Xₙ = add_noise(X, rng)

# Function to add noise to the solution
function add_noise(X, rng, noise_magnitude=5e-3)
    x̄ = mean(X, dims=2)
    return X .+ (noise_magnitude * x̄) .* randn(rng, eltype(X), size(X))
end

# Define the Neural ODE model
const NODE = Lux.Chain(
    Lux.Dense(1, 16, tanh),
    Lux.Dense(16, 16, tanh),
    Lux.Dense(16, 1)
)

# Initialize the Neural ODE parameters
p_node, st_node = Lux.setup(rng, NODE)

# Define the Neural ODE dynamics
function node_dynamics!(du, u, p, t)
    du[1] = NODE(u, p, st_node)[1][1]
end

# Create the Neural ODE problem
prob_node = ODEProblem(node_dynamics!, Xₙ[:, 1], tspan, p_node)

# Define the prediction function for Neural ODE
function predict_node(θ, X = Xₙ[:, 1], T = solution.t)
    _prob = remake(prob_node, u0 = X, tspan = (T[1], T[end]), p = θ)
    Array(solve(_prob, Tsit5(), saveat = T, abstol = 1e-6, reltol = 1e-6, sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))))
end

# Define the loss function for Neural ODE
function loss_node(θ, T, X_train)
    X̂ = predict_node(θ, X_train[:, 1], T)
    mean(abs2, X_train .- X̂)
end

# Function to train the Neural ODE model for a given time span
function train_node(t_end)
    T = solution.t[solution.t .<= t_end]
    X_train = Xₙ[:, 1:length(T)]
    
    # Define callback to print the current loss
    losses_node = Float64[]
    callback_node = function (p, l)
        push!(losses_node, l)
        if length(losses_node) % 50 == 0
            println("Current NODE loss after $(length(losses_node)) iterations: $l")
        end
        return false
    end
    
    optf_node = Optimization.OptimizationFunction((x, p) -> loss_node(x, T, X_train), Optimization.AutoZygote())
    optprob_node = Optimization.OptimizationProblem(optf_node, ComponentVector{Float64}(p_node))
    res1_node = Optimization.solve(optprob_node, ADAM(), callback = callback_node, maxiters = 500)
    optprob2_node = Optimization.OptimizationProblem(optf_node, res1_node.u)
    res2_node = Optimization.solve(optprob2_node, Optim.LBFGS(), callback = callback_node, maxiters = 100)
    
    return res2_node.u
end

# Function to forecast using the trained Neural ODE model
function node_forecast(p_trained_node, initial_condition, tspan)
    _prob = remake(prob_node, u0 = initial_condition, tspan = tspan, p = p_trained_node)
    sol = solve(_prob, Vern9(), saveat = 0.1, abstol = 1e-6, reltol = 1e-6)
    return sol, Array(sol)
end

# Function to plot the results for Case 1
function plot_case_1_node(p_trained_node, Xₙ, solution)
    train_t = 9.0
    forecast_t = 10.0
    train_solution, train_array = node_forecast(p_trained_node, Xₙ[:, 1], (0.0, train_t))
    forecast_solution, forecast_array = node_forecast(p_trained_node, train_array[:, end], (train_t, forecast_t))
    
    plot(solution.t, X[1, :], label="Underlying Data", xlabel="Time", ylabel="Capital per effective worker", legend=:topleft)
    plot!(train_solution.t, train_array[1, :], label="NODE Training Prediction (t ≤ $train_t)", color=:blue)
    plot!(forecast_solution.t, forecast_array[1, :], label="NODE Forecasting Prediction ($train_t < t ≤ $forecast_t)", color=:red)
    title!("Case 1: Train till t = $train_t, Forecast till t = $forecast_t")
    display(plot)
    savefig("case_1_node.png")  # Save the plot to a file
end

# Function to plot the results for Case 2
function plot_case_2_node(p_trained_node, Xₙ, solution)
    train_t = 7.0
    forecast_t = 10.0
    train_solution, train_array = node_forecast(p_trained_node, Xₙ[:, 1], (0.0, train_t))
    forecast_solution, forecast_array = node_forecast(p_trained_node, train_array[:, end], (train_t, forecast_t))
    
    plot(solution.t, X[1, :], label="Underlying Data", xlabel="Time", ylabel="Capital per effective worker", legend=:topleft)
    plot!(train_solution.t, train_array[1, :], label="NODE Training Prediction (t ≤ $train_t)", color=:blue)
    plot!(forecast_solution.t, forecast_array[1, :], label="NODE Forecasting Prediction ($train_t < t ≤ $forecast_t)", color=:red)
    title!("Case 2: Train till t = $train_t, Forecast till t = $forecast_t")
    display(plot)
    savefig("case_2_node.png")  # Save the plot to a file
end

# Function to plot the results for Case 3
function plot_case_3_node(p_trained_node, Xₙ, solution)
    train_t = 5.0
    forecast_t = 10.0
    train_solution, train_array = node_forecast(p_trained_node, Xₙ[:, 1], (0.0, train_t))
    forecast_solution, forecast_array = node_forecast(p_trained_node, train_array[:, end], (train_t, forecast_t))
    
    plot(solution.t, X[1, :], label="Underlying Data", xlabel="Time", ylabel="Capital per effective worker", legend=:topleft)
    plot!(train_solution.t, train_array[1, :], label="NODE Training Prediction (t ≤ $train_t)", color=:blue)
    plot!(forecast_solution.t, forecast_array[1, :], label="NODE Forecasting Prediction ($train_t < t ≤ $forecast_t)", color=:red)
    title!("Case 3: Train till t = $train_t, Forecast till t = $forecast_t")
    display(plot)
    savefig("case_3_node.png")  # Save the plot to a file
end

# Function to plot the results for Case 4
function plot_case_4_node(p_trained_node, Xₙ, solution)
    train_t = 3.0
    forecast_t = 10.0
    train_solution, train_array = node_forecast(p_trained_node, Xₙ[:, 1], (0.0, train_t))
    forecast_solution, forecast_array = node_forecast(p_trained_node, train_array[:, end], (train_t, forecast_t))
    
    plot(solution.t, X[1, :], label="Underlying Data", xlabel="Time", ylabel="Capital per effective worker", legend=:topleft)
    plot!(train_solution.t, train_array[1, :], label="NODE Training Prediction (t ≤ $train_t)", color=:blue)
    plot!(forecast_solution.t, forecast_array[1, :], label="NODE Forecasting Prediction ($train_t < t ≤ $forecast_t)", color=:red)
    title!("Case 4: Train till t = $train_t, Forecast till t = $forecast_t")
    display(plot)
    savefig("case_4_node.png")  # Save the plot to a file
end

# Function to plot the results for Case 5
function plot_case_5_node(p_trained_node, Xₙ, solution)
    train_t = 1.0
    forecast_t = 10.0
    train_solution, train_array = node_forecast(p_trained_node, Xₙ[:, 1], (0.0, train_t))
    forecast_solution, forecast_array = node_forecast(p_trained_node, train_array[:, end], (train_t, forecast_t))
    
    plot(solution.t, X[1, :], label="Underlying Data", xlabel="Time", ylabel="Capital per effective worker", legend=:topleft)
    plot!(train_solution.t, train_array[1, :], label="NODE Training Prediction (t ≤ $train_t)", color=:blue)
    plot!(forecast_solution.t, forecast_array[1, :], label="NODE Forecasting Prediction ($train_t < t ≤ $forecast_t)", color=:red)
    title!("Case 5: Train till t = $train_t, Forecast till t = $forecast_t")
    display(plot)
    savefig("case_5_node.png")  # Save the plot to a file
end

# Main script to train and plot for each case
for (case_num, train_t) in enumerate([9.0, 7.0, 5.0, 3.0, 1.0])
    println("Training for Case $case_num: Train till t = $train_t")
    p_trained_node = train_node(train_t)
    if case_num == 1
        plot_case_1_node(p_trained_node, Xₙ, solution)
    elseif case_num == 2
        plot_case_2_node(p_trained_node, Xₙ, solution)
    elseif case_num == 3
        plot_case_3_node(p_trained_node, Xₙ, solution)
    elseif case_num == 4
        plot_case_4_node(p_trained_node, Xₙ, solution)
    elseif case_num == 5
        plot_case_5_node(p_trained_node, Xₙ, solution)
    end
end

# Training for Case 1: Train till t = 9.0
#     Current NODE loss after 50 iterations: 0.0004335232335103837
#     Current NODE loss after 100 iterations: 0.00010460112434877047
#     Current NODE loss after 150 iterations: 9.714996116240077e-5
#     Current NODE loss after 200 iterations: 9.087936945268673e-5
#     Current NODE loss after 250 iterations: 8.455167872882977e-5
#     Current NODE loss after 300 iterations: 7.839652694437384e-5
#     Current NODE loss after 350 iterations: 7.259514548494936e-5
#     Current NODE loss after 400 iterations: 6.726529929494214e-5
#     Current NODE loss after 450 iterations: 6.24741285920589e-5
#     Current NODE loss after 500 iterations: 5.8248835653903655e-5
#     Current NODE loss after 550 iterations: 3.8268852859935865e-5
#     Current NODE loss after 600 iterations: 3.817735411920361e-5
#     Training for Case 2: Train till t = 7.0
#     Current NODE loss after 50 iterations: 0.00026694955516122043
#     Current NODE loss after 100 iterations: 8.497319727951959e-5
#     Current NODE loss after 150 iterations: 8.075451393468957e-5
#     Current NODE loss after 200 iterations: 7.751736648173572e-5
#     Current NODE loss after 250 iterations: 7.41374024284654e-5
#     Current NODE loss after 300 iterations: 7.071378837790514e-5
#     Current NODE loss after 350 iterations: 6.734040539857943e-5
#     Current NODE loss after 400 iterations: 6.408891576042984e-5
#     Current NODE loss after 450 iterations: 6.101223985859828e-5
#     Current NODE loss after 500 iterations: 5.814759325937956e-5
#     Current NODE loss after 550 iterations: 3.749496542629167e-5
#     Current NODE loss after 600 iterations: 3.742981868339769e-5
#     Training for Case 3: Train till t = 5.0
#     Current NODE loss after 50 iterations: 7.724449054741712e-5
#     Current NODE loss after 100 iterations: 4.228961385803557e-5
#     Current NODE loss after 150 iterations: 4.019998078135285e-5
#     Current NODE loss after 200 iterations: 3.9808149389616646e-5
#     Current NODE loss after 250 iterations: 3.938937673222317e-5
#     Current NODE loss after 300 iterations: 3.894766283750274e-5
#     Current NODE loss after 350 iterations: 3.8492420371534596e-5
#     Current NODE loss after 400 iterations: 3.8031716209255127e-5
#     Current NODE loss after 450 iterations: 3.7572455398107115e-5
#     Current NODE loss after 500 iterations: 3.712040057186164e-5
#     Current NODE loss after 550 iterations: 3.106381801760344e-5
#     Current NODE loss after 600 iterations: 3.1042812534571986e-5
#     Training for Case 4: Train till t = 3.0
#     Current NODE loss after 50 iterations: 3.421058027637098e-5
#     Current NODE loss after 100 iterations: 3.471559155045924e-5
#     Current NODE loss after 150 iterations: 3.3604501638068675e-5
#     Current NODE loss after 200 iterations: 3.356226424808463e-5
#     Current NODE loss after 250 iterations: 3.351505960855885e-5
#     Current NODE loss after 300 iterations: 3.346355723976739e-5
#     Current NODE loss after 350 iterations: 3.340840073216254e-5
#     Current NODE loss after 400 iterations: 3.335026224127833e-5
#     Current NODE loss after 450 iterations: 3.328962835545127e-5
#     Current NODE loss after 500 iterations: 3.322705025202459e-5
#     Current NODE loss after 550 iterations: 2.7411188457704383e-5
#     Current NODE loss after 600 iterations: 1.6259451898008666e-5
#     Training for Case 5: Train till t = 1.0
#     Current NODE loss after 50 iterations: 2.746293627479554e-5
#     Current NODE loss after 100 iterations: 2.5489232014396515e-5
#     Current NODE loss after 150 iterations: 2.538191135696793e-5
#     Current NODE loss after 200 iterations: 2.5371672281850216e-5
#     Current NODE loss after 250 iterations: 2.5360532662386236e-5
#     Current NODE loss after 300 iterations: 2.534808690711568e-5
#     Current NODE loss after 350 iterations: 2.533442527450681e-5
#     Current NODE loss after 400 iterations: 2.5319622389819867e-5
#     Current NODE loss after 450 iterations: 2.530371204760604e-5
#     Current NODE loss after 500 iterations: 2.5286747160949208e-5
#     Current NODE loss after 550 iterations: 2.1958166530400888e-5
#     Current NODE loss after 600 iterations: 2.180285332078268e-5
