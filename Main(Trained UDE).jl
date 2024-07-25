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

# Define the neural network
const U = Lux.Chain(
    Lux.Dense(1, 9, sigmoid),
    Lux.Dense(9, 1, identity),
    x -> x .^ 2
)

# Initialize the neural network parameters
p, st = Lux.setup(rng, U)
const _st = st

# Define the UDE dynamics
function ude_dynamics!(du, u, p, t, p_true)
    û = U(u, p, _st)[1]
    scaling_factor = 0.1
    du[1] = s * f(u[1]) - scaling_factor * û[1]
end

# Create the UDE problem
nn_dynamics!(du, u, p, t) = ude_dynamics!(du, u, p, t, p_)
prob_nn = ODEProblem(nn_dynamics!, Xₙ[:, 1], tspan, p)

# Define the prediction function
function predict(θ, X = Xₙ[:, 1], T = solution.t)
    _prob = remake(prob_nn, u0 = X, tspan = (T[1], T[end]), p = θ)
    Array(solve(_prob, Vern9(), saveat = T, abstol = 1e-6, reltol = 1e-6, sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))))
end

# Define the loss function
function loss(θ, T, X_train)
    X̂ = predict(θ, X_train[:, 1], T)
    mean(abs2, X_train .- X̂)
end

# Function to train the model for a given time span
function train_ude(t_end)
    T = solution.t[solution.t .<= t_end]
    X_train = Xₙ[:, 1:length(T)]
    
    # Define callback to print the current loss
    losses = Float64[]
    callback = function (p, l)
        push!(losses, l)
        if length(losses) % 50 == 0
            println("Current loss after $(length(losses)) iterations: $l")
        end
        return false
    end
    
    optf = Optimization.OptimizationFunction((x, p) -> loss(x, T, X_train), Optimization.AutoZygote())
    optprob = Optimization.OptimizationProblem(optf, ComponentVector{Float64}(p))
    res1 = Optimization.solve(optprob, ADAM(), callback = callback, maxiters = 500)
    optprob2 = Optimization.OptimizationProblem(optf, res1.u)
    res2 = Optimization.solve(optprob2, Optim.LBFGS(), callback = callback, maxiters = 100)
    
    return res2.u
end

# Function to forecast using the trained model
function ude_forecast(p_trained, initial_condition, tspan)
    _prob = remake(prob_nn, u0 = initial_condition, tspan = tspan, p = p_trained)
    sol = solve(_prob, Vern9(), saveat = 0.1, abstol = 1e-6, reltol = 1e-6)
    return sol, Array(sol)
end

# Function to plot the results for Case 1
function plot_case_1(p_trained, Xₙ, solution)
    train_t = 9.0
    forecast_t = 10.0
    train_solution, train_array = ude_forecast(p_trained, Xₙ[:, 1], (0.0, train_t))
    forecast_solution, forecast_array = ude_forecast(p_trained, train_array[:, end], (train_t, forecast_t))
    
    plot(solution.t, X[1, :], label="Underlying Data", xlabel="Time", ylabel="Capital per effective worker", legend=:topleft)
    plot!(train_solution.t, train_array[1, :], label="UDE Training Prediction (t ≤ $train_t)", color=:blue)
    plot!(forecast_solution.t, forecast_array[1, :], label="UDE Forecasting Prediction ($train_t < t ≤ $forecast_t)", color=:red)
    title!("Case 1: Train till t = $train_t, Forecast till t = $forecast_t")
    display(plot)
    savefig("case_1.png")  # Save the plot to a file
end

# Function to plot the results for Case 2
function plot_case_2(p_trained, Xₙ, solution)
    train_t = 7.0
    forecast_t = 10.0
    train_solution, train_array = ude_forecast(p_trained, Xₙ[:, 1], (0.0, train_t))
    forecast_solution, forecast_array = ude_forecast(p_trained, train_array[:, end], (train_t, forecast_t))
    
    plot(solution.t, X[1, :], label="Underlying Data", xlabel="Time", ylabel="Capital per effective worker", legend=:topleft)
    plot!(train_solution.t, train_array[1, :], label="UDE Training Prediction (t ≤ $train_t)", color=:blue)
    plot!(forecast_solution.t, forecast_array[1, :], label="UDE Forecasting Prediction ($train_t < t ≤ $forecast_t)", color=:red)
    title!("Case 2: Train till t = $train_t, Forecast till t = $forecast_t")
    display(plot)
    savefig("case_2.png")  # Save the plot to a file
end

# Function to plot the results for Case 3
function plot_case_3(p_trained, Xₙ, solution)
    train_t = 5.0
    forecast_t = 10.0
    train_solution, train_array = ude_forecast(p_trained, Xₙ[:, 1], (0.0, train_t))
    forecast_solution, forecast_array = ude_forecast(p_trained, train_array[:, end], (train_t, forecast_t))
    
    plot(solution.t, X[1, :], label="Underlying Data", xlabel="Time", ylabel="Capital per effective worker", legend=:topleft)
    plot!(train_solution.t, train_array[1, :], label="UDE Training Prediction (t ≤ $train_t)", color=:blue)
    plot!(forecast_solution.t, forecast_array[1, :], label="UDE Forecasting Prediction ($train_t < t ≤ $forecast_t)", color=:red)
    title!("Case 3: Train till t = $train_t, Forecast till t = $forecast_t")
    display(plot)
    savefig("case_3.png")  # Save the plot to a file
end

# Function to plot the results for Case 4
function plot_case_4(p_trained, Xₙ, solution)
    train_t = 3.0
    forecast_t = 10.0
    train_solution, train_array = ude_forecast(p_trained, Xₙ[:, 1], (0.0, train_t))
    forecast_solution, forecast_array = ude_forecast(p_trained, train_array[:, end], (train_t, forecast_t))
    
    plot(solution.t, X[1, :], label="Underlying Data", xlabel="Time", ylabel="Capital per effective worker", legend=:topleft)
    plot!(train_solution.t, train_array[1, :], label="UDE Training Prediction (t ≤ $train_t)", color=:blue)
    plot!(forecast_solution.t, forecast_array[1, :], label="UDE Forecasting Prediction ($train_t < t ≤ $forecast_t)", color=:red)
    title!("Case 4: Train till t = $train_t, Forecast till t = $forecast_t")
    display(plot)
    savefig("case_4.png")  # Save the plot to a file
end

# Function to plot the results for Case 5
function plot_case_5(p_trained, Xₙ, solution)
    train_t = 1.0
    forecast_t = 10.0
    train_solution, train_array = ude_forecast(p_trained, Xₙ[:, 1], (0.0, train_t))
    forecast_solution, forecast_array = ude_forecast(p_trained, train_array[:, end], (train_t, forecast_t))
    
    plot(solution.t, X[1, :], label="Underlying Data", xlabel="Time", ylabel="Capital per effective worker", legend=:topleft)
    plot!(train_solution.t, train_array[1, :], label="UDE Training Prediction (t ≤ $train_t)", color=:blue)
    plot!(forecast_solution.t, forecast_array[1, :], label="UDE Forecasting Prediction ($train_t < t ≤ $forecast_t)", color=:red)
    title!("Case 5: Train till t = $train_t, Forecast till t = $forecast_t")
    display(plot)
    savefig("case_5.png")  # Save the plot to a file
end

for (case_num, train_t) in enumerate([9.0, 7.0, 5.0, 3.0, 1.0])
    println("Training for Case $case_num: Train till t = $train_t")
    p_trained = train_ude(train_t)
    if case_num == 1
        plot_case_1(p_trained, Xₙ, solution)
    elseif case_num == 2
        plot_case_2(p_trained, Xₙ, solution)
    elseif case_num == 3
        plot_case_3(p_trained, Xₙ, solution)
    elseif case_num == 4
        plot_case_4(p_trained, Xₙ, solution)
    elseif case_num == 5
        plot_case_5(p_trained, Xₙ, solution)
    end
end

#     Training for Case 1: Train till t = 9.0
#     Current loss after 50 iterations: 0.5612950164515307
#     Current loss after 100 iterations: 0.01286421959843165
#     Current loss after 150 iterations: 0.0035974599138575776
#     Current loss after 200 iterations: 0.0035309847190179073
#     Current loss after 250 iterations: 0.0035274936500598484
#     Current loss after 300 iterations: 0.0035239582984186372
#     Current loss after 350 iterations: 0.003520070260726108
#     Current loss after 400 iterations: 0.0035158449632015517
#     Current loss after 450 iterations: 0.0035112929689750175
#     Current loss after 500 iterations: 0.003506423848790902
#     Current loss after 550 iterations: 3.8951317464031944e-5
#     Current loss after 600 iterations: 3.885096729440605e-5
#     Training for Case 2: Train till t = 7.0
#     Current loss after 50 iterations: 0.26196516937630004
#     Current loss after 100 iterations: 0.005737502768948344
#     Current loss after 150 iterations: 0.0016205907921758762
#     Current loss after 200 iterations: 0.001592096252074493
#     Current loss after 250 iterations: 0.001590792013078768
#     Current loss after 300 iterations: 0.0015894824508041491
#     Current loss after 350 iterations: 0.0015880415286658924
#     Current loss after 400 iterations: 0.0015864754089210946
#     Current loss after 450 iterations: 0.0015847873254135403
#     Current loss after 500 iterations: 0.0015829811627197908
#     Current loss after 550 iterations: 4.001795357174085e-5
#     Current loss after 600 iterations: 3.957781803628519e-5
#     Training for Case 3: Train till t = 5.0
#     Current loss after 50 iterations: 0.09771661738073212
#     Current loss after 100 iterations: 0.0017177960022469424
#     Current loss after 150 iterations: 0.00046058214618744937
#     Current loss after 200 iterations: 0.00045128309690539514
#     Current loss after 250 iterations: 0.00045100618372488296
#     Current loss after 300 iterations: 0.0004507455383660954
#     Current loss after 350 iterations: 0.00045045888778682354
#     Current loss after 400 iterations: 0.00045014690646030445
#     Current loss after 450 iterations: 0.0004498106990433406
#     Current loss after 500 iterations: 0.00044945091158293436
#     Current loss after 550 iterations: 3.2429456839510046e-5
#     Current loss after 600 iterations: 3.2326133294538495e-5
#     Training for Case 4: Train till t = 3.0
#     Current loss after 50 iterations: 0.025014994897195924
#     Current loss after 100 iterations: 0.0003281332845570461
#     Current loss after 150 iterations: 9.910468893228639e-5
#     Current loss after 200 iterations: 9.710738164357531e-5
#     Current loss after 250 iterations: 9.70774787454145e-5
#     Current loss after 300 iterations: 9.705556426707378e-5
#     Current loss after 350 iterations: 9.70314885892531e-5
#     Current loss after 400 iterations: 9.700536155699165e-5
#     Current loss after 450 iterations: 9.697706541454869e-5
#     Current loss after 500 iterations: 9.694679274668514e-5
#     Current loss after 550 iterations: 3.1959062612500325e-5
#     Current loss after 600 iterations: 3.180934160526798e-5
#     Training for Case 5: Train till t = 1.0
#     Current loss after 50 iterations: 0.0022626493113960993
#     Current loss after 100 iterations: 4.02825225953328e-5
#     Current loss after 150 iterations: 2.3354170043914615e-5
#     Current loss after 200 iterations: 2.319306775253447e-5
#     Current loss after 250 iterations: 2.3192132826120485e-5
#     Current loss after 300 iterations: 2.319206995883326e-5
#     Current loss after 350 iterations: 2.319199584201222e-5
#     Current loss after 400 iterations: 2.319192817948368e-5
#     Current loss after 450 iterations: 2.3191842258181137e-5
#     Current loss after 500 iterations: 2.3191747789737693e-5
#     Current loss after 550 iterations: 2.1979458632839823e-5
#     Current loss after 600 iterations: 2.1963641865945956e-5
