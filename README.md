# Revolutionizing Economic Growth Forecasting: Integrating Neural Ordinary Differential Equations and Universal Differential Equations with the Solow-Swan Model


## Overview

This repository contains an implementation of the Solow-Swan model using Universal Differential Equations (UDEs). The Solow-Swan model is a mathematical model used in economics to describe the long-run growth of an economy. UDEs are a type of machine learning model that can be used to learn the dynamics of complex systems.

## Code Structure

The code is written in Julia and is organized into several sections:

*   **Importing Libraries**: The code begins by importing the necessary libraries, including `OrdinaryDiffEq`, `ModelingToolkit`, `DataDrivenDiffEq`, `SciMLSensitivity`, `LinearAlgebra`, `Statistics`, `DataDrivenSparse`, `Optimization`, `OptimizationOptimisers`, `OptimizationOptimJL`, `ComponentArrays`, `Lux`, `Zygote`, `Plots`, and `StableRNGs`.
*   **Defining the Solow-Swan Model**: The Solow-Swan model is defined using the `capital_ode` function, which describes the dynamics of the capital stock over time.
*   **Defining the UDE Model**: The UDE model is defined using the `ude_dynamics` function, which learns the dynamics of the Solow-Swan model.
*   **Training the UDE Model**: The UDE model is trained using the `train_ude` function, which takes the training data and returns the trained model parameters.
*   **Forecasting with the UDE Model**: The trained UDE model is used to make forecasts using the `ude_forecast` function.
*   **Plotting the Results**: The results are plotted using the `plot_case_1`, `plot_case_2`, `plot_case_3`, `plot_case_4`, and `plot_case_5` functions.

## Running the Code

To run the code, simply execute the `main` function. This will train the UDE model and generate plots for each of the five cases.

## Case Studies

The code includes five case studies, each with a different training period:

*   **Case 1**: Train till t = 9.0
*   **Case 2**: Train till t = 7.0
*   **Case 3**: Train till t = 5.0
*   **Case 4**: Train till t = 3.0
*   **Case 5**: Train till t = 1.0

## Requirements

*   Julia 1.7 or higher
*   `OrdinaryDiffEq`, `ModelingToolkit`, `DataDrivenDiffEq`, `SciMLSensitivity`, `LinearAlgebra`, `Statistics`, `DataDrivenSparse`, `Optimization`, `OptimizationOptimisers`, `OptimizationOptimJL`, `ComponentArrays`, `Lux`, `Zygote`, `Plots`, and `StableRNGs` libraries

## Acknowledgments

This code is based on the work of Dr. Raj Abhijit Dandekar, Dr. Rajat Dandekar , Dr. Sreedath Panat and Satwik Sinha.
