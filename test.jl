# load modules and functions
using Plots
using DifferentialEquations
using LinearAlgebra
using Statistics
using Random
using Distributions
using DataFrames
using CSV

# read csv file
df = CSV.read("data_2.csv", DataFrame)

# plot step vs count_copepod
plot(df[:,:step], df[:,:count_copepod], label = "Copepod", xlabel = "Step", ylabel = "Count", title = "Copepod Population", legend = :topleft)