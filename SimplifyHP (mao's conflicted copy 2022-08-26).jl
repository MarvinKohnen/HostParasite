using Agents, Random

@agent Grazer GridAgent{2} begin
    energy::Float64
    reproduction_prob::Float64
    Δenergy::Float64
    Infection::Int64
    infection_prob::Float64
end

@agent Grazer GridAgent{2} begin
    energy::Float64
    reproduction_prob::Float64
    Δenergy::Float64
    Infection::Int64
    infection_prob::Float64
end

@agent Parasite GridAgent{2} begin
    energy::Float64
    reproduction_prob::Float64
    Δenergy::Float64
    Infection::Int64
    infection_prob::Float64
end



@agent Copepod GridAgent{2} begin
    energy::Float64
    reproduction_prob::Float64
    Δenergy::Float64
    Infection::Int64
    infection_prob::Float64
end

function initialize_model(;
    n_grazer = 100,
    n_copepods = 50,
    n_parasites = 20,
    dims = (20, 20),
    regrowth_time = 30,
    Δenergy_grazer = 4,
    Δenergy_copepod = 20,
    grazer_reproduce = 0.04,
    copepod_reproduce = 0.05,
    copepod_infection = 0.05,
    seed = 23182,
)

rng = MersenneTwister(seed)
space = GridSpace(dims, periodic = true)
# Model properties contain the algae as two arrays: whether it is fully grown
# and the time to regrow. Also have static parameter `regrowth_time`.
# Notice how the properties are a `NamedTuple` to ensure type stability.
properties = (
    fully_grown = falses(dims),
    countdown = zeros(Int, dims),
    regrowth_time = regrowth_time,
)
model = ABM(Union{Grazer, Copepod}, space;
    properties, rng, scheduler = Schedulers.randomly, warn = false
)
# Add agents

for _ in 1:n_parasites
    energy = rand(model.rng, 1:(Δenergy_grazer*2)) - 1
    add_agent!(Grazer, model, energy, grazer_reproduce, Δenergy_grazer, 0, 0)
end

for _ in 1:n_grazer
    energy = rand(model.rng, 1:(Δenergy_grazer*2)) - 1
    add_agent!(Grazer, model, energy, grazer_reproduce, Δenergy_grazer, 0, 0)
end
for _ in 1:n_copepods
    energy = rand(model.rng, 1:(Δenergy_copepod*2)) - 1
    add_agent!(Copepod, model, energy, copepod_reproduce, Δenergy_copepod, 0, copepod_infection)
end
# Add algae with random initial growth
for p in positions(model)
    fully_grown = rand(model.rng, Bool)
    countdown = fully_grown ? regrowth_time : rand(model.rng, 1:regrowth_time) - 1
    model.countdown[p...] = countdown
    model.fully_grown[p...] = fully_grown
end
return model
end

grazercopepodalgae = initialize_model()

function grazercopepod_step!(grazer::Grazer, model)
    walk!(grazer, rand, model)
    grazer.energy -= 1
    if grazer.energy < 0
        kill_agent!(grazer, model)
        return
    end
    eat!(grazer, model)
    if rand(model.rng) ≤ grazer.reproduction_prob
        reproduce!(grazer, model)
    end
end

function grazercopepod_step!(copepod::Copepod, model)
    walk!(copepod, rand, model)
    copepod.energy -= 1
    if copepod.energy < 0
        kill_agent!(copepod, model)
        return
    end
    # If there is any grazer on this grid cell, it's dinner time!
    dinner = first_grazer_in_position(copepod.pos, model)
    !isnothing(dinner) && eat!(copepod, dinner, model)
    
    if rand(model.rng) ≤ copepod.reproduction_prob
        reproduce!(copepod, model)
    end


    if rand(model.rng) ≤ copepod.reproduction_prob
        reproduce!(copepod, model)
    end
end

function first_grazer_in_position(pos, model)
    ids = ids_in_position(pos, model)
    j = findfirst(id -> model[id] isa Grazer, ids)
    isnothing(j) ? nothing : model[ids[j]]::Grazer
end


function eat!(grazer::Grazer, model)
    if model.fully_grown[grazer.pos...]
        grazer.energy += grazer.Δenergy
        model.fully_grown[grazer.pos...] = false
    end
    return
end

function eat!(copepod::Copepod, grazer::Grazer, model)
    kill_agent!(grazer, model)
    copepod.energy += copepod.Δenergy
    return
end

function reproduce!(agent::A, model) where {A}
    agent.energy /= 2
    id = nextid(model)
    offspring = A(id, agent.pos, agent.energy, agent.reproduction_prob, agent.Δenergy, 0, 0)
    add_agent_pos!(offspring, model)
    return
end

function algae_step!(model)
    @inbounds for p in positions(model) # we don't have to enable bound checking
        if !(model.fully_grown[p...])
            if model.countdown[p...] ≤ 0
                model.fully_grown[p...] = true
                model.countdown[p...] = model.regrowth_time
            else
                model.countdown[p...] -= 1
            end
        end
    end
end



using InteractiveDynamics
using CairoMakie

offset(a) = a isa Grazer ? (-0.1, -0.1*rand()) : (+0.1, +0.1*rand())
ashape(a) = a isa Grazer ? :circle : :utriangle
acolor(a) = a isa Grazer ? RGBAf(1.0, 1.0, 1.0, 0.8) : RGBAf(0.2, 0.2, 0.3, 0.8)

algaecolor(model) = model.countdown ./ model.regrowth_time

heatkwargs = (colormap = [:brown, :green], colorrange = (0, 1))

plotkwargs = (;
    ac = acolor,
    as = 25,
    am = ashape,
    offset,
    scatterkwargs = (strokewidth = 1.0, strokecolor = :black),
    heatarray = algaecolor,
    heatkwargs = heatkwargs,
)

grazercopepodalgae = initialize_model()

fig, ax, abmobs = abmplot(grazercopepodalgae;
    agent_step! = grazercopepod_step!,
    model_step! = algae_step!,
plotkwargs...)
fig



grazer(a) = a isa Grazer
copepod(a) = a isa Copepod
count_algae(model) = count(model.fully_grown)


grazercopepodalgae = initialize_model()
steps = 1000
adata = [(grazer, count), (copepod, count)]
mdata = [count_algae]
adf, mdf = run!(grazercopepodalgae, grazercopepod_step!, algae_step!, steps; adata, mdata)


function plot_population_timeseries(adf, mdf)
    figure = Figure(resolution = (600, 400))
    ax = figure[1, 1] = Axis(figure; xlabel = "Step", ylabel = "Population")
    grazerl = lines!(ax, adf.step, adf.count_grazer, color = :cornsilk4)
    copepodl = lines!(ax, adf.step, adf.count_copepod, color = RGBAf(0.8, 0.2, 0.8))
    algael = lines!(ax, mdf.step, mdf.count_algae, color = :green)
    figure[1, 2] = Legend(figure, [grazerl, copepodl, algael], ["Grazer", "Copepods", "Algae"])
    figure
end

plot_population_timeseries(adf, mdf)

stable_params = (;
    n_grazer = 140,
    n_copepods = 20,
    dims = (30, 30),
    Δenergy_grazer = 5,
    grazer_reproduce = 0.31,
    copepod_reproduce = 0.06,
    Δenergy_copepod = 30,
    seed = 71758,
)

grazercopepodalgae = initialize_model(;stable_params...)
adf, mdf = run!(grazercopepodalgae, grazercopepod_step!, algae_step!, 2000; adata, mdata)
plot_population_timeseries(adf, mdf)