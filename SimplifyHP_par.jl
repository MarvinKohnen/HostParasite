using Agents, Random, Statistics, Distributions



function HDI(samples; credible_mass=0.95)
	# Computes highest density interval from a sample of representative values,
	# estimated as the shortest credible interval
	# Takes Arguments posterior_samples (samples from posterior) and credible mass (normally .95)
	# Originally from https://stackoverflow.com/questions/22284502/highest-posterior-density-region-and-central-credible-region
	# Adapted to Julialang
	sorted_points = sort(samples)
	ciIdxInc = Int(ceil(credible_mass * length(sorted_points)))
	nCIs = length(sorted_points) - ciIdxInc
	ciWidth = repeat([0.0],nCIs)
	for i in range(1, stop=nCIs)
		ciWidth[i] = sorted_points[i + ciIdxInc] - sorted_points[i]
	end
	HDImin = sorted_points[findfirst(isequal(minimum(ciWidth)),ciWidth)]
	HDImax = sorted_points[findfirst(isequal(minimum(ciWidth)),ciWidth)+ciIdxInc]
	return([HDImin, HDImax])
end




function My_Logit(z)
    l = log(z / (1 - (z)))
    return l 
end 




function My_Logistic(x)
    l = 1/(1+exp(-x))
    return l 
end 

My_Logistic(0)
My_Logit(0.5)

@agent Grazer GridAgent{2} begin
    energy::Float64
    reproduction_prob::Float64
    Δenergy::Float64
    Infection::Int64
    # infection_prob::Float64
    α::Float64  # intercep feding rate
    σ::Float64  # variance
    FR::Float64 # feeding rate
end


@agent Parasite GridAgent{2} begin
    energy::Float64
    reproduction_prob::Float64
    Δenergy::Float64
    Infection::Int64
    
    α::Float64
    σ::Float64
    FR::Float64
end



@agent Copepod GridAgent{2} begin
    energy::Float64
    reproduction_prob::Float64
    Δenergy::Float64
    Infection::Int64
    # infection_prob::Float64
    α::Float64
    σ::Float64
    FR::Float64
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
    Feeding_rate = 0.5,
    virulence = 0.01,
    seed = 23182,
)

rng = MersenneTwister(seed)
space = GridSpace(dims, periodic = true)
# Model properties contain the algae as two arrays: whether it is fully grown
# and the time to regrow. Also have static parameter `regrowth_time`.
# Notice how the properties are a `NamedTuple` to ensure type stability.
properties = Dict(
    :fully_grown => falses(dims),
    :countdown => zeros(Int, dims),
    :regrowth_time => regrowth_time,
    
    :n_copepods        => n_copepods,
    :dims              => dims,
    :Δenergy_copepod   => Δenergy_copepod,
    :copepod_reproduce => copepod_reproduce,
    :virulence         => virulence,
    :n_grazer          => n_grazer,
    :grazer_reproduce  => grazer_reproduce,
    :Feeding_rate      => Feeding_rate,
    :Δenergy_grazer    => Δenergy_grazer,
    :n_parasites       => n_parasites,
    :regrowth_time     => regrowth_time,
    :seed              => seed,
)
model = ABM(Union{Grazer, Copepod, Parasite}, space;
    properties = properties , rng, scheduler = Agents.Schedulers.randomly, warn = false
)
# Add agents
# energy::Float64
# reproduction_prob::Float64
# Δenergy::Float64
# Infection::Int64
# infection_prob::Float64
# α::Float64
# σ::Float64
# FR::Float64
for _ in 1:n_parasites
    energy = rand(model.rng, 1:20) - 1
    σ = 0.01 # add individual variation for the rand(Exponential(0.01),1)
    α = virulence# rand(Truncated(Normal(virulence, σ), -3, 3 ), 1))
    FR = 1
    add_agent!(Parasite, model, energy, 0, 15, 0, α, σ, FR)
end

for _ in 1:n_grazer
    energy = rand(model.rng, 1:(Δenergy_grazer*2)) - 1
    σ = 0 # add individual variation for the rand(Exponential(0.01),1)
    α = 0# rand(Truncated(Normal(virulence, σ), -3, 3 ), 1))
    FR = 1
    add_agent!(Grazer, model, energy, grazer_reproduce, Δenergy_grazer, 0,  α, σ, FR)
end


for _ in 1:n_copepods
    energy = rand(model.rng, 1:(Δenergy_copepod*2)) - 1
    σ = 0.01 # add individual variation for the rand(Exponential(0.01),1)
    α = My_Logit(Feeding_rate) # rand(Truncated(Normal(virulence, σ), -3, 3 ), 1))
    FR =My_Logistic.(rand(Truncated(Normal(α, σ), -3, 3 ), 1))[1]
    add_agent!(Copepod, model, energy, copepod_reproduce, Δenergy_copepod, 0,  α, σ, FR)
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

function agent_step!(grazer::Grazer, model)
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



function agent_step!(parasite::Parasite, model)
    walk!(parasite, rand, model)    
    parasite.Infection += 1

    if parasite.Infection ≥ 500
        walk!(parasite, rand, model)
        #print("Parasite walk")
        parasite.energy -= 1
        # if parasite.energy < 0
        #     kill_agent!(parasite, model)
        #   #  print("parasite kill")
        #     return
        # end
    end


end


function agent_step!(copepod::Copepod, model)
   
    walk!(copepod, rand, model)
    copepod.energy -= 1
    if copepod.energy < 0
        kill_agent!(copepod, model)
        return
    end
    

   
    # If there is any grazer on this grid cell, it's dinner time!
   
    dinner = first_grazer_in_position(copepod.pos, model)

    μ =(rand(Truncated(Normal(copepod.α, copepod.σ), -3, 3 ), 1))[1]
    copepod.FR = My_Logistic(μ)
    if rand(model.rng) ≤ copepod.FR
        !isnothing(dinner) && eat!(copepod, dinner, model)
    end 

    # Infect copepods    
    walk!(copepod, rand, model)
   
    infection = first_parasite_in_position(copepod.pos, model)
    if !isnothing(infection) && infection.Infection ≥ 500 
        
        if copepod.Infection == 0
            
            copepod.Infection = 1
            copepod.σ = copepod.σ + infection.α 
            reproduce!(infection, model)
            kill_agent!(infection, model)
            # kill_agent!(infection, model)
            # return

        else
            kill_agent!(infection, model)
        end
      
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


function first_parasite_in_position(pos, model)
    ids = ids_in_position(pos, model)
    j = findfirst(id -> model[id] isa Parasite, ids)
    isnothing(j) ? nothing : model[ids[j]]::Parasite
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
    
    if agent isa Parasite
        for _ in 1:100
            id = nextid(model)
            energy = rand(model.rng, 1:(25*2)) - 1
               σ = 1 # add individual variation for the rand(Exponential(0.01),1)
               α = 1 # rand(Truncated(Normal(virulence, σ), -3, 3 ), 1))
               FR = 1
            offspring = A(id, agent.pos, agent.energy, agent.reproduction_prob, agent.Δenergy, 480, α, σ,  FR)
            add_agent_pos!(offspring, model)
        
        end
        
    else
        id = nextid(model)
        offspring = A(id, agent.pos, agent.energy, agent.reproduction_prob, agent.Δenergy, 0, agent.α, agent.σ,  agent.FR)
        add_agent_pos!(offspring, model)
    end
  
    return
end




function model_step!(model)
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
    agent_step! = agent_step!,
    model_step! = model_step!,
plotkwargs...)
fig



grazer(a) = a isa Grazer
copepod(a) = a isa Copepod
parasite(a) = a isa Parasite
Inf_copepod(a) = a isa Copepod && a.Infection == 1
count_algae(model) = count(model.fully_grown)


grazercopepodalgae = initialize_model()
steps = 10
adata = [(grazer, count), (copepod, count), (Inf_copepod, count), (parasite, count)]
mdata = [count_algae]
adf, mdf = run!(grazercopepodalgae, agent_step!, model_step!, steps; adata, mdata)


function plot_population_timeseries(adf, mdf, title)
    figure = Figure(resolution = (600, 400))
    ax = figure[1, 1] = Axis(figure; xlabel = "Step", ylabel = "Population", title = title)
    ylims!(0,2000)
    grazerl = lines!(ax, adf.step, adf.count_grazer, color = :cornsilk4)
    copepodl = lines!(ax, adf.step, adf.count_copepod, color = RGBAf(0.8, 0.2, 0.8))
    inf_copepodl = lines!(ax, adf.step, adf.count_Inf_copepod, color = RGBAf(0, 0.2, 0.8), lw = 3)
    algael = lines!(ax, mdf.step, mdf.count_algae, color = :green)
    figure[1, 2] = Legend(figure, [grazerl, copepodl, inf_copepodl, algael], ["Grazer", "Copepods", "Inf Cop", "Algae"])
    figure
end






grazercopepodalgae = initialize_model()
n = 10
adata = [(grazer, count), (copepod, count), (Inf_copepod, count), (parasite, count)]
mdata = [count_algae]
adf, mdf = run!(grazercopepodalgae, agent_step!, model_step!, n; adata, mdata)





params = Dict(
    :n_grazer => 300,
    :n_copepods => 100,
    :n_parasites => 10,
    :dims => (50, 50),
    :regrowth_time => 30,
    :Δenergy_grazer => 4,
    :Δenergy_copepod => 30,
    :grazer_reproduce => 0.3,
    :copepod_reproduce => 0.02,
    :Feeding_rate => 0.5,
    :virulence => collect(0:0.1:0.5),
    :seed => rand(UInt8, 10)
)



adata = [(grazer, count), (copepod, count), (Inf_copepod, count), (parasite, count)]
mdata = [count_algae]

adf, mdf = paramscan(params, initialize_model; adata, mdata, agent_step!, model_step!, n = 1000)


using DataFrames

adf.count_algae .= mdf.count_algae

dfp = select(groupby(adf, [:step, :virulence]), [:count_algae, :count_grazer, :count_copepod, :count_Inf_copepod, :count_parasite] .=> [mean std])

dfp[(end-10):end,7:end]

# p2 = plot_population_timeseries(adf, mdf, "Virulence:  σ + 0.8")

# p2
# save("V08.pdf", p2, pt_per_unit = 2) # size = 1600 x 1200 pt

