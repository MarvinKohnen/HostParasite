#Energy transmission (0.75 from prey to predator) doesnt work, does it?
#Parasite Introduction in model_step function doesnt work -> adding fish, if fish infected, doesnt reproduce, but if it gets infected it produces parasite eggs, also preys on grazers or zooplankton in general
#Size? multiplicate vision radius with size?
#numbers: merles data: some repetitions, almost equal distribution of copepod and grazer (just chydoridae and cyclopods)
#mortality for copepods (simulate sticklebacks)
#mortality for phytoplankton (simulate all other zooplankton)
#have copepods feed on phytoplankton? yes for early stages 
#how to incorporate larval stages of copepod and grazers, give them a mortality as well? yes 50%
#dont stack agents on top of each other in one position
#limit amount of agents in general
#sticklebacks lifespan much longer 
#90% loss of energy each trophic level; metabolic cost  
#validation via ensemble run 
#infected copepods dont escape sticklebacks !!! (3 scenarios: fleeing, not fleeing, actively searching for sticklebacks)
#adding classes of death (dead by fish, dead by energy loss, dead by mortality) ???
#fish are fastest
#movement based on relative sizes 

#"The reductionist approach in ecology is relatively easy to apply if we assume that population members are either identical or that they differ only by sex and age." - include in title?
#allagents(model)
#has_empty_positions(model)
#think about walk funciton


using Random
using Agents
using Agents.Pathfinding
using FileIO
using Distributions
using InteractiveDynamics
using CairoMakie
using GLMakie
using Images
using FileIO
using ImageMagick


mutable struct CopepodGrazerParasitePhytoplankton <: AbstractAgent
    id::Int #Id of the Agent
    pos::NTuple{2, Float64} #position in the Space
    type::Symbol # :Copepod or :Parasite or :Grazer or :Phytoplankton or :Stickleback
    energy::Float64 
    reproduction_prob::Float64  
    Δenergy::Float64
    infected::Bool
    age::Int  # 19.5 days for Macrocyclops albidus
    gender::Int  # 1 = female , 2 = male
    size::Float64 #bigger copepods eat more Grazer and vice versa  
    # -> mean for Macrocyclops albidus: mean (+- SD) in mm: Females: 1.56 +- 0.097 ; males: 1.11 +- 0.093
    # -> Chydoridae: use Alona rectangula, https://onlinelibrary.wiley.com/doi/epdf/10.1111/j.1365-2427.1988.tb01719.x Table 6
end

function Copepod(id, pos, energy, repr, Δe, age, size)
    CopepodGrazerParasitePhytoplankton(id, pos, :copepod, energy, repr, Δe, :false, age, rand(1:2), size)
end

function Parasite(id, pos, energy, repr, Δe, age, size)
    CopepodGrazerParasitePhytoplankton(id, pos, :parasite, energy, repr, Δe,:false, age, 1, size)
end
    
function Grazer(id, pos, energy, repr, Δe, age, size)
    CopepodGrazerParasitePhytoplankton(id, pos, :grazer, energy, repr, Δe, :false, age, rand(1:2), size)
end

function Phytoplankton(id, pos, energy, age)
    CopepodGrazerParasitePhytoplankton(id, pos, :phytoplankton, energy, 0.0, 10, :false, age, 1, 0.01)
end 

function Stickleback(id, pos, repr, size)
    CopepodGrazerParasitePhytoplankton(id, pos, :stickleback, 100, repr, 10, :false, 10, rand(1:2), size)
end

norm(vec) = √sum(vec .^ 2)

function initialize_model(;
    n_copepod = 500,
    n_phytoplankton = 500,
    n_grazer = 500, # Grazer being Chydoridae, Daphniidae and Sididae (All Branchiopoda)
    n_parasite = 200, 
    n_stickleback = 20,
    #n_eggs = 200, #continuous stream of "newly introduced parasites"
    Δenergy_copepod = 96, #4 days
    Δenergy_grazer = 96, #4 days
    Δenergy_parasite = 24,# 2 days   
    #Δenergy_stickleback = 96,
    copepod_vision = 1,  # how far copepods can see grazer to hunt
    grazer_vision = 1,  # how far grazer see phytoplankton to feed on
    parasite_vision = 0.5,  # how far parasites can see copepods to stay in their general vicinit
    stickleback_vision = 3,
    copepod_reproduce = 0.05, #changes if infected, see copepod_reproduce function
    grazer_reproduce = 0.05, #are not infected -> steady reproduction rate
    parasite_reproduce = 0, 
    stickleback_reproduce = 0.05,
    copepod_age = 0,
    grazer_age = 0,
    parasite_age = 0,
    #stickleback_age =0,
    copepod_size = 1,
    grazer_size = 0.5,
    parasite_size = 0.1,
    stickleback_size = 3,
    phytoplankton_age = 0,
    phytoplankton_energy = 0,
    copepod_mortality = 0.05,
    #grazer_mortality = 0.1,
    phytoplankton_mortality = 0.1,
    #stickleback_mortality = 0.2,
    copepod_vel = 0.7,
    grazer_vel = 0.5,
    parasite_vel = 0.2,
    stickleback_vel = 1,
    hatch_prob = 0.20, #probability for eggs to hatch, 20% as to Merles results (Parasite_eggs excel in Dropbox)
    seed = 23182,
    dt = 0.1,    
    )

    rng = MersenneTwister(seed) #MersenneTwister: pseudo random number generator
    space = ContinuousSpace((500., 500.); periodic = false)
    heightmap_path = "C:\\Users\\Marvin\\OneDrive\\Dokumente\\GitHub\\HostParasite\\WhiteSpace.jpg"
    heightmap = load(heightmap_path)
    #heightmap_url = "https://github.com/MarvinKohnen/HostParasite/blob/2D-Beta/WhiteSpace.jpg"
    #heightmap = load(download(heightmap_url))
    dims = (size(heightmap))
    water_walkmap= BitArray(falses(dims))
    

    properties = (
        pathfinder = AStar(space; walkmap = water_walkmap),
        Δenergy_copepod = Δenergy_copepod,
        Δenergy_grazer = Δenergy_grazer,
        Δenergy_parasite =Δenergy_parasite,
        #Δenergy_stickleback = Δenergy_stickleback,
        copepod_vision = copepod_vision,
        grazer_vision = grazer_vision,
        parasite_vision = parasite_vision,
        stickleback_vision = stickleback_vision,
        copepod_reproduce = copepod_reproduce,
        grazer_reproduce = grazer_reproduce,
        parasite_reproduce = parasite_reproduce, 
        stickleback_reproduce = stickleback_reproduce,
        copepod_age = copepod_age,
        grazer_age = grazer_age,
        parasite_age = parasite_age,
        #stickleback_age = stickleback_age,
        copepod_size = copepod_size,
        grazer_size = grazer_size,
        parasite_size = parasite_size,
        stickleback_size = stickleback_size,
        phytoplankton_age = phytoplankton_age,
        phytoplankton_energy = phytoplankton_energy,
        hatch_prob = hatch_prob,
        copepod_mortality = copepod_mortality,
        #grazer_mortality = grazer_mortality,
        phytoplankton_mortality = phytoplankton_mortality,
        #stickleback_mortality = stickleback_mortality,
        copepod_vel = copepod_vel,
        grazer_vel = grazer_vel,
        parasite_vel = parasite_vel,
        stickleback_vel = stickleback_vel,
        dt = dt,
    )
    
    model = ABM(CopepodGrazerParasitePhytoplankton, space; properties, rng, scheduler = Schedulers.randomly)
    
    for _ in 1:n_copepod
        add_agent_pos!(
            Copepod(
                nextid(model),
                random_walkable(random_position(model), model, model.pathfinder, model.copepod_vision),
                rand(1:(Δenergy_copepod*1)) - 1,
                copepod_reproduce,
                Δenergy_copepod,
                0,
                copepod_size,
            ),
            model,
        )
    end

    for _ in 1:n_stickleback
        add_agent_pos!(
            Stickleback(
                nextid(model),
                random_walkable(random_position(model),model, model.pathfinder, model.stickleback_vision),
                stickleback_reproduce,
                stickleback_size,
            ),
            model,
        )
    end
    
    for _ in 1:n_grazer
        add_agent_pos!(
            Grazer(
                nextid(model),
                random_walkable(random_position(model), model, model.pathfinder, model.grazer_vision),
                rand(1:(Δenergy_grazer*1)) - 1,
                grazer_reproduce,
                Δenergy_grazer,
                0,
                grazer_size,
            ),
            model,
        )
    end
    
    for _ in 1:n_parasite
        add_agent_pos!(
            Parasite(
                nextid(model),
                random_walkable(random_position(model), model, model.pathfinder),
                rand(1:(Δenergy_parasite*1)) - 1,
                parasite_reproduce,
                Δenergy_parasite,
                0,
                parasite_size,
            ),
            model,
        )
    end
    
    for _ in 1:n_phytoplankton
        add_agent_pos!(
            Phytoplankton(
                nextid(model),
                random_position(model),
                0,
                rand(1:10),
            ),
            model,
        )
    end
    return model
end
    
function model_step!(agent::CopepodGrazerParasitePhytoplankton, model)
    if agent.type == :copepod 
        copepod_step!(agent, model)
    elseif agent.type == :grazer 
        grazer_step!(agent, model)
    elseif agent.type == :parasite
        parasite_step!(agent, model)
    elseif agent.type == :stickleback
        stickleback_step!(agent, model)
    else 
        phytoplankton_step!(agent, model)
    end
end


function phytoplankton_step!(phytoplankton, model)
    phytoplankton.age += 1
    if phytoplankton.age >= 48 #"a couple of days" e.g. 2 up to 23 days (https://acp.copernicus.org/articles/10/9295/2010/)??? 
        kill_agent!(phytoplankton, model, model.pathfinder)
        return
    end
    if rand(model.rng) < model.phytoplankton_mortality
        kill_agent!(phytoplankton, model, model.pathfinder)
        return
    end
    phytoplankton.energy += 1
    phytoplankton_reproduce!(phytoplankton, model)
end
    
function parasite_step!(parasite, model) #in lab: 2 days max, copepod 4 days max without food 
    parasite.energy -= 1
    if parasite.energy < 0
        kill_agent!(parasite, model, model.pathfinder)
        return
    end
    for _ in rand(5:24)
        walk!(parasite, rand, model)
    end
end


function grazer_step!(grazer, model) 
    grazer_eat!(grazer, model)
    grazer.age += 1
    grazer.energy -=model.dt
    if grazer.energy < 0
        kill_agent!(grazer, model, model.pathfinder)
        return
    end
    #if rand(model.rng) < model.grazer_mortality
        #kill_agent!(grazer, model)
        #return
        
    #end
    
    if rand(model.rng) <= grazer.reproduction_prob * model.dt
        grazer_reproduce!(grazer, model)
    end
        
    predators = [
        x.pos for x in nearby_agents(grazer, model, model.grazer_vision) if 
            x.type == :copepod || x.type == :stickleback
    ]

    if !isempty(predators) && is_stationary(grazer, model.pathfinder)
        direction = (0., 0.)
        away_direction = []
        for i in 1:length(predators)
            if i == 1
                away_direction = (grazer.pos .- predators[i]) 
            else    
                away_direction = away_direction .- predators[i]
            end
            
            #direction away from predators
            all(away_direction .≈ 0.) && continue  #predator at our location -> move anywhere
            direction = direction .+ away_direction ./norm(away_direction) ^2  #set new direction, closer predators contribute more to direction 
        end
        if all(direction .≈ 0.)
            #move anywhere
            chosen_position = random_walkable(grazer.pos, model, model.pathfinder, model.grazer_vision) 
        else
            #Normalize the resultant direction and get the ideal position to move it
            direction = direction ./norm(direction)
            #move to a random position in the general direction of away from predators
            position = grazer.pos .+ direction .* (model.grazer_vision / 2.)
            chosen_position = random_walkable(position, model, model.pathfinder,model.grazer_vision / 2.)
        end
        set_target!(grazer, chosen_position, model.pathfinder)
    end 

    if is_stationary(grazer, model.pathfinder)
        set_target!(
            grazer,
            random_walkable(grazer.pos, model, model.pathfinder, model.grazer_vision),
            model.pathfinder
        )
    end
    move_along_route!(grazer, model, model.pathfinder, model.grazer_vel, model.dt)  
end
 
function copepod_step!(copepod, model) #Copepod is able to detect pray at 1mm (parasties want to stay in that vicinity)
    copepod_eat!(copepod, model)  
    copepod.age += 1
    copepod.energy -= model.dt
    if copepod.energy < 0
        kill_agent!(copepod, model, model.pathfinder)
        return
    end
    if rand(model.rng) < model.copepod_mortality
        kill_agent!(copepod, model)
        return
    end
    if rand(model.rng) <= copepod.reproduction_prob * model.dt
        copepod_reproduce!(copepod, model)
    end

    if is_stationary(copepod, model.pathfinder)  #ADD FLIGHT FROM Stickleback
        prey = [x for x in nearby_agents(copepod, model, model.copepod_vision) if x.type == :grazer && x.age >= 10]
        if isempty(prey)
            #move anywhere if no prey nearby
            set_target!(
                copepod,
                random_walkable(copepod.pos, model, model.pathfinder, model.copepod_vision),
                model.pathfinder
            )
            return
        end
        set_target!(copepod, rand(model.rng, map(x -> x.pos, prey)), model.pathfinder)
    end
    move_along_route!(copepod, model, model.pathfinder, model.copepod_vel, model.dt)
    if copepod.infected == true
        copepod_eat!(copepod, model)
        copepod.energy -= model.dt
        move_along_route!(copepod, model, model.pathfinder, model.copepod_vel, model.dt)
    end
end

function stickleback_step!(stickleback, model) 
    stickleback_eat!(stickleback, model)  
    
    if (rand(model.rng) <= stickleback.reproduction_prob) && (stickleback.infected == true)
        parasite_reproduce!(model)
    end

    if is_stationary(stickleback, model.pathfinder)
        hunt = [x for x in nearby_agents(stickleback, model, model.stickleback_vision) if (x.type == :grazer && x.age >= 10) || (x.type == :copepod && x.age >= 19)] #only eating adult copepods and grazers
        
        if isempty(hunt)
            #move anywhere if no hunt nearby
            set_target!(
                stickleback,
                random_walkable(stickleback.pos, model, model.pathfinder, model.stickleback_vision),
                model.pathfinder
            )
            return
        end
        set_target!(stickleback, rand(model.rng, map(x -> x.pos, hunt)), model.pathfinder)
    end
    move_along_route!(stickleback, model, model.pathfinder, model.stickleback_vel, model.dt)
end

function stickleback_eat!(stickleback, model)
    chase = [x for x in nearby_agents(stickleback, model, model.stickleback_vision) if x.type == :copepod || x.type == :grazer]
    if !isempty(chase)
        for x in chase
            if x.infected == true
                stickleback.infected == true
            end
        kill_agent!(rand(model.rng, chase), model, model.pathfinder)
        end
    end
end

function copepod_eat!(copepod, model) #copepod eat around their general vicinity
    food = [x for x in nearby_agents(copepod, model, model.copepod_vision) if x.type == :grazer]
    if !isempty(food)
        kill_agent!(rand(model.rng, food), model, model.pathfinder)
        copepod.energy += copepod.Δenergy
    end

    infection = [x for x in nearby_agents(copepod, model, model.copepod_vision) if x.type == :parasite]
    if !isempty(infection)
        kill_agent!(rand(model.rng, infection), model, model.pathfinder)
        copepod.infected = true
    end
end

function grazer_eat!(grazer, model)        
    plankton = [x for x in nearby_agents(grazer, model) if x.type == :phytoplankton]
    if !isempty(plankton)
        #plankton = rand(model.rng, phytoplankton)
        grazer.energy += grazer.Δenergy
        kill_agent!(rand(model.rng, plankton), model, model.pathfinder)
    end
end

#Clutch size for Macrocyclops albidus: 72.0 
# add time to grow up: mean time to maturity for Macrocyclops albidus: 19.5 days 
function copepod_reproduce!(copepod, model) 
    if copepod.type == :copepod && copepod.infected == true 
    elseif copepod.gender == 1 && copepod.age > 19
       
        copepod.energy /= 2

        for _ in 1:(rand(Normal(72, 5))) #50% survive
            id = nextid(model)
            offspring = CopepodGrazerParasitePhytoplankton(
                id,
                random_walkable(model, model.pathfinder),
                copepod.type,
                copepod.energy,
                copepod.reproduction_prob,
                copepod.Δenergy,
                :false,
                0,
                rand(1:2),
                copepod.size,
            )
        add_agent_pos!(offspring, model)
        end
    return
    end
end

function grazer_reproduce!(grazer, model) 
    if grazer.gender == 1 && grazer.age > 10
       
        grazer.energy /= 2
        for _ in 1:(rand(Normal(72, 5)))
            id = nextid(model)
            offspring = CopepodGrazerParasitePhytoplankton(
                id,
                random_walkable(model, model.pathfinder),
                grazer.type,
                grazer.energy,
                grazer.reproduction_prob,
                grazer.Δenergy,
                :false,
                0,
                rand(1:2),
                grazer.size,
            )
        add_agent_pos!(offspring, model)
        end
    return
    end
end

function phytoplankton_reproduce!(phytoplankton, model) 
    if phytoplankton.age >= 10
        phytoplankton.energy /= 2

        id = nextid(model)
        offspring = Phytoplankton(
            id,
            random_walkable(random_position(model),model, model.pathfinder),
            0,
            0,
        )
    add_agent_pos!(offspring, model)
    return
    end
end

function parasite_reproduce!(model)
    for _ in 1:rand(1000:500000)
        id = nextid(model)
        if rand(model.rng) <= model.hatch_prob
            eggs = CopepodGrazerParasitePhytoplankton(
                id,
                random_walkable(model, model.pathfinder),
                :parasite,
                rand(1:(Δenergy_parasite))-1,
                parasite_reproduce,
                Δenergy_parasite,
                :false,
                0,
                1,
                parasite_size,
            )
        end
    add_agent!(eggs, model)
    end
return
end

function offset(a)
    a.type == :copepod ? (-0.7, -0.5) : (-0.3, -0.5)
end

function ashape(a)
    if a.type == :copepod 
        :circle 
    elseif a.type == :grazer
        :utriangle
    elseif a.type == :parasite
        :hline
    elseif a.type == :stickleback
        :rect
    else 
        :diamond
    end
end

function acolor(a)
    if a.type == :copepod
        :black 
    elseif a.infected == true  #what else is there to try? 
        :red
    elseif a.type == :grazer 
        :yellow
    elseif a.type == :parasite
        :magenta
    elseif a.type == :stickleback
        :blue
    else
        :green
    end
end

plotkwargs = (
    ac = acolor,
    as = 5,
    am = ashape,
    offset = offset,
)

model = initialize_model()

fig, _ = abm_plot(model; plotkwargs...)
fig

grazer(a) = a.type == :grazer
stickleback(a) = a.type == :copepod
copepodInf(a) = a.type == :copepod && a.infected == true
parasite(a) = a.type == :parasite
phytoplankton(a) = a.type == :phytoplankton
stickleback(a) = a.type == :stickleback

n=5
adata = [(grazer, count), (parasite, count), (phytoplankton, count), (copepodInf, count), (stickleback, count)]
adf = run!(model, model_step!, n; adata)

#adf = adf[1]

using Plots
#plot(adf.count_copepod, adf.count_grazer)

# function plot_population_timeseries(adf)
#     figure = Figure(resolution = (600, 400))
#     ax = figure[1, 1] = Axis(figure; xlabel = "Step", ylabel = "Population")
#     grazerl = lines!(ax, adf.count_grazer, color = :yellow)
#     copepodl = lines!(ax, adf.count_copepod, color = :black)
#     parasitel = lines!(ax,  adf.count_parasite, color = :magenta)
#     phytoplanktonl = lines!(ax,  adf.count_phytoplankton, color = :green)
#     figure[1, 2] = Legend(figure, [grazerl, copepodl, parasitel, phytoplanktonl], ["Grazers", "Copepods", "Parasites", "Phytoplankton"])
#     figure
# end

#plot_population_timeseries(adf)

abm_video(
    "HostParasiteModel.mp4",
    model,
    model_step!;
    frames = 25, 
    framerate = 8,
    plotkwargs...,
)

