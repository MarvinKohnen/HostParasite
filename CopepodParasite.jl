#TO DO:
# Infected copepods?
# Incoming Parasites? New function? maybe hatch function or another reproduce 
# Influence on infected Copepod behaviour apart from "moving more"
# Statistics: add infected copepods somehow 
# only differentiate between phytoplancton and no phytoplancton, count biomass as patcehes where phytocplancton is found

#Source of information for parameters: Maier1994

#Hallo Marius 

using Agents
using Agents.Pathfinding
using Random
using FileIO
#using Distributions
#Normal(5,2) mu sigma
#size= rand(Normal(5,2)n_copepod)
#in for loop:
# size[_]


#reproduction for loop 1:72.5
mutable struct CopepodGrazerParasite <: AbstractAgent
    id::Int #Id of the Agent
    pos::NTuple{3, Float64} #position in the Space
    type::Symbol # :Copepod or :Parasite or :Grazer 
    energy::Float64 
    reproduction_prob::Float64  #Clutch size for Macrocyclops albidus: 72.0 
    Δenergy::Float64
    infected::Bool
    age::Int  # 19.5 days for Macrocyclops albidus
    gender::Int  # 1 = female , 2 = male
    size::Float64 #bigger copepods eat more Grazer and vice versa  
    # -> mean for Macrocyclops albidus: mean (+- SD) in mm: Females: 1.56 +- 0.097 ; males: 1.11 +- 0.093
    # -> Chydoridae: use Alona rectangula, https://onlinelibrary.wiley.com/doi/epdf/10.1111/j.1365-2427.1988.tb01719.x Table 6
end

function Copepod(id, pos, energy, repr, Δe, age, size)
CopepodGrazerParasite(id, pos, :copepod, energy, repr, Δe, :false, rand(1:2), age, size)
end

function Parasite(id, pos, energy, repr, Δe, age, size)
    CopepodGrazerParasite(id, pos, :parasite, energy, repr, Δe,:false, 1, age, size)
end

function Grazer(id, pos, energy, repr, Δe, age, size)
    CopepodGrazerParasite(id, pos, :grazer, energy, repr, Δe, :false, rand(1:2), age, size)
end
norm(vec) = √sum(vec .^ 2)

function initialize_model(
    heightmap_url,
    ground_level= 0,
    water_level = 35,  
    n_copepod = 100,
    n_grazer = 200, # Grazer being Chydoridae, Daphniidae and Sididae (All Branchiopoda)
    n_parasite = 100, #continuous stream of "newly introduced parasites": x amount of bird introduce each: 8000 eggs, only 20% hatch (Merle), check literature 
    Δenergy_copepod = 20, #??? 
    Δenergy_grazer = 20, #??? 
    Δenergy_parasite = 10,#???   
    copepod_vision = 2,  # how far copepods can see grazer to hunt
    grazer_vision = 1,  # how far grazer see phytoplancton to feed on
    parasite_vision = 2,  # how far parasites can see copepods to stay in their general vicinity
    copepod_speed = 1.2,
    parasite_speed = 1.3,
    grazer_speed = 1.0,
    copepod_reproduce = 0.05, #changes if infected, see copepod_reproduce function
    grazer_reproduce = 0.05, #are not infected -> steady reproduction rate
    parasite_reproduce = 0, 
    copepod_age = 0,
    grazer_age = 0,
    parasite_age = 0,
    copepod_size = rand(7:10),
    grazer_size = rand(4:6),
    parasite_size = rand(1:3),
    regrowth_chance = 0.03, #regrowth chance of Phytoplankton in "steps"
    dt = 0.1, #timestep for modle 
    seed = 23182,    
    )
    
    heightmap = floor.(Int, convert.(Float64, load(download(heightmap_url))) * 39) .+ 1
    dims = (size(heightmap)..., 50) #3 Dimensional space
    water_walkmap = BitArray(falses(dims...))

    for i in 1:dims[1], j in 1:dims[2]
        if ground_level < heightmap[i,j] < water_level
            water_walkmap[i, j, heightmap[i, j]+1] = true
        end
    end

    rng = MersenneTwister(seed) #MersenneTwister: pseudo random number generator
    space = ContinuousSpace((100., 100., 50.); periodic = false)
    
    
    phytoplancton = BitArray(           #?? 
        rand(rng, dims[1:2]...) .< (water_level .- heightmap)
    )
        
    properties = (
        pathfinder = AStar(space; walkmap = water_walkmap),
        regrowth_chance = regrowth_chance,
        Δenergy_copepod = Δenergy_copepod,
        Δenergy_grazer = Δenergy_grazer,
        Δenergy_parasite =Δenergy_parasite,
        copepod_vision = copepod_vision,
        grazer_vision = grazer_vision,
        parasite_vision = parasite_vision,
        copepod_speed = copepod_speed,
        parasite_speed = parasite_speed,
        grazer_speed = grazer_speed,
        copepod_reproduce = copepod_reproduce,
        grazer_reproduce = grazer_reproduce,
        parasite_reproduce = parasite_reproduce, 
        copepod_age = copepod_age,
        grazer_age = grazer_age,
        parasite_age = parasite_age,
        copepod_size = copepod_size,
        grazer_size = grazer_size,
        parasite_size = parasite_size,
        dt = dt,
        heightmap = heightmap,
        phytoplancton = phytoplancton,
        water_level = water_level,
        ground_level = ground_level,
    )

    model = ABM(CopepodGrazerParasite, space; properties, rng)
    
    for _ in 1:n_copepod
        add_agent_pos!(
            Copepod(
                nextid(model),
                random_walkable(model, model.pathfinder),
                rand(1:(Δenergy_copepod*2)) - 1,
                copepod_reproduce,
                Δenergy_copepod,
                0,
                copepod_size,
            ),
            model,
        )
    end
    
    for _ in 1:n_grazer
        add_agent_pos!(
            Grazer(
                nextid(model),
                random_walkable(model, model.pathfinder),
                rand(1:(Δenergy_grazer*2)) - 1,
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
                random_walkable(model, model.pathfinder),
                rand(1:(Δenergy_parasite*2)) - 1,
                parasite_reproduce,
                Δenergy_parasite,
                0,
                parasite_size,
            ),
            model,
        )
    end
    return model
end


function model_step!(agent::CopepodGrazerParasite, model)
    if agent.type == :copepod
        copepod_step!(agent, model)
    elseif agent.type == :grazer 
        grazer_step!(agent, model)
    else 
        parasite_step!(agent, model)
    end
end

function parasite_step!(parasite, model) #in lab: 2 days max (Parasites move really quickly, maybe even follow copepods), copepod 4 days max without food 
    parasite.energy -= model.dt     
    if parasite.energy < 0
        kill_agent!(parasite, model, model.pathfinder)
        return
    end
    host = [x for x in nearby_agents(parasite, model, model.parasite_vision) if x.type == :copepod]
    if is_stationary(parasite, model.pathfinder)
        if isempty(host)
            set_target!(
                parasite, 
                random_walkable(parasite.pos, model, model.pathfinder, model.parasite_vision),
                model.pathfinder,
            )
        else 
            set_target!(parasite, rand(model.rng,map(x -> x.pos, host)), model.pathfinder)
        end 
        move_along_route!(parasite, model, model.pathfinder, model.parasite_speed, model.dt)
    end
end

function grazer_step!(grazer, model) 
    grazer_eat!(grazer, model)
    grazer.age += 1
    grazer.energy -= model.dt 
    if grazer.energy < 0
        kill_agent!(grazer, model,model.pathfinder)
        return
    end
    
    if rand(model.rng) <= grazer.reproduction_prob
        reproduce!(grazer, model)
    end
        
    predators = [
        x.pos for x in nearby_agents(grazer, model, model.grazer_vision) if 
            x.type == :copepod
    ]

    if !isempty(predators) && is_stationary(grazer, model.pathfinder)
        direction = (0. , 0., 0.)
        for predator in predators 
            away_direction = (grazer.pos .- predators) #direction away from predators
            all(away_direction .≈ 0.) && continue  #predator at our location -> move anywhere
            direction = direction .+ away_direction ./norm_(away_direction) ^2  #set new direction, closer predators contribute more to direction 
        end
        if all(direction .≈ 0.)
            #move anywhere
            chosen_position = random_walkable(grazer.pos, model, model.pathfinder, model.grazer_vision)
        else
            #Normalize the resultant direction and get the ideal position to move it
            direction = direction ./norm(direction)
            #move to a random position in the general direction of away from predators
            position = grazer.pos .+ direction .* (model.grazer_vision/ 2.)
            chosen_position = random.walkable(position, model, model.pathfinder, model.grazer_vision /2.)
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
    move_along_route!(grazer, model, model.pathfinder, model.grazer_speed, model.dt)
    grazer.age += 1
end
 

function copepod_step!(copepod, model) #Copepod is able to detect pray at 1mm (parasties want to stay in that vicinity)
    food = [x for x in nearby_agents(copepod, model) if x.type == :grazer]
    infection = [x for x in nearby_agents(copepod, model) if x.type == :parasite] 
    copepod_eat!(copepod, food, infection, model)  
    copepod.age += 1
    copepod.energy -= model.dt
    if copepod.energy < 0
        kill_agent!(copepod, model, model.pathfinder)
        return
    end

    if rand(model.rng) <= copepod.reproduction_prob 
        reproduce!(copepod, model)
    end

    if is_stationary(copepod, model.pathfinder)
        prey = [x for x in nearby_agents(copepod, model, model.copepod_vision) if x.type == :grazer]
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
    move_along_route!(copepod, model, model.pathfinder, model.copepod_speed, model.dt)
end

function copepod_eat!(copepod, food, infection, model) #copepod eat around their general vicinity
    if !isempty(food)
        kill_agent!(rand(model.rng, food), model, model.pathfinder)
        copepod.energy += copepod.Δenergy
    end
    if !isempty(infection)
        kill_agent!(rand(model.rng, infection), model, model.pathfinder)
        copepod.infected = true
    end
end

function grazer_eat!(grazer, model)        
    if get_spatial_property(grazer.pos, model.phytoplancton, model) == 1
        model.phytoplancton[get_spatial_index(grazer.pos, model.phytoplancton, model)]=0
        grazer.energy += grazer.Δenergy
    end
end

#Clutch size for Macrocyclops albidus: 72.0 
# add time to grow up: mean time to maturity for Macrocyclops albidus: 19.5 days 
function reproduce!(agent, model) 
    if agent.type == :copepod && agent.infected == true 
    elseif agent.gender == 1
    agent.energy /= 2
    id = nextid(model)
    offspring = CopepodGrazerParasite(
        id,
        agent.pos,
        agent.type,
        agent.energy,
        agent.reproduction_prob,
        agent.Δenergy,
        agent.infected,
        0,
        rand(1:2),
        agent.size,
    )
    add_agent_pos!(offspring, model)
    return
    end
end

function phytoplancton_step!(model)  
    growable = view(
        model.phytocplancton,
        model.phytocplancton .== 0 .& model.ground_level .< model.heightmap .< model.water_level,
    )
    growable .= rand(model.rng, length(growable)) .< model.regrowth_chance * model.dt
end 



using InteractiveDynamics
using CairoMakie
using GLMakie




function acolor(a)
    if a.type == :copepod
        :black 
    elseif (a.type == :copepod) && (a.infected == true)
        :red
    elseif a.type == :grazer 
        :yellow
    else
        :magenta
    end
end

function static_preplot!(ax, model)
    surface!(
        ax,
        (100/205):(100/205):100,
        (100/205):(100/205):100,
        model.heightmap;
        colormap = :terrain
    )
end

heightmap_url =
    "https://raw.githubusercontent.com/JuliaDynamics/" *
    "JuliaDynamics/master/videos/agents/rabbit_fox_hawk_heightmap.png"
model = initialize_model(heightmap_url)

abm_video(
    "CopepodGrazerParasite.mp4",
    model,
    model_step!,
    phytoplancton_step!;
    resolution = (700, 700),
    frames = 300,
    framerate = 20,
    ac = acolor,
    as = 1.0,
    static_preplot!
)