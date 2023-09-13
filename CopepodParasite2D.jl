# upon infection of copepods less evasion of fish after a couple of days (infection days 13! Haferr-Hamann 2019 ) [check]
# initialize fish as some infected some not [check], no parasites for first 20 days
# have infected fish introduce parasites after 20 days 


# virulence = amount of eaten grazers while infected 


# TODO:

# Visualize 






#https://onlinelibrary.wiley.com/doi/epdf/10.1111/j.1461-0248.2006.00995.x  :   independent mortality = death by parasite = 0.05 (p. 49)  

#for grazer use d. dentifera 
#Model specific values:
#   1. Clutch size: M. albidus: 72            ; Grazer: see paper ; Parasites 504-1694 see paper 7                              
#   2. age at maturity : macrocyclops albidus: 19.5 days ; Grazer: see paper 3                                                  done 
#   3. max Lifespan:  Copepods: 45 days (jaime); Grazer: d.dentifera 20 days ; Parasite 4-5 depending on temp (paper 6)         done 
#   4. lifespan without food: macrocyclops: 5 days                                                                              done
#   5. reproduction rate:  m. albidus : once every 7 days at 10 degrees   ; Grazer:  once every 2.5 days                        done       
#   6. Numbers -> use equal numbers 
#   7. Vision radius: absolutely no data
#   8. Velocity: absolutely no data    

#papers: 
#1. p. 346: https://reader.elsevier.com/reader/sd/pii/S0924796397000857?token=BDD92FD299CA569D5058AB729D2CE9429B1905261D75D3CC7176ECFC3EAD4FEE94B5B65E141F97EE5B48C0728F618017&originRegion=eu-west-1&originCreation=20220118110130
#2. p. 827: https://academic.oup.com/plankt/article/9/5/821/1492634?login=true 
#3. p. 81: https://onlinelibrary.wiley.com/doi/epdf/10.1111/j.1365-2427.1988.tb01719.x
#4. https://www.journals.uchicago.edu/doi/abs/10.1899/0887-3593(2004)023%3C0806%3AGRAPDO%3E2.0.CO%3B2
#5. p. 103: https://www.researchgate.net/profile/B-K-Sharma/publication/338108307_Sharma_Sumita_and_Sharma_B_K_1998_Observations_on_the_longevity_instar_durations_fecundity_and_growth_in_Alonella_excisa_Fischer_Cladocera_Chydoridae_Indian_Journal_of_Animal_Sciences_68_101-104/links/5dff30a64585159aa490129b/Sharma-Sumita-and-Sharma-B-K-1998-Observations-on-the-longevity-instar-durations-fecundity-and-growth-in-Alonella-excisa-Fischer-Cladocera-Chydoridae-Indian-Journal-of-Animal-Sciences-68-101-104.pdf
#6. p. 267: https://reader.elsevier.com/reader/sd/pii/S0014489411002815?token=7019A29B14322B0C2D554F79BC309CF9A86E57DC66A0A300A46654D8814B177160946E8151AB7365279B35134BE8F18D&originRegion=eu-west-1&originCreation=20220118120608
#7: p. 1053: https://www.jstor.org/stable/3283228?seq=5#metadata_info_tab_contents
#8: https://link.springer.com/content/pdf/10.1007/BF00006104.pdf
#9 https://www.nature.com/articles/s41598-019-51705-9 for functional stuff and 5 eatings per day 

# using Pkg

# Pkg.add(["Random", "Agents", "FileIO", "Distributions", "InteractiveDynamics", "Images", "ImageMagick", "DataFrames"])

using Random
using Agents
using Agents.Pathfinding
using FileIO
using Distributions
using InteractiveDynamics
#using CairoMakie
#using GLMakie
using Images #use for url load 
using ImageMagick
using DataFrames
using Logging 

#Open new file to log
io = open("CopepodLogging.txt", "w+")

#create logger
logger = SimpleLogger(io)

#initiate Stream and flush old file 
flush(io)

#make logger global so we can use it later
global_logger(logger)


pwd()
#cd("Dropbox/Jaime M/Projects_JM/Muenster/Marvin/project/HostParasite/HostParasite/")

mutable struct CopepodGrazerParasitePhytoplankton <: AbstractAgent
    id::Int 
    pos::NTuple{3, Float64} 
    type::Symbol 
    energy::Float64 
    reproduction_prob::Float64 
    Δenergy::Float64 
    infected::Int   
    infectiondays::Int
    gender::Int  
    fullness::Int 
    age::Int
end

function Copepod(id, pos, energy, repr, Δe)
    CopepodGrazerParasitePhytoplankton(id, pos, :copepod, energy, repr, Δe, 0, 0, rand(1:2),0,0)
end

function Parasite(id, pos, energy, repr, Δe)
    CopepodGrazerParasitePhytoplankton(id, pos, :parasite, energy, repr, Δe,0, 0, 0, 0,0)
end
function Grazer(id, pos, energy, repr, Δe)
    CopepodGrazerParasitePhytoplankton(id, pos, :grazer, energy, repr, Δe, 0, 0, rand(1:2), 0,0)
end

function Phytoplankton(id, pos, energy, repr, Δe, age)
    CopepodGrazerParasitePhytoplankton(id, pos, :phytoplankton, energy, repr, Δe, 0, 0, 1, 0,age)
end 

function Stickleback(id, pos, repr, infected)
    CopepodGrazerParasitePhytoplankton(id, pos, :stickleback, 100, repr, 10, infected, 10, rand(1:2), 0, 0)
end

eunorm(vec) = √sum(vec .^ 2)

function initialize_model(;
    #initial amount of Agents
    n_copepod = 150, 
    n_phytoplankton = 400, 
    n_grazer = 300, 
    n_parasite = 2000,  
    n_stickleback = 20,

    #starting energy 
    starting_energy_copepod = 5,
    starting_energy_grazer = 5,
    starting_energy_parasite = 15,

    # Energy gain on Phytoplankton kill 
    Δenergy_copepod = 0.1, 
    Δenergy_grazer = 0.2, 
    Δenergy_parasite = 0, 
    Δenergy_phytoplankton = 0, 

    #feeding rate
    infected_feeding_rate = 4,
    feeding_rate = 1,

    #Vision Radius
    copepod_vision = 10,
    grazer_vision = 10,
    stickleback_vision = 10,
    parasite_vision = 1,

    #Reproduction probability
    copepod_reproduce = 0.15,
    grazer_reproduce = 0.2, 
    phytoplankton_reproduce = 0.14, #0.10
    stickleback_reproduce = 1.0, 
    parasite_reproduce = 0, 

    # Mortality rates per day
    copepod_mortality = 0.0001,
    grazer_mortality = 0.0001,
    phytoplankton_mortality = 0.0001,

    # Movement rates 
    copepod_vel = 1.2,
    copepod_infected_vel = 1.0,
    grazer_vel = 1.1,
    parasite_vel = 1.0,
    stickleback_vel = 1.4,

    stickleback_eat_chance = 0.5,

    #logging stuff
    grazers_born = 0,
    grazers_eaten = 0,
    copepods_born = 0,
    copepods_eaten = 0,
    phytoplankton_born = 0,
    phytoplankton_eaten = 0, 
    grazers_energyloss = 0,
    copepods_energyloss = 0,
    model_step_counter = 0,
    hatch_prob = 0.2, #probability for eggs to hatch, 20% as to Merles results (Parasite_eggs/hatching rates excel in Dropbox)
    seed = 23182,
    dt = 1.0,
    )


    rng = MersenneTwister(seed) #MersenneTwister: pseudo random number generator
    space = ContinuousSpace((100., 100., 50.); periodic = true)
    heightmap_path = "WhiteSpace.jpg"
    heightmap = load(heightmap_path)
    dims = (size(heightmap)..., 50)
    @info("dims are " * string(dims))
    water_walkmap= BitArray(trues(dims))
    

    # NEEDS TO BE DICTIONARY
    properties = Dict(
        :pathfinder => AStar(space, walkmap = water_walkmap),
        :n_copepod => n_copepod, 
        :n_phytoplankton => n_phytoplankton, 
        :n_grazer => n_grazer, # Grazer being Chydoridae, Daphniidae and Sididae (All Branchiopoda)
        :n_parasite => n_parasite,
        :n_stickleback => n_stickleback,
        :Δenergy_copepod => Δenergy_copepod,
        :Δenergy_grazer => Δenergy_grazer,
        :Δenergy_parasite => Δenergy_parasite,
        :Δenergy_phytoplankton => Δenergy_phytoplankton,
        :copepod_vision => copepod_vision,
        :grazer_vision => grazer_vision,
        :parasite_vision => parasite_vision,
        :stickleback_vision => stickleback_vision,
        :copepod_reproduce => copepod_reproduce,
        :grazer_reproduce => grazer_reproduce,
        :stickleback_reproduce => stickleback_reproduce,
        :phytoplankton_reproduce => phytoplankton_reproduce,
        :parasite_reproduce => parasite_reproduce,
        :hatch_prob => hatch_prob,
        :copepod_mortality => copepod_mortality,
        :grazer_mortality => grazer_mortality,
        :phytoplankton_mortality => phytoplankton_mortality,
        :copepod_vel => copepod_vel,
        :copepod_infected_vel => copepod_infected_vel,
        :grazer_vel => grazer_vel,
        :parasite_vel => parasite_vel,
        :stickleback_vel => stickleback_vel,
        :stickleback_eat_chance => stickleback_eat_chance,
        :starting_energy_copepod => starting_energy_copepod,
        :starting_energy_parasite => starting_energy_parasite,
        :starting_energy_grazer => starting_energy_grazer,
        :feeding_rate => feeding_rate,
        :infected_feeding_rate => infected_feeding_rate,
        :dt => dt,
        :seed => seed,
        :model_step_counter => model_step_counter,
        :grazers_born => grazers_born,
        :copepods_born => copepods_born,
        :phytoplankton_born => phytoplankton_born, 
        :copepods_eaten => copepods_eaten,
        :grazers_eaten => grazers_eaten,
        :phytoplankton_eaten => phytoplankton_eaten,
        :grazers_energyloss => grazers_energyloss,
        :copepods_energyloss => copepods_energyloss
    )
    
    model = ABM(CopepodGrazerParasitePhytoplankton, space; properties, rng, scheduler = Schedulers.randomly)   
    
    for _ in 1:n_grazer
        add_agent_pos!(
            Grazer(
                nextid(model),
                random_walkable(random_position(model), model, model.pathfinder, model.grazer_vision),
                rand(model.rng,(1*model.starting_energy_grazer:2*model.starting_energy_grazer)),
                grazer_reproduce,
                Δenergy_grazer,
            ),
            model,
        )
    end

    for _ in 1:n_copepod
        add_agent_pos!(
            Copepod(
                nextid(model),
                random_walkable(random_position(model), model, model.pathfinder, model.copepod_vision),
                rand(model.rng, 0.8*model.starting_energy_copepod:1.2*model.starting_energy_copepod),
                copepod_reproduce,
                Δenergy_copepod,
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
                0,
            ),
            model,
        )
    end
    
    for _ in 1:n_parasite
        add_agent_pos!(
            Parasite(
                nextid(model),
                random_walkable(random_position(model), model, model.pathfinder),
                rand(model.starting_energy_parasite:2*model.starting_energy_parasite),
                parasite_reproduce,
                Δenergy_parasite,
            ),
            model,
        )
    end
    
    for _ in 1:n_phytoplankton
        add_agent_pos!(
            Phytoplankton(
                nextid(model),
                random_position(model),
                rand(Δenergy_phytoplankton:2*Δenergy_phytoplankton),
                phytoplankton_reproduce,
                Δenergy_phytoplankton,
                rand(1:8)
            ),
            model,
        )
    end
    return model
end
    
function agent_step!(agent::CopepodGrazerParasitePhytoplankton, model)  #agent
    if agent.type == :grazer 
        @info("now stepping agent of type grazer with ID: " * string(agent.id))
        grazer_step!(agent, model)
    elseif agent.type == :copepod 
        @info("now stepping agent of type copepod with ID: " * string(agent.id))
        copepod_step!(agent, model)
    elseif agent.type == :parasite
        @info("now stepping agent of type parasite with ID: " * string(agent.id))
        parasite_step!(agent, model)

    elseif agent.type == :stickleback
        @info("now stepping agent of type Stickleback with ID: " * string(agent.id))
        stickleback_step!(agent, model)

    else 
        @info("now stepping agent of type phytoplankton with ID: " * string(agent.id))
        phytoplankton_step!(agent, model)
    end
end

function phytoplankton_step!(phytoplankton, model)
    if rand(model.rng) < model.phytoplankton_mortality
        remove_agent!(phytoplankton, model, model.pathfinder)
        @info("Phytoplankton with ID: " * string(phytoplankton.id) * " died of mortality at model step: " *string(model.model_step_counter))
        return
    end

    phytoplankton.age += 1

    if phytoplankton.age >= 10
        remove_agent!(phytoplankton, model)
    end 

    if rand(model.rng) <= model.phytoplankton_reproduce * model.dt
        phytoplankton_reproduce!(phytoplankton, model)
    end
end
    
function parasite_step!(parasite, model) 
    parasite.energy -= 1
    @info("This parasite with ID: " * string(parasite.id) * " has this much energy left: " * string(parasite.energy))
    if parasite.energy < 0
        remove_agent!(parasite, model, model.pathfinder)
        @info("This grazer with ID: " * string(parasite.id) * " has this much energy left: " * string(parasite.energy))
        return
    end
    walk!(parasite, rand, model) #periodic = false
end


function grazer_step!(grazer, model)
    @info("This Grazer with ID: " * string(grazer.id) * " has this much energy left: " * string(grazer.energy))
    if grazer.energy <= 20
        @info("This Grazer called grazer_eat! with ID: " * string(grazer.id) * ", since he has this much energy left: " * string(grazer.energy))
        grazer_eat!(grazer, model)
    end
    grazer.energy -=model.dt
    @info("This Grazer with ID: " * string(grazer.id) * " has now this much energy left: " * string(grazer.energy) * ", because model.dt was: " * string(model.dt))
    if grazer.energy < 0
        remove_agent!(grazer, model, model.pathfinder)
        @info("This Grazer with ID: " * string(grazer.id) * " died of energyloss  at model step: " * string(model.model_step_counter))
        model.grazers_energyloss += 1
        return
    end

    if rand(model.rng) < model.grazer_mortality
        remove_agent!(grazer, model, model.pathfinder)
        @info("This grazer with ID: " * string(grazer.id) * " died of mortality  at model step:" * string(model.model_step_counter))
        return
    end
    
    if rand(model.rng) <= grazer.reproduction_prob * model.dt && grazer.gender == 1 && grazer.age >= 1
        @info("This grazer with ID: " * string(grazer.id) * " reproduced at model step: " * string(model.model_step_counter))
        grazer_reproduce!(grazer, model)
    end
        
    predators = [x.pos for x in nearby_agents(grazer, model, model.grazer_vision) if x.type == :copepod || x.type == :stickleback]

    if !isempty(predators) && is_stationary(grazer, model.pathfinder)
        @info("This grazer with ID: " * string(grazer.id) * " is fleeing from predators")
        direction = (0., 0., 0.)
        away_direction = []
        for i in eachindex(predators)
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
            gchosen_position = random_walkable(grazer.pos, model, model.pathfinder, model.grazer_vision) 
        else
            #Normalize the resultant direction and get the ideal position to move it
            direction = direction ./norm(direction)
            #move to a random position in the general direction of away from predators
            position = grazer.pos .+ direction .* (model.grazer_vision / 2.)
            gchosen_position = random_walkable(position, model, model.pathfinder, model.grazer_vision / 2.)
        end
        plan_route!(grazer, gchosen_position, model.pathfinder)
    end 

    if is_stationary(grazer, model.pathfinder) && isempty(predators)
        plan_route!(
            grazer,
            random_walkable(grazer.pos, model, model.pathfinder, model.grazer_vision),
            model.pathfinder
        )
        return
    end
    move_along_route!(grazer, model, model.pathfinder, model.grazer_vel, model.dt)  
    grazer.age += 1 
end

function copepod_step!(copepod, model) #Copepod is able to detect pray at 1mm (parasites want to stay in that vicinity)
    copepod.age += 1
    if rand(model.rng) < model.copepod_mortality
        remove_agent!(copepod, model, model.pathfinder)
        @info("This copepod with ID: " * string(copepod.id) * " died of Mortality  at model step:" * string(model.model_step_counter))
        return
    end

    if rand(model.rng) <= copepod.reproduction_prob * model.dt && copepod.infected == 0 && copepod.gender == 1 && copepod.age >= 2
        @info("This Copepod with ID: " * string(copepod.id) * " reproduced at model step: " * string(model.model_step_counter))
        copepod_reproduce!(copepod, model)
    end

    if copepod.energy <= 3
        @info("This Copepod called copepod_eat! with ID: " * string(copepod.id) * ", since he has this much energy left: " * string(copepod.energy))
        copepod_eat!(copepod, model)
    end
    copepod.energy -= model.dt
    @info("This Copepod with ID: " * string(copepod.id) * ", has now this much energy left " * string(copepod.energy))
    
    if copepod.energy < 0
        remove_agent!(copepod, model, model.pathfinder)
        @info("This Copepod with ID: " * string(copepod.id) * ", died of energyloss at model step: " * string(model.model_step_counter))
        model.copepods_energyloss += 1
        return
    end
    
    if copepod.infected == 0 
        #get an iterable of nearby prey and predators
        prey = [g.pos for g in nearby_agents(copepod, model, model.copepod_vision) if g.type == :grazer]
        cpredators = [s.pos for s in nearby_agents(copepod, model, model.copepod_vision) if s.type == :stickleback]
       
        #if there are predators nearby, override any other movement and flee from it 
        if !isempty(cpredators) 
            # Try and get an ideal direction away from predators
            direction = (0., 0., 0.)
            for predator in cpredators
                # Get the direction away from the predator
                away_direction = (copepod.pos .- predator)
                # In case there is already a predator at our location, moving anywhere is
                # moving away from it, so it doesn't contribute to `direction`
                all(away_direction .≈ 0.) && continue
                # Add this to the overall direction, scaling inversely with distance.
                # As a result, closer predators contribute more to the direction to move in
                direction = direction .+ away_direction ./ eunorm(away_direction) ^ 2
            end
           
            if all(direction .≈ 0.)

                chosen_position = random_walkable(copepod.pos, model, model.pathfinder, model.copepod_vision)
            else
                
                direction = direction ./ eunorm(direction)
                
                position = copepod.pos .+ direction .* (model.copepod_vision / 2.)
                chosen_position = random_walkable(position, model, model.pathfinder, model.copepod_vision / 2.)
            end
            plan_route!(copepod, chosen_position, model.pathfinder)
        end
                
        #only if there are no predators nearby, the copepod will look for prey
        if !isempty(prey) && isempty(cpredators)
            direction = (0., 0., 0.)
            for agent in prey
                toward_direction = (copepod.pos .+ agent)
                direction = direction .+ toward_direction ./ eunorm(toward_direction) ^ 2
            end

            if all(direction .≈ 0.)

                chosen_position = random_walkable(copeod.pos, model, model.pathfinder, model.copepod_vision)

            else
                
                direction = direction ./ eunorm(direction)
                
                position = copepod.pos .+ direction .* (model.copepod_vision / 2.)
                chosen_position = random_walkable(position, model, model.pathfinder, model.copepod_vision / 2.)
            end
            plan_route!(copepod, chosen_position, model.pathfinder)
        end

        #if no prey and no predator nearby, move to a random location
        if is_stationary(copepod, model.pathfinder)
            plan_route!(
                copepod,
                random_walkable(copepod.pos, model, model.pathfinder, model.copepod_vision),
                model.pathfinder,
            )
        end

    elseif copepod.infected == 1 && copepod.infectiondays < 13 

        #get an iterable of nearby prey and predators
        prey = [g.pos for g in nearby_agents(copepod, model, model.copepod_vision) if g.type == :grazer]
        cpredators = [s.pos for s in nearby_agents(copepod, model, model.copepod_vision) if s.type == :stickleback]
       
        #if there are predators nearby, override any other movement and flee from it 
        if !isempty(cpredators) 
            # Try and get an ideal direction away from predators
            direction = (0., 0., 0.)
            for predator in cpredators
                # Get the direction away from the predator
                away_direction = (copepod.pos .- predator)
                # In case there is already a predator at our location, moving anywhere is
                # moving away from it, so it doesn't contribute to `direction`
                all(away_direction .≈ 0.) && continue
                # Add this to the overall direction, scaling inversely with distance.
                # As a result, closer predators contribute more to the direction to move in
                direction = direction .+ away_direction ./ eunorm(away_direction) ^ 2
            end
           
            if all(direction .≈ 0.)

                chosen_position = random_walkable(copepod.pos, model, model.pathfinder, model.copepod_vision)
            else
                
                direction = direction ./ eunorm(direction)
                
                position = copepod.pos .+ direction .* (model.copepod_vision / 2.)
                chosen_position = random_walkable(position, model, model.pathfinder, model.copepod_vision / 2.)
            end
            plan_route!(copepod, chosen_position, model.pathfinder)
        end
                
        #only if there are no predators nearby, the copepod will look for prey
        if !isempty(prey) && isempty(cpredators)
            direction = (0., 0., 0.)
            for agent in prey
                toward_direction = (copepod.pos .+ agent)
                direction = direction .+ toward_direction ./ eunorm(toward_direction) ^ 2
            end

            if all(direction .≈ 0.)

                chosen_position = random_walkable(copeod.pos, model, model.pathfinder, model.copepod_vision)

            else
                
                direction = direction ./ eunorm(direction)
                
                position = copepod.pos .+ direction .* (model.copepod_vision / 2.)
                chosen_position = random_walkable(position, model, model.pathfinder, model.copepod_vision / 2.)
            end
            plan_route!(copepod, chosen_position, model.pathfinder)
        end

        #if no prey and no predator nearby, move to a random location
        if is_stationary(copepod, model.pathfinder)
            plan_route!(
                copepod,
                random_walkable(copepod.pos, model, model.pathfinder, model.copepod_vision),
                model.pathfinder,
            )
        end
        copepod_eat!(copepod, model)
        copepod.infectiondays +=1
        @info("copepod with ID: " * string(copepod.id) * " is now infected for " * string(copepod.infectiondays) * "steps")

    # if the copepod is infected for more than 12 days 
    else 
        #get an iterable of nearby prey and predators
        prey = [g.pos for g in nearby_agents(copepod, model, model.copepod_vision) if g.type == :grazer]
        cpredators = [s.pos for s in nearby_agents(copepod, model, model.copepod_vision) if s.type == :stickleback]

        if !isempty(cpredators) 
            # Try and get an ideal direction away from predators
            direction = (0., 0., 0.)
            for predator in cpredators
                # Get the direction away from the predator
                away_direction = (copepod.pos .- predator)
                # In case there is already a predator at our location, moving anywhere is
                # moving away from it, so it doesn't contribute to `direction`
                all(away_direction .≈ 0.) && continue
                # Add this to the overall direction, scaling inversely with distance.
                # As a result, closer predators contribute more to the direction to move in
                direction = direction .+ away_direction ./ eunorm(away_direction) ^ 2
            end

            #the infected copepod exudes risk behaviour
            if !isempty(prey) 
                direction = (0.,0.,0.)
                for agent in prey
                    toward_direction = (copepod.pos .+ agent)
                    direction = direction .+ toward_direction ./ eunorm(toward_direction) ^ 2
                end
            end 

            if all(direction .≈ 0.)

                chosen_position = random_walkable(copeod.pos, model, model.pathfinder, model.copepod_vision)
            else
                
                direction = direction ./ eunorm(direction)
                
                position = copepod.pos .+ direction .* (model.copepod_vision / 2.)
                chosen_position = random_walkable(position, model, model.pathfinder, model.copepod_vision / 2.)
            end
            plan_route!(copepod, chosen_position, model.pathfinder)
        end

        if !isempty(prey) && isempty(cpredators)
            direction = (0.,0.,0.)
            for agent in prey
                toward_direction = (copepod.pos .+ agent)
                direction = direction .+ toward_direction ./ eunorm(toward_direction) ^ 2
            end

            if all(direction .≈ 0.)

                chosen_position = random_walkable(copeod.pos, model, model.pathfinder, model.copepod_vision)
            else
                
                direction = direction ./ eunorm(direction)
                
                position = copepod.pos .+ direction .* (model.copepod_vision / 2.)
                chosen_position = random_walkable(position, model, model.pathfinder, model.copepod_vision / 2.)
            end
            plan_route!(copepod, chosen_position, model.pathfinder)
        end

        #if no prey and no predator nearby, move to a random location
        if is_stationary(copepod, model.pathfinder)
            plan_route!(
                copepod,
                random_walkable(copepod.pos, model, model.pathfinder, model.copepod_vision),
                model.pathfinder,
            )
        end
        move_along_route!(copepod, model, model.pathfinder, model.copepod_infected_vel, model.dt) 
        copepod_eat!(copepod, model)
        copepod.infectiondays +=1
        @info("copepod with ID: " * string(copepod.id) * " is now infected for " * string(copepod.infectiondays) * "steps")
    end

end



function stickleback_step!(stickleback, model)
    @info("stepping stickleback with ID: " *string(stickleback.id)) 
    if (stickleback.infected == 1) && (rand(model.rng) <= stickleback.reproduction_prob) 
        parasite_reproduce!(model)
        stickleback.infected = 0
        @info("Stickleback with ID: " * string(stickleback.id) * " is no longer infected")
    end

    if is_stationary(stickleback, model.pathfinder)
        hunt = [x for x in nearby_agents(stickleback, model, model.stickleback_vision) if x.type == :copepod || x.type == :grazer]
        if isempty(hunt)
          
            plan_route!(
                stickleback,
                random_walkable(stickleback.pos, model, model.pathfinder, model.stickleback_vision),
                model.pathfinder,
            )
        else
            # Move toward a random agent in hunt
            plan_route!(stickleback, rand(model.rng, map(x -> x.pos, hunt)), model.pathfinder)
        end   
    end

    move_along_route!(stickleback, model, model.pathfinder, model.stickleback_vel, model.dt) 

    if rand(model.rng) <= model.stickleback_eat_chance
        stickleback_infection!(stickleback, model)
        stickleback_eat!(stickleback, model) 
    end 
    
end
function stickleback_infection!(stickleback, model)
    
    infection = [y for y in nearby_agents(stickleback, model, 1.01 * model.stickleback_vision) if y.infected == 1]
    infection_possible = [y for y in nearby_agents(stickleback, model, 1.01 * model.stickleback_vision) if y.type == :copepod]
    counter = 0
    counter_p = 0 
    for y in infection
        counter += 1
    end
    for y in infection_possible
        counter_p += 1
    end
    @info("infection array contained this many elements: " * string(counter))
    @info("infection possible array contained this many elements: " * string(counter_p))
    if !isempty(infection)
        @info("Stickleback with ID: " * string(stickleback.id) * " got infected")
        for y in infection
            stickleback.infected = 1
        end
    end
end

function stickleback_eat!(stickleback, model)

    eat = [x for x in nearby_agents(stickleback, model) if x.type == :copepod || x.type == :grazer]
    if !isempty(eat)
        counter = 0 
        for x in eat
            counter += 1
            @info("This agent of type:" * string(x.type) * " with ID: " * string(x.id) * " at position" * string(x.pos) * " was eaten by Stickleback with ID: " * string(stickleback.id) * " at position" * string(stickleback.pos) * " at model step: " * string(model.model_step_counter) * " at age: " * string(x.age))
            remove_agent!(x, model, model.pathfinder)
            if x.type == :copepod
                model.copepods_eaten +=1 
            end
            if x.type == :grazer
                model.grazers_eaten += 1
            end 
        end
        @info("Stickleback with ID: " * string(stickleback.id) * "at position " * string(stickleback.pos) * "ate in total: " * string(counter) * " Agents")
    end
end 

function copepod_eat!(copepod, model) 
    if copepod.infected == 1
        r = model.infected_feeding_rate
    end

    if copepod.infected == 0 
        r = model.feeding_rate
    end 

    food = [x for x in nearby_agents(copepod, model, r) if (x.type == :grazer) || (x.type == :phytoplankton)]
    if !isempty(food)  
        counter = 0
        for x in food 

            if copepod.fullness >= 10
                @info("Copepod with I: " * string(copepod.id) * "is full and wont be eating any more grazers")
                return
            end 

            if x.type == :grazer
                copepod.energy += x.energy * 0.5
                copepod.fullness += 1 
                model.grazers_eaten += 1
                counter +=1
                remove_agent!(x, model, model.pathfinder)
                @info("This agent of type:" * string(x.type) * " with ID: " * string(x.id) * " at position" * string(x.pos) * " was eaten by Copepod with ID: " * string(copepod.id) * " at position" * string(copepod.pos) * " at model step: " * string(model.model_step_counter) * " at prey age: " * string(x.age) * "and Copepod gained " * string(x.energy*0.5) * " energy")
            end

            if x.type == :phytoplankton
                if copepod.infected == 1 
                    @info("copepod is infected and ignores phytoplankton") 
                else 
                    copepod.energy += copepod.Δenergy
                    counter +=1
                    model.phytoplankton_eaten += 1 
                    remove_agent!(x, model, model.pathfinder)
                    @info("This agent of type:" * string(x.type) * " with ID: " * string(x.id) * " at position" * string(x.pos) * " was eaten by Copepod with ID: " * string(copepod.id) * " at position" * string(copepod.pos) * " at model step: " * string(model.model_step_counter) * " at age: " * string(x.age))
                end
            end

        end
        @info("Copepod with ID: " * string(copepod.id) * "at position " * string(copepod.pos) * "ate in total: " * string(counter) * " grazer")
    end

    if copepod.infected == 0
        copepod_infection_prob = 0.3
        infection = [x for x in nearby_agents(copepod, model) if x.type == :parasite]
        if !isempty(infection)
            for x in infection
                if rand(model.rng) <= copepod_infection_prob
                    remove_agent!(x, model, model.pathfinder)
                    copepod.infected = 1
                    @info("This Copepod with ID: " * string(copepod.id) * " got infected by Parasite with ID: " * string(x.id))
                end 
            end
        end
    end 
end

function grazer_eat!(grazer, model)        
    plankton = [x for x in nearby_agents(grazer, model) if x.type == :phytoplankton]
    if !isempty(plankton)
        counter = 0
        for x in plankton
            model.phytoplankton_eaten += 1
            counter +=1
            grazer.energy += grazer.Δenergy
            remove_agent!(x, model, model.pathfinder)
            @info("Grazer with ID: " * string(grazer.id) * "at position " * string(grazer.pos) * " ate phytoplankton with ID: " * string(x.id) * " at pos: " * string(x.pos) * " at model step: " * string(model.model_step_counter))
        end
        @info("Grazer with ID: " * string(grazer.id) * "at position " * string(grazer.pos) * " ate in total: " * string(counter) * " phytoplankton")
    end
end

function grazer_reproduce!(grazer, model) 
        
    for _ in 1:rand(1:4)
        id = nextid(model)
        offspring = CopepodGrazerParasitePhytoplankton(
            id,
            random_position(model),
            grazer.type,
            model.starting_energy_grazer,
            grazer.reproduction_prob,
            grazer.Δenergy,
            0,
            0,
            rand(1:2),
            0,
            0
        )
    x = offspring.id
    y = offspring.pos
    
    model.grazers_born += 1
    @info("Added new Grazer to model with ID: " * string(x) * ", at pos: " * string(y) * "at model step: " * string(model.model_step_counter))
    add_agent_pos!(offspring, model)
    end
    return
end

#Clutch size for Macrocyclops albidus: 92.0 at 10 degrees
# add time to grow up: mean time to maturity for Macrocyclops albidus: 19.5 days 
function copepod_reproduce!(copepod, model) 

    for _ in 1:(rand(1:4))#(rand(Normal(92, 5))) 
        id = nextid(model)
        offspring = CopepodGrazerParasitePhytoplankton(
            id,
            copepod.pos,
            copepod.type,
            model.starting_energy_copepod,
            copepod.reproduction_prob,
            copepod.Δenergy,
            0,
            0,
            rand(1:2),
            0,
            0
        )
        x = offspring.id
        y = offspring.pos
        
        @info("Added new Copepod to model with ID: " * string(x) * ", at pos: " * string(y) * "at model step: " * string(model.model_step_counter))
        add_agent_pos!(offspring, model)
        model.copepods_born += 1
    end
    return
end


function phytoplankton_reproduce!(phytoplankton, model) 
   

    id = nextid(model)
    offspring = Phytoplankton(
        id,
        random_walkable(random_position(model),model, model.pathfinder),
        0,
        phytoplankton.reproduction_prob,
        phytoplankton.Δenergy,
        0
    )
    x = offspring.id
    y = offspring.pos
    
    model.phytoplankton_born += 1
    @info("Added new Phytoplankton to model with ID: " * string(x) * ", at pos: " * string(y) * "at model step: " * string(model.model_step_counter))
    add_agent_pos!(offspring, model)
    return
end

function parasite_reproduce!(model)
    dry_w = rand(Normal(150,50)) ./ 10000
    epg = 39247 .* dry_w .- 47 
    parasite_reproduce = 0
    Δenergy_parasite = 0
    @info("amount of parasite eggs introduced: " * string(epg))
    for _ in 1:epg
        if rand(model.rng) <= model.hatch_prob
            id = nextid(model)
            egg = Parasite(
                id,
                random_walkable(random_position(model), model, model.pathfinder),
                rand(model.rng, (model.starting_energy_parasite:model.starting_energy_parasite*2)),
                parasite_reproduce,
                Δenergy_parasite,
            )
            x = egg.id
            y = egg.pos
        
            @info("Added new Parasite to model with ID: " * string(x) * ", at pos: " * string(y) * "at model step: " * string(model.model_step_counter))
            add_agent!(egg, model)
        end
    end
    
return
end


function model_step!(model)

    @info("There were this many phytoplankton born: " * string(model.phytoplankton_born) * " at model step: " * string(model.model_step_counter))
    @info("There were this many phytoplankton eaten: " * string(model.phytoplankton_eaten) * " at model step: " * string(model.model_step_counter))
    @info("There were this many grazer born: " * string(model.grazers_born) * " at model step: " * string(model.model_step_counter))
    @info("There were this many grazer eaten: " * string(model.grazers_eaten) * " at model step: " * string(model.model_step_counter))
    @info("There were this many grazer killed by energyloss: " * string(model.grazers_energyloss) * " at model step: " * string(model.model_step_counter))
    @info("There were this many copepod born: " * string(model.copepods_born) * " at model step: " * string(model.model_step_counter))
    @info("There were this many copepod eaten: " * string(model.copepods_eaten) * " at model step: " * string(model.model_step_counter)) 
    @info("There were this many copepod killed by energyloss: " * string(model.copepods_energyloss) * " at model step: " * string(model.model_step_counter))
    @info("Advancing the model one step!")
    model.model_step_counter +=1 
    model.phytoplankton_born = 0
    model.phytoplankton_eaten = 0
    model.grazers_born = 0 
    model.grazers_eaten = 0
    model.copepods_born = 0
    model.copepods_eaten = 0
    model.grazers_energyloss = 0
    model.copepods_energyloss = 0
    
    agents = collect(allagents(model))
    x1 = filter!(x -> x.type == :phytoplankton, agents)
    ids = []

    for i in eachindex(x1)
        push!(ids, x1[i].id) 
    end

    n = length(ids) 
    K =  5000 # carrying capacity
    if n > K

        p_overpopulation = n-K
        @info("had to kill this many agents bc of phyotplankton overpopulation: " * string(p_overpopulation))
        to_kill = rand(1:length(ids), n-K)

        for j in 1:length(to_kill)
            remove_agent!(x1[j], model)
        end
    end   

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
    elseif a.infected == 1  
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

grazer(a) = a.type == :grazer
copepod(a) = a.type == :copepod && a.infected == 0
copepodInf(a) = a.type == :copepod && a.infected == 1
parasite(a) = a.type == :parasite
phytoplankton(a) = a.type == :phytoplankton
stickleback(a) = a.type == :stickleback
sticklebackInf(a) = a.type ==:stickleback && a.infected == 1



#main
n = 60
model = initialize_model()

adata = [(grazer, count), (parasite, count), (phytoplankton, count),(copepod, count), (copepodInf, count), (stickleback, count), (sticklebackInf, count)]
adf = run!(model, agent_step!, model_step!, n; adata)
adf = adf[1]
close(io)

show(adf, allrows=true, allcols=true)

using CSV

CSV.write("datainitial.csv", adf)

# Plotting for Thesis
#using Plots

#t = adf.step 

#Plots.plot(t, (adf.count_copepod), lab = "Copepods")
#plot!(t, (adf.count_phytoplankton), lab = "Phytoplankton")
#plot!(t, (adf.count_grazer), lab = "Grazers")
#plot!(t, (adf.count_stickleback), lab = "Fish")
#plot!(t, (adf.count_parasite), lab = "Parasite")
#plot!(t, (adf.count_copepodInf), lab = "Infected Copepod")
#Plots.ylims!(0,600)


params = Dict(
        :n_copepod => 150, 
        :n_phytoplankton => 400, 
        :n_grazer => 300,
        :n_parasite => 2000,
        :n_stickleback => 20,
        :Δenergy_copepod =>  0.1,
        :Δenergy_grazer => 0.2,
        :Δenergy_parasite => 0,
        :Δenergy_phytoplankton => 0,
        :copepod_vision => 10,
        :grazer_vision => 10,
        :parasite_vision => 1,
        :stickleback_vision => 10,
        :copepod_reproduce => 0.15,
        :grazer_reproduce => 0.2,
        :stickleback_reproduce => 1.0,
        :phytoplankton_reproduce => 0.14,
        :parasite_reproduce => 0,
        :hatch_prob => 0.2,
        :copepod_mortality => 0.0001,
        :grazer_mortality => 0.0001,
        :phytoplankton_mortality => 0.0001,
        :copepod_vel => 1.2,
        :copepod_infected_vel => collect(1.0:0.1:1.5),
        :grazer_vel => 1.1,
        :parasite_vel => 1.0,
        :stickleback_vel => 1.4,
        :stickleback_eat_chance => collect(0.5:0.1:0.7),
        :starting_energy_copepod => 5,
        :starting_energy_parasite => 15,
        :starting_energy_grazer => 5,
        :feeding_rate => 1,
        :infected_feeding_rate => collect(2:10),
        :dt => 1.0,
        :seed => rand(UInt8, 1),
        :model_step_counter => 0,
        :grazers_born => 0,
        :copepods_born => 0,
        :phytoplankton_born => 0, 
        :copepods_eaten => 0,
        :grazers_eaten => 0,
        :phytoplankton_eaten => 0,
        :grazers_energyloss => 0,
        :copepods_energyloss => 0
)



adata = [(grazer, count), (parasite, count), (phytoplankton, count),(copepod, count), (copepodInf, count), (stickleback, count), (sticklebackInf, count)]
bdf = paramscan(params, initialize_model; adata, agent_step!, model_step!, n = 60)


CSV.write("dataParamscan.csv", bdf)

show(bdf, allrows=true, allcols=true)





#global sensitivity -> ensemblerun  : https://github.com/SciML/GlobalSensitivity.jl
#Salib py - sobol julia


