# upon infection of copepods less evasion of fish after a couple of days (infection days 13! Haferr-Hamann 2019 ) [check]
# initialize fish as some infected some not [check], no parasites for first 20 days
# have infected fish introduce parasites after 20 days 


# virulence = amount of eaten grazers while infected 


# TODO:
# Energy gain copepod differ for phytoplankton and grazer (also maybe only prey on grazer?)
# check movement of hunting agents (maybe only chase one agent not all)
# problems: stepping uses random agents -> we should use a rigid order?

#Changed: Stickleback instant reproduction and also only eats infected beings
# Energy parasite high, amount of parasite high 




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

#Logging.disable_logging(Logging.Warn)

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
    id::Int #Id of the Agent
    pos::NTuple{2, Float64} #position in the Space
    type::Symbol # :Copepod or :Parasite or :Grazer or :Phytoplankton or :Stickleback
    energy::Float64 
    reproduction_prob::Float64  
    Δenergy::Float64
    infected::Int   # 0 = false, 1 = true
    infectiondays::Int
    gender::Int  # 1 = female , 2 = male
    fullness::Int 
    age::Int
    # -> mean for Macrocyclops albidus: mean (+- SD) in mm: Females: 1.56 +- 0.097 ; males: 1.11 +- 0.093
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

function Phytoplankton(id, pos, energy, repr, Δe)
    CopepodGrazerParasitePhytoplankton(id, pos, :phytoplankton, energy, repr, Δe, 0, 0, 1, 0,0)
end 

function Stickleback(id, pos, repr, infected)
    CopepodGrazerParasitePhytoplankton(id, pos, :stickleback, 100, repr, 10, infected, 10, rand(1:2), 0, 0)
end


norm(vec) = √sum(vec .^ 2)

function initialize_model(;
    n_copepod = 100, #30, #100
    n_phytoplankton = 400, 
    n_grazer = 160, #100 # Grazer being Chydoridae, Daphniidae and Sididae (All Branchiopoda)
    n_parasite = 4000, #300 
    n_stickleback = 10,

    # Energy parameters
    Δenergy_copepod = 30, #5 days
    Δenergy_grazer = 30, #1 days
    Δenergy_parasite = 50,# 4 days
    Δenergy_phytoplankton = 10,   
    #Δenergy_stickleback = 96,

    copepod_vision = 9,  # how far copepods can see grazer to hunt
    grazer_vision = 1,  # how far grazer see phytoplankton to feed on
    parasite_vision = 2,  # how far parasites can see copepods to stay in their general vicinity
    stickleback_vision = 8, # location to location in grid = 1 
    
    copepod_reproduce = 0.03,
    grazer_reproduce = 0.03,
    parasite_reproduce = 0, 
    phytoplankton_reproduce = 0.2,
    stickleback_reproduce = 1.0, #once per day
    
    phytoplankton_energy = 0,

    # Mortality rates per day
    copepod_mortality = 0.0001,
    grazer_mortality = 0.0001,
    phytoplankton_mortality = 0.0001,
    #stickleback_mortality = 0.2,

    # Movement rates 
    copepod_vel = 1.3,
    grazer_vel = 1.1,
    parasite_vel = 1.0,
    stickleback_vel = 1.4,

    #stickleback_infected = rand((0,1)),
    hatch_prob = 0.2, #probability for eggs to hatch, 20% as to Merles results (Parasite_eggs/hatching rates excel in Dropbox)
    seed = 23182,
    dt = 1.0,
    )

    rng = MersenneTwister(seed) #MersenneTwister: pseudo random number generator
    space = ContinuousSpace((100., 100.); periodic = true)

    heightmap_path = "WhiteSpace.jpg"
    heightmap = load(heightmap_path)
    dims = (size(heightmap))
    water_walkmap= BitArray(falses(dims))
 

    # NEEDS TO BE DICTIONARY
    properties = Dict(
        :pathfinder => AStar(space; walkmap = water_walkmap),
        :n_copepod => n_copepod, 
        :n_phytoplankton => n_phytoplankton, 
        :n_grazer => n_grazer, # Grazer being Chydoridae, Daphniidae and Sididae (All Branchiopoda)
        :n_parasite => n_parasite,
        :n_stickleback => n_stickleback,
        :Δenergy_copepod => Δenergy_copepod,
        :Δenergy_grazer => Δenergy_grazer,
        :Δenergy_parasite => Δenergy_parasite,
        :copepod_vision => copepod_vision,
        :grazer_vision => grazer_vision,
        :parasite_vision => parasite_vision,
        :stickleback_vision => stickleback_vision,
        :copepod_reproduce => copepod_reproduce,
        :grazer_reproduce => grazer_reproduce,
        :parasite_reproduce => parasite_reproduce, 
        :stickleback_reproduce => stickleback_reproduce,
        :phytoplankton_reproduce => phytoplankton_reproduce,
        :Δenergy_phytoplankton => Δenergy_phytoplankton,
        :hatch_prob => hatch_prob,
        :copepod_mortality => copepod_mortality,
        :grazer_mortality => grazer_mortality,
        :phytoplankton_mortality => phytoplankton_mortality,
        :copepod_vel => copepod_vel,
        :grazer_vel => grazer_vel,
        :parasite_vel => parasite_vel,
        :stickleback_vel => stickleback_vel,
        :dt => dt,
        :seed => seed,
    )
    
    model = ABM(CopepodGrazerParasitePhytoplankton, space; properties, rng, scheduler = Schedulers.randomly)  #Random Step order of agents! 
    
    for _ in 1:n_grazer
        add_agent_pos!(
            Grazer(
                nextid(model),
                random_walkable(random_position(model), model, model.pathfinder, model.grazer_vision),
                rand(model.rng,(0.8*Δenergy_grazer:1.2*Δenergy_grazer)),
                grazer_reproduce,
                Δenergy_grazer,
            ),
            model,
        )
        #@info("added Grazer to model with ID: " + Grazer.id + "at Position: " + Grazer.pos)
    end

    for _ in 1:n_copepod
        add_agent_pos!(
            Copepod(
                nextid(model),
                random_walkable(random_position(model), model, model.pathfinder, model.copepod_vision),
                rand(model.rng, 0.8*Δenergy_copepod:1.2*Δenergy_copepod),
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
        #@info("added Stickleback to Map, with Id:" + Stickleback.id + " and position:" + Stickleback.pos)
        #doesnt work here??? why not 
    end
    
    for _ in 1:n_parasite
        add_agent_pos!(
            Parasite(
                nextid(model),
                random_walkable(random_position(model), model, model.pathfinder),
                rand(Δenergy_parasite:2*Δenergy_parasite),
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
            ),
            model,
        )
    end
    return model
end
    
function agent_step!(agent::CopepodGrazerParasitePhytoplankton, model)  #agent
    if agent.type == :grazer 
        grazer_step!(agent, model)
        @info("now stepping agent of type grazer with ID: " * string(agent.id))
    elseif agent.type == :copepod 
        copepod_step!(agent, model)
        @info("now stepping agent of type copepod with ID: " * string(agent.id))
    elseif agent.type == :parasite
        parasite_step!(agent, model)
        @info("now stepping agent of type parasite with ID: " * string(agent.id))
    elseif agent.type == :stickleback
        stickleback_step!(agent, model)
        @info("now stepping agent of type Stickleback with ID: " * string(agent.id))
    else 
        phytoplankton_step!(agent, model)
        @info("now stepping agent of type phytoplankton with ID: " * string(agent.id))
    end
end

#function model_step!(model)
 #   for p in positions(model)
  #      n = length(agents_in_position(p,model))
   #     K = 10
    #    if n>=K
     #       ids = ids_in_position(p, model)
      #      remove_all!(model, rand(ids,(n-K)))
       # end
    #end
#end

function phytoplankton_step!(phytoplankton, model)
    if rand(model.rng) < model.phytoplankton_mortality
        remove_agent!(phytoplankton, model, model.pathfinder)
        return
    end
    phytoplankton.energy += 3
    if rand(model.rng) <= model.phytoplankton_reproduce * model.dt
        phytoplankton_reproduce!(phytoplankton, model)
    end
end
    
function parasite_step!(parasite, model) #in lab: 2 days max, copepod 4 days max without food 
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
    grazer.age += 1 
    @info("This grazer with Id: " * string(grazer.id) * " has this much energy left: " * string(grazer.energy))
    if grazer.energy <= 15
        @info("This grazer called grazer_eat! with ID: " * string(grazer.id) * ", since he has this much energy left: " * string(grazer.energy))
        grazer_eat!(grazer, model)
    end
    grazer.energy -=model.dt
    @info("This grazer with ID: " * string(grazer.id) * " has now this much energy left: " * string(grazer.energy) * ", because model.dt was: " * string(model.dt))
    if grazer.energy < 0
        remove_agent!(grazer, model, model.pathfinder)
        @info("This grazer with ID: " * string(grazer.id) * " died of energy loss")
        return
    end
    if rand(model.rng) < model.grazer_mortality
        remove_agent!(grazer, model)
        @info("This grazer with ID: " * string(grazer.id) * " died of mortality")
        return
    end
    
    if rand(model.rng) <= grazer.reproduction_prob * model.dt
        @info("This grazer with ID: " * string(grazer.id) * " reproduced")
        grazer_reproduce!(grazer, model)
    end
        
    predators = [x.pos for x in nearby_agents(grazer, model, model.grazer_vision) if x.type == :copepod || x.type == :stickleback]

    if !isempty(predators) && is_stationary(grazer, model.pathfinder)
        @info("This grazer with ID: " * string(grazer.id) * " is fleeing from predators")
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
            gchosen_position = random_walkable(grazer.pos, model, model.pathfinder, model.grazer_vision) 
        else
            #Normalize the resultant direction and get the ideal position to move it
            direction = direction ./norm(direction)
            #move to a random position in the general direction of away from predators
            position = grazer.pos .+ direction .* (model.grazer_vision / 2.)
            gchosen_position = random_walkable(position, model, model.pathfinder, model.grazer_vision / 2.)
        end
        set_target!(grazer, gchosen_position, model.pathfinder)
    end 

    if is_stationary(grazer, model.pathfinder) && isempty(predators)
        set_target!(
            grazer,
            random_walkable(grazer.pos, model, model.pathfinder, model.grazer_vision),
            model.pathfinder
        )
        return
    end
    move_along_route!(grazer, model, model.pathfinder, model.grazer_vel, model.dt)  
end

function copepod_step!(copepod, model) #Copepod is able to detect pray at 1mm (parasites want to stay in that vicinity)
    copepod.age += 1
    if rand(model.rng) < model.copepod_mortality
        remove_agent!(copepod, model)
        @info("This copepod with ID: " * string(copepod.id) * " died of Mortality")
        return
    end
    if rand(model.rng) <= copepod.reproduction_prob * model.dt
        copepod_reproduce!(copepod, model)
    end
   
    if copepod.energy <= 15
        @info("This copepod called copepod_eat! with ID: " * string(copepod.id) * ", since he has this much energy left: " * string(copepod.energy))
        copepod_eat!(copepod, model)
    end
    copepod.energy -= model.dt
    @info("This copepod with Id: " * string(copepod.id) * ", has now this much energy left " * string(copepod.energy))
    
    if copepod.energy < 0
        remove_agent!(copepod, model, model.pathfinder)
        @info("This copepod with Id: " * string(copepod.id) * ", died of energyloss")
        return
    end
    @info("infection status copepod: " * string(copepod.infected))
    if copepod.infected == 1
        move_along_route!(copepod, model, model.pathfinder, model.copepod_vel, model.dt)
        copepod_eat!(copepod, model)
        copepod.infectiondays +=1
        @info("copepod with ID: " * string(copepod.id) * " is now infected for " * string(copepod.infectiondays) * "steps")
        copepod.energy -= model.dt
    end
    
    
    if copepod.infectiondays <= 2#12 
        prey = [x.pos for x in nearby_agents(copepod, model, model.copepod_vision) if x.type == :grazer]
        cpredators = [x.pos for x in nearby_agents(copepod, model, model.copepod_vision) if x.type == :stickleback]
        for predator in cpredators
            @info("copepod with ID: " * string(copepod.id) * " is fleeing from the Stickleback with ID: " * string(predator.pos))
        end
        cdirection = (0., 0.)
        caway_direction = (0.,0.)
        if !isempty(cpredators) 
            @info("This copepod with ID: " * string(copepod.id) * ", is fleeing from predators")
            caway_direction = []
            caway_direction = (copepod.pos .- cpredators[1]) 
            chosen_position = random_walkable(caway_direction, model, model.pathfinder, model.copepod_vision / 2.)
        end 
        
        ctoward_direction = (0.,0.)
        if !isempty(prey)
            ctoward_direction = []
            for i in 1:length(prey)
                if i == 1
                    ctoward_direction = (copepod.pos .+ prey[i])
                else
                    ctoward_direction = ctoward_direction .+ prey[i]
                end
            end 
        end
            
        if all(caway_direction .≈ 0.) #meaning the sticklebacks are on top of the copepod
            #move anywhere
            chosen_position = random_walkable(copepod.pos, model, model.pathfinder, model.copepod_vision) 
        else
            #Normalize the resultant direction and get the ideal position to move it
            cdirection = cdirection ./norm(cdirection)
            #move to a random position in the general direction away from predators and toward prey
            cposition = copepod.pos .+ cdirection .* (model.copepod_vision / 2.)
            chosen_position = random_walkable(cposition, model, model.pathfinder, model.copepod_vision / 2.)
        end
        set_target!(copepod, chosen_position, model.pathfinder)
        
        if isempty(prey) && isempty(cpredators)
            #move anywhere if no prey nearby
            set_target!(
                copepod,
                random_walkable(copepod.pos, model, model.pathfinder, model.copepod_vision),
                model.pathfinder
            )
            return
        end

    else #if is_stationary(copepod, model.pathfinder)
        prey = [x.pos for x in nearby_agents(copepod, model, model.copepod_vision) if x.type == :grazer]
        cdirection = (0., 0.)
        ctoward_direction = (0.,0.)
        if !isempty(prey)
            @info("This copepod with ID: " * string(copepod.id) * ", is hunting prey")
            ctoward_direction = []
            for i in 1:length(prey)
                if i == 1
                    ctoward_direction = (copepod.pos .+ prey[i])
                else
                    ctoward_direction = ctoward_direction .+ prey[i]
                end
            end 


            cdirection = cdirection .+ ctoward_direction ./ norm(ctoward_direction) .^2   #set new direction 
                
            if all(cdirection .≈ 0.) #meaning the sticklebacks are on top of the copepod
                #move anywhere
                chosen_position = random_walkable(copepod.pos, model, model.pathfinder, model.copepod_vision) 
            else
                #Normalize the resultant direction and get the ideal position to move it
                cdirection = cdirection ./norm(cdirection)
                #move to a random position in the general direction away from predators and toward prey
                cposition = copepod.pos .+ cdirection .* (model.copepod_vision / 2.) 
                chosen_position = random_walkable(cposition, model, model.pathfinder, model.copepod_vision / 2.)
            end
            set_target!(copepod, chosen_position, model.pathfinder)
        end 

        if isempty(prey) 
            #move anywhere if no prey nearby
            set_target!(
                copepod,
                random_walkable(copepod.pos, model, model.pathfinder, model.copepod_vision),
                model.pathfinder
            )
            return
        end

    move_along_route!(copepod, model, model.pathfinder, model.copepod_vel, model.dt)
end 
end

function stickleback_step!(stickleback, model) 
    
    hunt = [x.pos for x in nearby_agents(stickleback, model, model.stickleback_vision) if  (x.type == :copepod)]
    if is_stationary(stickleback, model.pathfinder) && !isempty(hunt)
        #@info("This Stickleback with Id: " * string(stickleback.id) * ", is hunting the following agents: " )
        sdirection = (0., 0.)
        
        stoward_direction = []
        for i in 1:length(hunt)

            if i == 1
                stoward_direction = (stickleback.pos .+ hunt[i])
            else
                stoward_direction = stoward_direction .+ hunt[i]
            end
            sdirection = sdirection .+ stoward_direction ./ norm(stoward_direction) ^2
        end
        
        sdirection = sdirection ./ norm(sdirection)
        sposition = stickleback.pos .+ sdirection .* (model.stickleback_vision / 2.)
        schosen_position = random_walkable(sposition, model, model.pathfinder, model.stickleback_vision / 2.)
        set_target!(stickleback, schosen_position, model.pathfinder)
    end

    if is_stationary(stickleback, model.pathfinder) && isempty(hunt)
        set_target!(
            stickleback,
            random_walkable(stickleback.pos, model, model.pathfinder, model.stickleback_vision),
            model.pathfinder
        )
        return
    end
    move_along_route!(stickleback, model, model.pathfinder, model.stickleback_vel, model.dt) 


    stickleback_eat!(stickleback, model)  
    if (stickleback.infected == 1) && (rand(model.rng) <= stickleback.reproduction_prob) 
        parasite_reproduce!(model)
        stickleback.infected = 0
    end
    
end


function stickleback_eat!(stickleback, model)
    infection = [y for y in nearby_agents(stickleback, model, model.stickleback_vision) if y.infected == 1]
    eat = [x for x in nearby_agents(stickleback, model, model.stickleback_vision) if x.type == :copepod || x.type == :grazer]
    if !isempty(infection)
        stickleback.infected = 1
        @info("Stickleback with ID: " * string(stickleback.id) * "got infected")
    end

    if !isempty(eat)
        counter = 0 
        for x in eat
            counter += 1
            @info("This agent of type:" * string(x.type) * " and ID: " * string(x.id) * " at position" * string(x.pos) * " was eaten by Stickleback with ID: " * string(stickleback.id) * " at position" * string(stickleback.pos))
            remove_agent!(x, model, model.pathfinder)
        end
        @info("Stickleback with ID: " * string(stickleback.id) * "at position " * string(stickleback.pos) * "ate in total: " * string(counter) * " Agents")
    end

    
end

function copepod_eat!(copepod, model) 
    food = [x for x in nearby_agents(copepod, model, model.copepod_vision) if (x.type == :grazer)]# || (x.type == :phytoplankton)]
    if !isempty(food)  
        counter = 0
        for x in food 
            counter +=1
            if x.type == :grazer  
                copepod.fullness += 1 
            end
            @info("This agent of type:" * string(x.type) * " and ID: " * string(x.id) * " at position" * string(x.pos) * " was eaten by Copepod with ID: " * string(copepod.id) * " at position" * string(copepod.pos))
            remove_agent!(x, model, model.pathfinder)
            copepod.energy += 5
        end
        @info("Copepod with ID: " * string(copepod.id) * "at position " * string(copepod.pos) * "ate in total: " * string(counter) * " grazer")

    end


    infection = [x for x in nearby_agents(copepod, model, model.copepod_vision) if x.type == :parasite]
    if !isempty(infection)
        for x in infection
            remove_agent!(x, model, model.pathfinder)
            
            copepod.infected = 1
        end
        @info("This Copepod with ID: " * string(copepod.id) * " got infected")
    end
end

function grazer_eat!(grazer, model)        
    plankton = [x for x in nearby_agents(grazer, model) if x.type == :phytoplankton]
    if !isempty(plankton)
        counter = 0
        for x in plankton
            counter +=1
            grazer.energy += 3
            remove_agent!(x, model, model.pathfinder)
            grazer_pos = grazer.pos
            plankton_pos = x.pos
            @info("Grazer with ID: " * string(grazer.id) * "at position " * string(grazer_pos) * "ate phytoplankton at pos: " * string(plankton_pos))
        end
        @info("Grazer with ID: " * string(grazer.id) * "at position " * string(grazer_pos) * "ate in total: " * string(counter) * " phytoplankton")
    end
end

function grazer_reproduce!(grazer, model) 
    if grazer.gender == 1
        
        #grazer.energy /= 2
        for _ in 1:rand(1:5)
            id = nextid(model)
            offspring = CopepodGrazerParasitePhytoplankton(
                id,
                random_position(model),
                grazer.type,
                model.Δenergy_grazer,
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
        
        @info("Added new Grazer to model with ID: " * string(x) * ", at pos: " * string(y))
        add_agent_pos!(offspring, model)
        end
        return
    end
end

#Clutch size for Macrocyclops albidus: 92.0 at 10 degrees
# add time to grow up: mean time to maturity for Macrocyclops albidus: 19.5 days 
function copepod_reproduce!(copepod, model) 
    #if copepod.type == :copepod && copepod.infected == 1
    #elseif
    if copepod.gender == 1 
       
        #copepod.energy /= 2

        for _ in 1:(rand(1:5))#(rand(Normal(92, 5))) 
            id = nextid(model)
            offspring = CopepodGrazerParasitePhytoplankton(
                id,
                copepod.pos,
                copepod.type,
                copepod.energy,
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
        
        @info("Added new Copepod to model with ID: " * string(x) * ", at pos: " * string(y))
        add_agent_pos!(offspring, model)
        end
    return
    end
end


function phytoplankton_reproduce!(phytoplankton, model) 
   
    phytoplankton.energy /= 2

    id = nextid(model)
    offspring = Phytoplankton(
        id,
        random_walkable(random_position(model),model, model.pathfinder),
        0,
        phytoplankton.reproduction_prob,
        phytoplankton.Δenergy,
    )
    x = offspring.id
    y = offspring.pos
        
    @info("Added new Phytoplankton to model with id: " * string(x) * ", at pos: " * string(y))
    add_agent_pos!(offspring, model)
    Agents.remove_agent!(phytoplankton, model)
    return
end

function parasite_reproduce!(model)
    dry_w = rand(Normal(150,50)) ./ 10000
    epg = 39247 .* dry_w .- 47 
    Δenergy_parasite = 96
    parasite_reproduce = 0
    @info("amount of parasite eggs introduced: " * epg)
    for _ in 1:epg
        if rand(model.rng) <= model.hatch_prob
            id = nextid(model)
            egg = Parasite(
                id,
                random_walkable(random_position(model), model, model.pathfinder),
                rand(model.rng, (Δenergy_parasite:Δenergy_parasite*2)),
                parasite_reproduce,
                Δenergy_parasite,
            )
            x = egg.id
            y = egg.pos
        
            @info("Added new Parasite to model with ID: " * string(x) * ", at pos: " * string(y))
            add_agent!(egg, model)
        end
    end

return
end


function model_step!(model)
    @info("Advancing the model one step!")
    agents = collect(allagents(model))
    x1 = filter!(x -> x.type == :phytoplankton, agents)
    ids = []

    for i in 1:length(x1)
        push!(ids, x1[i].id) 
    end

    n = length(ids) 
    K =  5000 # carrying capacity
    if n > K
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
    elseif a.infected == 1  #what else is there to try? 
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
copepodInf(a) = a.type == :copepod
parasite(a) = a.type == :parasite
phytoplankton(a) = a.type == :phytoplankton
stickleback(a) = a.type == :stickleback
sticklebackInf(a) = a.type ==:stickleback && a.infected == 1



#main
n= 60
model = initialize_model()

adata = [(grazer, count), (parasite, count), (phytoplankton, count),(copepod, count), (copepodInf, count), (stickleback, count), (sticklebackInf, count)]
adf = run!(model, agent_step!, model_step!, n; adata)
adf = adf[1]
close(io)

show(adf, allrows=true)




# params = Dict(
#     :n_copepod =>  [1, 300, 500]  ,#collect(0:100:500), # 
#     :n_phytoplankton =>  3000,  #collect(1000:500:4000), 
#     :n_grazer =>  300,  #collect(1:100:500), # Grazer being Chydoridae, Daphniidae and Sididae (All Branchiopoda)
#     :n_parasite => 2000, #collect(1:1000:3000),
#     :n_stickleback => 100, #[1:10:40],
#     :Δenergy_copepod => 24*5,
#     :Δenergy_grazer => 24,
#     :Δenergy_parasite => 24*4,
#     #Δenergy_stickleback = Δenergy_stickleback,
#     :copepod_vision => 4,
#     :grazer_vision => 2,
#     :parasite_vision => 1,
#     :stickleback_vision => 8, #[4, 6, 8, 10],  #best to always alter one? 
#     :copepod_reproduce => (1/(24)),
#     :grazer_reproduce => (1/(24)),
#     :parasite_reproduce => 0, 
#     :stickleback_reproduce => 0.8,
#     :phytoplankton_reproduce => (1/(24)),
#     :copepod_age => 0,
#     :grazer_age => 0,
#     :parasite_age => 0,
#     :copepod_size => 1,
#     :grazer_size => 1,
#     :parasite_size => 1,
#     :stickleback_size => 3,
#     :phytoplankton_age => 0,
#     :phytoplankton_energy => 0,
#     :hatch_prob => 0.2,
#     :copepod_mortality => (1/24)*0.05,
#     :grazer_mortality => (1/24)*0.1,
#     :phytoplankton_mortality => (1/24) * 0.01,
#     #stickleback_mortality = stickleback_mortality,
#     :copepod_vel => 0.5,
#     :grazer_vel => 0.25,
#     :parasite_vel => 0.1,
#     :stickleback_vel => 0.7, #[0.5, 0.6, 0.7],
#     #:stickleback_infected => rand((0,1))
#     :dt => 1.0,
#     :seed => rand(UInt8, 1),
# )


# params = Dict(
#     :n_copepod => [300], 
#     :n_phytoplankton => 3000, 
#     :n_grazer => 300,
#     :n_parasite => 2000,
#     :n_stickleback => 100,
#     :Δenergy_copepod => 24 * 5,
#     :Δenergy_grazer => 24,
#     :Δenergy_parasite => 24 * 4,
#     :copepod_vision => 4,
#     :grazer_vision => 2,
#     :parasite_vision => 1,
#     :stickleback_vision => 8,
#     :copepod_reproduce => 1 / 24,
#     :grazer_reproduce => 1 / 24,
#     :parasite_reproduce => 0, 
#     :stickleback_reproduce => 0.8,
#     :phytoplankton_reproduce => 1 / 24,
#     :copepod_age => 0,
#     :grazer_age => 0,
#     :parasite_age => 0,
#     :copepod_size => 1,
#     :grazer_size => 1,
#     :parasite_size => 1,
#     :stickleback_size => 3,
#     :phytoplankton_age => 0,
#     :phytoplankton_energy => 0,
#     :hatch_prob => 0.2,
#     :copepod_mortality => (1 / 24) * 0.05,
#     :grazer_mortality => (1 / 24) * 0.1,
#     :phytoplankton_mortality => (1 / 24) * 0.01,
#     :copepod_vel => 0.5,
#     :grazer_vel => 0.25,
#     :parasite_vel => 0.1,
#     :stickleback_vel => 0.7,
#     :dt => 1.0,
#     :seed => rand(UInt8, 1),
# )

###adata = [(grazer, count), (parasite, count), (phytoplankton, count),(copepod, count), (copepodInf, count), (stickleback, count), (sticklebackInf, count)]
###adf = paramscan(params, initialize_model; adata, agent_step!, model_step!, n = 5)

###adata = [(grazer, count), (parasite, count), (phytoplankton, count),(copepod, count), (copepodInf, count), (stickleback, count), (sticklebackInf, count)]
###adf = paramscan(params, initialize_model; adata, agent_step!, model_step!, n = 24) 

####println(adf[1])


## Save results
#####using CSV, DataFrames
# write out a DataFrame to csv file
#####CSV.write("data_2.csv", adf[1])
# CSV.write("/scratch/tmp/janayaro/My_projects/Copepod/data_2.csv", adf[1])



# n=60
# model = initialize_model()
# adata = [(grazer, count), (parasite, count), (phytoplankton, count),(copepod, count), (copepodInf, count), (stickleback, count), (sticklebackInf, count)]
# adf = run!(model, agent_step!, model_step!, n; adata)
# adf = adf[1]
# show(adf, allrows=true)


#plot(adf.count_copepod, adf.count_grazer, adf.count_parasite, adf.count_phytoplankton, adf.count_copepodInf, adf.count_stickleback, adf.count_sticklebackInf)

#df = adf[1]

##names(df)





#t = adf[1].step ./ 24

#using Plots

#Plots.plot(t, (df.count_phytoplankton), lab = "Phytoplankton")
#plot!(t, (df.count_copepod), lab = "Copepods")
#plot!(t, (adf.count_grazer), lab = "Grazers")
#plot!(t, (adf.count_stickleback), lab = "Fish")


#global sensitivity -> ensemblerun  : https://github.com/SciML/GlobalSensitivity.jl
#Salib py - sobol julia


