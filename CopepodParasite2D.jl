
# upon infection of copepods less evasion of fish after a couple of days (infection days 13! Haferr-Hamann 2019 ) [check]
# initialize fish as some infected some not [check], no parasites for first 20 days
# have infected fish introduce parasites after 20 days 


# virulence = amount of eaten grazers while infected 



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


pwd()
#cd("Dropbox/Jaime M/Projects_JM/Muenster/Marvin/project/HostParasite/HostParasite/")

mutable struct CopepodGrazerParasitePhytoplankton <: AbstractAgent
    id::Int #Id of the Agent
    pos::NTuple{2, Float64} #position in the Space
    type::Symbol # :Copepod or :Parasite or :Grazer or :Phytoplankton or :Stickleback
    energy::Float64 
    reproduction_prob::Float64  
    Δenergy::Float64
    infected::Int   # ::Int
    infectiondays::Int
    gender::Int  # 1 = female , 2 = male
    fullness::Int 
    # -> mean for Macrocyclops albidus: mean (+- SD) in mm: Females: 1.56 +- 0.097 ; males: 1.11 +- 0.093
end

function Copepod(id, pos, energy, repr, Δe)
    CopepodGrazerParasitePhytoplankton(id, pos, :copepod, energy, repr, Δe, 0, 0, rand(1:2),0)
end

function Parasite(id, pos, energy, repr, Δe)
    CopepodGrazerParasitePhytoplankton(id, pos, :parasite, energy, repr, Δe,0, 0, 0, 0)
end
function Grazer(id, pos, energy, repr, Δe)
    CopepodGrazerParasitePhytoplankton(id, pos, :grazer, energy, repr, Δe, 0, 0, rand(1:2), 0)
end

function Phytoplankton(id, pos, energy, repr, Δe)
    CopepodGrazerParasitePhytoplankton(id, pos, :phytoplankton, energy, repr, Δe, 0, 0, 1, 0)
end 

function Stickleback(id, pos, repr, infected)
    CopepodGrazerParasitePhytoplankton(id, pos, :stickleback, 100, repr, 10, infected, 10, rand(1:2), 0)
end

norm(vec) = √sum(vec .^ 2)

function initialize_model(;
    n_copepod = 30, #100
    n_phytoplankton = 400, 
    n_grazer = 160, #100 # Grazer being Chydoridae, Daphniidae and Sididae (All Branchiopoda)
    n_parasite = 300, #1000
    n_stickleback = 10,

    # Energy parameters
    Δenergy_copepod = 30, #5 days
    Δenergy_grazer = 30, #1 days
    Δenergy_parasite = 30,# 4 days
    Δenergy_phytoplankton = 10,   
    #Δenergy_stickleback = 96,


    copepod_vision = 10,  # how far copepods can see grazer to hunt
    grazer_vision = 6,  # how far grazer see phytoplankton to feed on
    parasite_vision = 6,  # how far parasites can see copepods to stay in their general vicinity
    stickleback_vision = 15, # location to location in grid = 1 
    
    copepod_reproduce = 0.03,
    grazer_reproduce = 0.06,
    parasite_reproduce = 0, 
    phytoplankton_reproduce = 0.2,
    stickleback_reproduce = 0.8, #once per day
    
    phytoplankton_energy = 0,

    # Mortality rates per day
    copepod_mortality = 0.0001,
    grazer_mortality = 0.0001,
    phytoplankton_mortality = 0.0001,
    #stickleback_mortality = 0.2,

    # Movement rates per day
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
    space = ContinuousSpace((100., 100.); periodic = false)

    heightmap_path = "WhiteSpace.jpg"
    heightmap = load(heightmap_path)
    dims = (size(heightmap))
    water_walkmap= BitArray(falses(dims))
 
#  
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
                rand(model.rng,(Δenergy_grazer:2*Δenergy_grazer)),
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
                rand(model.rng, Δenergy_copepod:2*Δenergy_copepod),
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
    elseif agent.type == :copepod 
        copepod_step!(agent, model)
    elseif agent.type == :parasite
        parasite_step!(agent, model)
    elseif agent.type == :stickleback
        stickleback_step!(agent, model)
    else 
        phytoplankton_step!(agent, model)
    end
end

#function model_step!(model)
 #   for p in positions(model)
  #      n = length(agents_in_position(p,model))
   #     K = 10
    #    if n>=K
     #       ids = ids_in_position(p, model)
      #      genocide!(model, rand(ids,(n-K)))
       # end
    #end
#end

function phytoplankton_step!(phytoplankton, model)
    if rand(model.rng) < model.phytoplankton_mortality
        kill_agent!(phytoplankton, model, model.pathfinder)
        return
    end
    phytoplankton.energy += 3
    if rand(model.rng) <= model.phytoplankton_reproduce * model.dt
        phytoplankton_reproduce!(phytoplankton, model)
    end
end
    
function parasite_step!(parasite, model) #in lab: 2 days max, copepod 4 days max without food 
    parasite.energy -= 1
    if parasite.energy < 0
        kill_agent!(parasite, model, model.pathfinder)
        return
    end
    walk!(parasite, rand, model) #periodic = false
end


function grazer_step!(grazer, model) 
    if grazer.energy <= 15
        grazer_eat!(grazer, model)
    end
    grazer.energy -=model.dt

    if grazer.energy < 0
        kill_agent!(grazer, model, model.pathfinder)
        return
    end
    if rand(model.rng) < model.grazer_mortality
        kill_agent!(grazer, model)
        return
    end
    
    if rand(model.rng) <= grazer.reproduction_prob * model.dt
        grazer_reproduce!(grazer, model)
    end
        
    predators = [x.pos for x in nearby_agents(grazer, model, model.grazer_vision) if x.type == :copepod || x.type == :stickleback]

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
    #print(copepod.Δenergy)
    if rand(model.rng) < model.copepod_mortality
        kill_agent!(copepod, model)
        return
    end
    if rand(model.rng) <= copepod.reproduction_prob * model.dt
        copepod_reproduce!(copepod, model)
    end
    

    copepod_eat!(copepod, model)
    copepod.energy -= model.dt
    
    if copepod.energy < 0
        kill_agent!(copepod, model, model.pathfinder)
        return
    end
    
        
    if copepod.infectiondays <= 12 
        prey = [x.pos for x in nearby_agents(copepod, model, model.copepod_vision) if x.type == :grazer]
        cpredators = [x.pos for x in nearby_agents(copepod, model, model.copepod_vision) if x.type == :Stickleback]
        cdirection = (0., 0.)
        caway_direction = (0.,0.)
        if !isempty(cpredators) 
            caway_direction = []
            for i in 1:length(cpredators)
                if i == 1
                    caway_direction = (copepod.pos .- cpredators[i]) 
                else    
                    caway_direction = caway_direction .- cpredators[i]
                end
            end
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

        cdirection = cdirection .+ ctoward_direction ./ norm(ctoward_direction) .^2 .+ caway_direction ./ norm(caway_direction) .^2   #set new direction 
            
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
            ctoward_direction = []
            for i in 1:length(prey)
                if i == 1
                    ctoward_direction = (copepod.pos .+ prey[i])
                else
                    ctoward_direction = ctoward_direction .+ prey[i]
                end
            end 
        end

        cdirection = cdirection .+ ctoward_direction ./ norm(ctoward_direction) .^2   #set new direction 
            
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
    if copepod.infected == 1
        copepod_eat!(copepod, model)
        infectiondays +=1
        copepod.energy -= model.dt
        move_along_route!(copepod, model, model.pathfinder, model.copepod_vel, model.dt)
    end
end 
end

function stickleback_step!(stickleback, model) 
    
    
    hunt = [x.pos for x in nearby_agents(stickleback, model, model.stickleback_vision) if  (x.type == :copepod)]
    if is_stationary(stickleback, model.pathfinder) && !isempty(hunt)
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
        print("parasite reproduced")
    end
    
end


function stickleback_eat!(stickleback, model)
    chase = [x for x in nearby_agents(stickleback, model, model.stickleback_vision) if x.type == :copepod || x.type == :grazer]
    if !isempty(chase)
        for x in chase
            if x.infected == 1
                stickleback.infected = 1
                print("SB infected")
            end
            kill_agent!(x, model, model.pathfinder)
        end
    end
end

function copepod_eat!(copepod, model) 
    food = [x for x in nearby_agents(copepod, model, model.copepod_vision) if (x.type == :grazer) || (x.type == :phytoplankton)]
    if !isempty(food)  
        for x in food 
            if x.type == :grazer  
                copepod.fullness += 1 
            end
            #if x.type == :phytoplankton
                
            #end
            kill_agent!(x, model, model.pathfinder)
            copepod.energy += copepod.Δenergy
        end
    end


    infection = [x for x in nearby_agents(copepod, model, model.copepod_vision) if x.type == :parasite]
    if !isempty(infection)
        for x in infection
            kill_agent!(x, model, model.pathfinder)
            copepod.infected = 1
        end
    end
end

function grazer_eat!(grazer, model)        
    plankton = [x for x in nearby_agents(grazer, model) if x.type == :phytoplankton]
    if !isempty(plankton)
        for x in plankton
        #plankton = rand(model.rng, phytoplankton)
        grazer.energy += grazer.Δenergy
        #println("$grazer.energy \n" )
        kill_agent!(x, model, model.pathfinder)
        end
    end
end

function grazer_reproduce!(grazer, model) 
    if grazer.gender == 1
        
        #grazer.energy /= 2
        for _ in 1:rand(10:72)
            id = nextid(model)
            offspring = CopepodGrazerParasitePhytoplankton(
                id,
                grazer.pos,
                grazer.type,
                model.Δenergy_grazer,
                grazer.reproduction_prob,
                grazer.Δenergy,
                0,
                0,
                rand(1:2),
                0
            )
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
       
        copepod.energy /= 2

        for _ in 1:(rand(Normal(92, 5))) 
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
                0
            )
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
    add_agent_pos!(offspring, model)
    kill_agent!(phytoplankton, model)
    return
end

function parasite_reproduce!(model)
    dry_w = rand(Normal(150,50)) ./ 10000
    epg = 39247 .* dry_w .- 47 
    Δenergy_parasite = 96
    parasite_reproduce = 0
    print(epg)
    for _ in 1:epg
        if rand(model.rng) <= model.hatch_prob
            id = nextid(model)
            eggs = Parasite(
                id,
                random_walkable(random_position(model), model, model.pathfinder),
                rand(model.rng, (Δenergy_parasite:Δenergy_parasite*2)),
                parasite_reproduce,
                Δenergy_parasite,
            )
            add_agent!(eggs, model)
        end
    end

return
end


function model_step!(model)
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
            kill_agent!(x1[j], model)
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
copepod(a) = a.type == :copepod
copepodInf(a) = a.type == :copepod && a.infected == 1
parasite(a) = a.type == :parasite
phytoplankton(a) = a.type == :phytoplankton
stickleback(a) = a.type == :stickleback
sticklebackInf(a) = a.type ==:stickleback && a.infected == 1

n= 4
model = initialize_model()
adata = [(grazer, count), (parasite, count), (phytoplankton, count),(copepod, count), (copepodInf, count), (stickleback, count), (sticklebackInf, count)]
adf = run!(model, agent_step!, model_step!, n; adata)
adf = adf[1]
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


params = Dict(
    :n_copepod => [300], 
    :n_phytoplankton => 3000, 
    :n_grazer => 300,
    :n_parasite => 2000,
    :n_stickleback => 100,
    :Δenergy_copepod => 24 * 5,
    :Δenergy_grazer => 24,
    :Δenergy_parasite => 24 * 4,
    :copepod_vision => 4,
    :grazer_vision => 2,
    :parasite_vision => 1,
    :stickleback_vision => 8,
    :copepod_reproduce => 1 / 24,
    :grazer_reproduce => 1 / 24,
    :parasite_reproduce => 0, 
    :stickleback_reproduce => 0.8,
    :phytoplankton_reproduce => 1 / 24,
    :copepod_age => 0,
    :grazer_age => 0,
    :parasite_age => 0,
    :copepod_size => 1,
    :grazer_size => 1,
    :parasite_size => 1,
    :stickleback_size => 3,
    :phytoplankton_age => 0,
    :phytoplankton_energy => 0,
    :hatch_prob => 0.2,
    :copepod_mortality => (1 / 24) * 0.05,
    :grazer_mortality => (1 / 24) * 0.1,
    :phytoplankton_mortality => (1 / 24) * 0.01,
    :copepod_vel => 0.5,
    :grazer_vel => 0.25,
    :parasite_vel => 0.1,
    :stickleback_vel => 0.7,
    :dt => 1.0,
    :seed => rand(UInt8, 1),
)

adata = [(grazer, count), (parasite, count), (phytoplankton, count),(copepod, count), (copepodInf, count), (stickleback, count), (sticklebackInf, count)]
adf = paramscan(params, initialize_model; adata, agent_step!, model_step!, n = 5)

adata = [(grazer, count), (parasite, count), (phytoplankton, count),(copepod, count), (copepodInf, count), (stickleback, count), (sticklebackInf, count)]
adf = paramscan(params, initialize_model; adata, agent_step!, model_step!, n = 24) 

println(adf[1])


## Save results
using CSV, DataFrames
# write out a DataFrame to csv file
CSV.write("data_2.csv", adf[1])
# CSV.write("/scratch/tmp/janayaro/My_projects/Copepod/data_2.csv", adf[1])

#FIRST GRAZER IN POSITION???


# n=60
# model = initialize_model()
# adata = [(grazer, count), (parasite, count), (phytoplankton, count),(copepod, count), (copepodInf, count), (stickleback, count), (sticklebackInf, count)]
# adf = run!(model, agent_step!, model_step!, n; adata)
# adf = adf[1]
# show(adf, allrows=true)


#plot(adf.count_copepod, adf.count_grazer, adf.count_parasite, adf.count_phytoplankton, adf.count_copepodInf, adf.count_stickleback, adf.count_sticklebackInf)

df = adf[1]

names(df)





# t = adf[1].step ./ 24

# using Plots

# Plots.plot(t, (df.count_phytoplankton), lab = "Phytoplankton")
# plot!(t, (df.count_copepod), lab = "Copepods")
# plot!(t, (adf.count_grazer), lab = "Grazers")
# plot!(t, (adf.count_stickleback), lab = "Fish")


#global sensitivity -> ensemblerun  : https://github.com/SciML/GlobalSensitivity.jl
#Salib py - sobol julia


