#Return functionality check
#Trophic cascades: energy loss and provision from agents to higher trophic levels

#Size? multiplicate vision radius with size?
#mortality for copepods (simulate sticklebacks)
#mortality for phytoplankton (simulate all other zooplankton)
#have copepods feed on phytoplankton? yes for early stages 
#dont stack agents on top of each other in one position
#limit amount of agents in general 
#90% loss of energy each trophic level; metabolic cost  
#adding classes of death (dead by fish, dead by energy loss, dead by mortality) 

#"The reductionist approach in ecology is relatively easy to apply if we assume that population members are either identical or that they differ only by sex and age." - include in title?
#allagents(model)
#has_empty_positions(model)
#think about walk funciton
#copepod eat phytoplankton before age 19.5 
#age 1 day = 24 steps!! 
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


#errors:
#Stickleback not implemented for ContinuousSpace
#Plotting

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

#cd("Dropbox/Jaime M/Projects_JM/Muenster/Marvin/project/HostParasite/HostParasite/")

mutable struct CopepodGrazerParasitePhytoplankton <: AbstractAgent
    id::Int #Id of the Agent
    pos::NTuple{2, Float64} #position in the Space
    type::Symbol # :Copepod or :Parasite or :Grazer or :Phytoplankton or :Stickleback
    energy::Float64 
    reproduction_prob::Float64  
    Δenergy::Float64
    infected::Bool
    age::Int  
    gender::Int  # 1 = female , 2 = male
    size::Float64 #bigger copepods eat more Grazer and vice versa 
    fullness::Int 
    # -> mean for Macrocyclops albidus: mean (+- SD) in mm: Females: 1.56 +- 0.097 ; males: 1.11 +- 0.093
end

function Copepod(id, pos, energy, repr, Δe, age, size)
    CopepodGrazerParasitePhytoplankton(id, pos, :copepod, energy, repr, Δe, :false, age, rand(1:2), size, 0)
end

function Parasite(id, pos, energy, repr, Δe, age, size)
    CopepodGrazerParasitePhytoplankton(id, pos, :parasite, energy, repr, Δe,:false, age, 1, size, 0)
end
    
function Grazer(id, pos, energy, repr, Δe, age, size)
    CopepodGrazerParasitePhytoplankton(id, pos, :grazer, energy, repr, Δe, :false, age, rand(1:2), size, 0)
end

function Phytoplankton(id, pos, energy, age)
    CopepodGrazerParasitePhytoplankton(id, pos, :phytoplankton, energy, 0.0, 10, :false, age, 1, 0.01, 0)
end 

function Stickleback(id, pos, repr, size)
    CopepodGrazerParasitePhytoplankton(id, pos, :stickleback, 100, repr, 10, :false, 10, rand(1:2), size, 0)
end

norm(vec) = √sum(vec .^ 2)

function initialize_model(;
    n_copepod = 500,
    n_phytoplankton = 10000,
    n_grazer = 1000, # Grazer being Chydoridae, Daphniidae and Sididae (All Branchiopoda)
    n_parasite = 5000, 
    n_stickleback = 30,
    #n_eggs = 200, #continuous stream of "newly introduced parasites"
    Δenergy_copepod = 120, #5 days
    Δenergy_grazer = 72, #3 days
    Δenergy_parasite = 96,# 4 days   
    #Δenergy_stickleback = 96,
    copepod_vision = 0.05,  # how far copepods can see grazer to hunt
    grazer_vision = 0.05,  # how far grazer see phytoplankton to feed on
    parasite_vision = 0.005,  # how far parasites can see copepods to stay in their general vicinity
    stickleback_vision = 0.5,
    copepod_reproduce = 0.00595, 
    grazer_reproduce = 0.01666,         
    parasite_reproduce = 0, 
    stickleback_reproduce = 0.041, #once per day
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
    phytoplankton_mortality = 0.01,
    #stickleback_mortality = 0.2,
    copepod_vel = 0.7,
    grazer_vel = 0.5,
    parasite_vel = 0.2,
    stickleback_vel = 1.,
    hatch_prob = 0.20, #probability for eggs to hatch, 20% as to Merles results (Parasite_eggs/hatching rates excel in Dropbox)
    seed = 23182,
    dt = 0.1,    
    )

    rng = MersenneTwister(seed) #MersenneTwister: pseudo random number generator
    space = ContinuousSpace((2000., 2000.); periodic = true)
    #heightmap_path = "C:\\Users\\Marvin\\OneDrive\\Dokumente\\GitHub\\HostParasite\\WhiteSpace.jpg"
    heightmap_path ="WhiteSpace.jpg"
    #heightmap_path = "WhiteSpace.jpg"
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


function phytoplankton_step!(phytoplankton, model)
    phytoplankton.age += 1
    if phytoplankton.age >= 48 #"a couple of days" e.g. 2 up to 23 days (https://acp.copernicus.org/articles/10/9295/2010/)??? 
        kill_agent!(phytoplankton, model, model.pathfinder)
        print("phytoplankton dead by age")
        return
    end
    if rand(model.rng) < model.phytoplankton_mortality
        kill_agent!(phytoplankton, model, model.pathfinder)
        print("phytoplankton dead by mortality")
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
        walk!(parasite, rand, model; periodic = false)
    end
end


function grazer_step!(grazer, model) 
    grazer_eat!(grazer, model)
    grazer.age += 1
    if grazer.age >= 480
        kill_agent!(grazer, model, model.pathfinder)
        return
    end
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
    if copepod.fullness < 9
        copepod_eat!(copepod, model)  
    end 
    copepod.age += 1
    if copepod.age >= 1080
        kill_agent!(copepod, model, model.pathfinder)
        return
    end
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

    if is_stationary(copepod, model.pathfinder)  
        prey = [x.pos for x in nearby_agents(copepod, model, model.copepod_vision) if x.type == :grazer && x.age >= 10]
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
        stickleback.infected = false
    end
    
    hunt = [x.pos for x in nearby_agents(stickleback, model, model.stickleback_vision) if (x.type == :grazer && x.age >= 10) || (x.type == :copepod && x.age >= 19)] #only eating adult copepods and grazers

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
end



function stickleback_eat!(stickleback, model)
    chase = [x for x in nearby_agents(stickleback, model, model.stickleback_vision) if x.type == :copepod || x.type == :grazer]
    if !isempty(chase)
        for x in chase
            if x.infected == true
                stickleback.infected = true
            end
            kill_agent!(x, model, model.pathfinder)
        end
    end
end

function copepod_eat!(copepod, model) 
    food = [x for x in nearby_agents(copepod, model, model.copepod_vision) if (x.type == :grazer && copepod.age >= 480) || (x.type == :phytoplankton && copepod.age <= 480)]
    if !isempty(food)  
        for x in food 
            if x.type == :grazer  
                copepod.fullness += 1 
            end
            if x.type == :phytoplankton
                print("phytoplankton dead by copepod")
            end
        kill_agent!(x, model, model.pathfinder)
        copepod.energy += copepod.Δenergy
        end
    end

    if copepod.age % 24 == 0 
        copepod.fullness = 0
    end 

    infection = [x for x in nearby_agents(copepod, model, model.copepod_vision) if x.type == :parasite]
    if !isempty(infection)
        for x in infection
            kill_agent!(x, model, model.pathfinder)
            copepod.infected = true
        end
    end
end

function grazer_eat!(grazer, model)        
    plankton = [x for x in nearby_agents(grazer, model) if x.type == :phytoplankton]
    if !isempty(plankton)
        #plankton = rand(model.rng, phytoplankton)
        grazer.energy += grazer.Δenergy
        kill_agent!(rand(model.rng, plankton), model, model.pathfinder)
        print("phytoplankton dead by grazer")
        return
    end
end

function grazer_reproduce!(grazer, model) 
    if grazer.gender == 1 && grazer.age > 168
       
        grazer.energy /= 2
        for _ in 1:4
            id = nextid(model)
            offspring = CopepodGrazerParasitePhytoplankton(
                id,
                grazer.pos,
                grazer.type,
                grazer.energy,
                grazer.reproduction_prob,
                grazer.Δenergy,
                :false,
                0,
                rand(1:2),
                grazer.size,
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
    if copepod.type == :copepod && copepod.infected == true 
    elseif copepod.gender == 1 && copepod.age > 468
       
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
                :false,
                0,
                rand(1:2),
                copepod.size,
                0
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
    dry_w = rand(Normal(150,50)) ./ 10000
    epg = 39247 .* dry_w .- 47 
    for _ in 1:epg
        Δenergy_parasite = 96
        parasite_reproduce = 0
        parasite_size = 0.1
        if rand(model.rng) <= model.hatch_prob
            id = nextid(model)
            eggs = Parasite(
                id,
                random_walkable(random_position(model), model, model.pathfinder),
                rand(1:(Δenergy_parasite*1))-1,
                parasite_reproduce,
                Δenergy_parasite,
                0,
                parasite_size
            )
        add_agent!(eggs, model)
        end
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

#fig, _ = abm_plot(model; plotkwargs...)
#fig

grazer(a) = a.type == :grazer
copepod(a) = a.type == :copepod
copepodInf(a) = a.type == :copepod && a.infected == true
parasite(a) = a.type == :parasite
phytoplankton(a) = a.type == :phytoplankton
stickleback(a) = a.type == :stickleback
sticklebackInf(a) = a.type ==:stickleback && a.infected == true

n=5
adata = [(grazer, count), (parasite, count), (phytoplankton, count),(copepod, count), (copepodInf, count), (stickleback, count), (sticklebackInf, count)]
adf = run!(model, model_step!, n; adata)

model = initialize_model()

n = 50
adf = run!(model, model_step!, n; adata)


adf = adf[1]


using Plots

#plot(adf.count_copepod, adf.count_grazer, adf.count_parasite, adf.count_phytoplankton, adf.count_copepodInf, adf.count_stickleback, adf.count_sticklebackInf)

Plots.plot(adf.count_copepod, adf.step)
Plots.plot!(adf.count_grazer, adf.step)
Plots.plot!(adf.count_phytoplankton, adf.step)
Plots.plot!(adf.count_stickleback, adf.step)


#function plot_population_timeseries(adf)
 #   figure = Figure(resolution = (600, 600))
 #   ax = figure[1, 1] = Axis(figure; xlabel="Step",ylabel = "Population")
 #   grazerl = lines!(ax, adf.step, adf.count_grazer, color = :yellow)
 #   copepodl = lines!(ax, adf.step, adf.count_copepod, color = :black)
 #   parasitel = lines!(ax, adf.step, adf.count_parasite, color = :magenta)
 #   phytoplanktonl = lines!(ax, adf.step, adf.count_phytoplankton, color = :green)
 #   sticklebackl = lines!(ax, adf.step, adf.count_stickleback, color = :blue)
 #   copepodInfl = lines!(ax, adf.step, adf.count_copepodInf, color = :red)
 #   sticklebackInfl = lines!(ax, adf.step, adf.count_sticklebackInf, color = :orange)
 #   figure[1, 2] = Legend(figure, [grazerl, copepodl, parasitel, phytoplanktonl, sticklebackl, copepodInfl, sticklebackInfl], ["Grazers", "Copepods", "Parasites", "Phytoplankton", "Stickleback", "CopepodInfected", "Stickleback Infected"])
 #   figure
#end

#plot_population_timeseries(adf)
#abm_video(
#    "HostParasiteModel.mp4",
 #   model,
  #  model_step!;
   # frames = 25, 
    #framerate = 8,
    #plotkwargs...,
#)

