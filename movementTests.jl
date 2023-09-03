caway_direction = (0.,0.)
cdirection = (0., 0.)


copepodpos = (4.5,3.0)
cpredators = [(10.0, 13.0), (14.0,19.0),(5.0, 54.5)]
prey = [(15.0,13.0)]
if !isempty(cpredators) 
    caway_direction = []
    caway_direction = (copepodpos .- cpredators[1]) 
end 

print(caway_direction)


ctoward_direction = (0.,0.)
if !isempty(prey)
    ctoward_direction = copepodpos .+ prey[1]
end

print(ctoward_direction)

cdirection = cdirection .+ ctoward_direction .+ caway_direction   #set new direction 

print(cdirection)

