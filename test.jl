using CSV
using DataFrames
using PlotlyJS
using IterTools
using Combinatorics

# Read the CSV file into a DataFrame
df = CSV.File("dataParamscan.csv") |> DataFrame

# Group the DataFrame by the copepod_infected_vel column
grouped_civ = groupby(df, [:copepod_infected_vel, :infected_feeding_rate])

civ_keys = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
ifr_keys = [2,3,4,5,6,7,8,9,10]

x_data = collect(0:60)

color_vec = []
for _ in 0:60
    r = rand(0:255)
    g = rand(0:255)
    b = rand(0:255)
    a = 0.5
    
    rgba_str = "rgba($r, $g, $b, $a)"
    push!(color_vec, rgba_str)
end

y_data = []

for civ_key in civ_keys
    for ifr_key in ifr_keys
        count_copepods = []
        for row in eachrow(grouped_civ[(civ_key, ifr_key)])
            push!(count_copepods, row.count_copepod)
        end
        push!(y_data,count_copepods)
    end
end

traces = [
    box(
        y = yd,
        x = xd,
        boxpoints="all",
        jitter=0.5,
        whiskerwidth=0.2,
        fillcolor = cls,
        marker_size=2,
        line_width=1
    )
    for (xd, yd, cls) in zip(x_data, y_data, color_vec)
]

layout = Layout(
    title="Copepod Variation by Steps",
    yaxis=attr(
        autorange=true,
        showgrid=true,
        zerline=true,
        dtick=5,
        gridcolor="rgb(255,255,255)",
        zerlinecolor="rgb(255,255,255)",
        zerolinewidth=2
    ),
    margin=attr(
        l=40,
        r=30,
        b=80,
        t=100
    ),
    paper_bgcolor="rgb(243,243,243)",
    plot_bgcolor="rgb(243,243,243)",
    showlegend=false
)
print(typeof(traces))
plot(traces, layout)


