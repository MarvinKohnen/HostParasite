using CSV
using DataFrames
using Statistics


# Load the CSV file
data = CSV.File("Data/dataHighEnd.csv") |> DataFrame

# Group the data by the 'step' column and calculate the mean and std for 'count_copepod'
groupedDF = combine(groupby(data, :step), :count_copepod => mean, :count_copepod => std, :count_copepodInf => mean, :count_copepodInf => std, :count_grazer => mean, :count_grazer => std, :count_parasite => mean, :count_parasite => std, :count_phytoplankton => mean, :count_phytoplankton => std)

# Extract mean, and standard deviation into separate arrays
mean_count_copepod = groupedDF.count_copepod_mean
std_count_copepod = groupedDF.count_copepod_std

mean_count_copepodInf = groupedDF.count_copepodInf_mean
std_count_copepodInf = groupedDF.count_copepodInf_std

mean_count_grazer = groupedDF.count_grazer_mean
std_count_grazer = groupedDF.count_grazer_std


mean_count_parasite = groupedDF.count_parasite_mean
std_count_parasite = groupedDF.count_parasite_std


mean_count_phytoplankton = groupedDF.count_phytoplankton_mean
std_count_phytoplankton = groupedDF.count_phytoplankton_std

using PlotlyJS

x = groupedDF.step
y = [0, 100,200,300, 400, 500, 600]
y_upper_copepod = mean_count_copepod + std_count_copepod
y_lower_copepod = mean_count_copepod + -1 * std_count_copepod

y_upper_copepodInf = mean_count_copepodInf + std_count_copepodInf
y_lower_copepodInf = mean_count_copepodInf + -1 * std_count_copepodInf

y_upper_grazer = mean_count_grazer + std_count_grazer
y_lower_grazer = mean_count_grazer + -1 * std_count_grazer


y_upper_parasite= mean_count_parasite + std_count_parasite
y_lower_parasite = mean_count_parasite + -1 * std_count_parasite 


y_upper_phytoplankton = mean_count_phytoplankton + std_count_phytoplankton
y_lower_phytoplankton = mean_count_phytoplankton+ -1 * std_count_phytoplankton 

traces=[
    scatter(
        name = "Mean count Copepod",
        x=x,
        y=mean_count_copepod,
        line=attr(color="rgb(52,94,235)"),
        mode="lines"
    )
    scatter(
        name = "Std count Copepod",
        x=vcat(x, reverse(x)), # 
        y=vcat(y_upper_copepod, reverse(y_lower_copepod)),        
        fill="toself",
        fillcolor="rgba(52,94,235,0.2)",
        line=attr(color="rgba(52,94,235,0)"),
        hoverinfo="skip",
        showlegend=false
    )
    scatter(
        name = "Mean count infected Copepod ",
        x=x,
        y=mean_count_copepodInf,
        line=attr(color="rgb(235,70,52)"),
        mode="lines"
    )
    scatter(
        name = "Std count infected Copepod",
        x=vcat(x, reverse(x)), # 
        y=vcat(y_upper_copepodInf, reverse(y_lower_copepodInf)),        
        fill="toself",
        fillcolor="rgba(235,70,52,0.2)",
        line=attr(color="rgba(235,70,52,0)"),
        hoverinfo="skip",
        showlegend=false
    )
    scatter(
        name = "Mean count Grazer ",
        x=x,
        y=mean_count_grazer,
        line=attr(color="rgb(235,229,52)"),
        mode="lines"
    )
    scatter(
        name = "Std count Grazer",
        x=vcat(x, reverse(x)), # 
        y=vcat(y_upper_grazer, reverse(y_lower_grazer)),        
        fill="toself",
        fillcolor="rgba(235,229,52,0.2)",
        line=attr(color="rgba(235,229,52,0)"),
        hoverinfo="skip",
        showlegend=false
    )
    
    scatter(
        name = "Mean count Phytoplankton ",
        x=x,
        y=mean_count_phytoplankton,
        line=attr(color="rgb(52,235,88)"),
        mode="lines"
    )
    scatter(
        name = "Std count Phytoplankton",
        x=vcat(x, reverse(x)), # 
        y=vcat(y_upper_phytoplankton, reverse(y_lower_phytoplankton)),        
        fill="toself",
        fillcolor="rgba(52,235,88,0.2)",
        line=attr(color="rgba(52,235,88,0)"),
        hoverinfo="skip",
        showlegend=false
    )
]
layout = Layout(
    legend=attr(
        x=1,
        y=1.00,
        yanchor="bottom",
        xanchor="right",
        orientation="h"
    ),
    title="Copepod Infected Velocity: 1.5, Infected Feeding Rate 10",
    xaxis_title="Steps",
    yaxis_title="Population Size",
    paper_bgcolor="white",
    plot_bgcolor="white",  
    yaxis = attr(range=[0, 750])
)
plot(traces, layout)

#savefig("DataAnalysisHighEnd.png")