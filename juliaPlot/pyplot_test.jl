using PyPlot

#= turn of interactive plotting
ioff() 
=#

a = randn(30000, 32);
a[:, 3] .= NaN;
imshow(a, aspect="auto");
PyPlot.plt[:colorbar]();

# Set na.color = "black"
cm = PyPlot.cm_get_cmap()
cm[:set_bad](color="black")

savefig("heatmap.pdf")
