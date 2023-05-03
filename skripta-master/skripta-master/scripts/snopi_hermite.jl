using Plots
n = 300
f(x, y) = exp(-(x^2+y^2)/2)
h0(x) = 1
h1(x) = 2x
h2(x) = 4x^2 - 2
h3(x) = 8x^3 - 12x
h = [h0, h1, h2, h3]
x = LinRange(-3, 3, n)
y = x



plots = []
for i = 0:3, j= 0:3
  z = [ ((h[i+1](xi)*h[j+1](yi)*f(xi, yi)))^2 for xi in x, yi in y ]
  p = heatmap(x, y, z,
                aspect_ratio=:equal, axis= false,
                legend=false, grid=false,
                color=ColorGradient([:white, :red2]))
  push!(plots, p)
  # savefig("slike/hsnop$i$j.png")
end
plot(plots..., layout=16)
savefig("slike/hermitovi_snopi.png")
