using Plots
n = 300
f(x, y) = exp(-(x^2+y^2)/2)
l0(x, m) = 1
l1(x, m) = -x + m + 1
l2(x, m) = (x^2 - 2(m+2)*x + (m+1)*(m+2))/2
l = [l0, l1, l2]
x = LinRange(-4, 4, n)
y = LinRange(-4, 4, n)
r(x, y) = sqrt(x^2 + y^2)


plots = []
for  i = 0:2, m = 0:3
  z = [ (r(xi, yi)^m * l[i+1](r(xi,yi)^2, m) * f(xi, yi))^2 for
                                                    xi in x, yi in y ]
  p = heatmap(x, y, z,
                aspect_ratio=1, axis= false,
                legend=false, grid=false,
                color=ColorGradient([:white, :red2]))
  push!(plots, p)
end
plot(plots..., layout=grid(3, 4))
savefig("slike/laguerrovi_snopi.png")
