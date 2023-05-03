import Pkg
Pkg.activate(".")
using Plots
n = 2000
f(x, y) = exp(-(x^2 + y^2) / 2)
l0(x, m) = 1
l1(x, m) = -x + m + 1
l2(x, m) = (x^2 - 2 * (m + 2) * x + (m + 1) * (m + 2)) / 2
l = [l0, l1, l2]
x = LinRange(-6, 6, n)
y = LinRange(-6, 6, n)
r(x, y) = sqrt(x^2 + y^2)
"Cosine of a multiply of polar angle for a point in plane."
cos_m(x, y, m) = cos(m * atan(y / x))

argon=[]
push!(argon,RGB(0.0,247/255,255/255))
push!(argon,RGB(0/255,196/255,255/255))
push!(argon,RGB(26/255,255/255,0.0))
#push!(argon,RGB(0/255,92/255,255/255))

plots = []
for i = 0:2, m = [0,2,3,4]
  color = coloralpha(argon[i+1],0)
  c_gradient = ColorGradient([color, argon[i+1]])
  z = [abs((r(xi, yi)^m * l[i+1](r(xi, yi)^2, m) * f(xi, yi)) *
           cos_m(xi, yi, m)) for xi in x, yi in y]
  p = heatmap(x, y, z,
    aspect_ratio = 1,
    axis = false,
    legend = false,
    grid = false,
    color = c_gradient,
    background_color = color,
    size = (2000,2000)
  )
  push!(plots, p)
  savefig("../slike/laguerrovi_snopi_faza_$(i)_$m.png")
end
plot(plots..., layout = (3, 4),background_color=:gray2)
