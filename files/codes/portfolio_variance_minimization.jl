#Julia implementation to compute optimal portfolio with risky and risk-free assets
#The objective is to minimize portfolio variance subject to 1. a certain level of portfolio return, 2. short-selling constraint
#Referenced Ch. 4 of Introduction to Computational Economics Using Fortran by Hans Fehr and Fabian Kindermann 
#Author: Hanjo Terry Kim (terryhanjokim@gmail.com) 
#Last updated: 9/3/2024 

using DataFrames, StatsBase, Optim, NLsolve, Plots, CSV

#Create a sample data of stock prices 
data = DataFrame(:year=>[0; 1; 2; 3; 4; 5], :stock_A=>[1.; 1.02; 1.17; 1.08; 1.16; 1.26], :stock_B=>[2.0; 2.65; 2.40; 2.70; 2.85; 2.75], :stock_C=>[3.0; 2.80; 4.5; 4.2; 3.2; 4.2]); 

#get returns, its covariance and historical means
r = Array(data[2:end,2:end])./Array(data[1:end-1,2:end]).-1; 
Σ = cov(r, corrected=false)
μ = vec(mean(r, dims=1))
r_f = 0.04; 

"""
```
dc = get_omega(X)
```
Input: X (2*NN+1) where NN is the number of stocks, first NN are omegas (optimal investment shares), next NN+1 are the multipliers (short-selling constraint), μ is a vector of historical mean of returns, Σ is a covariance matrix of returns, r_f is the risk-free rate, and γ is the risk aversion parameter

Output: residuals of first-order conditions that characterize optimal investment shares among stocks and risk-free asset plus Lagrange multipliers 

We re-formulate the KKT problem with Fischer-Burmeister function to make it computationally smooth
minimizing variance subject to some levels of return and short-selling constraint.  
"""
function get_omegas(X, μ, Σ; r_f=0.04, γ=8)
    NN = length(μ); 
    ω = exp.(X[1:NN]); 
    ωf = 1-sum(ω); 
    λ = exp.(X[NN+1:2*NN]); 
    λf = exp.(X[end]); 
    F = zeros(eltype(X), 2*NN+1); 
    F[1:NN]      .= μ .- r_f .- γ.*Σ*ω .+ λ .- λf; #FOC wrt ω_i
    F[NN+1:2*NN] .= ω .+ λ .- sqrt.(ω.^2 .+ λ.^2); #KKT condition (reformulated to be smooth) 
    F[end]      = ωf + λf - sqrt(ωf^2 + λf^2)
    return F
end

NN = length(μ); 
x_init = zeros(2*NN+1); 
x_init[1:NN] .= log(1/NN); 
res = nlsolve(x->get_omegas(x, μ, Σ, r_f=r_f, γ=5), x_init, show_trace=true, autodiff=:forward); 
ω   = exp.(res.zero)[1:NN]; 
ωf = 1-sum(ω); 

#Solving with Monte Carlo simulation (good to avoid curse of dimensionality and in case the root-finding doesn't converge)
#Make sure that μ and Σ includes the risk-free rate 
NN = length(μ); 
μ2 = [r_f; μ]; 
Σ2 = [zeros(NN+1)'; 
      zeros(NN,1) Σ]; 
function get_omegas_mc(μ, Σ; γ=5, NP=5_000, maxit=5_000)
    NN = length(μ); 
    ω_mc = rand(NN, NP); 
    ω_mc .= ω_mc./sum(ω_mc, dims=1)
 
    ω = zeros(NN); 
    μ_mc = zeros(NP); 
    Σ_mc = zeros(NP); 
    U_mc = zeros(NP); 
 
    for it=1:maxit
        #Get simulated returns and covariance
        μ_mc .= vec(sum(μ.*ω_mc, dims=1)); 
        Σ_mc .= vec([ω_mc[:,i]'Σ*ω_mc[:,i] for i=1:NP]); 
        
        #Get utilities and maximize  
        U_mc .= μ_mc .- γ/2.0.*Σ_mc; 
        _, inds = findmax(U_mc); 
        ω .= ω_mc[:,inds]; 
        
        #Update guess around the new maximum 
        ω_mc .= rand(NN, NP); 
        ω_mc .= ω_mc./sum(ω_mc, dims=1)
        ω_mc .= 0.9.*ω .+ 0.1.*ω_mc; 
    end
    return ω
end
ω2 = get_omegas_mc(μ2, Σ2); 

println("Optimal portfolio shares through root-finding:")
println("Stock A: $(round(ω[1], digits=3)).")
println("Stock B: $(round(ω[2], digits=3)).")
println("Stock C: $(round(ω[3], digits=3)).")
println("Risk-free: $(round(ωf, digits=3)).")

println("Optimal portfolio shares through Monte Carlo:")
println("Stock A: $(round(ω2[2], digits=3)).")
println("Stock B: $(round(ω2[3], digits=3)).")
println("Stock C: $(round(ω2[4], digits=3)).")
println("Risk-free: $(round(ω2[1], digits=3)).")

