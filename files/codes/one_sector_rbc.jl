#one sector RBC model
#Author: Hanjo Terry Kim (terryhanjokim@gmail.com)
#last update 9/3/2024 

using Plots, QuantEcon
include("SGU.jl")
import .SGU
using Parameters, SymEngine

function model()
    #Step 1: Declare parameter and variables 
    params_keys = "beta, delta, theta, gamma, alpha, rho_z, stdeps_z"; 
    vars_keys = "c, k, y, h, lambda, iv, z"; 
    params, vars, varsp = SGU.create_variables(params_keys, vars_keys); 
    shocks = "z"
    std_eps = "stdeps_z"
    
    #Optional step 1.5: unpack params, vars, and varsp; 
    @unpack beta, delta, theta, gamma, alpha, rho_z, stdeps_z = params;  
    @unpack c, k, y, h, lambda, iv, z = vars; 
    @unpack cp, kp, yp, hp, lambdap, ivp, zp = varsp; 

    #Step 2: Model equations 
    F = Array{Basic}(undef,length(vars));
    F.= 0.;
    
    F[1] = (exp(c)^theta*(1-exp(h))^(1-theta))^(-gamma)*theta*(exp(c)/(1-exp(h)))^(theta-1) - exp(lambda); 
    F[2] = (exp(c)^theta*(1-exp(h))^(1-theta))^(-gamma)*(1-theta)*(exp(c)/(1-exp(h)))^(theta) - exp(lambda)*exp(z)*(1-alpha)*(exp(k)/exp(h))^alpha; 
    F[3] = exp(lambda)-beta*exp(lambdap)*(1-delta+exp(zp)*alpha*(exp(kp)/exp(hp))^(alpha-1)); 
    F[4] = exp(c)+exp(kp)-(1-delta)*exp(k)-exp(z)*exp(k)^alpha*exp(h)^(1-alpha); 
    F[5] = exp(iv)-exp(kp)+(1-delta)*exp(k);  
    F[6] = exp(y)-exp(z)*exp(k)^alpha*exp(h)^(1-alpha); 
    F[7] = zp-rho_z*z; 

    statevar = [k; z];
    controlvar = [c; h; y; iv; lambda]; 
    
    FF = SGU.linearize(F, statevar, controlvar, params, vars, shocks, stdeps_z; lambdifying=true); 
    return FF
end

function steadystate!(FF)
    #Step 3: Assign parameter values (make sure the names are exact matches) 
    alpha = 0.33; 
    delta = 0.025; 
    beta  = 0.98; 
    rho_z   = 0.9; 
    gamma = 2; 
    theta = 0.33; 
    stdeps_z = 0.01; 
    @pack! FF.params = alpha, delta, beta, rho_z, gamma, theta, stdeps_z;  

    #Step 4: solve the steady state 
    kh = ((1/beta+delta-1)/alpha)^(1/(alpha-1)); 
    h = theta/(1-theta)*(1-alpha)*kh^alpha/(kh^alpha-delta*kh+theta/(1-theta)*(1-alpha)*kh^alpha)
    k = kh*h;
    y = kh^alpha*h; 
    z = 1
    iv = delta*k; 
    c = y-iv; 
    lambda = (c^theta*(1-h)^(1-theta))^(-gamma)*theta*(c/(1-h))^(theta-1);  
    #written in logs
    k, z, c, h, y, iv, lambda = log(k), log(z), log(c), log(h), log(y), log(iv), log(lambda); 
    @pack! FF.vars = k, z, c, h, y, iv, lambda; 
    
    #@pack! FF = vars, params;  
end

FF = model(); 
steadystate!(FF); 

#Step 4: num_eval 
G = SGU.num_eval(FF)

#IRF to one percent positive productivity shock
x0 = zeros(G.nx); 
x0[G.ix[:z]] = 1; 
T = 48; 
_, IRy, IRx = SGU.ir(G.gx, G.hx, x0, T)
p1 = plot(IRy[:,G.iy[:c]], title="Consumption", label=false); 
p2 = plot(IRy[:,G.iy[:h]], title="Labor", label=false); 
p3 = plot(IRy[:,G.iy[:iv]], title="Investment", label=false); 
p4 = plot(IRy[:,G.iy[:y]], title="Output", label=false); 
plot(p1, p2, p3, p4)

#Simulate and calculate simulated moments
Xsim, Ysim = SGU.simu_1st(G.gx, G.hx, G.nETASHOCK, 1500); 
Tdrop = 500; 
using StatsBase
std(hp_filter(Ysim[Tdrop+1:end,G.iy[:iv]], 1600)[2])
std(hp_filter(Ysim[Tdrop+1:end,G.iy[:y]], 1600)[2])
std(hp_filter(Ysim[Tdrop+1:end,G.iy[:c]], 1600)[2])

#Calculate theoretical second moments 
sigy, sigx = SGU.mom(G.gx, G.hx, G.nvarshock); 
@show sigy[G.iy[:y]]
@show sigy[G.iy[:c]]
@show sigy[G.iy[:iv]]



