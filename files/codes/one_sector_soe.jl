#One sector small open economy model as in Chapter 4 of Open Economy Macroeconomics textbook by Uribe and Schmitt-grohe 
#Author: Hanjo Terry Kim (terryhanjokim@gmail.com)
#Last updated: 9/2/2024 
using SymEngine, Parameters, Plots
include("SGU.jl")
import .SGU 

function model()
    params_keys = "rstar,betta,dbar,delta,alfa,phi,rho,etatilde,sigg,omega,pssi"
    vars_keys = "la,c,k,kfu,h,d,output,ivv,tb,tby,ca,cay,r,a,muu"
    shocks = "a,muu"
    stdeps = "etatilde,etatilde"
    params, vars, varsp =  SGU.create_variables(params_keys,vars_keys)
    @unpack rstar,betta,dbar,delta,alfa,phi,rho,etatilde,sigg,omega,pssi=params
    @unpack la,c,k,kfu,h,d,output,ivv,tb,tby,ca,cay,r,a,muu=vars
    @unpack lap,cp,kp,kfup,hp,dp,outputp,ivvp,tbp,tbyp,cap,cayp,rp,ap,muup=varsp

    e1  = -exp(la) + (exp(c)-exp(h)^omega/omega)^(-sigg)
    e2  = -exp(ivv) + exp(kp) - (1-delta)*exp(k);
    e3  = -exp(r) + rstar + pssi * (exp(dp-dbar) -1) + exp(exp(muu)-1)-1;
    e4  = -exp(output) + exp(a) * exp(k)^alfa * exp(h)^(1-alfa)
    e5  = -exp(h)^(omega-1) + (1-alfa) *exp(a) * (exp(k)/exp(h))^alfa
    e6  = -dp + (1+exp(r)) * (d) +exp(c) + exp(ivv) + phi/2 * (exp(kp)-exp(k))^2 - exp(output)
    e7  = -exp(kfu)+exp(kp);
    e8  = -tb + exp(output) - exp(c) - exp(ivv);
    e9  = -tby + tb/exp(output);
    e10 = -(ca) - (dp) + (d);
    e11 = -(cay) + (ca)/exp(output);

    e12 = -exp(la) + betta * (1+exp(r)) * exp(lap)
    e13 = -exp(la)* (1+phi*(exp(kp)-exp(k))) + betta * exp(lap) * (1-delta + alfa * exp(ap) * (exp(kp)/exp(hp))^(alfa-1) + phi * (exp(kfup)-exp(kp)))
    e14 = -ap + rho * a;
    e15 = -muup + rho * muu;

    #Create function f
    f = [e1;e2;e3;e4;e5;e6;e7;e8;e9;e10;e11;e12;e13;e14;e15];
    statevar = [d;k;a;muu]
    controlvar = [c; ivv; r; output; h; la; kfu;tb; tby; ca; cay;]
    FF = SGU.linearize(f, statevar, controlvar, params, vars, shocks, stdeps)
    return FF
end

function steadystate!(FF)
    sigg = 2. 
    delta = 0.1 
    rstar = 0.04
    alfa = 0.32 
    omega = 1.455 
    dbar =  0.74421765717098 
    pssi = 0.11135/150 
    phi = 0.028 
    rho = 0.42
    etatilde = 0.0129 
    betta = 1/(1+rstar)

    r      = rstar
    d      = dbar
    kapa   = ((1/betta - (1-delta)) / alfa)^(1/(alfa-1))
    h      = ((1-alfa)*kapa^alfa)^(1/(omega -1))
    k      = kapa * h
    kfu    = k
    output = kapa^alfa * h
    c      = output-delta*k-rstar*dbar
    ivv    = delta * k
    tb     = output - ivv - c
    tby    = tb/output
    ca     = -r*d+tb
    cay    = ca/output
    
    la     = log(((c - h^omega/omega))^(-sigg))
    c      = log(c); 
    h      = log(h)
    k      = log(k); 
    kfu    = log(kfu); 
    output = log(output); 
    ivv    = log(ivv); 
    a      = log(1.)
    r      = log(r)
    tb     = (tb)
    tby    = (tby)
    ca     = (ca)
    cay    = (cay)
    muu    = log(1);

    @pack! FF.params = sigg, delta, rstar, alfa, omega, dbar, pssi, phi, rho, etatilde, betta 
    @pack! FF.vars = la, c, k, kfu, h, d, output, ivv, tb, tby, ca, cay, r, a, muu 
end

@time FF = model(); 
@time steadystate!(FF); 
@time G = SGU.num_eval(FF); 

x0 = zeros(G.nx); 
x0[G.ix[:a]] = 1; 
IR, IRy, IRx = SGU.ir(G.gx, G.hx, x0, 12); 

p1 = plot(IRy[:,G.iy[:output]], title="output"); 
p2 = plot(IRy[:,G.iy[:c]], title="consumption"); 
p3 = plot(IRy[:,G.iy[:ivv]], title="investment"); 
p4 = plot(IRy[:,G.iy[:tby]], title="tby"); 
p5 = plot(IRy[:,G.iy[:cay]], title="cay"); 
p6 = plot(IRy[:,G.iy[:h]], title="labor"); 
plot(p1, p2, p3, p4, p5, p6)

