#Solving IFP with policy function iteration 
#Author: Hanjo Kim (terryhanjokim@gmail.com) 
#Last updated 8/30/2024
using Plots, NLsolve, QuantEcon, Parameters, BasisMatrices

get_params = @with_kw (β=0.94, γ=2) 

function get_settings()
    ny = 5
    ρy = 0.9
    σe = 0.1
    mc = rouwenhorst(ny, ρy, σe)
    ygrid = exp.(mc.state_values); 
    pmat = mc.p 

    na = 150
    amin = -2
    amax = 80
    agrid = range(amin, amax, length=na); 
    
    maxit = 300
    tol = 1e-8

    fspace = Basis(SplineParams(agrid, 0, 1), SplineParams(ygrid, 0, 1)) 
    Grid, Nodes = nodes(fspace) 
    Phi = BasisMatrix(fspace, Expanded(), Grid, 0); 

    bigP = kron(pmat, ones(na, 1)) 
    ns = size(Grid, 1) 
    return (ny=ny, na=na, ygrid=ygrid, pmat=pmat, agrid=agrid, amin=amin, amax=amax, maxit=maxit, tol=tol, ns=ns, bigP=bigP, Grid=Grid, Phi=Phi, fspace=fspace)
end


@with_kw mutable struct policies
    ns::Int
    par_ap = zeros(ns)
    ap = ones(ns)
    c = ones(ns) 
end 

function solve_hh(r, w, z, params, settings; disp=true) 
    @unpack ns, maxit, tol = settings; 
    pols = policies(ns=ns); 
    old_par = zeros(ns)
    new_par = zeros(ns) 
    @views @. pols.ap = settings.Grid[:,1]    
    it=0
    dc=1
    @inbounds while it<=maxit && dc>tol 
        it += 1 
        update_pols!(r, w, z, pols, params, settings) 
        @views @. new_par = pols.par_ap; 
        dc = findmax(abs.(new_par.-old_par))[1]; 
        @views @. old_par = pols.par_ap; 
        if mod(it, 100)==0
            println("Iterations: $(it). DC: $(dc).")
        end
    end 
    return pols
end

function update_pols!(r, w, z, pols, params, settings) 
    @unpack γ, β = params; 
    @unpack ns, fspace, Grid, Phi = settings; 
    @unpack par_ap, ap, c = pols; 
    for i=1:ns 
        f(x) = eulerres(r, w, z, x[1], i, pols, params, settings)     
        res = nlsolve(f, [ap[i]], iterations=100, show_trace=false) 
        ap[i] = res.zero[1]
    end
    @views par_ap .= Phi.vals[1]\ap 
    @views @. c = (1+r)*Grid[:,1] - ap + w*Grid[:,2]
    @pack! pols = par_ap, ap, c
end 

function eulerres(r, w, z, x, i, pols, params, settings) 
    @unpack ns, ny, Grid, fspace, bigP, ygrid, agrid = settings; 
    @unpack γ, β = params; 
    @unpack par_ap = pols; 
    c = (1+r)*Grid[i,1] - x + w*Grid[i,2]
    if c<=0
        LHS = 1e100
    else
        LHS = c^(-γ)
    end 
    RHS = 0
    @inbounds @simd for k=1:ny 
        apnext = funeval(par_ap, fspace, [x; ygrid[k]])[1]
        cnext = (1+r)*x - apnext + w*ygrid[k]
        if cnext<=0 
            RHS += 1e100
        else
            RHS += bigP[i,k]*β*(1+r)*cnext^(-γ)
        end
    end 
    return min((LHS-RHS)/LHS, x-agrid[1])
end

params = get_params(); 
settings = get_settings(); 

#Steady state prices
r = 0.04 #Interest rate
w = 1.01 #Wage rates
z = 1.00 #TFP

@time pols = solve_hh(r, w, z, params, settings); 
p1 = plot(settings.agrid, reshape(pols.c, settings.na, settings.ny), title="Consumption")
p2 = plot(settings.agrid[1:50], reshape(pols.ap, settings.na, settings.ny)[1:50,[1,end]], title="Savings")
plot(p1, p2)


