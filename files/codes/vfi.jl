#solving income fluctuation problems with value function iterations
#Original basismatrices algorithm taken the example in https://github.com/QuantEcon/BasisMatrices.jl/blob/master/demo/examples.ipynb 
#Author: Hanjo Terry Kim (terryhanjokim@gmail.com)  
#Last updated: Aug. 30th 2024
using Parameters, BasisMatrices, QuantEcon, Optim, SparseArrays, LinearAlgebra, Plots

get_params = @with_kw (β=0.94, γ=2, ϕ=0., ρ=0.95, σ=0.1, r=0.04)

function get_settings(params)
    @unpack β, γ, ϕ, ρ, σ = params; 
    #Discretize Shocks
    ny = 11
    pmat, ygrid = rouwenhorst(ny, ρ, σ).p, exp.(rouwenhorst(ny, ρ, σ).state_values)   
    
    #Set grid
    amin = ϕ
    amax = 100
    na = 200
    agrid = range(amin, amax, length=na); 

    #Create Interpolation matrix
    fspace = Basis(SplineParams(agrid, 0, 1), SplineParams(ygrid, 0, 1)); 
    Grid, Nodes = nodes(fspace); 
    Phi = BasisMatrix(fspace, Expanded(), Grid, 0).vals[1]
    Phi_y = BasisMatrix(fspace, Direct(), Grid, 0).vals[2] 
    Emat = kron(pmat, sparse(I, na, na))

    ns = na*ny 
    return (pmat=pmat, ygrid=ygrid, amin=amin, amax=amax, agrid=agrid, fspace=fspace, Grid=Grid, Nodes=Nodes, Phi=Phi, na=na, ny=ny, ns=ns, Emat=Emat, Phi_y=Phi_y) 
end

function solve_hh(params, settings; Nbell=15, maxit=50, tol=1e-8, disp=false)
    @unpack Emat, Phi, ns = settings;
    
    #policy functions
    ap = zeros(ns)
    c = zeros(ns) 
    v = zeros(ns) 

    #coefficients and jacobian
    old_par = Phi\(Emat*v)
    new_par = Phi\(Emat*v)
    D = zeros(ns, ns)

    #Initialize iteration and distance
    it = 0 
    dc = 1

    while it<=maxit && dc>tol 
        it +=1
        update_valfun!(ap, c, v, D, old_par, new_par, params, settings)
        if it<=Nbell 
            new_par .= Phi\(Emat*v); 
            dc = findmax(abs.(old_par.-new_par))[1];
            old_par .= new_par;
        else it>Nbell 
            new_par .= old_par .- D\(Phi*old_par.-Emat*v);
            dc = findmax(abs.(old_par.-new_par))[1];
            old_par.=new_par;
        end
        if disp
            if mod(it, 1) == 0
                println("Bellman iterations: $(it). DC: $(dc).")
            end
        end
    end

    if disp 
        println("Bellman iteration converged in $(it) iterations. DC: $(dc).")
    end 
    return (ap=ap, c=c, old_par=old_par)
end

function update_valfun!(ap, c, v, D, old_par, new_par, params, settings)
    @unpack ϕ, r, β = params;
    @unpack ns, Grid, Phi_y, fspace, Phi, Emat = settings;

    #For each state, solve the value function
    lower_bound = ϕ.*ones(ns)
    upper_bound = (1+r).*Grid[:,1].+Grid[:,2]
    @inbounds for i=1:ns
        f(x) = -value_function(x, old_par, i, params, settings) 
        res = optimize(f, lower_bound[i], upper_bound[i]) 
        ap[i] = res.minimizer[1]
        v[i] = -res.minimum[1]
    end

    #Calculate jacobian
    Phiap = BasisMatrix(fspace[1], Expanded(), ap, 0).vals[1]
    Phiprime = row_kron(Phi_y, Phiap) 
    D .= Phi .- Emat*β*Phiprime
    
    c .= (1+r).*Grid[:,1] .+ Grid[:,2] .- ap
end

function value_function(api, old_par, i, params, settings)
    @unpack β, r, γ = params;
    @unpack Grid, fspace, Phi_y = settings
    ctoday = (1+r)*Grid[i,1]+Grid[i,2]-api 
    Phiap = BasisMatrix(fspace[1], Expanded(), [api], 0).vals[1]
    Phiprime = vec(kron(Phi_y[i,:], Phiap)');
    EV = dot(Phiprime, old_par)
    if ctoday<=0
        v = -1e100
    else
        v = ctoday^(1-γ)/(1-γ)+β*EV 
    end
    return v
end

params = get_params(); 
settings = get_settings(params);
@time policy = solve_hh(params, settings, disp=true);

p1 = plot(settings.Nodes[1], reshape(policy.c, settings.na, settings.ny), title="Consumption"); 
p2 = plot(settings.agrid, policy.ap[:,[1,5,11]], title="Savings")
plot(p1, p2)

