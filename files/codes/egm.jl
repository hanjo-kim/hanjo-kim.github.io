#solving income fluctuation problems with endogenous grid method 
#Author: Hanjo Terry Kim (terryhanjokim@gmail.com)
#Last updated: Aug. 30th 2024
using Parameters, Plots, QuantEcon, Interpolations

@with_kw mutable struct policies
    na::Int
    ny::Int
    c  = zeros(na, ny)
    dv = zeros(na, ny)
    ap = zeros(na, ny)
    h  = zeros(na, ny)
end

get_params = @with_kw (γ=2, β=0.94, r=0.04, ψ=1/2) 

function get_settings(params)
    ny = 11
    mc = tauchen(ny, 0.9, 0.1); 
    pmat  = mc.p; 
    ygrid = exp.(mc.state_values)

    amin = 0 
    amax = 80
    na=300
    agrid = range(amin, amax, length=na) 
    return (pmat=pmat, ygrid=ygrid, agrid=agrid, na=na, ny=ny, nafine=na*2)
end

function get_policies(params, settings, maxit=500, tol=1e-5)
    @unpack na, ny, agrid, ygrid = settings
    @unpack β, γ, r = params; 

    pols = policies(na=na, ny=ny) 
    pols.dv=ones(na, ny).*(1 ./(agrid.+1))
    old_dv = zeros(na, ny)
    new_dv = zeros(na, ny)
    it=0
    dc=1
    @inbounds while it<maxit && dc>tol
        it+=1
        old_dv .= pols.dv
        update_pols!(pols, params, settings)
        new_dv .= pols.dv
        dc = findmax(abs.(old_dv.-new_dv))[1]
        if mod(it, 100)==0
            println("Iteration: $(it). Distance: $(dc).")
        end
    end
    return pols
end

function update_pols!(pols, params, settings)
    @unpack na, ny, agrid, ygrid, pmat = settings
    @unpack β, γ, r, ψ = params
    @unpack c, ap, dv, h = pols 
    
    temp_c = zeros(na, ny)
    temp_a = zeros(na, ny) 
    temp_dv = zeros(na, ny)
    temp_h = zeros(na, ny)

    Edv = dv*pmat' 

    for j=eachindex(ygrid)
        for i=eachindex(agrid)
            temp_c[i,j] = (β*Edv[i,j])^(-1/γ)
            temp_h[i,j] = (temp_c[i,j]^(-γ)/ygrid[j])^ψ
            temp_a[i,j] = (-ygrid[j]+temp_c[i,j]+agrid[i])/(1+r)
            temp_dv[i,j] = temp_c[i,j].^(-γ)*(1+r)
        end
        
        #interpolate back to a'(a, y)
        ord = sortperm(temp_a[:,j])
        a_ = temp_a[ord,j]
        ap_ = agrid[ord]
        h_  = temp_h[ord,j]; 
        li = linear_interpolation(a_, ap_, extrapolation_bc=Line())
        lih = linear_interpolation(a_, h_, extrapolation_bc=Line())
        ap[:,j] = max.(agrid[1], li.(agrid))
        h[:,j]  = min.(1, max.(1e-4, lih.(agrid)))
        c[:,j] .= (1+r).*agrid .+ ygrid[j] .- ap[:,j]
        dv[:,j] .= c[:,j].^(-γ)*(1+r)
    end 
    
    @pack! pols = c, ap, dv, h 
end

params= get_params();
settings = get_settings(params);

@time pols = get_policies(params, settings);
p1 = plot(settings.agrid, pols.c[:,[1,5,11]], title="Consumption")
p2 = plot(settings.agrid[1:50], reshape(pols.ap, settings.na, settings.ny)[1:50,[1,end]], title="Savings")
plot(p1, p2)

