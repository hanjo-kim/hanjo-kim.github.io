---
layout: archive
title: "Code"
permalink: /code/
author_profile: true
---

This page contains Julia code to solve various economics and finance problems that I have written over the years. 

This code is provided "as is" without warranty of any kind, express or implied. While efforts have been made to ensure accuracy, all errors or omissions are my own. 

Any comments, suggestions, and corrections are welcome and can be emailed to terryhanjokim@gmail.com.

### Local methods 
This code translates Schmitt-grohe and Uribe's Matlab codes to log-linearize and solve models to Julia. In particular, I use SymEngine to symbolically differentiate first-order conditions and uses lambdify to quickly evaluate them numerically. 

Package: [SGU.jl](https://www.dropbox.com/scl/fi/1bdlrpq06nehhvtwavgs1/SGU.jl?rlkey=e3nio2ovkllw326xhsbfxzyaz&st=60uwgb6w&dl=1)

Examples: 
- Closed economy one-sector RBC: [one_sector_rbc.jl](https://www.dropbox.com/scl/fi/s9xy6tr2a69dh1i1o218z/one_sector_rbc.jl?rlkey=p8zvgar91985hqv01cuig6gbx&dl=1)
- Small open-economy one-sector RBC: [one_sector_soe.jl](https://www.dropbox.com/scl/fi/l6wznkzx95eyc1jcfwz26/one_sector_soe.jl?rlkey=gs63rtnrjf3pud8gngjguiscu&dl=1)

[comment]: # (- Two country one-sector RBC: [one_sector_two_country.jl](https://www.dropbox.com/scl/fi/hpnfrsei7q0bmykzkmred/one_sector_two_country.jl?rlkey=0e9727nt4n1wo10axzk6hthd0&dl=1))

### Global methods: 
Three ways to solve the income fluctuation problem.

- Value function iteration: [vfi.jl](https://www.dropbox.com/scl/fi/yzyba0vp15cpbsk61itg6/vfi.jl?rlkey=ys2ktl8962kzkzs2s4aa2fz8a&dl=1)
- Policy function iteration: [pfi.jl](https://www.dropbox.com/scl/fi/lmpwtqzb6v2ehpxiewbm7/pfi.jl?rlkey=xn0vgm3yvyxdckoewmrdy2zho&dl=1)
- Endogenous grid method: [egm.jl](https://www.dropbox.com/scl/fi/0wgatpagmjj9my8qm8mnp/egm.jl?rlkey=mbbj6k8yzgkmjip6j37wi15aq&dl=1)


### Portfolio Optimization: 
Solve for the optimal portfolio that minimizes variance subject to a short-selling constraint using both the Monte Carlo method and the root-finding method. 

- Portfolio optimization: [portfolio_variance_minimization](https://www.dropbox.com/scl/fi/8a9bwk6nv2cwh4du1azm0/portfolio_variance_minimization.jl?rlkey=15q0wncizu3gnovu95y18i46b&dl=1) 
