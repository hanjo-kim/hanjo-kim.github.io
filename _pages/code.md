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

Package: [SGU.jl](https://hanjo-kim.github.io/files/codes/SGU.jl)

Examples: 
- Closed economy one-sector RBC: [one_sector_rbc.jl](https://hanjo-kim.github.io/files/codes/one_sector_rbc.jl)
- Small open-economy one-sector RBC: [one_sector_soe.jl](https://hanjo-kim.github.io/files/codes/one_sector_soe.jl)

[comment]: # (- Two country one-sector RBC: [one_sector_two_country.jl](https://hanjo-kim.github.io/files/codes/one_sector_two_country.jl))

### Global methods: 
Three ways to solve the income fluctuation problem.

- Value function iteration: [vfi.jl](https://hanjo-kim.github.io/files/codes/vfi.jl)
- Policy function iteration: [pfi.jl](https://hanjo-kim.github.io/files/codes/pfi.jl)
- Endogenous grid method: [egm.jl](https://hanjo-kim.github.io/files/codes/egm.jl)


### Portfolio Optimization: 
Solve for the optimal portfolio that minimizes variance subject to a short-selling constraint using both the Monte Carlo method and the root-finding method. 

- Portfolio optimization: [portfolio_variance_minimization.jl](https://hanjo-kim.github.io/files/codes/portfolio_variance_minimization.jl) 
