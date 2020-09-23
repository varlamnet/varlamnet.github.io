---
layout: post
title: Factor Analysis
subtitle: Classical vs high-dimensional
comments: false
---
<!-- MathJAx Import -->
$$\newcommand{\abs}[1]{\left\lvert#1\right\rvert}$$
$$\newcommand{\norm}[1]{\left\lVert#1\right\rVert}$$
$$\newcommand{\inner}[1]{\left\langle#1\right\rangle}$$
$$\DeclareMathOperator*{\argmin}{arg\,min}$$
$$\DeclareMathOperator*{\argmax}{arg\,max}$$
$$\DeclareMathOperator*{\E}{\mathbb{E}}$$
<!-- MathJAx End -->
<p style="margin-bottom:-2cm;"></p>

<div class="alert">
  <span class="closebtn" onclick="this.parentElement.style.display='none';">&times;</span> 
  <i class="fas fa-exclamation-circle fa-lg"></i><strong> This section is under development!</strong>
</div>

Factor models used for Asset Pricing, Business Cycle Analysis, Monitoring/Forecasting, Consumer Theory, etc.

$$\boxed{X_{it} = \lambda_i'F_t + e_{it}} \quad i\le N, \; t\le T.$$

## Classical FA
- $$T \to \infty$$, $$N$$ fixed
- $$e_{it}$$ are _iid_ over $$t$$ and ind. over $$i$$, i.e. $$\Omega := \E(e_te_t')$$ is a diagonal matrix
- $$F_t$$ are _iid_ and ind. of $$e_{it}$$
- $$\widehat{\Sigma}$$ is assumed to be $$\sqrt{T}$$-consistent for $$\Sigma$$
- $$\lambda_i$$ can be consistently estimated, but not $$F_t$$ (Anderson 1984)

## Approximate FA (Chamberlain & Rothschild 1983)
- nondiagonal $$\Omega := \E(e_te_t')$$ allowed
- $$\Omega := \E(e_te_t')$$ has bounded eigenvalues
- Hence largest eigenvalue bounded by $$\boxed{\underset{i}{\max} \sum_{j=1}^N\abs{\Omega_{ij}}}$$
- PCA ~ FA, when $$N\to\infty$$ (assuming $$\Sigma$$ known)
- Connor & Korajczyk: unknown $$\Sigma$$ and $$N\gg T$$

## Strict FA (APT of Ross 1976)
- $$e_{it}$$ is uncorrelated across $$i$$

## HD FA (Bai 2003)
- $$N,T \to \infty$$ (also $$N\to \infty, T$$ fixed)
- serial and cross-section dependence for the idiosyncratic errors
- heteroskedasticity in both dimensions

# POET (Fan 2013)

$$\boxed{y_{it} = b_i'f_t + u_{it}} \quad i\le p, \; t\le T.$$\\
$$\Sigma = B cov(f_t)B' + \Sigma_u$$

