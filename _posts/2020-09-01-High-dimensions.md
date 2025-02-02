---
layout: post
title: Statistics in high dimensions
subtitle: What could go wrong?
thumbnail-img: assets/img/cov.png
comments: false
---

<!-- MathJAx Import -->

$$\newcommand{\abs}[1]{\left\lvert#1\right\rvert}$$
$$\newcommand{\norm}[1]{\left\lVert#1\right\rVert}$$
$$\newcommand{\as}{\overset{a.s.}{\to}}$$
$$\DeclareMathOperator*{\E}{\mathbb{E}}$$

<!-- MathJAx End -->
<p style="margin-bottom:-2cm;"></p>

<style>
.dropcap {
  margin: 5px 7px -30px 0;
  }
</style>

# Classical vs High-dimensional

<p class="dropcap">L</p>et $n$ be # of observations, $p$ be # of variables.
The <b>classical</b> regime allows $n$ to diverge, but assumes $p$ fixed. In contrast, the <b>high-dimensional</b> regime permits both $n$ and $p$ to diverge, $ p/n \to \gamma > 0$. Many of the classical results break down in that case. Here I consider eigenvalues and eigenvectors of a high-dimensional covariance matrix. This has immediate implications for covariance estimation, but also for all the statistical tools based on covariance estimates: PCA, GLS, GMM, classification, portfolio optimization, etc.

Consider a simple case &nbsp; $X_i \overset{iid}{\sim} \mathcal{N}_p(\mathbf{0}, \Sigma),\quad i=1,\ldots, n.$

**How to estimate $\Sigma$?**

{: .box-note}
Notation:
<span style="display:block; height: 10px;"></span>
Sample covariance estimator  &nbsp; $S = \frac{1}{n}\sum_i^n X_iX_i' = \frac{1}{n} X'X.$
<span style="display:block; height: 10px;"></span>
Eigendecompositions  &nbsp; $\Sigma = ULU' = \sum_j^p \ell_j \mathrm{u}_j \mathrm{u}_j', \quad S = V\Lambda V' = \sum_j^p \lambda_j \mathrm{v}_j \mathrm{v}_j'.$
<span style="display:block; height: 10px;"></span>
Eigenvalues distinct, sorted in decreasing order.
<span style="display:block; height: 10px;"></span>
Eigenvectors chosen with the first element positive.

# Clasical Regime
In a classical regime, $S$ is a very good estimator (Anderson 1963, Van der Vaart 2000): <span style="display:block; height: 10px;"></span>
<i class="fas fa-check-circle"></i> Unbiased  &nbsp; $\E(S) = \Sigma.$ \\
<i class="fas fa-check-circle"></i> Consistent  &nbsp; $S \as \Sigma$ as $n\to\infty.$\\
<!-- <i class="fas fa-check-circle"></i> Eigenvalues converge  &nbsp; $\lambda_j \as \ell_j$ as $n\to\infty, \quad j=1,\ldots,p.$\\ -->
<i class="fas fa-check-circle"></i> Asymptotically normal eigenvalues  &nbsp; $$ \sqrt{n}(\lambda_i-\ell_i) \overset{d}{\to} \mathcal{N}(0,2\ell_i^2), \quad j=1,\ldots,p.$$\\
<i class="fas fa-check-circle"></i> Is invertible.\\
<!-- <i class="fas fa-check-circle"></i> Eigenvectors converge  &nbsp; $\mathrm{v}_j \as \mathrm{u}_j$ as $n\to\infty, \quad j=1,\ldots,p.$ -->

<i class="fas fa-exclamation-triangle" style="color:#f44336"></i> <em>It gets trickier in high dimensions</em> 
<span style="display:block; height: 10px;"></span>
It is especially interesting what happens to eigenvalues and eigenvectors in high dimensions. There are three key features: <b>eigenvalue spreading</b>, <b>eigenvalue bias</b> and <b>eigenvectors inconsistency</b>.

<!-- LINE & JUMP -->
<span style="display:block; height: 0px;"></span>
<hr style="border-top: 1px solid grey">

# High-dimensional Regime
## Eigenvalue spreading
#### Marchenko-Pastur (1967)
In high dimensions, sample eigenvalues $\lambda_j$ are more spread out than their population counterparts $\ell_j.$ In fact, the higher the dimension, the more is the spreading. 

Consider the case when $\Sigma = I_p,$ i.e. $\ell_1 = \ldots = \ell_p = 1,$ and $p/n \to \gamma \le 1.$ 

{: .box-note}
Empirical d'n of eigenvalues of sample covariance  &nbsp; $$F_p(x) := \frac{1}{p} \# \{ \lambda_j\le x  \}$$

Ukranian mathematicians Marchenko & Pastur (MP) showed that this empirical d'n converges $F_p(x) \to F(x),$ with the limit pdf given by:

{: .box-success}
$$f^{MP}(x) = \frac{\sqrt{(\lambda_+-x)(x-\lambda_-)}}{2\pi x \gamma}, \quad \lambda_+ = (1+\sqrt{\gamma})^2, \quad \lambda_- = (1-\sqrt{\gamma})^2.$$

{: .box-success}
$$\begin{split} F(x) = & \frac{1}{2} + \frac{1}{2\pi \gamma} \Big[\sqrt{(\lambda_+-x)(x-\lambda_-)} \\ & + (1+\gamma)\arcsin(\frac{x-1-\gamma}{2\sqrt{\gamma}}) + (1-\gamma)\arcsin(\frac{(1-\gamma)^2-(1+\gamma)x}{2x\sqrt{\gamma}})\Big]. \end{split}$$


<iframe src="/assets/html/high_dim_plot_1.html" width="100%" height="500px" style="border:none;"></iframe>

Some properties: <span style="display:block; height: 10px;"></span>
<i class="fas fa-check"></i> Mean $1,$\\
<i class="fas fa-check"></i> Mode $\frac{(1-\gamma)^2}{1+\gamma},$ \\
<i class="fas fa-check"></i> Median $$m(\gamma),$$ with $$1 - (\sqrt{2}-1)\gamma < m(\gamma) < 1$$ and $$\underset{\gamma\to 0}{\lim} m(\gamma) = 1-\frac{\gamma}{3} + \mathcal{o}(\gamma).$$

#### Quarter circle Law
An interesting special case is when $\gamma = 1.$ Then the d'n of normalized sample singular values of $X,$ $s_i/\sqrt{n},$ converges to the "quarter circle" law:

$$f^{Q}(x) = \frac{\sqrt{(4-x^2)}}{\pi}, \quad 0\le x \le 2,$$

that is, the singular values of a random normal square matrix lie on a quarter circle. Moreover, its moments are Catalan numbers.

#### Bai & Yin's (1993) Law
Also when $\Sigma = I_p$ and $\gamma \le 1$, the largest and smallest eigenvalues converge almost surely to the corresponding boundaries of the support, 

$$\lambda_1 \as \lambda_+ \quad \text{and} \quad \lambda_p \as \lambda_-.$$

Notice that the larger is $\gamma$, the wider is the spreding and the stronger is the eigenvalues bias! This phenomenon is very general and is not limited to the identity case.

<hr style="border-top: 1px solid grey">

If $\gamma>1$, then the sample covariance has only $n$ positive eigenvalues, while the remaining $p-n$ equal zero.
In that case the limit distribution has a differential form and an isolated point zero is added to the support:

$$F(dx) = (1-1/\gamma) \delta_0(dx) + f^{MP}(x)dx,$$

where $\delta_0$ is the Dirac delta at $0$. 


## Eigenvalue bias
Let's consider a covariance with a few "spiked" eigenvalues.
#### BBP (2005) Phase transition

{: .box-note}
&#8226; $$X_i \overset{iid}{\sim} \mathcal{N}_p(0,\Sigma), \quad i=1,\ldots,n,$$ \\
&#8226; $$p/n \to \gamma, \quad 0< \gamma \le 1,\;$$ as $$\; n\to\infty,$$\\
&#8226; $$\Sigma = diag(\ell_1, \ldots, \ell_r, 1,\ldots,1), \quad \ell_r \ge 1$$

Top $r$ sample eigenvalues will converge, but not to their true counterparts. Depending on where the true counterparts are positioned wrt to the so-called <b>Baik-Ben Arous-Peche (BBP) transition point $$\lambda_+^{1/2}$$, </b>

{: .box-success}
$$\lambda_j \overset{as}{\longrightarrow} 
\begin{cases} 
    \lambda_+, \; & \ell_j < \lambda_+^{1/2}, \\ 
    \ell_j + \gamma \frac{\ell_j}{\ell_j-1}, \; & \ell_j > \lambda_+^{1/2},
\end{cases} \quad \text{for} \quad j = 1,\ldots, r.$$

<iframe src="/assets/html/high_dim_plot_2.html" width="100%" height="500px" style="border:none;"></iframe>

<i class="fas fa-exclamation-triangle" style="color:#f44336"></i> <em>$\lambda_1$ is asymptotically upward biased, while $\lambda_p$ will be downward biased</em> <i class="fas fa-exclamation-triangle" style="color:#f44336"></i>

<hr style="border-top: 1px solid grey">

#### Tracy-Widom (1996)
The exact asymptotic d'n is also known for both cases! 
Below the BBP transition point the top eigenvalues are distributed with <b>Tracy-Widom d'n</b> with rate $n^{2/3}$, above with Normal with rate $n^{1/2}:$

{: .box-success}
$$n^{2/3}\lambda_1 \overset{d}{\to} TW_1 \left(\lambda_+, \; \left(\frac{\lambda_+}{\gamma^{1/4}}\right)^{4/3} \right) \quad \text{if} \; \ell_j < \lambda_+^{1/2}$$

{: .box-success}
$$n^{1/2} \lambda_1 \overset{d}{\to} \mathcal{N}\left( \ell_j + \gamma \frac{\ell_j}{\ell_j-1}, \; 2\ell_j^2 \left(1-\frac{\gamma}{(\ell_j-1)^2} \right)\right) \quad \text{if} \; \ell_j > \lambda_+^{1/2}$$


That is, if the true spikes are not large enough, the sample eigendistribution will look like that of $\Sigma = I_p$, i.e. according to MP d'n.
In the opposite case, the spiked sample eigenvalues will overshoot the true counterparts and lie above the MP sea.



## Eigenvector inconsistency
Paul (2007) showed that when $$p/n \to \gamma \in (0,\infty)$$, the sample eigenvectors are not consistent and hence PCA would generally be inconsistent.
Their Theorem 4 characterizes precisely how bad this inconsistency is

{: .box-success}
$$\langle \mathrm{v}_j, \mathrm{u}_j \rangle^2 \as 
\begin{cases} 
    0, \quad & \ell_j < \lambda_+^{1/2}, \\
    \frac{1-\gamma/(\ell_j-1)^2}{1+\gamma/(\ell_j-1)}, \quad & \ell_j > \lambda_+^{1/2},  
\end{cases}\\
\abs{\langle \mathrm{v}_j, \mathrm{u}_k \rangle} \as 0, \quad \text{for} \quad j\ne k.\quad\quad\quad\quad\quad\;$$

In the special case where $$\Sigma = I$$ and the $$X_{ij}$$ are iid standard (real or complex) Gaussian random variables, it is known that the matrix of sample eigenvectors is Haar distributed.
<hr class="new1" style="border-top: 1px solid grey"> 


## PCA in high dimensions
#### Johnstone Lu (2009), Thm 1
Assume a $p$-dimensional one-factor model 

$$\mathrm{x}_i = v_i\rho + \sigma z_i, \quad i=1,\ldots,n,$$

and that $$\frac{p}{n} \to c$$ and $$\frac{\|\rho\|^2}{\sigma^2} \to \omega > 0$$ and define the normalized inner product (cos of the angle)

$$ R(\hat{\rho},\rho) = \frac{\hat{\rho}'\rho}{\|\hat{\rho}\|\|\rho\|}. $$

Then 

{: .box-success}
$$\lim R^2(\widehat{\rho}, \rho) = \frac{(\omega^2-c)_+}{\omega^2 +c\omega} \quad a.s.$$

<i class="fas fa-exclamation-triangle" style="color:#f44336"></i>  <em>i.e. PCA eigenvector estimate is consistent iff $$p/n \to 0$$.</em>

Paul (2007) shows that this is also true for spiked covariance.

Luckily, consistency can be recovered if there exists a sparse representation in some basis. In that case, PCA on a subset of variables with sufficiently high variability can yield consistent estimates.
