---
layout: post
title: Norms and Inequalities
subtitle:
comments: false
---

<!-- MathJAx Import -->

$$\newcommand{\abs}[1]{\left\lvert#1\right\rvert}$$
$$\newcommand{\norm}[1]{\left\lVert#1\right\rVert}$$
$$\newcommand{\inner}[1]{\left\langle#1\right\rangle}$$
$$\DeclareMathOperator*{\argmin}{arg\,min}$$
$$\DeclareMathOperator*{\argmax}{arg\,max}$$
$$\DeclareMathOperator*{\E}{\mathbb{E}}$$
$$\DeclareMathOperator*{\V}{\mathbb{V}}$$

<!-- MathJAx End -->
<p style="margin-bottom:-2cm;"></p>

## Vector Norms

For $\mathrm{x} = (x_i)_{i=1}^n \in \mathbb{R}^n$:

- General ℓₚ-norm: $\norm{\mathrm{x}}_p = \left(\sum_i^n \abs{x_i}^p\right)^{1/p}$, $p \in [1,\infty)$
- Manhattan (ℓ₁)-norm: $\norm{\mathrm{x}}_1 = \sum_i^n \abs{x_i}$
- Euclidean (ℓ₂)-norm: $\norm{\mathrm{x}}_2 = \left(\sum_i^n x_i^2\right)^{1/2}$
- Maximum (ℓ∞)-norm: $\norm{\mathrm{x}}_\infty = \max_i \abs{x_i}$
- ℓ₀ "norm": Count of nonzero entries.

Usually, ℓ₂ is common for distances, ℓ₁ for sparsity, ℓ∞ for robustness.

## Key Vector Inequalities

{: .box-success}
$$\norm{\mathrm{x}}_q \le \norm{\mathrm{x}}_p \le n^{1/p - 1/q} \norm{\mathrm{x}}_q, \quad (0 < p < q \le \infty)$$

For example:

$$\norm{\mathrm{x}}_\infty \le \norm{\mathrm{x}}_2 \le \norm{\mathrm{x}}_1 \le \sqrt{n}\norm{\mathrm{x}}_2 \le n\norm{\mathrm{x}}_\infty$$

**Hölder's inequality** bounds the inner product,

$$\norm{\mathrm{x}^\top \mathrm{y}}_1 \le \norm{\mathrm{x}}_p \norm{\mathrm{y}}_q, \quad 1/p + 1/q = 1$$

which implies **Cauchy-Schwarz inequality** when $p=q=2$,

$$\norm{\mathrm{x}^\top \mathrm{y}}_1 \le \norm{\mathrm{x}}_2 \norm{\mathrm{y}}_2$$

and implies another interesting special case when $p=1, q=\infty$,

$$\norm{\mathrm{x}^\top \mathrm{y}}_1 \le \norm{\mathrm{x}}_1 \norm{\mathrm{y}}_\infty$$

**Minkowski inequality** is a triangle inequality for $L^p$ spaces,

$$\norm{\mathrm{x}+\mathrm{y}}_p \le \norm{\mathrm{x}}_p + \norm{\mathrm{y}}_p, \quad p \in [1,\infty]$$

which confirms that the shortest path between two points is a straight line, even when you change how you measure "distance" (the $p$-norm).

**Jensen's inequality** states that for a convex function of the average is less than or equal to the average of the function,

$$ f \left(\sum_i^n a_i x_i \right) \le \sum_i^n a_i f( x_i), \quad \sum_i^n a_i = 1 $$

## Matrix Norms

For $A \in \mathbb{R}^{m \times n}$, rank $r$, singular values $s_1 \ge \cdots \ge s_r$.

- Spectral (operator) norm:

$$
\norm{A} = \underset{x\in\mathbb{R}^n\backslash \{0\}}{\max} \frac{\norm{Ax}_2}{\norm{x}_2} = \underset{x\in S^{n-1}}{\max} \norm{Ax}_2 =
\underset{x\in S^{n-1},\ y\in S^{m-1}}{\max} \inner{Ax,y} = s_1(A) = \norm{s}_\infty
$$

- Frobenius (Euclidean) norm:

$$\norm{A}_F = \sqrt{\sum_i^m\sum_j^n \abs{A_{ij}}^2} = \sqrt{\sum_i^r s_i^2(A_{i})} = \sqrt{tr(A'A)} = \sqrt{\inner{A,A}} = \norm{s}_2$$

- Nuclear norm: $\norm{A}_* = \sum_i^r s_i$, sum of singular values.
- Hilbert-Schmidt norm: $\norm{A}_{HS} = \sqrt{\inner{A,A}_F}$, for operators.

## Key Matrix Inequalities

{: .box-success}
$$\norm{A} \le \norm{A}_F \le \sqrt{r} \norm{A}$$
Singular values bound: $$s_i \le \norm{A}_F / \sqrt{i}$$
{: .box-warning}

## SVD and Decompositions

Singular Value Decomposition: $$A = \sum_i^r s_i u_i v_i^T$$, where $$s_i = \sqrt{\lambda_i(A^T A)}$$.

### Courant-Fisher min-max Thm

Courant–Fischer variational representation of max eigenvalue & eigenvector:

$$\mathrm{v}_1(\widehat{\Sigma}) = \underset{\norm{z}_2 = 1}{\argmax} \; z'\widehat{\Sigma}z$$

Alternative equivalent variational representation is in terms of the semidefinite program (SDP):

$$Z^* = \underset{Z\in\mathbb{S}^p_+, \, tr(Z)=1}{\argmax} \, tr(\widehat{\Sigma}Z)$$

### Eckart-Young

Truncated SVD gives best low-rank approximation, key for dimensionality reduction in ML.

## Asymptotic Notation

For sequences $f_n, g_n$:

- $\mathcal{O}(g_n)$: $f_n$ grows no faster than $g_n$.
- $\Omega(g_n)$: No slower.
- $\Theta(g_n)$: Same rate.

Probabilistic versions:

- $\mathcal{O}_p(g_n)$: Bounded in probability.
- $\mathcal{o}_p(g_n)$: Converges to zero in probability.

Used in convergence proofs for stochastic gradient descent.

#### $$\mathcal{O}_p$$ and $$\mathcal{o}_p$$

$$\boxed{X_n = \mathcal{O}_p(g_n) \Longleftrightarrow \forall \epsilon>0, \, \exists M>0 \; \text{ s.t. } \; \mathbb{P}(\abs{X_n/g_n}\ge M) < \epsilon}$$
$$\boxed{X_n = \mathcal{o}_p(g_n) \Longleftrightarrow \forall \epsilon>0 \quad \underset{n\to\infty}{\lim} \mathbb{P}(\abs{X_n/g_n}\ge \epsilon) = 0}$$

if $$X_n = \mathcal{O}_p(f_n)$$ and $$Y_n = \mathcal{O}_p(g_n)$$:

- $$X_n Y_n = \mathcal{O}_p(f_ng_n) $$
- $$\abs{X_n}^s = \mathcal{O}_p(f_n^s), \quad s>0$$
- $$X_n + Y_n = \max\{\mathcal{O}_p(f_n), \mathcal{O}_p(g_n)\}$$

if $$X_n = \mathcal{o}_p(f_n)$$ and $$Y_n = \mathcal{o}_p(g_n)$$:

- $$X_n Y_n = \mathcal{o}_p(f_ng_n) $$
- $$\abs{X_n}^s = \mathcal{o}_p(f_n^s), \quad s>0$$
- $$X_n + Y_n = \max\{\mathcal{o}_p(f_n), \mathcal{o}_p(g_n)\}$$

if $$X_n = \mathcal{o}_p(f_n)$$ and $$Y_n = \mathcal{O}_p(g_n)$$:

- $$X_n Y_n = \mathcal{o}_p(f_ng_n) $$
- $$X_n + Y_n = \mathcal{O}_p(g_n)$$

### Continuous mapping Thm

Given $$f: \mathbb{R}^k \to \mathbb{R}^m$$ is "almost surely continuous"

- $$X_n \overset{d}{\to} X \Longrightarrow f(X_n) \overset{d}{\to} f(X)$$
- $$X_n \overset{d}{\to} X \Longrightarrow f(X_n) \overset{p}{\to} f(X)$$
- $$X_n \overset{d}{\to} X \Longrightarrow f(X_n) \overset{as}{\to} f(X)$$

## Slutsky’s Lemma

if $$X_n \overset{d}{\to} X$$ and $$Y_n \overset{d}{\to} c$$:

- $$X_n + Y_n \overset{d}{\to} X + c$$
- $$X_nY_n \overset{d}{\to} cX$$
- $$X_n/Y_n \overset{d}{\to} X/c$$

### Johnson–Lindenstrauss

Given a set $Q$ of $q$ points in $\mathbb{R}^N$ with $N$ typically large, we would like to embed
these points into a lower-dimensional Euclidean space $\mathbb{R}^n$ while approximately preserving
the relative distances between any two of these points. The question is how
small can we make $n$ (relative to $q$) and which types of embeddings work?

{: .box-success}
Given $\epsilon \in (0, 1),$ for every set $Q$ of
$q$ points in $\mathbb{R}^N$, if $n$ is a positive integer such that $$n>n_0 = O(\ln(q)/\epsilon^2)$$,
there exists a Lipschitz f'n $f: \mathbb{R}^N \to \mathbb{R}^n$ s.t. \\
$$(1-\epsilon)\norm{u-v}^2_2 \le \norm{f(u) - f(v)}^2_2 \le (1+\epsilon)\norm{u-v}^2_2, \quad \forall \; u,v\in Q.$$
