---
layout: post
title: Perturbation bounds
subtitle: Davis-Kahan, Weyl and others
# cover-img: assets/img/path.jpg
# thumbnail-img: assets/img/cov.png
# share-img: assets/img/path.jpg
# gh-repo: varlamnet
# gh-badge: [star, fork, follow]
# tags: [Thms]
comments: true
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

# Perturbation bounds
Symmetric $$\Sigma, \widehat{\Sigma} \in \mathbb{R}^{p\times p}$$ w/ descending $$\{\lambda\}_{i=1}^p$$,
$$\{\widehat{\lambda}\}_{i=1}^p$$ and 
$$\{\mathrm{v}\}_{i=1}^p$$,
$$\{\widehat{\mathrm{v}}\}_{i=1}^p$$:

### **Davis-Kahan Thm** (1970): 
$$ \norm{\widehat{\mathrm{v}}_i - \mathrm{v}_i}
\le \dfrac{\sqrt{2}\norm{\widehat{\Sigma} - \Sigma}}{\min\{|\widehat{\lambda}_{i-1} - \lambda_i|,|\lambda_i - \widehat{\lambda}_{i+1}|\}},$$

### **Weyl's Thm**:
$$ |\widehat{\lambda}_i - \lambda_i| \le \norm{\widehat{\Sigma} - \Sigma}, \quad \forall i=1,\ldots,p.$$

### **Dual Weyl**
Symmetric $$A, B \in \mathbb{R}^{p\times p}$$, then $$\forall j = 1, . . . , p$$,

$$\begin{Bmatrix}  \lambda_j(A) & + & \lambda_p(B)\\ \lambda_{j+1}(A) & + & \lambda_{p-1}(B) \\ & \vdots & \\ \lambda_p(A) & + & \lambda_j(B) \end{Bmatrix} \le 
\lambda_j(A+B) \le 
\begin{Bmatrix}  \lambda_j(A) & + & \lambda_1(B)\\ \lambda_{j-1}(A) & + & \lambda_2(B) \\ & \vdots & \\ \lambda_1(A) & + & \lambda_j(B) \end{Bmatrix}.$$

(From TAO 254A Note 3a) Weyl:

$$\lambda_{i+j-1}(A+B) \le \lambda_{i}(A) + \lambda_{j}(B), \quad i,j\ge 1, \; i+j-1\le n.$$

### **Ky Fan inequality**
$$\lambda_{1}(A+B) + \cdots + \lambda_{k}(A+B) \le \lambda_{1}(A) + \cdots +\lambda_{k}(A) + \lambda_{1}(B) + \cdots +\lambda_{k}(B)$$

### **Eigenvalue stability inequality**
$$\abs{\lambda_{i}(A+B) -\lambda_{i}(A)}\le \norm{B}_{op}$$

that is, the spectrum of $$A+B$$ is close to that of $$A$$ if $$\norm{B}_{op}$$ is small.

### **Lindskii inequality**
$$\lambda_{i_1}(A+B) + \cdots + \lambda_{i_k}(A+B) \le \lambda_{i_1}(A) + \cdots +\lambda_{i_k}(A) + \lambda_{1}(B) + \cdots +\lambda_{k}(B),$$ 

for all $$1\le i_1 \le \cdots i_k \le n.$$

### **Dual Lindskii inequality**
$$\lambda_{i_1}(A+B) + \cdots + \lambda_{i_k}(A+B) \ge \lambda_{i_1}(A) + \cdots +\lambda_{i_k}(A) + \lambda_{n-k+1}(B) + \cdots +\lambda_{n}(B),$$ 

for all $$1\le i_1 \le \cdots i_k \le n.$$

### **Dual Weyl inequality**
$$\lambda_{i+j-n}(A+B) \ge \lambda_{i}(A) + \lambda_{j}(B), \quad 1\le i, j, i+j-n \le n.$$


### Cauchy's Eigenvalue Interlacing
$$A \in \mathbb{R}^{n\times n}$$ is symmetric, $$B \in \mathbb{R}^{m\times m},\; m< n$$ is a principal submatrix of $$A$$ (or a projection of $$A$$ onto $$m$$ coordinates). Then, their eigenvalues are interlaced,

$$\lambda_{i}(A) \ge \lambda_i(B) \ge \lambda_{i+n-m}(A), \quad i=1,\ldots,m.$$

E.g. if $$m=n-1$$

$$\lambda_{1}(A) \ge \lambda_1(B) \ge \ldots \ge \lambda_{n-1}(B) \ge \lambda_{n}(A).$$


### Weilandt-Hoffmann inequality:

$$\sum_{i=1}^n \abs{\lambda_i(A+B) -\lambda_i(A)}^2 \le \norm{B}_F^2 $$


### Curvature lemma
$$(\lambda_d - \lambda_{d+1}) \norm{\widehat{\Pi}_d - \Pi_d}_F^2 \le 2 \mathrm{Tr}(\Sigma(\Pi_d - \widehat{\Pi}_d))$$

<hr class="new1" style="border-top: 1px solid grey">

### ~Random fact
For matrix $$\mathrm{M}\in\mathbb{R}^{n\times p}$$ and a unit vector $$\mathrm{u}\in\mathbb{R}^{p}$$:

$$\lambda_1(\mathrm{M'M}) \ge \norm{\mathrm{Mu}}_2^2 \ge \lambda_p(\mathrm{M'M})$$

