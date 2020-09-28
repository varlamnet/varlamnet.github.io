---
layout: post
title: Nonparametric estimation
subtitle: Nadaraya-Watson and other basics
thumbnail-img: assets/img/nonparametric.png
comments: false
---
<!-- MathJAx Import -->
$$\newcommand{\abs}[1]{\left\lvert#1\right\rvert}$$
$$\newcommand{\norm}[1]{\left\lVert#1\right\rVert}$$
$$\newcommand{\inner}[1]{\left\langle#1\right\rangle}$$
$$\newcommand{\as}{\overset{a.s.}{\to}}$$
$$\newcommand{\d}{\overset{d}{\to}}$$
$$\DeclareMathOperator*{\argmin}{arg\,min}$$
$$\DeclareMathOperator*{\argmax}{arg\,max}$$
$$\DeclareMathOperator*{\E}{\mathbb{E}}$$
<!-- MathJAx End -->
<p style="margin-bottom:-2cm;"></p>

<style>
.dropcap {
  margin: 5px 7px -30px 0;
  }
</style>

<div class="alert">
  <span class="closebtn" onclick="this.parentElement.style.display='none';">&times;</span> 
  <i class="fas fa-exclamation-circle fa-lg"></i><strong> This section is under development!</strong>
</div>

Add: Bandwidth Estimation, As properties, LC, LL

# Nonparametric Density Estimation
- For discrete $X$:\\
$$\displaystyle \; \hat{f}(x)=\frac{n_0}{n}=\frac{1}{h}\frac{\text{# of $x_i=x$}}{n}=\frac{1}{n}\sum_{i=1}^{n} \mathbb{1} \{x_i=x \}$$

- For continuous $X$:\\
$$\displaystyle \hat{f}(x)=\frac{1}{h}\frac{n_0}{n}=\frac{1}{h}\frac{\text{# of $x_i \in $}(x-h/2,x+h/2)}{n}=\frac{1}{hn}\sum_{i=1}^{n} \mathbb{1} \{\frac{x_i-x}{h} \}$$ \\
Note: $$ f(x)=\frac{d}{dx} F(x)={\displaystyle\lim_{h\to0}}\frac{F(x+h/2)-F(x-h/2)}{h}= {\displaystyle\lim_{h \to 0}}\frac{P(x-h/2< X < x+h/2)}{h}$$

- $\hat{f}(x)$ is not differentiable, use kernel, Rosenblatt (1952): $$\hat{f}(x)=\dfrac{1}{nh}\sum_{i=1}^{n} \mathbb{K}(\dfrac{x_i-x}{h})$$ \\
(i) standard normal  $$\mathbb{K}(\psi)=\frac{1}{\sqrt{2\pi}}\exp^{-\frac{1}{2}\psi^2}$$\\
(ii) uniform  $$\mathbb{K}(\psi)=(2c)^{-1}, \text{ for } -c < \psi < c $$ and $0$ o/w.

# Bias and Variance of $\hat{f}$
Denote $\hat{f}=\frac{1}{n}\sum_{i=1}^{n}z_i$, where $z_i = \frac{1}{h}\mathbb{K}(\frac{x_i-x}{h})$.

- $$\displaystyle \E{\hat{f}(x)} = \E{z_1} = \frac{1}{h}\int_{x_1}\mathbb{K}(\underbrace{\frac{x_1-x}{h}}_{\equiv\psi})\underbrace{f(x_1)}_{unknown}dx_1 = \frac{1}{h} \int_{\psi}\mathbb{K}(\psi)f(x+h\psi)h d\psi $$ \\
$$ \approx \int_{\psi}\mathbb{K}(\psi)\big[f(x) + h\psi f^{(1)}(x) + \frac{h^2\psi^2}{2!} f^{(2)}(x) \big]d\psi $$\\
$$= f(x) \times 1 + hf^{(1)}(x) \times 0 + \frac{h^2}{2}f^{(2)}(x)\underbrace{\int_{\psi}\psi^2\mathbb{K}(\psi)d\psi}_{\equiv \mu_2}$$\\
$$= f(x) + \frac{h^2}{2}f^{(2)}(x)\mu_2$$
<hr class="new1" style="border-top: 1px solid grey">

{: .box-success}
$$\displaystyle \text{BIAS}(\hat{f}(x)) = \frac{h^2}{2}f^{(2)}(x)\mu_2 = O(h^2)$$\\
$$\displaystyle \mathbb{V}(\hat{f}(x)) = \frac{1}{n}\mathbb{V}(z_1) = \frac{1}{n}\big[ \E{z_1^2-(\E{z_1})^2} \big] =$$

- $$\displaystyle \E{z_1^2}=\frac{1}{h^2}\int_{x_1}\mathbb{K}^2(\frac{x_1-x}{h})f(x_1)dx_1
=\frac{1}{h^2} \int_{\psi}\mathbb{K}^2(\psi)f(x+h\psi)hd\psi$$\\
$$\stackrel{\textit{Taylor}}{\approx}
\frac{1}{h} \int_{\psi}\mathbb{K}^2(\psi)\big[ f(x) + h\psi f^{(1)}(x) \big] d\psi\\
=\frac{f(x)}{h}\int_{\psi}\mathbb{K}^2(\psi)d\psi + f^{(1)}(x)\int_{\psi}\psi\mathbb{K}^2(\psi)d\psi $$


$$= \displaystyle  \underbrace{\frac{f(x)}{nh}\int_{\psi}\mathbb{K}^2(\psi)d\psi} _{O(\frac{1}{nh})} + \underbrace{\frac{f^{(1)}(x)}{n}\int _{\psi}\psi\mathbb{K}^2(\psi)d\psi} _{O(\frac{1}{n})} - \underbrace{\frac{1}{n} (\E{z_1})^2} _{O(\frac{1}{n})+O(\frac{h^2}{n})}  \approx \underbrace{\frac{f(x)}{nh}\int _{\psi}\mathbb{K}^2(\psi)d\psi}_{O(\frac{1}{nh})}$$

<hr class="new1" style="border-top: 1px solid grey">

- $$ \displaystyle \stackrel{Local}{\text{MSE}}(\hat{f}(x))=\text{BIAS}^2(\hat{f}(x)) + \mathbb{V}(\hat{f}(x))$$ \\
  $$= \frac{h^4}{4}(f^{(2)}(x))^2\mu_2^2 + \frac{f(x)}{nh}\int_{\psi}\mathbb{K}^2(\psi)d\psi $$\\
  $$= h^4\lambda_1(x)+\frac{1}{nh}\lambda_2(x) $$

    - $$\displaystyle \lambda_1(x) \equiv \frac{1}{4}(f^{(2)}(x))^2\mu_2^2$$
    - $$\displaystyle \lambda_2(x) \equiv f(x)\int_{\psi}\mathbb{K}^2(\psi)d\psi,$$
 	
- $$ \displaystyle \stackrel{Global}{\text{IMSE}}(\hat{f}(x))= \int_{x}\stackrel{Local}{\text{MSE}}(\hat{f}(x))dx$$ \\
$$= h^4\int_{x}\frac{1}{4}(f^{(2)}(x))^2\mu_2^2dx + \frac{1}{nh}\int_{x}f(x)\int_{\psi}\mathbb{K}^2(\psi)d\psi dx$$ \\
$$= h^4\frac{1}{4}\mu_2^2\int_{x}(f^{(2)}(x))^2dx + \frac{1}{nh}\int_{\psi}\mathbb{K}^2(\psi)d\psi 1 $$ \\
$$= h^4\lambda_1+\frac{1}{nh}\lambda_2 $$

    - $$\displaystyle \lambda_1 \equiv \frac{1}{4}\mu_2^2\int_{x}(f^{(2)}(x))^2dx \leftarrow unknown$$
    - $$\displaystyle \lambda_2 \equiv \int_{\psi}\mathbb{K}^2(\psi)d\psi$$
	
- Choose bandwidth $$h$$ to minimize IMSE\\
	$$ \displaystyle \frac{\partial \text{IMSE}}{\partial h}= 4h^3 \lambda_1 - \frac{1}{nh^2}\lambda_2 =0 \rightarrow h_{opt}=n^{-1/5}(\frac{\lambda_2}{4\lambda_1})^{1/5} \propto n^{-1/5}$$
	
- Substitute $$h_{opt}$$ into IMSE and minimize wrt to $$K(\psi)$$ s.t. $$\int K{\psi} d\psi =1$$ and $$\int \psi^2 K{\psi} = 1$$ to obtain\\
	$$ \displaystyle K^{opt}(\psi)=\begin{cases} \frac{3}{4}(1-\psi^2), |\psi|\le 1 \\ 0, \text{ o/w} \end{cases}, \text{ Bartlett's (Epanechnikov's) Kernel}$$

