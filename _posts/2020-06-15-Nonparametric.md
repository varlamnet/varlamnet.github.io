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
<!-- \newtheorem*{lemma}{Lemma} % Stars mean no numbering -->

<!-- MathJAx End -->
<p style="margin-bottom:-2cm;"></p>

<style>
.dropcap {
  margin: 5px 7px -30px 0;
  }
</style>

<div class="alert">
  <span class="closebtn" onclick="this.parentElement.style.display='none';">&times;</span> 
  <i class="fas fa-exclamation-circle fa-lg"></i><strong> lecture notes from many years ago </strong>
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

# Bandwidth Estimation

Note that $\lambda_1$ is unknown because of unknown $f(x)$, hence to obtain $h$ can use one of the following methods.

- **Ad-hoc Method**\\
Assume $f(x) \sim N(0, \sigma_x^2)$. For $K(\psi)$ normal, it can be shown that $h_{opt}=1.06\sigma_x n^{-1/5}$

- **Plug-in Method**\\
Repeat the following loop until the difference becomes small
\begin{enumerate}
  - Start with ad-hoc $h_{opt}$ and estimate $\hat{f}(x)$
  - Calculate $\hat{\lambda}_1$ using $\hat{f}^{(2)}(x)$ instead of $f^{(2)}(x)$ 
  - Use $\hat{\lambda}_1$ to calculate new $\hat{h}_{new}$
  - Repeat the loop using the last $\hat{h}_{new}$ obtained
\end{enumerate}

- **ISE (Cross-Validation) Method**\\
Minimize ISE($h$) wrt $h$ (use grid search)

  $$\displaystyle
\text{ISE}(h)= \int_{x} (\hat{f}(x) - f(x))^2 dx $$
$$\displaystyle  = \int_{x} \hat{f}^2(x) dx +\underbrace{\int_{x} f^2(x) dx}_{\text{can be dropped}} - 2\int_{x} \hat{f}(x)f(x) dx \\
= \int_{x} \hat{f}^2(x) dx - 2\E\hat{f}(x) 
= \int_{x} \hat{f}^2(x) dx - 2 \frac{1}{n}\sum_{i=1}^{n}\hat{f}(x_i) 
$$ <i class="far fa-arrow-alt-circle-right"></i>


  - $$\hat{f}(x)=\dfrac{1}{nh}\sum_{i=1}^{n} \mathbb{K}(\dfrac{x_i-x}{h})$$

  - $$\hat{f}^2(x)=\dfrac{1}{n^2h^2}\sum_{i=1}^{n}\sum_{j=1}^{n} \mathbb{K}(\dfrac{x_i-x}{h}) \mathbb{K}(\dfrac{x_j-x}{h}) $$


<i class="far fa-arrow-alt-circle-right"></i>
$$\displaystyle
\frac{1}{n^2h^2}\sum_{i=1}^{n}\sum_{j=1}^{n} \int_{x} \mathbb{K}(\frac{x_i-x}{h})\mathbb{K}(\dfrac{x_j-x}{h})dx - \frac{2}{n^2h}\sum_{i=1}^{n}\sum_{j=1}^{n} \mathbb{K}(\frac{x_i-x_j}{h})\\
=\frac{1}{n^2h^2}\sum_{i=1}^{n}\sum_{j=1}^{n} \int_{x} \mathbb{K}(\frac{x_i-x}{h})\mathbb{K}(\dfrac{x_j-x}{h})dx -
\frac{2}{n(n-1)h} \underset{i\ne j}{\sum^{n}\sum^{n}} \mathbb{K}(\frac{x_i-x_j}{h})
$$

# Asymptotic Properties of $\hat{f}$

{: .box-success}
Under assumptions (A1), (A4), (A8) and (A9) we have\\
$$\displaystyle
\frac{1}{h} \E K^r \left( \frac{x_1-x}{h} \right) = \int K^r(\psi) f(h\psi +x) d\psi 
\to f(x) \int K^r (\psi) d\psi \text{ as } n \to \infty$$ 



- **Asymptotic Mean**  <span style="display:block; height: 10px;"></span>
$$\displaystyle 
\E \hat{f}(x) = \E z_1  = \frac{1}{h} \E K(\frac{x_i-x}{h}) \overset{Lemma}{\to} f(x) \int_{\psi} K(\psi) d \psi = f(x) \text{ as } n\to \infty $$

- **Asymptotic Variance** <span style="display:block; height: 10px;"></span>
$$\displaystyle 
\mathbb{V} (\hat{f}(x)) = \mathbb{V} (\frac{1}{n} \sum_{i=1}^{n} z_i)$$
$$= \frac{1}{n} (\E z_1^2 -(\E z_1)^2) = \frac{1}{nh} \left[ \frac{1}{h} \E K^2(\frac{x_i-x}{h})\right] - \frac{1}{n}(\E z_1)^2 $$
$$nh \mathbb{V} (\hat{f}(x)) = \frac{1}{h}\E K^2(\frac{x_i-x}{h}) - h (\E z_1)^2 \overset{Lemma}{\to} f(x) \int K^2 (\psi) d\psi - 0 \cdot f^2(x)$$
$$= f(x) \int K^2 (\psi) d\psi  \text{ as } n\to \infty $$

- **Weak Consistency** <span style="display:block; height: 10px;"></span>
We have that MSE$$(\hat{f}(x)) \to 0$$ as $$n \to \infty$$. Use Chebyshev inequality to show that
$$\displaystyle P[|\hat{f} - f|\ge \epsilon] \le \frac{\E(\hat{f}-f)^2}{\epsilon^2}$$, and hence
$$\displaystyle P[|\hat{f} - f|\le \epsilon] \to 1$$, i.e. $$\displaystyle p \lim_{n \to n} \hat{f} = f$$

- **Asymptotic Normality** <span style="display:block; height: 10px;"></span>
$$\displaystyle z \equiv \frac{\hat{f}(x) - \E \hat{f}(x)}{\sqrt{\mathbb{V} (\hat{f}(x))}}$$
$$= \frac{\frac{1}{n} \sum_{i=1}^{n}(z_i - \E z_i)}{\sqrt{\frac{1}{n} \mathbb{V}(z_1)}} $$
$$=\sum_{i=1}^{n} L_{n,i}, \text{ where } L_{n,i} \equiv \frac{1}{n} \frac{z_i - \E z_i}{\frac{1}{n} V(z_1)} $$



### Lyapunov CLT 

{: .box-success}
Let $$\{ X_{n,i}\}$$ be a sequence of independent (not necessarily identically distributed) RVs, with $$\E X_{n,i} = \mu_{n,i}$$ and $$\mathbb{V} (X_{n,i}) = \sigma^2_n < \infty$$. Denote $$ \displaystyle L_{n,i} \equiv \frac{X_{n,i} - \mu_{n,i}}{\sigma_n}$$. \\
If for some $$\delta>0$$ the condition $$\lim_{n \to \infty} \sum_{i=1}^{n} \E \abs{L_{n,i}}^{2+\delta}=0$$ is satisfied, then $$ \displaystyle \sum_{i=1}^{n} L _{n,i} \overset{d}{\to} \mathcal{N}(0,1)$$
