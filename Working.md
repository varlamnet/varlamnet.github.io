---
layout: page
title: Research
subtitle:
---

<style>
.wrapper {
max-width: 1024px;
margin: 0 auto;
}
.wrapper > * {
background-color: var(--headbackcol);
border-radius: 5px;
padding: 12px;
}
.wrapper {
display: grid;
grid-template-columns: repeat(12, [col-start] 1fr);
grid-gap: 5px;
}
.item1 {
grid-column: col-start 1 / span 9;
grid-row: 1/2;
}
.item2 {
grid-column: col-start 1 / span 12 ;
grid-row: 2 / 7;
padding: 15px;
}
.item3 {
grid-column: col-start 10 / span 3;
grid-row: 1/2;
  display: flex;
  justify-content: center;
  align-items: center;
} 
.item11 {
grid-column: col-start 1 / span 9;
grid-row: 1/2;
}
.item33 {
grid-column: col-start 10 / span 3;
grid-row: 1/2;
  display: flex;
  justify-content: center;
  align-items: center;
} 
.badge{
  padding: 3px 8px 5px;
  font-size: inherit;
  font-weight: inherit;
  background-color:var(--backcol);
  /* color: var(--backcol); */
  border-radius: 9px;
}
.badge:hover, .badge:focus{
    background-color:var(--footertextcol);
    color: var(--backcol);
  /* color:var(--headbackcol); */
  /* background-color:var(--footertextcol); */
  cursor: pointer;
}
</style>

<span style="display:block; height: 0px;"></span>

<div class="wrapper">
  <div class="item1">
    <b>"Doubly Sparse Estimator for High-Dimensional Covariance Matrices"</b> with Seregina, <i>Econometrics & Statistics</i>, 2024
  </div>
  <div class="item2">

 <p class="dropcap">T</p>he classical sample covariance estimator lacks desirable properties such as consistency and suffers from eigenvalue spreading in high-dimensional settings. Improved estimators have been proposed that shrink sample eigenvalues but retain the eigenvectors of the sample covariance estimator. In high dimensions, however, sample eigenvectors are generally strongly inconsistent, rendering eigenvalue shrinkage estimators suboptimal. A Doubly Sparse Covariance Estimator (DSCE) is developed that goes beyond mere eigenvalue shrinkage: a covariance matrix is decomposed into a signal part, where sparse eigenvectors are estimated via truncation, and an idiosyncratic part, estimated via thresholding. It is shown that accurate estimation is possible if the leading eigenvectors are sufficiently sparse affecting proportionately less than $\sqrt{p}$ of the variables. DSCE fills the gap for empirical applications that fall in-between fully sparse settings and conditionally sparse settings: DSCE takes advantage of conditional sparsity implied by factor models while allowing only a subset of variables to load on factors, which relaxes pervasiveness assumption of traditional factor models. An empirical application to the constituents of the S&P 1500 illustrates that DSCE-based portfolios outperform competing methods in terms of Sharpe ratio, maximum drawdown, and cumulative return for monthly and daily data..<br> <br>

    <b>Keywords:</b>
    <span class="badge"> Sparse recovery </span>
    <span class="badge"> Rotation equivariance </span>
    <span class="badge"> Random matrix theory </span>
    <span class="badge"> Large-dimensional asymptotics </span>
    <span class="badge"> Principal components </span>

  </div>
  <div class="item3">
    <center> 
      <a href="https://www.sciencedirect.com/science/article/abs/pii/S2452306224000364" type="button" class="btn btn-new btn-sm" data-toggle="tooltip" data-placement="bottom"><i class="fas fa-file-pdf fa-lg"></i><b> Paper</b></a>  
    </center>
  </div>
</div>

<span style="display:block; height: 0px;"></span>

<div class="wrapper">
  <div class="item11">
    <b>"The Kernel Trick for Nonlinear Factor Modeling"</b> <br> <i>International Journal of Forecasting</i>, 2022
  </div>
  <div class="item2">
    <p class="dropcap">F</p>actor modeling is a powerful statistical technique that permits to capture the common dynamics in a large panel of data with a few latent variables, or factors, thus alleviating the curse of dimensionality. Despite its popularity and widespread use for various applications ranging from genomics to finance, this methodology has predominantly remained linear. This study estimates factors nonlinearly through the kernel method, which allows flexible nonlinearities while still avoiding the curse of dimensionality. We focus on factor-augmented forecasting of a single time series in a high-dimensional setting, known as diffusion index forecasting in macroeconomics literature. Our main contribution is twofold. First, we show that the proposed estimator is consistent and it nests linear PCA estimator as well as some nonlinear estimators introduced in the literature as specific examples. Second, our empirical application to a classical macroeconomic dataset demonstrates that this approach can offer substantial advantages over mainstream methods. <br> <br>

    <b>Keywords:</b>
    <span class="badge">Forecasting</span>
    <span class="badge">Latent factor model</span>
    <span class="badge">Nonlinear time series</span>
    <span class="badge">Kernel PCA</span>
    <span class="badge">Neural networks</span>
    <span class="badge">Econometric models </span>

  </div>
  <div class="item33">
    <center> 
      <a href="https://www.sciencedirect.com/science/article/abs/pii/S0169207021000741" type="button" class="btn btn-new btn-sm" data-toggle="tooltip" data-placement="bottom" ><i class="fas fa-file-pdf fa-lg"></i><b> Paper</b></a> 
    </center>
  </div>
</div>

<span style="display:block; height: 0px;"></span>

<div class="wrapper">
  <div class="item11">
    <b>"Fast and Efficient Data Science Techniques for COVID-19 Group Testing"</b> with Seregina, <i>Journal of Data Science</i>, 2021
  </div>
  <div class="item2">
    <p class="dropcap">R</p>esearchers and public officials tend to agree that until a vaccine is developed, stopping SARS-CoV-2 transmission is the name of the game. Testing is the key to preventing the spread, especially by asymptomatic individuals. With testing capacity restricted, group testing is an appealing alternative for comprehensive screening and has recently received FDA emergency authorization. This technique tests pools of individual samples, thereby often requiring fewer testing resources while potentially providing multiple folds of speedup. We approach group testing from a data science perspective and offer two contributions. First, we provide an extensive empirical comparison of modern group testing techniques based on simulated data. Second, we propose a simple one-round method based on $\ell_1$-norm sparse recovery, which outperforms current state-of-the-art approaches at certain disease prevalence rates. <br> <br>

    <b>Keywords:</b>
    <span class="badge">Pooled Testing</span>
    <span class="badge">Compressed Sensing</span>
    <span class="badge">Sparse Recovery</span>
    <span class="badge">Lasso</span>
    <span class="badge">Sensing Matrix</span>
    <span class="badge">SARS-CoV-2</span>

  </div>
  <div class="item33">
    <center> 
      <a href="https://jds-online.org/journal/JDS/article/561/info" type="button" class="btn btn-new btn-sm" data-toggle="tooltip" data-placement="bottom" ><i class="fas fa-file-pdf fa-lg"></i><b> Paper</b></a> 
    </center>
  </div>
</div>
