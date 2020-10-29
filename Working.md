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
grid-column: col-start 1 / span 7;
grid-row: 1/2;
}
.item33 {
grid-column: col-start 8 / span 5;
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


<div class="wrapper">
  <div class="item11">
    <b>"The Kernel Trick for Nonlinear Factor Modeling"</b> <i>(JMP, submitted)</i>
  </div>
  <div class="item2">
    <p class="dropcap">F</p>actor modeling is a powerful statistical technique that permits to capture the common dynamics in a large panel of data with a few latent variables, or factors, thus alleviating the curse of dimensionality. Despite its popularity and widespread use for various applications ranging from genomics to finance, this methodology has predominantly remained linear. This study estimates factors nonlinearly through the kernel method, which allows flexible nonlinearities while still avoiding the curse of dimensionality. We focus on factor-augmented forecasting of a single time series in a high-dimensional setting, known as diffusion index forecasting in macroeconomics literature. Our main contribution is twofold. First, we show that the proposed estimator is consistent and it nests linear PCA estimator as well as some nonlinear estimators introduced in the literature as specific examples. Second, our empirical application to a classical macroeconomic dataset demonstrates that this approach can offer substantial advantages over mainstream methods. <br> <br>

    <b>Keywords:</b> 
    <span class="badge">Forecasting</span>
    <span class="badge">Latent factor model</span>
    <span class="badge">Nonlinear time series</span>
    <span class="badge">kernel PCA</span>
    <span class="badge">Neural networks</span>
    <span class="badge">Econometric models </span>
    
  </div>
  <div class="item33">
    <center> 
      <a href="/pdfs/JMP.pdf" type="button" class="btn btn-new btn-sm" data-toggle="tooltip" data-placement="bottom" ><i class="fas fa-file-pdf fa-lg"></i><b> Paper</b></a> 
      <a href="/pdfs/JMP_pres.pdf" type="button" class="btn btn-new btn-sm" data-toggle="tooltip" data-placement="bottom" title="Short version"><i class="fas fa-file-pdf fa-lg"></i><b> Slides</b></a>
      <a href="https://drive.google.com/file/d/1XT9-gnNdrSYtsPTP0-95ULq01hxgO0h5/view?usp=sharing" type="button" class="btn btn-new btn-sm" data-toggle="tooltip" data-placement="bottom" title="Video"><i class="fas fa-video fa-lg"></i></a>
    </center>
  </div>
</div>

<span style="display:block; height: 0px;"></span>


<div class="wrapper">
  <div class="item1">
    <b>"Nonlinear Shrinkage Covariance Matrix Estimation"</b>
  </div>
  <div class="item2">

 <p class="dropcap">C</p>ovariance matrix estimates are required in a wide range of applied problems in multivariate data analysis, including portfolio and risk management in finance, factor models and testing in economics, and graphical models and classification in machine learning.
In modern applications, where often the model dimensionality is comparable or even larger than the sample size, 
the classical sample covariance estimator lacks desirable properties, such as consistency, and suffers from eigenvalue spreading.
In recent years, improved estimators have been proposed based on the idea of regularization.  Specifically, such estimators, known as rotation-equivariant estimators, shrink the sample eigenvalues, while keeping the eigenvectors of the sample covariance estimator. In high dimensions, however, the sample eigenvectors will generally be strongly inconsistent, rendering eigenvalue shrinkage estimators suboptimal. 
We consider an estimator that goes beyond mere eigenvalue shrinkage and employs recent advancements in random matrix theory to account for eigenvector inconsistency in a large-dimensional setting. 
We provide the theoretical guarantees and an empirical evaluation demonstrating the superior performance of the proposed estimator. <br> <br>

    <b>Keywords:</b> 
    <span class="badge">Shrinkage estimator</span>
    <span class="badge">Rotation equivariance</span>
    <span class="badge">Random matrix theory</span>
    <span class="badge">Large-dimensional asymptotics</span>
    <span class="badge">Bias correction</span>
    <span class="badge">Principal components</span>
  </div>
  <div class="item3">
    <center> 
      <a href="/pdfs/Cov_abstract.pdf" type="button" class="btn btn-new btn-sm" data-toggle="tooltip" data-placement="bottom" title="Paper available soon!"><i class="fas fa-file-pdf fa-lg"></i><b> Abstract</b></a>  
    </center>
  </div>
</div>

<span style="display:block; height: 0px;"></span>

<div class="wrapper">
  <div class="item11">
    <b>"Fast and Efficient Data Science Techniques for COVID-19 Group Testing"</b> (with E. Seregina)
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
      <a href="/pdfs/Covid_paper.pdf" type="button" class="btn btn-new btn-sm" data-toggle="tooltip" data-placement="bottom" ><i class="fas fa-file-pdf fa-lg"></i><b> Paper</b></a> 
      <a href="/pdfs/Covid_pres.pdf" type="button" class="btn btn-new btn-sm" data-toggle="tooltip" data-placement="bottom" ><i class="fas fa-file-pdf fa-lg"></i><b> Slides</b></a>
      <a href="/pdfs/Covid_poster.pdf" type="button" class="btn btn-new btn-sm" data-toggle="tooltip" data-placement="bottom" ><i class="fas fa-file-pdf fa-lg"></i><b> Poster</b></a>
    </center>
  </div>
</div>
