---
title: 'Ensembles'
hide:
  - toc
---

<div class="listing-header" markdown>

# Ensembles

<p class="page-meta" markdown="span">79 papers · page 1 of 3 · <a href="../../bib/ensembles.bib" download>:material-download: BibTeX for this topic</a></p>

</div>

<div class="grid cards" markdown>

-   #### A dataset of one-dimensional idealized probabilistic fields { #2609.25720 }

    *Gregor Skok, Romain Pic* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.25720">Verification of probabilistic weather forecasts remains a crucial aspect of numerical weather prediction, as new AI-based models become more widely used alongside the more traditional physics-based...</span><span class="abstract-full" id="full-2609.25720" hidden>Verification of probabilistic weather forecasts remains a crucial aspect of numerical weather prediction, as new AI-based models become more widely used alongside the more traditional physics-based ensemble forecasting systems that continue to be developed and improved. We present a first-of-its-kind idealized probabilistic dataset composed of one-dimensional cases aimed at analyzing the behavior and properties of verification methods for probabilistic forecasts and comparing their behavior. It covers a wide range of probabilistic cases, such as constant, localized events, gradients, fronts, noisy, bimodal, and limiting cases. Moreover, the code associated with the dataset provides great flexibility for customizing the experiments it covers. The dataset represents the first building block of the more extensive comparison dataset of the Bridging The Gap project, which aims to facilitate the development and comparison of spatial verification methods for probabilistic forecasts.</span> <span class="abstract-toggle" data-id="2609.25720">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.25720v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.25720v1) · [:material-content-copy: BibTeX](../../bibtex/2609.25720.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=benchmarks-datasets" data-tag="benchmarks-datasets">Benchmarks & datasets</a>
    { .paper-tags }

-   #### Predictability-Guided Multiscale Probabilistic Forecasting of Wind Direction under Extreme Shear { #2609.16707 }

    *Hailong Shu* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.16707">Accurate multi-horizon wind direction forecasting is critical for turbine yaw control and grid security. Rapid directional shear (turning $\ge 90^\circ$) challenges models via non-Euclidean geometry...</span><span class="abstract-full" id="full-2609.16707" hidden>Accurate multi-horizon wind direction forecasting is critical for turbine yaw control and grid security. Rapid directional shear (turning $\ge 90^\circ$) challenges models via non-Euclidean geometry on $S^1$, multiscale dynamics, and regime-dependent uncertainty. Conventional discrete models and foundation models suffer from mid-frequency phase lag and turning misalignments. We show that directional predictability decays at disparate rates across frequency subbands, rendering monolithic mechanisms suboptimal. We propose a predictability-guided paradigm: slow synoptic drift $\to$ deterministic regression; intermediate turning $\to$ continuous latent differential flows; unresolved turbulence $\to$ conditional residual diffusion; followed by causal recalibration. On a 10,000-sequence multi-year benchmark, our framework maintains calm-weather accuracy (Test MCE $38.48^\circ$) while reducing extreme-turning error (Case 1 MCE $60.69^\circ$ vs $70.42^\circ$ for zero-shot foundation models). The circular CRPS reaches $22.36^\circ$, with 93.88% coverage at nominal 95% (91.01% out-of-distribution). Density estimation further reveals near-antipodal bimodal structure under severe shear (13.39%--15.43% tail mass $\ge 135^\circ$), exposing a geometric bound where single-center calibration under-covers (81.56%), motivating multimodal circular manifold learning.</span> <span class="abstract-toggle" data-id="2609.16707">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.16707v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.16707v1) · [:material-content-copy: BibTeX](../../bibtex/2609.16707.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=foundation-models" data-tag="foundation-models">Foundation models</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### A Self-Diagnosing Structural Error-Aware Parameter Estimation Method for Earth System Models { #2609.16210 }

    *Qingyuan Yang, Addisu G Semie, Brian Medeiros, Gregory S Elsaesser, Da Fan, Wayne Chuang* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.16210">We propose a fully automated, structural error-aware, interpretable climate model parameter estimation method that leverages Perturbed Parameter Ensembles (PPEs). It is based on history matching and...</span><span class="abstract-full" id="full-2609.16210" hidden>We propose a fully automated, structural error-aware, interpretable climate model parameter estimation method that leverages Perturbed Parameter Ensembles (PPEs). It is based on history matching and aligns with an increasingly-used iterative simulation-emulation-calibration methodology. The method is motivated by the negative impacts of structural error and emulator and observational uncertainties on climate model parameter estimation efforts, as well as the problems associated with sparsely-sampled PPEs. To address these challenges, the method explicitly builds simpler emulators that avoid overfitting, detect structural error, avoids compensating for structural error through inflated mismatch tolerances, and sequentially excludes structurally inconsistent variables for parameter estimation. The method decomposes the high-dimensional calibration problem into linked low-dimensional subproblems, and integrates their constraints to reconstruct the jointly plausible region of the full parameter space. The method is applied to a 100-member PPE with 34 perturbed parameters generated by a version of CAM6 with machine learning-based warm rain microphysics parameterization. Through iterative application, the method greatly reduces the ensemble spread and improves the matching between simulated and observed zonal climatologies. The method also finds ensemble members that outperform the default CAM6 configuration in root mean square error across multiple diagnostics. Controlled experiments demonstrate that overly-conservative emulator uncertainty could lead to neglect of informative observations, and tolerance of the structural error, in the context of this method, biases the estimated parameters toward compensating for structural error. Our work also emphasizes the value of interpretability for diagnosing structural error and informing parameter estimation in PPE-based calibration.</span> <span class="abstract-toggle" data-id="2609.16210">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.16210v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.16210v1) · [:material-content-copy: BibTeX](../../bibtex/2609.16210.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=interpretability" data-tag="interpretability">Interpretability</a>
    { .paper-tags }

-   #### Statistical versus machine learning-based spatial interpolation of post-processed ensemble weather forecasts { #2609.07512 }

    *Mária Lakatos* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.07512">Statistical post-processing improves ensemble weather forecasts, but generating calibrated predictions at locations without observations remains challenging. This study compares statistical and...</span><span class="abstract-full" id="full-2609.07512" hidden>Statistical post-processing improves ensemble weather forecasts, but generating calibrated predictions at locations without observations remains challenging. This study compares statistical and machine-learning-based methods for post-processing ECMWF 2-m temperature and 10-m wind speed forecasts at observed and unobserved stations in Germany. We consider EMOS-based approaches, distributional regression networks, Transformers, and graph neural networks under both limited and extended predictor settings. For temperature, we also investigate linear forecast combinations and propose an altitude-aware linear pool (ALP). The results show that post-processing improves upon the raw ensemble in most settings, but no single method performs best across all variables, station groups, and evaluation metrics. The proposed ALP provides a small but significant improvement over the standard linear pool at unobserved locations.</span> <span class="abstract-toggle" data-id="2609.07512">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.07512v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.07512v1) · [:material-content-copy: BibTeX](../../bibtex/2609.07512.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=transformers" data-tag="transformers">Transformers</a> <a class="md-tag" href="/explore/?t=graph-neural-networks" data-tag="graph-neural-networks">Graph neural networks</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### FarSky: Task-Aware Latent-Space Coupling for Generative Intra-Hour Solar Forecasting { #2608.11254 }

    *Yann Fabel, Bijan Nouri, Milon Miah, Niklas Blum, Luis F. Zarzalejo, Julia Kowalski et al.* · Aug 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2608.11254">Accurate solar irradiance forecasting is essential for the reliable integration of photovoltaic power into modern electricity grids. All-sky imagers (ASI) provide high-resolution observations of...</span><span class="abstract-full" id="full-2608.11254" hidden>Accurate solar irradiance forecasting is essential for the reliable integration of photovoltaic power into modern electricity grids. All-sky imagers (ASI) provide high-resolution observations of clouds, making them well suited for intra-hour forecasting. Recent deep learning approaches have substantially improved forecast accuracy but are often limited by deterministic predictions and a reduced capability to anticipate ramp events. This work proposes FarSky, a generative forecasting framework that leverages latent-space coupling to learn task-aware representations of sky images. A multi-task autoencoder first learns a shared latent representation for image reconstruction and irradiance estimation. A latent diffusion model then generates future latent states conditioned on recent observations, from which irradiance forecasts are directly decoded. Probabilistic forecasts are inherently obtained through stochastic sampling. The framework is developed using a multi-year ASI dataset acquired at the Plataforma Solar de Almería, Spain, and evaluated on two independent test datasets against persistence, state-of-the-art end-to-end, and generative forecasting approaches. FarSky achieves the best overall deterministic and probabilistic forecasting performance, improving forecast skill by up to 11 percentage points. Furthermore, it substantially improves ramp event detection over existing methods, achieving F1-scores above 60%. These results demonstrate the potential of combining generative models with task-aware latent-space coupling for solar forecasting.</span> <span class="abstract-toggle" data-id="2608.11254">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2608.11254v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2608.11254v1) · [:material-content-copy: BibTeX](../../bibtex/2608.11254.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=energy" data-tag="energy">Energy</a>
    { .paper-tags }

-   #### Do AI Forecast Ensembles Sample the Correct Conditional Distribution? { #2608.08954 }

    *Lucas J. Howard, Elizabeth A. Barnes* · Aug 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2608.08954">Ensemble forecasting aims to sample the conditional distribution of outcomes; whether AI forecast ensembles do this correctly in a joint sense remains largely untested. We train a diffusion model for...</span><span class="abstract-full" id="full-2608.08954" hidden>Ensemble forecasting aims to sample the conditional distribution of outcomes; whether AI forecast ensembles do this correctly in a joint sense remains largely untested. We train a diffusion model for probabilistic subseasonal coastal sea level forecasts at eight US East Coast tide gauge stations, with sea level derived from reanalysis, and find that marginal and joint forecast quality decouple: positive skill at every station and lead time marginally, while joint spatial structure is worse than climatological draws. A shuffle-based permutation decomposition reveals this failure is invisible to the energy score but detected by the variogram score. Lorenz-96 experiments across 0.7-170 equivalent years show the gap persists regardless of training volume and is reproduced by a linear baseline, indicating structural inadequacy of the learned distribution. A dynamical ensemble does not replicate the failure while a deterministic emulator does, suggesting it is specific to learned emulators rather than ensemble forecasting generally.</span> <span class="abstract-toggle" data-id="2608.08954">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2608.08954v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2608.08954v1) · [:material-content-copy: BibTeX](../../bibtex/2608.08954.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### Uncertainty quantification via conformal prediction in data assimilation { #2606.27001 }

    *Catherine George, Alireza Javanmardi, Tijana Janjić, Eyke Hüllermeier* · Jun 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2606.27001">Quantifying the evolution of uncertainty is critical to both probabilistic forecasting and data assimilation in numerical weather prediction. In this study, we investigate the applicability of...</span><span class="abstract-full" id="full-2606.27001" hidden>Quantifying the evolution of uncertainty is critical to both probabilistic forecasting and data assimilation in numerical weather prediction. In this study, we investigate the applicability of conformal prediction (CP), a recent machine learning (ML) method, to quantify uncertainty in a controlled, idealized setting. We use the one dimensional modified shallow water model, designed to mimic the convective process. CP provides a set of possible outcomes with a chosen confidence level. Here, we compare and evaluate the average empirical coverage, the average interval length, miss low, miss high and average interval score loss (AISL) for three variants of CP, namely a) Standard CP, b) Normalized CP and c) Conformalized Quantile Regression. We further compare these CP-based uncertainty estimates with traditional ensemble-based measures such as standard deviation intervals and ensemble spread. In addition, we investigate the integration of CP-derived uncertainty within the data assimilation cycle through CP perturbations. Our results highlight the strengths and limitations of each approach, providing insight into the effectiveness of CP to complement common ensemble-based uncertainty quantification in simplified atmospheric models.</span> <span class="abstract-toggle" data-id="2606.27001">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2606.27001v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2606.27001v1) · [:material-content-copy: BibTeX](../../bibtex/2606.27001.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### Exascale Hybrid Numerical-AI Ensembles for Operational Flood-Season Forecasting in East Asia: 15-km Decadal Hindcasts and 1-km High-Resolution Capability { #2605.24896 }

    *Mengxuan Chen, Yunpu Xu, Qiuyan Sun, Han Zhang, Jiayi Lai, Zheng Zhou, Juepeng Zheng, Hongsong Meng et al.* · May 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2605.24896">Seasonal forecasting of summer rainfall in East Asia remains a grand challenge, as predictability at 3 to 6 month lead times is constrained by the spring predictability barrier, weak large-scale...</span><span class="abstract-full" id="full-2605.24896" hidden>Seasonal forecasting of summer rainfall in East Asia remains a grand challenge, as predictability at 3 to 6 month lead times is constrained by the spring predictability barrier, weak large-scale signals, and localized nonlinear convective extremes. We address this challenge with CAPES, which integrates a kilometer-resolution coupled regional model with atmosphere, land, and ocean components and a data-driven AI seasonal forecasting system. At 15 km resolution, the fused workflow combines 174 numerical members from varying start times, physics schemes, and parameter perturbations with 1,600 AI members generated from initial and physical perturbations. Using the full LineShine system, CAPES completes ten annual 1,774-member hindcasts for 2016 to 2025 within 14.6 hours, improving the mean prediction score from ECMWF's 71.8 to 75.9 and delivering a major gain in operational forecasting capability. The 1-km configuration further enables fine-scale typhoon simulation and establishes the feasibility of kilometer-scale fused ensemble forecasting on a one-week timescale.</span> <span class="abstract-toggle" data-id="2605.24896">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2605.24896v3) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2605.24896v3) · [:material-content-copy: BibTeX](../../bibtex/2605.24896.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a> <a class="md-tag" href="/explore/?t=extremes" data-tag="extremes">Extremes</a> <a class="md-tag" href="/explore/?t=subseasonal-to-seasonal" data-tag="subseasonal-to-seasonal">Subseasonal to seasonal</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a> <a class="md-tag" href="/explore/?t=km-scale" data-tag="km-scale">Km-scale</a>
    { .paper-tags }

-   #### Deep Learning Surrogates for Emulating Stochastic Climate Tipping Dynamics { #2605.20580 }

    *Adeline Hillier, Jennifer Sleeman, Jay Brett, Caroline Tang, Jenelle Millison, Anand Gnanadesikan* · May 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2605.20580">This work explores a dynamics-informed Temporal Fusion Transformer (TFT) as a data-driven surrogate for computationally intensive Earth system simulations. Focusing on multivariate time series...</span><span class="abstract-full" id="full-2605.20580" hidden>This work explores a dynamics-informed Temporal Fusion Transformer (TFT) as a data-driven surrogate for computationally intensive Earth system simulations. Focusing on multivariate time series describing global ocean transport, we demonstrate the surrogate's ability to forecast tip events across thousands of time steps. The data involve up to 21 non-stationary time series in addition to static covariates describing free parameters and initial conditions. Modifications to the architecture and objective function yield a surrogate that anticipates the timing of Atlantic and Pacific collapses to high fidelity and captures the stochastic uncertainty in transition timing across ensemble predictions. The learned surrogate achieves a 465x computational speedup over the numerical simulator while maintaining differentiability with respect to parameters and initial conditions.</span> <span class="abstract-toggle" data-id="2605.20580">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2605.20580v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2605.20580v1) · [:material-content-copy: BibTeX](../../bibtex/2605.20580.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=transformers" data-tag="transformers">Transformers</a> <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a>
    { .paper-tags }

-   #### Njord: A Probabilistic Graph Neural Network for Ensemble Ocean Forecasting { #2605.15470 }

    *Daniel Holmberg, Joel Oskarsson, Erik Wikingsson, Fredrik Lindsten, Teemu Roos* · May 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2605.15470">Ocean dynamics are inherently chaotic, yet existing machine learning ocean models produce only deterministic forecasts. We introduce Njord, a probabilistic data-driven model for ocean forecasting,...</span><span class="abstract-full" id="full-2605.15470" hidden>Ocean dynamics are inherently chaotic, yet existing machine learning ocean models produce only deterministic forecasts. We introduce Njord, a probabilistic data-driven model for ocean forecasting, applicable to both global and regional domains. Njord combines a deep latent variable framework with a graph neural network architecture, enabling sampling each forecast step in a single forward pass. We apply Njord globally at 0.25° resolution and regionally to the Baltic Sea at 2 km resolution. To scale to these large ocean grids we introduce K-means cluster meshes that adapt to irregular sea surface geometry. Experiments demonstrate strong performance on both domains compared to deterministic machine learning baselines, while also providing uncertainty estimates from the sampled ensemble forecasts. On the global OceanBench benchmark, Njord achieves the lowest errors on average across upper-ocean variables when evaluated against real-world observations, with the largest improvements in surface temperature prediction.</span> <span class="abstract-toggle" data-id="2605.15470">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2605.15470v2) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2605.15470v2) · [:material-content-copy: BibTeX](../../bibtex/2605.15470.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=graph-neural-networks" data-tag="graph-neural-networks">Graph neural networks</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a> <a class="md-tag" href="/explore/?t=km-scale" data-tag="km-scale">Km-scale</a> <a class="md-tag" href="/explore/?t=quarter-degree" data-tag="quarter-degree">0.25°</a>
    { .paper-tags }

-   #### Tyche: One Step Flow for Efficient Probabilistic Weather Forecasting { #2605.06916 }

    *Fan Xu, Yuan Gao, Kun Wang, Rui Su, Fenghua Ling, Hao Wu, Wanli Ouyang* · May 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2605.06916">Probabilistic weather forecasting requires not only accurate trajectories, but calibrated distributions over plausible atmospheric futures. Recent data-driven systems have achieved remarkable...</span><span class="abstract-full" id="full-2605.06916" hidden>Probabilistic weather forecasting requires not only accurate trajectories, but calibrated distributions over plausible atmospheric futures. Recent data-driven systems have achieved remarkable deterministic skill, and diffusion-based ensemble forecasters have substantially improved sample realism and uncertainty quantification. However, their inference cost scales with forecast horizon, ensemble size, and the number of denoising steps required for each transition, making large operational ensembles expensive. To address this, we present Tyche, a one-step conditional flow model for efficient probabilistic weather forecasting. Tyche models the conditional forecast distribution with a destination-aware average-velocity flow that maps Gaussian noise directly to future weather states in a single function evaluation (1-NFE). To make this one-step transport learnable in high-dimensional geophysical fields, we derive a JVP-regularized rectification objective that enforces temporal self-consistency across source and destination flow timesteps without explicitly forming Jacobians. The transport field is parameterized by an isotropic Swin-style transformer that preserves fine-scale spatial structure while remaining scalable on global grids. To improve ensemble reliability under autoregressive forecasting, we further introduce a rollout-based finetuning stage with curriculum CRPS calibration supervision. Experiments on ERA5 at 1.5$^\circ$ and 6-hour resolution show that our Tyche, using merely a single NFE, matches or exceeds the forecast skill and calibration of state-of-the-art multi-step generative baselines and the operational ECMWF IFS ensemble.</span> <span class="abstract-toggle" data-id="2605.06916">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2605.06916v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2605.06916v1) · [:material-content-copy: BibTeX](../../bibtex/2605.06916.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=transformers" data-tag="transformers">Transformers</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a> <a class="md-tag" href="/explore/?t=coarse" data-tag="coarse">Coarse (≥1°)</a> <a class="md-tag" href="/explore/?t=6-hourly" data-tag="6-hourly">6-hourly</a>
    { .paper-tags }

-   #### Cast3: Translating numerical weather prediction principles into data-driven forecasting { #2605.01599 }

    *Congyi Nai, Baoxiang Pan, Yuan Liang, Xi Chen* · May 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2605.01599">Data-driven weather models have made rapid advances in recent years, reaching and in some metrics surpassing the large-scale forecast skill of operational numerical weather prediction. This progress,...</span><span class="abstract-full" id="full-2605.01599" hidden>Data-driven weather models have made rapid advances in recent years, reaching and in some metrics surpassing the large-scale forecast skill of operational numerical weather prediction. This progress, however, has been built almost entirely on the reanalysis data that NWP produced, while the methodological knowledge that the NWP community distilled over decades of multi-scale atmospheric modelling remains largely unused. Here we present Cast3, a generative forecasting framework that systematically absorbs NWP meta-knowledge to close this gap. Cast3 operates on variable-resolution cubed-sphere grids for scale-aware representation and constructs structurally diverse super-ensembles that sample the complementary biases of different grid discretizations, delivering state-of-the-art ensemble prediction. It further introduces generative nudging, a posterior-sampling strategy that distils the collective information of the full ensemble into a single forecast possessing both the large-scale accuracy of the ensemble mean and the mesoscale realism of a high-resolution member. Evaluated across synoptic-scale skill, spectral fidelity, station-level surface verification, and tropical cyclone prediction, Cast3 outperforms established deterministic and generative baselines across various dimensions. More broadly, these results demonstrate that the design principles embedded in computational atmospheric science offer a rich and largely untapped foundation for the next generation of data-driven Earth system modelling.</span> <span class="abstract-toggle" data-id="2605.01599">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2605.01599v2) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2605.01599v2) · [:material-content-copy: BibTeX](../../bibtex/2605.01599.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### CycloneMAE: A Scalable Multi-Task Learning Model for Global Tropical Cyclone Probabilistic Forecasting { #2604.12180 }

    *Renlong Hang, Zihao Xu, Jiuwei Zhao, Runling Yu, Leye Cheng, Qingshan Liu* · Apr 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2604.12180">Tropical cyclones (TCs) rank among the most destructive natural hazards, yet their forecasting faces fundamental trade-offs: numerical weather prediction (NWP) models are computationally prohibitive...</span><span class="abstract-full" id="full-2604.12180" hidden>Tropical cyclones (TCs) rank among the most destructive natural hazards, yet their forecasting faces fundamental trade-offs: numerical weather prediction (NWP) models are computationally prohibitive and struggle to leverage historical data, while existing deep learning (DL)-based intelligent models are variable-specific and deterministic, which fail to generalize across different forecasting variables. Here we present CycloneMAE, a scalable multi-task forecasting model that learns transferable TC representations from multi-modal data using a TC structure-aware masked autoencoder. By coupling a discrete probabilistic gridding mechanism with a pre-train/fine-tune paradigm, CycloneMAE simultaneously delivers deterministic forecasts and probability distributions. Evaluated across five global ocean basins, CycloneMAE outperforms leading NWP systems in pressure and wind forecasting up to 120 hours and in track forecasting up to 24 hours. Attribution analysis via integrated gradients reveals physically interpretable learning dynamics: short-term forecasts rely predominantly on the internal core convective structure from satellite imagery, whereas longer-term forecasts progressively shift attention to external environmental factors. Our framework establishes a scalable, probabilistic, and interpretable pathway for operational TC forecasting.</span> <span class="abstract-toggle" data-id="2604.12180">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2604.12180v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2604.12180v1) · [:material-content-copy: BibTeX](../../bibtex/2604.12180.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=interpretability" data-tag="interpretability">Interpretability</a>
    { .paper-tags }

-   #### Diffusion-based Probabilistic Air Quality Forecasting with Mechanistic Insight { #2603.21131 }

    *Ao Ding, Aoxing Zhang, Tzung-May Fu, Yuanlong Huang, Qianjie Chen, Yuyang Chen, Jiajia Mo, Wei Tao et al.* · Mar 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2603.21131">Current operational air quality forecasts are computationally expensive, sensitive to errors in physics and emissions, and often neglect weather-related uncertainty. To address these limitations, we...</span><span class="abstract-full" id="full-2603.21131" hidden>Current operational air quality forecasts are computationally expensive, sensitive to errors in physics and emissions, and often neglect weather-related uncertainty. To address these limitations, we present AirFusion, a hybrid, diffusion-based framework that synergistically integrates knowledge from chemical transport models with real-world observational constraints to enable accurate and efficient probabilistic regional air quality prediction. We apply AirFusion to generate operational 6-day, 30-member ensemble forecasts of surface ozone across China, initialized with observations and driven by ensemble weather forecasts. AirFusion outperforms existing operational benchmarks, achieving substantially lower forecast errors against surface measurements, while also providing ensemble-based diagnostics that explicitly quantify the impacts of weather uncertainty on air quality predictability. Moreover, AirFusion can rapidly adapt to evolving emissions through fine-tuning with only one month of recent observations. These attributes establish AirFusion as a powerful and extensible framework for next-generation probabilistic air quality forecasting, with clear potential for application to other pollutants and regions.</span> <span class="abstract-toggle" data-id="2603.21131">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2603.21131v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2603.21131v1) · [:material-content-copy: BibTeX](../../bibtex/2603.21131.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### Developing Machine Learning-Based Watch-to-Warning Severe Weather Guidance from the Warn-on-Forecast System { #2603.20250 }

    *Montgomery Flora, Samuel Varga, Corey Potvin, Noah Lang* · Mar 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2603.20250">While machine learning (ML) post-processing of convection-allowing model (CAM) output for severe weather hazards (large hail, damaging winds, and/or tornadoes) has shown promise for very short lead...</span><span class="abstract-full" id="full-2603.20250" hidden>While machine learning (ML) post-processing of convection-allowing model (CAM) output for severe weather hazards (large hail, damaging winds, and/or tornadoes) has shown promise for very short lead times (0-3 hours), its application to slightly longer forecast windows remains relatively underexplored. In this study, we develop and evaluate a grid-based ML framework to predict the probability of severe weather hazards over the next 2-6 hours using forecast output from the Warn-on-Forecast System (WoFS). Our dataset includes WoFS ensemble forecasts valid every 5 minutes out to 6 hours from 108 days during the 2019--2023 NOAA Hazardous Weather Testbed Spring Forecasting Experiments. We train ML models to generate probabilistic forecasts of severe weather akin to Storm Prediction Center outlooks (i.e., likelihood of a tornado, severe wind, or severe hail event within 36 km of each point). We compare a histogram gradient-boosted tree (HGBT) model and a deep learning U-Net approach against a carefully calibrated baseline generated from 2-5 km updraft helicity. Results indicate that the HGBT and U-Net outperform the baseline, particularly at higher probability thresholds. The HGBT achieves the best performance metrics, but predicted probabilities cap at 60% while the U-net forecasts extend to 100%. Similar to previous studies, the U-Net produces spatially smoother guidance than the tree-based method. These findings add to the growing evidence of the effectiveness of ML-based CAM post-processing for providing short-term severe weather guidance.</span> <span class="abstract-toggle" data-id="2603.20250">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2603.20250v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2603.20250v1) · [:material-content-copy: BibTeX](../../bibtex/2603.20250.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a> <a class="md-tag" href="/explore/?t=extremes" data-tag="extremes">Extremes</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=sub-hourly" data-tag="sub-hourly">Sub-hourly</a>
    { .paper-tags }

-   #### Probabilistic NDVI Forecasting from Sparse Satellite Time Series and Weather Covariates { #2602.17683 }

    *Irene Iele, Giulia Romoli, Daniele Molino, Elena Mulero Ayllón, Filippo Ruffini, Paolo Soda et al.* · Feb 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2602.17683">Short-term forecasting of vegetation dynamics is a key enabler for data-driven decision support in precision agriculture. Normalized Difference Vegetation Index (NDVI) forecasting from satellite...</span><span class="abstract-full" id="full-2602.17683" hidden>Short-term forecasting of vegetation dynamics is a key enabler for data-driven decision support in precision agriculture. Normalized Difference Vegetation Index (NDVI) forecasting from satellite observations, however, remains challenging due to sparse and irregular sampling caused by cloud masking, as well as the heterogeneous climatic conditions under which crops evolve. In this work, we propose a probabilistic forecasting framework for field-level NDVI prediction under sparse, irregular clear-sky acquisitions. The architecture separates the encoding of historical NDVI and meteorological observations from future exogenous covariates, fusing both representations for multi-step quantile prediction. To address irregular revisit patterns and horizon-dependent uncertainty, we introduce a temporal-distance weighted quantile loss that aligns the training objective with the effective forecasting horizon. In addition, we incorporate cumulative and extreme-weather feature engineering to capture delayed meteorological effects relevant to vegetation response. Experiments on European satellite data show that the proposed approach outperforms statistical, deep learning, and time-series baselines on both pointwise and probabilistic evaluation metrics. Ablation studies confirm that target history is the primary driver of performance, with meteorological covariates providing additional gains in the full multimodal setting. The code is available at https://github.com/arco-group/ndvi-forecasting.</span> <span class="abstract-toggle" data-id="2602.17683">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2602.17683v3) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2602.17683v3) · [:fontawesome-brands-github: Code](https://github.com/arco-group/ndvi-forecasting) · [:material-content-copy: BibTeX](../../bibtex/2602.17683.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### Distillation and Interpretability of Ensemble Forecasts of ENSO Phase using Entropic Learning { #2602.16857 }

    *Michael Groom, Davide Bassetti, Illia Horenko, Terence J. O'Kane* · Feb 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2602.16857">This paper introduces a distillation framework for an ensemble of entropy-optimal Sparse Probabilistic Approximation (eSPA) models, trained exclusively on satellite-era observational and reanalysis...</span><span class="abstract-full" id="full-2602.16857" hidden>This paper introduces a distillation framework for an ensemble of entropy-optimal Sparse Probabilistic Approximation (eSPA) models, trained exclusively on satellite-era observational and reanalysis data to predict ENSO phase up to 24 months in advance. While eSPA ensembles yield state-of-the-art forecast skill, they are harder to interpret than individual eSPA models. We show how to compress the ensemble into a compact set of "distilled" models by aggregating the structure of only those ensemble members that make correct predictions. This process yields a single, diagnostically tractable model for each forecast lead time that preserves forecast performance while also enabling diagnostics that are impractical to implement on the full ensemble.   An analysis of the regime persistence of the distilled model "superclusters", as well as cross-lead clustering consistency, shows that the discretised system accurately captures the spatiotemporal dynamics of ENSO. By considering the effective dimension of the feature importance vectors, the complexity of the input space required for correct ENSO phase prediction is shown to peak when forecasts must cross the boreal spring predictability barrier. Spatial importance maps derived from the feature importance vectors are introduced to identify where predictive information resides in each field and are shown to include known physical precursors at certain lead times. Case studies of key events are also presented, showing how fields reconstructed from distilled model centroids trace the evolution from extratropical and inter-basin precursors to the mature ENSO state. Overall, the distillation framework enables a rigorous investigation of long-range ENSO predictability that complements real-time data-driven operational forecasts.</span> <span class="abstract-toggle" data-id="2602.16857">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2602.16857v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2602.16857v1) · [:material-content-copy: BibTeX](../../bibtex/2602.16857.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=subseasonal-to-seasonal" data-tag="subseasonal-to-seasonal">Subseasonal to seasonal</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=interpretability" data-tag="interpretability">Interpretability</a>
    { .paper-tags }

-   #### Ensemble-size-dependence of deep-learning post-processing methods that minimize an (un)fair score: motivating examples and a proof-of-concept solution { #2602.15830 }

    *Christopher David Roberts* · Feb 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2602.15830">Fair scores reward ensemble forecast members that behave like samples from the same distribution as the verifying observations. They are therefore an attractive choice as loss functions to train...</span><span class="abstract-full" id="full-2602.15830" hidden>Fair scores reward ensemble forecast members that behave like samples from the same distribution as the verifying observations. They are therefore an attractive choice as loss functions to train data-driven ensemble forecasts or post-processing methods when large training ensembles are either unavailable or computationally prohibitive. The adjusted continuous ranked probability score (aCRPS) is fair and unbiased with respect to ensemble size, provided forecast members are exchangeable and interpretable as conditionally independent draws from an underlying predictive distribution. However, distribution-aware post-processing methods that introduce structural dependency between members can violate this assumption, rendering aCRPS unfair. We demonstrate this effect using two approaches designed to minimize the expected aCRPS of a finite ensemble: (1) a linear member-by-member calibration, which couples members through a common dependency on the sample ensemble mean, and (2) a deep-learning method, which couples members via transformer self-attention across the ensemble dimension. In both cases, the results are sensitive to ensemble size and apparent gains in aCRPS can correspond to systematic unreliability characterized by over-dispersion. We introduce trajectory transformers as a proof-of-concept that ensemble-size independence can be achieved. This approach is an adaptation of the Post-processing Ensembles with Transformers (PoET) framework and applies self-attention over lead time while preserving the conditional independence required by aCRPS. When applied to weekly mean $T_{2m}$ forecasts from the ECMWF subseasonal forecasting system, this approach successfully reduces systematic model biases whilst also improving or maintaining forecast reliability regardless of the ensemble size used in training (3 vs 9 members) or real-time forecasts (9 vs 100 members).</span> <span class="abstract-toggle" data-id="2602.15830">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2602.15830v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2602.15830v1) · [:material-content-copy: BibTeX](../../bibtex/2602.15830.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=transformers" data-tag="transformers">Transformers</a> <a class="md-tag" href="/explore/?t=subseasonal-to-seasonal" data-tag="subseasonal-to-seasonal">Subseasonal to seasonal</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### Blackening Cryosphere: Revealing Hotspot Shifts and HGB-Based Forecasting of Absorbing Aerosol Threats over the Himalayan Frozen Frontiers { #2602.15052 }

    *Abira Sengupta, Ayoti Banerjee, Sarbani Palit, Brendon Woodford* · Feb 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2602.15052">Black carbon and mineral dust are key absorbing aerosols that influence atmospheric radiation and increasingly threaten global cryospheric stability. This study examines the long-range transport and...</span><span class="abstract-full" id="full-2602.15052" hidden>Black carbon and mineral dust are key absorbing aerosols that influence atmospheric radiation and increasingly threaten global cryospheric stability. This study examines the long-range transport and seasonal variability of these aerosols over Pakistan and their movement toward the western Himalayas. Using satellite-derived Absorption Aerosol Optical Depth (AAOD) data from 2019 to mid-2025, we analyse their spatiotemporal behaviour across Pakistan's urban lowlands and high-altitude regions. Fifteen-day aggregated AAOD fields are used to track seasonal transport into glaciated terrain, where deposited aerosols can darken snow and ice and accelerate melt. For high-AAOD events, a probabilistic forecasting approach based on machine learning (ML) was developed. Using geographical, seasonal, and lagged indicators, a histogram-based gradient boosting classifier was trained to predict AAOD exceedance one step in advance. ROC-AUC, PR-AUC, and the Brier score were used to assess the model's performance. The results show high predictive capacity and good probability calibration, with values of 0.791, 0.269, and 0.028, respectively. Forecasts indicate that areas adjacent to Himalayan glaciers consistently exhibit the highest probability of increasing AAOD, signalling an elevated risk of aerosol-induced snowmelt.</span> <span class="abstract-toggle" data-id="2602.15052">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2602.15052v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2602.15052v1) · [:material-content-copy: BibTeX](../../bibtex/2602.15052.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=classical-ml" data-tag="classical-ml">Classical ML</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a>
    { .paper-tags }

-   #### Probabilistic Wind Power Forecasting with Tree-Based Machine Learning and Weather Ensembles { #2602.13010 }

    *Max Bruninx, Diederik van Binsbergen, Timothy Verstraeten, Ann Nowé, Jan Helsen* · Feb 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2602.13010">Accurate production forecasts are essential for the integration of renewable energy sources into the power grid. This paper illustrates how to obtain probabilistic forecasts of wind power generation...</span><span class="abstract-full" id="full-2602.13010" hidden>Accurate production forecasts are essential for the integration of renewable energy sources into the power grid. This paper illustrates how to obtain probabilistic forecasts of wind power generation using gradient boosting trees and an ensemble of weather forecasts. To this end, we perform a comparative analysis across three state-of-the-art probabilistic prediction methods-conformalized quantile regression, natural gradient boosting and conditional diffusion models-all of which can be combined with tree-based machine learning. The methods are validated using four years of data for all Belgian offshore wind farms. We benchmark the models against the power curve and a calibrated wake model as well as a probabilistic method using stochastic variational Gaussian process regression. The tree-based models significantly reduce the mean absolute error in comparison to the deterministic baselines. Additionally, all three methods outperform the Gaussian process baseline in probabilistic skill, while two out of the three also improve point forecast accuracy. The conditional diffusion model attains the best performance, with improvements of 5% in mean absolute error and 12% in continuous rank probability score compared to the probabilistic baseline. Last, the results indicate an average improvement in point forecast accuracy of 17% by using an ensemble of weather forecasts instead of a single provider.</span> <span class="abstract-toggle" data-id="2602.13010">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2602.13010v2) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2602.13010v2) · [:material-content-copy: BibTeX](../../bibtex/2602.13010.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=classical-ml" data-tag="classical-ml">Classical ML</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=energy" data-tag="energy">Energy</a>
    { .paper-tags }

-   #### PuYun-LDM: A Latent Diffusion Model for High-Resolution Ensemble Weather Forecasts { #2602.11807 }

    *Lianjun Wu, Shengchen Zhu, Yuxuan Liu, Liuyu Kai, Xiaoduan Feng, Duomin Wang, Wenshuo Liu et al.* · Feb 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2602.11807">Latent diffusion models (LDMs) suffer from limited diffusability in high-resolution (<=0.25°) ensemble weather forecasting, where diffusability characterizes how easily a latent data distribution can...</span><span class="abstract-full" id="full-2602.11807" hidden>Latent diffusion models (LDMs) suffer from limited diffusability in high-resolution (<=0.25°) ensemble weather forecasting, where diffusability characterizes how easily a latent data distribution can be modeled by a diffusion process. Unlike natural image fields, meteorological fields lack task-agnostic foundation models and explicit semantic structures, making VFM-based regularization inapplicable. Moreover, existing frequency-based approaches impose identical spectral regularization across channels under a homogeneity assumption, which leads to uneven regularization strength under the inter-variable spectral heterogeneity in multivariate meteorological data. To address these challenges, we propose a 3D Masked AutoEncoder (3D-MAE) that encodes weather-state evolution features as an additional conditioning for the diffusion model, together with a Variable-Aware Masked Frequency Modeling (VA-MFM) strategy that adaptively selects thresholds based on the spectral energy distribution of each variable. Together, we propose PuYun-LDM, which enhances latent diffusability and achieves superior performance to ENS at short lead times while remaining comparable to ENS at longer horizons. PuYun-LDM generates a 15-day global forecast with a 6-hour temporal resolution in five minutes on a single NVIDIA H200 GPU, while ensemble forecasts can be efficiently produced in parallel.</span> <span class="abstract-toggle" data-id="2602.11807">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2602.11807v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2602.11807v1) · [:material-content-copy: BibTeX](../../bibtex/2602.11807.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a> <a class="md-tag" href="/explore/?t=quarter-degree" data-tag="quarter-degree">0.25°</a>
    { .paper-tags }

-   #### Reduced-Order Surrogates for Forced Flexible Mesh Coastal-Ocean Models { #2602.05416 }

    *Freja Høgholm Petersen, Jesper Sandvig Mariegaard, Rocco Palmitessa, Allan P. Engsig-Karup* · Feb 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2602.05416">While proper orthogonal decomposition (POD)-based surrogates are widely explored for hydrodynamic applications, the use of Koopman autoencoders for real-world coastal-ocean modelling remains...</span><span class="abstract-full" id="full-2602.05416" hidden>While proper orthogonal decomposition (POD)-based surrogates are widely explored for hydrodynamic applications, the use of Koopman autoencoders for real-world coastal-ocean modelling remains relatively limited. This paper introduces a flexible Koopman autoencoder formulation that incorporates meteorological forcings and boundary conditions, and systematically compares its performance against POD-based surrogates. The Koopman autoencoder employs a learned linear temporal operator in latent space, enabling eigenvalue regularization to promote temporal stability. This strategy is evaluated alongside temporal unrolling techniques for achieving stable and accurate long-term predictions. The models are assessed on three test cases spanning distinct dynamical regimes, with prediction horizons up to one year at 30-minute temporal resolution. Across all cases, the reduced order surrogates with temporal unrolling achieve high accuracy with relative root-mean-squared-errors of 0.0068-0.14 and $R^2$-values of 0.61-0.995, where prediction errors are largest for current velocities, and smallest for water surface elevations. In two of the three cases, the Koopman Autoencoder have higher accuracy than the POD-based surrogates. Comparing to in-situ observations, the surrogate yields -0.64% to 12% increase in water surface elevation prediction error when compared to prediction errors of the physics-based model. These error levels, corresponding to a few centimeters, are acceptable for many practical applications, while inference speed-ups of 300-1400x enables workflows such as ensemble forecasting and long climate simulations for coastal-ocean modelling.</span> <span class="abstract-toggle" data-id="2602.05416">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2602.05416v2) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2602.05416v2) · [:material-content-copy: BibTeX](../../bibtex/2602.05416.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=sub-hourly" data-tag="sub-hourly">Sub-hourly</a>
    { .paper-tags }

-   #### Large-Ensemble Simulations Reveal Links Between Atmospheric Blocking Frequency and Sea Surface Temperature Variability { #2602.05083 }

    *Zilu Meng, Gregory J. Hakim, Wenchang Yang, Gabriel A. Vecchi* · Feb 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2602.05083">Atmospheric blocking events drive persistent weather extremes in midlatitudes, but isolating the influence of sea surface temperature (SST) from chaotic internal atmospheric variability on these...</span><span class="abstract-full" id="full-2602.05083" hidden>Atmospheric blocking events drive persistent weather extremes in midlatitudes, but isolating the influence of sea surface temperature (SST) from chaotic internal atmospheric variability on these events remains a challenge. We address this challenge using century-long (1900-2010), large-ensemble simulations with two computationally efficient deep-learning general circulation models. We find these models skillfully reproduce the observed blocking climatology, matching or exceeding the performance of a traditional high-resolution model and representative CMIP6 models. Averaging the large ensembles filters internal atmospheric noise to isolate the SST-forced component of blocking variability, yielding substantially higher correlations with reanalysis than for individual ensemble members. We identify robust teleconnections linking Greenland blocking frequency to North Atlantic SST and El Niño-like patterns. Furthermore, SST-forced trends in blocking frequency show a consistent decline in winter over Greenland, and an increase over Europe. These results demonstrate that SST variability exerts a significant and physically interpretable influence on blocking frequency and establishes large ensembles from deep learning models as a powerful tool for separating forced SST signals from internal noise.</span> <span class="abstract-toggle" data-id="2602.05083">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2602.05083v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2602.05083v1) · [:material-content-copy: BibTeX](../../bibtex/2602.05083.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=subseasonal-to-seasonal" data-tag="subseasonal-to-seasonal">Subseasonal to seasonal</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a>
    { .paper-tags }

-   #### Resilient Load Forecasting under Climate Change: Adaptive Conditional Neural Processes for Few-Shot Extreme Load Forecasting { #2602.04609 }

    *Chenxi Hu, Yue Ma, Yifan Wu, Yunhe Hou* · Feb 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2602.04609">Extreme weather can substantially change electricity consumption behavior, causing load curves to exhibit sharp spikes and pronounced volatility. If forecasts are inaccurate during those periods,...</span><span class="abstract-full" id="full-2602.04609" hidden>Extreme weather can substantially change electricity consumption behavior, causing load curves to exhibit sharp spikes and pronounced volatility. If forecasts are inaccurate during those periods, power systems are more likely to face supply shortfalls or localized overloads, forcing emergency actions such as load shedding and increasing the risk of service disruptions and public-safety impacts. This problem is inherently difficult because extreme events can trigger abrupt regime shifts in load patterns, while relevant extreme samples are rare and irregular, making reliable learning and calibration challenging. We propose AdaCNP, a probabilistic forecasting model for data-scarce condition. AdaCNP learns similarity in a shared embedding space. For each target data, it evaluates how relevant each historical context segment is to the current condition and reweights the context information accordingly. This design highlights the most informative historical evidence even when extreme samples are rare. It enables few-shot adaptation to previously unseen extreme patterns. AdaCNP also produces predictive distributions for risk-aware decision-making without expensive fine-tuning on the target domain. We evaluate AdaCNP on real-world power-system load data and compare it against a range of representative baselines. The results show that AdaCNP is more robust during extreme periods, reducing the mean squared error by 22% relative to the strongest baseline while achieving the lowest negative log-likelihood, indicating more reliable probabilistic outputs. These findings suggest that AdaCNP can effectively mitigate the combined impact of abrupt distribution shifts and scarce extreme samples, providing a more trustworthy forecasting for resilient power system operation under extreme events.</span> <span class="abstract-toggle" data-id="2602.04609">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2602.04609v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2602.04609v1) · [:material-content-copy: BibTeX](../../bibtex/2602.04609.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=extremes" data-tag="extremes">Extremes</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=energy" data-tag="energy">Energy</a>
    { .paper-tags }

-   #### WIND: Weather Inverse Diffusion for Zero-Shot Atmospheric Modeling { #2602.03924 }

    *Michael Aich, Andreas Fürst, Florian Sestak, Carlos Ruiz-Gonzalez, Niklas Boers et al.* · Feb 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2602.03924">Deep learning has revolutionized weather forecasting, but many challenges remain, including climate modeling. Moreover, the current landscape remains fragmented: highly specialized models are...</span><span class="abstract-full" id="full-2602.03924" hidden>Deep learning has revolutionized weather forecasting, but many challenges remain, including climate modeling. Moreover, the current landscape remains fragmented: highly specialized models are typically trained individually for distinct tasks. To unify this landscape, we introduce WIND, a single pre-trained foundation model capable of replacing specialized baselines across a vast array of tasks. Crucially, in contrast to previous atmospheric foundation models, we achieve this without any task-specific fine-tuning. To learn a robust, task-agnostic prior of the atmosphere, we pre-train WIND with a self-supervised video reconstruction objective, utilizing an unconditional video diffusion model to iteratively reconstruct atmospheric dynamics from a noisy state. At inference, we frame diverse domain-specific problems strictly as inverse problems and solve them via posterior sampling. This unified approach allows us to tackle highly relevant weather and climate problems, including probabilistic forecasting, spatial and temporal downscaling, reconstruction of spatial fields from sparse observations and enforcing global dry air mass conservation. We further demonstrate how WIND can be applied to explore extreme weather events under prescribed out-of-distribution thermodynamic perturbations. By combining generative video modeling with inverse problem solving, WIND offers a computationally efficient alternative for AI-based atmospheric modeling.</span> <span class="abstract-toggle" data-id="2602.03924">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2602.03924v2) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2602.03924v2) · [:material-content-copy: BibTeX](../../bibtex/2602.03924.bib){ .bibtex-link }
    { .paper-links }

-   #### Long-Term Probabilistic Forecast of Vegetation Conditions Using Climate Attributes in the Four Corners Region { #2601.16347 }

    *Erika McPhillips, Hyeongseong Lee, Xiangyu Xie, Kathy Baylis, Chris Funk, Mengyang Gu* · Jan 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2601.16347">Weather conditions can drastically alter the state of crops and rangelands, and in turn, impact the incomes and food security of individuals worldwide. Satellite-based remote sensing offers an...</span><span class="abstract-full" id="full-2601.16347" hidden>Weather conditions can drastically alter the state of crops and rangelands, and in turn, impact the incomes and food security of individuals worldwide. Satellite-based remote sensing offers an effective way to monitor vegetation and climate variables on regional and global scales. The annual peak Normalized Difference Vegetation Index (NDVI), derived from satellite observations, is closely associated with crop development, rangeland biomass, and vegetation growth. Although various machine learning methods have been developed to forecast NDVI over short time ranges, such as one-month-ahead predictions, long-term forecasting approaches, such as one-year-ahead predictions of vegetation conditions, are not yet available. To fill this gap, we develop a two-phase machine learning model to forecast the one-year-ahead peak NDVI over high-resolution grids, using the Four Corners region of the Southwestern United States as a testbed. In phase one, we identify informative climate attributes, including precipitation and maximum vapor pressure deficit, and develop the generalized parallel Gaussian process that captures the relationship between climate attributes and NDVI. In phase two, we forecast these climate attributes using historical data at least one year before the NDVI prediction month, which then serve as inputs to forecast the peak NDVI at each spatial grid. We developed open-source tools that outperform alternative methods for both gross NDVI and grid-based NDVI one-year forecasts, providing information that can help farmers and ranchers make actionable plans a year in advance.</span> <span class="abstract-toggle" data-id="2601.16347">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2601.16347v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2601.16347v1) · [:material-content-copy: BibTeX](../../bibtex/2601.16347.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=classical-ml" data-tag="classical-ml">Classical ML</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a>
    { .paper-tags }

-   #### STIPP: Space-time in situ postprocessing over the French Alps using proper scoring rules { #2601.02882 }

    *David Landry, Isabelle Gouttevin, Hugo Merizen, Claire Monteleoni, Anastase Charantonis* · Jan 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2601.02882">We propose Space-time in situ postprocessing (STIPP), a machine learning model that generates spatio-temporally consistent weather forecasts for a network of station locations. Gridded forecasts from...</span><span class="abstract-full" id="full-2601.02882" hidden>We propose Space-time in situ postprocessing (STIPP), a machine learning model that generates spatio-temporally consistent weather forecasts for a network of station locations. Gridded forecasts from classical numerical weather prediction or data-driven models often lack the necessary precision due to unresolved local effects. Typical statistical postprocessing methods correct these biases, but often degrade spatio-temporal correlation structures in doing so. Recent works based on generative modeling successfully improve spatial correlation structures but have to forecast every lead time independently. In contrast, STIPP makes joint spatio-temporal forecasts which have increased accuracy for surface temperature, wind, relative humidity and precipitation when compared to baseline methods. It makes hourly ensemble predictions given only a six-hourly deterministic forecast, blending the boundaries of postprocessing and temporal interpolation. By leveraging a multivariate proper scoring rule for training, STIPP contributes to ongoing work data-driven atmospheric models supervised only with distribution marginals.</span> <span class="abstract-toggle" data-id="2601.02882">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2601.02882v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2601.02882v1) · [:material-content-copy: BibTeX](../../bibtex/2601.02882.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=station-point" data-tag="station-point">Station / point</a> <a class="md-tag" href="/explore/?t=hourly" data-tag="hourly">Hourly</a> <a class="md-tag" href="/explore/?t=6-hourly" data-tag="6-hourly">6-hourly</a>
    { .paper-tags }

-   #### Latent-Constrained Conditional VAEs for Augmenting Large-Scale Climate Ensembles { #2601.00915 }

    *Jacquelyn Shelton, Przemyslaw Polewski, Alexander Robel, Matthew Hoffman, Stephen Price* · Jan 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2601.00915">Large climate-model ensembles are computationally expensive; yet many downstream analyses would benefit from additional, statistically consistent realizations of spatiotemporal climate variables. We...</span><span class="abstract-full" id="full-2601.00915" hidden>Large climate-model ensembles are computationally expensive; yet many downstream analyses would benefit from additional, statistically consistent realizations of spatiotemporal climate variables. We study a generative modeling approach for producing new realizations from a limited set of available runs by transferring structure learned across an ensemble. Using monthly near-surface temperature time series from ten independent reanalysis realizations (ERA5), we find that a vanilla conditional variational autoencoder (CVAE) trained jointly across realizations yields a fragmented latent space that fails to generalize to unseen ensemble members. To address this, we introduce a latent-constrained CVAE (LC-CVAE) that enforces cross-realization homogeneity of latent embeddings at a small set of shared geographic 'anchor' locations. We then use multi-output Gaussian process regression in the latent space to predict latent coordinates at unsampled locations in a new realization, followed by decoding to generate full time series fields. Experiments and ablations demonstrate (i) instability when training on a single realization, (ii) diminishing returns after incorporating roughly five realizations, and (iii) a trade-off between spatial coverage and reconstruction quality that is closely linked to the average neighbor distance in latent space.</span> <span class="abstract-toggle" data-id="2601.00915">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2601.00915v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2601.00915v1) · [:material-content-copy: BibTeX](../../bibtex/2601.00915.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=classical-ml" data-tag="classical-ml">Classical ML</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=monthly" data-tag="monthly">Monthly</a>
    { .paper-tags }

-   #### Rainfall forecasts in daily use over East Africa improved by machine learning { #2512.24525 }

    *Fenwick C. Cooper, Shruti Nath, Andrew T. T. McRae, Bobby Antonio, Antje Weisheimer, Tim Palmer et al.* · Dec 2025
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2512.24525">Ensemble forecasting has proven over the years to be a vital tool for predicting extreme or only partially predictable weather events. In particular life-threatening weather events. Many National...</span><span class="abstract-full" id="full-2512.24525" hidden>Ensemble forecasting has proven over the years to be a vital tool for predicting extreme or only partially predictable weather events. In particular life-threatening weather events. Many National Meteorological Services in East Africa do not have the computing resources to enable them to run their local area models in full ensemble mode over the full period of the 2 week medium range. As a result, weather users in these countries are not being given sufficient information about weather risk that is needed to make reliable decisions about taking preventative action. Consequently, society in many parts of the world is not as resilient to weather events as they could be. In this paper we test the performance of our forecast system, cGAN, which is the only high-resolution (10 km) ensemble rainfall product that does real-time, probabilistic correction of global forecasts for East Africa. Compared to existing state-of-the-art AI models, our system offers higher spatial resolution. It is cheap to train/run and requires no additional post-processing. It is run on laptops and can generate many thousands of ensemble members at little computational cost (compared with physical local area models). It is ideally suited to Meteorological Services with limited computational facilities.</span> <span class="abstract-toggle" data-id="2512.24525">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2512.24525v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2512.24525v1) · [:material-content-copy: BibTeX](../../bibtex/2512.24525.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a> <a class="md-tag" href="/explore/?t=daily" data-tag="daily">Daily</a>
    { .paper-tags }

-   #### Lazy Diffusion: Mitigating spectral collapse in generative diffusion-based stable autoregressive emulation of turbulent flows { #2512.09572 }

    *Anish Sambamurthy, Ashesh Chattopadhyay* · Dec 2025
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2512.09572">Turbulent flows posses broadband, power-law spectra in which multiscale interactions couple high-wavenumber fluctuations to large-scale dynamics. Although diffusion-based generative models offer a...</span><span class="abstract-full" id="full-2512.09572" hidden>Turbulent flows posses broadband, power-law spectra in which multiscale interactions couple high-wavenumber fluctuations to large-scale dynamics. Although diffusion-based generative models offer a principled probabilistic forecasting framework, we show that standard DDPMs induce a fundamental <em>spectral collapse</em>: a Fourier-space analysis of the forward SDE reveals a closed-form, mode-wise signal-to-noise ratio (SNR) that decays monotonically in wavenumber, $|k|$ for spectra $S(k)\!\propto\!|k|^{-λ}$, rendering high-wavenumber modes indistinguishable from noise and producing an intrinsic spectral bias. We reinterpret the noise schedule as a spectral regularizer and introduce power-law schedules $β(τ)\!\propto\!τ^γ$ that preserve fine-scale structure deeper into diffusion time, along with <em>Lazy Diffusion</em>, a one-step distillation method that leverages the learned score geometry to bypass long reverse-time trajectories and prevent high-$k$ degradation. Applied to high-Reynolds-number 2D Kolmogorov turbulence and $1/12^\circ$ Gulf of Mexico ocean reanalysis, these methods resolve spectral collapse, stabilize long-horizon autoregression, and restore physically realistic inertial-range scaling. Together, they show that naïve Gaussian scheduling is structurally incompatible with power-law physics and that physics-aware diffusion processes can yield accurate, efficient, and fully probabilistic surrogates for multiscale dynamical systems.</span> <span class="abstract-toggle" data-id="2512.09572">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2512.09572v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2512.09572v1) · [:material-content-copy: BibTeX](../../bibtex/2512.09572.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

</div>

<nav class="pager" markdown="span">**1** [2](2.md) [3](3.md) [Older :material-arrow-right:](2.md){ .pager-step }</nav>

