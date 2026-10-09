---
hide:
  - navigation
  - toc
title: weatherml
---

A collection of papers on AI for weather forecasting, climate modelling and atmospheric science.

<p class="page-meta" markdown="span">1537 papers · updated 2026-10-09 · <a href="feed.xml">:material-rss: RSS</a> · <a href="all_papers.bib" download>:material-download: BibTeX</a> · <a href="https://github.com/weatherml/weatherml.github.io/issues/new?template=suggest-paper.yml">:material-plus: Suggest a paper</a></p>

## Browse by Topic

<div class="grid cards topics" markdown>

-   [Global Models](papers/global-models/index.md) <span class="topic-count">365</span>
-   [Regional Models](papers/regional-models/index.md) <span class="topic-count">65</span>
-   [Nowcasting](papers/nowcasting/index.md) <span class="topic-count">102</span>
-   [Downscaling](papers/downscaling/index.md) <span class="topic-count">100</span>
-   [Post-processing](papers/post-processing/index.md) <span class="topic-count">26</span>
-   [Data Assimilation](papers/data-assimilation/index.md) <span class="topic-count">78</span>
-   [Climate Modeling](papers/climate-modeling/index.md) <span class="topic-count">299</span>
-   [Hydrology](papers/hydrology/index.md) <span class="topic-count">37</span>
-   [Ocean & Sea Ice](papers/ocean-sea-ice/index.md) <span class="topic-count">88</span>
-   [Air Quality & Composition](papers/air-quality-composition/index.md) <span class="topic-count">71</span>
-   [Remote Sensing](papers/remote-sensing/index.md) <span class="topic-count">100</span>
-   [Other](papers/other/index.md) <span class="topic-count">206</span>

</div>

## Recent Additions

<div class="grid cards" markdown data-search-exclude>

-   #### Learning Kilometer-Scale Weather Prediction with Global-Regional Alignment { #2610.12401 }

    *Guowen Li, Yang Liu, Yujie Wang, Qiuyan Sun, Haoyuan Liang, Juepeng Zheng, Hong Cheng, Haohuan Fu* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.12401">Kilometer-scale regional weather forecasting is essential for local weather warnings and weather-sensitive decisions. Existing data-driven approaches often rely on numerical forecasts for large-scale...</span><span class="abstract-full" id="full-2610.12401" hidden>Kilometer-scale regional weather forecasting is essential for local weather warnings and weather-sensitive decisions. Existing data-driven approaches often rely on numerical forecasts for large-scale guidance or require additional training of global forecasting components. Pretrained global weather models offer an efficient source of large-scale forecasts, motivating their reuse to guide high-resolution regional prediction. However, this coupling requires aligning global and regional representations across different grids and integrating global guidance with local interactions to advance regional states. We propose ScaleCast, a regional forecasting framework that addresses these challenges through Global-Regional Alignment. Its Global-Regional Conversion module aligns joint global and regional representations with regional locations, while the Global-Regional Alignment and Dynamics block combines aligned guidance with regional neighborhood interactions. Experiments using ERA5 global analyses on a 0.25-degree grid and CERRA regional reanalysis at 5.5 km spacing demonstrate improved regional forecasts across surface and upper-air variables, with a single trained model supporting multiple global forecast drivers (i.e., Pangu-Weather, GraphCast, and HRES) without specific retraining. Fine-tuning on HRRR at 3 km spacing further demonstrates the framework's adaptability to a different regional domain and spatial resolution. Windstorm case studies show improved cyclone positioning and core-pressure estimates, while comparisons with HadISD station observations show closer agreement with local temperature and humidity changes.</span> <span class="abstract-toggle" data-id="2610.12401">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.12401v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.12401v1) · [:material-content-copy: BibTeX](bibtex/2610.12401.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a> <a class="md-tag" href="/explore/?t=km-scale" data-tag="km-scale">Km-scale</a> <a class="md-tag" href="/explore/?t=quarter-degree" data-tag="quarter-degree">0.25°</a>
    { .paper-tags }

-   #### Just Weather Scoring: Efficient End-to-end Nowcasting with Distributional Diffusion { #2610.12189 }

    *Jannik Wiese, Johannes Schusterbauer, Tommaso Martorella, Björn Ommer* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.12189">Generative diffusion models are well-suited for probabilistic precipitation nowcasting, but existing approaches often rely on separately trained compression or deterministic forecasting components...</span><span class="abstract-full" id="full-2610.12189" hidden>Generative diffusion models are well-suited for probabilistic precipitation nowcasting, but existing approaches often rely on separately trained compression or deterministic forecasting components and remain costly at inference due to iterative denoising. We introduce Just Weather Scoring (JWS), a single-stage, end-to-end diffusion model which addresses both issues by forecasting directly in radar space and enabling few-step generation. Radar-space modeling greatly simplifies training and inference and eliminates uncertainty arising from lossy compression. JWS combines Masked Asynchronous Diffusion, a timestep-sampling scheme that preserves clean context while adapting diffusion training to high-dimensional spatio-temporal data, with a simple scoring-rule objective that aligns training with probabilistic forecasting and unlocks few-step generation. On the SEVIR and MeteoNet benchmarks, JWS achieves state-of-the-art probabilistic forecasting performance at reduced training and inference cost. Even our smallest model remains competitive using substantially fewer parameters and more than 17x faster inference.</span> <span class="abstract-toggle" data-id="2610.12189">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.12189v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.12189v1) · [:material-content-copy: BibTeX](bibtex/2610.12189.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a>
    { .paper-tags }

-   #### legoESM: a modular, differentiable, multiscale, AI-ready Earth system model built with AI agents { #2610.11883 }

    *Pierre Gentine, Dhruv Balwada, Aytaç Paçal, Linnia Hawkins, Alistair Adcroft, Hang Fan et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.11883">Earth system models (ESMs) have grown tremendously in realism, yet key uncertainties persist in the climate response to greenhouse-gas forcing, particularly due to cloud radiative feedbacks. In...</span><span class="abstract-full" id="full-2610.11883" hidden>Earth system models (ESMs) have grown tremendously in realism, yet key uncertainties persist in the climate response to greenhouse-gas forcing, particularly due to cloud radiative feedbacks. In addition, their software architecture was not designed for accelerator hardware or modern artificial intelligence (AI). Here we present legoESM, a composable, differentiable, multiscale ESM written in JAX. It builds on decades of community-developed parameterizations and numerical methods, recast in a unified framework by AI coding agents under a human-specified scientific contract and verified through benchmarking. Dynamical cores, physics schemes, grids, complexity levels and components are swappable like building blocks, and can use conventional physics or machine-learned emulators. A single code base spans metre-scale large-eddy simulation to global simulations and weather to climate. End-to-end differentiability enables gradient-based calibration, variational data assimilation and online training. legoESM modular architecture enables systematic evaluation of diverse model variants to explore structural uncertainty and test hypotheses. legoESM produces realistic simulations across scales, reduces land-surface temperature bias through gradient-based calibration, and scales efficiently on GPUs to kilometer-scale simulations. It offers an open, community infrastructure for hypothesis testing, research and teaching in Earth sciences and a template for multiscale physical systems.</span> <span class="abstract-toggle" data-id="2610.11883">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.11883v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.11883v1) · [:material-content-copy: BibTeX](bibtex/2610.11883.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=km-scale" data-tag="km-scale">Km-scale</a>
    { .paper-tags }

-   #### A Physics-Constrained Implicit Profile Network for Continuous Reconstruction of Tropical Cyclone Near-Surface Wind Profiles { #2610.11405 }

    *Jian Ma, Yilin Yang, Robert Rogers, Jun A. Zhang, Jie Tang* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.11405">Near the ocean surface, tropical cyclone winds change rapidly with height, but direct measurements are limited because aircraft dropsondes provide only sparse and irregular observations. Continuous...</span><span class="abstract-full" id="full-2610.11405" hidden>Near the ocean surface, tropical cyclone winds change rapidly with height, but direct measurements are limited because aircraft dropsondes provide only sparse and irregular observations. Continuous wind profiles are important for understanding hurricane boundary-layer processes, improving storm-surge prediction, supporting offshore engineering, and assessing coastal hazards. In this study, we developed an artificial intelligence model that combines machine learning with physical principles to reconstruct continuous wind profiles from sparse observations. The model is designed to preserve the observed surface winds while generating realistic changes in wind speed and direction with height. Tests using more than two decades of NOAA hurricane observations show that the method accurately reproduces the vertical structure of tropical cyclone winds over a wide range of storm intensities. The framework can also extend satellite-derived surface wind measurements into three-dimensional near-surface wind fields, providing new opportunities for hurricane research, operational forecasting, and engineering applications.</span> <span class="abstract-toggle" data-id="2610.11405">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.11405v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.11405v1) · [:material-content-copy: BibTeX](bibtex/2610.11405.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=physics-ml-hybrid" data-tag="physics-ml-hybrid">Physics–ML hybrid</a> <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a>
    { .paper-tags }

-   #### A Graph Neural Network for Global Daily Fire Radiative Power Prediction at Medium-Range Lead Times { #2610.11022 }

    *Li Zhang, Jun Wang, Isidora Jankov, Yongxin Liu, Gonzalo A. Ferrada, Ravan Ahmadov, Ligia Bernardet et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.11022">Skillful prediction of biomass-burning activity several days in advance is important for air-quality forecasting and aerosol prediction. Two operational constraints motivate this work. First, the...</span><span class="abstract-full" id="full-2610.11022" hidden>Skillful prediction of biomass-burning activity several days in advance is important for air-quality forecasting and aerosol prediction. Two operational constraints motivate this work. First, the GBBEPx satellite fire radiative power (FRP) product used to initialize NOAA's GEFS-Aerosols is available with about a 1.5-day latency, so each forecast cycle relies on the most recently available, but already outdated, fire observations. Second, these fire inputs are then held fixed throughout the subsequent 5-day operational forecast, or 7 days in the GSL experimental system, effectively assuming no evolution in fire activity. We develop a data-driven model that predicts global FRP one to seven days ahead from the most recent available observations. The model adapts a spatiotemporal graph neural network using reanalysis meteorology, land-cover and vegetation information, recent fire history, and GBBEPx FRP as the training target. It is trained on 2020-2022 data and evaluated for 2023-2024. The model reproduces the global seasonal cycle and substantially outperforms persistence. At 0.1$^\circ$ resolution, mean squared error is reduced by 32% at one-day lead and 43% at seven days in 2023, and by 24% and 40% in 2024. At 1$^\circ$ resolution, the critical success index ranges from 0.32 to 0.60. Detection skill declines only modestly with lead time, whereas intensity skill degrades more rapidly. Large fires are detected reliably, but their radiative power is systematically underestimated. These results demonstrate useful predictability of fire activity several days ahead and identify intensity calibration and small-fire placement as the main remaining challenges before predicted FRP can support operational aerosol forecasts.</span> <span class="abstract-toggle" data-id="2610.11022">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.11022v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.11022v1) · [:material-content-copy: BibTeX](bibtex/2610.11022.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=graph-neural-networks" data-tag="graph-neural-networks">Graph neural networks</a> <a class="md-tag" href="/explore/?t=coarse" data-tag="coarse">Coarse (≥1°)</a> <a class="md-tag" href="/explore/?t=daily" data-tag="daily">Daily</a>
    { .paper-tags }

-   #### Low-rank tensor structure of precipitation and its application to satellite-reference merging { #2610.11000 }

    *Ryan Solgi, Rohan Shankar, Hugo A. Loaiciga* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.11000">The intermittent and variable nature of precipitation makes its accurate estimation over extended domains difficult, yet its spatiotemporal structure suggests that a low-rank representation may be...</span><span class="abstract-full" id="full-2610.11000" hidden>The intermittent and variable nature of precipitation makes its accurate estimation over extended domains difficult, yet its spatiotemporal structure suggests that a low-rank representation may be possible. This work represents daily precipitation over the contiguous United States (CONUS) as spatiotemporal tensors and applies CANDECOMP/PARAFAC factorization, showing that preserving the native spatial and temporal modes yields more accurate reconstruction than factorizing independent daily fields or unfolded space--time matrices. Building on this finding, this work presents TMerge, a tensor-based framework that integrates satellite precipitation with sparse reference observations through shared low-rank spatial and temporal factors. TMerge was applied to correct the IMERG Final Run product with climate prediction center reference observations over CONUS. During 2019-2022, TMerge increased correlation from 0.53 to 0.85 and reduced root-mean-square error and mean absolute error by 48.2% and 29.3%, respectively. TMerge consistently outperformed linear bias correction, quantile mapping, and neural networks across seasons, precipitation-intensity regimes, and regions. Improvements were spatially coherent and largest in coastal regions where IMERG errors were greatest. These results demonstrate that low-rank tensor structure parsimoniously approximates the dominant spatiotemporal variability of precipitation and provides a practical mechanism for improving satellite estimates under limited reference observations over extended domains.</span> <span class="abstract-toggle" data-id="2610.11000">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.11000v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.11000v1) · [:material-content-copy: BibTeX](bibtex/2610.11000.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a> <a class="md-tag" href="/explore/?t=daily" data-tag="daily">Daily</a>
    { .paper-tags }

-   #### Strategic Governance of AI Models in Earth Science { #2610.10560 }

    *Makoto Kelp, Amirhossein Arzani, Patricia Castellanos, Paul Griffiths, Ivan Higuera-Mendieta et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.10560">AI foundation models pretrained on weather and climate data are increasingly fine-tuned to Earth science tasks well beyond weather forecasting. Their development and adoption are outpacing the...</span><span class="abstract-full" id="full-2610.10560" hidden>AI foundation models pretrained on weather and climate data are increasingly fine-tuned to Earth science tasks well beyond weather forecasting. Their development and adoption are outpacing the scientific community's ability to evaluate them. These models are judged almost entirely by benchmark skill metrics, which measure how closely a forecast reproduces a reference product but not whether a model represents the physical processes governing the system it predicts. Forecast skill and physical reliability are therefore distinct properties. The distinction is most consequential under the nonstationary conditions of a changing climate for which these models were never trained. We identify five priorities for the physical evaluation of AI models in Earth science from task-specific emulators to foundation models, spanning training data, fine-tuning, behavioral testing, mechanistic interpretability, and output validation. We recommend three activities for the coming decade: 1) open AI-ready evaluation datasets, 2) a shared reporting standard for physics-based evaluation, and 3) a dedicated research program on the safety of these models.</span> <span class="abstract-toggle" data-id="2610.10560">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.10560v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.10560v1) · [:material-content-copy: BibTeX](bibtex/2610.10560.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=foundation-models" data-tag="foundation-models">Foundation models</a> <a class="md-tag" href="/explore/?t=evaluation" data-tag="evaluation">Evaluation</a> <a class="md-tag" href="/explore/?t=interpretability" data-tag="interpretability">Interpretability</a>
    { .paper-tags }

-   #### SciExam for ENSO: Can AI Agents Build Climate Models? { #2610.10513 }

    *Yinling Zhang, Langchen Liu, Dongbin Xiu, Xueyan Zou, Xu Kuang, Mengdi Wang, Shilong Liu* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.10513">Language-model agents are increasingly asked to carry out open-ended scientific research, yet their results are usually graded against a known answer, a rubric, or a language-model reviewer, none of...</span><span class="abstract-full" id="full-2610.10513" hidden>Language-model agents are increasingly asked to carry out open-ended scientific research, yet their results are usually graded against a known answer, a rubric, or a language-model reviewer, none of which can tell whether a new scientific model is valid. The AI Science Exam for El Nino-Southern Oscillation (SciExam for ENSO) is a benchmark in which agents build low-order stochastic models of ENSO, the dominant mode of interannual climate variability, from real observations. Within a six-hour budget, agents process the observations, write their own diagnostics, which are then frozen, and develop a model using only these diagnostics as feedback. Hidden graders then test whether the model reproduces ENSO's statistics, recovers unobserved variables, and forecasts held-out years, and score a published model in the same way. Across twelve agent systems, six produce models that score higher than the published model, mainly through better reconstruction and forecasting. The simplified forms of the stronger models are each compatible with one of the two competing explanations of ENSO's warm-cold asymmetry, an open debate that the task never mentions. Controlled runs of the top system under varied information suggest that its scores do not come from recalling the dated observational record and that the information it receives shapes how it builds its model. SciExam for ENSO can thus evaluate agent research where no answer is known, and the results suggest that agents can already build competitive models whose structures bear on questions that scientists still debate.</span> <span class="abstract-toggle" data-id="2610.10513">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.10513v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.10513v1) · [:fontawesome-brands-github: Code](https://github.com/ylzhang2447/SciExam-ENSO-code) · [:material-content-copy: BibTeX](bibtex/2610.10513.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=subseasonal-to-seasonal" data-tag="subseasonal-to-seasonal">Subseasonal to seasonal</a>
    { .paper-tags }

-   #### Conditional Flow Matching for Generation of 3D Multi-variable Instantaneous Urban Microclimate Fields { #2610.10430 }

    *Peng Liu, Shaoxiang Qin, Theodore Potsis, Lili Ji, Dingyang Geng, Liangzhu Leon Wang* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.10430">Rapid and accurate prediction of urban wind and temperature fields is important for urban microclimate design and climate adaptation. Large-eddy simulation (LES) effectively resolves these...</span><span class="abstract-full" id="full-2610.10430" hidden>Rapid and accurate prediction of urban wind and temperature fields is important for urban microclimate design and climate adaptation. Large-eddy simulation (LES) effectively resolves these instantaneous fields, but its application is limited in iterative design of urban microclimate applications due to high computational cost. Existing regressive data-driven models offers quick outputs, but they produce only deterministic point predictions that inherently fail to represent turbulent stochasticity. This paper adopts a novel generative framework of Conditional Flow Matching (CFM) that uses building geometry and mean flow as guidance to generate plausible three-dimensional instantaneous velocity and temperature fields for urban microclimate in seconds. To overcome the GPU memory bottleneck of pixel space 3D generation, the model operates in parallel on overlapping pixel space through a shared-noise initialization that preserves high spatial continuity of flow structure across the entire domain. Against reference LES data, the CFM surrogate can rapidly and accurately restore the first-order statistics with Normalized Root Mean Square Error (NRMSE) of 2.99% for wind and 1.77% for temperature, second-order turbulence metrics with NRMSE of 7.17% for wind and 8.84% for temperature, turbulent kinetic energy with NRMSE of 7%, probability density function and vertical profiles in representative locations. Wind engineering application of local gust prediction demonstrate that the speed and accuracy of CFM, supporting the use of generative AI for making turbulence-aware resilient urban design and climate adaptation more computationally feasible.</span> <span class="abstract-toggle" data-id="2610.10430">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.10430v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.10430v1) · [:material-content-copy: BibTeX](bibtex/2610.10430.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a>
    { .paper-tags }

-   #### WxFM-XL: Adapting Univariate Foundation Models to Multi-Station Weather Forecasting { #2610.10057 }

    *Xiao Wang, Changjian Chen, Zhuo Tang, Rongwen Li, Hongwu Liu, Kenli Li* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.10057">With the rise of univariate time series foundation models (e.g., Sundial, Timer), initial efforts have been made to extend them to multivariate settings. However, these models mainly focus on...</span><span class="abstract-full" id="full-2610.10057" hidden>With the rise of univariate time series foundation models (e.g., Sundial, Timer), initial efforts have been made to extend them to multivariate settings. However, these models mainly focus on modeling correlations among variables. When they are applied to multi-station weather forecasting, two important factors are often overlooked: (1) the spatial information of stations, and (2) different error priors of different stations relative to the foundation model. In this paper, we propose WxFM-XL, a model for adapting univariate time series foundation models to multi-station weather forecasting. WxFM-XL introduces a cross-station error correlation prior graph to capture stationwise error priors with respect to the foundation model. Building on this, we further propose a dynamic fusion mechanism that adaptively integrates a spatial correlation graph with the error correlation prior graph. Experiments on multiple datasets demonstrate that our model outperforms state of the art baselines.</span> <span class="abstract-toggle" data-id="2610.10057">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.10057v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.10057v1) · [:material-content-copy: BibTeX](bibtex/2610.10057.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=foundation-models" data-tag="foundation-models">Foundation models</a> <a class="md-tag" href="/explore/?t=station-point" data-tag="station-point">Station / point</a>
    { .paper-tags }

-   #### Learning joint probabilistic weather forecasts from station observations alone { #2610.09898 }

    *Chaeyeon Yi, Yun Am Seo* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.09898">Assessing compound weather risks requires forecasts representing dependence between variables. CLARA (Calibrated Advection-Routing Attention) learns joint Gaussian predictive distributions of five...</span><span class="abstract-full" id="full-2610.09898" hidden>Assessing compound weather risks requires forecasts representing dependence between variables. CLARA (Calibrated Advection-Routing Attention) learns joint Gaussian predictive distributions of five surface variables from station observations alone, without numerical weather prediction or reanalysis; the approximately 28,000-parameter model supports CPU training and prediction. Across six multi-year folds on 96 stations, its lead-mean energy score is 4.9% lower than that of a learned comparator with matched temporal inputs (4.7% with a similar parameter count) and 11-65% lower than those of statistical baselines. Holding marginal variances fixed, removing learned correlations worsens joint negative log-likelihood by 1.0-2.8 nats per station. A covariance-scale estimator, proved consistent under stated assumptions, improves short-lead calibration but over-corrects at long leads. Synthetic interventions show an attention-bias coefficient alone does not measure forecast influence. Retrained in ten regions on six continents, CLARA outperforms persistence in all 60 multi-year region-lead comparisons and a similarly sized learned model in 57 of 60.</span> <span class="abstract-toggle" data-id="2610.09898">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.09898v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.09898v1) · [:material-content-copy: BibTeX](bibtex/2610.09898.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=station-point" data-tag="station-point">Station / point</a>
    { .paper-tags }

-   #### Generative and deterministic deep learning models comparison for fine-scale precipitation retrievals from infrared brightness temperature { #2610.09859 }

    *Matthieu Meignin, Cécile Mallet, Nicolas Viltard* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.09859">Accurate precipitation estimation at fine spatial scales is critical for hydrology, agriculture, and climate studies. Infrared brightness temperatures from geostationary satellites offer excellent...</span><span class="abstract-full" id="full-2610.09859" hidden>Accurate precipitation estimation at fine spatial scales is critical for hydrology, agriculture, and climate studies. Infrared brightness temperatures from geostationary satellites offer excellent temporal coverage over continental-scale domains. However, because these measurements primarily characterize cloud-top properties rather than precipitation processes near the surface, their correlation with rainfall intensity remains limited, making quantitative precipitation estimation challenging. In this study, we conduct a systematic inter-comparison of state-of-the-art deep learning models for high-resolution precipitation retrieval from Meteosat Second Generation infrared brightness temperatures over metropolitan France. These models include deterministic U-Nets, transformer-based architectures, conditional GANs, and diffusion models. We construct a curated dataset spanning 2008--2023, combining M{é}t{é}o-France radar mosaics as reference with multi-channel infrared observations, and design preprocessing and sampling strategies to address the heavy-tailed, intermittent nature of rainfall. Our results show that deterministic models provide robust mean estimates and excel in pixel-wise accuracy, but systematically underestimate extreme precipitation. In contrast, generative models better capture the full precipitation distribution, including rare and heavy rainfall events, producing more realistic spatial structures at the cost of reduced pixel-wise fidelity. These results highlight a trade-off between pixel-wise accuracy and precipitation variability, showing that generative approaches are advantageous for extreme-event detection and probabilistic applications. This work establishes a reproducible framework for evaluating infrared- based precipitation retrieval methods and provides guidance for designing models that balance precision, variability, and extreme-event representation.</span> <span class="abstract-toggle" data-id="2610.09859">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.09859v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.09859v1) · [:material-content-copy: BibTeX](bibtex/2610.09859.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=gans" data-tag="gans">GANs</a> <a class="md-tag" href="/explore/?t=transformers" data-tag="transformers">Transformers</a> <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=benchmarks-datasets" data-tag="benchmarks-datasets">Benchmarks & datasets</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a>
    { .paper-tags }

-   #### Artificial intelligence pathways from weather to climate { #2610.09770 }

    *Tom Beucler, J. David Neelin, Hui Su, Shivanshi Asthana, Chris Bretherton, Will Chapman et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.09770">Deep learning has made rapid advances in weather forecasting: autoregressive models trained on atmospheric reanalyses now rival dynamical models across nowcasting, medium-range, and...</span><span class="abstract-full" id="full-2610.09770" hidden>Deep learning has made rapid advances in weather forecasting: autoregressive models trained on atmospheric reanalyses now rival dynamical models across nowcasting, medium-range, and subseasonal-to-seasonal lead times, producing well-calibrated ensemble forecasts at reduced cost. We review these advances and consider their extension to climate horizons, where the challenge shifts from initial-condition skill to producing reliable statistical responses under altered forcings. AI-powered climate prediction systems must produce credible forced responses to drivers (e.g., greenhouse gases, land-use change) typically outside the observed record. We propose two minimum requirements for AI in climate modeling: (i) external forcing agents must enter explicitly enough to support interventions in which they vary independently; and (ii) robustness must be stress-tested in out-of-distribution regimes, including extremes and counterfactual trajectories. Using leading AI autoregressive emulators and hybrid physics-AI models, we identify development and coupling challenges. Comparing the reported throughput of these models with that of GPU-ported dynamical models highlights how AI can reduce time-to-solution by advancing only the target variables at the required resolution and using longer time steps, rather than integrating a full high-frequency, multivariate state. Diverse AI downscaling strategies can partially substitute for explicit fine-scale resolution, paving the way toward inexpensive local hazard assessment across prediction horizons.</span> <span class="abstract-toggle" data-id="2610.09770">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.09770v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.09770v1) · [:material-content-copy: BibTeX](bibtex/2610.09770.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=physics-ml-hybrid" data-tag="physics-ml-hybrid">Physics–ML hybrid</a>
    { .paper-tags }

-   #### SoftSEEPS improves ML-based precipitation forecasting { #2610.09752 }

    *Jost Arndt, Utku Isil, Noelia Otero, Rodrigo Almeida, Wojciech Samek, Jackie Ma* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.09752">In this paper we have developed a differentiable approximation of the well-known SEEPS score, which we name SoftSEEPS. This allows the training of a Machine Learning model to forecast precipitation...</span><span class="abstract-full" id="full-2610.09752" hidden>In this paper we have developed a differentiable approximation of the well-known SEEPS score, which we name SoftSEEPS. This allows the training of a Machine Learning model to forecast precipitation directly. We test SoftSEEPS on the IMERG dataset (0.1 degree resolution) by training a decoder for precipitation on the latent space of a pre-trained low-resolution forecasting model. Combining SoftSEEPS and RMSE in a joint objective is possible with marginal trade-offs in either metric.</span> <span class="abstract-toggle" data-id="2610.09752">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.09752v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.09752v1) · [:material-content-copy: BibTeX](bibtex/2610.09752.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a>
    { .paper-tags }

-   #### EC-EarthFlow: Probabilistic emulation of daily transient global climate model simulations with flow matching { #2610.09715 }

    *Kirien Whan, Nikolaj T. Mücke, Karin van der Wiel* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.09715">We introduce EC-EarthFlow, a generative flow matching model that emulates simulations from the physical climate model EC-Earth3. The model is trained on transient simulations from EC-Earth3...</span><span class="abstract-full" id="full-2610.09715" hidden>We introduce EC-EarthFlow, a generative flow matching model that emulates simulations from the physical climate model EC-Earth3. The model is trained on transient simulations from EC-Earth3 (1950-2166, SSP2-4.5) to predict the day ahead temperature field from the previous days temperature as well as annual mean temperature. Predictions are made auto-regressively with rollout periods of between a month and an extended season. Using only this variable of interest, we are able to reproduce the daily variability, spatial patterns, annual cycle and long-term trend from EC-Earth3 at a substantially lower computational cost than the physical model. We demonstrate that EC-EarthFlow is stable for long inference periods, and that it can learn the physical relationships as simulated in EC-Earth3.</span> <span class="abstract-toggle" data-id="2610.09715">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.09715v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.09715v1) · [:material-content-copy: BibTeX](bibtex/2610.09715.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a> <a class="md-tag" href="/explore/?t=daily" data-tag="daily">Daily</a>
    { .paper-tags }

-   #### Beyond the Doppler Dilemma: Improved Fast Weather Radar Unambiguous Doppler Velocity Spectrum Reconstruction from Sparse Aperiodic Sweeps { #2610.09172 }

    *Tworit Dash, S. A. K. Syed Mohamed, Oleg Krasnov, Alexander Yarovoy* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.09172">The problem of fast Doppler counter-aliasing for short-dwell weather-radar measurements is addressed. The proposed approach realizes a log-periodic slow-time design by changing only the waiting time...</span><span class="abstract-full" id="full-2610.09172" hidden>The problem of fast Doppler counter-aliasing for short-dwell weather-radar measurements is addressed. The proposed approach realizes a log-periodic slow-time design by changing only the waiting time between otherwise unchanged frequency-modulated continuous-wave (FMCW) chirps. The fast-time waveform, bandwidth, and range-processing chain are preserved while the acquisition changes the Doppler ambiguity structure. We formulate a novel moment-space ambiguity function for distributed weather spectra. It measures statistical ambiguity jointly in mean Doppler velocity and spectral width and expresses the competing sidelobes through their expected relative likelihood. A complex Gaussian process (CGP) maximum-likelihood estimator reconstructs the spectrum from the measured aperiodic samples. A deterministic basin-selection and optimization strategy accelerates the same CGP likelihood without defining a different statistical estimator. Real precipitation measurements establish central-branch stability over an extended search domain. Using 64 slow-time sweeps, we compare three acquisitions and their associated estimators. Periodic data are processed by the discrete Fourier transform (DFT) and parametric spectral estimation (PSE), log-periodic data by the nonuniform discrete Fourier transform (NUDFT) and CGP, and staggered data by velocity-difference and CGP estimators. The differences become more pronounced when the dwell is reduced to 32 sweeps. The results show that aperiodic time sampling enables extended-domain branch identification without degrading spectral-width estimation.</span> <span class="abstract-toggle" data-id="2610.09172">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.09172v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.09172v1) · [:material-content-copy: BibTeX](bibtex/2610.09172.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=classical-ml" data-tag="classical-ml">Classical ML</a>
    { .paper-tags }

-   #### Multi-model ocean oxygen fields predicted by conditional diffusion models { #2610.08523 }

    *Linus Vogt, Laure Zanna* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.08523">Dissolved oxygen is important for the ocean's ecosystems and biogeochemical cycles. Yet, Earth System Models (ESMs) vary in their simulations of the present-day and future ocean oxygen inventory. To...</span><span class="abstract-full" id="full-2610.08523" hidden>Dissolved oxygen is important for the ocean's ecosystems and biogeochemical cycles. Yet, Earth System Models (ESMs) vary in their simulations of the present-day and future ocean oxygen inventory. To narrow down the uncertainty in estimates of ocean oxygen content, we train a conditional generative diffusion model on outputs of a multi-model ESM ensemble to learn the conditional distribution of upper-ocean oxygen given physical input variables such as temperature and salinity. This generative model has considerable skill in the Atlantic and Southern Oceans, and can generate realistic oxygen samples under conditions not seen in the training data. We validate this model using observational datasets, and use it to generate oxygen fields for models without oxygen data using only temperature and salinity as conditioning inputs. This physics-conditioned extrapolation suggests that model biases in the tropical Pacific Oxygen Minimum Zone may be smaller than currently assumed when considering a larger set of physical ocean states. Our approach provides a complementary way to represent probabilistic multi-model climate distributions.</span> <span class="abstract-toggle" data-id="2610.08523">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.08523v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.08523v1) · [:material-content-copy: BibTeX](bibtex/2610.08523.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a>
    { .paper-tags }

-   #### Mechanistic Interpretability of Atmospheric Rivers in GraphCast { #2610.07583 }

    *Madelyn Mathai, Timothy B. Higgins, Kevin M. Grise, Chirag Agarwal, Antonios Mamalakis* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.07583">While AI weather models now rival operational forecasts, how they represent the atmosphere internally remains an open question: feature attribution reveals which input patterns matter, not what the...</span><span class="abstract-full" id="full-2610.07583" hidden>While AI weather models now rival operational forecasts, how they represent the atmosphere internally remains an open question: feature attribution reveals which input patterns matter, not what the model computes or how it combines information internally. We train sparse autoencoders (SAEs) on GraphCast to uncover its learned concepts, using atmospheric rivers as our phenomenon of focus. Both standard and Matryoshka SAEs show GraphCast computes atmospheric river intensity, measured by integrated vapor transport (IVT), as a stable internal variable, despite IVT being neither an input nor a target. In contrast to the unstructured concept retrieval of the standard SAE, the Matryoshka SAE orders concepts by importance and exposes their relations. Atmospheric river concepts persist across depth and direct interventions confirm causality. This method offers a way to find internal variables and determine which of them the model actually relies on, which is a prerequisite for asking whether those variables remain meaningful as the phenomenon changes under a warming climate.</span> <span class="abstract-toggle" data-id="2610.07583">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.07583v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.07583v1) · [:material-content-copy: BibTeX](bibtex/2610.07583.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=interpretability" data-tag="interpretability">Interpretability</a>
    { .paper-tags }

-   #### The interface of data assimilation and machine learning { #2610.07496 }

    *Eviatar Bach* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.07496">Data assimilation (DA) is the process of combining forecasts from a model with observations in order to optimally estimate the state of a system. This is critical for chaotic systems, such as the...</span><span class="abstract-full" id="full-2610.07496" hidden>Data assimilation (DA) is the process of combining forecasts from a model with observations in order to optimally estimate the state of a system. This is critical for chaotic systems, such as the atmosphere, since if observations are not continually assimilated the model will quickly lose skill. DA is routinely performed (usually every 6 hours) at operational forecasting centres around the world.   In this article we discuss the interface of machine learning (ML) and DA. This is still an emerging and quickly developing field, and this article tries to give an overview of some of the main topics and methods.</span> <span class="abstract-toggle" data-id="2610.07496">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.07496v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.07496v1) · [:material-content-copy: BibTeX](bibtex/2610.07496.bib){ .bibtex-link }
    { .paper-links }

-   #### Skillful Data-Driven Subseasonal Soil Moisture Forecasting: Prospects and Limits for Flash Drought Prediction { #2610.07060 }

    *Noelia Otero, Atahan Özer, Miguel-Ángel Fernández-Torres, Jackie Ma* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.07060">Despite substantial progress in short-to-medium-range weather forecasting, predicting high-impact events such as flash droughts remains a key challenge for both early warning operations and...</span><span class="abstract-full" id="full-2610.07060" hidden>Despite substantial progress in short-to-medium-range weather forecasting, predicting high-impact events such as flash droughts remains a key challenge for both early warning operations and physically-based subseasonal-to-seasonal (S2S) prediction systems. Here we demonstrate that, for S2S soil-moisture forecasting over Europe, forecast skill depends as much on how the prediction problem is formulated as on the forecasting model itself. Using a Vision Transformer-based architecture with dual-pathway temporal and spatial attention, we show that residual learning is essential to outperform persistence. This advantage is realized only when forecasting root-zone soil moisture in physical units rather than standardized anomalies, revealing that the target representation itself constrains predictability. A probabilistic extension via quantile-head fine-tuning further provides well-calibrated predictive distributions. Benchmarked against deep-learning and operational ECMWF S2S baselines over 2021-2022, our model achieves the highest deterministic and probabilistic skill at all lead times and reliably detects anomalously dry root-zone states (below the 20th percentile). Yet flash drought onset, defined by multi-pentad intensification criteria, remains a fundamental challenge shared across all current S2S systems. These findings advance data-driven S2S soil-moisture forecasting while highlighting the remaining challenge of predicting rapid drought development.</span> <span class="abstract-toggle" data-id="2610.07060">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.07060v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.07060v1) · [:material-content-copy: BibTeX](bibtex/2610.07060.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=extremes" data-tag="extremes">Extremes</a> <a class="md-tag" href="/explore/?t=subseasonal-to-seasonal" data-tag="subseasonal-to-seasonal">Subseasonal to seasonal</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a>
    { .paper-tags }

-   #### Xaurora: Generative Weather Forecasting with Denoising Stochastic Interpolants from a Foundation Model Prior { #2610.06509 }

    *Eliot Walt, Miltiadis Kofinas, Nikolaj Mücke, Efstratios Gavves, Dim Coumou* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.06509">Deep learning has revolutionised weather forecasting in recent years, especially through atmospheric foundation models, which offer competitive skill for a fraction of the computational costs of...</span><span class="abstract-full" id="full-2610.06509" hidden>Deep learning has revolutionised weather forecasting in recent years, especially through atmospheric foundation models, which offer competitive skill for a fraction of the computational costs of classic physics-based models. However, most existing foundation models are deterministic, limiting the generation of large ensembles for accurate uncertainty quantification, extreme weather risk assessment, and long-range weather forecasting. Furthermore, these models incur a large, often prohibitive, computational overhead to train from scratch. To address these shortcomings, we turn a pretrained deterministic prior model, namely the Aurora foundation model, into a generative ensemble-prediction model. To that end, we introduce a novel generative method, Denoising Stochastic Interpolants, combined with a replay buffer for Stochastic Differential Equation (SDE) rollout, enabling probabilistic training of SDE trajectories. Our stochastic foundation model, Xaurora, is finetuned from the small Aurora version, yet it approaches the state-of-the-art on global ensemble metrics and is competitive with the large version of Aurora. Our method is parameter and sample efficient, and generates skilful 15-day forecasts in 13 minutes. Our results demonstrate that deterministic foundation models can be efficiently extended into even stronger stochastic models.</span> <span class="abstract-toggle" data-id="2610.06509">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.06509v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.06509v1) · [:material-content-copy: BibTeX](bibtex/2610.06509.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=foundation-models" data-tag="foundation-models">Foundation models</a> <a class="md-tag" href="/explore/?t=sub-hourly" data-tag="sub-hourly">Sub-hourly</a>
    { .paper-tags }

-   #### FlexCast: Adaptive Weather Forecasting from Arbitrary Field Sets { #2610.05296 }

    *Yuang Zhang, Chen Hui, Weisi Lin, Haiqi Zhu, Xiulai Wang, Sun-Yuan Kung, Feng Jiang* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.05296">Most deep learning weather models assign a fixed set of variables and pressure levels to predefined channels, limiting transfer across atmospheric field configurations. This dependence on a fixed...</span><span class="abstract-full" id="full-2610.05296" hidden>Most deep learning weather models assign a fixed set of variables and pressure levels to predefined channels, limiting transfer across atmospheric field configurations. This dependence on a fixed field set limits the transferability of trained models across atmospheric field configurations. We propose FlexCast, a field-adaptive weather forecasting model that uses a single set of parameters to produce identity-aligned forecasts for variable-cardinality subsets drawn from a 69-field ERA5 registry. Specifically, a metadata-conditioned adapter the first encodes variable identity, pressure level, and field type and combines them with spatial features. Then, shared rank-16 projec?tions are modulated by metadata-dependent gates to produce field?specific features, while masked set fusion aggregates the available fields into a fixed-width representation. Subsequently, a multiscale U-Transformer processes the fused atmospheric features, while an identity-aware query decoder produces forecasts for the requested fields. Finally, FlexCast learns a standardized six-hour increment and applies it recursively to generate forecasts at longer lead times. Experiments on the 2020 ERA5 test set demonstrate that FlexCast operates across varying field configurations. Compatible cross-field context is associated with lower forecast errors, whereas mismatched context increases them.</span> <span class="abstract-toggle" data-id="2610.05296">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.05296v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.05296v1) · [:material-content-copy: BibTeX](bibtex/2610.05296.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=transformers" data-tag="transformers">Transformers</a>
    { .paper-tags }

-   #### ClimateBench v2.0: Probabilistic Climate Model Benchmarking { #2610.04558 }

    *Duncan Watson-Parris, Willa Tobin, Aytaç Paçal, Manuel Schlund, V. Balaji, Kevin Bowman et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.04558">We present ClimateBench v2, a standardized protocol for evaluating climate models on diagnostics expected to be informative for their skill in projecting mid-century regional temperature and...</span><span class="abstract-full" id="full-2610.04558" hidden>We present ClimateBench v2, a standardized protocol for evaluating climate models on diagnostics expected to be informative for their skill in projecting mid-century regional temperature and precipitation changes. The protocol is designed to evaluate any physics-based, data-driven, or hybrid climate model on equal footing using a common set of observational and out-of-distribution tests. We define three tiers of evaluation. Tier I establishes physical credibility through entry-ticket tests of energy conservation, coupled (co-)variability, and basic forced responses. Tier II scores models against post-2015 observations of surface temperature, precipitation, radiative fluxes, sea ice, and key modes of variability using fair CRPS as the primary probabilistic score, complemented by distributional and ensemble-consistency diagnostics. Tier III tests out-of-distribution generalization through paleoclimate simulations spanning the Last Interglacial, Last Glacial Maximum, and Mid-Holocene, and through perfect-model experiments in which data-driven models must predict the future climate of existing Earth system models from historical data alone. We reserve all observational data after 2015 for testing, and submissions must include multiple ensemble members to enable probabilistic evaluation. This reservation exploits a new opportunity provided by the decade of observations accumulated since the end of the CMIP6 historical experiment, which constitutes an out-of-sample record of forced climate change (and internal variability) for the current generation of models, and we quantify, in an idealized setting, the information it carries about mid-century warming. We provide the evaluation code, observational reference datasets, and perfect-model training data as an open benchmark to drive measurable progress in climate projection across all modeling approaches.</span> <span class="abstract-toggle" data-id="2610.04558">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.04558v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.04558v1) · [:material-content-copy: BibTeX](bibtex/2610.04558.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### S$^3$N: A Spherical Spiral Scanning Network for Weather Forecasting { #2610.04338 }

    *Fan Yan, Chen Hui, Weisi Lin, Haiqi Zhu, Feng Jiang, Sun-Yuan Kung, Wei Zhang* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.04338">Machine learning-based weather prediction (MLWP) has achieved strong performance in global weather forecasting. Recent Hierarchical Equal Area isoLatitude Pixelation (HEALPix)-based methods use the...</span><span class="abstract-full" id="full-2610.04338" hidden>Machine learning-based weather prediction (MLWP) has achieved strong performance in global weather forecasting. Recent Hierarchical Equal Area isoLatitude Pixelation (HEALPix)-based methods use the HEALPix (HP) grid to avoid area distortion near the poles of conventional latitude-longitude (LL) grids. However, existing HP-based approaches often use pointwise mapping methods and process HP pixels within separate base faces or local windows. Consequently, the mapping may introduce reconstruction errors and cross-face communication depends on handcrafted boundary handling or shifted windows. We propose the Spherical Spiral Scanning Network (S$^3$N) to address both limitations. First, L2Proj provides a bidirectional method for mapping atmospheric fields between the LL and HP grids through an $L^2$ projection of their continuous finite-element representations. Second, the Attention-Guided Quad-Spiral State-Space Scanning (AQSS) block uses cross-latitude attention to guide selective state-space updates along four global pole-to-pole spiral paths. This design enables continuous information propagation across HP base-face boundaries without additional boundary-processing mechanisms. Experiments show that S$^3$N achieves better results at 4-, 7-, and 10-day lead times, and exhibits slower error growth in long-range forecasting.</span> <span class="abstract-toggle" data-id="2610.04338">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.04338v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.04338v1) · [:material-content-copy: BibTeX](bibtex/2610.04338.bib){ .bibtex-link }
    { .paper-links }

</div>

