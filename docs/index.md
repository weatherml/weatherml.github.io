---
hide:
  - navigation
  - toc
title: weatherml
---

A collection of papers on AI for weather forecasting, climate modelling and atmospheric science.

<p class="page-meta" markdown="span">1639 papers · updated 2026-11-11 · <a href="feed.xml">:material-rss: RSS</a> · <a href="all_papers.bib" download>:material-download: BibTeX</a> · <a href="https://github.com/weatherml/weatherml.github.io/issues/new?template=suggest-paper.yml">:material-plus: Suggest a paper</a></p>

## Browse by Topic

<div class="grid cards topics" markdown>

-   [Global Models](papers/global-models/index.md) <span class="topic-count">387</span>
-   [Regional Models](papers/regional-models/index.md) <span class="topic-count">70</span>
-   [Nowcasting](papers/nowcasting/index.md) <span class="topic-count">109</span>
-   [Downscaling](papers/downscaling/index.md) <span class="topic-count">109</span>
-   [Post-processing](papers/post-processing/index.md) <span class="topic-count">27</span>
-   [Data Assimilation](papers/data-assimilation/index.md) <span class="topic-count">82</span>
-   [Climate Modeling](papers/climate-modeling/index.md) <span class="topic-count">315</span>
-   [Hydrology](papers/hydrology/index.md) <span class="topic-count">40</span>
-   [Ocean & Sea Ice](papers/ocean-sea-ice/index.md) <span class="topic-count">91</span>
-   [Air Quality & Composition](papers/air-quality-composition/index.md) <span class="topic-count">79</span>
-   [Remote Sensing](papers/remote-sensing/index.md) <span class="topic-count">108</span>
-   [Other](papers/other/index.md) <span class="topic-count">222</span>

</div>

## Recent Additions

<div class="grid cards" markdown data-search-exclude>

-   #### Discovering Equations for the Thermodynamics of South Asian Monsoon Precipitation { #2611.14401 }

    *Savannah L. Ferretti, Jane W. Baldwin, Michael S. Pritchard, Tom Beucler* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.14401">South Asian monsoon precipitation depends on the thermodynamic structure of the lower troposphere, but how its ingredients combine to shape precipitation remains unclear, partly because this...</span><span class="abstract-full" id="full-2611.14401" hidden>South Asian monsoon precipitation depends on the thermodynamic structure of the lower troposphere, but how its ingredients combine to shape precipitation remains unclear, partly because this dependence is nonlocal in the vertical. We use symbolic distillation, which combines neural networks and symbolic regression, to discover interpretable precipitation equations from 21 years of June-August reanalysis data. Neural networks first learn the vertical thermodynamic structure most relevant to precipitation, condensing each variable into a single interpretable feature not tied to a particular vertical grid. Then, symbolic regression derives closed-form equations from these features and the surface state. Our best equation explains 1.5 times as much precipitation variance as a widely used buoyancy-based diagnostic and recovers about 85% of neural network skill while using about a thousand times fewer parameters. Unlike the buoyancy diagnostic, which adds moisture and instability, the equations treat them as competing controls, with the heaviest precipitation occurring when the lower free troposphere is humid. Over ocean, sensible heating is associated with additional precipitation mainly when near-surface air is warm and moist, improving predicted precipitation and its diurnal cycle. Although the learned features were optimized only for precipitation prediction, the equations largely respond to each feature as expected from convective physics, consistent with the reanalysis. These responses reduce the dependence of monsoon precipitation on the entire vertical column to a few simple, testable constraints that could guide data-driven precipitation models to make them more physically consistent.</span> <span class="abstract-toggle" data-id="2611.14401">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.14401v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.14401v1) · [:material-content-copy: BibTeX](bibtex/2611.14401.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=interpretability" data-tag="interpretability">Interpretability</a>
    { .paper-tags }

-   #### PARDIG: Global Precipitation Estimates Learned from Sparse Satellite Observations { #2611.13941 }

    *Guy Shalev, Stephan Rasp, Boris Babenko, Andrew El-Kadi, Mohammed Alewi Hassen, Samier Merchant et al.* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.13941">Estimating precipitation at the global scale is vital for Earth system science and disaster response, yet observational coverage and measurement precision remain constrained. Existing solutions...</span><span class="abstract-full" id="full-2611.13941" hidden>Estimating precipitation at the global scale is vital for Earth system science and disaster response, yet observational coverage and measurement precision remain constrained. Existing solutions provide options that trade off desirable traits such as skill, latency, geographical coverage, temporal coverage, and spatial and temporal resolution. We present PARDIG, an artificial intelligence framework based on a spherical graph-mesh transformer architecture that produces near-real-time global precipitation estimates. PARDIG is trained on dense multi-modal inputs, including atmospheric state analyses and multi-spectral geostationary satellite imagery, to predict sparse, high-precision precipitation measurements from spaceborne radar. At inference, the trained model is applied densely across the globe at $0.1^\circ$ spatial and 15-minute temporal resolution, producing complete near-real-time precipitation fields from 2016 to the present, alongside a historical variant extending the record back to 1999. Evaluation against three independent radar networks (over the US, Europe, and Japan) and against over 36,000 surface stations worldwide shows that PARDIG consistently outperforms IMERG V07 (Final and Early), MSWEP V3, ERA5, and HRES across precipitation thresholds and regions, with the largest improvements at extreme intensities. Case studies of extreme precipitation events demonstrate that PARDIG successfully captures their spatial structure and timing. The improvements also extend to forecasting: WeatherNext 3, a global weather model, produces more skillful precipitation forecasts when trained to predict PARDIG. Both the historical and near-real-time PARDIG datasets are openly available under a CC BY 4.0 license.</span> <span class="abstract-toggle" data-id="2611.13941">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.13941v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.13941v1) · [:material-content-copy: BibTeX](bibtex/2611.13941.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=transformers" data-tag="transformers">Transformers</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a> <a class="md-tag" href="/explore/?t=sub-hourly" data-tag="sub-hourly">Sub-hourly</a>
    { .paper-tags }

-   #### Reconstructing Boundary Layer Wind Profiles with Machine Learning: A Comparative Study on IGRA Radiosonde Data { #2611.13359 }

    *Fedor Buzaev, Dmitry Efremenko, Daria Pugacheva, Fedor Ratnikov, Egor Bugaev, Vladimir Gordin et al.* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.13359">Wind profiles in the atmospheric boundary layer (ABL) drive short-range forecasting and wind-energy siting, yet sparse radiosonde coverage forces practitioners to interpolate them from a handful of...</span><span class="abstract-full" id="full-2611.13359" hidden>Wind profiles in the atmospheric boundary layer (ABL) drive short-range forecasting and wind-energy siting, yet sparse radiosonde coverage forces practitioners to interpolate them from a handful of physical predictors. We benchmark modern tabular learners against analytical parameterizations on 158,097 IGRA soundings. CatBoost, TabM, and FT-Transformer reduce RMSE to 1.3-1.6 m/s, against approximately 3 m/s for power-law and Monin-Obukhov fits; once the predictor set is fixed the three architectures differ by less than 0.05 m/s, so accuracy is driven by predictors rather than architecture. A cohort analysis of within-bin profile dispersion places an empirical floor of approximately 1.77 m/s on any point-wise predictor that uses standard radiosonde features, consistent with prior eddy-viscosity inversions. A one-dimensional Fourier Neural Operator applied as a post-hoc corrector to frozen tabular predictions reduces near-surface RMSE on U.S. stations from 1.5 to 1.2 m/s, while leaving upper-layer error unchanged. Further gains should come from new predictors rather than larger models.</span> <span class="abstract-toggle" data-id="2611.13359">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.13359v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.13359v1) · [:material-content-copy: BibTeX](bibtex/2611.13359.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=neural-operators" data-tag="neural-operators">Neural operators</a>
    { .paper-tags }

-   #### Direct Multispectral Satellite Nowcasting of Gridded Surface Solar Radiation { #2611.12999 }

    *Tianyu Xie, Zhaoliang Zeng, Zhehao Liang, Ke Gui, Xutao Zhang, Yaqiang Wang, Dabo Guan, Huizheng Che* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.12999">Surface shortwave radiation (SWR) can change within minutes as clouds evolve, but operational forecasts often trade spatial detail and update frequency for forecast range. We present a...</span><span class="abstract-full" id="full-2611.12999" hidden>Surface shortwave radiation (SWR) can change within minutes as clouds evolve, but operational forecasts often trade spatial detail and update frequency for forecast range. We present a numerical-weather-prediction-independent, observation-to-forecast framework that converts sequences of Himawari-8 Level-1 multispectral imagery and solar geometry directly into 24 gridded SWR fields at 10-min intervals for the next 4 h. A controlled six-model experiment separates the effects of satellite information from those of temporal self-attention. In the 0-1 h window, the full-multispectral attention model achieves a mean absolute error of 37.9 W/m2 and a root mean square error of 63.7 W/m2, reductions of 21.0% and 19.5% relative to a historical-SWR baseline. Improvements extend across nine subregions and include weaker spatial bias and better structural consistency. Temporal attention does not improve forecasts from historical SWR alone, but becomes effective when supplied with multispectral cloud and thermodynamic signals. Direct multispectral observations therefore provide the precursor information required for high-frequency, spatially resolved SWR nowcasting.</span> <span class="abstract-toggle" data-id="2611.12999">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.12999v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.12999v1) · [:material-content-copy: BibTeX](bibtex/2611.12999.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=transformers" data-tag="transformers">Transformers</a> <a class="md-tag" href="/explore/?t=sub-hourly" data-tag="sub-hourly">Sub-hourly</a>
    { .paper-tags }

-   #### MarsNet: Learning the Martian Atmosphere from Observations Alone { #2611.12787 }

    *Sebastian Mengwall, J. Michael Battalio* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.12787">We present MarsNet, a machine-learned model of the Martian atmosphere trained only on observations. Existing reanalyses assimilate spacecraft retrievals into general circulation models (GCMs), so...</span><span class="abstract-full" id="full-2611.12787" hidden>We present MarsNet, a machine-learned model of the Martian atmosphere trained only on observations. Existing reanalyses assimilate spacecraft retrievals into general circulation models (GCMs), so where observations are sparse their fields largely reflect the model. We therefore use extant reanalyses only for evaluation. MarsNet alternates a two-hour forecast with an update that assimilates the observations of each window, and both steps are learned jointly. Because the Martian record is too sparse for an unrestricted network, learning is confined within a fixed physical structure: the state is in physical units, the forecast uses physical tendencies of fixed form and learned strength, and the update is a bounded Kalman update. MarsNet is trained on Mars Years (MY) 24--31 and evaluated on independent observations. On three seasons of MGS radio-occultation temperatures assimilated by no dataset, MarsNet matches or improves on the three reanalyses in the lowest three scale heights, with a near-surface root-mean-square error (RMSE) of 4.2 K against 4.4--5.4 K, but is less accurate aloft. The model's surface pressures are the most accurate of the four products (RMSE 10 Pa against Curiosity in MY32, compared with 20--46 Pa). On withheld Mars Climate Sounder profiles of MY32, its temperatures beat an observed climatology and persistence, including in ten-sol forecasts without new observations, and its stated uncertainty is close to calibrated. Much of the temperature skill is present before training and appears to come from the structure and tables fitted to the observations, while training mainly improves the surface pressures, at some cost to dust. The model cycles stably through nine Mars years, and the influence of each observation can be traced. Its speed makes experiments on the observing record practical, and it offers an independent reference for GCMs and reanalyses.</span> <span class="abstract-toggle" data-id="2611.12787">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.12787v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.12787v1) · [:material-content-copy: BibTeX](bibtex/2611.12787.bib){ .bibtex-link }
    { .paper-links }

-   #### One2N: Recursive refinement breaks the resolution barrier for Earth-system AI { #2611.12249 }

    *Yanyan Tan, Xiangjun Tian, Can Wang, Hongqin Zhang, Yinhai Luo, Min Zhao, Zhe Jin, Xingchao Yang et al.* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.12249">Earth-system AI models approach kilometre-scale resolution, but halving grid spacing quadruples the grid-point count and end-to-end training cost. We introduce One2N, a recursive incremental...</span><span class="abstract-full" id="full-2611.12249" hidden>Earth-system AI models approach kilometre-scale resolution, but halving grid spacing quadruples the grid-point count and end-to-end training cost. We introduce One2N, a recursive incremental framework extending a coarse model through $N$ levels of cross-resolution residual correction. At each level the parent is frozen and excluded from backpropagation, so its activations and optimizer states are never retained; only a tier-specific module is trained. A single refinement level on the NeuralGCM-0.7$^\circ$ baseline yields the two-level One2N-0.07$^\circ$, trained on one 80 GB GPU. Against HRES-0.07$^\circ$ and ERA5-0.25$^\circ$ analyses it matches its parent's tropospheric skill while reducing upper-stratospheric errors and resolving finer structure. Tropical-cyclone track errors nearly match the parent's and are lower than Pangu-Weather-0.25$^\circ$ at medium range; RMSE is lower than that model's on 93.46% of common targets. One2N thus turns an existing forecasting system, or an inexpensively trained coarse baseline, into a reusable foundation for high-resolution development, breaking the resolution barrier.</span> <span class="abstract-toggle" data-id="2611.12249">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.12249v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.12249v1) · [:material-content-copy: BibTeX](bibtex/2611.12249.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=km-scale" data-tag="km-scale">Km-scale</a>
    { .paper-tags }

-   #### Perils and opportunities of AI for public climate services { #2611.12106 }

    *Francesca Larosa, Sandro Calmanti, Matteo De Felice, Marcello Petitta* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.12106">The rapid integration of artificial intelligence (AI), particularly generative and machine-learning systems, in climate services development and provision is transforming how climate information is...</span><span class="abstract-full" id="full-2611.12106" hidden>The rapid integration of artificial intelligence (AI), particularly generative and machine-learning systems, in climate services development and provision is transforming how climate information is produced, accessed, and interpreted. AI-enabled climate services offer significant opportunities for process automation, optimisation, tailoring, and the translation of complex climate data into actionable knowledge for diverse users. However, the growing reliance on privately controlled algorithms, data infrastructures, and computing facilities raises critical concerns related to governance, transparency, equity and economic sustainability, among others. While the technical architecture of climate services increasingly relies on advanced machine learning and large-scale climate models, their legitimacy as public services may be weakened in the absence of arrangements that prioritise public value and long-lasting societal benefit over profit maximisation. The tension between the public mandate of climate services and the private nature of AI infrastructure challenges traditional notions of openness. Using a consensus-based elicitation method coupled with the PESTLE framework, we analyse the political, economic, social, technological, legal, and environmental risks and opportunities shaping the co-production value chain of AI-enabled climate services. We draw policy-relevant conclusions for data, model and infrastructure supporting the next generation of upstream and downstream climate services as critical applications for society. Our work presents potential and plausible challenges and benefits for a pan-European, decentralised public climate service built on sovereign AI infrastructure and open governance principles.</span> <span class="abstract-toggle" data-id="2611.12106">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.12106v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.12106v1) · [:material-content-copy: BibTeX](bibtex/2611.12106.bib){ .bibtex-link }
    { .paper-links }

-   #### Ensemble Covariance Propagation with Machine Learning Weather Prediction Models in a Realistic Data Assimilation System { #2611.11918 }

    *Simon Toedtli, David Ahijevych, Wei Wang, Chris Snyder* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.11918">Integrating machine learning weather prediction (MLWP) models into classical data assimilation (DA) workflows requires that they accurately evolve ensemble covariances. We evaluate this capability in...</span><span class="abstract-full" id="full-2611.11918" hidden>Integrating machine learning weather prediction (MLWP) models into classical data assimilation (DA) workflows requires that they accurately evolve ensemble covariances. We evaluate this capability in a two-way coupled DA system, where an ensemble variational (EnVar) component evolves the mean and an ensemble Kalman filter (EnKF) evolves the covariances. The system is cycled over one month and assimilates a wide range of real-world observations, including microwave radiances. Using the MLWP model GraphCast for the EnKF and the physics-based MPAS model for the EnVar yields ensemble spread and covariances comparable to a reference configuration in which MPAS evolves both components. The resulting EnVar mean achieves consistently lower RMSE against GFS analyses, except near the model top, where spurious correlations degrade performance. These results demonstrate that MLWP models can propagate covariance information in a realistic DA cycling system and provide a proof of concept for their integration into future operational ensemble systems.</span> <span class="abstract-toggle" data-id="2611.11918">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.11918v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.11918v1) · [:material-content-copy: BibTeX](bibtex/2611.11918.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### PROLA: Principal-Orthogonal Low-rank Adaptation for Predictive Spatiotemporal Weather Downscaling { #2611.11318 }

    *Minseo Yoon, Minseong Bae, Sojin Lee, Hyunwoo J. Kim* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.11318">High-resolution weather prediction is important for resolving local atmospheric patterns missed by coarse global forecasts. Large pretrained weather backbones have improved global forecasting, but...</span><span class="abstract-full" id="full-2611.11318" hidden>High-resolution weather prediction is important for resolving local atmospheric patterns missed by coarse global forecasts. Large pretrained weather backbones have improved global forecasting, but adapting them to produce future high-resolution fields from recent coarse states remains challenging and costly. We formulate this setting as Predictive Spatiotemporal Weather Downscaling (PSWD), a task that jointly refines spatial and temporal resolution while predicting future high-resolution trajectories. To this end, we propose a framework that decomposes prediction into spatial and temporal stages using a shared pretrained backbone, fixed numerical scaffolds, and lightweight residual heads. For backbone adaptation, we introduce PROLA (PRincipal-Orthogonal Low-rank Adaptation), which allocates a fixed low-rank budget between pretrained weights' principal directions and their orthogonal complement. PROLA further rescales rank-1 optimizer updates using gradient signals, improving adaptation without increasing trainable rank. Experiments in the PSWD setting show that PROLA outperforms representative low-rank adaptation baselines. Further experiments across diverse datasets and tasks support the broader applicability of PROLA, showing that its benefits extend beyond the primary weather downscaling setting.</span> <span class="abstract-toggle" data-id="2611.11318">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.11318v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.11318v1) · [:material-content-copy: BibTeX](bibtex/2611.11318.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a>
    { .paper-tags }

-   #### Analysis increments for model bias diagnosis and correction: Theoretical foundations { #2611.11162 }

    *Eviatar Bach* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.11162">A common method for diagnosing and correcting model bias in weather and climate prediction is based on analysis increments: the difference between the analysis and the forecast at each data...</span><span class="abstract-full" id="full-2611.11162" hidden>A common method for diagnosing and correcting model bias in weather and climate prediction is based on analysis increments: the difference between the analysis and the forecast at each data assimilation cycle. Their time average is used to estimate and correct the bias, and machine learning is increasingly used to map the forecast state to the increments, giving a state-dependent correction. Despite its widespread use, the method lacks theoretical foundations.   We first analyse a Kalman filter whose forecast model has a constant bias. We show what the averaged increment measures, and prove that iterating the correction removes all of the bias seen through the observations; for linear dynamics, no method can identify the rest.   We then extend the analysis to state-dependent biases. For linear dynamics, each increment records the model error over the filter's memory, as seen through the observations. We use this to determine what the increments measure when averaged over a long period, averaged over periods with similar slowly varying conditions, and used without averaging, as in machine-learned corrections.   Numerical experiments confirm the theory for a linear model and suggest that its main conclusions carry over to the nonlinear Lorenz '96 model, including for a correction learned by a neural network.</span> <span class="abstract-toggle" data-id="2611.11162">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.11162v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.11162v1) · [:material-content-copy: BibTeX](bibtex/2611.11162.bib){ .bibtex-link }
    { .paper-links }

-   #### Rescene-ESM: an Earth system model built around a frozen AI weather model { #2611.11066 }

    *Minjong Cheon, Kaoru Kakinuma* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.11066">Earth system models underpin climate projections, but their cost limits how many scenarios, ensemble members and impact pathways can be explored. Machine-learning weather models forecast as...</span><span class="abstract-full" id="full-2611.11066" hidden>Earth system models underpin climate projections, but their cost limits how many scenarios, ensemble members and impact pathways can be explored. Machine-learning weather models forecast as skillfully as physics-based models at a fraction of the cost, yet coupled to an ocean they have produced weak El Nino variability or have learned it from climate-model output, and none carries the climate through rivers to the people exposed to it. Here we present Rescene-ESM, an Earth system model whose coupler, YAHO, yokes a frozen 20.8-million-parameter AI weather model to physically formulated ocean, sea-ice, land, river, carbon and energy components constrained by observations. Adding the couplings one at a time shows what each contributes: over a climatological ocean the AI atmosphere varies from year to year by a third of the observed amount over the oceans, and an interactive mixed layer, an equatorial wave ocean, the residual of the observed sea surface temperature (SST) equation as stochastic forcing and the land's moisture flux raise this step by step and bring El Nino and its teleconnections, leaving the mean climate almost unchanged. The full model simulates about 1,000 years per day on one graphics processing unit, holds the observed climate for a thousand years without drift, reproduces the observed 1979-2022 global warming, carries forced change with a physically based structure around the frozen network (transient climate response 1.60 K), routes runoff to 212 river gauges with a median seasonal-cycle correlation of 0.79 and, driven by emissions, separates the heatwave exposure of an SSP1 and an SSP3 world by a factor of 4.7 in 2081-2100, three-quarters of it through the warmer climate.</span> <span class="abstract-toggle" data-id="2611.11066">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.11066v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.11066v1) · [:material-content-copy: BibTeX](bibtex/2611.11066.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=subseasonal-to-seasonal" data-tag="subseasonal-to-seasonal">Subseasonal to seasonal</a>
    { .paper-tags }

-   #### Guiding future global temperature distributions with simulations and observations { #2611.10882 }

    *Francesco Immorlano, Elijah Tavares, Felix Draxler, Juan Nathaniel, Violette Launeau et al.* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.10882">Assessing climate risks requires projections of daily temperature distributions, including rare and spatially widespread extremes. Climate simulations are costly and biased, while correcting them...</span><span class="abstract-full" id="full-2611.10882" hidden>Assessing climate risks requires projections of daily temperature distributions, including rare and spatially widespread extremes. Climate simulations are costly and biased, while correcting them toward observations risks suppressing inter-model differences. We train a single machine-learning emulator, a conditional diffusion model, on seventeen CMIP6 models and ERA5 reanalysis to generate spatially coherent daily temperature fields conditioned on model identity, date, and CO2-equivalent concentration. At sampling time, diffusion guidance reduces model biases by steering each emulated climate-model distribution toward observations, with its strength selected through perfect-model experiments using withheld simulated futures. Observational influence decreases with projection horizon and forcing. Across seventeen synthetic targets, guidance more than halves median biases in the 1st and 99th percentiles of daily temperature, improves probabilistic scores and reliability, and reduces median annual global-mean temperature RMSE by 32-57% across three emission scenarios. In a detailed benchmark, guidance also outperforms established bias corrections on daily-field probabilistic scores. Efficient sampling generates large ensembles without further climate simulations. For example, our framework corrects CMIP6's roughly threefold historical overestimate of extreme-heat coverage, yet projects that concurrent extremes will affect nearly half of global land on a typical day in the 2080s under SSP3-7.0.</span> <span class="abstract-toggle" data-id="2611.10882">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.10882v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.10882v1) · [:material-content-copy: BibTeX](bibtex/2611.10882.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=daily" data-tag="daily">Daily</a>
    { .paper-tags }

-   #### SWXAI: A Unified Multimodal Explainability Library for Space Weather Forecasting and Heliophysics { #2611.10861 }

    *Benedict Antonio Mervyn, Anli Ji, Manolis K. Georgoulis, Berkay Aydin* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.10861">Space weather forecasting and heliophysics increasingly rely on machine learning models that process heterogeneous inputs such as multi-viewpoint multi-wavelength solar images and rasters,...</span><span class="abstract-full" id="full-2611.10861" hidden>Space weather forecasting and heliophysics increasingly rely on machine learning models that process heterogeneous inputs such as multi-viewpoint multi-wavelength solar images and rasters, multivariate time series, spatial annotations, and tabular event-level data. However, explainability workflows remain fragmented across libraries, model interfaces, and output formats, limiting reuse and reproducibility. We present SWXAI, an open-source library that unifies explainable AI (XAI) workflows for image, time-series, and tabular data in space weather forecasting and heliophysics. SWXAI provides a common, extensible interface across commonly used vision, time-series, and tabular learning models, with validation, adaptation, and persistent explanation objects. It also includes domain-grounded examples, reproducible notebooks, tests, and documentation, enabling users to inspect outputs, swap models or explainers, and extend workflows to new heliophysics datasets. SWXAI reduces modality-specific engineering overhead and supports comparable, reusable explanations across heterogeneous scientific machine learning pipelines. A benchmark against direct backend implementations shows that SWXAI substantially reduces the code users write to produce an explanation, with no practical runtime cost.</span> <span class="abstract-toggle" data-id="2611.10861">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.10861v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.10861v1) · [:material-content-copy: BibTeX](bibtex/2611.10861.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=interpretability" data-tag="interpretability">Interpretability</a>
    { .paper-tags }

-   #### RainFlow: Flow Matching for Infrared-Only Quantitative Precipitation Estimation { #2611.09663 }

    *Matthieu Meignin, Cécile Mallet, Nicolas Viltard* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.09663">Geostationary infrared imagers observe precipitating systems every 10 to 15 minutes over large areas, but they sense cloud tops rather than the rain beneath. Their brightness temperatures constrain...</span><span class="abstract-full" id="full-2611.09663" hidden>Geostationary infrared imagers observe precipitating systems every 10 to 15 minutes over large areas, but they sense cloud tops rather than the rain beneath. Their brightness temperatures constrain surface rain only weakly, so retrievals trained by regression return a smooth estimate that misses heavy rain and carries no measure of uncertainty. We treat infrared-only retrieval as conditional generation and introduce RainFlow, a flow-matching model that generates ensembles of rain fields from MSG/SEVIRI infrared observations, in a direct and a residual variant. Evaluated against radar over metropolitan France for a full year, the generated members are realistic rain fields, with the observed dry areas, heavy rain and spatial structure, and the ensemble mean is as accurate as a deterministic baseline. Used as probabilities, the ensembles locate heavy rain at scales where the baseline has no skill, and their spread captures most of the retrieval error, with only slight underdispersion. Trained with a time-sampling density that favours the noise end of the generative trajectory, the two variants reach similar skill. The direct variant depends strongly on this choice of density, whereas the residual variant is nearly insensitive to it. Evaluating the generated rain as a function of brightness temperature shows that the ensembles reproduce how rain depends on it over most of the range. It also isolates the remaining gap, an underestimate of both the amount and the variability of rain under the coldest cloud tops, which scores computed over all pixels do not reveal.</span> <span class="abstract-toggle" data-id="2611.09663">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.09663v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.09663v1) · [:material-content-copy: BibTeX](bibtex/2611.09663.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=sub-hourly" data-tag="sub-hourly">Sub-hourly</a>
    { .paper-tags }

-   #### Half-precision inference of AI weather prediction models: negligible loss of skill at a fraction of the cost { #2611.09573 }

    *Stefan Siegert* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.09573">AI weather prediction models are released with 32-bit (fp32) weights and run in fp32 by default, although current GPUs compute much faster in 16-bit formats. We run five publicly available models --...</span><span class="abstract-full" id="full-2611.09573" hidden>AI weather prediction models are released with 32-bit (fp32) weights and run in fp32 by default, although current GPUs compute much faster in 16-bit formats. We run five publicly available models -- Aurora, Pangu-Weather, FourCastNet, SFNO and FastNet -- on a single 16 GB GPU from 52 initial dates in 2024, in fp32 and with their networks in half precision (fp16), and verify the forecasts against ERA5. The physical state must stay in fp32: converting the whole model rounds it in physical units, which in fp16 overflows and in bf16 discards meaningful information. With the state, its normalisation and denormalisation in fp32, running the network in fp16 changes the global RMSE by at most 1.4%, and by about 0.5% or less for four of the five models, equivalent to less than one hour of lead time up to day 8 and at most two hours at day 10. The differences show no systematic pattern by region, scale or for tropical and extratropical storms, apart from a small tropical penalty in FourCastNet, and fp16 leaves the CRPS, spread and calibration of Aurora's ensemble almost unchanged. No value overflowed fp16's numeric range (+/-65,504); the largest stay 38 to 544 times below it. A 10-day forecast in fp16 is 1.3-3.5 times faster than in fp32 and uses 1.4-3.7 times less energy. Sensitivity experiments show that rules of thumb from other applications of reduced precision carry over only partly: the first and last layers dominate for some models but not for others, and no type of layer is sensitive on its own. For running AI weather models on a single GPU, fp16 with the state in fp32 is practically lossless at a fraction of the cost. bf16, often preferred for its wider range, is as fast but less accurate than fp16 for every model, and simple 8-bit quantisation costs far more skill.</span> <span class="abstract-toggle" data-id="2611.09573">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.09573v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.09573v1) · [:fontawesome-brands-github: Code](https://github.com/sieste/ai-nwp-fp16) · [:material-content-copy: BibTeX](bibtex/2611.09573.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=neural-operators" data-tag="neural-operators">Neural operators</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=efficiency" data-tag="efficiency">Efficiency</a>
    { .paper-tags }

-   #### Evaluation of a Multi-Link Machine Learning Approach for Rainfall Detection Using Satellite Microwave Links { #2611.09156 }

    *Mostafa Ftouni, Margherita Spalla, Matteo Colli, Andrea Randazzo, Christian Gianoglio* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.09156">The impacts of climate change have increased the need for reliable real-time rainfall monitoring systems. This study investigates rainfall detection using a network of opportunistic sensors based on...</span><span class="abstract-full" id="full-2611.09156" hidden>The impacts of climate change have increased the need for reliable real-time rainfall monitoring systems. This study investigates rainfall detection using a network of opportunistic sensors based on satellite microwave links (SMLs) combined with machine learning (ML) classifiers. Two models, a random forest (RF) and a convolutional neural network (CNN), were trained on aggregated SML data, including information from neighboring links. The results demonstrate that leveraging information from multiple links substantially enhances the detection of both wet and dry periods compared to single-link approaches. Moreover, the RF model consistently provides a more stable and reliable trade-off than CNN, improving rainfall detection and reducing misclassifications overall, with particularly strong performance during harsh weather conditions. The study spanned up to 32 rainy days between 2021 and 2024.</span> <span class="abstract-toggle" data-id="2611.09156">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.09156v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.09156v1) · [:material-content-copy: BibTeX](bibtex/2611.09156.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a> <a class="md-tag" href="/explore/?t=classical-ml" data-tag="classical-ml">Classical ML</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a>
    { .paper-tags }

-   #### Nowcasting of Tropical Cyclone Wind Fields and Intensity from Geostationary Imagery { #2611.08255 }

    *Simon Donike, Pritthijit Nath, Cristina Radin, Arthur Avenas, Ruben Cartuyvels, Emiliano Diaz et al.* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.08255">High-resolution tropical cyclone surface wind speeds derived from synthetic aperture radar (SAR) observations can support storm structure and intensity modelling, but remain spatially and temporally...</span><span class="abstract-full" id="full-2611.08255" hidden>High-resolution tropical cyclone surface wind speeds derived from synthetic aperture radar (SAR) observations can support storm structure and intensity modelling, but remain spatially and temporally sparse and unavailable near-real-time. Geostationary satellites capture cloud tops continuously in space and time, but do not directly resolve surface winds. Here, we present simple neural networks to predict 2D wind fields and storm-scale intensity metrics from geostationary imagery. Through a series of ablation studies, we find that joint prediction of 2D fields and scalar metrics from the model latent space improves the retrieval of the (radius of) maximum wind, especially for rapid intensification events.</span> <span class="abstract-toggle" data-id="2611.08255">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.08255v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.08255v1) · [:material-content-copy: BibTeX](bibtex/2611.08255.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a>
    { .paper-tags }

-   #### Detecting Weather Fronts from Satellite Observations: What Does Reanalysis Add? { #2611.08148 }

    *Ioannis Dravilas, Miruna Stoicescu, Roope Tervo, Carlos Horn, Oriol Hinojo Comellas, Sina Montazeri et al.* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.08148">Weather fronts are transition regions between air masses with contrasting thermodynamic properties and are associated with mid-latitude precipitation and severe weather. We investigate machine...</span><span class="abstract-full" id="full-2611.08148" hidden>Weather fronts are transition regions between air masses with contrasting thermodynamic properties and are associated with mid-latitude precipitation and severe weather. We investigate machine learning (ML) methods for detecting weather fronts and quantify the contribution of reanalysis data to performance. We compare two configurations of the same ML model for pixel-wise frontal classification over the North Atlantic and Europe: (1) using only geostationary infrared satellite observations, and (2) additionally using 28 ERA5 reanalysis fields. The satellite-only model achieves a neighbourhood critical success index at a 150 km tolerance (CSI@150) of $0.402$, which lies within the range of agreement between analyses from different meteorological services. The satellite$+$ERA5 model achieves $0.575$, an improvement of $0.173$. Our analysis shows that low-level winds are the least predictable from the satellite representation ($R^2 = 0.23$, compared with $0.88$ for temperature), while removing winds or moisture causes the largest decreases in detection performance.</span> <span class="abstract-toggle" data-id="2611.08148">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.08148v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.08148v1) · [:material-content-copy: BibTeX](bibtex/2611.08148.bib){ .bibtex-link }
    { .paper-links }

-   #### Estimating Vertical Ocean Velocity from Horizontal Currents: Linear and Depth-Conditional Neural-Network Approaches on a High-Resolution Regional Ocean Model { #2611.07969 }

    *Francesco M. Ventrella, Cristóbal López, Emilio Hernandez-Garcia* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.07969">The vertical component of the ocean velocity, w, is several orders of magnitude smaller than the horizontal components u and v, yet it controls vertical tracer transport, upwelling, and mixed-layer...</span><span class="abstract-full" id="full-2611.07969" hidden>The vertical component of the ocean velocity, w, is several orders of magnitude smaller than the horizontal components u and v, yet it controls vertical tracer transport, upwelling, and mixed-layer ventilation. Because w is extremely difficult to measure directly and is normally obtained only diagnostically from the continuity equation, predicting it from the horizontal velocity field is a problem of both practical and physical interest. We address this problem with two data-driven methods, a physically-motivated linear baseline and a convolutional neural network conditioned on depth through Feature-wise Linear Modulation (FiLM). Both methods are applied to a high-resolution regional simulation of the Coastal and Regional Ocean Community model over the Western Mediterranean, using a sub-region of the south-west Mediterranean Sea as a training region and two independent, out-of-sample regions for validation: the Alboran Sea, with a distinct dynamical regime driven by Atlantic inflow through the Strait of Gibraltar, and a transition region between Almería and Cartagena, between the Alboran Sea and the open western Mediterranean. Once the FiLM network is regularized enough to avoid overfitting to a single training domain, we find that its comparison with the physically interpretable linear baseline is domain- and depth-dependent rather than uniform: near the reference depth, essentially all of the achievable out-of-sample performance on both validation domains is already captured by the linear model, but at the deepest levels tested the FiLM network outperforms the linear baseline on the transition domain, whereas on Alboran it does not.</span> <span class="abstract-toggle" data-id="2611.07969">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.07969v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.07969v1) · [:material-content-copy: BibTeX](bibtex/2611.07969.bib){ .bibtex-link }
    { .paper-links }

-   #### Robust Ranking of Port Impacts from Impending Tropical Cyclones: A Two-Stage Recommendation System { #2611.07296 }

    *Chenchen Kuai, Shixuan Tang, Chaolun Ma, Xiaoqiang Kong, Xiubin Bruce Wang, Yunlong Zhang, Zihao Li* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.07296">When a tropical cyclone (TC) threatens several ports, preparedness planning depends on anticipating which ports are likely to be affected and where disruptions may last longest. Limited historical...</span><span class="abstract-full" id="full-2611.07296" hidden>When a tropical cyclone (TC) threatens several ports, preparedness planning depends on anticipating which ports are likely to be affected and where disruptions may last longest. Limited historical observations at individual ports and uncertainty in landfall location make these relative impacts difficult to estimate. We use a two-stage recommendation framework that first selects candidate ports based on their proximity to the expected landfall location and then ranks them by their estimated number of impact days. Using AIS-derived impact days for 37 ports and 31 TCs in the Gulf of Mexico study region, we compare three ways of representing port--cyclone interactions: pairwise feature interactions, tree-based feature combinations, and explicit feature crosses combined with learned embeddings. In tests on Hurricanes Nicole and Ian, the retrieval stage consistently identifies all observed affected ports under landfall uncertainty. Among the ranking models, Wide & Deep Learning achieves the closest agreement with the observed port rankings. Its mean ranking agreement exceeds that of the distance-based baseline by 85.5% for Nicole and 55.5% for Ian and is less sensitive to uncertainty in the landfall location. These rankings could help port authorities prioritize post-cyclone inspections and allocate recovery resources across ports threatened by the same event.</span> <span class="abstract-toggle" data-id="2611.07296">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.07296v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.07296v1) · [:material-content-copy: BibTeX](bibtex/2611.07296.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a>
    { .paper-tags }

-   #### A Deep-Learning Framework for the Generalized Emulation of Atmospheric Radiative Transfer at Microwave Frequencies { #2611.07094 }

    *Spencer R. Jones, Simon Pfreundschuh, Christian D. Kummerow* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.07094">This paper presents an application of machine learning to the problem of the computational expense of atmospheric radiative transfer calculations from physics-based models. While in recent studies...</span><span class="abstract-full" id="full-2611.07094" hidden>This paper presents an application of machine learning to the problem of the computational expense of atmospheric radiative transfer calculations from physics-based models. While in recent studies individual emulators have been created that emulate certain radiative transfer schemes for different sensors, we show that a general framework can be applied to emulate the radiative transfer method itself. A fully connected deep neural network has been trained to predict top-of-atmosphere microwave radiances in all-sky conditions over various land, ocean and sea ice surfaces at all frequencies from 1 to 300 GHz, and for all incidence angles from 0 to 65 degrees. The reference models against which the emulator has been trained include an Eddington two-stream radiative transfer model for one-dimensional and pseudo-three-dimensional microwave brightness temperature simulations. This neural network architecture has also been separately trained to reproduce the explicit calculations of the Community Radiative Transfer Model (CRTM) for a single microwave imager. In all cases, the model architecture is successfully able predict microwave brightness temperatures by being trained directly on the optical properties of the atmosphere and surface and therefore directly emulating the radiative transfer method. Compared to the explicit schemes used to produce the training data, the GPU-enabled emulator is able to produce brightness temperature simulations more than three orders of magnitude faster and with a root-mean-squared-error less than 0.0348 K, 0.0531 K, and 0.0246 K against the independent test data for the two Eddington models and CRTM, respectively. Additionally, preliminary training using a full 3D Monte Carlo scheme for high-resolution simulations shows promising results for the application of this method to "exact" radiative transfer methods, and these results are discussed in Section IV.</span> <span class="abstract-toggle" data-id="2611.07094">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.07094v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.07094v1) · [:material-content-copy: BibTeX](bibtex/2611.07094.bib){ .bibtex-link }
    { .paper-links }

-   #### Evaluating the prediction of subgrid eddy kinetic energy from a machine learning model { #2611.06539 }

    *Thomas Wilder, Ranjini Swaminathan, David Storkey, Till Kuhlbrodt* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.06539">Global climate model resolution is still too coarse to fully resolve mesoscale processes, therefore subgrid closures are required to approximate the feedback of eddies onto the resolved flow. With...</span><span class="abstract-full" id="full-2611.06539" hidden>Global climate model resolution is still too coarse to fully resolve mesoscale processes, therefore subgrid closures are required to approximate the feedback of eddies onto the resolved flow. With high-resolution datasets now readily available, there has been a rapid growth in the use of machine learning to develop new eddy closures that could help reduce model biases and uncertainty. In this work we employ an offline learning approach with a convolutional neural network to predict subgrid eddy kinetic energy ($KE_e$), laying the initial groundwork for a machine-learned scale-aware GEOMETRIC eddy closure for the UK's Earth System Model. We address the question of whether global parameterizations are needed or if regional closures are sufficient by evaluating a number of neural network models through region and feature ablation experiments. ML parameterization models generalize to unseen regions when their training data share similar feature and target distributions with those regions. Models that fail to generalize incur distributional shift, or employ different filtering scales for the true eddy kinetic energy field. When holding out features, there is improvement among the generalizable ML models, with some feature spaces leading to improvements in some regions but degrading performance in others. Generally, power spectral density of $KE_e$ tend to diverge at small scales, though some feature spaces are able to minimize this divergence. Overall, the best performing model trained on all regions achieves time-mean and anomaly R2 scores of R2$\approx 0.95$ and R2$\approx 0.8 - 0.9$, respectively.</span> <span class="abstract-toggle" data-id="2611.06539">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.06539v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.06539v1) · [:material-content-copy: BibTeX](bibtex/2611.06539.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a> <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a>
    { .paper-tags }

-   #### Distinguishing Initial-Condition and Model Error Using Machine-Learning Weather Models { #2611.05592 }

    *P. Trent Vonich, Gregory J. Hakim* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.05592">Differentiable weather models allow atmospheric initial conditions to be directly optimized to shadow a known trajectory, which we take from a reanalysis dataset. Here, we extend this optimization to...</span><span class="abstract-full" id="full-2611.05592" hidden>Differentiable weather models allow atmospheric initial conditions to be directly optimized to shadow a known trajectory, which we take from a reanalysis dataset. Here, we extend this optimization to include the forecast model, GraphCast, using low-rank adaptation (LoRA). We compare model optimization with initial-condition (IC) optimization, and then joint optimization, where both are tuned simultaneously. When compared against a control forecast initialized with reanalysis data, LoRA and IC optimization individually produce similar improvements. We find similar results when IC and LoRA are jointly optimized across a spectrum of solutions where a tuning coefficient smoothly shifts the weight of the joint optimization from all-LoRA to all-IC; forecast improvement is not unique to the relative contributions of model and IC corrections. To better discriminate the source of error, we introduce a second model, Pangu-Weather, as a supervisor that gates IC updates during joint optimization. This gated optimization is the same as before, except that IC updates at each epoch are accepted only when they improve the forecasts of both models; otherwise, only the LoRA update is accepted. Results show that this cross-model constraint substantially improves agreement between the two models in their response to the same optimized ICs, narrowing the gap in mean forecast error reduction from 64% to 10%. Moreover, when used together, the LoRA and IC corrections from the gated optimization yield GraphCast forecast improvements similar to those from the ungated optimization. These results imply that model error that is encoded in the ungated ICs is shifted to the LoRA corrections via gating, yielding IC corrections that may provide better analysis estimates.</span> <span class="abstract-toggle" data-id="2611.05592">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.05592v2) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.05592v2) · [:material-content-copy: BibTeX](bibtex/2611.05592.bib){ .bibtex-link }
    { .paper-links }

-   #### A Multidimensional Evaluation Framework for Artificial Intelligence-Based Ocean State Forecasts { #2611.04892 }

    *Ziqing Zu, Ye Yuan, Tiejun Ling, Zhi Chen, Xiang Li, Zhiqiang Li, Qiang Wang, Jiangjiang Xia et al.* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.04892">Artificial intelligence (AI) based ocean state forecasts now approach or surpass those produced by operational ocean forecasting systems as measured by conventional pointwise metrics. However, these...</span><span class="abstract-full" id="full-2611.04892" hidden>Artificial intelligence (AI) based ocean state forecasts now approach or surpass those produced by operational ocean forecasting systems as measured by conventional pointwise metrics. However, these metrics may favor spatially smoothed forecasts and cannot determine whether forecast variables remain physically coherent. Forecast evaluation should therefore move beyond overall pointwise accuracy toward assessing forecast trustworthiness. We propose a multidimensional framework comprising overall pointwise accuracy, physical consistency, and distributional consistency. Overall pointwise accuracy summarizes forecast and reference agreement over all valid samples; physical consistency evaluates dynamical balances using model outputs alone; and distributional consistency evaluates distribution tails and multiscale spatial structures. Three AI models were deployed as representative testbeds in a unified operational environment. This comprehensive assessment shows that lower overall pointwise errors cannot preclude substantial dynamical imbalances and suppressed fine-scale variance. Conversely, higher pointwise errors may arise from spatial displacement of coherent structures rather than a poor representation of their amplitude. The results demonstrate that the evaluation of AI ocean forecasting models should be based on the multi-dimensional assessment of forecast trustworthiness, rather than solely relying on a single metric. The study provides a preliminary framework for AI model intercomparison, which is also informative for targeted model refinement and operational implementation.</span> <span class="abstract-toggle" data-id="2611.04892">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.04892v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.04892v1) · [:material-content-copy: BibTeX](bibtex/2611.04892.bib){ .bibtex-link }
    { .paper-links }

</div>

