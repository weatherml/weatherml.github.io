---
hide:
  - navigation
  - toc
title: weatherml
---

A collection of papers on AI for weather forecasting, climate modelling and atmospheric science.

<p class="page-meta" markdown="span">1616 papers · updated 2026-11-04 · <a href="feed.xml">:material-rss: RSS</a> · <a href="all_papers.bib" download>:material-download: BibTeX</a> · <a href="https://github.com/weatherml/weatherml.github.io/issues/new?template=suggest-paper.yml">:material-plus: Suggest a paper</a></p>

## Browse by Topic

<div class="grid cards topics" markdown>

-   [Global Models](papers/global-models/index.md) <span class="topic-count">382</span>
-   [Regional Models](papers/regional-models/index.md) <span class="topic-count">69</span>
-   [Nowcasting](papers/nowcasting/index.md) <span class="topic-count">106</span>
-   [Downscaling](papers/downscaling/index.md) <span class="topic-count">108</span>
-   [Post-processing](papers/post-processing/index.md) <span class="topic-count">27</span>
-   [Data Assimilation](papers/data-assimilation/index.md) <span class="topic-count">81</span>
-   [Climate Modeling](papers/climate-modeling/index.md) <span class="topic-count">308</span>
-   [Hydrology](papers/hydrology/index.md) <span class="topic-count">40</span>
-   [Ocean & Sea Ice](papers/ocean-sea-ice/index.md) <span class="topic-count">91</span>
-   [Air Quality & Composition](papers/air-quality-composition/index.md) <span class="topic-count">79</span>
-   [Remote Sensing](papers/remote-sensing/index.md) <span class="topic-count">103</span>
-   [Other](papers/other/index.md) <span class="topic-count">222</span>

</div>

## Recent Additions

<div class="grid cards" markdown data-search-exclude>

-   #### A Multidimensional Evaluation Framework for Artificial Intelligence-Based Ocean State Forecasts { #2611.04892 }

    *Ziqing Zu, Ye Yuan, Tiejun Ling, Zhi Chen, Xiang Li, Zhiqiang Li, Qiang Wang, Jiangjiang Xia et al.* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.04892">Artificial intelligence (AI) based ocean state forecasts now approach or surpass those produced by operational ocean forecasting systems as measured by conventional pointwise metrics. However, these...</span><span class="abstract-full" id="full-2611.04892" hidden>Artificial intelligence (AI) based ocean state forecasts now approach or surpass those produced by operational ocean forecasting systems as measured by conventional pointwise metrics. However, these metrics may favor spatially smoothed forecasts and cannot determine whether forecast variables remain physically coherent. Forecast evaluation should therefore move beyond overall pointwise accuracy toward assessing forecast trustworthiness. We propose a multidimensional framework comprising overall pointwise accuracy, physical consistency, and distributional consistency. Overall pointwise accuracy summarizes forecast and reference agreement over all valid samples; physical consistency evaluates dynamical balances using model outputs alone; and distributional consistency evaluates distribution tails and multiscale spatial structures. Three AI models were deployed as representative testbeds in a unified operational environment. This comprehensive assessment shows that lower overall pointwise errors cannot preclude substantial dynamical imbalances and suppressed fine-scale variance. Conversely, higher pointwise errors may arise from spatial displacement of coherent structures rather than a poor representation of their amplitude. The results demonstrate that the evaluation of AI ocean forecasting models should be based on the multi-dimensional assessment of forecast trustworthiness, rather than solely relying on a single metric. The study provides a preliminary framework for AI model intercomparison, which is also informative for targeted model refinement and operational implementation.</span> <span class="abstract-toggle" data-id="2611.04892">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.04892v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.04892v1) · [:material-content-copy: BibTeX](bibtex/2611.04892.bib){ .bibtex-link }
    { .paper-links }

-   #### Machine learning emulators of dynamic global vegetation models for fast gridded carbon budget predictions { #2611.04536 }

    *Piyu Ke, Xiaofan Gui, Stephen Sitch, Zhu Liu, Xun Li, Jiang Bian, Pierre Friedlingstein et al.* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.04536">The land carbon sink explains most of the variability of the atmospheric CO2 growth rate, but the Global Carbon Budget estimates it with dynamic global vegetation models (DGVMs) run once a year....</span><span class="abstract-full" id="full-2611.04536" hidden>The land carbon sink explains most of the variability of the atmospheric CO2 growth rate, but the Global Carbon Budget estimates it with dynamic global vegetation models (DGVMs) run once a year. Emulators could provide it sooner if they reproduce each model outside its training period. Here we show that convolutional neural network emulators of 19 of these models, driven by ERA5 climate and their own past predictions, reproduce the global land sink of these models and its spread over two years not used in training. Global monthly net biome production (NBP) anomalies were reproduced with a median R^2 of 0.81. The error of the multi-model mean annual NBP was at most 0.23 GtC yr^-1. At the grid-cell level, the median R^2 of monthly anomalies was 0.30, partly because of errors that accumulate in the emulated NBP history.</span> <span class="abstract-toggle" data-id="2611.04536">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.04536v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.04536v1) · [:material-content-copy: BibTeX](bibtex/2611.04536.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a> <a class="md-tag" href="/explore/?t=monthly" data-tag="monthly">Monthly</a>
    { .paper-tags }

-   #### From Sea Surface Height to Velocity - A Machine Learning Approach { #2611.04270 }

    *Yaswanth Reddy Yaradoddi, Jinbo Wang, Elyes Serghine* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.04270">Inferring upper-ocean velocity from altimetric sea surface height (SSH) remains challenging below 50~km, scales now resolved by the Surface Water and Ocean Topography (SWOT) mission. Conventional...</span><span class="abstract-full" id="full-2611.04270" hidden>Inferring upper-ocean velocity from altimetric sea surface height (SSH) remains challenging below 50~km, scales now resolved by the Surface Water and Ocean Topography (SWOT) mission. Conventional methods rely on geostrophic or cyclogeostrophic balance. Here, we demonstrate that deep learning can predict the full velocity vector from SSH and environmental data and recover derivative-based diagnostics, including spectral energy flux. We trained a U-Net encoder--decoder on hourly fields from the pre-SWOT Cape Basin extraction of the LLC4320 ocean simulation, with 1.5-km grid spacing ($11.8$--$15.8^\circ$E, $41$--$45^\circ$S). Inputs were SSH, sea surface temperature (SST), and wind stress; targets were horizontal velocities at the surface and 50~m, where wind influence is weaker. At 50~m, the best model achieves a speed RMSE of $0.049\,\mathrm{m\,s^{-1}}$, 29% lower than the geostrophic approximation. Predicted relative vorticity has a correlation of 0.93, compared with 0.75 for the geostrophic approximation. The model also recovers divergence with a correlation of 0.66, whereas geostrophic velocity is nondivergent by definition. Tests of training strategies yield four findings: SSH gradients outperform raw SSH as inputs; wind stress is the most useful covariate for total velocity prediction; SST benefits vary seasonally; and including vorticity and divergence in the loss improves accuracy. These results offer a path for transferring high-fidelity simulation physics to SWOT applications, where kilometer-scale SSH snapshots contain superposed balanced and unbalanced signals.</span> <span class="abstract-toggle" data-id="2611.04270">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.04270v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.04270v1) · [:material-content-copy: BibTeX](bibtex/2611.04270.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a> <a class="md-tag" href="/explore/?t=km-scale" data-tag="km-scale">Km-scale</a> <a class="md-tag" href="/explore/?t=hourly" data-tag="hourly">Hourly</a>
    { .paper-tags }

-   #### Tropical Cyclone-Following Dataset for Limited-Area Machine Learning Applications { #2611.03925 }

    *Minh-Khanh Luong, Chanh Kieu* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.03925">This work presents a lightweight dataset of tropical cyclone (TC)-following meteorological variables to facilitate the development and application of machine learning (ML) methods in TC research....</span><span class="abstract-full" id="full-2611.03925" hidden>This work presents a lightweight dataset of tropical cyclone (TC)-following meteorological variables to facilitate the development and application of machine learning (ML) methods in TC research. Using outputs from our previous climate downscaling and idealized simulations, the dataset is constructed as multichannel video sequences that track TC centers throughout their lifetimes. To demonstrate the dataset utility, we present an application in which a regional ML model is trained to predict TC intensity with explicit incorporation of lateral boundary conditions. Our results highlight the importance of lateral boundary information for predicting TC intensity with ML models inside a limited-area domain, underscoring the need to properly account for lateral boundaries when developing regional ML models for TC intensity forecast.</span> <span class="abstract-toggle" data-id="2611.03925">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.03925v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.03925v1) · [:material-content-copy: BibTeX](bibtex/2611.03925.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a> <a class="md-tag" href="/explore/?t=benchmarks-datasets" data-tag="benchmarks-datasets">Benchmarks & datasets</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a>
    { .paper-tags }

-   #### Can LLM Agents Improve Imperfect Earth System Model Parameterizations? { #2611.03543 }

    *Kara D. Lamb* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.03543">Earth system models (ESMs) rely on parameterizations to represent unresolved processes such as convection, cloud microphysics, and turbulence. However, translating advances in physical understanding...</span><span class="abstract-full" id="full-2611.03543" hidden>Earth system models (ESMs) rely on parameterizations to represent unresolved processes such as convection, cloud microphysics, and turbulence. However, translating advances in physical understanding into stable, validated parameterizations and integrating them into ESMs often takes years. Agentic AI could accelerate this process and enable more systematic exploration of design choices. Using warm-rain collision-coalescence as a test case, we evaluate agents' ability to synthesize scientific literature, reimplement published bulk parameterizations, and propose new functional forms as differentiable JAX code. We compare conventional parameterizations, data-driven emulators, and agent-generated closures against high-fidelity superdroplet simulations, repeating unattended experiments across ten language models. With expert guidance or iterative feedback, agents produced executable, differentiable, mass-conserving code, although unattended performance depended strongly on the scheme and language model. Iterative feedback lowered median rollout error for eight of ten models. Retuned physical schemes matched or outperformed many learned parameterizations at substantially lower computational cost, but two of 100 unattended agent-generated closures achieved slightly lower rain-mass error than the most accurate retuned scheme on held-out simulations.</span> <span class="abstract-toggle" data-id="2611.03543">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.03543v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.03543v1) · [:material-content-copy: BibTeX](bibtex/2611.03543.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=llms-agents" data-tag="llms-agents">LLMs & agents</a>
    { .paper-tags }

-   #### Six-hourly Probabilistic Precipitation Forecasts Created From Hourly Probabilities { #2611.03326 }

    *Thomas M Hamill* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.03326">A lightweight neural network is described that produces six-hourly, spatially detailed probabilistic quantitative precipitation forecasts (PQPF) from a sequence of hourly spatially detailed...</span><span class="abstract-full" id="full-2611.03326" hidden>A lightweight neural network is described that produces six-hourly, spatially detailed probabilistic quantitative precipitation forecasts (PQPF) from a sequence of hourly spatially detailed probabilistic forecasts, for lead times from 6 to 72\,h. The hourly forecasts were generated by a “zero-inflated” Attention Residual U-Net that outputs the parameters of a probability distribution including a probability of no precipitation and a weighted mixture of two Gamma distributions at each grid point for each hourly forecast lead time. The six-hourly network passes the six consecutive hourly parameter sets, together with summary statistics of the spatial texture of the raw deterministic forecast, through a small recurrent (GRU) layer and a multi-layer perceptron, and maps them to a corresponding zero-inflated mixture of three Gamma distributions describing the six-hourly accumulated precipitation. Training uses sets of hourly probabilistic forecasts, drawn preferentially from wet and heavy-precipitation cases (importance sampling), and matched Multi-Radar Multi-Sensor (MRMS) six-hourly precipitation accumulations, subject to a quality threshold. The resulting six-hourly probabilistic forecasts are well calibrated and skillful, as demonstrated by reliability diagrams, performance diagrams and Brier Skill Scores evaluated over an out-of-sample period spanning day 12 through the end of every month of 2025. The Brier Skill Scores are consistently and significantly higher than those of a baseline that sums independent draws from the hourly distributions. Forecasts show realistic modulation of precipitation related to terrain variations</span> <span class="abstract-toggle" data-id="2611.03326">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.03326v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.03326v1) · [:material-content-copy: BibTeX](bibtex/2611.03326.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a> <a class="md-tag" href="/explore/?t=recurrent-networks" data-tag="recurrent-networks">Recurrent networks</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=hourly" data-tag="hourly">Hourly</a> <a class="md-tag" href="/explore/?t=6-hourly" data-tag="6-hourly">6-hourly</a>
    { .paper-tags }

-   #### New Wine in an Old Bottle -- NeuralPOM: A Hybrid Ocean Forecast Model with Differentiable Dynamics and Staged Learning { #2611.02460 }

    *Ruiqi Shu, Guangliang Liu, Ruijian Gou, Tianrun Gao, Haomin Wen, Hao Li, Xiaomeng Huang* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.02460">Coarse-grid ocean circulation models rely on empirical subgrid parameterizations that struggle to capture mesoscale eddies, whereas purely data-driven neural forecasters accumulate trajectory drift,...</span><span class="abstract-full" id="full-2611.02460" hidden>Coarse-grid ocean circulation models rely on empirical subgrid parameterizations that struggle to capture mesoscale eddies, whereas purely data-driven neural forecasters accumulate trajectory drift, violate tracer conservation, and damp small-scale variability over extended integrations. Here we present NeuralPOM, a hybrid ocean forecast model that couples differentiable explicit mode-split and semi-implicit free-surface implementations of the Princeton Ocean Model (POM) to a conservation-constrained neural corrector. The three-dimensional primitive-equation solver advances the ocean state, while the neural network periodically updates the physical fields at high-frequency (1 h) or low-frequency (1 day) intervals under volume, heat, and salt conservation constraints. Using a fine-grid reference dataset exceeding 100,000 model days in a wind-driven double-gyre basin, we introduce a three-stage training procedure comprising offline single-step pretraining, multi-step differentiable training through the ocean solver, and multi-day closed-loop rollout training. Over 100-day forecasts, NeuralPOM reduces errors across all prognostic fields relative to coarse-grid physical closures and direct neural forecasters, preserving mesoscale kinetic energy while improving eddy phase alignment. Controlled ablations show that closed-loop training suppresses multi-day drift, whereas backpropagating gradients through the differentiable solver stabilizes high-frequency coupling. Finally, because high-frequency tracer updates can introduce density anomalies via the equation of state that excite spurious baroclinic pressure gradients, we introduce Density-Normal Damping (DND) to attenuate the density-modifying (diapycnal) tracer component while retaining the density-compensated (isopycnal) correction.</span> <span class="abstract-toggle" data-id="2611.02460">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.02460v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.02460v1) · [:material-content-copy: BibTeX](bibtex/2611.02460.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=physics-ml-hybrid" data-tag="physics-ml-hybrid">Physics–ML hybrid</a>
    { .paper-tags }

-   #### Sonny-ENS: demystifying machine-learning ensemble weather forecasting { #2611.01614 }

    *Minjong Cheon* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.01614">Every weather forecast is uncertain, and decisions from flood warnings to energy trading depend on how well that uncertainty is quantified. Machine-learning (ML) ensembles now quantify it better than...</span><span class="abstract-full" id="full-2611.01614" hidden>Every weather forecast is uncertain, and decisions from flood warnings to energy trading depend on how well that uncertainty is quantified. Machine-learning (ML) ensembles now quantify it better than the best physics-based ensemble, yet two things about them remain opaque: whether their skill requires hundreds to thousands of accelerator-days of training, and how a network with a simple source of randomness produces a spread that follows the flow. Here we introduce Sonny-ENS, an ensemble of three 68.9-million-parameter networks with Monte Carlo dropout, trained on the continuous ranked probability score (CRPS) in 2.7 A40 GPU-days. On 2022, it has a lower CRPS than the European Centre for Medium-Range Weather Forecasts ensemble (ENS) on 96% of 570 targets and an 8.3% lower mean CRPS at lead times of 1-10 days. It outperforms the operational GenCast on 69% of the targets, and a 51-member, 15-day forecast takes 69 s on one GPU. Ablations identify the ingredients of this skill, and a Kolmogorov-Arnold sparse autoencoder shows that dropout injects uncertainty mainly in the tropics, whereas the spread grows where the atmosphere is baroclinically unstable (spatial correlation 0.91 with the Eady growth rate). Skillful probabilistic weather forecasts are therefore within reach of a single GPU server.</span> <span class="abstract-toggle" data-id="2611.01614">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.01614v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.01614v1) · [:material-content-copy: BibTeX](bibtex/2611.01614.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### Forecasting Visitor Demand from Open AI-Camera Counts and Weather: Lessons from Japan for Saudi Heritage Sites { #2611.01033 }

    *Amil Khanzada, Mohammed Sohail, Jana Wasfi, Ahmad Showail* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.01033">Small heritage and nature sites need a few days' notice of busy days to plan staff, overflow parking and shuttles, but most have neither sensors nor forecasting expertise. We test what one artificial...</span><span class="abstract-full" id="full-2611.01033" hidden>Small heritage and nature sites need a few days' notice of busy days to plan staff, overflow parking and shuttles, but most have neither sensors nor forecasting expertise. We test what one artificial intelligence (AI) camera counter per site, a public holiday calendar and a free weather forecast can predict. Using 22 months of open AI-camera counts from three contrasting sites in Fukui, Japan (a train station, a coastal cliff and a mountain summit park), we measure how weather changes visits, forecast daily counts up to a week ahead with only the weather forecast available at the time, test busy-day warnings tied to a staffing decision, and ask how much history a new counter needs. The more exposed the site, the more weather matters: 10 mm of rain comes with 3% fewer detections at the station, 9% fewer at the cliff and 37% fewer at the summit park. With the forecasts actually issued, three-day-ahead weighted absolute percentage error (WAPE) at the summit park falls from 38.5% to 31.9% (Diebold-Mariano p<0.001), and the gain holds to five days; the sheltered station gains nothing. Three-day busy-day warnings with forecast weather reach, on past data and under the primary definitions, an area under the receiver operating characteristic curve (AUC) of 0.87-0.93, and weather lifts the overflow car park warning's precision from 0.60 to 0.82. For a new counter, blending a learned model with last week's same-hour counts cuts hourly error after only three days of training, pending a real installation. All inputs are off the shelf, and we outline how the approach could apply to heritage sites in Saudi Arabia.</span> <span class="abstract-toggle" data-id="2611.01033">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.01033v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.01033v1) · [:fontawesome-brands-github: Code](https://github.com/amilkh/paper-icaisc-supplement) · [:material-content-copy: BibTeX](bibtex/2611.01033.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=hourly" data-tag="hourly">Hourly</a> <a class="md-tag" href="/explore/?t=daily" data-tag="daily">Daily</a>
    { .paper-tags }

-   #### Value of Adding AI Weather Model Ensembles to a Multi-model Probabilistic Blend { #2610.42130 }

    *Harris Sop Nkuiate, Gavin Evans, Christopher A. T. Ferro, Frank Kwasniok, Caroline Bain et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.42130">Many weather forecasting organizations use a multi-model blend of ensemble members from different prediction systems to obtain a more accurate forecast. Recent studies have shown that blending...</span><span class="abstract-full" id="full-2610.42130" hidden>Many weather forecasting organizations use a multi-model blend of ensemble members from different prediction systems to obtain a more accurate forecast. Recent studies have shown that blending deterministic AI-based models alongside physics-based models improves the overall forecast skill. However, the incorporation of an ensemble AI-based model into a multi-model blend is yet to be investigated. This study explores the value of blending the ensemble version of the Artificial Intelligence Forecasting System (AIFS-CRPS) with the Met Office's convection permitting physics-based ensemble, MOGREPS-UK. We show that blending site temperature (and wind speed) forecasts over the United Kingdom from both ensemble physics-based and ensemble AI-based models leads to an increase in overall forecast skill spatially, across lead times and across exceedance thresholds, especially when the models have different error characteristics. By decomposing the continuous ranked probability score of the blend into component means and diversity terms, we show that a more accurate blend can be achieved by assigning more weight to more skillful models and using models with complementary errors. For example, we show that a blend of MOGREPS-UK with either AIFS-CRPS or the physics-based Integrated Forecasting System (IFS) is better than a blend of IFS and AIFS-CRPS which are not very diverse. This has implications for the importance of weather model diversity and may lend weight to different model development approaches being pursued in tandem. We also demonstrate that calibrating AIFS-CRPS forecasts yields an increase in forecast skill comparable to calibrated ensemble physics-based forecasts, and calibrating the source forecasts before blending and the blended forecast after blending improves the overall forecast accuracy.</span> <span class="abstract-toggle" data-id="2610.42130">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.42130v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.42130v1) · [:material-content-copy: BibTeX](bibtex/2610.42130.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=evaluation" data-tag="evaluation">Evaluation</a>
    { .paper-tags }

-   #### SATFRONT: Detecting Atmospheric Fronts with Satellite Data using Machine Learning { #2610.42111 }

    *Ioannis Dravilas, Miruna Stoicescu, Roope Tervo, Carlos Horn, Oriol Hinojo Comellas, Sina Montazeri et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.42111">Atmospheric fronts are associated with much of the high-impact weather in the mid-latitudes, yet their identification and classification remain largely manual and subjective. Existing deep learning...</span><span class="abstract-full" id="full-2610.42111" hidden>Atmospheric fronts are associated with much of the high-impact weather in the mid-latitudes, yet their identification and classification remain largely manual and subjective. Existing deep learning approaches rely primarily on gridded reanalysis or forecast data, limiting their applicability to satellite observations available for time-critical applications. We introduce SATFRONT, a machine learning framework for detecting cold, warm, and occluded fronts over Europe from geostationary satellite observations, with reanalysis data as an optional additional input. The model is trained on frontal analyses from five European national meteorological services and evaluated against UK Met Office analyses for 2024. Using satellite observations alone, SATFRONT achieves a Critical Success Index (CSI) of 0.428 at a 150 km tolerance, within the range of agreement between analyses from different meteorological services (0.277-0.501) and higher than that of an operational rule-based front detector that uses satellite data (0.133). Adding ERA5 reanalysis increases the CSI to 0.591. ERA5 substantially improves the placement of individual fronts, although its effect on their annual spatial distribution varies by front type. These results demonstrate that cold, warm, and occluded fronts can be detected directly from geostationary satellite observations, providing a basis for automated frontal analyses at the high temporal frequency required for time-critical applications.</span> <span class="abstract-toggle" data-id="2610.42111">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.42111v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.42111v1) · [:material-content-copy: BibTeX](bibtex/2610.42111.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a>
    { .paper-tags }

-   #### Patch-Based Deep Kernel Learning for Data Assimilation and Bias Correction with Extremely Sparse Observations { #2610.41596 }

    *Emanuele Mele, Angelo Coluccia, Massimo Cafaro, Italo Epicoco, Paola Nassisi* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.41596">Data Assimilation (DA) reconstructs the state of a spatio-temporal system by combining sensor observations with the output of a numerical model. Since model states are high-dimensional while...</span><span class="abstract-full" id="full-2610.41596" hidden>Data Assimilation (DA) reconstructs the state of a spatio-temporal system by combining sensor observations with the output of a numerical model. Since model states are high-dimensional while observations cover only a small fraction of the domain, DA is a challenging large-scale estimation problem, central to weather forecasting and oceanography. Classical variational methods such as 3D-Var rely on a hand-crafted background error covariance that assumes unbiased errors, and therefore cannot capture state-dependent correlations or systematic model biases. Moreover, their cost grows with the size of the domain. Deep learning (DL) methods can perform DA and bias correction (BC) non-linearly, but typically offer limited interpretability and no calibrated uncertainty estimates. We propose a hybrid framework that couples a convolutional neural network (CNN) with a learnable-kernel Gaussian Process (GP) to jointly perform DA and BC from extremely sparse observations. The model is trained end-to-end by maximizing the marginal log-likelihood, learns a data-driven covariance over the whole grid, and provides both the Maximum A Posteriori analysis and its uncertainty by design. A patch-based strategy reduces the $O(N^3)$ training cost of exact GPs over $N$ grid points to $O(N)$ for a fixed patch size. Inference remains exact and only requires solving a linear system whose $O(m^3)$ cost depends on the number of observations $m \ll N$, not on the domain. The learned covariance can also replace the hand-crafted one in 3D-Var, improving its accuracy without changing the variational formulation. Experiments on sea surface temperature and salinity, with both synthetic and real, extremely sparse in-situ observations, show that the framework outperforms 3D-Var, a classical GP and a competing DL model, while reducing inference time by more than 98% with respect to 3D-Var.</span> <span class="abstract-toggle" data-id="2610.41596">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.41596v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.41596v1) · [:material-content-copy: BibTeX](bibtex/2610.41596.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a> <a class="md-tag" href="/explore/?t=physics-ml-hybrid" data-tag="physics-ml-hybrid">Physics–ML hybrid</a> <a class="md-tag" href="/explore/?t=classical-ml" data-tag="classical-ml">Classical ML</a>
    { .paper-tags }

-   #### Generated storm lifecycles fill the rare-event tail of short rainfall records { #2610.41365 }

    *Valerie Tsao, Nathaniel W. Chaney, Manolis Veveakis, Marta Zaniolo* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.41365">Historically rare ''one-in-a-generation'' storms are occurring with unprecedented frequency, causing catastrophic damage and mounting casualties, and exposing the vulnerability of infrastructure...</span><span class="abstract-full" id="full-2610.41365" hidden>Historically rare ''one-in-a-generation'' storms are occurring with unprecedented frequency, causing catastrophic damage and mounting casualties, and exposing the vulnerability of infrastructure designed around short observational records. To overcome the limitations of sparse data, we present StormGen, a generative machine-learning model that transforms an observed storm's initial hours into complete, physically plausible hourly rainfall lifecycles in minutes. Evaluated on a decade of southeast Texas rainfall, StormGen produced close counterparts for 4\5% of the most severe 2% of storms--compared to just 3% for stochastic storm transposition--and maintained spatial coherence over full 24-hour sequences. Crucially, from a single hour of initial rainfall, the model successfully reconstructed the specific lifecycle, spatial pattern, and extreme catchment totals of Hurricane Harvey itself, despite the event being entirely absent in training. Because storm timing alone can more than double peak flows, these complete generated scenarios provide a vital mechanism to stress-test flood protections against unprecedented extremes.</span> <span class="abstract-toggle" data-id="2610.41365">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.41365v2) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.41365v2) · [:material-content-copy: BibTeX](bibtex/2610.41365.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a> <a class="md-tag" href="/explore/?t=hourly" data-tag="hourly">Hourly</a>
    { .paper-tags }

-   #### Machine Learning Sensor Fusion for Water Vapor Radiometry based Tropospheric Delay Calibration { #2610.41052 }

    *T. K. Sridharan, Yoshiharu Asaki, Brian E. Svoboda* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.41052">The next-generation Very Large Array (ngVLA) intends to use 22 GHz water vapor radiometry (WVR) to correct for tropospheric phase fluctuations in its high-frequency bands. However, current...</span><span class="abstract-full" id="full-2610.41052" hidden>The next-generation Very Large Array (ngVLA) intends to use 22 GHz water vapor radiometry (WVR) to correct for tropospheric phase fluctuations in its high-frequency bands. However, current delay-retrieval methods do not reach the required accuracy and reliability. As a remedy, we propose a machine-learning based retrieval concept and a preliminary architecture, framed as heterogeneous, multi-rate sensor fusion. Atmospheric reanalysis profiles and surface weather set the context in which per-antenna WVR spectra are used to estimate excess path by a Kalman filter whose parameters are learned from high-cadence calibrator phases. The method is being developed, with planned first validation on the 183 GHz WVR system of the Atacama Large Millimeter/submillimeter Array (ALMA).</span> <span class="abstract-toggle" data-id="2610.41052">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.41052v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.41052v1) · [:material-content-copy: BibTeX](bibtex/2610.41052.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### From extreme weather to agricultural consequences with agentic AI { #2610.40641 }

    *Minghan Yu, Mengyu Dai, Yixin Wen, Haizhao Yang* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.40641">Extreme weather is forecast days ahead, yet its consequences for agriculture remain hard to predict: a forecast says where a storm will strike, not which crops will suffer or how much is at risk, and...</span><span class="abstract-full" id="full-2610.40641" hidden>Extreme weather is forecast days ahead, yet its consequences for agriculture remain hard to predict: a forecast says where a storm will strike, not which crops will suffer or how much is at risk, and the records that could tell span many sources. We present WxCrop, the first agent that judges agricultural hazards from physically interpretable weather indices examined by causal analysis, and the first that combines weather and agricultural information to predict crop loss reliably for past and forecast dates. We demonstrate it in Florida on hurricanes, freezes and dry heat, with a loss model calibrated on insurance records for each, and the same procedure extends to other hazards and areas. Three checks show reliability: the hurricane index detects recorded cyclone days with an AUC of 0.98, recorded losses in held-out years are reproduced to within 2%, and independent production records confirm the county rankings (rank correlation 0.63). Users ask the agent for each crop group's loss as a fraction of expected season value, every number sourced, so that governments, growers and scientists respond to agricultural disasters quickly.</span> <span class="abstract-toggle" data-id="2610.40641">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.40641v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.40641v1) · [:material-content-copy: BibTeX](bibtex/2610.40641.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=llms-agents" data-tag="llms-agents">LLMs & agents</a> <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a> <a class="md-tag" href="/explore/?t=extremes" data-tag="extremes">Extremes</a>
    { .paper-tags }

-   #### Causal-prior-constrained deep learning for rapid-intensification warning of western North Pacific tropical cyclones { #2610.40304 }

    *Xinrui Li, Yijun Jin, Yifan Dong, Shoujuan Shu* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.40304">Rapid intensification (RI) is one of the most dangerous and unpredictable phases of tropical cyclones, and its environmental and structural drivers are difficult to identify and separate in...</span><span class="abstract-full" id="full-2610.40304" hidden>Rapid intensification (RI) is one of the most dangerous and unpredictable phases of tropical cyclones, and its environmental and structural drivers are difficult to identify and separate in observations. Here we introduce CausalVisionRI, a causal-prior-constrained deep-learning system for 24-h RI warning in the western North Pacific. It integrates infrared remote sensing imagery, atmospheric and oceanic fields, and an observationally discovered causal graph into a unified model. The graph and eight sign-verified effects on 24-h intensity change provide an auditable causal prior for training. Discovered from 522 cyclones in 2000-2017, it identifies causal pathways linking atmospheric and oceanic conditions, the cyclone's dynamical and structural state, and its 24-h intensity change. On a forward holdout covering 2020-2024 and 3,004 records, the five-member ensemble achieves a Brier skill score (BSS) of 0.259 relative to a fixed development climatology, significantly exceeding four reference systems; the three LDA baselines reach at most 0.105. It also attains higher detection rates than a published knowledge-block scheme and a reconstructed contrastive-learning scheme at matched false-detection rates under our USA 30-kt standard and the published RITCF standard. Ablations show that the graph pathway adds 0.032 BSS across the holdout and 0.063 over 2020-2021, with a significant contribution in one ensemble member. A deep-learning model constrained by observationally discovered causal structure thus offers a new route to physically informed RI warning.</span> <span class="abstract-toggle" data-id="2610.40304">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.40304v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.40304v1) · [:material-content-copy: BibTeX](bibtex/2610.40304.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a>
    { .paper-tags }

-   #### Operator-Aware Posterior Sampling for Multivariate Weather Downscaling { #2610.39592 }

    *Carlo Saccardi, Rudolf Saathof, Geethu Joseph, Justin Dauwels* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.39592">Statistical downscaling is the task of reconstructing high-resolution meteorological fields from coarse inputs. A principled approach formulates downscaling as a Bayesian inverse problem, using a...</span><span class="abstract-full" id="full-2610.39592" hidden>Statistical downscaling is the task of reconstructing high-resolution meteorological fields from coarse inputs. A principled approach formulates downscaling as a Bayesian inverse problem, using a diffusion prior over high-resolution fields and incorporating the coarse field only at inference through a likelihood. This decouples the prior from the coarse source, allowing one pretrained model to adapt to different sources. However, two challenges limit this flexibility: (1) when coarse and target fields come from different reanalysis systems, no fixed coarsening operator can capture their relationship; and (2) guidance gradients can differ by orders of magnitude across variables, making a single guidance strength ineffective. This paper addresses both challenges within the guidance loop, leaving the prior unchanged. We learn the forward operator from paired data and normalize the gradient per channel. We show that this normalization corresponds to steepest descent under a block-max norm, allowing a single guidance strength across variables. Our code is publicly available at https://github.com/CarloSaccardi/operator-aware-weather-downscaling.</span> <span class="abstract-toggle" data-id="2610.39592">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.39592v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.39592v1) · [:fontawesome-brands-github: Code](https://github.com/CarloSaccardi/operator-aware-weather-downscaling) · [:material-content-copy: BibTeX](bibtex/2610.39592.bib){ .bibtex-link }
    { .paper-links }

-   #### Skillful and realistic precipitation forecasting with reinforced diffusion { #2610.38775 }

    *Ozan Sener, Anand Avati, David Hafner, Jierui Lin, Laura Manduchi, Xiaoying Pang, Langwen Huang et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.38775">Accurate precipitation forecasting remains elusive as physics-based models struggle to resolve mesoscale convection and machine learning often yields physically inconsistent, blurred results that...</span><span class="abstract-full" id="full-2610.38775" hidden>Accurate precipitation forecasting remains elusive as physics-based models struggle to resolve mesoscale convection and machine learning often yields physically inconsistent, blurred results that consistently underestimate extreme events. We present FitzRoy, the first system that simultaneously achieves state-of-the-art predictive skill and physical realism over the continental United States. Through a reinforced diffusion framework, where a conditional discrete diffusion model is refined via reinforcement learning, FitzRoy effectively resolves the long-standing trade-off between pointwise accuracy and structural fidelity. Across lead times up to 24 hours, FitzRoy consistently matches or exceeds state-of-the-art models while preserving structural realism and capturing extreme events. Its low-latency inference further enables deployment in real-time operational settings. In case studies such as the 2025 Texas Hill Country floods and the 2026 Missouri floods, FitzRoy signals record-breaking accumulations that existing models largely missed. By bridging the gap between realism and skill, FitzRoy establishes a new frontier for high-fidelity precipitation forecasting.</span> <span class="abstract-toggle" data-id="2610.38775">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.38775v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.38775v1) · [:material-content-copy: BibTeX](bibtex/2610.38775.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=reinforcement-learning" data-tag="reinforcement-learning">Reinforcement learning</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a>
    { .paper-tags }

-   #### Curriculum-Based Deep Reinforcement Learning for Robust Autonomous Rocket Landing under Atmospheric Uncertainty { #2610.38636 }

    *Atin Chowdhury, Piyush Joshi* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.38636">Autonomous far-range rocket landing is a challenging control problem because navigation, descent-rate regulation, attitude management, and fuel use must be coordinated in the presence of uncertain...</span><span class="abstract-full" id="full-2610.38636" hidden>Autonomous far-range rocket landing is a challenging control problem because navigation, descent-rate regulation, attitude management, and fuel use must be coordinated in the presence of uncertain atmospheric and vehicle dynamics. This paper presents a curriculum-driven proximal policy optimization (PPO) framework for three-dimensional rocket landing in a stochastic environment incorporating altitude-dependent aerodynamic drag, wind shear, temporally correlated turbulence, discrete gusts, uncertain propulsion and aerodynamic parameters, thrust-vector misalignment, engine lag, actuator-gain variation, noisy observations, and command latency. Learning is organised into far-range approach, soft descent, and integrated landing stages. Policy transfer, critic warm-up after reward changes, rehearsal of earlier tasks, and progressive expansion of weather severity are used to support learning across these coupled phases. Across 100,000 evaluation episodes, the policy achieves 96,656 successful landings, corresponding to a success rate of 96.656% and a 95% Wilson confidence interval of 96.543--96.766%. Success decreases from 98.84% under zero-disturbance conditions to 88.07% in the highest weather-severity band. Successful touchdowns exhibit small miss distances and low residual horizontal and vertical velocities. An exploratory logistic-regression analysis identifies mean wind speed as the strongest measured correlate of success, with initial distance, initial altitude, initial vertical velocity, and wind-shear exponent also associated with outcomes. The results quantify both the effectiveness of the learned policy over the evaluated disturbance distribution and the degradation that occurs under severe compound atmospheric conditions.</span> <span class="abstract-toggle" data-id="2610.38636">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.38636v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.38636v1) · [:material-content-copy: BibTeX](bibtex/2610.38636.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=reinforcement-learning" data-tag="reinforcement-learning">Reinforcement learning</a>
    { .paper-tags }

-   #### RainCheck: Harnessing Historical Weather Data with Machine Learning for Rainfall Forecasting in Bangladesh { #2610.38211 }

    *Shahran Rahman Alve, Muhammad Zawad Mahmud, Samiha Islam, Al Jubayer Pial* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.38211">Machine-learning rainfall forecasts are often evaluated by shuffling station records into random training and test sets. For rainfall, this lets a model see what happened at neighbouring stations on...</span><span class="abstract-full" id="full-2610.38211" hidden>Machine-learning rainfall forecasts are often evaluated by shuffling station records into random training and test sets. For rainfall, this lets a model see what happened at neighbouring stations on the very day it is asked to forecast, and it hides the fact that a seasonal lookup table already explains most of the variance. We quantify both effects on 618,650 station-days from 35 Bangladesh Meteorological Department stations (1948-2014). We compare random, temporal (train <=1999, test 2005-2014) and spatio-temporal (unseen stations and unseen years) evaluation, calendar-only versus antecedent-rainfall features, and six learners against climatology, persistence and SARIMA. Three findings stand out. First, a station-month climatology alone classifies high-rainfall months with 88.4% accuracy (AUC 0.950), and no learner improved on it by more than 0.8 points under temporal evaluation. Second, a calendar-only random forest reached a next-day wet-day AUC of 0.889 under a random split, higher than any model with real predictors under temporal evaluation, yet fell to 0.835, below climatology (0.859), on future years. Third, models with lagged rainfall add genuine but modest skill for next-day forecasts: a Brier skill score of 0.074 (95% CI 0.064-0.088) over climatology, rising to 0.091 when station and year inputs are removed and to 0.131 when other stations' same-day observations are added. A GRU sequence model matches but does not exceed gradient boosting, and skill disappears beyond a two-day lead. For monthly totals, no learner beat climatology significantly. We release a leakage-aware protocol and code so that future station-based studies report skill against climatology on held-out years.</span> <span class="abstract-toggle" data-id="2610.38211">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.38211v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.38211v1) · [:material-content-copy: BibTeX](bibtex/2610.38211.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=recurrent-networks" data-tag="recurrent-networks">Recurrent networks</a> <a class="md-tag" href="/explore/?t=classical-ml" data-tag="classical-ml">Classical ML</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=station-point" data-tag="station-point">Station / point</a> <a class="md-tag" href="/explore/?t=monthly" data-tag="monthly">Monthly</a>
    { .paper-tags }

-   #### Multi-Agent Reinforcement Learning for Global Climate Policy Cooperation { #2610.37870 }

    *Anne Moll-Elsborg, Oskar Bohn Lassen, Francisco Camara Pereira* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.37870">International climate policy requires coordination among countries whose mitigation decisions impose local costs but generate global benefits. Classical game-theoretic and integrated assessment...</span><span class="abstract-full" id="full-2610.37870" hidden>International climate policy requires coordination among countries whose mitigation decisions impose local costs but generate global benefits. Classical game-theoretic and integrated assessment approaches can study these incentives, but typically rely on prescribed behavioural assumptions or equilibrium concepts. Recent work has therefore explored Multi-Agent Reinforcement Learning as a way to model climate-policy cooperation as an emergent outcome of repeated strategic interaction. We extend an existing MARL climate framework with a multi-stage negotiation mechanism in which agents propose agreements, accept or reject commitments, and face penalties for non-compliance. This allows us to study how agreement design affects learned mitigation behaviour, compliance, and climate outcomes. In an idealized setting with four homogeneous agents, we find that non-binding agreements do little to overcome free-riding, while moderate non-compliance penalties improve cooperation and reduce warming. However, excessively strong penalties discourage ambitious commitments, revealing a trade-off between enforcement and participation. We position the framework as a methodological contribution for testing agreement design rather than a predictive model of real negotiations.</span> <span class="abstract-toggle" data-id="2610.37870">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.37870v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.37870v1) · [:material-content-copy: BibTeX](bibtex/2610.37870.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=reinforcement-learning" data-tag="reinforcement-learning">Reinforcement learning</a> <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a>
    { .paper-tags }

-   #### KPPax: preprocess differentiable kinetic ODE for machine learning in atmospheric chemistry { #2610.37655 }

    *Wenqing Peng, Michael Boy, Zhi-Song Liu* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.37655">Atmospheric chemical mechanisms are commonly distributed in the language of the Kinetic PreProcessor (KPP), which generates efficient Fortran or C code but does not fit into the differentiable...</span><span class="abstract-full" id="full-2610.37655" hidden>Atmospheric chemical mechanisms are commonly distributed in the language of the Kinetic PreProcessor (KPP), which generates efficient Fortran or C code but does not fit into the differentiable programming frameworks in which machine-learning models are developed. We present KPPax, a kinetic preprocessor that generates differentiable, just-in-time compiled, and vectorisable chemistry code in JAX from mechanisms written in the KPP language. KPPax evaluates the rate law with sparse operations in log space, which keeps it differentiable, and integrates the stiff system with the implicit solvers and the forward- and reverse-mode adjoints of the <code>diffrax</code> library, so that exact sensitivities are obtained without hand-derived adjoint code. We demonstrate two applications with the POLLU benchmark mechanism. In the first, the Fisher information matrix computed from an automatically differentiated Jacobian predicts, before any fit, which of the 25 rate coefficients a flow-tube-like design can identify, and noisy synthetic estimation confirms the predicted uncertainties of the nine identifiable rate coefficients to within about 30% while revealing clusters of fast reactions of which only ratios are constrained. In the second, a neural-network correction of an emission inventory is trained offline on per-scenario 4D-Var retrievals and then fine-tuned online through the chemical model, which reduces the error of the biogenic correction by a factor of 2.4--5 and recovers information on low-temperature emissions that the individual retrievals lose. KPPax makes KPP mechanisms directly usable for parameter estimation and hybrid physics--machine-learning modelling.</span> <span class="abstract-toggle" data-id="2610.37655">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.37655v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.37655v1) · [:material-content-copy: BibTeX](bibtex/2610.37655.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=physics-ml-hybrid" data-tag="physics-ml-hybrid">Physics–ML hybrid</a>
    { .paper-tags }

-   #### An Input Skip Can Reparameterize a VAE Without Improving the Generator { #2610.37456 }

    *Kevin Lee* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.37456">An input skip can reduce reconstruction error while leaving a model's distribution of new samples unchanged. For a Gaussian variational autoencoder (VAE) with constant input skip $0\leqα<1$, a model...</span><span class="abstract-full" id="full-2610.37456" hidden>An input skip can reduce reconstruction error while leaving a model's distribution of new samples unchanged. For a Gaussian variational autoencoder (VAE) with constant input skip $0\leqα<1$, a model without a skip can be written in skip form by multiplying its decoder output by $b=1-α$ and its noise variance by $b^2$. The distribution of observations and latent variables and its exact generative scores stay the same, while reconstruction mean squared error (MSE) is multiplied by $b^2$. Separate training can produce different generators because optimization depends on the parameter coordinates. In a Gaussian experiment with known density, changing skip and observation variance together reduces MSE but worsens energy score and the estimate of negative log density from importance sampling. On seven annual Deep-ocean Assessment and Reporting of Tsunamis (DART) records, energy score differences at the same observation variance have both signs on four original records and favor the skip on three records from 2018. Probabilistic principal component analysis (PPCA) has lower mean energy scores than every neural DART fit. With the encoder and decoder kept the same, estimating observation variance from training reconstruction error can also worsen energy score. Assessing generation requires more than reconstruction error.</span> <span class="abstract-toggle" data-id="2610.37456">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.37456v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.37456v1) · [:fontawesome-brands-github: Code](https://github.com/magnaprog/vae-skip-reparam) · [:material-content-copy: BibTeX](bibtex/2610.37456.bib){ .bibtex-link }
    { .paper-links }

-   #### Scale-aware and spatially-localised optimisation of kilometre-scale MLWP models { #2610.37325 }

    *James Chappell, Sophie Luise Arana, Jack Bowyer, Alejandro Coca-Castro, Evangeline Corcoran et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.37325">Preserving fine-scale structure without degrading spatial accuracy is a central challenge for kilometre-scale machine-learning weather prediction (MLWP). We introduce a loss based on the dual-tree...</span><span class="abstract-full" id="full-2610.37325" hidden>Preserving fine-scale structure without degrading spatial accuracy is a central challenge for kilometre-scale machine-learning weather prediction (MLWP). We introduce a loss based on the dual-tree complex wavelet transform that constrains atmospheric fields jointly by scale, location, and orientation. We evaluate its performance using an experimental joint global--regional FastNet configuration, which produces 1.5 km forecasts over the UK alongside $\sim$31 km global forecasts. Relative to both mean-squared error (MSE) and an FFT-based spectral objective, the wavelet loss substantially improves spectral fidelity. Compared with MSE training, it maintains similar pointwise performance, improves neighbourhood-based verification, and produces sharper precipitation fields with more realistic intensity distributions. These results demonstrate that spatially localised wavelet constraints can preserve multiscale atmospheric structure while largely avoiding the loss of pointwise accuracy associated with globally supported spectral objectives.</span> <span class="abstract-toggle" data-id="2610.37325">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.37325v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.37325v1) · [:material-content-copy: BibTeX](bibtex/2610.37325.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a> <a class="md-tag" href="/explore/?t=km-scale" data-tag="km-scale">Km-scale</a>
    { .paper-tags }

</div>

