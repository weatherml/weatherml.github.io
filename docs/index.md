---
hide:
  - navigation
  - toc
title: weatherml
---

A collection of papers on AI for weather forecasting, climate modelling and atmospheric science.

<p class="page-meta" markdown="span">1468 papers · updated 2026-09-25 · <a href="feed.xml">:material-rss: RSS</a> · <a href="all_papers.bib" download>:material-download: BibTeX</a> · <a href="https://github.com/weatherml/weatherml.github.io/issues/new?template=suggest-paper.yml">:material-plus: Suggest a paper</a></p>

## Browse by Topic

<div class="grid cards topics" markdown>

-   [Global Models](papers/global-models/index.md) <span class="topic-count">347</span>
-   [Regional Models](papers/regional-models/index.md) <span class="topic-count">60</span>
-   [Nowcasting](papers/nowcasting/index.md) <span class="topic-count">97</span>
-   [Downscaling](papers/downscaling/index.md) <span class="topic-count">95</span>
-   [Post-processing](papers/post-processing/index.md) <span class="topic-count">25</span>
-   [Data Assimilation](papers/data-assimilation/index.md) <span class="topic-count">74</span>
-   [Climate Modeling](papers/climate-modeling/index.md) <span class="topic-count">289</span>
-   [Hydrology](papers/hydrology/index.md) <span class="topic-count">37</span>
-   [Ocean & Sea Ice](papers/ocean-sea-ice/index.md) <span class="topic-count">83</span>
-   [Air Quality & Composition](papers/air-quality-composition/index.md) <span class="topic-count">71</span>
-   [Remote Sensing](papers/remote-sensing/index.md) <span class="topic-count">98</span>
-   [Other](papers/other/index.md) <span class="topic-count">192</span>

</div>

## Recent Additions

<div class="grid cards" markdown data-search-exclude>

-   #### Analysis of trade-offs in urban heat mitigation using a Bayesian Optimization framework for an urban canopy layer model { #2609.25953 }

    *Rebekka Walter, Johanna Gelhaus, David Anton, Henning Wessles, Stephan Weber* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.25953">To mitigate the challenges of climate change and intensifying heat stress in urban areas, local adaptation strategies are discussed and introduced in cities worldwide. To understand processes and...</span><span class="abstract-full" id="full-2609.25953" hidden>To mitigate the challenges of climate change and intensifying heat stress in urban areas, local adaptation strategies are discussed and introduced in cities worldwide. To understand processes and potential trade-offs of these strategies a Bayesian optimization and surrogate modeling framework was employed to investigate urban parameter ranges of heat mitigation strategies with focus on three thermal metrics: daytime air temperature, Universal Thermal Climate Index (UTCI), and nighttime air temperature. Based on an urban street canyon configuration, it was shown that heat mitigation measures that reduce daytime air temperature and UTCI are often associated with higher nighttime temperatures. This results in a curved Pareto front that reflects the trade-off between daytime and nighttime thermal comfort. Under identical forcing conditions, different urban configurations, varying in geometry, vegetation, and surface characteristics, are shown to alter peak canyon air temperature by up to 5.2~$^\circ$C during the day and 2.6~$^\circ$C at night, while UTCI varies by up to 7.9~$^\circ$C, demonstrating that favorable urban configurations can substantially mitigate microclimatic heat stress. These findings suggest that combining multi-objective Bayesian optimization and surrogate modeling can help bridge the gap between computationally intensive climate simulations and practical decision-making in urban planning. An interactive visualization tool was developed to explore these trade-offs, making the often opposing relationships between urban parameters and the three thermal metrics directly accessible to planners.</span> <span class="abstract-toggle" data-id="2609.25953">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.25953v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.25953v1) · [:material-content-copy: BibTeX](bibtex/2609.25953.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a>
    { .paper-tags }

-   #### A dataset of one-dimensional idealized probabilistic fields { #2609.25720 }

    *Gregor Skok, Romain Pic* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.25720">Verification of probabilistic weather forecasts remains a crucial aspect of numerical weather prediction, as new AI-based models become more widely used alongside the more traditional physics-based...</span><span class="abstract-full" id="full-2609.25720" hidden>Verification of probabilistic weather forecasts remains a crucial aspect of numerical weather prediction, as new AI-based models become more widely used alongside the more traditional physics-based ensemble forecasting systems that continue to be developed and improved. We present a first-of-its-kind idealized probabilistic dataset composed of one-dimensional cases aimed at analyzing the behavior and properties of verification methods for probabilistic forecasts and comparing their behavior. It covers a wide range of probabilistic cases, such as constant, localized events, gradients, fronts, noisy, bimodal, and limiting cases. Moreover, the code associated with the dataset provides great flexibility for customizing the experiments it covers. The dataset represents the first building block of the more extensive comparison dataset of the Bridging The Gap project, which aims to facilitate the development and comparison of spatial verification methods for probabilistic forecasts.</span> <span class="abstract-toggle" data-id="2609.25720">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.25720v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.25720v1) · [:material-content-copy: BibTeX](bibtex/2609.25720.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=benchmarks-datasets" data-tag="benchmarks-datasets">Benchmarks & datasets</a>
    { .paper-tags }

-   #### West-WRF AI 2-km: High-Resolution Prediction of Integrated Vapor Transport and Precipitation { #2609.25512 }

    *Nazak Rouzegari, Vesta Afzali Gorooh, Agniv Sengupta, Phu Nguyen, Kuo-Lin Hsu, Amir AghaKouchak et al.* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.25512">We introduce a stretched-grid artificial intelligence (AI) weather forecasting model with 2-km resolution over the western United States and part of the Northeast Pacific and approximately 31-km...</span><span class="abstract-full" id="full-2609.25512" hidden>We introduce a stretched-grid artificial intelligence (AI) weather forecasting model with 2-km resolution over the western United States and part of the Northeast Pacific and approximately 31-km resolution elsewhere globally. Forecasting over the western U.S. is challenging because complex topography and atmospheric rivers (ARs) strongly influence orographic precipitation. West-WRF AI 2-km builds on a global model pretrained with a 40-year European Centre for Medium-Range Weather Forecasts Reanalysis v5 (ERA5) dataset and is fine-tuned with the Center for Western Weather and Water Extremes (CW3E) 2-km regional reanalysis to produce autoregressive 6-hourly forecasts of precipitation and integrated vapor transport (IVT). Forecasts are evaluated over winters 2020-2023 using gridded precipitation observations, rain gauges, and AR Reconnaissance dropsondes and are benchmarked against coarser-resolution AI forecasts and regional and global numerical weather prediction (NWP) systems. West-WRF AI 2-km reproduces observed precipitation-intensity distributions, retains fine-scale spectral variability, and produces sharper narrow coastal precipitation bands and localized, terrain-sensitive extremes. Its broader-scale performance remains comparable to coarser-resolution configurations while preserving large-scale skill despite higher resolution. Dropsonde verification shows lower errors and improved categorical skill at the most extreme IVT threshold. Overall, West-WRF AI 2-km provides its greatest value for localized precipitation extremes and intense AR-related moisture transport.</span> <span class="abstract-toggle" data-id="2609.25512">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.25512v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.25512v1) · [:material-content-copy: BibTeX](bibtex/2609.25512.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a> <a class="md-tag" href="/explore/?t=km-scale" data-tag="km-scale">Km-scale</a> <a class="md-tag" href="/explore/?t=quarter-degree" data-tag="quarter-degree">0.25°</a> <a class="md-tag" href="/explore/?t=6-hourly" data-tag="6-hourly">6-hourly</a>
    { .paper-tags }

-   #### FAST-ML: A Hybrid Physics-Machine Learning Framework for Tropical Cyclone Intensity Forecasting { #2609.25505 }

    *Shijie Xiao, Jonathan Lin, Thomas Ehrmann, Ali Sarhadi* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.25505">Rapid intensification (RI) remains one of the most consequential and difficult aspects of tropical cyclone (TC) forecasting. Although full-physics numerical weather prediction models can represent...</span><span class="abstract-full" id="full-2609.25505" hidden>Rapid intensification (RI) remains one of the most consequential and difficult aspects of tropical cyclone (TC) forecasting. Although full-physics numerical weather prediction models can represent the processes governing RI, resolving storm-environment interactions remains computationally expensive, while purely data-driven approaches often lack physical interpretability. We present FAST-ML, a hybrid framework that bridges data-driven efficiency with physical constraints. A physically informed dual-stream neural parameterization ingests 3D ERA5 fields to diagnose ventilation controls---environmental wind shear and mid-level entropy deficit. By optimizing these parameters end-to-end through a differentiable FAST intensity model, this architecture establishes a robust new paradigm for observation-driven parameter optimization, ensuring storm evolution remains strictly governed by thermodynamic principles. By better capturing the storm's continuous intensity evolution, FAST-ML improves upon its physical baseline, reducing ensemble CRPS across forecast lead times, with a reduction of approximately 31% at 60 h and nearly halving the RI false alarm ratio without sacrificing detection skill. In a 100-member ensemble configuration, FAST-ML produces intensity forecasts comparable to FNV3 for selected storms under the evaluated input configurations. Furthermore, zero-shot tests on selected Eastern Pacific storms provide encouraging evidence of cross-basin transferability. FAST-ML provides a modular intensity forecasting framework that can be coupled with externally supplied storm tracks and environmental fields. It demonstrates that observation-driven parameter learning within physically constrained dynamics simultaneously enhances accuracy, interpretability, and computational efficiency.</span> <span class="abstract-toggle" data-id="2609.25505">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.25505v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.25505v1) · [:material-content-copy: BibTeX](bibtex/2609.25505.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=physics-ml-hybrid" data-tag="physics-ml-hybrid">Physics–ML hybrid</a> <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a> <a class="md-tag" href="/explore/?t=interpretability" data-tag="interpretability">Interpretability</a> <a class="md-tag" href="/explore/?t=efficiency" data-tag="efficiency">Efficiency</a>
    { .paper-tags }

-   #### Learning Prognostic Variables for AI Convective Parameterizations via Symbolic Distillation { #2609.24882 }

    *Jurij Schönfeld, Tom Beucler, Julien Savre, Steven Sherwood, Veronika Eyring* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.24882">Hybrid AI-physics climate modeling aims to improve coarse (~100km-resolution) Earth system models by learning to parameterize subgrid processes from high-fidelity data. However, this so far mostly...</span><span class="abstract-full" id="full-2609.24882" hidden>Hybrid AI-physics climate modeling aims to improve coarse (~100km-resolution) Earth system models by learning to parameterize subgrid processes from high-fidelity data. However, this so far mostly involves local-in-time, diagnostic parameterizations, in which the subgrid state depends only on the current coarse state with no memory of previous states, which is unrealistic for processes such as convection that have intrinsic persistence. To address this, we enhance local-in-time parameterizations by learning prognostic variables that compactly carry important, additional past information where no explicit sub-grid information is available. First we compress past information into a low-dimensional latent space using an autoencoder, which then informs a neural network trained to parameterize targeted subgrid-scale processes. We then replace the autoencoder with symbolic equations that govern the time evolution of the latent variables, yielding additional prognostic memory variables that can be integrated alongside the resolved atmospheric state. We evaluate this approach on two systems: the Lorenz-96 model (online) and surface precipitation from high-resolution atmospheric simulations (offline). A forced multivariate linear ordinary differential equation recovers most of the added value achieved by the autoencoder-based approach in both experiments. Benchmarked against diagnostic parameterizations without memory, our memory-informed approach improves climate statistics and temporal structure, including a realistic diurnal cycle of tropical land precipitation.</span> <span class="abstract-toggle" data-id="2609.24882">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.24882v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.24882v1) · [:material-content-copy: BibTeX](bibtex/2609.24882.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a>
    { .paper-tags }

-   #### Inference of Unknown Dynamical Components Using Next Generation Reservoir Computing: From Chaotic Systems to Climate Data { #2609.24754 }

    *Jule Budnick, Andrew Keane, Serhiy Yanchuk* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.24754">We investigate next generation reservoir computing (NGRC) as a data-driven approach for inferring unseen components of dynamical systems. We compare NGRC with traditional reservoir computing (RC)...</span><span class="abstract-full" id="full-2609.24754" hidden>We investigate next generation reservoir computing (NGRC) as a data-driven approach for inferring unseen components of dynamical systems. We compare NGRC with traditional reservoir computing (RC) using the Lorenz and Rössler system, where two unknown components are inferred from one given component. For both systems, NGRC achieves accurate results while requiring fewer training data and less computational time than RC. We identified an inverse proportional behavior between the number of time-delayed steps needed for NGRC and the temporal resolution, indicating that the physical time span covered by the delay interval is an important factor in determining the required number of delayed steps. Finally, we apply NGRC to the observational climate data of ENSO (El Niño--Southern Oscillation) and infer one observable from the remaining variables. Despite the noise and complexity of the real-world data, the NGRC shows promising results. Our findings demonstrate the potential of NGRC for efficient inference of unseen components in both controlled dynamical systems and real-world data.</span> <span class="abstract-toggle" data-id="2609.24754">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.24754v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.24754v1) · [:material-content-copy: BibTeX](bibtex/2609.24754.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=subseasonal-to-seasonal" data-tag="subseasonal-to-seasonal">Subseasonal to seasonal</a>
    { .paper-tags }

-   #### Climate Variability Modulates the Impact of Price Spikes on Food Insecurity { #2609.24394 }

    *Jordi Cerdà-Bautista, Vasileios Sitokonstantinou, Homer Durand, Gherardo Varando, Michele Ronco et al.* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.24394">Climate variability influences whether a market disruption escalates into a food crisis, yet broad climate patterns like El Niño, tracked months before they alter hydro-climatic conditions, are still...</span><span class="abstract-full" id="full-2609.24394" hidden>Climate variability influences whether a market disruption escalates into a food crisis, yet broad climate patterns like El Niño, tracked months before they alter hydro-climatic conditions, are still not incorporated as an early-warning component in food-security responses. We address this gap by introducing sensitivity regimes, a stratification of regions by the direction and strength of their vegetation response to the El Niño Southern Oscillation, and using them to estimate how food price spikes affect acute food insecurity across sub-Saharan Africa. Integrating remote sensing, socioeconomic data, and causal machine learning, we find that in regions where ENSO systematically suppresses vegetation, a price spike raises the share of the population at acute risk by 5.4 percentage points in the following month. In regions where vegetation is unaffected by or positively linked to ENSO, the estimated effect is smaller (around 2 percentage points) and statistically insignificant. These results demonstrate that climate context is critical for understanding food security vulnerabilities. Sensitivity regimes can be combined with operational price-spike triggers to stage anticipatory action: the ENSO state flags vulnerable regions months ahead, and a pre-positioned response in those regions to a price spike would avert the largest jump in acute food insecurity.</span> <span class="abstract-toggle" data-id="2609.24394">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.24394v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.24394v1) · [:material-content-copy: BibTeX](bibtex/2609.24394.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=subseasonal-to-seasonal" data-tag="subseasonal-to-seasonal">Subseasonal to seasonal</a>
    { .paper-tags }

-   #### From Regional to Global: Transfer Learning for Atmospheric Transport Emulators { #2609.23838 }

    *Jeff Clark, Elena Fillola, Nawid Keshtmand, Raul Santos-Rodriguez, Matthew Rigby* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.23838">Greenhouse gas emissions estimates can be derived using inverse methods by combining atmospheric concentration observations with chemical transport models. The latter traditionally use physics-driven...</span><span class="abstract-full" id="full-2609.23838" hidden>Greenhouse gas emissions estimates can be derived using inverse methods by combining atmospheric concentration observations with chemical transport models. The latter traditionally use physics-driven simulators such as Lagrangian Particle Dispersion Models (LPDMs), which are expensive to run and do not scale well to modern satellites' high resolution data. Previously we developed a performant atmospheric transport emulator that approximates LPDM outputs ("footprints") over South America ~1,000X faster than the UK Met Office's LPDM. Expanding towards global emulation is not straightforward, as atmospheric transport is regionally heterogeneous. This paper evaluates spatial transferability capabilities of models across four world regions: South America, East Asia, South Asia, North Africa using both region-specific and multi-region models, and leave-one-region-out experiments. Regional differences are characterised in the context of input variable and output footprint distributions. This work builds intuition in cross-region generalisation and transfer learning, aiding regional performance towards efficient global emissions estimates.</span> <span class="abstract-toggle" data-id="2609.23838">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.23838v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.23838v1) · [:material-content-copy: BibTeX](bibtex/2609.23838.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a>
    { .paper-tags }

-   #### ClimTip-GML: A global bias-corrected and downscaled dataset for assessing impacts of climate tipping events { #2609.23149 }

    *Philipp Hess, Sebastian Bathiany, Lucas Ferreira Correa, Laura C. Jackson, Casey R. Patrizio et al.* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.23149">Assessing the impacts of future climate scenarios including tipping events of major Earth system components such as the Amazon rainforest (ARF) or the Atlantic meridional overturning circulation...</span><span class="abstract-full" id="full-2609.23149" hidden>Assessing the impacts of future climate scenarios including tipping events of major Earth system components such as the Amazon rainforest (ARF) or the Atlantic meridional overturning circulation (AMOC), requires accurate and high-resolution simulations. Here, we present ClimTip-GML, the first globally bias-corrected and downscaled climate dataset for impact assessment of large-scale tipping scenarios, comprising eight key variables at 0.25° spatial resolution from three general circulation models (GCMs): CESM1-CAM5, HadGEM3-GC31-MM, and MPI-ESM1-2-HR. The dataset includes 100-year-long climate simulations with preindustrial and historical conditions, as well as scenarios at a +2°C warming level with and without tipping transitions of the AMOC or ARF. We apply generative machine learning (GML) techniques trained on reanalysis data to bias-correct and downscale the GCMs in a manner that is physically consistent across space, time, and all eight variables. Comprehensive validation shows substantially reduced biases, improved small-scale spatial variability, multivariate correlations, and consistent long-term climate responses to the external forcing and tipping events. The results hence permit substantially improved impact assessments of tipping transitions of the ARF and AMOC, directly informing mitigation and adaptation policies.</span> <span class="abstract-toggle" data-id="2609.23149">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.23149v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.23149v1) · [:material-content-copy: BibTeX](bibtex/2609.23149.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a> <a class="md-tag" href="/explore/?t=quarter-degree" data-tag="quarter-degree">0.25°</a>
    { .paper-tags }

-   #### Diffusion-Based Super-Resolution of Adriatic Sea Oceanographic Fields { #2609.22574 }

    *Rajat Srivastava, Muhammad Sarmad, Emanuele Mele, Massimo Cafaro, Marco Pulimeno, Italo Epicoco* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.22574">High-resolution oceanographic fields are critical for resolving mesoscale and sub-mesoscale coastal dynamics, yet their generation remains constrained by both computational cost and observational...</span><span class="abstract-full" id="full-2609.22574" hidden>High-resolution oceanographic fields are critical for resolving mesoscale and sub-mesoscale coastal dynamics, yet their generation remains constrained by both computational cost and observational sparsity. We present OcDiffSR, a conditional denoising diffusion probabilistic model (DDPM) for oceanographic super-resolution that reconstructs high-resolution sea-surface fields from coarse-resolution reanalysis inputs. The model is trained on ten years (2011-2020) of paired low-resolution (GLORYS12V1, 1/12) and high-resolution (Mediterranean Sea Physics Reanalysis, Med MFC, 1/24) data, and evaluated on an independent test year (2009) over the Adriatic Sea. OcDiffSR employs a conditional U-Net augmented with multi-scale low-resolution encoders, cross-attention bottleneck layers, and sinusoidal seasonal embeddings via Feature-wise Linear Modulation (FiLM), enabling joint super-resolution of sea-surface temperature (SST), salinity (SSS), and horizontal velocity components with visually coherent circulation patterns. Benchmarked against bilinear interpolation and the state-of-the-art residual diffusion model CorrDiff, OcDiffSR achieves substantially lower reconstruction errors for scalar fields (RMSESST=0.477 C, RMSESSS=0.346 psu), near-unity Pearson correlation (PCC >= 0.999), and high structural similarity (SSIM >= 0.964). For dynamical vector fields, OcDiffSR outperforms both baselines in absolute error and spatial coherence, though moderate correlation (PCC = 0.64) reflects the intrinsic stochasticity of oceanic velocity fields. Daily and monthly evaluations confirm temporal robustness across all seasons. These results establish OcDiffSR as a reliable framework for high-fidelity oceanographic downscaling and reanalysis enhancement, producing fields that are visually consistent with known ocean dynamics.</span> <span class="abstract-toggle" data-id="2609.22574">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.22574v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.22574v1) · [:material-content-copy: BibTeX](bibtex/2609.22574.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a> <a class="md-tag" href="/explore/?t=daily" data-tag="daily">Daily</a> <a class="md-tag" href="/explore/?t=monthly" data-tag="monthly">Monthly</a>
    { .paper-tags }

-   #### Spatial Aggregation of ROC and Precision-Recall Curves { #2609.19517 }

    *Romain Pic, Zhongwei Zhang, Sebastian Engelke, Johanna Ziegel* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.19517">Receiver Operating Characteristic (ROC) and Precision-Recall (PR) curves are widely used to assess the discrimination ability of forecasts for binary events, such as threshold exceedances or warnings...</span><span class="abstract-full" id="full-2609.19517" hidden>Receiver Operating Characteristic (ROC) and Precision-Recall (PR) curves are widely used to assess the discrimination ability of forecasts for binary events, such as threshold exceedances or warnings of extreme events. In weather forecasting, forecasts are provided as spatial fields, yielding location-wise ROC and PR curves that are often aggregated to facilitate comparison. However, the effect of the aggregation strategy on performance assessment remains poorly understood.   We investigate how different aggregation strategies for ROC and PR curves affect the assessment of discrimination ability. In particular, we identify conditions under which aggregation strategies satisfy two desirable properties for fair comparison: preservation of dominance between forecasts and preservation of concavity or achievability of the curves. We obtain sufficient conditions and propose two strategies satisfying them. They are compared with existing strategies from the literature, and we analyze their properties and highlight potential pitfalls that may lead to misleading interpretations. Based on these findings, we provide practical guidelines for the interpretation of aggregated ROC and PR curves. The proposed framework is illustrated with AI-based global weather forecasts, showing how different aggregation strategies can yield different rankings of competing forecasts.</span> <span class="abstract-toggle" data-id="2609.19517">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.19517v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.19517v1) · [:fontawesome-brands-github: Code](https://github.com/pic-romain/spatial-agg-roc-pr) · [:material-content-copy: BibTeX](bibtex/2609.19517.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a>
    { .paper-tags }

-   #### A more predictable Madden-Julian Oscillation index derived from Koopman spectral analysis { #2609.19435 }

    *Claire Valva, Edwin P. Gerber* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.19435">The Madden-Julian oscillation (MJO) is a major source of subseasonal-to-seasonal (S2S) predictability. The MJO is commonly defined and tracked with indices such as the Real-time Multivariate MJO...</span><span class="abstract-full" id="full-2609.19435" hidden>The Madden-Julian oscillation (MJO) is a major source of subseasonal-to-seasonal (S2S) predictability. The MJO is commonly defined and tracked with indices such as the Real-time Multivariate MJO (RMM) index. Although the RMM provides a useful description of the MJO, its evolution can be noisy and difficult to predict. We define an MJO index using a data-driven approximation of the Koopman operator. The Koopman index captures similar tropical circulation and convection patterns to the RMM but evolves more smoothly and predictably. Skillful prediction extends to 46 days for the Koopman index compared to 11 days for the RMM under the same prediction framework. While this new approach does not recover the RMM as well as operational S2S models, which provide skillful forecasts up to 35 days, the Koopman index could complement existing MJO diagnostics in evaluating and developing extended-range forecast systems.</span> <span class="abstract-toggle" data-id="2609.19435">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.19435v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.19435v1) · [:material-content-copy: BibTeX](bibtex/2609.19435.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=subseasonal-to-seasonal" data-tag="subseasonal-to-seasonal">Subseasonal to seasonal</a>
    { .paper-tags }

-   #### Butterfly Effect and the Kinetic Energy Cascade in Probabilistic Machine Learning Weather Prediction Models { #2609.18489 }

    *Jiakai Chen, Joel Oskarsson, Simon Driscoll, Sebastian Schemm* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.18489">This study analyses kinetic energy (KE) spectra, difference kinetic energy (DKE) spectra, and signatures of KE transfer across spatial scales in four state-of-the-art probabilistic machine learning...</span><span class="abstract-full" id="full-2609.18489" hidden>This study analyses kinetic energy (KE) spectra, difference kinetic energy (DKE) spectra, and signatures of KE transfer across spatial scales in four state-of-the-art probabilistic machine learning weather prediction (MLWP) models: NeuralGCM-ENS, FourCastNet 3, AIFS-ENS, and GenCast. Results are compared with those from the physics-based numerical weather prediction model IFS-ENS. While NeuralGCM-ENS successfully reproduces the expected upscale transfer of KE, noise injection at its encoder stage underestimates mesoscale KE. Conversely, AIFS-ENS, GenCast, and FourCastNet 3 produce realistic KE spectral magnitudes but do not capture the expected upscale transfer of KE. In particular, AIFS-ENS and GenCast, which employ spatially uncorrelated stochastic perturbations, exhibit enhanced accumulation of KE at high wavenumbers. All examined models exhibit upscale error growth, reflected by the progressive shift of the DKE spectral peak toward larger wavelengths over time. However, the MLWP models struggle to reproduce the rapid initial growth of ensemble spread at small spatial scales associated with the butterfly effect. The results show that MLWP models can misrepresent the known scale transfer of kinetic energy despite producing skilful weather forecasts.</span> <span class="abstract-toggle" data-id="2609.18489">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.18489v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.18489v1) · [:material-content-copy: BibTeX](bibtex/2609.18489.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=physics-ml-hybrid" data-tag="physics-ml-hybrid">Physics–ML hybrid</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### Every Fixed Metric Has a Blind Spot: A Learned Atmospheric Critic for Scoring Forecast Realism { #2609.18381 }

    *Younes Elberkennou, Dmitri Demler, Thierry Meier, Luca Rispoli, Fanny Lehmann, Joel Oskarsson* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.18381">Despite their high accuracy on point-wise metrics, machine learning weather forecasting models can exhibit different failure modes such as blurring, periodic irregularities, and other unphysical...</span><span class="abstract-full" id="full-2609.18381" hidden>Despite their high accuracy on point-wise metrics, machine learning weather forecasting models can exhibit different failure modes such as blurring, periodic irregularities, and other unphysical spatial artifacts. This has motivated a variety of metrics to detect known failure cases. Existing metrics fix a representation or transformation in advance, and that choice limits the artifacts they can detect. We propose to train a discriminator for separating reference data from the model's output, and using its output logit to obtain a divergence-like realism score. The discriminator learns whatever separates the model's fields from real weather, adapting to whichever failure mode that model exhibits. We compare our learned atmospheric critic to existing metrics using various synthetic corruptions applied to ERA5 reanalysis data. Our method successfully identifies the corruptions and ranks their severity, while existing metrics fail on at least one corruption. Additionally, we evaluate forecasts from real weather models, and find that the realism score degrades with longer lead times and the metric generally assigns higher realism to numerical models than to machine learning models.</span> <span class="abstract-toggle" data-id="2609.18381">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.18381v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.18381v1) · [:material-content-copy: BibTeX](bibtex/2609.18381.bib){ .bibtex-link }
    { .paper-links }

-   #### IRENE: A Convolutional GRU Ensemble Model for Radar Precipitation Nowcasting over Italy { #2609.17175 }

    *Alessandro Camilletti, Gabriele Franch, Elena Tomasi, Marco Cristoforetti* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.17175">We present IRENE (Italian Radar Ensemble Nowcasting Experiment), a deep learning model for probabilistic short-range precipitation nowcasting over the Italian domain at 1 km spatial and 5 min...</span><span class="abstract-full" id="full-2609.17175" hidden>We present IRENE (Italian Radar Ensemble Nowcasting Experiment), a deep learning model for probabilistic short-range precipitation nowcasting over the Italian domain at 1 km spatial and 5 min temporal resolution. IRENE adopts an encoder--forecaster architecture built on multi-scale Convolutional Gated Recurrent Units (ConvGRUs), trained on the national radar composite produced by the Italian Civil Protection Department (DPC). An importance-sampling scheme focuses training on precipitation-relevant events, while the almost-fair Continuous Ranked Probability Score (afCRPS) is adopted as the primary probabilistic loss function. Two additional training configurations are proposed: an adversarial (GAN) variant, IRENE-GAN, designed to improve the spatial sharpness of the generated forecasts, and a spectrally constrained variant, IRENE-GAN-RAPSD, in which the adversarial objective is complemented by an explicit penalty on the radially averaged power spectral density. The three configurations are evaluated against the stochastic extrapolation method STEPS and the pre-trained deep learning model DGMR. All IRENE configurations attain a lower Continuous Ranked Probability Score than both benchmarks at every lead time and rank histograms closer to uniformity, indicating better probabilistic skill and ensemble calibration. In terms of ensemble-mean mean absolute error the advantage is confined to the first 90 min, beyond which the strongly damped DGMR fields and, to a lesser extent, STEPS become competitive. Spectral analysis shows that the adversarial training removes the progressive loss of small-scale variance exhibited by IRENE, at the cost of an excess of fine-scale power at long lead times that the spectral penalty only partially controls.</span> <span class="abstract-toggle" data-id="2609.17175">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.17175v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.17175v1) · [:material-content-copy: BibTeX](bibtex/2609.17175.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=gans" data-tag="gans">GANs</a> <a class="md-tag" href="/explore/?t=recurrent-networks" data-tag="recurrent-networks">Recurrent networks</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a> <a class="md-tag" href="/explore/?t=sub-hourly" data-tag="sub-hourly">Sub-hourly</a>
    { .paper-tags }

-   #### Predictability-Guided Multiscale Probabilistic Forecasting of Wind Direction under Extreme Shear { #2609.16707 }

    *Hailong Shu* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.16707">Accurate multi-horizon wind direction forecasting is critical for turbine yaw control and grid security. Rapid directional shear (turning $\ge 90^\circ$) challenges models via non-Euclidean geometry...</span><span class="abstract-full" id="full-2609.16707" hidden>Accurate multi-horizon wind direction forecasting is critical for turbine yaw control and grid security. Rapid directional shear (turning $\ge 90^\circ$) challenges models via non-Euclidean geometry on $S^1$, multiscale dynamics, and regime-dependent uncertainty. Conventional discrete models and foundation models suffer from mid-frequency phase lag and turning misalignments. We show that directional predictability decays at disparate rates across frequency subbands, rendering monolithic mechanisms suboptimal. We propose a predictability-guided paradigm: slow synoptic drift $\to$ deterministic regression; intermediate turning $\to$ continuous latent differential flows; unresolved turbulence $\to$ conditional residual diffusion; followed by causal recalibration. On a 10,000-sequence multi-year benchmark, our framework maintains calm-weather accuracy (Test MCE $38.48^\circ$) while reducing extreme-turning error (Case 1 MCE $60.69^\circ$ vs $70.42^\circ$ for zero-shot foundation models). The circular CRPS reaches $22.36^\circ$, with 93.88% coverage at nominal 95% (91.01% out-of-distribution). Density estimation further reveals near-antipodal bimodal structure under severe shear (13.39%--15.43% tail mass $\ge 135^\circ$), exposing a geometric bound where single-center calibration under-covers (81.56%), motivating multimodal circular manifold learning.</span> <span class="abstract-toggle" data-id="2609.16707">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.16707v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.16707v1) · [:material-content-copy: BibTeX](bibtex/2609.16707.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=foundation-models" data-tag="foundation-models">Foundation models</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### A Self-Diagnosing Structural Error-Aware Parameter Estimation Method for Earth System Models { #2609.16210 }

    *Qingyuan Yang, Addisu G Semie, Brian Medeiros, Gregory S Elsaesser, Da Fan, Wayne Chuang* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.16210">We propose a fully automated, structural error-aware, interpretable climate model parameter estimation method that leverages Perturbed Parameter Ensembles (PPEs). It is based on history matching and...</span><span class="abstract-full" id="full-2609.16210" hidden>We propose a fully automated, structural error-aware, interpretable climate model parameter estimation method that leverages Perturbed Parameter Ensembles (PPEs). It is based on history matching and aligns with an increasingly-used iterative simulation-emulation-calibration methodology. The method is motivated by the negative impacts of structural error and emulator and observational uncertainties on climate model parameter estimation efforts, as well as the problems associated with sparsely-sampled PPEs. To address these challenges, the method explicitly builds simpler emulators that avoid overfitting, detect structural error, avoids compensating for structural error through inflated mismatch tolerances, and sequentially excludes structurally inconsistent variables for parameter estimation. The method decomposes the high-dimensional calibration problem into linked low-dimensional subproblems, and integrates their constraints to reconstruct the jointly plausible region of the full parameter space. The method is applied to a 100-member PPE with 34 perturbed parameters generated by a version of CAM6 with machine learning-based warm rain microphysics parameterization. Through iterative application, the method greatly reduces the ensemble spread and improves the matching between simulated and observed zonal climatologies. The method also finds ensemble members that outperform the default CAM6 configuration in root mean square error across multiple diagnostics. Controlled experiments demonstrate that overly-conservative emulator uncertainty could lead to neglect of informative observations, and tolerance of the structural error, in the context of this method, biases the estimated parameters toward compensating for structural error. Our work also emphasizes the value of interpretability for diagnosing structural error and informing parameter estimation in PPE-based calibration.</span> <span class="abstract-toggle" data-id="2609.16210">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.16210v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.16210v1) · [:material-content-copy: BibTeX](bibtex/2609.16210.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=interpretability" data-tag="interpretability">Interpretability</a>
    { .paper-tags }

-   #### Aries: A Proprietary Medium-Range Weather Prediction Model for the Energy Industry { #2609.13292 }

    *Lukas Hedegaard Morsing, Arian Bakhtiarnia, Jonas Lynge Olesen, Tómas Bragi Björnsson Leth et al.* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.13292">Medium-range weather forecasting underpins operational and planning decisions across the energy industry. Developing competitive weather models was once the domain of national meteorological centers,...</span><span class="abstract-full" id="full-2609.13292" hidden>Medium-range weather forecasting underpins operational and planning decisions across the energy industry. Developing competitive weather models was once the domain of national meteorological centers, but recent advances in machine-learned weather prediction (MLWP) have opened the field to industry. We present Aries, a SwinTransformer-based MLWP model developed at InCommodities. Aries is trained on ERA5 reanalysis data at 0.25°{} resolution, predicting 74 prognostic and 11 diagnostic atmospheric variables. We evaluate the model on 2025 ECMWF Analysis initializations, ensuring a recent and strictly out-of-sample test period for all models compared. On 10-metre wind speed, Aries outperforms both ECMWF HRES and AIFS in terms of RMSE for lead times up to four days, while on 2-metre temperature it achieves RMSE on par with AIFS operational. These results demonstrate that proprietary development of competitive weather models is technically viable, supporting a broader set of forecasts available for operational and planning applications in the energy industry.</span> <span class="abstract-toggle" data-id="2609.13292">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.13292v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.13292v1) · [:material-content-copy: BibTeX](bibtex/2609.13292.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=transformers" data-tag="transformers">Transformers</a>
    { .paper-tags }

-   #### A Physics--ML Multi-Fidelity Strategy for Earth System Model Parameter Optimization: A QG Proof-of-Concept { #2609.13275 }

    *Abdullah A. Fahad, Manmeet Singh, Donifan Barahona, Anton Darmenov, Andrea Molod* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.13275">Earth System Models rely on tunable subgrid-scale parameterizations, but optimizing these parameters is computationally expensive, particularly when nonlinear interactions require many simulations....</span><span class="abstract-full" id="full-2609.13275" hidden>Earth System Models rely on tunable subgrid-scale parameterizations, but optimizing these parameters is computationally expensive, particularly when nonlinear interactions require many simulations. We present a hybrid Physics-ML multi-fidelity framework that combines Green's Function Optimization (GFO) with Gaussian Process or Neural Network surrogate optimization. Using a quasi-geostrophic turbulence model, GFO first ranks parameter sensitivities in normalized coordinates and selects a reduced active subset. Nonlinear surrogates then explore this subset using inexpensive 30-day simulations before refining promising candidates with 180-day simulations. Across seven strategies and a 35-member ensemble, GFO-MultiGP and GFO-MultiNN achieved mean improvements of 64.6 percent and 65.2 percent, respectively, while reaching practical saturation after 3,060 and 2,520 simulation-days. The corresponding standalone GP and NN achieved 61.1 percent and 42.0 percent improvements and required 7,740 and 6,660 simulation-days. These results demonstrate an end-to-end sample-efficiency advantage for the tested hybrid pipelines. Because screening, dimensionality reduction, initialization, and fidelity scheduling change simultaneously, their individual contributions are not isolated.</span> <span class="abstract-toggle" data-id="2609.13275">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.13275v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.13275v1) · [:material-content-copy: BibTeX](bibtex/2609.13275.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=physics-ml-hybrid" data-tag="physics-ml-hybrid">Physics–ML hybrid</a> <a class="md-tag" href="/explore/?t=classical-ml" data-tag="classical-ml">Classical ML</a>
    { .paper-tags }

-   #### 4D Parallelism Unlocks Exascale Bayesian Neural Networks for High-Fidelity Atmospheric Modeling { #2609.12815 }

    *Deifilia Kieckhefen, Juan Pedro Gutiérrez Hermosillo Muriedas, Lars Helge Heyen, Mathis Bode et al.* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.12815">We present BEAST, the first-ever Bayesian Swin Transformer for atmospheric forecasting on 0.25$^\circ$ global resolution able to accurately quantify both aleatoric and epistemic uncertainty. To...</span><span class="abstract-full" id="full-2609.12815" hidden>We present BEAST, the first-ever Bayesian Swin Transformer for atmospheric forecasting on 0.25$^\circ$ global resolution able to accurately quantify both aleatoric and epistemic uncertainty. To overcome the associated computational bottlenecks, we devise an orthogonal 4D-parallelization scheme that introduces a unique domain-tensor-parallelism strategy and a novel uncertainty parallel method, enabling us to fully leverage GPU capacity and efficiently scale model training. For a 2.4-billion-parameter model, we achieve a peak performance of 3.96 EFLOP/s on 20,480 NVIDIA GH200 GPUs on the JUPITER supercomputer. We train BEAST as a 700-million-parameter model with 96 random weight samples on 384 nodes on 40 years of data for nearly one million gradient updates. This model achieves predictive skill scores competitive with state-of-the-art probabilistic atmospheric AI models and numerical models, and can predict extreme events with exceptional skill, while generating large ensembles 3 to 4 times faster than the current-best AI model. Our contribution unlocks the potential of high-fidelity uncertainty quantification in atmospheric AI models, heralding a new era for AI-based models in climate and Earth system sciences.</span> <span class="abstract-toggle" data-id="2609.12815">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.12815v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.12815v1) · [:material-content-copy: BibTeX](bibtex/2609.12815.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=transformers" data-tag="transformers">Transformers</a> <a class="md-tag" href="/explore/?t=quarter-degree" data-tag="quarter-degree">0.25°</a>
    { .paper-tags }

-   #### Optimizing Geoengineering Interventions Using Differentiable Climate Models { #2609.12528 }

    *Pulkit Dubey, Dorian S. Abbot, Ashesh Chattopadhyay* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.12528">The deployment of a geoengineering program to cool Earth's climate may be imminent. It is crucial that tools be developed to ensure that such a program would achieve its objectives while minimizing...</span><span class="abstract-full" id="full-2609.12528" hidden>The deployment of a geoengineering program to cool Earth's climate may be imminent. It is crucial that tools be developed to ensure that such a program would achieve its objectives while minimizing disruption. Here we exploit recently developed differentiable atmospheric models to demonstrate a novel geoengineering control strategy. In the differentiable primitive-equation atmospheric model JAX-GCM we impose a uniform $+4$\,K ocean warming and ask what pattern of sea-surface temperature cooling -- in five ocean-masked zonal bands of prescribed SST forcings whose amplitudes are free -- returns land near-surface air temperature closest to the model's own unwarmed climatology. This idealized set-up represents a cooling pattern that could be delivered physically either by marine cloud brightening or stratospheric aerosol injection. Gradients through chaotic dynamics decorrelate from the true sensitivity beyond the Lyapunov horizon, so we optimize greedily over segments of 8 to 14 days, following receding-horizon control. The learned strategy removes $92.3 \pm 0.4\%$ of the realized land warming across a ten-member ensemble of two-year rollouts, and a three-year run sustains it. If we use the spatial pattern of land temperature as the optimization objective, the distributions of precipitation, evaporation, and specific humidity over land are restored as well, even though they are not included in the objective function. The learned strategy from JAX-GCM replayed in the AI emulators LUCIE and NeuralGCM without re-optimization is successful, suggesting robustness. These promising results demonstrate a strategy for designing optimal climate interventions that can be applied broadly for geoengineering scenarios under consideration.</span> <span class="abstract-toggle" data-id="2609.12528">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.12528v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.12528v1) · [:material-content-copy: BibTeX](bibtex/2609.12528.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=physics-ml-hybrid" data-tag="physics-ml-hybrid">Physics–ML hybrid</a>
    { .paper-tags }

-   #### Automated Detection and Structuring of Social Tipping Point Evidence in Climate related Documents: A Modular AI Framework { #2609.12254 }

    *Kavindu Perera, Mohammad Abaeiani, Ekaterina Gilman, Lauri Loven, Mourad Oussalah, Tassos Kanellos et al.* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.12254">The climate literature has grown faster than review teams can read it. That gap matters most for a concept like the environmental social tipping point, the threshold at which a small change triggers...</span><span class="abstract-full" id="full-2609.12254" hidden>The climate literature has grown faster than review teams can read it. That gap matters most for a concept like the environmental social tipping point, the threshold at which a small change triggers rapid, self-reinforcing change in a social system. Evidence of this kind of shift is usually contained in one or two paragraphs within a longer document. As a result, existing text mining tools-which categorize entire documents by topic or highlight isolated claims-leave an expanding set of important evidence without any systematic method for discovery or organization. This paper presents an open and modular transformer-based framework that detects and structures social tipping point evidence at the passage level. The framework joins five components into a single deployable workflow: a DistilBERT boundary splitter for segmentation, an iteratively augmented RoBERTa classifier for detection, a Mistral 7B model that rewrites each detected passage for clarity, a LLaMA 3.2 3B model that rates the passage against five published social tipping point criteria, and a Milvus vector store for semantic retrieval. The system is wrapped in a Streamlit interface backed by MinIO object storage. Evaluated on a 163-passage benchmark labelled by GPT-4.1 and a 51-passage set reviewed by experts, the splitter surpassed three competing methods on a nine-metric composite score (6.137). The tuned RoBERTa model achieved 71.4 percent accuracy with a Cohen's kappa of 0.337 on the full benchmark, and 87.5 percent accuracy with a kappa of 0.742 on passages with labels, outperforming both a climate-focused model and untuned language models.</span> <span class="abstract-toggle" data-id="2609.12254">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.12254v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.12254v1) · [:material-content-copy: BibTeX](bibtex/2609.12254.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=transformers" data-tag="transformers">Transformers</a>
    { .paper-tags }

-   #### WIND-Bench: A Benchmark Dataset for In-Situ Near-Surface Wind Speed Observations Across the Conterminous United States { #2609.12228 }

    *Kyla Bazlen, Grant Buster, Brandon Benton, Lauren North, Ansley Baring, David D. Turner et al.* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.12228">Accurate wind forecasts are essential for operational decision-making and public safety, yet forecasts tend to miss near-surface high wind speeds in complex terrain. In response, advances in machine...</span><span class="abstract-full" id="full-2609.12228" hidden>Accurate wind forecasts are essential for operational decision-making and public safety, yet forecasts tend to miss near-surface high wind speeds in complex terrain. In response, advances in machine learning (ML) weather prediction methods have demonstrated the ability to improve forecast skill beyond traditional numerical weather prediction (NWP) models. However, the absence of a benchmark dataset to evaluate NWP and ML models with sufficient, quality-controlled wind speed observations in complex terrain poses challenges to the development and intercomparison of high-quality surface wind forecasts across the Conterminous United States (CONUS). We develop the Wind IN-situ Data Benchmark (WIND-Bench), a benchmark dataset from in-situ observations in the Meteorological Assimilation Data Ingest System (MADIS) observational network. WIND-Bench integrates multiple sensor networks with quality control that distinguishes sensor failures from high-wind conditions, using a framework that validates observations against forecasts from the National Oceanic and Atmospheric Administration (NOAA) High-Resolution Rapid Refresh (HRRR) model. WIND-Bench provides a standardized benchmark for evaluating ML and NWP models and for quantifying forecast skill, accelerating the development, evaluation, and operational deployment of skilled near-surface wind forecasts.</span> <span class="abstract-toggle" data-id="2609.12228">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.12228v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.12228v1) · [:material-content-copy: BibTeX](bibtex/2609.12228.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=benchmarks-datasets" data-tag="benchmarks-datasets">Benchmarks & datasets</a> <a class="md-tag" href="/explore/?t=evaluation" data-tag="evaluation">Evaluation</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a> <a class="md-tag" href="/explore/?t=station-point" data-tag="station-point">Station / point</a>
    { .paper-tags }

-   #### Stress-Testing Dynamical and Generative Downscaling Using Subseasonal Extreme Precipitation Forecasts { #2609.11696 }

    *Mauricio Lima, Marika Koukoula, Romain Pilon, Monika Feldmann, Erwan Koch, Daniela I. V. Domeisen et al.* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.11696">Coarse spatial resolution limits the ability of subseasonal prediction models to resolve extreme precipitation. Downscaling with either dynamical or deep generative models can overcome this issue,...</span><span class="abstract-full" id="full-2609.11696" hidden>Coarse spatial resolution limits the ability of subseasonal prediction models to resolve extreme precipitation. Downscaling with either dynamical or deep generative models can overcome this issue, but the comparative performance of these models for extremes across different atmospheric regimes remains poorly understood. In this work, we evaluate the Weather Research and Forecasting (WRF) model against a diffusion-based generative model by downscaling two physically distinct, extreme precipitation events up to lead times of 3 weeks. For a fair comparison with WRF, which can downscale boundary conditions from different driving models without model-specific training, the diffusion model is trained in an unpaired fashion. Both approaches improve upon the raw European Centre for Medium-Range Weather Forecasts forecasts, in comparison to fused rain gauge-radar observations in Switzerland (CombiPrecip), but exhibit regime-dependent strengths. WRF achieves the highest probabilistic skill for a multicell, non-stationary event. Conversely, the diffusion model is more consistent across different performance metrics for the two events, outperforming WRF in a more stationary supercell event. These results demonstrate that explicit dynamical modeling can add value for specific precipitation events for subseasonal lead times, and that generative downscaling adds value more broadly in different situations.</span> <span class="abstract-toggle" data-id="2609.11696">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.11696v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.11696v1) · [:material-content-copy: BibTeX](bibtex/2609.11696.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=extremes" data-tag="extremes">Extremes</a> <a class="md-tag" href="/explore/?t=subseasonal-to-seasonal" data-tag="subseasonal-to-seasonal">Subseasonal to seasonal</a>
    { .paper-tags }

</div>

