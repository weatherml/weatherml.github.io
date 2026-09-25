---
hide:
  - navigation
title: Deep Learning in Weather
---

A collection of papers on deep learning and machine learning applied to weather forecasting, climate modeling, and atmospheric science.

*Last updated: 2026-09-25*

## Recent Additions

<div class="grid cards" markdown>

-   #### Analysis of trade-offs in urban heat mitigation using a Bayesian Optimization framework for an urban canopy layer model

    ---

    *Rebekka Walter, Johanna Gelhaus, David Anton, Henning Wessles, Stephan Weber* · 2026

    <span class="abstract-snippet" id="snip-2609.25953">To mitigate the challenges of climate change and intensifying heat stress in urban areas, local adaptation strategies are discussed and introduced in cities worldwide. To understand processes and...</span><span class="abstract-full" id="full-2609.25953" hidden>To mitigate the challenges of climate change and intensifying heat stress in urban areas, local adaptation strategies are discussed and introduced in cities worldwide. To understand processes and potential trade-offs of these strategies a Bayesian optimization and surrogate modeling framework was employed to investigate urban parameter ranges of heat mitigation strategies with focus on three thermal metrics: daytime air temperature, Universal Thermal Climate Index (UTCI), and nighttime air temperature. Based on an urban street canyon configuration, it was shown that heat mitigation measures that reduce daytime air temperature and UTCI are often associated with higher nighttime temperatures. This results in a curved Pareto front that reflects the trade-off between daytime and nighttime thermal comfort. Under identical forcing conditions, different urban configurations, varying in geometry, vegetation, and surface characteristics, are shown to alter peak canyon air temperature by up to 5.2~$^\circ$C during the day and 2.6~$^\circ$C at night, while UTCI varies by up to 7.9~$^\circ$C, demonstrating that favorable urban configurations can substantially mitigate microclimatic heat stress. These findings suggest that combining multi-objective Bayesian optimization and surrogate modeling can help bridge the gap between computationally intensive climate simulations and practical decision-making in urban planning. An interactive visualization tool was developed to explore these trade-offs, making the often opposing relationships between urban parameters and the three thermal metrics directly accessible to planners.</span> <span class="abstract-toggle" data-id="2609.25953">more</span>

    [:material-file-document: 2609.25953](https://arxiv.org/abs/2609.25953v1) · [:material-content-copy: BibTeX](bibtex/2609.25953.bib){ .bibtex-link }

    <span class="md-tag">probabilistic</span>

-   #### A dataset of one-dimensional idealized probabilistic fields

    ---

    *Gregor Skok, Romain Pic* · 2026

    <span class="abstract-snippet" id="snip-2609.25720">Verification of probabilistic weather forecasts remains a crucial aspect of numerical weather prediction, as new AI-based models become more widely used alongside the more traditional physics-based...</span><span class="abstract-full" id="full-2609.25720" hidden>Verification of probabilistic weather forecasts remains a crucial aspect of numerical weather prediction, as new AI-based models become more widely used alongside the more traditional physics-based ensemble forecasting systems that continue to be developed and improved. We present a first-of-its-kind idealized probabilistic dataset composed of one-dimensional cases aimed at analyzing the behavior and properties of verification methods for probabilistic forecasts and comparing their behavior. It covers a wide range of probabilistic cases, such as constant, localized events, gradients, fronts, noisy, bimodal, and limiting cases. Moreover, the code associated with the dataset provides great flexibility for customizing the experiments it covers. The dataset represents the first building block of the more extensive comparison dataset of the Bridging The Gap project, which aims to facilitate the development and comparison of spatial verification methods for probabilistic forecasts.</span> <span class="abstract-toggle" data-id="2609.25720">more</span>

    [:material-file-document: 2609.25720](https://arxiv.org/abs/2609.25720v1) · [:material-content-copy: BibTeX](bibtex/2609.25720.bib){ .bibtex-link }

    <span class="md-tag">physics-informed</span> <span class="md-tag">probabilistic</span>

-   #### West-WRF AI 2-km: High-Resolution Prediction of Integrated Vapor Transport and Precipitation

    ---

    *Nazak Rouzegari, Vesta Afzali Gorooh, Agniv Sengupta, Phu Nguyen, Kuo-Lin Hsu, Amir AghaKouchak et al.* · 2026

    <span class="abstract-snippet" id="snip-2609.25512">We introduce a stretched-grid artificial intelligence (AI) weather forecasting model with 2-km resolution over the western United States and part of the Northeast Pacific and approximately 31-km...</span><span class="abstract-full" id="full-2609.25512" hidden>We introduce a stretched-grid artificial intelligence (AI) weather forecasting model with 2-km resolution over the western United States and part of the Northeast Pacific and approximately 31-km resolution elsewhere globally. Forecasting over the western U.S. is challenging because complex topography and atmospheric rivers (ARs) strongly influence orographic precipitation. West-WRF AI 2-km builds on a global model pretrained with a 40-year European Centre for Medium-Range Weather Forecasts Reanalysis v5 (ERA5) dataset and is fine-tuned with the Center for Western Weather and Water Extremes (CW3E) 2-km regional reanalysis to produce autoregressive 6-hourly forecasts of precipitation and integrated vapor transport (IVT). Forecasts are evaluated over winters 2020-2023 using gridded precipitation observations, rain gauges, and AR Reconnaissance dropsondes and are benchmarked against coarser-resolution AI forecasts and regional and global numerical weather prediction (NWP) systems. West-WRF AI 2-km reproduces observed precipitation-intensity distributions, retains fine-scale spectral variability, and produces sharper narrow coastal precipitation bands and localized, terrain-sensitive extremes. Its broader-scale performance remains comparable to coarser-resolution configurations while preserving large-scale skill despite higher resolution. Dropsonde verification shows lower errors and improved categorical skill at the most extreme IVT threshold. Overall, West-WRF AI 2-km provides its greatest value for localized precipitation extremes and intense AR-related moisture transport.</span> <span class="abstract-toggle" data-id="2609.25512">more</span>

    [:material-file-document: 2609.25512](https://arxiv.org/abs/2609.25512v1) · [:material-content-copy: BibTeX](bibtex/2609.25512.bib){ .bibtex-link }

-   #### FAST-ML: A Hybrid Physics-Machine Learning Framework for Tropical Cyclone Intensity Forecasting

    ---

    *Shijie Xiao, Jonathan Lin, Thomas Ehrmann, Ali Sarhadi* · 2026

    <span class="abstract-snippet" id="snip-2609.25505">Rapid intensification (RI) remains one of the most consequential and difficult aspects of tropical cyclone (TC) forecasting. Although full-physics numerical weather prediction models can represent...</span><span class="abstract-full" id="full-2609.25505" hidden>Rapid intensification (RI) remains one of the most consequential and difficult aspects of tropical cyclone (TC) forecasting. Although full-physics numerical weather prediction models can represent the processes governing RI, resolving storm-environment interactions remains computationally expensive, while purely data-driven approaches often lack physical interpretability. We present FAST-ML, a hybrid framework that bridges data-driven efficiency with physical constraints. A physically informed dual-stream neural parameterization ingests 3D ERA5 fields to diagnose ventilation controls---environmental wind shear and mid-level entropy deficit. By optimizing these parameters end-to-end through a differentiable FAST intensity model, this architecture establishes a robust new paradigm for observation-driven parameter optimization, ensuring storm evolution remains strictly governed by thermodynamic principles. By better capturing the storm's continuous intensity evolution, FAST-ML improves upon its physical baseline, reducing ensemble CRPS across forecast lead times, with a reduction of approximately 31% at 60 h and nearly halving the RI false alarm ratio without sacrificing detection skill. In a 100-member ensemble configuration, FAST-ML produces intensity forecasts comparable to FNV3 for selected storms under the evaluated input configurations. Furthermore, zero-shot tests on selected Eastern Pacific storms provide encouraging evidence of cross-basin transferability. FAST-ML provides a modular intensity forecasting framework that can be coupled with externally supplied storm tracks and environmental fields. It demonstrates that observation-driven parameter learning within physically constrained dynamics simultaneously enhances accuracy, interpretability, and computational efficiency.</span> <span class="abstract-toggle" data-id="2609.25505">more</span>

    [:material-file-document: 2609.25505](https://arxiv.org/abs/2609.25505v1) · [:material-content-copy: BibTeX](bibtex/2609.25505.bib){ .bibtex-link }

-   #### Learning Prognostic Variables for AI Convective Parameterizations via Symbolic Distillation

    ---

    *Jurij Schönfeld, Tom Beucler, Julien Savre, Steven Sherwood, Veronika Eyring* · 2026

    <span class="abstract-snippet" id="snip-2609.24882">Hybrid AI-physics climate modeling aims to improve coarse (~100km-resolution) Earth system models by learning to parameterize subgrid processes from high-fidelity data. However, this so far mostly...</span><span class="abstract-full" id="full-2609.24882" hidden>Hybrid AI-physics climate modeling aims to improve coarse (~100km-resolution) Earth system models by learning to parameterize subgrid processes from high-fidelity data. However, this so far mostly involves local-in-time, diagnostic parameterizations, in which the subgrid state depends only on the current coarse state with no memory of previous states, which is unrealistic for processes such as convection that have intrinsic persistence. To address this, we enhance local-in-time parameterizations by learning prognostic variables that compactly carry important, additional past information where no explicit sub-grid information is available. First we compress past information into a low-dimensional latent space using an autoencoder, which then informs a neural network trained to parameterize targeted subgrid-scale processes. We then replace the autoencoder with symbolic equations that govern the time evolution of the latent variables, yielding additional prognostic memory variables that can be integrated alongside the resolved atmospheric state. We evaluate this approach on two systems: the Lorenz-96 model (online) and surface precipitation from high-resolution atmospheric simulations (offline). A forced multivariate linear ordinary differential equation recovers most of the added value achieved by the autoencoder-based approach in both experiments. Benchmarked against diagnostic parameterizations without memory, our memory-informed approach improves climate statistics and temporal structure, including a realistic diurnal cycle of tropical land precipitation.</span> <span class="abstract-toggle" data-id="2609.24882">more</span>

    [:material-file-document: 2609.24882](https://arxiv.org/abs/2609.24882v1) · [:material-content-copy: BibTeX](bibtex/2609.24882.bib){ .bibtex-link }

-   #### Inference of Unknown Dynamical Components Using Next Generation Reservoir Computing: From Chaotic Systems to Climate Data

    ---

    *Jule Budnick, Andrew Keane, Serhiy Yanchuk* · 2026

    <span class="abstract-snippet" id="snip-2609.24754">We investigate next generation reservoir computing (NGRC) as a data-driven approach for inferring unseen components of dynamical systems. We compare NGRC with traditional reservoir computing (RC)...</span><span class="abstract-full" id="full-2609.24754" hidden>We investigate next generation reservoir computing (NGRC) as a data-driven approach for inferring unseen components of dynamical systems. We compare NGRC with traditional reservoir computing (RC) using the Lorenz and Rössler system, where two unknown components are inferred from one given component. For both systems, NGRC achieves accurate results while requiring fewer training data and less computational time than RC. We identified an inverse proportional behavior between the number of time-delayed steps needed for NGRC and the temporal resolution, indicating that the physical time span covered by the delay interval is an important factor in determining the required number of delayed steps. Finally, we apply NGRC to the observational climate data of ENSO (El Niño--Southern Oscillation) and infer one observable from the remaining variables. Despite the noise and complexity of the real-world data, the NGRC shows promising results. Our findings demonstrate the potential of NGRC for efficient inference of unseen components in both controlled dynamical systems and real-world data.</span> <span class="abstract-toggle" data-id="2609.24754">more</span>

    [:material-file-document: 2609.24754](https://arxiv.org/abs/2609.24754v1) · [:material-content-copy: BibTeX](bibtex/2609.24754.bib){ .bibtex-link }

-   #### Climate Variability Modulates the Impact of Price Spikes on Food Insecurity

    ---

    *Jordi Cerdà-Bautista, Vasileios Sitokonstantinou, Homer Durand, Gherardo Varando, Michele Ronco et al.* · 2026

    <span class="abstract-snippet" id="snip-2609.24394">Climate variability influences whether a market disruption escalates into a food crisis, yet broad climate patterns like El Niño, tracked months before they alter hydro-climatic conditions, are still...</span><span class="abstract-full" id="full-2609.24394" hidden>Climate variability influences whether a market disruption escalates into a food crisis, yet broad climate patterns like El Niño, tracked months before they alter hydro-climatic conditions, are still not incorporated as an early-warning component in food-security responses. We address this gap by introducing sensitivity regimes, a stratification of regions by the direction and strength of their vegetation response to the El Niño Southern Oscillation, and using them to estimate how food price spikes affect acute food insecurity across sub-Saharan Africa. Integrating remote sensing, socioeconomic data, and causal machine learning, we find that in regions where ENSO systematically suppresses vegetation, a price spike raises the share of the population at acute risk by 5.4 percentage points in the following month. In regions where vegetation is unaffected by or positively linked to ENSO, the estimated effect is smaller (around 2 percentage points) and statistically insignificant. These results demonstrate that climate context is critical for understanding food security vulnerabilities. Sensitivity regimes can be combined with operational price-spike triggers to stage anticipatory action: the ENSO state flags vulnerable regions months ahead, and a pre-positioned response in those regions to a price spike would avert the largest jump in acute food insecurity.</span> <span class="abstract-toggle" data-id="2609.24394">more</span>

    [:material-file-document: 2609.24394](https://arxiv.org/abs/2609.24394v1) · [:material-content-copy: BibTeX](bibtex/2609.24394.bib){ .bibtex-link }

-   #### From Regional to Global: Transfer Learning for Atmospheric Transport Emulators

    ---

    *Jeff Clark, Elena Fillola, Nawid Keshtmand, Raul Santos-Rodriguez, Matthew Rigby* · 2026

    <span class="abstract-snippet" id="snip-2609.23838">Greenhouse gas emissions estimates can be derived using inverse methods by combining atmospheric concentration observations with chemical transport models. The latter traditionally use physics-driven...</span><span class="abstract-full" id="full-2609.23838" hidden>Greenhouse gas emissions estimates can be derived using inverse methods by combining atmospheric concentration observations with chemical transport models. The latter traditionally use physics-driven simulators such as Lagrangian Particle Dispersion Models (LPDMs), which are expensive to run and do not scale well to modern satellites' high resolution data. Previously we developed a performant atmospheric transport emulator that approximates LPDM outputs ("footprints") over South America ~1,000X faster than the UK Met Office's LPDM. Expanding towards global emulation is not straightforward, as atmospheric transport is regionally heterogeneous. This paper evaluates spatial transferability capabilities of models across four world regions: South America, East Asia, South Asia, North Africa using both region-specific and multi-region models, and leave-one-region-out experiments. Regional differences are characterised in the context of input variable and output footprint distributions. This work builds intuition in cross-region generalisation and transfer learning, aiding regional performance towards efficient global emissions estimates.</span> <span class="abstract-toggle" data-id="2609.23838">more</span>

    [:material-file-document: 2609.23838](https://arxiv.org/abs/2609.23838v1) · [:material-content-copy: BibTeX](bibtex/2609.23838.bib){ .bibtex-link }

-   #### ClimTip-GML: A global bias-corrected and downscaled dataset for assessing impacts of climate tipping events

    ---

    *Philipp Hess, Sebastian Bathiany, Lucas Ferreira Correa, Laura C. Jackson, Casey R. Patrizio et al.* · 2026

    <span class="abstract-snippet" id="snip-2609.23149">Assessing the impacts of future climate scenarios including tipping events of major Earth system components such as the Amazon rainforest (ARF) or the Atlantic meridional overturning circulation...</span><span class="abstract-full" id="full-2609.23149" hidden>Assessing the impacts of future climate scenarios including tipping events of major Earth system components such as the Amazon rainforest (ARF) or the Atlantic meridional overturning circulation (AMOC), requires accurate and high-resolution simulations. Here, we present ClimTip-GML, the first globally bias-corrected and downscaled climate dataset for impact assessment of large-scale tipping scenarios, comprising eight key variables at 0.25° spatial resolution from three general circulation models (GCMs): CESM1-CAM5, HadGEM3-GC31-MM, and MPI-ESM1-2-HR. The dataset includes 100-year-long climate simulations with preindustrial and historical conditions, as well as scenarios at a +2°C warming level with and without tipping transitions of the AMOC or ARF. We apply generative machine learning (GML) techniques trained on reanalysis data to bias-correct and downscale the GCMs in a manner that is physically consistent across space, time, and all eight variables. Comprehensive validation shows substantially reduced biases, improved small-scale spatial variability, multivariate correlations, and consistent long-term climate responses to the external forcing and tipping events. The results hence permit substantially improved impact assessments of tipping transitions of the ARF and AMOC, directly informing mitigation and adaptation policies.</span> <span class="abstract-toggle" data-id="2609.23149">more</span>

    [:material-file-document: 2609.23149](https://arxiv.org/abs/2609.23149v1) · [:material-content-copy: BibTeX](bibtex/2609.23149.bib){ .bibtex-link }

-   #### Diffusion-Based Super-Resolution of Adriatic Sea Oceanographic Fields

    ---

    *Rajat Srivastava, Muhammad Sarmad, Emanuele Mele, Massimo Cafaro, Marco Pulimeno, Italo Epicoco* · 2026

    <span class="abstract-snippet" id="snip-2609.22574">High-resolution oceanographic fields are critical for resolving mesoscale and sub-mesoscale coastal dynamics, yet their generation remains constrained by both computational cost and observational...</span><span class="abstract-full" id="full-2609.22574" hidden>High-resolution oceanographic fields are critical for resolving mesoscale and sub-mesoscale coastal dynamics, yet their generation remains constrained by both computational cost and observational sparsity. We present OcDiffSR, a conditional denoising diffusion probabilistic model (DDPM) for oceanographic super-resolution that reconstructs high-resolution sea-surface fields from coarse-resolution reanalysis inputs. The model is trained on ten years (2011-2020) of paired low-resolution (GLORYS12V1, 1/12) and high-resolution (Mediterranean Sea Physics Reanalysis, Med MFC, 1/24) data, and evaluated on an independent test year (2009) over the Adriatic Sea. OcDiffSR employs a conditional U-Net augmented with multi-scale low-resolution encoders, cross-attention bottleneck layers, and sinusoidal seasonal embeddings via Feature-wise Linear Modulation (FiLM), enabling joint super-resolution of sea-surface temperature (SST), salinity (SSS), and horizontal velocity components with visually coherent circulation patterns. Benchmarked against bilinear interpolation and the state-of-the-art residual diffusion model CorrDiff, OcDiffSR achieves substantially lower reconstruction errors for scalar fields (RMSESST=0.477 C, RMSESSS=0.346 psu), near-unity Pearson correlation (PCC >= 0.999), and high structural similarity (SSIM >= 0.964). For dynamical vector fields, OcDiffSR outperforms both baselines in absolute error and spatial coherence, though moderate correlation (PCC = 0.64) reflects the intrinsic stochasticity of oceanic velocity fields. Daily and monthly evaluations confirm temporal robustness across all seasons. These results establish OcDiffSR as a reliable framework for high-fidelity oceanographic downscaling and reanalysis enhancement, producing fields that are visually consistent with known ocean dynamics.</span> <span class="abstract-toggle" data-id="2609.22574">more</span>

    [:material-file-document: 2609.22574](https://arxiv.org/abs/2609.22574v1) · [:material-content-copy: BibTeX](bibtex/2609.22574.bib){ .bibtex-link }

    <span class="md-tag">transformer</span> <span class="md-tag">diffusion</span> <span class="md-tag">CNN</span> <span class="md-tag">probabilistic</span>

</div>

