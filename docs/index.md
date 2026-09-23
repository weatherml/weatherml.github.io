---
hide:
  - navigation
title: Deep Learning in Weather
---

A collection of papers on deep learning and machine learning applied to weather forecasting, climate modeling, and atmospheric science.

*Last updated: 2026-09-23*

## Starred Papers

<div class="grid cards" markdown>

-   #### <span class="star-marker">:material-star:</span> AIFS-DOP: End-to-End Medium-Range Weather Prediction from Observations Alone with Machine Learning

    ---

    <span class="paper-meta"><em>Ewan Pinnington, Peter Lean, Mihai Alexe, Eulalie Boucher, Simon Lang, Patrick Laloyaux et al.</em> · 2026</span>

    <span class="abstract-snippet" id="snip-2606.19093" data-search-exclude>We introduce the Artificial Intelligence Forecasting System for Direct Observation Prediction (AIFS-DOP). AIFS-DOP is trained on a 40-year harmonized dataset of gridded observations, without using...</span><span class="abstract-full" id="full-2606.19093" data-search-exclude hidden>We introduce the Artificial Intelligence Forecasting System for Direct Observation Prediction (AIFS-DOP). AIFS-DOP is trained on a 40-year harmonized dataset of gridded observations, without using numerical weather prediction (NWP) reanalysis or model data. The resulting model is competitive with ECMWF's Integrated Forecasting System (IFS) when scored on a one year period of forecasts across 2021/2022. This progress on Direct Observation Prediction represents the first time that a data-driven model, trained solely on observations, is competitive with the IFS at medium ranges for several key upper-air and surface headline scores, when verified against observation data.</span> <span class="abstract-toggle" data-id="2606.19093">more</span>

    <span class="paper-links">[:material-file-document: 2606.19093](https://arxiv.org/abs/2606.19093v1) · [:material-content-copy: BibTeX](bibtex/2606.19093.bib){ .bibtex-link }</span>

-   #### <span class="star-marker">:material-star:</span> (Sparse) Attention to the Details: Preserving Spectral Fidelity in ML-based Weather Forecasting Models

    ---

    <span class="paper-meta"><em>Maksim Zhdanov, Ana Lucic, Max Welling, Jan-Willem van de Meent</em> · 2026</span>

    <span class="abstract-snippet" id="snip-2604.16429" data-search-exclude>We introduce Mosaic, a probabilistic weather forecasting model that addresses three failure modes of spectral degradation in ML-based weather prediction: spectral damping (statistical),...</span><span class="abstract-full" id="full-2604.16429" data-search-exclude hidden>We introduce Mosaic, a probabilistic weather forecasting model that addresses three failure modes of spectral degradation in ML-based weather prediction: spectral damping (statistical), high-frequency aliasing (architectural), and residual high-frequency leakage (parametric). Mosaic generates ensemble members through learned functional perturbations and operates on native-resolution grids via mesh-aligned block-sparse attention, a hardware-aligned mechanism that captures long-range dependencies at linear cost by sharing keys and values across spatially adjacent queries. At 1.5° resolution with 214M parameters, Mosaic matches or outperforms models trained on 6$\times$ finer resolution on key variables and achieves state-of-the-art results among 1.5° models, producing well-calibrated ensembles whose individual members exhibit near-perfect spectral alignment across all resolved frequencies. A 24-member, 10-day forecast takes under 12s on a single H100~GPU. Code is available at https://github.com/maxxxzdn/mosaic.</span> <span class="abstract-toggle" data-id="2604.16429">more</span>

    <span class="paper-links">[:material-file-document: 2604.16429](https://arxiv.org/abs/2604.16429v3) · [:fontawesome-brands-github:](https://github.com/maxxxzdn/mosaic) · [:material-content-copy: BibTeX](bibtex/2604.16429.bib){ .bibtex-link }</span>

    <a class="md-tag" href="/tags/#probabilistic">probabilistic</a>

-   #### <span class="star-marker">:material-star:</span> U-Cast: A Surprisingly Simple and Efficient Frontier Probabilistic AI Weather Forecaster

    ---

    <span class="paper-meta"><em>Salva Rühling Cachay, Duncan Watson-Parris, Rose Yu</em> · 2026</span>

    <span class="abstract-snippet" id="snip-2604.09041" data-search-exclude>AI-based weather forecasting now rivals traditional physics-based ensembles, but state-of-the-art (SOTA) models rely on specialized architectures and massive computational budgets, creating a high...</span><span class="abstract-full" id="full-2604.09041" data-search-exclude hidden>AI-based weather forecasting now rivals traditional physics-based ensembles, but state-of-the-art (SOTA) models rely on specialized architectures and massive computational budgets, creating a high barrier to entry. We demonstrate that such complexity is unnecessary for frontier performance. We introduce \ours, a probabilistic forecaster built on a standard U-Net backbone trained with a simple recipe: deterministic pre-training on Mean Absolute Error followed by short probabilistic fine-tuning on the Continuous Ranked Probability Score (CRPS) using Monte Carlo Dropout for stochasticity. As a result, our model matches or exceeds the probabilistic skill of GenCast and IFS ENS at $1.5^\circ$ resolution while reducing training compute by over $10\times$ compared to leading CRPS-based models and inference latency by over $10\times$ compared to diffusion-based models. U-Cast trains in under 12 H200 GPU-days and generates a 15-day ensemble forecast in 3 seconds. These results suggest that scalable, general-purpose architectures paired with efficient training curricula can match complex domain-specific designs at a fraction of the cost, opening the training of frontier probabilistic weather models to the broader community.</span> <span class="abstract-toggle" data-id="2604.09041">more</span>

    <span class="paper-links">[:material-file-document: 2604.09041](https://arxiv.org/abs/2604.09041v2) · [:fontawesome-brands-github:](https://github.com/Rose-STL-Lab/u-cast) · [:material-content-copy: BibTeX](bibtex/2604.09041.bib){ .bibtex-link }</span>

    <a class="md-tag" href="/tags/#cnn">CNN</a> <a class="md-tag" href="/tags/#physics-informed">physics-informed</a> <a class="md-tag" href="/tags/#probabilistic">probabilistic</a>

-   #### <span class="star-marker">:material-star:</span> Using data assimilation tools to dissect GraphDOP

    ---

    <span class="paper-meta"><em>Patrick Laloyaux, Mihai Alexe, Eulalie Boucher, Peter Lean, Ewan Pinnington, Simon Lang et al.</em> · 2025</span>

    <span class="abstract-snippet" id="snip-2510.27388" data-search-exclude>The Data Assimilation (DA) community has been developing various diagnostics to understand the importance of the observing system in accurately forecasting the weather. They usually rely on the...</span><span class="abstract-full" id="full-2510.27388" data-search-exclude hidden>The Data Assimilation (DA) community has been developing various diagnostics to understand the importance of the observing system in accurately forecasting the weather. They usually rely on the ability to compute the derivatives of the physical model output with respect to its initial condition. For example, the Forecast Sensitivity-based Observation Impact (FSOI) estimates the impact on the forecast error of each observation processed in the DA system. This paper presents how these DA diagnostic tools are transferred to Machine Learning (ML) models, as their derivatives are readily available through automatic differentiation. We specifically explore the interpretability and explainability of the observation-driven GraphDOP model developed at the European Centre for Medium-Range Weather Forecasts (ECMWF). The interpretability study demonstrates the effectiveness of GraphDOP's sliding attention window to learn the meteorological features present in the observation datasets and to learn the spatial relationships between different regions. Making these relationships more transparent confirms that GraphDOP captures real, physically meaningful processes, such as the movement of storm systems. The explainability of GraphDOP is explored by applying the FSOI tool to study the impact of the different observations on the forecast error. This inspection reveals that GraphDOP creates an internal representation of the Earth system by combining the information from conventional and satellite observations.</span> <span class="abstract-toggle" data-id="2510.27388">more</span>

    <span class="paper-links">[:material-file-document: 2510.27388](https://arxiv.org/abs/2510.27388v1) · [:material-content-copy: BibTeX](bibtex/2510.27388.bib){ .bibtex-link }</span>

-   #### <span class="star-marker">:material-star:</span> GraphDOP: Towards skilful data-driven medium-range weather forecasts learnt and initialised directly from observations

    ---

    <span class="paper-meta"><em>Mihai Alexe, Eulalie Boucher, Peter Lean, Ewan Pinnington, Patrick Laloyaux, Anthony McNally et al.</em> · 2024</span>

    <span class="abstract-snippet" id="snip-2412.15687" data-search-exclude>We introduce GraphDOP, a new data-driven, end-to-end forecast system developed at the European Centre for Medium-Range Weather Forecasts (ECMWF) that is trained and initialised exclusively from Earth...</span><span class="abstract-full" id="full-2412.15687" data-search-exclude hidden>We introduce GraphDOP, a new data-driven, end-to-end forecast system developed at the European Centre for Medium-Range Weather Forecasts (ECMWF) that is trained and initialised exclusively from Earth System observations, with no physics-based (re)analysis inputs or feedbacks. GraphDOP learns the correlations between observed quantities - such as brightness temperatures from polar orbiters and geostationary satellites - and geophysical quantities of interest (that are measured by conventional observations), to form a coherent latent representation of Earth System state dynamics and physical processes, and is capable of producing skilful predictions of relevant weather parameters up to five days into the future.</span> <span class="abstract-toggle" data-id="2412.15687">more</span>

    <span class="paper-links">[:material-file-document: 2412.15687](https://arxiv.org/abs/2412.15687v1) · [:material-content-copy: BibTeX](bibtex/2412.15687.bib){ .bibtex-link }</span>

    <a class="md-tag" href="/tags/#physics-informed">physics-informed</a>

</div>

## Recent Additions

<div class="grid cards" markdown>

-   #### Analysis of trade-offs in urban heat mitigation using a Bayesian Optimization framework for an urban canopy layer model

    ---

    <span class="paper-meta"><em>Rebekka Walter, Johanna Gelhaus, David Anton, Henning Wessles, Stephan Weber</em> · 2026</span>

    <span class="abstract-snippet" id="snip-2609.25953" data-search-exclude>To mitigate the challenges of climate change and intensifying heat stress in urban areas, local adaptation strategies are discussed and introduced in cities worldwide. To understand processes and...</span><span class="abstract-full" id="full-2609.25953" data-search-exclude hidden>To mitigate the challenges of climate change and intensifying heat stress in urban areas, local adaptation strategies are discussed and introduced in cities worldwide. To understand processes and potential trade-offs of these strategies a Bayesian optimization and surrogate modeling framework was employed to investigate urban parameter ranges of heat mitigation strategies with focus on three thermal metrics: daytime air temperature, Universal Thermal Climate Index (UTCI), and nighttime air temperature. Based on an urban street canyon configuration, it was shown that heat mitigation measures that reduce daytime air temperature and UTCI are often associated with higher nighttime temperatures. This results in a curved Pareto front that reflects the trade-off between daytime and nighttime thermal comfort. Under identical forcing conditions, different urban configurations, varying in geometry, vegetation, and surface characteristics, are shown to alter peak canyon air temperature by up to 5.2~$^\circ$C during the day and 2.6~$^\circ$C at night, while UTCI varies by up to 7.9~$^\circ$C, demonstrating that favorable urban configurations can substantially mitigate microclimatic heat stress. These findings suggest that combining multi-objective Bayesian optimization and surrogate modeling can help bridge the gap between computationally intensive climate simulations and practical decision-making in urban planning. An interactive visualization tool was developed to explore these trade-offs, making the often opposing relationships between urban parameters and the three thermal metrics directly accessible to planners.</span> <span class="abstract-toggle" data-id="2609.25953">more</span>

    <span class="paper-links">[:material-file-document: 2609.25953](https://arxiv.org/abs/2609.25953v1) · [:material-content-copy: BibTeX](bibtex/2609.25953.bib){ .bibtex-link }</span>

    <a class="md-tag" href="/tags/#probabilistic">probabilistic</a>

-   #### A dataset of one-dimensional idealized probabilistic fields

    ---

    <span class="paper-meta"><em>Gregor Skok, Romain Pic</em> · 2026</span>

    <span class="abstract-snippet" id="snip-2609.25720" data-search-exclude>Verification of probabilistic weather forecasts remains a crucial aspect of numerical weather prediction, as new AI-based models become more widely used alongside the more traditional physics-based...</span><span class="abstract-full" id="full-2609.25720" data-search-exclude hidden>Verification of probabilistic weather forecasts remains a crucial aspect of numerical weather prediction, as new AI-based models become more widely used alongside the more traditional physics-based ensemble forecasting systems that continue to be developed and improved. We present a first-of-its-kind idealized probabilistic dataset composed of one-dimensional cases aimed at analyzing the behavior and properties of verification methods for probabilistic forecasts and comparing their behavior. It covers a wide range of probabilistic cases, such as constant, localized events, gradients, fronts, noisy, bimodal, and limiting cases. Moreover, the code associated with the dataset provides great flexibility for customizing the experiments it covers. The dataset represents the first building block of the more extensive comparison dataset of the Bridging The Gap project, which aims to facilitate the development and comparison of spatial verification methods for probabilistic forecasts.</span> <span class="abstract-toggle" data-id="2609.25720">more</span>

    <span class="paper-links">[:material-file-document: 2609.25720](https://arxiv.org/abs/2609.25720v1) · [:material-content-copy: BibTeX](bibtex/2609.25720.bib){ .bibtex-link }</span>

    <a class="md-tag" href="/tags/#physics-informed">physics-informed</a> <a class="md-tag" href="/tags/#probabilistic">probabilistic</a>

-   #### West-WRF AI 2-km: High-Resolution Prediction of Integrated Vapor Transport and Precipitation

    ---

    <span class="paper-meta"><em>Nazak Rouzegari, Vesta Afzali Gorooh, Agniv Sengupta, Phu Nguyen, Kuo-Lin Hsu, Amir AghaKouchak et al.</em> · 2026</span>

    <span class="abstract-snippet" id="snip-2609.25512" data-search-exclude>We introduce a stretched-grid artificial intelligence (AI) weather forecasting model with 2-km resolution over the western United States and part of the Northeast Pacific and approximately 31-km...</span><span class="abstract-full" id="full-2609.25512" data-search-exclude hidden>We introduce a stretched-grid artificial intelligence (AI) weather forecasting model with 2-km resolution over the western United States and part of the Northeast Pacific and approximately 31-km resolution elsewhere globally. Forecasting over the western U.S. is challenging because complex topography and atmospheric rivers (ARs) strongly influence orographic precipitation. West-WRF AI 2-km builds on a global model pretrained with a 40-year European Centre for Medium-Range Weather Forecasts Reanalysis v5 (ERA5) dataset and is fine-tuned with the Center for Western Weather and Water Extremes (CW3E) 2-km regional reanalysis to produce autoregressive 6-hourly forecasts of precipitation and integrated vapor transport (IVT). Forecasts are evaluated over winters 2020-2023 using gridded precipitation observations, rain gauges, and AR Reconnaissance dropsondes and are benchmarked against coarser-resolution AI forecasts and regional and global numerical weather prediction (NWP) systems. West-WRF AI 2-km reproduces observed precipitation-intensity distributions, retains fine-scale spectral variability, and produces sharper narrow coastal precipitation bands and localized, terrain-sensitive extremes. Its broader-scale performance remains comparable to coarser-resolution configurations while preserving large-scale skill despite higher resolution. Dropsonde verification shows lower errors and improved categorical skill at the most extreme IVT threshold. Overall, West-WRF AI 2-km provides its greatest value for localized precipitation extremes and intense AR-related moisture transport.</span> <span class="abstract-toggle" data-id="2609.25512">more</span>

    <span class="paper-links">[:material-file-document: 2609.25512](https://arxiv.org/abs/2609.25512v1) · [:material-content-copy: BibTeX](bibtex/2609.25512.bib){ .bibtex-link }</span>

-   #### FAST-ML: A Hybrid Physics-Machine Learning Framework for Tropical Cyclone Intensity Forecasting

    ---

    <span class="paper-meta"><em>Shijie Xiao, Jonathan Lin, Thomas Ehrmann, Ali Sarhadi</em> · 2026</span>

    <span class="abstract-snippet" id="snip-2609.25505" data-search-exclude>Rapid intensification (RI) remains one of the most consequential and difficult aspects of tropical cyclone (TC) forecasting. Although full-physics numerical weather prediction models can represent...</span><span class="abstract-full" id="full-2609.25505" data-search-exclude hidden>Rapid intensification (RI) remains one of the most consequential and difficult aspects of tropical cyclone (TC) forecasting. Although full-physics numerical weather prediction models can represent the processes governing RI, resolving storm-environment interactions remains computationally expensive, while purely data-driven approaches often lack physical interpretability. We present FAST-ML, a hybrid framework that bridges data-driven efficiency with physical constraints. A physically informed dual-stream neural parameterization ingests 3D ERA5 fields to diagnose ventilation controls---environmental wind shear and mid-level entropy deficit. By optimizing these parameters end-to-end through a differentiable FAST intensity model, this architecture establishes a robust new paradigm for observation-driven parameter optimization, ensuring storm evolution remains strictly governed by thermodynamic principles. By better capturing the storm's continuous intensity evolution, FAST-ML improves upon its physical baseline, reducing ensemble CRPS across forecast lead times, with a reduction of approximately 31% at 60 h and nearly halving the RI false alarm ratio without sacrificing detection skill. In a 100-member ensemble configuration, FAST-ML produces intensity forecasts comparable to FNV3 for selected storms under the evaluated input configurations. Furthermore, zero-shot tests on selected Eastern Pacific storms provide encouraging evidence of cross-basin transferability. FAST-ML provides a modular intensity forecasting framework that can be coupled with externally supplied storm tracks and environmental fields. It demonstrates that observation-driven parameter learning within physically constrained dynamics simultaneously enhances accuracy, interpretability, and computational efficiency.</span> <span class="abstract-toggle" data-id="2609.25505">more</span>

    <span class="paper-links">[:material-file-document: 2609.25505](https://arxiv.org/abs/2609.25505v1) · [:material-content-copy: BibTeX](bibtex/2609.25505.bib){ .bibtex-link }</span>

-   #### Learning Prognostic Variables for AI Convective Parameterizations via Symbolic Distillation

    ---

    <span class="paper-meta"><em>Jurij Schönfeld, Tom Beucler, Julien Savre, Steven Sherwood, Veronika Eyring</em> · 2026</span>

    <span class="abstract-snippet" id="snip-2609.24882" data-search-exclude>Hybrid AI-physics climate modeling aims to improve coarse (~100km-resolution) Earth system models by learning to parameterize subgrid processes from high-fidelity data. However, this so far mostly...</span><span class="abstract-full" id="full-2609.24882" data-search-exclude hidden>Hybrid AI-physics climate modeling aims to improve coarse (~100km-resolution) Earth system models by learning to parameterize subgrid processes from high-fidelity data. However, this so far mostly involves local-in-time, diagnostic parameterizations, in which the subgrid state depends only on the current coarse state with no memory of previous states, which is unrealistic for processes such as convection that have intrinsic persistence. To address this, we enhance local-in-time parameterizations by learning prognostic variables that compactly carry important, additional past information where no explicit sub-grid information is available. First we compress past information into a low-dimensional latent space using an autoencoder, which then informs a neural network trained to parameterize targeted subgrid-scale processes. We then replace the autoencoder with symbolic equations that govern the time evolution of the latent variables, yielding additional prognostic memory variables that can be integrated alongside the resolved atmospheric state. We evaluate this approach on two systems: the Lorenz-96 model (online) and surface precipitation from high-resolution atmospheric simulations (offline). A forced multivariate linear ordinary differential equation recovers most of the added value achieved by the autoencoder-based approach in both experiments. Benchmarked against diagnostic parameterizations without memory, our memory-informed approach improves climate statistics and temporal structure, including a realistic diurnal cycle of tropical land precipitation.</span> <span class="abstract-toggle" data-id="2609.24882">more</span>

    <span class="paper-links">[:material-file-document: 2609.24882](https://arxiv.org/abs/2609.24882v1) · [:material-content-copy: BibTeX](bibtex/2609.24882.bib){ .bibtex-link }</span>

-   #### Inference of Unknown Dynamical Components Using Next Generation Reservoir Computing: From Chaotic Systems to Climate Data

    ---

    <span class="paper-meta"><em>Jule Budnick, Andrew Keane, Serhiy Yanchuk</em> · 2026</span>

    <span class="abstract-snippet" id="snip-2609.24754" data-search-exclude>We investigate next generation reservoir computing (NGRC) as a data-driven approach for inferring unseen components of dynamical systems. We compare NGRC with traditional reservoir computing (RC)...</span><span class="abstract-full" id="full-2609.24754" data-search-exclude hidden>We investigate next generation reservoir computing (NGRC) as a data-driven approach for inferring unseen components of dynamical systems. We compare NGRC with traditional reservoir computing (RC) using the Lorenz and Rössler system, where two unknown components are inferred from one given component. For both systems, NGRC achieves accurate results while requiring fewer training data and less computational time than RC. We identified an inverse proportional behavior between the number of time-delayed steps needed for NGRC and the temporal resolution, indicating that the physical time span covered by the delay interval is an important factor in determining the required number of delayed steps. Finally, we apply NGRC to the observational climate data of ENSO (El Niño--Southern Oscillation) and infer one observable from the remaining variables. Despite the noise and complexity of the real-world data, the NGRC shows promising results. Our findings demonstrate the potential of NGRC for efficient inference of unseen components in both controlled dynamical systems and real-world data.</span> <span class="abstract-toggle" data-id="2609.24754">more</span>

    <span class="paper-links">[:material-file-document: 2609.24754](https://arxiv.org/abs/2609.24754v1) · [:material-content-copy: BibTeX](bibtex/2609.24754.bib){ .bibtex-link }</span>

-   #### Climate Variability Modulates the Impact of Price Spikes on Food Insecurity

    ---

    <span class="paper-meta"><em>Jordi Cerdà-Bautista, Vasileios Sitokonstantinou, Homer Durand, Gherardo Varando, Michele Ronco et al.</em> · 2026</span>

    <span class="abstract-snippet" id="snip-2609.24394" data-search-exclude>Climate variability influences whether a market disruption escalates into a food crisis, yet broad climate patterns like El Niño, tracked months before they alter hydro-climatic conditions, are still...</span><span class="abstract-full" id="full-2609.24394" data-search-exclude hidden>Climate variability influences whether a market disruption escalates into a food crisis, yet broad climate patterns like El Niño, tracked months before they alter hydro-climatic conditions, are still not incorporated as an early-warning component in food-security responses. We address this gap by introducing sensitivity regimes, a stratification of regions by the direction and strength of their vegetation response to the El Niño Southern Oscillation, and using them to estimate how food price spikes affect acute food insecurity across sub-Saharan Africa. Integrating remote sensing, socioeconomic data, and causal machine learning, we find that in regions where ENSO systematically suppresses vegetation, a price spike raises the share of the population at acute risk by 5.4 percentage points in the following month. In regions where vegetation is unaffected by or positively linked to ENSO, the estimated effect is smaller (around 2 percentage points) and statistically insignificant. These results demonstrate that climate context is critical for understanding food security vulnerabilities. Sensitivity regimes can be combined with operational price-spike triggers to stage anticipatory action: the ENSO state flags vulnerable regions months ahead, and a pre-positioned response in those regions to a price spike would avert the largest jump in acute food insecurity.</span> <span class="abstract-toggle" data-id="2609.24394">more</span>

    <span class="paper-links">[:material-file-document: 2609.24394](https://arxiv.org/abs/2609.24394v1) · [:material-content-copy: BibTeX](bibtex/2609.24394.bib){ .bibtex-link }</span>

-   #### From Regional to Global: Transfer Learning for Atmospheric Transport Emulators

    ---

    <span class="paper-meta"><em>Jeff Clark, Elena Fillola, Nawid Keshtmand, Raul Santos-Rodriguez, Matthew Rigby</em> · 2026</span>

    <span class="abstract-snippet" id="snip-2609.23838" data-search-exclude>Greenhouse gas emissions estimates can be derived using inverse methods by combining atmospheric concentration observations with chemical transport models. The latter traditionally use physics-driven...</span><span class="abstract-full" id="full-2609.23838" data-search-exclude hidden>Greenhouse gas emissions estimates can be derived using inverse methods by combining atmospheric concentration observations with chemical transport models. The latter traditionally use physics-driven simulators such as Lagrangian Particle Dispersion Models (LPDMs), which are expensive to run and do not scale well to modern satellites' high resolution data. Previously we developed a performant atmospheric transport emulator that approximates LPDM outputs ("footprints") over South America ~1,000X faster than the UK Met Office's LPDM. Expanding towards global emulation is not straightforward, as atmospheric transport is regionally heterogeneous. This paper evaluates spatial transferability capabilities of models across four world regions: South America, East Asia, South Asia, North Africa using both region-specific and multi-region models, and leave-one-region-out experiments. Regional differences are characterised in the context of input variable and output footprint distributions. This work builds intuition in cross-region generalisation and transfer learning, aiding regional performance towards efficient global emissions estimates.</span> <span class="abstract-toggle" data-id="2609.23838">more</span>

    <span class="paper-links">[:material-file-document: 2609.23838](https://arxiv.org/abs/2609.23838v1) · [:material-content-copy: BibTeX](bibtex/2609.23838.bib){ .bibtex-link }</span>

-   #### ClimTip-GML: A global bias-corrected and downscaled dataset for assessing impacts of climate tipping events

    ---

    <span class="paper-meta"><em>Philipp Hess, Sebastian Bathiany, Lucas Ferreira Correa, Laura C. Jackson, Casey R. Patrizio et al.</em> · 2026</span>

    <span class="abstract-snippet" id="snip-2609.23149" data-search-exclude>Assessing the impacts of future climate scenarios including tipping events of major Earth system components such as the Amazon rainforest (ARF) or the Atlantic meridional overturning circulation...</span><span class="abstract-full" id="full-2609.23149" data-search-exclude hidden>Assessing the impacts of future climate scenarios including tipping events of major Earth system components such as the Amazon rainforest (ARF) or the Atlantic meridional overturning circulation (AMOC), requires accurate and high-resolution simulations. Here, we present ClimTip-GML, the first globally bias-corrected and downscaled climate dataset for impact assessment of large-scale tipping scenarios, comprising eight key variables at 0.25° spatial resolution from three general circulation models (GCMs): CESM1-CAM5, HadGEM3-GC31-MM, and MPI-ESM1-2-HR. The dataset includes 100-year-long climate simulations with preindustrial and historical conditions, as well as scenarios at a +2°C warming level with and without tipping transitions of the AMOC or ARF. We apply generative machine learning (GML) techniques trained on reanalysis data to bias-correct and downscale the GCMs in a manner that is physically consistent across space, time, and all eight variables. Comprehensive validation shows substantially reduced biases, improved small-scale spatial variability, multivariate correlations, and consistent long-term climate responses to the external forcing and tipping events. The results hence permit substantially improved impact assessments of tipping transitions of the ARF and AMOC, directly informing mitigation and adaptation policies.</span> <span class="abstract-toggle" data-id="2609.23149">more</span>

    <span class="paper-links">[:material-file-document: 2609.23149](https://arxiv.org/abs/2609.23149v1) · [:material-content-copy: BibTeX](bibtex/2609.23149.bib){ .bibtex-link }</span>

-   #### Diffusion-Based Super-Resolution of Adriatic Sea Oceanographic Fields

    ---

    <span class="paper-meta"><em>Rajat Srivastava, Muhammad Sarmad, Emanuele Mele, Massimo Cafaro, Marco Pulimeno, Italo Epicoco</em> · 2026</span>

    <span class="abstract-snippet" id="snip-2609.22574" data-search-exclude>High-resolution oceanographic fields are critical for resolving mesoscale and sub-mesoscale coastal dynamics, yet their generation remains constrained by both computational cost and observational...</span><span class="abstract-full" id="full-2609.22574" data-search-exclude hidden>High-resolution oceanographic fields are critical for resolving mesoscale and sub-mesoscale coastal dynamics, yet their generation remains constrained by both computational cost and observational sparsity. We present OcDiffSR, a conditional denoising diffusion probabilistic model (DDPM) for oceanographic super-resolution that reconstructs high-resolution sea-surface fields from coarse-resolution reanalysis inputs. The model is trained on ten years (2011-2020) of paired low-resolution (GLORYS12V1, 1/12) and high-resolution (Mediterranean Sea Physics Reanalysis, Med MFC, 1/24) data, and evaluated on an independent test year (2009) over the Adriatic Sea. OcDiffSR employs a conditional U-Net augmented with multi-scale low-resolution encoders, cross-attention bottleneck layers, and sinusoidal seasonal embeddings via Feature-wise Linear Modulation (FiLM), enabling joint super-resolution of sea-surface temperature (SST), salinity (SSS), and horizontal velocity components with visually coherent circulation patterns. Benchmarked against bilinear interpolation and the state-of-the-art residual diffusion model CorrDiff, OcDiffSR achieves substantially lower reconstruction errors for scalar fields (RMSESST=0.477 C, RMSESSS=0.346 psu), near-unity Pearson correlation (PCC >= 0.999), and high structural similarity (SSIM >= 0.964). For dynamical vector fields, OcDiffSR outperforms both baselines in absolute error and spatial coherence, though moderate correlation (PCC = 0.64) reflects the intrinsic stochasticity of oceanic velocity fields. Daily and monthly evaluations confirm temporal robustness across all seasons. These results establish OcDiffSR as a reliable framework for high-fidelity oceanographic downscaling and reanalysis enhancement, producing fields that are visually consistent with known ocean dynamics.</span> <span class="abstract-toggle" data-id="2609.22574">more</span>

    <span class="paper-links">[:material-file-document: 2609.22574](https://arxiv.org/abs/2609.22574v1) · [:material-content-copy: BibTeX](bibtex/2609.22574.bib){ .bibtex-link }</span>

    <a class="md-tag" href="/tags/#transformer">transformer</a> <a class="md-tag" href="/tags/#diffusion">diffusion</a> <a class="md-tag" href="/tags/#cnn">CNN</a> <a class="md-tag" href="/tags/#probabilistic">probabilistic</a>

</div>

