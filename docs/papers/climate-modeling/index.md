---
title: 'Climate Modeling'
hide:
  - toc
---

<div class="listing-header" markdown>

# Climate Modeling

<p class="page-meta" markdown="span">299 papers · page 1 of 10 · <a href="../../bib/climate-modeling.bib" download>:material-download: BibTeX for this topic</a></p>

</div>

<div class="grid cards" markdown>

-   #### SciExam for ENSO: Can AI Agents Build Climate Models? { #2610.10513 }

    *Yinling Zhang, Langchen Liu, Dongbin Xiu, Xueyan Zou, Xu Kuang, Mengdi Wang, Shilong Liu* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.10513">Language-model agents are increasingly asked to carry out open-ended scientific research, yet their results are usually graded against a known answer, a rubric, or a language-model reviewer, none of...</span><span class="abstract-full" id="full-2610.10513" hidden>Language-model agents are increasingly asked to carry out open-ended scientific research, yet their results are usually graded against a known answer, a rubric, or a language-model reviewer, none of which can tell whether a new scientific model is valid. The AI Science Exam for El Nino-Southern Oscillation (SciExam for ENSO) is a benchmark in which agents build low-order stochastic models of ENSO, the dominant mode of interannual climate variability, from real observations. Within a six-hour budget, agents process the observations, write their own diagnostics, which are then frozen, and develop a model using only these diagnostics as feedback. Hidden graders then test whether the model reproduces ENSO's statistics, recovers unobserved variables, and forecasts held-out years, and score a published model in the same way. Across twelve agent systems, six produce models that score higher than the published model, mainly through better reconstruction and forecasting. The simplified forms of the stronger models are each compatible with one of the two competing explanations of ENSO's warm-cold asymmetry, an open debate that the task never mentions. Controlled runs of the top system under varied information suggest that its scores do not come from recalling the dated observational record and that the information it receives shapes how it builds its model. SciExam for ENSO can thus evaluate agent research where no answer is known, and the results suggest that agents can already build competitive models whose structures bear on questions that scientists still debate.</span> <span class="abstract-toggle" data-id="2610.10513">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.10513v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.10513v1) · [:fontawesome-brands-github: Code](https://github.com/ylzhang2447/SciExam-ENSO-code) · [:material-content-copy: BibTeX](../../bibtex/2610.10513.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=subseasonal-to-seasonal" data-tag="subseasonal-to-seasonal">Subseasonal to seasonal</a>
    { .paper-tags }

-   #### Artificial intelligence pathways from weather to climate { #2610.09770 }

    *Tom Beucler, J. David Neelin, Hui Su, Shivanshi Asthana, Chris Bretherton, Will Chapman et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.09770">Deep learning has made rapid advances in weather forecasting: autoregressive models trained on atmospheric reanalyses now rival dynamical models across nowcasting, medium-range, and...</span><span class="abstract-full" id="full-2610.09770" hidden>Deep learning has made rapid advances in weather forecasting: autoregressive models trained on atmospheric reanalyses now rival dynamical models across nowcasting, medium-range, and subseasonal-to-seasonal lead times, producing well-calibrated ensemble forecasts at reduced cost. We review these advances and consider their extension to climate horizons, where the challenge shifts from initial-condition skill to producing reliable statistical responses under altered forcings. AI-powered climate prediction systems must produce credible forced responses to drivers (e.g., greenhouse gases, land-use change) typically outside the observed record. We propose two minimum requirements for AI in climate modeling: (i) external forcing agents must enter explicitly enough to support interventions in which they vary independently; and (ii) robustness must be stress-tested in out-of-distribution regimes, including extremes and counterfactual trajectories. Using leading AI autoregressive emulators and hybrid physics-AI models, we identify development and coupling challenges. Comparing the reported throughput of these models with that of GPU-ported dynamical models highlights how AI can reduce time-to-solution by advancing only the target variables at the required resolution and using longer time steps, rather than integrating a full high-frequency, multivariate state. Diverse AI downscaling strategies can partially substitute for explicit fine-scale resolution, paving the way toward inexpensive local hazard assessment across prediction horizons.</span> <span class="abstract-toggle" data-id="2610.09770">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.09770v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.09770v1) · [:material-content-copy: BibTeX](../../bibtex/2610.09770.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=physics-ml-hybrid" data-tag="physics-ml-hybrid">Physics–ML hybrid</a>
    { .paper-tags }

-   #### EC-EarthFlow: Probabilistic emulation of daily transient global climate model simulations with flow matching { #2610.09715 }

    *Kirien Whan, Nikolaj T. Mücke, Karin van der Wiel* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.09715">We introduce EC-EarthFlow, a generative flow matching model that emulates simulations from the physical climate model EC-Earth3. The model is trained on transient simulations from EC-Earth3...</span><span class="abstract-full" id="full-2610.09715" hidden>We introduce EC-EarthFlow, a generative flow matching model that emulates simulations from the physical climate model EC-Earth3. The model is trained on transient simulations from EC-Earth3 (1950-2166, SSP2-4.5) to predict the day ahead temperature field from the previous days temperature as well as annual mean temperature. Predictions are made auto-regressively with rollout periods of between a month and an extended season. Using only this variable of interest, we are able to reproduce the daily variability, spatial patterns, annual cycle and long-term trend from EC-Earth3 at a substantially lower computational cost than the physical model. We demonstrate that EC-EarthFlow is stable for long inference periods, and that it can learn the physical relationships as simulated in EC-Earth3.</span> <span class="abstract-toggle" data-id="2610.09715">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.09715v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.09715v1) · [:material-content-copy: BibTeX](../../bibtex/2610.09715.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a> <a class="md-tag" href="/explore/?t=daily" data-tag="daily">Daily</a>
    { .paper-tags }

-   #### Skillful Data-Driven Subseasonal Soil Moisture Forecasting: Prospects and Limits for Flash Drought Prediction { #2610.07060 }

    *Noelia Otero, Atahan Özer, Miguel-Ángel Fernández-Torres, Jackie Ma* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.07060">Despite substantial progress in short-to-medium-range weather forecasting, predicting high-impact events such as flash droughts remains a key challenge for both early warning operations and...</span><span class="abstract-full" id="full-2610.07060" hidden>Despite substantial progress in short-to-medium-range weather forecasting, predicting high-impact events such as flash droughts remains a key challenge for both early warning operations and physically-based subseasonal-to-seasonal (S2S) prediction systems. Here we demonstrate that, for S2S soil-moisture forecasting over Europe, forecast skill depends as much on how the prediction problem is formulated as on the forecasting model itself. Using a Vision Transformer-based architecture with dual-pathway temporal and spatial attention, we show that residual learning is essential to outperform persistence. This advantage is realized only when forecasting root-zone soil moisture in physical units rather than standardized anomalies, revealing that the target representation itself constrains predictability. A probabilistic extension via quantile-head fine-tuning further provides well-calibrated predictive distributions. Benchmarked against deep-learning and operational ECMWF S2S baselines over 2021-2022, our model achieves the highest deterministic and probabilistic skill at all lead times and reliably detects anomalously dry root-zone states (below the 20th percentile). Yet flash drought onset, defined by multi-pentad intensification criteria, remains a fundamental challenge shared across all current S2S systems. These findings advance data-driven S2S soil-moisture forecasting while highlighting the remaining challenge of predicting rapid drought development.</span> <span class="abstract-toggle" data-id="2610.07060">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.07060v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.07060v1) · [:material-content-copy: BibTeX](../../bibtex/2610.07060.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=extremes" data-tag="extremes">Extremes</a> <a class="md-tag" href="/explore/?t=subseasonal-to-seasonal" data-tag="subseasonal-to-seasonal">Subseasonal to seasonal</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a>
    { .paper-tags }

-   #### ClimateBench v2.0: Probabilistic Climate Model Benchmarking { #2610.04558 }

    *Duncan Watson-Parris, Willa Tobin, Aytaç Paçal, Manuel Schlund, V. Balaji, Kevin Bowman et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.04558">We present ClimateBench v2, a standardized protocol for evaluating climate models on diagnostics expected to be informative for their skill in projecting mid-century regional temperature and...</span><span class="abstract-full" id="full-2610.04558" hidden>We present ClimateBench v2, a standardized protocol for evaluating climate models on diagnostics expected to be informative for their skill in projecting mid-century regional temperature and precipitation changes. The protocol is designed to evaluate any physics-based, data-driven, or hybrid climate model on equal footing using a common set of observational and out-of-distribution tests. We define three tiers of evaluation. Tier I establishes physical credibility through entry-ticket tests of energy conservation, coupled (co-)variability, and basic forced responses. Tier II scores models against post-2015 observations of surface temperature, precipitation, radiative fluxes, sea ice, and key modes of variability using fair CRPS as the primary probabilistic score, complemented by distributional and ensemble-consistency diagnostics. Tier III tests out-of-distribution generalization through paleoclimate simulations spanning the Last Interglacial, Last Glacial Maximum, and Mid-Holocene, and through perfect-model experiments in which data-driven models must predict the future climate of existing Earth system models from historical data alone. We reserve all observational data after 2015 for testing, and submissions must include multiple ensemble members to enable probabilistic evaluation. This reservation exploits a new opportunity provided by the decade of observations accumulated since the end of the CMIP6 historical experiment, which constitutes an out-of-sample record of forced climate change (and internal variability) for the current generation of models, and we quantify, in an idealized setting, the information it carries about mid-century warming. We provide the evaluation code, observational reference datasets, and perfect-model training data as an open benchmark to drive measurable progress in climate projection across all modeling approaches.</span> <span class="abstract-toggle" data-id="2610.04558">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.04558v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.04558v1) · [:material-content-copy: BibTeX](../../bibtex/2610.04558.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### AEGIS: Differentiable Mars Climate Model with Neural Closures { #2610.04081 }

    *Sameera S Kashyap, Victor Cruz, Angel Yepez, Razvan Marinescu* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.04081">General circulation models (GCMs) are the primary tool for simulating planetary atmospheres. They play a vital role in understanding Mars's atmosphere, as forecasting its unique weather is...</span><span class="abstract-full" id="full-2610.04081" hidden>General circulation models (GCMs) are the primary tool for simulating planetary atmospheres. They play a vital role in understanding Mars's atmosphere, as forecasting its unique weather is mission-critical for operations such as entry, descent, and landing. Mars poses unusual challenges for these models, as observations are sparse compared to Earth. In addition, a thin \co{} atmosphere alongside a radiatively active dust cycle creates a volatile atmosphere with large diurnal temperature swings and no true terrestrial analog for validation. Existing Mars GCMs, including the LMD PCM, the NASA Ames Mars GCM, and PlanetWRF, are mature and physically detailed but are implemented in legacy Fortran with finite-difference or finite-volume solvers, and they do not expose gradients for calibration or machine learning. Here we present AEGIS, a modular differentiable Mars climate model that couples Mars's unique atmospheric physics to the Dinosaur dynamical core, with interfaces for neural closures. We showcase stable ten-Mars-year simulations that reproduce the seasonal \co{} cycle while conserving the total \co{} inventory, capture realistic large-scale surface-temperature structure, and produce surface pressure that follows Mars Orbiter Laser Altimeter (MOLA) topography. Gradients through coupled trajectories agree with finite differences and support physical calibration and neural training. We compare with conventional GCMs, highlighting the framework's computational efficiency and differentiability.</span> <span class="abstract-toggle" data-id="2610.04081">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.04081v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.04081v1) · [:material-content-copy: BibTeX](../../bibtex/2610.04081.bib){ .bibtex-link }
    { .paper-links }

-   #### S2S-JEPA: Predicting the Predictable at Subseasonal-to-Seasonal Timescales { #2610.03106 }

    *Chenyu Dong, Gianmarco Mengaldo* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.03106">The subseasonal-to-seasonal (S2S) timescale, roughly from two weeks to two months ahead, is a critical forecast window for sectors such as agriculture, energy, and water management. Yet, it is widely...</span><span class="abstract-full" id="full-2610.03106" hidden>The subseasonal-to-seasonal (S2S) timescale, roughly from two weeks to two months ahead, is a critical forecast window for sectors such as agriculture, energy, and water management. Yet, it is widely known as the ‘predictability desert’. Recent AI weather models excel up to two weeks ahead but deteriorate beyond, largely because they are trained to predict fine-scale details that are neither predictable nor essential at S2S timescales. We argue that a more physically grounded objective is to forecast only the slowly varying components that remain predictable. Computer vision reached the same conclusion with the Joint-Embedding Predictive Architecture (JEPA), which predicts in latent space, discarding unpredictable details. In this work, we introduce S2S-JEPA, which brings the JEPA paradigm to S2S forecasting. It is tailored to this task through design elements from state-of-the-art AI weather models. S2S-JEPA achieves comparable skill to the gold-standard ECMWF physics-based ensemble and surpasses it on multiple metrics at weeks 5 to 6.</span> <span class="abstract-toggle" data-id="2610.03106">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.03106v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.03106v1) · [:material-content-copy: BibTeX](../../bibtex/2610.03106.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=subseasonal-to-seasonal" data-tag="subseasonal-to-seasonal">Subseasonal to seasonal</a>
    { .paper-tags }

-   #### Safe Greenhouse Climate Control Using Lagrangian-Constrained PPO with Kolmogorov-Arnold Networks { #2609.34966 }

    *Hangzun Liu, Yuling Fan, Fang Tian, Zhilong Bie, Zaiwen Feng, Yongliang Qiao* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.34966">Greenhouse climate control balances economic return with maintaining temperature, humidity and CO2 within crop-adapted growth ranges. Conventional reinforcement learning (RL) greenhouse controllers...</span><span class="abstract-full" id="full-2609.34966" hidden>Greenhouse climate control balances economic return with maintaining temperature, humidity and CO2 within crop-adapted growth ranges. Conventional reinforcement learning (RL) greenhouse controllers use fixed reward penalties to limit climate constraint violations, yet such heuristic penalties cannot explicitly constrain long-term cumulative violations. Poorly tuned weights either lead to overly conservative policies and lower yields, or fail to suppress persistent climate deviations that harm photosynthesis and induce crop diseases. To address this issue, we formulate greenhouse climate regulation as a Constrained Markov Decision Process (CMDP) and use a Lagrangian safe RL framework RCPO-PPO to separate economic optimization and cumulative safety constraints, enabling adaptive penalty adjustment without manual tuning. To handle strong nonlinear, time-varying coupling between greenhouse microclimate and crop growth, Kolmogorov-Arnold Networks (KANs) replace Multi-Layer Perceptrons (MLPs) as policy and value approximators for improved nonlinear representation. Sinusoidal cyclic time features are embedded in observations to capture diurnal environmental periodicity. Simulations use a classic winter lettuce greenhouse model driven by 40-day real weather disturbances. Compared with vanilla penalty-based PPO, our method cuts cumulative climate violations by 18.65% and raises lettuce economic profit by 2.91%, keeping violations stable near the safety threshold. This decoupled CMDP optimization with KAN-based policy representation mitigates long-term climate risks and boosts planting profits, offering a constraint-aware control strategy for precision greenhouse cultivation.</span> <span class="abstract-toggle" data-id="2609.34966">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.34966v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.34966v1) · [:material-content-copy: BibTeX](../../bibtex/2609.34966.bib){ .bibtex-link }
    { .paper-links }

-   #### Learning Hierarchical Causal Representations of the Effects of Forcings on Temperature in Climate Models { #2609.30995 }

    *Shan Zhao, Ilija Trajkovic, Julia Kaltenborn, Yaniv Gurwicz, Peer Nowack, David Rolnick et al.* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.30995">Machine learning (ML) emulators provide a fast and cost-effective method to simulate climate change scenarios after being trained on Earth System Models projections. However, the black-box nature of...</span><span class="abstract-full" id="full-2609.30995" hidden>Machine learning (ML) emulators provide a fast and cost-effective method to simulate climate change scenarios after being trained on Earth System Models projections. However, the black-box nature of those data-driven approaches limit the usability and trustworthiness of their outputs and in particular their use as causal attribution tools. Here, we develop a hierarchical causal representation learning framework applied to sea surface temperature fields from a state-of-the-art global climate model. As a key advance over previous work, our framework explicitly models both atmospheric dynamical interactions arising from internal climate variability and forced responses due to changes in atmospheric greenhouse gas and aerosol concentrations. When trained on future climate change scenarios, our method accurately predicts the long-term global mean and regional temperature evolution and shows physically realistic responses to perturbations in greenhouse gas and aerosol concentrations when evaluated on unseen scenarios. Our results underline the potential of causal representation learning frameworks for advancing climate model emulation.</span> <span class="abstract-toggle" data-id="2609.30995">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.30995v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.30995v1) · [:material-content-copy: BibTeX](../../bibtex/2609.30995.bib){ .bibtex-link }
    { .paper-links }

-   #### Understanding Perturbed Parameter Ensemble Sensitivities Using A Contrastive Learning Approach { #2609.30420 }

    *Da Fan, David John Gagne, Gregory S Elsaesser, Brian Medeiros, Addisu G Semie, Qingyuan Yang et al.* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.30420">Perturbed parameter ensembles (PPEs) reveal how physics parameters affect climate simulations, but interpreting parameter sensitivities across multivariate, spatially structured outputs remains...</span><span class="abstract-full" id="full-2609.30420" hidden>Perturbed parameter ensembles (PPEs) reveal how physics parameters affect climate simulations, but interpreting parameter sensitivities across multivariate, spatially structured outputs remains challenging, particularly when calibrating models against observations. We develop an explainable contrastive learning model that maps 5 monthly cloud and radiation fields into a shared representation space. We train the model on the fields of two 100-member Community Atmosphere Model version 6 (CAM6) PPEs, spanning 34 parameters, that only differ in the warm rain microphysics scheme: KK2000, the default bulk microphysics scheme, and TAU-ML, a neural network emulator of a bin microphysics scheme. The learned representations separates two PPEs with over 94% linear classification accuracy while preserving the seasonal variability and ensemble spread due to parameter perturbations. In the shared representation space, the representations of satellite observations occupy the same low-dimensional manifold as the PPEs but are displaced from them most strongly during boreal spring and autumn. TAU-ML PPE has a lower distance to observations compared to KK2000 in the representation space. Integrated Gradients attributions highlights the contributions in subtropical low-cloud regions, Northern and Southern Hemisphere storm track regions, and tropical convection regions to differences between PPEs and observations. Regional attributions correlate most strongly with parameters associated with cloud microphysics, boundary layer turbulence, and deep convection. These results demonstrate that explainable representations of climate fields can attribute model differences to specific variables, regions, seasons, and physical parameters.</span> <span class="abstract-toggle" data-id="2609.30420">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.30420v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.30420v1) · [:material-content-copy: BibTeX](../../bibtex/2609.30420.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=interpretability" data-tag="interpretability">Interpretability</a> <a class="md-tag" href="/explore/?t=monthly" data-tag="monthly">Monthly</a>
    { .paper-tags }

-   #### Analysis of trade-offs in urban heat mitigation using a Bayesian Optimization framework for an urban canopy layer model { #2609.25953 }

    *Rebekka Walter, Johanna Gelhaus, David Anton, Henning Wessles, Stephan Weber* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.25953">To mitigate the challenges of climate change and intensifying heat stress in urban areas, local adaptation strategies are discussed and introduced in cities worldwide. To understand processes and...</span><span class="abstract-full" id="full-2609.25953" hidden>To mitigate the challenges of climate change and intensifying heat stress in urban areas, local adaptation strategies are discussed and introduced in cities worldwide. To understand processes and potential trade-offs of these strategies a Bayesian optimization and surrogate modeling framework was employed to investigate urban parameter ranges of heat mitigation strategies with focus on three thermal metrics: daytime air temperature, Universal Thermal Climate Index (UTCI), and nighttime air temperature. Based on an urban street canyon configuration, it was shown that heat mitigation measures that reduce daytime air temperature and UTCI are often associated with higher nighttime temperatures. This results in a curved Pareto front that reflects the trade-off between daytime and nighttime thermal comfort. Under identical forcing conditions, different urban configurations, varying in geometry, vegetation, and surface characteristics, are shown to alter peak canyon air temperature by up to 5.2~$^\circ$C during the day and 2.6~$^\circ$C at night, while UTCI varies by up to 7.9~$^\circ$C, demonstrating that favorable urban configurations can substantially mitigate microclimatic heat stress. These findings suggest that combining multi-objective Bayesian optimization and surrogate modeling can help bridge the gap between computationally intensive climate simulations and practical decision-making in urban planning. An interactive visualization tool was developed to explore these trade-offs, making the often opposing relationships between urban parameters and the three thermal metrics directly accessible to planners.</span> <span class="abstract-toggle" data-id="2609.25953">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.25953v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.25953v1) · [:material-content-copy: BibTeX](../../bibtex/2609.25953.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a>
    { .paper-tags }

-   #### Learning Prognostic Variables for AI Convective Parameterizations via Symbolic Distillation { #2609.24882 }

    *Jurij Schönfeld, Tom Beucler, Julien Savre, Steven Sherwood, Veronika Eyring* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.24882">Hybrid AI-physics climate modeling aims to improve coarse (~100km-resolution) Earth system models by learning to parameterize subgrid processes from high-fidelity data. However, this so far mostly...</span><span class="abstract-full" id="full-2609.24882" hidden>Hybrid AI-physics climate modeling aims to improve coarse (~100km-resolution) Earth system models by learning to parameterize subgrid processes from high-fidelity data. However, this so far mostly involves local-in-time, diagnostic parameterizations, in which the subgrid state depends only on the current coarse state with no memory of previous states, which is unrealistic for processes such as convection that have intrinsic persistence. To address this, we enhance local-in-time parameterizations by learning prognostic variables that compactly carry important, additional past information where no explicit sub-grid information is available. First we compress past information into a low-dimensional latent space using an autoencoder, which then informs a neural network trained to parameterize targeted subgrid-scale processes. We then replace the autoencoder with symbolic equations that govern the time evolution of the latent variables, yielding additional prognostic memory variables that can be integrated alongside the resolved atmospheric state. We evaluate this approach on two systems: the Lorenz-96 model (online) and surface precipitation from high-resolution atmospheric simulations (offline). A forced multivariate linear ordinary differential equation recovers most of the added value achieved by the autoencoder-based approach in both experiments. Benchmarked against diagnostic parameterizations without memory, our memory-informed approach improves climate statistics and temporal structure, including a realistic diurnal cycle of tropical land precipitation.</span> <span class="abstract-toggle" data-id="2609.24882">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.24882v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.24882v1) · [:material-content-copy: BibTeX](../../bibtex/2609.24882.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a>
    { .paper-tags }

-   #### Climate Variability Modulates the Impact of Price Spikes on Food Insecurity { #2609.24394 }

    *Jordi Cerdà-Bautista, Vasileios Sitokonstantinou, Homer Durand, Gherardo Varando, Michele Ronco et al.* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.24394">Climate variability influences whether a market disruption escalates into a food crisis, yet broad climate patterns like El Niño, tracked months before they alter hydro-climatic conditions, are still...</span><span class="abstract-full" id="full-2609.24394" hidden>Climate variability influences whether a market disruption escalates into a food crisis, yet broad climate patterns like El Niño, tracked months before they alter hydro-climatic conditions, are still not incorporated as an early-warning component in food-security responses. We address this gap by introducing sensitivity regimes, a stratification of regions by the direction and strength of their vegetation response to the El Niño Southern Oscillation, and using them to estimate how food price spikes affect acute food insecurity across sub-Saharan Africa. Integrating remote sensing, socioeconomic data, and causal machine learning, we find that in regions where ENSO systematically suppresses vegetation, a price spike raises the share of the population at acute risk by 5.4 percentage points in the following month. In regions where vegetation is unaffected by or positively linked to ENSO, the estimated effect is smaller (around 2 percentage points) and statistically insignificant. These results demonstrate that climate context is critical for understanding food security vulnerabilities. Sensitivity regimes can be combined with operational price-spike triggers to stage anticipatory action: the ENSO state flags vulnerable regions months ahead, and a pre-positioned response in those regions to a price spike would avert the largest jump in acute food insecurity.</span> <span class="abstract-toggle" data-id="2609.24394">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.24394v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.24394v1) · [:material-content-copy: BibTeX](../../bibtex/2609.24394.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=subseasonal-to-seasonal" data-tag="subseasonal-to-seasonal">Subseasonal to seasonal</a>
    { .paper-tags }

-   #### A more predictable Madden-Julian Oscillation index derived from Koopman spectral analysis { #2609.19435 }

    *Claire Valva, Edwin P. Gerber* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.19435">The Madden-Julian oscillation (MJO) is a major source of subseasonal-to-seasonal (S2S) predictability. The MJO is commonly defined and tracked with indices such as the Real-time Multivariate MJO...</span><span class="abstract-full" id="full-2609.19435" hidden>The Madden-Julian oscillation (MJO) is a major source of subseasonal-to-seasonal (S2S) predictability. The MJO is commonly defined and tracked with indices such as the Real-time Multivariate MJO (RMM) index. Although the RMM provides a useful description of the MJO, its evolution can be noisy and difficult to predict. We define an MJO index using a data-driven approximation of the Koopman operator. The Koopman index captures similar tropical circulation and convection patterns to the RMM but evolves more smoothly and predictably. Skillful prediction extends to 46 days for the Koopman index compared to 11 days for the RMM under the same prediction framework. While this new approach does not recover the RMM as well as operational S2S models, which provide skillful forecasts up to 35 days, the Koopman index could complement existing MJO diagnostics in evaluating and developing extended-range forecast systems.</span> <span class="abstract-toggle" data-id="2609.19435">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.19435v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.19435v1) · [:material-content-copy: BibTeX](../../bibtex/2609.19435.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=subseasonal-to-seasonal" data-tag="subseasonal-to-seasonal">Subseasonal to seasonal</a>
    { .paper-tags }

-   #### A Self-Diagnosing Structural Error-Aware Parameter Estimation Method for Earth System Models { #2609.16210 }

    *Qingyuan Yang, Addisu G Semie, Brian Medeiros, Gregory S Elsaesser, Da Fan, Wayne Chuang* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.16210">We propose a fully automated, structural error-aware, interpretable climate model parameter estimation method that leverages Perturbed Parameter Ensembles (PPEs). It is based on history matching and...</span><span class="abstract-full" id="full-2609.16210" hidden>We propose a fully automated, structural error-aware, interpretable climate model parameter estimation method that leverages Perturbed Parameter Ensembles (PPEs). It is based on history matching and aligns with an increasingly-used iterative simulation-emulation-calibration methodology. The method is motivated by the negative impacts of structural error and emulator and observational uncertainties on climate model parameter estimation efforts, as well as the problems associated with sparsely-sampled PPEs. To address these challenges, the method explicitly builds simpler emulators that avoid overfitting, detect structural error, avoids compensating for structural error through inflated mismatch tolerances, and sequentially excludes structurally inconsistent variables for parameter estimation. The method decomposes the high-dimensional calibration problem into linked low-dimensional subproblems, and integrates their constraints to reconstruct the jointly plausible region of the full parameter space. The method is applied to a 100-member PPE with 34 perturbed parameters generated by a version of CAM6 with machine learning-based warm rain microphysics parameterization. Through iterative application, the method greatly reduces the ensemble spread and improves the matching between simulated and observed zonal climatologies. The method also finds ensemble members that outperform the default CAM6 configuration in root mean square error across multiple diagnostics. Controlled experiments demonstrate that overly-conservative emulator uncertainty could lead to neglect of informative observations, and tolerance of the structural error, in the context of this method, biases the estimated parameters toward compensating for structural error. Our work also emphasizes the value of interpretability for diagnosing structural error and informing parameter estimation in PPE-based calibration.</span> <span class="abstract-toggle" data-id="2609.16210">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.16210v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.16210v1) · [:material-content-copy: BibTeX](../../bibtex/2609.16210.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=interpretability" data-tag="interpretability">Interpretability</a>
    { .paper-tags }

-   #### A Physics--ML Multi-Fidelity Strategy for Earth System Model Parameter Optimization: A QG Proof-of-Concept { #2609.13275 }

    *Abdullah A. Fahad, Manmeet Singh, Donifan Barahona, Anton Darmenov, Andrea Molod* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.13275">Earth System Models rely on tunable subgrid-scale parameterizations, but optimizing these parameters is computationally expensive, particularly when nonlinear interactions require many simulations....</span><span class="abstract-full" id="full-2609.13275" hidden>Earth System Models rely on tunable subgrid-scale parameterizations, but optimizing these parameters is computationally expensive, particularly when nonlinear interactions require many simulations. We present a hybrid Physics-ML multi-fidelity framework that combines Green's Function Optimization (GFO) with Gaussian Process or Neural Network surrogate optimization. Using a quasi-geostrophic turbulence model, GFO first ranks parameter sensitivities in normalized coordinates and selects a reduced active subset. Nonlinear surrogates then explore this subset using inexpensive 30-day simulations before refining promising candidates with 180-day simulations. Across seven strategies and a 35-member ensemble, GFO-MultiGP and GFO-MultiNN achieved mean improvements of 64.6 percent and 65.2 percent, respectively, while reaching practical saturation after 3,060 and 2,520 simulation-days. The corresponding standalone GP and NN achieved 61.1 percent and 42.0 percent improvements and required 7,740 and 6,660 simulation-days. These results demonstrate an end-to-end sample-efficiency advantage for the tested hybrid pipelines. Because screening, dimensionality reduction, initialization, and fidelity scheduling change simultaneously, their individual contributions are not isolated.</span> <span class="abstract-toggle" data-id="2609.13275">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.13275v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.13275v1) · [:material-content-copy: BibTeX](../../bibtex/2609.13275.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=physics-ml-hybrid" data-tag="physics-ml-hybrid">Physics–ML hybrid</a> <a class="md-tag" href="/explore/?t=classical-ml" data-tag="classical-ml">Classical ML</a>
    { .paper-tags }

-   #### Stress-Testing Dynamical and Generative Downscaling Using Subseasonal Extreme Precipitation Forecasts { #2609.11696 }

    *Mauricio Lima, Marika Koukoula, Romain Pilon, Monika Feldmann, Erwan Koch, Daniela I. V. Domeisen et al.* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.11696">Coarse spatial resolution limits the ability of subseasonal prediction models to resolve extreme precipitation. Downscaling with either dynamical or deep generative models can overcome this issue,...</span><span class="abstract-full" id="full-2609.11696" hidden>Coarse spatial resolution limits the ability of subseasonal prediction models to resolve extreme precipitation. Downscaling with either dynamical or deep generative models can overcome this issue, but the comparative performance of these models for extremes across different atmospheric regimes remains poorly understood. In this work, we evaluate the Weather Research and Forecasting (WRF) model against a diffusion-based generative model by downscaling two physically distinct, extreme precipitation events up to lead times of 3 weeks. For a fair comparison with WRF, which can downscale boundary conditions from different driving models without model-specific training, the diffusion model is trained in an unpaired fashion. Both approaches improve upon the raw European Centre for Medium-Range Weather Forecasts forecasts, in comparison to fused rain gauge-radar observations in Switzerland (CombiPrecip), but exhibit regime-dependent strengths. WRF achieves the highest probabilistic skill for a multicell, non-stationary event. Conversely, the diffusion model is more consistent across different performance metrics for the two events, outperforming WRF in a more stationary supercell event. These results demonstrate that explicit dynamical modeling can add value for specific precipitation events for subseasonal lead times, and that generative downscaling adds value more broadly in different situations.</span> <span class="abstract-toggle" data-id="2609.11696">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.11696v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.11696v1) · [:material-content-copy: BibTeX](../../bibtex/2609.11696.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=extremes" data-tag="extremes">Extremes</a> <a class="md-tag" href="/explore/?t=subseasonal-to-seasonal" data-tag="subseasonal-to-seasonal">Subseasonal to seasonal</a>
    { .paper-tags }

-   #### Neptune: An AI model for Global Ocean Subseasonal Prediction { #2609.08606 }

    *Davide Donno, Italo Epicoco, Massimo Cafaro, Gabriele Accarino, Mohammad M. Amirian et al.* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.08606">Subseasonal-to-seasonal (S2S) forecasting is societally critical, supporting decision-making in sectors ranging from water and agricultural management to disaster risk reduction, energy planning, and...</span><span class="abstract-full" id="full-2609.08606" hidden>Subseasonal-to-seasonal (S2S) forecasting is societally critical, supporting decision-making in sectors ranging from water and agricultural management to disaster risk reduction, energy planning, and insurance. Achieving reliable predictions at these timescales requires representing the ocean and its dynamics, but traditional physics-based Ocean General Circulation Models (OGCMs), are computationally expensive and difficult to develop and improve because of the code complexity. In this work, we propose Neptune, an end-to-end data-driven framework for global ocean and sea-ice components emulation tailored for S2S timescales, up to 60 days. Neptune combines Convolutional Neural Networks (CNNs) and Spherical Fourier Neural Operators (SFNOs) to effectively capture local features and global cross-scale interactions, thereby obtaining a coherent representation of the ocean state. Forced by prescribed daily atmospheric fields, Neptune emulates ocean state variables, from temperature and salinity, to zonal and meridional currents, from sea surface height to sea ice thickness and concentration, with daily outputs at the ocean surface and through the water column. Specifically, we propose two variants of Neptune, Neptune-1 and Neptune-025, capable of emulating the ocean state at 1° and 0.25° resolution, respectively. Evaluated against a suite of metrics, including statistics (RMSE, CRPS and ACC), physical coherency (Ocean Heat Content, Eddy Kinetic Energy and Ice Brier Score) and climate indices (ENSO and Z20 metric, IOD), Neptune successfully reproduces the spatio-temporal evolution of the oceanic fields up to 60 days, and is stable over long timescales. Neptune provides compelling evidence that end-to-end data-driven ocean emulators can become a powerful component of next-generation S2S forecasting systems, emulating ocean state at high spatio-temporal resolution.</span> <span class="abstract-toggle" data-id="2609.08606">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.08606v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.08606v1) · [:material-content-copy: BibTeX](../../bibtex/2609.08606.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=neural-operators" data-tag="neural-operators">Neural operators</a> <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a> <a class="md-tag" href="/explore/?t=subseasonal-to-seasonal" data-tag="subseasonal-to-seasonal">Subseasonal to seasonal</a> <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a> <a class="md-tag" href="/explore/?t=quarter-degree" data-tag="quarter-degree">0.25°</a> <a class="md-tag" href="/explore/?t=coarse" data-tag="coarse">Coarse (≥1°)</a> <a class="md-tag" href="/explore/?t=daily" data-tag="daily">Daily</a>
    { .paper-tags }

-   #### GCMagicc v1: a fast generative emulator for multivariate climate-impact ensembles { #2609.08383 }

    *Nicolai Meinshausen, Malte Meinshausen, Jared Lewis, Zebedee Nicholls, Sarah Schöngart et al.* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.08383">Projecting the impacts of climate change requires large ensembles of climate variables that match historical observations, align with the warming ranges assessed by the IPCC, and can efficiently run...</span><span class="abstract-full" id="full-2609.08383" hidden>Projecting the impacts of climate change requires large ensembles of climate variables that match historical observations, align with the warming ranges assessed by the IPCC, and can efficiently run new future emissions scenarios, including the newest generation of climate model scenarios (CMIP7) and pathways consistent with countries' Paris Agreement pledges. Generating such ensembles at the scale needed for impact studies is normally computationally prohibitive. We close this gap with GCMagicc, a hybrid model that pairs a simple physical climate model with machine learning to generate ensembles of 10 climate variables at the resolution of full-scale Earth system models, without relying on GPU resources or retraining for new scenarios. Trained on 32 CMIP6 Earth system models and observational/reanalysis data, GCMagicc complements rather than replaces Earth system models. We apply it to a range of future pathways: the canonical SSP scenarios of the latest IPCC report (1.2-6.1°C warming, min-max across scenarios of 5-95 percentile ranges), current policies (2.3-4.0°C), national pledges under the Paris Agreement (1.5-3.3°C) and the CMIP7 range from the 'VL' to 'H' scenarios (1.2-4.2°C), releasing a large public dataset. As an illustration, we perform an attribution analysis of the severe 2025 Iranian drought using GCMagicc ensembles, with three CMIP6 large ensembles for comparison, with and without anthropogenic forcings. The results suggest a strong anthropogenic signal: a median probability of drought at least as severe as observed of 29% with anthropogenic forcing, and zero under natural-forcing-only simulations. In the future, drought conditions are projected to materially worsen, amplifying the potential for agricultural and food security impacts and geopolitical conflicts that use water scarcity as a weapon. GCMagicc data is available at https://gcmagicc.org.</span> <span class="abstract-toggle" data-id="2609.08383">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.08383v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.08383v1) · [:material-content-copy: BibTeX](../../bibtex/2609.08383.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=physics-ml-hybrid" data-tag="physics-ml-hybrid">Physics–ML hybrid</a> <a class="md-tag" href="/explore/?t=extremes" data-tag="extremes">Extremes</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### EastAsiaClimateExtremes: An AI-Ready Dataset of Weekly Atmospheric and Oceanic Extremes over East Asia for Subseasonal Prediction Research { #2609.08241 }

    *Miae Kim, Yun-Young Lee, Uran Chung* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.08241">Despite growing interest in AI-based prediction of climate extremes, event- or label-based AI-ready extreme climate datasets remain limited, constraining efforts to systematically characterize and...</span><span class="abstract-full" id="full-2609.08241" hidden>Despite growing interest in AI-based prediction of climate extremes, event- or label-based AI-ready extreme climate datasets remain limited, constraining efforts to systematically characterize and forecast such phenomena. To address this gap, we present EastAsiaClimateExtremes, an open dataset that provides ERA5/OISST reanalysis-based weekly extreme labels and event-based metrics for anomalously high temperature (AHT), heavy rainfall (HR), and marine heatwaves (MHW) over East Asia, together with analysis workflows hosted on GitHub to facilitate reproducibility and adaptation. The dataset is fully documented and co-registered with ECMWF S2S hindcast outputs on a common spatial grid and temporal framework, thereby enabling direct comparison between reanalysis-derived labels and dynamical model forecasts. This unified dataset serves as a reference framework for East Asian climate extreme research and AI-based subseasonal-to-seasonal prediction. It supports both quantitative characterization of the spatiotemporal occurrence of regional extremes and systematic diagnosis of S2S model skill in reproducing extreme signals. Beyond these immediate applications, the dataset enables a broader range of studies such as extreme event attribution and compound risk analysis. The accompanying analysis workflows characterize the historical statistics of reanalysis-based weekly extremes-including occurrence frequency and mean and maximum intensity-together with their climatological means and trend characteristics, and further assess the skill of the ECMWF hindcast.</span> <span class="abstract-toggle" data-id="2609.08241">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.08241v2) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.08241v2) · [:material-content-copy: BibTeX](../../bibtex/2609.08241.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=subseasonal-to-seasonal" data-tag="subseasonal-to-seasonal">Subseasonal to seasonal</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a>
    { .paper-tags }

-   #### Radiative and Dynamical Controls on the Land-Ocean Warming Contrast in Climate Models { #2609.03658 }

    *Paolo Giani, Arlene M. Fiore, Raffaele Ferrari, Paul A. O'Gorman, Vincent T. Cooper, Noelle E. Selin* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.03658">Surface air over land warms substantially more than over the ocean under greenhouse forcing, a phenomenon known as the land-ocean warming contrast. Current explanations for this contrast are commonly...</span><span class="abstract-full" id="full-2609.03658" hidden>Surface air over land warms substantially more than over the ocean under greenhouse forcing, a phenomenon known as the land-ocean warming contrast. Current explanations for this contrast are commonly expressed either in terms of energetic constraints, from top-of-atmosphere and surface energy balance, or dynamical constraints, from large-scale atmospheric dynamics. We show that these perspectives are complementary when viewed through the lens of atmospheric moist static energy (MSE) transport, and that connecting them yields new insight into the controls of the warming contrast and the spread in climate models. We use this framework to construct an interpretable emulator that reproduces the land-ocean warming response across 22 models from the latest Coupled Model Intercomparison Project (CMIP6). We find that the strength of the land-ocean warming contrast emerges from the interplay between a model-dependent radiative baseline and a robust dynamical restoring mechanism that favors greater warming over land. This interplay leads to two broad model regimes that align with climate sensitivity. In low-climate-sensitivity models, more stabilizing radiative feedbacks over the ocean directly favor greater land warming. In high-climate-sensitivity models, radiative feedbacks alone would instead favor greater ocean warming, but a strong MSE-transport feedback (approximately 0.2 PW/K) more than compensates for this tendency. The intermodel spread in the land-ocean warming contrast is closely related to the ratio of radiative feedbacks over land and ocean, highlighting a broader connection between climate sensitivity and the land-ocean warming contrast.</span> <span class="abstract-toggle" data-id="2609.03658">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.03658v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.03658v1) · [:material-content-copy: BibTeX](../../bibtex/2609.03658.bib){ .bibtex-link }
    { .paper-links }

-   #### A Checklist to assess the energy and carbon impacts of ML/AI applications in Earth System Modeling { #2609.00847 }

    *Filippo Dainelli, Amirpasha Mozaffari, Marina Castaño, Aina Gaya i Àvila, Lluís Palma Garcia et al.* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.00847">As machine learning and artificial intelligence find their way into nearly every aspect of climate, weather, and Earth system modeling, it is worth pausing to consider what our design decisions imply...</span><span class="abstract-full" id="full-2609.00847" hidden>As machine learning and artificial intelligence find their way into nearly every aspect of climate, weather, and Earth system modeling, it is worth pausing to consider what our design decisions imply for the science and for the computational resources we consume. A growing body of literature addresses the ethical and sustainable development of ML/AI, yet translating these principles into day-to-day research practice remains a challenge as most of best practices are dispersed across multiple studies and commentaries. Here, we distill these discussions into a practical checklist that ML/AI and Earth system science practitioners can use to assess and reduce the environmental footprint of their own applications, organised around the successive stages of the model development pipeline. We complement the checklist with a selection of metrics drawn from the literature for estimating the energy consumption and carbon footprint of a project. For each question, we point to concrete examples and actionable suggestions from recent literature, aiming to bridge the gap between aspirational principles and the decisions researchers face at every stage of the development cycle.</span> <span class="abstract-toggle" data-id="2609.00847">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.00847v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.00847v1) · [:material-content-copy: BibTeX](../../bibtex/2609.00847.bib){ .bibtex-link }
    { .paper-links }

-   #### SimCast-S2S: An Efficient Generative Model for Subseasonal Precipitation Forecasting via Transfer Learning from Climate Simulations { #2608.26594 }

    *Hiep V. Dang, Antonios Mamalakis* · Aug 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2608.26594">Subseasonal-to-seasonal (S2S) precipitation forecasting has substantial financial and societal impact, yet remains challenging because of weak predictive signals, high associated uncertainty, and the...</span><span class="abstract-full" id="full-2608.26594" hidden>Subseasonal-to-seasonal (S2S) precipitation forecasting has substantial financial and societal impact, yet remains challenging because of weak predictive signals, high associated uncertainty, and the computational cost of operational systems, which constrains simulation fidelity. We introduce SimCast-S2S, a generative latent-diffusion framework for probabilistic S2S precipitation forecasting that addresses three major bottlenecks in data-driven prediction. First, because S2S prediction requires uncertainty quantification rather than only deterministic point forecasts, SimCast-S2S is the first data-driven system that uses a diffusion-based generative pipeline for S2S prediction, enabling effective sampling from the underlying conditional distribution. Second, since generating large probabilistic ensembles is computationally costly in physical space, SimCast-S2S instead operates in a compact latent space learned by variational autoencoders, enabling efficient large-ensemble generation. Third, diffusion models typically require large training datasets; SimCast-S2S overcomes this via transfer learning with low-rank adaptation (LoRA), pretraining on large ensembles of climate simulations before fine-tuning on limited reanalysis data. On reanalysis data, SimCast-S2S outperforms deep learning baselines, including convolutional neural networks and U-Net architectures. Notably, despite using only a subset of atmospheric input variables and no post-processing, bias correction, or calibration, SimCast-S2S remains competitive with, and in many cases outperforms, state-of-the-art operational systems such as the ECMWF-S2S baseline. These results indicate that latent generative modeling combined with simulation-to-reanalysis transfer learning offers an efficient and scalable path toward data-driven probabilistic S2S precipitation forecasting.</span> <span class="abstract-toggle" data-id="2608.26594">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2608.26594v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2608.26594v1) · [:material-content-copy: BibTeX](../../bibtex/2608.26594.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=subseasonal-to-seasonal" data-tag="subseasonal-to-seasonal">Subseasonal to seasonal</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=efficiency" data-tag="efficiency">Efficiency</a>
    { .paper-tags }

-   #### UHI-Bench: Benchmarking Dual-Source Urban Heat Island Modeling Across Cities in Diverse Climate Regimes { #2608.23857 }

    *Wanyun Ling, Chenxi Liu, Yi Xie, Aopu Xu, Zhuoqi Zeng, Ziyue Li* · Aug 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2608.23857">Urban heat islands (UHIs) are intensifying under climate change, exacerbating thermal exposure risks. Their two primary observations, land surface temperature UHI (LST-UHI) and near-surface air...</span><span class="abstract-full" id="full-2608.23857" hidden>Urban heat islands (UHIs) are intensifying under climate change, exacerbating thermal exposure risks. Their two primary observations, land surface temperature UHI (LST-UHI) and near-surface air temperature UHI (AirT-UHI), capture physically distinct aspects of urban heat. However, most studies rely on a single source, and substituting one for the other can substantially bias the magnitude and spatial variability of human heat exposure. Accurate UHI modeling also requires dynamic meteorological drivers and static urban morphology features, but spatiotemporal incompatibilities hinder their alignment. Cloud gaps in LST observations and sparse AirT station networks further limit dual-source UHI modeling, motivating cross-city transfer across diverse climates. To bridge these gaps, we introduce UHI-Bench, the first UHI benchmark for dual-source UHI modeling that integrates dynamic and static environmental context. Following a unified signal, mechanism, and transfer framework, it evaluates over 20 baselines from four model families on five tasks across 20 cities and nine Köppen climate classes. Results show that no model is uniformly best, although foundation models remain consistently competitive and stable. Environmental covariates generally improve performance, but their utility varies across sources and tasks. Cross-city transferability is better explained by overlap in UHI regimes than by climate-zone similarity. With the dataset and standardized pipeline, our work provides practical guidance for urban heat modeling, promotes climate data equity, and supports future advances in climate research.</span> <span class="abstract-toggle" data-id="2608.23857">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2608.23857v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2608.23857v1) · [:material-content-copy: BibTeX](../../bibtex/2608.23857.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=foundation-models" data-tag="foundation-models">Foundation models</a>
    { .paper-tags }

-   #### DySCo: Dynamically consistent data-driven downscaling of extremes in climate projections { #2608.21998 }

    *S. Stamatelopoulos, M. Wang, I. Lopez-Gomez, L. Zepeda-Nunez, Z. Y. Wan, R. Carver, F. Sha et al.* · Aug 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2608.21998">Regional climate risk assessment is critical for applications such as infrastructure design, disaster forecasting, and insurance resource allocation. However, estimating regional (i.e.,...</span><span class="abstract-full" id="full-2608.21998" hidden>Regional climate risk assessment is critical for applications such as infrastructure design, disaster forecasting, and insurance resource allocation. However, estimating regional (i.e., high-spatial-resolution) risk with global climate models (GCMs) remains computationally prohibitive, which has driven the development of downscaling methods for coarse GCM outputs. Downscaling is vital for rare events, since quantifying their extreme properties requires high spatial resolution and very long GCM simulations. These methods non-intrusively increase GCM resolution while correcting statistical biases from unresolved fine-scale processes, thereby improving the accuracy of extreme event statistics with long return periods. A key challenge is preserving dynamical consistency, as freely evolving GCM trajectories are not expected to track the observational dataset used for training the correction operator. This is critical for causal extreme event analyses, where storyline-based risk assessment, i.e., extreme event catalogs, is necessary for effective planning. We address this challenge by introducing Dynamically and Statistically Consistent downscaling (DySCo), a non-intrusive framework yielding high-resolution climate projections consistent with coarse GCM dynamics. DySCo relies on a data-driven reformulation of nudging to create dynamically paired training trajectories without intrusive GCM modifications. Using these paired trajectories, we train a dynamically and statistically consistent, two-stage operator. We evaluate the method by downscaling the Community Earth System Model v2 Large Ensemble (LENS2) in time and space towards historical reanalysis. Results show DySCo achieves superior dynamical consistency with the coarse GCM trajectories, essentially applying a minimal, causal correction to the GCM, preserving top statistical performance comparable to state-of-the-art unsupervised models.</span> <span class="abstract-toggle" data-id="2608.21998">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2608.21998v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2608.21998v1) · [:material-content-copy: BibTeX](../../bibtex/2608.21998.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=extremes" data-tag="extremes">Extremes</a> <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a>
    { .paper-tags }

-   #### Interpretable AI predicts a 2026 summer dry anomaly in central China { #2608.19163 }

    *Anran Wang, Wen Shi, Yong Luo, Jianbin Huang, Lijuan Chen, Junhu Zhao, Weixin Jin, Huihui Yuan* · Aug 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2608.19163">Seasonal precipitation anomalies are largely regulated by atmospheric circulation, which dynamical models predict with greater reliability than precipitation itself. Here, we employ a deep learning...</span><span class="abstract-full" id="full-2608.19163" hidden>Seasonal precipitation anomalies are largely regulated by atmospheric circulation, which dynamical models predict with greater reliability than precipitation itself. Here, we employ a deep learning model that translates dynamical circulation predictions into precipitation estimates. Predictions initialized from March to May consistently indicate a dry anomaly over central China in summer 2026. Retrospective evaluations revealed higher predictive skill in the analogue years, which also tended to feature central equatorial Pacific warming persisting from the preceding winter into summer. This warming favors an anomalous cyclonic circulation over the western North Pacific-South China Sea-South China region, which induces northerly winds and moisture divergence that jointly suppress rainfall over central China. Supporting this mechanism, layer-wise relevance propagation (LRP) independently identifies these northerly winds as the dominant driver of the prediction among all model inputs. Perturbation tests supported this attribution: removing LRP-identified features effectively eliminates the dry anomaly. Our framework thus provides physically interpretable explanations for AI-derived regional climate projections, facilitating evidence-based assessment before observational data become available.</span> <span class="abstract-toggle" data-id="2608.19163">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2608.19163v2) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2608.19163v2) · [:material-content-copy: BibTeX](../../bibtex/2608.19163.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=interpretability" data-tag="interpretability">Interpretability</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a>
    { .paper-tags }

-   #### Developing an Offshore Machine Learning Surface Layer Scheme { #2608.14935 }

    *Susan Dettling, Sue Ellen Haupt, Thomas Brummet, Patrick Hawbecker, Branko Kosović, David John Gagne* · Aug 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2608.14935">Turbulent fluxes between the surface and the atmosphere are typically parameterized using empirically fit relationships. Here we test machine learning techniques for fitting the relationship for the...</span><span class="abstract-full" id="full-2608.14935" hidden>Turbulent fluxes between the surface and the atmosphere are typically parameterized using empirically fit relationships. Here we test machine learning techniques for fitting the relationship for the offshore environment. To do that, data from three offshore sites are used: the Martha's Vineyard Coastal Observatory (MVCO) air-sea interaction tower, the FINO1 research platform, and the CASPER-West FLIP research vessel deployed off the coast of California. Two machine learning methods were employed: Neural Networks (NN) and Random Forests (RF). Because the observational sites had towers with measurements at different levels, the vertical differences were input as gradients. Models were built for both momentum flux and heat flux. ML models trained at the individual sites were competitive with and in some cases, better than the physically-based COARE-3 model tailored to offshore fluxes. The heat flux ML models generally outperformed the physics-based parameterizations for most metrics, but the results were mixed for momentum flux, with only the site with the most training data (MVCO) producing results better than COARE-3. When the ML models from that site were applied to the other sites, results were degraded from using data from the site being tested. ML models built from data combined from the three sites generally showed improvements for the sites with less available training data. When assessing which variables were most important, the wind speed was most important for momentum flux and temperature gradient for heat flux.</span> <span class="abstract-toggle" data-id="2608.14935">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2608.14935v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2608.14935v1) · [:material-content-copy: BibTeX](../../bibtex/2608.14935.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=classical-ml" data-tag="classical-ml">Classical ML</a>
    { .paper-tags }

-   #### Paleoclimate Boundary Conditions as an Out-of-Sample Test for the Forced Response of Ocean Climate Emulators { #2608.13494 }

    *Adam Subel, Laure Zanna* · Aug 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2608.13494">AI weather emulators benefit from clear objectives and metrics, which have led to the rapid development of models that outperform traditional benchmarks. In contrast, long-term climate emulators must...</span><span class="abstract-full" id="full-2608.13494" hidden>AI weather emulators benefit from clear objectives and metrics, which have led to the rapid development of models that outperform traditional benchmarks. In contrast, long-term climate emulators must reliably reproduce forced responses over months to centuries, while relying on training objectives that span a small number of model time steps. We assess autoregressive, full-depth ocean emulators using data from the midHolocene experiment of a numerical climate model to examine their skill in responding to surface forcings from an in-distribution, out-of-sample climate. We demonstrate that these emulators generalize to new orbital forcings, reproducing the spatial structure of the large-scale response as well as changes in seasonal patterns and in the spatial structure of ocean variability, while underestimating their amplitude. Baselines that infer the ocean state directly from the boundary forcings also recover much of the large-scale pattern, but only near the surface, and capture neither the seasonal nor the variability changes, indicating that these require some representation of dynamics. Despite these successes, the emulators fail to reproduce the slow, internally driven evolution of the ocean interior. We then show that the emulators' total forced response is well reconstructed by linearly composing their independent responses to each forcing component. Tracking response across training epochs, we find that convergence on mean state metrics in the training climate does not guarantee that the emulators capture the dynamics necessary for a skillful response. Together, these experiments establish the midHolocene as a controlled, ground-truthed setting for diagnosing forced-response failures before emulators are pushed to out-of-distribution climates.</span> <span class="abstract-toggle" data-id="2608.13494">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2608.13494v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2608.13494v1) · [:material-content-copy: BibTeX](../../bibtex/2608.13494.bib){ .bibtex-link }
    { .paper-links }

-   #### Do AI Forecast Ensembles Sample the Correct Conditional Distribution? { #2608.08954 }

    *Lucas J. Howard, Elizabeth A. Barnes* · Aug 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2608.08954">Ensemble forecasting aims to sample the conditional distribution of outcomes; whether AI forecast ensembles do this correctly in a joint sense remains largely untested. We train a diffusion model for...</span><span class="abstract-full" id="full-2608.08954" hidden>Ensemble forecasting aims to sample the conditional distribution of outcomes; whether AI forecast ensembles do this correctly in a joint sense remains largely untested. We train a diffusion model for probabilistic subseasonal coastal sea level forecasts at eight US East Coast tide gauge stations, with sea level derived from reanalysis, and find that marginal and joint forecast quality decouple: positive skill at every station and lead time marginally, while joint spatial structure is worse than climatological draws. A shuffle-based permutation decomposition reveals this failure is invisible to the energy score but detected by the variogram score. Lorenz-96 experiments across 0.7-170 equivalent years show the gap persists regardless of training volume and is reproduced by a linear baseline, indicating structural inadequacy of the learned distribution. A dynamical ensemble does not replicate the failure while a deterministic emulator does, suggesting it is specific to learned emulators rather than ensemble forecasting generally.</span> <span class="abstract-toggle" data-id="2608.08954">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2608.08954v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2608.08954v1) · [:material-content-copy: BibTeX](../../bibtex/2608.08954.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### Probabilistic Deep Learning for Drought Forecasting: Role of Internal Climate Variability { #2608.01864 }

    *Henri Funk, Cornelia Gruber, Göran Kauermann, Helmut Küchenhoff, Magdalena Mittermeier* · Aug 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2608.01864">Predicting drought risk is essential for anticipating impacts on water resources, agriculture, ecosystems, and climate adaptation planning. Yet drought forecasts remain uncertain because variability...</span><span class="abstract-full" id="full-2608.01864" hidden>Predicting drought risk is essential for anticipating impacts on water resources, agriculture, ecosystems, and climate adaptation planning. Yet drought forecasts remain uncertain because variability can substantially alter regional precipitation and evaporative demand. Treating this variability as unstructured noise ignores the fact that internal variability has spatial, seasonal, and temporal structure and thus contains information that can be used to improve drought forecasting. We propose a deep-learning-based forecasting framework for European drought prediction and extend it with an uncertainty-aware drought bound that explicitly incorporates internal forecast variability from a large climate model ensemble. This bound represents a physically plausible lower-tail trajectory of future drought conditions and marks how severe drought could plausibly become under an unfavourable realisation of internal variability, giving adaptation planning a conservative, risk-averse reference. We compare the proposed bound with a lower bound derived from reanalysis data only and show that our proposed ensemble-informed bound is better calibrated across most regions and seasons. This is specifically true during anomalously dry conditions, when historical reanalysis alone underestimates lower-tail drought risk. Our results show that internal variability should be treated as a forecast quantity in its own right. More broadly, large ensembles provide a practical way to transfer physically plausible climate variability into machine-learning drought forecasts, yielding risk-aware bounds that are more informative for drought assessment under shifting climate conditions.</span> <span class="abstract-toggle" data-id="2608.01864">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2608.01864v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2608.01864v1) · [:material-content-copy: BibTeX](../../bibtex/2608.01864.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=extremes" data-tag="extremes">Extremes</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

</div>

<nav class="pager" markdown="span">**1** [2](2.md) [3](3.md) [4](4.md) [5](5.md) [6](6.md) [7](7.md) [8](8.md) [9](9.md) [10](10.md) [Older :material-arrow-right:](2.md){ .pager-step }</nav>

