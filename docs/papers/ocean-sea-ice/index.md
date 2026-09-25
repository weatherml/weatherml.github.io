---
title: 'Ocean & Sea Ice'
hide:
  - toc
---

<div class="listing-header" markdown>

# Ocean & Sea Ice

<p class="page-meta" markdown="span">57 papers · page 1 of 2 · <a href="../../bib/ocean-sea-ice.bib" download>:material-download: BibTeX for this topic</a></p>

</div>

<div class="grid cards" markdown>

-   #### Diffusion-Based Super-Resolution of Adriatic Sea Oceanographic Fields { #2609.22574 }

    *Rajat Srivastava, Muhammad Sarmad, Emanuele Mele, Massimo Cafaro, Marco Pulimeno, Italo Epicoco* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.22574">High-resolution oceanographic fields are critical for resolving mesoscale and sub-mesoscale coastal dynamics, yet their generation remains constrained by both computational cost and observational...</span><span class="abstract-full" id="full-2609.22574" hidden>High-resolution oceanographic fields are critical for resolving mesoscale and sub-mesoscale coastal dynamics, yet their generation remains constrained by both computational cost and observational sparsity. We present OcDiffSR, a conditional denoising diffusion probabilistic model (DDPM) for oceanographic super-resolution that reconstructs high-resolution sea-surface fields from coarse-resolution reanalysis inputs. The model is trained on ten years (2011-2020) of paired low-resolution (GLORYS12V1, 1/12) and high-resolution (Mediterranean Sea Physics Reanalysis, Med MFC, 1/24) data, and evaluated on an independent test year (2009) over the Adriatic Sea. OcDiffSR employs a conditional U-Net augmented with multi-scale low-resolution encoders, cross-attention bottleneck layers, and sinusoidal seasonal embeddings via Feature-wise Linear Modulation (FiLM), enabling joint super-resolution of sea-surface temperature (SST), salinity (SSS), and horizontal velocity components with visually coherent circulation patterns. Benchmarked against bilinear interpolation and the state-of-the-art residual diffusion model CorrDiff, OcDiffSR achieves substantially lower reconstruction errors for scalar fields (RMSESST=0.477 C, RMSESSS=0.346 psu), near-unity Pearson correlation (PCC >= 0.999), and high structural similarity (SSIM >= 0.964). For dynamical vector fields, OcDiffSR outperforms both baselines in absolute error and spatial coherence, though moderate correlation (PCC = 0.64) reflects the intrinsic stochasticity of oceanic velocity fields. Daily and monthly evaluations confirm temporal robustness across all seasons. These results establish OcDiffSR as a reliable framework for high-fidelity oceanographic downscaling and reanalysis enhancement, producing fields that are visually consistent with known ocean dynamics.</span> <span class="abstract-toggle" data-id="2609.22574">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.22574v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.22574v1) · [:material-content-copy: BibTeX](../../bibtex/2609.22574.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a> <a class="md-tag" href="/explore/?t=daily" data-tag="daily">Daily</a> <a class="md-tag" href="/explore/?t=monthly" data-tag="monthly">Monthly</a>
    { .paper-tags }

-   #### Neptune: An AI model for Global Ocean Subseasonal Prediction { #2609.08606 }

    *Davide Donno, Italo Epicoco, Massimo Cafaro, Gabriele Accarino, Mohammad M. Amirian et al.* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.08606">Subseasonal-to-seasonal (S2S) forecasting is societally critical, supporting decision-making in sectors ranging from water and agricultural management to disaster risk reduction, energy planning, and...</span><span class="abstract-full" id="full-2609.08606" hidden>Subseasonal-to-seasonal (S2S) forecasting is societally critical, supporting decision-making in sectors ranging from water and agricultural management to disaster risk reduction, energy planning, and insurance. Achieving reliable predictions at these timescales requires representing the ocean and its dynamics, but traditional physics-based Ocean General Circulation Models (OGCMs), are computationally expensive and difficult to develop and improve because of the code complexity. In this work, we propose Neptune, an end-to-end data-driven framework for global ocean and sea-ice components emulation tailored for S2S timescales, up to 60 days. Neptune combines Convolutional Neural Networks (CNNs) and Spherical Fourier Neural Operators (SFNOs) to effectively capture local features and global cross-scale interactions, thereby obtaining a coherent representation of the ocean state. Forced by prescribed daily atmospheric fields, Neptune emulates ocean state variables, from temperature and salinity, to zonal and meridional currents, from sea surface height to sea ice thickness and concentration, with daily outputs at the ocean surface and through the water column. Specifically, we propose two variants of Neptune, Neptune-1 and Neptune-025, capable of emulating the ocean state at 1° and 0.25° resolution, respectively. Evaluated against a suite of metrics, including statistics (RMSE, CRPS and ACC), physical coherency (Ocean Heat Content, Eddy Kinetic Energy and Ice Brier Score) and climate indices (ENSO and Z20 metric, IOD), Neptune successfully reproduces the spatio-temporal evolution of the oceanic fields up to 60 days, and is stable over long timescales. Neptune provides compelling evidence that end-to-end data-driven ocean emulators can become a powerful component of next-generation S2S forecasting systems, emulating ocean state at high spatio-temporal resolution.</span> <span class="abstract-toggle" data-id="2609.08606">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.08606v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.08606v1) · [:material-content-copy: BibTeX](../../bibtex/2609.08606.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=neural-operators" data-tag="neural-operators">Neural operators</a> <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a> <a class="md-tag" href="/explore/?t=subseasonal-to-seasonal" data-tag="subseasonal-to-seasonal">Subseasonal to seasonal</a> <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a> <a class="md-tag" href="/explore/?t=quarter-degree" data-tag="quarter-degree">0.25°</a> <a class="md-tag" href="/explore/?t=coarse" data-tag="coarse">Coarse (≥1°)</a> <a class="md-tag" href="/explore/?t=daily" data-tag="daily">Daily</a>
    { .paper-tags }

-   #### How well is surface ocean carbon represented in observations and ocean models? { #2609.00133 }

    *Viviana Acquaviva, Romina Wild, Alessandro Laio, Amanda R. Fay, Thea H. Heimdal, Galen A. McKinley* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.00133">We introduce a general framework for quantifying the information content and representation quality of complex geophysical datasets based on the intrinsic dimension and differentiable information...</span><span class="abstract-full" id="full-2609.00133" hidden>We introduce a general framework for quantifying the information content and representation quality of complex geophysical datasets based on the intrinsic dimension and differentiable information imbalance of data manifolds. We use it to derive and compare optimal representations of surface ocean carbon in the SOCAT database of observations and in global ocean biogeochemistry models (GOBMs) and to assess the robustness of the information we can extract from existing data. We find that within the most widely used feature set, the complexity of the data space of SOCAT observations is not fully captured by GOBMs, but the ranking and relative importance of variables learned through GOBMs are substantially correct. We observe that the learned representation of ocean carbon is less accurate in some regions, including the Southern Ocean, but doesn't appear to have evolved significantly over the last two decades. Finally, we show how the optimal representations can be used to improve the skill of distance-based machine learning models and demonstrate it for ocean carbon, and we propose two new metrics to compare models and observations that can be used to build more accurate weighted ensembles of estimates.</span> <span class="abstract-toggle" data-id="2609.00133">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.00133v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.00133v1) · [:material-content-copy: BibTeX](../../bibtex/2609.00133.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=benchmarks-datasets" data-tag="benchmarks-datasets">Benchmarks & datasets</a> <a class="md-tag" href="/explore/?t=evaluation" data-tag="evaluation">Evaluation</a>
    { .paper-tags }

-   #### Decadal wave reconstruction in the Mediterranean Sea with graph neural networks { #2608.16449 }

    *Federica Benassi, Lorenzo Mentaschi, Salvatore Causio, Daniel Holmberg, Ivan Federico, Nadia Pinardi* · Aug 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2608.16449">Accurate simulation and prediction of ocean waves are essential for coastal risk management and climate studies. Deep learning has shown promising results for wave modeling, but most approaches still...</span><span class="abstract-full" id="full-2608.16449" hidden>Accurate simulation and prediction of ocean waves are essential for coastal risk management and climate studies. Deep learning has shown promising results for wave modeling, but most approaches still operate on regular grids and on forecasting time scales, and do not generalize to unstructured discretization or to long time horizons. Here we present WaveGraph, a model based on Graph Neural Networks (GNNs) that emulates basin-scale wave dynamics directly on unstructured meshes with high resolution along the coasts (up to 2-3 km). Trained on bias-corrected simulation data over the Mediterranean Sea, WaveGraph uses a multiscale architecture combining the unstructured model mesh with a uniform graph, allowing simultaneous representation of local coastal interactions and large-scale wave dynamics. The model reconstructs the evolution of significant wave height, mean period, and mean direction, and is applied autoregressively for a continuous 17-year period without reinitialization or drift. Validation against buoy and satellite observations shows skill comparable to the input data set, and ablation experiments indicate that wind forcing drives most of the long-term stability while wave history improves swell-driven and basin-scale dynamics. These results show that GNNs can provide stable and efficient emulators of spectral wave models on unstructured domains, enabling decadal wave reconstructions.</span> <span class="abstract-toggle" data-id="2608.16449">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2608.16449v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2608.16449v1) · [:material-content-copy: BibTeX](../../bibtex/2608.16449.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=graph-neural-networks" data-tag="graph-neural-networks">Graph neural networks</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a> <a class="md-tag" href="/explore/?t=km-scale" data-tag="km-scale">Km-scale</a>
    { .paper-tags }

-   #### Stochastic Emulation of a Fully Coupled Preindustrial E3SMv3 Simulation { #2608.10277 }

    *Elynn Wu, James P. C. Duncan, Troy Arcomano, Jeremy McGibbon, Oliver Watt-Meyer et al.* · Aug 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2608.10277">We present a stochastic coupled emulator of E3SM version 3, built on the SamudrACE framework, which couples an atmosphere emulator (ACE2) with a full-depth ocean emulator (Samudra). We replace the...</span><span class="abstract-full" id="full-2608.10277" hidden>We present a stochastic coupled emulator of E3SM version 3, built on the SamudrACE framework, which couples an atmosphere emulator (ACE2) with a full-depth ocean emulator (Samudra). We replace the deterministic atmosphere emulator with its stochastic counterpart, ACE2S, and fine-tune the coupled system with a probabilistic objective, so that the atmosphere acts as a source of internal variability for the ocean. Trained on 105 years of a pre-industrial control simulation and evaluated on an independent 400 years, the emulator reproduces E3SMv3's mean climate state with biases much smaller than existing model-to-observation differences. Relative to a deterministic baseline, stochastic training maintains internal variability across timescales, most notably in the ENSO power spectrum, eddy-rich SST anomalies, and sea ice variability in the marginal ice zone. The emulator captures daily precipitation accurately up to the 99.99th percentile, but underestimates the rarest tropical extremes. These results show that stochastic coupled emulators can reproduce long-timescale variability with high fidelity, while extrapolation to unseen extremes remains a key challenge.</span> <span class="abstract-toggle" data-id="2608.10277">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2608.10277v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2608.10277v1) · [:material-content-copy: BibTeX](../../bibtex/2608.10277.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=daily" data-tag="daily">Daily</a>
    { .paper-tags }

-   #### FESOM2-JAX v1.0: a differentiable shadow of the ocean-sea-ice model FESOM2, cast onto GPUs { #2608.01546 }

    *Nikolay V. Koldunov, Sergey Danilov, Suvarchal Cheedela, Dmitry Sidorenko, Sebastian Beyer et al.* · Aug 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2608.01546">We present FESOM2-JAX, a Python re-implementation of the Finite-volumE Sea ice-Ocean Model (FESOM2) in JAX. The model retains the unstructured-mesh, cell-vertex finite-volume formulation of the...</span><span class="abstract-full" id="full-2608.01546" hidden>We present FESOM2-JAX, a Python re-implementation of the Finite-volumE Sea ice-Ocean Model (FESOM2) in JAX. The model retains the unstructured-mesh, cell-vertex finite-volume formulation of the original, runs unchanged from a laptop CPU to 256 GPUs, and is end-to-end differentiable. FESOM2-JAX is a code shadow of the Fortran model: a projection onto the Python ecosystem, translated with large language models and verified kernel by kernel against the original. It is built to lower the barrier to experimentation, from new numerics and parameterizations to gradient-based calibration and hybrid physics-machine-learning components, while remaining close enough to the original so that what is developed in the shadow can be transferred back. In a 1958-2019 hindcast at 1$^{\circ}$ equivalent resolution with identical physics and forcing, the mean states of the JAX and Fortran versions differ from each other by two orders of magnitude less than either differs from observations, and the two runs agree for six decades in global temperature, salinity, heat content, and sea ice. The complete 1$^{\circ}$ configuration fits on a single GPU, a node of four GH200 superchips integrates $\sim$113 simulated years per wall-clock day, and meshes of up to 7.4 million surface vertices ($\sim$5 km) scale to 128 GPUs. What limits the model is communication rather than arithmetic. What the shadow adds to the original is the gradient: a single reverse-mode pass through the full time loop returns the sensitivity of a model diagnostic to a parameter at every mesh vertex, verified against finite differences. To our knowledge, FESOM2-JAX is the first global ocean-sea-ice model of CMIP-class complexity written natively in a differentiable framework, and the first on an unstructured mesh.</span> <span class="abstract-toggle" data-id="2608.01546">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2608.01546v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2608.01546v1) · [:material-content-copy: BibTeX](../../bibtex/2608.01546.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=physics-ml-hybrid" data-tag="physics-ml-hybrid">Physics–ML hybrid</a> <a class="md-tag" href="/explore/?t=llms-agents" data-tag="llms-agents">LLMs & agents</a> <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a> <a class="md-tag" href="/explore/?t=coarse" data-tag="coarse">Coarse (≥1°)</a>
    { .paper-tags }

-   #### Memory compression and physical state augmentation favor different AMOC prediction tasks { #2607.28468 }

    *Mauricio Herrera-Marín* · Jul 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2607.28468">The Atlantic Meridional Overturning Circulation is monitored and emulated through reduced indices, but such projections discard thermohaline structure and may require either explicit physical state...</span><span class="abstract-full" id="full-2607.28468" hidden>The Atlantic Meridional Overturning Circulation is monitored and emulated through reduced indices, but such projections discard thermohaline structure and may require either explicit physical state or memory of the observed index. We compare these strategies in 30 branch-consistent CMIP6 trajectories from eight model families using leave-one-family-out validation. Salinity, temperature and density information improves direct 20-year forecasts, whereas compact scalar memory is top-ranked at every recursive horizon and yields the lowest case-averaged Brier score. A matched ablation confirms that feedback from memory improves long-horizon prediction. Physical state and recent trends also predict future ocean-state changes beyond the emissions pathway, most robustly at five years. NorESM under SSP5--8.5 identifies a forcing-dependent limit of scalar compression, while MIROC shows negative long-horizon transfer. A resolvent analysis explains why stable memory components do not guarantee stability of the complete learned model. Physical augmentation and memory compression therefore serve different AMOC prediction tasks.</span> <span class="abstract-toggle" data-id="2607.28468">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2607.28468v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2607.28468v1) · [:material-content-copy: BibTeX](../../bibtex/2607.28468.bib){ .bibtex-link }
    { .paper-links }

-   #### Locally stationary Argo ocean heat content estimates: Modeling, validation and uncertainty quantification { #2606.31957 }

    *Thea Sukianto, Mikael Kuusela, Donata Giglio, Anirban Mondal, Pulong Ma, Douglas W. Nychka* · Jun 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2606.31957">Argo profiling floats measure seawater temperature and salinity in the upper 2000 meters of the ocean. These floats are uniquely capable of measuring the global Ocean Heat Content (OHC), a quantity...</span><span class="abstract-full" id="full-2606.31957" hidden>Argo profiling floats measure seawater temperature and salinity in the upper 2000 meters of the ocean. These floats are uniquely capable of measuring the global Ocean Heat Content (OHC), a quantity that is of central importance for understanding Earth Energy Imbalance. Yet, producing Argo-based OHC estimates with reliable uncertainties is statistically challenging due to the complex structure and large size of the Argo dataset. Here we present an end-to-end mapping and uncertainty quantification framework for Argo-based OHC estimation using state-of-the-art methods from spatio-temporal statistics. The framework is based on modeling vertically integrated Argo temperature profiles as a locally stationary Gaussian process defined over space and time. This enables us to produce computationally tractable OHC anomaly maps based on data-driven decorrelation scales estimated from the Argo observations. Our modeling choices are validated using statistical cross-validation, which demonstrates the importance of including a climatological time trend in the mean field and accounting for time in the covariance function. We quantify the uncertainty of these maps using local conditional simulation ensembles, a novel approach that leads to principled spatially and temporally correlated uncertainty quantification. A new paired cross-validation technique is presented to validate these uncertainties. The mapping framework is implemented in an open-source codebase that is designed to be modular, reproducible and extensible. To demonstrate the mapping and uncertainty quantification capabilities of this approach, we present new Argo OHC maps with uncertainties for 2004-2022 and report on various downstream climatological estimates and their uncertainties.</span> <span class="abstract-toggle" data-id="2606.31957">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2606.31957v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2606.31957v1) · [:material-content-copy: BibTeX](../../bibtex/2606.31957.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=classical-ml" data-tag="classical-ml">Classical ML</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a>
    { .paper-tags }

-   #### Towards bridging the gap between data-driven and theoretical turbulence closures in stratified flows { #2606.20901 }

    *Laure Zanna, Pavel Perezhogin* · Jun 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2606.20901">Turbulence closure models are essential for solving the equations of motion in realistic systems, where fully resolving all relevant scales of motion is computationally infeasible. Developing...</span><span class="abstract-full" id="full-2606.20901" hidden>Turbulence closure models are essential for solving the equations of motion in realistic systems, where fully resolving all relevant scales of motion is computationally infeasible. Developing turbulence closures remains one of the most challenging problems in fluid dynamics. Specifically, the Navier-Stokes equations, when filtered to isolate large-scale motions, introduce new terms representing the influence of subgrid-scale turbulent stresses. These terms, which can only be computed directly by resolving the turbulence itself, therefore lead to the closure problem: we must add new equations or introduce assumptions to relate the unresolved scales of motions to the resolved flow. Here we consider the closure problem for oceanic flows, i.e., stratified, Boussinesq, incompressible, in a rotating frame of reference. In particular, we focus on a closure for ocean mesoscale eddies, which have horizontal scales of 10-100km and are key to the redistribution of momentum, energy, and tracers in the ocean. In particular, mesoscale eddies can reinject energy and momentum into the large-scale flow through an inverse energy cascade. Here, we explore a range of theoretical and data-driven ocean mesoscale closures and examine their connections using analytical and data-driven methods. This note aims to bridge the gap between novel methods from artificial intelligence (AI) and machine learning and theoretical fluid dynamics to address significant challenges in the physics of turbulence.</span> <span class="abstract-toggle" data-id="2606.20901">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2606.20901v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2606.20901v1) · [:material-content-copy: BibTeX](../../bibtex/2606.20901.bib){ .bibtex-link }
    { .paper-links }

-   #### Volador 1.0: A Data-Driven Air-Sea Full-Coupling Regional Forecast Model with Submesoscale-Permitting Based on MOE-Swin-Transformer Framework { #2605.24032 }

    *Yuhang Zhu, Jianxin Wang, Yu-kun Qian, Yineng Li, Yahui Liu, Yankun Gong, Shilin Tang, Shiqiu Peng et al.* · May 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2605.24032">A data-driven air-sea full-coupling regional forecast model with submesoscale-permitting, named "Volador 1.0", is developed for the South China Sea (SCS). The model features a Swin-Transformer...</span><span class="abstract-full" id="full-2605.24032" hidden>A data-driven air-sea full-coupling regional forecast model with submesoscale-permitting, named "Volador 1.0", is developed for the South China Sea (SCS). The model features a Swin-Transformer framework integrated with a Mixture-of-Experts (MoE) system, a latent space interaction architecture based on Cross-Grid Bidirectional Cross-Attention, and a fast-slow dual-branch architecture. Both the three-month hindcast test and the 15-day operational real-time forecasting demonstrate that Volador 1.0 has a very encouraging and promising performance in 0-72h forecasting of temperature and salinity in the 0-500m upper ocean as well as the sea surface height with root-mean-square-error (RMSE) or mean absolute error (MAE) smaller than or at least comparable to those from the reanalysis datasets REDOS V2.0 and GLORYS12 and the state-of-the-art regional numerical model Regional Ocean Modeling System (ROMS). In particular, Volador 1.0 demonstrates its capability of capturing/forecasting submesoscale processes including internal waves, with an energy spectrum well representing sub- to mesoscale energy cascade as expected by the classical turbulence theory. Further analysis based on ablation experiments shows that the air-sea full-coupling framework, which takes into account the dynamic exchanges of momentum and heat fluxes between the atmosphere and the ocean, indeed helps improve the model's performance compared to the non-full-coupling one. Volador 1.0, though still subject to refinement in the coming future with a large space for improvement, blazes a path for an accurate, fine and fast marine environment forecasting, and thus could help promote our capability of disaster prevention and mitigation in the SCS as well as in other coastal regions where these innovative techniques can be applied.</span> <span class="abstract-toggle" data-id="2605.24032">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2605.24032v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2605.24032v1) · [:material-content-copy: BibTeX](../../bibtex/2605.24032.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=transformers" data-tag="transformers">Transformers</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a>
    { .paper-tags }

-   #### Prediction and Predictability of the Wet-Season Rainfall over Southeast India { #2605.01326 }

    *Harini S, Devabrat Sharma, Yogenraj Patil, Gaurav Chopra, Shruti Tandon, B. N. Goswami, R. I. Sujith* · May 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2605.01326">The challenge in predicting sub-regional climate within the Indian monsoon region is exacerbated by its increasing variability in a warming world. While exploring the seasonal predictability of...</span><span class="abstract-full" id="full-2605.01326" hidden>The challenge in predicting sub-regional climate within the Indian monsoon region is exacerbated by its increasing variability in a warming world. While exploring the seasonal predictability of rainfall over the state of Tamil Nadu in southeast India, we identify an overall increase in the monthly rainfall and its variability in recent years due to an increase in surface temperature, water vapour and moisture convergence. We attribute the increasing excess rainfall to a long-term reduction in convective inhibition. We further find an increasing trend in the length of the rainy season due to an earlier onset and a delayed withdrawal of the large-scale monsoon over the southeastern and southwestern regions of southern peninsular India, respectively. Further, the simultaneous (0- month lead) predictability of the primary wet-season (October-December, OND) rainfall over Tamil Nadu is dominated by sea surface temperature (SST) anomalies in the North Indian Ocean. However, a global tropical SST climate network reveals a high potential predictability and potential to realize significant forecast skill at a lead time of up to 10 months. The long-lead predictability arises from SST and rainfall interactions across the tropical Indo-Pacific and equatorial Atlantic regions. Our findings provide a robust data-driven methodology for skillful seasonal rainfall prediction over Tamil Nadu, despite the increasing rainfall variability.</span> <span class="abstract-toggle" data-id="2605.01326">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2605.01326v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2605.01326v1) · [:material-content-copy: BibTeX](../../bibtex/2605.01326.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a> <a class="md-tag" href="/explore/?t=monthly" data-tag="monthly">Monthly</a>
    { .paper-tags }

-   #### Optimal sensor placement for the reconstruction of ocean states using differentiable Gumbel-Softmax sampling operator { #2604.22511 }

    *Oscar Chapron, Ronan Fablet, Yann Stéphan* · Apr 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2604.22511">Accurately reconstructing and forecasting ocean fields from sparse observations is critical for both operational and scientific purposes. Optimizing sensor placement to maximize reconstruction skill...</span><span class="abstract-full" id="full-2604.22511" hidden>Accurately reconstructing and forecasting ocean fields from sparse observations is critical for both operational and scientific purposes. Optimizing sensor placement to maximize reconstruction skill remains challenging due to evolving ocean dynamics and practical deployment constraints. Traditional approaches, such as Empirical Orthogonal Functions, greedy search, or Gaussian processes, either assume static observation networks or scale poorly in high-resolution and non-stationary regimes.   We introduce a differentiable adaptive sensor placement framework based on a Gumbel-Softmax sampling operator. Given an ensemble of forecasts or simulations, the method jointly optimizes a probabilistic sampling mask and the reconstruction mapping (e.g., Optimal Interpolation correlation lengths) under strict observation budgets. Numerical experiments are conducted for Sea Surface Height reconstruction in a Gulf Stream region through Observing-System Simulation Experiments using a state-of-the-art high-resolution ocean simulation.   With a sensor budget of only 0.1% (fewer than 100 point-wise observations on a 14°x14° domain) the optimized sampling reduces the reconstruction RMSE by more than half (0.0908 m versus 0.1750 m) and increases explained variance by about 20% (93.1% versus 74.4%) compared with a uniform random strategy. The method remains robust when trained on noisy ensembles with significant spatial displacement (up to 1°), demonstrating practical applicability under forecast uncertainty.   Overall, the framework provides a scalable, budget-aware approach to designing observation networks. Beyond improved skill, it yields interpretable sampling patterns that consistently target energetic regions such as eddies and fronts, offering a transferable tool for adaptive sensing in geophysical systems.</span> <span class="abstract-toggle" data-id="2604.22511">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2604.22511v2) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2604.22511v2) · [:material-content-copy: BibTeX](../../bibtex/2604.22511.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### What's in the latent space? Exploring coupled tropical Pacific variability within a Multi-branch $β$-Variational Autoencoder { #2604.07137 }

    *Emily F. Wisinski, Maria J. Molina, Kyle J. C. Hall, Hannah Bao, Salil Mahajan, Nan Rosenbloom et al.* · Apr 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2604.07137">What is encoded in the latent space of a multi-branch $β$-variational autoencoder ($β$-VAE) trained on coupled tropical Pacific climate fields? To answer this question, we assess the reconstruction...</span><span class="abstract-full" id="full-2604.07137" hidden>What is encoded in the latent space of a multi-branch $β$-variational autoencoder ($β$-VAE) trained on coupled tropical Pacific climate fields? To answer this question, we assess the reconstruction skill and physical interpretability of the latent space of a multi-branch $β$-VAE trained on sea surface temperature, ocean heat content, and outgoing longwave radiation across the tropical Pacific from a 500-year preindustrial control simulation. The model generalizes well, with only modest degradation from training to test performance, and preserves the dominant basin-scale structure of all three fields. Latent-space diagnostics show that variability is organized unevenly across dimensions: sea surface temperature is concentrated in a smaller subset of latent dimensions, whereas ocean heat content and outgoing longwave radiation are more broadly distributed across multiple dimensions. Comparisons with conventional tropical Pacific diagnostics further show that several latent dimensions align with known El Niño and La Niña variability, while others capture related coupled ocean-atmosphere variability on decadal or longer timescales. Sensitivity experiments and latent traversals identify dimensions associated with eastern-Pacific-like, central-Pacific-like, coastal, subsurface-dominant, and atmosphere-dominant variability. Together, these results show that the multi-branch $β$-variational autoencoder yields a skillful and physically informative reduced representation of coupled tropical Pacific variability.</span> <span class="abstract-toggle" data-id="2604.07137">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2604.07137v2) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2604.07137v2) · [:material-content-copy: BibTeX](../../bibtex/2604.07137.bib){ .bibtex-link }
    { .paper-links }

-   #### Impact of geophysical fields on Deep Learning-based Lagrangian drift simulations { #2604.03292 }

    *Daria Botvynko, Carlos Granero-Belinchon, Simon Van Gennip, Abdesslam Benzinou, Ronan Fablet* · Apr 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2604.03292">We assess the influence of different Eulerian geophysical input fields on Lagrangian drift simulations using DriftNet, a learning-based method designed to simulate Lagrangian drift on the sea...</span><span class="abstract-full" id="full-2604.03292" hidden>We assess the influence of different Eulerian geophysical input fields on Lagrangian drift simulations using DriftNet, a learning-based method designed to simulate Lagrangian drift on the sea surface. Two experiments are conducted: a fully numerical experiment (Benchmark B1) and a real-world drifters-based experiment (Benchmark B2). Both experiments are performed in two regions with different ocean dynamics: North East Pacific and Gulf Stream regions. The performance of DrifNet is evaluated with three different metrics: separation distance between simulated and ground-truth trajectories, the normalized cumulative Lagrangian separation and the autocorrelation of Lagrangian velocities. In both regions, results from B1 show that combining assimilated sea surface currents (SSC) with fully observed sea surface height (SSH) leads to greatest improvement in trajectory simulation. This configuration reduces separation distance by over 50% and significantly decreases normalized cumulative Lagrangian separation and metrics related to velocities autocorrelation functions compared to the baseline using SSC alone. On the other hand, the inclusion of sea surface temperature (SST) either alone or in combination with SSC generally degrades performance. In B2, using satellite-derived SSH, Ekman and winds velocities improves surface drifters trajectories simulation, particularly in the North East Pacific. While the satellite-derived SST in combination with reanalysis-based SSC configuration leads to better trajectories simulation in the Gulf Stream. Overall, we highlight the added value of combining multiple geophysical fields to improve Lagrangian drift simulation on both numerical and real-world experiments.</span> <span class="abstract-toggle" data-id="2604.03292">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2604.03292v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2604.03292v1) · [:material-content-copy: BibTeX](../../bibtex/2604.03292.bib){ .bibtex-link }
    { .paper-links }

-   #### High-resolution probabilistic estimation of three-dimensional regional ocean dynamics from sparse surface observations { #2604.02850 }

    *Niloofar Asefi, Tianning Wu, Ruoying He, Ashesh Chattopadhyay* · Apr 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2604.02850">The ocean interior regulates Earth's climate but remains sparsely observed due to limited in situ measurements, while satellite observations are restricted to the surface. We present a depth-aware...</span><span class="abstract-full" id="full-2604.02850" hidden>The ocean interior regulates Earth's climate but remains sparsely observed due to limited in situ measurements, while satellite observations are restricted to the surface. We present a depth-aware generative framework for reconstructing high-resolution three-dimensional ocean states from extremely sparse surface data. Our approach employs a conditional denoising diffusion probabilistic model (DDPM) trained on sea surface height and temperature observations with up to 99.9 percent sparsity, without reliance on a background dynamical model. By incorporating continuous depth embeddings, the model learns a unified vertical representation of the ocean states and generalizes to previously unseen depths. Applied to the Gulf of Mexico, the framework accurately reconstructs subsurface temperature, salinity, and velocity fields across multiple depths. Evaluations using statistical metrics, spectral analysis, and heat transport diagnostics demonstrate recovery of both large-scale circulation and multiscale variability. These results establish generative diffusion models as a scalable approach for probabilistic ocean reconstruction in data-limited regimes, with implications for climate monitoring and forecasting.</span> <span class="abstract-toggle" data-id="2604.02850">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2604.02850v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2604.02850v1) · [:material-content-copy: BibTeX](../../bibtex/2604.02850.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### FloeNet: A mass-conserving global sea ice emulator that generalizes across climates { #2603.12449 }

    *William Gregory, Mitchell Bushuk, James Duncan, Elynn Wu, Adam Subel, Spencer K. Clark, Bill Hurlin et al.* · Mar 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2603.12449">We introduce FloeNet, a machine-learning emulator trained on the Geophysical Fluid Dynamics Laboratory global sea ice model, SIS2. FloeNet is a mass-conserving model, emulating 6-hour mass and area...</span><span class="abstract-full" id="full-2603.12449" hidden>We introduce FloeNet, a machine-learning emulator trained on the Geophysical Fluid Dynamics Laboratory global sea ice model, SIS2. FloeNet is a mass-conserving model, emulating 6-hour mass and area budget tendencies related to sea ice and snow-on-sea-ice growth, melt, and advection. We train FloeNet using simulated data from a reanalysis-forced ice-ocean simulation and test its ability to generalize to pre-industrial control and 1% CO2 climates. FloeNet outperforms a non-conservative model at reproducing sea ice and snow-on-sea-ice mean state, trends, and inter-annual variability, with volume anomaly correlations above 0.96 in the Antarctic and 0.76 in the Arctic, across all forcings. FloeNet also produces the correct thermodynamic vs dynamic response to forcing, enabling physical interpretability of emulator output. Finally, we show that FloeNet outputs high-fidelity coupling-related variables, including ice-surface skin temperature, ice-to-ocean salt flux, and melting energy fluxes. We hypothesize that FloeNet will improve polar climate processes within existing atmosphere and ocean emulators.</span> <span class="abstract-toggle" data-id="2603.12449">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2603.12449v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2603.12449v1) · [:material-content-copy: BibTeX](../../bibtex/2603.12449.bib){ .bibtex-link }
    { .paper-links }

-   #### Data Driven Air Entrainment Velocity Parameterization by Breaking Waves { #2602.04067 }

    *Xiaohui Zhou, Anton S. Darmenov, Kianoosh Yousefi* · Feb 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2602.04067">Wave breaking injects turbulence and bubbles into the upper ocean, modulating air-sea exchange of momentum, heat, gases, and sea-spray aerosols. These fluxes depend nonlinearly on sea state but...</span><span class="abstract-full" id="full-2602.04067" hidden>Wave breaking injects turbulence and bubbles into the upper ocean, modulating air-sea exchange of momentum, heat, gases, and sea-spray aerosols. These fluxes depend nonlinearly on sea state but remain poorly represented in coupled atmosphere-wave-ocean models, where air-entrainment velocity is often parameterized using wind speed or significant wave height alone. We develop a global machine-learning parameterization of Va trained on a 43-year WAVEWATCH III simulation that resolves the breaker-front distribution and associated energetics. A multilayer perceptron with seven physically motivated predictors (wind speed, wave height, wave age, steepness, direction, and depth) reproduces spectral-reference Va with high skill. The model reduces longstanding biases in bulk formulas, notably overestimation in swell-dominated low latitudes and underestimation in storm tracks. Applied globally, it improves bubble-mediated CO2 transfer velocity and sea-salt aerosol emission, reducing errors by an order of magnitude. Validation against independent HiWinGS observations supports robust deep-water performance.</span> <span class="abstract-toggle" data-id="2602.04067">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2602.04067v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2602.04067v1) · [:material-content-copy: BibTeX](../../bibtex/2602.04067.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a>
    { .paper-tags }

-   #### Hybrid physics-data-driven modeling for sea ice thermodynamics and transfer learning { #2601.23190 }

    *Giovanni De Cillis, Alberto Carrassi, Julien Brajard, Laurent Bertino, Matteo Broccoli et al.* · Jan 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2601.23190">This study explores a physics-data driven hybrid approach for sea-ice column physics models, in which a machine learning (ML) component acts as a state-dependent parameterization of forecast errors....</span><span class="abstract-full" id="full-2601.23190" hidden>This study explores a physics-data driven hybrid approach for sea-ice column physics models, in which a machine learning (ML) component acts as a state-dependent parameterization of forecast errors. We examine how perturbations in snow thermodynamics and sea-ice radiative properties affect forecast errors, and train dedicated neural networks (NNs) for each model configuration. The performance of the hybrid models is evaluated for long lead-time forecasts and compared against a benchmark system based on climatological forecast-error estimates. The NN-based hybrids prove to be stable, robust to initial condition and atmospheric forcing errors, and consistently outperform their climatology-based counterpart. To derive guiding principles for efficiently handling possible physical model updates, we perform transfer learning experiments to test whether pretrained NNs optimized for one model configuration can be successfully adapted to another. Results indicate that direct evaluation of pretrained networks on the target task provides useful insights into their adaptability, recommending transfer learning whenever performance exceeds a trivial baseline. Finally, a feature-importance analysis shows that atmospheric forcing inputs have negligible influence on NN predictive skill, while ice-layer enthalpies play a key role in achieving satisfactory performance.</span> <span class="abstract-toggle" data-id="2601.23190">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2601.23190v2) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2601.23190v2) · [:material-content-copy: BibTeX](../../bibtex/2601.23190.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=physics-ml-hybrid" data-tag="physics-ml-hybrid">Physics–ML hybrid</a>
    { .paper-tags }

-   #### Estimation of temperature and precipitation uncertainties using quantile neural networks { #2601.17243 }

    *Andrew Brettin, Laure Zanna* · Jan 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2601.17243">Extreme events pose significant risks and are challenging to predict. Assessing climate hazards requires placing quantitative constraints on geophysical fields under observable but fluctuating...</span><span class="abstract-full" id="full-2601.17243" hidden>Extreme events pose significant risks and are challenging to predict. Assessing climate hazards requires placing quantitative constraints on geophysical fields under observable but fluctuating conditions. We propose a framework for estimating uncertainties -- a ReLU-bias loss quantile neural network (RBLQNN) -- with two novel modifications to the loss function to enforce uniform quantile accuracy and reduce degenerate predicted probability distributions. We evaluate the RBLQNN against other probabilistic baselines on a suite of datasets: synthetic datasets, observed daily temperature maxima from 1,501 NOAA Global Surface Summary of the Day (GSOD) weather stations, and altimetry-observed precipitation from the Tropical Rainfall Measuring Mission (TRMM). On synthetic datasets, the RBLQNN accurately predicts conditional distributions where more restrictive methods like linear quantile regression (LQR) or mean-variance estimation (MVE) neural networks fail, mitigates shortcomings of some other quantile neural networks, and converges stably under a range of hyperparameters. When applied to daily temperature maxima, the RBLQNN reveals that temperature distributions are relatively well described by Gaussian statistics, though nonlinear dependencies on local sea level pressure and geopotential heights appear important. For precipitation statistics, the RBLQNN strongly outperforms both LQR and MVE baselines, demonstrating its capacity to capture highly nonlinear and non-Gaussian conditional distributions. The RBLQNN's performance across varied datasets demonstrates it is a flexible and general approach for constraining uncertainties in geophysical quantities with nonlinear or non-Gaussian conditional dependencies.</span> <span class="abstract-toggle" data-id="2601.17243">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2601.17243v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2601.17243v1) · [:material-content-copy: BibTeX](../../bibtex/2601.17243.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=daily" data-tag="daily">Daily</a>
    { .paper-tags }

-   #### Super-resolution of satellite-derived SST data via Generative Adversarial Networks { #2511.22610 }

    *Claudia Fanelli, Tiany Li, Luca Biferale, Bruno Buongiorno Nardelli, Daniele Ciani, Andrea Pisano et al.* · Nov 2025
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2511.22610">In this work, we address the super-resolution problem of satellite-derived sea surface temperature (SST) using deep generative models. Although standard gap-filling techniques are effective in...</span><span class="abstract-full" id="full-2511.22610" hidden>In this work, we address the super-resolution problem of satellite-derived sea surface temperature (SST) using deep generative models. Although standard gap-filling techniques are effective in producing spatially complete datasets, they inherently smooth out fine-scale features that may be critical for a better understanding of the ocean dynamics. We investigate the use of deep learning models as Autoencoders (AEs) and generative models as Conditional-Generative Adversarial Networks (C-GANs), to reconstruct small-scale structures lost during interpolation. Our supervised -- model free -- training is based on SST observations of the Mediterranean Sea, with a focus on learning the conditional distribution of high-resolution fields given their low-resolution counterparts. We apply a tiling and merging strategy to deal with limited observational coverage and to ensure spatial continuity. Quantitative evaluations based on mean squared error metrics, spectral analysis, and gradient statistics show that while the AE reduces reconstruction error, it fails to recover high-frequency variability. In contrast, the C-GAN effectively restores the statistical properties of the true SST field at the cost of increasing the pointwise discrepancy with the ground truth observation. Our results highlight the potential of deep generative models to enhance the physical and statistical realism of gap-filled satellite data in oceanographic applications.</span> <span class="abstract-toggle" data-id="2511.22610">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2511.22610v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2511.22610v1) · [:material-content-copy: BibTeX](../../bibtex/2511.22610.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=gans" data-tag="gans">GANs</a>
    { .paper-tags }

-   #### SSTODE: Ocean-Atmosphere Physics-Informed Neural ODEs for Sea Surface Temperature Prediction { #2511.05629 }

    *Zheng Jiang, Wei Wang, Gaowei Zhang, Yi Wang* · Nov 2025
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2511.05629">Sea Surface Temperature (SST) is crucial for understanding upper-ocean thermal dynamics and ocean-atmosphere interactions, which have profound economic and social impacts. While data-driven models...</span><span class="abstract-full" id="full-2511.05629" hidden>Sea Surface Temperature (SST) is crucial for understanding upper-ocean thermal dynamics and ocean-atmosphere interactions, which have profound economic and social impacts. While data-driven models show promise in SST prediction, their black-box nature often limits interpretability and overlooks key physical processes. Recently, physics-informed neural networks have been gaining momentum but struggle with complex ocean-atmosphere dynamics due to 1) inadequate characterization of seawater movement (e.g., coastal upwelling) and 2) insufficient integration of external SST drivers (e.g., turbulent heat fluxes). To address these challenges, we propose SSTODE, a physics-informed Neural Ordinary Differential Equations (Neural ODEs) framework for SST prediction. First, we derive ODEs from fluid transport principles, incorporating both advection and diffusion to model ocean spatiotemporal dynamics. Through variational optimization, we recover a latent velocity field that explicitly governs the temporal dynamics of SST. Building upon ODE, we introduce an Energy Exchanges Integrator (EEI)-inspired by ocean heat budget equations-to account for external forcing factors. Thus, the variations in the components of these factors provide deeper insights into SST dynamics. Extensive experiments demonstrate that SSTODE achieves state-of-the-art performances in global and regional SST forecasting benchmarks. Furthermore, SSTODE visually reveals the impact of advection dynamics, thermal diffusion patterns, and diurnal heating-cooling cycles on SST evolution. These findings demonstrate the model's interpretability and physical consistency.</span> <span class="abstract-toggle" data-id="2511.05629">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2511.05629v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2511.05629v1) · [:material-content-copy: BibTeX](../../bibtex/2511.05629.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=physics-ml-hybrid" data-tag="physics-ml-hybrid">Physics–ML hybrid</a> <a class="md-tag" href="/explore/?t=interpretability" data-tag="interpretability">Interpretability</a>
    { .paper-tags }

-   #### OceanAI: A Conversational Platform for Accurate, Transparent, Near-Real-Time Oceanographic Insights { #2511.01019 }

    *Bowen Chen, Jayesh Gajbhar, Gregory Dusek, Rob Redmon, Patrick Hogan, Paul Liu et al.* · Nov 2025
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2511.01019">Artificial intelligence is transforming the sciences, yet general conversational AI systems often generate unverified "hallucinations" undermining scientific rigor. We present OceanAI, a...</span><span class="abstract-full" id="full-2511.01019" hidden>Artificial intelligence is transforming the sciences, yet general conversational AI systems often generate unverified "hallucinations" undermining scientific rigor. We present OceanAI, a conversational platform that integrates the natural-language fluency of open-source large language models (LLMs) with real-time, parameterized access to authoritative oceanographic data streams hosted by the National Oceanic and Atmospheric Administration (NOAA). Each query such as "What was Boston Harbor's highest water level in 2024?" triggers real-time API calls that identify, parse, and synthesize relevant datasets into reproducible natural-language responses and data visualizations. In a blind comparison with three widely used AI chat-interface products, only OceanAI produced NOAA-sourced values with original data references; others either declined to answer or provided unsupported results. Designed for extensibility, OceanAI connects to multiple NOAA data products and variables, supporting applications in marine hazard forecasting, ecosystem assessment, and water-quality monitoring. By grounding outputs and verifiable observations, OceanAI advances transparency, reproducibility, and trust, offering a scalable framework for AI-enabled decision support within the oceans. A public demonstration is available at https://oceanai.ai4ocean.xyz.</span> <span class="abstract-toggle" data-id="2511.01019">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2511.01019v2) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2511.01019v2) · [:material-content-copy: BibTeX](../../bibtex/2511.01019.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=llms-agents" data-tag="llms-agents">LLMs & agents</a>
    { .paper-tags }

-   #### Sensitivity Analysis for Climate Science with Generative Flow Models { #2511.00663 }

    *Alex Dobra, Jakiw Pidstrigach, Tim Reichelt, Paolo Fraccaro, Anne Jones, Johannes Jakubik et al.* · Nov 2025
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2511.00663">Sensitivity analysis is a cornerstone of climate science, essential for understanding phenomena ranging from storm intensity to long-term climate feedbacks. However, computing these sensitivities...</span><span class="abstract-full" id="full-2511.00663" hidden>Sensitivity analysis is a cornerstone of climate science, essential for understanding phenomena ranging from storm intensity to long-term climate feedbacks. However, computing these sensitivities using traditional physical models is often prohibitively expensive in terms of both computation and development time. While modern AI-based generative models are orders of magnitude faster to evaluate, computing sensitivities with them remains a significant bottleneck. This work addresses this challenge by applying the adjoint state method for calculating gradients in generative flow models. We apply this method to the cBottle generative model, trained on ERA5 and ICON data, to perform sensitivity analysis of any atmospheric variable with respect to sea surface temperatures. We quantitatively validate the computed sensitivities against the model's own outputs. Our results provide initial evidence that this approach can produce reliable gradients, reducing the computational cost of sensitivity analysis from weeks on a supercomputer with a physical model to hours on a GPU, thereby simplifying a critical workflow in climate science. The code can be found at https://github.com/Kwartzl8/cbottle_adjoint_sensitivity.</span> <span class="abstract-toggle" data-id="2511.00663">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2511.00663v3) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2511.00663v3) · [:fontawesome-brands-github: Code](https://github.com/Kwartzl8/cbottle_adjoint_sensitivity) · [:material-content-copy: BibTeX](../../bibtex/2511.00663.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=efficiency" data-tag="efficiency">Efficiency</a>
    { .paper-tags }

-   #### Digestible Pieces: comparing three options for partitioning the Northeast Pacific Coast for S2S sea surface height prediction { #2510.18133 }

    *Laura Thapa, Marybeth Arcodia, Elizabeth A. Barnes* · Oct 2025
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2510.18133">We discuss the utility of applying clustering as a preprocessing step for identifying subseasonal to seasonal forecasts of opportunity of coastal sea level using convolutional neural networks (CNNs)....</span><span class="abstract-full" id="full-2510.18133" hidden>We discuss the utility of applying clustering as a preprocessing step for identifying subseasonal to seasonal forecasts of opportunity of coastal sea level using convolutional neural networks (CNNs). Clustering leverages potential covariance among points along the same coastline or in the same ocean basin. To evaluate the utility of clustering for reliably identifying forecasts of opportunity, we compare CNNs trained to predict sea level probability distributions in three ways: over the whole Northeast Pacific Coast simultaneously, over predetermined clusters within this coastline, and at individual gridpoints near tide gauges. All CNN prediction tasks (Whole Coast, Cluster, Point), outperform climatology by a similar margin at Week 3 when the entire test set is used to evaluate CNN skill. However, when comparing the skill of each tasks' 20% most confident predictions, we find the skill of the Cluster and Point tasks to be on par with each other and substantially more skillful than the Whole Coast task. Of the Cluster and Point task, the Cluster task represents all gridpoints in the Northeast Pacific Coast with minimal tunable parameters. Throughout this exercise we learned that clustering gridpoints as a pre-processing step is the preferred approach between the three for making S2S predictions of coastal sea level.</span> <span class="abstract-toggle" data-id="2510.18133">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2510.18133v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2510.18133v1) · [:material-content-copy: BibTeX](../../bibtex/2510.18133.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a> <a class="md-tag" href="/explore/?t=subseasonal-to-seasonal" data-tag="subseasonal-to-seasonal">Subseasonal to seasonal</a>
    { .paper-tags }

-   #### Principled Operator Learning in Ocean Dynamics: The Role of Temporal Structure { #2510.09792 }

    *Vahidreza Jahanmard, Ali Ramezani-Kebrya, Robinson Hordoir* · Oct 2025
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2510.09792">Neural operators are becoming the default tools to learn solutions to governing partial differential equations (PDEs) in weather and ocean forecasting applications. Despite early promising...</span><span class="abstract-full" id="full-2510.09792" hidden>Neural operators are becoming the default tools to learn solutions to governing partial differential equations (PDEs) in weather and ocean forecasting applications. Despite early promising achievements, significant challenges remain, including long-term prediction stability and adherence to physical laws, particularly for high-frequency processes. In this paper, we take a step toward addressing these challenges in high-resolution ocean prediction by incorporating temporal Fourier modes, demonstrating how this modification enhances physical fidelity. This study compares the standard Fourier Neural Operator (FNO) with its variant, FNOtD, which has been modified to internalize the dispersion relation while learning the solution operator for ocean PDEs. The results demonstrate that entangling space and time in the training of integral kernels enables the model to capture multiscale wave propagation and effectively learn ocean dynamics. FNOtD substantially improves long-term prediction stability and consistency with underlying physical dynamics in challenging high-frequency settings compared to the standard FNO. It also provides competitive predictive skill relative to a state-of-the-art numerical ocean model, while requiring significantly lower computational cost.</span> <span class="abstract-toggle" data-id="2510.09792">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2510.09792v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2510.09792v1) · [:material-content-copy: BibTeX](../../bibtex/2510.09792.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=neural-operators" data-tag="neural-operators">Neural operators</a>
    { .paper-tags }

-   #### Down-scale marine hydrodynamic analysis at the Norwegian coast -- the NORA-SARAH open framework { #2509.21329 }

    *Widar Weizhi Wang, Konstantinos Christakos, Csaba Pakozdi, Hans Bihs* · Sep 2025
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2509.21329">Offshore wave studies often assume Gaussian processes and homogeneous wave fields. However, as waves approach the shoreline, complex coastal topo-bathymetry induces transformations such as shoaling,...</span><span class="abstract-full" id="full-2509.21329" hidden>Offshore wave studies often assume Gaussian processes and homogeneous wave fields. However, as waves approach the shoreline, complex coastal topo-bathymetry induces transformations such as shoaling, refraction, diffraction, reflection, and breaking, leading to increased nonlinearity and site-specific wave characteristics. This complexity necessitates detailed site-specific studies for coastal infrastructure design and blue economy planning. This work presents a downscaling procedure for analyzing wave-structure interactions from offshore metocean conditions. The open-access NORA3 and NORA10EI hindcast databases define offshore sea states, which are propagated to nearshore regions using the phase-averaged wave model SWAN. The outputs inform phase-resolving simulations with the fully nonlinear potential flow solver REEF3D::FNPF, incorporating an Arbitrary Eulerian-Lagrangian (ALE) method to compute wave forces via Morisons formulation and to screen for extreme events. Extreme wave loads are further examined using the fully viscous Navier-Stokes solver REEF3D::CFD. A one-way hydrodynamic coupling (HDC) between the potential flow and viscous solvers ensures accurate information transfer. The proposed NORA-SARAH framework, integrating NORA databases with SWAN, REEF3D, ALE, and HDC, offers a robust approach for complex coastal environments. A case study in Southern Norway demonstrates its advantages over traditional significant wave height (Hs)-based or phase-averaged modeling practices, highlighting the necessity of this downscaling method.</span> <span class="abstract-toggle" data-id="2509.21329">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2509.21329v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2509.21329v1) · [:material-content-copy: BibTeX](../../bibtex/2509.21329.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=classical-ml" data-tag="classical-ml">Classical ML</a>
    { .paper-tags }

-   #### Climate-Adaptive and Cascade-Constrained Machine Learning Prediction for Sea Surface Height under Greenhouse Warming { #2509.18741 }

    *Tianmu Zheng, Ru Chen, Xin Su, Julian Mak, Gang Huang, Bingzheng Yan* · Sep 2025
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2509.18741">Machine learning (ML) has achieved remarkable success in climate and marine science. Given that greenhouse warming fundamentally reshapes ocean conditions such as stratification, circulation patterns...</span><span class="abstract-full" id="full-2509.18741" hidden>Machine learning (ML) has achieved remarkable success in climate and marine science. Given that greenhouse warming fundamentally reshapes ocean conditions such as stratification, circulation patterns and eddy activity, evaluating the climate adaptability of the ML models is crucial. While physical constraints have been shown to enhance the performance of ML models, kinetic energy (KE) cascade has not been used as a constraint despite its importance in regulating multi-scale ocean motions. Here we develop two sea surface height (SSH) prediction models (with and without KE cascade constraint) and quantify their climate adaptability at the Kuroshio Extension. Both models exhibit only slight performance degradation under greenhouse warming conditions. Incorporating the KE cascade as a physical constraint significantly improves the model performance, reducing eddy kinetic energy errors by 14.7% in the present climate and 15.9% under greenhouse warming. Additional validations using satellite observations and in the Gulf Stream region further confirm the robustness of the proposed models. Compared with the KE spectrum constraint, both constraints improve the cross-scale transfer and spectrum of KE, but the KE cascade constraint yields larger improvements in the cross-scale transfer. This work presents the first application of the KE cascade as a physical constraint for ML-based ocean state prediction and demonstrates its robust adaptability across climates, offering guidance for the further development of global ML models for both present and future conditions.</span> <span class="abstract-toggle" data-id="2509.18741">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2509.18741v2) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2509.18741v2) · [:material-content-copy: BibTeX](../../bibtex/2509.18741.bib){ .bibtex-link }
    { .paper-links }

-   #### Artificial neural networks ensemble methodology to predict significant wave height { #2509.14020 }

    *Felipe Crivellaro Minuzzi, Leandro Farina* · Sep 2025
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2509.14020">The forecast of wave variables are important for several applications that depend on a better description of the ocean state. Due to the chaotic behaviour of the differential equations which model...</span><span class="abstract-full" id="full-2509.14020" hidden>The forecast of wave variables are important for several applications that depend on a better description of the ocean state. Due to the chaotic behaviour of the differential equations which model this problem, a well know strategy to overcome the difficulties is basically to run several simulations, by for instance, varying the initial condition, and averaging the result of each of these, creating an ensemble. Moreover, in the last few years, considering the amount of available data and the computational power increase, machine learning algorithms have been applied as surrogate to traditional numerical models, yielding comparative or better results. In this work, we present a methodology to create an ensemble of different artificial neural networks architectures, namely, MLP, RNN, LSTM, CNN and a hybrid CNN-LSTM, which aims to predict significant wave height on six different locations in the Brazilian coast. The networks are trained using NOAA's numerical reforecast data and target the residual between observational data and the numerical model output. A new strategy to create the training and target datasets is demonstrated. Results show that our framework is capable of producing high efficient forecast, with an average accuracy of $80\%$, that can achieve up to $88\%$ in the best case scenario, which means $5\%$ reduction in error metrics if compared to NOAA's numerical model, and a increasingly reduction of computational cost.</span> <span class="abstract-toggle" data-id="2509.14020">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2509.14020v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2509.14020v1) · [:material-content-copy: BibTeX](../../bibtex/2509.14020.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a> <a class="md-tag" href="/explore/?t=recurrent-networks" data-tag="recurrent-networks">Recurrent networks</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### DiffTopo: Solver in the Loop for Inverse Topography via Condition Diffusion Generation { #2509.00007 }

    *Aoming Liang, Qi Liu, Weicheng Cui* · Sep 2025
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2509.00007">Inferring seabed topography from wave height observations is fundamental to tsunami hazard assessment, coastal planning, and large scale ocean circulation modeling. Classical inversion models...</span><span class="abstract-full" id="full-2509.00007" hidden>Inferring seabed topography from wave height observations is fundamental to tsunami hazard assessment, coastal planning, and large scale ocean circulation modeling. Classical inversion models typically rely on direct sensing or optimization based schemes that must contend with the strongly nonlinear coupling between free surface dynamics and topography. However, data driven approaches are capable of tackling strongly nonlinear problems by learning the underlying data distributions. This study introduces DiffTopo, a conditional diffusion model that reconstructs topography from surface wave field data governed by shallow water equations. Leveraging classifier free guidance, DiffTopo not only generates a series of solutions but also applies a thresholding mechanism that ensures, via the solver, the validation results are physically plausible. This study evaluates both observed wave fields and three distinct topography configurations, demonstrating that DiffTopo exhibits robust generalization and remains consistent with the shallow water equations even under full observations. These results underscore the potential of diffusion based generative modeling for addressing ill posed inverse problems in geophysics.</span> <span class="abstract-toggle" data-id="2509.00007">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2509.00007v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2509.00007v1) · [:material-content-copy: BibTeX](../../bibtex/2509.00007.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a>
    { .paper-tags }

-   #### Ensembles of Neural Surrogates for Parametric Sensitivity in Ocean Modeling { #2508.16489 }

    *Yixuan Sun, Romain Egele, Sri Hari Krishna Narayanan, Luke Van Roekel, Carmelo Gonzales et al.* · Aug 2025
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2508.16489">Accurate simulations of the oceans are crucial in understanding the Earth system. Despite their efficiency, simulations at lower resolutions must rely on various uncertain parameterizations to...</span><span class="abstract-full" id="full-2508.16489" hidden>Accurate simulations of the oceans are crucial in understanding the Earth system. Despite their efficiency, simulations at lower resolutions must rely on various uncertain parameterizations to account for unresolved processes. However, model sensitivity to parameterizations is difficult to quantify, making it challenging to tune these parameterizations to reproduce observations. Deep learning surrogates have shown promise for efficient computation of the parametric sensitivities in the form of partial derivatives, but their reliability is difficult to evaluate without ground truth derivatives. In this work, we leverage large-scale hyperparameter search and ensemble learning to improve both forward predictions, autoregressive rollout, and backward adjoint sensitivity estimation. Particularly, the ensemble method provides epistemic uncertainty of function value predictions and their derivatives, providing improved reliability of the neural surrogates in decision making.</span> <span class="abstract-toggle" data-id="2508.16489">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2508.16489v2) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2508.16489v2) · [:material-content-copy: BibTeX](../../bibtex/2508.16489.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

</div>

<nav class="pager" markdown="span">**1** [2](2.md) [Older :material-arrow-right:](2.md){ .pager-step }</nav>

