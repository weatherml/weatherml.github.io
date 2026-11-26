---
hide:
  - navigation
  - toc
title: weatherml
---

A collection of papers on AI for weather forecasting, climate modelling and atmospheric science.

<p class="page-meta" markdown="span">1682 papers · updated 2026-11-26 · <a href="feed.xml">:material-rss: RSS</a> · <a href="all_papers.bib" download>:material-download: BibTeX</a> · <a href="https://github.com/weatherml/weatherml.github.io/issues/new?template=suggest-paper.yml">:material-plus: Suggest a paper</a></p>

## Browse by Topic

<div class="grid cards topics" markdown>

-   [Global Models](papers/global-models/index.md) <span class="topic-count">401</span>
-   [Regional Models](papers/regional-models/index.md) <span class="topic-count">70</span>
-   [Nowcasting](papers/nowcasting/index.md) <span class="topic-count">111</span>
-   [Downscaling](papers/downscaling/index.md) <span class="topic-count">111</span>
-   [Post-processing](papers/post-processing/index.md) <span class="topic-count">27</span>
-   [Data Assimilation](papers/data-assimilation/index.md) <span class="topic-count">84</span>
-   [Climate Modeling](papers/climate-modeling/index.md) <span class="topic-count">320</span>
-   [Hydrology](papers/hydrology/index.md) <span class="topic-count">41</span>
-   [Ocean & Sea Ice](papers/ocean-sea-ice/index.md) <span class="topic-count">94</span>
-   [Air Quality & Composition](papers/air-quality-composition/index.md) <span class="topic-count">79</span>
-   [Remote Sensing](papers/remote-sensing/index.md) <span class="topic-count">113</span>
-   [Other](papers/other/index.md) <span class="topic-count">231</span>

</div>

## Recent Additions

<div class="grid cards" markdown data-search-exclude>

-   #### Investigating the efficacy of vortex beam speckle patterns for the recognition of atmospheric turbulence and fractional topological charge with a deep learning model { #2611.34573 }

    *Ratikanta Swain, Allarakha Shikder, Naseem Babu, Niladri Sarkar, Aryan Raj, Raghavan K. Easwaran* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.34573">Fractional topological charge (TC) enlarges the usable mode set of orbital angular momentum (OAM) links, but reading it reliably after atmospheric turbulence remains difficult. Interferometric and...</span><span class="abstract-full" id="full-2611.34573" hidden>Fractional topological charge (TC) enlarges the usable mode set of orbital angular momentum (OAM) links, but reading it reliably after atmospheric turbulence remains difficult. Interferometric and diffractive methods need an undistorted beam and a careful alignment, while most learned approaches treat integer or coarsely spaced charges and do not include turbulence. Here we ask whether the charge of a scalar vortex beam can be regressed directly from a single speckle image, and at which turbulence strength this stops working. We propagate vortex beams over 1500~m through 150 von Kármán phase screens, scatter them with a fixed random diffuser, and record $256\times256$, 16-bit speckle images for TC from 0 to 10 in steps of 0.01 and turbulence strengths $D/r_0$ from 0 to 50; the results reported here use a subset of 20\,000 images, and a data set of $10^{6}$ images is in preparation. A convolutional neural network estimates the TC and, through an auxiliary head, the turbulence category. On charges withheld from training, the mean absolute error is 0.031 without turbulence (with a common intensity scale) and rises to 0.78 at $D/r_0=10$ and 1.85 at $D/r_0=50$, against 2.5 for a constant prediction, and the turbulence category is identified correctly in 78% of the images. The results show that, at the present training size, the charge is read accurately only without turbulence, and that the charge and the turbulence oth act on the speckle grain size, which may limit the readout. Experimental validation and the extension to vector vortex beams are left for future work.</span> <span class="abstract-toggle" data-id="2611.34573">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.34573v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.34573v1) · [:material-content-copy: BibTeX](bibtex/2611.34573.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a>
    { .paper-tags }

-   #### LatENSO v1.0: a learned latent state of the tropical Pacific for seasonal ENSO operators and forecasting { #2611.34296 }

    *Milad Panahi, Annalisa Bracco* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.34296">Here we present LatENSO v1.0, a machine-learning forecast system for the El Niño-Southern Oscillation (ENSO) built around a learned, 32-dimensional state of the tropical Pacific. A Swin-transformer...</span><span class="abstract-full" id="full-2611.34296" hidden>Here we present LatENSO v1.0, a machine-learning forecast system for the El Niño-Southern Oscillation (ENSO) built around a learned, 32-dimensional state of the tropical Pacific. A Swin-transformer autoencoder compresses thirteen monthly-averaged ocean and atmosphere fields into 32 numbers, trained to reconstruct the fields, with no forecast target. The 32 numbers rebuild the fields as well as O(100) empirical orthogonal functions. Two seasonal operators use this state. The first is the extended nonlinear recharge oscillator (XRO) with two directions of the learned state among its variables; we call it XRO-lat. The second is a Floquet modal operator (FMO) of three damped oscillators, which forecasts a correction added to a least-squares base forecast made from four observed predictors. We evaluate both against the recharge oscillator, XRO and a deep echo-state network (DESN), all refitted on identical data, cross-validation folds and verification windows (1981-2024). Over lead times of 1-18 months, XRO-lat reaches an anomaly correlation coefficient (ACC) of 0.673, the highest of all tropical-Pacific-only (in-domain) systems and statistically indistinguishable from XRO and DESN with their eight remote-basin indices (0.685 and 0.667). From lead 15 to lead 24 it is the most skillful of all systems. Over leads 1-18, the FMO forecast (0.619) is statistically indistinguishable from every in-domain reference model (0.631-0.654). What the FMO adds to its base forecast, 0.054 in ACC, is carried by its Floquet propagation, whose generator consistently holds a prominent quasi-biennial oscillator. At leads 3-9 the FMO trails XRO, DESN and the in-domain XRO, a deficit partly attributable to its base. LatENSO combines competitive skill with trustworthiness: a compact state whose physical content can be inspected, and fully open access to code, weights and data.</span> <span class="abstract-toggle" data-id="2611.34296">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.34296v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.34296v1) · [:material-content-copy: BibTeX](bibtex/2611.34296.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=transformers" data-tag="transformers">Transformers</a> <a class="md-tag" href="/explore/?t=subseasonal-to-seasonal" data-tag="subseasonal-to-seasonal">Subseasonal to seasonal</a> <a class="md-tag" href="/explore/?t=monthly" data-tag="monthly">Monthly</a>
    { .paper-tags }

-   #### AlignPre: Skill-Aligned Post-Training for Latent-Free Generative Precipitation Nowcasting { #2611.34101 }

    *Demin Yu, Kenghong Lin, Xutao Li* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.34101">Precipitation nowcasting requires accurate short-term predictions under rapidly evolving spatiotemporal dynamics. Generative diffusion models can produce detailed and competitive radar forecasts by...</span><span class="abstract-full" id="full-2611.34101" hidden>Precipitation nowcasting requires accurate short-term predictions under rapidly evolving spatiotemporal dynamics. Generative diffusion models can produce detailed and competitive radar forecasts by modeling conditional distributions. However, most existing approaches rely on separately trained components, such as deterministic branches or latent compression modules. More fundamentally, distribution matching alone fails to explicitly optimize operational forecast skill. We propose AlignPre, a streamlined pixel-space nowcasting framework that bridges this gap via skill-aligned post-training (SAPT). AlignPre employs an end-to-end flow-matching forecaster with a hierarchical spatiotemporal architecture for distribution matching and direct radar-field synthesis, without latent compression. During post-training, SAPT exploits the analytic structure of forecast verification scores to construct a differentiable objective. By backpropagating gradients of this objective from radar fields to the generator, SAPT turns forecast verification into direct training feedback. Across three real-world benchmarks spanning different forecast horizons, AlignPre consistently surpasses state-of-the-art deterministic and generative baselines in forecast skill, demonstrating that aligning generative modeling with domain-specific verification provides an effective paradigm for meteorological forecasting.</span> <span class="abstract-toggle" data-id="2611.34101">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.34101v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.34101v1) · [:material-content-copy: BibTeX](bibtex/2611.34101.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=evaluation" data-tag="evaluation">Evaluation</a>
    { .paper-tags }

-   #### Evaluation of Machine-Learning Weather Models in Reproducing Mesoscale Tornado-Favorable Convective Environments { #2611.31564 }

    *Yiding Luo, Rui Yuan, Shetian Xia, Royce Lee, Alexandra Anderson-Frey* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.31564">Machine-learning (ML)-based weather prediction models have developed rapidly in recent years and are already being used in operational weather forecasting. These ML-based models' performance has...</span><span class="abstract-full" id="full-2611.31564" hidden>Machine-learning (ML)-based weather prediction models have developed rapidly in recent years and are already being used in operational weather forecasting. These ML-based models' performance has exceeded that of traditional numerical weather prediction (NWP) methods in certain implementations, with substantially lower computational cost. However, most ML-based models were trained on global data at a horizontal grid spacing of 0.25 degrees, which may be too coarse to capture tornado-favorable parameters. The ability of these ML-based models to predict the parameters relevant for tornado-favorable convection remains unclear. Here, we compare the performance of three ML-based models, Pangu-Weather, FengWu (versions 1 and 2), and FuXi Weather, in forecasting ten distinct convective events associated with strong tornadoes that occurred in Oklahoma, USA. The proposed methodology incorporates five lead times, ranging from 24 to 168 h, for each event and uses reanalysis data as both model input and ground truth. We evaluate the models by forecasting eight tornado-relevant parameters and by conducting spatial and value-consistency checks between model outputs and ground truth. Performance varies across parameters, models, and events. Pangu-Weather performs well for positive storm-relative helicity, while FuXi Weather performs well for several thermodynamic parameters, albeit with some overprediction of magnitude. Increasing lead time is associated with loss of extreme-value and spatial-structure fidelity. In the Global Forecast System (GFS) case study, ML-based models retain convective signals at longer lead times but poorly represent fine-scale structures such as cold pool-like features. These findings indicate that model-specific biases and spatial extremes should be assessed alongside domain-wide forecast scores.</span> <span class="abstract-toggle" data-id="2611.31564">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.31564v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.31564v1) · [:material-content-copy: BibTeX](bibtex/2611.31564.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=evaluation" data-tag="evaluation">Evaluation</a> <a class="md-tag" href="/explore/?t=quarter-degree" data-tag="quarter-degree">0.25°</a>
    { .paper-tags }

-   #### A Comparison of Uncertainty in AI and Dynamical Climate Ensembles { #2611.31064 }

    *Jacob B. Landsberg, Devin M. McAfee, M. A. Fernandez, Elizabeth A. Barnes* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.31064">Climate model ensembles and their uncertainty estimates are essential for evaluating risk and designing climate policy. Yet, in the assessment of newly developed AI climate models, evaluating...</span><span class="abstract-full" id="full-2611.31064" hidden>Climate model ensembles and their uncertainty estimates are essential for evaluating risk and designing climate policy. Yet, in the assessment of newly developed AI climate models, evaluating uncertainty is often secondary to diagnosing mean-state climate biases and trends. In this work, we conduct the first systematic comparison of uncertainty representation in ensembles of state-of-the-art AI and dynamical atmospheric climate models. Examining 2-meter temperature, 500 hPa geopotential height, and precipitable water, we find that the AI ensemble exhibits reduced internal variability and model uncertainty across variables relative to the dynamical ensemble. We show that this dampened variability causes the AI ensemble to produce more limited climate extremes and yields underdispersion relative to ERA5. Under a uniform warming scenario, the AI ensemble shows increased model uncertainty. Dynamical models grapple with their own challenges, exhibiting stronger miscalibration (overdispersion) relative to ERA5 and larger climate biases than the AI models. As AI and dynamical climate models are mutually integrated into operational climate prediction, careful consideration of their differences in uncertainty representations is crucial for grounded assessment of climate risk.</span> <span class="abstract-toggle" data-id="2611.31064">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.31064v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.31064v1) · [:material-content-copy: BibTeX](bibtex/2611.31064.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### An exact and differentiable rewrite of a coupled weather and fire model { #2611.30866 }

    *John S. Schreck, John Dennis, Iris Linck, Maria Frediani, Pedro A. Jimenez Munoz, Matthew Rehme et al.* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.30866">AI is changing weather and climate prediction by replacing physical models with learned ones. Here we show a different path, in which AI coding agents rebuild the physical model itself in a...</span><span class="abstract-full" id="full-2611.30866" hidden>AI is changing weather and climate prediction by replacing physical models with learned ones. Here we show a different path, in which AI coding agents rebuild the physical model itself in a machine-learning framework and reproduce its forecast bit for bit. We reconstruct selected configurations of the Weather Research and Forecasting model and its coupled fire physics in PyTorch, treating every difference from the corrected Fortran code as a bug. Across twelve hours of a Marshall Fire simulation, the two independently advancing codes agree bit for bit at all 365 checkpoints, in the atmosphere, fire and firebrand particles. This agreement holds when the code runs one operation at a time, with the Fortran code's mathematical library. On a separate fast route, the reconstructed model runs about five times faster on one GPU than the double-precision Fortran code on a 128-core CPU node. One backward pass maps the fire's sensitivity to the wind at every grid point. In synthetic tests, its derivatives show how to change wind and fuel moisture to correct a misplaced fire front. Neural networks replace the radiation and fire-spread calculations in 24-hour coupled forecasts, and we measure their effect against the physical calculation. An established model thus becomes fast, differentiable and open to learned components, with its tested calculation intact.</span> <span class="abstract-toggle" data-id="2611.30866">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.30866v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.30866v1) · [:material-content-copy: BibTeX](bibtex/2611.30866.bib){ .bibtex-link }
    { .paper-links }

-   #### Geospatial foundation models and deep learning for climate resilience: a review of transfer, uncertainty and decision readiness { #2611.30198 }

    *Parastoo Farajpoor, Mohammadreza Narimani* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.30198">Geospatial foundation models can reuse information learned from large Earth-observation archives, but their value for climate resilience depends on whether that information transfers to unfamiliar...</span><span class="abstract-full" id="full-2611.30198" hidden>Geospatial foundation models can reuse information learned from large Earth-observation archives, but their value for climate resilience depends on whether that information transfers to unfamiliar events, regions and observing conditions. This review connects advances in deep learning with the evidence needed for hazard monitoring and early warning. A descriptive audit of 137 full-text hazard-application studies is interpreted alongside a separate corpus of 37 model and benchmark records. Ninety-eight application studies reported an out-of-distribution experiment, whereas headline evaluations included held-out events in 31 studies, regions in 25 and sensors in only two. None of the ten studies coded for geospatial foundation-model fine-tuning used sensor holdout as a headline test. Uncertainty-related methods were recorded in 16 studies, including two with explicitly calibrated probabilistic output. These overlapping counts describe evaluation practice within the coded corpus, not a ranking of model families. We distinguish Earth-observation representation learning from atmospheric forecasting and examine how labels, observability, forecast lead time and decision requirements constrain their contribution across flood, wildfire, drought, heat, cryosphere and cascading-hazard applications. We then develop a decision-backward evaluation framework: define the action and information available when it must be taken, reproduce the relevant shift, retain a matched task-specific baseline, and report misses, false alarms, reliability and end-to-end latency. The framework turns transferability from a broad model claim into a testable requirement for climate adaptation and risk management.</span> <span class="abstract-toggle" data-id="2611.30198">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.30198v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.30198v1) · [:material-content-copy: BibTeX](bibtex/2611.30198.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=foundation-models" data-tag="foundation-models">Foundation models</a> <a class="md-tag" href="/explore/?t=extremes" data-tag="extremes">Extremes</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### ESM-Bench: A Benchmark for Evaluating Whether AI Agents Understand Earth System Model Physics and Code { #2611.28477 }

    *Koutian Wu, Yuxuan Cao, Gengchen Mai, Zeping Liu* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.28477">AI coding agents that modify Earth System Model (ESM) code must preserve both software interfaces and the physical assumptions encoded in them. We introduce ESM-Bench, a benchmark of 60...</span><span class="abstract-full" id="full-2611.28477" hidden>AI coding agents that modify Earth System Model (ESM) code must preserve both software interfaces and the physical assumptions encoded in them. We introduce ESM-Bench, a benchmark of 60 developer-commit tasks from 6 repositories: Noah-MP, CLM5, SUMMA, ParFlow, MOM6, and VIC. The v5 corpus contains 17 physics-based bug fixes, 16 process-representation modifications, 13 parameterization-scheme selections, and 14 parameter-optimization tasks. Each task includes a scientific objective, parent-commit source files, a reference diff, and three stored prompt variants. An artifact audit confirms that all 60 reference diffs apply to their supplied starting files. We define an eight-item patch-review rubric and audit five archived patch fixtures against the implemented local checkers. None of the five fixture diffs applies to its supplied starting files, yet three text-pattern checkers return a pass. A ParFlow expression replay disagrees with the reference slopes, and a MOM6 pattern check detects an omitted update. Inspection also identifies a retained VIC reference assignment that overwrites the intended vegetation-specific albedo. These findings expose limits of reference patches and local proxies as correctness evidence. The release provides a reproducible corpus and diagnostic tools; complete expert validation, comparative agent performance, and full-model execution remain unmeasured on v5.</span> <span class="abstract-toggle" data-id="2611.28477">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.28477v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.28477v1) · [:material-content-copy: BibTeX](bibtex/2611.28477.bib){ .bibtex-link }
    { .paper-links }

-   #### Stochastic climate emulation at reduced cost with ACE2.5-ERA5 { #2611.28315 }

    *Jeremy McGibbon, Troy Arcomano, Spencer K. Clark, James P. C. Duncan, Brian Henn, Oliver Watt-Meyer* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.28315">We introduce ACE2.5-ERA5 (Ai2 Climate Emulator of ERA5, version 2.5), a stochastic autoregressive weather-climate AI model that produces calibrated weather ensembles with realistic small-scale...</span><span class="abstract-full" id="full-2611.28315" hidden>We introduce ACE2.5-ERA5 (Ai2 Climate Emulator of ERA5, version 2.5), a stochastic autoregressive weather-climate AI model that produces calibrated weather ensembles with realistic small-scale structure on timescales from days to centuries. Its deterministic predecessor, ACE2-ERA5, retains 36% of its ERA5 target's precipitation spectral power at total wavenumbers 100--179; ACE2.5-ERA5 retains 86%. In weather forecasts its ensemble spread matches its error to within 4% from day 3. Time-mean climate is on par with or better than ACE2-ERA5, the precipitation response to ENSO is improved, and the model is stable over 1000 years. Interannual variability of global mean surface temperature is shown not to be calibrated in either ACE2.5-ERA5 or its predecessor. ACE2.5-ERA5 differs from ACE2-ERA5 in three ways. Noise-conditioned layer normalization, trained with a combined CRPS and spectral energy score loss in place of mean squared error, makes it stochastic. Spectral bottlenecking with grouped convolutions cuts the Spherical Fourier Neural Operator's spectral filter parameters by more than two orders of magnitude and leaves a model a tenth the size of ACE2-ERA5. Its time step is daily rather than 6-hourly. ACE2.5-ERA5 has 3.2 times the inference throughput of ACE2-ERA5. A 6-hourly variant has weather prediction skill similar to ACE2-ERA5. A 4-degree variant reaches comparable climate skill at an order of magnitude lower training cost, enabling rapid iteration during development. We release 1-degree daily, 4-degree daily and 1-degree 6-hourly checkpoints.</span> <span class="abstract-toggle" data-id="2611.28315">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.28315v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.28315v1) · [:material-content-copy: BibTeX](bibtex/2611.28315.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=6-hourly" data-tag="6-hourly">6-hourly</a> <a class="md-tag" href="/explore/?t=daily" data-tag="daily">Daily</a>
    { .paper-tags }

-   #### Beyond Resolution: Inference-Time Parameterized Temporal Consistency in Generative Regional Climate Downscaling { #2611.27641 }

    *Luca Glawion, Julius Polz, Harald Kunstmann, Benjamin Fersch, Patrick Ludwig et al.* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.27641">GCM-to-RCM downscaling emulation is dominated by generative diffusion models, but most existing work evaluates skill only per timestep, treating each field as an independent snapshot. This leaves...</span><span class="abstract-full" id="full-2611.27641" hidden>GCM-to-RCM downscaling emulation is dominated by generative diffusion models, but most existing work evaluates skill only per timestep, treating each field as an independent snapshot. This leaves open whether generated sequences are temporally coherent or merely independently plausible, a gap that matters increasingly for sub-daily resolutions since downstream impacts depend on accumulation and persistence, not just on isolated states. This study asks whether autoregressive conditioning can deliver coherent rollouts without error accumulation and ensemble-spread collapse, and whether a single model can expose the resulting coherence-diversity trade-off as a controllable inference choice rather than a fixed design commitment. We introduce a noise-conditioned teacher-forcing training scheme conditioning a diffusion U-Net downscaling model on a noised previous high-resolution state, so the trained model interpolates continuously between autoregressive (AR) and per-timestep independent (PTI) sampling via one noise parameter. We apply the scheme to an hourly, European-domain historical simulation downscaling AWI-CM1 (1$^\circ$) with ICON (0.11$^\circ$, regridded to 0.25$^\circ$) for precipitation and 2-meter temperature. The noise parameter traces a smooth interpolation between ICON's autocorrelation and PTI generation. AR rollouts of up to six months show no progressive degradation, with ensemble spread contracting only slightly within the first 24 hours. In contrast, the same architecture trained on noise-free previous states diverges. For precipitation PTI underestimates maxima by up to 2x in daily aggregations, while AR tracks ICON's maxima closely. For temperature, differences between modes are negligible. These results show that in diffusion downscaling temporal consistency can be achieved architecture-agnostically and offered as a controllable, not fixed, inference choice.</span> <span class="abstract-toggle" data-id="2611.27641">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.27641v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.27641v1) · [:material-content-copy: BibTeX](bibtex/2611.27641.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a> <a class="md-tag" href="/explore/?t=quarter-degree" data-tag="quarter-degree">0.25°</a> <a class="md-tag" href="/explore/?t=hourly" data-tag="hourly">Hourly</a> <a class="md-tag" href="/explore/?t=daily" data-tag="daily">Daily</a>
    { .paper-tags }

-   #### Adapting AI Weather Foundation Models to a Changing Climate { #2611.27605 }

    *Xinlei Xiong, Wenbo Hu, Shuxun Zhou, Lingxi Xie, Richang Hong, Qi Tian* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.27605">AI weather foundation models learn from historical archives, while the climate distribution encountered during deployment continues to evolve. This growing model--climate mismatch motivates continual...</span><span class="abstract-full" id="full-2611.27605" hidden>AI weather foundation models learn from historical archives, while the climate distribution encountered during deployment continues to evolve. This growing model--climate mismatch motivates continual adaptation without repeated full-model retraining, but the structure of weather forecast errors determines how such adaptation should be performed. We develop an online adaptation framework that exploits two complementary error structures: geographically persistent biases and flow-dependent errors that vary with the atmospheric state. Analysis of Aurora-1.3B reveals strongly concentrated forecast-loss sensitivity in its decoder, motivating flow-dependent corrections within fixed low-dimensional sensitivity subspaces. A residual memory tracks persistent errors remaining after this correction, while a variable-wise online gate controls the contribution of the persistent component. We evaluate Aurora-1.3B using ERA5 from 1959--1968 for offline initialization and a strictly chronological 1969--2022 stream for online evaluation, with updates restricted to previously available verification targets. For 2-metre temperature, distributional departure from the historical reference, measured directly from ERA5, is associated with larger frozen-model errors and greater adaptation gains, while the adapted model maintains more stable skill across evaluation periods. Across 69 output fields and five lead times from 6 to 72 hours, our method reduces MSE and MAE by 6.90% and 6.45%, respectively, relative to $δ$-Adapter, the strongest of six online adaptation baselines.</span> <span class="abstract-toggle" data-id="2611.27605">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.27605v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.27605v1) · [:material-content-copy: BibTeX](bibtex/2611.27605.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=foundation-models" data-tag="foundation-models">Foundation models</a>
    { .paper-tags }

-   #### Democratizing Cross-Project Knowledge for Transforming Water Governance and Climate Resilience: an Agentic AI Ecosystem { #2611.26398 }

    *Oriol Alàs, Ian Palacín, Iván Cester, Edgar Rubión* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.26398">EU-funded projects produce extensive evidence in the form of deliverables, reports, and case studies. However, this knowledge is typically scattered across projects and document structures, making it...</span><span class="abstract-full" id="full-2611.26398" hidden>EU-funded projects produce extensive evidence in the form of deliverables, reports, and case studies. However, this knowledge is typically scattered across projects and document structures, making it difficult to locate relevant, contextual information when making decisions. This paper introduces an Agentic AI Ecosystem that consolidates heterogeneous project documentation into a domain-aware knowledge base and provides adaptive access through an agentic Retrieval-Augmented Generation pipeline. Documents from three EU-funded projects (IMPETUS, NATALIE, and InnWater) are reconstructed into hierarchical section trees and indexed with structured metadata. Retrieval is supported by two domain-specific embedding models fine-tuned on domain data (water and climate). On top of this knowledge base, a ReAct-style controller iteratively orchestrates retrieval actions by allocating retrieval budget across domain collections and accumulating evidence until a stopping criterion is met. In a retrieval benchmark over the corpus, the proposed approach improves Mean Reciprocal Rank by 26.7% (relative) compared to a single-domain embedding baseline, indicating more consistent retrieval of relevant evidence to support cross-project knowledge synthesis and climate-resilient decision-making.</span> <span class="abstract-toggle" data-id="2611.26398">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.26398v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.26398v1) · [:material-content-copy: BibTeX](bibtex/2611.26398.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=llms-agents" data-tag="llms-agents">LLMs & agents</a>
    { .paper-tags }

-   #### Interpretable Prototype-Conditioned Adaptation for Weather Forecasting { #2611.25848 }

    *Zhixuan Li, Hao Jiang, Chen Hui, Kai Geng, Weisi Lin* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.25848">Data-driven weather forecasting requires distinguishing local states and their temporal evolution. However, learned patterns remain implicit in neural forecasters, making their influence difficult to...</span><span class="abstract-full" id="full-2611.25848" hidden>Data-driven weather forecasting requires distinguishing local states and their temporal evolution. However, learned patterns remain implicit in neural forecasters, making their influence difficult to interpret. Failing to distinguish local evolution can lead to similar forecasts for locations that develop differently, increasing prediction error. This motivates explicit references for distinguishing and interpreting local patterns. Prototypes provide explicit pattern references, but sample- or task-level conditioning leaves open how to adapt to distinct locations and variable groups. To deal with this problem, we design ProtoCast, which uses group-specific prototype banks to adapt local features while accounting for cross-variable dependencies. Specifically, (1) we combine each group's historical features and temporal change with forecast context from cross-variable attention for evolution-aware prototype matching. (2) Sparse mixtures of prototype-specific low-rank operators tailor corrections to each group and location. A gated attention block then exchanges the corrected features across groups before prediction. Each group's correction decomposes into explicit prototype contributions before this fusion. (3) We jointly train the forecasting network and operators from scratch and refit each group's prototypes before every training epoch. Slot tracking preserves prototype-operator associations, making each prototype's contribution to local corrections more interpretable. Experiments on GlobalForecast and TyphoonTracking show improved forecasting accuracy, while prototype deletion links local feature contributions to changes in prediction error. This work advances the development of more accurate and interpretable data-driven weather forecasting models. Project page: https://zhixuanli.github.io/projects/li2026protocast</span> <span class="abstract-toggle" data-id="2611.25848">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.25848v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.25848v1) · [:material-content-copy: BibTeX](bibtex/2611.25848.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a> <a class="md-tag" href="/explore/?t=interpretability" data-tag="interpretability">Interpretability</a>
    { .paper-tags }

-   #### Reinforcement-Aligned Probabilistic Weather Forecasting with Discrete Tokens { #2611.25570 }

    *Jinhan Li, Zhisheng Chen, Yuxuan Li, Yuan Gao, Fan Xu, Hao Wu, Jinlong Du, Kun Wang* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.25570">Global medium-range weather forecasting must not only generate ensembles efficiently to characterize uncertainty in future states, but also serve downstream applications such as extreme-event...</span><span class="abstract-full" id="full-2611.25570" hidden>Global medium-range weather forecasting must not only generate ensembles efficiently to characterize uncertainty in future states, but also serve downstream applications such as extreme-event warnings and cross-sphere coupled simulations. We introduce WeatherCraft, a framework that unifies probabilistic generation and reinforcement learning alignment through discrete weather tokens, bringing the GPT-style pretraining--post-training paradigm to global medium-range weather forecasting. A multiscale tokenizer represents atmospheric states as a coarse-to-fine token hierarchy, while a Transformer models causal dependencies across time and scale and predicts tokens within each scale in parallel. Direct sampling generates diverse ensembles, and exact token log-probabilities enable Group Relative Policy Optimization (GRPO) to maximize downstream task rewards without differentiating through evaluators or coupled models. On the ERA5 benchmark, WeatherCraft's eight-member ensemble mean reduces day-15 normalized RMSE by 17.6% relative to the strongest evaluated deterministic baseline and, with KV caching, achieves an approximately $18\times$ inference speedup over GenCast. Direct fair-CRPS adaptation improves ensemble probabilistic skill, while reward-based post-training improves forecasts of persistent land temperature extremes and coupled ocean heat extremes. This work establishes discrete autoregressive modeling as a general foundation for scalable, task-alignable Earth system foundation models.</span> <span class="abstract-toggle" data-id="2611.25570">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.25570v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.25570v1) · [:fontawesome-brands-github: Code](https://github.com/Jinhan-Li/WeatherCraft) · [:material-content-copy: BibTeX](bibtex/2611.25570.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=transformers" data-tag="transformers">Transformers</a> <a class="md-tag" href="/explore/?t=foundation-models" data-tag="foundation-models">Foundation models</a> <a class="md-tag" href="/explore/?t=reinforcement-learning" data-tag="reinforcement-learning">Reinforcement learning</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a>
    { .paper-tags }

-   #### Auditing Learned Dynamics in Weather Models { #2611.25119 }

    *Skylar Gu, Christina Last* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.25119">Learned weather models are experimentally accessible computational systems whose internal representations can be inspected and manipulated. Using sparse autoencoder features from GraphCast,...</span><span class="abstract-full" id="full-2611.25119" hidden>Learned weather models are experimentally accessible computational systems whose internal representations can be inspected and manipulated. Using sparse autoencoder features from GraphCast, DeepMind's autoregressive weather forecasting model, we test whether causal relevance requires physical semantics, a question that matters for auditing whether AI forecasters used in high-assurance settings rely on reliable mechanisms. We also look at whether physically associated latent directions can be steered to improve extreme-event forecasts, where interventions are applied to the model's internal activations during the rollout. We find that ablating a feature locked to GraphCast's computational mesh increases global geopotential height error despite lacking conventional physical semantics. Second, suppressing convection- and low-level-vorticity-associated features weakens predicted tropical-cyclone deepening (the fall in central pressure during intensification), while amplifying them reduces intensity error, with amplification of convection features through feature scaling achieving a median of 51% error reduction over seven storms. Together, these results separate interpretable, physical association from causal utility and establish sparse features as experimental coordinates for auditing and intervening on learned weather dynamics.</span> <span class="abstract-toggle" data-id="2611.25119">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.25119v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.25119v1) · [:material-content-copy: BibTeX](bibtex/2611.25119.bib){ .bibtex-link }
    { .paper-links }

-   #### Evaluation of machine-learning and deep-learning SPEI drought forecasts against persistence across timescales and lead times in Bangladesh { #2611.25075 }

    *Mahir Ahnaf, Showmitra Kumar Sarkar* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.25075">Bangladesh, home to nearly 178 million people, is highly vulnerable to recurring droughts that threaten food security and rural livelihoods. Machine-learning and deep-learning models are increasingly...</span><span class="abstract-full" id="full-2611.25075" hidden>Bangladesh, home to nearly 178 million people, is highly vulnerable to recurring droughts that threaten food security and rural livelihoods. Machine-learning and deep-learning models are increasingly used to forecast drought indices such as the Standardized Precipitation Evapotranspiration Index (SPEI). Most studies report accuracy without a persistence reference, although accumulated indices are strongly autocorrelated, and seldom test drought detection, longer leads or variation between training runs. This study forecast SPEI at 3-, 6- and 12-month timescales one to three months ahead for the eight divisions of Bangladesh, using ERA5 reanalysis checked against 35 Bangladesh Meteorological Department stations. Five models, XGBoost, CNN, Bi-LSTM, CNN-BiLSTM and CEEMDAN-BiLSTM, were trained with five seeds each, scored on accuracy, skill against persistence and drought detection, and interpreted with Shapley values. At a one-month lead, the CNN-BiLSTM and Bi-LSTM beat persistence clearly at SPEI-3 and SPEI-6 (skill 0.408 and 0.350), while at SPEI-12 a Nash-Sutcliffe efficiency of 0.872 barely exceeded persistence (0.850). Skill and detection could disagree: at SPEI-3 and a three-month lead, XGBoost had the highest skill (0.439) yet detected none of the 119 drought months. Only at SPEI-6 did the forecasts beat persistence significantly and match its detection at all three leads, with fewer false alarms. At SPEI-12, gains were small or absent and varied between seeds. The forecasts kept 76-80% of their correlation against station-derived SPEI and relied mainly on precipitation and lagged SPEI. Forecasts should therefore be judged on skill against persistence and detection across several seeds, not on accuracy alone. In Bangladesh, SPEI-6 forecasts can support early warning up to three months ahead and SPEI-3 forecasts one month ahead, while SPEI-12 is better suited to monitoring.</span> <span class="abstract-toggle" data-id="2611.25075">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.25075v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.25075v1) · [:fontawesome-brands-github: Code](https://github.com/mahirahnaf1727/bangladesh-spei-drought-forecasting) · [:material-content-copy: BibTeX](bibtex/2611.25075.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a> <a class="md-tag" href="/explore/?t=recurrent-networks" data-tag="recurrent-networks">Recurrent networks</a> <a class="md-tag" href="/explore/?t=classical-ml" data-tag="classical-ml">Classical ML</a> <a class="md-tag" href="/explore/?t=extremes" data-tag="extremes">Extremes</a> <a class="md-tag" href="/explore/?t=evaluation" data-tag="evaluation">Evaluation</a>
    { .paper-tags }

-   #### Distribution-Free Uncertainty Quantification for Weather Forecasting: Conformal Prediction Under Spatiotemporal Dependence { #2611.24329 }

    * Mariya* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.24329">Conformal prediction gives finite-sample, distribution-free marginal coverage and has begun to be applied to gridded weather forecasts. Marginal coverage concerns the long-run fraction of covered...</span><span class="abstract-full" id="full-2611.24329" hidden>Conformal prediction gives finite-sample, distribution-free marginal coverage and has begun to be applied to gridded weather forecasts. Marginal coverage concerns the long-run fraction of covered points at a location, not how coverage behaves jointly across space and time on a given day. I benchmark eight interval-construction procedures (Gaussian, pooled and per-cell split conformal, per-season, adaptive conformal inference and rolling windows) for gridded 2 m temperature forecasts at eight lead times from 6 hours to 10 days, using ERA5 at 5.625 degree resolution, a deliberately modest convolutional forecaster and ECMWF's operational HRES, with calibration in 2017-2018 and testing in 2019-2021. Alongside coverage, width and interval score, I report dependence diagnostics: a spatial overdispersion index, quantiles of daily spatial coverage and miss persistence relative to a shuffled-days null. Per-cell methods reach near-nominal marginal coverage, yet the day-to-day variance of the fraction of cells covered is roughly 3 to 10 times what independent cells would give, and misses cluster in time; both effects grow with lead time for the CNN, also appear for HRES, and are shared by all methods. An oracle that divides each day's normalised errors by that day's mean removes 81 to 92 percent of the excess dispersion for the CNN and 93 to 98 percent for HRES, so days of large errors across much of the grid account for most of it. A day-scale predicted from seasonal terms and already-verified errors lowers the overdispersion index without widening intervals but improves interval score by at most 0.6 percent; on HRES from 12 hours to 3 days it has the lowest overdispersion index while ACI or season-conditional calibration has the best interval score, so no method is best on both. The study covers one variable, one coarse grid, a reanalysis target and two forecasters.</span> <span class="abstract-toggle" data-id="2611.24329">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.24329v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.24329v1) · [:fontawesome-brands-github: Code](https://github.com/Mariyaaleem/conformal-wx) · [:material-content-copy: BibTeX](bibtex/2611.24329.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=coarse" data-tag="coarse">Coarse (≥1°)</a> <a class="md-tag" href="/explore/?t=daily" data-tag="daily">Daily</a>
    { .paper-tags }

-   #### Enabling joint chemical and meteorological data assimilation with an atmospheric foundation model { #2611.24020 }

    *Eliot J. Kim, Eric J. Mei, Peter D. Ivatt, Lee T. Murray, Alexander J. Turner* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.24020">The hydroxyl radical (OH) is the main oxidant in the troposphere and accounts for 88% of the methane sink, yet its abundance and trends remain poorly constrained. Satellites observe many of the...</span><span class="abstract-full" id="full-2611.24020" hidden>The hydroxyl radical (OH) is the main oxidant in the troposphere and accounts for 88% of the methane sink, yet its abundance and trends remain poorly constrained. Satellites observe many of the chemical and meteorological drivers of OH, but assimilating both requires a model that represents chemistry and meteorology together. Existing chemistry-climate models are too expensive for joint data assimilation. Here, we present AuroraChem, a fine-tuned version of the Aurora atmospheric foundation model that jointly simulates meteorology and the GEOS-Chem chemical fields needed to constrain OH. A one-year AuroraChem rollout takes 12 minutes on one GPU. Free rollouts remain stable for one year but do not fully capture seasonal cycles. Using idealized data assimilation, we find that chemical observations can constrain the meteorological state and meteorological observations can constrain the chemical state. Inserting only the chemical fields raises the annual mean spatial correlation of 500 hPa temperature from 0.89 to 0.92 and recovers the moisture and circulation pattern of the Indian summer monsoon. Inserting only the meteorological fields raises the 500 hPa OH correlation from 0.89 to 0.93, close to the 0.94 from inserting the chemical fields themselves, and recovers the OH seasonal cycle. Inserting only NO2 columns sampled at TROPOMI pixels gives an OH correlation of 0.93. These relationships likely reflect learned covariances rather than physical coupling, and covariances are what data assimilation uses to let an observation of one variable update the others. AuroraChem therefore provides a new pathway to joint chemical and meteorological data assimilation.</span> <span class="abstract-toggle" data-id="2611.24020">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.24020v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.24020v1) · [:material-content-copy: BibTeX](bibtex/2611.24020.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=foundation-models" data-tag="foundation-models">Foundation models</a> <a class="md-tag" href="/explore/?t=sub-hourly" data-tag="sub-hourly">Sub-hourly</a>
    { .paper-tags }

-   #### Comparing Data-Driven and Numerical Weather Prediction Forecasts of Extreme Weather: A Review { #2611.23863 }

    *Gabriele Messori, Hiu-Wang Lau, Chenyu Dong, Gianmarco Mengaldo, Leonardo Olivetti* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.23863">Data-driven weather forecasting models, chiefly trained on reanalysis data, now match or beat physics-based numerical weather prediction (NWP) models on most headline scores, but whether this...</span><span class="abstract-full" id="full-2611.23863" hidden>Data-driven weather forecasting models, chiefly trained on reanalysis data, now match or beat physics-based numerical weather prediction (NWP) models on most headline scores, but whether this advantage extends to extreme weather is contested. We review peer-reviewed studies that directly compare data-driven forecasts with NWP for extremes, from nowcasting to subseasonal lead times. We discuss grey literature in the outlook but do not include it in the formal review. There is good agreement on the fact that data-driven models mostly outperform NWP on temperature extremes, tropical cyclone tracks and the location of synoptic features across time scales. The evidence is mixed for extreme precipitation and feature-based extreme event detection such as extremes related to atmospheric rivers, while it leans toward better NWP performance for estimating event intensity. For record-breaking events that lie outside the training distribution, the evidence is narrower. Nonetheless, both a systematic global evaluation and several case studies find that physics-based models retain the advantage, with data-driven models underestimating both the frequency and the magnitude of records. Hybrid systems that post-process NWP with machine-learning (ML), or couple data-driven forecasts with physics-based models, deliver some of the largest and most consistent gains. We argue that headline scores, verification choices and case selection explain much of the apparent disagreement in the literature, and provide recommendations for a more trustworthy and interpretable evaluation of data-driven models for extreme and record-breaking weather.</span> <span class="abstract-toggle" data-id="2611.23863">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.23863v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.23863v1) · [:material-content-copy: BibTeX](bibtex/2611.23863.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=extremes" data-tag="extremes">Extremes</a> <a class="md-tag" href="/explore/?t=subseasonal-to-seasonal" data-tag="subseasonal-to-seasonal">Subseasonal to seasonal</a> <a class="md-tag" href="/explore/?t=evaluation" data-tag="evaluation">Evaluation</a>
    { .paper-tags }

-   #### Improving Eddy Detection from Wide-Swath Altimetry with Multimodal Data-Driven Models { #2611.22601 }

    *Matteo Broccoli, Andrea Cipollone, Antonio Bonaduce, Elisa Carli, Florian Le Guillou* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.22601">Physically-based eddy detection algorithms usually rely on hard-coded thresholds and constraints that make the detected structures sensitive to small variations in the underlying fields. One way to...</span><span class="abstract-full" id="full-2611.22601" hidden>Physically-based eddy detection algorithms usually rely on hard-coded thresholds and constraints that make the detected structures sensitive to small variations in the underlying fields. One way to relax such conditions is the use of machine-learning methodologies that transform the full problem into an optimization process. We present a multimodal, multi-task, data-driven eddy detection framework based on a shallow convolutional U-Net, trained to jointly segment eddy structures and reconstruct the co-located spatial sea surface temperature (SST) anomaly using a dynamically-based reconstruction as target. Two model variants are developed: NADIRnet, trained on conventional nadir altimetry, and SWOTnet, trained through a curriculum that incorporates SWOT-enriched maps. Both models achieve high pixel- and structure-wise detection scores against the ground-truth, with most residual disagreement attributable to boundary placement rather than missed or spurious detections. The coupled SST regression task, enforced through boundary-coupling and gradient-matching loss terms, is shown to sharpen and regularise the detected eddy boundaries. Compared to the dynamically-based reference, both models recover a substantially larger and finer eddy population; notably, SWOTnet's higher-resolution training signal transfers to conventional nadir altimetry thanks to the higher-resolution SST channel, detecting more eddies than a nadir-trained model on the same data. The framework further generalises, even without retraining, to altimetry products and processing levels unseen during training, including SWOT L3 wide-swath retrievals and an independent gridded product. Together with its low computational cost, this framework is thus promising for near-real-time operational deployment and further downstream applications.</span> <span class="abstract-toggle" data-id="2611.22601">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.22601v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.22601v1) · [:fontawesome-brands-github: Code](https://github.com/MatteoBroccoli/WATER-ML-Eddy-Detection) · [:material-content-copy: BibTeX](bibtex/2611.22601.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a>
    { .paper-tags }

-   #### DARWIN: a Geospatial Foundation Model for Marine Intelligence { #2611.22496 }

    *Coen van den Elsen, Laurens Geffert* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.22496">We present DARWIN, a geospatial foundation model for marine intelligence. Recent Earth-observation foundation models have concentrated on terrestrial domains. The marine realm poses distinct...</span><span class="abstract-full" id="full-2611.22496" hidden>We present DARWIN, a geospatial foundation model for marine intelligence. Recent Earth-observation foundation models have concentrated on terrestrial domains. The marine realm poses distinct challenges: missingness, multi-modal physical and biogeochemical variables, sparse labels, and dynamics that are transient, multi-scale, and non stationary over time. DARWIN adapts a masked-autoencoder backbone to these properties through missingness-aware encoding, expanded spatial and temporal masking, and Fourier-style positional embeddings, and is pretrained on 26 years of global Copernicus reanalysis together with GEBCO bathymetry.   On predicting chlorophyll-a, the pretrained backbone attains the lowest monthly relative error in every month of 2022, outperforming both no-encoder baselines and an identically architected random-init encoder. At 10% of the supervised training data, DARWIN already surpasses the random-init model trained on 100% of the data. Embedding analysis and qualitative reconstructions show that DARWIN captures coherent oceanographic structure rather than memorising input statistics, providing a scalable backbone for downstream marine applications such as anomaly detection, species distribution modelling, and environmental monitoring.</span> <span class="abstract-toggle" data-id="2611.22496">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.22496v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.22496v1) · [:material-content-copy: BibTeX](bibtex/2611.22496.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=foundation-models" data-tag="foundation-models">Foundation models</a> <a class="md-tag" href="/explore/?t=monthly" data-tag="monthly">Monthly</a>
    { .paper-tags }

-   #### SeasonRainBench: Global Seasonal Precipitation { #2611.22281 }

    *Qiuyan Sun, Mengxuan Chen, Haoyuan Liang, Guowen Li, Zhen Huang, Jianhao Lin, Juepeng Zheng et al.* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.22281">Accurate seasonal precipitation forecasts help societies prepare for drought, flooding, and crop failure months in advance. While data-driven methods have made rapid progress in weather and...</span><span class="abstract-full" id="full-2611.22281" hidden>Accurate seasonal precipitation forecasts help societies prepare for drought, flooding, and crop failure months in advance. While data-driven methods have made rapid progress in weather and subseasonal forecasting, operational seasonal precipitation prediction still relies on numerical ensemble systems. These systems are computationally expensive and exhibit biases and drift that increase with lead time, motivating more efficient data-driven alternatives. Yet this transition faces three linked challenges: comparing data-driven forecasts with operational ones requires (1) a global benchmark that does not exist, scoring them requires (2) metrics that capture operational priorities beyond anomaly correlation (ACC) and RMSE, and training them must (3) rely on a single, short observational record. To address the first two challenges, we introduce \BenchName, the first such benchmark, covering monthly precipitation forecasts 1--5 months ahead. It pairs global $1^\circ$ satellite precipitation with oceanic and atmospheric predictors, includes seven operational centers, and provides 310,944 simulated months from 52 climate models for training. Beyond standard scores, we introduce two operationally motivated metrics for anomaly strength and monthly rain-belt position. To address the third challenge, we develop \MODEL, which learns from simulations what the record lacks and transfers it to Earth through three modules: SimForced aligns the climate background, SimMemory transfers a history encoder pretrained on simulations, and SimAnalog supplements Earth labels with simulated analog outcomes. Extensive experiments show that \MODEL{} improves ACC, CRPS, anomaly strength, and rain-belt position by 12.0%, 3.6%, 2.9%, and 0.3% over the strongest operational or data-driven model, opening a path from simulated worlds to Earth system prediction.</span> <span class="abstract-toggle" data-id="2611.22281">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.22281v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.22281v1) · [:material-content-copy: BibTeX](bibtex/2611.22281.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=foundation-models" data-tag="foundation-models">Foundation models</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=extremes" data-tag="extremes">Extremes</a> <a class="md-tag" href="/explore/?t=subseasonal-to-seasonal" data-tag="subseasonal-to-seasonal">Subseasonal to seasonal</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=monthly" data-tag="monthly">Monthly</a>
    { .paper-tags }

-   #### Learning Calibrated and Samplable Predictive Distributions of Surface-Ocean $Δp$CO$_2$ Using Tabular Diffusion Models { #2611.21769 }

    *Gabriele Accarino, Mohammad M. Amirian, Thea H. Heimdal, Amanda R. Fay, Galen A. McKinley et al.* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.21769">The ocean absorbs almost 30% of anthropogenic CO$_2$ emissions, but quantifying this sink requires estimating surface-ocean CO$_2$ from observations that are sparse in space and time. Most existing...</span><span class="abstract-full" id="full-2611.21769" hidden>The ocean absorbs almost 30% of anthropogenic CO$_2$ emissions, but quantifying this sink requires estimating surface-ocean CO$_2$ from observations that are sparse in space and time. Most existing machine-learning estimates of this field are deterministic, giving a single value per grid point, with uncertainty approximated post hoc via residuals, ensembles, or prescribed distributions. We instead learn a conditional predictive distribution for each observation: using Treeffuser, a conditional diffusion model with gradient-boosted-tree score functions, we estimate the full predictive distribution of $Δp$CO$_2$ (the sea-air $p$CO$_2$ difference) from 13 environmental, spatial, and temporal predictors. On a held-out SOCAT test set, Treeffuser attains the lowest continuous ranked probability score among the uncertainty-quantification methods evaluated (8.229 $μ$atm) and the coverage closest to the nominal 68% / 95% levels (0.686 / 0.930), while retaining point-prediction skill comparable to a deterministic XGBoost baseline (RMSE 18.4 vs. 17.7 $μ$atm; $R^2$ 0.77 vs. 0.79). These results show that conditional diffusion can provide well-calibrated, observation-level uncertainties while largely preserving point-prediction skill. Because Treeffuser is generative, its samples can support probabilistic analyses of $Δp$CO$_2$; extending this to spatially coherent field reconstruction will require modeling dependence across grid cells and evaluation on spatially held-out data.</span> <span class="abstract-toggle" data-id="2611.21769">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.21769v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.21769v1) · [:material-content-copy: BibTeX](bibtex/2611.21769.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=classical-ml" data-tag="classical-ml">Classical ML</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### Quantifying the added value of human forecaster discussions for tropical cyclone intensity prediction { #2611.21602 }

    *Ilia Azizi, Frederick Iat-Hin Tam, Marc-Olivier Boldi, Valérie Chavez-Demoulin, Tom Beucler* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.21602">The National Hurricane Center routinely releases Tropical Cyclone Discussions that offer insights into its tropical cyclone forecasts. This study examines the added value of these discussions for...</span><span class="abstract-full" id="full-2611.21602" hidden>The National Hurricane Center routinely releases Tropical Cyclone Discussions that offer insights into its tropical cyclone forecasts. This study examines the added value of these discussions for intensity forecasting beyond a set of numerical predictors. For 637 Atlantic and eastern North Pacific tropical cyclones from the 2002-2023 hurricane seasons, we analyze the discussion text and its combination with 13 hurricane environmental predictors from SHIPS and ERA5 10-meter wind fields, for forecasting intensity changes at lead times of 24 to 120 hours using three modeling methods. In one approach, an end-to-end deep learning framework for the text shows that the discussions alone offer comparable value for 24-hour intensity forecasts to the SHIPS predictors. Adding discussion information to the SHIPS predictors in an ensemble of five such models reduces the mean absolute error of 24-hour maximum sustained wind forecasts for 2022-2023 cyclones from 9.89 to 8.90 kt, with a confidence interval that excludes zero when whole storms are resampled. The ERA5 wind fields have a minor effect on mean absolute error in wind predictions, and model rankings differ across predicted variables, lead times and model types. We provide prediction intervals for these forecasts, whose coverage falls below the 90% target during rapid intensification. Analysis of the text shows that words related to intensity and intensity change (such as intensify, weaken, eyewall and category) are important for the model predictions. This retrospective study shows the added value of the forecasters' discussions and suggests that multi-modal learning can help understand the predictive contribution of expert human judgment. The reanalysis data differ from real-time analyses, and the discussions are issued at least three hours after the valid time of the other inputs, so these results should not be interpreted operationally.</span> <span class="abstract-toggle" data-id="2611.21602">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.21602v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.21602v1) · [:material-content-copy: BibTeX](bibtex/2611.21602.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a>
    { .paper-tags }

</div>

