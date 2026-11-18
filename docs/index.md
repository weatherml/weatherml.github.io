---
hide:
  - navigation
  - toc
title: weatherml
---

A collection of papers on AI for weather forecasting, climate modelling and atmospheric science.

<p class="page-meta" markdown="span">1663 papers · updated 2026-11-18 · <a href="feed.xml">:material-rss: RSS</a> · <a href="all_papers.bib" download>:material-download: BibTeX</a> · <a href="https://github.com/weatherml/weatherml.github.io/issues/new?template=suggest-paper.yml">:material-plus: Suggest a paper</a></p>

## Browse by Topic

<div class="grid cards topics" markdown>

-   [Global Models](papers/global-models/index.md) <span class="topic-count">393</span>
-   [Regional Models](papers/regional-models/index.md) <span class="topic-count">70</span>
-   [Nowcasting](papers/nowcasting/index.md) <span class="topic-count">110</span>
-   [Downscaling](papers/downscaling/index.md) <span class="topic-count">110</span>
-   [Post-processing](papers/post-processing/index.md) <span class="topic-count">27</span>
-   [Data Assimilation](papers/data-assimilation/index.md) <span class="topic-count">83</span>
-   [Climate Modeling](papers/climate-modeling/index.md) <span class="topic-count">317</span>
-   [Hydrology](papers/hydrology/index.md) <span class="topic-count">40</span>
-   [Ocean & Sea Ice](papers/ocean-sea-ice/index.md) <span class="topic-count">94</span>
-   [Air Quality & Composition](papers/air-quality-composition/index.md) <span class="topic-count">79</span>
-   [Remote Sensing](papers/remote-sensing/index.md) <span class="topic-count">112</span>
-   [Other](papers/other/index.md) <span class="topic-count">228</span>

</div>

## Recent Additions

<div class="grid cards" markdown data-search-exclude>

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

-   #### Run it Back: Autoregression vs. Video Diffusion for Final-Conditioned Weather Trajectories { #2611.21521 }

    *Jerry Lin, Elizabeth A. Barnes* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.21521">Low-likelihood, high-impact extremes are projected to increase in severity and frequency in a warming climate. Sampling the precursors that lead to these events traditionally requires massive...</span><span class="abstract-full" id="full-2611.21521" hidden>Low-likelihood, high-impact extremes are projected to increase in severity and frequency in a warming climate. Sampling the precursors that lead to these events traditionally requires massive ensembles that scale poorly with lead time and rarity, and typical rare-event sampling and ensemble-boosting methods do not show how a single extreme might arise from diverse synoptic precursors. Recently, Lin et al.\ (2026) showed this is possible with final-conditioned 66-hour trajectories (trajectories constrained to end at an observed state) from cBottle-video, a pre-trained, off-the-shelf video diffusion climate emulator not explicitly fine-tuned for this task (Brenowitz et al., 2025). However, its ensembles were underdispersed and geostrophically inconsistent, and it reproduces conditioned frames only approximately, hindering autoregressive rollout. Here, we train Rewind-AR, a backward-in-time probabilistic autoregressive model with a FourCastNet~3 backbone (Bonev et al., 2025), and generate final-conditioned ensembles for the same three cases: the 2021 Pacific Northwest heatwave, Hurricane Ian, and Superstorm Sandy (in the Supplementary Information). The leading modes of Rewind-AR's antecedent 500 hPa geopotential height explain far more of the across-member variance than cBottle-video's for both hurricane position and heatwave surface temperature, and Rewind-AR better reproduces the day-over-day heatwave intensification observed in ERA5. Finally, geostrophic inconsistencies close to the conditioning frame seen in cBottle-video are not reproduced by Rewind-AR. Nevertheless, up to 22% of Rewind-AR members become numerically unstable within 66 hours, so improving rollout physical consistency and stability over long lead times is a natural next step.</span> <span class="abstract-toggle" data-id="2611.21521">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.21521v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.21521v1) · [:material-content-copy: BibTeX](bibtex/2611.21521.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### Improving Tropical Cyclone Forecasts through Objective Ensemble Track-Error Reduction { #2611.20251 }

    *Shuochen Wang, Mark S. Allen, Auroop R. Ganguly* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.20251">Accurate tropical cyclone (TC) track guidance is essential for warnings, evacuation planning, and risk communication, but outlying members and distinct track scenarios can degrade raw ensemble...</span><span class="abstract-full" id="full-2611.20251" hidden>Accurate tropical cyclone (TC) track guidance is essential for warnings, evacuation planning, and risk communication, but outlying members and distinct track scenarios can degrade raw ensemble forecasts. Statistical and machine-learning (ML) methods proposed to improve ensemble guidance are difficult to compare because previous studies often use different forecast systems, basins, periods, and verification methods. Using a common verification framework, this study compares six objective methods (four statistical trimming and two ML methods) with the full ensemble mean for 138 TCs forecast by GEFS, MOGREPS-G, and IFS during 2016-2024. Dynamic trimming (DT) is the most consistent statistical method, reducing mean total track error by 8.0%, 13.4%, and 16.1% at short, medium, and long ranges. Its largest benefits occur in high-spread cases, where it can reach 70% of the low-error bound. Short-lead-error trimming (SLET) provides useful delayed guidance after early observations become available, but its improvement decreases with lead time. Regression-mixture clustering (RC) shows potential in low-spread cases and provides the largest improvement over the control run. The gradient boosted tree (GBT) produces the largest medium-range improvement and exceeds the theoretical trimming bound, suggesting strong potential, particularly when the observed track is poorly represented by the ensemble family. However, its 12.8% short-range degradation and gains concentrated in a small subset of cases indicate overcorrection when the ensemble mean is already accurate, highlighting the need for additional constraints. These results provide a consistent basis for using objective error-reduction methods to strengthen TC track guidance for operational decision-making.</span> <span class="abstract-toggle" data-id="2611.20251">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.20251v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.20251v1) · [:material-content-copy: BibTeX](bibtex/2611.20251.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=classical-ml" data-tag="classical-ml">Classical ML</a> <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### A plug-in energy-balance freezing module for transformer frost nowcasting without leaf-sensor inputs { #2611.20114 }

    *Juyeon Kim, Chaeyeon Yi, Yun Am Seo* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.20114">Frost protection requires local warnings where leaf-sensor inputs are unavailable. In the frost literature we surveyed, physics entered machine-learning models through inputs, numerical-model...</span><span class="abstract-full" id="full-2611.20114" hidden>Frost protection requires local warnings where leaf-sensor inputs are unavailable. In the frost literature we surveyed, physics entered machine-learning models through inputs, numerical-model coupling, or losses, rather than model structure. We develop a transformer-compatible module that embeds leaf energy-balance physics in freezing-probability forecasts: an energy-balance temperature reference and a learned residual predict leaf temperature, which drives a multiplicative gate imposing a temperature-dependent probability cap. We mathematically prove conditional monotonicity and an alarm boundary, and test whether placing the same information in the output pathway or in attention yields different benefits. We evaluated 12 hourly leads at 19 Korean stations over five frost seasons, with and without leaf-sensor inputs. In the 2025-2026 hold-out season without these inputs, our time-token transformer (PET-E) achieved a lead-mean critical success index (CSI) of 0.516 versus 0.478 for fixed-configuration, joint-lead LightGBM, with significant gains at all 12 leads, and the fraction of alarms at observed leaf temperatures of at least 2 degrees C fell from 0.087 to 0.051 relative to the unmodified backbone. Across the hold-out and a retrospective 2024-2025 evaluation, the module reduced Brier scores and warm-surface alarm fractions in both the time-token transformer and iTransformer, with no significant CSI decrease at any lead. At held-out stations without static or leaf-sensor inputs, both module-equipped backbones exceeded LightGBM at all 12 leads (CSI 0.468 and 0.467 versus 0.437). In the time-token backbone, the temperature pathway reduced this alarm fraction in all four settings, whereas the attention prior increased CSI in three. These findings link an explicit temperature-dependent probability cap to improved forecasts and distinguish output-side from attention-based embedding.</span> <span class="abstract-toggle" data-id="2611.20114">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.20114v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.20114v1) · [:material-content-copy: BibTeX](bibtex/2611.20114.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=transformers" data-tag="transformers">Transformers</a> <a class="md-tag" href="/explore/?t=classical-ml" data-tag="classical-ml">Classical ML</a> <a class="md-tag" href="/explore/?t=station-point" data-tag="station-point">Station / point</a> <a class="md-tag" href="/explore/?t=hourly" data-tag="hourly">Hourly</a>
    { .paper-tags }

-   #### Physics-Spawned Neural Integration of Rotating Shallow-Water Dynamics { #2611.20054 }

    *Kaushal Gianchandani, Luka Todorovic, Sai Ravela* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.20054">Equation-derived neural integration can execute prescribed dynamics without training, while informative ensemble learning can estimate uncertain parameters and identify governing interactions without...</span><span class="abstract-full" id="full-2611.20054" hidden>Equation-derived neural integration can execute prescribed dynamics without training, while informative ensemble learning can estimate uncertain parameters and identify governing interactions without requiring differentiability. Together, these approaches motivate Physics-Spawned Informative Networks ($Φ$SpIN), a framework for prescribed and adaptive hybrid dynamical systems. As a first step toward flood modeling applications, we present $Φ$SpIN-SW, its prescribed-physics realization for the rotating shallow-water equations. Compatible C-grid operators on spherical Voronoi meshes define a computational graph that retains non-polynomial operations and advances the state through recurrent neural integration compiled with JAX. Analytical benchmarks characterize numerical errors, and global and coastal experiments closely reproduce reference solutions generated using the Model Prediction Across Scales (MPAS) model. At approximately 14~km global resolution, A100, H200 and B200 GPUs attain 9-, 19- and 34-fold higher timestep throughput than a 96-core MPAS-SW CPU reference under matched workloads. These results extend training-free neural integration beyond polynomial dynamics to complex geophysical simulation, establishing a computational foundation for subsequent adjoint-free ensemble estimation of physical and neural parameters and informative adaptation of operator structure.</span> <span class="abstract-toggle" data-id="2611.20054">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.20054v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.20054v1) · [:material-content-copy: BibTeX](bibtex/2611.20054.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=recurrent-networks" data-tag="recurrent-networks">Recurrent networks</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### GPROF V08: Improving Satellite Precipitation Estimates Across a Heterogeneous Constellation of Passive-Microwave Sensors { #2611.19978 }

    *Simon Pfreundschuh, Paula J. Brown, Christian D. Kummerow* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.19978">Precipitation estimates from the passive microwave observations (PMW) of the Global Precipitation Measurements (GPM) mission serve a wide range of scientific, societal, and economic applications. The...</span><span class="abstract-full" id="full-2611.19978" hidden>Precipitation estimates from the passive microwave observations (PMW) of the Global Precipitation Measurements (GPM) mission serve a wide range of scientific, societal, and economic applications. The Goddard Profiling Algorithm (GPROF) is the operational retrieval used to produce the precipitation estimates for all GPM PMW sensors. This study presents the implementation GPROF V08 -- the most recent version of GPROF and a major update aimed at improving the accuracy of precipitation estimates from the GPM constellation.   GPROF V08 replaces the conventional Bayesian scheme used by earlier GPROF versions by a convolutional neural network (CNN). The first part of this two-part study described a novel ML framework for generating synthetic observations required for training CNN retrievals for all GPM sensors. This second part describes the final, operational implementation, which adds a second finetuning stage to the training.Retrievals from the new GPROF V08 algorithm for all principal GPM sensors are validated against reference precipitation estimates from various sources. Our results show that, while the magnitude of improvements vary by sensor and surface type, they are robust across sensors and metrics and can be as high as 40 % for the linear correlation coefficient for precipitation estimates from SSMIS and ATMS over ocean compared to combined radar and radiometer retrievals from GPM.   While some challenges remain -- such as an increased sensitivity to sensor degradation and the persistence of climatological biases, which remain of the same order as for GPROF V07 -- GPROF V08 constitutes a substantial advancement in spaceborne precipitation monitoring. Since the improvements are robust across all sensor of the constellation, the update is expected to improve the full record of GPM precipitation estimates and thus benefit a wide range of downstream datasets and applications.</span> <span class="abstract-toggle" data-id="2611.19978">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.19978v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.19978v1) · [:material-content-copy: BibTeX](bibtex/2611.19978.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a>
    { .paper-tags }

-   #### Towards GPROF V08: A Machine-Learning Framework for Simulating Passive Microwave Observations of Precipitation { #2611.19966 }

    *Simon Pfreundschuh, Christian D. Kummerow, Paula J. Brown* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.19966">Spatially resolved precipitation retrievals based on convolutional neural networks (CNNs) can improve precipitation estimates from passive microwave (PMW) observations and will be used to implement...</span><span class="abstract-full" id="full-2611.19966" hidden>Spatially resolved precipitation retrievals based on convolutional neural networks (CNNs) can improve precipitation estimates from passive microwave (PMW) observations and will be used to implement the next operational PMW retrieval algorithm for the Global Precipitation Measurement (GPM) mission. However, the conventional radiative transfer simulations used to extend the retrieval algorithm to the various sensors of the GPM constellation do not provide the spatial coverage required to train CNN-based retrievals.   To address this, we introduce a machine-learning-based simulation framework that enables the training of the CNN-based GPROF-NN 3D retrieval algorithm for the full GPM constellation. At the core of this framework is Satformer, a new neural-network architecture that can translate PMW observations between any pair of GPM sensors. The resulting synthetic observations exhibit lower errors than conventional radiative transfer simulations and lead to more accurate precipitation retrievals when used to train GPROF-NN precipitation retrievals. We further demonstrate the applicability of the framework to new instruments by developing GPROF-NN retrievals for the Microwave Imager aboard the Weather System Follow-on Microwave satellite. While GPROF-NN retrievals based on the new simulation framework achieve lower instantaneous retrieval errors and higher correlations compared to independent reference data than conventional retrievals, residual simulation errors can introduce systematic biases in the resulting retrievals.   The proposed simulation framework provides a key component for extending GPROF-NN to the full GPM constellation as part of the upcoming operational GPROF V08 retrieval algorithm. The results demonstrate the potential of machine-learning-based simulations to support spatially resolved precipitation retrievals across heterogeneous satellite constellations.</span> <span class="abstract-toggle" data-id="2611.19966">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.19966v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.19966v1) · [:material-content-copy: BibTeX](bibtex/2611.19966.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a>
    { .paper-tags }

-   #### Spatiotemporal Reconstruction of Auroral Precipitations with Neural Fields { #2611.19757 }

    *Nazar Misyats, Mizuki Fukizawa, Yoshimasa Tanaka, Satoshi Ikehata* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.19757">Auroras provide remote optical observations of energetic electrons precipitating into the upper atmosphere, and whose energy distribution encodes key information about large scale magnetospheric...</span><span class="abstract-full" id="full-2611.19757" hidden>Auroras provide remote optical observations of energetic electrons precipitating into the upper atmosphere, and whose energy distribution encodes key information about large scale magnetospheric processes. Recovering the underlying electron precipitation field from ground-based camera images of auroras remains a challenging inverse problem. Existing auroral tomography techniques are often limited to simplified static settings, discrete spatial and energy grids, or strong assumptions on the electron energy spectrum. We propose to model the dynamic auroral electron precipitation field via a temporally conditioned neural network. The state of the precipitation field at each frame is represented by a per-frame latent code driven by learned discrete dynamics. A shared decoder maps horizontal positions and latent states to electron energy spectra. The predicted flux is converted to multi-wavelength volume emission rates and rendered into camera images accounting for delayed emissions. We evaluate the method on synthetic dynamic auroral sequences and real multi-view auroral videos, showing that the proposed representation produces temporally coherent precipitation reconstructions and image projections consistent with the observations.</span> <span class="abstract-toggle" data-id="2611.19757">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.19757v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.19757v1) · [:material-content-copy: BibTeX](bibtex/2611.19757.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a>
    { .paper-tags }

-   #### TyphoonWorld: A Residual Token World Model for Probabilistic Forecasting of Typhoon Imagery { #2611.18740 }

    *Xiaoyu Chen, Tianyi Li, Xuanyi Zhang* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.18740">A satellite forecast of a typhoon should give a low-error picture of the next two days, together with an ensemble whose members are coherent fields and whose spread responds to the storm at hand. A...</span><span class="abstract-full" id="full-2611.18740" hidden>A satellite forecast of a typhoon should give a low-error picture of the next two days, together with an ensemble whose members are coherent fields and whose spread responds to the storm at hand. A small direct regression network already gives the lowest squared error on held-out storms, so a generative model has to earn its cost through the part of the future that the regression smooths away. We present TyphoonWorld, a discrete-token world model that generates only that part. Conditioned on tokens of the regression forecast, the latest frames and the best-track state at forecast time, it samples residual fields lead by lead, and members add these residuals to the regression center. We compare it under one protocol with per-pixel Gaussian networks, a diffusion model of the same residuals, residuals from other storms and an unconditional token world model, on 276 forecast windows from 46 held-out storms of Digital Typhoon. TyphoonWorld matches the squared error of the regression, lowers the CRPS of the area colder than 220 K, a direct measure of deep convection, by 14 to 39% against every other ensemble, and ties its spread to forecast error as closely as the Gaussian network, with a correlation of 0.78 across windows. It uses 56% of the parameters of the unconditional world model.</span> <span class="abstract-toggle" data-id="2611.18740">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.18740v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.18740v1) · [:material-content-copy: BibTeX](bibtex/2611.18740.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### WeatherDreamer: Learning a World Model from Multimodal Earth Observations { #2611.18393 }

    *Kate Duffy, Elise Penn, Tristan Ballard, August Posch, Thomas J. Vandal* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.18393">Machine learning weather models now rival operational global forecasts but still start from analyses produced by separate, expensive data assimilation (DA) systems, which for global models are...</span><span class="abstract-full" id="full-2611.18393" hidden>Machine learning weather models now rival operational global forecasts but still start from analyses produced by separate, expensive data assimilation (DA) systems, which for global models are typically updated every six hours. Observation-to-observation models avoid this dependence, but most omit the recursive filtering cycle that lets DA systems carry a background state through sparse, irregular observations. We introduce WeatherDreamer, a world model that learns this cycle end-to-end from 18 global observing systems, without reanalysis. It infers a latent state from observations, carries it forward hour by hour with learned dynamics, corrects it each hour with new observations, and can predict what any of its observing systems would measure at any location. At held-out stations, its ensembles beat NOAA's Global Ensemble Forecast System (GEFS) for temperature, dew point and wind out to 12 h and are better calibrated than GEFS and the ECMWF ensemble. Precipitation, cloud-top height and solar irradiance, never inputs at inference, are inferred with skill matching or exceeding operational products at short leads. Against the first guess of NOAA's Gridpoint Statistical Interpolation (GSI) DA system, WeatherDreamer's short-range departures are within 10% of GSI's at surface stations and within 25% for moisture-sensitive microwave channels, but up to five times larger for temperature-sounding channels. Never given observation errors, its learned update assumes errors close to known instrument and retrieval errors of microwave sounders and scatterometer winds. WeatherDreamer also separates viewing geometry from atmospheric state, recovering 83-96% of observed limb darkening and producing seamless nadir-view imagery. Producing an hourly global analysis in about 1.5 minutes on one GPU, WeatherDreamer shows that assimilation and forecasting can be learned together directly from observations.</span> <span class="abstract-toggle" data-id="2611.18393">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.18393v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.18393v1) · [:material-content-copy: BibTeX](bibtex/2611.18393.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=energy" data-tag="energy">Energy</a> <a class="md-tag" href="/explore/?t=sub-hourly" data-tag="sub-hourly">Sub-hourly</a> <a class="md-tag" href="/explore/?t=hourly" data-tag="hourly">Hourly</a>
    { .paper-tags }

-   #### Tracing Uncertainties of Extreme El Niño Prediction to Mesoscale Ocean Currents { #2611.17854 }

    *Bo Qin, Zeyun Yang, Qiaofei Xue, Yanan Ding, Jinyu Wang, Chaopeng Ji, Shijin Yuan, Bin Mu et al.* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.17854">The predictability of the El Niño-Southern Oscillation (ENSO) has historically been linked to large-scale errors in models' initial conditions such as heat content and surface wind stress. Here we...</span><span class="abstract-full" id="full-2611.17854" hidden>The predictability of the El Niño-Southern Oscillation (ENSO) has historically been linked to large-scale errors in models' initial conditions such as heat content and surface wind stress. Here we demonstrate that these slowly evolving errors may ultimately originate from mesoscale errors in initial oceanic states that are common in current models and ocean observation systems. This hidden source of ENSO forecast uncertainty, in terms of mesoscale ocean-current errors, is convincingly exposed by the conditional nonlinear optimal perturbation (CNOP) method applied to a differentiable ocean AI model. The CNOP-identified most influential current errors are strongly event-dependent. For the 1997/98 El Niño, the initial current errors in the eastern Pacific results in the largest forecast error through locally suppressing original subsurface warming. For the 2015/16 event, the dominate error originates from the western Pacific, which reorganizes upper ocean temperature structure and affects the eastern Pacific through the equatorial waveguide. Further analysis reveals that these current errors perturb mesoscale eddy count and eddy kinetic energy triggering large-scale temperature errors that hinder El Niño development. The timing of the maximum growth rate of the obtained CNOP error is well aligned with the spring predictability barrier of ENSO, and the error structures are reflected in simulated analysis errors caused by unresolved mesoscale currents. These findings extend the potential source of ENSO predictability beyond large-scale thermal signals, reveal for the first time an uncharted route of ENSO forecast failure, and provide an important physical basis for targeted ocean observations and data assimilation in the future.</span> <span class="abstract-toggle" data-id="2611.17854">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.17854v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.17854v1) · [:material-content-copy: BibTeX](bibtex/2611.17854.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=subseasonal-to-seasonal" data-tag="subseasonal-to-seasonal">Subseasonal to seasonal</a>
    { .paper-tags }

-   #### Activation Steering the Nan-Shan in Typhoon { #2611.17768 }

    *Megan Wei, Damon Pham, John Cayley, Ellie Pavlick* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.17768">Typhoon is an audiovisual installation based on Joseph Conrad's novella, in which Captain MacWhirr steers the steamer Nan-Shan directly into and through a catastrophic storm. The novella is a study...</span><span class="abstract-full" id="full-2611.17768" hidden>Typhoon is an audiovisual installation based on Joseph Conrad's novella, in which Captain MacWhirr steers the steamer Nan-Shan directly into and through a catastrophic storm. The novella is a study of what happens when the scale of a phenomenon exceeds the practical and conceptual frameworks available to comprehend it and of the paralysis that sometimes follows a demand for action in such overwhelming circumstances. We chose the text of Typhoon to reflect on human culture's current AI moment. We prompt a large language model, seeded with Conrad's original language, to feed endlessly on its own output, and explore when and how the text becomes increasingly alien and unwieldy as time progresses, including deviations from the narrative, digressions into the model's own instructions and thinking, and long periods of incoherent rumination. We employ methods from activation steering, including a novel adaptive steering method we introduce, to control the rhythm of the narrative and the model's digressions over long stretches of generation. The simultaneous efficacy and unpredictability of this steering showcases the tension between the perceived interpretability of AI, and the reality of a system that remains outside our comprehension and control.</span> <span class="abstract-toggle" data-id="2611.17768">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.17768v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.17768v1) · [:material-content-copy: BibTeX](bibtex/2611.17768.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=llms-agents" data-tag="llms-agents">LLMs & agents</a> <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a>
    { .paper-tags }

-   #### An AI-enabled high-resolution chlorophyll-a forecasting model that outperforms the operational global biogeochemical system { #2611.17704 }

    *Yong Jiang, Zheng Jiang, Gaowei Zhang, Zengzhou Hao, Suixiang Shi, Chi Zhang, Wei Wang* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.17704">Surface chlorophyll-a concentration is a key indicator of marine phytoplankton variability, and short-term forecasts of its distribution provide advance information for marine ecosystem monitoring...</span><span class="abstract-full" id="full-2611.17704" hidden>Surface chlorophyll-a concentration is a key indicator of marine phytoplankton variability, and short-term forecasts of its distribution provide advance information for marine ecosystem monitoring and management. However, interactions between physical transport and biological processes complicate chlorophyll-a forecasting, and simulating these coupled processes with conventional numerical models imposes substantial computational demands. Here we introduce ChlCast, an AI model that generates daily forecasts of surface chlorophyll-a concentration over a 10-day horizon at approximately 4-km resolution between 60$^\circ$S and 60$^\circ$N. ChlCast combines factorized spatiotemporal convolutions with a multiscale architecture to learn from historical chlorophyll-a fields and climatological information. Compared with the operational 0.25$^\circ$ Copernicus global ocean biogeochemical forecasting system, ChlCast reduces mean absolute error and root mean square error by 70.74% and 47.63%, respectively, against satellite-derived Level-4 chlorophyll-a fields, and by 35.03% and 13.07% against independent BGC-Argo in situ observations. Coastal case studies further show that the 4-km forecasts retain coastal gradients and localized concentration maxima that are weakened by coarse-grid averaging. These results demonstrate the potential of observation-driven deep learning for high-resolution forecasting of surface chlorophyll-a variability at the near-global scale.</span> <span class="abstract-toggle" data-id="2611.17704">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.17704v2) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.17704v2) · [:material-content-copy: BibTeX](bibtex/2611.17704.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a> <a class="md-tag" href="/explore/?t=km-scale" data-tag="km-scale">Km-scale</a> <a class="md-tag" href="/explore/?t=daily" data-tag="daily">Daily</a>
    { .paper-tags }

-   #### Climate Temporal Temperature Prediction via an Adaptive Stacked Bidirectional Residual Mamba Network { #2611.16322 }

    *Yongxiang Lei* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.16322">Accurate temperatures forecasting relies on traditional meteorological parameters that are essential for monitoring weather informatics and guiding forecasting efforts. This study presents Deep...</span><span class="abstract-full" id="full-2611.16322" hidden>Accurate temperatures forecasting relies on traditional meteorological parameters that are essential for monitoring weather informatics and guiding forecasting efforts. This study presents Deep DualMamba, a novel deep learning framework for temporal temperature prediction, built upon an enhanced variant of the recently proposed Mamba network. Unlike conventional Transformer-based architectures, which dominate foundation models but suffer from computational inefficiencies on long sequences, Deep DualMamba leverages selective structured state space models (SSMs) to achieve both scalability and performance. We identify a core limitation of prior subquadratic-time models-namely, their inability to perform long-term and robust forecasting and introduce two key innovations. First, we propose an adaptive bidirectional Mamba as encoders to learn complex and efficient features. Second, we develop a hardware-aware, parallel recurrent algorithm to compensate for the loss of convolutional efficiency, ensuring practical viability at scale. Last, the Deep DualMamba's approximation ability and convergence is theoretical analyzed. These design choices culminate in a simplified, attention- and MLP-free neural architecture that exhibits linear complexity with sequence length and significantly improved inference throughput. To demonstrate its applicability in climate informatics, we implement a full forecasting pipeline using historical meteorological data from UK monitoring stations.</span> <span class="abstract-toggle" data-id="2611.16322">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.16322v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.16322v1) · [:material-content-copy: BibTeX](bibtex/2611.16322.bib){ .bibtex-link }
    { .paper-links }

-   #### HybridCT: ML Accelerated Tomography for Climate Relevant 3D Cloud Observations { #2611.16255 }

    *Linda Forster, Nicholas J. LaHaye, Marcin J. Kurowski, Anna Jungbluth, Bernhard Mayer et al.* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.16255">Low-level clouds are one of the largest sources of uncertainty in climate predictions, partly because we lack detailed three-dimensional (3D) measurements of their internal structure. Here, we...</span><span class="abstract-full" id="full-2611.16255" hidden>Low-level clouds are one of the largest sources of uncertainty in climate predictions, partly because we lack detailed three-dimensional (3D) measurements of their internal structure. Here, we present HybridCT, a cloud tomographic reconstruction scheme accelerated by machine learning (ML) that recovers 3D cloud extinction directly from passive multi-angle observations. An ML emulator maps radiances from multiple viewing angles to cloud optical thickness (COT), producing path-integrated projections, and a Simultaneous Algebraic Reconstruction Technique (SART) explicitly inverts these projections to retrieve 3D extinction. Training observations are generated from synthetic 3D cloud fields using radiative transfer simulations for nine viewing angles between $\pm$70$^{\circ}$. HybridCT reconstructs extinction with RMSE 13.3 km$^{-1}$ in 3.5 s, compared to 22.7 km$^{-1}$ in approx. 30 min for the state-of-the-art physics-based tomography framework AT3D, a speedup exceeding 500$\times$. HybridCT thus provides a scalable route toward large-scale observations of 3D cloud structure relevant to radiative-budget interpretation and evaluation of high-resolution cloud simulations.</span> <span class="abstract-toggle" data-id="2611.16255">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.16255v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.16255v1) · [:material-content-copy: BibTeX](bibtex/2611.16255.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=sub-hourly" data-tag="sub-hourly">Sub-hourly</a>
    { .paper-tags }

-   #### A hybrid machine learning-analog framework for downscaling unprecedented climate extremes { #2611.15284 }

    *Nina Effenberger, Florian E. Roemer, Maybritt Schillinger, Maxim Samarin, Erich Fischer, Reto Knutti* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.15284">Extreme weather events are critical for climate impact assessments. While physical models can simulate unprecedented extremes, for example, through ensemble boosting, this approach is computationally...</span><span class="abstract-full" id="full-2611.15284" hidden>Extreme weather events are critical for climate impact assessments. While physical models can simulate unprecedented extremes, for example, through ensemble boosting, this approach is computationally demanding, often restricting output to coarse spatial resolutions. Probabilistic generative downscaling offers a solution by enabling efficient sampling from the entire climate distribution, including its extreme tails. Here, we first demonstrate that machine learning generated regional climate model heatwaves over Paris are physically realistic. We then leverage an emulated machine learning dataset to perform analog-based downscaling of coarse global climate model ensemble boosting events. Our hybrid framework effectively bridges high-resolution regional detail with rare-event sampling, providing an efficient tool for extreme event downscaling and capturing multi-model uncertainty.</span> <span class="abstract-toggle" data-id="2611.15284">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.15284v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.15284v1) · [:material-content-copy: BibTeX](bibtex/2611.15284.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=physics-ml-hybrid" data-tag="physics-ml-hybrid">Physics–ML hybrid</a> <a class="md-tag" href="/explore/?t=extremes" data-tag="extremes">Extremes</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a>
    { .paper-tags }

-   #### DaYu-TC: a global-regional nested AI system for tropical cyclone forecasting { #2611.15266 }

    *Runzepu Hou, Feng Zhang, Xin Wu, Xujun Wei, Fenghua Ling, Kan Dai, Gaozhen Nie, Leiming Ma et al.* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.15266">Accurate tropical cyclone intensity forecasts are essential for disaster preparedness, yet persistent intensity underestimation limits the value of artificial intelligence (AI) weather models for...</span><span class="abstract-full" id="full-2611.15266" hidden>Accurate tropical cyclone intensity forecasts are essential for disaster preparedness, yet persistent intensity underestimation limits the value of artificial intelligence (AI) weather models for predicting these destructive storms. Here we present DaYu-TC, a global-regional nested system that links the preservation of strong-wind and low-pressure signals with high-resolution regional prediction. Its global and regional models share an architecture and extreme-aware training objectives. The regional model combines evolving local states with global forecasts valid at each target time, connecting large-scale environmental information with finer-scale cyclone evolution. In an independent evaluation of 16 western North Pacific tropical cyclones in 2023, the global model retains competitive conventional forecast skill while reducing five-day maximum-wind root-mean-square error (RMSE) by 6-22% relative to other evaluated global AI models. Relative to the global model, regional nesting reduces maximum-wind RMSE by 22.0-33.1% and minimum-pressure RMSE by 28.6-39.8% across 12-120 h. The complete system outperforms concurrent operational high-resolution forecasts from the European Centre for Medium-Range Weather Forecasts for minimum pressure at 12-120 h and maximum wind speed at 48-96 h, with respective RMSE reductions of 26% and 12% at 72 h. Without retraining, the same regional model enhances three other global AI models, reducing overall maximum-wind RMSE by 22.6-28.6%. Five-day forecasts take under one minute on a single GPU. These results establish a transferable multiscale approach to reducing cyclone intensity underestimation, extending advances in AI weather forecasting towards more accurate prediction of high-impact weather events.</span> <span class="abstract-toggle" data-id="2611.15266">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.15266v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.15266v1) · [:material-content-copy: BibTeX](bibtex/2611.15266.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a> <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a>
    { .paper-tags }

-   #### Winter wheat improves soil carbon sequestration with reduced gains in warmer climates { #2611.15232 }

    *Georgios Giannarakis, Vasileios Sitokonstantinou, Dimitrios Bormpoudakis, Ilias Tsoumas et al.* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.15232">Sustainable agricultural intensification requires a precise understanding of how land use decisions influence soil organic carbon (SOC) sequestration under varying environmental conditions. We apply...</span><span class="abstract-full" id="full-2611.15232" hidden>Sustainable agricultural intensification requires a precise understanding of how land use decisions influence soil organic carbon (SOC) sequestration under varying environmental conditions. We apply causal machine learning to evaluate the heterogeneous impact of winter wheat-based rotations on SOC at the field-level across Lithuania. Using a nationwide dataset collected over five years (2018-2022), including crop types, management practices, soil properties, and environmental conditions, we estimate how the effects of winter wheat-based rotation vary across space and climate. Our results indicate that winter wheat generally enhances SOC levels, but its effectiveness is modulated by climate. Cooler regions experience higher SOC gains, whereas warmer regions show diminished or even negative effects. Future climate projections suggest that the potential for winter wheat carbon sequestration may decline under high-emission scenarios. These findings highlight the importance of localized agricultural strategies that account for climate variability to maximize SOC sequestration and ensure long-term soil sustainability.</span> <span class="abstract-toggle" data-id="2611.15232">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.15232v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.15232v1) · [:material-content-copy: BibTeX](bibtex/2611.15232.bib){ .bibtex-link }
    { .paper-links }

-   #### PhysXAI: A Theoretical Framework for Diagnosing Physical Hallucinations in Weather Foundation Models { #2611.15053 }

    *Anjan Mahapatra, Nikhil Raj* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.15053">Modern machine learning models for global weather forecasting achieve remarkable predictive accuracy, yet they frequently produce states that violate fundamental physical laws, including conservation...</span><span class="abstract-full" id="full-2611.15053" hidden>Modern machine learning models for global weather forecasting achieve remarkable predictive accuracy, yet they frequently produce states that violate fundamental physical laws, including conservation of mass, hydrostatic balance, or spectral energy scaling. We term these outputs physical hallucinations and propose PhysXAI, a rigorous diagnostic framework that detects, localises, and attributes such inconsistencies. Unlike existing post-hoc metric suites, PhysXAI is a learnable system that explicitly decouples forecast error from physical plausibility. We formalise the problem through a novel joint condition on numerical and physical residuals, and introduce a multi-branch encoder that extracts representations from conservation, dynamical, spectral, cross-variable, and spatiotemporal residuals. A key innovation is a physics-consistency contrastive learning objective, which we theoretically show the latent representation to be approximately isometric to the physical residual space, ensuring that diagnostic decisions remain grounded in physically meaningful deviations. We further derive an adaptive multi-objective loss with gradient-based weighting to balance competing diagnostic objectives. The framework produces spatially resolved hallucination maps, violation-type classifications, and calibrated Bayesian uncertainty, enabling both detection and interpretable attribution of physical inconsistencies. We conduct a comprehensive comparative evaluation of PhysXAI across ERA5, WeatherBench 2, and MERRA-2, using forecasts from GraphCast, Pangu-Weather, and NeuralGCM, with Python-written code and a fully reproducible experimental protocol. Our theoretical analysis and empirical evaluation establish PhysXAI as a principled framework for diagnosing physical hallucinations and advancing trustworthy AI for weather forecasting.</span> <span class="abstract-toggle" data-id="2611.15053">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.15053v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.15053v1) · [:material-content-copy: BibTeX](bibtex/2611.15053.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=foundation-models" data-tag="foundation-models">Foundation models</a> <a class="md-tag" href="/explore/?t=physics-ml-hybrid" data-tag="physics-ml-hybrid">Physics–ML hybrid</a> <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a>
    { .paper-tags }

-   #### Data-driven Reconstruction of the Phase Dynamics for the El Niño-Southern Oscillation Reveals Seasonality and Statistical Synchronization with the Annual Cycle { #2611.14868 }

    *Takahiro Arai, Yoji Kawamura* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.14868">In this study, we treat the El Niño-Southern Oscillation (ENSO) as a rhythmic system periodically forced by the annual cycle. We examine ENSO event seasonality and synchronization between the annual...</span><span class="abstract-full" id="full-2611.14868" hidden>In this study, we treat the El Niño-Southern Oscillation (ENSO) as a rhythmic system periodically forced by the annual cycle. We examine ENSO event seasonality and synchronization between the annual and ENSO cycles. We compute phase time series for both cycles from sea surface temperature data using a data-driven Koopman operator approach. We then use Bayesian inference to reconstruct phase equations describing how the phases of the two oscillators evolve. The reconstructed two-variable phase equations show that the annual cycle modulates the instantaneous frequency of the fundamental ENSO mode. Stochastic simulations of these equations incorporating the estimated stationary noise indicate that El Niño and La Niña events tend to last longer when they start in boreal summer and that the variability of their durations is greater when they start in boreal spring. Furthermore, the averaged form of the phase equations indicates statistical $3:1$ phase locking between the annual and ENSO cycles, whereas no statistical phase locking is found for the other $m:n$ ratios.</span> <span class="abstract-toggle" data-id="2611.14868">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.14868v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.14868v1) · [:material-content-copy: BibTeX](bibtex/2611.14868.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=subseasonal-to-seasonal" data-tag="subseasonal-to-seasonal">Subseasonal to seasonal</a>
    { .paper-tags }

</div>

