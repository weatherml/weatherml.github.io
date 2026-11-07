---
hide:
  - navigation
  - toc
title: weatherml
---

A collection of papers on AI for weather forecasting, climate modelling and atmospheric science.

<p class="page-meta" markdown="span">1623 papers · updated 2026-11-07 · <a href="feed.xml">:material-rss: RSS</a> · <a href="all_papers.bib" download>:material-download: BibTeX</a> · <a href="https://github.com/weatherml/weatherml.github.io/issues/new?template=suggest-paper.yml">:material-plus: Suggest a paper</a></p>

## Browse by Topic

<div class="grid cards topics" markdown>

-   [Global Models](papers/global-models/index.md) <span class="topic-count">383</span>
-   [Regional Models](papers/regional-models/index.md) <span class="topic-count">70</span>
-   [Nowcasting](papers/nowcasting/index.md) <span class="topic-count">107</span>
-   [Downscaling](papers/downscaling/index.md) <span class="topic-count">108</span>
-   [Post-processing](papers/post-processing/index.md) <span class="topic-count">27</span>
-   [Data Assimilation](papers/data-assimilation/index.md) <span class="topic-count">81</span>
-   [Climate Modeling](papers/climate-modeling/index.md) <span class="topic-count">309</span>
-   [Hydrology](papers/hydrology/index.md) <span class="topic-count">40</span>
-   [Ocean & Sea Ice](papers/ocean-sea-ice/index.md) <span class="topic-count">91</span>
-   [Air Quality & Composition](papers/air-quality-composition/index.md) <span class="topic-count">79</span>
-   [Remote Sensing](papers/remote-sensing/index.md) <span class="topic-count">106</span>
-   [Other](papers/other/index.md) <span class="topic-count">222</span>

</div>

## Recent Additions

<div class="grid cards" markdown data-search-exclude>

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

</div>

