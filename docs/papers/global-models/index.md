---
title: 'Global Models'
hide:
  - toc
---

<div class="listing-header" markdown>

# Global Models

<p class="page-meta" markdown="span">387 papers · page 1 of 13 · <a href="../../bib/global-models.bib" download>:material-download: BibTeX for this topic</a></p>

</div>

<div class="grid cards" markdown>

-   #### One2N: Recursive refinement breaks the resolution barrier for Earth-system AI { #2611.12249 }

    *Yanyan Tan, Xiangjun Tian, Can Wang, Hongqin Zhang, Yinhai Luo, Min Zhao, Zhe Jin, Xingchao Yang et al.* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.12249">Earth-system AI models approach kilometre-scale resolution, but halving grid spacing quadruples the grid-point count and end-to-end training cost. We introduce One2N, a recursive incremental...</span><span class="abstract-full" id="full-2611.12249" hidden>Earth-system AI models approach kilometre-scale resolution, but halving grid spacing quadruples the grid-point count and end-to-end training cost. We introduce One2N, a recursive incremental framework extending a coarse model through $N$ levels of cross-resolution residual correction. At each level the parent is frozen and excluded from backpropagation, so its activations and optimizer states are never retained; only a tier-specific module is trained. A single refinement level on the NeuralGCM-0.7$^\circ$ baseline yields the two-level One2N-0.07$^\circ$, trained on one 80 GB GPU. Against HRES-0.07$^\circ$ and ERA5-0.25$^\circ$ analyses it matches its parent's tropospheric skill while reducing upper-stratospheric errors and resolving finer structure. Tropical-cyclone track errors nearly match the parent's and are lower than Pangu-Weather-0.25$^\circ$ at medium range; RMSE is lower than that model's on 93.46% of common targets. One2N thus turns an existing forecasting system, or an inexpensively trained coarse baseline, into a reusable foundation for high-resolution development, breaking the resolution barrier.</span> <span class="abstract-toggle" data-id="2611.12249">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.12249v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.12249v1) · [:material-content-copy: BibTeX](../../bibtex/2611.12249.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=km-scale" data-tag="km-scale">Km-scale</a>
    { .paper-tags }

-   #### Ensemble Covariance Propagation with Machine Learning Weather Prediction Models in a Realistic Data Assimilation System { #2611.11918 }

    *Simon Toedtli, David Ahijevych, Wei Wang, Chris Snyder* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.11918">Integrating machine learning weather prediction (MLWP) models into classical data assimilation (DA) workflows requires that they accurately evolve ensemble covariances. We evaluate this capability in...</span><span class="abstract-full" id="full-2611.11918" hidden>Integrating machine learning weather prediction (MLWP) models into classical data assimilation (DA) workflows requires that they accurately evolve ensemble covariances. We evaluate this capability in a two-way coupled DA system, where an ensemble variational (EnVar) component evolves the mean and an ensemble Kalman filter (EnKF) evolves the covariances. The system is cycled over one month and assimilates a wide range of real-world observations, including microwave radiances. Using the MLWP model GraphCast for the EnKF and the physics-based MPAS model for the EnVar yields ensemble spread and covariances comparable to a reference configuration in which MPAS evolves both components. The resulting EnVar mean achieves consistently lower RMSE against GFS analyses, except near the model top, where spurious correlations degrade performance. These results demonstrate that MLWP models can propagate covariance information in a realistic DA cycling system and provide a proof of concept for their integration into future operational ensemble systems.</span> <span class="abstract-toggle" data-id="2611.11918">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.11918v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.11918v1) · [:material-content-copy: BibTeX](../../bibtex/2611.11918.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### SWXAI: A Unified Multimodal Explainability Library for Space Weather Forecasting and Heliophysics { #2611.10861 }

    *Benedict Antonio Mervyn, Anli Ji, Manolis K. Georgoulis, Berkay Aydin* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.10861">Space weather forecasting and heliophysics increasingly rely on machine learning models that process heterogeneous inputs such as multi-viewpoint multi-wavelength solar images and rasters,...</span><span class="abstract-full" id="full-2611.10861" hidden>Space weather forecasting and heliophysics increasingly rely on machine learning models that process heterogeneous inputs such as multi-viewpoint multi-wavelength solar images and rasters, multivariate time series, spatial annotations, and tabular event-level data. However, explainability workflows remain fragmented across libraries, model interfaces, and output formats, limiting reuse and reproducibility. We present SWXAI, an open-source library that unifies explainable AI (XAI) workflows for image, time-series, and tabular data in space weather forecasting and heliophysics. SWXAI provides a common, extensible interface across commonly used vision, time-series, and tabular learning models, with validation, adaptation, and persistent explanation objects. It also includes domain-grounded examples, reproducible notebooks, tests, and documentation, enabling users to inspect outputs, swap models or explainers, and extend workflows to new heliophysics datasets. SWXAI reduces modality-specific engineering overhead and supports comparable, reusable explanations across heterogeneous scientific machine learning pipelines. A benchmark against direct backend implementations shows that SWXAI substantially reduces the code users write to produce an explanation, with no practical runtime cost.</span> <span class="abstract-toggle" data-id="2611.10861">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.10861v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.10861v1) · [:material-content-copy: BibTeX](../../bibtex/2611.10861.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=interpretability" data-tag="interpretability">Interpretability</a>
    { .paper-tags }

-   #### Half-precision inference of AI weather prediction models: negligible loss of skill at a fraction of the cost { #2611.09573 }

    *Stefan Siegert* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.09573">AI weather prediction models are released with 32-bit (fp32) weights and run in fp32 by default, although current GPUs compute much faster in 16-bit formats. We run five publicly available models --...</span><span class="abstract-full" id="full-2611.09573" hidden>AI weather prediction models are released with 32-bit (fp32) weights and run in fp32 by default, although current GPUs compute much faster in 16-bit formats. We run five publicly available models -- Aurora, Pangu-Weather, FourCastNet, SFNO and FastNet -- on a single 16 GB GPU from 52 initial dates in 2024, in fp32 and with their networks in half precision (fp16), and verify the forecasts against ERA5. The physical state must stay in fp32: converting the whole model rounds it in physical units, which in fp16 overflows and in bf16 discards meaningful information. With the state, its normalisation and denormalisation in fp32, running the network in fp16 changes the global RMSE by at most 1.4%, and by about 0.5% or less for four of the five models, equivalent to less than one hour of lead time up to day 8 and at most two hours at day 10. The differences show no systematic pattern by region, scale or for tropical and extratropical storms, apart from a small tropical penalty in FourCastNet, and fp16 leaves the CRPS, spread and calibration of Aurora's ensemble almost unchanged. No value overflowed fp16's numeric range (+/-65,504); the largest stay 38 to 544 times below it. A 10-day forecast in fp16 is 1.3-3.5 times faster than in fp32 and uses 1.4-3.7 times less energy. Sensitivity experiments show that rules of thumb from other applications of reduced precision carry over only partly: the first and last layers dominate for some models but not for others, and no type of layer is sensitive on its own. For running AI weather models on a single GPU, fp16 with the state in fp32 is practically lossless at a fraction of the cost. bf16, often preferred for its wider range, is as fast but less accurate than fp16 for every model, and simple 8-bit quantisation costs far more skill.</span> <span class="abstract-toggle" data-id="2611.09573">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.09573v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.09573v1) · [:fontawesome-brands-github: Code](https://github.com/sieste/ai-nwp-fp16) · [:material-content-copy: BibTeX](../../bibtex/2611.09573.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=neural-operators" data-tag="neural-operators">Neural operators</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=efficiency" data-tag="efficiency">Efficiency</a>
    { .paper-tags }

-   #### Distinguishing Initial-Condition and Model Error Using Machine-Learning Weather Models { #2611.05592 }

    *P. Trent Vonich, Gregory J. Hakim* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.05592">Differentiable weather models allow atmospheric initial conditions to be directly optimized to shadow a known trajectory, which we take from a reanalysis dataset. Here, we extend this optimization to...</span><span class="abstract-full" id="full-2611.05592" hidden>Differentiable weather models allow atmospheric initial conditions to be directly optimized to shadow a known trajectory, which we take from a reanalysis dataset. Here, we extend this optimization to include the forecast model, GraphCast, using low-rank adaptation (LoRA). We compare model optimization with initial-condition (IC) optimization, and then joint optimization, where both are tuned simultaneously. When compared against a control forecast initialized with reanalysis data, LoRA and IC optimization individually produce similar improvements. We find similar results when IC and LoRA are jointly optimized across a spectrum of solutions where a tuning coefficient smoothly shifts the weight of the joint optimization from all-LoRA to all-IC; forecast improvement is not unique to the relative contributions of model and IC corrections. To better discriminate the source of error, we introduce a second model, Pangu-Weather, as a supervisor that gates IC updates during joint optimization. This gated optimization is the same as before, except that IC updates at each epoch are accepted only when they improve the forecasts of both models; otherwise, only the LoRA update is accepted. Results show that this cross-model constraint substantially improves agreement between the two models in their response to the same optimized ICs, narrowing the gap in mean forecast error reduction from 64% to 10%. Moreover, when used together, the LoRA and IC corrections from the gated optimization yield GraphCast forecast improvements similar to those from the ungated optimization. These results imply that model error that is encoded in the ungated ICs is shifted to the LoRA corrections via gating, yielding IC corrections that may provide better analysis estimates.</span> <span class="abstract-toggle" data-id="2611.05592">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.05592v2) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.05592v2) · [:material-content-copy: BibTeX](../../bibtex/2611.05592.bib){ .bibtex-link }
    { .paper-links }

-   #### Sonny-ENS: demystifying machine-learning ensemble weather forecasting { #2611.01614 }

    *Minjong Cheon* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.01614">Every weather forecast is uncertain, and decisions from flood warnings to energy trading depend on how well that uncertainty is quantified. Machine-learning (ML) ensembles now quantify it better than...</span><span class="abstract-full" id="full-2611.01614" hidden>Every weather forecast is uncertain, and decisions from flood warnings to energy trading depend on how well that uncertainty is quantified. Machine-learning (ML) ensembles now quantify it better than the best physics-based ensemble, yet two things about them remain opaque: whether their skill requires hundreds to thousands of accelerator-days of training, and how a network with a simple source of randomness produces a spread that follows the flow. Here we introduce Sonny-ENS, an ensemble of three 68.9-million-parameter networks with Monte Carlo dropout, trained on the continuous ranked probability score (CRPS) in 2.7 A40 GPU-days. On 2022, it has a lower CRPS than the European Centre for Medium-Range Weather Forecasts ensemble (ENS) on 96% of 570 targets and an 8.3% lower mean CRPS at lead times of 1-10 days. It outperforms the operational GenCast on 69% of the targets, and a 51-member, 15-day forecast takes 69 s on one GPU. Ablations identify the ingredients of this skill, and a Kolmogorov-Arnold sparse autoencoder shows that dropout injects uncertainty mainly in the tropics, whereas the spread grows where the atmosphere is baroclinically unstable (spatial correlation 0.91 with the Eady growth rate). Skillful probabilistic weather forecasts are therefore within reach of a single GPU server.</span> <span class="abstract-toggle" data-id="2611.01614">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.01614v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.01614v1) · [:material-content-copy: BibTeX](../../bibtex/2611.01614.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### Value of Adding AI Weather Model Ensembles to a Multi-model Probabilistic Blend { #2610.42130 }

    *Harris Sop Nkuiate, Gavin Evans, Christopher A. T. Ferro, Frank Kwasniok, Caroline Bain et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.42130">Many weather forecasting organizations use a multi-model blend of ensemble members from different prediction systems to obtain a more accurate forecast. Recent studies have shown that blending...</span><span class="abstract-full" id="full-2610.42130" hidden>Many weather forecasting organizations use a multi-model blend of ensemble members from different prediction systems to obtain a more accurate forecast. Recent studies have shown that blending deterministic AI-based models alongside physics-based models improves the overall forecast skill. However, the incorporation of an ensemble AI-based model into a multi-model blend is yet to be investigated. This study explores the value of blending the ensemble version of the Artificial Intelligence Forecasting System (AIFS-CRPS) with the Met Office's convection permitting physics-based ensemble, MOGREPS-UK. We show that blending site temperature (and wind speed) forecasts over the United Kingdom from both ensemble physics-based and ensemble AI-based models leads to an increase in overall forecast skill spatially, across lead times and across exceedance thresholds, especially when the models have different error characteristics. By decomposing the continuous ranked probability score of the blend into component means and diversity terms, we show that a more accurate blend can be achieved by assigning more weight to more skillful models and using models with complementary errors. For example, we show that a blend of MOGREPS-UK with either AIFS-CRPS or the physics-based Integrated Forecasting System (IFS) is better than a blend of IFS and AIFS-CRPS which are not very diverse. This has implications for the importance of weather model diversity and may lend weight to different model development approaches being pursued in tandem. We also demonstrate that calibrating AIFS-CRPS forecasts yields an increase in forecast skill comparable to calibrated ensemble physics-based forecasts, and calibrating the source forecasts before blending and the blended forecast after blending improves the overall forecast accuracy.</span> <span class="abstract-toggle" data-id="2610.42130">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.42130v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.42130v1) · [:material-content-copy: BibTeX](../../bibtex/2610.42130.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=evaluation" data-tag="evaluation">Evaluation</a>
    { .paper-tags }

-   #### Scale-aware and spatially-localised optimisation of kilometre-scale MLWP models { #2610.37325 }

    *James Chappell, Sophie Luise Arana, Jack Bowyer, Alejandro Coca-Castro, Evangeline Corcoran et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.37325">Preserving fine-scale structure without degrading spatial accuracy is a central challenge for kilometre-scale machine-learning weather prediction (MLWP). We introduce a loss based on the dual-tree...</span><span class="abstract-full" id="full-2610.37325" hidden>Preserving fine-scale structure without degrading spatial accuracy is a central challenge for kilometre-scale machine-learning weather prediction (MLWP). We introduce a loss based on the dual-tree complex wavelet transform that constrains atmospheric fields jointly by scale, location, and orientation. We evaluate its performance using an experimental joint global--regional FastNet configuration, which produces 1.5 km forecasts over the UK alongside $\sim$31 km global forecasts. Relative to both mean-squared error (MSE) and an FFT-based spectral objective, the wavelet loss substantially improves spectral fidelity. Compared with MSE training, it maintains similar pointwise performance, improves neighbourhood-based verification, and produces sharper precipitation fields with more realistic intensity distributions. These results demonstrate that spatially localised wavelet constraints can preserve multiscale atmospheric structure while largely avoiding the loss of pointwise accuracy associated with globally supported spectral objectives.</span> <span class="abstract-toggle" data-id="2610.37325">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.37325v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.37325v1) · [:material-content-copy: BibTeX](../../bibtex/2610.37325.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a> <a class="md-tag" href="/explore/?t=km-scale" data-tag="km-scale">Km-scale</a>
    { .paper-tags }

-   #### GNN-VIEW: Visual Inspection and Explanation of Graph-based Earth-System Models { #2610.35693 }

    *Alessandro Lucatelli, Ferdinand Unterhuber, Katharina Krenzer, Mennatallah El-Assady, Benedikt Soja et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.35693">Machine learning Earth-system models are becoming increasingly important, and so is their development. Graph-based models, one of the most prominent approaches, allow developers to constrain the...</span><span class="abstract-full" id="full-2610.35693" hidden>Machine learning Earth-system models are becoming increasingly important, and so is their development. Graph-based models, one of the most prominent approaches, allow developers to constrain the learning process before training via graph connectivity. Yet, developing these models remains challenging: the graph is built algorithmically with little intuition about the final result beyond static plotting, which requires extra work and is often not detailed enough. Even after successful training, how the model arrives at forecasts remains obscure. We present GNN-VIEW, an interactive browser-based tool that supports the development of such systems. GNN-VIEW exposes the graph's information pathways, allowing model developers to debug the graph structure before training. After training, GNN-VIEW uses attribution-based explainability to reveal which input features are used by the trained model for a given forecast. This helps developers understand why a model fails. We showcase the usefulness of GNN-VIEW in the construction stage of a graph for an ocean forecasting model, and in explaining poorly predicted precipitation of a trained weather forecasting model during Hurricane Laura.</span> <span class="abstract-toggle" data-id="2610.35693">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.35693v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.35693v1) · [:material-content-copy: BibTeX](../../bibtex/2610.35693.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=graph-neural-networks" data-tag="graph-neural-networks">Graph neural networks</a> <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a>
    { .paper-tags }

-   #### Can We Trust Evaluations of AI Weather Forecasts for Extremes? How Evaluation Protocols Shape Model Rankings { #2610.31237 }

    *Tobias Biegert, Sam Allen, Zhongwei Zhang, Sebastian Lerch* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.31237">AI weather models perform competitively with traditional physics-based models at a fraction of the computational cost, and they are thus already used operationally. However, studies reach conflicting...</span><span class="abstract-full" id="full-2610.31237" hidden>AI weather models perform competitively with traditional physics-based models at a fraction of the computational cost, and they are thus already used operationally. However, studies reach conflicting conclusions about their ability to forecast extreme weather events, and comparisons across studies are complicated by differing evaluation protocols. We discuss the challenges that arise when comparing AI and physics-based weather forecasts for extreme events. Within a common experimental design, we find that model rankings are highly sensitive to the choice of evaluation metrics: the considered physics-based model tends to perform best when evaluation is restricted to observed extreme events, whereas AI models generally perform better when evaluated using weighted scoring functions. A contingency table analysis shows that AI models underpredict the frequency of extremes, so that their scores are generally dominated by misses, while the physics-based model forecasts extremes more frequently and its scores are dominated by false alarms. Different evaluation metrics penalise these errors differently. Further score decompositions confirm that the score differences arise primarily from miscalibration rather than discrimination ability. This sheds light on the contrasting results in the literature, and highlights the importance of the evaluation metrics used to evaluate AI models.</span> <span class="abstract-toggle" data-id="2610.31237">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.31237v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.31237v1) · [:material-content-copy: BibTeX](../../bibtex/2610.31237.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=extremes" data-tag="extremes">Extremes</a> <a class="md-tag" href="/explore/?t=evaluation" data-tag="evaluation">Evaluation</a>
    { .paper-tags }

-   #### From Weather to Enhanced Weathering: Weather Foundation Models for Carbon Dioxide Removal at Operational Scale { #2610.28973 }

    *Sasankh Munukutla, Felipe Oviedo, Danrong Zhang, Kenji Takeda, Juan M Lavista Ferres* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.28973">Enhanced Rock Weathering (ERW) is a carbon removal solution with the potential to remove gigatons of CO$_2$ per year, but quantifying that removal today requires dense soil sampling and laboratory...</span><span class="abstract-full" id="full-2610.28973" hidden>Enhanced Rock Weathering (ERW) is a carbon removal solution with the potential to remove gigatons of CO$_2$ per year, but quantifying that removal today requires dense soil sampling and laboratory analysis. We evaluate five weather Foundation Models (FMs) as frozen encoders feeding lightweight decoders, under a leave-one-field-out protocol on 5,192 chronosequences from 264 fields across Brazil and the US. To our knowledge, this is the first holistic evaluation of weather FMs for ERW and the largest evaluation on operational ERW data. We find that across weather FMs, representations transfer to ERW: our best model, TerraNova-ERW, built on Aurora 1.5, positively orders measurements within fields it has never seen. Zero-shot prediction of a new field's mean removal is not yet feasible, and we attribute this to data scale: the same architecture excels on $\sim$$10^5$ natural-weathering sequences, anchoring the scale we expect ERW data must reach. Few-shot prediction, blending the model with a few samples per field, is supported today: TerraNova-ERW recovers over 60% of the highest-removal fields with two samples per field, roughly halves the samples needed to estimate a field's mean removal, and can cut sampling on our densest fields by 28-98%.</span> <span class="abstract-toggle" data-id="2610.28973">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.28973v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.28973v1) · [:material-content-copy: BibTeX](../../bibtex/2610.28973.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=foundation-models" data-tag="foundation-models">Foundation models</a>
    { .paper-tags }

-   #### Tropical Cyclone Attributes in Sparse Autoencoder Features of an AI Weather Model { #2610.28904 }

    *Jiapei Liu, Ruidong Li, Ting Sun, Guangheng Ni* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.28904">Artificial intelligence (AI) weather models now forecast weather events skillfully, but how they represent these events internally remains unknown. Sparse autoencoders (SAEs) provide sparse features...</span><span class="abstract-full" id="full-2610.28904" hidden>Artificial intelligence (AI) weather models now forecast weather events skillfully, but how they represent these events internally remains unknown. Sparse autoencoders (SAEs) provide sparse features for probing hidden states, but how information about multiple physical attributes of an event is organized across them remains unclear. Using tropical cyclones in GraphCast as a test case, we analyze 2,115 six-hourly timesteps from 205 cyclones, relating SAE activations to position, intensity, and pressure structure across five Layer-8 decompositions and 16 processor layers. Each attribute relates to multiple features; position and pressure-structure relations overlap in a smaller subset. This organization recurs across independent SAEs and layers without feature alignment. In deeper layers, the leading position feature aligns better with position and pressure structure six hours ahead than at present. These results suggest that the organization of physical-attribute relations provides a reproducible level of description for interpreting cyclone-related information in GraphCast.</span> <span class="abstract-toggle" data-id="2610.28904">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.28904v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.28904v1) · [:material-content-copy: BibTeX](../../bibtex/2610.28904.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a> <a class="md-tag" href="/explore/?t=6-hourly" data-tag="6-hourly">6-hourly</a>
    { .paper-tags }

-   #### GraphCast-Mamba: Learning Non-Markovian Corrections for Weather Forecasting { #2610.28512 }

    *Ilya Vilkoviskiy, Liang-Hong Mo, Suozhi Huang, Dmitry A. Abanin* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.28512">Unresolved atmospheric processes can introduce history dependence beyond the short input windows used by many weather forecasting models. We introduce GraphCast--Mamba, which augments a frozen,...</span><span class="abstract-full" id="full-2610.28512" hidden>Unresolved atmospheric processes can introduce history dependence beyond the short input windows used by many weather forecasting models. We introduce GraphCast--Mamba, which augments a frozen, pretrained $1^\circ$ GraphCast backbone with a trainable Mamba correction branch combining graph message passing and Mamba layers. The branch retains a temporal memory state across successive six-hour forecast steps, learning corrections informed by temporal context beyond GraphCast's two-frame input. Training is confined to the correction branch, using ERA5 data from 2015--2021. After 2,500 optimizer updates, including warm-up, the selected model achieves substantial improvements relative to the uncorrected GraphCast backbone. It reduces the GraphCast loss (normalized weighted mean-squared error; MSE) by 14.42% when aggregated over the full ten-day forecast and by 21.66% at day 10 (240 h after forecast initialization). At this final forecast lead time, root-mean-square error (RMSE) decreases by 7.09--14.60% across all 11 evaluated variable groups, spanning temperature, wind, pressure, geopotential, specific humidity, and precipitation. These results demonstrate broad improvements in ten-day weather forecasts through a short correction-training schedule while retaining the pretrained backbone.</span> <span class="abstract-toggle" data-id="2610.28512">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.28512v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.28512v1) · [:material-content-copy: BibTeX](../../bibtex/2610.28512.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=graph-neural-networks" data-tag="graph-neural-networks">Graph neural networks</a>
    { .paper-tags }

-   #### Beyond the record: evaluating probabilistic AI weather forecasts deep in the tail { #2610.25545 }

    *Bastiaan Van Velthoven, Cas Decancq, Diego G. Miralles, Stijn Luca, Thomas Mortier* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.25545">Can probabilistic artificial intelligence (AI) weather forecasts retain skill beyond their training domain? We compare four state-of-the-art AI models (ArchesWeatherGen, U-Cast, WeatherNext2 and...</span><span class="abstract-full" id="full-2610.25545" hidden>Can probabilistic artificial intelligence (AI) weather forecasts retain skill beyond their training domain? We compare four state-of-the-art AI models (ArchesWeatherGen, U-Cast, WeatherNext2 and GenCast-mini) with the physics-based IFS-ENS over land in 2022, at 1.5° resolution and lead times of 2--10 days. Monthly ERA5 records from 1979--2021 define common historical boundaries, and we characterize new records set in 2022 by their depth, i.e., the exceedance margin relative to local historical variability. Two proper scores, record-threshold weighted CRPS and Brier score, assess the magnitude and probability of record exceedances across all eligible cases, retaining both missed extremes and false alarms. Verified against ERA5, AI models outperform IFS-ENS for heat and cold records at almost every lead and depth. For heat, the advantage strengthens with depth, partly because IFS-ENS over-forecasts deep records, and largely persists for deeper records under HRES-fc0 verification. For wind, gains erode for deeper records and longer leads. Better tail scores can nevertheless coexist with aggregate probability deficits or surpluses, and some improvements arise chiefly on non-exceedances. Increasing underprediction on progressively more extreme observed subsets can also arise through outcome selection, even for an ideal conditional-mean forecast. These findings establish that native probabilistic AI ensembles retain, and for heat increase, their advantage over IFS-ENS as thresholds move beyond local records, although all advantages narrow with lead time. They caution against generalizing the extrapolation failure reported for deterministic AI forecasts to probabilistic systems, and motivate common verification conventions that separate comparative tail skill, performance on realized extremes, and probabilistic calibration.</span> <span class="abstract-toggle" data-id="2610.25545">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.25545v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.25545v1) · [:material-content-copy: BibTeX](../../bibtex/2610.25545.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=evaluation" data-tag="evaluation">Evaluation</a> <a class="md-tag" href="/explore/?t=coarse" data-tag="coarse">Coarse (≥1°)</a> <a class="md-tag" href="/explore/?t=monthly" data-tag="monthly">Monthly</a>
    { .paper-tags }

-   #### UtoMe: Observation-Uncertainty-Guided Token Merging for Weather Foundation Models { #2610.25001 }

    *Yunbei Zhang, Janet Wang, Xi Xiao, Jihun Hamm, Xiao Wang* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.25001">Weather transformers process dense reanalysis fields whose reliability varies across variables, locations, and weather regimes. Token merging can reduce computation, but feature similarity alone does...</span><span class="abstract-full" id="full-2610.25001" hidden>Weather transformers process dense reanalysis fields whose reliability varies across variables, locations, and weather regimes. Token merging can reduce computation, but feature similarity alone does not account for this heterogeneous input uncertainty. We introduce UtoMe, an observation-uncertainty-guided token merging framework that combines dynamics saliency with ensemble data assimilation spread as a relative input-reliability proxy. The resulting importance score protects selected tokens as anchors and calibrates bipartite matching between the remaining tokens. Three variants use external uncertainty at both training and inference, during training only, or neither stage. In ERA5 72-hour forecasting at two spatial resolutions, UtoMe improves Overall error over train-time ToMe at matched token budgets. At $1.40625^\circ$ and 50% token reduction, UtoMe-E lowers Overall error from 67.32 to 53.00 at the same 0.55x encoder FLOPs. Small reductions remain close to the full-token baseline; larger reductions incur measurable accuracy losses. Component ablations and spatial uncertainty controls support combining dynamics with aligned uncertainty when allocating a weather transformer's token budget.</span> <span class="abstract-toggle" data-id="2610.25001">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.25001v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.25001v1) · [:material-content-copy: BibTeX](../../bibtex/2610.25001.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=transformers" data-tag="transformers">Transformers</a> <a class="md-tag" href="/explore/?t=foundation-models" data-tag="foundation-models">Foundation models</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### Interpretability of AI weather models via intermediate decoding { #2610.24110 }

    *Matthias Beylich, Kirsten I. Tempest, Tobias Selz, George C. Craig* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.24110">AI weather prediction models are often regarded as black boxes. Despite their impressive skill, there is little understanding of how the input data is processed during inference to produce the...</span><span class="abstract-full" id="full-2610.24110" hidden>AI weather prediction models are often regarded as black boxes. Despite their impressive skill, there is little understanding of how the input data is processed during inference to produce the forecast. This limitation arises from the use of high-dimensional latent representations in which individual dimensions are no longer directly interpretable in physical terms. To gain insight into how forecasts are constructed, we propose the method of intermediate decoding for AI weather models and apply it to GraphCast, Aurora and ArchesWeather. In this approach, latent states from intermediate depths are decoded back into meteorological fields. To do this, we use the decoder that has been trained to output the forecast at the end of processing. Since latent states at different depths do not share a consistent basis, we train affine transformations that align intermediate latent states with the decoder's expected input space. Applying this framework to the three models yields the following results. First, the skill of forecasts at intermediate stages of processing improves gradually through the processor layers. Second, bypassing all processor blocks shows that the encoder-decoder pair is not an identity mapping returning the input, but instead plays a significant role in the forecast construction. Third, by examining increments of individual processor blocks across model depth, we show that all three models progressively refine the forecast from coarse to finer spatial scales. Finally, we assess the physical consistency of the intermediate decoding forecasts and find that large scales remain physically balanced throughout processing, while small scales become more balanced with each additional processor block.</span> <span class="abstract-toggle" data-id="2610.24110">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.24110v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.24110v1) · [:material-content-copy: BibTeX](../../bibtex/2610.24110.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=interpretability" data-tag="interpretability">Interpretability</a>
    { .paper-tags }

-   #### Can AI Forecast the Extremes? Evaluating Probabilistic Weather Models on Tropical Cyclones { #2610.22077 }

    *Nathan Chalumeau, Xavier Renard, Remi Meynadier, Hugo Rakotoarimanga, Marcin Detyniecki et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.22077">Weather prediction is undergoing an important shift, with data-driven AI models outperforming leading numerical weather prediction systems on average at a fraction of the computational cost. Yet,...</span><span class="abstract-full" id="full-2610.22077" hidden>Weather prediction is undergoing an important shift, with data-driven AI models outperforming leading numerical weather prediction systems on average at a fraction of the computational cost. Yet, whether this success extends to extreme events remains contested. Earlier evaluations have exposed important shortcomings, but have largely focused on deterministic models and metrics, leaving the capabilities of the latest probabilistic models less understood. We conduct an extensive benchmark of tropical cyclone forecasting, comparing GenCast, FourCastNet 3, and FGN with ECMWF's operational physics-based ensemble, IFS-ENS. We develop a suite of probabilistic metrics and diagnostic analyses to assess cyclone tracking, intensity biases, uncertainty, and the wind-pressure relationship that characterizes storm structure. We find that, despite remaining limitations in intensity and ensemble spread, leading AI models substantially outperform IFS-ENS in cyclone tracking: FGN achieves better probabilistic location scores for 90% of evaluated storms in 2025. Further analyses reveal model-specific failure modes and suggest a lingering influence of pretraining data, pointing to opportunities for targeted improvement. Nonetheless, these results suggest that AI models are capable of capturing physical relationships that remain predictive well beyond typical weather conditions, motivating further investigation into how these capabilities emerge and how to strengthen them.</span> <span class="abstract-toggle" data-id="2610.22077">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.22077v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.22077v1) · [:material-content-copy: BibTeX](../../bibtex/2610.22077.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### Forecast lead time and the timing of crew deployment for weather-driven power outages using AI weather models { #2610.20918 }

    *Mohammad Kian Golkar, Luciano Alves de Oliveira, Mohamad Alipour* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.20918">Utilities request mutual aid crews to restore power after severe storms, and requesting early secures more crews but relies on less accurate forecasts. We examined how forecast lead time affects the...</span><span class="abstract-full" id="full-2610.20918" hidden>Utilities request mutual aid crews to restore power after severe storms, and requesting early secures more crews but relies on less accurate forecasts. We examined how forecast lead time affects the value of outage predictions for crew deployment, using operational forecasts from two AI weather models, GraphCast and FourCastNet, to predict outage occurrence for counties across the contiguous United States 24 to 120 hours ahead. Forecasts were evaluated by the share of outage burden captured by the counties where crews would be staged. Captured burden declined with lead time: the loss between three and five days was significant in all analyses, whereas the loss within three days was significant only for outages that had not yet started. Early deployment with all requested crews outperformed late deployment with fewer, and we report the availability ratio at which the two break even. Antecedent land surface predictors computed at forecast valid time include weather from the forecast period, which inflates their skill by 42 to 53% and reverses their apparent advantage over forecast meteorology at long lead times. Computed correctly, they improved occurrence skill by 17.5% at 24 hours but did not significantly change captured burden, a gap that attribution analysis linked to their static component. Prediction intervals covered only 0.476 of tropical cyclone events against a nominal 0.90. Calibration by hazard class corrected most of this shortfall at an 11% cost in interval width, as a diagnostic only, because hazard class is known only after the event.</span> <span class="abstract-toggle" data-id="2610.20918">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.20918v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.20918v1) · [:material-content-copy: BibTeX](../../bibtex/2610.20918.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a>
    { .paper-tags }

-   #### Establishing baseline model performances for optical turbulence forecasting { #2610.18702 }

    *M. De Sepibus, E. Masciadri, C. Weinberger, A. Otarola, S. Miesske, C. Veillet, S. Ragland* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.18702">Accurate forecasting of optical turbulence (OT) is useful to optimise telescope scheduling and maximising observing efficiency, as the performance of adaptive optics systems strongly depends on...</span><span class="abstract-full" id="full-2610.18702" hidden>Accurate forecasting of optical turbulence (OT) is useful to optimise telescope scheduling and maximising observing efficiency, as the performance of adaptive optics systems strongly depends on turbulence conditions. While various numerical weather prediction and machine learning models are developed to predict these complex turbulence properties, properly evaluating their added value requires well-defined baseline reference models. This study establishes baseline performance thresholds for key astroclimatic parameters using two distinct models: one forecasting the parameter average over a specific timescale, and the other forecasting the temporal evolution over the same specific timescale. We apply these frameworks to extensive, multi-year datasets from the sites of the Very Large Telescope (VLT) and the Large Binocular Telescope (LBT) and discuss the possibility of extrapolating reference values of forecast performances for other sites based on their optical turbulence statistical characteristics. Our results show that baseline predictive accuracy varies according to the observing site, the specific parameter, the chosen timescale, and the forecasting approach. Therefore, an OT forecasting model's accuracy cannot be quantified in absolute terms, but relatively to the baseline.</span> <span class="abstract-toggle" data-id="2610.18702">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.18702v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.18702v1) · [:material-content-copy: BibTeX](../../bibtex/2610.18702.bib){ .bibtex-link }
    { .paper-links }

-   #### Multivariate phase-space interpretation of ensemble forecasts: A comparison of physical and AI models { #2610.16422 }

    *Martin Bonte, Stéphane Vannitsem* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.16422">Ensemble forecasts from the ECMWF's Integrated Forecasting System (IFS) and the Artificial Intelligence Forecasting System (AIFS-ENS) are compared using a multi-dimensional evaluation framework....</span><span class="abstract-full" id="full-2610.16422" hidden>Ensemble forecasts from the ECMWF's Integrated Forecasting System (IFS) and the Artificial Intelligence Forecasting System (AIFS-ENS) are compared using a multi-dimensional evaluation framework. Within this framework, the uncertainty is interpreted as the linear subspace spanned by the ensemble members in phase space, from which various geometric properties are extracted. As revealed by the analysis, the error does not entirely lie in this subspace, but is reasonably well calibrated in all directions. Moreover, the framework allows to quantify that this subspace critically depends on the number of ensemble members, suggesting that the current setups only partially represent the uncertainty subspace. This phase-space interpretation furthermore allows a direct comparison of the error-representation patterns of AIFS-ENS and IFS by computing the principal angles between the subspaces: IFS and AIFS ensemble subspaces do not entirely overlap in phase space, even though they were shown to have similar abilities to represent the amplitude and the structure of the error.</span> <span class="abstract-toggle" data-id="2610.16422">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.16422v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.16422v1) · [:material-content-copy: BibTeX](../../bibtex/2610.16422.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### WeatherBridge: Forecast-Aware Multi-Source Input Adaptation for Pretrained Weather Forecasting Models { #2610.14863 }

    *Mutian Xi, Xinyu Su, Xiaohui Zhong, Hao Li* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.14863">Data-driven weather forecasting models are commonly trained on ERA5, yet operational deployment must use timely initial states from physics-based numerical weather prediction (NWP) systems whose...</span><span class="abstract-full" id="full-2610.14863" hidden>Data-driven weather forecasting models are commonly trained on ERA5, yet operational deployment must use timely initial states from physics-based numerical weather prediction (NWP) systems whose distributions differ substantially from ERA5. Direct initialization with these states can degrade autoregressive forecasts, motivating input adaptation that keeps the pretrained forecasting model unchanged. Here, we introduce WeatherBridge, a unified, source-conditioned adapter that maps heterogeneous operational NWP initial states to ERA5 reference states without updating the pretrained forecasting model. WeatherBridge predicts the structured residual corrections with a Swin-UNet and injects source embeddings across multiple spatial scales, allowing the model to capture shared atmospheric structure and source-specific discrepancies. Its training objective combines latitude-weighted ERA5 reconstruction with One-step Rollout Alignment (ORA), which backpropagates one-step forecast error through a frozen FuXi model to optimize the adapted states for both initial state fidelity and downstream forecast compatibility. We evaluate WeatherBridge using daily GFS, HRES, and CMA initializations from 1 January to 22 November 2025 across six variables and lead times up to 240 hours. Relative to raw source initialization of FuXi, WeatherBridge reduces the mean RMSE of FuXi over the 6- 240-hour rollout for 17 of 18 source-variable combinations, with a median reduction of 10.2% and a maximum reduction of 46.3%. The adapter trained with frozen FuXi also improves FengWu forecasts on GFS inputs without retraining either the adapter or FengWu. These results demonstrate that forecast-aware input adaptation can improve forecasts from heterogeneous operational NWP inputs and transfer across pretrained ERA5-based forecasting models.</span> <span class="abstract-toggle" data-id="2610.14863">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.14863v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.14863v1) · [:material-content-copy: BibTeX](../../bibtex/2610.14863.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=transformers" data-tag="transformers">Transformers</a> <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a> <a class="md-tag" href="/explore/?t=daily" data-tag="daily">Daily</a>
    { .paper-tags }

-   #### Diffusion Precipitation Generation with Medium-range Weather Forecasting { #2610.12976 }

    *Guowen Li, Yang Liu, Mengxuan Chen, Jiashun Cheng, Haoyuan Liang, Shilei Cao, Yujie Wang et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.12976">Accurate daily precipitation forecasts are important for weather-sensitive decisions. Although recent Weather Foundation Models (WFMs) have achieved significant progress, they are trained on...</span><span class="abstract-full" id="full-2610.12976" hidden>Accurate daily precipitation forecasts are important for weather-sensitive decisions. Although recent Weather Foundation Models (WFMs) have achieved significant progress, they are trained on reanalysis data, which exhibit regional and intensity-dependent biases. Since retraining WFMs on observation-based targets is costly, we study Dynamics-Precipitation Decoupling, a plug-and-play framework that uses a separately trained generator to map non-precipitation forecasts from a frozen weather model to observation-aligned daily precipitation. Its key challenges are to capture heterogeneous rainfall distributions and align daily accumulations with atmospheric signals that evolve across space and time. To address them, we train a diffusion generator conditional on atmospheric states and satellite-gauge estimates of precipitation. A multi-scale recurrent encoder summarizes atmospheric evolution, while cross-attention aligns atmospheric and precipitation features during denoising. Extensive evaluation of various backbones, including Pangu-Weather, GraphCast, and HRES, on global and regional forecasting shows improved daily precipitation accuracy and event detection, with better representation of localized accumulation patterns. Global ensemble evaluation shows lower Brier scores than ENS and training-from-scratch GenCast at all reported lead times, with lower CRPS in most comparisons. Remarkably, a heavy-rainfall case study using 2,288 stations in China further supports improved daily accumulation patterns and event detection against ground observations.</span> <span class="abstract-toggle" data-id="2610.12976">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.12976v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.12976v1) · [:material-content-copy: BibTeX](../../bibtex/2610.12976.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a> <a class="md-tag" href="/explore/?t=daily" data-tag="daily">Daily</a>
    { .paper-tags }

-   #### A Graph Neural Network for Global Daily Fire Radiative Power Prediction at Medium-Range Lead Times { #2610.11022 }

    *Li Zhang, Jun Wang, Isidora Jankov, Yongxin Liu, Gonzalo A. Ferrada, Ravan Ahmadov, Ligia Bernardet et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.11022">Skillful prediction of biomass-burning activity several days in advance is important for air-quality forecasting and aerosol prediction. Two operational constraints motivate this work. First, the...</span><span class="abstract-full" id="full-2610.11022" hidden>Skillful prediction of biomass-burning activity several days in advance is important for air-quality forecasting and aerosol prediction. Two operational constraints motivate this work. First, the GBBEPx satellite fire radiative power (FRP) product used to initialize NOAA's GEFS-Aerosols is available with about a 1.5-day latency, so each forecast cycle relies on the most recently available, but already outdated, fire observations. Second, these fire inputs are then held fixed throughout the subsequent 5-day operational forecast, or 7 days in the GSL experimental system, effectively assuming no evolution in fire activity. We develop a data-driven model that predicts global FRP one to seven days ahead from the most recent available observations. The model adapts a spatiotemporal graph neural network using reanalysis meteorology, land-cover and vegetation information, recent fire history, and GBBEPx FRP as the training target. It is trained on 2020-2022 data and evaluated for 2023-2024. The model reproduces the global seasonal cycle and substantially outperforms persistence. At 0.1$^\circ$ resolution, mean squared error is reduced by 32% at one-day lead and 43% at seven days in 2023, and by 24% and 40% in 2024. At 1$^\circ$ resolution, the critical success index ranges from 0.32 to 0.60. Detection skill declines only modestly with lead time, whereas intensity skill degrades more rapidly. Large fires are detected reliably, but their radiative power is systematically underestimated. These results demonstrate useful predictability of fire activity several days ahead and identify intensity calibration and small-fire placement as the main remaining challenges before predicted FRP can support operational aerosol forecasts.</span> <span class="abstract-toggle" data-id="2610.11022">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.11022v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.11022v1) · [:material-content-copy: BibTeX](../../bibtex/2610.11022.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=graph-neural-networks" data-tag="graph-neural-networks">Graph neural networks</a> <a class="md-tag" href="/explore/?t=coarse" data-tag="coarse">Coarse (≥1°)</a> <a class="md-tag" href="/explore/?t=daily" data-tag="daily">Daily</a>
    { .paper-tags }

-   #### Strategic Governance of AI Models in Earth Science { #2610.10560 }

    *Makoto Kelp, Amirhossein Arzani, Patricia Castellanos, Paul Griffiths, Ivan Higuera-Mendieta et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.10560">AI foundation models pretrained on weather and climate data are increasingly fine-tuned to Earth science tasks well beyond weather forecasting. Their development and adoption are outpacing the...</span><span class="abstract-full" id="full-2610.10560" hidden>AI foundation models pretrained on weather and climate data are increasingly fine-tuned to Earth science tasks well beyond weather forecasting. Their development and adoption are outpacing the scientific community's ability to evaluate them. These models are judged almost entirely by benchmark skill metrics, which measure how closely a forecast reproduces a reference product but not whether a model represents the physical processes governing the system it predicts. Forecast skill and physical reliability are therefore distinct properties. The distinction is most consequential under the nonstationary conditions of a changing climate for which these models were never trained. We identify five priorities for the physical evaluation of AI models in Earth science from task-specific emulators to foundation models, spanning training data, fine-tuning, behavioral testing, mechanistic interpretability, and output validation. We recommend three activities for the coming decade: 1) open AI-ready evaluation datasets, 2) a shared reporting standard for physics-based evaluation, and 3) a dedicated research program on the safety of these models.</span> <span class="abstract-toggle" data-id="2610.10560">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.10560v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.10560v1) · [:material-content-copy: BibTeX](../../bibtex/2610.10560.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=foundation-models" data-tag="foundation-models">Foundation models</a> <a class="md-tag" href="/explore/?t=evaluation" data-tag="evaluation">Evaluation</a> <a class="md-tag" href="/explore/?t=interpretability" data-tag="interpretability">Interpretability</a>
    { .paper-tags }

-   #### WxFM-XL: Adapting Univariate Foundation Models to Multi-Station Weather Forecasting { #2610.10057 }

    *Xiao Wang, Changjian Chen, Zhuo Tang, Rongwen Li, Hongwu Liu, Kenli Li* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.10057">With the rise of univariate time series foundation models (e.g., Sundial, Timer), initial efforts have been made to extend them to multivariate settings. However, these models mainly focus on...</span><span class="abstract-full" id="full-2610.10057" hidden>With the rise of univariate time series foundation models (e.g., Sundial, Timer), initial efforts have been made to extend them to multivariate settings. However, these models mainly focus on modeling correlations among variables. When they are applied to multi-station weather forecasting, two important factors are often overlooked: (1) the spatial information of stations, and (2) different error priors of different stations relative to the foundation model. In this paper, we propose WxFM-XL, a model for adapting univariate time series foundation models to multi-station weather forecasting. WxFM-XL introduces a cross-station error correlation prior graph to capture stationwise error priors with respect to the foundation model. Building on this, we further propose a dynamic fusion mechanism that adaptively integrates a spatial correlation graph with the error correlation prior graph. Experiments on multiple datasets demonstrate that our model outperforms state of the art baselines.</span> <span class="abstract-toggle" data-id="2610.10057">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.10057v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.10057v1) · [:material-content-copy: BibTeX](../../bibtex/2610.10057.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=foundation-models" data-tag="foundation-models">Foundation models</a> <a class="md-tag" href="/explore/?t=station-point" data-tag="station-point">Station / point</a>
    { .paper-tags }

-   #### Learning joint probabilistic weather forecasts from station observations alone { #2610.09898 }

    *Chaeyeon Yi, Yun Am Seo* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.09898">Assessing compound weather risks requires forecasts representing dependence between variables. CLARA (Calibrated Advection-Routing Attention) learns joint Gaussian predictive distributions of five...</span><span class="abstract-full" id="full-2610.09898" hidden>Assessing compound weather risks requires forecasts representing dependence between variables. CLARA (Calibrated Advection-Routing Attention) learns joint Gaussian predictive distributions of five surface variables from station observations alone, without numerical weather prediction or reanalysis; the approximately 28,000-parameter model supports CPU training and prediction. Across six multi-year folds on 96 stations, its lead-mean energy score is 4.9% lower than that of a learned comparator with matched temporal inputs (4.7% with a similar parameter count) and 11-65% lower than those of statistical baselines. Holding marginal variances fixed, removing learned correlations worsens joint negative log-likelihood by 1.0-2.8 nats per station. A covariance-scale estimator, proved consistent under stated assumptions, improves short-lead calibration but over-corrects at long leads. Synthetic interventions show an attention-bias coefficient alone does not measure forecast influence. Retrained in ten regions on six continents, CLARA outperforms persistence in all 60 multi-year region-lead comparisons and a similarly sized learned model in 57 of 60.</span> <span class="abstract-toggle" data-id="2610.09898">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.09898v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.09898v1) · [:material-content-copy: BibTeX](../../bibtex/2610.09898.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=station-point" data-tag="station-point">Station / point</a>
    { .paper-tags }

-   #### Mechanistic Interpretability of Atmospheric Rivers in GraphCast { #2610.07583 }

    *Madelyn Mathai, Timothy B. Higgins, Kevin M. Grise, Chirag Agarwal, Antonios Mamalakis* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.07583">While AI weather models now rival operational forecasts, how they represent the atmosphere internally remains an open question: feature attribution reveals which input patterns matter, not what the...</span><span class="abstract-full" id="full-2610.07583" hidden>While AI weather models now rival operational forecasts, how they represent the atmosphere internally remains an open question: feature attribution reveals which input patterns matter, not what the model computes or how it combines information internally. We train sparse autoencoders (SAEs) on GraphCast to uncover its learned concepts, using atmospheric rivers as our phenomenon of focus. Both standard and Matryoshka SAEs show GraphCast computes atmospheric river intensity, measured by integrated vapor transport (IVT), as a stable internal variable, despite IVT being neither an input nor a target. In contrast to the unstructured concept retrieval of the standard SAE, the Matryoshka SAE orders concepts by importance and exposes their relations. Atmospheric river concepts persist across depth and direct interventions confirm causality. This method offers a way to find internal variables and determine which of them the model actually relies on, which is a prerequisite for asking whether those variables remain meaningful as the phenomenon changes under a warming climate.</span> <span class="abstract-toggle" data-id="2610.07583">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.07583v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.07583v1) · [:material-content-copy: BibTeX](../../bibtex/2610.07583.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=interpretability" data-tag="interpretability">Interpretability</a>
    { .paper-tags }

-   #### Xaurora: Generative Weather Forecasting with Denoising Stochastic Interpolants from a Foundation Model Prior { #2610.06509 }

    *Eliot Walt, Miltiadis Kofinas, Nikolaj Mücke, Efstratios Gavves, Dim Coumou* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.06509">Deep learning has revolutionised weather forecasting in recent years, especially through atmospheric foundation models, which offer competitive skill for a fraction of the computational costs of...</span><span class="abstract-full" id="full-2610.06509" hidden>Deep learning has revolutionised weather forecasting in recent years, especially through atmospheric foundation models, which offer competitive skill for a fraction of the computational costs of classic physics-based models. However, most existing foundation models are deterministic, limiting the generation of large ensembles for accurate uncertainty quantification, extreme weather risk assessment, and long-range weather forecasting. Furthermore, these models incur a large, often prohibitive, computational overhead to train from scratch. To address these shortcomings, we turn a pretrained deterministic prior model, namely the Aurora foundation model, into a generative ensemble-prediction model. To that end, we introduce a novel generative method, Denoising Stochastic Interpolants, combined with a replay buffer for Stochastic Differential Equation (SDE) rollout, enabling probabilistic training of SDE trajectories. Our stochastic foundation model, Xaurora, is finetuned from the small Aurora version, yet it approaches the state-of-the-art on global ensemble metrics and is competitive with the large version of Aurora. Our method is parameter and sample efficient, and generates skilful 15-day forecasts in 13 minutes. Our results demonstrate that deterministic foundation models can be efficiently extended into even stronger stochastic models.</span> <span class="abstract-toggle" data-id="2610.06509">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.06509v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.06509v1) · [:material-content-copy: BibTeX](../../bibtex/2610.06509.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=foundation-models" data-tag="foundation-models">Foundation models</a> <a class="md-tag" href="/explore/?t=sub-hourly" data-tag="sub-hourly">Sub-hourly</a>
    { .paper-tags }

-   #### FlexCast: Adaptive Weather Forecasting from Arbitrary Field Sets { #2610.05296 }

    *Yuang Zhang, Chen Hui, Weisi Lin, Haiqi Zhu, Xiulai Wang, Sun-Yuan Kung, Feng Jiang* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.05296">Most deep learning weather models assign a fixed set of variables and pressure levels to predefined channels, limiting transfer across atmospheric field configurations. This dependence on a fixed...</span><span class="abstract-full" id="full-2610.05296" hidden>Most deep learning weather models assign a fixed set of variables and pressure levels to predefined channels, limiting transfer across atmospheric field configurations. This dependence on a fixed field set limits the transferability of trained models across atmospheric field configurations. We propose FlexCast, a field-adaptive weather forecasting model that uses a single set of parameters to produce identity-aligned forecasts for variable-cardinality subsets drawn from a 69-field ERA5 registry. Specifically, a metadata-conditioned adapter the first encodes variable identity, pressure level, and field type and combines them with spatial features. Then, shared rank-16 projec?tions are modulated by metadata-dependent gates to produce field?specific features, while masked set fusion aggregates the available fields into a fixed-width representation. Subsequently, a multiscale U-Transformer processes the fused atmospheric features, while an identity-aware query decoder produces forecasts for the requested fields. Finally, FlexCast learns a standardized six-hour increment and applies it recursively to generate forecasts at longer lead times. Experiments on the 2020 ERA5 test set demonstrate that FlexCast operates across varying field configurations. Compatible cross-field context is associated with lower forecast errors, whereas mismatched context increases them.</span> <span class="abstract-toggle" data-id="2610.05296">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.05296v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.05296v1) · [:material-content-copy: BibTeX](../../bibtex/2610.05296.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=transformers" data-tag="transformers">Transformers</a>
    { .paper-tags }

-   #### S$^3$N: A Spherical Spiral Scanning Network for Weather Forecasting { #2610.04338 }

    *Fan Yan, Chen Hui, Weisi Lin, Haiqi Zhu, Feng Jiang, Sun-Yuan Kung, Wei Zhang* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.04338">Machine learning-based weather prediction (MLWP) has achieved strong performance in global weather forecasting. Recent Hierarchical Equal Area isoLatitude Pixelation (HEALPix)-based methods use the...</span><span class="abstract-full" id="full-2610.04338" hidden>Machine learning-based weather prediction (MLWP) has achieved strong performance in global weather forecasting. Recent Hierarchical Equal Area isoLatitude Pixelation (HEALPix)-based methods use the HEALPix (HP) grid to avoid area distortion near the poles of conventional latitude-longitude (LL) grids. However, existing HP-based approaches often use pointwise mapping methods and process HP pixels within separate base faces or local windows. Consequently, the mapping may introduce reconstruction errors and cross-face communication depends on handcrafted boundary handling or shifted windows. We propose the Spherical Spiral Scanning Network (S$^3$N) to address both limitations. First, L2Proj provides a bidirectional method for mapping atmospheric fields between the LL and HP grids through an $L^2$ projection of their continuous finite-element representations. Second, the Attention-Guided Quad-Spiral State-Space Scanning (AQSS) block uses cross-latitude attention to guide selective state-space updates along four global pole-to-pole spiral paths. This design enables continuous information propagation across HP base-face boundaries without additional boundary-processing mechanisms. Experiments show that S$^3$N achieves better results at 4-, 7-, and 10-day lead times, and exhibits slower error growth in long-range forecasting.</span> <span class="abstract-toggle" data-id="2610.04338">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.04338v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.04338v1) · [:material-content-copy: BibTeX](../../bibtex/2610.04338.bib){ .bibtex-link }
    { .paper-links }

</div>

<nav class="pager" markdown="span">**1** [2](2.md) [3](3.md) [4](4.md) [5](5.md) [6](6.md) [7](7.md) [8](8.md) [9](9.md) [10](10.md) [11](11.md) [12](12.md) [13](13.md) [Older :material-arrow-right:](2.md){ .pager-step }</nav>

