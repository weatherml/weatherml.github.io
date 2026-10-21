---
hide:
  - navigation
  - toc
title: weatherml
---

A collection of papers on AI for weather forecasting, climate modelling and atmospheric science.

<p class="page-meta" markdown="span">1574 papers · updated 2026-10-21 · <a href="feed.xml">:material-rss: RSS</a> · <a href="all_papers.bib" download>:material-download: BibTeX</a> · <a href="https://github.com/weatherml/weatherml.github.io/issues/new?template=suggest-paper.yml">:material-plus: Suggest a paper</a></p>

## Browse by Topic

<div class="grid cards topics" markdown>

-   [Global Models](papers/global-models/index.md) <span class="topic-count">375</span>
-   [Regional Models](papers/regional-models/index.md) <span class="topic-count">66</span>
-   [Nowcasting](papers/nowcasting/index.md) <span class="topic-count">104</span>
-   [Downscaling](papers/downscaling/index.md) <span class="topic-count">106</span>
-   [Post-processing](papers/post-processing/index.md) <span class="topic-count">26</span>
-   [Data Assimilation](papers/data-assimilation/index.md) <span class="topic-count">79</span>
-   [Climate Modeling](papers/climate-modeling/index.md) <span class="topic-count">304</span>
-   [Hydrology](papers/hydrology/index.md) <span class="topic-count">39</span>
-   [Ocean & Sea Ice](papers/ocean-sea-ice/index.md) <span class="topic-count">89</span>
-   [Air Quality & Composition](papers/air-quality-composition/index.md) <span class="topic-count">74</span>
-   [Remote Sensing](papers/remote-sensing/index.md) <span class="topic-count">101</span>
-   [Other](papers/other/index.md) <span class="topic-count">211</span>

</div>

## Recent Additions

<div class="grid cards" markdown data-search-exclude>

-   #### GraphCast-Mamba: Learning Non-Markovian Corrections for Weather Forecasting { #2610.28512 }

    *Ilya Vilkoviskiy, Liang-Hong Mo, Suozhi Huang, Dmitry A. Abanin* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.28512">Unresolved atmospheric processes can introduce history dependence beyond the short input windows used by many weather forecasting models. We introduce GraphCast--Mamba, which augments a frozen,...</span><span class="abstract-full" id="full-2610.28512" hidden>Unresolved atmospheric processes can introduce history dependence beyond the short input windows used by many weather forecasting models. We introduce GraphCast--Mamba, which augments a frozen, pretrained $1^\circ$ GraphCast backbone with a trainable Mamba correction branch combining graph message passing and Mamba layers. The branch retains a temporal memory state across successive six-hour forecast steps, learning corrections informed by temporal context beyond GraphCast's two-frame input. Training is confined to the correction branch, using ERA5 data from 2015--2021. After 2,500 optimizer updates, including warm-up, the selected model achieves substantial improvements relative to the uncorrected GraphCast backbone. It reduces the GraphCast loss (normalized weighted mean-squared error; MSE) by 14.42% when aggregated over the full ten-day forecast and by 21.66% at day 10 (240 h after forecast initialization). At this final forecast lead time, root-mean-square error (RMSE) decreases by 7.09--14.60% across all 11 evaluated variable groups, spanning temperature, wind, pressure, geopotential, specific humidity, and precipitation. These results demonstrate broad improvements in ten-day weather forecasts through a short correction-training schedule while retaining the pretrained backbone.</span> <span class="abstract-toggle" data-id="2610.28512">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.28512v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.28512v1) · [:material-content-copy: BibTeX](bibtex/2610.28512.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=graph-neural-networks" data-tag="graph-neural-networks">Graph neural networks</a>
    { .paper-tags }

-   #### Observation Assimilation Transformer: Probabilistic Weather Analysis from Sparse Station Observations { #2610.28327 }

    *Christian Donner, Michele Cattaneo, Maxim Samarin, Francesco Zanetta, Icíar Lloréns Jover et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.28327">We present the Observation Assimilation Transformer (OAT), a geometry-aware neural process that corrects and refines a prior numerical weather prediction (NWP) field into a high-resolution posterior...</span><span class="abstract-full" id="full-2610.28327" hidden>We present the Observation Assimilation Transformer (OAT), a geometry-aware neural process that corrects and refines a prior numerical weather prediction (NWP) field into a high-resolution posterior field by assimilating sparse observations. OAT is an encoder-decoder transformer that considers a set of observed locations and their NWP prior to condition predictions at a set of arbitrary unobserved query locations. Its sparse attention mechanism makes it scalable to thousands of station observations and depends on relative positions between locations, never on the absolute coordinates. The model allows for partially missing observations, predictions at arbitrary continuous locations without grid projection, and outputs are parametric distributions. A station-randomized training algorithm is developed with continuous ranked probability score (CRPS) loss. We test OAT on two datasets - one at regional scale with hundreds of stations, and one at continental scale with thousands of stations. On these datasets OAT outperforms several baselines and shows well calibrated predictions. In an illustrative case study, OAT's downscaling capabilities are demonstrated by downscaling ERA5 (~31km) to COSMO (1km), where OAT achieves lower mean absolute error (MAE) and CRPS for temperature and precipitation on test stations compared to COSMO.</span> <span class="abstract-toggle" data-id="2610.28327">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.28327v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.28327v1) · [:material-content-copy: BibTeX](bibtex/2610.28327.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=transformers" data-tag="transformers">Transformers</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a> <a class="md-tag" href="/explore/?t=station-point" data-tag="station-point">Station / point</a>
    { .paper-tags }

-   #### Disentangling forced and internal components in observed climate evolution using spatiotemporal deep learning { #2610.28219 }

    *Parvathi Kooloth, Wenyu Zhou, L. Ruby Leung, Shang-Ping Xie, Hailong Wang, Qiang Fu* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.28219">Attribution of observed historical climate evolution to externally forced change and internal variability is fundamental to understanding climate change and evaluating climate models. Yet such...</span><span class="abstract-full" id="full-2610.28219" hidden>Attribution of observed historical climate evolution to externally forced change and internal variability is fundamental to understanding climate change and evaluating climate models. Yet such attribution remains challenging because the forced response is intertwined with internal variability in observations. Here, we develop a convolutional long short-term memory network (ConvLSTM) to disentangle these two components. Trained and evaluated using large-ensemble simulations from multiple climate models, ConvLSTM consistently outperforms existing model-based, statistical, and machine-learning approaches. Applying ConvLSTM to observations reveals an evolving spatial pattern of the forced response and substantial contributions from internal variability to recent multidecadal climate trends. For several key climate features whose attribution remains debated, ConvLSTM indicates that internal variability amplified the forced reversal of Atlantic interhemispheric temperature asymmetry, dominated the observed La Niña-like warming pattern over the equatorial Pacific and the associated precipitation decline over southwestern North America since 1980, and modulated Arctic amplification around its nearly constant threefold forced value. Comparison between the ConvLSTM-estimated and model-simulated forced responses reveals broadly consistent patterns but also important discrepancies, with climate models simulating stronger Northern Hemisphere warming, weaker subtropical high intensification, and smaller land precipitation declines. Overall, ConvLSTM provides a robust framework for attributing historical climate evolution and evaluating the fidelity of climate models</span> <span class="abstract-toggle" data-id="2610.28219">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.28219v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.28219v1) · [:material-content-copy: BibTeX](bibtex/2610.28219.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=recurrent-networks" data-tag="recurrent-networks">Recurrent networks</a>
    { .paper-tags }

-   #### TerraNostra-Bench: Benchmarking Spatial Downscaling and Temporal Rollout for Global 1-km Land-Use Reconstruction { #2610.27793 }

    *Amirpasha Mozaffari, Marina Castaño, Stefano Materia, Joan Vedrí, Jordi Varela-Agrelo et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.27793">Uncertainty in the terrestrial carbon cycle remains a major constraint in climate projections, partly due to imperfect land surface representations in Earth system models. We propose a data-driven...</span><span class="abstract-full" id="full-2610.27793" hidden>Uncertainty in the terrestrial carbon cycle remains a major constraint in climate projections, partly due to imperfect land surface representations in Earth system models. We propose a data-driven framework for high-resolution land-use reconstruction, treated as a semantic-segmentation task, that combines coarse-resolution scenario data and climate classification with static geophysical features to produce annual land-use maps at 1 km resolution. We introduce TerraNostra-Bench, a change-aware evaluation protocol that separates spatial downscaling from autoregressive temporal rollout and scores models on the rare land-use-change pixels rather than the dominant static background. Using it, we compare convolutional (U-Net) and transformer (Swin) backbones in autoregressive and non-autoregressive schemes against persistence, linear, and shallow convolutional baselines. The autoregressive prior is the most informative input yet accumulates error over long rollouts: autoregressive models lose 5.7-7.2 accuracy points over 13 years, against 2.2-2.8 for non-autoregressive models. This trade-off is governed by the autoregressive choice rather than the backbone. It can be tuned by an inference-time prior-trust threshold that requires no retraining and by the training-time weighting of reconstruction and change objectives; the two levers trace complementary frontiers, with calibration recovering up to 10 accuracy points while keeping change sensitivity above the persistence floor. Extrapolating beyond the training window, backward to 1955-1959 and forward to 2015-2019 under scenario forcing, the models retain most of their in-window skill and show the same trade-off, indicating it is intrinsic to the design. This work provides the methodological foundation and evaluation surface for a forthcoming temporally continuous land-use product.</span> <span class="abstract-toggle" data-id="2610.27793">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.27793v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.27793v1) · [:material-content-copy: BibTeX](bibtex/2610.27793.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=km-scale" data-tag="km-scale">Km-scale</a>
    { .paper-tags }

-   #### Wet-Biased Data Pipelines Mislead the Training and Evaluation of Generative Precipitation Nowcasting { #2610.26486 }

    *Christoph Metzl, Stefan Kapsreiter, Kianusch Vahid Yousefnia, Virginia Poli, Richard Müller et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.26486">Generative models for precipitation nowcasting are trained and evaluated predominantly on wet-biased datasets, which over-represent precipitation events to emphasize rare extremes. Selecting test...</span><span class="abstract-full" id="full-2610.26486" hidden>Generative models for precipitation nowcasting are trained and evaluated predominantly on wet-biased datasets, which over-represent precipitation events to emphasize rare extremes. Selecting test cases based on target observations is known to favor forecasts that match the selected distribution rather than the intended deployment distribution. We show analytically how this target-dependent sampling also shifts the conditional distribution learned by generative models. When training and test sampling match, even an ideal model learns precisely the distribution favored by evaluation, allowing the evaluation bias to conceal the training bias. Using a fixed satellite-to-radar diffusion model, we demonstrate that target-dependent test sampling can reverse the calibration assessment. The model appears under-predictive on a wet-biased test set but over-predictive on a deployment-representative test set. This result is replicated outside the model's training domain after a seven-year gap. Utilizing classifier-free guidance as an example, we demonstrate how the contradictory calibration assessment can misdirect downstream model development.</span> <span class="abstract-toggle" data-id="2610.26486">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.26486v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.26486v1) · [:material-content-copy: BibTeX](bibtex/2610.26486.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### Spatio-temporal precipitation downscaling at hourly resolution based on km-scale regional simulations { #2610.26145 }

    *Henry Addison, Elizabeth J. Kendon, Peter A. G. Watson* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.26145">Effectively preparing for changes in extreme flooding requires understanding present and future rainfall extremes with return periods up to $\sim$100-1000 years, including changes in intensity,...</span><span class="abstract-full" id="full-2610.26145" hidden>Effectively preparing for changes in extreme flooding requires understanding present and future rainfall extremes with return periods up to $\sim$100-1000 years, including changes in intensity, spatial structure at km-scale and sub-daily temporal variability. Conventional km-scale climate models give greatly increased realism compared to typical global climate models (GCMs) but are too computationally costly to allow enough simulations for robustly quantifying changes in extremes. This work presents an approach based on video diffusion to generate realistic precipitation at 8.8km-hourly resolution by spatially and temporally downscaling daily, GCM-scale atmospheric variables. The emulator is trained to produce similar output to a $4\times$-coarsening of the Met Office's 2.2km-resolution UK convection-permitting model. An application for such an emulator is to downscale large-ensemble GCM simulations to allow quantification of $\sim$100-1000-year return level rainfall, which in turn can be used to drive hydrological and flood models for improved projections of future changes in flood risk.</span> <span class="abstract-toggle" data-id="2610.26145">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.26145v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.26145v1) · [:material-content-copy: BibTeX](bibtex/2610.26145.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=extremes" data-tag="extremes">Extremes</a> <a class="md-tag" href="/explore/?t=km-scale" data-tag="km-scale">Km-scale</a> <a class="md-tag" href="/explore/?t=hourly" data-tag="hourly">Hourly</a> <a class="md-tag" href="/explore/?t=daily" data-tag="daily">Daily</a>
    { .paper-tags }

-   #### AirMPA: A Meteorology-to-Pollution Adapter for Global Air Quality Forecasting { #2610.25797 }

    *Yiheng Wang, Kai Zheng, Yuetan Lin, Fanglu Fan, Chenliang Tao, Guochao Chen, Hongliang Zhang, Hao Li* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.25797">Accurate real-time forecasting of atmospheric pollutants is essential for reducing exposure risks and improving public health. Traditional numerical forecasting systems suffer from incomplete...</span><span class="abstract-full" id="full-2610.25797" hidden>Accurate real-time forecasting of atmospheric pollutants is essential for reducing exposure risks and improving public health. Traditional numerical forecasting systems suffer from incomplete parameterizations, uncertain inputs, and high computational cost. Existing deep learning approaches improve efficiency, but often entangle meteorological and pollutant variables within unified forecasting frameworks, limiting flexibility and pollutant-specific representation learning. To address these challenges, we propose AirMPA, a decoupled autoregressive framework for meteorology-to-pollution forecasting. AirMPA predicts 13 atmospheric pollutants at 0.4° spatial resolution and 12-h temporal resolution using meteorological fields, emission priors, and historical pollutant concentrations. The model is built on an improved Swin-Transformer architecture with zonal release attention and dual-branch fusion, enabling more effective modeling of global transport continuity and heterogeneous pollutant dynamics. During training, AirMPA uses the fifth-generation atmospheric reanalysis from the European Centre for Medium-Range Weather Forecasts (ERA5) as meteorological forcing, while during inference it directly ingests forecast fields from external meteorological models, enabling stable autoregressive prediction up to 5 days ahead. Experiments show that AirMPA consistently outperforms Aurora and surpasses the Copernicus Atmosphere Monitoring Service (CAMS) in approximately 90% of pollutant-lead-time evaluation cases over the 5-day forecast horizon, while outperforming CAMS on all variables beyond 48 h. These results demonstrate the potential of decoupled meteorology-to-pollution forecasting for global air quality prediction.</span> <span class="abstract-toggle" data-id="2610.25797">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.25797v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.25797v1) · [:material-content-copy: BibTeX](bibtex/2610.25797.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=transformers" data-tag="transformers">Transformers</a>
    { .paper-tags }

-   #### Zephyrus-R1: Agentic Reinforcement Learning for Weather Reporting { #2610.25625 }

    *Fengfei Yu, Marshall Fisher, Isadora White, Sumanth Varambally, Yasaman Jafari, Shivam Yadav et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.25625">Reinforcement learning (RL) has emerged as a key method for training AI models in scientific applications, but RL remains difficult when outputs cannot easily be verified against a known ground-truth...</span><span class="abstract-full" id="full-2610.25625" hidden>Reinforcement learning (RL) has emerged as a key method for training AI models in scientific applications, but RL remains difficult when outputs cannot easily be verified against a known ground-truth answer. Natural-language weather forecast reports are one such domain: they translate atmospheric data into assessments of weather evolution, hazards, and uncertainty for both the general public and key decision-makers. Unlike numerical weather-analysis questions, they admit many valid outputs, which limits deterministic verification. To address this, we develop Zephyrus-R1, a practical framework for RL on scientific weather forecast reporting. We focus on three key components of the agentic RL pipeline: reward design, curriculum learning, and an analysis of human-expert alignment. To encourage both report legibility and factual accuracy, we design a reward function that combines a meteorological rubric with bidirectional claim entailment, both evaluated against National Weather Service reference reports. Our framework optimizes this reward in a simulated tool-calling environment, enabling vision-language agents to gather and analyze forecast evidence across regions and lead times before producing a report. We use annotations from three meteorologists to audit the claim labels produced by our LLM judges. Finally, a curriculum helps small models first develop tool use and scientific reasoning through RL on verifiable weather-analysis questions, then optimize the open-ended report reward. After training Qwen3.5-4B with Zephyrus-R1, we improve the mean across six judge-metric scores by 26% over the base model and achieve results competitive with frontier open-weight models. For Qwen3.8-27B, we improve the same mean by 13% over the base model, by 6.7% over Kimi-K3, and by 2.7% over GPT-5.6-Luna.</span> <span class="abstract-toggle" data-id="2610.25625">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.25625v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.25625v1) · [:material-content-copy: BibTeX](bibtex/2610.25625.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=llms-agents" data-tag="llms-agents">LLMs & agents</a> <a class="md-tag" href="/explore/?t=reinforcement-learning" data-tag="reinforcement-learning">Reinforcement learning</a>
    { .paper-tags }

-   #### Beyond the record: evaluating probabilistic AI weather forecasts deep in the tail { #2610.25545 }

    *Bastiaan Van Velthoven, Cas Decancq, Diego G. Miralles, Stijn Luca, Thomas Mortier* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.25545">Can probabilistic artificial intelligence (AI) weather forecasts retain skill beyond their training domain? We compare four state-of-the-art AI models (ArchesWeatherGen, U-Cast, WeatherNext2 and...</span><span class="abstract-full" id="full-2610.25545" hidden>Can probabilistic artificial intelligence (AI) weather forecasts retain skill beyond their training domain? We compare four state-of-the-art AI models (ArchesWeatherGen, U-Cast, WeatherNext2 and GenCast-mini) with the physics-based IFS-ENS over land in 2022, at 1.5° resolution and lead times of 2--10 days. Monthly ERA5 records from 1979--2021 define common historical boundaries, and we characterize new records set in 2022 by their depth, i.e., the exceedance margin relative to local historical variability. Two proper scores, record-threshold weighted CRPS and Brier score, assess the magnitude and probability of record exceedances across all eligible cases, retaining both missed extremes and false alarms. Verified against ERA5, AI models outperform IFS-ENS for heat and cold records at almost every lead and depth. For heat, the advantage strengthens with depth, partly because IFS-ENS over-forecasts deep records, and largely persists for deeper records under HRES-fc0 verification. For wind, gains erode for deeper records and longer leads. Better tail scores can nevertheless coexist with aggregate probability deficits or surpluses, and some improvements arise chiefly on non-exceedances. Increasing underprediction on progressively more extreme observed subsets can also arise through outcome selection, even for an ideal conditional-mean forecast. These findings establish that native probabilistic AI ensembles retain, and for heat increase, their advantage over IFS-ENS as thresholds move beyond local records, although all advantages narrow with lead time. They caution against generalizing the extrapolation failure reported for deterministic AI forecasts to probabilistic systems, and motivate common verification conventions that separate comparative tail skill, performance on realized extremes, and probabilistic calibration.</span> <span class="abstract-toggle" data-id="2610.25545">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.25545v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.25545v1) · [:material-content-copy: BibTeX](bibtex/2610.25545.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=evaluation" data-tag="evaluation">Evaluation</a> <a class="md-tag" href="/explore/?t=coarse" data-tag="coarse">Coarse (≥1°)</a> <a class="md-tag" href="/explore/?t=monthly" data-tag="monthly">Monthly</a>
    { .paper-tags }

-   #### UtoMe: Observation-Uncertainty-Guided Token Merging for Weather Foundation Models { #2610.25001 }

    *Yunbei Zhang, Janet Wang, Xi Xiao, Jihun Hamm, Xiao Wang* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.25001">Weather transformers process dense reanalysis fields whose reliability varies across variables, locations, and weather regimes. Token merging can reduce computation, but feature similarity alone does...</span><span class="abstract-full" id="full-2610.25001" hidden>Weather transformers process dense reanalysis fields whose reliability varies across variables, locations, and weather regimes. Token merging can reduce computation, but feature similarity alone does not account for this heterogeneous input uncertainty. We introduce UtoMe, an observation-uncertainty-guided token merging framework that combines dynamics saliency with ensemble data assimilation spread as a relative input-reliability proxy. The resulting importance score protects selected tokens as anchors and calibrates bipartite matching between the remaining tokens. Three variants use external uncertainty at both training and inference, during training only, or neither stage. In ERA5 72-hour forecasting at two spatial resolutions, UtoMe improves Overall error over train-time ToMe at matched token budgets. At $1.40625^\circ$ and 50% token reduction, UtoMe-E lowers Overall error from 67.32 to 53.00 at the same 0.55x encoder FLOPs. Small reductions remain close to the full-token baseline; larger reductions incur measurable accuracy losses. Component ablations and spatial uncertainty controls support combining dynamics with aligned uncertainty when allocating a weather transformer's token budget.</span> <span class="abstract-toggle" data-id="2610.25001">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.25001v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.25001v1) · [:material-content-copy: BibTeX](bibtex/2610.25001.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=transformers" data-tag="transformers">Transformers</a> <a class="md-tag" href="/explore/?t=foundation-models" data-tag="foundation-models">Foundation models</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### Spherical Physics-informed Neural Operator with Multi-scale Coupling for Meteorological Downscaling { #2610.24708 }

    *Yiqiang Ye, Yichi Wang, Haodi Zhang, Jiawei Wen, Jiahui Jiang, Zhaoyu Zhong, Jiangjian Yu et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.24708">Meteorological downscaling is crucial for high-resolution regional climate forecasting and disaster early warning. While neural operators have emerged as a promising paradigm for modeling complex...</span><span class="abstract-full" id="full-2610.24708" hidden>Meteorological downscaling is crucial for high-resolution regional climate forecasting and disaster early warning. While neural operators have emerged as a promising paradigm for modeling complex spatiotemporal mappings, existing frameworks often struggle with spherical manifold geometric distortions, inherent atmospheric multi-scale coupling mismatches, and lack of explicit atmospheric laws. We propose the Spherical Physics-informed Neural Operator, which utilizes a Spherical Laplacian Decomposition to partition atmospheric fields into hierarchical frequency components, maintaining exact point-wise correspondence across scales. To evaluate these representations at arbitrary locations, we introduce a localized spherical integral operator that approximates continuous kernel transforms via geometry-aware attention. Dynamical consistency is further enforced by embedding differentiable constraints into the learning process. Extensive experiments demonstrate that our framework attains superior accuracy and zero-shot generalization across various meteorological variables and unseen queries, representing a robust and interpretable solution for global-to-regional meteorological downscaling.</span> <span class="abstract-toggle" data-id="2610.24708">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.24708v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.24708v1) · [:material-content-copy: BibTeX](bibtex/2610.24708.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=neural-operators" data-tag="neural-operators">Neural operators</a> <a class="md-tag" href="/explore/?t=physics-ml-hybrid" data-tag="physics-ml-hybrid">Physics–ML hybrid</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a>
    { .paper-tags }

-   #### Interpretability of AI weather models via intermediate decoding { #2610.24110 }

    *Matthias Beylich, Kirsten I. Tempest, Tobias Selz, George C. Craig* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.24110">AI weather prediction models are often regarded as black boxes. Despite their impressive skill, there is little understanding of how the input data is processed during inference to produce the...</span><span class="abstract-full" id="full-2610.24110" hidden>AI weather prediction models are often regarded as black boxes. Despite their impressive skill, there is little understanding of how the input data is processed during inference to produce the forecast. This limitation arises from the use of high-dimensional latent representations in which individual dimensions are no longer directly interpretable in physical terms. To gain insight into how forecasts are constructed, we propose the method of intermediate decoding for AI weather models and apply it to GraphCast, Aurora and ArchesWeather. In this approach, latent states from intermediate depths are decoded back into meteorological fields. To do this, we use the decoder that has been trained to output the forecast at the end of processing. Since latent states at different depths do not share a consistent basis, we train affine transformations that align intermediate latent states with the decoder's expected input space. Applying this framework to the three models yields the following results. First, the skill of forecasts at intermediate stages of processing improves gradually through the processor layers. Second, bypassing all processor blocks shows that the encoder-decoder pair is not an identity mapping returning the input, but instead plays a significant role in the forecast construction. Third, by examining increments of individual processor blocks across model depth, we show that all three models progressively refine the forecast from coarse to finer spatial scales. Finally, we assess the physical consistency of the intermediate decoding forecasts and find that large scales remain physically balanced throughout processing, while small scales become more balanced with each additional processor block.</span> <span class="abstract-toggle" data-id="2610.24110">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.24110v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.24110v1) · [:material-content-copy: BibTeX](bibtex/2610.24110.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=interpretability" data-tag="interpretability">Interpretability</a>
    { .paper-tags }

-   #### Energy-Aware Earth System Science: A Community Roadmap { #2610.23216 }

    *Ioana Colfescu, Annalisa Bracco, Kara Moraw, Chen Lu, Michael Bareford, Tom Beucler et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.23216">Earth system models underpin the evidence that informs net-zero policy, yet their energy demands are rarely considered when judging model quality. Using archived output from 159 models spanning five...</span><span class="abstract-full" id="full-2610.23216" hidden>Earth system models underpin the evidence that informs net-zero policy, yet their energy demands are rarely considered when judging model quality. Using archived output from 159 models spanning five generations of the Coupled Model Intercomparison Project (CMIP), we show that model error has fallen at a constant rate of 13.5 per cent per doubling of compute since 1995. Halving CMIP6 error would require 28 times more compute. Improvements are uneven, with sea-level pressure and surface air temperature errors declining 3.5-fold and 2.7-fold since CMIP1, but precipitation error by only 7 per cent. Machine-learning weather models match or exceed operational forecast skill at a fraction of the runtime, but none publishes measured energy, and including training changes how they rank. We propose a skill-energy framework that combines an empirical Pareto front with a ranking score. These findings argue for standardised reporting of energy use and performance ahead of CMIP8.</span> <span class="abstract-toggle" data-id="2610.23216">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.23216v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.23216v1) · [:material-content-copy: BibTeX](bibtex/2610.23216.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=evaluation" data-tag="evaluation">Evaluation</a> <a class="md-tag" href="/explore/?t=energy" data-tag="energy">Energy</a>
    { .paper-tags }

-   #### Network-Wide Freeway Speed Nowcasting and Forecasting from Sparse Fixed Sensors with a Probe-Supervised Graph Convolutional Recurrent Network { #2610.22630 }

    *Ran Tu, Zihao Li, Arash Mirzaei* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.22630">Traffic agencies need current and near-future speeds on every link of their network, but their fixed sensors cover only part of it. We study joint nowcasting and next-hour forecasting of hourly...</span><span class="abstract-full" id="full-2610.22630" hidden>Traffic agencies need current and near-future speeds on every link of their network, but their fixed sensors cover only part of it. We study joint nowcasting and next-hour forecasting of hourly speeds on all 4{,}127 directional freeway links of the Dallas--Fort Worth region from sensors on 643 links (15.6%). Historical probe-vehicle data supervise every link in training; only sensor readings are used at inference. We call this setting probe-supervised sparse sensing. We propose GCRN+, a graph convolutional recurrent network that predicts deviations from a historical speed profile. It combines a parameter-free, direction-aware kernel interpolation of sensor anomalies, used as input features and as an output baseline, with a diffusion-convolutional GRU on the directed road graph, a low-rank adaptive graph and learnable link embeddings. It scales linearly with network size and trains on a CPU. Sensor readings are emulated from probe speeds, which isolates spatio-temporal inference from detector calibration. On four held-out test days, GCRN+ attains an RMSE of 2.21~mph for the current hour and 2.39~mph for the next hour on unsensored links, against 3.23~mph for the historical average. The best baselines that use the same interpolation features reach 2.53 and 2.75~mph. Adapted recent models without these features (STID, STAEformer, ImputeFormer, MoGERNN) reach at best 2.54 and 2.53~mph; this comparison is between complete pipelines rather than architectures. GCRN+ is best in every distance band except links not connected to any sensor through the road graph. Ablations within one recurrent backbone attribute most of the gain to the link embeddings and the adaptive graph, which can be trained because probe labels exist for every link. For congestion detection on unsensored links, recall rises from 77% (historical average) and 86% (best competitor) to 89% at aprecision of 0.94.</span> <span class="abstract-toggle" data-id="2610.22630">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.22630v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.22630v1) · [:material-content-copy: BibTeX](bibtex/2610.22630.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=hourly" data-tag="hourly">Hourly</a>
    { .paper-tags }

-   #### Atmospheric Chemistry-Climate Modeling with an AI Foundation Model { #2610.22555 }

    *James Weber, Makoto Kelp* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.22555">Atmospheric composition and chemistry-climate interactions are central to predicting future climate change. Yet simulating chemical and aerosol processes is typically the most expensive component of...</span><span class="abstract-full" id="full-2610.22555" hidden>Atmospheric composition and chemistry-climate interactions are central to predicting future climate change. Yet simulating chemical and aerosol processes is typically the most expensive component of an atmospheric model and is often highly simplified or prescribed entirely. AI weather forecasting foundation models have shown skill rivalling process-based models and are increasingly fine-tuned for new atmospheric tasks. Here, we present the first foundation model to capture interactions between emissions, atmospheric composition, and radiation. We fine-tune the Aurora Earth system foundation model on simulations from the chemistry-climate model UKCA. Our model reproduces daily meteorology, composition, aerosol, and radiation fields of the coupled system with 23 of 27 predicted fields achieving $R^2$ above 0.85 in both the troposphere and stratosphere. It reproduces ozone, CO, and temperature with annual mean biases below 4% and the downwelling shortwave and longwave fluxes to within 6%. Our model remains stable in autoregressive simulation over 50 years while running ~$100\times$ faster than UKCA. By training on paired emissions scenarios, our model learns responses to perturbations by capturing tropospheric ozone's response to changes in NO and CO emissions and surface radiation's response to aerosol. We further show that incorporating MLS satellite ozone retrievals into training corrects biases in the underlying UKCA fields, reducing zonal mean ozone error against collocated observations by 23% between 250 and 20 hPa. Our approach does not suggest process-based modeling can be replaced but demonstrates that a foundation model can be adapted to respond to forcings and learn from observations at a relatively small computational cost.</span> <span class="abstract-toggle" data-id="2610.22555">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.22555v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.22555v1) · [:material-content-copy: BibTeX](bibtex/2610.22555.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=foundation-models" data-tag="foundation-models">Foundation models</a> <a class="md-tag" href="/explore/?t=daily" data-tag="daily">Daily</a>
    { .paper-tags }

-   #### Can AI Forecast the Extremes? Evaluating Probabilistic Weather Models on Tropical Cyclones { #2610.22077 }

    *Nathan Chalumeau, Xavier Renard, Remi Meynadier, Hugo Rakotoarimanga, Marcin Detyniecki et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.22077">Weather prediction is undergoing an important shift, with data-driven AI models outperforming leading numerical weather prediction systems on average at a fraction of the computational cost. Yet,...</span><span class="abstract-full" id="full-2610.22077" hidden>Weather prediction is undergoing an important shift, with data-driven AI models outperforming leading numerical weather prediction systems on average at a fraction of the computational cost. Yet, whether this success extends to extreme events remains contested. Earlier evaluations have exposed important shortcomings, but have largely focused on deterministic models and metrics, leaving the capabilities of the latest probabilistic models less understood. We conduct an extensive benchmark of tropical cyclone forecasting, comparing GenCast, FourCastNet 3, and FGN with ECMWF's operational physics-based ensemble, IFS-ENS. We develop a suite of probabilistic metrics and diagnostic analyses to assess cyclone tracking, intensity biases, uncertainty, and the wind-pressure relationship that characterizes storm structure. We find that, despite remaining limitations in intensity and ensemble spread, leading AI models substantially outperform IFS-ENS in cyclone tracking: FGN achieves better probabilistic location scores for 90% of evaluated storms in 2025. Further analyses reveal model-specific failure modes and suggest a lingering influence of pretraining data, pointing to opportunities for targeted improvement. Nonetheless, these results suggest that AI models are capable of capturing physical relationships that remain predictive well beyond typical weather conditions, motivating further investigation into how these capabilities emerge and how to strengthen them.</span> <span class="abstract-toggle" data-id="2610.22077">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.22077v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.22077v1) · [:material-content-copy: BibTeX](bibtex/2610.22077.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### Digital Typhoon Dataset V3: Benchmarking for Multi-spectral Extension and Analyzing Systematic Bias in Intensity Regression { #2610.21654 }

    *Asanobu Kitamoto, Victor Le Louarn, Noelia Falczuk, Ngoc Anh Tong, Anja Veleva* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.21654">This paper presents the Digital Typhoon Dataset V3, a new version of the longest typhoon satellite image dataset spanning nearly 50 years, designed to benchmark machine learning models on long-term...</span><span class="abstract-full" id="full-2610.21654" hidden>This paper presents the Digital Typhoon Dataset V3, a new version of the longest typhoon satellite image dataset spanning nearly 50 years, designed to benchmark machine learning models on long-term spatio-temporal data. The latest addition to Dataset V3 is multispectral satellite data, enabling multi-channel models. We investigate data-splitting strategies and present benchmarking results for the Northern and Southern Hemispheres. Experimental results show that, under the current dataset construction and experimental setting, multi-channel inputs do not clearly outperform single-channel inputs for pressure regression. Among the evaluated model architectures, image types, and spectral channels, ResNet-18, full-resolution images, and Channel 1 generally provide the best single-channel performance for pressure regression. Performance also differs between the WP and AU datasets by regression target, with lower RMSE on WP for pressure regression and on AU for wind regression. These results highlight the importance of considering dataset-specific and task-specific characteristics when interpreting benchmark performance. Additionally, we analyzed systematic bias in intensity regression, which leads to underestimation of intensity for strong typhoons. We tested imbalance-aware weighting strategies, including inverse-density weighting inspired by BalancedMSE, and showed that these approaches are not sufficient to remove the bias. The dataset is publicly available at http://agora.ex.nii.ac.jp/digital-typhoon/dataset/ and https://github.com/kitamoto-lab/digital-typhoon/.</span> <span class="abstract-toggle" data-id="2610.21654">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.21654v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.21654v1) · [:fontawesome-brands-github: Code](https://github.com/kitamoto-lab/digital-typhoon) · [:material-content-copy: BibTeX](bibtex/2610.21654.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a> <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a> <a class="md-tag" href="/explore/?t=benchmarks-datasets" data-tag="benchmarks-datasets">Benchmarks & datasets</a>
    { .paper-tags }

-   #### Energy Flow Shapes the Decomposition Dynamics of a Reactive Atmospheric Intermediate { #2610.21259 }

    *Mitradip Das, Cangtao Yin, M. Meuwly* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.21259">The unimolecular dynamics of Criegee intermediates depend strongly on their internal energy, which can differ substantially between nascent atmospheric species and laboratory-prepared reactants....</span><span class="abstract-full" id="full-2610.21259" hidden>The unimolecular dynamics of Criegee intermediates depend strongly on their internal energy, which can differ substantially between nascent atmospheric species and laboratory-prepared reactants. Here, the decomposition dynamics of <em>anti-</em>CH$_3$CHOO are investigated using molecular dynamics simulations on a neural network-based potential energy surface under thermal and mode-selective vibrational excitation. Simulations were performed at internal energies representative of those released during ozonolysis (chemical/thermal preparation, TP) or following vibrational excitation (VibEx) along the CH-stretching mode with two or three quanta, as can be done in laboratory studies. These excitation conditions strongly affect reaction yields and kinetics, with VibEx promoting passage across the initial cyclization barrier. For TP, 35 kcal/mol of internal energy are required to form molecular products, whereas for VibEx, 27 kcal/mol are sufficient. Although TP and VibEx ensembles access largely the same fragmentation pathways with comparable relative yields, the absolute product yields, characteristic timescales, and kinetic behavior differ substantially. Methyldioxirane is found to be a kinetic trap along the pathway. The resulting molecular-level description of <em>anti-</em>CH$_3$CHOO decomposition provides guidance for interpreting experiments and contributes to a broader understanding of energy-dependent Criegee-intermediate dynamics.</span> <span class="abstract-toggle" data-id="2610.21259">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.21259v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.21259v1) · [:material-content-copy: BibTeX](bibtex/2610.21259.bib){ .bibtex-link }
    { .paper-links }

-   #### Variable Masking for Climate Emulation in ACE { #2610.21012 }

    *Alexey Yermakov, Troy Arcomano, Jeremy McGibbon, Brian Henn, Oliver Watt-Meyer* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.21012">In the recent AI Model Intercomparison Project (AIMIP) benchmark, the ACE2.1-ERA5 emulator was shown to underestimate the true near-surface warming trend both during the training period and the...</span><span class="abstract-full" id="full-2610.21012" hidden>In the recent AI Model Intercomparison Project (AIMIP) benchmark, the ACE2.1-ERA5 emulator was shown to underestimate the true near-surface warming trend both during the training period and the out-of-sample extrapolation period. In this work, we present M-ACE2S (Masked-ACE2S), a modified version of the stochastic ACE2S model, which uses variable masking during training and reduces the reliance of M-ACE2S on any single input channel. We show that M-ACE2S improves long-term near-surface air temperature prediction. We also present a novel data normalization strategy called "global mean removal", which, combined with masking, further improves the near-surface temperature forecasts. We show that M-ACE2S outperforms the ACE2S baseline on in-sample data for ERA5 while improving training stability and generalizability of near-surface temperature to unseen forcing when paired with global mean removal.</span> <span class="abstract-toggle" data-id="2610.21012">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.21012v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.21012v1) · [:fontawesome-brands-github: Code](https://github.com/ai2cm/ace) · [:material-content-copy: BibTeX](bibtex/2610.21012.bib){ .bibtex-link }
    { .paper-links }

-   #### Forecast lead time and the timing of crew deployment for weather-driven power outages using AI weather models { #2610.20918 }

    *Mohammad Kian Golkar, Luciano Alves de Oliveira, Mohamad Alipour* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.20918">Utilities request mutual aid crews to restore power after severe storms, and requesting early secures more crews but relies on less accurate forecasts. We examined how forecast lead time affects the...</span><span class="abstract-full" id="full-2610.20918" hidden>Utilities request mutual aid crews to restore power after severe storms, and requesting early secures more crews but relies on less accurate forecasts. We examined how forecast lead time affects the value of outage predictions for crew deployment, using operational forecasts from two AI weather models, GraphCast and FourCastNet, to predict outage occurrence for counties across the contiguous United States 24 to 120 hours ahead. Forecasts were evaluated by the share of outage burden captured by the counties where crews would be staged. Captured burden declined with lead time: the loss between three and five days was significant in all analyses, whereas the loss within three days was significant only for outages that had not yet started. Early deployment with all requested crews outperformed late deployment with fewer, and we report the availability ratio at which the two break even. Antecedent land surface predictors computed at forecast valid time include weather from the forecast period, which inflates their skill by 42 to 53% and reverses their apparent advantage over forecast meteorology at long lead times. Computed correctly, they improved occurrence skill by 17.5% at 24 hours but did not significantly change captured burden, a gap that attribution analysis linked to their static component. Prediction intervals covered only 0.476 of tropical cyclone events against a nominal 0.90. Calibration by hazard class corrected most of this shortfall at an 11% cost in interval width, as a diagnostic only, because hazard class is known only after the event.</span> <span class="abstract-toggle" data-id="2610.20918">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.20918v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.20918v1) · [:material-content-copy: BibTeX](bibtex/2610.20918.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a>
    { .paper-tags }

-   #### Improved Precipitation Variability in the Community Atmosphere Model through Implementation of a Neural-Network Deep Convection Scheme { #2610.20405 }

    *Xavier J. Levine, Griffin Mooers, Jack W. Atkinson, Marat F. Khairoutdinov, Janni Yuval et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.20405">Conventional deep convection schemes remain a primary source of uncertainty in global climate models, yet replacing them with data-driven alternatives while maintaining physical consistency and...</span><span class="abstract-full" id="full-2610.20405" hidden>Conventional deep convection schemes remain a primary source of uncertainty in global climate models, yet replacing them with data-driven alternatives while maintaining physical consistency and numerical stability is an open challenge. We present a hybrid configuration of the Community Atmosphere Model version 6 (CAM6), in which a fully connected deep neural network replaces the Zhang-McFarlane deep convection scheme (ZM), while radiation, turbulence, and other cloud processes are left unchanged. The network is trained on output from a storm-resolving global model - the global System for Atmospheric Modeling (gSAM) - using a coarse-graining approach to derive subgrid-scale convective fluxes and tendencies. This data-driven parameterization is ported to CAM6 and found to run stably for more than a decade while conserving energy and moisture. It produces a broadly realistic climatology without retuning the remaining parameterizations. Relative to the default Zhang-McFarlane scheme, the neural network parameterization substantially improves some long-standing model deficiencies. In particular, the biases in light precipitation and extreme precipitation are markedly reduced, and the diurnal cycle of tropical land precipitation shifts from a spurious midday peak over many regions to the observed late-afternoon peak. These results support targeted ML-based convection schemes as a viable path toward reducing systematic model errors in next-generation climate models.</span> <span class="abstract-toggle" data-id="2610.20405">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.20405v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.20405v1) · [:material-content-copy: BibTeX](bibtex/2610.20405.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a> <a class="md-tag" href="/explore/?t=km-scale" data-tag="km-scale">Km-scale</a>
    { .paper-tags }

-   #### PrecipCorrDiff: A Deep Learning Framework for Generating 0.0625$^\circ$ High-Resolution IMD Gridded Precipitation Data { #2610.19994 }

    *Bipin Kumar, Rajib Chattopadhyay, Harsh P. Bachhav,  Jannu, Bhupendra Bahadur Singh, Ramesh Ragala et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.19994">High-resolution precipitation datasets are essential for weather forecasting, hydrological modeling, and climate risk assessment, yet observation-based gridded products often lack the spatial detail...</span><span class="abstract-full" id="full-2610.19994" hidden>High-resolution precipitation datasets are essential for weather forecasting, hydrological modeling, and climate risk assessment, yet observation-based gridded products often lack the spatial detail required for regional applications. Here we present PrecipCorrDiff, a diffusion-based downscaling framework built by modifying the original corrector diffusion based archetecture that incorporates a customized loss function and a multi-scale residual generation strategy to improve the precipitation especially the representation of extremes. PrecipCorrDiff generates 0.0625° resolution India Meteorological Department (IMD) precipitation fields from 0.25° gridded data while preserving precipitation statistics and reconstructing fine-scale spatial variability. Validation against independent IMD station observations, comparing the proposed framework with the SRGAN-based model and the original CorrDiff model, indicates that the proposed framework substantially improves the representation of spatial precipitation patterns, rainfall intensity distributions, and extreme rainfall events, while eliminating the systematic negative bias present in the previously developed SRGAN-based model. Across four climatologically distinct homogeneous Indian monsoon regions, PrecipCorrDiff reproduces observed precipitation characteristics with near-zero spatial bias, demonstrating the potential of diffusion-based downscaling for generating physically consistent, high-resolution precipitation datasets for hydrological, meteorological, and climate applications.</span> <span class="abstract-toggle" data-id="2610.19994">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.19994v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.19994v1) · [:material-content-copy: BibTeX](bibtex/2610.19994.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=quarter-degree" data-tag="quarter-degree">0.25°</a>
    { .paper-tags }

-   #### Generative modelling for multivariate spatio-temporal downscaling of monthly climate fields { #2610.19885 }

    *Yu Huang, Shangshang Yang, Sebastian Bathiany, Michael Aich, Philipp Hess, Niklas Boers* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.19885">Inferring climate fields at higher spatio-temporal resolutions from coarse representations is essential for accessing socioeconomic and environmental impacts. The downscaling of modelled climate...</span><span class="abstract-full" id="full-2610.19885" hidden>Inferring climate fields at higher spatio-temporal resolutions from coarse representations is essential for accessing socioeconomic and environmental impacts. The downscaling of modelled climate fields faces a trade-off between data storage constraints, simulation efficiency, and resolution requirements. While many methods for spatial downscaling have been proposed, joint spatio-temporal downscaling has received comparable little attention. When deriving, for example, daily fields from climate datasets that are available only as monthly averages at coarser spatial resolution, it is highly challenging to faithfully reproduce extreme event statistics and spatiotemporal consistency. To address this task, we introduce a generative machine learning approach, using a diffusion model to jointly downscale monthly temperature and precipitation fields into temporally consistent daily fields at higher spatial resolution. We demonstrate that our approach can generate high-dimensional multivariate climate fields at daily temporal and 0.25° spatial resolution from monthly input fields at 1° spatial resolution, exhibiting coherent temporal dynamics and physically plausible spatial structures. In particular, by incorporating a conditional guidance technique, we show that a pre-trained diffusion model can be directly leveraged to efficiently generate daily high-resolution fields conditioned on monthly inputs, without requiring additional fine-tuning or retraining. Our probabilistic approach thus provides an efficient and generalizable framework for spatio-temporal climate downscaling, enabling flexible climate fields generation for climate impact assessments and digital Earth simulator constructions.</span> <span class="abstract-toggle" data-id="2610.19885">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.19885v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.19885v1) · [:material-content-copy: BibTeX](bibtex/2610.19885.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=quarter-degree" data-tag="quarter-degree">0.25°</a> <a class="md-tag" href="/explore/?t=coarse" data-tag="coarse">Coarse (≥1°)</a> <a class="md-tag" href="/explore/?t=daily" data-tag="daily">Daily</a> <a class="md-tag" href="/explore/?t=monthly" data-tag="monthly">Monthly</a>
    { .paper-tags }

-   #### Unmasking Compensating Errors in Numerical Weather Models: A Process-Additive Neural Network Approach { #2610.19772 }

    *Yuiko Ichikawa, Edward Groot, Jack Atkinson* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.19772">This work introduces a Process Additive Neural Network (PANN) framework to disentangle compensating biases in atmospheric models, where errors in different physical processes can offset each other...</span><span class="abstract-full" id="full-2610.19772" hidden>This work introduces a Process Additive Neural Network (PANN) framework to disentangle compensating biases in atmospheric models, where errors in different physical processes can offset each other and obscure their origins. Using a single-column model, each PANN branch represents the tendency error associated with a separate physical process, and the summed branch outputs approximate the total error relative to a reference tendency. Once trained successfully, the branch outputs are expected to provide process-level error attributions and, when averaged, estimates of process-level bias. Initial experiments focus on cloud-related processes in OpenIFS SCM case studies, where cloud and convection tendencies are strongly coupled and compensation is difficult to diagnose using standard approaches. The aim is not to replace human interpretation of physical processes, but to extend it to regimes where model complexity, process interaction, and compensating errors make direct diagnosis difficult. By providing a physically structured form of machine-learning interpretability, PANN can help connect complex model behaviour with process-level explanations. This work supports both improved understanding of compensating errors and the development of more physically consistent weather and climate models.</span> <span class="abstract-toggle" data-id="2610.19772">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.19772v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.19772v1) · [:material-content-copy: BibTeX](bibtex/2610.19772.bib){ .bibtex-link }
    { .paper-links }

</div>

