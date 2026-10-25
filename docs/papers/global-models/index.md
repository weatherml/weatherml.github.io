---
title: 'Global Models'
hide:
  - toc
---

<div class="listing-header" markdown>

# Global Models

<p class="page-meta" markdown="span">378 papers · page 1 of 13 · <a href="../../bib/global-models.bib" download>:material-download: BibTeX for this topic</a></p>

</div>

<div class="grid cards" markdown>

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

-   #### Global Evaluation of AI and NWP Precipitation Forecasts During Atmospheric River Events { #2610.03758 }

    *Marina Vicens-Miquel, Taylor Mandelbaum, Amy McGovern, Aaron J. Hill, Daniel Rothenberg* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.03758">Atmospheric rivers (ARs) produce many of the world's most extreme precipitation events and hydrometeorological hazards. Although artificial intelligence weather prediction (AIWP) models have...</span><span class="abstract-full" id="full-2610.03758" hidden>Atmospheric rivers (ARs) produce many of the world's most extreme precipitation events and hydrometeorological hazards. Although artificial intelligence weather prediction (AIWP) models have demonstrated skill comparable to or exceeding numerical weather prediction (NWP) systems for large-scale atmospheric variables, their ability to forecast AR-related precipitation remains insufficiently characterized globally. Here, we evaluate 24-hour precipitation forecasts from the Global Forecast System (GFS), Global Ensemble Forecast System (GEFS), GraphCast, and Artificial Intelligence Forecasting System (AIFS) from Day 1 through Day 10 globally and across North America, Europe, and Australia and New Zealand. Using the Extreme Weather Bench framework, forecasts are evaluated against Integrated Multi-satellitE Retrievals for GPM (IMERG) observations using measures of precipitation magnitude, spatial structure, and localization. GraphCast and AIFS exhibit greater spatial skill than GFS and GEFS, particularly for heavy precipitation and at longer lead times, and better preserve the spatial organization of AR-related precipitation through Day 10. However, this improved spatial skill does not translate into accurate precipitation magnitudes. AIWP models tend to overpredict moderate-to-heavy accumulations while underpredicting the heaviest precipitation at longer lead times, whereas NWP systems develop pronounced dry biases. These results reveal distinct strengths and limitations of AIWP for high-impact precipitation forecasting and provide a reproducible benchmark.</span> <span class="abstract-toggle" data-id="2610.03758">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.03758v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.03758v1) · [:material-content-copy: BibTeX](../../bibtex/2610.03758.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=extremes" data-tag="extremes">Extremes</a> <a class="md-tag" href="/explore/?t=benchmarks-datasets" data-tag="benchmarks-datasets">Benchmarks & datasets</a> <a class="md-tag" href="/explore/?t=evaluation" data-tag="evaluation">Evaluation</a> <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a>
    { .paper-tags }

-   #### SDECast: Probabilistic Weather Forecasting in Continuous Time with Neural SDEs { #2610.03313 }

    *Maria Marchenko, Martin Andrae, Fredrik Lindsten, Christian A. Naesseth* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.03313">Existing machine learning weather forecasting models typically generate forecasts through autoregressive rollouts at a fixed temporal resolution. While highly efficient for long-range prediction,...</span><span class="abstract-full" id="full-2610.03313" hidden>Existing machine learning weather forecasting models typically generate forecasts through autoregressive rollouts at a fixed temporal resolution. While highly efficient for long-range prediction, this formulation can suffer from severe error accumulation when used with shorter time steps and does not explicitly encode the locality and temporal continuity of atmospheric dynamics. To address these limitations, we introduce **SDECast**, a Neural Stochastic Differential Equation (SDE) framework for continuous-time probabilistic weather forecasting. SDECast extends SDE Matching to learn stochastic dynamics directly in physical space, without requiring repeated SDE simulation during training. On a simulated geophysical flow, we show that SDECast recovers meaningful drift dynamics and faithfully reproduces the underlying continuous-time behavior. We then demonstrate its scalability to global weather forecasting at hourly resolution, where SDECast produces skillful probabilistic forecasts for lead times of up to five days.</span> <span class="abstract-toggle" data-id="2610.03313">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.03313v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.03313v1) · [:material-content-copy: BibTeX](../../bibtex/2610.03313.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a> <a class="md-tag" href="/explore/?t=hourly" data-tag="hourly">Hourly</a>
    { .paper-tags }

-   #### Post-Training Quantization of Autoregressive Weather Models { #2610.02511 }

    *Ananyo Bhattacharya, Swastik Bhattacharya, Christiane Jablonowski* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.02511">Advancements in high-resolution numerical weather prediction (NWP) and data assimilation (DA) have shaped the developments in deep learning (DL) architectures emulating atmospheric dynamics....</span><span class="abstract-full" id="full-2610.02511" hidden>Advancements in high-resolution numerical weather prediction (NWP) and data assimilation (DA) have shaped the developments in deep learning (DL) architectures emulating atmospheric dynamics. Emulators for weather forecasting exhibit forecast quality comparable to physics based models at forecast horizon scaling from few days to subseasonal time scales. The emulators are driven by hardware-accelerated matrix multiplication in autoregressive inferences, significantly reducing the computation time and resources required for NWP. Optimization of the matrix multiplication processes in GPU architectures provides opportunities to scale towards high-resolution domain, and offers implementation of out of the box solutions. Post-training quantization (PTQ) has been demonstrated across multiple DL architectures to accelerate and increase the number of computations in unit time while consuming less power, enabling applications on edge hardware. In this study, we investigate the effect of PTQ on pre-trained AI emulators for global-scale weather forecasting. We implement PTQ algorithms in Deep Learning Weather Prediction (DLWP) and FourCastNet (FCN) models as a proof of concept for geophysical fluid dynamics applications. We systematically investigate the effect of PTQ on emulator inferences over short-range forecast horizons. Evaluation of PTQ configurations using simulated quantization hints at qualitatively meaningful forecasts over short-time horizons. These results provide a first benchmark of PTQ for autoregressive weather emulators and a basis for quantization-based optimization of DL models for dynamical systems.</span> <span class="abstract-toggle" data-id="2610.02511">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.02511v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.02511v1) · [:material-content-copy: BibTeX](../../bibtex/2610.02511.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=subseasonal-to-seasonal" data-tag="subseasonal-to-seasonal">Subseasonal to seasonal</a> <a class="md-tag" href="/explore/?t=benchmarks-datasets" data-tag="benchmarks-datasets">Benchmarks & datasets</a>
    { .paper-tags }

-   #### Weather Jiu-Jitsu: Exploring the Feasibility of Control Paradigms in Weather Foundation Models { #2610.00792 }

    *Prakriti Biswas, Kobi Abayomi, Upmanu Lall* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.00792">Weather Jiu-Jitsu is a control paradigm for extreme climatological events, inspired by chaos theory. As a proposition, small, precise, targeted, and cost-inexpensive perturbations can redirect...</span><span class="abstract-full" id="full-2610.00792" hidden>Weather Jiu-Jitsu is a control paradigm for extreme climatological events, inspired by chaos theory. As a proposition, small, precise, targeted, and cost-inexpensive perturbations can redirect trajectories of a large dynamical system. This strategy has been demonstrated analytically in the Lorenz-63 system, where a naturally chaotic trajectory switching between two attractors can be confined to a single attractor, indefinitely, via arbitrarily small perturbations. This paper examines the feasibility of Microsoft's Aurora -- a 1.3 billion parameter global atmospheric model -- as a test bed for this strategy. This paper explores three questions: (1) Is Aurora a reliable enough simulation environment to serve as a meaningful testbed? (2) Are the perturbations required to redirect its trajectories small enough to be physically plausible? (3) Does Aurora's learned latent space (the parametric estimators on climatological attributes) yield any apparent, structured, and/or perhaps interpretable features that can convey a geo/atmospheric response to initial conditions? We find evidence consistent with all three: Aurora's modeled trajectories respond to perturbations beyond measurement drift, the perturbation magnitudes required are small relative to the model's own forecast uncertainty, and its latent representations exhibit directional structure that responds to Jiu-Jitsu-type interventions, even though that structure does not separate extreme from normal states outright. These results should be read as feasibility diagnostics rather than a demonstration of control: we do not implement or test an actual steering intervention on Aurora, and several of our findings, particularly around the model's latent-space geometry, are exploratory.</span> <span class="abstract-toggle" data-id="2610.00792">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.00792v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.00792v1) · [:material-content-copy: BibTeX](../../bibtex/2610.00792.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=foundation-models" data-tag="foundation-models">Foundation models</a> <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a>
    { .paper-tags }

-   #### STCFormer: Adaptive Spatio-Temporal Modeling with Dynamic Cluster Transformer for Station-based Weather Forecasting { #2610.00377 }

    *Rongwen Li, Haixin Xie, Mingyang Wang, Hongwu Liu, Kun Fang, Changjian Chen, Zhuo Tang, Kenli Li* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.00377">Station-based weather forecasting supports daily life and economic activity, yet accurate forecasts require modeling complex spatial dependencies among stations. Recent clustering-based selective...</span><span class="abstract-full" id="full-2610.00377" hidden>Station-based weather forecasting supports daily life and economic activity, yet accurate forecasts require modeling complex spatial dependencies among stations. Recent clustering-based selective modeling offers a promising alternative to dense inter-station interactions. However, a grouping shared across an observation window may obscure local changes in station relationships, while intra-cluster interactions alone may miss important global context. The theoretical advantages of selective interactions over dense connectivity also remain insufficiently understood. We therefore propose STCFormer, an adaptive spatio-temporal Transformer that dynamically groups stations according to their local evolution within each temporal patch. Its Cluster-Guided Attention Block combines fine-grained local attention within clusters and global attention over regional state summaries, allowing each station to access information beyond its own cluster. We further show that a derived Lipschitz upper bound for cluster-conditioned local attention is no larger than its fully connected counterpart, explaining a potential robustness benefit and motivating the design of InfoLoss. Experiments on three real-world weather datasets spanning eight temperature and wind forecasting tasks show that STCFormer achieves the lowest 24-hour mean squared error on all eight tasks and ranks first or second in 47 of 48 comparisons across metrics and forecasting horizons. Ablations and case studies further confirm the benefits of locally adaptive grouping and complementary local-global interactions. Our code can be obtained at https://github.com/hnu-vis/STCFormer.</span> <span class="abstract-toggle" data-id="2610.00377">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.00377v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.00377v1) · [:fontawesome-brands-github: Code](https://github.com/hnu-vis/STCFormer) · [:material-content-copy: BibTeX](../../bibtex/2610.00377.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=transformers" data-tag="transformers">Transformers</a> <a class="md-tag" href="/explore/?t=station-point" data-tag="station-point">Station / point</a> <a class="md-tag" href="/explore/?t=daily" data-tag="daily">Daily</a>
    { .paper-tags }

-   #### Butterfly Effect Confirmed in Global AI Weather Models: Evidence from Tropical Cyclone Forecasting { #2609.39379 }

    *Jeremy Cheuk-Hin Leung, Daosheng Xu, Weiye Yu, Shaojing Zhang, Xiaodong Zeng, Gaozhen Nie, Jie Feng et al.* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.39379">A paradox recently emerged in artificial intelligence (AI) weather prediction research. While some claim AI weather models cannot simulate atmospheric butterfly effect, this conflicts with AI models'...</span><span class="abstract-full" id="full-2609.39379" hidden>A paradox recently emerged in artificial intelligence (AI) weather prediction research. While some claim AI weather models cannot simulate atmospheric butterfly effect, this conflicts with AI models' limited predictability and advances in AI ensemble forecasting. This study demonstrates via counterexamples that the butterfly effect does exist in AI weather predictions. For Super Typhoon Khanun, AI predictions are constrained by a double-attractor system. Minor initial perturbations confined to two regions trigger state transitions between two local attractors, causing a 1006-km difference in the predicted storm position on Day 7. This behavior is consistent with numerical weather prediction models and observed in ~12% of tropical cyclones in the past 5 years. These findings verify AI's ability to capture atmospheric chaos and provide the physical basis for AI ensemble forecasting.</span> <span class="abstract-toggle" data-id="2609.39379">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.39379v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.39379v1) · [:material-content-copy: BibTeX](../../bibtex/2609.39379.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### Proper Scoring Rule-based Diffusion for Probabilistic Weather Forecasting { #2609.38632 }

    *Joonhyeong Park, Giung Nam, Hyungi Lee, Kyunghyun Cho, Byoungwoo Park, Juho Lee* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.38632">Recent probabilistic weather forecasters train stochastic predictors with the continuous ranked probability score (CRPS) to generate each ensemble member in a single forward pass. These models learn...</span><span class="abstract-full" id="full-2609.38632" hidden>Recent probabilistic weather forecasters train stochastic predictors with the continuous ranked probability score (CRPS) to generate each ensemble member in a single forward pass. These models learn the predictive distribution from the forecast context alone, which becomes difficult at longer forecast horizons where uncertainty is high. To learn the predictive distribution more effectively, we introduce auxiliary conditional denoising tasks that predict the same future state from the context and its corrupted version, which provides partial future information that can reduce prediction ambiguity. Building on distributional diffusion models, we learn the conditional distributions of these tasks with a single stochastic predictor by minimizing a proper scoring rule across noise levels. At inference, the predictor can still generate each ensemble member in a single forward pass at the fully corrupted endpoint. Standard CRPS training is recovered as the endpoint-only special case of our formulation, so our framework extends existing CRPS-based forecasters with only additional conditioning inputs. Controlled experiments show that the auxiliary tasks improve one-step forecasting across architectures, with larger gains at longer forecast horizons. The gains extend to high-dimensional global weather forecasting under both training from scratch and fine-tuning, along with improved calibration and potential benefits for generalization under distribution shift.</span> <span class="abstract-toggle" data-id="2609.38632">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.38632v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.38632v1) · [:material-content-copy: BibTeX](../../bibtex/2609.38632.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a>
    { .paper-tags }

-   #### An Input-Frugal Deep Learning Framework for Weather-Driven National Crop-Yield Forecasting: A Case Study of Brazilian Soybean { #2609.38447 }

    *Fernando Dupin da Cunha Mello, Prashant Kumar, Erick G. Sperandio Nascimento* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.38447">Reliable, timely crop-yield forecasts are essential for market stability and risk management, yet many approaches rely on costly or hard-to-scale inputs. We present a frugal, transferable, and...</span><span class="abstract-full" id="full-2609.38447" hidden>Reliable, timely crop-yield forecasts are essential for market stability and risk management, yet many approaches rely on costly or hard-to-scale inputs. We present a frugal, transferable, and architecture-agnostic deep learning framework that uses routine weather as the only time-varying input plus two lightweight static context inputs (crop year and an agro-environmental label) to capture long-run change and regional heterogeneity, while supporting multiple sequence encoders under identical data requirements. Using a 20-season Brazilian soybean case study (2001/02-2020/21) with leave-one-year-out cross-validation, we benchmark MLP, CNN, LSTM, CNN-LSTM, a Transformer encoder and the Mamba state-space model against linear ridge regression and a five-year moving-average "farmer" baseline. All deep learning variants outperform ridge, and all sequential encoders surpass the non-sequential MLP. The Transformer achieves the best national accuracy (RMSE 149 kg ha^-1; rRMSE 5.3%; R^2 = 0.784), reducing error by 47.6% relative to the farmer baseline. In-season forecasts improve monotonically from early- to late-season issuance, reaching approximately 50% lower error than the baseline at the latest forecast point. Ablations indicate that the agro-environmental label and spatial instance expansion (multiple grid-node weather sequences per municipality-year) contribute positively without increasing input complexity. SHAP diagnostics suggest crop year explains most of the long-run trajectory, whereas within-season weather and agro-environmental context primarily drive interannual deviations, with moisture/cloud and thermal-demand variables dominating. Overall, the framework is straightforward to deploy across other crops and geographic regions and is naturally compatible with operational weather forecasts for routine monitoring.</span> <span class="abstract-toggle" data-id="2609.38447">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.38447v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.38447v1) · [:material-content-copy: BibTeX](../../bibtex/2609.38447.bib){ .bibtex-link }
    { .paper-links }

-   #### A neural network-based Universal Thermal Climate Index for reliable global thermal-stress classification across extreme weather { #2609.35949 }

    *Bikem Pastine, Milan Klöwer, Tianning Tang, Sarah Wilson Kemsley, Louise Slater* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.35949">Extreme temperatures are the leading cause of climate-related mortality world-wide. Climate-health research and operational weather forecasting require accurate estimates of human thermal stress. The...</span><span class="abstract-full" id="full-2609.35949" hidden>Extreme temperatures are the leading cause of climate-related mortality world-wide. Climate-health research and operational weather forecasting require accurate estimates of human thermal stress. The Universal Thermal Climate Index (UTCI) is among the most sophisticated and widely used feels-like temperature metrics. However, its ubiquitous polynomial approximation does not generalize well to extreme weather conditions. Here, we introduce Neural-UTCI, a neural network that calculates UTCI with substantially higher accuracy across global conditions at a lower computational cost for operational use. Neural-UTCI reduces the polynomial approximation RMSE from 2.78°C to 0.36 °C, an 87% improvement, and lowers thermal stress misclassification rates from 5.3% to 1.7%, with consistent performance across resampling experiments. These differences affect thermal exposure metrics. For example, during the 2003 European heatwave summer in Rome, Italy, the number of very strong heat stress days increases from 15 to 35 days when using Neural-UTCI compared to operational products like ERA5-HEAT. Simultaneously, Neural-UTCI reliably classifies extreme cold stress conditions, allowing continuous global application. By improving UTCI accuracy, Neural-UTCI can strengthen climate-health risk assessments and public weather warning systems, especially as global warming increases the incidence of extreme events.</span> <span class="abstract-toggle" data-id="2609.35949">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.35949v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.35949v1) · [:material-content-copy: BibTeX](../../bibtex/2609.35949.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=extremes" data-tag="extremes">Extremes</a>
    { .paper-tags }

</div>

<nav class="pager" markdown="span">**1** [2](2.md) [3](3.md) [4](4.md) [5](5.md) [6](6.md) [7](7.md) [8](8.md) [9](9.md) [10](10.md) [11](11.md) [12](12.md) [13](13.md) [Older :material-arrow-right:](2.md){ .pager-step }</nav>

