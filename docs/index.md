---
hide:
  - navigation
title: Deep Learning in Weather
---

A collection of papers on deep learning and machine learning applied to weather forecasting, climate modeling, and atmospheric science.

*Last updated: 2026-09-06*

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

-   #### From Nowcasting to Forecasting: Adapting a Reanalysis-Trained

    ---

    <span class="paper-meta"><em>Mikko Partio, Leila Hieta, Ossi Laine</em> · 2026</span>

    <span class="abstract-snippet" id="snip-2609.03763" data-search-exclude>Accurate cloud-cover forecasts are important for temperature prediction, radiation forecasting, and solar-power operations. Short-range forecasting methods can preserve observed cloud placement...</span><span class="abstract-full" id="full-2609.03763" data-search-exclude hidden>Accurate cloud-cover forecasts are important for temperature prediction, radiation forecasting, and solar-power operations. Short-range forecasting methods can preserve observed cloud placement during the first forecast hours, but their skill decreases when cloud fields evolve through formation, dissipation and deformation. Longer lead times require accounting for atmospheric evolution, but operational numerical weather prediction (NWP) forecasts may not accurately represent the satellite-observed cloud state at initialization. We develop CloudCast v2, a machine-learning model for 12-hour cloud-cover forecasting from observation-based initial conditions. The model is first trained on the Copernicus European Regional Reanalysis (Ridal2024) to learn cloud-evolution dynamics, and is then adapted to satellite-derived cloud fields using conditional flow matching (Lipman2023), a generative method that transforms noise into cloud-cover forecasts conditioned on the observed initial cloud fields and NWP inputs. CloudCast v2 reduces mean absolute error by 10% relative to its predecessor, CloudCast v1 (Partio2025), over the 1-12 h range. It also overtakes CloudCast v1 in fractions skill score, a neighborhood-based measure of spatial agreement, after approximately 3-6 h, depending on the cloudiness category. These results show that observation-initialized machine-learning forecasts can extend beyond the usual 1-3-hour nowcasting range while retaining spatial detail from satellite cloud fields.</span> <span class="abstract-toggle" data-id="2609.03763">more</span>

    <span class="paper-links">[:material-file-document: 2609.03763](https://arxiv.org/abs/2609.03763v1) · [:material-content-copy: BibTeX](bibtex/2609.03763.bib){ .bibtex-link }</span>

    <a class="md-tag" href="/tags/#diffusion">diffusion</a>

-   #### Radiative and Dynamical Controls on the Land-Ocean Warming Contrast in Climate Models

    ---

    <span class="paper-meta"><em>Paolo Giani, Arlene M. Fiore, Raffaele Ferrari, Paul A. O'Gorman, Vincent T. Cooper, Noelle E. Selin</em> · 2026</span>

    <span class="abstract-snippet" id="snip-2609.03658" data-search-exclude>Surface air over land warms substantially more than over the ocean under greenhouse forcing, a phenomenon known as the land-ocean warming contrast. Current explanations for this contrast are commonly...</span><span class="abstract-full" id="full-2609.03658" data-search-exclude hidden>Surface air over land warms substantially more than over the ocean under greenhouse forcing, a phenomenon known as the land-ocean warming contrast. Current explanations for this contrast are commonly expressed either in terms of energetic constraints, from top-of-atmosphere and surface energy balance, or dynamical constraints, from large-scale atmospheric dynamics. We show that these perspectives are complementary when viewed through the lens of atmospheric moist static energy (MSE) transport, and that connecting them yields new insight into the controls of the warming contrast and the spread in climate models. We use this framework to construct an interpretable emulator that reproduces the land-ocean warming response across 22 models from the latest Coupled Model Intercomparison Project (CMIP6). We find that the strength of the land-ocean warming contrast emerges from the interplay between a model-dependent radiative baseline and a robust dynamical restoring mechanism that favors greater warming over land. This interplay leads to two broad model regimes that align with climate sensitivity. In low-climate-sensitivity models, more stabilizing radiative feedbacks over the ocean directly favor greater land warming. In high-climate-sensitivity models, radiative feedbacks alone would instead favor greater ocean warming, but a strong MSE-transport feedback (approximately 0.2 PW/K) more than compensates for this tendency. The intermodel spread in the land-ocean warming contrast is closely related to the ratio of radiative feedbacks over land and ocean, highlighting a broader connection between climate sensitivity and the land-ocean warming contrast.</span> <span class="abstract-toggle" data-id="2609.03658">more</span>

    <span class="paper-links">[:material-file-document: 2609.03658](https://arxiv.org/abs/2609.03658v1) · [:material-content-copy: BibTeX](bibtex/2609.03658.bib){ .bibtex-link }</span>

-   #### WeatherNext 3: Increasing resolution and performance of global weather models with raw observations

    ---

    <span class="paper-meta"><em>Stephan Rasp, Boris Babenko, Dominic Masters, Andrew El-Kadi, Samier Merchant, Guy Shalev et al.</em> · 2026</span>

    <span class="abstract-snippet" id="snip-2609.03582" data-search-exclude>State-of-the-art AI weather models have shown impressive medium-range forecast skill and computational efficiency, but suffer two key shortcomings: their forecasts have lower spatial and temporal...</span><span class="abstract-full" id="full-2609.03582" data-search-exclude hidden>State-of-the-art AI weather models have shown impressive medium-range forecast skill and computational efficiency, but suffer two key shortcomings: their forecasts have lower spatial and temporal resolution than the best physics-based models and they are exclusively initialized with and trained on analysis data. As a result, they cannot directly make use of observations, and any biases in the analysis are inherited by the forecast. WeatherNext 3 addresses these shortcomings and establishes a new state-of-the-art for probabilistic medium-range forecasting skill. First, WeatherNext 3 generates new forecasts every hour (rather than every 6 hours like traditional global models) by ingesting low-latency geostationary satellite data. Second, WeatherNext 3's temporal and spatial resolution are on par with physics-based global models, with hourly time steps and 0.1 degree resolution for single-level variables, including solar radiation and cloud cover. Third, WeatherNext 3 moves beyond traditional analysis variables by learning to predict satellite-derived precipitation estimates, as well as tropical cyclone and station observations. Modelling sparse station data allows WeatherNext 3 to make 2m temperature and dewpoint predictions at any location and time, conditioned on local geographical features, with substantially lower error than competing global models, even when evaluated against unseen stations. Together, WeatherNext 3's capabilities move operational AI-based weather forecasting beyond emulating the traditionally distinct stages of data assimilation, forecasting and post-processing, which helps to further push the frontier of performance and granularity for global weather prediction.</span> <span class="abstract-toggle" data-id="2609.03582">more</span>

    <span class="paper-links">[:material-file-document: 2609.03582](https://arxiv.org/abs/2609.03582v1) · [:material-content-copy: BibTeX](bibtex/2609.03582.bib){ .bibtex-link }</span>

    <a class="md-tag" href="/tags/#physics-informed">physics-informed</a> <a class="md-tag" href="/tags/#probabilistic">probabilistic</a>

-   #### Improving precipitation forecasts in an AI weather model using observational data

    ---

    <span class="paper-meta"><em>Julian F. Schmitt, Bertrand Delorme, Robert C. King, Yashica Patodia, Tapio Schneider et al.</em> · 2026</span>

    <span class="abstract-snippet" id="snip-2609.03210" data-search-exclude>Artificial intelligence weather prediction (AIWP) systems now surpass state-of-the-art physical models for medium-range weather forecasting. Current global AIWP models are trained almost exclusively...</span><span class="abstract-full" id="full-2609.03210" data-search-exclude hidden>Artificial intelligence weather prediction (AIWP) systems now surpass state-of-the-art physical models for medium-range weather forecasting. Current global AIWP models are trained almost exclusively using one reanalysis dataset, ERA5, but it has known biases, particularly for precipitation. Here we fine-tune a graph-transformer architecture with IMERG precipitation data at 0.25° resolution. The resulting model improves medium-range continuous ranked probability scores by up to 19%, while also demonstrating superior skill for tropical storms and drizzle events. Our model exceeds the Brier skill score of state-of-the-art operational models on extreme rainfall prediction by 57% globally; however, a physics-based operational model remains more reliable for the heaviest precipitation events. Our results demonstrate that incorporating observations-based precipitation data directly into training can substantially improve precipitation forecasts.</span> <span class="abstract-toggle" data-id="2609.03210">more</span>

    <span class="paper-links">[:material-file-document: 2609.03210](https://arxiv.org/abs/2609.03210v1) · [:material-content-copy: BibTeX](bibtex/2609.03210.bib){ .bibtex-link }</span>

    <a class="md-tag" href="/tags/#transformer">transformer</a> <a class="md-tag" href="/tags/#physics-informed">physics-informed</a>

-   #### Distilling deep optical flow stereo methods to retrieve dense three-dimensional wind fields

    ---

    <span class="paper-meta"><em>Thomas J. Vandal, Dong L. Wu, James L. Carr, Derek J. Posselt, Elise Penn, Tristan Ballard et al.</em> · 2026</span>

    <span class="abstract-snippet" id="snip-2609.03100" data-search-exclude>Geostationary atmospheric motion vectors (AMVs) provide the dense horizontal wind vectors (u,v) and heights ingested into data assimilation systems. Traditional AMVs track features using window-based...</span><span class="abstract-full" id="full-2609.03100" data-search-exclude hidden>Geostationary atmospheric motion vectors (AMVs) provide the dense horizontal wind vectors (u,v) and heights ingested into data assimilation systems. Traditional AMVs track features using window-based cross-correlation and estimate heights via infrared brightness temperatures paired with numerical weather prediction (NWP) background states, creating a circular dependency that yields inaccurate heights, high computational cost, and sparse retrievals. Stereo winds from GEO-GEO and GEO-LEO geometrically resolve heights from parallax shifts across different poses, eliminating NWP dependence and improving accuracy, but they remain computationally heavy with limited coverage. In this work, we replace window-based tracking in stereo matching with deep optical flow for efficient, improved retrieval. Fine-tuning balances a self-supervised geometric residual loss with supervised radiosonde reconstruction. To eliminate multi-satellite overlap requirements, we distill the stereo teacher into a single-satellite student model. Chi-square and height uncertainties from the teacher are emulated by the student for quality assurance. The student generates winds across full-disk GEO imagery globally. Validation compares stereo and student models against radiosondes, operational AMVs, ERA5 reanalysis, and EarthCARE cloud profiles. Results through triple collocation show that stereo winds improve performance beyond operational AMVs for water vapor bands (6.2, 6.9, and 7.3 μm), wit degradation in the long-wave infrared (11.2 μm) band.</span> <span class="abstract-toggle" data-id="2609.03100">more</span>

    <span class="paper-links">[:material-file-document: 2609.03100](https://arxiv.org/abs/2609.03100v1) · [:material-content-copy: BibTeX](bibtex/2609.03100.bib){ .bibtex-link }</span>

-   #### Efficient All-in-One Weather Restoration using Spectral Harmonization

    ---

    <span class="paper-meta"><em>Paula Garrido-Mellado, Daniel Feijoo, Yuning Cui, Alvaro Garcia, Marcos V. Conde</em> · 2026</span>

    <span class="abstract-snippet" id="snip-2609.02839" data-search-exclude>Adverse weather conditions such as rain, haze, and snow significantly degrade image quality, posing challenges for both human perception and physical AI. Existing restoration methods require large...</span><span class="abstract-full" id="full-2609.02839" data-search-exclude hidden>Adverse weather conditions such as rain, haze, and snow significantly degrade image quality, posing challenges for both human perception and physical AI. Existing restoration methods require large computational budgets, struggling to process high-resolution images and handle different degradations. In this paper, we present Frequency Reconstruction via Spectral Harmonization, a novel lightweight all-in-one restoration method that explicitly decomposes feature representations into high- and low-frequency components at each scale of a hierarchical encoder-decoder architecture. By combining spectral decomposition with spatial processing through Fourier-based skip connections, FReSH-IR captures complementary frequency information without sacrificing spatial detail. Our approach achieves similar restoration quality with 80% fewer parameters and operations than transformer-based models. Extensive experiments demonstrate that our method offers a great efficiency-performance trade-off, highlighting its practical applications in constrained-resource systems.</span> <span class="abstract-toggle" data-id="2609.02839">more</span>

    <span class="paper-links">[:material-file-document: 2609.02839](https://arxiv.org/abs/2609.02839v1) · [:material-content-copy: BibTeX](bibtex/2609.02839.bib){ .bibtex-link }</span>

    <a class="md-tag" href="/tags/#transformer">transformer</a>

-   #### Uncertainty-Guided Adverse Weather Restoration via Gated Transformer Network

    ---

    <span class="paper-meta"><em>Zheke Jin, Yuning Cui, Tianle Jin, Alois Knoll, Hu Cao</em> · 2026</span>

    <span class="abstract-snippet" id="snip-2609.02434" data-search-exclude>Restoring images degraded by adverse weather remains challenging due to spatially heterogeneous degradations. Many existing weather-specific restoration models rely on weather-agnostic global...</span><span class="abstract-full" id="full-2609.02434" data-search-exclude hidden>Restoring images degraded by adverse weather remains challenging due to spatially heterogeneous degradations. Many existing weather-specific restoration models rely on weather-agnostic global aggregation, naive cross-scale fusion, and deterministic objectives, which struggle to handle heterogeneous degradations in all-in-one adverse-weather settings. To address these limitations, we propose an Uncertainty-guided Adverse-weather Restoration Network (UAR-Net), a weather-specific AiO framework that integrates a gated transformer with balanced multi-scale skip connections. Specifically, we employ Gated Dual-scale Transformer Blocks (GDTB) to jointly model selective global interactions and multi-scale local structures, a progressive Balanced Multi-scale Skip Connection (BMSC) for balanced multi-scale feature integration, and an Uncertainty-Aware Refinement Head (URH) that performs artifact removal, detail enhancement, and predictive uncertainty estimation. The model is supervised by a Brightness-Aware Energy Loss (BAE-Loss) to encourage accurate reconstruction with well-calibrated uncertainty. Extensive experiments demonstrate that our method achieves state-of-the-art performance across multiple adverse-weather benchmarks. The codes will open source upon acceptance.</span> <span class="abstract-toggle" data-id="2609.02434">more</span>

    <span class="paper-links">[:material-file-document: 2609.02434](https://arxiv.org/abs/2609.02434v1) · [:material-content-copy: BibTeX](bibtex/2609.02434.bib){ .bibtex-link }</span>

    <a class="md-tag" href="/tags/#transformer">transformer</a>

-   #### TC-Next: Zero-Shot Multimodal Cyclone Forecasting

    ---

    <span class="paper-meta"><em>Zhe Wang, Sijie Chen, Yiming Luo, Daehyun Kim, Chien-Yi Chang</em> · 2026</span>

    <span class="abstract-snippet" id="snip-2609.02085" data-search-exclude>We present TropicalCycloneNext (TC-Next), a multimodal deep learning model that forecasts tropical cyclone track and intensity at $6$-$24$ h leads by leveraging a foundation model's forecast fields...</span><span class="abstract-full" id="full-2609.02085" data-search-exclude hidden>We present TropicalCycloneNext (TC-Next), a multimodal deep learning model that forecasts tropical cyclone track and intensity at $6$-$24$ h leads by leveraging a foundation model's forecast fields of atmospheric kinematic and thermodynamic fields and GridSat infrared satellite imagery. Trained only on GraphCast forecasts over the Western Pacific (WP), yet reliant only on generic atmospheric variables, TC-Next on GraphCast lowers track error by $15$-$44\%$ and intensity error by a factor of $3$-$6$ relative to a conventional, rule-based tracker, TempestExtremes; applied without retraining to the forecast fields of Pangu-Weather and IFS HRES, it stays ahead of TempestExtremes on both. Applied zero-shot to the generic weather fields of WeatherNext Cyclones on the 2025 WP season, TC-Next attains lower intensity error at every lead time, and lower or comparable track error, compared to that model's specialized direct tracker in a deterministic comparison. Our ablation studies show that our multimodal model is able to utilize the additional modality to improve performance in tracking errors at every lead time and in intensity prediction at longer lead times.</span> <span class="abstract-toggle" data-id="2609.02085">more</span>

    <span class="paper-links">[:material-file-document: 2609.02085](https://arxiv.org/abs/2609.02085v1) · [:material-content-copy: BibTeX](bibtex/2609.02085.bib){ .bibtex-link }</span>

    <a class="md-tag" href="/tags/#foundation-model">foundation-model</a>

-   #### Kilometer-Scale AI Downscaling of Atlantic Hurricanes with Generative Ensembles

    ---

    <span class="paper-meta"><em>Yingkai Sha, Talea L. Mayo, Ethan D. Gutmann, Lulin Xue, Andrew Newman</em> · 2026</span>

    <span class="abstract-snippet" id="snip-2609.02034" data-search-exclude>This study presents an AI-based dynamical downscaling system for Tropical Cyclones (TCs). The system incorporates an AI-based limited-area model that downscales 3-hourly low-resolution boundary...</span><span class="abstract-full" id="full-2609.02034" data-search-exclude hidden>This study presents an AI-based dynamical downscaling system for Tropical Cyclones (TCs). The system incorporates an AI-based limited-area model that downscales 3-hourly low-resolution boundary forcings into hourly high-resolution fields autoregressively, and a diffusion model that converts the outputs into ensembles of hazard-relevant variables. The system is trained on the regridded CONUS404 data with ERA5 forcings, and is evaluated on 20 TCs in 2020--2024. Verification shows stable downscaling performance across Atlantic hurricane seasons, with energy spectra closely matching the CONUS404 reference. The system is also verified to produce skillful TC-relevant weather extremes, largely improved over a deterministic AI baseline. The system performs well with forcing data from other models (GDAS/FNL) and can produce detailed eyewall, rainband, and landfall structures in TC case studies. The study provides a good example of how AI-based dynamical downscaling systems can be designed to resolve small-scale extreme weather events.</span> <span class="abstract-toggle" data-id="2609.02034">more</span>

    <span class="paper-links">[:material-file-document: 2609.02034](https://arxiv.org/abs/2609.02034v1) · [:material-content-copy: BibTeX](bibtex/2609.02034.bib){ .bibtex-link }</span>

    <a class="md-tag" href="/tags/#diffusion">diffusion</a>

-   #### A Sensor-Adaptive Incremental Learning Framework for Artifact Detection in Satellite Precipitation Data

    ---

    <span class="paper-meta"><em>Andres F. Monsalve, Hernan A. Moreno, Christian D. Kummerow</em> · 2026</span>

    <span class="abstract-snippet" id="snip-2609.01514" data-search-exclude>Historically, retrieving rainfall data from satellite imagery has been the domain of space agencies. However, in recent years, the development of cheaper, more compact satellites (SmallSats) capable...</span><span class="abstract-full" id="full-2609.01514" data-search-exclude hidden>Historically, retrieving rainfall data from satellite imagery has been the domain of space agencies. However, in recent years, the development of cheaper, more compact satellites (SmallSats) capable of detecting rainfall proxies has led to a significant increase in private-sector initiatives for satellite launch and surface precipitation products. This rapid growth has yet to be matched by data validation efforts. Consequently, the need for a robust tool to detect anomalies in near-real-time data before it is disseminated to the public has become critical. In this paper, we present the development of an anomaly-detection system to identify artifacts in global satellite-based rainfall products. The developed framework leverages pre-trained computer vision models and incorporates scarce human-labeled data to detect specific anomalies. Our proposed anomaly detection strategy is tested on data from the Special Sensor Microwave Imager (SSMI) and the Special Sensor Microwave Imager/Sounder (SSMIS). Results demonstrate the efficacy of our approach at separating regular orbits from artifact-containing orbits for each satellite, with performance comparable to state-of-the-art in-place methods. Additionally, the framework offers explainability and the capacity for iterative refinement following false-positive or false-negative classifications.</span> <span class="abstract-toggle" data-id="2609.01514">more</span>

    <span class="paper-links">[:material-file-document: 2609.01514](https://arxiv.org/abs/2609.01514v1) · [:material-content-copy: BibTeX](bibtex/2609.01514.bib){ .bibtex-link }</span>

    <a class="md-tag" href="/tags/#foundation-model">foundation-model</a>

</div>

