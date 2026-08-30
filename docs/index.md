---
hide:
  - navigation
title: Deep Learning in Weather
---

A collection of papers on deep learning and machine learning applied to weather forecasting, climate modeling, and atmospheric science.

*Last updated: 2026-08-30*

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

-   #### Bridging short- and medium-range weather forecasting with machine learning

    ---

    <span class="paper-meta"><em>Timothy A. Smith, Mariah Pope, Sergey Frolov, Brett Basarab, Daniel Abdi, Paul Madden et al.</em> · 2026</span>

    <span class="abstract-snippet" id="snip-2608.26822" data-search-exclude>The National Oceanic and Atmospheric Administration (NOAA) employs independent prediction systems for distinct forecast products. While some separation is practical, we argue that combining short-...</span><span class="abstract-full" id="full-2608.26822" data-search-exclude hidden>The National Oceanic and Atmospheric Administration (NOAA) employs independent prediction systems for distinct forecast products. While some separation is practical, we argue that combining short- and medium-range weather into a single prediction system would provide the public with a useful distillation of global weather and its impacts. To this end, we present Nested-EAGLE (Experimental Artificial intelligence Global and Limited-area Ensemble): a 0.25° global weather model with a 6 km refinement over the Contiguous United States (CONUS). The model achieves significantly lower mean-squared error in near-surface and low-level quantities over CONUS compared to NOAA's Global Forecast System and High-Resolution Rapid Refresh (HRRR), while remaining competitive throughout the rest of the global atmosphere. We show that the skill gains for near-surface fields stem from incorporating high-resolution regional analysis data into training through the nesting process. Forecasts of precipitation amounts are less skillful than those from HRRR, owing to deterministic training. However, we show that Nested-EAGLE provides the most accurate forecasts of storm locations at longer leads, despite blurred extrema. Our results motivate future work to extend the skill gains beyond CONUS and improve precipitation representation.</span> <span class="abstract-toggle" data-id="2608.26822">more</span>

    <span class="paper-links">[:material-file-document: 2608.26822](https://arxiv.org/abs/2608.26822v1) · [:material-content-copy: BibTeX](bibtex/2608.26822.bib){ .bibtex-link }</span>

-   #### SimCast-S2S: An Efficient Generative Model for Subseasonal Precipitation Forecasting via Transfer Learning from Climate Simulations

    ---

    <span class="paper-meta"><em>Hiep V. Dang, Antonios Mamalakis</em> · 2026</span>

    <span class="abstract-snippet" id="snip-2608.26594" data-search-exclude>Subseasonal-to-seasonal (S2S) precipitation forecasting has substantial financial and societal impact, yet remains challenging because of weak predictive signals, high associated uncertainty, and the...</span><span class="abstract-full" id="full-2608.26594" data-search-exclude hidden>Subseasonal-to-seasonal (S2S) precipitation forecasting has substantial financial and societal impact, yet remains challenging because of weak predictive signals, high associated uncertainty, and the computational cost of operational systems, which constrains simulation fidelity. We introduce SimCast-S2S, a generative latent-diffusion framework for probabilistic S2S precipitation forecasting that addresses three major bottlenecks in data-driven prediction. First, because S2S prediction requires uncertainty quantification rather than only deterministic point forecasts, SimCast-S2S is the first data-driven system that uses a diffusion-based generative pipeline for S2S prediction, enabling effective sampling from the underlying conditional distribution. Second, since generating large probabilistic ensembles is computationally costly in physical space, SimCast-S2S instead operates in a compact latent space learned by variational autoencoders, enabling efficient large-ensemble generation. Third, diffusion models typically require large training datasets; SimCast-S2S overcomes this via transfer learning with low-rank adaptation (LoRA), pretraining on large ensembles of climate simulations before fine-tuning on limited reanalysis data. On reanalysis data, SimCast-S2S outperforms deep learning baselines, including convolutional neural networks and U-Net architectures. Notably, despite using only a subset of atmospheric input variables and no post-processing, bias correction, or calibration, SimCast-S2S remains competitive with, and in many cases outperforms, state-of-the-art operational systems such as the ECMWF-S2S baseline. These results indicate that latent generative modeling combined with simulation-to-reanalysis transfer learning offers an efficient and scalable path toward data-driven probabilistic S2S precipitation forecasting.</span> <span class="abstract-toggle" data-id="2608.26594">more</span>

    <span class="paper-links">[:material-file-document: 2608.26594](https://arxiv.org/abs/2608.26594v1) · [:material-content-copy: BibTeX](bibtex/2608.26594.bib){ .bibtex-link }</span>

    <a class="md-tag" href="/tags/#diffusion">diffusion</a> <a class="md-tag" href="/tags/#cnn">CNN</a> <a class="md-tag" href="/tags/#variational">variational</a> <a class="md-tag" href="/tags/#probabilistic">probabilistic</a>

-   #### When Does Forecast-Error Energy Grow Logistically in Geophysical Turbulence?

    ---

    <span class="paper-meta"><em>Malaquias Peña</em> · 2026</span>

    <span class="abstract-snippet" id="snip-2608.26492" data-search-exclude>Coarse-graining can yield a simple macroscopic growth curve in a bounded chaotic system even when constituent scales follow different clocks. The distinction matters as reduced-order and generative...</span><span class="abstract-full" id="full-2608.26492" data-search-exclude hidden>Coarse-graining can yield a simple macroscopic growth curve in a bounded chaotic system even when constituent scales follow different clocks. The distinction matters as reduced-order and generative models compress multiscale forecast uncertainty into learned coordinates. We ask when forecast-error energy admits a logistic law. From the exact twin-error budget and correlated and decorrelated spectra, we derive two scalar limits: an invariant decorrelation amplitude, logistic only when contributing scales share one shape and one clock, and a self-similar upscale error front whose law depends on spectral slope and front speed. With local-strain scaling, the front predicts exponential error-energy growth for the canonical barotropic-vorticity spectrum and linear growth for the surface-quasigeostrophic spectrum. Stationary forced surface-quasigeostrophic twins test the logistic admission conditions. A response-blind partition of 16 trajectories gives cluster-mean logistic root-mean-square deviations 0.080 and 0.093, although every trajectory has resolved clock heterogeneity. An exact averaging identity shows how signed shape and clock corrections cancel, producing a nearly logistic aggregate while constituent scales retain distinct clocks. Mechanism identification therefore requires more than goodness of fit: independent shape, clock, and residual tests are required. These admission conditions provide physics-based guardrails for compact representations of chaotic systems and generative forecast ensembles.</span> <span class="abstract-toggle" data-id="2608.26492">more</span>

    <span class="paper-links">[:material-file-document: 2608.26492](https://arxiv.org/abs/2608.26492v1) · [:material-content-copy: BibTeX](bibtex/2608.26492.bib){ .bibtex-link }</span>

    <a class="md-tag" href="/tags/#physics-informed">physics-informed</a>

-   #### Precipitation Downscaling Using Foundation Model-Conditioned Diffusion

    ---

    <span class="paper-meta"><em>Victor Nascimento Ribeiro, Jorge Guevara, Jorge Sebastian Moraga, Chris Lucas, Natalie Lord et al.</em> · 2026</span>

    <span class="abstract-snippet" id="snip-2608.25858" data-search-exclude>High-resolution precipitation fields are essential for hydrological impact assessment, yet global climate model outputs are too coarse and biased for direct use. AI-based statistical downscaling with...</span><span class="abstract-full" id="full-2608.25858" data-search-exclude hidden>High-resolution precipitation fields are essential for hydrological impact assessment, yet global climate model outputs are too coarse and biased for direct use. AI-based statistical downscaling with diffusion models offers a promising approach, but the mechanism by which large-scale atmospheric predictors condition generation remains largely unexplored. We investigate three conditioning strategies for a denoising diffusion probabilistic model applied to daily precipitation downscaling: channel concatenation of upsampled coarse predictors, cross-attention conditioning with a learned convolutional encoder, and cross-attention conditioning with the frozen encoder of the pretrained Prithvi WxC weather foundation model. All strategies are evaluated against an unconditioned baseline under identical conditions using probabilistic, distributional, spectral, and extreme-event metrics for the Colorado River Basin. Concatenation conditioning achieves the lowest point-wise CRPS and MSE, but tends to produce over-smoothed fields that suppress high-intensity events. In contrast, cross-attention conditioning provides substantially better distributional realism and modest improvements in spectral fidelity. Improvements are greatest for extremes: the Prithvi-WxC conditioned model retains over half of >100mm/day events, although estimates are uncertain due to limited samples. When trained on the full dataset, the learned convolutional model performs similarly to the foundation model-conditioned approach while requiring lower computational resources. However, the Prithvi-WxC-conditioned model achieves comparable performance with only five years of training data. These results indicate that cross-attention conditioning offers advantages over simple concatenation for probabilistic precipitation downscaling, and that pre-trained foundation model representations may offer benefits in data-limited settings.</span> <span class="abstract-toggle" data-id="2608.25858">more</span>

    <span class="paper-links">[:material-file-document: 2608.25858](https://arxiv.org/abs/2608.25858v1) · [:material-content-copy: BibTeX](bibtex/2608.25858.bib){ .bibtex-link }</span>

    <a class="md-tag" href="/tags/#transformer">transformer</a> <a class="md-tag" href="/tags/#diffusion">diffusion</a> <a class="md-tag" href="/tags/#foundation-model">foundation-model</a> <a class="md-tag" href="/tags/#probabilistic">probabilistic</a>

-   #### Missing the Butterfly and Predicting the Past: Features or Bugs of Accurate AI Weather Models?

    ---

    <span class="paper-meta"><em>Pedram Hassanzadeh, Weidong Li, Y. Qiang Sun, Jiangdi Wang, Alexander Wikner, Justin Finkel et al.</em> · 2026</span>

    <span class="abstract-snippet" id="snip-2608.25835" data-search-exclude>AI weather prediction (AIWP) models rival physics-based models, yet the sources of their unexpected forecast accuracy and the degree of their physical fidelity remain unclear. Here, across a...</span><span class="abstract-full" id="full-2608.25835" data-search-exclude hidden>AI weather prediction (AIWP) models rival physics-based models, yet the sources of their unexpected forecast accuracy and the degree of their physical fidelity remain unclear. Here, across a hierarchy spanning observation-based reanalysis, a general circulation model, and the multi-scale Lorenz system, we show that AI models can be trained to skillfully predict the past (backcast), though backcasts are systematically less accurate than forecasts. However, skillful backcasting appears to violate the second law of thermodynamics, and all these forecasting and backcasting models miss the butterfly effect. We trace the surprising forecast accuracy, missing butterfly, and skillful backcasting to a single cause: inevitable coarse-graining of training data, which removes fast, small scales and/or some variables. From the Lorenz system to official Pangu-Weather models, reducing coarse-graining makes AI predictions more physics-like (arrow of time and butterfly-like effects emerge), but forecast accuracy declines. Results offer an explanation for AIWP models' forecast skill: unlike physics-based models, they implicitly learn how fast, small scales affect large scales without inheriting their rapid error growth. Broader implications are that AI models' proliferation calls for revisiting predictability theories and long-term climate emulation strategies, and backcasting offers a useful, new lens for such analyses.</span> <span class="abstract-toggle" data-id="2608.25835">more</span>

    <span class="paper-links">[:material-file-document: 2608.25835](https://arxiv.org/abs/2608.25835v1) · [:material-content-copy: BibTeX](bibtex/2608.25835.bib){ .bibtex-link }</span>

    <a class="md-tag" href="/tags/#physics-informed">physics-informed</a>

-   #### Tropospheric temperature and humidity profile retrieval from Meteosat Flexible Combined Imager based on deep learning

    ---

    <span class="paper-meta"><em>Alejandro Salgueiro, Johannes Rausch, Julie Thérèse Villinger, Angela Meyer</em> · 2026</span>

    <span class="abstract-snippet" id="snip-2608.25700" data-search-exclude>The Meteosat Third Generation (MTG) Flexible Combined Imager (FCI) offers new opportunities for tropospheric temperature and humidity profiling, at higher spatio-temporal resolutions and expanded...</span><span class="abstract-full" id="full-2608.25700" data-search-exclude hidden>The Meteosat Third Generation (MTG) Flexible Combined Imager (FCI) offers new opportunities for tropospheric temperature and humidity profiling, at higher spatio-temporal resolutions and expanded spectral coverage relative to its predecessor. Vertically resolved retrievals from broadband imagers are inherently challenging, and operational retrieval algorithms typically rely on numerical weather prediction (NWP) background fields to compensate for limited infrared spectral resolution, reducing the retrievals' independence. We develop a spatially aware deep learning framework to retrieve all-sky tropospheric temperature and humidity profiles from FCI, without forecast profiles as input. A Residual U-Net that exploits spatial context across all 16 FCI channels was trained on 14 months of collocated FCI observations and CERRA reanalysis targets over Europe. Validated against independent radiosondes, retrieved temperatures show biases below 0.4 K and standard deviations of 1.5-1.9 K. Retrieved relative humidity standard deviations range from 12-20 %, compared to 9-19 % for CERRA. Performance degrades modestly under clouds, with standard deviation increases below 0.4 K and 3 % RH beneath cloud tops despite limited direct radiative information. Ablation experiments show that spatial context improves retrievals, with the largest gains below cloud tops. Feature sensitivity analysis indicates broad consistency with FCI bands' established radiative transfer characteristics. Visible and near-infrared channels contribute despite not being commonly used in physics-based profile inversions. These results demonstrate that spatially aware deep learning models can extract statistically reliable tropospheric profiles from geostationary imager observations, independent of NWP forecast fields, enabling more rapid autonomous monitoring of the atmosphere.</span> <span class="abstract-toggle" data-id="2608.25700">more</span>

    <span class="paper-links">[:material-file-document: 2608.25700](https://arxiv.org/abs/2608.25700v1) · [:material-content-copy: BibTeX](bibtex/2608.25700.bib){ .bibtex-link }</span>

    <a class="md-tag" href="/tags/#cnn">CNN</a> <a class="md-tag" href="/tags/#physics-informed">physics-informed</a>

-   #### Frequency-aware forecasting for short-term typhoon gust prediction

    ---

    <span class="paper-meta"><em>Xuefei Wang, Tingyi Liu, Heng Zhang, Shengjun Zhang</em> · 2026</span>

    <span class="abstract-snippet" id="snip-2608.25604" data-search-exclude>Accurate gust forecasting under typhoon conditions remains challenging due to the highly non-stationary and multi-scale characteristics of extreme wind fluctuations. Existing deep learning models...</span><span class="abstract-full" id="full-2608.25604" data-search-exclude hidden>Accurate gust forecasting under typhoon conditions remains challenging due to the highly non-stationary and multi-scale characteristics of extreme wind fluctuations. Existing deep learning models often struggle to simultaneously capture long-term trends and rapid local variations, resulting in degraded performance during extreme events. We propose WDANet, a frequency-aware forecasting framework that integrates stationary wavelet decomposition, a Feature-wise Linear Modulation (FiLM) strategy, and a dual-branch encoder-decoder architecture, enabling separate modeling of trend and fluctuation components. Taking the offshore regions of the Western Pacific in China as an example, we conduct fine-grid wind gust prediction research. The results demonstrate that WDANet shows advantages for short lead times under the experimental setting across a 24-h forecasting horizon and achieves higher prediction accuracy than ECMWF-HRES within the first 6 h. During extreme wind events, WDANet more accurately captures gust peaks and attains the best RMSE and MAE performance. These results highlight its potential for offshore wind power operation, disaster warning, and risk mitigation.</span> <span class="abstract-toggle" data-id="2608.25604">more</span>

    <span class="paper-links">[:material-file-document: 2608.25604](https://arxiv.org/abs/2608.25604v1) · [:material-content-copy: BibTeX](bibtex/2608.25604.bib){ .bibtex-link }</span>

-   #### Energy Yield and Lifetime Climate Classification via Machine Learning for Optimizing Photovoltaic Module Design and Materials

    ---

    <span class="paper-meta"><em>Youri Blom, Sofia Dutto, Alexandru Costache, Rowan Richie, Ruben Pelsser, Wesley Berger, Jing Sun et al.</em> · 2026</span>

    <span class="abstract-snippet" id="snip-2608.25448" data-search-exclude>To resiliently and sustainably meet our future energy demand, photovoltaic (PV) modules must be deployed across a broad and diverse range of geographical regions with varying operating conditions. As...</span><span class="abstract-full" id="full-2608.25448" data-search-exclude hidden>To resiliently and sustainably meet our future energy demand, photovoltaic (PV) modules must be deployed across a broad and diverse range of geographical regions with varying operating conditions. As these conditions strongly affect both performance and optimal system design, a dedicated PV-specific climate classification can be of great use. In this work, we develop a climate classification framework tailored to PV applications using a variety of machine learning (ML) techniques. Building on previous studies, our approach incorporates both energy yield, and for the first time, also the module lifetime with climate dependent degradation. We generate an interpolated dataset containing twelve input features and two target variables (i.e. energy yield and module lifetime). Feature importance analysis shows that annual global horizontal irradiation and ambient temperature are the most influential predictors. The most accurate regression model achieves root mean square errors (RMSE) of 0.007 MWh for energy yield and 1.5 years for lifetime prediction. The calculated feature importance scores are then integrated into a hierarchical clustering framework, resulting in 6 primary climate clusters (Tropical, Desert, Continental, Temperate, Boreal, and Polar) and 15 corresponding subclusters. Our analysis shows that the low temperature continental climate offers the highest discounted lifetime energy yield. These results can support a wide range of applications, including PV module optimization, system siting decisions, and comparative performance studies.</span> <span class="abstract-toggle" data-id="2608.25448">more</span>

    <span class="paper-links">[:material-file-document: 2608.25448](https://arxiv.org/abs/2608.25448v1) · [:material-content-copy: BibTeX](bibtex/2608.25448.bib){ .bibtex-link }</span>

-   #### AFDBench: A Reasoning-First AI Scientist for NationalWeather Service Forecast Discussions

    ---

    <span class="paper-meta"><em>Manmeet Singh, Somnath Luitel, Prabhjot Singh, Manraaj Banga, Naveen Sudharsan, Josh Durkee</em> · 2026</span>

    <span class="abstract-snippet" id="snip-2608.24954" data-search-exclude>Large language models (LLMs) hallucinate numerical values when generating high-stakes meteorological text, posing risks for weather communication. We present AFDBench, an AI meteorologist that...</span><span class="abstract-full" id="full-2608.24954" data-search-exclude hidden>Large language models (LLMs) hallucinate numerical values when generating high-stakes meteorological text, posing risks for weather communication. We present AFDBench, an AI meteorologist that generates professional Area Forecast Discussions (AFDs) by reasoning through structured AI weather forecast data from Google's WeatherNext 2. We introduce AFDBench, the first benchmark for evaluating generative meteorological reasoning, comprising 7,732 expert written discussions from 13 National Weather Service (NWS) offices paired with real AI weather forecast inputs, and three complementary metrics: Met-Align (numerical accuracy), Style-Align (professional dialect adherence), and Input-Grounding (fidelity to source weather data). Zero-shot evaluations reveal that open-source LLMs achieve low Style-Align (~0.33) and moderate Input-Grounding (~0.88), failing to write in the professional NWS register or faithfully use their input data. We apply Group Relative Policy Optimization (GRPO) with domain-specific rewards targeting temperature accuracy, synoptic correctness, and format compliance. On 1,033 held-out samples from two unseen NWS offices, GRPO nearly doubles Style-Align from 0.318 to 0.619 and improves Input-Grounding from 0.881 to 0.940, demonstrating that reinforcement learning teaches a 7B-parameter model to write like a professional meteorologist and faithfully interpret AI weather data.</span> <span class="abstract-toggle" data-id="2608.24954">more</span>

    <span class="paper-links">[:material-file-document: 2608.24954](https://arxiv.org/abs/2608.24954v1) · [:material-content-copy: BibTeX](bibtex/2608.24954.bib){ .bibtex-link }</span>

    <a class="md-tag" href="/tags/#reinforcement-learning">reinforcement-learning</a>

-   #### Deep Learning Super Resolution for Satellite Cloud Mask Downscaling

    ---

    <span class="paper-meta"><em>Angelos Georgakis, Valentina Kanaki, Giorgos Giannopoulos, Stella Girtsou, Ioannis Kontogiorgakis et al.</em> · 2026</span>

    <span class="abstract-snippet" id="snip-2608.24715" data-search-exclude>A vast amount of optical satellite data is being transmitted to Earth-based servers every day, and more than half of this data is affected by haze or clouds. Additionally, this data suffers from the...</span><span class="abstract-full" id="full-2608.24715" data-search-exclude hidden>A vast amount of optical satellite data is being transmitted to Earth-based servers every day, and more than half of this data is affected by haze or clouds. Additionally, this data suffers from the fundamental trade-off between spatial and temporal resolution, which remains largely unresolved, making the acquisition of continuous high-resolution satellite observations of clouds an ongoing challenge. This work addresses this challenge by proposing two Deep Learning super-resolution methods for the accurate downscaling of SEVIRI cloud mask products, as well as a novel cross-sensor cloud mask dataset called SEVMOD-CM, created by spatially and temporally matching MODIS and SEVIRI satellite observations. The two proposed models are a CNN-based (SpatialCNN) and a GAN-based (SpatialGAN) Neural Network. Trained on the SEVIRI spectral and cloud mask products, the proposed methods predict the corresponding MODIS Cloud masks, achieving a 4x spatial enhancement across sensor domains. Both approaches are evaluated experimentally, and compared against the standard bicubic interpolation upsampling technique. The experimental results demonstrate the value of the proposed models and dataset for the remote sensing community, highlighting the benefits of applying super-resolution techniques to geostationary-derived cloud mask products for applications such as atmospheric monitoring, weather forecasting, disaster risk reduction, solar energy forecasting, and climate research.</span> <span class="abstract-toggle" data-id="2608.24715">more</span>

    <span class="paper-links">[:material-file-document: 2608.24715](https://arxiv.org/abs/2608.24715v1) · [:material-content-copy: BibTeX](bibtex/2608.24715.bib){ .bibtex-link }</span>

</div>

