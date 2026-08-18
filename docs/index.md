---
hide:
  - navigation
title: Deep Learning in Weather
---

A collection of papers on deep learning and machine learning applied to weather forecasting, climate modeling, and atmospheric science.

*Last updated: 2026-08-18*

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

-   #### Decadal wave reconstruction in the Mediterranean Sea with graph neural networks

    ---

    <span class="paper-meta"><em>Federica Benassi, Lorenzo Mentaschi, Salvatore Causio, Daniel Holmberg, Ivan Federico, Nadia Pinardi</em> · 2026</span>

    <span class="abstract-snippet" id="snip-2608.16449" data-search-exclude>Accurate simulation and prediction of ocean waves are essential for coastal risk management and climate studies. Deep learning has shown promising results for wave modeling, but most approaches still...</span><span class="abstract-full" id="full-2608.16449" data-search-exclude hidden>Accurate simulation and prediction of ocean waves are essential for coastal risk management and climate studies. Deep learning has shown promising results for wave modeling, but most approaches still operate on regular grids and on forecasting time scales, and do not generalize to unstructured discretization or to long time horizons. Here we present WaveGraph, a model based on Graph Neural Networks (GNNs) that emulates basin-scale wave dynamics directly on unstructured meshes with high resolution along the coasts (up to 2-3 km). Trained on bias-corrected simulation data over the Mediterranean Sea, WaveGraph uses a multiscale architecture combining the unstructured model mesh with a uniform graph, allowing simultaneous representation of local coastal interactions and large-scale wave dynamics. The model reconstructs the evolution of significant wave height, mean period, and mean direction, and is applied autoregressively for a continuous 17-year period without reinitialization or drift. Validation against buoy and satellite observations shows skill comparable to the input data set, and ablation experiments indicate that wind forcing drives most of the long-term stability while wave history improves swell-driven and basin-scale dynamics. These results show that GNNs can provide stable and efficient emulators of spectral wave models on unstructured domains, enabling decadal wave reconstructions.</span> <span class="abstract-toggle" data-id="2608.16449">more</span>

    <span class="paper-links">[:material-file-document: 2608.16449](https://arxiv.org/abs/2608.16449v1) · [:material-content-copy: BibTeX](bibtex/2608.16449.bib){ .bibtex-link }</span>

    <a class="md-tag" href="/tags/#gnn">GNN</a>

-   #### Rainfall Sensing via Mobile Communication Signals

    ---

    <span class="paper-meta"><em>Zhongqin Wang, J. Andrew Zhang, Kai Wu, Y. Jay Guo</em> · 2026</span>

    <span class="abstract-snippet" id="snip-2608.16088" data-search-exclude>Rainfall monitoring is important for hydrological observation, disaster warning, and environmental sensing, but conventional rain gauges and weather radars suffer from sparse deployment and high...</span><span class="abstract-full" id="full-2608.16088" data-search-exclude hidden>Rainfall monitoring is important for hydrological observation, disaster warning, and environmental sensing, but conventional rain gauges and weather radars suffer from sparse deployment and high infrastructure costs. This paper proposes PMN-RainSense, a rainfall sensing framework using sub-6-GHz mobile communication signals that supports practical single-antenna deployment. Unlike attenuation-based approaches, which are unreliable at sub-6 GHz because rain-induced attenuation over short mobile access links is only on the order of hundredths of a decibel, the proposed framework exploits fine-grained dynamics. A spectral-temporal channel state information (CSI) compensation method suppresses packet-wise timing and phase distortions while preserving sensing-relevant information. Rainfall-sensitive features are extracted from the delay-Doppler domain to mitigate environmental interference, with angle-domain filtering as an optional extension for multi-antenna receivers. Under bandwidth and antenna constraints, rainfall-correlated Doppler fluctuations serve as the dominant sensing signature, while Doppler-domain normalization improves robustness across links and deployments. Controlled WiFi experiments demonstrate rainfall-associated Doppler broadening and achieve a three-class classification accuracy of 95.48% using a random forest classifier. Long-Term Evolution (LTE) CSI measurements collected from cellular base stations over 11 carrier frequencies from 0.763 to 2.68 GHz yield a mean absolute error (MAE) of 0.25-0.27 mm/h for rainfall intensity estimation using a one-dimensional convolutional network.</span> <span class="abstract-toggle" data-id="2608.16088">more</span>

    <span class="paper-links">[:material-file-document: 2608.16088](https://arxiv.org/abs/2608.16088v1) · [:material-content-copy: BibTeX](bibtex/2608.16088.bib){ .bibtex-link }</span>

-   #### Generative data assimilation highlights fronts as key regulators of ocean energy cascade

    ---

    <span class="paper-meta"><em>Scott A. Martin, Georgy E. Manucharyan, Patrice Klein</em> · 2026</span>

    <span class="abstract-snippet" id="snip-2608.14955" data-search-exclude>Mesoscale eddies are fundamental to the ocean circulation, yet the extent to which submesoscale motions, a few kilometers across, influence mesoscale eddy energetics through a kinetic energy cascade...</span><span class="abstract-full" id="full-2608.14955" data-search-exclude hidden>Mesoscale eddies are fundamental to the ocean circulation, yet the extent to which submesoscale motions, a few kilometers across, influence mesoscale eddy energetics through a kinetic energy cascade remains uncertain. High-resolution simulations predict that submesoscale fronts are key regulators of the cascade, transferring energy both downscale towards dissipation and upscale to sustain and shape the seasonality of mesoscale eddies. Testing these predictions has remained difficult because existing observations and state estimates cannot resolve submesoscale currents over sufficiently broad domains. Here we map the ocean's submesoscale energy cascade by combining multi-source satellite observations with a generative deep learning framework, reconstructing gap-free, kilometer-scale surface currents with physically plausible dynamics learned from simulations. Applying this to the eddy-rich Agulhas Current system, we find that submesoscales energize the mesoscale through an upscale energy cascade above 10 km, contributing to the seasonality of mesoscale eddies. Below 10 km, convergence at submesoscale fronts drives a downscale cascade towards dissipation. Both upscale and downscale pathways concentrate within fronts, where cross-scale transfer is up to an order of magnitude more efficient. Despite their limited extent, fronts account for a substantial fraction of the domain-integrated cascade, establishing them as key regulators of the cascade and targets for next-generation eddy parameterizations.</span> <span class="abstract-toggle" data-id="2608.14955">more</span>

    <span class="paper-links">[:material-file-document: 2608.14955](https://arxiv.org/abs/2608.14955v1) · [:material-content-copy: BibTeX](bibtex/2608.14955.bib){ .bibtex-link }</span>

-   #### Developing an Offshore Machine Learning Surface Layer Scheme

    ---

    <span class="paper-meta"><em>Susan Dettling, Sue Ellen Haupt, Thomas Brummet, Patrick Hawbecker, Branko Kosović, David John Gagne</em> · 2026</span>

    <span class="abstract-snippet" id="snip-2608.14935" data-search-exclude>Turbulent fluxes between the surface and the atmosphere are typically parameterized using empirically fit relationships. Here we test machine learning techniques for fitting the relationship for the...</span><span class="abstract-full" id="full-2608.14935" data-search-exclude hidden>Turbulent fluxes between the surface and the atmosphere are typically parameterized using empirically fit relationships. Here we test machine learning techniques for fitting the relationship for the offshore environment. To do that, data from three offshore sites are used: the Martha's Vineyard Coastal Observatory (MVCO) air-sea interaction tower, the FINO1 research platform, and the CASPER-West FLIP research vessel deployed off the coast of California. Two machine learning methods were employed: Neural Networks (NN) and Random Forests (RF). Because the observational sites had towers with measurements at different levels, the vertical differences were input as gradients. Models were built for both momentum flux and heat flux. ML models trained at the individual sites were competitive with and in some cases, better than the physically-based COARE-3 model tailored to offshore fluxes. The heat flux ML models generally outperformed the physics-based parameterizations for most metrics, but the results were mixed for momentum flux, with only the site with the most training data (MVCO) producing results better than COARE-3. When the ML models from that site were applied to the other sites, results were degraded from using data from the site being tested. ML models built from data combined from the three sites generally showed improvements for the sites with less available training data. When assessing which variables were most important, the wind speed was most important for momentum flux and temperature gradient for heat flux.</span> <span class="abstract-toggle" data-id="2608.14935">more</span>

    <span class="paper-links">[:material-file-document: 2608.14935](https://arxiv.org/abs/2608.14935v1) · [:material-content-copy: BibTeX](bibtex/2608.14935.bib){ .bibtex-link }</span>

    <a class="md-tag" href="/tags/#physics-informed">physics-informed</a>

-   #### Meteorology-driven Causal Nowcasting of Fugitive Landfill Emissions Enables Proactive Public Health Response

    ---

    <span class="paper-meta"><em>Timothy C. Pearce, David J. T. Smith, Alec Dobney, Alessia Freddo</em> · 2026</span>

    <span class="abstract-snippet" id="snip-2608.14254" data-search-exclude>Fugitive emissions from waste sites increasingly expose communities to toxic and odorous gases, yet public-health responses remain largely retrospective, with episodes investigated only after...</span><span class="abstract-full" id="full-2608.14254" data-search-exclude hidden>Fugitive emissions from waste sites increasingly expose communities to toxic and odorous gases, yet public-health responses remain largely retrospective, with episodes investigated only after residents have been exposed. Here we show that the meteorological drivers of elevated hydrogen sulphide (HS) at a long-monitored European landfill, and the timescales over which they act, can be identified directly from routine monitoring data. We introduce CAIRN (Causal-Anchored Inference for Receptor Nowcasting), a machine-learning framework whose internal memory is matched to these measured timescales: a fast component tracking hour-scale wind-borne transport and a slow component tracking multi-hour weather changes. Trained to predict gas measurements, CAIRN operates using only routine weather variables and the calendar, without hand-engineered features. Its behaviour is consistent with the identified transport mechanisms, and the framework transfers unchanged to a second monitoring station and to co-emitted methane. Combining four such nowcasters produces a site-level, tiered alert aligned with WHO odour guidance that closely reproduces the alert generated by a direct sensor network and tracks an independent record of community odour complaints. Weather-driven nowcasting can therefore estimate community impact as an emission episode unfolds, providing public-health authorities with a validated, graded trigger for intervention and enabling exposure to be reduced during events rather than after them.</span> <span class="abstract-toggle" data-id="2608.14254">more</span>

    <span class="paper-links">[:material-file-document: 2608.14254](https://arxiv.org/abs/2608.14254v1) · [:material-content-copy: BibTeX](bibtex/2608.14254.bib){ .bibtex-link }</span>

-   #### Paleoclimate Boundary Conditions as an Out-of-Sample Test for the Forced Response of Ocean Climate Emulators

    ---

    <span class="paper-meta"><em>Adam Subel, Laure Zanna</em> · 2026</span>

    <span class="abstract-snippet" id="snip-2608.13494" data-search-exclude>AI weather emulators benefit from clear objectives and metrics, which have led to the rapid development of models that outperform traditional benchmarks. In contrast, long-term climate emulators must...</span><span class="abstract-full" id="full-2608.13494" data-search-exclude hidden>AI weather emulators benefit from clear objectives and metrics, which have led to the rapid development of models that outperform traditional benchmarks. In contrast, long-term climate emulators must reliably reproduce forced responses over months to centuries, while relying on training objectives that span a small number of model time steps. We assess autoregressive, full-depth ocean emulators using data from the midHolocene experiment of a numerical climate model to examine their skill in responding to surface forcings from an in-distribution, out-of-sample climate. We demonstrate that these emulators generalize to new orbital forcings, reproducing the spatial structure of the large-scale response as well as changes in seasonal patterns and in the spatial structure of ocean variability, while underestimating their amplitude. Baselines that infer the ocean state directly from the boundary forcings also recover much of the large-scale pattern, but only near the surface, and capture neither the seasonal nor the variability changes, indicating that these require some representation of dynamics. Despite these successes, the emulators fail to reproduce the slow, internally driven evolution of the ocean interior. We then show that the emulators' total forced response is well reconstructed by linearly composing their independent responses to each forcing component. Tracking response across training epochs, we find that convergence on mean state metrics in the training climate does not guarantee that the emulators capture the dynamics necessary for a skillful response. Together, these experiments establish the midHolocene as a controlled, ground-truthed setting for diagnosing forced-response failures before emulators are pushed to out-of-distribution climates.</span> <span class="abstract-toggle" data-id="2608.13494">more</span>

    <span class="paper-links">[:material-file-document: 2608.13494](https://arxiv.org/abs/2608.13494v1) · [:material-content-copy: BibTeX](bibtex/2608.13494.bib){ .bibtex-link }</span>

-   #### Machine learning correction of satellite precipitation is governed by mechanism purity, not algorithmic complexity: a proof-of-concept study in Hunan, China, with pre-registered cross-regional validation

    ---

    <span class="paper-meta"><em>Yi Xu</em> · 2026</span>

    <span class="abstract-snippet" id="snip-2608.12988" data-search-exclude>Satellite precipitation products such as IMERG exhibit biases that vary with terrain, season, and precipitation regime, leaving the applicability boundaries of machine learning correction unclear....</span><span class="abstract-full" id="full-2608.12988" data-search-exclude hidden>Satellite precipitation products such as IMERG exhibit biases that vary with terrain, season, and precipitation regime, leaving the applicability boundaries of machine learning correction unclear. This study proposes the Terrain-Moisture-Intensity (TMI) framework, centered on mechanism purity, extending the correction problem from purely algorithmic optimization to physical consistency diagnosis. A proof-of-concept study in Hunan Province employs IMERG V07, SRTM DEM, and ERA5 variables (tcwv, u10, v10). Ablation results indicate that, under the conditions of this study, terrain-moisture relationships are predominantly additive: RF-Full yields merely +0.001 R^2 gain over LR-Full, while bias rises to 1.282 mm d^-1; MAE decreases by approximately 14%, reflecting a trade-off between tail-fitting improvement and mean shift. SHAP diagnostics identify three categories of boundaries. Spatially, Central Hunan exhibits significant degradation (R^2=0.133) despite strong variable activation, consistent with mechanism fragmentation induced by mixed terrain. Temporally, u10 undergoes directional reversal between summer and spring (+0.096 to -0.156), presenting "silent failure." Extreme precipitation (>=50 mm d^-1) approximates a mechanism saturation frontier rather than isolated out-of-distribution samples, with DEM showing the largest relative amplification in SHAP disorder (+150%). The results demonstrate that machine learning correction performance is primarily constrained by mechanism purity. A pre-registered cross-regional test (Hunan, Guangxi, Guangdong) confirms this screening capability out of sample: a priori coherence proxies predict correction efficiency with a mean absolute error of 2.6 percentage points, while the transfer-versus-retraining contrast separates mechanism mismatch (coastal Guangdong) from portability (Guangxi), establishing the framework as a validated applicability screen.</span> <span class="abstract-toggle" data-id="2608.12988">more</span>

    <span class="paper-links">[:material-file-document: 2608.12988](https://arxiv.org/abs/2608.12988v1) · [:material-content-copy: BibTeX](bibtex/2608.12988.bib){ .bibtex-link }</span>

-   #### High-resolution Calibrated Probabilistic Hourly Precipitation from a Deterministic Forecast

    ---

    <span class="paper-meta"><em>Thomas M Hamill</em> · 2026</span>

    <span class="abstract-snippet" id="snip-2608.12685" data-search-exclude>An ``Attention Residual U-Net'' method is described for probabilistic quantitative precipitation forecasting (PQPF) that predicts the hourly probability of no precipitation plus the distribution of...</span><span class="abstract-full" id="full-2608.12685" data-search-exclude hidden>An ``Attention Residual U-Net'' method is described for probabilistic quantitative precipitation forecasting (PQPF) that predicts the hourly probability of no precipitation plus the distribution of positive precipitation from a weighted mixture of two Gamma distributions. The neural network is trained on patches of numerical weather prediction (NWP) hourly precipitation from The Weather Company's convection-permitting GRAF (Global high-Resolution Atmospheric Forecasting) model along with terrain information and column-average relative humidity from the National Oceanic and Atmospheric Administration's (NOAA's) Global Forecast System (GFS). The target data are NOAA's Multi-Radar, Multi-Sensor (MRMS) gauge-corrected, quality controlled radar data sampled to the same grid as the GRAF data. The network outputs distributional parameters for each model grid point. Training uses negative log-likelihood as a proper scoring rule, with climatological initialization for stable convergence. Inference is performed as a single forward pass over the contiguous United States (CONUS) domain, with edge-replication padding to satisfy the network's spatial-divisibility requirement. The subsequent forecasts are spatially detailed, highly reliable, and skillful with respect to climatology and a simpler reference forecast method. The method is particularly useful for estimating probabilities in regions with large terrain variation.</span> <span class="abstract-toggle" data-id="2608.12685">more</span>

    <span class="paper-links">[:material-file-document: 2608.12685](https://arxiv.org/abs/2608.12685v1) · [:material-content-copy: BibTeX](bibtex/2608.12685.bib){ .bibtex-link }</span>

    <a class="md-tag" href="/tags/#cnn">CNN</a> <a class="md-tag" href="/tags/#probabilistic">probabilistic</a>

-   #### Earth observation embeddings are effective sub-grid descriptors for probabilistic weather downscaling

    ---

    <span class="paper-meta"><em>Pedro Sousa, Will Tebbutt, Sadiq Jaffer, Robin Young, Anil Madhavapeddy, Richard E. Turner</em> · 2026</span>

    <span class="abstract-snippet" id="snip-2608.12271" data-search-exclude>Global weather reanalyses and forecasts resolve the evolving atmospheric state on coarse grids, but site-specific applications require predictions at arbitrary locations where near-surface conditions...</span><span class="abstract-full" id="full-2608.12271" data-search-exclude hidden>Global weather reanalyses and forecasts resolve the evolving atmospheric state on coarse grids, but site-specific applications require predictions at arbitrary locations where near-surface conditions also depend on unresolved terrain and land-surface properties. Existing probabilistic downscalers address this gap using hand-crafted topographic descriptors. We ask instead whether Earth observation foundation models can provide transferable sub-grid surface representations for probabilistic weather downscaling.   We augment a convolutional conditional neural process that downscales coarse ERA5 reanalysis fields at ~25 km resolution with a learned local surface descriptor, obtained by compressing a patch of TESSERA embeddings at 10 m resolution. Although these embeddings summarise surface conditions over annual timescales, they improve downscaling of instantaneous 2 m temperature and 10 m wind speed by encoding persistent surface properties that capture a location's departure from the coarse-grid atmospheric state. Across five climatically diverse regions, the embedding improves point and probabilistic skill at stations held out in both space and time, overall improving CRPS skill by 11.5% for 2 m temperature and 6.2% for 10 m wind speed. We further analyse how its contribution differs by variable, finding that topography explains more of temperature's sub-grid structure, while TESSERA provides additional surface information for wind speed.   These improvements persist when the coarse input is changed from ERA5 to forecasts from the Aurora AI forecasting model, and when predicting at newly deployed stations with no regional history. To our knowledge, this is the first evidence that long-timescale Earth-observation embeddings can support short-timescale weather downscaling where sub-grid departures are systematically structured by persistent surface properties.</span> <span class="abstract-toggle" data-id="2608.12271">more</span>

    <span class="paper-links">[:material-file-document: 2608.12271](https://arxiv.org/abs/2608.12271v1) · [:material-content-copy: BibTeX](bibtex/2608.12271.bib){ .bibtex-link }</span>

    <a class="md-tag" href="/tags/#foundation-model">foundation-model</a> <a class="md-tag" href="/tags/#probabilistic">probabilistic</a>

-   #### DLESyM-Ocean: A Deep Learning Probabilistic Global Model for Simulating Present-Day Upper Ocean and Sea Ice

    ---

    <span class="paper-meta"><em>Zachary I Espinosa, Nathaniel Cresswell-Clay, William Yik, Cecilia M. Bitz et al.</em> · 2026</span>

    <span class="abstract-snippet" id="snip-2608.11545" data-search-exclude>While AI has shown remarkable promise in atmospheric and meteorological forecasting, accurately simulating other components of the Earth system with AI remains an active frontier. We present...</span><span class="abstract-full" id="full-2608.11545" data-search-exclude hidden>While AI has shown remarkable promise in atmospheric and meteorological forecasting, accurately simulating other components of the Earth system with AI remains an active frontier. We present DLESyM-Ocean, a Deep Learning Earth System Model that simulates global present-day sea ice and upper ocean conditions. Unlike conventional probabilistic models optimized via diffusion objectives or losses such as continuous-ranked probability score, DLESyM-Ocean is trained using a patch energy score loss. When driven by atmospheric forcing, DLESyM-Ocean produces a well-calibrated, spatially coherent, and skillful ensemble of sea ice and upper ocean conditions with minimal bias relative to reanalysis products. DLESyM-Ocean is stable when autoregressively run for multi-year simulations and produces a climatology and variability with minimal bias compared with reanalysis. We evaluate case studies including a recent sea ice extreme, a severe marine heatwave, the 2023 El Niño transition, and the 2023 spike in global mean temperature. In all of these case studies, DLESyM-Ocean produces realistic surface and subsurface trajectories and ample ensemble diversity in response to common atmospheric forcing, suggestive of learned autoregressive ocean dynamics. When coupled with other Earth system components, such as the atmosphere, the computational efficiency of DLESyM-Ocean makes it a promising tool for subseasonal to seasonal forecasting.</span> <span class="abstract-toggle" data-id="2608.11545">more</span>

    <span class="paper-links">[:material-file-document: 2608.11545](https://arxiv.org/abs/2608.11545v1) · [:material-content-copy: BibTeX](bibtex/2608.11545.bib){ .bibtex-link }</span>

    <a class="md-tag" href="/tags/#probabilistic">probabilistic</a>

</div>

