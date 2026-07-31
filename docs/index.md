---
hide:
  - navigation
title: Deep Learning in Weather
---

A collection of papers on deep learning and machine learning applied to weather forecasting, climate modeling, and atmospheric science.

*Last updated: 2026-07-31*

## Starred Papers

<div class="grid cards" markdown>

-   #### :material-star: AIFS-DOP: End-to-End Medium-Range Weather Prediction from Observations Alone with Machine Learning

    ---

    *Ewan Pinnington, Peter Lean, Mihai Alexe, Eulalie Boucher, Simon Lang, Patrick Laloyaux et al.* · 2026

    <span class="abstract-snippet" id="snip-2606.19093">We introduce the Artificial Intelligence Forecasting System for Direct Observation Prediction (AIFS-DOP). AIFS-DOP is trained on a 40-year harmonized dataset of gridded observations, without using...</span><span class="abstract-full" id="full-2606.19093" hidden>We introduce the Artificial Intelligence Forecasting System for Direct Observation Prediction (AIFS-DOP). AIFS-DOP is trained on a 40-year harmonized dataset of gridded observations, without using numerical weather prediction (NWP) reanalysis or model data. The resulting model is competitive with ECMWF's Integrated Forecasting System (IFS) when scored on a one year period of forecasts across 2021/2022. This progress on Direct Observation Prediction represents the first time that a data-driven model, trained solely on observations, is competitive with the IFS at medium ranges for several key upper-air and surface headline scores, when verified against observation data.</span> <span class="abstract-toggle" data-id="2606.19093">more</span>

    [:material-file-document: 2606.19093](https://arxiv.org/abs/2606.19093v1) · [:material-content-copy: BibTeX](bibtex/2606.19093.bib){ .bibtex-link }

-   #### :material-star: (Sparse) Attention to the Details: Preserving Spectral Fidelity in ML-based Weather Forecasting Models

    ---

    *Maksim Zhdanov, Ana Lucic, Max Welling, Jan-Willem van de Meent* · 2026

    <span class="abstract-snippet" id="snip-2604.16429">We introduce Mosaic, a probabilistic weather forecasting model that addresses three failure modes of spectral degradation in ML-based weather prediction: spectral damping (statistical),...</span><span class="abstract-full" id="full-2604.16429" hidden>We introduce Mosaic, a probabilistic weather forecasting model that addresses three failure modes of spectral degradation in ML-based weather prediction: spectral damping (statistical), high-frequency aliasing (architectural), and residual high-frequency leakage (parametric). Mosaic generates ensemble members through learned functional perturbations and operates on native-resolution grids via mesh-aligned block-sparse attention, a hardware-aligned mechanism that captures long-range dependencies at linear cost by sharing keys and values across spatially adjacent queries. At 1.5° resolution with 214M parameters, Mosaic matches or outperforms models trained on 6$\times$ finer resolution on key variables and achieves state-of-the-art results among 1.5° models, producing well-calibrated ensembles whose individual members exhibit near-perfect spectral alignment across all resolved frequencies. A 24-member, 10-day forecast takes under 12s on a single H100~GPU. Code is available at https://github.com/maxxxzdn/mosaic.</span> <span class="abstract-toggle" data-id="2604.16429">more</span>

    [:material-file-document: 2604.16429](https://arxiv.org/abs/2604.16429v3) · [:fontawesome-brands-github:](https://github.com/maxxxzdn/mosaic) · [:material-content-copy: BibTeX](bibtex/2604.16429.bib){ .bibtex-link }

    <span class="md-tag">probabilistic</span>

-   #### :material-star: U-Cast: A Surprisingly Simple and Efficient Frontier Probabilistic AI Weather Forecaster

    ---

    *Salva Rühling Cachay, Duncan Watson-Parris, Rose Yu* · 2026

    <span class="abstract-snippet" id="snip-2604.09041">AI-based weather forecasting now rivals traditional physics-based ensembles, but state-of-the-art (SOTA) models rely on specialized architectures and massive computational budgets, creating a high...</span><span class="abstract-full" id="full-2604.09041" hidden>AI-based weather forecasting now rivals traditional physics-based ensembles, but state-of-the-art (SOTA) models rely on specialized architectures and massive computational budgets, creating a high barrier to entry. We demonstrate that such complexity is unnecessary for frontier performance. We introduce \ours, a probabilistic forecaster built on a standard U-Net backbone trained with a simple recipe: deterministic pre-training on Mean Absolute Error followed by short probabilistic fine-tuning on the Continuous Ranked Probability Score (CRPS) using Monte Carlo Dropout for stochasticity. As a result, our model matches or exceeds the probabilistic skill of GenCast and IFS ENS at $1.5^\circ$ resolution while reducing training compute by over $10\times$ compared to leading CRPS-based models and inference latency by over $10\times$ compared to diffusion-based models. U-Cast trains in under 12 H200 GPU-days and generates a 15-day ensemble forecast in 3 seconds. These results suggest that scalable, general-purpose architectures paired with efficient training curricula can match complex domain-specific designs at a fraction of the cost, opening the training of frontier probabilistic weather models to the broader community.</span> <span class="abstract-toggle" data-id="2604.09041">more</span>

    [:material-file-document: 2604.09041](https://arxiv.org/abs/2604.09041v2) · [:fontawesome-brands-github:](https://github.com/Rose-STL-Lab/u-cast) · [:material-content-copy: BibTeX](bibtex/2604.09041.bib){ .bibtex-link }

    <span class="md-tag">CNN</span> <span class="md-tag">physics-informed</span> <span class="md-tag">probabilistic</span>

-   #### :material-star: Using data assimilation tools to dissect GraphDOP

    ---

    *Patrick Laloyaux, Mihai Alexe, Eulalie Boucher, Peter Lean, Ewan Pinnington, Simon Lang et al.* · 2025

    <span class="abstract-snippet" id="snip-2510.27388">The Data Assimilation (DA) community has been developing various diagnostics to understand the importance of the observing system in accurately forecasting the weather. They usually rely on the...</span><span class="abstract-full" id="full-2510.27388" hidden>The Data Assimilation (DA) community has been developing various diagnostics to understand the importance of the observing system in accurately forecasting the weather. They usually rely on the ability to compute the derivatives of the physical model output with respect to its initial condition. For example, the Forecast Sensitivity-based Observation Impact (FSOI) estimates the impact on the forecast error of each observation processed in the DA system. This paper presents how these DA diagnostic tools are transferred to Machine Learning (ML) models, as their derivatives are readily available through automatic differentiation. We specifically explore the interpretability and explainability of the observation-driven GraphDOP model developed at the European Centre for Medium-Range Weather Forecasts (ECMWF). The interpretability study demonstrates the effectiveness of GraphDOP's sliding attention window to learn the meteorological features present in the observation datasets and to learn the spatial relationships between different regions. Making these relationships more transparent confirms that GraphDOP captures real, physically meaningful processes, such as the movement of storm systems. The explainability of GraphDOP is explored by applying the FSOI tool to study the impact of the different observations on the forecast error. This inspection reveals that GraphDOP creates an internal representation of the Earth system by combining the information from conventional and satellite observations.</span> <span class="abstract-toggle" data-id="2510.27388">more</span>

    [:material-file-document: 2510.27388](https://arxiv.org/abs/2510.27388v1) · [:material-content-copy: BibTeX](bibtex/2510.27388.bib){ .bibtex-link }

-   #### :material-star: GraphDOP: Towards skilful data-driven medium-range weather forecasts learnt and initialised directly from observations

    ---

    *Mihai Alexe, Eulalie Boucher, Peter Lean, Ewan Pinnington, Patrick Laloyaux, Anthony McNally et al.* · 2024

    <span class="abstract-snippet" id="snip-2412.15687">We introduce GraphDOP, a new data-driven, end-to-end forecast system developed at the European Centre for Medium-Range Weather Forecasts (ECMWF) that is trained and initialised exclusively from Earth...</span><span class="abstract-full" id="full-2412.15687" hidden>We introduce GraphDOP, a new data-driven, end-to-end forecast system developed at the European Centre for Medium-Range Weather Forecasts (ECMWF) that is trained and initialised exclusively from Earth System observations, with no physics-based (re)analysis inputs or feedbacks. GraphDOP learns the correlations between observed quantities - such as brightness temperatures from polar orbiters and geostationary satellites - and geophysical quantities of interest (that are measured by conventional observations), to form a coherent latent representation of Earth System state dynamics and physical processes, and is capable of producing skilful predictions of relevant weather parameters up to five days into the future.</span> <span class="abstract-toggle" data-id="2412.15687">more</span>

    [:material-file-document: 2412.15687](https://arxiv.org/abs/2412.15687v1) · [:material-content-copy: BibTeX](bibtex/2412.15687.bib){ .bibtex-link }

    <span class="md-tag">physics-informed</span>

</div>

## Recent Additions

<div class="grid cards" markdown>

-   #### Memory compression and physical state augmentation favor different AMOC prediction tasks

    ---

    *Mauricio Herrera-Marín* · 2026

    <span class="abstract-snippet" id="snip-2607.28468">The Atlantic Meridional Overturning Circulation is monitored and emulated through reduced indices, but such projections discard thermohaline structure and may require either explicit physical state...</span><span class="abstract-full" id="full-2607.28468" hidden>The Atlantic Meridional Overturning Circulation is monitored and emulated through reduced indices, but such projections discard thermohaline structure and may require either explicit physical state or memory of the observed index. We compare these strategies in 30 branch-consistent CMIP6 trajectories from eight model families using leave-one-family-out validation. Salinity, temperature and density information improves direct 20-year forecasts, whereas compact scalar memory is top-ranked at every recursive horizon and yields the lowest case-averaged Brier score. A matched ablation confirms that feedback from memory improves long-horizon prediction. Physical state and recent trends also predict future ocean-state changes beyond the emissions pathway, most robustly at five years. NorESM under SSP5--8.5 identifies a forcing-dependent limit of scalar compression, while MIROC shows negative long-horizon transfer. A resolvent analysis explains why stable memory components do not guarantee stability of the complete learned model. Physical augmentation and memory compression therefore serve different AMOC prediction tasks.</span> <span class="abstract-toggle" data-id="2607.28468">more</span>

    [:material-file-document: 2607.28468](https://arxiv.org/abs/2607.28468v1) · [:material-content-copy: BibTeX](bibtex/2607.28468.bib){ .bibtex-link }

-   #### Weather Emulators at the Frontier of Heat Extremes Predictability

    ---

    *Cas Decancq, Thomas Mortier, Jessica Keune, Diego G. Miralles* · 2026

    <span class="abstract-snippet" id="snip-2607.28220">Atmospheric predictability declines rapidly beyond the next ten days, such that forecasts at longer lead times primarily convey large-scale trends rather than specific states. Yet in a warming world,...</span><span class="abstract-full" id="full-2607.28220" hidden>Atmospheric predictability declines rapidly beyond the next ten days, such that forecasts at longer lead times primarily convey large-scale trends rather than specific states. Yet in a warming world, improving early warnings of extreme heat is an increasingly critical challenge. Here we evaluate six state-of-the-art deep learning weather emulators - Pangu-Weather, FuXi, ArchesWeather, AIFS, GraphCast and Aurora - alongside leading dynamical systems and statistical baselines in forecasting global near-surface temperature and extreme heat at lead times of 10-15 days. We find that several emulators rival or even surpass physics-based forecasts in deterministic temperature skill, but do so at the cost of reduced spectral fidelity, in a process widely known as blurring. While all models show some degree of predictive skill for extreme heat, most emulators under-represent peak intensities, and IFS recall is greater than that of any of the emulators. These results highlight both the emerging potential of AI to enhance extended range temperature prediction, and the remaining challenges in delivering reliable, actionable early warnings in a changing climate.</span> <span class="abstract-toggle" data-id="2607.28220">more</span>

    [:material-file-document: 2607.28220](https://arxiv.org/abs/2607.28220v1) · [:material-content-copy: BibTeX](bibtex/2607.28220.bib){ .bibtex-link }

    <span class="md-tag">physics-informed</span>

-   #### Meteosat Third Generation imagery improves CNN-based SSI retrieval

    ---

    *Gordei Pribõtkin, Piia Post, Velle Toll* · 2026

    <span class="abstract-snippet" id="snip-2607.28093">Accurate Surface Solar Irradiance (SSI) estimation is increasingly important for photovoltaic energy monitoring and forecasting. The recently introduced Meteosat Third Generation (MTG) satellite...</span><span class="abstract-full" id="full-2607.28093" hidden>Accurate Surface Solar Irradiance (SSI) estimation is increasingly important for photovoltaic energy monitoring and forecasting. The recently introduced Meteosat Third Generation (MTG) satellite constellation provides imaging data with higher spatial resolution compared to the Meteosat Second Generation (MSG) satellite constellation, but its benefits for machine-learning-based SSI retrieval have not been well established. In this work, we introduce a multi-imager and multi-resolution convolutional neural network architecture for 10-minute SSI retrieval over Northern Europe (Estonia) using MSG/SEVIRI and MTG/FCI satellite imagery together with solar-geometry and clear-sky irradiance features. Model performance is evaluated against ground-based pyranometer measurements from eight Estonian meteorological stations using site-based cross-validation and multiple training seeds. Model performance is also compared with the SARAH-3 physics-based satellite SSI product. The hybrid SEVIRI-FCI model significantly outperformed the SEVIRI-only model under overcast and cloudy conditions, reducing RMSE by 8.2 W m$^{-2}$ and 5.7 W m$^{-2}$, respectively. However, under partly cloudy or clear skies, no statistically significant difference in RMSE was observed between the SEVIRI-FCI hybrid and the SEVIRI-only models. Compared with physics-based SARAH-3, the hybrid model yielded skill scores of 35 % under overcast conditions, 21 % under cloudy conditions, and 20 % overall. Furthermore, both models underperformed SARAH-3 in clear-sky conditions. These results show that higher-resolution MTG/FCI imagery improves CNN-based SSI retrieval when clouds dominate irradiance variability, but also indicate that higher spatial resolution alone is insufficient to address clear-sky limitations in machine-learning-based SSI retrieval.</span> <span class="abstract-toggle" data-id="2607.28093">more</span>

    [:material-file-document: 2607.28093](https://arxiv.org/abs/2607.28093v1) · [:material-content-copy: BibTeX](bibtex/2607.28093.bib){ .bibtex-link }

    <span class="md-tag">CNN</span> <span class="md-tag">physics-informed</span>

-   #### Benchmarking ConvLSTM for One-Day-Ahead IMDAA Rainfall-Field Prediction across Four Indian Cities

    ---

    *Tanmay Ghosh, Shaurabh Anand, Rakesh Gomaji Nannewar, Nithin Nagaraj* · 2026

    <span class="abstract-snippet" id="snip-2607.26581">Convolutional long short-term memory networks (ConvLSTMs) are widely used for precipitation forecasting, but most evidence for their performance comes from dense, high-frequency radar sequences. This...</span><span class="abstract-full" id="full-2607.26581" hidden>Convolutional long short-term memory networks (ConvLSTMs) are widely used for precipitation forecasting, but most evidence for their performance comes from dense, high-frequency radar sequences. This study tests whether convolutional recurrence improves one-day-ahead rainfall-field prediction on small daily reanalysis grids. Indian Monsoon Data Assimilation and Analysis (IMDAA) fields for June-September 1998-2020 were analysed for Bengaluru, Delhi, Kolkata and Mumbai. Ten naive, statistical, tree-based and neural approaches were compared using atmospheric-only and rainfall-history-plus-atmospheric inputs. Performance was assessed for complete fields, domain-mean rainfall, spatial anomalies and high-rainfall days.   ConvLSTM did not consistently outperform simpler alternatives. FC-LSTM produced the numerically lowest domain-mean rainfall error in Bengaluru, Kolkata and Mumbai, whereas persistence performed best in Delhi. ConvLSTM produced the numerically lowest spatial-anomaly error only in Mumbai, where rainfall fields showed greater short-term spatial continuity and rainfall-history inputs improved all three neural architectures. The difference between ConvLSTM and FC-LSTM was nevertheless small. Neural models underestimated rainfall magnitude and predicted too few threshold exceedances on high-rainfall days, while persistence achieved the highest detection performance in every city. Post-hoc analyses showed that the selected models were most sensitive to the latest input day, with broader recent-lag sensitivity in Mumbai. These findings show that gridded inputs alone do not justify ConvLSTM and that architecture choice should follow strong benchmarking across average, spatial and high-rainfall performance.</span> <span class="abstract-toggle" data-id="2607.26581">more</span>

    [:material-file-document: 2607.26581](https://arxiv.org/abs/2607.26581v1) · [:material-content-copy: BibTeX](bibtex/2607.26581.bib){ .bibtex-link }

    <span class="md-tag">recurrent</span>

-   #### From Conceptual Hydrologic Models to Conceptually Interpretable Neural Networks: A Snow-Water Mass-Conserving-Perceptron Framework for Discovering Catchment-Scale Precipitation-Storage-Runoff Representations

    ---

    *Yuan-Heng Wang, Hoshin V. Gupta* · 2026

    <span class="abstract-snippet" id="snip-2607.26492">The Mass-Conserving Perceptron (MCP) establishes a modeling paradigm in which conceptual hydrologic models can be reformulated as physically constrained, conceptually interpretable neural networks....</span><span class="abstract-full" id="full-2607.26492" hidden>The Mass-Conserving Perceptron (MCP) establishes a modeling paradigm in which conceptual hydrologic models can be reformulated as physically constrained, conceptually interpretable neural networks. Here, we develop a snow-water MCP network framework and evaluate it across 513 CAMELS-US basins. We first recast a coupled two-state SOIL-MCP and SNOWMCP conceptual model as a mass-conserving neural network and show that the hydrologic-model and neural-network formulations achieve comparable predictive performance. We then examine cross-node state-information sharing within two-state HYDROMCP architectures and evaluate broader single-layer networks constructed from three types of interpretable MCP units with one to five states. Across CONUS, the median KGEss increases from 0.82 for one-state networks to 0.89 for two-state networks and 0.90 for five-state networks, suggesting diminishing aggregate gains beyond two states. Basin-specific MCP and LSTM selection yields the same median KGEss of 0.90, while the selected MCP networks use fewer parameters on average. Complementary AIC- and KGE-based selection identifies compact, basin-specific directed-graph representations that balance predictive accuracy and model complexity. These analyses provide an empirical basis for identifying the numbers, types, and interactions of states needed for hydrologic representation. Future studies should test joint training against multiple hydrologic responses, such as streamflow, snow water equivalent, and groundwater storage.</span> <span class="abstract-toggle" data-id="2607.26492">more</span>

    [:material-file-document: 2607.26492](https://arxiv.org/abs/2607.26492v1) · [:material-content-copy: BibTeX](bibtex/2607.26492.bib){ .bibtex-link }

    <span class="md-tag">recurrent</span>

-   #### From Heat Stress to Perception: Interpretable Data-Driven Models of Human Thermal Sensation

    ---

    *Abed Hammoud, Xinjie Huang, Qinqin Kong, Marialena Nikolopoulou, Elie Bou-Zeid* · 2026

    <span class="abstract-snippet" id="snip-2607.25850">Heat stress indices are designed to quantify physiological thermal stress, but their relevance for inferring the thermal perception of individuals remains unclear. In this study, we show that thermal...</span><span class="abstract-full" id="full-2607.25850" hidden>Heat stress indices are designed to quantify physiological thermal stress, but their relevance for inferring the thermal perception of individuals remains unclear. In this study, we show that thermal stress and thermal sensation often diverge, as evidenced by distinct global sensitivity patterns with respect to environmental drivers. Using thermal sensation vote survey data, we demonstrate that the dominant sensitivities of stress-based metrics do not align with those governing reported human thermal sensation. Given the multitude of globally-applicable thermal stress indices and the lack of comparable general thermal sensation metrics, we develop two complementary data-driven modeling frameworks for thermal sensation. First, we construct polynomial chaos expansion (PCE) surrogates to represent thermal sensation as a function of meteorological variables, enabling efficient variance-based sensitivity analysis and explicit identification of influential inputs and interactions. Second, we develop multilayer perceptron (MLP) classifiers that capture the nonlinear and subjective nature of thermal perception, while achieving high predictive accuracy. The PCE models provide physically interpretable sensitivities that can explain the drivers of thermal sensation, while the MLPs offer flexible predictive capability suited to complex environments. We apply both modeling approaches at city- and continent-scales, revealing systematic differences in sensitivity structure and performance across climates. In particular, we find that the sensitivity of TSV-based models to the variability of meteorological conditions across geoclimatic zone encodes distinct dependencies on temperature, radiation, humidity, and wind that vary geographically, and are generally different from those of heat stress indices.</span> <span class="abstract-toggle" data-id="2607.25850">more</span>

    [:material-file-document: 2607.25850](https://arxiv.org/abs/2607.25850v1) · [:material-content-copy: BibTeX](bibtex/2607.25850.bib){ .bibtex-link }

-   #### A Physics-Informed Neural Operator for Thermal Ranking of Low-Cost Wall Materials in Hot-Dry Climates

    ---

    *Muhammad Akbar Khan, Fahim Raees, Ubaida Fatima* · 2026

    <span class="abstract-snippet" id="snip-2607.25668">Identifying cost-effective indigenous building materials that minimise heat penetration through walls is critical for indoor thermal comfort in low-income rural housing in hot-dry climates, where...</span><span class="abstract-full" id="full-2607.25668" hidden>Identifying cost-effective indigenous building materials that minimise heat penetration through walls is critical for indoor thermal comfort in low-income rural housing in hot-dry climates, where summer temperatures routinely exceed 45 C. We present a two-stage computational framework for thermal ranking of five low-cost indigenous wall materials: mud brick, clay-straw adobe, lime-stabilised bamboo panel, fired clay brick, and lime-mud composite. First, a validated Crank-Nicolson finite difference method (FDM) solves the one-dimensional transient heat equation with Robin boundary conditions under diurnal solar and outdoor air-temperature forcing, generating 1500 periodic-day solutions across a nine-dimensional parameter space by Latin Hypercube sampling. Second, a Physics-Informed Neural Operator (PINO) with a Fourier Neural Operator (FNO) backbone learns the parameter-to-solution operator mu -> T(x,t), enforcing both data fidelity and PDE consistency. The trained PINO attains a relative L2 field error of 5.14e-4 and a 0.201 K mean absolute error on the peak inner surface temperature, preserving the FDM material ranking exactly; PINO trained on 150 FDM samples matches a data-only FNO trained on twice as many, so the physics loss is most valuable when data are scarce. The periodic-day formulation also yields the ISO 13786 time lag and decrement factor, reproduced to within 0.99 h and 0.010. At nominal hot-dry summer conditions, clay-straw adobe achieves the best cost-performance index among widely available materials. A climate sweep, confirmed by FDM spot checks, reveals a regime boundary: under sub-ambient outdoor conditions the ranking inverts to conductive fired clay brick, delineating heat-exclusion and heat-rejection regimes. The framework supports evidence-based material selection for post-flood reconstruction in hot-dry regions.</span> <span class="abstract-toggle" data-id="2607.25668">more</span>

    [:material-file-document: 2607.25668](https://arxiv.org/abs/2607.25668v1) · [:material-content-copy: BibTeX](bibtex/2607.25668.bib){ .bibtex-link }

    <span class="md-tag">physics-informed</span> <span class="md-tag">operator-learning</span>

-   #### Anomalous Diffusion of Tropical Cyclones Observed in Huge Ensembles of Hindcasts

    ---

    *Abdoul R. Zeba, William D. Collins, Ankur Mahesh, Boris Bonev, Karthik Kashinath, Thorsten Kurth et al.* · 2026

    <span class="abstract-snippet" id="snip-2607.21954">We examine whether tropical cyclones (TCs) obey ordinary Brownian or anomalous diffusion using a huge ensemble (HENS) of hindcasts for summer 2023. Anomalous diffusion has been inferred for actual...</span><span class="abstract-full" id="full-2607.21954" hidden>We examine whether tropical cyclones (TCs) obey ordinary Brownian or anomalous diffusion using a huge ensemble (HENS) of hindcasts for summer 2023. Anomalous diffusion has been inferred for actual TCs from the fluctuations in their tracks from the shortest paths between the initiation and termination of each cyclone. We reproduce the same anomalous diffusion power laws connecting spatial position and time using HENS. In addition, we show that the variance in the position of a single TC across HENS since initiation follows a scaling law with time that, in some cases, corresponds to ballistic motion of the TC through the background atmospheric flow. This determination was enabled by the exceptional statistics determined from thousands of plausible yet counterfactual recreations of 34 individual TCs. HENS consists of 7424 15-day hindcasts initiated from observed atmospheric conditions each day from June 1, 2023 to August 31, 2023 using the ECMWF ERA5 meteorological reanalysis. The hindcasts were generated using NVIDIA's Spherical Fourier Neural Operator (SFNO) machine-learning-based weather and climate emulator. We identify tropical cyclones in HENS using a variant of the Tempest Extremes detection and tracking frameworks for TCs with adjustments to the disposable parameters to minimize the numbers of false positives and negatives relative to the International Best Track Archive for Climate Stewardship (IBTrACS) records for TCs observed in summer 2023. We conclude with the implications of our findings for the predictability of TC tracks and landfall locations on lead times of days to weeks.</span> <span class="abstract-toggle" data-id="2607.21954">more</span>

    [:material-file-document: 2607.21954](https://arxiv.org/abs/2607.21954v1) · [:material-content-copy: BibTeX](bibtex/2607.21954.bib){ .bibtex-link }

    <span class="md-tag">operator-learning</span>

-   #### MAPCast: A Convection Allowing MPAS Emulator for Ensemble-based Background Error Covariance Estimation Toward Multi-Scale Data Assimilation

    ---

    *Yongming Wang, Xuguang Wang* · 2026

    <span class="abstract-snippet" id="snip-2607.21917">Machine learning (ML) emulators offer a cost-efficient alternative to numerical weather prediction models for generating convection-allowing background ensembles in ensemble-based data assimilation...</span><span class="abstract-full" id="full-2607.21917" hidden>Machine learning (ML) emulators offer a cost-efficient alternative to numerical weather prediction models for generating convection-allowing background ensembles in ensemble-based data assimilation (DA). However, few studies have explored ML-based surrogate background ensembles for estimating background-error covariances (BECs). This study develops a convection-allowing emulator, MAPCast, trained on historical convection-allowing simulations from the Model for Prediction Across Scales (MPAS), and evaluates its ability to estimate BECs, paving the way toward multiscale DA. The evaluation uses 10 retrospective convective cases at 15- and 60-min forecast lead times corresponding to subhourly and hourly DA. MAPCast reproduces MPAS forecasts with good fidelity, including realistic storm coverage, temporal evolution, and similar spatial and spectral characteristics of state variables. Discrepancies are primarily confined to small spatial scales near sharp gradients and convective-scale features and variables. For BEC statistics, MAPCast captures ensemble spread magnitude and spatial distribution for most variables, although larger errors occur for storm-related fields that are vertical velocity and reflectivity. Correlation structures are reproduced most faithfully at mesoscale and above, followed by at convective scales, whereas cross-variable correlations are less accurately represented than univariate correlations, indicating that multivariate coupling remains the principal limitation. MAPCast shows weaker replication of full-scale versus decomposed large and small-scale correlations. BEC estimates derived from 15-min forecasts consistently outperform those from 60-min forecasts, suggesting that shorter lead times better preserve flow-dependent error structures.</span> <span class="abstract-toggle" data-id="2607.21917">more</span>

    [:material-file-document: 2607.21917](https://arxiv.org/abs/2607.21917v1) · [:material-content-copy: BibTeX](bibtex/2607.21917.bib){ .bibtex-link }

-   #### Flexible generation of daily Earth system model projections across radiative forcing scenarios

    ---

    *Yu Huang, Sebastian Bathiany, Shangshang Yang, Philipp Hess, Michael Aich, Niklas Boers* · 2026

    <span class="abstract-snippet" id="snip-2607.21382">Earth system model (ESM) projections of the climate system's response to anthropogenic forcing are central to assess the impacts of climate change and inform adaptation and mitigation policies....</span><span class="abstract-full" id="full-2607.21382" hidden>Earth system model (ESM) projections of the climate system's response to anthropogenic forcing are central to assess the impacts of climate change and inform adaptation and mitigation policies. However, given their high computational cost, projections are only made for a limited set of standardized forcing scenarios with limited temporal extent, such as the Shared Socioeconomic Pathways (SSPs), the spatiotemporal resolution remains too low for direct impact assessments, and uncertainties cannot be comprehensively quantified. Recent data-driven models offer efficient and accurate high-resolution simulations for weather prediction, but cannot extrapolate to future greenhouse gas concentrations because they cannot capture the responses to unprecedented forcing, limiting their value for climate change projections. Here, we combine response theory with a tailored generative machine learning framework to address this challenge. Our approach extracts the physical forced response to radiative forcing from monthly low-resolution ESM fields, and uses this response to guide a generative model to infer consistent daily global high-resolution temperature and precipitation projections. Our probabilistic approach generalizes across ESMs and provides long-term, bias-corrected responses to radiative forcing at high spatiotemporal resolution. It efficiently generates large ensembles needed for uncertainty quantification, effectively fills the gaps between existing SSPs, and readily extends climate projections to 2300 and beyond. Our framework hence complements ESM projections by providing efficient, stable, and high spatiotemporal resolution long-term climate projection ensembles across emission scenarios, enabling detailed impact assessment and exploration of long-term climate commitment.</span> <span class="abstract-toggle" data-id="2607.21382">more</span>

    [:material-file-document: 2607.21382](https://arxiv.org/abs/2607.21382v1) · [:material-content-copy: BibTeX](bibtex/2607.21382.bib){ .bibtex-link }

    <span class="md-tag">probabilistic</span>

</div>

