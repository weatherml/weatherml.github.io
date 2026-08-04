---
hide:
  - navigation
title: Deep Learning in Weather
---

A collection of papers on deep learning and machine learning applied to weather forecasting, climate modeling, and atmospheric science.

*Last updated: 2026-08-04*

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

-   #### Probabilistic Deep Learning for Drought Forecasting: Role of Internal Climate Variability

    ---

    *Henri Funk, Cornelia Gruber, Göran Kauermann, Helmut Küchenhoff, Magdalena Mittermeier* · 2026

    <span class="abstract-snippet" id="snip-2608.01864">Predicting drought risk is essential for anticipating impacts on water resources, agriculture, ecosystems, and climate adaptation planning. Yet drought forecasts remain uncertain because variability...</span><span class="abstract-full" id="full-2608.01864" hidden>Predicting drought risk is essential for anticipating impacts on water resources, agriculture, ecosystems, and climate adaptation planning. Yet drought forecasts remain uncertain because variability can substantially alter regional precipitation and evaporative demand. Treating this variability as unstructured noise ignores the fact that internal variability has spatial, seasonal, and temporal structure and thus contains information that can be used to improve drought forecasting. We propose a deep-learning-based forecasting framework for European drought prediction and extend it with an uncertainty-aware drought bound that explicitly incorporates internal forecast variability from a large climate model ensemble. This bound represents a physically plausible lower-tail trajectory of future drought conditions and marks how severe drought could plausibly become under an unfavourable realisation of internal variability, giving adaptation planning a conservative, risk-averse reference. We compare the proposed bound with a lower bound derived from reanalysis data only and show that our proposed ensemble-informed bound is better calibrated across most regions and seasons. This is specifically true during anomalously dry conditions, when historical reanalysis alone underestimates lower-tail drought risk. Our results show that internal variability should be treated as a forecast quantity in its own right. More broadly, large ensembles provide a practical way to transfer physically plausible climate variability into machine-learning drought forecasts, yielding risk-aware bounds that are more informative for drought assessment under shifting climate conditions.</span> <span class="abstract-toggle" data-id="2608.01864">more</span>

    [:material-file-document: 2608.01864](https://arxiv.org/abs/2608.01864v1) · [:material-content-copy: BibTeX](bibtex/2608.01864.bib){ .bibtex-link }

    <span class="md-tag">probabilistic</span>

-   #### FESOM2-JAX v1.0: a differentiable shadow of the ocean-sea-ice model FESOM2, cast onto GPUs

    ---

    *Nikolay V. Koldunov, Sergey Danilov, Suvarchal Cheedela, Dmitry Sidorenko, Sebastian Beyer et al.* · 2026

    <span class="abstract-snippet" id="snip-2608.01546">We present FESOM2-JAX, a Python re-implementation of the Finite-volumE Sea ice-Ocean Model (FESOM2) in JAX. The model retains the unstructured-mesh, cell-vertex finite-volume formulation of the...</span><span class="abstract-full" id="full-2608.01546" hidden>We present FESOM2-JAX, a Python re-implementation of the Finite-volumE Sea ice-Ocean Model (FESOM2) in JAX. The model retains the unstructured-mesh, cell-vertex finite-volume formulation of the original, runs unchanged from a laptop CPU to 256 GPUs, and is end-to-end differentiable. FESOM2-JAX is a code shadow of the Fortran model: a projection onto the Python ecosystem, translated with large language models and verified kernel by kernel against the original. It is built to lower the barrier to experimentation, from new numerics and parameterizations to gradient-based calibration and hybrid physics-machine-learning components, while remaining close enough to the original so that what is developed in the shadow can be transferred back. In a 1958-2019 hindcast at 1$^{\circ}$ equivalent resolution with identical physics and forcing, the mean states of the JAX and Fortran versions differ from each other by two orders of magnitude less than either differs from observations, and the two runs agree for six decades in global temperature, salinity, heat content, and sea ice. The complete 1$^{\circ}$ configuration fits on a single GPU, a node of four GH200 superchips integrates $\sim$113 simulated years per wall-clock day, and meshes of up to 7.4 million surface vertices ($\sim$5 km) scale to 128 GPUs. What limits the model is communication rather than arithmetic. What the shadow adds to the original is the gradient: a single reverse-mode pass through the full time loop returns the sensitivity of a model diagnostic to a parameter at every mesh vertex, verified against finite differences. To our knowledge, FESOM2-JAX is the first global ocean-sea-ice model of CMIP-class complexity written natively in a differentiable framework, and the first on an unstructured mesh.</span> <span class="abstract-toggle" data-id="2608.01546">more</span>

    [:material-file-document: 2608.01546](https://arxiv.org/abs/2608.01546v1) · [:material-content-copy: BibTeX](bibtex/2608.01546.bib){ .bibtex-link }

-   #### Climate-Dyna Deep Hedging for XVAs: Model-Based Reinforcement Learning, Residual Climate HVA, and Hedge-Instrument Discovery

    ---

    *Xiaozhen Wang, Francois Buet-Golfouse* · 2026

    <span class="abstract-snippet" id="snip-2608.01208">For a trading desk, residual climate hedging valuation adjustment (HVA) is the climate cost left after its inherited hedge and any admissible overlay have been taken into account; it therefore cannot...</span><span class="abstract-full" id="full-2608.01208" hidden>For a trading desk, residual climate hedging valuation adjustment (HVA) is the climate cost left after its inherited hedge and any admissible overlay have been taken into account; it therefore cannot be inferred from a stand-alone stress loss. We obtain this residual by comparing paired climate-on and baseline worlds and reoptimizing the overlay for each hedge universe, which also turns hedge-instrument discovery into a valuation problem: an instrument is useful to the extent that it lowers the optimized residual cost. The linear-Gaussian case has an exact finite-horizon Riccati solution; Climate-Dyna starts from that hedge and learns the remaining nonlinear correction from paired world-model rollouts, with an independent gate deciding whether to deploy the update. In a public-data-calibrated semi-synthetic EU ETS study, crediting the inherited hedge lowers the mean climate charge from 1.517 to 0.906, and the learned overlay lowers it to 0.831 against a 0.821 exact floor; residual Dyna cuts regret by 93% relative to replay with one quarter as many trajectories, while adaptation from only 25 target transitions retains 60.7% of the exact-assisted gain.</span> <span class="abstract-toggle" data-id="2608.01208">more</span>

    [:material-file-document: 2608.01208](https://arxiv.org/abs/2608.01208v1) · [:material-content-copy: BibTeX](bibtex/2608.01208.bib){ .bibtex-link }

    <span class="md-tag">reinforcement-learning</span>

-   #### A Sequence-to-Sequence ConvLSTM Approach for Leaf Area Index Forecasting over the South-Central United States

    ---

    *Zhixing Ruan, Lixin Lu* · 2026

    <span class="abstract-snippet" id="snip-2608.00879">Leaf Area Index (LAI) is a fundamental biophysical variable governing land-atmosphere interactions; however, LAI forecasting at high spatial resolution remains an unsolved challenge. While recent...</span><span class="abstract-full" id="full-2608.00879" hidden>Leaf Area Index (LAI) is a fundamental biophysical variable governing land-atmosphere interactions; however, LAI forecasting at high spatial resolution remains an unsolved challenge. While recent machine learning approaches have demonstrated LAI estimation at point or regional scales, none provides a gridded, meteorology-driven prognostic forecast suitable for subseasonal land surface and climate modeling applications. Here we present a sequence-to-sequence Convolutional LSTM (ConvLSTM) framework that generates daily 1-km LAI forecasts up to 30 days ahead, driven by historical LAI sequences and daily meteorological forcing including temperature and precipitation. Trained and evaluated over the South-Central United States -- a region of strong climate gradients and diverse vegetation -- the model achieves a domain-averaged RMSE of 0.36 at a 30-day lead time, more than a third lower than the persistence baseline. Forecast skill remains robust across seasons, geographic distributions, and plant functional types, including forests, grasslands, shrublands, and croplands. To our knowledge, this is the first demonstration of skillful LAI forecasting at a 30-day horizon at 1-km resolution.</span> <span class="abstract-toggle" data-id="2608.00879">more</span>

    [:material-file-document: 2608.00879](https://arxiv.org/abs/2608.00879v1) · [:material-content-copy: BibTeX](bibtex/2608.00879.bib){ .bibtex-link }

    <span class="md-tag">recurrent</span>

-   #### A Machine Learning-based Non-precipitating Clouds Estimation for THz Dual-Frequency Radar

    ---

    *Kazuhiko Tamesue, Zheng Wen, Shotaro Yamaguchi, Hiroyuki Kasai, Wataru Kameyama, Toshio Sato et al.* · 2026

    <span class="abstract-snippet" id="snip-2608.00653">Accurate measurement of non-precipitable clouds is important for early prediction of heavy rainfall disasters caused by extreme weather events. However, microwave cloud radar cannot observe the early...</span><span class="abstract-full" id="full-2608.00653" hidden>Accurate measurement of non-precipitable clouds is important for early prediction of heavy rainfall disasters caused by extreme weather events. However, microwave cloud radar cannot observe the early stages of cloud development from non-precipitation clouds (cumulus) to cumulonimbus. In this paper, we propose a terahertz dual-frequency cloud radar using 150 GHz and 95 GHz bands to detect cloud particles in cumulus smaller than 10 μm. Using a dataset generated by the ITU-R radio propagation model, we estimate the liquid water content of non-precipitation clouds and water vapor content in atmospheric gases, respectively, by using a machine learning-based approach. The effectiveness of using the dual wavelength ratio as an explanatory variable is examined.</span> <span class="abstract-toggle" data-id="2608.00653">more</span>

    [:material-file-document: 2608.00653](https://arxiv.org/abs/2608.00653v1) · [:material-content-copy: BibTeX](bibtex/2608.00653.bib){ .bibtex-link }

-   #### Generative Models for Modeling and Synthesizing MIMO Channels in Adverse Weather Conditions

    ---

    *Vignesh Nandakumar, Faraz Barati, Brian L. Evans* · 2026

    <span class="abstract-snippet" id="snip-2608.00156">The push for broader coverage in future cellular networks depends on reliable service, yet this is increasingly harder to do as we encounter more instances of extreme weather conditions. In extreme...</span><span class="abstract-full" id="full-2608.00156" hidden>The push for broader coverage in future cellular networks depends on reliable service, yet this is increasingly harder to do as we encounter more instances of extreme weather conditions. In extreme weather conditions, we have difficulty evaluating coverage due to limited access to channel measurements. In this paper, we generate channel state information (CSI) in low and moderate weather conditions to synthesize realistic MIMO CSI under adverse weather conditions. Our primary contributions are to (1) synthesize MIMO channel datasets incorporating three weather types, each with three intensity levels, representative of practical 5G/6G scenarios; (2) train a diffusion model conditioned on weather using channel samples obtained through conventional pilot-based estimation under low and moderate weather intensities, and subsequently use it to generate channel realizations for severe weather conditions; and (3) evaluate the downlink Bit Error Rate (BER) and Outage Probability measures using the generated channels. The results show that diffusion-based generative models provide a scalable, data-driven alternative for channel modeling in harsh environments and can generalize to severe weather conditions using only low- and moderate-intensity training data.</span> <span class="abstract-toggle" data-id="2608.00156">more</span>

    [:material-file-document: 2608.00156](https://arxiv.org/abs/2608.00156v1) · [:material-content-copy: BibTeX](bibtex/2608.00156.bib){ .bibtex-link }

    <span class="md-tag">diffusion</span>

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

</div>

