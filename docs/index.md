---
hide:
  - navigation
title: Deep Learning in Weather
---

A collection of papers on deep learning and machine learning applied to weather forecasting, climate modeling, and atmospheric science.

*Last updated: 2026-08-13*

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

-   #### FarSky: Task-Aware Latent-Space Coupling for Generative Intra-Hour Solar Forecasting

    ---

    <span class="paper-meta"><em>Yann Fabel, Bijan Nouri, Milon Miah, Niklas Blum, Luis F. Zarzalejo, Julia Kowalski et al.</em> · 2026</span>

    <span class="abstract-snippet" id="snip-2608.11254" data-search-exclude>Accurate solar irradiance forecasting is essential for the reliable integration of photovoltaic power into modern electricity grids. All-sky imagers (ASI) provide high-resolution observations of...</span><span class="abstract-full" id="full-2608.11254" data-search-exclude hidden>Accurate solar irradiance forecasting is essential for the reliable integration of photovoltaic power into modern electricity grids. All-sky imagers (ASI) provide high-resolution observations of clouds, making them well suited for intra-hour forecasting. Recent deep learning approaches have substantially improved forecast accuracy but are often limited by deterministic predictions and a reduced capability to anticipate ramp events. This work proposes FarSky, a generative forecasting framework that leverages latent-space coupling to learn task-aware representations of sky images. A multi-task autoencoder first learns a shared latent representation for image reconstruction and irradiance estimation. A latent diffusion model then generates future latent states conditioned on recent observations, from which irradiance forecasts are directly decoded. Probabilistic forecasts are inherently obtained through stochastic sampling. The framework is developed using a multi-year ASI dataset acquired at the Plataforma Solar de Almería, Spain, and evaluated on two independent test datasets against persistence, state-of-the-art end-to-end, and generative forecasting approaches. FarSky achieves the best overall deterministic and probabilistic forecasting performance, improving forecast skill by up to 11 percentage points. Furthermore, it substantially improves ramp event detection over existing methods, achieving F1-scores above 60%. These results demonstrate the potential of combining generative models with task-aware latent-space coupling for solar forecasting.</span> <span class="abstract-toggle" data-id="2608.11254">more</span>

    <span class="paper-links">[:material-file-document: 2608.11254](https://arxiv.org/abs/2608.11254v1) · [:material-content-copy: BibTeX](bibtex/2608.11254.bib){ .bibtex-link }</span>

    <a class="md-tag" href="/tags/#diffusion">diffusion</a> <a class="md-tag" href="/tags/#probabilistic">probabilistic</a>

-   #### Observational Evidence Revises Presumed Large Ozone Worsening from Nitrogen Oxides Cuts

    ---

    <span class="paper-meta"><em>Xiang Weng, Xiao Lu, Jiawei Li, Grant Forster, Jessica Chapman, Beckie George, Yunbo Lu, Guowen He et al.</em> · 2026</span>

    <span class="abstract-snippet" id="snip-2608.10399" data-search-exclude>Many air quality models indicate that rapid reductions in nitrogen oxides (NOx), without comparable controls on volatile organic compounds, have worsened summertime ozone pollution in urban China,...</span><span class="abstract-full" id="full-2608.10399" data-search-exclude hidden>Many air quality models indicate that rapid reductions in nitrogen oxides (NOx), without comparable controls on volatile organic compounds, have worsened summertime ozone pollution in urban China, producing a short-term strong ozone penalty. Other models, however, simulate the opposite response, suggesting that cutting down NOx has already helped mitigate ozone pollution. This contradiction obscures understanding of atmospheric chemistry and weakens guidance on control policy design. Here, we reconcile this disagreement and reveal the underestimated benefits of NOx emission reductions using a machine learning framework integrated with an observational constraint. We first constrain ozone responses under a 30% NOx reduction, comparable to the magnitude of NOx emission declines across major Chinese city clusters between 2015 and 2023. The constrained results indicate that ozone decreases prevail across urban China, with only small increases mainly in July 2015. This challenges the widespread ozone worsening that many models predict. We then extend the constraint across 10-60% NOx reductions, establishing its use for rapid ozone sensitivity diagnosis without exhaustive scenario modeling. This diagnosis shows that sustained NOx control increasingly favored ozone mitigation during 2015-2023, benefiting a growing share of China's population. These results underscore that continued NOx reductions can deliver larger ozone mitigation benefits than many models suggest.</span> <span class="abstract-toggle" data-id="2608.10399">more</span>

    <span class="paper-links">[:material-file-document: 2608.10399](https://arxiv.org/abs/2608.10399v1) · [:material-content-copy: BibTeX](bibtex/2608.10399.bib){ .bibtex-link }</span>

-   #### Stochastic Emulation of a Fully Coupled Preindustrial E3SMv3 Simulation

    ---

    <span class="paper-meta"><em>Elynn Wu, James P. C. Duncan, Troy Arcomano, Jeremy McGibbon, Oliver Watt-Meyer et al.</em> · 2026</span>

    <span class="abstract-snippet" id="snip-2608.10277" data-search-exclude>We present a stochastic coupled emulator of E3SM version 3, built on the SamudrACE framework, which couples an atmosphere emulator (ACE2) with a full-depth ocean emulator (Samudra). We replace the...</span><span class="abstract-full" id="full-2608.10277" data-search-exclude hidden>We present a stochastic coupled emulator of E3SM version 3, built on the SamudrACE framework, which couples an atmosphere emulator (ACE2) with a full-depth ocean emulator (Samudra). We replace the deterministic atmosphere emulator with its stochastic counterpart, ACE2S, and fine-tune the coupled system with a probabilistic objective, so that the atmosphere acts as a source of internal variability for the ocean. Trained on 105 years of a pre-industrial control simulation and evaluated on an independent 400 years, the emulator reproduces E3SMv3's mean climate state with biases much smaller than existing model-to-observation differences. Relative to a deterministic baseline, stochastic training maintains internal variability across timescales, most notably in the ENSO power spectrum, eddy-rich SST anomalies, and sea ice variability in the marginal ice zone. The emulator captures daily precipitation accurately up to the 99.99th percentile, but underestimates the rarest tropical extremes. These results show that stochastic coupled emulators can reproduce long-timescale variability with high fidelity, while extrapolation to unseen extremes remains a key challenge.</span> <span class="abstract-toggle" data-id="2608.10277">more</span>

    <span class="paper-links">[:material-file-document: 2608.10277](https://arxiv.org/abs/2608.10277v1) · [:material-content-copy: BibTeX](bibtex/2608.10277.bib){ .bibtex-link }</span>

    <a class="md-tag" href="/tags/#probabilistic">probabilistic</a>

-   #### Deep Learning-Based Statistical Downscaling of Sea Surface Temperature Using a Residual Corrective Neural Network

    ---

    <span class="paper-meta"><em>Onkar Jadhav, Tim French, Ivica Janekovic, Nicole L. Jones, Matthew Rayson</em> · 2026</span>

    <span class="abstract-snippet" id="snip-2608.10022" data-search-exclude>The large-scale oceanic and atmospheric forecasts provided by global climate models typically lack sufficient resolution to accurately capture the response of the coastal ocean to atmospheric forcing...</span><span class="abstract-full" id="full-2608.10022" data-search-exclude hidden>The large-scale oceanic and atmospheric forecasts provided by global climate models typically lack sufficient resolution to accurately capture the response of the coastal ocean to atmospheric forcing and coastal circulation that drive fine-scale SST variability. Dynamical downscaling is computationally prohibitive, when applied to extensive coastlines, predictive ensembles, or long time periods. Therefore, this work presents a statistical downscaling of sea surface temperature (SST) from the seasonal coupled ocean-atmosphere forecast system (ACCESS-S2) using machine learning techniques. This study proposes a novel deep learning framework that uses a U-Net to generate an initial high-resolution SST estimate, which is subsequently refined using a residual corrective approach. The target SST fields are derived from the Regional Ocean Modeling System (ROMS). This two step approach called Residual Corrective Neural Network (RCNN) progressively refines initial U-Net predictions by incorporating dynamically scaled residuals at each step, enabling accurate capture of broad patterns and fine-grained features such as eddies and fronts. We also introduce a custom loss-assisted RCNN variant to improve performance during extreme events, which may be absent from training data due to climate-driven shifts in SST extremes. The framework efficiently downscales SST along the west coast of Australia. A 2011 marine heatwave case study shows that the RCNN improves ACCESS-S2 SST predictions by increasing horizontal resolution from 25 km to 2 km, enabling identification of fine-scale anomalies unresolved in the ACCESS-S2 dataset. This balance between computational efficiency and accuracy supports applications in coastal impact assessment and marine ecosystem studies.</span> <span class="abstract-toggle" data-id="2608.10022">more</span>

    <span class="paper-links">[:material-file-document: 2608.10022](https://arxiv.org/abs/2608.10022v1) · [:material-content-copy: BibTeX](bibtex/2608.10022.bib){ .bibtex-link }</span>

    <a class="md-tag" href="/tags/#cnn">CNN</a>

-   #### Do AI weather models miss extremes?

    ---

    <span class="paper-meta"><em>Marvin Vincent Gabler, Roberto Molinaro, Niall Siegenheim, Henry Martin, Mark Frey, Niels Poulsen et al.</em> · 2026</span>

    <span class="abstract-snippet" id="snip-2608.09972" data-search-exclude>First-generation AI weather models are often reported to underperform at extremes, mostly in reanalysis-based evaluations of deterministic regression systems. We verify eleven physical and AI...</span><span class="abstract-full" id="full-2608.09972" data-search-exclude hidden>First-generation AI weather models are often reported to underperform at extremes, mostly in reanalysis-based evaluations of deterministic regression systems. We verify eleven physical and AI forecast systems against European synoptic, solar, and rain-gauge stations over ten months for 10 m wind, 2 m temperature, hourly shortwave accumulation, and hourly precipitation, scoring mean absolute error (MAE) against ECMWF IFS in ERA5 1991-2020 climatological regimes. Among these systems, AI models do not show a uniform relative-skill deficit in the tails. Jua EPT-2.1 Europa leads all-conditions wind (+8.4%), while Jua EPT-2 HRRR leads temperature overall (+12.1%) and in the heat regime (+19.6 +/- 2.2%). EPT-2.1 Europa and DWD ICON Global lead at gale-force wind. Jua EPT-2.1 Helios leads solar overall (+10.2 +/- 1.7%), in overcast conditions (+16.4 +/- 3.4%), and in the clear-sky tail (+24.8 +/- 5.4%). For precipitation, three Jua models gain 14-15% at moderate intensity and 9-11% at P75-P95; EPT-2 Reasoning remains ahead above P95 (+1.7 +/- 0.5%). Failures are model-specific: ECMWF AIFS loses 4.9 +/- 2.0% in the heat tail, while NOAA GFS loses 22.8 +/- 2.0% there. Every model, including numerical weather prediction systems, shows a shared conditional bias toward the centre of the observed distribution, with an inter-model spread several times smaller than the shared signal. Missing relative skill at extremes is therefore not a property of AI weather models as a class, but of particular AI and physical models.</span> <span class="abstract-toggle" data-id="2608.09972">more</span>

    <span class="paper-links">[:material-file-document: 2608.09972](https://arxiv.org/abs/2608.09972v1) · [:material-content-copy: BibTeX](bibtex/2608.09972.bib){ .bibtex-link }</span>

-   #### Real-Time Climate Risk Assessment for Supply Chain Resilience: A Data-Driven Nowcasting Framework for Colombian Agriculture

    ---

    <span class="paper-meta"><em>Hernan J. Silva-Sosa</em> · 2026</span>

    <span class="abstract-snippet" id="snip-2608.09846" data-search-exclude>This paper presents a methodological framework for real-time climate risk assessment using data-driven nowcasting techniques to enhance supply chain resilience in Colombian agricultural contexts....</span><span class="abstract-full" id="full-2608.09846" data-search-exclude hidden>This paper presents a methodological framework for real-time climate risk assessment using data-driven nowcasting techniques to enhance supply chain resilience in Colombian agricultural contexts. Climate variability in Colombia, characterized by irregular rainfall, temperature fluctuations, and recurrent extreme events, has a direct impact on agricultural production and logistics, particularly for time sensitive crops. The proposed approach integrates short term climate forecasting based on historical meteorological observations with supply chain risk modeling to establish a conceptual early warning system architecture. A prototype implementation developed in a controlled computational environment demonstrates the feasibility of the framework using historical meteorological and agricultural time series derived from official statistics and reanalysis products, without reliance on satellite imagery or computer vision components. The methodology addresses the integration of climate nowcasting with supply chain decision making through explicit risk mapping, threshold-based categorization, and stakeholder-oriented risk signals. Results from synthetic and historical data experiments indicate that short term precipitation nowcasts can be translated into actionable risk indicators for agricultural supply chains, supporting anticipatory decisions related to inventory, sourcing, and transport.</span> <span class="abstract-toggle" data-id="2608.09846">more</span>

    <span class="paper-links">[:material-file-document: 2608.09846](https://arxiv.org/abs/2608.09846v1) · [:material-content-copy: BibTeX](bibtex/2608.09846.bib){ .bibtex-link }</span>

-   #### Deep Learning Imputation of Missing Radius of Maximum Winds (Rmax) Values in Tropical Cyclone Best-Track Data

    ---

    <span class="paper-meta"><em>Swastik Agrawal, Nishkal Hundia, Ziyue Liu, Michelle Bensi</em> · 2026</span>

    <span class="abstract-snippet" id="snip-2608.09683" data-search-exclude>Probabilistic coastal hazard assessments require accurate characterization of tropical cyclone (TC) parameters, yet datasets often contain missing records for the radius of maximum winds (Rmax), a...</span><span class="abstract-full" id="full-2608.09683" data-search-exclude hidden>Probabilistic coastal hazard assessments require accurate characterization of tropical cyclone (TC) parameters, yet datasets often contain missing records for the radius of maximum winds (Rmax), a key variable in Joint Probability Method analyses. This study evaluates data-driven approaches for Rmax imputation, including one-dimensional Convolutional Neural Networks (1DCNNs), Long Short-Term Memory (LSTM) networks, and conventional machine learning models. We examine physics-informed input augmentation, temporal modeling, and transfer learning using synthetic RAFT and STORM datasets for pre-training and observational IBTrACS data for fine-tuning. Including the radius of 34-knot winds (R34) substantially improves performance across all model types. Temporal models achieve higher average correlations than non-temporal models despite using approximately an order of magnitude fewer samples, indicating better preservation of relative Rmax variability across storms. This advantage is more pronounced when R34 is unavailable, suggesting temporal information can partially compensate for missing storm-size predictors. Transfer learning does not improve performance, likely because synthetic datasets have lower and less variable Rmax distributions than IBTrACS. These findings demonstrate the potential of temporal deep learning for reconstructing incomplete TC records and highlight the importance of physics-informed inputs, observational data availability, and distributional consistency in coastal hazard assessment.</span> <span class="abstract-toggle" data-id="2608.09683">more</span>

    <span class="paper-links">[:material-file-document: 2608.09683](https://arxiv.org/abs/2608.09683v1) · [:material-content-copy: BibTeX](bibtex/2608.09683.bib){ .bibtex-link }</span>

    <a class="md-tag" href="/tags/#cnn">CNN</a> <a class="md-tag" href="/tags/#physics-informed">physics-informed</a> <a class="md-tag" href="/tags/#recurrent">recurrent</a> <a class="md-tag" href="/tags/#probabilistic">probabilistic</a>

-   #### VeinCast: Physics-Guided Dynamic Field Graphs with Graph-Conditioned Fusion for Global Medium-Range Weather Forecasting

    ---

    <span class="paper-meta"><em>Zhisheng Chen, Jinhan Li, Yuxuan Li, Yuan Gao, Hao Wu, Zheng Lu, Jinlong Du, Kun Wang, Bo An</em> · 2026</span>

    <span class="abstract-snippet" id="snip-2608.09286" data-search-exclude>Global medium-range weather forecasting requires modeling structured yet state-dependent interactions among heterogeneous atmospheric fields. Existing data-driven models largely learn these...</span><span class="abstract-full" id="full-2608.09286" data-search-exclude hidden>Global medium-range weather forecasting requires modeling structured yet state-dependent interactions among heterogeneous atmospheric fields. Existing data-driven models largely learn these interactions implicitly, whereas equation-level physical constraints may inherit approximation and model-form biases. We present VeinCast, a physics-guided dynamic field graph and graph-conditioned fusion framework that jointly forecasts 69 surface and upper-air fields. Within each local window, its Physics-Guided Dynamic Field Graph combines predefined atmospheric relations with state-dependent Top-K residual edges and adapts Earth-window attention using the resulting graph context. Graph-Conditioned Latent Fusion further employs graph context and source-node centrality to guide field-to-latent aggregation, while bounded feedback preserves field-specific information. On the $1.5^\circ$ ERA5 benchmark, VeinCast demonstrates competitive forecasting performance across all 69 meteorological fields at lead times of up to 14 days, compared with representative global weather forecasting models including FuXi, Pangu-Weather, GraphCast, FengWu, and ARROW. Ablations confirm that the two modules provide complementary gains, demonstrating the effectiveness of relational-level physical guidance for data-driven weather forecasting.</span> <span class="abstract-toggle" data-id="2608.09286">more</span>

    <span class="paper-links">[:material-file-document: 2608.09286](https://arxiv.org/abs/2608.09286v1) · [:material-content-copy: BibTeX](bibtex/2608.09286.bib){ .bibtex-link }</span>

</div>

