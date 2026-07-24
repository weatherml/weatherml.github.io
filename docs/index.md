---
hide:
  - navigation
title: Deep Learning in Weather
---

A collection of papers on deep learning and machine learning applied to weather forecasting, climate modeling, and atmospheric science.

*Last updated: 2026-07-24*

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

-   #### Flexible generation of daily Earth system model projections across radiative forcing scenarios

    ---

    *Yu Huang, Sebastian Bathiany, Shangshang Yang, Philipp Hess, Michael Aich, Niklas Boers* · 2026

    <span class="abstract-snippet" id="snip-2607.21382">Earth system model (ESM) projections of the climate system's response to anthropogenic forcing are central to assess the impacts of climate change and inform adaptation and mitigation policies....</span><span class="abstract-full" id="full-2607.21382" hidden>Earth system model (ESM) projections of the climate system's response to anthropogenic forcing are central to assess the impacts of climate change and inform adaptation and mitigation policies. However, given their high computational cost, projections are only made for a limited set of standardized forcing scenarios with limited temporal extent, such as the Shared Socioeconomic Pathways (SSPs), the spatiotemporal resolution remains too low for direct impact assessments, and uncertainties cannot be comprehensively quantified. Recent data-driven models offer efficient and accurate high-resolution simulations for weather prediction, but cannot extrapolate to future greenhouse gas concentrations because they cannot capture the responses to unprecedented forcing, limiting their value for climate change projections. Here, we combine response theory with a tailored generative machine learning framework to address this challenge. Our approach extracts the physical forced response to radiative forcing from monthly low-resolution ESM fields, and uses this response to guide a generative model to infer consistent daily global high-resolution temperature and precipitation projections. Our probabilistic approach generalizes across ESMs and provides long-term, bias-corrected responses to radiative forcing at high spatiotemporal resolution. It efficiently generates large ensembles needed for uncertainty quantification, effectively fills the gaps between existing SSPs, and readily extends climate projections to 2300 and beyond. Our framework hence complements ESM projections by providing efficient, stable, and high spatiotemporal resolution long-term climate projection ensembles across emission scenarios, enabling detailed impact assessment and exploration of long-term climate commitment.</span> <span class="abstract-toggle" data-id="2607.21382">more</span>

    [:material-file-document: 2607.21382](https://arxiv.org/abs/2607.21382v1) · [:material-content-copy: BibTeX](bibtex/2607.21382.bib){ .bibtex-link }

    <span class="md-tag">probabilistic</span>

-   #### Nipping the Butterfly Effect in the Bud: Self-Output Fine-Tuning for Autoregressive Weather Prediction

    ---

    *Yun-Ye Cai, Hsuan-Tien Lin* · 2026

    <span class="abstract-snippet" id="snip-2607.21080">Long-horizon weather forecasting is a fundamental challenge in atmospheric science, for which autoregressive Deep Learning Weather Prediction (DLWP) has emerged as the primary paradigm. Although the...</span><span class="abstract-full" id="full-2607.21080" hidden>Long-horizon weather forecasting is a fundamental challenge in atmospheric science, for which autoregressive Deep Learning Weather Prediction (DLWP) has emerged as the primary paradigm. Although the autoregressive pipeline is highly scalable and flexible, its prediction errors grow rapidly over long forecasting horizons. In this work, we study this error growth phenomenon from both theoretical and empirical perspectives. Our analysis reveals that the growth is driven by a feedback loop between output errors and input distribution shifts. Specifically, the autoregressive process amplifies small initial output errors, which progressively corrupt subsequent input distributions, echoing the butterfly effect in atmospheric science and ultimately deteriorating forecasting accuracy over longer horizons. Furthermore, we show that this distributional shift originates at the earliest stage of inference, with out-of-distribution signatures detectable as early as the first autoregressive step. To mitigate this issue, we propose \textbf{Self-Output Fine-Tuning (SOFT)}, a plug-and-play strategy that leverages the model's own one-step predictions to calibrate the biased input distribution encountered at the first step. Extensive experiments demonstrate that, despite its simplicity, SOFT achieves state-of-the-art performance on long-horizon forecasting tasks and substantially reduces both prediction errors and distributional discrepancy. The success of SOFT highlights the importance of reexamining the fundamental pipeline of deep learning weather prediction, representing a critical pipeline advance for atmospheric science.</span> <span class="abstract-toggle" data-id="2607.21080">more</span>

    [:material-file-document: 2607.21080](https://arxiv.org/abs/2607.21080v1) · [:material-content-copy: BibTeX](bibtex/2607.21080.bib){ .bibtex-link }

-   #### Toward Mechanistic Interpretability of an AI Foundation Model Fine-Tuned for Atmospheric Chemistry

    ---

    *Jason Y. Hu, Ivan Higuera-Mendieta, Patrick Obin Sturm, Makoto M. Kelp* · 2026

    <span class="abstract-snippet" id="snip-2607.20778">Weather forecasting foundation models (FMs) are increasingly fine-tuned to predict air quality, offering fast global pollution forecasts at lower computational cost than conventional chemical...</span><span class="abstract-full" id="full-2607.20778" hidden>Weather forecasting foundation models (FMs) are increasingly fine-tuned to predict air quality, offering fast global pollution forecasts at lower computational cost than conventional chemical transport models. These FMs are typically trained on reanalysis data and generate forecasts through autoregressive rollout. They do not explicitly represent governing physical or chemical processes. Therefore, high forecast skill does not reveal whether a model has learned physical mechanisms or exploits statistical regularities in its training data. Here, we present the first study of what a FM fine-tuned for atmospheric chemistry has learned by examining Microsoft's Aurora model. We impose controlled chemical perturbations on its forecasts and test them against known photochemical relationships. We then examine the internal representations that generate these forecasts. We find that Aurora captures a first-order ozone response to reactive nitrogen but does not enforce the chemical constraints that a process-based model encodes. It generates chemically inconsistent combinations of related species and relaxes localized emission features such as wildfire plumes toward background. Internally, its representations remain largely organized around the meteorology inherited during pretraining, with little structure specific to chemistry. Using sparse autoencoders, we identify internal components that causally control the chemical forecast but do not map cleanly onto individual atmospheric processes. This work provides a framework for testing whether AI forecasting systems learn atmospheric chemistry from reanalysis data. As these models are increasingly positioned to inform environmental policy decisions, we argue that composition forecasts should also be judged by their internal mechanisms rather than by benchmark skill alone.</span> <span class="abstract-toggle" data-id="2607.20778">more</span>

    [:material-file-document: 2607.20778](https://arxiv.org/abs/2607.20778v1) · [:material-content-copy: BibTeX](bibtex/2607.20778.bib){ .bibtex-link }

    <span class="md-tag">foundation-model</span>

-   #### Spatial Generalization Tests for Machine Learning-based Weather Models to Assess Physical Consistency

    ---

    *Maren Höver, Milan Klöwer, Christian Schroeder de Witt, Hannah M. Christensen* · 2026

    <span class="abstract-snippet" id="snip-2607.20716">Machine learning-based weather prediction is revolutionizing weather forecasting by learning from weather data in present-day climate. However, generalization to other climates remains a major...</span><span class="abstract-full" id="full-2607.20716" hidden>Machine learning-based weather prediction is revolutionizing weather forecasting by learning from weather data in present-day climate. However, generalization to other climates remains a major challenge. With melting sea ice, land-use change, and increasing ocean temperatures, boundary conditions are changing. Therefore, generalization in time depends on generalization in space. Here, we present three test cases to evaluate whether machine learning-based weather and climate models generalize in space and apply them to GraphCast and NeuralGCM. We reverse or rotate the planet in longitude or latitude under the model's coordinate system and adapt all boundary conditions and forcings accordingly. Physics-based general circulation models simulate a rotated/reversed planet with only rounding errors, but GraphCast and NeuralGCM fail these tests. The analyses furthermore revealed unphysical variable mappings based on correlation rather than causation. We argue that machine learning-based climate models should be designed to pass generalization tests to prevent overfitting on present-day regional climate.</span> <span class="abstract-toggle" data-id="2607.20716">more</span>

    [:material-file-document: 2607.20716](https://arxiv.org/abs/2607.20716v1) · [:material-content-copy: BibTeX](bibtex/2607.20716.bib){ .bibtex-link }

    <span class="md-tag">physics-informed</span>

-   #### Geospatial Diffusion-based Evolution Synthesis (GeoDES) for Storm-Centered Weather Augmentation

    ---

    *Sonia Cromp, Satya Sai Srinath Namburi GNVV, Youran Wang, Grace Kisslinger, Frederic Sala et al.* · 2026

    <span class="abstract-snippet" id="snip-2607.19522">While machine learning-based weather models hold significant promise, they struggle to predict the detailed structure of large-scale weather systems such as cyclonic storms. Regional models are...</span><span class="abstract-full" id="full-2607.19522" hidden>While machine learning-based weather models hold significant promise, they struggle to predict the detailed structure of large-scale weather systems such as cyclonic storms. Regional models are constrained by limited historical records within fixed geographic boundaries, while global models are computationally expensive and often operate at resolutions too coarse to capture fine-grained storm dynamics. To bridge this gap, we introduce the Geospatial Diffusion-based Evolution Synthesis (GeoDES) model, a custom image-to-video diffusion model. By focusing generation strictly on the evolving storm structure, GeoDES synthesizes physically consistent, high-fidelity weather events suitable for stress-testing forecast models and expanding meteorological datasets. Evaluations demonstrate that GeoDES outperforms prior methods on key metrics, achieving $52\%$ lower Peak Vorticity Error and $8\%$ higher Anomaly Correlation Coefficient than the next strongest methods on the North Atlantic test set.</span> <span class="abstract-toggle" data-id="2607.19522">more</span>

    [:material-file-document: 2607.19522](https://arxiv.org/abs/2607.19522v1) · [:material-content-copy: BibTeX](bibtex/2607.19522.bib){ .bibtex-link }

    <span class="md-tag">diffusion</span>

-   #### Aircast-Mars: A Mars Foundation Model for Global Weather Forecasting with HEALPix-Aware Convolutions

    ---

    *Manmeet Singh, Saptarishi Dhanuka, Naveen Sudharsan, Houman Owhadi, Krista M. Soderlund et al.* · 2026

    <span class="abstract-snippet" id="snip-2607.19370">Foundation models for planetary atmospheres promise fast, lightweight surrogates of expensive general circulation models (GCMs) for mission planning and scientific inquiry. Here we present...</span><span class="abstract-full" id="full-2607.19370" hidden>Foundation models for planetary atmospheres promise fast, lightweight surrogates of expensive general circulation models (GCMs) for mission planning and scientific inquiry. Here we present Aircast-Mars, a deep-learning weather prediction system for Mars trained on the Ensemble Mars Atmosphere Reanalysis System (EMARS) v1.0. We regrid temperature, zonal wind, and meridional wind fields across 28 vertical levels onto a hierarchical equal-area isolatitude pixelization (HEALPix) mesh at Nside = 64 (~110 km resolution) and train a HEALPix-aware 2D U-Net inspired by the DLESyM architecture to predict the next hourly atmospheric state. The model employs custom inter-face padding that respects the topology of the 12-face HEALPix sphere and modern ConvNeXt residual blocks with capped Gaussian Error Linear Unit (GELU) activations. While containing 4.3 million trainable parameters, a compact size compared to terrestrial weather foundation models, the network achieves a best validation Mean Squared Error (MSE) of 1.58e-5 in normalized units. Recursive autoregressive rollouts remain stable and physically coherent for 25 hours (one Martian sol), with Root Mean Square Error (RMSE) growing monotonically from ~0.004 at t + 1 h to ~0.031 at t + 25 h without divergence. Compared to a baseline 3D U-Net, the HEALPix-aware architecture reduces validation loss by more than an order of magnitude while using fewer parameters. The model generates a one-hour forecast in approximately 0.5 seconds on a single GPU, offering several orders-of-magnitude speedup over traditional numerical GCMs. These results demonstrate that parsimonious, geometry-respecting neural architectures can capture synoptic-scale Martian atmospheric dynamics and provide a foundation for planetary-scale weather forecasting.</span> <span class="abstract-toggle" data-id="2607.19370">more</span>

    [:material-file-document: 2607.19370](https://arxiv.org/abs/2607.19370v1) · [:material-content-copy: BibTeX](bibtex/2607.19370.bib){ .bibtex-link }

    <span class="md-tag">CNN</span> <span class="md-tag">foundation-model</span>

-   #### On the sensitivity of machine-learned probabilistic weather forecast models to scale-aware scoring rules

    ---

    *Simon Lang, Martin Leutbecher, Sam Hatfield* · 2026

    <span class="abstract-snippet" id="snip-2607.19161">Probabilistic forecast models can be machine-learned from data using loss functions based on scoring rules such as the Continuous Ranked Probability Score (CRPS). This note summarises a preliminary...</span><span class="abstract-full" id="full-2607.19161" hidden>Probabilistic forecast models can be machine-learned from data using loss functions based on scoring rules such as the Continuous Ranked Probability Score (CRPS). This note summarises a preliminary study comparing versions of AIFS-CRPS, a global weather forecast model, trained with different univariate and multivariate scoring rules that aim to explicitly represent scale-awareness in the loss function. In the first part, we compare the (almost) fair CRPS, a fair global energy score, and a graph energy score based on node neighbourhoods. Across standard verification metrics, forecast skill is broadly similar. In the extratropics we find only small differences, while in the tropics the graph energy score setup performs somewhat better and the global energy score shows some degradation. These results suggest that multivariate scores are a viable alternative to CRPS-based training for global machine-learned weather forecasting. In the second part of the study, we analyse how different scoring rules and scale-aware loss constraints shape the spectra of forecast fields. It is apparent that any form of explicit scale-awareness improves realism. Here, the largest differences are likely associated with different effective weights per scale.</span> <span class="abstract-toggle" data-id="2607.19161">more</span>

    [:material-file-document: 2607.19161](https://arxiv.org/abs/2607.19161v1) · [:material-content-copy: BibTeX](bibtex/2607.19161.bib){ .bibtex-link }

    <span class="md-tag">probabilistic</span>

-   #### Physics-Informed Super-Resolution of Atmospheric Data

    ---

    *Chang Xu, Gencer Sumbul, Hugo Porta, Manon Béchaz, Sebastian Schemm, Devis Tuia* · 2026

    <span class="abstract-snippet" id="snip-2607.18877">In the context of global warming, extreme events have become more frequent and intense, making their trustworthy detection and forecasting more important than ever. Yet, atmospheric observations lack...</span><span class="abstract-full" id="full-2607.18877" hidden>In the context of global warming, extreme events have become more frequent and intense, making their trustworthy detection and forecasting more important than ever. Yet, atmospheric observations lack sufficient spatial resolution, motivating atmospheric data downscaling as a way to reconstruct high-resolution data from coarse observations. This task is now being formulated as a super-resolution (SR) problem with machine learning methods featuring high efficiency. Nevertheless, it remains unclear whether the super-resolved atmospheric data still satisfies fundamental physics governing the Earth system, raising concerns about their trustworthiness in climate-related applications. In this work, we address this challenge by constraining SR models to respect hydrostatic primitive equations that represent multivariate atmospheric physics. First, we propose a Physics-Informed Super-Resolution (PISR) method involving multi-scale physics-informed objectives based on primitive equations. PISR favors the SR outputs to respect these equations and therefore naturally encodes inter-variable relationships. In addition, we propose a metric called Normalized Physical Consistency (NPC) derived from said primitive equations to measure the physical consistency of super-resolved data. Experiments on ERA5, CERRA, and COSMO demonstrate that PISR enhances the reconstruction fidelity by improving physical consistency, SR accuracy, and downstream detection of extreme events, as demonstrated by case studies in heatwaves and extreme winds.</span> <span class="abstract-toggle" data-id="2607.18877">more</span>

    [:material-file-document: 2607.18877](https://arxiv.org/abs/2607.18877v1) · [:material-content-copy: BibTeX](bibtex/2607.18877.bib){ .bibtex-link }

    <span class="md-tag">physics-informed</span>

-   #### Hard conservation correctors can hide a degrading model when training autoregressive emulators

    ---

    *William E. Chapman, John Schreck, Yingkai Sha* · 2026

    <span class="abstract-snippet" id="snip-2607.18416">AI weather and climate emulators increasingly incorporate physical principles into their formulation. One approach is to apply hard correctors that modify network outputs so that global mass, water,...</span><span class="abstract-full" id="full-2607.18416" hidden>AI weather and climate emulators increasingly incorporate physical principles into their formulation. One approach is to apply hard correctors that modify network outputs so that global mass, water, or energy budgets close. Prior work introduced such training-time correctors in the CREDIT framework and reported reduced precipitation bias and improved stability. Motivated by those results, we fine-tuned a global atmosphere emulator with a water-budget corrector, using the corrected prediction in the supervised loss and evaluating through post-correction budget closure. By that measure, training appeared successful. Every delivered field closed the moisture budget to machine precision. However, raw precipitation developed a growing global low bias over 18 training epochs, while the required correction increased from about 2% to roughly 24%. The cause is a scale degeneracy. A uniform change in raw precipitation amplitude is offset by a compensating change in the correction factor, leaving the corrected field, and therefore the supervised loss, unchanged. This invariance removes the restoring force on raw precipitation amplitude, allowing other training pressures to drive drift. Two changes recovered stable behavior. We supervised the pre-correction prediction and penalized its raw budget imbalance, while the hard correction remained in place for the delivered field. The required correction returned to less than 1% within the next epoch. A controlled 2x2 ablation showed that the runaway occurred only when corrected-output supervision was combined with no imbalance penalty. Exact post-correction closure therefore says little about whether the raw model has learned the budget. When a corrector removes information from the loss, the raw fields and the applied correction need to be tracked.</span> <span class="abstract-toggle" data-id="2607.18416">more</span>

    [:material-file-document: 2607.18416](https://arxiv.org/abs/2607.18416v1) · [:material-content-copy: BibTeX](bibtex/2607.18416.bib){ .bibtex-link }

-   #### Fourier Geometric Wind Power Forecasting with Numerical Weather Prediction

    ---

    *Shiyuan Piao, Fan Zehui, Yang Liu, Hong Cheng, Juepeng Zheng, Jie Zhou, Fugee Tsung* · 2026

    <span class="abstract-snippet" id="snip-2607.17095">Accurate short-term wind power forecasting is essential for grid stability and operational planning, yet remains challenging due to the complex interactions between atmospheric conditions and turbine...</span><span class="abstract-full" id="full-2607.17095" hidden>Accurate short-term wind power forecasting is essential for grid stability and operational planning, yet remains challenging due to the complex interactions between atmospheric conditions and turbine dynamics. However, existing methods fail to effectively incorporate weather forecasting with wind turbine data (i.e., SCADA), leading to suboptimal solutions. To address this, we introduce a multimodal framework that integrates historical point-based SCADA data with grid-based Numerical Weather Prediction (NWP) forecasts, which is challenging due to heterogeneous input and the complex physical wind-turbine interactions. Our approach first explicitly decomposes inputs into scalar and vector features to better capture both site-specific and geometric dependencies and then incorporates a geometric encoder to extract rotation-invariant features from wind vectors. We further leverages a Fourier Neural Operator (FNO) architecture, which performs global convolutions in the frequency domain to efficiently model long-range spatiotemporal relationships. Extensive experiments on three real-world wind farms, with weather forecasting data, demonstrate that our model consistently outperforms state-of-the-art baselines, highlighting the effectiveness of its physically-informed design. The core implementation of our method is publicly available at: https://github.com/shawn-sypiao/GWPF.</span> <span class="abstract-toggle" data-id="2607.17095">more</span>

    [:material-file-document: 2607.17095](https://arxiv.org/abs/2607.17095v1) · [:fontawesome-brands-github:](https://github.com/shawn-sypiao/GWPF) · [:material-content-copy: BibTeX](bibtex/2607.17095.bib){ .bibtex-link }

    <span class="md-tag">operator-learning</span>

</div>

