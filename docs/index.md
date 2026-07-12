---
hide:
  - navigation
title: Deep Learning in Weather
---

A collection of papers on deep learning and machine learning applied to weather forecasting, climate modeling, and atmospheric science.

*Last updated: 2026-07-12*

## Recent Additions

<div class="grid cards" markdown>

-   #### Domain-Adaptive Climate Downscaling Under Temporal Distribution Shift

    ---

    *Shuochen Wang, Nishant Yadav, Auroop R. Ganguly* · 2026

    <span class="abstract-snippet" id="snip-2607.05645">Deep-learning-based climate downscaling aims to learn relationships from historical low-resolution (LR) and high-resolution (HR) climate data to generate HR climate projections. However, this setting...</span><span class="abstract-full" id="full-2607.05645" hidden>Deep-learning-based climate downscaling aims to learn relationships from historical low-resolution (LR) and high-resolution (HR) climate data to generate HR climate projections. However, this setting faces a temporal out-of-distribution (OOD) challenge: models trained on historical data are commonly applied to future projections whose distributions may differ substantially from the training period. This study investigates temporal OOD shift for daily temperature downscaling over the Continental United States using paired LR-HR model simulations. We propose a temporal domain-adaptive downscaling framework that combines supervised HR reconstruction on historical data with domain alignment between historical and future climate distributions. Experiments across future validation periods show that the proposed domain-adaptive model consistently outperforms statistical and deep-learning-based bias-correction methods, with the largest gains occurring when the temporal distribution shift is strongest. Spatial analyses indicate stronger improvements over high-elevation and topographically complex regions, along with higher spatiotemporal correlation with the HR target. The extreme analysis shows that domain adaptation also reduces upper-tail temperature bias relative to the non-adaptive model. These results demonstrate that temporal domain adaptation can improve the robustness of HR climate projections under non-stationary climate conditions.</span> <span class="abstract-toggle" data-id="2607.05645">more</span>

    [:material-file-document: 2607.05645](https://arxiv.org/abs/2607.05645v1) · [:material-content-copy: BibTeX](bibtex/2607.05645.bib){ .bibtex-link }

-   #### Exploring Convolutional Neural Processes for Weather Downscaling

    ---

    *Francisco Passos* · 2026

    <span class="abstract-snippet" id="snip-2607.04190">Global reanalysis products such as ERA5-Land provide spatially complete weather fields but at resolutions too coarse for local applications, particularly in mountainous regions where temperature can...</span><span class="abstract-full" id="full-2607.04190" hidden>Global reanalysis products such as ERA5-Land provide spatially complete weather fields but at resolutions too coarse for local applications, particularly in mountainous regions where temperature can vary by several degrees over short distances. This project investigates Convolutional Conditional Neural Processes (ConvCNPs) for statistical downscaling of daily maximum temperature from the ~11km resolution ERA5-Land grid to ~1km resolution over Switzerland, building upon the architecture of Vaughan et al. (2022) and adapting it to the topographically complex Swiss domain with high-resolution elevation features from the swisstopo DHM25. The best model, trained on ten years of data (2014-2023) with five-fold temporal cross-validation, achieves a mean absolute error of 1.31 Celsius and a CRPS-based skill score of 0.524 relative to bilinear interpolation, reducing the expected prediction error by more than half. An ablation study reveals that the elevation MLP is the indispensable component - without it, the model diverges entirely - while explicit seasonal features and Topographic Position Index provide secondary benefits. Under sparse on-grid input the model degrades gracefully, maintaining positive skill down to approximately 10% of the input grid; however, zero-shot deployment on off-grid station observations does not achieve positive skill at any density tested. All configurations exhibit severely overconfident uncertainty estimates, a structural limitation of the Gaussian likelihood training objective. These results demonstrate that ConvCNPs are a viable and effective approach to climate downscaling in complex terrain, and identify uncertainty calibration and native support for non-gridded input as the key challenges for operational deployment.</span> <span class="abstract-toggle" data-id="2607.04190">more</span>

    [:material-file-document: 2607.04190](https://arxiv.org/abs/2607.04190v1) · [:material-content-copy: BibTeX](bibtex/2607.04190.bib){ .bibtex-link }

    <span class="md-tag">CNN</span>

-   #### CORDEX-ML-Bench: A Benchmark for Data-Driven Regional Climate Downscaling -Experiment Design and Overview

    ---

    *Neelesh Rampal, José González-Abad, Henry Addison, Jorge Baño-Medina, Maria Laura Bettolli et al.* · 2026

    <span class="abstract-snippet" id="snip-2606.29172">Machine learning (ML) has emerged as a cost-effective approach to complement dynamical downscaling for producing high-resolution regional climate projections. However, the absence of standardised...</span><span class="abstract-full" id="full-2606.29172" hidden>Machine learning (ML) has emerged as a cost-effective approach to complement dynamical downscaling for producing high-resolution regional climate projections. However, the absence of standardised training and evaluation protocols, applied consistently across multiple domains, continues to hinder meaningful model intercomparison. We introduce CORDEX-ML-Bench, a benchmark aligned with CORDEX, which constitutes the first phase of a community initiative to advance data-driven downscaling toward operational readiness, and complement future dynamical downscaling efforts under CMIP7. The framework targets downscaled daily maximum temperature and precipitation to ~10 km resolution (20x increase) across three pilot regions; European Alps, New Zealand, and Southern Africa. Using a perfect-model experimental design, we evaluate 40 ML configurations developed independently, spanning traditional ML, convolutional U-Nets, vision transformers, graph neural networks, and generative models based on diffusion, flow matching, and generative adversarial networks. Models are trained under two experimental periods, an empirical-statistical downscaling pseudo-reality (historical period only) and Emulator (historical and future periods) -and are evaluated against a core set of metrics developed specifically for assessing downscaling skill. Generative models consistently outperform deterministic approaches for precipitation, better capturing fine-scale variability and extremes. For temperature, the generative advantage narrows and deterministic architectures remain competitive. Models trained solely on the historical period systematically underestimate future climate-change signals while those additionally trained on a future period perform better. These findings raise concerns about historically trained models widely used in an operational setting, underscoring the need for rigorous extrapolation testing.</span> <span class="abstract-toggle" data-id="2606.29172">more</span>

    [:material-file-document: 2606.29172](https://arxiv.org/abs/2606.29172v1) · [:material-content-copy: BibTeX](bibtex/2606.29172.bib){ .bibtex-link }

    <span class="md-tag">transformer</span> <span class="md-tag">diffusion</span> <span class="md-tag">GAN</span> <span class="md-tag">CNN</span> <span class="md-tag">GNN</span>

-   #### Pointwise is Pointless? A Multimodal Ablation Study for Precipitation Nowcasting with Graph Neural Networks

    ---

    *Ophélia Miralles, Máté Mile, Christoffer Artturi, Thomas Nipen, Ivar Seierstad* · 2026

    <span class="abstract-snippet" id="snip-2606.18436">Sparse point observations are increasingly available for precipitation nowcasting, but it is unclear how much they improve dense radar-field forecasts. We partially address this question with a...</span><span class="abstract-full" id="full-2606.18436" hidden>Sparse point observations are increasingly available for precipitation nowcasting, but it is unclear how much they improve dense radar-field forecasts. We partially address this question with a multimodal graph neural network nowcasting system over the Nordic radar domain. The model predicts rain rate every five minutes up to two hours ahead and is trained with different combinations of radar history, MEPS numerical weather prediction, Netatmo surface observations, MSG satellite channels, stochastic noise, and CRPS-based ensemble losses. The study is designed as an ablation of operationally relevant information sources and training objectives. We compare radar-only, NWP-informed, station-informed, satellite-informed, noise-augmented, and CRPS-based configurations using complementary diagnostics on the radar grid, at station locations, for rain onset, and through oracle, displacement, and amplitude scores. The results show that each source improves a different part of the forecast problem. MEPS stabilises radar-only extrapolation, Netatmo observations improve local station and onset diagnostics, and satellite predictors reduce some station-level biases but may activate rain too early when used deterministically. CRPS-based configurations provide the most consistent radar-grid gains, while the combined satellite and CRPS setup gives the best overall oracle/DAS score. These results do not support the conclusion that point observations are uninformative for nowcasting, but they show that local observational skill and spatially coherent radar-field skill are distinct targets. The practical implication is that sparse observations can provide useful local constraints, but their benefit for radar-like fields depends on the training loss, uncertainty representation, and how observation support is encoded in the model.</span> <span class="abstract-toggle" data-id="2606.18436">more</span>

    [:material-file-document: 2606.18436](https://arxiv.org/abs/2606.18436v2) · [:material-content-copy: BibTeX](bibtex/2606.18436.bib){ .bibtex-link }

    <span class="md-tag">GNN</span>

-   #### When the Past Matters: FlashBack Memory for Precipitation Nowcasting

    ---

    *Yuhao Du, Boxiao Huang, Chengrong Wu, Jiankai Zhang* · 2026

    <span class="abstract-snippet" id="snip-2606.16342">Accurate precipitation nowcasting is crucial for disaster mitigation and socio-economic planning, yet existing methods often struggle with false alarms, missed events, and long range dependency...</span><span class="abstract-full" id="full-2606.16342" hidden>Accurate precipitation nowcasting is crucial for disaster mitigation and socio-economic planning, yet existing methods often struggle with false alarms, missed events, and long range dependency modeling at high spatiotemporal resolution. To address these challenges, we propose FlashBack Memory (FB), a module that dynamically retrieves key historical states and integrates them via an adaptive fusion gate, enhancing the spatiotemporal representation capability of recurrent-based models. We incorporate FB into PredRNN, PredRNNpp, MIM, MotionRNN, and PredRNN-V2, and evaluate on CIKM2017, Shanghai2020, and SEVIR datasets. Experimental results demonstrate that FB significantly improves MSE, MAE, SSIM, and CSI metrics, particularly for high-intensity rainfall and long-sequence predictions, while reducing false alarms and missed events and enhancing temporal consistency and spatial localization. The proposed method provides a general and efficient memory enhancement mechanism, improving the overall performance of recurrent-based precipitation nowcasting models.</span> <span class="abstract-toggle" data-id="2606.16342">more</span>

    [:material-file-document: 2606.16342](https://arxiv.org/abs/2606.16342v1) · [:material-content-copy: BibTeX](bibtex/2606.16342.bib){ .bibtex-link }

-   #### Temporal Context Conditioning for Seasonality-Aware Precipitation Nowcasting of High-Intensity Rainfall

    ---

    *Gijs van Nieuwkoop, Siamak Mehrkanoon* · 2026

    <span class="abstract-snippet" id="snip-2606.09959">Precipitation nowcasting is increasingly being approached with deep learning models that learn directly from recent radar observations. Although such models can efficiently capture short-term...</span><span class="abstract-full" id="full-2606.09959" hidden>Precipitation nowcasting is increasingly being approached with deep learning models that learn directly from recent radar observations. Although such models can efficiently capture short-term precipitation motion, they often lack broader contextual information about the meteorological conditions under which rainfall develops. This paper investigates whether lightweight temporal context can improve radar-based nowcasting, particularly for high-intensity rainfall. We propose the Time-Aware Small-Attention U-Net (TA-SmaAt-UNet), which extends the core SmaAt-UNet model with temporal conditioning layers that use cyclical encodings of time-of-day and time-of-year to modulate intermediate feature representations. Experiments on KNMI radar precipitation data show that temporal conditioning is most beneficial for rare, high-intensity precipitation events, while also improving the representation of seasonal variability and predicted rainfall-intensity distributions. A layer conductance analysis further indicates that the added temporal conditioning layers are actively used by the model despite their small parameter cost. These findings suggest that simple, physically motivated temporal context can improve the realism and reliability of deep learning-based precipitation nowcasts. The implementation of our models and training setup is available on \href{https://github.com/gijsvn/TA-SmaAt-UNet}{GitHub}.</span> <span class="abstract-toggle" data-id="2606.09959">more</span>

    [:material-file-document: 2606.09959](https://arxiv.org/abs/2606.09959v1) · [:fontawesome-brands-github:](https://github.com/gijsvn/TA-SmaAt-UNet) · [:material-content-copy: BibTeX](bibtex/2606.09959.bib){ .bibtex-link }

    <span class="md-tag">CNN</span>

-   #### Learning to Solve Generative ODEs Beyond the Linear Span

    ---

    *Sihyeon Kim, Seunghun Lee, Vikas Singh, Hyunwoo J. Kim* · 2026

    <span class="abstract-snippet" id="snip-2606.08672">Diffusion and flow generative models sample by integrating a learned ODE, but high quality still requires many sequential model evaluations. Solver learning reduces this cost by adapting scalar...</span><span class="abstract-full" id="full-2606.08672" hidden>Diffusion and flow generative models sample by integrating a learned ODE, but high quality still requires many sequential model evaluations. Solver learning reduces this cost by adapting scalar coefficients, timesteps, or both, while keeping the backbone model fixed. In this work, we identify a structural bottleneck in this update family: each step remains span-limited. Since the scalar-coefficient update lies in the span of buffered velocity evaluations, it can fit only the in-span component while leaving any out-of-span residual unreachable by scalar recombination alone. We propose SpanLift, a lightweight neural solver that augments scalar-coefficient updates with a spatial residual operator. SpanLift keeps a fixed base solver as an in-span prior and learns a spatial residual operator over the state and velocity buffer. The operator is trained by endpoint teacher matching, preserves the pretrained backbone, and adds no model NFEs. Empirically, the learned correction transfers across base solvers and is predominantly out-of-span. Across pixel-space diffusion, latent flow matching, and precipitation nowcasting, SpanLift achieves state-of-the-art few-step sampling. With only 3 NFE, it improves CIFAR-10 FID from 8.16 to 5.69 and ImageNet FID from 17.37 to 11.83.</span> <span class="abstract-toggle" data-id="2606.08672">more</span>

    [:material-file-document: 2606.08672](https://arxiv.org/abs/2606.08672v1) · [:material-content-copy: BibTeX](bibtex/2606.08672.bib){ .bibtex-link }

    <span class="md-tag">diffusion</span>

-   #### Temporal Coverage over Density: Parsimonious Training-Set Design for ML Climate Downscaling

    ---

    *Karandeep Singh, Stefan Rahimi, Chad W. Thackeray, Stephen Cropper, Alex Hall* · 2026

    <span class="abstract-snippet" id="snip-2606.07898">High-resolution regional climate simulations provide critical information for climate impacts assessments but remain computationally expensive, motivating the development of machine-learning...</span><span class="abstract-full" id="full-2606.07898" hidden>High-resolution regional climate simulations provide critical information for climate impacts assessments but remain computationally expensive, motivating the development of machine-learning downscalers and emulators. A key challenge is determining how limited high-resolution simulations should be distributed across a changing climate trajectory to capture both forced climate response and internal variability. Using the CESM2 Large Ensemble over the western United States, we compare three training-year selection strategies under fixed data budgets: a contiguous block of historical years, years drawn from both the beginning and end of the simulation period, and years distributed throughout the full climate trajectory. Including both historical and future years consistently outperforms training on historical years alone, demonstrating the importance of exposing downscaling models to climate states outside the historical record and highlighting limitations of stationarity assumptions common in statistical downscaling. Training on years distributed throughout the full climate trajectory performs best overall, indicating that broad sampling of internal variability provides additional information beyond exposure to the forced climate response alone. Models trained on temporally distributed subsets more successfully reproduce variability in unseen ensemble members while retaining strong performance across a wide range of climate diagnostics. Even when trained on only one-tenth of the available high-resolution years, temporally distributed models remain highly competitive with full-data training. These results suggest that, under fixed computational budgets, broad sampling of climate states is more valuable than temporal continuity when allocating scarce high-resolution simulations. The findings provide practical guidance for regional climate downscaling and large-ensemble projection workflows.</span> <span class="abstract-toggle" data-id="2606.07898">more</span>

    [:material-file-document: 2606.07898](https://arxiv.org/abs/2606.07898v1) · [:material-content-copy: BibTeX](bibtex/2606.07898.bib){ .bibtex-link }

-   #### Learning to Refine: Spectral-Decoupled Iterative Refinement Framework for Precipitation Nowcasting

    ---

    *Yunlong Zhou, Chen Zhao, Danyang Peng, Fanfan Ji, Xiao-Tong Yuan* · 2026

    <span class="abstract-snippet" id="snip-2606.02661">Accurate precipitation nowcasting is vital for disaster mitigation, but deep learning methods face a key trade-off: regression models produce over-smoothed, spectrally decaying predictions that blur...</span><span class="abstract-full" id="full-2606.02661" hidden>Accurate precipitation nowcasting is vital for disaster mitigation, but deep learning methods face a key trade-off: regression models produce over-smoothed, spectrally decaying predictions that blur convective details and violate turbulence power laws; diffusion models generate realistic yet unanchored hallucinations lacking physical grounding. We propose Spectral-Decoupled Iterative Refinement (SDIR), a deterministic framework that reformulates nowcasting as progressive frequency-decoupled refinement. SDIR first extracts a stable low-frequency synoptic skeleton, then iteratively refines high-frequency textures under physical constraints, eliminating both blurring and hallucinations. It features a dual-path design: the Synoptic Frequency-Guided Former (SFG-Former) with Scale-Adaptive Transformers for global structure, and the Fourier Residual Refiner (FR-Refiner) with Scale-Conditioned Fourier Neural Operators for fine residuals. A Physically Consistent Power Spectral Density (PCPSD) loss with dynamic masking enforces a turbulence-consistent spectral distribution. Experiments on three benchmarks show SDIR significantly outperforms SOTA methods in spatial accuracy while achieving spectral fidelity competitive with diffusion-based methods, enabling reliable high-resolution operational nowcasting. Code link: https://github.com/RuntimeWarning/SDIR.</span> <span class="abstract-toggle" data-id="2606.02661">more</span>

    [:material-file-document: 2606.02661](https://arxiv.org/abs/2606.02661v1) · [:fontawesome-brands-github:](https://github.com/RuntimeWarning/SDIR) · [:material-content-copy: BibTeX](bibtex/2606.02661.bib){ .bibtex-link }

    <span class="md-tag">transformer</span> <span class="md-tag">diffusion</span> <span class="md-tag">operator-learning</span>

-   #### Probabilistic Precipitation Nowcasting with Rectified Flow Transformers

    ---

    *Johannes Schusterbauer, Jannik Wiese, Nick Stracke, Timy Phan, Björn Ommer* · 2026

    <span class="abstract-snippet" id="snip-2605.31204">Accurate weather forecasts are essential across various domains and are safety-critical in extreme weather conditions. Compared to simulation-based forecasting, data-driven approaches show greater...</span><span class="abstract-full" id="full-2605.31204" hidden>Accurate weather forecasts are essential across various domains and are safety-critical in extreme weather conditions. Compared to simulation-based forecasting, data-driven approaches show greater efficiency, enabling short-term, high-resolution nowcasting. In particular, diffusion models proved effective in weather nowcasting due to their strong probabilistic foundation. However, existing methods rely on deterministic compression to reduce the complexity of high-dimensional weather data, limiting their ability to capture uncertainty in the decoding process. In this work, we introduce $\textbf{FREUD}$, a $\textbf{Fr}$ame-wise $\textbf{E}$ncoder and $\textbf{U}$nited $\textbf{D}$ecoder model based on rectified flow transformers for efficient compression of spatio-temporal weather data. Frame-wise encoding enables continuous forecast updates, while the unified video decoder ensures temporal consistency. Our uncertainty-preserving first stage allows us to capture aleatoric uncertainty via ensembling, which is particularly beneficial for extreme weather events with high decoding variability. We achieve state-of-the-art performance in precipitation nowcasting with a compact latent-space rectified flow transformer on the SEVIR benchmark and show further performance gains by model and test-time scaling. Code available here: https://github.com/CompVis/weather-rf</span> <span class="abstract-toggle" data-id="2605.31204">more</span>

    [:material-file-document: 2605.31204](https://arxiv.org/abs/2605.31204v1) · [:fontawesome-brands-github:](https://github.com/CompVis/weather-rf) · [:material-content-copy: BibTeX](bibtex/2605.31204.bib){ .bibtex-link }

    <span class="md-tag">transformer</span> <span class="md-tag">diffusion</span> <span class="md-tag">probabilistic</span>

</div>

