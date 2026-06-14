---
hide:
  - navigation
title: Deep Learning in Weather
---

A collection of papers on deep learning and machine learning applied to weather forecasting, climate modeling, and atmospheric science.

*Last updated: 2026-06-14*

## Recent Additions

<div class="grid cards" markdown>

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

-   #### Forecasting threshold exceedance of atmospheric variables at a specific location

    ---

    *Roberta Baggio, Jean-François Muzy* · 2026

    <span class="abstract-snippet" id="snip-2605.31079">This study compares two methodological approaches for predicting, at a given site, threshold exceedances of atmospheric variables such as temperature and wind speed: (i) direct probabilistic methods,...</span><span class="abstract-full" id="full-2605.31079" hidden>This study compares two methodological approaches for predicting, at a given site, threshold exceedances of atmospheric variables such as temperature and wind speed: (i) direct probabilistic methods, which treat exceedance as a binary classification problem, and (ii) full distribution probabilistic methods, which model the complete conditional probability law of the target variable. Using theoretical analysis and numerical simulations on a toy model, alongside real-world data from the MeteoNet dataset (2016--2018) for southeastern France, we demonstrate that the full distribution approach consistently outperforms the direct method for rare, extreme events. This advantage arises because the full distribution approach effectively learns the parameters of the conditional distribution from moderate and mild intensity events, thereby achieving better calibration and discrimination in the tails. We find that the specific parametric shape of the chosen distribution plays a secondary role compared to accurately capturing predictable shifts in its bulk properties (i.e., mean and variance). This empirical indistinguishability is also informative about the physical mechanics driving atmospheric extremes, suggesting that extreme exceedances are primarily driven by significant conditional displacements of the entire distribution rather than by unpredictable, fat-tailed anomalies within a static climatology. Our results are validated for both strong surface wind speeds and intense hourly rainfall, with performance evaluated using proper scoring rules (Brier score, logarithmic score) and deterministic skill scores (Peirce Skill Score, CSI, HSS). These findings highlight the critical importance of modeling the full probability distribution for rare-event forecasting and provide practical guidance for improving extreme weather prediction in operational meteorology.</span> <span class="abstract-toggle" data-id="2605.31079">more</span>

    [:material-file-document: 2605.31079](https://arxiv.org/abs/2605.31079v1) · [:material-content-copy: BibTeX](bibtex/2605.31079.bib){ .bibtex-link }

    <span class="md-tag">probabilistic</span>

-   #### Beyond MSE: Improving Precipitation Nowcasting with Multi-Quantile Regression

    ---

    *Gijs van Nieuwkoop, Siamak Mehrkanoon* · 2026

    <span class="abstract-snippet" id="snip-2605.30122">Deep-learning precipitation nowcasting models are often optimized using pointwise losses such as mean squared error or mean absolute error, which can lead to overly smooth forecasts and poor...</span><span class="abstract-full" id="full-2605.30122" hidden>Deep-learning precipitation nowcasting models are often optimized using pointwise losses such as mean squared error or mean absolute error, which can lead to overly smooth forecasts and poor representation of heavy rainfall. This study investigates whether the predictive performance of an established deterministic nowcasting architecture can be improved by reformulating training as a multi-quantile regression problem. Using SmaAt-UNet as a core model, we compare MSE, MAE, and multi-quantile pinball-loss training on radar precipitation nowcasting over the Netherlands. The results show that multi-quantile training improves the central deterministic forecast, decreasing test-set MSE by 8.6\% compared to a model trained using MSE, while also producing upper-quantile outputs that are useful for risk-sensitive prediction of heavy precipitation. These findings suggest that quantile regression provides a simple alternative to standard pointwise losses without requiring a new architecture or generative sampling procedure. The implementation of our models and training setup is available on \href{https://github.com/gijsvn/Multi-Quantile-Precipitation-Nowcasting}{GitHub}.</span> <span class="abstract-toggle" data-id="2605.30122">more</span>

    [:material-file-document: 2605.30122](https://arxiv.org/abs/2605.30122v2) · [:fontawesome-brands-github:](https://github.com/gijsvn/Multi-Quantile-Precipitation-Nowcasting) · [:material-content-copy: BibTeX](bibtex/2605.30122.bib){ .bibtex-link }

    <span class="md-tag">CNN</span>

-   #### Beyond MSE: Improving Precipitation Nowcasting with Multi-Quantile Regression

    ---

    *Gijs van Nieuwkoop, Siamak Mehrkanoon* · 2026

    <span class="abstract-snippet" id="snip-2605.30122">Deep-learning precipitation nowcasting models are often optimized using pointwise losses such as mean squared error or mean absolute error, which can lead to overly smooth forecasts and poor...</span><span class="abstract-full" id="full-2605.30122" hidden>Deep-learning precipitation nowcasting models are often optimized using pointwise losses such as mean squared error or mean absolute error, which can lead to overly smooth forecasts and poor representation of heavy rainfall. This study investigates whether the predictive performance of an established deterministic nowcasting architecture can be improved by reformulating training as a multi-quantile regression problem. Using SmaAt-UNet as a core model, we compare MSE, MAE, and multi-quantile pinball-loss training on radar precipitation nowcasting over the Netherlands. The results show that multi-quantile training improves the central deterministic forecast, decreasing test-set MSE by 8.6\% compared to a model trained using MSE, while also producing upper-quantile outputs that are useful for risk-sensitive prediction of heavy precipitation. These findings suggest that quantile regression provides a simple alternative to standard pointwise losses without requiring a new architecture or generative sampling procedure. The implementation of our models and training setup is available on \href{https://github.com/gijsvn/Multi-Quantile-Precipitation-Nowcasting}{GitHub}.</span> <span class="abstract-toggle" data-id="2605.30122">more</span>

    [:material-file-document: 2605.30122](https://arxiv.org/abs/2605.30122v1) · [:fontawesome-brands-github:](https://github.com/gijsvn/Multi-Quantile-Precipitation-Nowcasting) · [:material-content-copy: BibTeX](bibtex/2605.30122.bib){ .bibtex-link }

    <span class="md-tag">CNN</span>

-   #### MambaRain: Multi-Scale Mamba-Attention Framework for 0-3 Hour Precipitation Nowcasting

    ---

    *Chunlei Shi, Cui Wu, Xiang Xu, Hao Li, Ni Fan, Xue Han, Yongchao Feng, Yufeng Zhu, Boyu Liu et al.* · 2026

    <span class="abstract-snippet" id="snip-2605.14606">Accurate precipitation nowcasting over extended horizons (0-3 hours) is essential for disaster mitigation and operational decision-making, yet remains a critical challenge in the field. Existing...</span><span class="abstract-full" id="full-2605.14606" hidden>Accurate precipitation nowcasting over extended horizons (0-3 hours) is essential for disaster mitigation and operational decision-making, yet remains a critical challenge in the field. Existing deterministic approaches are predominantly constrained to shorter prediction windows (0-2 hours), exhibiting severe performance degradation beyond 90 minutes owing to their inherent difficulty in capturing long-range spatiotemporal dependencies from radar-derived observations. To address these fundamental limitations, we propose MambaRain, a novel multi-scale encoder-decoder architecture that synergistically integrates Mamba's linear-complexity long-range temporal modeling with self-attention mechanisms for explicit spatial correlation capture. The core innovation lies in a hybrid design paradigm wherein Mamba blocks leverage selective state space mechanisms to model global temporal dynamics across extended sequences with computational efficiency, while self-attention modules explicitly characterize spatial correlations within precipitation fields - a capability inherently absent in Mamba's sequential processing paradigm. This complementary synergy enables comprehensive spatiotemporal representation learning, effectively extending the viable forecasting horizon to 2-3 hours with substantial accuracy improvements. Furthermore, we introduce a spectral loss formulation to mitigate blurring artifacts characteristic of chaotic precipitation systems, thereby preserving fine-scale motion details critical for nowcasting accuracy. Experimental validation demonstrates that MambaRain substantially outperforms existing deterministic methodologies in 0-3 hour nowcasting tasks, with particularly pronounced performance gains in the challenging 2-3 hour prediction range.</span> <span class="abstract-toggle" data-id="2605.14606">more</span>

    [:material-file-document: 2605.14606](https://arxiv.org/abs/2605.14606v1) · [:material-content-copy: BibTeX](bibtex/2605.14606.bib){ .bibtex-link }

    <span class="md-tag">transformer</span>

-   #### VMU-Diff: A Coarse-to-fine Multi-source Data Fusion Framework for Precipitation Nowcasting

    ---

    *Chunlei Shi, Hao Li, Yufeng Zhu, Boyu Liu, Yongchao Feng, Zengliang Zang, Hongbin Wang, Yanlan Yang et al.* · 2026

    <span class="abstract-snippet" id="snip-2605.14597">Precipitation nowcasting is a vital spatio-temporal prediction task for meteorological applications but faces challenges due to the chaotic property of precipitation systems. Existing methods...</span><span class="abstract-full" id="full-2605.14597" hidden>Precipitation nowcasting is a vital spatio-temporal prediction task for meteorological applications but faces challenges due to the chaotic property of precipitation systems. Existing methods predominantly rely on single-source radar data to build either deterministic or probabilistic models for extrapolation. However, the single deterministic model suffers from blurring due to MSE convergence. The single probabilistic model, typically represented by diffusion models, can generate fine details but suffers from spurious artifacts that compromise accuracy and computational inefficiency. To address these challenges, this paper proposes a novel coarse-to-fine Vision Mamba Unet and residual Diffusion (VMU-Diff) based precipitation nowcasting framework. It realizes precipitation nowcasting through a two-stage process, i.e., a deterministic model-based coarse stage to predict global motion trends and a probabilistic model-based fine stage to generate fine prediction details. In the coarse prediction stage, rather than single-source radar data, both radar and multi-band satellite data are taken as input. A spatial-temporal attention block and several Vision mamba state-space blocks realize multi-source data fusion, and predict the future echo global dynamics. The fine-grained stage is realized by a spatio-temporal refine generator based on residual conditional diffusion models. It first obtains spatio-temporal residual features based on coarse prediction and ground truth, and further reconstructs the residual via conditional Mamba state-space module. Experiments on Jiangsu SWAN datasets demonstrate the improvements of our method over state-of-the-art methods, particularly in short-term forecasts.</span> <span class="abstract-toggle" data-id="2605.14597">more</span>

    [:material-file-document: 2605.14597](https://arxiv.org/abs/2605.14597v1) · [:material-content-copy: BibTeX](bibtex/2605.14597.bib){ .bibtex-link }

    <span class="md-tag">diffusion</span> <span class="md-tag">CNN</span> <span class="md-tag">probabilistic</span>

</div>

