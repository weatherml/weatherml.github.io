---
hide:
  - navigation
title: Deep Learning in Weather
---

A collection of papers on deep learning and machine learning applied to weather forecasting, climate modeling, and atmospheric science.

*Last updated: 2026-06-08*

## Recent Additions

<div class="grid cards" markdown>

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

-   #### ForcingDAS: Unified and Robust Data Assimilation via Diffusion Forcing

    ---

    *Yixuan Jia, Siyi Chen, Yida Pan, Xiao Li, Lianghe Shi, Chanyong Jung, Haijie Yuan, Ismail Alkhouri et al.* · 2026

    <span class="abstract-snippet" id="snip-2605.14285">Data assimilation (DA) estimates the state of an evolving dynamical system from noisy, partial observations, and is widely used in scientific simulation as well as weather and climate science. In...</span><span class="abstract-full" id="full-2605.14285" hidden>Data assimilation (DA) estimates the state of an evolving dynamical system from noisy, partial observations, and is widely used in scientific simulation as well as weather and climate science. In practice, filtering methods rely on frame-to-frame transition models. However, these models are fragile when observations are non-Markovian (when they form only a partial slice of a higher-dimensional latent state as in real-world weather data): they tend to accumulate errors over long horizons. At the same time, learned DA methods typically commit to a single regime, either filtering (nowcasting, real-time forecasting) or smoothing (retrospective reanalysis), which splits what should be a shared prior across application-specific pipelines. To address both issues, we introduce ForcingDAS, a unified and robust DA framework. Built on Diffusion Forcing with an independent noise level assigned to each frame, ForcingDAS learns a joint-trajectory prior instead of frame-to-frame transitions. This allows it to capture long-horizon temporal dependencies and reduce error accumulation. In addition, the same trained model spans the full filtering to smoothing spectrum at inference time. Specifically, nowcasting, fixed-lag smoothing, and batch reanalysis are selected through the inference schedule alone, without retraining. We evaluate ForcingDAS on 2D Navier-Stokes vorticity, precipitation nowcasting, and global atmospheric state estimation. Across all settings, a single model is competitive with or outperforms both learned and classical baselines that are specialized for individual regimes, with the largest gains observed on real-world weather benchmarks.</span> <span class="abstract-toggle" data-id="2605.14285">more</span>

    [:material-file-document: 2605.14285](https://arxiv.org/abs/2605.14285v1) · [:material-content-copy: BibTeX](bibtex/2605.14285.bib){ .bibtex-link }

-   #### McCast: Memory-Guided Latent Drift Correction for Long-Horizon Precipitation Nowcasting

    ---

    *Penghui Wen, Yu Luo, Lintao Wang, Mengwei He, Patrick Filippi, Thomas Francis Bishop, Zhiyong Wang* · 2026

    <span class="abstract-snippet" id="snip-2605.13197">Existing precipitation nowcasting methods typically adopt an autoregressive formulation, where future states are predicted from previous outputs. However, such an approach accumulates errors over...</span><span class="abstract-full" id="full-2605.13197" hidden>Existing precipitation nowcasting methods typically adopt an autoregressive formulation, where future states are predicted from previous outputs. However, such an approach accumulates errors over long rollouts, causing forecasts to drift away from physically plausible evolution trajectories. Although various studies have attempted to alleviate this problem by improving step-wise prediction accuracy, they largely neglect the global temporal evolution of meteorological systems and lack mechanisms to actively correct drift during rollouts. To address this issue, we propose McCast, a memory-guided latent drift correction method for precipitation nowcasting. Rather than treating memory as an unordered dictionary of latent states for passive conditioning, McCast leverages temporally organized memory to actively correct autoregressive latent evolution. Specifically, McCast introduces a Drift-Corrective Memory Bank (DCBank) that explicitly estimates the temporally consistent drift corrections to calibrate the divergent trajectory. DCBank performs drift correction in two stages: a Corrective Latent Extractor first predicts an initial correction from the current prediction and a reference latent state, and a Correction-Aware Memory Retrieval module then refines the initial correction using temporally organized historical memory. By explicitly correcting latent evolution, instead of improving step-wise prediction accuracy only, McCast produces more temporally coherent and reliable long-horizon forecasts. Experiments on two widely used benchmarks, SEVIR and MeteoNet, show that McCast achieves state-of-the-art performance, particularly in challenging long-horizon forecasting scenarios.</span> <span class="abstract-toggle" data-id="2605.13197">more</span>

    [:material-file-document: 2605.13197](https://arxiv.org/abs/2605.13197v1) · [:material-content-copy: BibTeX](bibtex/2605.13197.bib){ .bibtex-link }

-   #### Stable Attention Response for Reliable Precipitation Nowcasting

    ---

    *Penghui Wen, Zexin Hu, Sen Zhang, Patrick Filippi, Xiaogang Zhu, Allen Benter, Thomas Bishop et al.* · 2026

    <span class="abstract-snippet" id="snip-2605.13181">Precipitation nowcasting remains challenging due to the highly localized, rapidly evolving, and heterogeneous nature of atmospheric dynamics. Although recent methods increasingly adopt...</span><span class="abstract-full" id="full-2605.13181" hidden>Precipitation nowcasting remains challenging due to the highly localized, rapidly evolving, and heterogeneous nature of atmospheric dynamics. Although recent methods increasingly adopt attention-based architectures in both unimodal and multimodal settings, they mainly emphasize stronger representation learning and prediction capacity, while paying less attention to the stability of attention responses across samples. In this work, we show that cross-sample instability of attention-response energy is an important and previously underexplored source of forecasting unreliability. Empirically, inaccurate forecasts are associated with larger attention-response energy variance across heads and layers. Theoretically, we show that cross-sample variability can propagate through self-attention, and enlarge a lower bound on prediction error. Based on this insight, we propose HARECast, a Head-wise Attention Response Energy-regulated framework for precipitation nowcasting. HARECast explicitly models head-wise attention-response energy and stabilizes it through a group-wise regularization objective that reduces cross-sample fluctuations. The proposed formulation is generic and applicable to both unimodal and multimodal nowcasting architectures. We instantiate HARECast in a standard forecasting pipeline with reconstruction branches and a diffusion-based predictor, and evaluate it on commonly used benchmarks--SEVIR and MeteoNet. Experimental results demonstrate that HARECast achieves state-of-the-art performance.</span> <span class="abstract-toggle" data-id="2605.13181">more</span>

    [:material-file-document: 2605.13181](https://arxiv.org/abs/2605.13181v1) · [:material-content-copy: BibTeX](bibtex/2605.13181.bib){ .bibtex-link }

    <span class="md-tag">transformer</span>

</div>

