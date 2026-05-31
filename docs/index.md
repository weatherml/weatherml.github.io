---
hide:
  - navigation
title: Deep Learning in Weather
---

A collection of papers on deep learning and machine learning applied to weather forecasting, climate modeling, and atmospheric science.

*Last updated: 2026-05-31*

## Recent Additions

<div class="grid cards" markdown>

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

-   #### Generative climate downscaling enables high-resolution compound risk assessment by preserving multivariate dependencies

    ---

    *Takuro Kutsuna, Noriko N. Ishizaki, Norihiro Oyama, Hiroaki Yoshida* · 2026

    <span class="abstract-snippet" id="snip-2605.11531">Physics-based climate projections using general circulation models are essential for assessing future risks, but their coarse resolution limits regional decision-making. Statistical downscaling can...</span><span class="abstract-full" id="full-2605.11531" hidden>Physics-based climate projections using general circulation models are essential for assessing future risks, but their coarse resolution limits regional decision-making. Statistical downscaling can efficiently add detail, yet many methods treat variables independently, degrading inter-variable relationships that govern compound hazards such as heat stress, drought, and wildfire. Here we show that a diffusion-based multivariate generative framework, combined with bias correction, recovers degraded inter-variable correlations even under a 50$\times$ increase in linear resolution. When applied to five meteorological variables over Japan, the framework reduces inter-variable correlation errors by more than fourfold relative to existing baselines while improving both univariate and spatial accuracy, leading to more accurate detection of severe drought. These results demonstrate that multivariate generative downscaling improves the reliability of compound risk assessment under large resolution gaps.</span> <span class="abstract-toggle" data-id="2605.11531">more</span>

    [:material-file-document: 2605.11531](https://arxiv.org/abs/2605.11531v1) · [:material-content-copy: BibTeX](bibtex/2605.11531.bib){ .bibtex-link }

    <span class="md-tag">physics-informed</span>

-   #### PixelFlowCast: Latent-Free Precipitation Nowcasting via Pixel Mean Flows

    ---

    *Yufeng Zhu, Chunlei Shi, Yongchao Feng, Dan Niu* · 2026

    <span class="abstract-snippet" id="snip-2605.10046">Precipitation nowcasting aims to forecast short-term radar echo sequences for extreme weather warning, where both prediction fidelity and inference efficiency are critical for real-world deployment....</span><span class="abstract-full" id="full-2605.10046" hidden>Precipitation nowcasting aims to forecast short-term radar echo sequences for extreme weather warning, where both prediction fidelity and inference efficiency are critical for real-world deployment. However, diffusion-based models, despite their strong generative capability, suffer from slow inference due to multi-step sampling trajectories, limiting their practical usability. Conditional Flow Matching (CFM) improves efficiency via straightened trajectories, but relies on latent space compression, which inevitably discards high-frequency physical details and degrades fine-grained prediction quality. To address these limitations, we propose PixelFlowCast, a two-stage probabilistic forecasting framework that achieves both high-efficiency and high-fidelity prediction without latent compression. Specifically, in the first stage, a deterministic model first produces coarse forecasts to capture global evolution trends. In the subsequent stage, the proposed KANCondNet extracts deep spatiotemporal evolution features to provide accurate conditional guidance. Based on this, a latent-free, few-step Pixel Mean Flows (PMF) predictor employs an $x$-prediction mechanism to generate high-quality predictions, effectively preserving fine-grained structures while maintaining fast inference. Experiments on the publicly available SEVIR dataset demonstrate that PixelFlowCast outperforms existing mainstream methods in both prediction accuracy and inference efficiency, particularly for long sequence forecasting, highlighting its strong potential for real-world operational deployment.</span> <span class="abstract-toggle" data-id="2605.10046">more</span>

    [:material-file-document: 2605.10046](https://arxiv.org/abs/2605.10046v1) · [:material-content-copy: BibTeX](bibtex/2605.10046.bib){ .bibtex-link }

    <span class="md-tag">diffusion</span> <span class="md-tag">probabilistic</span>

-   #### M3R: Localized Rainfall Nowcasting with Meteorology-Informed MultiModal Attention

    ---

    *Sanjeev Panta, Rhett M Morvant, Xu Yuan, Li Chen, Nian-Feng Tzeng* · 2026

    <span class="abstract-snippet" id="snip-2604.15377">Accurate and timely rainfall nowcasting is crucial for disaster mitigation and water resource management. Despite recent advances in deep learning, precipitation prediction remains challenging due to...</span><span class="abstract-full" id="full-2604.15377" hidden>Accurate and timely rainfall nowcasting is crucial for disaster mitigation and water resource management. Despite recent advances in deep learning, precipitation prediction remains challenging due to limitations in effectively leveraging diverse multimedia data sources. We introduce M3R, a Meteorology-informed MultiModal attention-based architecture for direct Rainfall prediction that synergistically combines visual NEXRAD radar imagery with numerical Personal Weather Station (PWS) measurements, using a comprehensive pipeline for temporal alignment of heterogeneous meteorological data. With specialized multimodal attention mechanisms, M3R novelly leverages weather station time series as queries to selectively attend to spatial radar features, enabling focused extraction of precipitation signatures. Experimental results for three spatial areas of 100 km * 100 km centered at NEXRAD radar stations demonstrate that M3R outperforms existing approaches, achieving substantial improvements in accuracy, efficiency, and precipitation detection capabilities. Our work establishes new benchmarks for multimedia-based precipitation nowcasting and provides practical tools for operational weather prediction systems. The source code is available at https://github.com/Sanjeev97/M3Rain</span> <span class="abstract-toggle" data-id="2604.15377">more</span>

    [:material-file-document: 2604.15377](https://arxiv.org/abs/2604.15377v1) · [:fontawesome-brands-github:](https://github.com/Sanjeev97/M3Rain) · [:material-content-copy: BibTeX](bibtex/2604.15377.bib){ .bibtex-link }

    <span class="md-tag">transformer</span>

-   #### Capturing Aleatoric Uncertainty in Climate Models

    ---

    *Cornelia Gruber, Henri Funk, Magdalena Mittermeier, Helmut Küchenhoff, Göran Kauermann* · 2026

    <span class="abstract-snippet" id="snip-2604.15067">Internal climate variability arises from the climate system's inherently chaotic dynamics. Quantifying it is essential for climate science, as it enables risk-based decision-making and differentiates...</span><span class="abstract-full" id="full-2604.15067" hidden>Internal climate variability arises from the climate system's inherently chaotic dynamics. Quantifying it is essential for climate science, as it enables risk-based decision-making and differentiates between externally forced change and internal fluctuations. In statistical terms, natural variability corresponds to aleatoric uncertainty, i.e., irreducible stochastic variability. Despite this close conceptual alignment, the link between internal climate variability and aleatoric uncertainty has not yet been formalized. We establish a theoretical link by showing that member-to-member differences in single-model large ensembles provide a direct representation of aleatoric uncertainty. To quantify the spatio-temporal structure of aleatoric uncertainty, we employ generalized additive models. The proposed framework is validated through comparison with ERA5-Land reanalysis data, demonstrating that ensemble-derived estimates reproduce key spatial and temporal patterns of real-world variability. Applied to the water balance over the Iberian Peninsula, our approach reveals coherent variability structures and pronounced regional heterogeneity. We find a decline in variability in drought-prone regions and seasons, a pattern that strengthens under +3 °C global warming, implying an increased risk of persistent summer drought conditions. Beyond this application, the framework is climate-model agnostic and transferable to other variables and spatial scales, providing a statistical basis for quantifying internal climate variability as aleatoric uncertainty.</span> <span class="abstract-toggle" data-id="2604.15067">more</span>

    [:material-file-document: 2604.15067](https://arxiv.org/abs/2604.15067v1) · [:material-content-copy: BibTeX](bibtex/2604.15067.bib){ .bibtex-link }

</div>

