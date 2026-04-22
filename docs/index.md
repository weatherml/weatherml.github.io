---
hide:
  - navigation
title: Deep Learning in Weather
---

A collection of papers on deep learning and machine learning applied to weather forecasting, climate modeling, and atmospheric science.

*Last updated: 2026-04-22*

## Recent Additions

<div class="grid cards" markdown>

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

-   #### Generative 3D Gaussian Splatting for Arbitrary-ResolutionAtmospheric Downscaling and Forecasting

    ---

    *Tao Hana, Zhibin Wen, Zhenghao Chen, Fenghua Lin, Junyu Gao, Song Guo, Lei Bai* · 2026

    <span class="abstract-snippet" id="snip-2604.07928">While AI-based numerical weather prediction (NWP) enables rapid forecasting, generating high-resolution outputs remains computationally demanding due to limited multi-scale adaptability and...</span><span class="abstract-full" id="full-2604.07928" hidden>While AI-based numerical weather prediction (NWP) enables rapid forecasting, generating high-resolution outputs remains computationally demanding due to limited multi-scale adaptability and inefficient data representations. We propose the 3D Gaussian splatting-based scale-aware vision transformer (GSSA-ViT), a novel framework for arbitrary-resolution forecasting and flexible downscaling of high-dimensional atmospheric fields. Specifically, latitude-longitude grid points are treated as centers of 3D Gaussians. A generative 3D Gaussian prediction scheme is introduced to estimate key parameters, including covariance, attributes, and opacity, for unseen samples, improving generalization and mitigating overfitting. In addition, a scale-aware attention module is designed to capture cross-scale dependencies, enabling the model to effectively integrate information across varying downscaling ratios and support continuous resolution adaptation. To our knowledge, this is the first NWP approach that combines generative 3D Gaussian modeling with scale-aware attention for unified multi-scale prediction. Experiments on ERA5 show that the proposed method accurately forecasts 87 atmospheric variables at arbitrary resolutions, while evaluations on ERA5 and CMIP6 demonstrate its superior performance in downscaling tasks. The proposed framework provides an efficient and scalable solution for high-resolution, multi-scale atmospheric prediction and downscaling. Code is available at: https://github.com/binbin2xs/weather-GS.</span> <span class="abstract-toggle" data-id="2604.07928">more</span>

    [:material-file-document: 2604.07928](https://arxiv.org/abs/2604.07928v1) · [:fontawesome-brands-github:](https://github.com/binbin2xs/weather-GS) · [:material-content-copy: BibTeX](bibtex/2604.07928.bib){ .bibtex-link }

    <span class="md-tag">transformer</span>

-   #### El Nino Prediction Based on Weather Forecast and Geographical Time-series Data

    ---

    *Viet Trinh, Ha-Vy Luu, Quoc-Khiem Nguyen-Pham, Hung Tong, Thanh-Huyen Tran, Hoai-Nam Nguyen Dang* · 2026

    <span class="abstract-snippet" id="snip-2604.04998">This paper proposes a novel framework for enhancing the prediction accuracy and lead time of El Niño events, crucial for mitigating their global climatic, economic, and societal impacts. Traditional...</span><span class="abstract-full" id="full-2604.04998" hidden>This paper proposes a novel framework for enhancing the prediction accuracy and lead time of El Niño events, crucial for mitigating their global climatic, economic, and societal impacts. Traditional prediction models often rely on oceanic and atmospheric indices, which may lack the granularity or dynamic interplay captured by comprehensive meteorological and geographical datasets. Our framework integrates real-time global weather forecast data with anomalies, subsurface ocean heat content, and atmospheric pressure across various temporal and spatial resolutions. Leveraging a hybrid deep learning architecture that combines a Convolutional Neural Network (CNN) for spatial feature extraction and a Long Short-Term Memory (LSTM) network for temporal dependency modeling, the framework aims to identify complex precursors and evolving patterns of El Niño events.</span> <span class="abstract-toggle" data-id="2604.04998">more</span>

    [:material-file-document: 2604.04998](https://arxiv.org/abs/2604.04998v1) · [:material-content-copy: BibTeX](bibtex/2604.04998.bib){ .bibtex-link }

    <span class="md-tag">CNN</span> <span class="md-tag">recurrent</span>

-   #### Super-Resolving Coarse-Resolution Weather Forecasts With Flow Matching

    ---

    *Aymeric Delefosse, Anastase Charantonis, Dominique Béréziat* · 2026

    <span class="abstract-snippet" id="snip-2604.00897">Machine learning-based weather forecasting models now surpass state-of-the-art numerical weather prediction systems, but training and operating these models at high spatial resolution remains...</span><span class="abstract-full" id="full-2604.00897" hidden>Machine learning-based weather forecasting models now surpass state-of-the-art numerical weather prediction systems, but training and operating these models at high spatial resolution remains computationally expensive. We present a modular framework that decouples forecasting from spatial resolution by applying learned generative super-resolution as a post-processing step to coarse-resolution forecast trajectories. We formulate super-resolution as a stochastic inverse problem, using a residual formulation to preserve large-scale structure while reconstructing unresolved variability. The model is trained with flow matching exclusively on reanalysis data and is applied to global medium-range forecasts. We evaluate (i) design consistency by re-coarsening super-resolved forecasts and comparing them to the original coarse trajectories, and (ii) high-resolution forecast quality using standard ensemble verification metrics and spectral diagnostics. Results show that super-resolution preserves large-scale structure and variance after re-coarsening, introduces physically consistent small-scale variability, and achieves competitive probabilistic forecast skill at 0.25° resolution relative to an operational ensemble baseline, while requiring only a modest additional training cost compared with end-to-end high-resolution forecasting.</span> <span class="abstract-toggle" data-id="2604.00897">more</span>

    [:material-file-document: 2604.00897](https://arxiv.org/abs/2604.00897v1) · [:material-content-copy: BibTeX](bibtex/2604.00897.bib){ .bibtex-link }

    <span class="md-tag">diffusion</span> <span class="md-tag">probabilistic</span>

-   #### Deep-Learned Observation Operators for Artificial Intelligence Weather Forecasting Models

    ---

    *Kelsey Lieberman, Laura Slivinski, Matt Bender, Chris Miller, Josh DaRosa, Nick Krall et al.* · 2026

    <span class="abstract-snippet" id="snip-2604.00082">Satellite observation operators play an essential role in atmospheric data assimilation by translating model state variables into observation space. Previous work has shown that deep-learned...</span><span class="abstract-full" id="full-2604.00082" hidden>Satellite observation operators play an essential role in atmospheric data assimilation by translating model state variables into observation space. Previous work has shown that deep-learned emulators can effectively predict the outputs of classic observation operators, like the Community Radiative Transfer Model (CRTM), with reduced inference time. This study expands previous work to show the potential for integrating observation operators into artificial intelligence (AI) weather forecasting models. Specifically, this study shows that (1) deep-learned models can effectively predict the innovations (or differences between the simulated and observed radiances) used by data assimilation models and (2) deep-learned observation models suffer only minor degradations in performance when the model state is represented with fewer vertical levels, as is commonly used by AI forecasting models. Experiments were performed using the Unified Forecast System (UFS) replay dataset, including Gridpoint Statistical Interpolation (GSI) observational data for the Advanced Technology Microwave Sounder (ATMS) sensor from 2022 and 2023. Code is available at https://github.com/mitre/deep-obs.</span> <span class="abstract-toggle" data-id="2604.00082">more</span>

    [:material-file-document: 2604.00082](https://arxiv.org/abs/2604.00082v1) · [:fontawesome-brands-github:](https://github.com/mitre/deep-obs) · [:material-content-copy: BibTeX](bibtex/2604.00082.bib){ .bibtex-link }

-   #### 30-meter Land Surface Temperature from Landsat via Progressive Self-Training Downscaling

    ---

    *Huanfeng Shen, Chan Li, Menghui Jiang, Penghai Wu, Guanhao Zhang, Tian Xie* · 2026

    <span class="abstract-snippet" id="snip-2603.29478">Land surface temperature (LST) is a critical parameter for characterizing surface energy balance and hydrothermal processes. While Landsat provides invaluable LST observations at medium spatial...</span><span class="abstract-full" id="full-2603.29478" hidden>Land surface temperature (LST) is a critical parameter for characterizing surface energy balance and hydrothermal processes. While Landsat provides invaluable LST observations at medium spatial resolution for over 40 years, its native spatial resolution of thermal bands (e.g., 100 m) remains insufficient compared to its 30 m optical bands, failing to meet the demands of fine-scale studies. To address this issues, this study proposes a progressive self-training framework for downscaling Landsat LST to 30 m without relying on fine-scale ground truth, while maintaining minimal data dependence. The framework progressively optimizes a cross-modal fusion network to refine thermal details in a coarse-to-fine manner, characterized by one pre-training and two fine-tuning stages. Spatial validation against SDGSAT-1 30 m LST and temporal validation using in situ measurements confirm its reliability and accuracy, with both station-averaged MAE and RMSE outperforming the official cubic product by approximately 0.4 K. Further performance comparison experiments demonstrate that the proposed framework consistently reconstructs coherent fine-scale thermal patterns while preserving spatial heterogeneity. Multi spatial resolution evaluations and ablation studies verify the effectiveness of the proposed strategy and network design. Overall, the framework provides a stable pathway for enhancing the spatial resolution of Landsat LST, providing fine-resolution data support for fine-scale surface process studies and localized environmental monitoring.</span> <span class="abstract-toggle" data-id="2603.29478">more</span>

    [:material-file-document: 2603.29478](https://arxiv.org/abs/2603.29478v1) · [:material-content-copy: BibTeX](bibtex/2603.29478.bib){ .bibtex-link }

-   #### Improving Ensemble Forecasts of Abnormally Deflecting Tropical Cyclones with Fused Atmosphere-Ocean-Terrain Data

    ---

    *Qixiang Li, Yuan Zhou, Shuwei Huo, Chong Wang, Xiaofeng Li* · 2026

    <span class="abstract-snippet" id="snip-2603.29200">Deep learning-based tropical cyclone (TC) forecasting methods have demonstrated significant potential and application advantages, as they feature much lower computational cost and faster operation...</span><span class="abstract-full" id="full-2603.29200" hidden>Deep learning-based tropical cyclone (TC) forecasting methods have demonstrated significant potential and application advantages, as they feature much lower computational cost and faster operation speed than numerical weather prediction models. However, existing deep learning methods still have key limitations: they can only process a single type of sequential trajectory data or homogeneous meteorological variables, and fail to achieve accurate forecasting of abnormal deflected TCs. To address these challenges, we present two groundbreaking contributions. First, we have constructed a multimodal and multi-source dataset named AOT-TCs for TC forecasting in the Northwest Pacific basin. As the first dataset of its kind, it innovatively integrates heterogeneous variables from the atmosphere, ocean, and land, thus obtaining a comprehensive and information-rich meteorological dataset. Second, based on the AOT-TCs dataset, we propose a forecasting model that can handle both normal and abnormally deflected TCs. This is the first TC forecasting model to adopt an explicit atmosphere-ocean-terrain coupling architecture, enabling it to effectively capture complex interactions across physical domains. Extensive experiments on all TC cases in the Northwest Pacific from 2017 to 2024 show that our model achieves state-of-the-art performance in TC forecasting: it not only significantly improves the forecasting accuracy of normal TCs but also breaks through the technical bottleneck in forecasting abnormally deflected TCs.</span> <span class="abstract-toggle" data-id="2603.29200">more</span>

    [:material-file-document: 2603.29200](https://arxiv.org/abs/2603.29200v2) · [:material-content-copy: BibTeX](bibtex/2603.29200.bib){ .bibtex-link }

-   #### Skillful Kilometer-Scale Regional Weather Forecasting via Global and Regional Coupling

    ---

    *Weiqi Chen, Wenwei Wang, Qilong Yuan, Lefei Shen, Bingqing Peng, Jiawei Chen, Bo Wu, Liang Sun* · 2026

    <span class="abstract-snippet" id="snip-2603.28173">Data-driven weather models have advanced global medium-range forecasting, yet high-resolution regional prediction remains challenging due to unresolved multiscale interactions between large-scale...</span><span class="abstract-full" id="full-2603.28173" hidden>Data-driven weather models have advanced global medium-range forecasting, yet high-resolution regional prediction remains challenging due to unresolved multiscale interactions between large-scale dynamics and small-scale processes such as terrain-induced circulations and coastal effects. This paper presents a global-regional coupling framework for kilometer-scale regional weather forecasting that synergistically couples a pretrained Transformer-based global model with a high-resolution regional network via a novel bidirectional coupling module, ScaleMixer. ScaleMixer dynamically identifies meteorologically critical regions through adaptive key-position sampling and enables cross-scale feature interaction through dedicated attention mechanisms. The framework produces forecasts at $0.05^\circ$ ($\sim 5 \mathrm{km}$ ) and 1-hour resolution over China, significantly outperforming operational NWP and AI baselines on both gridded reanalysis data and real-time weather station observations. It exhibits exceptional skill in capturing fine-grained phenomena such as orographic wind patterns and Foehn warming, demonstrating effective global-scale coherence with high-resolution fidelity. The code is available at https://anonymous.4open.science/r/ScaleMixer-6B66.</span> <span class="abstract-toggle" data-id="2603.28173">more</span>

    [:material-file-document: 2603.28173](https://arxiv.org/abs/2603.28173v1) · [:material-content-copy: BibTeX](bibtex/2603.28173.bib){ .bibtex-link }

    <span class="md-tag">transformer</span>

-   #### Self-Organizing Score-based Data Assimilation

    ---

    *Yuma Yamaoka, Seiichi Uchida, Shoji Toyota* · 2026

    <span class="abstract-snippet" id="snip-2603.28048">A state-space model is a statistical framework for inferring latent states from observed time-series data. However, inference with nonlinear and high-dimensional state-space models remains...</span><span class="abstract-full" id="full-2603.28048" hidden>A state-space model is a statistical framework for inferring latent states from observed time-series data. However, inference with nonlinear and high-dimensional state-space models remains challenging. To this end, an approach based on diffusion models-a powerful class of deep generative models-has been developed, known as Score-based Data Assimilation (SDA). However, SDA cannot be directly applied when the latent-state transition depends on unknown parameters that must be inferred jointly with the latent states. To overcome this limitation, we propose a framework that enables SDA to handle latent states with unknown parameters. A key feature of the proposed method is the incorporation of the self-organization technique, which has been used in classical state-space modeling for the joint estimation of latent states and parameters. By integrating this classical technique into modern SDA, our method enables joint inference of latent states and unknown parameters while maintaining the high training efficiency of SDA. The effectiveness of the proposed approach is validated through numerical experiments on dynamical systems arising in neuroscience and atmospheric science. In addition, its scalability is demonstrated using a high-dimensional Kolmogorov flow, with the data dimension on the order of several hundred thousand.</span> <span class="abstract-toggle" data-id="2603.28048">more</span>

    [:material-file-document: 2603.28048](https://arxiv.org/abs/2603.28048v2) · [:material-content-copy: BibTeX](bibtex/2603.28048.bib){ .bibtex-link }

    <span class="md-tag">diffusion</span>

</div>

