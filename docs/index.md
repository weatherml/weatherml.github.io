---
hide:
  - navigation
  - toc
title: weatherml
---

A collection of papers on AI for weather forecasting, climate modelling and atmospheric science.

<p class="page-meta" markdown="span">1485 papers · updated 2026-09-29 · <a href="feed.xml">:material-rss: RSS</a> · <a href="all_papers.bib" download>:material-download: BibTeX</a> · <a href="https://github.com/weatherml/weatherml.github.io/issues/new?template=suggest-paper.yml">:material-plus: Suggest a paper</a></p>

## Browse by Topic

<div class="grid cards topics" markdown>

-   [Global Models](papers/global-models/index.md) <span class="topic-count">348</span>
-   [Regional Models](papers/regional-models/index.md) <span class="topic-count">60</span>
-   [Nowcasting](papers/nowcasting/index.md) <span class="topic-count">99</span>
-   [Downscaling](papers/downscaling/index.md) <span class="topic-count">98</span>
-   [Post-processing](papers/post-processing/index.md) <span class="topic-count">25</span>
-   [Data Assimilation](papers/data-assimilation/index.md) <span class="topic-count">74</span>
-   [Climate Modeling](papers/climate-modeling/index.md) <span class="topic-count">292</span>
-   [Hydrology](papers/hydrology/index.md) <span class="topic-count">37</span>
-   [Ocean & Sea Ice](papers/ocean-sea-ice/index.md) <span class="topic-count">85</span>
-   [Air Quality & Composition](papers/air-quality-composition/index.md) <span class="topic-count">71</span>
-   [Remote Sensing](papers/remote-sensing/index.md) <span class="topic-count">99</span>
-   [Other](papers/other/index.md) <span class="topic-count">197</span>

</div>

## Recent Additions

<div class="grid cards" markdown data-search-exclude>

-   #### Explainable Deep Learning for Probabilistic Nowcasting of Radar Reflectivity in Tornadic Storms { #2609.35675 }

    *Nathan Erickson, Amy McGovern, Aaron Hill* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.35675">Tornadoes pose substantial risk to human life and property in the United States, causing more than 50 fatalities and \$100 million of property damage on average annually. When tornadoes are likely,...</span><span class="abstract-full" id="full-2609.35675" hidden>Tornadoes pose substantial risk to human life and property in the United States, causing more than 50 fatalities and \$100 million of property damage on average annually. When tornadoes are likely, weather radar provides critical information for forecasters by providing information on storm morphology, storm motion, and intensity trends. Additional tools such as satellite and numerical weather prediction model runs can provide useful short-term information for understanding changes in storm characteristics. This work demonstrates a U-Net deep-learning system for nowcasting the evolution of radar reflectivity following tornadogenesis, which can provide value to forecasters by synthesizing large amounts of input data (e.g., radar imagery, near-storm environment data) and generating predictions of radar reflectivity from its inputs. Inputs to the model are radar imagery from the Multi-Radar Multi-Sensor (MRMS) dataset and near-storm environment data from the High-Resolution Rapid Refresh (HRRR) numerical weather prediction model. The U-Net is trained on a dataset of tornadic storms to produce 30 minutes of probabilistic predictions of radar reflectivity following tornadogenesis, with probabilistic predictions obtained by predicting parameters of the SinhArcSinh, or SHASH, distribution. The model produces physically realistic predictions of radar evolution, achieves comparable skill to next-hour forecasts from the HRRR, demonstrates reasonable probabilistic calibration and is accompanied by a variety of explainability methods to improve understanding by end users. Additionally, predictions from the model can be obtained much more quickly than those from a numerical weather prediction model. With further development, this model could be extended to nowcast radar reflectivity evolution in an operational setting.</span> <span class="abstract-toggle" data-id="2609.35675">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.35675v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.35675v1) · [:material-content-copy: BibTeX](bibtex/2609.35675.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=interpretability" data-tag="interpretability">Interpretability</a>
    { .paper-tags }

-   #### Safe Greenhouse Climate Control Using Lagrangian-Constrained PPO with Kolmogorov-Arnold Networks { #2609.34966 }

    *Hangzun Liu, Yuling Fan, Fang Tian, Zhilong Bie, Zaiwen Feng, Yongliang Qiao* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.34966">Greenhouse climate control balances economic return with maintaining temperature, humidity and CO2 within crop-adapted growth ranges. Conventional reinforcement learning (RL) greenhouse controllers...</span><span class="abstract-full" id="full-2609.34966" hidden>Greenhouse climate control balances economic return with maintaining temperature, humidity and CO2 within crop-adapted growth ranges. Conventional reinforcement learning (RL) greenhouse controllers use fixed reward penalties to limit climate constraint violations, yet such heuristic penalties cannot explicitly constrain long-term cumulative violations. Poorly tuned weights either lead to overly conservative policies and lower yields, or fail to suppress persistent climate deviations that harm photosynthesis and induce crop diseases. To address this issue, we formulate greenhouse climate regulation as a Constrained Markov Decision Process (CMDP) and use a Lagrangian safe RL framework RCPO-PPO to separate economic optimization and cumulative safety constraints, enabling adaptive penalty adjustment without manual tuning. To handle strong nonlinear, time-varying coupling between greenhouse microclimate and crop growth, Kolmogorov-Arnold Networks (KANs) replace Multi-Layer Perceptrons (MLPs) as policy and value approximators for improved nonlinear representation. Sinusoidal cyclic time features are embedded in observations to capture diurnal environmental periodicity. Simulations use a classic winter lettuce greenhouse model driven by 40-day real weather disturbances. Compared with vanilla penalty-based PPO, our method cuts cumulative climate violations by 18.65% and raises lettuce economic profit by 2.91%, keeping violations stable near the safety threshold. This decoupled CMDP optimization with KAN-based policy representation mitigates long-term climate risks and boosts planting profits, offering a constraint-aware control strategy for precision greenhouse cultivation.</span> <span class="abstract-toggle" data-id="2609.34966">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.34966v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.34966v1) · [:material-content-copy: BibTeX](bibtex/2609.34966.bib){ .bibtex-link }
    { .paper-links }

-   #### MW-Nowcast: Six-hour ensemble nowcasting of extreme precipitation { #2609.34836 }

    *Ning Wang, Zuliang Fang, Weixin Jin, Zhongjian Lv, Shuang Qin, Pengcheng Zhao, Siqi Xiang et al.* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.34836">Extending reliable nowcasting of extreme precipitation could provide critical additional time for warnings and emergency response during high-impact events such as flash floods. Radar-based...</span><span class="abstract-full" id="full-2609.34836" hidden>Extending reliable nowcasting of extreme precipitation could provide critical additional time for warnings and emergency response during high-impact events such as flash floods. Radar-based generative machine-learning models have enabled skilful hyperlocal precipitation nowcasting, but accurate prediction of intense precipitation remains confined to the first few hours. Because storm-scale structure is predictable for longer than individual cells, a natural strategy is to predict that structure while generatively modelling only the uncertain local growth, decay, reorganisation and initiation of storms. Here we present Microsoft Weather Nowcast (MW-Nowcast), a six-hour ensemble radar nowcasting model that jointly learns a deterministic predictor to capture organised precipitation structure shared across ensemble members, and a generator to produce diverse local residuals around this shared prediction. Across independent test data from the United States, Europe and China, MW-Nowcast achieves higher detection skill than leading methods for heavy and extreme precipitation throughout the 6 h horizon. For the most intense rainfall, MW-Nowcast doubles the available warning time across all three regions, delivering 6 h forecasts with skill previously limited to 3 h for the leading generative baseline. A cost-loss decision analysis shows that MW-Nowcast retains substantial value for a broad range of applications even at 4-6 h, where alternative methods offer little benefit. These additional hours can give forecasters and emergency managers the time to warn and act before extreme rainfall strikes, helping to protect lives and property.</span> <span class="abstract-toggle" data-id="2609.34836">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.34836v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.34836v1) · [:material-content-copy: BibTeX](bibtex/2609.34836.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=extremes" data-tag="extremes">Extremes</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### Predicting Delayed Train Trajectories on the Dutch Railway Network: Explainable AI Evaluation of Topological, Operational and Weather Features with Tree Based Ensemble Methods { #2609.34692 }

    *Jia Long Bao, Ali Mohammed Mansoor Alsahag, Seyed Sahand Mohammadi Ziabari* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.34692">The reliable prediction of passenger train delays is a critical component of railway management. While contemporary research frequently attempts to maximize absolute accuracy by deploying opaque deep...</span><span class="abstract-full" id="full-2609.34692" hidden>The reliable prediction of passenger train delays is a critical component of railway management. While contemporary research frequently attempts to maximize absolute accuracy by deploying opaque deep learning architectures, the underlying data mechanics driving longitudinal predictive decay remain underexplored. Consequently, this study provides an explainable temporal robustness analysis of network-wide railway delay prediction. Focusing on the Dutch railway network, this research utilizes interpretable tree-based ensembles to integrate granular topological, environmental, and operational features. The overarching finding establishes that while feature-rich tree-based models improve simultaneous (within-month) prediction, predictive performance systematically degrades when evaluated across non-simultaneous (future) months. Furthermore, multi-horizon SHAP and dispersion analyses explicitly link this degradation to environmental feature volatility and instability within the statistical target definition. Ultimately, this thesis demonstrates that richer feature sets alone are insufficient to resolve long-term forecasting constraints, underscoring the necessity to transition toward dynamic, season-aware architectures anchored by absolute operational boundaries.</span> <span class="abstract-toggle" data-id="2609.34692">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.34692v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.34692v1) · [:material-content-copy: BibTeX](bibtex/2609.34692.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=interpretability" data-tag="interpretability">Interpretability</a>
    { .paper-tags }

-   #### Low latency global carbon budget reveals strong land sink recovery in 2025 { #2609.34226 }

    *Philippe Ciais, Piyu Ke, Xiangjun Tian, Stephen Sitch, Wei Li, Xiaomeng Du, Xiaofan Gui et al.* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.34226">The atmospheric CO2 growth rate fell sharply in 2025, from a record 3.76 $\pm$ 0.09 ppm yr-1 in 2024 to 2.06 $\pm$ 0.09 ppm yr-1 (NOAA marine boundary layer observations), below the 2015-2022 mean of...</span><span class="abstract-full" id="full-2609.34226" hidden>The atmospheric CO2 growth rate fell sharply in 2025, from a record 3.76 $\pm$ 0.09 ppm yr-1 in 2024 to 2.06 $\pm$ 0.09 ppm yr-1 (NOAA marine boundary layer observations), below the 2015-2022 mean of 2.47 ppm yr-1, even as fossil CO2 emissions rose by 0.7% to 10.38 GtC yr-1. Here we present a low-latency global and regional carbon budget for 2025, combining three dynamic global vegetation models (DGVMs) and ocean model emulators with four atmospheric inversions constrained by OCO-2 satellite retrievals. The global net land sink reached 2.36 $\pm$ 0.16 GtC yr-1 in 2025 (DGVMs: 2.04 $\pm$ 0.24; inversions: 2.68 $\pm$ 0.20 GtC yr-1), strengthening by 2.81 $\pm$ 0.31 GtC yr-1 from 2024 and exceeding the 2015-2022 mean by 0.71 $\pm$ 0.13 GtC yr-1. Ocean uptake (3.11 $\pm$ 0.36 GtC yr-1) remained similar to 2024, making the land sink rebound the dominant driver of the slowdown in CO2 growth. Tropical lands shifted from net sources in 2024 to net sinks in 2025, with enhanced uptake across much of Africa and northern Eurasia, and land flux anomalies covaried with GRACE terrestrial water storage. Where the sink had weakened substantially in 2023-2024, about 80% of the area showed some recovery, with overall recovery of 87.3% (DGVMs) to 99.5% (inversions). Recovery exceeded 100% in the tropics but remained incomplete in the northern extratropics, indicating a strong but spatially uneven rebound of the land carbon sink.</span> <span class="abstract-toggle" data-id="2609.34226">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.34226v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.34226v1) · [:material-content-copy: BibTeX](bibtex/2609.34226.bib){ .bibtex-link }
    { .paper-links }

-   #### StatD2GAN: When Calibration Masks Generator Quality in Held-Out Evaluation of Synthetic Weather Sequences { #2609.33761 }

    *Mustafa Ozaytac, Ozge Karadag Atas* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.33761">Generative models for multivariate weather series are routinely evaluated with pooled distributional metrics computed after marginal calibration. We show this practice can invalidate architectural...</span><span class="abstract-full" id="full-2609.33761" hidden>Generative models for multivariate weather series are routinely evaluated with pooled distributional metrics computed after marginal calibration. We show this practice can invalidate architectural conclusions, and rebuild the evaluation of StatD2GAN, a three-discriminator GAN with evolutionary weight adaptation, around a held-out protocol: the final two calendar years of each dataset are held out behind a 168 hour embargo, calibration is fitted on the training block only, and all metrics are computed on the held-out block. Evidence comes from 25 matched (location, seed) pairs across five Koppen-Geiger climates, tested with Wilcoxon signed-rank tests under Holm correction. Four results follow. First, isotonic calibration drives the Kolmogorov-Smirnov distance to within 2% of a per-location noise-and-shift floor for every architecture tested, including a deliberately weak RCGAN baseline, so calibrated marginal metrics cannot discriminate between architectures. Second, the sorted-representation discriminator is the only component whose removal significantly degrades cross-variable dependence (Kendall tau MAE +0.080, Holm p = 0.009), with a regime-dependent effect: near zero in Ankara, above 115% in Dubai and Yakutsk. A rank-transformed variant isolates the mechanism as quantile supervision of the marginals rather than copula matching. Third, physical constraint violations are injected by calibration, not the generator; projection removes them at negligible cost (deltaKS <= 0.003). Fourth, pooled metrics conceal a collapse of between-sequence weekly-mean variability, a proxy for seasonal and regime diversity, in TimeGAN that only sequence-level statistics expose. We recommend floor-referenced marginal evaluation, matched-pair testing, and sequence-level variance decomposition as minimum requirements for calibrated generative pipelines.</span> <span class="abstract-toggle" data-id="2609.33761">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.33761v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.33761v1) · [:material-content-copy: BibTeX](bibtex/2609.33761.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=gans" data-tag="gans">GANs</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### Suitable Measures for the Potential Operational Utility of AI NWP Rainfall Forecasts Over Africa { #2609.31775 }

    *Shruti Nath, Docko Sow, Koomi Toussaint Amoussouvi, Fenwick Cooper, Josiah Kiarie Kimani et al.* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.31775">Artificial intelligence (AI)-based weather prediction is approaching the skill of physical numerical weather prediction (NWP) systems at a fraction of the computational cost. This is particularly...</span><span class="abstract-full" id="full-2609.31775" hidden>Artificial intelligence (AI)-based weather prediction is approaching the skill of physical numerical weather prediction (NWP) systems at a fraction of the computational cost. This is particularly promising for Africa, where rainfall extremes are intensifying and many forecasting centres lack the infrastructure to run physical models at extended lead times. We present a calibrated comparison of GraphCast, GenCast and the Functional Generative Network (FGN) against the physical NWP model IFS for rainfall prediction across Africa. Deterministic and probabilistic forecasts are postprocessed using Isotonic Distributional Regression and evaluated with the Continuous Ranked Probability Score against IMERG, RFEv2 and CHIRPS across seasons, wet and dry regimes, elevation zones and lead times. All models retain skill beyond climatology across most seasons and at extended lead times. AI models generally outperform IFS in wet regions, whereas IFS performs better in dry, high-elevation areas, where its finer resolution better represents orographic controls on rainfall. Across observational datasets and seasons, AI models achieve a median improvement of approximately 5% over IFS. GraphCast achieves calibrated skill comparable to the ensemble-based FGN, although FGN provides greater significant skill at longer lead times. These results highlight the potential of calibrated AI weather prediction to provide accessible and computationally efficient rainfall forecasts across Africa, while demonstrating the continuing importance of spatial resolution, ensemble design and regional characteristics.</span> <span class="abstract-toggle" data-id="2609.31775">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.31775v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.31775v1) · [:material-content-copy: BibTeX](bibtex/2609.31775.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=efficiency" data-tag="efficiency">Efficiency</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a>
    { .paper-tags }

-   #### Learning Hierarchical Causal Representations of the Effects of Forcings on Temperature in Climate Models { #2609.30995 }

    *Shan Zhao, Ilija Trajkovic, Julia Kaltenborn, Yaniv Gurwicz, Peer Nowack, David Rolnick et al.* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.30995">Machine learning (ML) emulators provide a fast and cost-effective method to simulate climate change scenarios after being trained on Earth System Models projections. However, the black-box nature of...</span><span class="abstract-full" id="full-2609.30995" hidden>Machine learning (ML) emulators provide a fast and cost-effective method to simulate climate change scenarios after being trained on Earth System Models projections. However, the black-box nature of those data-driven approaches limit the usability and trustworthiness of their outputs and in particular their use as causal attribution tools. Here, we develop a hierarchical causal representation learning framework applied to sea surface temperature fields from a state-of-the-art global climate model. As a key advance over previous work, our framework explicitly models both atmospheric dynamical interactions arising from internal climate variability and forced responses due to changes in atmospheric greenhouse gas and aerosol concentrations. When trained on future climate change scenarios, our method accurately predicts the long-term global mean and regional temperature evolution and shows physically realistic responses to perturbations in greenhouse gas and aerosol concentrations when evaluated on unseen scenarios. Our results underline the potential of causal representation learning frameworks for advancing climate model emulation.</span> <span class="abstract-toggle" data-id="2609.30995">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.30995v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.30995v1) · [:material-content-copy: BibTeX](bibtex/2609.30995.bib){ .bibtex-link }
    { .paper-links }

-   #### Mechanism-Aware Ensemble Conditioning for Data-Limited Emulation of Extreme Events { #2609.30746 }

    *Isabella S. Thiel, Juan Bello-Rivas, Yannis G. Kevrekidis, Themistoklis P. Sapsis* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.30746">Extreme events in chaotic systems are difficult to learn from short trajectories because they are controlled by transient finite-time instability rather than by frequently observed bulk dynamics. We...</span><span class="abstract-full" id="full-2609.30746" hidden>Extreme events in chaotic systems are difficult to learn from short trajectories because they are controlled by transient finite-time instability rather than by frequently observed bulk dynamics. We propose a mechanism-aware conditioning plug-in framework that turns a nudged coarse ensemble into a non-intrusive sensor of local instability geometry. In the small-noise regime, the ensemble covariance aggregates the same finite-time deformation kernels that govern local instability, providing a Jacobian-free proxy for the local amplification structure around a synchronized coarse trajectory. A small FiLM module injects statistics of this ensemble geometry into an otherwise unchanged backbone while leaving the coarse simulator unchanged. We demonstrate this interface in two distinct pipelines: a Transformer-style residual-attention corrector for a controlled low-dimensional chaotic system and a probabilistic recurrent STORN corrector for topographic two-layer quasi-geostrophic (QG) flow. In the low-dimensional benchmark, ensemble covariance directions co-activate with OTD modes and FiLM conditioning improves 99th-percentile exceedance-frequency errors over an identical no-context Transformer baseline. In QG, a fixed ensemble-conditioned FiLM-STORN model trained on only \(50\) time units substantially improves long-horizon rare-event statistics in the data-limited regime, including density-tail errors, exceedance frequencies, and spatial exceedance-area distributions relative to an unconditioned STORN trained on the same data; on averaged high-threshold exceedance diagnostics, it also outperforms the baseline STORN trained with $20$ times more high-resolution data. These results show that local instability geometry is not merely interpretable post hoc, but an actionable conditioning signal for data-efficient rare-event emulation.</span> <span class="abstract-toggle" data-id="2609.30746">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.30746v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.30746v1) · [:material-content-copy: BibTeX](bibtex/2609.30746.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=transformers" data-tag="transformers">Transformers</a> <a class="md-tag" href="/explore/?t=extremes" data-tag="extremes">Extremes</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### On the Limits of Univariate Deep Learning for Significant Wave Height Forecasting { #2609.30688 }

    *Yilin Zhai, Hongyuan Shi, Zaijin You* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.30688">This study conducts a systematic hyperparameter search across five deep learning architectures, DLinear, LSTM, PatchTST, ResAttLstm, and Mamba2, and nine context lengths (1-168 h) for single-station...</span><span class="abstract-full" id="full-2609.30688" hidden>This study conducts a systematic hyperparameter search across five deep learning architectures, DLinear, LSTM, PatchTST, ResAttLstm, and Mamba2, and nine context lengths (1-168 h) for single-station significant wave height (Hs) forecasting on NDBC buoy 41009, followed by re-evaluation of the best configurations on a 47-buoy, 37-year corpus. The five families converge to a common performance level on the multi-buoy evaluation (between-family SD = 0.0014 m^2, 0.8% of the grand mean), a spread dwarfed by the 4.83x cross-dataset MSE shift between buoy corpora. All multi-buoy trials beat persistence (mean skill +0.062), but no architecture consistently outperforms the others. On the single-buoy experiment, skill peaks at 12-24 h where five trials fall below persistence, per-family Q4/Q3 test MSE ratios range from 2.4 to 2.6, and deep models underperform persistence for the most extreme 1% of waves. These findings are consistent with the interpretation that persistence already captures the dominant linear-inertial signal in univariate Hs, and that architecture engineering under this univariate input setting has reached diminishing returns: cross-buoy variance, not model class, dominates forecast error. Future work should prioritise atmospheric covariates, zero-shot cross-buoy transfer, and decomposition of Hs into swell and wind-sea components. By establishing a rigorous reference baseline for what univariate Hs models can and cannot achieve, this study provides a benchmark against which future multivariate and physics-informed approaches can be calibrated, and offers practical guidance for lightweight buoy-level forecasting in mid-latitude storm-dominated and swell-mixed environments.</span> <span class="abstract-toggle" data-id="2609.30688">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.30688v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.30688v1) · [:material-content-copy: BibTeX](bibtex/2609.30688.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=recurrent-networks" data-tag="recurrent-networks">Recurrent networks</a>
    { .paper-tags }

-   #### Understanding Perturbed Parameter Ensemble Sensitivities Using A Contrastive Learning Approach { #2609.30420 }

    *Da Fan, David John Gagne, Gregory S Elsaesser, Brian Medeiros, Addisu G Semie, Qingyuan Yang et al.* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.30420">Perturbed parameter ensembles (PPEs) reveal how physics parameters affect climate simulations, but interpreting parameter sensitivities across multivariate, spatially structured outputs remains...</span><span class="abstract-full" id="full-2609.30420" hidden>Perturbed parameter ensembles (PPEs) reveal how physics parameters affect climate simulations, but interpreting parameter sensitivities across multivariate, spatially structured outputs remains challenging, particularly when calibrating models against observations. We develop an explainable contrastive learning model that maps 5 monthly cloud and radiation fields into a shared representation space. We train the model on the fields of two 100-member Community Atmosphere Model version 6 (CAM6) PPEs, spanning 34 parameters, that only differ in the warm rain microphysics scheme: KK2000, the default bulk microphysics scheme, and TAU-ML, a neural network emulator of a bin microphysics scheme. The learned representations separates two PPEs with over 94% linear classification accuracy while preserving the seasonal variability and ensemble spread due to parameter perturbations. In the shared representation space, the representations of satellite observations occupy the same low-dimensional manifold as the PPEs but are displaced from them most strongly during boreal spring and autumn. TAU-ML PPE has a lower distance to observations compared to KK2000 in the representation space. Integrated Gradients attributions highlights the contributions in subtropical low-cloud regions, Northern and Southern Hemisphere storm track regions, and tropical convection regions to differences between PPEs and observations. Regional attributions correlate most strongly with parameters associated with cloud microphysics, boundary layer turbulence, and deep convection. These results demonstrate that explainable representations of climate fields can attribute model differences to specific variables, regions, seasons, and physical parameters.</span> <span class="abstract-toggle" data-id="2609.30420">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.30420v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.30420v1) · [:material-content-copy: BibTeX](bibtex/2609.30420.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=interpretability" data-tag="interpretability">Interpretability</a> <a class="md-tag" href="/explore/?t=monthly" data-tag="monthly">Monthly</a>
    { .paper-tags }

-   #### Lightweight Probabilistic Downscaling from a Deterministic Base Model { #2609.29383 }

    *Joseph McLean, Tiffany Vlaar, Sigrid Passano Hellan, Linus Ericsson* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.29383">Climate data downscaling is the task of increasing the spatial resolution of climate data, typically by generating fine-resolution regional climate data from coarse global model output. Recent...</span><span class="abstract-full" id="full-2609.29383" hidden>Climate data downscaling is the task of increasing the spatial resolution of climate data, typically by generating fine-resolution regional climate data from coarse global model output. Recent machine learning (ML) work in the related task of weather forecasting has seen significant improvements due to newly devised training methods and architectural components, but these have not yet benefited downscaling. We adapt two of these methods to create a family of lightweight probabilistic ML downscaling models built on a modified U-Net backbone and evaluate them on the CORDEX-ML-Bench suite for daily maximum temperature and precipitation across three geographic regions: the Alps, New Zealand and South Africa. We find that a two-stage training curriculum, combining deterministic pretraining with probabilistic tuning, transfers well to downscaling, beating the state-of-the-art for RMSE. Our work provides an advancement towards lightweight, probabilistic downscaling models, reducing the current trade-off between computational intensity and distributional fit.</span> <span class="abstract-toggle" data-id="2609.29383">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.29383v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.29383v1) · [:material-content-copy: BibTeX](bibtex/2609.29383.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=efficiency" data-tag="efficiency">Efficiency</a> <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a> <a class="md-tag" href="/explore/?t=daily" data-tag="daily">Daily</a>
    { .paper-tags }

-   #### Generative Atmospheric Super-Resolution from Heterogeneous In Situ Observations through Composable Interfaces { #2609.29027 }

    *Yang Xu, Dibyajyoti Chakraborty, Haiwen Guan, Sen Wang, Romit Maulik* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.29027">Atmospheric observations are sparse, heterogeneous, and unevenly distributed, whereas many generative atmospheric models learn distributions over regularly gridded multivariate states. Once...</span><span class="abstract-full" id="full-2609.29027" hidden>Atmospheric observations are sparse, heterogeneous, and unevenly distributed, whereas many generative atmospheric models learn distributions over regularly gridded multivariate states. Once pretrained, diffusion models can supply atmospheric priors that can be combined with observation-derived likelihood factors in a Bayesian formulation. However, these observation sources differ substantially in geometry and sampling density, complicating the consistent use of their observations within a common inference framework. Here, we formulate this reconstruction problem as generative atmospheric super-resolution and introduce composable observation interfaces for conditioning a single pretrained 13-variable atmospheric diffusion model. The interfaces convert sparse radiosonde (R), clustered aircraft (A), and dense irregular surface-station (S) observations into source-specific likelihood factors that specify where observations constrain the gridded state, how residuals are counted under uneven sampling, and how strongly each source guides posterior sampling. We developed the aircraft and surface observation interfaces using 2019 observations and evaluated the selected interfaces throughout 2020 without further tuning. Compared with reconstructions conditioned only on radiosonde observations, the composed R+A+S interface reduces RMSE evaluated against ERA5 by $9.24\%$ across all 13 state variables over the CONUS domain. The aircraft and surface factors provide complementary improvements in upper-air and surface variables. The R+A+S combination also lowers the Continuous Ranked Probability Score (CRPS), while evaluations at held-out aircraft and surface-station observations show reduced prediction errors. Together, these results demonstrate a modular route for conditioning a pretrained atmospheric generative prior on heterogeneous in situ observations without retraining the underlying model.</span> <span class="abstract-toggle" data-id="2609.29027">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.29027v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.29027v1) · [:material-content-copy: BibTeX](bibtex/2609.29027.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=station-point" data-tag="station-point">Station / point</a>
    { .paper-tags }

-   #### Evaluating Cross-region Generalization for Wavelet-Diffusion Precipitation Downscaling { #2609.28749 }

    *Weikang Qian, Yixin Wen, Chugang Yi, Zhi Li, Lingcheng Li, Haizhao Yang* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.28749">Diffusion models have shown strong potential for kilometer-scale precipitation downscaling, but their performance in geographically unseen regions and event regimes remains insufficiently understood....</span><span class="abstract-full" id="full-2609.28749" hidden>Diffusion models have shown strong potential for kilometer-scale precipitation downscaling, but their performance in geographically unseen regions and event regimes remains insufficiently understood. Building on the wavelet diffusion model (WDM) framework, this study evaluates cross-region and cross-event generalization. Six 3 x 3 deg U.S. regions represent convective, winter, tropical, and atmospheric-river precipitation regimes. Low-resolution inputs are generated by block averaging NOAA Multi-Radar/Multi-Sensor (MRMS) composite reflectivity fields. A WDM trained only on Oklahoma (OK) samples and a WDM trained on all six regions are compared with nearest-neighbor and Bicubic interpolation. Model performance is evaluated using three metric families that measure image-domain reconstruction, spectral and distributional fidelity, and bin-wise precipitation detection. The OK-trained WDM remains competitive outside OK. Although the all-region WDM delivers the best and most consistent overall image-domain and detection performance, its gains are uneven across precipitation intensities. Bin-wise critical success index (CSI) over 5-dBZ reflectivity bins shows that WDM improvements concentrate in localized higher-reflectivity structures, which image-domain metrics partly obscure. In addition, the performance differences among samples are strongly associated with the spatial organization of the precipitation field, quantified by Moran's I as the spatial autocorrelation of each reflectivity bin. The sample-level Moran's I-CSI correlation stratified by sample intensity reaches 0.901 in all six regions, including regions unseen during training. Overall, these findings support future efforts to transfer downscaling models to regions with limited local training data and to generate globally consistent, high-resolution precipitation products.</span> <span class="abstract-toggle" data-id="2609.28749">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.28749v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.28749v1) · [:material-content-copy: BibTeX](bibtex/2609.28749.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a> <a class="md-tag" href="/explore/?t=km-scale" data-tag="km-scale">Km-scale</a>
    { .paper-tags }

-   #### HClimRep-Ocean: A Global Ocean Emulator on an Unstructured Mesh { #2609.28601 }

    *Kacper Nowak, Aleksei Koldunov, Nikolay Koldunov, Savvas Melidonis, Ankit Patnala, Simon Grasse et al.* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.28601">Machine-learning (ML) emulators for atmospheric processes have advanced rapidly in recent years, transforming weather forecasting. Although early ML ocean forecasting models now exist, they remain...</span><span class="abstract-full" id="full-2609.28601" hidden>Machine-learning (ML) emulators for atmospheric processes have advanced rapidly in recent years, transforming weather forecasting. Although early ML ocean forecasting models now exist, they remain less developed than their atmospheric counterparts. Unlike the atmosphere, much of the ocean's kinetic energy resides in mesoscale eddies whose characteristic spatial scales are approximately an order of magnitude smaller than those of comparable atmospheric features. Moreover, complex coastlines, narrow straits, and ice-covered seas make boundary representation a central challenge that atmospheric models do not face. Consequently, numerical ocean simulations commonly use locally refined or even completely unstructured meshes. However, their data-driven counterparts have so far been built around latitude-longitude grids. We present HClimRep-Ocean, an ocean emulator that operates directly on the native unstructured mesh of FESOM2. The emulator is trained on a 209-year AWI-CM3 control integration and is run without atmospheric forcing, receiving the atmospheric state only at initialisation time, which isolates the predictability carried by the ocean state itself. Skill is strongly field-dependent: for currents, HClimRep-Ocean outperforms every reference at 30 day forecast, whereas for temperature and salinity a damped-anomaly persistence forecast remains the more accurate estimator. This behaviour is physically interpretable: current variability is largely geostrophic and internally generated, whereas sea-surface temperature and salinity fluctuations are driven by atmospheric forcing through weather state. Evaluated independently on the OceanBench benchmark, a reanalysis-trained variant of HClimRep-Ocean achieves the lowest RMSE against GLORYS reanalysis among all assessed systems, confirming the competitiveness of the native-mesh approach.</span> <span class="abstract-toggle" data-id="2609.28601">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.28601v2) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.28601v2) · [:material-content-copy: BibTeX](bibtex/2609.28601.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a>
    { .paper-tags }

-   #### PISCES: Physics-Informed Solar-wind Convolutional autoEncoder for Space-weather Anomaly Detection and Early Warning { #2609.28022 }

    *Kevin Lee, Alison J. March* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.28022">Space weather early warning depends on detecting solar wind transients in in-situ measurements at the first Sun-Earth Lagrange point (L1), before they reach Earth. Fixed thresholds can miss combined...</span><span class="abstract-full" id="full-2609.28022" hidden>Space weather early warning depends on detecting solar wind transients in in-situ measurements at the first Sun-Earth Lagrange point (L1), before they reach Earth. Fixed thresholds can miss combined magnetic and plasma structure, and many learning methods provide a single anomaly score. We present the Physics-Informed Solar-wind Convolutional autoEncoder for Space-weather (PISCES), a convolutional autoencoder trained without catalog labels on OMNI solar wind measurements under physics constraints. Its loss includes magnetic field consistency, an empirical relation between temperature and velocity, the Parker spiral angle, and penalties on changes between consecutive one-minute samples in derived quantities calculated from the reconstruction. At inference, PISCES separates the anomaly score into magnetic and plasma reconstruction errors, physics relations, and residual corrections, and reports the magnitude of each contribution. Attenuation of the skip connections, selected on validation data, improves average precision for the trained models, while the untrained scores remain nearly the same. The trained models also give a more consistent ordering of these physical contributions. After smoothing with a trailing median, the alarms can precede independently observed sudden commencements, including positive sudden impulses.</span> <span class="abstract-toggle" data-id="2609.28022">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.28022v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.28022v1) · [:fontawesome-brands-github: Code](https://github.com/magnaprog/PISCES) · [:material-content-copy: BibTeX](bibtex/2609.28022.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=physics-ml-hybrid" data-tag="physics-ml-hybrid">Physics–ML hybrid</a>
    { .paper-tags }

-   #### Sparse-Observation Atmospheric Thermal Forecasting with Physics-Informed Neural Networks for Climate-Aware Digital Twins { #2609.27290 }

    *Tannaz Goodarzvand Chegini, Elyas Shivanian, Behzad Karimi, Faraz Dadgostari* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.27290">Short-horizon forecasts of atmospheric temperature are needed to support climate-aware digital-twin systems, but such forecasts must be produced where thermal observations are incomplete. This study...</span><span class="abstract-full" id="full-2609.27290" hidden>Short-horizon forecasts of atmospheric temperature are needed to support climate-aware digital-twin systems, but such forecasts must be produced where thermal observations are incomplete. This study evaluates a physics-informed neural network for potential-temperature forecasting, constrained by a pressure-coordinate thermodynamic advection-source equation and a diabatic-source closure fit from the preceding 12-hour period and frozen before future-time training. Using hourly ERA5 reanalysis at three pressure levels, the model is evaluated as a conditional hindcast at lead times of one, two and three hours against persistence, local-trend, and two matched neural-network baselines, one of which receives the same future meteorological forcing as the PINN, helping distinguish the physical constraint from access to future forcing. In an Oklahoma development case, mean RMSE improvement over the strongest baseline grew from 8.1% at one hour to 23.8% at three hours; under an observation-density sweep down to 5% of candidate locations, this 3-hour advantage remained 14.6--16.9%, with no evidence that lower density improves performance. Under a fixed protocol transferred to an Alabama heat event with three virtual-observation layouts, three-hour improvement ranged 19.7-24.4% with consistent origin-level wins. A parallel Montana stress test, in which fixed pressure levels intersected complex terrain, produced a three-hour degradation of roughly 17.5%, identifying a terrain-related applicability limit of the formulation. Together, these results indicate that the physics constraint's benefit grows with forecast horizon, persists under severe observation sparsity, and transfers across regions, but is bounded by the validity of a fixed vertical-coordinate representation over complex terrain, evidence relevant to physics-constrained components of climate-aware forecasting and digital-twin systems.</span> <span class="abstract-toggle" data-id="2609.27290">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.27290v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.27290v1) · [:material-content-copy: BibTeX](bibtex/2609.27290.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=physics-ml-hybrid" data-tag="physics-ml-hybrid">Physics–ML hybrid</a> <a class="md-tag" href="/explore/?t=hourly" data-tag="hourly">Hourly</a>
    { .paper-tags }

-   #### Analysis of trade-offs in urban heat mitigation using a Bayesian Optimization framework for an urban canopy layer model { #2609.25953 }

    *Rebekka Walter, Johanna Gelhaus, David Anton, Henning Wessles, Stephan Weber* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.25953">To mitigate the challenges of climate change and intensifying heat stress in urban areas, local adaptation strategies are discussed and introduced in cities worldwide. To understand processes and...</span><span class="abstract-full" id="full-2609.25953" hidden>To mitigate the challenges of climate change and intensifying heat stress in urban areas, local adaptation strategies are discussed and introduced in cities worldwide. To understand processes and potential trade-offs of these strategies a Bayesian optimization and surrogate modeling framework was employed to investigate urban parameter ranges of heat mitigation strategies with focus on three thermal metrics: daytime air temperature, Universal Thermal Climate Index (UTCI), and nighttime air temperature. Based on an urban street canyon configuration, it was shown that heat mitigation measures that reduce daytime air temperature and UTCI are often associated with higher nighttime temperatures. This results in a curved Pareto front that reflects the trade-off between daytime and nighttime thermal comfort. Under identical forcing conditions, different urban configurations, varying in geometry, vegetation, and surface characteristics, are shown to alter peak canyon air temperature by up to 5.2~$^\circ$C during the day and 2.6~$^\circ$C at night, while UTCI varies by up to 7.9~$^\circ$C, demonstrating that favorable urban configurations can substantially mitigate microclimatic heat stress. These findings suggest that combining multi-objective Bayesian optimization and surrogate modeling can help bridge the gap between computationally intensive climate simulations and practical decision-making in urban planning. An interactive visualization tool was developed to explore these trade-offs, making the often opposing relationships between urban parameters and the three thermal metrics directly accessible to planners.</span> <span class="abstract-toggle" data-id="2609.25953">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.25953v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.25953v1) · [:material-content-copy: BibTeX](bibtex/2609.25953.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a>
    { .paper-tags }

-   #### A dataset of one-dimensional idealized probabilistic fields { #2609.25720 }

    *Gregor Skok, Romain Pic* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.25720">Verification of probabilistic weather forecasts remains a crucial aspect of numerical weather prediction, as new AI-based models become more widely used alongside the more traditional physics-based...</span><span class="abstract-full" id="full-2609.25720" hidden>Verification of probabilistic weather forecasts remains a crucial aspect of numerical weather prediction, as new AI-based models become more widely used alongside the more traditional physics-based ensemble forecasting systems that continue to be developed and improved. We present a first-of-its-kind idealized probabilistic dataset composed of one-dimensional cases aimed at analyzing the behavior and properties of verification methods for probabilistic forecasts and comparing their behavior. It covers a wide range of probabilistic cases, such as constant, localized events, gradients, fronts, noisy, bimodal, and limiting cases. Moreover, the code associated with the dataset provides great flexibility for customizing the experiments it covers. The dataset represents the first building block of the more extensive comparison dataset of the Bridging The Gap project, which aims to facilitate the development and comparison of spatial verification methods for probabilistic forecasts.</span> <span class="abstract-toggle" data-id="2609.25720">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.25720v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.25720v1) · [:material-content-copy: BibTeX](bibtex/2609.25720.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=benchmarks-datasets" data-tag="benchmarks-datasets">Benchmarks & datasets</a>
    { .paper-tags }

-   #### West-WRF AI 2-km: High-Resolution Prediction of Integrated Vapor Transport and Precipitation { #2609.25512 }

    *Nazak Rouzegari, Vesta Afzali Gorooh, Agniv Sengupta, Phu Nguyen, Kuo-Lin Hsu, Amir AghaKouchak et al.* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.25512">We introduce a stretched-grid artificial intelligence (AI) weather forecasting model with 2-km resolution over the western United States and part of the Northeast Pacific and approximately 31-km...</span><span class="abstract-full" id="full-2609.25512" hidden>We introduce a stretched-grid artificial intelligence (AI) weather forecasting model with 2-km resolution over the western United States and part of the Northeast Pacific and approximately 31-km resolution elsewhere globally. Forecasting over the western U.S. is challenging because complex topography and atmospheric rivers (ARs) strongly influence orographic precipitation. West-WRF AI 2-km builds on a global model pretrained with a 40-year European Centre for Medium-Range Weather Forecasts Reanalysis v5 (ERA5) dataset and is fine-tuned with the Center for Western Weather and Water Extremes (CW3E) 2-km regional reanalysis to produce autoregressive 6-hourly forecasts of precipitation and integrated vapor transport (IVT). Forecasts are evaluated over winters 2020-2023 using gridded precipitation observations, rain gauges, and AR Reconnaissance dropsondes and are benchmarked against coarser-resolution AI forecasts and regional and global numerical weather prediction (NWP) systems. West-WRF AI 2-km reproduces observed precipitation-intensity distributions, retains fine-scale spectral variability, and produces sharper narrow coastal precipitation bands and localized, terrain-sensitive extremes. Its broader-scale performance remains comparable to coarser-resolution configurations while preserving large-scale skill despite higher resolution. Dropsonde verification shows lower errors and improved categorical skill at the most extreme IVT threshold. Overall, West-WRF AI 2-km provides its greatest value for localized precipitation extremes and intense AR-related moisture transport.</span> <span class="abstract-toggle" data-id="2609.25512">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.25512v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.25512v1) · [:material-content-copy: BibTeX](bibtex/2609.25512.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a> <a class="md-tag" href="/explore/?t=km-scale" data-tag="km-scale">Km-scale</a> <a class="md-tag" href="/explore/?t=quarter-degree" data-tag="quarter-degree">0.25°</a> <a class="md-tag" href="/explore/?t=6-hourly" data-tag="6-hourly">6-hourly</a>
    { .paper-tags }

-   #### FAST-ML: A Hybrid Physics-Machine Learning Framework for Tropical Cyclone Intensity Forecasting { #2609.25505 }

    *Shijie Xiao, Jonathan Lin, Thomas Ehrmann, Ali Sarhadi* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.25505">Rapid intensification (RI) remains one of the most consequential and difficult aspects of tropical cyclone (TC) forecasting. Although full-physics numerical weather prediction models can represent...</span><span class="abstract-full" id="full-2609.25505" hidden>Rapid intensification (RI) remains one of the most consequential and difficult aspects of tropical cyclone (TC) forecasting. Although full-physics numerical weather prediction models can represent the processes governing RI, resolving storm-environment interactions remains computationally expensive, while purely data-driven approaches often lack physical interpretability. We present FAST-ML, a hybrid framework that bridges data-driven efficiency with physical constraints. A physically informed dual-stream neural parameterization ingests 3D ERA5 fields to diagnose ventilation controls---environmental wind shear and mid-level entropy deficit. By optimizing these parameters end-to-end through a differentiable FAST intensity model, this architecture establishes a robust new paradigm for observation-driven parameter optimization, ensuring storm evolution remains strictly governed by thermodynamic principles. By better capturing the storm's continuous intensity evolution, FAST-ML improves upon its physical baseline, reducing ensemble CRPS across forecast lead times, with a reduction of approximately 31% at 60 h and nearly halving the RI false alarm ratio without sacrificing detection skill. In a 100-member ensemble configuration, FAST-ML produces intensity forecasts comparable to FNV3 for selected storms under the evaluated input configurations. Furthermore, zero-shot tests on selected Eastern Pacific storms provide encouraging evidence of cross-basin transferability. FAST-ML provides a modular intensity forecasting framework that can be coupled with externally supplied storm tracks and environmental fields. It demonstrates that observation-driven parameter learning within physically constrained dynamics simultaneously enhances accuracy, interpretability, and computational efficiency.</span> <span class="abstract-toggle" data-id="2609.25505">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.25505v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.25505v1) · [:material-content-copy: BibTeX](bibtex/2609.25505.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=physics-ml-hybrid" data-tag="physics-ml-hybrid">Physics–ML hybrid</a> <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a> <a class="md-tag" href="/explore/?t=interpretability" data-tag="interpretability">Interpretability</a> <a class="md-tag" href="/explore/?t=efficiency" data-tag="efficiency">Efficiency</a>
    { .paper-tags }

-   #### Learning Prognostic Variables for AI Convective Parameterizations via Symbolic Distillation { #2609.24882 }

    *Jurij Schönfeld, Tom Beucler, Julien Savre, Steven Sherwood, Veronika Eyring* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.24882">Hybrid AI-physics climate modeling aims to improve coarse (~100km-resolution) Earth system models by learning to parameterize subgrid processes from high-fidelity data. However, this so far mostly...</span><span class="abstract-full" id="full-2609.24882" hidden>Hybrid AI-physics climate modeling aims to improve coarse (~100km-resolution) Earth system models by learning to parameterize subgrid processes from high-fidelity data. However, this so far mostly involves local-in-time, diagnostic parameterizations, in which the subgrid state depends only on the current coarse state with no memory of previous states, which is unrealistic for processes such as convection that have intrinsic persistence. To address this, we enhance local-in-time parameterizations by learning prognostic variables that compactly carry important, additional past information where no explicit sub-grid information is available. First we compress past information into a low-dimensional latent space using an autoencoder, which then informs a neural network trained to parameterize targeted subgrid-scale processes. We then replace the autoencoder with symbolic equations that govern the time evolution of the latent variables, yielding additional prognostic memory variables that can be integrated alongside the resolved atmospheric state. We evaluate this approach on two systems: the Lorenz-96 model (online) and surface precipitation from high-resolution atmospheric simulations (offline). A forced multivariate linear ordinary differential equation recovers most of the added value achieved by the autoencoder-based approach in both experiments. Benchmarked against diagnostic parameterizations without memory, our memory-informed approach improves climate statistics and temporal structure, including a realistic diurnal cycle of tropical land precipitation.</span> <span class="abstract-toggle" data-id="2609.24882">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.24882v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.24882v1) · [:material-content-copy: BibTeX](bibtex/2609.24882.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a>
    { .paper-tags }

-   #### Inference of Unknown Dynamical Components Using Next Generation Reservoir Computing: From Chaotic Systems to Climate Data { #2609.24754 }

    *Jule Budnick, Andrew Keane, Serhiy Yanchuk* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.24754">We investigate next generation reservoir computing (NGRC) as a data-driven approach for inferring unseen components of dynamical systems. We compare NGRC with traditional reservoir computing (RC)...</span><span class="abstract-full" id="full-2609.24754" hidden>We investigate next generation reservoir computing (NGRC) as a data-driven approach for inferring unseen components of dynamical systems. We compare NGRC with traditional reservoir computing (RC) using the Lorenz and Rössler system, where two unknown components are inferred from one given component. For both systems, NGRC achieves accurate results while requiring fewer training data and less computational time than RC. We identified an inverse proportional behavior between the number of time-delayed steps needed for NGRC and the temporal resolution, indicating that the physical time span covered by the delay interval is an important factor in determining the required number of delayed steps. Finally, we apply NGRC to the observational climate data of ENSO (El Niño--Southern Oscillation) and infer one observable from the remaining variables. Despite the noise and complexity of the real-world data, the NGRC shows promising results. Our findings demonstrate the potential of NGRC for efficient inference of unseen components in both controlled dynamical systems and real-world data.</span> <span class="abstract-toggle" data-id="2609.24754">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.24754v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.24754v1) · [:material-content-copy: BibTeX](bibtex/2609.24754.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=subseasonal-to-seasonal" data-tag="subseasonal-to-seasonal">Subseasonal to seasonal</a>
    { .paper-tags }

-   #### Climate Variability Modulates the Impact of Price Spikes on Food Insecurity { #2609.24394 }

    *Jordi Cerdà-Bautista, Vasileios Sitokonstantinou, Homer Durand, Gherardo Varando, Michele Ronco et al.* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.24394">Climate variability influences whether a market disruption escalates into a food crisis, yet broad climate patterns like El Niño, tracked months before they alter hydro-climatic conditions, are still...</span><span class="abstract-full" id="full-2609.24394" hidden>Climate variability influences whether a market disruption escalates into a food crisis, yet broad climate patterns like El Niño, tracked months before they alter hydro-climatic conditions, are still not incorporated as an early-warning component in food-security responses. We address this gap by introducing sensitivity regimes, a stratification of regions by the direction and strength of their vegetation response to the El Niño Southern Oscillation, and using them to estimate how food price spikes affect acute food insecurity across sub-Saharan Africa. Integrating remote sensing, socioeconomic data, and causal machine learning, we find that in regions where ENSO systematically suppresses vegetation, a price spike raises the share of the population at acute risk by 5.4 percentage points in the following month. In regions where vegetation is unaffected by or positively linked to ENSO, the estimated effect is smaller (around 2 percentage points) and statistically insignificant. These results demonstrate that climate context is critical for understanding food security vulnerabilities. Sensitivity regimes can be combined with operational price-spike triggers to stage anticipatory action: the ENSO state flags vulnerable regions months ahead, and a pre-positioned response in those regions to a price spike would avert the largest jump in acute food insecurity.</span> <span class="abstract-toggle" data-id="2609.24394">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.24394v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.24394v1) · [:material-content-copy: BibTeX](bibtex/2609.24394.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=subseasonal-to-seasonal" data-tag="subseasonal-to-seasonal">Subseasonal to seasonal</a>
    { .paper-tags }

</div>

