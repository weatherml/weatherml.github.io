---
hide:
  - navigation
  - toc
title: weatherml
---

A collection of papers on AI for weather forecasting, climate modelling and atmospheric science.

<p class="page-meta" markdown="span">1671 papers · updated 2026-11-22 · <a href="feed.xml">:material-rss: RSS</a> · <a href="all_papers.bib" download>:material-download: BibTeX</a> · <a href="https://github.com/weatherml/weatherml.github.io/issues/new?template=suggest-paper.yml">:material-plus: Suggest a paper</a></p>

## Browse by Topic

<div class="grid cards topics" markdown>

-   [Global Models](papers/global-models/index.md) <span class="topic-count">398</span>
-   [Regional Models](papers/regional-models/index.md) <span class="topic-count">70</span>
-   [Nowcasting](papers/nowcasting/index.md) <span class="topic-count">110</span>
-   [Downscaling](papers/downscaling/index.md) <span class="topic-count">110</span>
-   [Post-processing](papers/post-processing/index.md) <span class="topic-count">27</span>
-   [Data Assimilation](papers/data-assimilation/index.md) <span class="topic-count">84</span>
-   [Climate Modeling](papers/climate-modeling/index.md) <span class="topic-count">317</span>
-   [Hydrology](papers/hydrology/index.md) <span class="topic-count">41</span>
-   [Ocean & Sea Ice](papers/ocean-sea-ice/index.md) <span class="topic-count">94</span>
-   [Air Quality & Composition](papers/air-quality-composition/index.md) <span class="topic-count">79</span>
-   [Remote Sensing](papers/remote-sensing/index.md) <span class="topic-count">113</span>
-   [Other](papers/other/index.md) <span class="topic-count">228</span>

</div>

## Recent Additions

<div class="grid cards" markdown data-search-exclude>

-   #### Democratizing Cross-Project Knowledge for Transforming Water Governance and Climate Resilience: an Agentic AI Ecosystem { #2611.26398 }

    *Oriol Alàs, Ian Palacín, Iván Cester, Edgar Rubión* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.26398">EU-funded projects produce extensive evidence in the form of deliverables, reports, and case studies. However, this knowledge is typically scattered across projects and document structures, making it...</span><span class="abstract-full" id="full-2611.26398" hidden>EU-funded projects produce extensive evidence in the form of deliverables, reports, and case studies. However, this knowledge is typically scattered across projects and document structures, making it difficult to locate relevant, contextual information when making decisions. This paper introduces an Agentic AI Ecosystem that consolidates heterogeneous project documentation into a domain-aware knowledge base and provides adaptive access through an agentic Retrieval-Augmented Generation pipeline. Documents from three EU-funded projects (IMPETUS, NATALIE, and InnWater) are reconstructed into hierarchical section trees and indexed with structured metadata. Retrieval is supported by two domain-specific embedding models fine-tuned on domain data (water and climate). On top of this knowledge base, a ReAct-style controller iteratively orchestrates retrieval actions by allocating retrieval budget across domain collections and accumulating evidence until a stopping criterion is met. In a retrieval benchmark over the corpus, the proposed approach improves Mean Reciprocal Rank by 26.7% (relative) compared to a single-domain embedding baseline, indicating more consistent retrieval of relevant evidence to support cross-project knowledge synthesis and climate-resilient decision-making.</span> <span class="abstract-toggle" data-id="2611.26398">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.26398v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.26398v1) · [:material-content-copy: BibTeX](bibtex/2611.26398.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=llms-agents" data-tag="llms-agents">LLMs & agents</a>
    { .paper-tags }

-   #### Interpretable Prototype-Conditioned Adaptation for Weather Forecasting { #2611.25848 }

    *Zhixuan Li, Hao Jiang, Chen Hui, Kai Geng, Weisi Lin* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.25848">Data-driven weather forecasting requires distinguishing local states and their temporal evolution. However, learned patterns remain implicit in neural forecasters, making their influence difficult to...</span><span class="abstract-full" id="full-2611.25848" hidden>Data-driven weather forecasting requires distinguishing local states and their temporal evolution. However, learned patterns remain implicit in neural forecasters, making their influence difficult to interpret. Failing to distinguish local evolution can lead to similar forecasts for locations that develop differently, increasing prediction error. This motivates explicit references for distinguishing and interpreting local patterns. Prototypes provide explicit pattern references, but sample- or task-level conditioning leaves open how to adapt to distinct locations and variable groups. To deal with this problem, we design ProtoCast, which uses group-specific prototype banks to adapt local features while accounting for cross-variable dependencies. Specifically, (1) we combine each group's historical features and temporal change with forecast context from cross-variable attention for evolution-aware prototype matching. (2) Sparse mixtures of prototype-specific low-rank operators tailor corrections to each group and location. A gated attention block then exchanges the corrected features across groups before prediction. Each group's correction decomposes into explicit prototype contributions before this fusion. (3) We jointly train the forecasting network and operators from scratch and refit each group's prototypes before every training epoch. Slot tracking preserves prototype-operator associations, making each prototype's contribution to local corrections more interpretable. Experiments on GlobalForecast and TyphoonTracking show improved forecasting accuracy, while prototype deletion links local feature contributions to changes in prediction error. This work advances the development of more accurate and interpretable data-driven weather forecasting models. Project page: https://zhixuanli.github.io/projects/li2026protocast</span> <span class="abstract-toggle" data-id="2611.25848">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.25848v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.25848v1) · [:material-content-copy: BibTeX](bibtex/2611.25848.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a> <a class="md-tag" href="/explore/?t=interpretability" data-tag="interpretability">Interpretability</a>
    { .paper-tags }

-   #### Reinforcement-Aligned Probabilistic Weather Forecasting with Discrete Tokens { #2611.25570 }

    *Jinhan Li, Zhisheng Chen, Yuxuan Li, Yuan Gao, Fan Xu, Hao Wu, Jinlong Du, Kun Wang* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.25570">Global medium-range weather forecasting must not only generate ensembles efficiently to characterize uncertainty in future states, but also serve downstream applications such as extreme-event...</span><span class="abstract-full" id="full-2611.25570" hidden>Global medium-range weather forecasting must not only generate ensembles efficiently to characterize uncertainty in future states, but also serve downstream applications such as extreme-event warnings and cross-sphere coupled simulations. We introduce WeatherCraft, a framework that unifies probabilistic generation and reinforcement learning alignment through discrete weather tokens, bringing the GPT-style pretraining--post-training paradigm to global medium-range weather forecasting. A multiscale tokenizer represents atmospheric states as a coarse-to-fine token hierarchy, while a Transformer models causal dependencies across time and scale and predicts tokens within each scale in parallel. Direct sampling generates diverse ensembles, and exact token log-probabilities enable Group Relative Policy Optimization (GRPO) to maximize downstream task rewards without differentiating through evaluators or coupled models. On the ERA5 benchmark, WeatherCraft's eight-member ensemble mean reduces day-15 normalized RMSE by 17.6% relative to the strongest evaluated deterministic baseline and, with KV caching, achieves an approximately $18\times$ inference speedup over GenCast. Direct fair-CRPS adaptation improves ensemble probabilistic skill, while reward-based post-training improves forecasts of persistent land temperature extremes and coupled ocean heat extremes. This work establishes discrete autoregressive modeling as a general foundation for scalable, task-alignable Earth system foundation models.</span> <span class="abstract-toggle" data-id="2611.25570">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.25570v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.25570v1) · [:fontawesome-brands-github: Code](https://github.com/Jinhan-Li/WeatherCraft) · [:material-content-copy: BibTeX](bibtex/2611.25570.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=transformers" data-tag="transformers">Transformers</a> <a class="md-tag" href="/explore/?t=foundation-models" data-tag="foundation-models">Foundation models</a> <a class="md-tag" href="/explore/?t=reinforcement-learning" data-tag="reinforcement-learning">Reinforcement learning</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a>
    { .paper-tags }

-   #### Auditing Learned Dynamics in Weather Models { #2611.25119 }

    *Skylar Gu, Christina Last* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.25119">Learned weather models are experimentally accessible computational systems whose internal representations can be inspected and manipulated. Using sparse autoencoder features from GraphCast,...</span><span class="abstract-full" id="full-2611.25119" hidden>Learned weather models are experimentally accessible computational systems whose internal representations can be inspected and manipulated. Using sparse autoencoder features from GraphCast, DeepMind's autoregressive weather forecasting model, we test whether causal relevance requires physical semantics, a question that matters for auditing whether AI forecasters used in high-assurance settings rely on reliable mechanisms. We also look at whether physically associated latent directions can be steered to improve extreme-event forecasts, where interventions are applied to the model's internal activations during the rollout. We find that ablating a feature locked to GraphCast's computational mesh increases global geopotential height error despite lacking conventional physical semantics. Second, suppressing convection- and low-level-vorticity-associated features weakens predicted tropical-cyclone deepening (the fall in central pressure during intensification), while amplifying them reduces intensity error, with amplification of convection features through feature scaling achieving a median of 51% error reduction over seven storms. Together, these results separate interpretable, physical association from causal utility and establish sparse features as experimental coordinates for auditing and intervening on learned weather dynamics.</span> <span class="abstract-toggle" data-id="2611.25119">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.25119v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.25119v1) · [:material-content-copy: BibTeX](bibtex/2611.25119.bib){ .bibtex-link }
    { .paper-links }

-   #### Evaluation of machine-learning and deep-learning SPEI drought forecasts against persistence across timescales and lead times in Bangladesh { #2611.25075 }

    *Mahir Ahnaf, Showmitra Kumar Sarkar* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.25075">Bangladesh, home to nearly 178 million people, is highly vulnerable to recurring droughts that threaten food security and rural livelihoods. Machine-learning and deep-learning models are increasingly...</span><span class="abstract-full" id="full-2611.25075" hidden>Bangladesh, home to nearly 178 million people, is highly vulnerable to recurring droughts that threaten food security and rural livelihoods. Machine-learning and deep-learning models are increasingly used to forecast drought indices such as the Standardized Precipitation Evapotranspiration Index (SPEI). Most studies report accuracy without a persistence reference, although accumulated indices are strongly autocorrelated, and seldom test drought detection, longer leads or variation between training runs. This study forecast SPEI at 3-, 6- and 12-month timescales one to three months ahead for the eight divisions of Bangladesh, using ERA5 reanalysis checked against 35 Bangladesh Meteorological Department stations. Five models, XGBoost, CNN, Bi-LSTM, CNN-BiLSTM and CEEMDAN-BiLSTM, were trained with five seeds each, scored on accuracy, skill against persistence and drought detection, and interpreted with Shapley values. At a one-month lead, the CNN-BiLSTM and Bi-LSTM beat persistence clearly at SPEI-3 and SPEI-6 (skill 0.408 and 0.350), while at SPEI-12 a Nash-Sutcliffe efficiency of 0.872 barely exceeded persistence (0.850). Skill and detection could disagree: at SPEI-3 and a three-month lead, XGBoost had the highest skill (0.439) yet detected none of the 119 drought months. Only at SPEI-6 did the forecasts beat persistence significantly and match its detection at all three leads, with fewer false alarms. At SPEI-12, gains were small or absent and varied between seeds. The forecasts kept 76-80% of their correlation against station-derived SPEI and relied mainly on precipitation and lagged SPEI. Forecasts should therefore be judged on skill against persistence and detection across several seeds, not on accuracy alone. In Bangladesh, SPEI-6 forecasts can support early warning up to three months ahead and SPEI-3 forecasts one month ahead, while SPEI-12 is better suited to monitoring.</span> <span class="abstract-toggle" data-id="2611.25075">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.25075v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.25075v1) · [:fontawesome-brands-github: Code](https://github.com/mahirahnaf1727/bangladesh-spei-drought-forecasting) · [:material-content-copy: BibTeX](bibtex/2611.25075.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a> <a class="md-tag" href="/explore/?t=recurrent-networks" data-tag="recurrent-networks">Recurrent networks</a> <a class="md-tag" href="/explore/?t=classical-ml" data-tag="classical-ml">Classical ML</a> <a class="md-tag" href="/explore/?t=extremes" data-tag="extremes">Extremes</a> <a class="md-tag" href="/explore/?t=evaluation" data-tag="evaluation">Evaluation</a>
    { .paper-tags }

-   #### Distribution-Free Uncertainty Quantification for Weather Forecasting: Conformal Prediction Under Spatiotemporal Dependence { #2611.24329 }

    * Mariya* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.24329">Conformal prediction gives finite-sample, distribution-free marginal coverage and has begun to be applied to gridded weather forecasts. Marginal coverage concerns the long-run fraction of covered...</span><span class="abstract-full" id="full-2611.24329" hidden>Conformal prediction gives finite-sample, distribution-free marginal coverage and has begun to be applied to gridded weather forecasts. Marginal coverage concerns the long-run fraction of covered points at a location, not how coverage behaves jointly across space and time on a given day. I benchmark eight interval-construction procedures (Gaussian, pooled and per-cell split conformal, per-season, adaptive conformal inference and rolling windows) for gridded 2 m temperature forecasts at eight lead times from 6 hours to 10 days, using ERA5 at 5.625 degree resolution, a deliberately modest convolutional forecaster and ECMWF's operational HRES, with calibration in 2017-2018 and testing in 2019-2021. Alongside coverage, width and interval score, I report dependence diagnostics: a spatial overdispersion index, quantiles of daily spatial coverage and miss persistence relative to a shuffled-days null. Per-cell methods reach near-nominal marginal coverage, yet the day-to-day variance of the fraction of cells covered is roughly 3 to 10 times what independent cells would give, and misses cluster in time; both effects grow with lead time for the CNN, also appear for HRES, and are shared by all methods. An oracle that divides each day's normalised errors by that day's mean removes 81 to 92 percent of the excess dispersion for the CNN and 93 to 98 percent for HRES, so days of large errors across much of the grid account for most of it. A day-scale predicted from seasonal terms and already-verified errors lowers the overdispersion index without widening intervals but improves interval score by at most 0.6 percent; on HRES from 12 hours to 3 days it has the lowest overdispersion index while ACI or season-conditional calibration has the best interval score, so no method is best on both. The study covers one variable, one coarse grid, a reanalysis target and two forecasters.</span> <span class="abstract-toggle" data-id="2611.24329">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.24329v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.24329v1) · [:fontawesome-brands-github: Code](https://github.com/Mariyaaleem/conformal-wx) · [:material-content-copy: BibTeX](bibtex/2611.24329.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=coarse" data-tag="coarse">Coarse (≥1°)</a> <a class="md-tag" href="/explore/?t=daily" data-tag="daily">Daily</a>
    { .paper-tags }

-   #### Enabling joint chemical and meteorological data assimilation with an atmospheric foundation model { #2611.24020 }

    *Eliot J. Kim, Eric J. Mei, Peter D. Ivatt, Lee T. Murray, Alexander J. Turner* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.24020">The hydroxyl radical (OH) is the main oxidant in the troposphere and accounts for 88% of the methane sink, yet its abundance and trends remain poorly constrained. Satellites observe many of the...</span><span class="abstract-full" id="full-2611.24020" hidden>The hydroxyl radical (OH) is the main oxidant in the troposphere and accounts for 88% of the methane sink, yet its abundance and trends remain poorly constrained. Satellites observe many of the chemical and meteorological drivers of OH, but assimilating both requires a model that represents chemistry and meteorology together. Existing chemistry-climate models are too expensive for joint data assimilation. Here, we present AuroraChem, a fine-tuned version of the Aurora atmospheric foundation model that jointly simulates meteorology and the GEOS-Chem chemical fields needed to constrain OH. A one-year AuroraChem rollout takes 12 minutes on one GPU. Free rollouts remain stable for one year but do not fully capture seasonal cycles. Using idealized data assimilation, we find that chemical observations can constrain the meteorological state and meteorological observations can constrain the chemical state. Inserting only the chemical fields raises the annual mean spatial correlation of 500 hPa temperature from 0.89 to 0.92 and recovers the moisture and circulation pattern of the Indian summer monsoon. Inserting only the meteorological fields raises the 500 hPa OH correlation from 0.89 to 0.93, close to the 0.94 from inserting the chemical fields themselves, and recovers the OH seasonal cycle. Inserting only NO2 columns sampled at TROPOMI pixels gives an OH correlation of 0.93. These relationships likely reflect learned covariances rather than physical coupling, and covariances are what data assimilation uses to let an observation of one variable update the others. AuroraChem therefore provides a new pathway to joint chemical and meteorological data assimilation.</span> <span class="abstract-toggle" data-id="2611.24020">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.24020v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.24020v1) · [:material-content-copy: BibTeX](bibtex/2611.24020.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=foundation-models" data-tag="foundation-models">Foundation models</a> <a class="md-tag" href="/explore/?t=sub-hourly" data-tag="sub-hourly">Sub-hourly</a>
    { .paper-tags }

-   #### Comparing Data-Driven and Numerical Weather Prediction Forecasts of Extreme Weather: A Review { #2611.23863 }

    *Gabriele Messori, Hiu-Wang Lau, Chenyu Dong, Gianmarco Mengaldo, Leonardo Olivetti* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.23863">Data-driven weather forecasting models, chiefly trained on reanalysis data, now match or beat physics-based numerical weather prediction (NWP) models on most headline scores, but whether this...</span><span class="abstract-full" id="full-2611.23863" hidden>Data-driven weather forecasting models, chiefly trained on reanalysis data, now match or beat physics-based numerical weather prediction (NWP) models on most headline scores, but whether this advantage extends to extreme weather is contested. We review peer-reviewed studies that directly compare data-driven forecasts with NWP for extremes, from nowcasting to subseasonal lead times. We discuss grey literature in the outlook but do not include it in the formal review. There is good agreement on the fact that data-driven models mostly outperform NWP on temperature extremes, tropical cyclone tracks and the location of synoptic features across time scales. The evidence is mixed for extreme precipitation and feature-based extreme event detection such as extremes related to atmospheric rivers, while it leans toward better NWP performance for estimating event intensity. For record-breaking events that lie outside the training distribution, the evidence is narrower. Nonetheless, both a systematic global evaluation and several case studies find that physics-based models retain the advantage, with data-driven models underestimating both the frequency and the magnitude of records. Hybrid systems that post-process NWP with machine-learning (ML), or couple data-driven forecasts with physics-based models, deliver some of the largest and most consistent gains. We argue that headline scores, verification choices and case selection explain much of the apparent disagreement in the literature, and provide recommendations for a more trustworthy and interpretable evaluation of data-driven models for extreme and record-breaking weather.</span> <span class="abstract-toggle" data-id="2611.23863">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.23863v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.23863v1) · [:material-content-copy: BibTeX](bibtex/2611.23863.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=extremes" data-tag="extremes">Extremes</a> <a class="md-tag" href="/explore/?t=subseasonal-to-seasonal" data-tag="subseasonal-to-seasonal">Subseasonal to seasonal</a> <a class="md-tag" href="/explore/?t=evaluation" data-tag="evaluation">Evaluation</a>
    { .paper-tags }

-   #### Improving Eddy Detection from Wide-Swath Altimetry with Multimodal Data-Driven Models { #2611.22601 }

    *Matteo Broccoli, Andrea Cipollone, Antonio Bonaduce, Elisa Carli, Florian Le Guillou* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.22601">Physically-based eddy detection algorithms usually rely on hard-coded thresholds and constraints that make the detected structures sensitive to small variations in the underlying fields. One way to...</span><span class="abstract-full" id="full-2611.22601" hidden>Physically-based eddy detection algorithms usually rely on hard-coded thresholds and constraints that make the detected structures sensitive to small variations in the underlying fields. One way to relax such conditions is the use of machine-learning methodologies that transform the full problem into an optimization process. We present a multimodal, multi-task, data-driven eddy detection framework based on a shallow convolutional U-Net, trained to jointly segment eddy structures and reconstruct the co-located spatial sea surface temperature (SST) anomaly using a dynamically-based reconstruction as target. Two model variants are developed: NADIRnet, trained on conventional nadir altimetry, and SWOTnet, trained through a curriculum that incorporates SWOT-enriched maps. Both models achieve high pixel- and structure-wise detection scores against the ground-truth, with most residual disagreement attributable to boundary placement rather than missed or spurious detections. The coupled SST regression task, enforced through boundary-coupling and gradient-matching loss terms, is shown to sharpen and regularise the detected eddy boundaries. Compared to the dynamically-based reference, both models recover a substantially larger and finer eddy population; notably, SWOTnet's higher-resolution training signal transfers to conventional nadir altimetry thanks to the higher-resolution SST channel, detecting more eddies than a nadir-trained model on the same data. The framework further generalises, even without retraining, to altimetry products and processing levels unseen during training, including SWOT L3 wide-swath retrievals and an independent gridded product. Together with its low computational cost, this framework is thus promising for near-real-time operational deployment and further downstream applications.</span> <span class="abstract-toggle" data-id="2611.22601">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.22601v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.22601v1) · [:fontawesome-brands-github: Code](https://github.com/MatteoBroccoli/WATER-ML-Eddy-Detection) · [:material-content-copy: BibTeX](bibtex/2611.22601.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a>
    { .paper-tags }

-   #### DARWIN: a Geospatial Foundation Model for Marine Intelligence { #2611.22496 }

    *Coen van den Elsen, Laurens Geffert* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.22496">We present DARWIN, a geospatial foundation model for marine intelligence. Recent Earth-observation foundation models have concentrated on terrestrial domains. The marine realm poses distinct...</span><span class="abstract-full" id="full-2611.22496" hidden>We present DARWIN, a geospatial foundation model for marine intelligence. Recent Earth-observation foundation models have concentrated on terrestrial domains. The marine realm poses distinct challenges: missingness, multi-modal physical and biogeochemical variables, sparse labels, and dynamics that are transient, multi-scale, and non stationary over time. DARWIN adapts a masked-autoencoder backbone to these properties through missingness-aware encoding, expanded spatial and temporal masking, and Fourier-style positional embeddings, and is pretrained on 26 years of global Copernicus reanalysis together with GEBCO bathymetry.   On predicting chlorophyll-a, the pretrained backbone attains the lowest monthly relative error in every month of 2022, outperforming both no-encoder baselines and an identically architected random-init encoder. At 10% of the supervised training data, DARWIN already surpasses the random-init model trained on 100% of the data. Embedding analysis and qualitative reconstructions show that DARWIN captures coherent oceanographic structure rather than memorising input statistics, providing a scalable backbone for downstream marine applications such as anomaly detection, species distribution modelling, and environmental monitoring.</span> <span class="abstract-toggle" data-id="2611.22496">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.22496v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.22496v1) · [:material-content-copy: BibTeX](bibtex/2611.22496.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=foundation-models" data-tag="foundation-models">Foundation models</a> <a class="md-tag" href="/explore/?t=monthly" data-tag="monthly">Monthly</a>
    { .paper-tags }

-   #### SeasonRainBench: Global Seasonal Precipitation { #2611.22281 }

    *Qiuyan Sun, Mengxuan Chen, Haoyuan Liang, Guowen Li, Zhen Huang, Jianhao Lin, Juepeng Zheng et al.* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.22281">Accurate seasonal precipitation forecasts help societies prepare for drought, flooding, and crop failure months in advance. While data-driven methods have made rapid progress in weather and...</span><span class="abstract-full" id="full-2611.22281" hidden>Accurate seasonal precipitation forecasts help societies prepare for drought, flooding, and crop failure months in advance. While data-driven methods have made rapid progress in weather and subseasonal forecasting, operational seasonal precipitation prediction still relies on numerical ensemble systems. These systems are computationally expensive and exhibit biases and drift that increase with lead time, motivating more efficient data-driven alternatives. Yet this transition faces three linked challenges: comparing data-driven forecasts with operational ones requires (1) a global benchmark that does not exist, scoring them requires (2) metrics that capture operational priorities beyond anomaly correlation (ACC) and RMSE, and training them must (3) rely on a single, short observational record. To address the first two challenges, we introduce \BenchName, the first such benchmark, covering monthly precipitation forecasts 1--5 months ahead. It pairs global $1^\circ$ satellite precipitation with oceanic and atmospheric predictors, includes seven operational centers, and provides 310,944 simulated months from 52 climate models for training. Beyond standard scores, we introduce two operationally motivated metrics for anomaly strength and monthly rain-belt position. To address the third challenge, we develop \MODEL, which learns from simulations what the record lacks and transfers it to Earth through three modules: SimForced aligns the climate background, SimMemory transfers a history encoder pretrained on simulations, and SimAnalog supplements Earth labels with simulated analog outcomes. Extensive experiments show that \MODEL{} improves ACC, CRPS, anomaly strength, and rain-belt position by 12.0%, 3.6%, 2.9%, and 0.3% over the strongest operational or data-driven model, opening a path from simulated worlds to Earth system prediction.</span> <span class="abstract-toggle" data-id="2611.22281">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.22281v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.22281v1) · [:material-content-copy: BibTeX](bibtex/2611.22281.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=foundation-models" data-tag="foundation-models">Foundation models</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=extremes" data-tag="extremes">Extremes</a> <a class="md-tag" href="/explore/?t=subseasonal-to-seasonal" data-tag="subseasonal-to-seasonal">Subseasonal to seasonal</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=monthly" data-tag="monthly">Monthly</a>
    { .paper-tags }

-   #### Learning Calibrated and Samplable Predictive Distributions of Surface-Ocean $Δp$CO$_2$ Using Tabular Diffusion Models { #2611.21769 }

    *Gabriele Accarino, Mohammad M. Amirian, Thea H. Heimdal, Amanda R. Fay, Galen A. McKinley et al.* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.21769">The ocean absorbs almost 30% of anthropogenic CO$_2$ emissions, but quantifying this sink requires estimating surface-ocean CO$_2$ from observations that are sparse in space and time. Most existing...</span><span class="abstract-full" id="full-2611.21769" hidden>The ocean absorbs almost 30% of anthropogenic CO$_2$ emissions, but quantifying this sink requires estimating surface-ocean CO$_2$ from observations that are sparse in space and time. Most existing machine-learning estimates of this field are deterministic, giving a single value per grid point, with uncertainty approximated post hoc via residuals, ensembles, or prescribed distributions. We instead learn a conditional predictive distribution for each observation: using Treeffuser, a conditional diffusion model with gradient-boosted-tree score functions, we estimate the full predictive distribution of $Δp$CO$_2$ (the sea-air $p$CO$_2$ difference) from 13 environmental, spatial, and temporal predictors. On a held-out SOCAT test set, Treeffuser attains the lowest continuous ranked probability score among the uncertainty-quantification methods evaluated (8.229 $μ$atm) and the coverage closest to the nominal 68% / 95% levels (0.686 / 0.930), while retaining point-prediction skill comparable to a deterministic XGBoost baseline (RMSE 18.4 vs. 17.7 $μ$atm; $R^2$ 0.77 vs. 0.79). These results show that conditional diffusion can provide well-calibrated, observation-level uncertainties while largely preserving point-prediction skill. Because Treeffuser is generative, its samples can support probabilistic analyses of $Δp$CO$_2$; extending this to spatially coherent field reconstruction will require modeling dependence across grid cells and evaluation on spatially held-out data.</span> <span class="abstract-toggle" data-id="2611.21769">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.21769v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.21769v1) · [:material-content-copy: BibTeX](bibtex/2611.21769.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=classical-ml" data-tag="classical-ml">Classical ML</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### Quantifying the added value of human forecaster discussions for tropical cyclone intensity prediction { #2611.21602 }

    *Ilia Azizi, Frederick Iat-Hin Tam, Marc-Olivier Boldi, Valérie Chavez-Demoulin, Tom Beucler* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.21602">The National Hurricane Center routinely releases Tropical Cyclone Discussions that offer insights into its tropical cyclone forecasts. This study examines the added value of these discussions for...</span><span class="abstract-full" id="full-2611.21602" hidden>The National Hurricane Center routinely releases Tropical Cyclone Discussions that offer insights into its tropical cyclone forecasts. This study examines the added value of these discussions for intensity forecasting beyond a set of numerical predictors. For 637 Atlantic and eastern North Pacific tropical cyclones from the 2002-2023 hurricane seasons, we analyze the discussion text and its combination with 13 hurricane environmental predictors from SHIPS and ERA5 10-meter wind fields, for forecasting intensity changes at lead times of 24 to 120 hours using three modeling methods. In one approach, an end-to-end deep learning framework for the text shows that the discussions alone offer comparable value for 24-hour intensity forecasts to the SHIPS predictors. Adding discussion information to the SHIPS predictors in an ensemble of five such models reduces the mean absolute error of 24-hour maximum sustained wind forecasts for 2022-2023 cyclones from 9.89 to 8.90 kt, with a confidence interval that excludes zero when whole storms are resampled. The ERA5 wind fields have a minor effect on mean absolute error in wind predictions, and model rankings differ across predicted variables, lead times and model types. We provide prediction intervals for these forecasts, whose coverage falls below the 90% target during rapid intensification. Analysis of the text shows that words related to intensity and intensity change (such as intensify, weaken, eyewall and category) are important for the model predictions. This retrospective study shows the added value of the forecasters' discussions and suggests that multi-modal learning can help understand the predictive contribution of expert human judgment. The reanalysis data differ from real-time analyses, and the discussions are issued at least three hours after the valid time of the other inputs, so these results should not be interpreted operationally.</span> <span class="abstract-toggle" data-id="2611.21602">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.21602v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.21602v1) · [:material-content-copy: BibTeX](bibtex/2611.21602.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a>
    { .paper-tags }

-   #### Run it Back: Autoregression vs. Video Diffusion for Final-Conditioned Weather Trajectories { #2611.21521 }

    *Jerry Lin, Elizabeth A. Barnes* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.21521">Low-likelihood, high-impact extremes are projected to increase in severity and frequency in a warming climate. Sampling the precursors that lead to these events traditionally requires massive...</span><span class="abstract-full" id="full-2611.21521" hidden>Low-likelihood, high-impact extremes are projected to increase in severity and frequency in a warming climate. Sampling the precursors that lead to these events traditionally requires massive ensembles that scale poorly with lead time and rarity, and typical rare-event sampling and ensemble-boosting methods do not show how a single extreme might arise from diverse synoptic precursors. Recently, Lin et al.\ (2026) showed this is possible with final-conditioned 66-hour trajectories (trajectories constrained to end at an observed state) from cBottle-video, a pre-trained, off-the-shelf video diffusion climate emulator not explicitly fine-tuned for this task (Brenowitz et al., 2025). However, its ensembles were underdispersed and geostrophically inconsistent, and it reproduces conditioned frames only approximately, hindering autoregressive rollout. Here, we train Rewind-AR, a backward-in-time probabilistic autoregressive model with a FourCastNet~3 backbone (Bonev et al., 2025), and generate final-conditioned ensembles for the same three cases: the 2021 Pacific Northwest heatwave, Hurricane Ian, and Superstorm Sandy (in the Supplementary Information). The leading modes of Rewind-AR's antecedent 500 hPa geopotential height explain far more of the across-member variance than cBottle-video's for both hurricane position and heatwave surface temperature, and Rewind-AR better reproduces the day-over-day heatwave intensification observed in ERA5. Finally, geostrophic inconsistencies close to the conditioning frame seen in cBottle-video are not reproduced by Rewind-AR. Nevertheless, up to 22% of Rewind-AR members become numerically unstable within 66 hours, so improving rollout physical consistency and stability over long lead times is a natural next step.</span> <span class="abstract-toggle" data-id="2611.21521">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.21521v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.21521v1) · [:material-content-copy: BibTeX](bibtex/2611.21521.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### Improving Tropical Cyclone Forecasts through Objective Ensemble Track-Error Reduction { #2611.20251 }

    *Shuochen Wang, Mark S. Allen, Auroop R. Ganguly* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.20251">Accurate tropical cyclone (TC) track guidance is essential for warnings, evacuation planning, and risk communication, but outlying members and distinct track scenarios can degrade raw ensemble...</span><span class="abstract-full" id="full-2611.20251" hidden>Accurate tropical cyclone (TC) track guidance is essential for warnings, evacuation planning, and risk communication, but outlying members and distinct track scenarios can degrade raw ensemble forecasts. Statistical and machine-learning (ML) methods proposed to improve ensemble guidance are difficult to compare because previous studies often use different forecast systems, basins, periods, and verification methods. Using a common verification framework, this study compares six objective methods (four statistical trimming and two ML methods) with the full ensemble mean for 138 TCs forecast by GEFS, MOGREPS-G, and IFS during 2016-2024. Dynamic trimming (DT) is the most consistent statistical method, reducing mean total track error by 8.0%, 13.4%, and 16.1% at short, medium, and long ranges. Its largest benefits occur in high-spread cases, where it can reach 70% of the low-error bound. Short-lead-error trimming (SLET) provides useful delayed guidance after early observations become available, but its improvement decreases with lead time. Regression-mixture clustering (RC) shows potential in low-spread cases and provides the largest improvement over the control run. The gradient boosted tree (GBT) produces the largest medium-range improvement and exceeds the theoretical trimming bound, suggesting strong potential, particularly when the observed track is poorly represented by the ensemble family. However, its 12.8% short-range degradation and gains concentrated in a small subset of cases indicate overcorrection when the ensemble mean is already accurate, highlighting the need for additional constraints. These results provide a consistent basis for using objective error-reduction methods to strengthen TC track guidance for operational decision-making.</span> <span class="abstract-toggle" data-id="2611.20251">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.20251v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.20251v1) · [:material-content-copy: BibTeX](bibtex/2611.20251.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=classical-ml" data-tag="classical-ml">Classical ML</a> <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### A plug-in energy-balance freezing module for transformer frost nowcasting without leaf-sensor inputs { #2611.20114 }

    *Juyeon Kim, Chaeyeon Yi, Yun Am Seo* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.20114">Frost protection requires local warnings where leaf-sensor inputs are unavailable. In the frost literature we surveyed, physics entered machine-learning models through inputs, numerical-model...</span><span class="abstract-full" id="full-2611.20114" hidden>Frost protection requires local warnings where leaf-sensor inputs are unavailable. In the frost literature we surveyed, physics entered machine-learning models through inputs, numerical-model coupling, or losses, rather than model structure. We develop a transformer-compatible module that embeds leaf energy-balance physics in freezing-probability forecasts: an energy-balance temperature reference and a learned residual predict leaf temperature, which drives a multiplicative gate imposing a temperature-dependent probability cap. We mathematically prove conditional monotonicity and an alarm boundary, and test whether placing the same information in the output pathway or in attention yields different benefits. We evaluated 12 hourly leads at 19 Korean stations over five frost seasons, with and without leaf-sensor inputs. In the 2025-2026 hold-out season without these inputs, our time-token transformer (PET-E) achieved a lead-mean critical success index (CSI) of 0.516 versus 0.478 for fixed-configuration, joint-lead LightGBM, with significant gains at all 12 leads, and the fraction of alarms at observed leaf temperatures of at least 2 degrees C fell from 0.087 to 0.051 relative to the unmodified backbone. Across the hold-out and a retrospective 2024-2025 evaluation, the module reduced Brier scores and warm-surface alarm fractions in both the time-token transformer and iTransformer, with no significant CSI decrease at any lead. At held-out stations without static or leaf-sensor inputs, both module-equipped backbones exceeded LightGBM at all 12 leads (CSI 0.468 and 0.467 versus 0.437). In the time-token backbone, the temperature pathway reduced this alarm fraction in all four settings, whereas the attention prior increased CSI in three. These findings link an explicit temperature-dependent probability cap to improved forecasts and distinguish output-side from attention-based embedding.</span> <span class="abstract-toggle" data-id="2611.20114">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.20114v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.20114v1) · [:material-content-copy: BibTeX](bibtex/2611.20114.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=transformers" data-tag="transformers">Transformers</a> <a class="md-tag" href="/explore/?t=classical-ml" data-tag="classical-ml">Classical ML</a> <a class="md-tag" href="/explore/?t=station-point" data-tag="station-point">Station / point</a> <a class="md-tag" href="/explore/?t=hourly" data-tag="hourly">Hourly</a>
    { .paper-tags }

-   #### Physics-Spawned Neural Integration of Rotating Shallow-Water Dynamics { #2611.20054 }

    *Kaushal Gianchandani, Luka Todorovic, Sai Ravela* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.20054">Equation-derived neural integration can execute prescribed dynamics without training, while informative ensemble learning can estimate uncertain parameters and identify governing interactions without...</span><span class="abstract-full" id="full-2611.20054" hidden>Equation-derived neural integration can execute prescribed dynamics without training, while informative ensemble learning can estimate uncertain parameters and identify governing interactions without requiring differentiability. Together, these approaches motivate Physics-Spawned Informative Networks ($Φ$SpIN), a framework for prescribed and adaptive hybrid dynamical systems. As a first step toward flood modeling applications, we present $Φ$SpIN-SW, its prescribed-physics realization for the rotating shallow-water equations. Compatible C-grid operators on spherical Voronoi meshes define a computational graph that retains non-polynomial operations and advances the state through recurrent neural integration compiled with JAX. Analytical benchmarks characterize numerical errors, and global and coastal experiments closely reproduce reference solutions generated using the Model Prediction Across Scales (MPAS) model. At approximately 14~km global resolution, A100, H200 and B200 GPUs attain 9-, 19- and 34-fold higher timestep throughput than a 96-core MPAS-SW CPU reference under matched workloads. These results extend training-free neural integration beyond polynomial dynamics to complex geophysical simulation, establishing a computational foundation for subsequent adjoint-free ensemble estimation of physical and neural parameters and informative adaptation of operator structure.</span> <span class="abstract-toggle" data-id="2611.20054">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.20054v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.20054v1) · [:material-content-copy: BibTeX](bibtex/2611.20054.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=recurrent-networks" data-tag="recurrent-networks">Recurrent networks</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### GPROF V08: Improving Satellite Precipitation Estimates Across a Heterogeneous Constellation of Passive-Microwave Sensors { #2611.19978 }

    *Simon Pfreundschuh, Paula J. Brown, Christian D. Kummerow* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.19978">Precipitation estimates from the passive microwave observations (PMW) of the Global Precipitation Measurements (GPM) mission serve a wide range of scientific, societal, and economic applications. The...</span><span class="abstract-full" id="full-2611.19978" hidden>Precipitation estimates from the passive microwave observations (PMW) of the Global Precipitation Measurements (GPM) mission serve a wide range of scientific, societal, and economic applications. The Goddard Profiling Algorithm (GPROF) is the operational retrieval used to produce the precipitation estimates for all GPM PMW sensors. This study presents the implementation GPROF V08 -- the most recent version of GPROF and a major update aimed at improving the accuracy of precipitation estimates from the GPM constellation.   GPROF V08 replaces the conventional Bayesian scheme used by earlier GPROF versions by a convolutional neural network (CNN). The first part of this two-part study described a novel ML framework for generating synthetic observations required for training CNN retrievals for all GPM sensors. This second part describes the final, operational implementation, which adds a second finetuning stage to the training.Retrievals from the new GPROF V08 algorithm for all principal GPM sensors are validated against reference precipitation estimates from various sources. Our results show that, while the magnitude of improvements vary by sensor and surface type, they are robust across sensors and metrics and can be as high as 40 % for the linear correlation coefficient for precipitation estimates from SSMIS and ATMS over ocean compared to combined radar and radiometer retrievals from GPM.   While some challenges remain -- such as an increased sensitivity to sensor degradation and the persistence of climatological biases, which remain of the same order as for GPROF V07 -- GPROF V08 constitutes a substantial advancement in spaceborne precipitation monitoring. Since the improvements are robust across all sensor of the constellation, the update is expected to improve the full record of GPM precipitation estimates and thus benefit a wide range of downstream datasets and applications.</span> <span class="abstract-toggle" data-id="2611.19978">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.19978v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.19978v1) · [:material-content-copy: BibTeX](bibtex/2611.19978.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a>
    { .paper-tags }

-   #### Towards GPROF V08: A Machine-Learning Framework for Simulating Passive Microwave Observations of Precipitation { #2611.19966 }

    *Simon Pfreundschuh, Christian D. Kummerow, Paula J. Brown* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.19966">Spatially resolved precipitation retrievals based on convolutional neural networks (CNNs) can improve precipitation estimates from passive microwave (PMW) observations and will be used to implement...</span><span class="abstract-full" id="full-2611.19966" hidden>Spatially resolved precipitation retrievals based on convolutional neural networks (CNNs) can improve precipitation estimates from passive microwave (PMW) observations and will be used to implement the next operational PMW retrieval algorithm for the Global Precipitation Measurement (GPM) mission. However, the conventional radiative transfer simulations used to extend the retrieval algorithm to the various sensors of the GPM constellation do not provide the spatial coverage required to train CNN-based retrievals.   To address this, we introduce a machine-learning-based simulation framework that enables the training of the CNN-based GPROF-NN 3D retrieval algorithm for the full GPM constellation. At the core of this framework is Satformer, a new neural-network architecture that can translate PMW observations between any pair of GPM sensors. The resulting synthetic observations exhibit lower errors than conventional radiative transfer simulations and lead to more accurate precipitation retrievals when used to train GPROF-NN precipitation retrievals. We further demonstrate the applicability of the framework to new instruments by developing GPROF-NN retrievals for the Microwave Imager aboard the Weather System Follow-on Microwave satellite. While GPROF-NN retrievals based on the new simulation framework achieve lower instantaneous retrieval errors and higher correlations compared to independent reference data than conventional retrievals, residual simulation errors can introduce systematic biases in the resulting retrievals.   The proposed simulation framework provides a key component for extending GPROF-NN to the full GPM constellation as part of the upcoming operational GPROF V08 retrieval algorithm. The results demonstrate the potential of machine-learning-based simulations to support spatially resolved precipitation retrievals across heterogeneous satellite constellations.</span> <span class="abstract-toggle" data-id="2611.19966">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.19966v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.19966v1) · [:material-content-copy: BibTeX](bibtex/2611.19966.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a>
    { .paper-tags }

-   #### Spatiotemporal Reconstruction of Auroral Precipitations with Neural Fields { #2611.19757 }

    *Nazar Misyats, Mizuki Fukizawa, Yoshimasa Tanaka, Satoshi Ikehata* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.19757">Auroras provide remote optical observations of energetic electrons precipitating into the upper atmosphere, and whose energy distribution encodes key information about large scale magnetospheric...</span><span class="abstract-full" id="full-2611.19757" hidden>Auroras provide remote optical observations of energetic electrons precipitating into the upper atmosphere, and whose energy distribution encodes key information about large scale magnetospheric processes. Recovering the underlying electron precipitation field from ground-based camera images of auroras remains a challenging inverse problem. Existing auroral tomography techniques are often limited to simplified static settings, discrete spatial and energy grids, or strong assumptions on the electron energy spectrum. We propose to model the dynamic auroral electron precipitation field via a temporally conditioned neural network. The state of the precipitation field at each frame is represented by a per-frame latent code driven by learned discrete dynamics. A shared decoder maps horizontal positions and latent states to electron energy spectra. The predicted flux is converted to multi-wavelength volume emission rates and rendered into camera images accounting for delayed emissions. We evaluate the method on synthetic dynamic auroral sequences and real multi-view auroral videos, showing that the proposed representation produces temporally coherent precipitation reconstructions and image projections consistent with the observations.</span> <span class="abstract-toggle" data-id="2611.19757">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.19757v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.19757v1) · [:material-content-copy: BibTeX](bibtex/2611.19757.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a>
    { .paper-tags }

-   #### TyphoonWorld: A Residual Token World Model for Probabilistic Forecasting of Typhoon Imagery { #2611.18740 }

    *Xiaoyu Chen, Tianyi Li, Xuanyi Zhang* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.18740">A satellite forecast of a typhoon should give a low-error picture of the next two days, together with an ensemble whose members are coherent fields and whose spread responds to the storm at hand. A...</span><span class="abstract-full" id="full-2611.18740" hidden>A satellite forecast of a typhoon should give a low-error picture of the next two days, together with an ensemble whose members are coherent fields and whose spread responds to the storm at hand. A small direct regression network already gives the lowest squared error on held-out storms, so a generative model has to earn its cost through the part of the future that the regression smooths away. We present TyphoonWorld, a discrete-token world model that generates only that part. Conditioned on tokens of the regression forecast, the latest frames and the best-track state at forecast time, it samples residual fields lead by lead, and members add these residuals to the regression center. We compare it under one protocol with per-pixel Gaussian networks, a diffusion model of the same residuals, residuals from other storms and an unconditional token world model, on 276 forecast windows from 46 held-out storms of Digital Typhoon. TyphoonWorld matches the squared error of the regression, lowers the CRPS of the area colder than 220 K, a direct measure of deep convection, by 14 to 39% against every other ensemble, and ties its spread to forecast error as closely as the Gaussian network, with a correlation of 0.78 across windows. It uses 56% of the parameters of the unconditional world model.</span> <span class="abstract-toggle" data-id="2611.18740">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.18740v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.18740v1) · [:material-content-copy: BibTeX](bibtex/2611.18740.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### WeatherDreamer: Learning a World Model from Multimodal Earth Observations { #2611.18393 }

    *Kate Duffy, Elise Penn, Tristan Ballard, August Posch, Thomas J. Vandal* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.18393">Machine learning weather models now rival operational global forecasts but still start from analyses produced by separate, expensive data assimilation (DA) systems, which for global models are...</span><span class="abstract-full" id="full-2611.18393" hidden>Machine learning weather models now rival operational global forecasts but still start from analyses produced by separate, expensive data assimilation (DA) systems, which for global models are typically updated every six hours. Observation-to-observation models avoid this dependence, but most omit the recursive filtering cycle that lets DA systems carry a background state through sparse, irregular observations. We introduce WeatherDreamer, a world model that learns this cycle end-to-end from 18 global observing systems, without reanalysis. It infers a latent state from observations, carries it forward hour by hour with learned dynamics, corrects it each hour with new observations, and can predict what any of its observing systems would measure at any location. At held-out stations, its ensembles beat NOAA's Global Ensemble Forecast System (GEFS) for temperature, dew point and wind out to 12 h and are better calibrated than GEFS and the ECMWF ensemble. Precipitation, cloud-top height and solar irradiance, never inputs at inference, are inferred with skill matching or exceeding operational products at short leads. Against the first guess of NOAA's Gridpoint Statistical Interpolation (GSI) DA system, WeatherDreamer's short-range departures are within 10% of GSI's at surface stations and within 25% for moisture-sensitive microwave channels, but up to five times larger for temperature-sounding channels. Never given observation errors, its learned update assumes errors close to known instrument and retrieval errors of microwave sounders and scatterometer winds. WeatherDreamer also separates viewing geometry from atmospheric state, recovering 83-96% of observed limb darkening and producing seamless nadir-view imagery. Producing an hourly global analysis in about 1.5 minutes on one GPU, WeatherDreamer shows that assimilation and forecasting can be learned together directly from observations.</span> <span class="abstract-toggle" data-id="2611.18393">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.18393v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.18393v1) · [:material-content-copy: BibTeX](bibtex/2611.18393.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=energy" data-tag="energy">Energy</a> <a class="md-tag" href="/explore/?t=sub-hourly" data-tag="sub-hourly">Sub-hourly</a> <a class="md-tag" href="/explore/?t=hourly" data-tag="hourly">Hourly</a>
    { .paper-tags }

-   #### Tracing Uncertainties of Extreme El Niño Prediction to Mesoscale Ocean Currents { #2611.17854 }

    *Bo Qin, Zeyun Yang, Qiaofei Xue, Yanan Ding, Jinyu Wang, Chaopeng Ji, Shijin Yuan, Bin Mu et al.* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.17854">The predictability of the El Niño-Southern Oscillation (ENSO) has historically been linked to large-scale errors in models' initial conditions such as heat content and surface wind stress. Here we...</span><span class="abstract-full" id="full-2611.17854" hidden>The predictability of the El Niño-Southern Oscillation (ENSO) has historically been linked to large-scale errors in models' initial conditions such as heat content and surface wind stress. Here we demonstrate that these slowly evolving errors may ultimately originate from mesoscale errors in initial oceanic states that are common in current models and ocean observation systems. This hidden source of ENSO forecast uncertainty, in terms of mesoscale ocean-current errors, is convincingly exposed by the conditional nonlinear optimal perturbation (CNOP) method applied to a differentiable ocean AI model. The CNOP-identified most influential current errors are strongly event-dependent. For the 1997/98 El Niño, the initial current errors in the eastern Pacific results in the largest forecast error through locally suppressing original subsurface warming. For the 2015/16 event, the dominate error originates from the western Pacific, which reorganizes upper ocean temperature structure and affects the eastern Pacific through the equatorial waveguide. Further analysis reveals that these current errors perturb mesoscale eddy count and eddy kinetic energy triggering large-scale temperature errors that hinder El Niño development. The timing of the maximum growth rate of the obtained CNOP error is well aligned with the spring predictability barrier of ENSO, and the error structures are reflected in simulated analysis errors caused by unresolved mesoscale currents. These findings extend the potential source of ENSO predictability beyond large-scale thermal signals, reveal for the first time an uncharted route of ENSO forecast failure, and provide an important physical basis for targeted ocean observations and data assimilation in the future.</span> <span class="abstract-toggle" data-id="2611.17854">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.17854v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.17854v1) · [:material-content-copy: BibTeX](bibtex/2611.17854.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=subseasonal-to-seasonal" data-tag="subseasonal-to-seasonal">Subseasonal to seasonal</a>
    { .paper-tags }

-   #### Activation Steering the Nan-Shan in Typhoon { #2611.17768 }

    *Megan Wei, Damon Pham, John Cayley, Ellie Pavlick* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.17768">Typhoon is an audiovisual installation based on Joseph Conrad's novella, in which Captain MacWhirr steers the steamer Nan-Shan directly into and through a catastrophic storm. The novella is a study...</span><span class="abstract-full" id="full-2611.17768" hidden>Typhoon is an audiovisual installation based on Joseph Conrad's novella, in which Captain MacWhirr steers the steamer Nan-Shan directly into and through a catastrophic storm. The novella is a study of what happens when the scale of a phenomenon exceeds the practical and conceptual frameworks available to comprehend it and of the paralysis that sometimes follows a demand for action in such overwhelming circumstances. We chose the text of Typhoon to reflect on human culture's current AI moment. We prompt a large language model, seeded with Conrad's original language, to feed endlessly on its own output, and explore when and how the text becomes increasingly alien and unwieldy as time progresses, including deviations from the narrative, digressions into the model's own instructions and thinking, and long periods of incoherent rumination. We employ methods from activation steering, including a novel adaptive steering method we introduce, to control the rhythm of the narrative and the model's digressions over long stretches of generation. The simultaneous efficacy and unpredictability of this steering showcases the tension between the perceived interpretability of AI, and the reality of a system that remains outside our comprehension and control.</span> <span class="abstract-toggle" data-id="2611.17768">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.17768v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.17768v1) · [:material-content-copy: BibTeX](bibtex/2611.17768.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=llms-agents" data-tag="llms-agents">LLMs & agents</a> <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a>
    { .paper-tags }

</div>

