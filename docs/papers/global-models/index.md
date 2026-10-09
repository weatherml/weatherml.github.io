---
title: 'Global Models'
hide:
  - toc
---

<div class="listing-header" markdown>

# Global Models

<p class="page-meta" markdown="span">365 papers · page 1 of 13 · <a href="../../bib/global-models.bib" download>:material-download: BibTeX for this topic</a></p>

</div>

<div class="grid cards" markdown>

-   #### A Graph Neural Network for Global Daily Fire Radiative Power Prediction at Medium-Range Lead Times { #2610.11022 }

    *Li Zhang, Jun Wang, Isidora Jankov, Yongxin Liu, Gonzalo A. Ferrada, Ravan Ahmadov, Ligia Bernardet et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.11022">Skillful prediction of biomass-burning activity several days in advance is important for air-quality forecasting and aerosol prediction. Two operational constraints motivate this work. First, the...</span><span class="abstract-full" id="full-2610.11022" hidden>Skillful prediction of biomass-burning activity several days in advance is important for air-quality forecasting and aerosol prediction. Two operational constraints motivate this work. First, the GBBEPx satellite fire radiative power (FRP) product used to initialize NOAA's GEFS-Aerosols is available with about a 1.5-day latency, so each forecast cycle relies on the most recently available, but already outdated, fire observations. Second, these fire inputs are then held fixed throughout the subsequent 5-day operational forecast, or 7 days in the GSL experimental system, effectively assuming no evolution in fire activity. We develop a data-driven model that predicts global FRP one to seven days ahead from the most recent available observations. The model adapts a spatiotemporal graph neural network using reanalysis meteorology, land-cover and vegetation information, recent fire history, and GBBEPx FRP as the training target. It is trained on 2020-2022 data and evaluated for 2023-2024. The model reproduces the global seasonal cycle and substantially outperforms persistence. At 0.1$^\circ$ resolution, mean squared error is reduced by 32% at one-day lead and 43% at seven days in 2023, and by 24% and 40% in 2024. At 1$^\circ$ resolution, the critical success index ranges from 0.32 to 0.60. Detection skill declines only modestly with lead time, whereas intensity skill degrades more rapidly. Large fires are detected reliably, but their radiative power is systematically underestimated. These results demonstrate useful predictability of fire activity several days ahead and identify intensity calibration and small-fire placement as the main remaining challenges before predicted FRP can support operational aerosol forecasts.</span> <span class="abstract-toggle" data-id="2610.11022">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.11022v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.11022v1) · [:material-content-copy: BibTeX](../../bibtex/2610.11022.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=graph-neural-networks" data-tag="graph-neural-networks">Graph neural networks</a> <a class="md-tag" href="/explore/?t=coarse" data-tag="coarse">Coarse (≥1°)</a> <a class="md-tag" href="/explore/?t=daily" data-tag="daily">Daily</a>
    { .paper-tags }

-   #### Strategic Governance of AI Models in Earth Science { #2610.10560 }

    *Makoto Kelp, Amirhossein Arzani, Patricia Castellanos, Paul Griffiths, Ivan Higuera-Mendieta et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.10560">AI foundation models pretrained on weather and climate data are increasingly fine-tuned to Earth science tasks well beyond weather forecasting. Their development and adoption are outpacing the...</span><span class="abstract-full" id="full-2610.10560" hidden>AI foundation models pretrained on weather and climate data are increasingly fine-tuned to Earth science tasks well beyond weather forecasting. Their development and adoption are outpacing the scientific community's ability to evaluate them. These models are judged almost entirely by benchmark skill metrics, which measure how closely a forecast reproduces a reference product but not whether a model represents the physical processes governing the system it predicts. Forecast skill and physical reliability are therefore distinct properties. The distinction is most consequential under the nonstationary conditions of a changing climate for which these models were never trained. We identify five priorities for the physical evaluation of AI models in Earth science from task-specific emulators to foundation models, spanning training data, fine-tuning, behavioral testing, mechanistic interpretability, and output validation. We recommend three activities for the coming decade: 1) open AI-ready evaluation datasets, 2) a shared reporting standard for physics-based evaluation, and 3) a dedicated research program on the safety of these models.</span> <span class="abstract-toggle" data-id="2610.10560">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.10560v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.10560v1) · [:material-content-copy: BibTeX](../../bibtex/2610.10560.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=foundation-models" data-tag="foundation-models">Foundation models</a> <a class="md-tag" href="/explore/?t=evaluation" data-tag="evaluation">Evaluation</a> <a class="md-tag" href="/explore/?t=interpretability" data-tag="interpretability">Interpretability</a>
    { .paper-tags }

-   #### WxFM-XL: Adapting Univariate Foundation Models to Multi-Station Weather Forecasting { #2610.10057 }

    *Xiao Wang, Changjian Chen, Zhuo Tang, Rongwen Li, Hongwu Liu, Kenli Li* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.10057">With the rise of univariate time series foundation models (e.g., Sundial, Timer), initial efforts have been made to extend them to multivariate settings. However, these models mainly focus on...</span><span class="abstract-full" id="full-2610.10057" hidden>With the rise of univariate time series foundation models (e.g., Sundial, Timer), initial efforts have been made to extend them to multivariate settings. However, these models mainly focus on modeling correlations among variables. When they are applied to multi-station weather forecasting, two important factors are often overlooked: (1) the spatial information of stations, and (2) different error priors of different stations relative to the foundation model. In this paper, we propose WxFM-XL, a model for adapting univariate time series foundation models to multi-station weather forecasting. WxFM-XL introduces a cross-station error correlation prior graph to capture stationwise error priors with respect to the foundation model. Building on this, we further propose a dynamic fusion mechanism that adaptively integrates a spatial correlation graph with the error correlation prior graph. Experiments on multiple datasets demonstrate that our model outperforms state of the art baselines.</span> <span class="abstract-toggle" data-id="2610.10057">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.10057v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.10057v1) · [:material-content-copy: BibTeX](../../bibtex/2610.10057.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=foundation-models" data-tag="foundation-models">Foundation models</a> <a class="md-tag" href="/explore/?t=station-point" data-tag="station-point">Station / point</a>
    { .paper-tags }

-   #### Learning joint probabilistic weather forecasts from station observations alone { #2610.09898 }

    *Chaeyeon Yi, Yun Am Seo* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.09898">Assessing compound weather risks requires forecasts representing dependence between variables. CLARA (Calibrated Advection-Routing Attention) learns joint Gaussian predictive distributions of five...</span><span class="abstract-full" id="full-2610.09898" hidden>Assessing compound weather risks requires forecasts representing dependence between variables. CLARA (Calibrated Advection-Routing Attention) learns joint Gaussian predictive distributions of five surface variables from station observations alone, without numerical weather prediction or reanalysis; the approximately 28,000-parameter model supports CPU training and prediction. Across six multi-year folds on 96 stations, its lead-mean energy score is 4.9% lower than that of a learned comparator with matched temporal inputs (4.7% with a similar parameter count) and 11-65% lower than those of statistical baselines. Holding marginal variances fixed, removing learned correlations worsens joint negative log-likelihood by 1.0-2.8 nats per station. A covariance-scale estimator, proved consistent under stated assumptions, improves short-lead calibration but over-corrects at long leads. Synthetic interventions show an attention-bias coefficient alone does not measure forecast influence. Retrained in ten regions on six continents, CLARA outperforms persistence in all 60 multi-year region-lead comparisons and a similarly sized learned model in 57 of 60.</span> <span class="abstract-toggle" data-id="2610.09898">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.09898v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.09898v1) · [:material-content-copy: BibTeX](../../bibtex/2610.09898.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=station-point" data-tag="station-point">Station / point</a>
    { .paper-tags }

-   #### Mechanistic Interpretability of Atmospheric Rivers in GraphCast { #2610.07583 }

    *Madelyn Mathai, Timothy B. Higgins, Kevin M. Grise, Chirag Agarwal, Antonios Mamalakis* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.07583">While AI weather models now rival operational forecasts, how they represent the atmosphere internally remains an open question: feature attribution reveals which input patterns matter, not what the...</span><span class="abstract-full" id="full-2610.07583" hidden>While AI weather models now rival operational forecasts, how they represent the atmosphere internally remains an open question: feature attribution reveals which input patterns matter, not what the model computes or how it combines information internally. We train sparse autoencoders (SAEs) on GraphCast to uncover its learned concepts, using atmospheric rivers as our phenomenon of focus. Both standard and Matryoshka SAEs show GraphCast computes atmospheric river intensity, measured by integrated vapor transport (IVT), as a stable internal variable, despite IVT being neither an input nor a target. In contrast to the unstructured concept retrieval of the standard SAE, the Matryoshka SAE orders concepts by importance and exposes their relations. Atmospheric river concepts persist across depth and direct interventions confirm causality. This method offers a way to find internal variables and determine which of them the model actually relies on, which is a prerequisite for asking whether those variables remain meaningful as the phenomenon changes under a warming climate.</span> <span class="abstract-toggle" data-id="2610.07583">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.07583v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.07583v1) · [:material-content-copy: BibTeX](../../bibtex/2610.07583.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=interpretability" data-tag="interpretability">Interpretability</a>
    { .paper-tags }

-   #### Xaurora: Generative Weather Forecasting with Denoising Stochastic Interpolants from a Foundation Model Prior { #2610.06509 }

    *Eliot Walt, Miltiadis Kofinas, Nikolaj Mücke, Efstratios Gavves, Dim Coumou* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.06509">Deep learning has revolutionised weather forecasting in recent years, especially through atmospheric foundation models, which offer competitive skill for a fraction of the computational costs of...</span><span class="abstract-full" id="full-2610.06509" hidden>Deep learning has revolutionised weather forecasting in recent years, especially through atmospheric foundation models, which offer competitive skill for a fraction of the computational costs of classic physics-based models. However, most existing foundation models are deterministic, limiting the generation of large ensembles for accurate uncertainty quantification, extreme weather risk assessment, and long-range weather forecasting. Furthermore, these models incur a large, often prohibitive, computational overhead to train from scratch. To address these shortcomings, we turn a pretrained deterministic prior model, namely the Aurora foundation model, into a generative ensemble-prediction model. To that end, we introduce a novel generative method, Denoising Stochastic Interpolants, combined with a replay buffer for Stochastic Differential Equation (SDE) rollout, enabling probabilistic training of SDE trajectories. Our stochastic foundation model, Xaurora, is finetuned from the small Aurora version, yet it approaches the state-of-the-art on global ensemble metrics and is competitive with the large version of Aurora. Our method is parameter and sample efficient, and generates skilful 15-day forecasts in 13 minutes. Our results demonstrate that deterministic foundation models can be efficiently extended into even stronger stochastic models.</span> <span class="abstract-toggle" data-id="2610.06509">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.06509v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.06509v1) · [:material-content-copy: BibTeX](../../bibtex/2610.06509.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=foundation-models" data-tag="foundation-models">Foundation models</a> <a class="md-tag" href="/explore/?t=sub-hourly" data-tag="sub-hourly">Sub-hourly</a>
    { .paper-tags }

-   #### FlexCast: Adaptive Weather Forecasting from Arbitrary Field Sets { #2610.05296 }

    *Yuang Zhang, Chen Hui, Weisi Lin, Haiqi Zhu, Xiulai Wang, Sun-Yuan Kung, Feng Jiang* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.05296">Most deep learning weather models assign a fixed set of variables and pressure levels to predefined channels, limiting transfer across atmospheric field configurations. This dependence on a fixed...</span><span class="abstract-full" id="full-2610.05296" hidden>Most deep learning weather models assign a fixed set of variables and pressure levels to predefined channels, limiting transfer across atmospheric field configurations. This dependence on a fixed field set limits the transferability of trained models across atmospheric field configurations. We propose FlexCast, a field-adaptive weather forecasting model that uses a single set of parameters to produce identity-aligned forecasts for variable-cardinality subsets drawn from a 69-field ERA5 registry. Specifically, a metadata-conditioned adapter the first encodes variable identity, pressure level, and field type and combines them with spatial features. Then, shared rank-16 projec?tions are modulated by metadata-dependent gates to produce field?specific features, while masked set fusion aggregates the available fields into a fixed-width representation. Subsequently, a multiscale U-Transformer processes the fused atmospheric features, while an identity-aware query decoder produces forecasts for the requested fields. Finally, FlexCast learns a standardized six-hour increment and applies it recursively to generate forecasts at longer lead times. Experiments on the 2020 ERA5 test set demonstrate that FlexCast operates across varying field configurations. Compatible cross-field context is associated with lower forecast errors, whereas mismatched context increases them.</span> <span class="abstract-toggle" data-id="2610.05296">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.05296v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.05296v1) · [:material-content-copy: BibTeX](../../bibtex/2610.05296.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=transformers" data-tag="transformers">Transformers</a>
    { .paper-tags }

-   #### S$^3$N: A Spherical Spiral Scanning Network for Weather Forecasting { #2610.04338 }

    *Fan Yan, Chen Hui, Weisi Lin, Haiqi Zhu, Feng Jiang, Sun-Yuan Kung, Wei Zhang* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.04338">Machine learning-based weather prediction (MLWP) has achieved strong performance in global weather forecasting. Recent Hierarchical Equal Area isoLatitude Pixelation (HEALPix)-based methods use the...</span><span class="abstract-full" id="full-2610.04338" hidden>Machine learning-based weather prediction (MLWP) has achieved strong performance in global weather forecasting. Recent Hierarchical Equal Area isoLatitude Pixelation (HEALPix)-based methods use the HEALPix (HP) grid to avoid area distortion near the poles of conventional latitude-longitude (LL) grids. However, existing HP-based approaches often use pointwise mapping methods and process HP pixels within separate base faces or local windows. Consequently, the mapping may introduce reconstruction errors and cross-face communication depends on handcrafted boundary handling or shifted windows. We propose the Spherical Spiral Scanning Network (S$^3$N) to address both limitations. First, L2Proj provides a bidirectional method for mapping atmospheric fields between the LL and HP grids through an $L^2$ projection of their continuous finite-element representations. Second, the Attention-Guided Quad-Spiral State-Space Scanning (AQSS) block uses cross-latitude attention to guide selective state-space updates along four global pole-to-pole spiral paths. This design enables continuous information propagation across HP base-face boundaries without additional boundary-processing mechanisms. Experiments show that S$^3$N achieves better results at 4-, 7-, and 10-day lead times, and exhibits slower error growth in long-range forecasting.</span> <span class="abstract-toggle" data-id="2610.04338">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.04338v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.04338v1) · [:material-content-copy: BibTeX](../../bibtex/2610.04338.bib){ .bibtex-link }
    { .paper-links }

-   #### Global Evaluation of AI and NWP Precipitation Forecasts During Atmospheric River Events { #2610.03758 }

    *Marina Vicens-Miquel, Taylor Mandelbaum, Amy McGovern, Aaron J. Hill, Daniel Rothenberg* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.03758">Atmospheric rivers (ARs) produce many of the world's most extreme precipitation events and hydrometeorological hazards. Although artificial intelligence weather prediction (AIWP) models have...</span><span class="abstract-full" id="full-2610.03758" hidden>Atmospheric rivers (ARs) produce many of the world's most extreme precipitation events and hydrometeorological hazards. Although artificial intelligence weather prediction (AIWP) models have demonstrated skill comparable to or exceeding numerical weather prediction (NWP) systems for large-scale atmospheric variables, their ability to forecast AR-related precipitation remains insufficiently characterized globally. Here, we evaluate 24-hour precipitation forecasts from the Global Forecast System (GFS), Global Ensemble Forecast System (GEFS), GraphCast, and Artificial Intelligence Forecasting System (AIFS) from Day 1 through Day 10 globally and across North America, Europe, and Australia and New Zealand. Using the Extreme Weather Bench framework, forecasts are evaluated against Integrated Multi-satellitE Retrievals for GPM (IMERG) observations using measures of precipitation magnitude, spatial structure, and localization. GraphCast and AIFS exhibit greater spatial skill than GFS and GEFS, particularly for heavy precipitation and at longer lead times, and better preserve the spatial organization of AR-related precipitation through Day 10. However, this improved spatial skill does not translate into accurate precipitation magnitudes. AIWP models tend to overpredict moderate-to-heavy accumulations while underpredicting the heaviest precipitation at longer lead times, whereas NWP systems develop pronounced dry biases. These results reveal distinct strengths and limitations of AIWP for high-impact precipitation forecasting and provide a reproducible benchmark.</span> <span class="abstract-toggle" data-id="2610.03758">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.03758v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.03758v1) · [:material-content-copy: BibTeX](../../bibtex/2610.03758.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=extremes" data-tag="extremes">Extremes</a> <a class="md-tag" href="/explore/?t=benchmarks-datasets" data-tag="benchmarks-datasets">Benchmarks & datasets</a> <a class="md-tag" href="/explore/?t=evaluation" data-tag="evaluation">Evaluation</a> <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a>
    { .paper-tags }

-   #### SDECast: Probabilistic Weather Forecasting in Continuous Time with Neural SDEs { #2610.03313 }

    *Maria Marchenko, Martin Andrae, Fredrik Lindsten, Christian A. Naesseth* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.03313">Existing machine learning weather forecasting models typically generate forecasts through autoregressive rollouts at a fixed temporal resolution. While highly efficient for long-range prediction,...</span><span class="abstract-full" id="full-2610.03313" hidden>Existing machine learning weather forecasting models typically generate forecasts through autoregressive rollouts at a fixed temporal resolution. While highly efficient for long-range prediction, this formulation can suffer from severe error accumulation when used with shorter time steps and does not explicitly encode the locality and temporal continuity of atmospheric dynamics. To address these limitations, we introduce **SDECast**, a Neural Stochastic Differential Equation (SDE) framework for continuous-time probabilistic weather forecasting. SDECast extends SDE Matching to learn stochastic dynamics directly in physical space, without requiring repeated SDE simulation during training. On a simulated geophysical flow, we show that SDECast recovers meaningful drift dynamics and faithfully reproduces the underlying continuous-time behavior. We then demonstrate its scalability to global weather forecasting at hourly resolution, where SDECast produces skillful probabilistic forecasts for lead times of up to five days.</span> <span class="abstract-toggle" data-id="2610.03313">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.03313v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.03313v1) · [:material-content-copy: BibTeX](../../bibtex/2610.03313.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a> <a class="md-tag" href="/explore/?t=hourly" data-tag="hourly">Hourly</a>
    { .paper-tags }

-   #### Post-Training Quantization of Autoregressive Weather Models { #2610.02511 }

    *Ananyo Bhattacharya, Swastik Bhattacharya, Christiane Jablonowski* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.02511">Advancements in high-resolution numerical weather prediction (NWP) and data assimilation (DA) have shaped the developments in deep learning (DL) architectures emulating atmospheric dynamics....</span><span class="abstract-full" id="full-2610.02511" hidden>Advancements in high-resolution numerical weather prediction (NWP) and data assimilation (DA) have shaped the developments in deep learning (DL) architectures emulating atmospheric dynamics. Emulators for weather forecasting exhibit forecast quality comparable to physics based models at forecast horizon scaling from few days to subseasonal time scales. The emulators are driven by hardware-accelerated matrix multiplication in autoregressive inferences, significantly reducing the computation time and resources required for NWP. Optimization of the matrix multiplication processes in GPU architectures provides opportunities to scale towards high-resolution domain, and offers implementation of out of the box solutions. Post-training quantization (PTQ) has been demonstrated across multiple DL architectures to accelerate and increase the number of computations in unit time while consuming less power, enabling applications on edge hardware. In this study, we investigate the effect of PTQ on pre-trained AI emulators for global-scale weather forecasting. We implement PTQ algorithms in Deep Learning Weather Prediction (DLWP) and FourCastNet (FCN) models as a proof of concept for geophysical fluid dynamics applications. We systematically investigate the effect of PTQ on emulator inferences over short-range forecast horizons. Evaluation of PTQ configurations using simulated quantization hints at qualitatively meaningful forecasts over short-time horizons. These results provide a first benchmark of PTQ for autoregressive weather emulators and a basis for quantization-based optimization of DL models for dynamical systems.</span> <span class="abstract-toggle" data-id="2610.02511">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.02511v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.02511v1) · [:material-content-copy: BibTeX](../../bibtex/2610.02511.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=subseasonal-to-seasonal" data-tag="subseasonal-to-seasonal">Subseasonal to seasonal</a> <a class="md-tag" href="/explore/?t=benchmarks-datasets" data-tag="benchmarks-datasets">Benchmarks & datasets</a>
    { .paper-tags }

-   #### Weather Jiu-Jitsu: Exploring the Feasibility of Control Paradigms in Weather Foundation Models { #2610.00792 }

    *Prakriti Biswas, Kobi Abayomi, Upmanu Lall* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.00792">Weather Jiu-Jitsu is a control paradigm for extreme climatological events, inspired by chaos theory. As a proposition, small, precise, targeted, and cost-inexpensive perturbations can redirect...</span><span class="abstract-full" id="full-2610.00792" hidden>Weather Jiu-Jitsu is a control paradigm for extreme climatological events, inspired by chaos theory. As a proposition, small, precise, targeted, and cost-inexpensive perturbations can redirect trajectories of a large dynamical system. This strategy has been demonstrated analytically in the Lorenz-63 system, where a naturally chaotic trajectory switching between two attractors can be confined to a single attractor, indefinitely, via arbitrarily small perturbations. This paper examines the feasibility of Microsoft's Aurora -- a 1.3 billion parameter global atmospheric model -- as a test bed for this strategy. This paper explores three questions: (1) Is Aurora a reliable enough simulation environment to serve as a meaningful testbed? (2) Are the perturbations required to redirect its trajectories small enough to be physically plausible? (3) Does Aurora's learned latent space (the parametric estimators on climatological attributes) yield any apparent, structured, and/or perhaps interpretable features that can convey a geo/atmospheric response to initial conditions? We find evidence consistent with all three: Aurora's modeled trajectories respond to perturbations beyond measurement drift, the perturbation magnitudes required are small relative to the model's own forecast uncertainty, and its latent representations exhibit directional structure that responds to Jiu-Jitsu-type interventions, even though that structure does not separate extreme from normal states outright. These results should be read as feasibility diagnostics rather than a demonstration of control: we do not implement or test an actual steering intervention on Aurora, and several of our findings, particularly around the model's latent-space geometry, are exploratory.</span> <span class="abstract-toggle" data-id="2610.00792">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.00792v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.00792v1) · [:material-content-copy: BibTeX](../../bibtex/2610.00792.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=foundation-models" data-tag="foundation-models">Foundation models</a> <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a>
    { .paper-tags }

-   #### STCFormer: Adaptive Spatio-Temporal Modeling with Dynamic Cluster Transformer for Station-based Weather Forecasting { #2610.00377 }

    *Rongwen Li, Haixin Xie, Mingyang Wang, Hongwu Liu, Kun Fang, Changjian Chen, Zhuo Tang, Kenli Li* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.00377">Station-based weather forecasting supports daily life and economic activity, yet accurate forecasts require modeling complex spatial dependencies among stations. Recent clustering-based selective...</span><span class="abstract-full" id="full-2610.00377" hidden>Station-based weather forecasting supports daily life and economic activity, yet accurate forecasts require modeling complex spatial dependencies among stations. Recent clustering-based selective modeling offers a promising alternative to dense inter-station interactions. However, a grouping shared across an observation window may obscure local changes in station relationships, while intra-cluster interactions alone may miss important global context. The theoretical advantages of selective interactions over dense connectivity also remain insufficiently understood. We therefore propose STCFormer, an adaptive spatio-temporal Transformer that dynamically groups stations according to their local evolution within each temporal patch. Its Cluster-Guided Attention Block combines fine-grained local attention within clusters and global attention over regional state summaries, allowing each station to access information beyond its own cluster. We further show that a derived Lipschitz upper bound for cluster-conditioned local attention is no larger than its fully connected counterpart, explaining a potential robustness benefit and motivating the design of InfoLoss. Experiments on three real-world weather datasets spanning eight temperature and wind forecasting tasks show that STCFormer achieves the lowest 24-hour mean squared error on all eight tasks and ranks first or second in 47 of 48 comparisons across metrics and forecasting horizons. Ablations and case studies further confirm the benefits of locally adaptive grouping and complementary local-global interactions. Our code can be obtained at https://github.com/hnu-vis/STCFormer.</span> <span class="abstract-toggle" data-id="2610.00377">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.00377v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.00377v1) · [:fontawesome-brands-github: Code](https://github.com/hnu-vis/STCFormer) · [:material-content-copy: BibTeX](../../bibtex/2610.00377.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=transformers" data-tag="transformers">Transformers</a> <a class="md-tag" href="/explore/?t=station-point" data-tag="station-point">Station / point</a> <a class="md-tag" href="/explore/?t=daily" data-tag="daily">Daily</a>
    { .paper-tags }

-   #### Butterfly Effect Confirmed in Global AI Weather Models: Evidence from Tropical Cyclone Forecasting { #2609.39379 }

    *Jeremy Cheuk-Hin Leung, Daosheng Xu, Weiye Yu, Shaojing Zhang, Xiaodong Zeng, Gaozhen Nie, Jie Feng et al.* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.39379">A paradox recently emerged in artificial intelligence (AI) weather prediction research. While some claim AI weather models cannot simulate atmospheric butterfly effect, this conflicts with AI models'...</span><span class="abstract-full" id="full-2609.39379" hidden>A paradox recently emerged in artificial intelligence (AI) weather prediction research. While some claim AI weather models cannot simulate atmospheric butterfly effect, this conflicts with AI models' limited predictability and advances in AI ensemble forecasting. This study demonstrates via counterexamples that the butterfly effect does exist in AI weather predictions. For Super Typhoon Khanun, AI predictions are constrained by a double-attractor system. Minor initial perturbations confined to two regions trigger state transitions between two local attractors, causing a 1006-km difference in the predicted storm position on Day 7. This behavior is consistent with numerical weather prediction models and observed in ~12% of tropical cyclones in the past 5 years. These findings verify AI's ability to capture atmospheric chaos and provide the physical basis for AI ensemble forecasting.</span> <span class="abstract-toggle" data-id="2609.39379">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.39379v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.39379v1) · [:material-content-copy: BibTeX](../../bibtex/2609.39379.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### Proper Scoring Rule-based Diffusion for Probabilistic Weather Forecasting { #2609.38632 }

    *Joonhyeong Park, Giung Nam, Hyungi Lee, Kyunghyun Cho, Byoungwoo Park, Juho Lee* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.38632">Recent probabilistic weather forecasters train stochastic predictors with the continuous ranked probability score (CRPS) to generate each ensemble member in a single forward pass. These models learn...</span><span class="abstract-full" id="full-2609.38632" hidden>Recent probabilistic weather forecasters train stochastic predictors with the continuous ranked probability score (CRPS) to generate each ensemble member in a single forward pass. These models learn the predictive distribution from the forecast context alone, which becomes difficult at longer forecast horizons where uncertainty is high. To learn the predictive distribution more effectively, we introduce auxiliary conditional denoising tasks that predict the same future state from the context and its corrupted version, which provides partial future information that can reduce prediction ambiguity. Building on distributional diffusion models, we learn the conditional distributions of these tasks with a single stochastic predictor by minimizing a proper scoring rule across noise levels. At inference, the predictor can still generate each ensemble member in a single forward pass at the fully corrupted endpoint. Standard CRPS training is recovered as the endpoint-only special case of our formulation, so our framework extends existing CRPS-based forecasters with only additional conditioning inputs. Controlled experiments show that the auxiliary tasks improve one-step forecasting across architectures, with larger gains at longer forecast horizons. The gains extend to high-dimensional global weather forecasting under both training from scratch and fine-tuning, along with improved calibration and potential benefits for generalization under distribution shift.</span> <span class="abstract-toggle" data-id="2609.38632">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.38632v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.38632v1) · [:material-content-copy: BibTeX](../../bibtex/2609.38632.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a>
    { .paper-tags }

-   #### An Input-Frugal Deep Learning Framework for Weather-Driven National Crop-Yield Forecasting: A Case Study of Brazilian Soybean { #2609.38447 }

    *Fernando Dupin da Cunha Mello, Prashant Kumar, Erick G. Sperandio Nascimento* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.38447">Reliable, timely crop-yield forecasts are essential for market stability and risk management, yet many approaches rely on costly or hard-to-scale inputs. We present a frugal, transferable, and...</span><span class="abstract-full" id="full-2609.38447" hidden>Reliable, timely crop-yield forecasts are essential for market stability and risk management, yet many approaches rely on costly or hard-to-scale inputs. We present a frugal, transferable, and architecture-agnostic deep learning framework that uses routine weather as the only time-varying input plus two lightweight static context inputs (crop year and an agro-environmental label) to capture long-run change and regional heterogeneity, while supporting multiple sequence encoders under identical data requirements. Using a 20-season Brazilian soybean case study (2001/02-2020/21) with leave-one-year-out cross-validation, we benchmark MLP, CNN, LSTM, CNN-LSTM, a Transformer encoder and the Mamba state-space model against linear ridge regression and a five-year moving-average "farmer" baseline. All deep learning variants outperform ridge, and all sequential encoders surpass the non-sequential MLP. The Transformer achieves the best national accuracy (RMSE 149 kg ha^-1; rRMSE 5.3%; R^2 = 0.784), reducing error by 47.6% relative to the farmer baseline. In-season forecasts improve monotonically from early- to late-season issuance, reaching approximately 50% lower error than the baseline at the latest forecast point. Ablations indicate that the agro-environmental label and spatial instance expansion (multiple grid-node weather sequences per municipality-year) contribute positively without increasing input complexity. SHAP diagnostics suggest crop year explains most of the long-run trajectory, whereas within-season weather and agro-environmental context primarily drive interannual deviations, with moisture/cloud and thermal-demand variables dominating. Overall, the framework is straightforward to deploy across other crops and geographic regions and is naturally compatible with operational weather forecasts for routine monitoring.</span> <span class="abstract-toggle" data-id="2609.38447">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.38447v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.38447v1) · [:material-content-copy: BibTeX](../../bibtex/2609.38447.bib){ .bibtex-link }
    { .paper-links }

-   #### A neural network-based Universal Thermal Climate Index for reliable global thermal-stress classification across extreme weather { #2609.35949 }

    *Bikem Pastine, Milan Klöwer, Tianning Tang, Sarah Wilson Kemsley, Louise Slater* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.35949">Extreme temperatures are the leading cause of climate-related mortality world-wide. Climate-health research and operational weather forecasting require accurate estimates of human thermal stress. The...</span><span class="abstract-full" id="full-2609.35949" hidden>Extreme temperatures are the leading cause of climate-related mortality world-wide. Climate-health research and operational weather forecasting require accurate estimates of human thermal stress. The Universal Thermal Climate Index (UTCI) is among the most sophisticated and widely used feels-like temperature metrics. However, its ubiquitous polynomial approximation does not generalize well to extreme weather conditions. Here, we introduce Neural-UTCI, a neural network that calculates UTCI with substantially higher accuracy across global conditions at a lower computational cost for operational use. Neural-UTCI reduces the polynomial approximation RMSE from 2.78°C to 0.36 °C, an 87% improvement, and lowers thermal stress misclassification rates from 5.3% to 1.7%, with consistent performance across resampling experiments. These differences affect thermal exposure metrics. For example, during the 2003 European heatwave summer in Rome, Italy, the number of very strong heat stress days increases from 15 to 35 days when using Neural-UTCI compared to operational products like ERA5-HEAT. Simultaneously, Neural-UTCI reliably classifies extreme cold stress conditions, allowing continuous global application. By improving UTCI accuracy, Neural-UTCI can strengthen climate-health risk assessments and public weather warning systems, especially as global warming increases the incidence of extreme events.</span> <span class="abstract-toggle" data-id="2609.35949">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.35949v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.35949v1) · [:material-content-copy: BibTeX](../../bibtex/2609.35949.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=extremes" data-tag="extremes">Extremes</a>
    { .paper-tags }

-   #### Suitable Measures for the Potential Operational Utility of AI NWP Rainfall Forecasts Over Africa { #2609.31775 }

    *Shruti Nath, Docko Sow, Koomi Toussaint Amoussouvi, Fenwick Cooper, Josiah Kiarie Kimani et al.* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.31775">Artificial intelligence (AI)-based weather prediction is approaching the skill of physical numerical weather prediction (NWP) systems at a fraction of the computational cost. This is particularly...</span><span class="abstract-full" id="full-2609.31775" hidden>Artificial intelligence (AI)-based weather prediction is approaching the skill of physical numerical weather prediction (NWP) systems at a fraction of the computational cost. This is particularly promising for Africa, where rainfall extremes are intensifying and many forecasting centres lack the infrastructure to run physical models at extended lead times. We present a calibrated comparison of GraphCast, GenCast and the Functional Generative Network (FGN) against the physical NWP model IFS for rainfall prediction across Africa. Deterministic and probabilistic forecasts are postprocessed using Isotonic Distributional Regression and evaluated with the Continuous Ranked Probability Score against IMERG, RFEv2 and CHIRPS across seasons, wet and dry regimes, elevation zones and lead times. All models retain skill beyond climatology across most seasons and at extended lead times. AI models generally outperform IFS in wet regions, whereas IFS performs better in dry, high-elevation areas, where its finer resolution better represents orographic controls on rainfall. Across observational datasets and seasons, AI models achieve a median improvement of approximately 5% over IFS. GraphCast achieves calibrated skill comparable to the ensemble-based FGN, although FGN provides greater significant skill at longer lead times. These results highlight the potential of calibrated AI weather prediction to provide accessible and computationally efficient rainfall forecasts across Africa, while demonstrating the continuing importance of spatial resolution, ensemble design and regional characteristics.</span> <span class="abstract-toggle" data-id="2609.31775">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.31775v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.31775v1) · [:material-content-copy: BibTeX](../../bibtex/2609.31775.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=efficiency" data-tag="efficiency">Efficiency</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a>
    { .paper-tags }

-   #### A dataset of one-dimensional idealized probabilistic fields { #2609.25720 }

    *Gregor Skok, Romain Pic* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.25720">Verification of probabilistic weather forecasts remains a crucial aspect of numerical weather prediction, as new AI-based models become more widely used alongside the more traditional physics-based...</span><span class="abstract-full" id="full-2609.25720" hidden>Verification of probabilistic weather forecasts remains a crucial aspect of numerical weather prediction, as new AI-based models become more widely used alongside the more traditional physics-based ensemble forecasting systems that continue to be developed and improved. We present a first-of-its-kind idealized probabilistic dataset composed of one-dimensional cases aimed at analyzing the behavior and properties of verification methods for probabilistic forecasts and comparing their behavior. It covers a wide range of probabilistic cases, such as constant, localized events, gradients, fronts, noisy, bimodal, and limiting cases. Moreover, the code associated with the dataset provides great flexibility for customizing the experiments it covers. The dataset represents the first building block of the more extensive comparison dataset of the Bridging The Gap project, which aims to facilitate the development and comparison of spatial verification methods for probabilistic forecasts.</span> <span class="abstract-toggle" data-id="2609.25720">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.25720v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.25720v1) · [:material-content-copy: BibTeX](../../bibtex/2609.25720.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=benchmarks-datasets" data-tag="benchmarks-datasets">Benchmarks & datasets</a>
    { .paper-tags }

-   #### FAST-ML: A Hybrid Physics-Machine Learning Framework for Tropical Cyclone Intensity Forecasting { #2609.25505 }

    *Shijie Xiao, Jonathan Lin, Thomas Ehrmann, Ali Sarhadi* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.25505">Rapid intensification (RI) remains one of the most consequential and difficult aspects of tropical cyclone (TC) forecasting. Although full-physics numerical weather prediction models can represent...</span><span class="abstract-full" id="full-2609.25505" hidden>Rapid intensification (RI) remains one of the most consequential and difficult aspects of tropical cyclone (TC) forecasting. Although full-physics numerical weather prediction models can represent the processes governing RI, resolving storm-environment interactions remains computationally expensive, while purely data-driven approaches often lack physical interpretability. We present FAST-ML, a hybrid framework that bridges data-driven efficiency with physical constraints. A physically informed dual-stream neural parameterization ingests 3D ERA5 fields to diagnose ventilation controls---environmental wind shear and mid-level entropy deficit. By optimizing these parameters end-to-end through a differentiable FAST intensity model, this architecture establishes a robust new paradigm for observation-driven parameter optimization, ensuring storm evolution remains strictly governed by thermodynamic principles. By better capturing the storm's continuous intensity evolution, FAST-ML improves upon its physical baseline, reducing ensemble CRPS across forecast lead times, with a reduction of approximately 31% at 60 h and nearly halving the RI false alarm ratio without sacrificing detection skill. In a 100-member ensemble configuration, FAST-ML produces intensity forecasts comparable to FNV3 for selected storms under the evaluated input configurations. Furthermore, zero-shot tests on selected Eastern Pacific storms provide encouraging evidence of cross-basin transferability. FAST-ML provides a modular intensity forecasting framework that can be coupled with externally supplied storm tracks and environmental fields. It demonstrates that observation-driven parameter learning within physically constrained dynamics simultaneously enhances accuracy, interpretability, and computational efficiency.</span> <span class="abstract-toggle" data-id="2609.25505">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.25505v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.25505v1) · [:material-content-copy: BibTeX](../../bibtex/2609.25505.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=physics-ml-hybrid" data-tag="physics-ml-hybrid">Physics–ML hybrid</a> <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a> <a class="md-tag" href="/explore/?t=interpretability" data-tag="interpretability">Interpretability</a> <a class="md-tag" href="/explore/?t=efficiency" data-tag="efficiency">Efficiency</a>
    { .paper-tags }

-   #### Spatial Aggregation of ROC and Precision-Recall Curves { #2609.19517 }

    *Romain Pic, Zhongwei Zhang, Sebastian Engelke, Johanna Ziegel* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.19517">Receiver Operating Characteristic (ROC) and Precision-Recall (PR) curves are widely used to assess the discrimination ability of forecasts for binary events, such as threshold exceedances or warnings...</span><span class="abstract-full" id="full-2609.19517" hidden>Receiver Operating Characteristic (ROC) and Precision-Recall (PR) curves are widely used to assess the discrimination ability of forecasts for binary events, such as threshold exceedances or warnings of extreme events. In weather forecasting, forecasts are provided as spatial fields, yielding location-wise ROC and PR curves that are often aggregated to facilitate comparison. However, the effect of the aggregation strategy on performance assessment remains poorly understood.   We investigate how different aggregation strategies for ROC and PR curves affect the assessment of discrimination ability. In particular, we identify conditions under which aggregation strategies satisfy two desirable properties for fair comparison: preservation of dominance between forecasts and preservation of concavity or achievability of the curves. We obtain sufficient conditions and propose two strategies satisfying them. They are compared with existing strategies from the literature, and we analyze their properties and highlight potential pitfalls that may lead to misleading interpretations. Based on these findings, we provide practical guidelines for the interpretation of aggregated ROC and PR curves. The proposed framework is illustrated with AI-based global weather forecasts, showing how different aggregation strategies can yield different rankings of competing forecasts.</span> <span class="abstract-toggle" data-id="2609.19517">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.19517v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.19517v1) · [:fontawesome-brands-github: Code](https://github.com/pic-romain/spatial-agg-roc-pr) · [:material-content-copy: BibTeX](../../bibtex/2609.19517.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a>
    { .paper-tags }

-   #### Butterfly Effect and the Kinetic Energy Cascade in Probabilistic Machine Learning Weather Prediction Models { #2609.18489 }

    *Jiakai Chen, Joel Oskarsson, Simon Driscoll, Sebastian Schemm* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.18489">This study analyses kinetic energy (KE) spectra, difference kinetic energy (DKE) spectra, and signatures of KE transfer across spatial scales in four state-of-the-art probabilistic machine learning...</span><span class="abstract-full" id="full-2609.18489" hidden>This study analyses kinetic energy (KE) spectra, difference kinetic energy (DKE) spectra, and signatures of KE transfer across spatial scales in four state-of-the-art probabilistic machine learning weather prediction (MLWP) models: NeuralGCM-ENS, FourCastNet 3, AIFS-ENS, and GenCast. Results are compared with those from the physics-based numerical weather prediction model IFS-ENS. While NeuralGCM-ENS successfully reproduces the expected upscale transfer of KE, noise injection at its encoder stage underestimates mesoscale KE. Conversely, AIFS-ENS, GenCast, and FourCastNet 3 produce realistic KE spectral magnitudes but do not capture the expected upscale transfer of KE. In particular, AIFS-ENS and GenCast, which employ spatially uncorrelated stochastic perturbations, exhibit enhanced accumulation of KE at high wavenumbers. All examined models exhibit upscale error growth, reflected by the progressive shift of the DKE spectral peak toward larger wavelengths over time. However, the MLWP models struggle to reproduce the rapid initial growth of ensemble spread at small spatial scales associated with the butterfly effect. The results show that MLWP models can misrepresent the known scale transfer of kinetic energy despite producing skilful weather forecasts.</span> <span class="abstract-toggle" data-id="2609.18489">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.18489v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.18489v1) · [:material-content-copy: BibTeX](../../bibtex/2609.18489.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=physics-ml-hybrid" data-tag="physics-ml-hybrid">Physics–ML hybrid</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### Every Fixed Metric Has a Blind Spot: A Learned Atmospheric Critic for Scoring Forecast Realism { #2609.18381 }

    *Younes Elberkennou, Dmitri Demler, Thierry Meier, Luca Rispoli, Fanny Lehmann, Joel Oskarsson* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.18381">Despite their high accuracy on point-wise metrics, machine learning weather forecasting models can exhibit different failure modes such as blurring, periodic irregularities, and other unphysical...</span><span class="abstract-full" id="full-2609.18381" hidden>Despite their high accuracy on point-wise metrics, machine learning weather forecasting models can exhibit different failure modes such as blurring, periodic irregularities, and other unphysical spatial artifacts. This has motivated a variety of metrics to detect known failure cases. Existing metrics fix a representation or transformation in advance, and that choice limits the artifacts they can detect. We propose to train a discriminator for separating reference data from the model's output, and using its output logit to obtain a divergence-like realism score. The discriminator learns whatever separates the model's fields from real weather, adapting to whichever failure mode that model exhibits. We compare our learned atmospheric critic to existing metrics using various synthetic corruptions applied to ERA5 reanalysis data. Our method successfully identifies the corruptions and ranks their severity, while existing metrics fail on at least one corruption. Additionally, we evaluate forecasts from real weather models, and find that the realism score degrades with longer lead times and the metric generally assigns higher realism to numerical models than to machine learning models.</span> <span class="abstract-toggle" data-id="2609.18381">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.18381v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.18381v1) · [:material-content-copy: BibTeX](../../bibtex/2609.18381.bib){ .bibtex-link }
    { .paper-links }

-   #### Aries: A Proprietary Medium-Range Weather Prediction Model for the Energy Industry { #2609.13292 }

    *Lukas Hedegaard Morsing, Arian Bakhtiarnia, Jonas Lynge Olesen, Tómas Bragi Björnsson Leth et al.* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.13292">Medium-range weather forecasting underpins operational and planning decisions across the energy industry. Developing competitive weather models was once the domain of national meteorological centers,...</span><span class="abstract-full" id="full-2609.13292" hidden>Medium-range weather forecasting underpins operational and planning decisions across the energy industry. Developing competitive weather models was once the domain of national meteorological centers, but recent advances in machine-learned weather prediction (MLWP) have opened the field to industry. We present Aries, a SwinTransformer-based MLWP model developed at InCommodities. Aries is trained on ERA5 reanalysis data at 0.25°{} resolution, predicting 74 prognostic and 11 diagnostic atmospheric variables. We evaluate the model on 2025 ECMWF Analysis initializations, ensuring a recent and strictly out-of-sample test period for all models compared. On 10-metre wind speed, Aries outperforms both ECMWF HRES and AIFS in terms of RMSE for lead times up to four days, while on 2-metre temperature it achieves RMSE on par with AIFS operational. These results demonstrate that proprietary development of competitive weather models is technically viable, supporting a broader set of forecasts available for operational and planning applications in the energy industry.</span> <span class="abstract-toggle" data-id="2609.13292">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.13292v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.13292v1) · [:material-content-copy: BibTeX](../../bibtex/2609.13292.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=transformers" data-tag="transformers">Transformers</a>
    { .paper-tags }

-   #### Optimizing Geoengineering Interventions Using Differentiable Climate Models { #2609.12528 }

    *Pulkit Dubey, Dorian S. Abbot, Ashesh Chattopadhyay* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.12528">The deployment of a geoengineering program to cool Earth's climate may be imminent. It is crucial that tools be developed to ensure that such a program would achieve its objectives while minimizing...</span><span class="abstract-full" id="full-2609.12528" hidden>The deployment of a geoengineering program to cool Earth's climate may be imminent. It is crucial that tools be developed to ensure that such a program would achieve its objectives while minimizing disruption. Here we exploit recently developed differentiable atmospheric models to demonstrate a novel geoengineering control strategy. In the differentiable primitive-equation atmospheric model JAX-GCM we impose a uniform $+4$\,K ocean warming and ask what pattern of sea-surface temperature cooling -- in five ocean-masked zonal bands of prescribed SST forcings whose amplitudes are free -- returns land near-surface air temperature closest to the model's own unwarmed climatology. This idealized set-up represents a cooling pattern that could be delivered physically either by marine cloud brightening or stratospheric aerosol injection. Gradients through chaotic dynamics decorrelate from the true sensitivity beyond the Lyapunov horizon, so we optimize greedily over segments of 8 to 14 days, following receding-horizon control. The learned strategy removes $92.3 \pm 0.4\%$ of the realized land warming across a ten-member ensemble of two-year rollouts, and a three-year run sustains it. If we use the spatial pattern of land temperature as the optimization objective, the distributions of precipitation, evaporation, and specific humidity over land are restored as well, even though they are not included in the objective function. The learned strategy from JAX-GCM replayed in the AI emulators LUCIE and NeuralGCM without re-optimization is successful, suggesting robustness. These promising results demonstrate a strategy for designing optimal climate interventions that can be applied broadly for geoengineering scenarios under consideration.</span> <span class="abstract-toggle" data-id="2609.12528">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.12528v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.12528v1) · [:material-content-copy: BibTeX](../../bibtex/2609.12528.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=physics-ml-hybrid" data-tag="physics-ml-hybrid">Physics–ML hybrid</a>
    { .paper-tags }

-   #### WIND-Bench: A Benchmark Dataset for In-Situ Near-Surface Wind Speed Observations Across the Conterminous United States { #2609.12228 }

    *Kyla Bazlen, Grant Buster, Brandon Benton, Lauren North, Ansley Baring, David D. Turner et al.* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.12228">Accurate wind forecasts are essential for operational decision-making and public safety, yet forecasts tend to miss near-surface high wind speeds in complex terrain. In response, advances in machine...</span><span class="abstract-full" id="full-2609.12228" hidden>Accurate wind forecasts are essential for operational decision-making and public safety, yet forecasts tend to miss near-surface high wind speeds in complex terrain. In response, advances in machine learning (ML) weather prediction methods have demonstrated the ability to improve forecast skill beyond traditional numerical weather prediction (NWP) models. However, the absence of a benchmark dataset to evaluate NWP and ML models with sufficient, quality-controlled wind speed observations in complex terrain poses challenges to the development and intercomparison of high-quality surface wind forecasts across the Conterminous United States (CONUS). We develop the Wind IN-situ Data Benchmark (WIND-Bench), a benchmark dataset from in-situ observations in the Meteorological Assimilation Data Ingest System (MADIS) observational network. WIND-Bench integrates multiple sensor networks with quality control that distinguishes sensor failures from high-wind conditions, using a framework that validates observations against forecasts from the National Oceanic and Atmospheric Administration (NOAA) High-Resolution Rapid Refresh (HRRR) model. WIND-Bench provides a standardized benchmark for evaluating ML and NWP models and for quantifying forecast skill, accelerating the development, evaluation, and operational deployment of skilled near-surface wind forecasts.</span> <span class="abstract-toggle" data-id="2609.12228">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.12228v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.12228v1) · [:material-content-copy: BibTeX](../../bibtex/2609.12228.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=benchmarks-datasets" data-tag="benchmarks-datasets">Benchmarks & datasets</a> <a class="md-tag" href="/explore/?t=evaluation" data-tag="evaluation">Evaluation</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a> <a class="md-tag" href="/explore/?t=station-point" data-tag="station-point">Station / point</a>
    { .paper-tags }

-   #### Stochastically Perturbed Weights: Ensembles from Deterministic Machine-Learning Weather Models { #2609.08412 }

    *Simon Adamov, Oliver Fuhrer, Reto Knutti, Sebastian Schemm* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.08412">Machine-learning weather models (MLWMs) now match or outperform operational numerical weather prediction (NWP) at global medium-range forecasting, at far lower inference cost. Many deployed MLWMs are...</span><span class="abstract-full" id="full-2609.08412" hidden>Machine-learning weather models (MLWMs) now match or outperform operational numerical weather prediction (NWP) at global medium-range forecasting, at far lower inference cost. Many deployed MLWMs are deterministic, producing a single forecast with no estimate of its own uncertainty, whereas a growing family of trained-probabilistic models generate calibrated ensembles directly, at the price of a dedicated training run. We ask instead how much uncertainty can be extracted from a deterministic checkpoint that already exists, without retraining it. Where physical ensembles represent model uncertainty by stochastically perturbing parametrisation tendencies, we perturb the network's raw weight tensors at inference time, a scheme we call stochastically perturbed weights (SPW). We also ask whether it works, where and on which scales to inject the noise, and where it fails. A three-phase ablation across four deterministic backbones, Aurora, GraphCast, SFNO, and AIFS, selects one production baseline per model, benchmarked against the trained-probabilistic AIFS-ENS, FourCastNet 3 and Atlas as well as the operational ECMWF ensemble (IFS-ENS) over 112 initialisation times. At a 240 h (10-day) lead time the SPW ensembles reach continuous ranked probability skill scores (CRPSS) between 0.04 and 0.13 below the best trained-probabilistic baseline, at zero marginal training cost. No injection site works across models: the productive tensor group is architecture-specific, so SPW is at present a tuning procedure rather than a plug-and-play recipe. Its main failure mode is a coherent whole-field offset that overdisperses the domain mean, and restricting the noise to coarse scales or perturbing the initial conditions each repair part of it.</span> <span class="abstract-toggle" data-id="2609.08412">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.08412v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.08412v1) · [:fontawesome-brands-github: Code](https://github.com/MeteoSwiss/ai-models-ensembles) · [:material-content-copy: BibTeX](../../bibtex/2609.08412.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### WeatherNext 3: Increasing resolution and performance of global weather models with raw observations { #2609.03582 }

    *Stephan Rasp, Boris Babenko, Dominic Masters, Andrew El-Kadi, Samier Merchant, Guy Shalev et al.* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.03582">State-of-the-art AI weather models have shown impressive medium-range forecast skill and computational efficiency, but suffer two key shortcomings: their forecasts have lower spatial and temporal...</span><span class="abstract-full" id="full-2609.03582" hidden>State-of-the-art AI weather models have shown impressive medium-range forecast skill and computational efficiency, but suffer two key shortcomings: their forecasts have lower spatial and temporal resolution than the best physics-based models and they are exclusively initialized with and trained on analysis data. As a result, they cannot directly make use of observations, and any biases in the analysis are inherited by the forecast. WeatherNext 3 addresses these shortcomings and establishes a new state-of-the-art for probabilistic medium-range forecasting skill. First, WeatherNext 3 generates new forecasts every hour (rather than every 6 hours like traditional global models) by ingesting low-latency geostationary satellite data. Second, WeatherNext 3's temporal and spatial resolution are on par with physics-based global models, with hourly time steps and 0.1 degree resolution for single-level variables, including solar radiation and cloud cover. Third, WeatherNext 3 moves beyond traditional analysis variables by learning to predict satellite-derived precipitation estimates, as well as tropical cyclone and station observations. Modelling sparse station data allows WeatherNext 3 to make 2m temperature and dewpoint predictions at any location and time, conditioned on local geographical features, with substantially lower error than competing global models, even when evaluated against unseen stations. Together, WeatherNext 3's capabilities move operational AI-based weather forecasting beyond emulating the traditionally distinct stages of data assimilation, forecasting and post-processing, which helps to further push the frontier of performance and granularity for global weather prediction.</span> <span class="abstract-toggle" data-id="2609.03582">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.03582v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.03582v1) · [:material-content-copy: BibTeX](../../bibtex/2609.03582.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a> <a class="md-tag" href="/explore/?t=hourly" data-tag="hourly">Hourly</a>
    { .paper-tags }

-   #### Improving precipitation forecasts in an AI weather model using observational data { #2609.03210 }

    *Julian F. Schmitt, Bertrand Delorme, Robert C. King, Yashica Patodia, Tapio Schneider et al.* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.03210">Artificial intelligence weather prediction (AIWP) systems now surpass state-of-the-art physical models for medium-range weather forecasting. Current global AIWP models are trained almost exclusively...</span><span class="abstract-full" id="full-2609.03210" hidden>Artificial intelligence weather prediction (AIWP) systems now surpass state-of-the-art physical models for medium-range weather forecasting. Current global AIWP models are trained almost exclusively using one reanalysis dataset, ERA5, but it has known biases, particularly for precipitation. Here we fine-tune a graph-transformer architecture with IMERG precipitation data at 0.25° resolution. The resulting model improves medium-range continuous ranked probability scores by up to 19%, while also demonstrating superior skill for tropical storms and drizzle events. Our model exceeds the Brier skill score of state-of-the-art operational models on extreme rainfall prediction by 57% globally; however, a physics-based operational model remains more reliable for the heaviest precipitation events. Our results demonstrate that incorporating observations-based precipitation data directly into training can substantially improve precipitation forecasts.</span> <span class="abstract-toggle" data-id="2609.03210">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.03210v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.03210v1) · [:material-content-copy: BibTeX](../../bibtex/2609.03210.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=transformers" data-tag="transformers">Transformers</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=quarter-degree" data-tag="quarter-degree">0.25°</a>
    { .paper-tags }

-   #### TC-Next: Zero-Shot Multimodal Cyclone Forecasting { #2609.02085 }

    *Zhe Wang, Sijie Chen, Yiming Luo, Daehyun Kim, Chien-Yi Chang* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.02085">We present TropicalCycloneNext (TC-Next), a multimodal deep learning model that forecasts tropical cyclone track and intensity at $6$-$24$ h leads by leveraging a foundation model's forecast fields...</span><span class="abstract-full" id="full-2609.02085" hidden>We present TropicalCycloneNext (TC-Next), a multimodal deep learning model that forecasts tropical cyclone track and intensity at $6$-$24$ h leads by leveraging a foundation model's forecast fields of atmospheric kinematic and thermodynamic fields and GridSat infrared satellite imagery. Trained only on GraphCast forecasts over the Western Pacific (WP), yet reliant only on generic atmospheric variables, TC-Next on GraphCast lowers track error by $15$-$44\%$ and intensity error by a factor of $3$-$6$ relative to a conventional, rule-based tracker, TempestExtremes; applied without retraining to the forecast fields of Pangu-Weather and IFS HRES, it stays ahead of TempestExtremes on both. Applied zero-shot to the generic weather fields of WeatherNext Cyclones on the 2025 WP season, TC-Next attains lower intensity error at every lead time, and lower or comparable track error, compared to that model's specialized direct tracker in a deterministic comparison. Our ablation studies show that our multimodal model is able to utilize the additional modality to improve performance in tracking errors at every lead time and in intensity prediction at longer lead times.</span> <span class="abstract-toggle" data-id="2609.02085">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.02085v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.02085v1) · [:material-content-copy: BibTeX](../../bibtex/2609.02085.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=foundation-models" data-tag="foundation-models">Foundation models</a> <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a>
    { .paper-tags }

</div>

<nav class="pager" markdown="span">**1** [2](2.md) [3](3.md) [4](4.md) [5](5.md) [6](6.md) [7](7.md) [8](8.md) [9](9.md) [10](10.md) [11](11.md) [12](12.md) [13](13.md) [Older :material-arrow-right:](2.md){ .pager-step }</nav>

