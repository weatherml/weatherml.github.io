---
hide:
  - navigation
  - toc
title: weatherml
---

A collection of papers on AI for weather forecasting, climate modelling and atmospheric science.

<p class="page-meta" markdown="span">1502 papers · updated 2026-10-02 · <a href="feed.xml">:material-rss: RSS</a> · <a href="all_papers.bib" download>:material-download: BibTeX</a> · <a href="https://github.com/weatherml/weatherml.github.io/issues/new?template=suggest-paper.yml">:material-plus: Suggest a paper</a></p>

## Browse by Topic

<div class="grid cards topics" markdown>

-   [Global Models](papers/global-models/index.md) <span class="topic-count">354</span>
-   [Regional Models](papers/regional-models/index.md) <span class="topic-count">63</span>
-   [Nowcasting](papers/nowcasting/index.md) <span class="topic-count">101</span>
-   [Downscaling](papers/downscaling/index.md) <span class="topic-count">98</span>
-   [Post-processing](papers/post-processing/index.md) <span class="topic-count">25</span>
-   [Data Assimilation](papers/data-assimilation/index.md) <span class="topic-count">75</span>
-   [Climate Modeling](papers/climate-modeling/index.md) <span class="topic-count">292</span>
-   [Hydrology](papers/hydrology/index.md) <span class="topic-count">37</span>
-   [Ocean & Sea Ice](papers/ocean-sea-ice/index.md) <span class="topic-count">85</span>
-   [Air Quality & Composition](papers/air-quality-composition/index.md) <span class="topic-count">71</span>
-   [Remote Sensing](papers/remote-sensing/index.md) <span class="topic-count">99</span>
-   [Other](papers/other/index.md) <span class="topic-count">202</span>

</div>

## Recent Additions

<div class="grid cards" markdown data-search-exclude>

-   #### AI Emulation of Stochastic Sudden Stratospheric Warming with Interpretable Latent Structure { #2610.02069 }

    *C. Daniel Boscu, Daniel Hernandez, Fabio Alvarez Ventura, Justin Finkel, Ashesh Chattopadhyay et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.02069">Rare weather regime transitions pose a challenge for data-driven modeling due to class imbalance. In this study, we develop a probabilistic deep learning emulator for a prototypical system with...</span><span class="abstract-full" id="full-2610.02069" hidden>Rare weather regime transitions pose a challenge for data-driven modeling due to class imbalance. In this study, we develop a probabilistic deep learning emulator for a prototypical system with regime transitions, the stochastic Holton--Mass model of stratospheric variability, and analyze the structure of its learned latent space. The Holton--Mass model exhibits two metastable regimes, a strong and a weak polar vortex, maintained by nonlinear wave--mean flow interactions, with weak stochastic forcing intermittently triggering rare transitions between these regimes that qualitatively represent SSW events. We employ a ResNet-inspired Conditional Variational Autoencoder with six-layer encoder and decoder layers and explicit current-state conditioning to model the distribution of the system's state at the next time step (one day). The emulator accurately reproduces short-term dynamics, steady-state probability distributions, regime persistence statistics, rare transition rates, the transition committor function, and the transition expected lead time of the physical model. Beyond emulation fidelity, we interrogate the learned latent representation to understand how the model internalizes the underlying metastable structure of the dynamics. Principal Component Analysis of the 32-dimensional latent space reveals a clear and unsupervised separation into four physically interpretable clusters corresponding to strong versus weak vortex regimes and stable versus transition-prone configurations. Such emergent regime separation in latent space is hard to identify for deep generative models applied to high-dimensional stochastic systems. Our results show that carefully designed probabilistic emulators can uncover physically meaningful manifolds governing extreme-event dynamics, potentially aiding the development of improved operational advanced warning systems.</span> <span class="abstract-toggle" data-id="2610.02069">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.02069v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.02069v1) · [:material-content-copy: BibTeX](bibtex/2610.02069.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=interpretability" data-tag="interpretability">Interpretability</a>
    { .paper-tags }

-   #### Unsupervised Domain Adaptation for Enhanced Radiometer Image Precipitation Estimation using Conditional Flow Matching { #2610.01890 }

    *Victor Enescu, Assaad Zeghina, Matthieu Meignin, Nicolas Viltard, Cécile Mallet* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.01890">Deep generative networks have recently achieved unprecedented performance in precise image and video editing using sophisticated textual prompts. However, the effectiveness of such models heavily...</span><span class="abstract-full" id="full-2610.01890" hidden>Deep generative networks have recently achieved unprecedented performance in precise image and video editing using sophisticated textual prompts. However, the effectiveness of such models heavily depends on access to very large supervised and annotated image datasets, which can be very difficult to obtain. This is particularly true for satellite instruments, which very rarely overlap with labelled data, and suffer from domain shifts in the rare occasions they do. In this paper, we investigate the potential of flow matching models for unsupervised domain adaptation of satellite radiometer images. Our main contribution is a novel unsupervised method that achieves precise domain alignment by leveraging parts of the deterministic ordinary differential equations in flow matching models, conditioned on different satellite instruments. A key strength of our approach is its ability to preserve essential information while adapting across any domains since the perturbations are in theory bijective. Extensive experiments conducted on the GPM-Core constellation show the benefit of our conditional domain adaptation, particularly in improving rain precipitation estimation from radiometer imagery.</span> <span class="abstract-toggle" data-id="2610.01890">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.01890v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.01890v1) · [:material-content-copy: BibTeX](bibtex/2610.01890.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a>
    { .paper-tags }

-   #### Varda-single-1.0: deterministic data-driven weather forecasting at 1 km resolution over Switzerland's complex topography { #2610.01835 }

    *Alberto Pennino, Francesco Zanetta, Michele Cattaneo, Claire Merker, Radi Radev, Jonas Bhend et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.01835">We present Varda-single-1.0, a medium-range data-driven weather prediction system built for the Alpine domain. It provides hourly deterministic regional forecasts on a mesh of 1 km resolution and...</span><span class="abstract-full" id="full-2610.01835" hidden>We present Varda-single-1.0, a medium-range data-driven weather prediction system built for the Alpine domain. It provides hourly deterministic regional forecasts on a mesh of 1 km resolution and global forecasts on a 31 km mesh. The system comprises two independently trained stretched-grid Graph Transformer models with encoder-processor-decoder architecture, developed in the Anemoi framework: a 6-hourly autoregressive forecaster and a temporal downscaler reconstructing hourly forecasts between the forecaster's steps. Its training curriculum includes pre-training on ERA5 reanalysis data, followed by training on a 20-year kilometre-scale regional reanalysis, and finally fine-tuning on operational kilometre-scale analyses. Verified over one year against operational analyses and surface station observations, Varda-single is competitive with or improves on MeteoSwiss' operational numerical weather prediction baselines for most headline scores and variables. It broadly matches the skill of the high-resolution 1 km ICON-CH1-EPS control at lead times up to +33 h and generally outperforms the 2 km ICON-CH2-EPS control at lead times up to +120 h. Despite competitive aggregate scores, Varda-single underestimates some local wind maxima and produces overly smooth convective precipitation fields, consistent with the smoothing associated with squared-error training. To gain insight into the model's behaviour, we investigate three case studies beyond the aggregated headline scores, and find particular weaknesses in Varda-single's representation of local winds over complex terrain. Varda-single represents an important step in the development of high-resolution ML forecasting over complex terrain, in complementing the operational regional numerical weather prediction models of MeteoSwiss with data-driven models and in providing a pretrained model for researchers and user-specific applications.</span> <span class="abstract-toggle" data-id="2610.01835">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.01835v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.01835v1) · [:material-content-copy: BibTeX](bibtex/2610.01835.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=transformers" data-tag="transformers">Transformers</a> <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a> <a class="md-tag" href="/explore/?t=km-scale" data-tag="km-scale">Km-scale</a> <a class="md-tag" href="/explore/?t=quarter-degree" data-tag="quarter-degree">0.25°</a> <a class="md-tag" href="/explore/?t=hourly" data-tag="hourly">Hourly</a> <a class="md-tag" href="/explore/?t=6-hourly" data-tag="6-hourly">6-hourly</a>
    { .paper-tags }

-   #### Explaining El Niño Forecasts with the Average Gradient Outer Product { #2610.01095 }

    *Yuan Hui, Dorian S. Abbot, Robert J. Webber* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.01095">An important and unresolved problem in the physical sciences is explaining the predictions made by neural networks. Several explainable artificial intelligence (XAI) methods have been proposed to...</span><span class="abstract-full" id="full-2610.01095" hidden>An important and unresolved problem in the physical sciences is explaining the predictions made by neural networks. Several explainable artificial intelligence (XAI) methods have been proposed to address this problem, including gradient XAI, Integrated Gradients, and GradientSHAP. We evaluate the baseline XAI methods according to four scores: sensitivity (XAI patterns strongly affect predictions), attribution (XAI patterns reproduce the change in prediction relative to a baseline), robustness (XAI patterns remain stable for nearby inputs), and coherence (XAI patterns are spatially smooth). We also introduce a new method, average gradient outer product (AGOP) XAI, that uses global gradient information to identify an important direction for a specific input. We apply XAI to neural network predictions of the El Niño-Southern Oscillation (ENSO) based on data from the Zebiak-Cane model.   AGOP XAI achieves the highest attribution, robustness, and coherence scores in the architecture and lead-time comparisons reported here. Its sensitivity is surpassed by gradient XAI, which is maximally sensitive by definition. Beyond diagnosing neural-network behavior, AGOP XAI can generate candidate hypotheses about physical mechanisms. The method highlights an equatorial thermocline-depth signal consistent with recharge oscillator physics, together with a southeastern-Pacific lobe that may be specific to the Zebiak-Cane model. Finally, we test the physical relevance of AGOP using optimized perturbations that move the Zebiak-Cane model along AGOP explanation coordinates. Such perturbations can suppress the selected extreme events or, from a near-neutral ensemble, generate strong El Niño or La Niña events 10 months later.</span> <span class="abstract-toggle" data-id="2610.01095">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.01095v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.01095v1) · [:fontawesome-brands-github: Code](https://github.com/rjwebber/agop-xai) · [:material-content-copy: BibTeX](bibtex/2610.01095.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=subseasonal-to-seasonal" data-tag="subseasonal-to-seasonal">Subseasonal to seasonal</a> <a class="md-tag" href="/explore/?t=interpretability" data-tag="interpretability">Interpretability</a>
    { .paper-tags }

-   #### Weather Jiu-Jitsu: Exploring the Feasibility of Control Paradigms in Weather Foundation Models { #2610.00792 }

    *Prakriti Biswas, Kobi Abayomi, Upmanu Lall* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.00792">Weather Jiu-Jitsu is a control paradigm for extreme climatological events, inspired by chaos theory. As a proposition, small, precise, targeted, and cost-inexpensive perturbations can redirect...</span><span class="abstract-full" id="full-2610.00792" hidden>Weather Jiu-Jitsu is a control paradigm for extreme climatological events, inspired by chaos theory. As a proposition, small, precise, targeted, and cost-inexpensive perturbations can redirect trajectories of a large dynamical system. This strategy has been demonstrated analytically in the Lorenz-63 system, where a naturally chaotic trajectory switching between two attractors can be confined to a single attractor, indefinitely, via arbitrarily small perturbations. This paper examines the feasibility of Microsoft's Aurora -- a 1.3 billion parameter global atmospheric model -- as a test bed for this strategy. This paper explores three questions: (1) Is Aurora a reliable enough simulation environment to serve as a meaningful testbed? (2) Are the perturbations required to redirect its trajectories small enough to be physically plausible? (3) Does Aurora's learned latent space (the parametric estimators on climatological attributes) yield any apparent, structured, and/or perhaps interpretable features that can convey a geo/atmospheric response to initial conditions? We find evidence consistent with all three: Aurora's modeled trajectories respond to perturbations beyond measurement drift, the perturbation magnitudes required are small relative to the model's own forecast uncertainty, and its latent representations exhibit directional structure that responds to Jiu-Jitsu-type interventions, even though that structure does not separate extreme from normal states outright. These results should be read as feasibility diagnostics rather than a demonstration of control: we do not implement or test an actual steering intervention on Aurora, and several of our findings, particularly around the model's latent-space geometry, are exploratory.</span> <span class="abstract-toggle" data-id="2610.00792">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.00792v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.00792v1) · [:material-content-copy: BibTeX](bibtex/2610.00792.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=foundation-models" data-tag="foundation-models">Foundation models</a> <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a>
    { .paper-tags }

-   #### Benchmarking Generative Models for Weather Data Assimilation on Real Station Observations { #2610.00728 }

    *Ruizhe Huang, Qidong Yang, Jonathan Giezendanner, Sherrie Wang* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.00728">Weather reanalysis products rely on computationally intensive numerical weather predictions followed by data assimilation that corrects the forecast toward observations. Deep generative models offer...</span><span class="abstract-full" id="full-2610.00728" hidden>Weather reanalysis products rely on computationally intensive numerical weather predictions followed by data assimilation that corrects the forecast toward observations. Deep generative models offer a cheaper alternative that shifts much of this cost from inference to offline training. However, existing generative approaches have been evaluated on synthetic observations or under different datasets and evaluation schemes, making it unclear which design choices actually improve real-world data assimilation. We present the first controlled benchmark of generative weather data assimilation on real weather station observations. Using 11,849 NOAA MADIS stations across the contiguous United States and four weather variables, we evaluate methods while holding the dataset, observation operator, and deep learning architecture fixed. The benchmark compares the major design choices, including diffusion versus flow matching, pixel versus latent-space formulations, and multiple inference-time conditioning strategies, against a classical 3D-Var baseline. The benchmark reveals three clear conclusions. First, learned generative priors outperform the Gaussian prior of 3D-Var (35.7% vs. 33.3% RMSE reduction over ERA5) despite using no ERA5 background field at inference. Second, full-gradient guidance consistently outperforms stop-gradient and initial-noise optimization. Third, other choices provide little measurable benefit: diffusion and flow matching perform nearly identically under matched conditions, and latent-space variable mixing does not help. We further evaluate both dense and sparse station settings and find advantages from generative AI and full-gradient guidance more pronounced under sparsity. Together, these results identify which components of generative weather data assimilation improve performance on real station observations and establish a standardized benchmark for future work.</span> <span class="abstract-toggle" data-id="2610.00728">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.00728v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.00728v1) · [:material-content-copy: BibTeX](bibtex/2610.00728.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=benchmarks-datasets" data-tag="benchmarks-datasets">Benchmarks & datasets</a> <a class="md-tag" href="/explore/?t=station-point" data-tag="station-point">Station / point</a>
    { .paper-tags }

-   #### STCFormer: Adaptive Spatio-Temporal Modeling with Dynamic Cluster Transformer for Station-based Weather Forecasting { #2610.00377 }

    *Rongwen Li, Haixin Xie, Mingyang Wang, Hongwu Liu, Kun Fang, Changjian Chen, Zhuo Tang, Kenli Li* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.00377">Station-based weather forecasting supports daily life and economic activity, yet accurate forecasts require modeling complex spatial dependencies among stations. Recent clustering-based selective...</span><span class="abstract-full" id="full-2610.00377" hidden>Station-based weather forecasting supports daily life and economic activity, yet accurate forecasts require modeling complex spatial dependencies among stations. Recent clustering-based selective modeling offers a promising alternative to dense inter-station interactions. However, a grouping shared across an observation window may obscure local changes in station relationships, while intra-cluster interactions alone may miss important global context. The theoretical advantages of selective interactions over dense connectivity also remain insufficiently understood. We therefore propose STCFormer, an adaptive spatio-temporal Transformer that dynamically groups stations according to their local evolution within each temporal patch. Its Cluster-Guided Attention Block combines fine-grained local attention within clusters and global attention over regional state summaries, allowing each station to access information beyond its own cluster. We further show that a derived Lipschitz upper bound for cluster-conditioned local attention is no larger than its fully connected counterpart, explaining a potential robustness benefit and motivating the design of InfoLoss. Experiments on three real-world weather datasets spanning eight temperature and wind forecasting tasks show that STCFormer achieves the lowest 24-hour mean squared error on all eight tasks and ranks first or second in 47 of 48 comparisons across metrics and forecasting horizons. Ablations and case studies further confirm the benefits of locally adaptive grouping and complementary local-global interactions. Our code can be obtained at https://github.com/hnu-vis/STCFormer.</span> <span class="abstract-toggle" data-id="2610.00377">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.00377v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.00377v1) · [:fontawesome-brands-github: Code](https://github.com/hnu-vis/STCFormer) · [:material-content-copy: BibTeX](bibtex/2610.00377.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=transformers" data-tag="transformers">Transformers</a> <a class="md-tag" href="/explore/?t=station-point" data-tag="station-point">Station / point</a> <a class="md-tag" href="/explore/?t=daily" data-tag="daily">Daily</a>
    { .paper-tags }

-   #### Less is more: error-distance scaling relation for data-efficient kilometer-scale downscaling of extreme heat { #2609.40140 }

    *Ahmed Marey, Henry Lu, Abhishek Gaur, Sherif Goubran, Malek Aloui, Theodore Potsis, David Rolnick et al.* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.40140">Extreme heat is where urban adaptation needs kilometer-scale data the most, but the simulations training a downscaler can cost more than they save, and how much is needed has not been identified. We...</span><span class="abstract-full" id="full-2609.40140" hidden>Extreme heat is where urban adaptation needs kilometer-scale data the most, but the simulations training a downscaler can cost more than they save, and how much is needed has not been identified. We measured it with CASPER, a U-Net with a structure-preserving loss downscaling 32 km reanalysis to 1 km temperature, humidity and wind, across 24 configurations of one to eight months. Held-out error grows linearly with climatological distance to the training data, RMSE = 0.83 + 2.95 d, explaining 90% of its variance against 7% for volume and predicting unseen months in advance. On held-out extreme summer weeks CASPER preserves the fine-scale structure and cross-variable physics that matched-budget baselines degrade, and matches station observations during documented heat waves to within 1.8 K. Transfer to a new region degrades geographically; 11 days of local simulation cuts Vancouver's held-out error from 3.8 to 1.3 K. Training periods should span the target climate: the same accuracy for four times less simulation, putting kilometer-scale downscaling of extreme heat within reach of groups without large computing facilities.</span> <span class="abstract-toggle" data-id="2609.40140">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.40140v2) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.40140v2) · [:material-content-copy: BibTeX](bibtex/2609.40140.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a> <a class="md-tag" href="/explore/?t=extremes" data-tag="extremes">Extremes</a> <a class="md-tag" href="/explore/?t=km-scale" data-tag="km-scale">Km-scale</a>
    { .paper-tags }

-   #### RainAtlas: A Multi-Continental Dataset for Precipitation Downscaling { #2609.39833 }

    *Pierre-Louis Lemaire, Luca Schmidt, Wietze Suijker, Alex Hernandez-Garcia, David Rolnick* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.39833">Extreme rainfall events are increasing in intensity and frequency as climate change accelerates. While kilometer-scale precipitation forecasts are critical for supporting local decision-making, the...</span><span class="abstract-full" id="full-2609.39833" hidden>Extreme rainfall events are increasing in intensity and frequency as climate change accelerates. While kilometer-scale precipitation forecasts are critical for supporting local decision-making, the limited availability of high-resolution precipitation observations hinders their accuracy, especially in under-resourced regions. Machine learning models are widely used to downscale precipitation data to km-scale, but their application to unseen geographies presents challenges. First, processing raw high-resolution precipitation datasets across regions requires significant engineering and domain expertise. Second, generalization across regions remains difficult. To help overcome these barriers, we release RainAtlas, a large-scale, ML-ready and multi-continental dataset for precipitation downscaling. Covering three continents, RainAtlas harmonizes heterogeneous hourly km-scale observations to a common 2-km grid. Each regional partition contains around 210,000 aligned low- and high-resolution precipitation pairs, respectively from ERA5 reanalysis and direct observations. We benchmark state-of-the-art ML-based downscaling models across RainAtlas using a wide range of metrics. Our evaluation reveals substantial variance in out-of-domain generalization depending on the training regions. This underscores the need for cross-regional, multi-source km-scale evaluation, establishing RainAtlas as a well-positioned benchmark for precipitation downscaling research.</span> <span class="abstract-toggle" data-id="2609.39833">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.39833v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.39833v1) · [:material-content-copy: BibTeX](bibtex/2609.39833.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a> <a class="md-tag" href="/explore/?t=km-scale" data-tag="km-scale">Km-scale</a> <a class="md-tag" href="/explore/?t=hourly" data-tag="hourly">Hourly</a>
    { .paper-tags }

-   #### A library for differentiable signal processing and machine learning on the sphere { #2609.39737 }

    *Thorsten Kurth, Max Rietmann, Mauro Bisson, Andrea Paris, Alberto Carpentieri, Jean Kossaifi et al.* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.39737">The two-dimensional sphere embedded in three-dimensional Euclidean space S2, plays a central role in a variety of scientific and engineering domains, including geophysics, planetary science, geodesy,...</span><span class="abstract-full" id="full-2609.39737" hidden>The two-dimensional sphere embedded in three-dimensional Euclidean space S2, plays a central role in a variety of scientific and engineering domains, including geophysics, planetary science, geodesy, atmospheric physics, quantum chemistry, cosmology, and virtual reality, among many others. As machine learning increasingly permeates these fields, the demand grows for robust tools that process and model functions on the sphere, while respecting the inherent topological and symmetry properties of the domain. We present torch-harmonics, a comprehensive library that offers efficient, differentiable implementations of advanced signal processing and machine learning (ML) methods for spherical data. These include the spherical harmonic transform (SHT), the spherical analogue of the Fourier transform, vector spherical harmonics, discrete-continuous and spectral convolutions, as well as both global and neighborhood spherical attention mechanisms. Beyond traditional representations, torch-harmonics provides the building blocks for state-of-the-art spherical ML architectures such as spherical transformers in order to enable scalable, rotationally-aware learning and inference in modern scientific and engineering applications.</span> <span class="abstract-toggle" data-id="2609.39737">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.39737v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.39737v1) · [:material-content-copy: BibTeX](bibtex/2609.39737.bib){ .bibtex-link }
    { .paper-links }

-   #### Butterfly Effect Confirmed in Global AI Weather Models: Evidence from Tropical Cyclone Forecasting { #2609.39379 }

    *Jeremy Cheuk-Hin Leung, Daosheng Xu, Weiye Yu, Shaojing Zhang, Xiaodong Zeng, Gaozhen Nie, Jie Feng et al.* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.39379">A paradox recently emerged in artificial intelligence (AI) weather prediction research. While some claim AI weather models cannot simulate atmospheric butterfly effect, this conflicts with AI models'...</span><span class="abstract-full" id="full-2609.39379" hidden>A paradox recently emerged in artificial intelligence (AI) weather prediction research. While some claim AI weather models cannot simulate atmospheric butterfly effect, this conflicts with AI models' limited predictability and advances in AI ensemble forecasting. This study demonstrates via counterexamples that the butterfly effect does exist in AI weather predictions. For Super Typhoon Khanun, AI predictions are constrained by a double-attractor system. Minor initial perturbations confined to two regions trigger state transitions between two local attractors, causing a 1006-km difference in the predicted storm position on Day 7. This behavior is consistent with numerical weather prediction models and observed in ~12% of tropical cyclones in the past 5 years. These findings verify AI's ability to capture atmospheric chaos and provide the physical basis for AI ensemble forecasting.</span> <span class="abstract-toggle" data-id="2609.39379">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.39379v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.39379v1) · [:material-content-copy: BibTeX](bibtex/2609.39379.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### Proper Scoring Rule-based Diffusion for Probabilistic Weather Forecasting { #2609.38632 }

    *Joonhyeong Park, Giung Nam, Hyungi Lee, Kyunghyun Cho, Byoungwoo Park, Juho Lee* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.38632">Recent probabilistic weather forecasters train stochastic predictors with the continuous ranked probability score (CRPS) to generate each ensemble member in a single forward pass. These models learn...</span><span class="abstract-full" id="full-2609.38632" hidden>Recent probabilistic weather forecasters train stochastic predictors with the continuous ranked probability score (CRPS) to generate each ensemble member in a single forward pass. These models learn the predictive distribution from the forecast context alone, which becomes difficult at longer forecast horizons where uncertainty is high. To learn the predictive distribution more effectively, we introduce auxiliary conditional denoising tasks that predict the same future state from the context and its corrupted version, which provides partial future information that can reduce prediction ambiguity. Building on distributional diffusion models, we learn the conditional distributions of these tasks with a single stochastic predictor by minimizing a proper scoring rule across noise levels. At inference, the predictor can still generate each ensemble member in a single forward pass at the fully corrupted endpoint. Standard CRPS training is recovered as the endpoint-only special case of our formulation, so our framework extends existing CRPS-based forecasters with only additional conditioning inputs. Controlled experiments show that the auxiliary tasks improve one-step forecasting across architectures, with larger gains at longer forecast horizons. The gains extend to high-dimensional global weather forecasting under both training from scratch and fine-tuning, along with improved calibration and potential benefits for generalization under distribution shift.</span> <span class="abstract-toggle" data-id="2609.38632">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.38632v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.38632v1) · [:material-content-copy: BibTeX](bibtex/2609.38632.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a>
    { .paper-tags }

-   #### Methodological Changes to the Attention ResUNet Hourly Precipitation Postprocessor { #2609.38609 }

    *Thomas M. Hamill* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.38609">This note is a technical companion to a previously published preprint describing an Attention Residual U-Net that postprocesses deterministic forecasts from The Weather Company's Global and Regional...</span><span class="abstract-full" id="full-2609.38609" hidden>This note is a technical companion to a previously published preprint describing an Attention Residual U-Net that postprocesses deterministic forecasts from The Weather Company's Global and Regional Atmospheric Forecast (GRAF) model into probabilistic hourly precipitation forecasts. It documents what has changed in that method since publication. Feature-wise Linear Modulation conditioning on calendar season and forecast lead time is used to produce a single trained model for each season, replacing 192 separately trained per-month, per-lead checkpoints. Lead time is extended from 48 to 72 h. Two new input channels are used, per-pixel local solar hour and a static, monthly-varying precipitation climatology. During verification, the climatological reference against which the Brier Skill Score is computed now has an added diurnal dimension, on top of the monthly resolution it already had. Brier Skill Score and reliability are compared between the new vs. the previous training. Forecasts generated with the new training show a modest, consistent improvement of the current training over the original.</span> <span class="abstract-toggle" data-id="2609.38609">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.38609v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.38609v1) · [:material-content-copy: BibTeX](bibtex/2609.38609.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=hourly" data-tag="hourly">Hourly</a> <a class="md-tag" href="/explore/?t=monthly" data-tag="monthly">Monthly</a>
    { .paper-tags }

-   #### An Input-Frugal Deep Learning Framework for Weather-Driven National Crop-Yield Forecasting: A Case Study of Brazilian Soybean { #2609.38447 }

    *Fernando Dupin da Cunha Mello, Prashant Kumar, Erick G. Sperandio Nascimento* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.38447">Reliable, timely crop-yield forecasts are essential for market stability and risk management, yet many approaches rely on costly or hard-to-scale inputs. We present a frugal, transferable, and...</span><span class="abstract-full" id="full-2609.38447" hidden>Reliable, timely crop-yield forecasts are essential for market stability and risk management, yet many approaches rely on costly or hard-to-scale inputs. We present a frugal, transferable, and architecture-agnostic deep learning framework that uses routine weather as the only time-varying input plus two lightweight static context inputs (crop year and an agro-environmental label) to capture long-run change and regional heterogeneity, while supporting multiple sequence encoders under identical data requirements. Using a 20-season Brazilian soybean case study (2001/02-2020/21) with leave-one-year-out cross-validation, we benchmark MLP, CNN, LSTM, CNN-LSTM, a Transformer encoder and the Mamba state-space model against linear ridge regression and a five-year moving-average "farmer" baseline. All deep learning variants outperform ridge, and all sequential encoders surpass the non-sequential MLP. The Transformer achieves the best national accuracy (RMSE 149 kg ha^-1; rRMSE 5.3%; R^2 = 0.784), reducing error by 47.6% relative to the farmer baseline. In-season forecasts improve monotonically from early- to late-season issuance, reaching approximately 50% lower error than the baseline at the latest forecast point. Ablations indicate that the agro-environmental label and spatial instance expansion (multiple grid-node weather sequences per municipality-year) contribute positively without increasing input complexity. SHAP diagnostics suggest crop year explains most of the long-run trajectory, whereas within-season weather and agro-environmental context primarily drive interannual deviations, with moisture/cloud and thermal-demand variables dominating. Overall, the framework is straightforward to deploy across other crops and geographic regions and is naturally compatible with operational weather forecasts for routine monitoring.</span> <span class="abstract-toggle" data-id="2609.38447">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.38447v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.38447v1) · [:material-content-copy: BibTeX](bibtex/2609.38447.bib){ .bibtex-link }
    { .paper-links }

-   #### Physics-Guided Flow-Map Matching for Precipitation Nowcasting { #2609.37487 }

    *Shunya Nagashima, Takumi Bannai, Makoto Misaizu, Keisuke Maeda, Takahiro Ogawa, Miki Haseyama* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.37487">Precipitation nowcasting, generating future radar fields from past observations, is critical for flood warning and disaster response. It is also a demanding benchmark for spatiotemporal generative...</span><span class="abstract-full" id="full-2609.37487" hidden>Precipitation nowcasting, generating future radar fields from past observations, is critical for flood warning and disaster response. It is also a demanding benchmark for spatiotemporal generative modeling, with chaotic dynamics, heavy-tailed intensities, and rare high-intensity structures that matter most. Deterministic models minimize a pixel loss and are driven toward the conditional mean, which blurs exactly those structures, while generative models that add a stochastic residual on top of a deterministic backbone inherit the same blur. We propose Physics-Guided Flow-Map Matching (PG-FMM), a conditional flow-map model that decouples predictable advection from uncertain small-scale detail. A frozen Lagrangian advection prior transports the radar field and supplies an explicit motion forecast, and a flow-map generative head, conditioned on the past frames and the prior rollout rather than summed onto it, produces sharp stochastic detail in four sampling steps. The prior serves only as guidance, so the head replaces blurred structure instead of inheriting it. Extensive experiments on four radar benchmarks show that PG-FMM outperforms state-of-the-art methods on 18 of 24 metrics, with the largest gains at heavy-rain thresholds, where the critical success index improves by up to 58.9%. The project page can be found at https://neurogica.github.io/PG-FMM.</span> <span class="abstract-toggle" data-id="2609.37487">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.37487v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.37487v1) · [:material-content-copy: BibTeX](bibtex/2609.37487.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=physics-ml-hybrid" data-tag="physics-ml-hybrid">Physics–ML hybrid</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a>
    { .paper-tags }

-   #### NowcastDiT: Diffusion Transformers are Effective Precipitation Nowcasters { #2609.37038 }

    *Haoran Xu, Xingzhuo Guo, Yuchen Zhang, Jincheng Zhong, Jianmin Wang, Mingsheng Long* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.37038">Precipitation nowcasting demands accurate short-term forecasts under strong spatiotemporal variability. Diffusion models are well suited to modeling complex precipitation distributions, yet existing...</span><span class="abstract-full" id="full-2609.37038" hidden>Precipitation nowcasting demands accurate short-term forecasts under strong spatiotemporal variability. Diffusion models are well suited to modeling complex precipitation distributions, yet existing approaches often introduce increasingly specialized designs, leaving the capability of a standard diffusion architecture underexplored. We show that a standard Diffusion Transformer already provides a simple and scalable foundation for precipitation nowcasting, with domain-specific requirements accommodated naturally within its design space. Based on this principle, we develop NowcastDiT and instantiate this flexibility through two complementary adaptations: a dynamics-aware noise prior for temporally coherent forecasts, and end-to-end reinforcement learning with timestep-aware rewards for meteorological skill. Experiments on SEVIR and MRMS benchmarks show that NowcastDiT achieves state-of-the-art performance in both perceptual quality and meteorological skill. These results suggest that standard DiT can serve as an effective foundation for precipitation nowcasting.</span> <span class="abstract-toggle" data-id="2609.37038">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.37038v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.37038v1) · [:material-content-copy: BibTeX](bibtex/2609.37038.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=transformers" data-tag="transformers">Transformers</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a>
    { .paper-tags }

-   #### A neural network-based Universal Thermal Climate Index for reliable global thermal-stress classification across extreme weather { #2609.35949 }

    *Bikem Pastine, Milan Klöwer, Tianning Tang, Sarah Wilson Kemsley, Louise Slater* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.35949">Extreme temperatures are the leading cause of climate-related mortality world-wide. Climate-health research and operational weather forecasting require accurate estimates of human thermal stress. The...</span><span class="abstract-full" id="full-2609.35949" hidden>Extreme temperatures are the leading cause of climate-related mortality world-wide. Climate-health research and operational weather forecasting require accurate estimates of human thermal stress. The Universal Thermal Climate Index (UTCI) is among the most sophisticated and widely used feels-like temperature metrics. However, its ubiquitous polynomial approximation does not generalize well to extreme weather conditions. Here, we introduce Neural-UTCI, a neural network that calculates UTCI with substantially higher accuracy across global conditions at a lower computational cost for operational use. Neural-UTCI reduces the polynomial approximation RMSE from 2.78°C to 0.36 °C, an 87% improvement, and lowers thermal stress misclassification rates from 5.3% to 1.7%, with consistent performance across resampling experiments. These differences affect thermal exposure metrics. For example, during the 2003 European heatwave summer in Rome, Italy, the number of very strong heat stress days increases from 15 to 35 days when using Neural-UTCI compared to operational products like ERA5-HEAT. Simultaneously, Neural-UTCI reliably classifies extreme cold stress conditions, allowing continuous global application. By improving UTCI accuracy, Neural-UTCI can strengthen climate-health risk assessments and public weather warning systems, especially as global warming increases the incidence of extreme events.</span> <span class="abstract-toggle" data-id="2609.35949">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.35949v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.35949v1) · [:material-content-copy: BibTeX](bibtex/2609.35949.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=extremes" data-tag="extremes">Extremes</a>
    { .paper-tags }

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

</div>

