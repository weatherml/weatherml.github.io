---
title: 'Global Models'
hide:
  - toc
---

<div class="listing-header" markdown>

# Global Models

<p class="page-meta" markdown="span">354 papers · page 1 of 12 · <a href="../../bib/global-models.bib" download>:material-download: BibTeX for this topic</a></p>

</div>

<div class="grid cards" markdown>

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

-   #### Uncertainty-Aware End-to-End AI Weather Forecasting: Disentangling Observation and Model Contributions { #2608.30795 }

    *Rodrigo Almeida, Noelia Otero, Jost Arndt, Simon Baur, Wojciech Samek, Jackie Ma* · Aug 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2608.30795">End-to-end weather forecasting systems produce skillful global gridded and station forecasts directly from raw Earth observations, replacing the numerical weather prediction pipeline, including data...</span><span class="abstract-full" id="full-2608.30795" hidden>End-to-end weather forecasting systems produce skillful global gridded and station forecasts directly from raw Earth observations, replacing the numerical weather prediction pipeline, including data assimilation, at a fraction of its cost. These systems are deterministic and issue no uncertainty. Here we render the Aardvark Weather model probabilistic by attaching one stochastic mechanism to each component: learned, input-dependent noise at the observation encoder, capturing aleatoric uncertainty inherited from the observing system, and Monte Carlo dropout in the processor, capturing epistemic uncertainty in the learned dynamics. The resulting nested ensemble attributes forecast spread to the two sources through a law-of-total-variance decomposition, cross-checked by withholding observation streams. Probabilistic finetuning significantly improves the mean forecast, by 4.2% on average across variables and lead times. The ensemble is calibrated against ERA5 through the medium range (spread-skill ratio 0.98), keeps station RMSE within 2.4% of the deterministic model while beating it in CRPS at every lead time, and trails the operational ECMWF ensemble. The encoder branch behaves as observation-driven uncertainty. Component-attributed uncertainty makes end-to-end forecasts more transparent, a step toward observation-driven digital twins of the atmosphere.</span> <span class="abstract-toggle" data-id="2608.30795">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2608.30795v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2608.30795v1) · [:material-content-copy: BibTeX](../../bibtex/2608.30795.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a>
    { .paper-tags }

-   #### Diffusion Distillation for Efficient Weather Ensembles { #2608.27728 }

    *Yiming Yang, Valentin Brekke, James Briant, Serge Guillas* · Aug 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2608.27728">Diffusion models generate skillful weather ensembles but require costly iterative sampling. We introduce a supervised energy-distance distillation method that compresses a multi-step diffusion...</span><span class="abstract-full" id="full-2608.27728" hidden>Diffusion models generate skillful weather ensembles but require costly iterative sampling. We introduce a supervised energy-distance distillation method that compresses a multi-step diffusion teacher into a single-step student by aligning student forecasts with teacher samples and ground-truth observations. Experiments on global forecasting and typhoon-track prediction show that our student outperforms existing distillation methods and preserves skill for extreme events. It matches or surpasses the teacher across key metrics using only one neural function evaluation per autoregressive step.</span> <span class="abstract-toggle" data-id="2608.27728">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2608.27728v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2608.27728v1) · [:material-content-copy: BibTeX](../../bibtex/2608.27728.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### Climate Physics Dynamic Matching { #2608.26907 }

    *Gurjeet Sangra Singh, Frantzeska Lavda, Alexandros Kalousis* · Aug 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2608.26907">Deep generative models such as flow matching and diffusion models have shown potential for learning complex dynamical systems, but typically act as black boxes that neglect underlying physical...</span><span class="abstract-full" id="full-2608.26907" hidden>Deep generative models such as flow matching and diffusion models have shown potential for learning complex dynamical systems, but typically act as black boxes that neglect underlying physical structure, while physics-based models governed by partial differential equations are often incomplete due to missing source terms, or uncertain parametrisations. We present Climate Physics Dynamic Matching (ClimPhyDM), a variational simulation-free dynamics informed framework for weather forecasting that combines an advection-type physics prior with data-driven components in a variational framework. % to capture the stochasticity and multi-modality of unresolved atmospheric dynamics. On the ERA5 benchmark at hourly (42-hour) and monthly (5-month) resolutions, ClimPhyDM outperforms ClimODE, and GB-DM, keeping the lower error at extended horizon, indicating improved temporal stability and resistance to error accumulation, while its simulation-free paradigm also enables training on a single modest 12 GB consumer GPU.</span> <span class="abstract-toggle" data-id="2608.26907">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2608.26907v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2608.26907v1) · [:material-content-copy: BibTeX](../../bibtex/2608.26907.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=hourly" data-tag="hourly">Hourly</a> <a class="md-tag" href="/explore/?t=monthly" data-tag="monthly">Monthly</a>
    { .paper-tags }

-   #### Bridging short- and medium-range weather forecasting with machine learning { #2608.26822 }

    *Timothy A. Smith, Mariah Pope, Sergey Frolov, Brett Basarab, Daniel Abdi, Paul Madden et al.* · Aug 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2608.26822">The National Oceanic and Atmospheric Administration (NOAA) employs independent prediction systems for distinct forecast products. While some separation is practical, we argue that combining short-...</span><span class="abstract-full" id="full-2608.26822" hidden>The National Oceanic and Atmospheric Administration (NOAA) employs independent prediction systems for distinct forecast products. While some separation is practical, we argue that combining short- and medium-range weather into a single prediction system would provide the public with a useful distillation of global weather and its impacts. To this end, we present Nested-EAGLE (Experimental Artificial intelligence Global and Limited-area Ensemble): a 0.25° global weather model with a 6 km refinement over the Contiguous United States (CONUS). The model achieves significantly lower mean-squared error in near-surface and low-level quantities over CONUS compared to NOAA's Global Forecast System and High-Resolution Rapid Refresh (HRRR), while remaining competitive throughout the rest of the global atmosphere. We show that the skill gains for near-surface fields stem from incorporating high-resolution regional analysis data into training through the nesting process. Forecasts of precipitation amounts are less skillful than those from HRRR, owing to deterministic training. However, we show that Nested-EAGLE provides the most accurate forecasts of storm locations at longer leads, despite blurred extrema. Our results motivate future work to extend the skill gains beyond CONUS and improve precipitation representation.</span> <span class="abstract-toggle" data-id="2608.26822">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2608.26822v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2608.26822v1) · [:material-content-copy: BibTeX](../../bibtex/2608.26822.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a>
    { .paper-tags }

-   #### Missing the Butterfly and Predicting the Past: Features or Bugs of Accurate AI Weather Models? { #2608.25835 }

    *Pedram Hassanzadeh, Weidong Li, Y. Qiang Sun, Jiangdi Wang, Alexander Wikner, Justin Finkel et al.* · Aug 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2608.25835">AI weather prediction (AIWP) models rival physics-based models, yet the sources of their unexpected forecast accuracy and the degree of their physical fidelity remain unclear. Here, across a...</span><span class="abstract-full" id="full-2608.25835" hidden>AI weather prediction (AIWP) models rival physics-based models, yet the sources of their unexpected forecast accuracy and the degree of their physical fidelity remain unclear. Here, across a hierarchy spanning observation-based reanalysis, a general circulation model, and the multi-scale Lorenz system, we show that AI models can be trained to skillfully predict the past (backcast), though backcasts are systematically less accurate than forecasts. However, skillful backcasting appears to violate the second law of thermodynamics, and all these forecasting and backcasting models miss the butterfly effect. We trace the surprising forecast accuracy, missing butterfly, and skillful backcasting to a single cause: inevitable coarse-graining of training data, which removes fast, small scales and/or some variables. From the Lorenz system to official Pangu-Weather models, reducing coarse-graining makes AI predictions more physics-like (arrow of time and butterfly-like effects emerge), but forecast accuracy declines. Results offer an explanation for AIWP models' forecast skill: unlike physics-based models, they implicitly learn how fast, small scales affect large scales without inheriting their rapid error growth. Broader implications are that AI models' proliferation calls for revisiting predictability theories and long-term climate emulation strategies, and backcasting offers a useful, new lens for such analyses.</span> <span class="abstract-toggle" data-id="2608.25835">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2608.25835v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2608.25835v1) · [:material-content-copy: BibTeX](../../bibtex/2608.25835.bib){ .bibtex-link }
    { .paper-links }

-   #### AFDBench: A Reasoning-First AI Scientist for NationalWeather Service Forecast Discussions { #2608.24954 }

    *Manmeet Singh, Somnath Luitel, Prabhjot Singh, Manraaj Banga, Naveen Sudharsan, Josh Durkee* · Aug 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2608.24954">Large language models (LLMs) hallucinate numerical values when generating high-stakes meteorological text, posing risks for weather communication. We present AFDBench, an AI meteorologist that...</span><span class="abstract-full" id="full-2608.24954" hidden>Large language models (LLMs) hallucinate numerical values when generating high-stakes meteorological text, posing risks for weather communication. We present AFDBench, an AI meteorologist that generates professional Area Forecast Discussions (AFDs) by reasoning through structured AI weather forecast data from Google's WeatherNext 2. We introduce AFDBench, the first benchmark for evaluating generative meteorological reasoning, comprising 7,732 expert written discussions from 13 National Weather Service (NWS) offices paired with real AI weather forecast inputs, and three complementary metrics: Met-Align (numerical accuracy), Style-Align (professional dialect adherence), and Input-Grounding (fidelity to source weather data). Zero-shot evaluations reveal that open-source LLMs achieve low Style-Align (~0.33) and moderate Input-Grounding (~0.88), failing to write in the professional NWS register or faithfully use their input data. We apply Group Relative Policy Optimization (GRPO) with domain-specific rewards targeting temperature accuracy, synoptic correctness, and format compliance. On 1,033 held-out samples from two unseen NWS offices, GRPO nearly doubles Style-Align from 0.318 to 0.619 and improves Input-Grounding from 0.881 to 0.940, demonstrating that reinforcement learning teaches a 7B-parameter model to write like a professional meteorologist and faithfully interpret AI weather data.</span> <span class="abstract-toggle" data-id="2608.24954">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2608.24954v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2608.24954v1) · [:material-content-copy: BibTeX](../../bibtex/2608.24954.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=llms-agents" data-tag="llms-agents">LLMs & agents</a> <a class="md-tag" href="/explore/?t=reinforcement-learning" data-tag="reinforcement-learning">Reinforcement learning</a>
    { .paper-tags }

-   #### AICON: An operational global machine learning weather forecasting model { #2608.24651 }

    *Tobias Goecke, Marek Jacob, Florian Prill, Michael Denhard, Felix Fundel, Jan Keller et al.* · Aug 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2608.24651">We introduce AICON, a global machine learning weather prediction (MLWP) model which generates forecasts at 13 km spatial resolution with a 3-hour time step, trained on the high-resolution,...</span><span class="abstract-full" id="full-2608.24651" hidden>We introduce AICON, a global machine learning weather prediction (MLWP) model which generates forecasts at 13 km spatial resolution with a 3-hour time step, trained on the high-resolution, non-hydrostatic ICON-DREAM dataset. AICON is in full operational use at Deutscher Wetterdienst since 2nd of March 2026. The model employs a graph neural network (GNN) architecture with an encoder-processor-decoder structure, where node updates are performed using a graph attention mechanism. A key feature of AICON is its use of an icosahedral multi-mesh derived from the native grid of the ICON model, ensuring consistency with the training data. ICON's terrain-following vertical SLEVE coordinate is one of the major distinctions from existing emulators. AICON's training strategy prioritizes small-scale fidelity by avoiding autoregressive multi-step rollout and longer forecast horizons during training, a design choice motivated by the hypothesis that this approach preserves fine-scale features often damped in models optimized for longer-range forecasts. We describe the prognostic and diagnostic variables used for training, the transfer learning protocol employed to accelerate convergence, and the model's performance across a range of evaluation metrics. An extensive evaluation, including routine verification against observation, a tropical cyclone case and spectral analysis reveal the strengths and limitations in the representation of atmospheric variability across scales. Routine verification against observations demonstrates competitive skill relative to the operational ICON model, particularly for near-surface variables in the short to medium forecast range.</span> <span class="abstract-toggle" data-id="2608.24651">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2608.24651v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2608.24651v1) · [:material-content-copy: BibTeX](../../bibtex/2608.24651.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=graph-neural-networks" data-tag="graph-neural-networks">Graph neural networks</a> <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a> <a class="md-tag" href="/explore/?t=evaluation" data-tag="evaluation">Evaluation</a>
    { .paper-tags }

-   #### Extremes on Rewind: Generating 1,000-Member Ensembles Initialized at a Final Condition { #2608.19008 }

    *Jerry Lin, Mu-Ting Chien, Mansi Sakarvadia, Elizabeth A. Barnes* · Aug 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2608.19008">Scenario planning for rare, high-impact events often requires massive ensembles to stochastically sample relevant trajectories. Although autoregressive weather emulators can efficiently generate such...</span><span class="abstract-full" id="full-2608.19008" hidden>Scenario planning for rare, high-impact events often requires massive ensembles to stochastically sample relevant trajectories. Although autoregressive weather emulators can efficiently generate such ensembles, isolating trajectories of interest requires sifting through petabytes of data, a challenge that grows exponentially with lead time and rarity. In contrast, a non-autoregressive foundation model like Climate in a Bottle video (cBottle-video) can directly sample trajectories terminating in extremes, avoiding large-ensemble search. We use cBottle-video to generate 1000-member ensembles with start- and/or end-conditioning across three extreme events---the 2021 Pacific Northwest (PNW) heatwave, Superstorm Sandy, and Hurricane Ian. Antecedent 500 hPa geopotential height ($z_{500}$) spread at the free end of end-conditioned ensembles reaches 84--89% of the final-state spread of start-conditioned ensembles, revealing substantial diversity consistent with each extreme event. For the 2021 PNW heatwave, end-conditioned ensemble members begin uniformly warmer than reanalysis and stay warm, replacing the observed rapid intensification with persistent antecedent heat. For Superstorm Sandy, the leading modes of $z_{500}$ at the antecedent end of the end-conditioned ensemble explain 44% of the variance in track latitude, and roughly 10% of ensemble members begin as stronger hurricanes than Sandy. For Hurricane Ian, variation in the first landfall location among end-conditioned trajectories underscores the importance of accounting for intermediate hazard exposure in risk planning.</span> <span class="abstract-toggle" data-id="2608.19008">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2608.19008v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2608.19008v1) · [:material-content-copy: BibTeX](../../bibtex/2608.19008.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=foundation-models" data-tag="foundation-models">Foundation models</a> <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a> <a class="md-tag" href="/explore/?t=extremes" data-tag="extremes">Extremes</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### Tianmu-TC: Physics-constraints Generative Artificial Intelligence for Global Tropical Cyclone Forecasting { #2608.18500 }

    *Shiqi Zhang, Pan Mu, Cheng Huang, Hanting Yan, Yuchao Zhu, Jinglin Zhang, Shengyong Chen et al.* · Aug 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2608.18500">Tropical cyclones (TCs) pose severe risks from strong winds and heavy rainfall. However, forecasting their track and intensity remains challenging due to chaotic atmosphere and the rapid...</span><span class="abstract-full" id="full-2608.18500" hidden>Tropical cyclones (TCs) pose severe risks from strong winds and heavy rainfall. However, forecasting their track and intensity remains challenging due to chaotic atmosphere and the rapid amplification of initial condition errors, leading to growing forecast uncertainty. While numerical weather prediction (NWP) and deep learning models have made progress, they remain computationally demanding and often fail under complex meteorological scenarios. Here, we present Tianmu-TC, a physics-constraints generative framework for global TC forecasting. Trained on Western North Pacific data, Tianmu-TC leverages physics-constraints to generate controllable outputs with reduced uncertainty thus improving forecast reliability. Experiments show Tianmu-TC outperforms deterministic and ensemble meteorological artificial intelligence models and authoritative NWP systems such as ECMWF in global ocean basins, with significantly lower computational cost. We further show Tianmu-TC performs well in challenging scenarios such as data sparsity, anomaly tracks, rapid intensification and weakening. These findings suggest physics-constraints generative AI offers a promising approach for reliable, efficient global TC forecasting.</span> <span class="abstract-toggle" data-id="2608.18500">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2608.18500v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2608.18500v1) · [:material-content-copy: BibTeX](../../bibtex/2608.18500.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a>
    { .paper-tags }

-   #### How Do AI Climate Models Respond to Warming Across Climate Zones? { #2608.17986 }

    *Charlotte C. Merchant, Milan Klöwer, Bradley Stanley-Clamp, Maren Höver, Simon L. L. Michel et al.* · Aug 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2608.17986">Regional climate zones are expected to shift under global warming. Whether AI climate models have learned to generalize climate-zone distributions under warming in a physically meaningful way affects...</span><span class="abstract-full" id="full-2608.17986" hidden>Regional climate zones are expected to shift under global warming. Whether AI climate models have learned to generalize climate-zone distributions under warming in a physically meaningful way affects their suitability for climate projection. We address this question by applying a Köppen-Geiger climate-zone decomposition to AIMIP Phase 1 models under prescribed +4K SST forcing and comparing their responses to physics-based AMIP models. Using this diagnostic, we compare baseline classification skill, per-zone responses in temperature, precipitation, and near-surface specific humidity, and the spatial structure of departures from physics-based models. All AI models considered reproduce the 1979-2014 ERA5 climatology within the physics-based models' range, but only the hybrid physics-AI model NeuralGCM-HRD reorganizes zones in agreement with established thermodynamic and hydrological scaling relations. The remaining emulators have distinct failure modes traceable to their architectural treatment of land cells. A physically consistent climate-zone response is therefore necessary for AI models intended for climate projection.</span> <span class="abstract-toggle" data-id="2608.17986">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2608.17986v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2608.17986v1) · [:material-content-copy: BibTeX](../../bibtex/2608.17986.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=physics-ml-hybrid" data-tag="physics-ml-hybrid">Physics–ML hybrid</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a>
    { .paper-tags }

-   #### Do AI weather models miss extremes? { #2608.09972 }

    *Marvin Vincent Gabler, Roberto Molinaro, Niall Siegenheim, Henry Martin, Mark Frey, Niels Poulsen et al.* · Aug 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2608.09972">First-generation AI weather models are often reported to underperform at extremes, mostly in reanalysis-based evaluations of deterministic regression systems. We verify eleven physical and AI...</span><span class="abstract-full" id="full-2608.09972" hidden>First-generation AI weather models are often reported to underperform at extremes, mostly in reanalysis-based evaluations of deterministic regression systems. We verify eleven physical and AI forecast systems against European synoptic, solar, and rain-gauge stations over ten months for 10 m wind, 2 m temperature, hourly shortwave accumulation, and hourly precipitation, scoring mean absolute error (MAE) against ECMWF IFS in ERA5 1991-2020 climatological regimes. Among these systems, AI models do not show a uniform relative-skill deficit in the tails. Jua EPT-2.1 Europa leads all-conditions wind (+8.4%), while Jua EPT-2 HRRR leads temperature overall (+12.1%) and in the heat regime (+19.6 +/- 2.2%). EPT-2.1 Europa and DWD ICON Global lead at gale-force wind. Jua EPT-2.1 Helios leads solar overall (+10.2 +/- 1.7%), in overcast conditions (+16.4 +/- 3.4%), and in the clear-sky tail (+24.8 +/- 5.4%). For precipitation, three Jua models gain 14-15% at moderate intensity and 9-11% at P75-P95; EPT-2 Reasoning remains ahead above P95 (+1.7 +/- 0.5%). Failures are model-specific: ECMWF AIFS loses 4.9 +/- 2.0% in the heat tail, while NOAA GFS loses 22.8 +/- 2.0% there. Every model, including numerical weather prediction systems, shows a shared conditional bias toward the centre of the observed distribution, with an inter-model spread several times smaller than the shared signal. Missing relative skill at extremes is therefore not a property of AI weather models as a class, but of particular AI and physical models.</span> <span class="abstract-toggle" data-id="2608.09972">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2608.09972v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2608.09972v1) · [:material-content-copy: BibTeX](../../bibtex/2608.09972.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=hourly" data-tag="hourly">Hourly</a>
    { .paper-tags }

</div>

<nav class="pager" markdown="span">**1** [2](2.md) [3](3.md) [4](4.md) [5](5.md) [6](6.md) [7](7.md) [8](8.md) [9](9.md) [10](10.md) [11](11.md) [12](12.md) [Older :material-arrow-right:](2.md){ .pager-step }</nav>

