---
title: 'Global Models'
hide:
  - toc
---

<div class="listing-header" markdown>

# Global Models

<p class="page-meta" markdown="span">398 papers · page 1 of 14 · <a href="../../bib/global-models.bib" download>:material-download: BibTeX for this topic</a></p>

</div>

<div class="grid cards" markdown>

-   #### Interpretable Prototype-Conditioned Adaptation for Weather Forecasting { #2611.25848 }

    *Zhixuan Li, Hao Jiang, Chen Hui, Kai Geng, Weisi Lin* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.25848">Data-driven weather forecasting requires distinguishing local states and their temporal evolution. However, learned patterns remain implicit in neural forecasters, making their influence difficult to...</span><span class="abstract-full" id="full-2611.25848" hidden>Data-driven weather forecasting requires distinguishing local states and their temporal evolution. However, learned patterns remain implicit in neural forecasters, making their influence difficult to interpret. Failing to distinguish local evolution can lead to similar forecasts for locations that develop differently, increasing prediction error. This motivates explicit references for distinguishing and interpreting local patterns. Prototypes provide explicit pattern references, but sample- or task-level conditioning leaves open how to adapt to distinct locations and variable groups. To deal with this problem, we design ProtoCast, which uses group-specific prototype banks to adapt local features while accounting for cross-variable dependencies. Specifically, (1) we combine each group's historical features and temporal change with forecast context from cross-variable attention for evolution-aware prototype matching. (2) Sparse mixtures of prototype-specific low-rank operators tailor corrections to each group and location. A gated attention block then exchanges the corrected features across groups before prediction. Each group's correction decomposes into explicit prototype contributions before this fusion. (3) We jointly train the forecasting network and operators from scratch and refit each group's prototypes before every training epoch. Slot tracking preserves prototype-operator associations, making each prototype's contribution to local corrections more interpretable. Experiments on GlobalForecast and TyphoonTracking show improved forecasting accuracy, while prototype deletion links local feature contributions to changes in prediction error. This work advances the development of more accurate and interpretable data-driven weather forecasting models. Project page: https://zhixuanli.github.io/projects/li2026protocast</span> <span class="abstract-toggle" data-id="2611.25848">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.25848v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.25848v1) · [:material-content-copy: BibTeX](../../bibtex/2611.25848.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a> <a class="md-tag" href="/explore/?t=interpretability" data-tag="interpretability">Interpretability</a>
    { .paper-tags }

-   #### Reinforcement-Aligned Probabilistic Weather Forecasting with Discrete Tokens { #2611.25570 }

    *Jinhan Li, Zhisheng Chen, Yuxuan Li, Yuan Gao, Fan Xu, Hao Wu, Jinlong Du, Kun Wang* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.25570">Global medium-range weather forecasting must not only generate ensembles efficiently to characterize uncertainty in future states, but also serve downstream applications such as extreme-event...</span><span class="abstract-full" id="full-2611.25570" hidden>Global medium-range weather forecasting must not only generate ensembles efficiently to characterize uncertainty in future states, but also serve downstream applications such as extreme-event warnings and cross-sphere coupled simulations. We introduce WeatherCraft, a framework that unifies probabilistic generation and reinforcement learning alignment through discrete weather tokens, bringing the GPT-style pretraining--post-training paradigm to global medium-range weather forecasting. A multiscale tokenizer represents atmospheric states as a coarse-to-fine token hierarchy, while a Transformer models causal dependencies across time and scale and predicts tokens within each scale in parallel. Direct sampling generates diverse ensembles, and exact token log-probabilities enable Group Relative Policy Optimization (GRPO) to maximize downstream task rewards without differentiating through evaluators or coupled models. On the ERA5 benchmark, WeatherCraft's eight-member ensemble mean reduces day-15 normalized RMSE by 17.6% relative to the strongest evaluated deterministic baseline and, with KV caching, achieves an approximately $18\times$ inference speedup over GenCast. Direct fair-CRPS adaptation improves ensemble probabilistic skill, while reward-based post-training improves forecasts of persistent land temperature extremes and coupled ocean heat extremes. This work establishes discrete autoregressive modeling as a general foundation for scalable, task-alignable Earth system foundation models.</span> <span class="abstract-toggle" data-id="2611.25570">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.25570v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.25570v1) · [:fontawesome-brands-github: Code](https://github.com/Jinhan-Li/WeatherCraft) · [:material-content-copy: BibTeX](../../bibtex/2611.25570.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=transformers" data-tag="transformers">Transformers</a> <a class="md-tag" href="/explore/?t=foundation-models" data-tag="foundation-models">Foundation models</a> <a class="md-tag" href="/explore/?t=reinforcement-learning" data-tag="reinforcement-learning">Reinforcement learning</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a>
    { .paper-tags }

-   #### Auditing Learned Dynamics in Weather Models { #2611.25119 }

    *Skylar Gu, Christina Last* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.25119">Learned weather models are experimentally accessible computational systems whose internal representations can be inspected and manipulated. Using sparse autoencoder features from GraphCast,...</span><span class="abstract-full" id="full-2611.25119" hidden>Learned weather models are experimentally accessible computational systems whose internal representations can be inspected and manipulated. Using sparse autoencoder features from GraphCast, DeepMind's autoregressive weather forecasting model, we test whether causal relevance requires physical semantics, a question that matters for auditing whether AI forecasters used in high-assurance settings rely on reliable mechanisms. We also look at whether physically associated latent directions can be steered to improve extreme-event forecasts, where interventions are applied to the model's internal activations during the rollout. We find that ablating a feature locked to GraphCast's computational mesh increases global geopotential height error despite lacking conventional physical semantics. Second, suppressing convection- and low-level-vorticity-associated features weakens predicted tropical-cyclone deepening (the fall in central pressure during intensification), while amplifying them reduces intensity error, with amplification of convection features through feature scaling achieving a median of 51% error reduction over seven storms. Together, these results separate interpretable, physical association from causal utility and establish sparse features as experimental coordinates for auditing and intervening on learned weather dynamics.</span> <span class="abstract-toggle" data-id="2611.25119">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.25119v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.25119v1) · [:material-content-copy: BibTeX](../../bibtex/2611.25119.bib){ .bibtex-link }
    { .paper-links }

-   #### Distribution-Free Uncertainty Quantification for Weather Forecasting: Conformal Prediction Under Spatiotemporal Dependence { #2611.24329 }

    * Mariya* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.24329">Conformal prediction gives finite-sample, distribution-free marginal coverage and has begun to be applied to gridded weather forecasts. Marginal coverage concerns the long-run fraction of covered...</span><span class="abstract-full" id="full-2611.24329" hidden>Conformal prediction gives finite-sample, distribution-free marginal coverage and has begun to be applied to gridded weather forecasts. Marginal coverage concerns the long-run fraction of covered points at a location, not how coverage behaves jointly across space and time on a given day. I benchmark eight interval-construction procedures (Gaussian, pooled and per-cell split conformal, per-season, adaptive conformal inference and rolling windows) for gridded 2 m temperature forecasts at eight lead times from 6 hours to 10 days, using ERA5 at 5.625 degree resolution, a deliberately modest convolutional forecaster and ECMWF's operational HRES, with calibration in 2017-2018 and testing in 2019-2021. Alongside coverage, width and interval score, I report dependence diagnostics: a spatial overdispersion index, quantiles of daily spatial coverage and miss persistence relative to a shuffled-days null. Per-cell methods reach near-nominal marginal coverage, yet the day-to-day variance of the fraction of cells covered is roughly 3 to 10 times what independent cells would give, and misses cluster in time; both effects grow with lead time for the CNN, also appear for HRES, and are shared by all methods. An oracle that divides each day's normalised errors by that day's mean removes 81 to 92 percent of the excess dispersion for the CNN and 93 to 98 percent for HRES, so days of large errors across much of the grid account for most of it. A day-scale predicted from seasonal terms and already-verified errors lowers the overdispersion index without widening intervals but improves interval score by at most 0.6 percent; on HRES from 12 hours to 3 days it has the lowest overdispersion index while ACI or season-conditional calibration has the best interval score, so no method is best on both. The study covers one variable, one coarse grid, a reanalysis target and two forecasters.</span> <span class="abstract-toggle" data-id="2611.24329">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.24329v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.24329v1) · [:fontawesome-brands-github: Code](https://github.com/Mariyaaleem/conformal-wx) · [:material-content-copy: BibTeX](../../bibtex/2611.24329.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=coarse" data-tag="coarse">Coarse (≥1°)</a> <a class="md-tag" href="/explore/?t=daily" data-tag="daily">Daily</a>
    { .paper-tags }

-   #### Comparing Data-Driven and Numerical Weather Prediction Forecasts of Extreme Weather: A Review { #2611.23863 }

    *Gabriele Messori, Hiu-Wang Lau, Chenyu Dong, Gianmarco Mengaldo, Leonardo Olivetti* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.23863">Data-driven weather forecasting models, chiefly trained on reanalysis data, now match or beat physics-based numerical weather prediction (NWP) models on most headline scores, but whether this...</span><span class="abstract-full" id="full-2611.23863" hidden>Data-driven weather forecasting models, chiefly trained on reanalysis data, now match or beat physics-based numerical weather prediction (NWP) models on most headline scores, but whether this advantage extends to extreme weather is contested. We review peer-reviewed studies that directly compare data-driven forecasts with NWP for extremes, from nowcasting to subseasonal lead times. We discuss grey literature in the outlook but do not include it in the formal review. There is good agreement on the fact that data-driven models mostly outperform NWP on temperature extremes, tropical cyclone tracks and the location of synoptic features across time scales. The evidence is mixed for extreme precipitation and feature-based extreme event detection such as extremes related to atmospheric rivers, while it leans toward better NWP performance for estimating event intensity. For record-breaking events that lie outside the training distribution, the evidence is narrower. Nonetheless, both a systematic global evaluation and several case studies find that physics-based models retain the advantage, with data-driven models underestimating both the frequency and the magnitude of records. Hybrid systems that post-process NWP with machine-learning (ML), or couple data-driven forecasts with physics-based models, deliver some of the largest and most consistent gains. We argue that headline scores, verification choices and case selection explain much of the apparent disagreement in the literature, and provide recommendations for a more trustworthy and interpretable evaluation of data-driven models for extreme and record-breaking weather.</span> <span class="abstract-toggle" data-id="2611.23863">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.23863v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.23863v1) · [:material-content-copy: BibTeX](../../bibtex/2611.23863.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=extremes" data-tag="extremes">Extremes</a> <a class="md-tag" href="/explore/?t=subseasonal-to-seasonal" data-tag="subseasonal-to-seasonal">Subseasonal to seasonal</a> <a class="md-tag" href="/explore/?t=evaluation" data-tag="evaluation">Evaluation</a>
    { .paper-tags }

-   #### Quantifying the added value of human forecaster discussions for tropical cyclone intensity prediction { #2611.21602 }

    *Ilia Azizi, Frederick Iat-Hin Tam, Marc-Olivier Boldi, Valérie Chavez-Demoulin, Tom Beucler* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.21602">The National Hurricane Center routinely releases Tropical Cyclone Discussions that offer insights into its tropical cyclone forecasts. This study examines the added value of these discussions for...</span><span class="abstract-full" id="full-2611.21602" hidden>The National Hurricane Center routinely releases Tropical Cyclone Discussions that offer insights into its tropical cyclone forecasts. This study examines the added value of these discussions for intensity forecasting beyond a set of numerical predictors. For 637 Atlantic and eastern North Pacific tropical cyclones from the 2002-2023 hurricane seasons, we analyze the discussion text and its combination with 13 hurricane environmental predictors from SHIPS and ERA5 10-meter wind fields, for forecasting intensity changes at lead times of 24 to 120 hours using three modeling methods. In one approach, an end-to-end deep learning framework for the text shows that the discussions alone offer comparable value for 24-hour intensity forecasts to the SHIPS predictors. Adding discussion information to the SHIPS predictors in an ensemble of five such models reduces the mean absolute error of 24-hour maximum sustained wind forecasts for 2022-2023 cyclones from 9.89 to 8.90 kt, with a confidence interval that excludes zero when whole storms are resampled. The ERA5 wind fields have a minor effect on mean absolute error in wind predictions, and model rankings differ across predicted variables, lead times and model types. We provide prediction intervals for these forecasts, whose coverage falls below the 90% target during rapid intensification. Analysis of the text shows that words related to intensity and intensity change (such as intensify, weaken, eyewall and category) are important for the model predictions. This retrospective study shows the added value of the forecasters' discussions and suggests that multi-modal learning can help understand the predictive contribution of expert human judgment. The reanalysis data differ from real-time analyses, and the discussions are issued at least three hours after the valid time of the other inputs, so these results should not be interpreted operationally.</span> <span class="abstract-toggle" data-id="2611.21602">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.21602v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.21602v1) · [:material-content-copy: BibTeX](../../bibtex/2611.21602.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a>
    { .paper-tags }

-   #### Run it Back: Autoregression vs. Video Diffusion for Final-Conditioned Weather Trajectories { #2611.21521 }

    *Jerry Lin, Elizabeth A. Barnes* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.21521">Low-likelihood, high-impact extremes are projected to increase in severity and frequency in a warming climate. Sampling the precursors that lead to these events traditionally requires massive...</span><span class="abstract-full" id="full-2611.21521" hidden>Low-likelihood, high-impact extremes are projected to increase in severity and frequency in a warming climate. Sampling the precursors that lead to these events traditionally requires massive ensembles that scale poorly with lead time and rarity, and typical rare-event sampling and ensemble-boosting methods do not show how a single extreme might arise from diverse synoptic precursors. Recently, Lin et al.\ (2026) showed this is possible with final-conditioned 66-hour trajectories (trajectories constrained to end at an observed state) from cBottle-video, a pre-trained, off-the-shelf video diffusion climate emulator not explicitly fine-tuned for this task (Brenowitz et al., 2025). However, its ensembles were underdispersed and geostrophically inconsistent, and it reproduces conditioned frames only approximately, hindering autoregressive rollout. Here, we train Rewind-AR, a backward-in-time probabilistic autoregressive model with a FourCastNet~3 backbone (Bonev et al., 2025), and generate final-conditioned ensembles for the same three cases: the 2021 Pacific Northwest heatwave, Hurricane Ian, and Superstorm Sandy (in the Supplementary Information). The leading modes of Rewind-AR's antecedent 500 hPa geopotential height explain far more of the across-member variance than cBottle-video's for both hurricane position and heatwave surface temperature, and Rewind-AR better reproduces the day-over-day heatwave intensification observed in ERA5. Finally, geostrophic inconsistencies close to the conditioning frame seen in cBottle-video are not reproduced by Rewind-AR. Nevertheless, up to 22% of Rewind-AR members become numerically unstable within 66 hours, so improving rollout physical consistency and stability over long lead times is a natural next step.</span> <span class="abstract-toggle" data-id="2611.21521">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.21521v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.21521v1) · [:material-content-copy: BibTeX](../../bibtex/2611.21521.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### Improving Tropical Cyclone Forecasts through Objective Ensemble Track-Error Reduction { #2611.20251 }

    *Shuochen Wang, Mark S. Allen, Auroop R. Ganguly* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.20251">Accurate tropical cyclone (TC) track guidance is essential for warnings, evacuation planning, and risk communication, but outlying members and distinct track scenarios can degrade raw ensemble...</span><span class="abstract-full" id="full-2611.20251" hidden>Accurate tropical cyclone (TC) track guidance is essential for warnings, evacuation planning, and risk communication, but outlying members and distinct track scenarios can degrade raw ensemble forecasts. Statistical and machine-learning (ML) methods proposed to improve ensemble guidance are difficult to compare because previous studies often use different forecast systems, basins, periods, and verification methods. Using a common verification framework, this study compares six objective methods (four statistical trimming and two ML methods) with the full ensemble mean for 138 TCs forecast by GEFS, MOGREPS-G, and IFS during 2016-2024. Dynamic trimming (DT) is the most consistent statistical method, reducing mean total track error by 8.0%, 13.4%, and 16.1% at short, medium, and long ranges. Its largest benefits occur in high-spread cases, where it can reach 70% of the low-error bound. Short-lead-error trimming (SLET) provides useful delayed guidance after early observations become available, but its improvement decreases with lead time. Regression-mixture clustering (RC) shows potential in low-spread cases and provides the largest improvement over the control run. The gradient boosted tree (GBT) produces the largest medium-range improvement and exceeds the theoretical trimming bound, suggesting strong potential, particularly when the observed track is poorly represented by the ensemble family. However, its 12.8% short-range degradation and gains concentrated in a small subset of cases indicate overcorrection when the ensemble mean is already accurate, highlighting the need for additional constraints. These results provide a consistent basis for using objective error-reduction methods to strengthen TC track guidance for operational decision-making.</span> <span class="abstract-toggle" data-id="2611.20251">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.20251v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.20251v1) · [:material-content-copy: BibTeX](../../bibtex/2611.20251.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=classical-ml" data-tag="classical-ml">Classical ML</a> <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### An AI-enabled high-resolution chlorophyll-a forecasting model that outperforms the operational global biogeochemical system { #2611.17704 }

    *Yong Jiang, Zheng Jiang, Gaowei Zhang, Zengzhou Hao, Suixiang Shi, Chi Zhang, Wei Wang* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.17704">Surface chlorophyll-a concentration is a key indicator of marine phytoplankton variability, and short-term forecasts of its distribution provide advance information for marine ecosystem monitoring...</span><span class="abstract-full" id="full-2611.17704" hidden>Surface chlorophyll-a concentration is a key indicator of marine phytoplankton variability, and short-term forecasts of its distribution provide advance information for marine ecosystem monitoring and management. However, interactions between physical transport and biological processes complicate chlorophyll-a forecasting, and simulating these coupled processes with conventional numerical models imposes substantial computational demands. Here we introduce ChlCast, an AI model that generates daily forecasts of surface chlorophyll-a concentration over a 10-day horizon at approximately 4-km resolution between 60$^\circ$S and 60$^\circ$N. ChlCast combines factorized spatiotemporal convolutions with a multiscale architecture to learn from historical chlorophyll-a fields and climatological information. Compared with the operational 0.25$^\circ$ Copernicus global ocean biogeochemical forecasting system, ChlCast reduces mean absolute error and root mean square error by 70.74% and 47.63%, respectively, against satellite-derived Level-4 chlorophyll-a fields, and by 35.03% and 13.07% against independent BGC-Argo in situ observations. Coastal case studies further show that the 4-km forecasts retain coastal gradients and localized concentration maxima that are weakened by coarse-grid averaging. These results demonstrate the potential of observation-driven deep learning for high-resolution forecasting of surface chlorophyll-a variability at the near-global scale.</span> <span class="abstract-toggle" data-id="2611.17704">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.17704v2) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.17704v2) · [:material-content-copy: BibTeX](../../bibtex/2611.17704.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a> <a class="md-tag" href="/explore/?t=km-scale" data-tag="km-scale">Km-scale</a> <a class="md-tag" href="/explore/?t=daily" data-tag="daily">Daily</a>
    { .paper-tags }

-   #### DaYu-TC: a global-regional nested AI system for tropical cyclone forecasting { #2611.15266 }

    *Runzepu Hou, Feng Zhang, Xin Wu, Xujun Wei, Fenghua Ling, Kan Dai, Gaozhen Nie, Leiming Ma et al.* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.15266">Accurate tropical cyclone intensity forecasts are essential for disaster preparedness, yet persistent intensity underestimation limits the value of artificial intelligence (AI) weather models for...</span><span class="abstract-full" id="full-2611.15266" hidden>Accurate tropical cyclone intensity forecasts are essential for disaster preparedness, yet persistent intensity underestimation limits the value of artificial intelligence (AI) weather models for predicting these destructive storms. Here we present DaYu-TC, a global-regional nested system that links the preservation of strong-wind and low-pressure signals with high-resolution regional prediction. Its global and regional models share an architecture and extreme-aware training objectives. The regional model combines evolving local states with global forecasts valid at each target time, connecting large-scale environmental information with finer-scale cyclone evolution. In an independent evaluation of 16 western North Pacific tropical cyclones in 2023, the global model retains competitive conventional forecast skill while reducing five-day maximum-wind root-mean-square error (RMSE) by 6-22% relative to other evaluated global AI models. Relative to the global model, regional nesting reduces maximum-wind RMSE by 22.0-33.1% and minimum-pressure RMSE by 28.6-39.8% across 12-120 h. The complete system outperforms concurrent operational high-resolution forecasts from the European Centre for Medium-Range Weather Forecasts for minimum pressure at 12-120 h and maximum wind speed at 48-96 h, with respective RMSE reductions of 26% and 12% at 72 h. Without retraining, the same regional model enhances three other global AI models, reducing overall maximum-wind RMSE by 22.6-28.6%. Five-day forecasts take under one minute on a single GPU. These results establish a transferable multiscale approach to reducing cyclone intensity underestimation, extending advances in AI weather forecasting towards more accurate prediction of high-impact weather events.</span> <span class="abstract-toggle" data-id="2611.15266">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.15266v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.15266v1) · [:material-content-copy: BibTeX](../../bibtex/2611.15266.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a> <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a>
    { .paper-tags }

-   #### PhysXAI: A Theoretical Framework for Diagnosing Physical Hallucinations in Weather Foundation Models { #2611.15053 }

    *Anjan Mahapatra, Nikhil Raj* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.15053">Modern machine learning models for global weather forecasting achieve remarkable predictive accuracy, yet they frequently produce states that violate fundamental physical laws, including conservation...</span><span class="abstract-full" id="full-2611.15053" hidden>Modern machine learning models for global weather forecasting achieve remarkable predictive accuracy, yet they frequently produce states that violate fundamental physical laws, including conservation of mass, hydrostatic balance, or spectral energy scaling. We term these outputs physical hallucinations and propose PhysXAI, a rigorous diagnostic framework that detects, localises, and attributes such inconsistencies. Unlike existing post-hoc metric suites, PhysXAI is a learnable system that explicitly decouples forecast error from physical plausibility. We formalise the problem through a novel joint condition on numerical and physical residuals, and introduce a multi-branch encoder that extracts representations from conservation, dynamical, spectral, cross-variable, and spatiotemporal residuals. A key innovation is a physics-consistency contrastive learning objective, which we theoretically show the latent representation to be approximately isometric to the physical residual space, ensuring that diagnostic decisions remain grounded in physically meaningful deviations. We further derive an adaptive multi-objective loss with gradient-based weighting to balance competing diagnostic objectives. The framework produces spatially resolved hallucination maps, violation-type classifications, and calibrated Bayesian uncertainty, enabling both detection and interpretable attribution of physical inconsistencies. We conduct a comprehensive comparative evaluation of PhysXAI across ERA5, WeatherBench 2, and MERRA-2, using forecasts from GraphCast, Pangu-Weather, and NeuralGCM, with Python-written code and a fully reproducible experimental protocol. Our theoretical analysis and empirical evaluation establish PhysXAI as a principled framework for diagnosing physical hallucinations and advancing trustworthy AI for weather forecasting.</span> <span class="abstract-toggle" data-id="2611.15053">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.15053v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.15053v1) · [:material-content-copy: BibTeX](../../bibtex/2611.15053.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=foundation-models" data-tag="foundation-models">Foundation models</a> <a class="md-tag" href="/explore/?t=physics-ml-hybrid" data-tag="physics-ml-hybrid">Physics–ML hybrid</a> <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a>
    { .paper-tags }

-   #### One2N: Recursive refinement breaks the resolution barrier for Earth-system AI { #2611.12249 }

    *Yanyan Tan, Xiangjun Tian, Can Wang, Hongqin Zhang, Yinhai Luo, Min Zhao, Zhe Jin, Xingchao Yang et al.* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.12249">Earth-system AI models approach kilometre-scale resolution, but halving grid spacing quadruples the grid-point count and end-to-end training cost. We introduce One2N, a recursive incremental...</span><span class="abstract-full" id="full-2611.12249" hidden>Earth-system AI models approach kilometre-scale resolution, but halving grid spacing quadruples the grid-point count and end-to-end training cost. We introduce One2N, a recursive incremental framework extending a coarse model through $N$ levels of cross-resolution residual correction. At each level the parent is frozen and excluded from backpropagation, so its activations and optimizer states are never retained; only a tier-specific module is trained. A single refinement level on the NeuralGCM-0.7$^\circ$ baseline yields the two-level One2N-0.07$^\circ$, trained on one 80 GB GPU. Against HRES-0.07$^\circ$ and ERA5-0.25$^\circ$ analyses it matches its parent's tropospheric skill while reducing upper-stratospheric errors and resolving finer structure. Tropical-cyclone track errors nearly match the parent's and are lower than Pangu-Weather-0.25$^\circ$ at medium range; RMSE is lower than that model's on 93.46% of common targets. One2N thus turns an existing forecasting system, or an inexpensively trained coarse baseline, into a reusable foundation for high-resolution development, breaking the resolution barrier.</span> <span class="abstract-toggle" data-id="2611.12249">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.12249v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.12249v1) · [:material-content-copy: BibTeX](../../bibtex/2611.12249.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=km-scale" data-tag="km-scale">Km-scale</a>
    { .paper-tags }

-   #### Ensemble Covariance Propagation with Machine Learning Weather Prediction Models in a Realistic Data Assimilation System { #2611.11918 }

    *Simon Toedtli, David Ahijevych, Wei Wang, Chris Snyder* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.11918">Integrating machine learning weather prediction (MLWP) models into classical data assimilation (DA) workflows requires that they accurately evolve ensemble covariances. We evaluate this capability in...</span><span class="abstract-full" id="full-2611.11918" hidden>Integrating machine learning weather prediction (MLWP) models into classical data assimilation (DA) workflows requires that they accurately evolve ensemble covariances. We evaluate this capability in a two-way coupled DA system, where an ensemble variational (EnVar) component evolves the mean and an ensemble Kalman filter (EnKF) evolves the covariances. The system is cycled over one month and assimilates a wide range of real-world observations, including microwave radiances. Using the MLWP model GraphCast for the EnKF and the physics-based MPAS model for the EnVar yields ensemble spread and covariances comparable to a reference configuration in which MPAS evolves both components. The resulting EnVar mean achieves consistently lower RMSE against GFS analyses, except near the model top, where spurious correlations degrade performance. These results demonstrate that MLWP models can propagate covariance information in a realistic DA cycling system and provide a proof of concept for their integration into future operational ensemble systems.</span> <span class="abstract-toggle" data-id="2611.11918">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.11918v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.11918v1) · [:material-content-copy: BibTeX](../../bibtex/2611.11918.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### SWXAI: A Unified Multimodal Explainability Library for Space Weather Forecasting and Heliophysics { #2611.10861 }

    *Benedict Antonio Mervyn, Anli Ji, Manolis K. Georgoulis, Berkay Aydin* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.10861">Space weather forecasting and heliophysics increasingly rely on machine learning models that process heterogeneous inputs such as multi-viewpoint multi-wavelength solar images and rasters,...</span><span class="abstract-full" id="full-2611.10861" hidden>Space weather forecasting and heliophysics increasingly rely on machine learning models that process heterogeneous inputs such as multi-viewpoint multi-wavelength solar images and rasters, multivariate time series, spatial annotations, and tabular event-level data. However, explainability workflows remain fragmented across libraries, model interfaces, and output formats, limiting reuse and reproducibility. We present SWXAI, an open-source library that unifies explainable AI (XAI) workflows for image, time-series, and tabular data in space weather forecasting and heliophysics. SWXAI provides a common, extensible interface across commonly used vision, time-series, and tabular learning models, with validation, adaptation, and persistent explanation objects. It also includes domain-grounded examples, reproducible notebooks, tests, and documentation, enabling users to inspect outputs, swap models or explainers, and extend workflows to new heliophysics datasets. SWXAI reduces modality-specific engineering overhead and supports comparable, reusable explanations across heterogeneous scientific machine learning pipelines. A benchmark against direct backend implementations shows that SWXAI substantially reduces the code users write to produce an explanation, with no practical runtime cost.</span> <span class="abstract-toggle" data-id="2611.10861">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.10861v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.10861v1) · [:material-content-copy: BibTeX](../../bibtex/2611.10861.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=interpretability" data-tag="interpretability">Interpretability</a>
    { .paper-tags }

-   #### Half-precision inference of AI weather prediction models: negligible loss of skill at a fraction of the cost { #2611.09573 }

    *Stefan Siegert* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.09573">AI weather prediction models are released with 32-bit (fp32) weights and run in fp32 by default, although current GPUs compute much faster in 16-bit formats. We run five publicly available models --...</span><span class="abstract-full" id="full-2611.09573" hidden>AI weather prediction models are released with 32-bit (fp32) weights and run in fp32 by default, although current GPUs compute much faster in 16-bit formats. We run five publicly available models -- Aurora, Pangu-Weather, FourCastNet, SFNO and FastNet -- on a single 16 GB GPU from 52 initial dates in 2024, in fp32 and with their networks in half precision (fp16), and verify the forecasts against ERA5. The physical state must stay in fp32: converting the whole model rounds it in physical units, which in fp16 overflows and in bf16 discards meaningful information. With the state, its normalisation and denormalisation in fp32, running the network in fp16 changes the global RMSE by at most 1.4%, and by about 0.5% or less for four of the five models, equivalent to less than one hour of lead time up to day 8 and at most two hours at day 10. The differences show no systematic pattern by region, scale or for tropical and extratropical storms, apart from a small tropical penalty in FourCastNet, and fp16 leaves the CRPS, spread and calibration of Aurora's ensemble almost unchanged. No value overflowed fp16's numeric range (+/-65,504); the largest stay 38 to 544 times below it. A 10-day forecast in fp16 is 1.3-3.5 times faster than in fp32 and uses 1.4-3.7 times less energy. Sensitivity experiments show that rules of thumb from other applications of reduced precision carry over only partly: the first and last layers dominate for some models but not for others, and no type of layer is sensitive on its own. For running AI weather models on a single GPU, fp16 with the state in fp32 is practically lossless at a fraction of the cost. bf16, often preferred for its wider range, is as fast but less accurate than fp16 for every model, and simple 8-bit quantisation costs far more skill.</span> <span class="abstract-toggle" data-id="2611.09573">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.09573v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.09573v1) · [:fontawesome-brands-github: Code](https://github.com/sieste/ai-nwp-fp16) · [:material-content-copy: BibTeX](../../bibtex/2611.09573.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=neural-operators" data-tag="neural-operators">Neural operators</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=efficiency" data-tag="efficiency">Efficiency</a>
    { .paper-tags }

-   #### Distinguishing Initial-Condition and Model Error Using Machine-Learning Weather Models { #2611.05592 }

    *P. Trent Vonich, Gregory J. Hakim* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.05592">Differentiable weather models allow atmospheric initial conditions to be directly optimized to shadow a known trajectory, which we take from a reanalysis dataset. Here, we extend this optimization to...</span><span class="abstract-full" id="full-2611.05592" hidden>Differentiable weather models allow atmospheric initial conditions to be directly optimized to shadow a known trajectory, which we take from a reanalysis dataset. Here, we extend this optimization to include the forecast model, GraphCast, using low-rank adaptation (LoRA). We compare model optimization with initial-condition (IC) optimization, and then joint optimization, where both are tuned simultaneously. When compared against a control forecast initialized with reanalysis data, LoRA and IC optimization individually produce similar improvements. We find similar results when IC and LoRA are jointly optimized across a spectrum of solutions where a tuning coefficient smoothly shifts the weight of the joint optimization from all-LoRA to all-IC; forecast improvement is not unique to the relative contributions of model and IC corrections. To better discriminate the source of error, we introduce a second model, Pangu-Weather, as a supervisor that gates IC updates during joint optimization. This gated optimization is the same as before, except that IC updates at each epoch are accepted only when they improve the forecasts of both models; otherwise, only the LoRA update is accepted. Results show that this cross-model constraint substantially improves agreement between the two models in their response to the same optimized ICs, narrowing the gap in mean forecast error reduction from 64% to 10%. Moreover, when used together, the LoRA and IC corrections from the gated optimization yield GraphCast forecast improvements similar to those from the ungated optimization. These results imply that model error that is encoded in the ungated ICs is shifted to the LoRA corrections via gating, yielding IC corrections that may provide better analysis estimates.</span> <span class="abstract-toggle" data-id="2611.05592">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.05592v2) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.05592v2) · [:material-content-copy: BibTeX](../../bibtex/2611.05592.bib){ .bibtex-link }
    { .paper-links }

-   #### Sonny-ENS: demystifying machine-learning ensemble weather forecasting { #2611.01614 }

    *Minjong Cheon* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.01614">Every weather forecast is uncertain, and decisions from flood warnings to energy trading depend on how well that uncertainty is quantified. Machine-learning (ML) ensembles now quantify it better than...</span><span class="abstract-full" id="full-2611.01614" hidden>Every weather forecast is uncertain, and decisions from flood warnings to energy trading depend on how well that uncertainty is quantified. Machine-learning (ML) ensembles now quantify it better than the best physics-based ensemble, yet two things about them remain opaque: whether their skill requires hundreds to thousands of accelerator-days of training, and how a network with a simple source of randomness produces a spread that follows the flow. Here we introduce Sonny-ENS, an ensemble of three 68.9-million-parameter networks with Monte Carlo dropout, trained on the continuous ranked probability score (CRPS) in 2.7 A40 GPU-days. On 2022, it has a lower CRPS than the European Centre for Medium-Range Weather Forecasts ensemble (ENS) on 96% of 570 targets and an 8.3% lower mean CRPS at lead times of 1-10 days. It outperforms the operational GenCast on 69% of the targets, and a 51-member, 15-day forecast takes 69 s on one GPU. Ablations identify the ingredients of this skill, and a Kolmogorov-Arnold sparse autoencoder shows that dropout injects uncertainty mainly in the tropics, whereas the spread grows where the atmosphere is baroclinically unstable (spatial correlation 0.91 with the Eady growth rate). Skillful probabilistic weather forecasts are therefore within reach of a single GPU server.</span> <span class="abstract-toggle" data-id="2611.01614">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.01614v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.01614v1) · [:material-content-copy: BibTeX](../../bibtex/2611.01614.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### Value of Adding AI Weather Model Ensembles to a Multi-model Probabilistic Blend { #2610.42130 }

    *Harris Sop Nkuiate, Gavin Evans, Christopher A. T. Ferro, Frank Kwasniok, Caroline Bain et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.42130">Many weather forecasting organizations use a multi-model blend of ensemble members from different prediction systems to obtain a more accurate forecast. Recent studies have shown that blending...</span><span class="abstract-full" id="full-2610.42130" hidden>Many weather forecasting organizations use a multi-model blend of ensemble members from different prediction systems to obtain a more accurate forecast. Recent studies have shown that blending deterministic AI-based models alongside physics-based models improves the overall forecast skill. However, the incorporation of an ensemble AI-based model into a multi-model blend is yet to be investigated. This study explores the value of blending the ensemble version of the Artificial Intelligence Forecasting System (AIFS-CRPS) with the Met Office's convection permitting physics-based ensemble, MOGREPS-UK. We show that blending site temperature (and wind speed) forecasts over the United Kingdom from both ensemble physics-based and ensemble AI-based models leads to an increase in overall forecast skill spatially, across lead times and across exceedance thresholds, especially when the models have different error characteristics. By decomposing the continuous ranked probability score of the blend into component means and diversity terms, we show that a more accurate blend can be achieved by assigning more weight to more skillful models and using models with complementary errors. For example, we show that a blend of MOGREPS-UK with either AIFS-CRPS or the physics-based Integrated Forecasting System (IFS) is better than a blend of IFS and AIFS-CRPS which are not very diverse. This has implications for the importance of weather model diversity and may lend weight to different model development approaches being pursued in tandem. We also demonstrate that calibrating AIFS-CRPS forecasts yields an increase in forecast skill comparable to calibrated ensemble physics-based forecasts, and calibrating the source forecasts before blending and the blended forecast after blending improves the overall forecast accuracy.</span> <span class="abstract-toggle" data-id="2610.42130">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.42130v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.42130v1) · [:material-content-copy: BibTeX](../../bibtex/2610.42130.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=evaluation" data-tag="evaluation">Evaluation</a>
    { .paper-tags }

-   #### Scale-aware and spatially-localised optimisation of kilometre-scale MLWP models { #2610.37325 }

    *James Chappell, Sophie Luise Arana, Jack Bowyer, Alejandro Coca-Castro, Evangeline Corcoran et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.37325">Preserving fine-scale structure without degrading spatial accuracy is a central challenge for kilometre-scale machine-learning weather prediction (MLWP). We introduce a loss based on the dual-tree...</span><span class="abstract-full" id="full-2610.37325" hidden>Preserving fine-scale structure without degrading spatial accuracy is a central challenge for kilometre-scale machine-learning weather prediction (MLWP). We introduce a loss based on the dual-tree complex wavelet transform that constrains atmospheric fields jointly by scale, location, and orientation. We evaluate its performance using an experimental joint global--regional FastNet configuration, which produces 1.5 km forecasts over the UK alongside $\sim$31 km global forecasts. Relative to both mean-squared error (MSE) and an FFT-based spectral objective, the wavelet loss substantially improves spectral fidelity. Compared with MSE training, it maintains similar pointwise performance, improves neighbourhood-based verification, and produces sharper precipitation fields with more realistic intensity distributions. These results demonstrate that spatially localised wavelet constraints can preserve multiscale atmospheric structure while largely avoiding the loss of pointwise accuracy associated with globally supported spectral objectives.</span> <span class="abstract-toggle" data-id="2610.37325">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.37325v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.37325v1) · [:material-content-copy: BibTeX](../../bibtex/2610.37325.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a> <a class="md-tag" href="/explore/?t=km-scale" data-tag="km-scale">Km-scale</a>
    { .paper-tags }

-   #### GNN-VIEW: Visual Inspection and Explanation of Graph-based Earth-System Models { #2610.35693 }

    *Alessandro Lucatelli, Ferdinand Unterhuber, Katharina Krenzer, Mennatallah El-Assady, Benedikt Soja et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.35693">Machine learning Earth-system models are becoming increasingly important, and so is their development. Graph-based models, one of the most prominent approaches, allow developers to constrain the...</span><span class="abstract-full" id="full-2610.35693" hidden>Machine learning Earth-system models are becoming increasingly important, and so is their development. Graph-based models, one of the most prominent approaches, allow developers to constrain the learning process before training via graph connectivity. Yet, developing these models remains challenging: the graph is built algorithmically with little intuition about the final result beyond static plotting, which requires extra work and is often not detailed enough. Even after successful training, how the model arrives at forecasts remains obscure. We present GNN-VIEW, an interactive browser-based tool that supports the development of such systems. GNN-VIEW exposes the graph's information pathways, allowing model developers to debug the graph structure before training. After training, GNN-VIEW uses attribution-based explainability to reveal which input features are used by the trained model for a given forecast. This helps developers understand why a model fails. We showcase the usefulness of GNN-VIEW in the construction stage of a graph for an ocean forecasting model, and in explaining poorly predicted precipitation of a trained weather forecasting model during Hurricane Laura.</span> <span class="abstract-toggle" data-id="2610.35693">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.35693v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.35693v1) · [:material-content-copy: BibTeX](../../bibtex/2610.35693.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=graph-neural-networks" data-tag="graph-neural-networks">Graph neural networks</a> <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a>
    { .paper-tags }

-   #### Can We Trust Evaluations of AI Weather Forecasts for Extremes? How Evaluation Protocols Shape Model Rankings { #2610.31237 }

    *Tobias Biegert, Sam Allen, Zhongwei Zhang, Sebastian Lerch* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.31237">AI weather models perform competitively with traditional physics-based models at a fraction of the computational cost, and they are thus already used operationally. However, studies reach conflicting...</span><span class="abstract-full" id="full-2610.31237" hidden>AI weather models perform competitively with traditional physics-based models at a fraction of the computational cost, and they are thus already used operationally. However, studies reach conflicting conclusions about their ability to forecast extreme weather events, and comparisons across studies are complicated by differing evaluation protocols. We discuss the challenges that arise when comparing AI and physics-based weather forecasts for extreme events. Within a common experimental design, we find that model rankings are highly sensitive to the choice of evaluation metrics: the considered physics-based model tends to perform best when evaluation is restricted to observed extreme events, whereas AI models generally perform better when evaluated using weighted scoring functions. A contingency table analysis shows that AI models underpredict the frequency of extremes, so that their scores are generally dominated by misses, while the physics-based model forecasts extremes more frequently and its scores are dominated by false alarms. Different evaluation metrics penalise these errors differently. Further score decompositions confirm that the score differences arise primarily from miscalibration rather than discrimination ability. This sheds light on the contrasting results in the literature, and highlights the importance of the evaluation metrics used to evaluate AI models.</span> <span class="abstract-toggle" data-id="2610.31237">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.31237v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.31237v1) · [:material-content-copy: BibTeX](../../bibtex/2610.31237.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=extremes" data-tag="extremes">Extremes</a> <a class="md-tag" href="/explore/?t=evaluation" data-tag="evaluation">Evaluation</a>
    { .paper-tags }

-   #### From Weather to Enhanced Weathering: Weather Foundation Models for Carbon Dioxide Removal at Operational Scale { #2610.28973 }

    *Sasankh Munukutla, Felipe Oviedo, Danrong Zhang, Kenji Takeda, Juan M Lavista Ferres* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.28973">Enhanced Rock Weathering (ERW) is a carbon removal solution with the potential to remove gigatons of CO$_2$ per year, but quantifying that removal today requires dense soil sampling and laboratory...</span><span class="abstract-full" id="full-2610.28973" hidden>Enhanced Rock Weathering (ERW) is a carbon removal solution with the potential to remove gigatons of CO$_2$ per year, but quantifying that removal today requires dense soil sampling and laboratory analysis. We evaluate five weather Foundation Models (FMs) as frozen encoders feeding lightweight decoders, under a leave-one-field-out protocol on 5,192 chronosequences from 264 fields across Brazil and the US. To our knowledge, this is the first holistic evaluation of weather FMs for ERW and the largest evaluation on operational ERW data. We find that across weather FMs, representations transfer to ERW: our best model, TerraNova-ERW, built on Aurora 1.5, positively orders measurements within fields it has never seen. Zero-shot prediction of a new field's mean removal is not yet feasible, and we attribute this to data scale: the same architecture excels on $\sim$$10^5$ natural-weathering sequences, anchoring the scale we expect ERW data must reach. Few-shot prediction, blending the model with a few samples per field, is supported today: TerraNova-ERW recovers over 60% of the highest-removal fields with two samples per field, roughly halves the samples needed to estimate a field's mean removal, and can cut sampling on our densest fields by 28-98%.</span> <span class="abstract-toggle" data-id="2610.28973">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.28973v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.28973v1) · [:material-content-copy: BibTeX](../../bibtex/2610.28973.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=foundation-models" data-tag="foundation-models">Foundation models</a>
    { .paper-tags }

-   #### Tropical Cyclone Attributes in Sparse Autoencoder Features of an AI Weather Model { #2610.28904 }

    *Jiapei Liu, Ruidong Li, Ting Sun, Guangheng Ni* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.28904">Artificial intelligence (AI) weather models now forecast weather events skillfully, but how they represent these events internally remains unknown. Sparse autoencoders (SAEs) provide sparse features...</span><span class="abstract-full" id="full-2610.28904" hidden>Artificial intelligence (AI) weather models now forecast weather events skillfully, but how they represent these events internally remains unknown. Sparse autoencoders (SAEs) provide sparse features for probing hidden states, but how information about multiple physical attributes of an event is organized across them remains unclear. Using tropical cyclones in GraphCast as a test case, we analyze 2,115 six-hourly timesteps from 205 cyclones, relating SAE activations to position, intensity, and pressure structure across five Layer-8 decompositions and 16 processor layers. Each attribute relates to multiple features; position and pressure-structure relations overlap in a smaller subset. This organization recurs across independent SAEs and layers without feature alignment. In deeper layers, the leading position feature aligns better with position and pressure structure six hours ahead than at present. These results suggest that the organization of physical-attribute relations provides a reproducible level of description for interpreting cyclone-related information in GraphCast.</span> <span class="abstract-toggle" data-id="2610.28904">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.28904v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.28904v1) · [:material-content-copy: BibTeX](../../bibtex/2610.28904.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a> <a class="md-tag" href="/explore/?t=6-hourly" data-tag="6-hourly">6-hourly</a>
    { .paper-tags }

-   #### GraphCast-Mamba: Learning Non-Markovian Corrections for Weather Forecasting { #2610.28512 }

    *Ilya Vilkoviskiy, Liang-Hong Mo, Suozhi Huang, Dmitry A. Abanin* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.28512">Unresolved atmospheric processes can introduce history dependence beyond the short input windows used by many weather forecasting models. We introduce GraphCast--Mamba, which augments a frozen,...</span><span class="abstract-full" id="full-2610.28512" hidden>Unresolved atmospheric processes can introduce history dependence beyond the short input windows used by many weather forecasting models. We introduce GraphCast--Mamba, which augments a frozen, pretrained $1^\circ$ GraphCast backbone with a trainable Mamba correction branch combining graph message passing and Mamba layers. The branch retains a temporal memory state across successive six-hour forecast steps, learning corrections informed by temporal context beyond GraphCast's two-frame input. Training is confined to the correction branch, using ERA5 data from 2015--2021. After 2,500 optimizer updates, including warm-up, the selected model achieves substantial improvements relative to the uncorrected GraphCast backbone. It reduces the GraphCast loss (normalized weighted mean-squared error; MSE) by 14.42% when aggregated over the full ten-day forecast and by 21.66% at day 10 (240 h after forecast initialization). At this final forecast lead time, root-mean-square error (RMSE) decreases by 7.09--14.60% across all 11 evaluated variable groups, spanning temperature, wind, pressure, geopotential, specific humidity, and precipitation. These results demonstrate broad improvements in ten-day weather forecasts through a short correction-training schedule while retaining the pretrained backbone.</span> <span class="abstract-toggle" data-id="2610.28512">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.28512v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.28512v1) · [:material-content-copy: BibTeX](../../bibtex/2610.28512.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=graph-neural-networks" data-tag="graph-neural-networks">Graph neural networks</a>
    { .paper-tags }

-   #### Beyond the record: evaluating probabilistic AI weather forecasts deep in the tail { #2610.25545 }

    *Bastiaan Van Velthoven, Cas Decancq, Diego G. Miralles, Stijn Luca, Thomas Mortier* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.25545">Can probabilistic artificial intelligence (AI) weather forecasts retain skill beyond their training domain? We compare four state-of-the-art AI models (ArchesWeatherGen, U-Cast, WeatherNext2 and...</span><span class="abstract-full" id="full-2610.25545" hidden>Can probabilistic artificial intelligence (AI) weather forecasts retain skill beyond their training domain? We compare four state-of-the-art AI models (ArchesWeatherGen, U-Cast, WeatherNext2 and GenCast-mini) with the physics-based IFS-ENS over land in 2022, at 1.5° resolution and lead times of 2--10 days. Monthly ERA5 records from 1979--2021 define common historical boundaries, and we characterize new records set in 2022 by their depth, i.e., the exceedance margin relative to local historical variability. Two proper scores, record-threshold weighted CRPS and Brier score, assess the magnitude and probability of record exceedances across all eligible cases, retaining both missed extremes and false alarms. Verified against ERA5, AI models outperform IFS-ENS for heat and cold records at almost every lead and depth. For heat, the advantage strengthens with depth, partly because IFS-ENS over-forecasts deep records, and largely persists for deeper records under HRES-fc0 verification. For wind, gains erode for deeper records and longer leads. Better tail scores can nevertheless coexist with aggregate probability deficits or surpluses, and some improvements arise chiefly on non-exceedances. Increasing underprediction on progressively more extreme observed subsets can also arise through outcome selection, even for an ideal conditional-mean forecast. These findings establish that native probabilistic AI ensembles retain, and for heat increase, their advantage over IFS-ENS as thresholds move beyond local records, although all advantages narrow with lead time. They caution against generalizing the extrapolation failure reported for deterministic AI forecasts to probabilistic systems, and motivate common verification conventions that separate comparative tail skill, performance on realized extremes, and probabilistic calibration.</span> <span class="abstract-toggle" data-id="2610.25545">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.25545v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.25545v1) · [:material-content-copy: BibTeX](../../bibtex/2610.25545.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=evaluation" data-tag="evaluation">Evaluation</a> <a class="md-tag" href="/explore/?t=coarse" data-tag="coarse">Coarse (≥1°)</a> <a class="md-tag" href="/explore/?t=monthly" data-tag="monthly">Monthly</a>
    { .paper-tags }

-   #### UtoMe: Observation-Uncertainty-Guided Token Merging for Weather Foundation Models { #2610.25001 }

    *Yunbei Zhang, Janet Wang, Xi Xiao, Jihun Hamm, Xiao Wang* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.25001">Weather transformers process dense reanalysis fields whose reliability varies across variables, locations, and weather regimes. Token merging can reduce computation, but feature similarity alone does...</span><span class="abstract-full" id="full-2610.25001" hidden>Weather transformers process dense reanalysis fields whose reliability varies across variables, locations, and weather regimes. Token merging can reduce computation, but feature similarity alone does not account for this heterogeneous input uncertainty. We introduce UtoMe, an observation-uncertainty-guided token merging framework that combines dynamics saliency with ensemble data assimilation spread as a relative input-reliability proxy. The resulting importance score protects selected tokens as anchors and calibrates bipartite matching between the remaining tokens. Three variants use external uncertainty at both training and inference, during training only, or neither stage. In ERA5 72-hour forecasting at two spatial resolutions, UtoMe improves Overall error over train-time ToMe at matched token budgets. At $1.40625^\circ$ and 50% token reduction, UtoMe-E lowers Overall error from 67.32 to 53.00 at the same 0.55x encoder FLOPs. Small reductions remain close to the full-token baseline; larger reductions incur measurable accuracy losses. Component ablations and spatial uncertainty controls support combining dynamics with aligned uncertainty when allocating a weather transformer's token budget.</span> <span class="abstract-toggle" data-id="2610.25001">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.25001v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.25001v1) · [:material-content-copy: BibTeX](../../bibtex/2610.25001.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=transformers" data-tag="transformers">Transformers</a> <a class="md-tag" href="/explore/?t=foundation-models" data-tag="foundation-models">Foundation models</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### Interpretability of AI weather models via intermediate decoding { #2610.24110 }

    *Matthias Beylich, Kirsten I. Tempest, Tobias Selz, George C. Craig* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.24110">AI weather prediction models are often regarded as black boxes. Despite their impressive skill, there is little understanding of how the input data is processed during inference to produce the...</span><span class="abstract-full" id="full-2610.24110" hidden>AI weather prediction models are often regarded as black boxes. Despite their impressive skill, there is little understanding of how the input data is processed during inference to produce the forecast. This limitation arises from the use of high-dimensional latent representations in which individual dimensions are no longer directly interpretable in physical terms. To gain insight into how forecasts are constructed, we propose the method of intermediate decoding for AI weather models and apply it to GraphCast, Aurora and ArchesWeather. In this approach, latent states from intermediate depths are decoded back into meteorological fields. To do this, we use the decoder that has been trained to output the forecast at the end of processing. Since latent states at different depths do not share a consistent basis, we train affine transformations that align intermediate latent states with the decoder's expected input space. Applying this framework to the three models yields the following results. First, the skill of forecasts at intermediate stages of processing improves gradually through the processor layers. Second, bypassing all processor blocks shows that the encoder-decoder pair is not an identity mapping returning the input, but instead plays a significant role in the forecast construction. Third, by examining increments of individual processor blocks across model depth, we show that all three models progressively refine the forecast from coarse to finer spatial scales. Finally, we assess the physical consistency of the intermediate decoding forecasts and find that large scales remain physically balanced throughout processing, while small scales become more balanced with each additional processor block.</span> <span class="abstract-toggle" data-id="2610.24110">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.24110v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.24110v1) · [:material-content-copy: BibTeX](../../bibtex/2610.24110.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=interpretability" data-tag="interpretability">Interpretability</a>
    { .paper-tags }

-   #### Can AI Forecast the Extremes? Evaluating Probabilistic Weather Models on Tropical Cyclones { #2610.22077 }

    *Nathan Chalumeau, Xavier Renard, Remi Meynadier, Hugo Rakotoarimanga, Marcin Detyniecki et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.22077">Weather prediction is undergoing an important shift, with data-driven AI models outperforming leading numerical weather prediction systems on average at a fraction of the computational cost. Yet,...</span><span class="abstract-full" id="full-2610.22077" hidden>Weather prediction is undergoing an important shift, with data-driven AI models outperforming leading numerical weather prediction systems on average at a fraction of the computational cost. Yet, whether this success extends to extreme events remains contested. Earlier evaluations have exposed important shortcomings, but have largely focused on deterministic models and metrics, leaving the capabilities of the latest probabilistic models less understood. We conduct an extensive benchmark of tropical cyclone forecasting, comparing GenCast, FourCastNet 3, and FGN with ECMWF's operational physics-based ensemble, IFS-ENS. We develop a suite of probabilistic metrics and diagnostic analyses to assess cyclone tracking, intensity biases, uncertainty, and the wind-pressure relationship that characterizes storm structure. We find that, despite remaining limitations in intensity and ensemble spread, leading AI models substantially outperform IFS-ENS in cyclone tracking: FGN achieves better probabilistic location scores for 90% of evaluated storms in 2025. Further analyses reveal model-specific failure modes and suggest a lingering influence of pretraining data, pointing to opportunities for targeted improvement. Nonetheless, these results suggest that AI models are capable of capturing physical relationships that remain predictive well beyond typical weather conditions, motivating further investigation into how these capabilities emerge and how to strengthen them.</span> <span class="abstract-toggle" data-id="2610.22077">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.22077v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.22077v1) · [:material-content-copy: BibTeX](../../bibtex/2610.22077.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### Forecast lead time and the timing of crew deployment for weather-driven power outages using AI weather models { #2610.20918 }

    *Mohammad Kian Golkar, Luciano Alves de Oliveira, Mohamad Alipour* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.20918">Utilities request mutual aid crews to restore power after severe storms, and requesting early secures more crews but relies on less accurate forecasts. We examined how forecast lead time affects the...</span><span class="abstract-full" id="full-2610.20918" hidden>Utilities request mutual aid crews to restore power after severe storms, and requesting early secures more crews but relies on less accurate forecasts. We examined how forecast lead time affects the value of outage predictions for crew deployment, using operational forecasts from two AI weather models, GraphCast and FourCastNet, to predict outage occurrence for counties across the contiguous United States 24 to 120 hours ahead. Forecasts were evaluated by the share of outage burden captured by the counties where crews would be staged. Captured burden declined with lead time: the loss between three and five days was significant in all analyses, whereas the loss within three days was significant only for outages that had not yet started. Early deployment with all requested crews outperformed late deployment with fewer, and we report the availability ratio at which the two break even. Antecedent land surface predictors computed at forecast valid time include weather from the forecast period, which inflates their skill by 42 to 53% and reverses their apparent advantage over forecast meteorology at long lead times. Computed correctly, they improved occurrence skill by 17.5% at 24 hours but did not significantly change captured burden, a gap that attribution analysis linked to their static component. Prediction intervals covered only 0.476 of tropical cyclone events against a nominal 0.90. Calibration by hazard class corrected most of this shortfall at an 11% cost in interval width, as a diagnostic only, because hazard class is known only after the event.</span> <span class="abstract-toggle" data-id="2610.20918">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.20918v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.20918v1) · [:material-content-copy: BibTeX](../../bibtex/2610.20918.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a>
    { .paper-tags }

-   #### Establishing baseline model performances for optical turbulence forecasting { #2610.18702 }

    *M. De Sepibus, E. Masciadri, C. Weinberger, A. Otarola, S. Miesske, C. Veillet, S. Ragland* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.18702">Accurate forecasting of optical turbulence (OT) is useful to optimise telescope scheduling and maximising observing efficiency, as the performance of adaptive optics systems strongly depends on...</span><span class="abstract-full" id="full-2610.18702" hidden>Accurate forecasting of optical turbulence (OT) is useful to optimise telescope scheduling and maximising observing efficiency, as the performance of adaptive optics systems strongly depends on turbulence conditions. While various numerical weather prediction and machine learning models are developed to predict these complex turbulence properties, properly evaluating their added value requires well-defined baseline reference models. This study establishes baseline performance thresholds for key astroclimatic parameters using two distinct models: one forecasting the parameter average over a specific timescale, and the other forecasting the temporal evolution over the same specific timescale. We apply these frameworks to extensive, multi-year datasets from the sites of the Very Large Telescope (VLT) and the Large Binocular Telescope (LBT) and discuss the possibility of extrapolating reference values of forecast performances for other sites based on their optical turbulence statistical characteristics. Our results show that baseline predictive accuracy varies according to the observing site, the specific parameter, the chosen timescale, and the forecasting approach. Therefore, an OT forecasting model's accuracy cannot be quantified in absolute terms, but relatively to the baseline.</span> <span class="abstract-toggle" data-id="2610.18702">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.18702v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.18702v1) · [:material-content-copy: BibTeX](../../bibtex/2610.18702.bib){ .bibtex-link }
    { .paper-links }

</div>

<nav class="pager" markdown="span">**1** [2](2.md) [3](3.md) [4](4.md) [5](5.md) [6](6.md) [7](7.md) [8](8.md) [9](9.md) [10](10.md) [11](11.md) [12](12.md) [13](13.md) [14](14.md) [Older :material-arrow-right:](2.md){ .pager-step }</nav>

