---
title: 'Other'
hide:
  - toc
---

<div class="listing-header" markdown>

# Other

<p class="page-meta" markdown="span">192 papers · page 1 of 7 · <a href="../../bib/other.bib" download>:material-download: BibTeX for this topic</a></p>

</div>

<div class="grid cards" markdown>

-   #### Inference of Unknown Dynamical Components Using Next Generation Reservoir Computing: From Chaotic Systems to Climate Data { #2609.24754 }

    *Jule Budnick, Andrew Keane, Serhiy Yanchuk* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.24754">We investigate next generation reservoir computing (NGRC) as a data-driven approach for inferring unseen components of dynamical systems. We compare NGRC with traditional reservoir computing (RC)...</span><span class="abstract-full" id="full-2609.24754" hidden>We investigate next generation reservoir computing (NGRC) as a data-driven approach for inferring unseen components of dynamical systems. We compare NGRC with traditional reservoir computing (RC) using the Lorenz and Rössler system, where two unknown components are inferred from one given component. For both systems, NGRC achieves accurate results while requiring fewer training data and less computational time than RC. We identified an inverse proportional behavior between the number of time-delayed steps needed for NGRC and the temporal resolution, indicating that the physical time span covered by the delay interval is an important factor in determining the required number of delayed steps. Finally, we apply NGRC to the observational climate data of ENSO (El Niño--Southern Oscillation) and infer one observable from the remaining variables. Despite the noise and complexity of the real-world data, the NGRC shows promising results. Our findings demonstrate the potential of NGRC for efficient inference of unseen components in both controlled dynamical systems and real-world data.</span> <span class="abstract-toggle" data-id="2609.24754">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.24754v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.24754v1) · [:material-content-copy: BibTeX](../../bibtex/2609.24754.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=subseasonal-to-seasonal" data-tag="subseasonal-to-seasonal">Subseasonal to seasonal</a>
    { .paper-tags }

-   #### Predictability-Guided Multiscale Probabilistic Forecasting of Wind Direction under Extreme Shear { #2609.16707 }

    *Hailong Shu* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.16707">Accurate multi-horizon wind direction forecasting is critical for turbine yaw control and grid security. Rapid directional shear (turning $\ge 90^\circ$) challenges models via non-Euclidean geometry...</span><span class="abstract-full" id="full-2609.16707" hidden>Accurate multi-horizon wind direction forecasting is critical for turbine yaw control and grid security. Rapid directional shear (turning $\ge 90^\circ$) challenges models via non-Euclidean geometry on $S^1$, multiscale dynamics, and regime-dependent uncertainty. Conventional discrete models and foundation models suffer from mid-frequency phase lag and turning misalignments. We show that directional predictability decays at disparate rates across frequency subbands, rendering monolithic mechanisms suboptimal. We propose a predictability-guided paradigm: slow synoptic drift $\to$ deterministic regression; intermediate turning $\to$ continuous latent differential flows; unresolved turbulence $\to$ conditional residual diffusion; followed by causal recalibration. On a 10,000-sequence multi-year benchmark, our framework maintains calm-weather accuracy (Test MCE $38.48^\circ$) while reducing extreme-turning error (Case 1 MCE $60.69^\circ$ vs $70.42^\circ$ for zero-shot foundation models). The circular CRPS reaches $22.36^\circ$, with 93.88% coverage at nominal 95% (91.01% out-of-distribution). Density estimation further reveals near-antipodal bimodal structure under severe shear (13.39%--15.43% tail mass $\ge 135^\circ$), exposing a geometric bound where single-center calibration under-covers (81.56%), motivating multimodal circular manifold learning.</span> <span class="abstract-toggle" data-id="2609.16707">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.16707v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.16707v1) · [:material-content-copy: BibTeX](../../bibtex/2609.16707.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=foundation-models" data-tag="foundation-models">Foundation models</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### 4D Parallelism Unlocks Exascale Bayesian Neural Networks for High-Fidelity Atmospheric Modeling { #2609.12815 }

    *Deifilia Kieckhefen, Juan Pedro Gutiérrez Hermosillo Muriedas, Lars Helge Heyen, Mathis Bode et al.* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.12815">We present BEAST, the first-ever Bayesian Swin Transformer for atmospheric forecasting on 0.25$^\circ$ global resolution able to accurately quantify both aleatoric and epistemic uncertainty. To...</span><span class="abstract-full" id="full-2609.12815" hidden>We present BEAST, the first-ever Bayesian Swin Transformer for atmospheric forecasting on 0.25$^\circ$ global resolution able to accurately quantify both aleatoric and epistemic uncertainty. To overcome the associated computational bottlenecks, we devise an orthogonal 4D-parallelization scheme that introduces a unique domain-tensor-parallelism strategy and a novel uncertainty parallel method, enabling us to fully leverage GPU capacity and efficiently scale model training. For a 2.4-billion-parameter model, we achieve a peak performance of 3.96 EFLOP/s on 20,480 NVIDIA GH200 GPUs on the JUPITER supercomputer. We train BEAST as a 700-million-parameter model with 96 random weight samples on 384 nodes on 40 years of data for nearly one million gradient updates. This model achieves predictive skill scores competitive with state-of-the-art probabilistic atmospheric AI models and numerical models, and can predict extreme events with exceptional skill, while generating large ensembles 3 to 4 times faster than the current-best AI model. Our contribution unlocks the potential of high-fidelity uncertainty quantification in atmospheric AI models, heralding a new era for AI-based models in climate and Earth system sciences.</span> <span class="abstract-toggle" data-id="2609.12815">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.12815v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.12815v1) · [:material-content-copy: BibTeX](../../bibtex/2609.12815.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=transformers" data-tag="transformers">Transformers</a> <a class="md-tag" href="/explore/?t=quarter-degree" data-tag="quarter-degree">0.25°</a>
    { .paper-tags }

-   #### Climate-ModernBERT: Revisiting Corpus Composition for Domain-Adaptive Continued Pretraining { #2609.07798 }

    *Yongan Yu, Shantam Raj, Jingwei Ni, Ario Saeid Vaghefi, Dominik Stammbach, Markus Leippold* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.07798">Natural Language Processing (NLP) in the climate domain requires models to process heterogeneous text sources, including scientific literature, policy disclosures, and synthetic reports. However, how...</span><span class="abstract-full" id="full-2609.07798" hidden>Natural Language Processing (NLP) in the climate domain requires models to process heterogeneous text sources, including scientific literature, policy disclosures, and synthetic reports. However, how to effectively combine diverse domain corpora during continued pretraining (CPT) remains underexplored. We introduce Climate-ModernBERT, a family of climate-adapted encoder models obtained through continued pretraining of ModernBERT-Base on three climate corpora: academic climate text, climate-filtered web data, and synthetic climate documents. We systematically compare joint continued pretraining on corpus mixtures with parameter-space merging of independently specialized checkpoints. Across nine climate NLP benchmarks, our best model achieves 76.3 average F_1, improving significantly over a vanilla ModernBERT baseline by 2.8 points. Within the climate NLP setting, the results show that academic climate corpora provide the strongest adaptation signal among the evaluated sources, while parameter-space merging improves over joint multi-source training and better preserves complementary information from heterogeneous climate corpora. We release all Climate-ModernBERT variants and training checkpoints to support future research in climate NLP and domain-adaptive pretraining.</span> <span class="abstract-toggle" data-id="2609.07798">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.07798v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.07798v1) · [:material-content-copy: BibTeX](../../bibtex/2609.07798.bib){ .bibtex-link }
    { .paper-links }

-   #### When Does Forecast-Error Energy Grow Logistically in Geophysical Turbulence? { #2608.26492 }

    *Malaquias Peña* · Aug 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2608.26492">Coarse-graining can yield a simple macroscopic growth curve in a bounded chaotic system even when constituent scales follow different clocks. The distinction matters as reduced-order and generative...</span><span class="abstract-full" id="full-2608.26492" hidden>Coarse-graining can yield a simple macroscopic growth curve in a bounded chaotic system even when constituent scales follow different clocks. The distinction matters as reduced-order and generative models compress multiscale forecast uncertainty into learned coordinates. We ask when forecast-error energy admits a logistic law. From the exact twin-error budget and correlated and decorrelated spectra, we derive two scalar limits: an invariant decorrelation amplitude, logistic only when contributing scales share one shape and one clock, and a self-similar upscale error front whose law depends on spectral slope and front speed. With local-strain scaling, the front predicts exponential error-energy growth for the canonical barotropic-vorticity spectrum and linear growth for the surface-quasigeostrophic spectrum. Stationary forced surface-quasigeostrophic twins test the logistic admission conditions. A response-blind partition of 16 trajectories gives cluster-mean logistic root-mean-square deviations 0.080 and 0.093, although every trajectory has resolved clock heterogeneity. An exact averaging identity shows how signed shape and clock corrections cancel, producing a nearly logistic aggregate while constituent scales retain distinct clocks. Mechanism identification therefore requires more than goodness of fit: independent shape, clock, and residual tests are required. These admission conditions provide physics-based guardrails for compact representations of chaotic systems and generative forecast ensembles.</span> <span class="abstract-toggle" data-id="2608.26492">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2608.26492v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2608.26492v1) · [:material-content-copy: BibTeX](../../bibtex/2608.26492.bib){ .bibtex-link }
    { .paper-links }

-   #### Frequency-aware forecasting for short-term typhoon gust prediction { #2608.25604 }

    *Xuefei Wang, Tingyi Liu, Heng Zhang, Shengjun Zhang* · Aug 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2608.25604">Accurate gust forecasting under typhoon conditions remains challenging due to the highly non-stationary and multi-scale characteristics of extreme wind fluctuations. Existing deep learning models...</span><span class="abstract-full" id="full-2608.25604" hidden>Accurate gust forecasting under typhoon conditions remains challenging due to the highly non-stationary and multi-scale characteristics of extreme wind fluctuations. Existing deep learning models often struggle to simultaneously capture long-term trends and rapid local variations, resulting in degraded performance during extreme events. We propose WDANet, a frequency-aware forecasting framework that integrates stationary wavelet decomposition, a Feature-wise Linear Modulation (FiLM) strategy, and a dual-branch encoder-decoder architecture, enabling separate modeling of trend and fluctuation components. Taking the offshore regions of the Western Pacific in China as an example, we conduct fine-grid wind gust prediction research. The results demonstrate that WDANet shows advantages for short lead times under the experimental setting across a 24-h forecasting horizon and achieves higher prediction accuracy than ECMWF-HRES within the first 6 h. During extreme wind events, WDANet more accurately captures gust peaks and attains the best RMSE and MAE performance. These results highlight its potential for offshore wind power operation, disaster warning, and risk mitigation.</span> <span class="abstract-toggle" data-id="2608.25604">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2608.25604v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2608.25604v1) · [:material-content-copy: BibTeX](../../bibtex/2608.25604.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a> <a class="md-tag" href="/explore/?t=extremes" data-tag="extremes">Extremes</a> <a class="md-tag" href="/explore/?t=energy" data-tag="energy">Energy</a>
    { .paper-tags }

-   #### Energy Yield and Lifetime Climate Classification via Machine Learning for Optimizing Photovoltaic Module Design and Materials { #2608.25448 }

    *Youri Blom, Sofia Dutto, Alexandru Costache, Rowan Richie, Ruben Pelsser, Wesley Berger, Jing Sun et al.* · Aug 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2608.25448">To resiliently and sustainably meet our future energy demand, photovoltaic (PV) modules must be deployed across a broad and diverse range of geographical regions with varying operating conditions. As...</span><span class="abstract-full" id="full-2608.25448" hidden>To resiliently and sustainably meet our future energy demand, photovoltaic (PV) modules must be deployed across a broad and diverse range of geographical regions with varying operating conditions. As these conditions strongly affect both performance and optimal system design, a dedicated PV-specific climate classification can be of great use. In this work, we develop a climate classification framework tailored to PV applications using a variety of machine learning (ML) techniques. Building on previous studies, our approach incorporates both energy yield, and for the first time, also the module lifetime with climate dependent degradation. We generate an interpolated dataset containing twelve input features and two target variables (i.e. energy yield and module lifetime). Feature importance analysis shows that annual global horizontal irradiation and ambient temperature are the most influential predictors. The most accurate regression model achieves root mean square errors (RMSE) of 0.007 MWh for energy yield and 1.5 years for lifetime prediction. The calculated feature importance scores are then integrated into a hierarchical clustering framework, resulting in 6 primary climate clusters (Tropical, Desert, Continental, Temperate, Boreal, and Polar) and 15 corresponding subclusters. Our analysis shows that the low temperature continental climate offers the highest discounted lifetime energy yield. These results can support a wide range of applications, including PV module optimization, system siting decisions, and comparative performance studies.</span> <span class="abstract-toggle" data-id="2608.25448">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2608.25448v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2608.25448v1) · [:material-content-copy: BibTeX](../../bibtex/2608.25448.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=energy" data-tag="energy">Energy</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a>
    { .paper-tags }

-   #### Predictability of El Niño from Delayed Observations { #2608.24428 }

    *Francisco J. Beron-Vera* · Aug 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2608.24428">Using monthly Niño-3.4 anomalies through July 2026, we investigate how much predictive information is contained in delayed observations of the index. Ridge regression identifies informative delays,...</span><span class="abstract-full" id="full-2608.24428" hidden>Using monthly Niño-3.4 anomalies through July 2026, we investigate how much predictive information is contained in delayed observations of the index. Ridge regression identifies informative delays, while multilayer perceptron and sparse identification of nonlinear dynamics (SINDy) models test whether nonlinear complexity provides additional direct forecast skill; gated recurrent unit (GRU) and long short-term memory (LSTM) networks provide a complementary test in which the temporal representation is learned internally. Delayed observations substantially improve forecasts over persistence and climatology at leads of up to six months, but increasing model complexity provides no systematic improvement. Historical recursive experiments favor a simple explicit SINDy recurrence and select shallow recurrent architectures, with no appreciable gain from learning the temporal representation internally. These results support a compact predictive representation of Niño-3.4 evolution in which the representation of past information is more consequential than model complexity. As a prospective application, the selected models are used to forecast the developing 2026 event beyond the last available observation and to compare its predicted evolution with completed historical El Niño events.</span> <span class="abstract-toggle" data-id="2608.24428">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2608.24428v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2608.24428v1) · [:material-content-copy: BibTeX](../../bibtex/2608.24428.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=recurrent-networks" data-tag="recurrent-networks">Recurrent networks</a> <a class="md-tag" href="/explore/?t=subseasonal-to-seasonal" data-tag="subseasonal-to-seasonal">Subseasonal to seasonal</a> <a class="md-tag" href="/explore/?t=monthly" data-tag="monthly">Monthly</a>
    { .paper-tags }

-   #### Tracing the Unlabeled Storm: Cross-Variable Transfer in a Lagrangian Atmospheric JEPA Framework { #2608.22358 }

    *K M Anirudh, S Sandeep, Hariprasad Kodamana* · Aug 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2608.22358">Deep atmospheric convection governs South Asian monsoon variability, yet attempting to learn its latent world model directly from zero-inflated, heavy-tailed precipitation yields suboptimal...</span><span class="abstract-full" id="full-2608.22358" hidden>Deep atmospheric convection governs South Asian monsoon variability, yet attempting to learn its latent world model directly from zero-inflated, heavy-tailed precipitation yields suboptimal predictive representations. Continuous atmospheric proxies, such as outgoing longwave radiation (OLR), express this convective organization far more coherently. We address this mismatch with <em>cross-variable proxy learning</em>: M-JEPA, a multiscale Monsoon Joint-Embedding Predictive Architecture, is pretrained on five continuous proxy fields over Lagrangian patches tracking moving convective systems---without rainfall supervision at any point. The resulting frozen representation is transferred to daily precipitation forecasts through a shared decoder trunk featuring parallel probabilistic and deterministic branches. Because rainfall is strictly unobserved during pretraining, downstream skill directly measures the predictive information captured in the latent rollout. A frozen-backbone probing framework with two controls (an identical architecture trained on rainfall alone, and a randomly initialized backbone) attributes the transfer specifically to proxy pretraining: direct rainfall training exhibits $36\%$ higher CRPS error ($7.52$ vs.\ $5.54$\,mm/day). Against the 51-member operational ECMWF ensemble, the transferred model attains a statistically resolved CRPS advantage ($6.81$ vs.\ $6.89$\,mm/day) and higher Brier skill ($+0.05$ vs.\ $-0.04$) using $15.4$M parameters on a single consumer GPU, concentrated at heavy-rain thresholds and fine spatial scales, while the ensemble retains an advantage in neighborhood skill and deterministic references on point metrics. The result provides a competitive monsoon precipitation forecast grounded in intraseasonal dynamics and a diagnostic framework for evaluating transferred atmospheric representations.</span> <span class="abstract-toggle" data-id="2608.22358">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2608.22358v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2608.22358v1) · [:material-content-copy: BibTeX](../../bibtex/2608.22358.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=foundation-models" data-tag="foundation-models">Foundation models</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=daily" data-tag="daily">Daily</a>
    { .paper-tags }

-   #### A Graph Neural Network Framework for Characterizing Rainfall Variability Regimes across India { #2608.20947 }

    *Pradyumnan Raghuveeran, Gaurav Chopra, Ajay Bankar, R. I. Sujith* · Aug 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2608.20947">The Indian Summer Monsoon shows significant spatial variation. While prior work primarily focused on forecasting rainfall amounts, little attention has been given to how consistently a location's...</span><span class="abstract-full" id="full-2608.20947" hidden>The Indian Summer Monsoon shows significant spatial variation. While prior work primarily focused on forecasting rainfall amounts, little attention has been given to how consistently a location's seasonal rainfall trajectory repeats from year to year. We introduce a graph-based machine learning framework to classify locations across India by this inter-annual consistency. Using 2001 to 2022 GSMaP ISRO data (excluding 2012), we constructed graphs for 29,026 grid points where nodes represent individual years and edges denote cosine similarity. A Graph Convolutional Network classified locations as either consistent or erratic with 96.8% accuracy. Applied to the Indian landmass, the model successfully identified the Western Ghats, Northeast India, and parts of central India as consistent regions. This classification was rigorously validated through statistical testing and temporal stability analysis, showing 93.6% agreement across two independent timeframes. Crucially, the results reveal a previously unreported coupling: regions with higher rainfall volumes are also the most temporally repeatable year-to-year, demonstrating an emergent, spatially coherent structure.</span> <span class="abstract-toggle" data-id="2608.20947">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2608.20947v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2608.20947v1) · [:material-content-copy: BibTeX](../../bibtex/2608.20947.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=graph-neural-networks" data-tag="graph-neural-networks">Graph neural networks</a> <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a>
    { .paper-tags }

-   #### An Agentic Approach for Active Data Collection, Travel Behavior Modeling, and Weather-Sensitive Demand Prediction { #2608.20320 }

    *Narges Ahmadi, Yubo Jiao, Jônatas Augusto Manzolli, Jiangbo Yu, Luis Miranda-Moreno* · Aug 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2608.20320">Travel behavior research increasingly combines digital data collection with predictive modeling, yet these stages are often developed and evaluated separately. This study proposes a three-agent...</span><span class="abstract-full" id="full-2608.20320" hidden>Travel behavior research increasingly combines digital data collection with predictive modeling, yet these stages are often developed and evaluated separately. This study proposes a three-agent workflow integrating conversational data collection, structured data processing, and behavioral prediction. A chatbot-administered, image-augmented stated-preference survey collected mode choices from student commuters across five predefined weather scenarios, yielding 454 respondent-scenario observations. Weather-related associations were analyzed using a multinomial logit model, while logistic regression and random forest provided machine-learning benchmarks. Nine locally deployed large language models (LLMs), ranging from 2 to 35 billion parameters, were evaluated across four zero-shot prompt-and-context conditions and extended through persona, few-shot, and vision-based configurations. Random forest achieved 69.6% five-class accuracy, while the best text-only zero-shot LLM reached 69.9% without task-specific fitting. Habitual travel information produced the most consistent gains, Expert framing generally outperformed Role-Play, and persona information was most useful when habitual travel information was unavailable. Few-shot prompting improved prediction for several models, with gains stabilizing after a small number of examples. Using the same weather images shown to respondents, the best vision-based configuration reached 71.5% five-class accuracy, indicating that visual context may provide additional predictive information for selected models. Overall, the study shows how conversational surveys, structured data processing, conventional behavioral modeling, machine learning, and multimodal LLM prediction can be coordinated within an auditable multi-agent workflow.</span> <span class="abstract-toggle" data-id="2608.20320">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2608.20320v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2608.20320v1) · [:material-content-copy: BibTeX](../../bibtex/2608.20320.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=llms-agents" data-tag="llms-agents">LLMs & agents</a> <a class="md-tag" href="/explore/?t=classical-ml" data-tag="classical-ml">Classical ML</a> <a class="md-tag" href="/explore/?t=benchmarks-datasets" data-tag="benchmarks-datasets">Benchmarks & datasets</a>
    { .paper-tags }

-   #### Machine learning correction of satellite precipitation is governed by mechanism purity, not algorithmic complexity: a proof-of-concept study in Hunan, China, with pre-registered cross-regional validation { #2608.12988 }

    *Yi Xu* · Aug 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2608.12988">Satellite precipitation products such as IMERG exhibit biases that vary with terrain, season, and precipitation regime, leaving the applicability boundaries of machine learning correction unclear....</span><span class="abstract-full" id="full-2608.12988" hidden>Satellite precipitation products such as IMERG exhibit biases that vary with terrain, season, and precipitation regime, leaving the applicability boundaries of machine learning correction unclear. This study proposes the Terrain-Moisture-Intensity (TMI) framework, centered on mechanism purity, extending the correction problem from purely algorithmic optimization to physical consistency diagnosis. A proof-of-concept study in Hunan Province employs IMERG V07, SRTM DEM, and ERA5 variables (tcwv, u10, v10). Ablation results indicate that, under the conditions of this study, terrain-moisture relationships are predominantly additive: RF-Full yields merely +0.001 R^2 gain over LR-Full, while bias rises to 1.282 mm d^-1; MAE decreases by approximately 14%, reflecting a trade-off between tail-fitting improvement and mean shift. SHAP diagnostics identify three categories of boundaries. Spatially, Central Hunan exhibits significant degradation (R^2=0.133) despite strong variable activation, consistent with mechanism fragmentation induced by mixed terrain. Temporally, u10 undergoes directional reversal between summer and spring (+0.096 to -0.156), presenting "silent failure." Extreme precipitation (>=50 mm d^-1) approximates a mechanism saturation frontier rather than isolated out-of-distribution samples, with DEM showing the largest relative amplification in SHAP disorder (+150%). The results demonstrate that machine learning correction performance is primarily constrained by mechanism purity. A pre-registered cross-regional test (Hunan, Guangxi, Guangdong) confirms this screening capability out of sample: a priori coherence proxies predict correction efficiency with a mean absolute error of 2.6 percentage points, while the transfer-versus-retraining contrast separates mechanism mismatch (coastal Guangdong) from portability (Guangxi), establishing the framework as a validated applicability screen.</span> <span class="abstract-toggle" data-id="2608.12988">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2608.12988v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2608.12988v1) · [:material-content-copy: BibTeX](../../bibtex/2608.12988.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a>
    { .paper-tags }

-   #### Deep Learning Imputation of Missing Radius of Maximum Winds (Rmax) Values in Tropical Cyclone Best-Track Data { #2608.09683 }

    *Swastik Agrawal, Nishkal Hundia, Ziyue Liu, Michelle Bensi* · Aug 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2608.09683">Probabilistic coastal hazard assessments require accurate characterization of tropical cyclone (TC) parameters, yet datasets often contain missing records for the radius of maximum winds (Rmax), a...</span><span class="abstract-full" id="full-2608.09683" hidden>Probabilistic coastal hazard assessments require accurate characterization of tropical cyclone (TC) parameters, yet datasets often contain missing records for the radius of maximum winds (Rmax), a key variable in Joint Probability Method analyses. This study evaluates data-driven approaches for Rmax imputation, including one-dimensional Convolutional Neural Networks (1DCNNs), Long Short-Term Memory (LSTM) networks, and conventional machine learning models. We examine physics-informed input augmentation, temporal modeling, and transfer learning using synthetic RAFT and STORM datasets for pre-training and observational IBTrACS data for fine-tuning. Including the radius of 34-knot winds (R34) substantially improves performance across all model types. Temporal models achieve higher average correlations than non-temporal models despite using approximately an order of magnitude fewer samples, indicating better preservation of relative Rmax variability across storms. This advantage is more pronounced when R34 is unavailable, suggesting temporal information can partially compensate for missing storm-size predictors. Transfer learning does not improve performance, likely because synthetic datasets have lower and less variable Rmax distributions than IBTrACS. These findings demonstrate the potential of temporal deep learning for reconstructing incomplete TC records and highlight the importance of physics-informed inputs, observational data availability, and distributional consistency in coastal hazard assessment.</span> <span class="abstract-toggle" data-id="2608.09683">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2608.09683v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2608.09683v1) · [:material-content-copy: BibTeX](../../bibtex/2608.09683.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=physics-ml-hybrid" data-tag="physics-ml-hybrid">Physics–ML hybrid</a> <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a>
    { .paper-tags }

-   #### Evaluating Explainable AI Methods for Geoscientific Regression: Insights from Applications and the Lorenz-63 System { #2608.07406 }

    *Ieuan Higgs, Todd Jones, Kieran Hunt, Anna-Louise Ellis* · Aug 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2608.07406">As artificial intelligence (AI) systems transition from research prototypes to operational tools in Earth system science and forecasting, establishing trust in their predictions becomes increasingly...</span><span class="abstract-full" id="full-2608.07406" hidden>As artificial intelligence (AI) systems transition from research prototypes to operational tools in Earth system science and forecasting, establishing trust in their predictions becomes increasingly important. Although model inputs and outputs are observable, the internal decision-making of modern AI models remains complex and hard to interpret, earning them the label “black boxes.” Explainable artificial intelligence (XAI) offers techniques to provide insight into these processes. However, most XAI methods were developed for classification tasks, raising questions about their suitability for the regression problems that dominate geoscientific applications. We review XAI approaches through this lens, organising them into a structured framework and examining both their theoretical foundations and practical behaviour. To ground this discussion, we apply a selection of methods to a machine learning emulator of the Lorenz 1963 system, an archetypal chaotic model that provides a tractable, physically meaningful setting for exposing the limitations and failure modes of general-purpose XAI in regression contexts. We then survey how these and related methods have been applied across a variety of Earth system sciences. We further situate XAI within the model development lifecycle, linking methodological choices to the needs of different stakeholder groups across operational Earth system science. We close by identifying gaps in existing methodologies and outlining a forward-looking research agenda, with practical recommendations for the responsible, effective use of XAI in regression applications of geoscientific modelling and forecasting.</span> <span class="abstract-toggle" data-id="2608.07406">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2608.07406v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2608.07406v1) · [:material-content-copy: BibTeX](../../bibtex/2608.07406.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=interpretability" data-tag="interpretability">Interpretability</a>
    { .paper-tags }

-   #### Climate-Dyna Deep Hedging for XVAs: Model-Based Reinforcement Learning, Residual Climate HVA, and Hedge-Instrument Discovery { #2608.01208 }

    *Xiaozhen Wang, Francois Buet-Golfouse* · Aug 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2608.01208">For a trading desk, residual climate hedging valuation adjustment (HVA) is the climate cost left after its inherited hedge and any admissible overlay have been taken into account; it therefore cannot...</span><span class="abstract-full" id="full-2608.01208" hidden>For a trading desk, residual climate hedging valuation adjustment (HVA) is the climate cost left after its inherited hedge and any admissible overlay have been taken into account; it therefore cannot be inferred from a stand-alone stress loss. We obtain this residual by comparing paired climate-on and baseline worlds and reoptimizing the overlay for each hedge universe, which also turns hedge-instrument discovery into a valuation problem: an instrument is useful to the extent that it lowers the optimized residual cost. The linear-Gaussian case has an exact finite-horizon Riccati solution; Climate-Dyna starts from that hedge and learns the remaining nonlinear correction from paired world-model rollouts, with an independent gate deciding whether to deploy the update. In a public-data-calibrated semi-synthetic EU ETS study, crediting the inherited hedge lowers the mean climate charge from 1.517 to 0.906, and the learned overlay lowers it to 0.831 against a 0.821 exact floor; residual Dyna cuts regret by 93% relative to replay with one quarter as many trajectories, while adaptation from only 25 target transitions retains 60.7% of the exact-assisted gain.</span> <span class="abstract-toggle" data-id="2608.01208">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2608.01208v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2608.01208v1) · [:material-content-copy: BibTeX](../../bibtex/2608.01208.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=reinforcement-learning" data-tag="reinforcement-learning">Reinforcement learning</a>
    { .paper-tags }

-   #### A Machine Learning-based Non-precipitating Clouds Estimation for THz Dual-Frequency Radar { #2608.00653 }

    *Kazuhiko Tamesue, Zheng Wen, Shotaro Yamaguchi, Hiroyuki Kasai, Wataru Kameyama, Toshio Sato et al.* · Aug 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2608.00653">Accurate measurement of non-precipitable clouds is important for early prediction of heavy rainfall disasters caused by extreme weather events. However, microwave cloud radar cannot observe the early...</span><span class="abstract-full" id="full-2608.00653" hidden>Accurate measurement of non-precipitable clouds is important for early prediction of heavy rainfall disasters caused by extreme weather events. However, microwave cloud radar cannot observe the early stages of cloud development from non-precipitation clouds (cumulus) to cumulonimbus. In this paper, we propose a terahertz dual-frequency cloud radar using 150 GHz and 95 GHz bands to detect cloud particles in cumulus smaller than 10 μm. Using a dataset generated by the ITU-R radio propagation model, we estimate the liquid water content of non-precipitation clouds and water vapor content in atmospheric gases, respectively, by using a machine learning-based approach. The effectiveness of using the dual wavelength ratio as an explanatory variable is examined.</span> <span class="abstract-toggle" data-id="2608.00653">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2608.00653v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2608.00653v1) · [:material-content-copy: BibTeX](../../bibtex/2608.00653.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a>
    { .paper-tags }

-   #### From Heat Stress to Perception: Interpretable Data-Driven Models of Human Thermal Sensation { #2607.25850 }

    *Abed Hammoud, Xinjie Huang, Qinqin Kong, Marialena Nikolopoulou, Elie Bou-Zeid* · Jul 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2607.25850">Heat stress indices are designed to quantify physiological thermal stress, but their relevance for inferring the thermal perception of individuals remains unclear. In this study, we show that thermal...</span><span class="abstract-full" id="full-2607.25850" hidden>Heat stress indices are designed to quantify physiological thermal stress, but their relevance for inferring the thermal perception of individuals remains unclear. In this study, we show that thermal stress and thermal sensation often diverge, as evidenced by distinct global sensitivity patterns with respect to environmental drivers. Using thermal sensation vote survey data, we demonstrate that the dominant sensitivities of stress-based metrics do not align with those governing reported human thermal sensation. Given the multitude of globally-applicable thermal stress indices and the lack of comparable general thermal sensation metrics, we develop two complementary data-driven modeling frameworks for thermal sensation. First, we construct polynomial chaos expansion (PCE) surrogates to represent thermal sensation as a function of meteorological variables, enabling efficient variance-based sensitivity analysis and explicit identification of influential inputs and interactions. Second, we develop multilayer perceptron (MLP) classifiers that capture the nonlinear and subjective nature of thermal perception, while achieving high predictive accuracy. The PCE models provide physically interpretable sensitivities that can explain the drivers of thermal sensation, while the MLPs offer flexible predictive capability suited to complex environments. We apply both modeling approaches at city- and continent-scales, revealing systematic differences in sensitivity structure and performance across climates. In particular, we find that the sensitivity of TSV-based models to the variability of meteorological conditions across geoclimatic zone encodes distinct dependencies on temperature, radiation, humidity, and wind that vary geographically, and are generally different from those of heat stress indices.</span> <span class="abstract-toggle" data-id="2607.25850">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2607.25850v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2607.25850v1) · [:material-content-copy: BibTeX](../../bibtex/2607.25850.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=interpretability" data-tag="interpretability">Interpretability</a> <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a>
    { .paper-tags }

-   #### A Physics-Informed Neural Operator for Thermal Ranking of Low-Cost Wall Materials in Hot-Dry Climates { #2607.25668 }

    *Muhammad Akbar Khan, Fahim Raees, Ubaida Fatima* · Jul 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2607.25668">Identifying cost-effective indigenous building materials that minimise heat penetration through walls is critical for indoor thermal comfort in low-income rural housing in hot-dry climates, where...</span><span class="abstract-full" id="full-2607.25668" hidden>Identifying cost-effective indigenous building materials that minimise heat penetration through walls is critical for indoor thermal comfort in low-income rural housing in hot-dry climates, where summer temperatures routinely exceed 45 C. We present a two-stage computational framework for thermal ranking of five low-cost indigenous wall materials: mud brick, clay-straw adobe, lime-stabilised bamboo panel, fired clay brick, and lime-mud composite. First, a validated Crank-Nicolson finite difference method (FDM) solves the one-dimensional transient heat equation with Robin boundary conditions under diurnal solar and outdoor air-temperature forcing, generating 1500 periodic-day solutions across a nine-dimensional parameter space by Latin Hypercube sampling. Second, a Physics-Informed Neural Operator (PINO) with a Fourier Neural Operator (FNO) backbone learns the parameter-to-solution operator mu -> T(x,t), enforcing both data fidelity and PDE consistency. The trained PINO attains a relative L2 field error of 5.14e-4 and a 0.201 K mean absolute error on the peak inner surface temperature, preserving the FDM material ranking exactly; PINO trained on 150 FDM samples matches a data-only FNO trained on twice as many, so the physics loss is most valuable when data are scarce. The periodic-day formulation also yields the ISO 13786 time lag and decrement factor, reproduced to within 0.99 h and 0.010. At nominal hot-dry summer conditions, clay-straw adobe achieves the best cost-performance index among widely available materials. A climate sweep, confirmed by FDM spot checks, reveals a regime boundary: under sub-ambient outdoor conditions the ranking inverts to conductive fired clay brick, delineating heat-exclusion and heat-rejection regimes. The framework supports evidence-based material selection for post-flood reconstruction in hot-dry regions.</span> <span class="abstract-toggle" data-id="2607.25668">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2607.25668v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2607.25668v1) · [:material-content-copy: BibTeX](../../bibtex/2607.25668.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=neural-operators" data-tag="neural-operators">Neural operators</a> <a class="md-tag" href="/explore/?t=physics-ml-hybrid" data-tag="physics-ml-hybrid">Physics–ML hybrid</a>
    { .paper-tags }

-   #### Predictive Modeling of High-Altitude Clear Air Turbulence in the United States: A Machine Learning Approach { #2607.11899 }

    *Kadir Gokdeniz, Irem Ulku* · Jul 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2607.11899">High-altitude Clear Air Turbulence (CAT) poses significant risks to aviation safety due to its unpredictability and challenges in detection. This study leverages machine learning models to improve...</span><span class="abstract-full" id="full-2607.11899" hidden>High-altitude Clear Air Turbulence (CAT) poses significant risks to aviation safety due to its unpredictability and challenges in detection. This study leverages machine learning models to improve CAT prediction within U.S. airspace at 200-350 hPa pressure levels, utilizing Pilot Reports (PIREPs), ERA5 reanalysis data, and aircraft aerodynamic parameters from the BADA database. Gradient boosting algorithms, particularly XGBoost, achieved the highest performance with an AUC of 0.904, demonstrating superior capability in capturing non-linear atmospheric dynamics. Key findings highlight the dominance of geographic coordinates (17.5% feature importance) and turbulence indices like TI3 in prediction, emphasizing the role of regional topography and upper-tropospheric instability. The integration of aerodynamic features such as drag force and wing loading improved the detection of moderate-to-severe perceived turbulence intensity (POD improved from 0.845 to 0.866), providing additional value to traditional aircraft-independent methods. Seasonal analysis revealed winter months as peak periods for CAT incidents, correlating with jet stream activity. While results align with global studies, limitations include geographic scope and aircraft-type diversity. This research underscores the potential of machine learning for operational CAT forecasting, with recommendations for future work focusing on global data integration and real-time telemetry to address climate-driven turbulence trends.</span> <span class="abstract-toggle" data-id="2607.11899">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2607.11899v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2607.11899v1) · [:material-content-copy: BibTeX](../../bibtex/2607.11899.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=classical-ml" data-tag="classical-ml">Classical ML</a>
    { .paper-tags }

-   #### Improved Global Ocean Heat Content Estimation by Modeling Vertical Spatio-Temporal Dependence { #2607.11832 }

    *Thea Sukianto, Donata Giglio, Mikael Kuusela* · Jul 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2607.11832">Estimating ocean heat content (OHC) with reliable uncertainties is critical for understanding and monitoring the evolution of Earth's climate, as the ocean has stored most of the energy accumulated...</span><span class="abstract-full" id="full-2607.11832" hidden>Estimating ocean heat content (OHC) with reliable uncertainties is critical for understanding and monitoring the evolution of Earth's climate, as the ocean has stored most of the energy accumulated in the climate system due to Earth Energy Imbalance. Here, we use Argo profiling float data from 2004-2022 to map OHC. As fewer Argo observations are available deeper in the water column, previous studies have partitioned the ocean into at least two pressure layers and mapped each separately, which complicates the estimation of uncertainties when the maps are summed to get the total OHC. In this work, we consider the case of two pressure layers and propose an improved mapping and uncertainty quantification method using bivariate locally stationary Gaussian processes and conditional simulations to map the two sections jointly while accounting for the correlation between them. We find that modeling this correlation results in improved OHC anomaly mapping and up to a 15 percent reduction of global OHC anomaly uncertainties in comparison to mapping the two layers separately without accounting for their dependence. These estimated uncertainties are essential to analyze the statistical significance of OHC anomalies on both regional and global scales, which we demonstrate using several climatological case studies.</span> <span class="abstract-toggle" data-id="2607.11832">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2607.11832v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2607.11832v1) · [:material-content-copy: BibTeX](../../bibtex/2607.11832.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=classical-ml" data-tag="classical-ml">Classical ML</a> <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a>
    { .paper-tags }

-   #### Tracing the space-time causal origins of Earth system extremes { #2607.10033 }

    *Jhayron S. Pérez-Carrasquilla, J. Jake Nichol, Vanessa Robledo, Diana Bull, Katherine Dagon et al.* · Jul 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2607.10033">Identifying the causes of Earth's extremes is challenging because counterfactual experiments are not possible in the observed world, while numerical experiments are computationally expensive and...</span><span class="abstract-full" id="full-2607.10033" hidden>Identifying the causes of Earth's extremes is challenging because counterfactual experiments are not possible in the observed world, while numerical experiments are computationally expensive and subject to biases. Data-driven causal discovery offers a complementary path, but existing approaches can fail in undersampled, high-dimensional regimes, and may not recover multi-timestep, multivariate pathways leading to particular events. We introduce Tracer of Causal Evolutions in Space and Time (TraCE-ST), a probabilistic Lagrangian approach that produces event-conditioned causal trajectories in multivariate gridded data. In synthetic experiments and real-world extreme events, TraCE-ST recovers known causal drivers and estimates their relative contributions, while also highlighting less-studied drivers, including orography-driven vorticity for Tropical Storm Debby (2006) and anomalous ocean-surface fluxes for the 2021 Pacific Northwest heatwave. Here, we propose causal tracking as an efficient data-driven framework for synthesizing causal evidence and generating testable hypotheses, complementing association analyses and numerical modeling while accelerating the study of high-impact events.</span> <span class="abstract-toggle" data-id="2607.10033">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2607.10033v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2607.10033v1) · [:material-content-copy: BibTeX](../../bibtex/2607.10033.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=extremes" data-tag="extremes">Extremes</a>
    { .paper-tags }

-   #### Spatial Support Matters: Geometry-Aware Graph Fusion for Rainfall Field Reconstruction { #2607.01621 }

    *Low Jun Yu, Niramay Kachhadiya, Herath Mudiyanselage Viraj Vidura Herath, Sanka Rasnayaka et al.* · Jul 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2607.01621">Fine-scale rainfall reconstruction is critical for urban flood modeling, but real rainfall sensing systems observe the field through incompatible spatial supports: gauges measure points, microwave...</span><span class="abstract-full" id="full-2607.01621" hidden>Fine-scale rainfall reconstruction is critical for urban flood modeling, but real rainfall sensing systems observe the field through incompatible spatial supports: gauges measure points, microwave links measure paths, and radar/satellite products measure gridded areas. These differences in measurement support impose geometrically distinct constraints on the rainfall field, yet existing heterogeneous graph approaches reconcile such sources in feature space, giving each its own embedding while discarding the geometry of its support. We propose a geometry-aware multi-support heterogeneous graph neural network that represents each observation according to its support type (0D point, 1D line, or 2D grid) as a distinct node layer, and fuses them through cross-support message passing into a point-support prediction layer from which the field is reconstructed. An inductive masked-node formulation decouples prediction resolution from sensing resolution, allowing the same trained model to reconstruct the field at user-defined target locations or display grids. On Singapore data, the proposed method reduces RMSE by 23.2% over the classical interpolation baseline, inverse-distance weighting, and consistently outperforms other neural architectures such as convolutional fusion and support-agnostic heterogeneous graph baselines. A generalization study using data from Sydney, Australia lets us characterize when multi-support fusion helps: the available skill appears to depend on gauge spacing relative to the spatial correlation length of the field, so fusion delivers the largest gains where the field is under-sampled relative to its correlation length and little when it is already resolved. Code and models will be open-sourced upon paper acceptance.</span> <span class="abstract-toggle" data-id="2607.01621">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2607.01621v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2607.01621v1) · [:material-content-copy: BibTeX](../../bibtex/2607.01621.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=graph-neural-networks" data-tag="graph-neural-networks">Graph neural networks</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a>
    { .paper-tags }

-   #### Wind-Aware Reinforcement Learning Control of a Small Quadrotor Using Learned Onboard Wind Estimation in Simulated Atmospheric Turbulence { #2607.01528 }

    *Abdullah Al Tasim, Wei Sun* · Jul 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2607.01528">Small multirotor aircraft are increasingly tasked with operations in the atmospheric boundary layer, where turbulent winds comparable to the vehicle's airspeed degrade trajectory tracking and can...</span><span class="abstract-full" id="full-2607.01528" hidden>Small multirotor aircraft are increasingly tasked with operations in the atmospheric boundary layer, where turbulent winds comparable to the vehicle's airspeed degrade trajectory tracking and can defeat conventional feedback control. This work illustrates a two-stage learning pipeline that first estimates the local wind from onboard kinematics and dynamics and then exploits that estimate inside a reinforcement learning (RL) flight controller. The wind estimator, an attention-augmented gated recurrent network trained on thousands of simulated flights through von Karman turbulence with power-law shear and veer, recovers the horizontal wind vector with a per-flight root-mean-square error of 0.40 m/s and a direction error of 3.2 degrees on unseen wind regimes, an accuracy near the floor imposed by unresolved turbulence, and generalizes to vertical ascent profiles with a skill score of 0.861 over a constant-wind reference. A proximal policy optimization controller receiving the frozen estimator's output reduces horizontal trajectory tracking error by 48% relative to a wind-blind proportional-derivative baseline across mean winds of 4 m/s to 12 m/s, winning on 100% of evaluation episodes. A three-way ablation decomposes this improvement into a kinematic component, available without wind information, and a wind-perception component; the perception share rises with wind speed, from small in light winds toward roughly half the total benefit in strong winds, consistent with the quadratic scaling of aerodynamic drag. The controller degrades gracefully on out-of-distribution winds of 13 m/s to 15 m/s, where the baseline fails catastrophically.</span> <span class="abstract-toggle" data-id="2607.01528">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2607.01528v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2607.01528v1) · [:material-content-copy: BibTeX](../../bibtex/2607.01528.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=reinforcement-learning" data-tag="reinforcement-learning">Reinforcement learning</a>
    { .paper-tags }

-   #### Conditional Tropical Cyclogenesis Rates via Rare-Event Sampling in a Neural Weather Emulator { #2606.30920 }

    *John S. Schreck, William Chapman, Charlie Becker, David John Gagne* · Jun 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2606.30920">We couple Forward Flux Sampling (FFS), a non-equilibrium rare-event technique from statistical mechanics, to a neural weather emulator (SDL-WXFormer, 1° grid spacing) to estimate conditional tropical...</span><span class="abstract-full" id="full-2606.30920" hidden>We couple Forward Flux Sampling (FFS), a non-equilibrium rare-event technique from statistical mechanics, to a neural weather emulator (SDL-WXFormer, 1° grid spacing) to estimate conditional tropical cyclogenesis rates, or how often a tropical cyclone achieves a hurricane-level central pressure, without modifying model dynamics. Tropical cyclogenesis rates vary by orders of magnitude across regimes, yet direct ensemble sampling cannot resolve this variability at operationally feasible ensemble sizes. FFS decomposes the rare disturbance to mature cyclone intensification path into a flux through an initial interface pressure and a product of conditional crossing probabilities across four intermediate interface pressures. We use the 1° emulator because FFS requires O(10^4) model trajectories per initial condition, and because the model's calibrated stochastic layers provide the necessary exploratory spread. Applied to 98 Atlantic basin initial conditions spanning 21 August - 8 October 2022, FFS resolves genesis rates spanning nearly three orders of magnitude, capturing a seasonal cycle qualitatively consistent with observations. A self-consistency check comparing FFS rates to independent direct-sampling rates yields a mean ratio of 1.03 +/- 0.15 across all initial conditions. Computational enhancement factors range from 3X (most active environment) to 140X (most suppressed), with a geometric mean of 14X. Three case studies illustrate the physical diagnostics the method provides: the rate-limiting step is initial tropical organization for the Earl environment, uniformly high crossing probabilities for the Fiona precursor environment, and a compound barrier at the final intensification stages for the Ian environment. More efficient emulators would enable application of FFS to finer resolutions.</span> <span class="abstract-toggle" data-id="2606.30920">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2606.30920v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2606.30920v1) · [:material-content-copy: BibTeX](../../bibtex/2606.30920.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=coarse" data-tag="coarse">Coarse (≥1°)</a>
    { .paper-tags }

-   #### An Integrated Two-Stage Deep-Learning Tool for Rapid Post-Hurricane Damage Identification and Repair Scheduling { #2606.29117 }

    *Hooman Torkaman, Ellis Oti Boateng, Jignesh Solanki, Anurag Srivastava* · Jun 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2606.29117">Post-hurricane damage assessment and repair scheduling can require computationally intensive simulation and optimization. This paper presents an integrated two-stage deep-learning tool for rapid...</span><span class="abstract-full" id="full-2606.29117" hidden>Post-hurricane damage assessment and repair scheduling can require computationally intensive simulation and optimization. This paper presents an integrated two-stage deep-learning tool for rapid damaged-line identification and repair-schedule computation. An available offline synthetic dataset for the IEEE 9500-node test feeder contains 1,700 hurricane scenarios with exposure features, grid metadata, fragility parameters, OpenDSS outputs, damaged-line labels, and Adaptive Large Neighborhood Search reference schedules. Stage 1 benchmarks MLP, ResMLP, and GraphSAGE, while Stage 2 compares MLP, DeepSets, and Set Transformer. The selected ResMLP-Set Transformer pipeline propagates Stage 1 errors into Stage 2 and achieves a damaged-job F1-score of 0.920, pairwise order agreement of 0.854, and start- and end-time mean absolute errors of 4.349 min and 4.486 min, respectively. The tool provides rapid initial repair-log decision support for new hurricane cases.</span> <span class="abstract-toggle" data-id="2606.29117">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2606.29117v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2606.29117v1) · [:material-content-copy: BibTeX](../../bibtex/2606.29117.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=transformers" data-tag="transformers">Transformers</a> <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a> <a class="md-tag" href="/explore/?t=sub-hourly" data-tag="sub-hourly">Sub-hourly</a>
    { .paper-tags }

-   #### Does Aurora Encode Atmospheric Structure? Latent Regime Analysis and Attribution { #2606.26361 }

    *Emma Kasteleyn, Ana Lucic* · Jun 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2606.26361">ML foundation models are able to emulate atmospheric dynamics accurately and efficiently but operate as opaque “black boxes”. We investigate the internal representations of the Aurora model using...</span><span class="abstract-full" id="full-2606.26361" hidden>ML foundation models are able to emulate atmospheric dynamics accurately and efficiently but operate as opaque “black boxes”. We investigate the internal representations of the Aurora model using spatially pooled PCA and layer-wise relevance propagation (LRP). We find evidence that Aurora's latent space is primarily organized by seasonal cycles, whereas extreme storm events do not form a linearly separable cluster. LRP indicates that the model attends to features consistent with the 3D vertical structure of the Great Storm of 1987. Perturbation tests show masking relevant regions degrades forecasts $3.31\times$ more than random masking. These findings suggest that Aurora learns meteorological coherence and vertical structure without explicit instruction.</span> <span class="abstract-toggle" data-id="2606.26361">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2606.26361v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2606.26361v1) · [:material-content-copy: BibTeX](../../bibtex/2606.26361.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=foundation-models" data-tag="foundation-models">Foundation models</a>
    { .paper-tags }

-   #### Short-Term Electricity Demand Forecasting for New England: A Comprehensive Machine Learning Benchmark with Weather, Calendar, and COVID-19 Indicators { #2606.20918 }

    *Reza Ghanavati, Behrooz Mosallaei* · Jun 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2606.20918">Accurate short-term electricity demand forecasting is critical for reliable power system operation, energy market planning, and infrastructure optimization. This paper benchmarks ten machine learning...</span><span class="abstract-full" id="full-2606.20918" hidden>Accurate short-term electricity demand forecasting is critical for reliable power system operation, energy market planning, and infrastructure optimization. This paper benchmarks ten machine learning models for daily electricity demand forecasting across the New England ISO (February 2020 - March 2023). The models span four families: tabular gradient-boosted trees (Random Forest, LightGBM, CatBoost, XGBoost), standalone neural architectures (LSTM, Transformer encoder), and hybrid Transformer+tree variants (Hybrid XGBoost, Hybrid LightGBM, Hybrid CatBoost, Hybrid RF). All models use meteorological data from six cities, calendar and holiday effects, autoregressive demand lags, and COVID-19 epidemiological variables. Hyperparameter optimization uses Optuna (300 trials, multivariate TPE, seed=42) under a leakage-free 70/15/15 chronological split. CatBoost achieves the best test performance: RMSE 8316 MWh, MAPE 1.87%, R-squared 0.917, followed by XGBoost (9066 MWh, R-squared 0.901), Hybrid CatBoost (9068 MWh, R-squared 0.901), and Hybrid XGBoost (9208 MWh, R-squared 0.898). Standalone neural architectures perform substantially worse (Transformer: 21294 MWh; LSTM: 22808 MWh), confirming the Transformer's role as a feature extractor rather than an end-to-end forecaster. An ablation on CatBoost shows that demand lags are the dominant predictor: removal degrades RMSE from 8316 to 11310 MWh (+36%), while weather and calendar features alone achieve an R-squared of 0.864. Removing COVID-19 features improves test RMSE by 1.7% while reducing training RMSE by 17.3%, a signature of temporal validity decay. SHAP analysis confirms this: 3 of 8 COVID features rank higher on the post-acute test set than during pandemic-active training, indicating the model over-applies stale pandemic patterns after behavioral adaptation was complete by August 2022.</span> <span class="abstract-toggle" data-id="2606.20918">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2606.20918v2) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2606.20918v2) · [:material-content-copy: BibTeX](../../bibtex/2606.20918.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=transformers" data-tag="transformers">Transformers</a> <a class="md-tag" href="/explore/?t=recurrent-networks" data-tag="recurrent-networks">Recurrent networks</a> <a class="md-tag" href="/explore/?t=classical-ml" data-tag="classical-ml">Classical ML</a> <a class="md-tag" href="/explore/?t=daily" data-tag="daily">Daily</a>
    { .paper-tags }

-   #### KFTD: Koopman-Fourier Time-Differentiable Network for Continuous Ocean Spatiotemporal Forecasting { #2606.17070 }

    *Qinghui Chen, Zekai Zhang, Hailong Liu, Jinglin Zhang, Cong Bai* · Jun 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2606.17070">Accurate oceanic forecasting is critical for climate monitoring and disaster early warning. However, ocean spatiotemporal forecasting encounters the double challenges of modeling complex dynamical...</span><span class="abstract-full" id="full-2606.17070" hidden>Accurate oceanic forecasting is critical for climate monitoring and disaster early warning. However, ocean spatiotemporal forecasting encounters the double challenges of modeling complex dynamical systems and ensuring computational efficiency. We present Koopman Fourier Time-Differentiable (KFTD) Network, a time continuous twostage paradigm that decouples interpolation from prediction to achieve efficient and scalable spatiotemporal modeling. We map complex nonlinear dynamics into the Koopman linear space and exploit Fourier analysis to enable continuous time interpolation at arbitrary sub-steps. A lightweight residual network consumes the high fidelity intermediate states to yield the final forecast. Unlike diffusion models, KFTD eliminates multi step noise sampling and directly evolves the system in continuous time, yielding a 4 computational speedup. We further introduce a DPP Loss that supports arbitrary PDE constraints in an endtoend manner, breaking the physical consistency bottleneck of pure data-driven approaches. Empirical results on four ocean datasets confirm that our continuous time framework reduces MSE by an average of 5.6% (up to 12.7% for SST) and improves efficiency over MCVD by 76.25%.</span> <span class="abstract-toggle" data-id="2606.17070">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2606.17070v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2606.17070v1) · [:material-content-copy: BibTeX](../../bibtex/2606.17070.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=efficiency" data-tag="efficiency">Efficiency</a>
    { .paper-tags }

-   #### Can Machine Learning Forecast Rice Yields in Data-Constrained Settings? Satellite Climate Data, National Crop Statistics, and Lessons from Sierra Leone { #2606.13959 }

    *Ibrahim Denis Fofanah* · Jun 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2606.13959">Sierra Leone's agriculture operates with almost no data-driven decision support, and no published machine learning study has examined the country's crop yields. We ask whether rice yield can be...</span><span class="abstract-full" id="full-2606.13959" hidden>Sierra Leone's agriculture operates with almost no data-driven decision support, and no published machine learning study has examined the country's crop yields. We ask whether rice yield can be forecast from data Sierra Leone currently has. Using 25 years of FAOSTAT production data (2000-2024) for nine major crops, we train XGBoost, Gradient Boosting, and Random Forest under a strict anti-leakage protocol with expanding-window walk-forward evaluation across seven held-out years, benchmarked against naive persistence. No model trained on crop statistics alone outperforms persistence. Augmenting with free satellite climate data (CHIRPS rainfall, NASA POWER temperature) reverses this result: a climate-only XGBoost reduces forecast error by one third (RMSE 284 vs 428 kg/ha), a gain that holds for a linear model and is robust to excluding the anomalous 2018 season. Early-season (May-June) rainfall is the dominant predictor, implying seasonal yield risk is observable months before harvest. No model anticipated the 2018 collapse, whose origins were institutional rather than climatic. We translate the findings into policy recommendations for Sierra Leone's Feed Salone Strategy, with a fully open-source pipeline.</span> <span class="abstract-toggle" data-id="2606.13959">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2606.13959v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2606.13959v1) · [:fontawesome-brands-github: Code](https://github.com/Denis060/sierraleone-agri-ml) · [:material-content-copy: BibTeX](../../bibtex/2606.13959.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=classical-ml" data-tag="classical-ml">Classical ML</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a>
    { .paper-tags }

-   #### AI Receptivity or AI Adoption Breadth? A Tool-Specific Reanalysis of the Lower-Literacy/Higher-Usage Link { #2606.13734 }

    *Hristo Inouzhe* · Jun 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2606.13734">Recent evidence reported by Tully, Longoni, and Appel (2025) suggests that lower artificial intelligence (AI) literacy predicts greater receptivity toward AI. We revisit this claim using the public...</span><span class="abstract-full" id="full-2606.13734" hidden>Recent evidence reported by Tully, Longoni, and Appel (2025) suggests that lower artificial intelligence (AI) literacy predicts greater receptivity toward AI. We revisit this claim using the public data from Study 3 of that article, which measures past usage of five AI tool categories on a five-point frequency scale. We first reproduce the negative association between AI literacy and aggregate AI usage using OLS on participant-level averages, binary logit, ordered logit, and multinomial logit specifications. We then show that the aggregate relationship masks substantial heterogeneity by tool type. In our demographic-adjusted primary specification, AI literacy does not significantly predict text AI usage (ordered-logit $β$ = -0.090, p = .387), whereas it remains a strong predictor of non-text AI adoption ($β$ = -0.377, p < .001). The non-text effect is also robust under Tully et al.'s original Study 3 control specification ($β$ = -0.502, p < .001). Binary, ordered-logit, and multinomial specifications suggest that the non-text relationship is primarily an adoption/non-adoption pattern rather than evidence of intensive use: the demographic-adjusted odds ratio of ever having used a non-text AI tool is 0.68. Thus, in the study that measures self-reported past usage rather than stated preferences, the evidence does not support a simple claim that lower AI literacy predicts greater receptivity to AI in general. It points instead to a narrower pattern of broader adoption across lower-penetration, non-text AI tools.</span> <span class="abstract-toggle" data-id="2606.13734">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2606.13734v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2606.13734v1) · [:material-content-copy: BibTeX](../../bibtex/2606.13734.bib){ .bibtex-link }
    { .paper-links }

</div>

<nav class="pager" markdown="span">**1** [2](2.md) [3](3.md) [4](4.md) [5](5.md) [6](6.md) [7](7.md) [Older :material-arrow-right:](2.md){ .pager-step }</nav>

