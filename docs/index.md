---
hide:
  - navigation
title: Deep Learning in Weather
---

A collection of papers on deep learning and machine learning applied to weather forecasting, climate modeling, and atmospheric science.

*Last updated: 2026-09-20*

## Starred Papers

<div class="grid cards" markdown>

-   #### <span class="star-marker">:material-star:</span> AIFS-DOP: End-to-End Medium-Range Weather Prediction from Observations Alone with Machine Learning

    ---

    <span class="paper-meta"><em>Ewan Pinnington, Peter Lean, Mihai Alexe, Eulalie Boucher, Simon Lang, Patrick Laloyaux et al.</em> · 2026</span>

    <span class="abstract-snippet" id="snip-2606.19093" data-search-exclude>We introduce the Artificial Intelligence Forecasting System for Direct Observation Prediction (AIFS-DOP). AIFS-DOP is trained on a 40-year harmonized dataset of gridded observations, without using...</span><span class="abstract-full" id="full-2606.19093" data-search-exclude hidden>We introduce the Artificial Intelligence Forecasting System for Direct Observation Prediction (AIFS-DOP). AIFS-DOP is trained on a 40-year harmonized dataset of gridded observations, without using numerical weather prediction (NWP) reanalysis or model data. The resulting model is competitive with ECMWF's Integrated Forecasting System (IFS) when scored on a one year period of forecasts across 2021/2022. This progress on Direct Observation Prediction represents the first time that a data-driven model, trained solely on observations, is competitive with the IFS at medium ranges for several key upper-air and surface headline scores, when verified against observation data.</span> <span class="abstract-toggle" data-id="2606.19093">more</span>

    <span class="paper-links">[:material-file-document: 2606.19093](https://arxiv.org/abs/2606.19093v1) · [:material-content-copy: BibTeX](bibtex/2606.19093.bib){ .bibtex-link }</span>

-   #### <span class="star-marker">:material-star:</span> (Sparse) Attention to the Details: Preserving Spectral Fidelity in ML-based Weather Forecasting Models

    ---

    <span class="paper-meta"><em>Maksim Zhdanov, Ana Lucic, Max Welling, Jan-Willem van de Meent</em> · 2026</span>

    <span class="abstract-snippet" id="snip-2604.16429" data-search-exclude>We introduce Mosaic, a probabilistic weather forecasting model that addresses three failure modes of spectral degradation in ML-based weather prediction: spectral damping (statistical),...</span><span class="abstract-full" id="full-2604.16429" data-search-exclude hidden>We introduce Mosaic, a probabilistic weather forecasting model that addresses three failure modes of spectral degradation in ML-based weather prediction: spectral damping (statistical), high-frequency aliasing (architectural), and residual high-frequency leakage (parametric). Mosaic generates ensemble members through learned functional perturbations and operates on native-resolution grids via mesh-aligned block-sparse attention, a hardware-aligned mechanism that captures long-range dependencies at linear cost by sharing keys and values across spatially adjacent queries. At 1.5° resolution with 214M parameters, Mosaic matches or outperforms models trained on 6$\times$ finer resolution on key variables and achieves state-of-the-art results among 1.5° models, producing well-calibrated ensembles whose individual members exhibit near-perfect spectral alignment across all resolved frequencies. A 24-member, 10-day forecast takes under 12s on a single H100~GPU. Code is available at https://github.com/maxxxzdn/mosaic.</span> <span class="abstract-toggle" data-id="2604.16429">more</span>

    <span class="paper-links">[:material-file-document: 2604.16429](https://arxiv.org/abs/2604.16429v3) · [:fontawesome-brands-github:](https://github.com/maxxxzdn/mosaic) · [:material-content-copy: BibTeX](bibtex/2604.16429.bib){ .bibtex-link }</span>

    <a class="md-tag" href="/tags/#probabilistic">probabilistic</a>

-   #### <span class="star-marker">:material-star:</span> U-Cast: A Surprisingly Simple and Efficient Frontier Probabilistic AI Weather Forecaster

    ---

    <span class="paper-meta"><em>Salva Rühling Cachay, Duncan Watson-Parris, Rose Yu</em> · 2026</span>

    <span class="abstract-snippet" id="snip-2604.09041" data-search-exclude>AI-based weather forecasting now rivals traditional physics-based ensembles, but state-of-the-art (SOTA) models rely on specialized architectures and massive computational budgets, creating a high...</span><span class="abstract-full" id="full-2604.09041" data-search-exclude hidden>AI-based weather forecasting now rivals traditional physics-based ensembles, but state-of-the-art (SOTA) models rely on specialized architectures and massive computational budgets, creating a high barrier to entry. We demonstrate that such complexity is unnecessary for frontier performance. We introduce \ours, a probabilistic forecaster built on a standard U-Net backbone trained with a simple recipe: deterministic pre-training on Mean Absolute Error followed by short probabilistic fine-tuning on the Continuous Ranked Probability Score (CRPS) using Monte Carlo Dropout for stochasticity. As a result, our model matches or exceeds the probabilistic skill of GenCast and IFS ENS at $1.5^\circ$ resolution while reducing training compute by over $10\times$ compared to leading CRPS-based models and inference latency by over $10\times$ compared to diffusion-based models. U-Cast trains in under 12 H200 GPU-days and generates a 15-day ensemble forecast in 3 seconds. These results suggest that scalable, general-purpose architectures paired with efficient training curricula can match complex domain-specific designs at a fraction of the cost, opening the training of frontier probabilistic weather models to the broader community.</span> <span class="abstract-toggle" data-id="2604.09041">more</span>

    <span class="paper-links">[:material-file-document: 2604.09041](https://arxiv.org/abs/2604.09041v2) · [:fontawesome-brands-github:](https://github.com/Rose-STL-Lab/u-cast) · [:material-content-copy: BibTeX](bibtex/2604.09041.bib){ .bibtex-link }</span>

    <a class="md-tag" href="/tags/#cnn">CNN</a> <a class="md-tag" href="/tags/#physics-informed">physics-informed</a> <a class="md-tag" href="/tags/#probabilistic">probabilistic</a>

-   #### <span class="star-marker">:material-star:</span> Using data assimilation tools to dissect GraphDOP

    ---

    <span class="paper-meta"><em>Patrick Laloyaux, Mihai Alexe, Eulalie Boucher, Peter Lean, Ewan Pinnington, Simon Lang et al.</em> · 2025</span>

    <span class="abstract-snippet" id="snip-2510.27388" data-search-exclude>The Data Assimilation (DA) community has been developing various diagnostics to understand the importance of the observing system in accurately forecasting the weather. They usually rely on the...</span><span class="abstract-full" id="full-2510.27388" data-search-exclude hidden>The Data Assimilation (DA) community has been developing various diagnostics to understand the importance of the observing system in accurately forecasting the weather. They usually rely on the ability to compute the derivatives of the physical model output with respect to its initial condition. For example, the Forecast Sensitivity-based Observation Impact (FSOI) estimates the impact on the forecast error of each observation processed in the DA system. This paper presents how these DA diagnostic tools are transferred to Machine Learning (ML) models, as their derivatives are readily available through automatic differentiation. We specifically explore the interpretability and explainability of the observation-driven GraphDOP model developed at the European Centre for Medium-Range Weather Forecasts (ECMWF). The interpretability study demonstrates the effectiveness of GraphDOP's sliding attention window to learn the meteorological features present in the observation datasets and to learn the spatial relationships between different regions. Making these relationships more transparent confirms that GraphDOP captures real, physically meaningful processes, such as the movement of storm systems. The explainability of GraphDOP is explored by applying the FSOI tool to study the impact of the different observations on the forecast error. This inspection reveals that GraphDOP creates an internal representation of the Earth system by combining the information from conventional and satellite observations.</span> <span class="abstract-toggle" data-id="2510.27388">more</span>

    <span class="paper-links">[:material-file-document: 2510.27388](https://arxiv.org/abs/2510.27388v1) · [:material-content-copy: BibTeX](bibtex/2510.27388.bib){ .bibtex-link }</span>

-   #### <span class="star-marker">:material-star:</span> GraphDOP: Towards skilful data-driven medium-range weather forecasts learnt and initialised directly from observations

    ---

    <span class="paper-meta"><em>Mihai Alexe, Eulalie Boucher, Peter Lean, Ewan Pinnington, Patrick Laloyaux, Anthony McNally et al.</em> · 2024</span>

    <span class="abstract-snippet" id="snip-2412.15687" data-search-exclude>We introduce GraphDOP, a new data-driven, end-to-end forecast system developed at the European Centre for Medium-Range Weather Forecasts (ECMWF) that is trained and initialised exclusively from Earth...</span><span class="abstract-full" id="full-2412.15687" data-search-exclude hidden>We introduce GraphDOP, a new data-driven, end-to-end forecast system developed at the European Centre for Medium-Range Weather Forecasts (ECMWF) that is trained and initialised exclusively from Earth System observations, with no physics-based (re)analysis inputs or feedbacks. GraphDOP learns the correlations between observed quantities - such as brightness temperatures from polar orbiters and geostationary satellites - and geophysical quantities of interest (that are measured by conventional observations), to form a coherent latent representation of Earth System state dynamics and physical processes, and is capable of producing skilful predictions of relevant weather parameters up to five days into the future.</span> <span class="abstract-toggle" data-id="2412.15687">more</span>

    <span class="paper-links">[:material-file-document: 2412.15687](https://arxiv.org/abs/2412.15687v1) · [:material-content-copy: BibTeX](bibtex/2412.15687.bib){ .bibtex-link }</span>

    <a class="md-tag" href="/tags/#physics-informed">physics-informed</a>

</div>

## Recent Additions

<div class="grid cards" markdown>

-   #### Spatial Aggregation of ROC and Precision-Recall Curves

    ---

    <span class="paper-meta"><em>Romain Pic, Zhongwei Zhang, Sebastian Engelke, Johanna Ziegel</em> · 2026</span>

    <span class="abstract-snippet" id="snip-2609.19517" data-search-exclude>Receiver Operating Characteristic (ROC) and Precision-Recall (PR) curves are widely used to assess the discrimination ability of forecasts for binary events, such as threshold exceedances or warnings...</span><span class="abstract-full" id="full-2609.19517" data-search-exclude hidden>Receiver Operating Characteristic (ROC) and Precision-Recall (PR) curves are widely used to assess the discrimination ability of forecasts for binary events, such as threshold exceedances or warnings of extreme events. In weather forecasting, forecasts are provided as spatial fields, yielding location-wise ROC and PR curves that are often aggregated to facilitate comparison. However, the effect of the aggregation strategy on performance assessment remains poorly understood.   We investigate how different aggregation strategies for ROC and PR curves affect the assessment of discrimination ability. In particular, we identify conditions under which aggregation strategies satisfy two desirable properties for fair comparison: preservation of dominance between forecasts and preservation of concavity or achievability of the curves. We obtain sufficient conditions and propose two strategies satisfying them. They are compared with existing strategies from the literature, and we analyze their properties and highlight potential pitfalls that may lead to misleading interpretations. Based on these findings, we provide practical guidelines for the interpretation of aggregated ROC and PR curves. The proposed framework is illustrated with AI-based global weather forecasts, showing how different aggregation strategies can yield different rankings of competing forecasts.</span> <span class="abstract-toggle" data-id="2609.19517">more</span>

    <span class="paper-links">[:material-file-document: 2609.19517](https://arxiv.org/abs/2609.19517v1) · [:fontawesome-brands-github:](https://github.com/pic-romain/spatial-agg-roc-pr) · [:material-content-copy: BibTeX](bibtex/2609.19517.bib){ .bibtex-link }</span>

-   #### A more predictable Madden-Julian Oscillation index derived from Koopman spectral analysis

    ---

    <span class="paper-meta"><em>Claire Valva, Edwin P. Gerber</em> · 2026</span>

    <span class="abstract-snippet" id="snip-2609.19435" data-search-exclude>The Madden-Julian oscillation (MJO) is a major source of subseasonal-to-seasonal (S2S) predictability. The MJO is commonly defined and tracked with indices such as the Real-time Multivariate MJO...</span><span class="abstract-full" id="full-2609.19435" data-search-exclude hidden>The Madden-Julian oscillation (MJO) is a major source of subseasonal-to-seasonal (S2S) predictability. The MJO is commonly defined and tracked with indices such as the Real-time Multivariate MJO (RMM) index. Although the RMM provides a useful description of the MJO, its evolution can be noisy and difficult to predict. We define an MJO index using a data-driven approximation of the Koopman operator. The Koopman index captures similar tropical circulation and convection patterns to the RMM but evolves more smoothly and predictably. Skillful prediction extends to 46 days for the Koopman index compared to 11 days for the RMM under the same prediction framework. While this new approach does not recover the RMM as well as operational S2S models, which provide skillful forecasts up to 35 days, the Koopman index could complement existing MJO diagnostics in evaluating and developing extended-range forecast systems.</span> <span class="abstract-toggle" data-id="2609.19435">more</span>

    <span class="paper-links">[:material-file-document: 2609.19435](https://arxiv.org/abs/2609.19435v1) · [:material-content-copy: BibTeX](bibtex/2609.19435.bib){ .bibtex-link }</span>

-   #### Physics-Informed Hemodynamic Modeling for Data-Free Prediction and Sparse-Data Assimilation

    ---

    <span class="paper-meta"><em>Xi Chen, Jianchuan Yang, Hongde Li, Guangxin He, Qiuyu Ye, Qiang Luo, Mao Chen, Wenqi Hu</em> · 2026</span>

    <span class="abstract-snippet" id="snip-2609.19290" data-search-exclude>Clinical decision-making for coronary intervention relies mainly on angiography and fractional flow reserve (FFR). However, angiography is two-dimensional and lacks depth information for 3D lesion...</span><span class="abstract-full" id="full-2609.19290" data-search-exclude hidden>Clinical decision-making for coronary intervention relies mainly on angiography and fractional flow reserve (FFR). However, angiography is two-dimensional and lacks depth information for 3D lesion characterization, while FFR provides only a single functional index, offering limited hemodynamic insight. Among existing methods, numerical analysis is computationally expensive, whereas learning-based approaches require extensive supervision and often lack physical consistency. To address these limitations, we propose physics-informed hemodynamic modeling, an integrated deep learning framework for 3D coronary blood flow analysis from dual-view angiography. First, an attention-enhanced CNN reconstructs coronary geometry from angiography. The resulting point clouds are then mapped to a reference domain and Fourier-encoded for joint representation. A decoupled network separately predicts velocity and pressure fields, with embedded physical priors enabling efficient transfer across physiological conditions. Across 32 clinical patients evaluated under four flow conditions, the trans-stenotic pressure-drop mean absolute percentage error was 2.02%, while the velocity and pressure relative-L2 errors were 0.054 and 0.023, respectively. Validation against hospital-measured FFR further achieved 93.8% diagnostic accuracy (30/32; exact 95% CI, 79.2%-99.2%). The framework also supports illustrative revascularization comparisons and sparse-data assimilation, with the full angiography-to-hemodynamics pipeline completed within 20 minutes per patient.</span> <span class="abstract-toggle" data-id="2609.19290">more</span>

    <span class="paper-links">[:material-file-document: 2609.19290](https://arxiv.org/abs/2609.19290v1) · [:material-content-copy: BibTeX](bibtex/2609.19290.bib){ .bibtex-link }</span>

    <a class="md-tag" href="/tags/#cnn">CNN</a> <a class="md-tag" href="/tags/#physics-informed">physics-informed</a>

-   #### Butterfly Effect and the Kinetic Energy Cascade in Probabilistic Machine Learning Weather Prediction Models

    ---

    <span class="paper-meta"><em>Jiakai Chen, Joel Oskarsson, Simon Driscoll, Sebastian Schemm</em> · 2026</span>

    <span class="abstract-snippet" id="snip-2609.18489" data-search-exclude>This study analyses kinetic energy (KE) spectra, difference kinetic energy (DKE) spectra, and signatures of KE transfer across spatial scales in four state-of-the-art probabilistic machine learning...</span><span class="abstract-full" id="full-2609.18489" data-search-exclude hidden>This study analyses kinetic energy (KE) spectra, difference kinetic energy (DKE) spectra, and signatures of KE transfer across spatial scales in four state-of-the-art probabilistic machine learning weather prediction (MLWP) models: NeuralGCM-ENS, FourCastNet 3, AIFS-ENS, and GenCast. Results are compared with those from the physics-based numerical weather prediction model IFS-ENS. While NeuralGCM-ENS successfully reproduces the expected upscale transfer of KE, noise injection at its encoder stage underestimates mesoscale KE. Conversely, AIFS-ENS, GenCast, and FourCastNet 3 produce realistic KE spectral magnitudes but do not capture the expected upscale transfer of KE. In particular, AIFS-ENS and GenCast, which employ spatially uncorrelated stochastic perturbations, exhibit enhanced accumulation of KE at high wavenumbers. All examined models exhibit upscale error growth, reflected by the progressive shift of the DKE spectral peak toward larger wavelengths over time. However, the MLWP models struggle to reproduce the rapid initial growth of ensemble spread at small spatial scales associated with the butterfly effect. The results show that MLWP models can misrepresent the known scale transfer of kinetic energy despite producing skilful weather forecasts.</span> <span class="abstract-toggle" data-id="2609.18489">more</span>

    <span class="paper-links">[:material-file-document: 2609.18489](https://arxiv.org/abs/2609.18489v1) · [:material-content-copy: BibTeX](bibtex/2609.18489.bib){ .bibtex-link }</span>

    <a class="md-tag" href="/tags/#physics-informed">physics-informed</a> <a class="md-tag" href="/tags/#probabilistic">probabilistic</a>

-   #### Every Fixed Metric Has a Blind Spot: A Learned Atmospheric Critic for Scoring Forecast Realism

    ---

    <span class="paper-meta"><em>Younes Elberkennou, Dmitri Demler, Thierry Meier, Luca Rispoli, Fanny Lehmann, Joel Oskarsson</em> · 2026</span>

    <span class="abstract-snippet" id="snip-2609.18381" data-search-exclude>Despite their high accuracy on point-wise metrics, machine learning weather forecasting models can exhibit different failure modes such as blurring, periodic irregularities, and other unphysical...</span><span class="abstract-full" id="full-2609.18381" data-search-exclude hidden>Despite their high accuracy on point-wise metrics, machine learning weather forecasting models can exhibit different failure modes such as blurring, periodic irregularities, and other unphysical spatial artifacts. This has motivated a variety of metrics to detect known failure cases. Existing metrics fix a representation or transformation in advance, and that choice limits the artifacts they can detect. We propose to train a discriminator for separating reference data from the model's output, and using its output logit to obtain a divergence-like realism score. The discriminator learns whatever separates the model's fields from real weather, adapting to whichever failure mode that model exhibits. We compare our learned atmospheric critic to existing metrics using various synthetic corruptions applied to ERA5 reanalysis data. Our method successfully identifies the corruptions and ranks their severity, while existing metrics fail on at least one corruption. Additionally, we evaluate forecasts from real weather models, and find that the realism score degrades with longer lead times and the metric generally assigns higher realism to numerical models than to machine learning models.</span> <span class="abstract-toggle" data-id="2609.18381">more</span>

    <span class="paper-links">[:material-file-document: 2609.18381](https://arxiv.org/abs/2609.18381v1) · [:material-content-copy: BibTeX](bibtex/2609.18381.bib){ .bibtex-link }</span>

-   #### IRENE: A Convolutional GRU Ensemble Model for Radar Precipitation Nowcasting over Italy

    ---

    <span class="paper-meta"><em>Alessandro Camilletti, Gabriele Franch, Elena Tomasi, Marco Cristoforetti</em> · 2026</span>

    <span class="abstract-snippet" id="snip-2609.17175" data-search-exclude>We present IRENE (Italian Radar Ensemble Nowcasting Experiment), a deep learning model for probabilistic short-range precipitation nowcasting over the Italian domain at \SI{1}{km} spatial and 5 min...</span><span class="abstract-full" id="full-2609.17175" data-search-exclude hidden>We present IRENE (Italian Radar Ensemble Nowcasting Experiment), a deep learning model for probabilistic short-range precipitation nowcasting over the Italian domain at \SI{1}{km} spatial and 5 min temporal resolution. IRENE adopts an encoder--forecaster architecture built on multi-scale Convolutional Gated Recurrent Units (ConvGRUs), trained on the national radar composite produced by the Italian Civil Protection Department (DPC). An importance-sampling scheme focuses training on precipitation-relevant events, while the almost-fair Continuous Ranked Probability Score (afCRPS) is adopted as the primary probabilistic loss function. Two additional training configurations are proposed: an adversarial (GAN) variant, IRENE-GAN, designed to improve the spatial sharpness of the generated forecasts, and a spectrally constrained variant, IRENE-GAN-RAPSD, in which the adversarial objective is complemented by an explicit penalty on the radially averaged power spectral density. The three configurations are evaluated against the stochastic extrapolation method STEPS and the pre-trained deep learning model DGMR. All IRENE configurations attain a lower Continuous Ranked Probability Score than both benchmarks at every lead time and rank histograms closer to uniformity, indicating better probabilistic skill and ensemble calibration. In terms of ensemble-mean mean absolute error the advantage is confined to the first 90 min, beyond which the strongly damped DGMR fields and, to a lesser extent, STEPS become competitive. Spectral analysis shows that the adversarial training removes the progressive loss of small-scale variance exhibited by IRENE, at the cost of an excess of fine-scale power at long lead times that the spectral penalty only partially controls.</span> <span class="abstract-toggle" data-id="2609.17175">more</span>

    <span class="paper-links">[:material-file-document: 2609.17175](https://arxiv.org/abs/2609.17175v1) · [:material-content-copy: BibTeX](bibtex/2609.17175.bib){ .bibtex-link }</span>

    <a class="md-tag" href="/tags/#foundation-model">foundation-model</a> <a class="md-tag" href="/tags/#recurrent">recurrent</a> <a class="md-tag" href="/tags/#probabilistic">probabilistic</a>

-   #### Predictability-Guided Multiscale Probabilistic Forecasting of Wind Direction under Extreme Shear

    ---

    <span class="paper-meta"><em>Hailong Shu</em> · 2026</span>

    <span class="abstract-snippet" id="snip-2609.16707" data-search-exclude>Accurate multi-horizon wind direction forecasting is critical for turbine yaw control and grid security. Rapid directional shear (turning $\ge 90^\circ$) challenges models via non-Euclidean geometry...</span><span class="abstract-full" id="full-2609.16707" data-search-exclude hidden>Accurate multi-horizon wind direction forecasting is critical for turbine yaw control and grid security. Rapid directional shear (turning $\ge 90^\circ$) challenges models via non-Euclidean geometry on $S^1$, multiscale dynamics, and regime-dependent uncertainty. Conventional discrete models and foundation models suffer from mid-frequency phase lag and turning misalignments. We show that directional predictability decays at disparate rates across frequency subbands, rendering monolithic mechanisms suboptimal. We propose a predictability-guided paradigm: slow synoptic drift $\to$ deterministic regression; intermediate turning $\to$ continuous latent differential flows; unresolved turbulence $\to$ conditional residual diffusion; followed by causal recalibration. On a 10,000-sequence multi-year benchmark, our framework maintains calm-weather accuracy (Test MCE $38.48^\circ$) while reducing extreme-turning error (Case 1 MCE $60.69^\circ$ vs $70.42^\circ$ for zero-shot foundation models). The circular CRPS reaches $22.36^\circ$, with 93.88\% coverage at nominal 95\% (91.01\% out-of-distribution). Density estimation further reveals near-antipodal bimodal structure under severe shear (13.39\%--15.43\% tail mass $\ge 135^\circ$), exposing a geometric bound where single-center calibration under-covers (81.56\%), motivating multimodal circular manifold learning.</span> <span class="abstract-toggle" data-id="2609.16707">more</span>

    <span class="paper-links">[:material-file-document: 2609.16707](https://arxiv.org/abs/2609.16707v1) · [:material-content-copy: BibTeX](bibtex/2609.16707.bib){ .bibtex-link }</span>

    <a class="md-tag" href="/tags/#foundation-model">foundation-model</a> <a class="md-tag" href="/tags/#probabilistic">probabilistic</a>

-   #### A Self-Diagnosing Structural Error-Aware Parameter Estimation Method for Earth System Models

    ---

    <span class="paper-meta"><em>Qingyuan Yang, Addisu G Semie, Brian Medeiros, Gregory S Elsaesser, Da Fan, Wayne Chuang</em> · 2026</span>

    <span class="abstract-snippet" id="snip-2609.16210" data-search-exclude>We propose a fully automated, structural error-aware, interpretable climate model parameter estimation method that leverages Perturbed Parameter Ensembles (PPEs). It is based on history matching and...</span><span class="abstract-full" id="full-2609.16210" data-search-exclude hidden>We propose a fully automated, structural error-aware, interpretable climate model parameter estimation method that leverages Perturbed Parameter Ensembles (PPEs). It is based on history matching and aligns with an increasingly-used iterative simulation-emulation-calibration methodology. The method is motivated by the negative impacts of structural error and emulator and observational uncertainties on climate model parameter estimation efforts, as well as the problems associated with sparsely-sampled PPEs. To address these challenges, the method explicitly builds simpler emulators that avoid overfitting, detect structural error, avoids compensating for structural error through inflated mismatch tolerances, and sequentially excludes structurally inconsistent variables for parameter estimation. The method decomposes the high-dimensional calibration problem into linked low-dimensional subproblems, and integrates their constraints to reconstruct the jointly plausible region of the full parameter space. The method is applied to a 100-member PPE with 34 perturbed parameters generated by a version of CAM6 with machine learning-based warm rain microphysics parameterization. Through iterative application, the method greatly reduces the ensemble spread and improves the matching between simulated and observed zonal climatologies. The method also finds ensemble members that outperform the default CAM6 configuration in root mean square error across multiple diagnostics. Controlled experiments demonstrate that overly-conservative emulator uncertainty could lead to neglect of informative observations, and tolerance of the structural error, in the context of this method, biases the estimated parameters toward compensating for structural error. Our work also emphasizes the value of interpretability for diagnosing structural error and informing parameter estimation in PPE-based calibration.</span> <span class="abstract-toggle" data-id="2609.16210">more</span>

    <span class="paper-links">[:material-file-document: 2609.16210](https://arxiv.org/abs/2609.16210v1) · [:material-content-copy: BibTeX](bibtex/2609.16210.bib){ .bibtex-link }</span>

-   #### Aries: A Proprietary Medium-Range Weather Prediction Model for the Energy Industry

    ---

    <span class="paper-meta"><em>Lukas Hedegaard Morsing, Arian Bakhtiarnia, Jonas Lynge Olesen, Tómas Bragi Björnsson Leth et al.</em> · 2026</span>

    <span class="abstract-snippet" id="snip-2609.13292" data-search-exclude>Medium-range weather forecasting underpins operational and planning decisions across the energy industry. Developing competitive weather models was once the domain of national meteorological centers,...</span><span class="abstract-full" id="full-2609.13292" data-search-exclude hidden>Medium-range weather forecasting underpins operational and planning decisions across the energy industry. Developing competitive weather models was once the domain of national meteorological centers, but recent advances in machine-learned weather prediction (MLWP) have opened the field to industry. We present Aries, a SwinTransformer-based MLWP model developed at InCommodities. Aries is trained on ERA5 reanalysis data at 0.25\textdegree{} resolution, predicting 74 prognostic and 11 diagnostic atmospheric variables. We evaluate the model on 2025 ECMWF Analysis initializations, ensuring a recent and strictly out-of-sample test period for all models compared. On 10-metre wind speed, Aries outperforms both ECMWF HRES and AIFS in terms of RMSE for lead times up to four days, while on 2-metre temperature it achieves RMSE on par with AIFS operational. These results demonstrate that proprietary development of competitive weather models is technically viable, supporting a broader set of forecasts available for operational and planning applications in the energy industry.</span> <span class="abstract-toggle" data-id="2609.13292">more</span>

    <span class="paper-links">[:material-file-document: 2609.13292](https://arxiv.org/abs/2609.13292v1) · [:material-content-copy: BibTeX](bibtex/2609.13292.bib){ .bibtex-link }</span>

    <a class="md-tag" href="/tags/#transformer">transformer</a>

-   #### A Physics--ML Multi-Fidelity Strategy for Earth System Model Parameter Optimization: A QG Proof-of-Concept

    ---

    <span class="paper-meta"><em>Abdullah A. Fahad, Manmeet Singh, Donifan Barahona, Anton Darmenov, Andrea Molod</em> · 2026</span>

    <span class="abstract-snippet" id="snip-2609.13275" data-search-exclude>Earth System Models rely on tunable subgrid-scale parameterizations, but optimizing these parameters is computationally expensive, particularly when nonlinear interactions require many simulations....</span><span class="abstract-full" id="full-2609.13275" data-search-exclude hidden>Earth System Models rely on tunable subgrid-scale parameterizations, but optimizing these parameters is computationally expensive, particularly when nonlinear interactions require many simulations. We present a hybrid Physics-ML multi-fidelity framework that combines Green's Function Optimization (GFO) with Gaussian Process or Neural Network surrogate optimization. Using a quasi-geostrophic turbulence model, GFO first ranks parameter sensitivities in normalized coordinates and selects a reduced active subset. Nonlinear surrogates then explore this subset using inexpensive 30-day simulations before refining promising candidates with 180-day simulations. Across seven strategies and a 35-member ensemble, GFO-MultiGP and GFO-MultiNN achieved mean improvements of 64.6 percent and 65.2 percent, respectively, while reaching practical saturation after 3,060 and 2,520 simulation-days. The corresponding standalone GP and NN achieved 61.1 percent and 42.0 percent improvements and required 7,740 and 6,660 simulation-days. These results demonstrate an end-to-end sample-efficiency advantage for the tested hybrid pipelines. Because screening, dimensionality reduction, initialization, and fidelity scheduling change simultaneously, their individual contributions are not isolated.</span> <span class="abstract-toggle" data-id="2609.13275">more</span>

    <span class="paper-links">[:material-file-document: 2609.13275](https://arxiv.org/abs/2609.13275v1) · [:material-content-copy: BibTeX](bibtex/2609.13275.bib){ .bibtex-link }</span>

</div>

