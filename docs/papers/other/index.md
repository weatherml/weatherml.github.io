---
title: 'Other'
hide:
  - toc
---

<div class="listing-header" markdown>

# Other

<p class="page-meta" markdown="span">202 papers · page 1 of 7 · <a href="../../bib/other.bib" download>:material-download: BibTeX for this topic</a></p>

</div>

<div class="grid cards" markdown>

-   #### AI Emulation of Stochastic Sudden Stratospheric Warming with Interpretable Latent Structure { #2610.02069 }

    *C. Daniel Boscu, Daniel Hernandez, Fabio Alvarez Ventura, Justin Finkel, Ashesh Chattopadhyay et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.02069">Rare weather regime transitions pose a challenge for data-driven modeling due to class imbalance. In this study, we develop a probabilistic deep learning emulator for a prototypical system with...</span><span class="abstract-full" id="full-2610.02069" hidden>Rare weather regime transitions pose a challenge for data-driven modeling due to class imbalance. In this study, we develop a probabilistic deep learning emulator for a prototypical system with regime transitions, the stochastic Holton--Mass model of stratospheric variability, and analyze the structure of its learned latent space. The Holton--Mass model exhibits two metastable regimes, a strong and a weak polar vortex, maintained by nonlinear wave--mean flow interactions, with weak stochastic forcing intermittently triggering rare transitions between these regimes that qualitatively represent SSW events. We employ a ResNet-inspired Conditional Variational Autoencoder with six-layer encoder and decoder layers and explicit current-state conditioning to model the distribution of the system's state at the next time step (one day). The emulator accurately reproduces short-term dynamics, steady-state probability distributions, regime persistence statistics, rare transition rates, the transition committor function, and the transition expected lead time of the physical model. Beyond emulation fidelity, we interrogate the learned latent representation to understand how the model internalizes the underlying metastable structure of the dynamics. Principal Component Analysis of the 32-dimensional latent space reveals a clear and unsupervised separation into four physically interpretable clusters corresponding to strong versus weak vortex regimes and stable versus transition-prone configurations. Such emergent regime separation in latent space is hard to identify for deep generative models applied to high-dimensional stochastic systems. Our results show that carefully designed probabilistic emulators can uncover physically meaningful manifolds governing extreme-event dynamics, potentially aiding the development of improved operational advanced warning systems.</span> <span class="abstract-toggle" data-id="2610.02069">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.02069v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.02069v1) · [:material-content-copy: BibTeX](../../bibtex/2610.02069.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=interpretability" data-tag="interpretability">Interpretability</a>
    { .paper-tags }

-   #### Unsupervised Domain Adaptation for Enhanced Radiometer Image Precipitation Estimation using Conditional Flow Matching { #2610.01890 }

    *Victor Enescu, Assaad Zeghina, Matthieu Meignin, Nicolas Viltard, Cécile Mallet* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.01890">Deep generative networks have recently achieved unprecedented performance in precise image and video editing using sophisticated textual prompts. However, the effectiveness of such models heavily...</span><span class="abstract-full" id="full-2610.01890" hidden>Deep generative networks have recently achieved unprecedented performance in precise image and video editing using sophisticated textual prompts. However, the effectiveness of such models heavily depends on access to very large supervised and annotated image datasets, which can be very difficult to obtain. This is particularly true for satellite instruments, which very rarely overlap with labelled data, and suffer from domain shifts in the rare occasions they do. In this paper, we investigate the potential of flow matching models for unsupervised domain adaptation of satellite radiometer images. Our main contribution is a novel unsupervised method that achieves precise domain alignment by leveraging parts of the deterministic ordinary differential equations in flow matching models, conditioned on different satellite instruments. A key strength of our approach is its ability to preserve essential information while adapting across any domains since the perturbations are in theory bijective. Extensive experiments conducted on the GPM-Core constellation show the benefit of our conditional domain adaptation, particularly in improving rain precipitation estimation from radiometer imagery.</span> <span class="abstract-toggle" data-id="2610.01890">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.01890v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.01890v1) · [:material-content-copy: BibTeX](../../bibtex/2610.01890.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a>
    { .paper-tags }

-   #### Explaining El Niño Forecasts with the Average Gradient Outer Product { #2610.01095 }

    *Yuan Hui, Dorian S. Abbot, Robert J. Webber* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.01095">An important and unresolved problem in the physical sciences is explaining the predictions made by neural networks. Several explainable artificial intelligence (XAI) methods have been proposed to...</span><span class="abstract-full" id="full-2610.01095" hidden>An important and unresolved problem in the physical sciences is explaining the predictions made by neural networks. Several explainable artificial intelligence (XAI) methods have been proposed to address this problem, including gradient XAI, Integrated Gradients, and GradientSHAP. We evaluate the baseline XAI methods according to four scores: sensitivity (XAI patterns strongly affect predictions), attribution (XAI patterns reproduce the change in prediction relative to a baseline), robustness (XAI patterns remain stable for nearby inputs), and coherence (XAI patterns are spatially smooth). We also introduce a new method, average gradient outer product (AGOP) XAI, that uses global gradient information to identify an important direction for a specific input. We apply XAI to neural network predictions of the El Niño-Southern Oscillation (ENSO) based on data from the Zebiak-Cane model.   AGOP XAI achieves the highest attribution, robustness, and coherence scores in the architecture and lead-time comparisons reported here. Its sensitivity is surpassed by gradient XAI, which is maximally sensitive by definition. Beyond diagnosing neural-network behavior, AGOP XAI can generate candidate hypotheses about physical mechanisms. The method highlights an equatorial thermocline-depth signal consistent with recharge oscillator physics, together with a southeastern-Pacific lobe that may be specific to the Zebiak-Cane model. Finally, we test the physical relevance of AGOP using optimized perturbations that move the Zebiak-Cane model along AGOP explanation coordinates. Such perturbations can suppress the selected extreme events or, from a near-neutral ensemble, generate strong El Niño or La Niña events 10 months later.</span> <span class="abstract-toggle" data-id="2610.01095">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.01095v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.01095v1) · [:fontawesome-brands-github: Code](https://github.com/rjwebber/agop-xai) · [:material-content-copy: BibTeX](../../bibtex/2610.01095.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=subseasonal-to-seasonal" data-tag="subseasonal-to-seasonal">Subseasonal to seasonal</a> <a class="md-tag" href="/explore/?t=interpretability" data-tag="interpretability">Interpretability</a>
    { .paper-tags }

-   #### A library for differentiable signal processing and machine learning on the sphere { #2609.39737 }

    *Thorsten Kurth, Max Rietmann, Mauro Bisson, Andrea Paris, Alberto Carpentieri, Jean Kossaifi et al.* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.39737">The two-dimensional sphere embedded in three-dimensional Euclidean space S2, plays a central role in a variety of scientific and engineering domains, including geophysics, planetary science, geodesy,...</span><span class="abstract-full" id="full-2609.39737" hidden>The two-dimensional sphere embedded in three-dimensional Euclidean space S2, plays a central role in a variety of scientific and engineering domains, including geophysics, planetary science, geodesy, atmospheric physics, quantum chemistry, cosmology, and virtual reality, among many others. As machine learning increasingly permeates these fields, the demand grows for robust tools that process and model functions on the sphere, while respecting the inherent topological and symmetry properties of the domain. We present torch-harmonics, a comprehensive library that offers efficient, differentiable implementations of advanced signal processing and machine learning (ML) methods for spherical data. These include the spherical harmonic transform (SHT), the spherical analogue of the Fourier transform, vector spherical harmonics, discrete-continuous and spectral convolutions, as well as both global and neighborhood spherical attention mechanisms. Beyond traditional representations, torch-harmonics provides the building blocks for state-of-the-art spherical ML architectures such as spherical transformers in order to enable scalable, rotationally-aware learning and inference in modern scientific and engineering applications.</span> <span class="abstract-toggle" data-id="2609.39737">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.39737v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.39737v1) · [:material-content-copy: BibTeX](../../bibtex/2609.39737.bib){ .bibtex-link }
    { .paper-links }

-   #### Methodological Changes to the Attention ResUNet Hourly Precipitation Postprocessor { #2609.38609 }

    *Thomas M. Hamill* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.38609">This note is a technical companion to a previously published preprint describing an Attention Residual U-Net that postprocesses deterministic forecasts from The Weather Company's Global and Regional...</span><span class="abstract-full" id="full-2609.38609" hidden>This note is a technical companion to a previously published preprint describing an Attention Residual U-Net that postprocesses deterministic forecasts from The Weather Company's Global and Regional Atmospheric Forecast (GRAF) model into probabilistic hourly precipitation forecasts. It documents what has changed in that method since publication. Feature-wise Linear Modulation conditioning on calendar season and forecast lead time is used to produce a single trained model for each season, replacing 192 separately trained per-month, per-lead checkpoints. Lead time is extended from 48 to 72 h. Two new input channels are used, per-pixel local solar hour and a static, monthly-varying precipitation climatology. During verification, the climatological reference against which the Brier Skill Score is computed now has an added diurnal dimension, on top of the monthly resolution it already had. Brier Skill Score and reliability are compared between the new vs. the previous training. Forecasts generated with the new training show a modest, consistent improvement of the current training over the original.</span> <span class="abstract-toggle" data-id="2609.38609">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.38609v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.38609v1) · [:material-content-copy: BibTeX](../../bibtex/2609.38609.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=hourly" data-tag="hourly">Hourly</a> <a class="md-tag" href="/explore/?t=monthly" data-tag="monthly">Monthly</a>
    { .paper-tags }

-   #### Predicting Delayed Train Trajectories on the Dutch Railway Network: Explainable AI Evaluation of Topological, Operational and Weather Features with Tree Based Ensemble Methods { #2609.34692 }

    *Jia Long Bao, Ali Mohammed Mansoor Alsahag, Seyed Sahand Mohammadi Ziabari* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.34692">The reliable prediction of passenger train delays is a critical component of railway management. While contemporary research frequently attempts to maximize absolute accuracy by deploying opaque deep...</span><span class="abstract-full" id="full-2609.34692" hidden>The reliable prediction of passenger train delays is a critical component of railway management. While contemporary research frequently attempts to maximize absolute accuracy by deploying opaque deep learning architectures, the underlying data mechanics driving longitudinal predictive decay remain underexplored. Consequently, this study provides an explainable temporal robustness analysis of network-wide railway delay prediction. Focusing on the Dutch railway network, this research utilizes interpretable tree-based ensembles to integrate granular topological, environmental, and operational features. The overarching finding establishes that while feature-rich tree-based models improve simultaneous (within-month) prediction, predictive performance systematically degrades when evaluated across non-simultaneous (future) months. Furthermore, multi-horizon SHAP and dispersion analyses explicitly link this degradation to environmental feature volatility and instability within the statistical target definition. Ultimately, this thesis demonstrates that richer feature sets alone are insufficient to resolve long-term forecasting constraints, underscoring the necessity to transition toward dynamic, season-aware architectures anchored by absolute operational boundaries.</span> <span class="abstract-toggle" data-id="2609.34692">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.34692v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.34692v1) · [:material-content-copy: BibTeX](../../bibtex/2609.34692.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=interpretability" data-tag="interpretability">Interpretability</a>
    { .paper-tags }

-   #### StatD2GAN: When Calibration Masks Generator Quality in Held-Out Evaluation of Synthetic Weather Sequences { #2609.33761 }

    *Mustafa Ozaytac, Ozge Karadag Atas* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.33761">Generative models for multivariate weather series are routinely evaluated with pooled distributional metrics computed after marginal calibration. We show this practice can invalidate architectural...</span><span class="abstract-full" id="full-2609.33761" hidden>Generative models for multivariate weather series are routinely evaluated with pooled distributional metrics computed after marginal calibration. We show this practice can invalidate architectural conclusions, and rebuild the evaluation of StatD2GAN, a three-discriminator GAN with evolutionary weight adaptation, around a held-out protocol: the final two calendar years of each dataset are held out behind a 168 hour embargo, calibration is fitted on the training block only, and all metrics are computed on the held-out block. Evidence comes from 25 matched (location, seed) pairs across five Koppen-Geiger climates, tested with Wilcoxon signed-rank tests under Holm correction. Four results follow. First, isotonic calibration drives the Kolmogorov-Smirnov distance to within 2% of a per-location noise-and-shift floor for every architecture tested, including a deliberately weak RCGAN baseline, so calibrated marginal metrics cannot discriminate between architectures. Second, the sorted-representation discriminator is the only component whose removal significantly degrades cross-variable dependence (Kendall tau MAE +0.080, Holm p = 0.009), with a regime-dependent effect: near zero in Ankara, above 115% in Dubai and Yakutsk. A rank-transformed variant isolates the mechanism as quantile supervision of the marginals rather than copula matching. Third, physical constraint violations are injected by calibration, not the generator; projection removes them at negligible cost (deltaKS <= 0.003). Fourth, pooled metrics conceal a collapse of between-sequence weekly-mean variability, a proxy for seasonal and regime diversity, in TimeGAN that only sequence-level statistics expose. We recommend floor-referenced marginal evaluation, matched-pair testing, and sequence-level variance decomposition as minimum requirements for calibrated generative pipelines.</span> <span class="abstract-toggle" data-id="2609.33761">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.33761v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.33761v1) · [:material-content-copy: BibTeX](../../bibtex/2609.33761.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=gans" data-tag="gans">GANs</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### Mechanism-Aware Ensemble Conditioning for Data-Limited Emulation of Extreme Events { #2609.30746 }

    *Isabella S. Thiel, Juan Bello-Rivas, Yannis G. Kevrekidis, Themistoklis P. Sapsis* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.30746">Extreme events in chaotic systems are difficult to learn from short trajectories because they are controlled by transient finite-time instability rather than by frequently observed bulk dynamics. We...</span><span class="abstract-full" id="full-2609.30746" hidden>Extreme events in chaotic systems are difficult to learn from short trajectories because they are controlled by transient finite-time instability rather than by frequently observed bulk dynamics. We propose a mechanism-aware conditioning plug-in framework that turns a nudged coarse ensemble into a non-intrusive sensor of local instability geometry. In the small-noise regime, the ensemble covariance aggregates the same finite-time deformation kernels that govern local instability, providing a Jacobian-free proxy for the local amplification structure around a synchronized coarse trajectory. A small FiLM module injects statistics of this ensemble geometry into an otherwise unchanged backbone while leaving the coarse simulator unchanged. We demonstrate this interface in two distinct pipelines: a Transformer-style residual-attention corrector for a controlled low-dimensional chaotic system and a probabilistic recurrent STORN corrector for topographic two-layer quasi-geostrophic (QG) flow. In the low-dimensional benchmark, ensemble covariance directions co-activate with OTD modes and FiLM conditioning improves 99th-percentile exceedance-frequency errors over an identical no-context Transformer baseline. In QG, a fixed ensemble-conditioned FiLM-STORN model trained on only \(50\) time units substantially improves long-horizon rare-event statistics in the data-limited regime, including density-tail errors, exceedance frequencies, and spatial exceedance-area distributions relative to an unconditioned STORN trained on the same data; on averaged high-threshold exceedance diagnostics, it also outperforms the baseline STORN trained with $20$ times more high-resolution data. These results show that local instability geometry is not merely interpretable post hoc, but an actionable conditioning signal for data-efficient rare-event emulation.</span> <span class="abstract-toggle" data-id="2609.30746">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.30746v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.30746v1) · [:material-content-copy: BibTeX](../../bibtex/2609.30746.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=transformers" data-tag="transformers">Transformers</a> <a class="md-tag" href="/explore/?t=extremes" data-tag="extremes">Extremes</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### PISCES: Physics-Informed Solar-wind Convolutional autoEncoder for Space-weather Anomaly Detection and Early Warning { #2609.28022 }

    *Kevin Lee, Alison J. March* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.28022">Space weather early warning depends on detecting solar wind transients in in-situ measurements at the first Sun-Earth Lagrange point (L1), before they reach Earth. Fixed thresholds can miss combined...</span><span class="abstract-full" id="full-2609.28022" hidden>Space weather early warning depends on detecting solar wind transients in in-situ measurements at the first Sun-Earth Lagrange point (L1), before they reach Earth. Fixed thresholds can miss combined magnetic and plasma structure, and many learning methods provide a single anomaly score. We present the Physics-Informed Solar-wind Convolutional autoEncoder for Space-weather (PISCES), a convolutional autoencoder trained without catalog labels on OMNI solar wind measurements under physics constraints. Its loss includes magnetic field consistency, an empirical relation between temperature and velocity, the Parker spiral angle, and penalties on changes between consecutive one-minute samples in derived quantities calculated from the reconstruction. At inference, PISCES separates the anomaly score into magnetic and plasma reconstruction errors, physics relations, and residual corrections, and reports the magnitude of each contribution. Attenuation of the skip connections, selected on validation data, improves average precision for the trained models, while the untrained scores remain nearly the same. The trained models also give a more consistent ordering of these physical contributions. After smoothing with a trailing median, the alarms can precede independently observed sudden commencements, including positive sudden impulses.</span> <span class="abstract-toggle" data-id="2609.28022">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.28022v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.28022v1) · [:fontawesome-brands-github: Code](https://github.com/magnaprog/PISCES) · [:material-content-copy: BibTeX](../../bibtex/2609.28022.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=physics-ml-hybrid" data-tag="physics-ml-hybrid">Physics–ML hybrid</a>
    { .paper-tags }

-   #### Sparse-Observation Atmospheric Thermal Forecasting with Physics-Informed Neural Networks for Climate-Aware Digital Twins { #2609.27290 }

    *Tannaz Goodarzvand Chegini, Elyas Shivanian, Behzad Karimi, Faraz Dadgostari* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.27290">Short-horizon forecasts of atmospheric temperature are needed to support climate-aware digital-twin systems, but such forecasts must be produced where thermal observations are incomplete. This study...</span><span class="abstract-full" id="full-2609.27290" hidden>Short-horizon forecasts of atmospheric temperature are needed to support climate-aware digital-twin systems, but such forecasts must be produced where thermal observations are incomplete. This study evaluates a physics-informed neural network for potential-temperature forecasting, constrained by a pressure-coordinate thermodynamic advection-source equation and a diabatic-source closure fit from the preceding 12-hour period and frozen before future-time training. Using hourly ERA5 reanalysis at three pressure levels, the model is evaluated as a conditional hindcast at lead times of one, two and three hours against persistence, local-trend, and two matched neural-network baselines, one of which receives the same future meteorological forcing as the PINN, helping distinguish the physical constraint from access to future forcing. In an Oklahoma development case, mean RMSE improvement over the strongest baseline grew from 8.1% at one hour to 23.8% at three hours; under an observation-density sweep down to 5% of candidate locations, this 3-hour advantage remained 14.6--16.9%, with no evidence that lower density improves performance. Under a fixed protocol transferred to an Alabama heat event with three virtual-observation layouts, three-hour improvement ranged 19.7-24.4% with consistent origin-level wins. A parallel Montana stress test, in which fixed pressure levels intersected complex terrain, produced a three-hour degradation of roughly 17.5%, identifying a terrain-related applicability limit of the formulation. Together, these results indicate that the physics constraint's benefit grows with forecast horizon, persists under severe observation sparsity, and transfers across regions, but is bounded by the validity of a fixed vertical-coordinate representation over complex terrain, evidence relevant to physics-constrained components of climate-aware forecasting and digital-twin systems.</span> <span class="abstract-toggle" data-id="2609.27290">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.27290v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.27290v1) · [:material-content-copy: BibTeX](../../bibtex/2609.27290.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=physics-ml-hybrid" data-tag="physics-ml-hybrid">Physics–ML hybrid</a> <a class="md-tag" href="/explore/?t=hourly" data-tag="hourly">Hourly</a>
    { .paper-tags }

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

</div>

<nav class="pager" markdown="span">**1** [2](2.md) [3](3.md) [4](4.md) [5](5.md) [6](6.md) [7](7.md) [Older :material-arrow-right:](2.md){ .pager-step }</nav>

