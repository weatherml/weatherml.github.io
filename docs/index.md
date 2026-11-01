---
hide:
  - navigation
  - toc
title: weatherml
---

A collection of papers on AI for weather forecasting, climate modelling and atmospheric science.

<p class="page-meta" markdown="span">1602 papers · updated 2026-11-01 · <a href="feed.xml">:material-rss: RSS</a> · <a href="all_papers.bib" download>:material-download: BibTeX</a> · <a href="https://github.com/weatherml/weatherml.github.io/issues/new?template=suggest-paper.yml">:material-plus: Suggest a paper</a></p>

## Browse by Topic

<div class="grid cards topics" markdown>

-   [Global Models](papers/global-models/index.md) <span class="topic-count">380</span>
-   [Regional Models](papers/regional-models/index.md) <span class="topic-count">67</span>
-   [Nowcasting](papers/nowcasting/index.md) <span class="topic-count">106</span>
-   [Downscaling](papers/downscaling/index.md) <span class="topic-count">108</span>
-   [Post-processing](papers/post-processing/index.md) <span class="topic-count">26</span>
-   [Data Assimilation](papers/data-assimilation/index.md) <span class="topic-count">80</span>
-   [Climate Modeling](papers/climate-modeling/index.md) <span class="topic-count">307</span>
-   [Hydrology](papers/hydrology/index.md) <span class="topic-count">39</span>
-   [Ocean & Sea Ice](papers/ocean-sea-ice/index.md) <span class="topic-count">89</span>
-   [Air Quality & Composition](papers/air-quality-composition/index.md) <span class="topic-count">78</span>
-   [Remote Sensing](papers/remote-sensing/index.md) <span class="topic-count">102</span>
-   [Other](papers/other/index.md) <span class="topic-count">220</span>

</div>

## Recent Additions

<div class="grid cards" markdown data-search-exclude>

-   #### From extreme weather to agricultural consequences with agentic AI { #2610.40641 }

    *Minghan Yu, Mengyu Dai, Yixin Wen, Haizhao Yang* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.40641">Extreme weather is forecast days ahead, yet its consequences for agriculture remain hard to predict: a forecast says where a storm will strike, not which crops will suffer or how much is at risk, and...</span><span class="abstract-full" id="full-2610.40641" hidden>Extreme weather is forecast days ahead, yet its consequences for agriculture remain hard to predict: a forecast says where a storm will strike, not which crops will suffer or how much is at risk, and the records that could tell span many sources. We present WxCrop, the first agent that judges agricultural hazards from physically interpretable weather indices examined by causal analysis, and the first that combines weather and agricultural information to predict crop loss reliably for past and forecast dates. We demonstrate it in Florida on hurricanes, freezes and dry heat, with a loss model calibrated on insurance records for each, and the same procedure extends to other hazards and areas. Three checks show reliability: the hurricane index detects recorded cyclone days with an AUC of 0.98, recorded losses in held-out years are reproduced to within 2%, and independent production records confirm the county rankings (rank correlation 0.63). Users ask the agent for each crop group's loss as a fraction of expected season value, every number sourced, so that governments, growers and scientists respond to agricultural disasters quickly.</span> <span class="abstract-toggle" data-id="2610.40641">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.40641v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.40641v1) · [:material-content-copy: BibTeX](bibtex/2610.40641.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=llms-agents" data-tag="llms-agents">LLMs & agents</a> <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a> <a class="md-tag" href="/explore/?t=extremes" data-tag="extremes">Extremes</a>
    { .paper-tags }

-   #### Causal-prior-constrained deep learning for rapid-intensification warning of western North Pacific tropical cyclones { #2610.40304 }

    *Xinrui Li, Yijun Jin, Yifan Dong, Shoujuan Shu* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.40304">Rapid intensification (RI) is one of the most dangerous and unpredictable phases of tropical cyclones, and its environmental and structural drivers are difficult to identify and separate in...</span><span class="abstract-full" id="full-2610.40304" hidden>Rapid intensification (RI) is one of the most dangerous and unpredictable phases of tropical cyclones, and its environmental and structural drivers are difficult to identify and separate in observations. Here we introduce CausalVisionRI, a causal-prior-constrained deep-learning system for 24-h RI warning in the western North Pacific. It integrates infrared remote sensing imagery, atmospheric and oceanic fields, and an observationally discovered causal graph into a unified model. The graph and eight sign-verified effects on 24-h intensity change provide an auditable causal prior for training. Discovered from 522 cyclones in 2000-2017, it identifies causal pathways linking atmospheric and oceanic conditions, the cyclone's dynamical and structural state, and its 24-h intensity change. On a forward holdout covering 2020-2024 and 3,004 records, the five-member ensemble achieves a Brier skill score (BSS) of 0.259 relative to a fixed development climatology, significantly exceeding four reference systems; the three LDA baselines reach at most 0.105. It also attains higher detection rates than a published knowledge-block scheme and a reconstructed contrastive-learning scheme at matched false-detection rates under our USA 30-kt standard and the published RITCF standard. Ablations show that the graph pathway adds 0.032 BSS across the holdout and 0.063 over 2020-2021, with a significant contribution in one ensemble member. A deep-learning model constrained by observationally discovered causal structure thus offers a new route to physically informed RI warning.</span> <span class="abstract-toggle" data-id="2610.40304">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.40304v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.40304v1) · [:material-content-copy: BibTeX](bibtex/2610.40304.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a>
    { .paper-tags }

-   #### Operator-Aware Posterior Sampling for Multivariate Weather Downscaling { #2610.39592 }

    *Carlo Saccardi, Rudolf Saathof, Geethu Joseph, Justin Dauwels* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.39592">Statistical downscaling is the task of reconstructing high-resolution meteorological fields from coarse inputs. A principled approach formulates downscaling as a Bayesian inverse problem, using a...</span><span class="abstract-full" id="full-2610.39592" hidden>Statistical downscaling is the task of reconstructing high-resolution meteorological fields from coarse inputs. A principled approach formulates downscaling as a Bayesian inverse problem, using a diffusion prior over high-resolution fields and incorporating the coarse field only at inference through a likelihood. This decouples the prior from the coarse source, allowing one pretrained model to adapt to different sources. However, two challenges limit this flexibility: (1) when coarse and target fields come from different reanalysis systems, no fixed coarsening operator can capture their relationship; and (2) guidance gradients can differ by orders of magnitude across variables, making a single guidance strength ineffective. This paper addresses both challenges within the guidance loop, leaving the prior unchanged. We learn the forward operator from paired data and normalize the gradient per channel. We show that this normalization corresponds to steepest descent under a block-max norm, allowing a single guidance strength across variables. Our code is publicly available at https://github.com/CarloSaccardi/operator-aware-weather-downscaling.</span> <span class="abstract-toggle" data-id="2610.39592">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.39592v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.39592v1) · [:fontawesome-brands-github: Code](https://github.com/CarloSaccardi/operator-aware-weather-downscaling) · [:material-content-copy: BibTeX](bibtex/2610.39592.bib){ .bibtex-link }
    { .paper-links }

-   #### Skillful and realistic precipitation forecasting with reinforced diffusion { #2610.38775 }

    *Ozan Sener, Anand Avati, David Hafner, Jierui Lin, Laura Manduchi, Xiaoying Pang, Langwen Huang et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.38775">Accurate precipitation forecasting remains elusive as physics-based models struggle to resolve mesoscale convection and machine learning often yields physically inconsistent, blurred results that...</span><span class="abstract-full" id="full-2610.38775" hidden>Accurate precipitation forecasting remains elusive as physics-based models struggle to resolve mesoscale convection and machine learning often yields physically inconsistent, blurred results that consistently underestimate extreme events. We present FitzRoy, the first system that simultaneously achieves state-of-the-art predictive skill and physical realism over the continental United States. Through a reinforced diffusion framework, where a conditional discrete diffusion model is refined via reinforcement learning, FitzRoy effectively resolves the long-standing trade-off between pointwise accuracy and structural fidelity. Across lead times up to 24 hours, FitzRoy consistently matches or exceeds state-of-the-art models while preserving structural realism and capturing extreme events. Its low-latency inference further enables deployment in real-time operational settings. In case studies such as the 2025 Texas Hill Country floods and the 2026 Missouri floods, FitzRoy signals record-breaking accumulations that existing models largely missed. By bridging the gap between realism and skill, FitzRoy establishes a new frontier for high-fidelity precipitation forecasting.</span> <span class="abstract-toggle" data-id="2610.38775">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.38775v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.38775v1) · [:material-content-copy: BibTeX](bibtex/2610.38775.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=reinforcement-learning" data-tag="reinforcement-learning">Reinforcement learning</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a>
    { .paper-tags }

-   #### Curriculum-Based Deep Reinforcement Learning for Robust Autonomous Rocket Landing under Atmospheric Uncertainty { #2610.38636 }

    *Atin Chowdhury, Piyush Joshi* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.38636">Autonomous far-range rocket landing is a challenging control problem because navigation, descent-rate regulation, attitude management, and fuel use must be coordinated in the presence of uncertain...</span><span class="abstract-full" id="full-2610.38636" hidden>Autonomous far-range rocket landing is a challenging control problem because navigation, descent-rate regulation, attitude management, and fuel use must be coordinated in the presence of uncertain atmospheric and vehicle dynamics. This paper presents a curriculum-driven proximal policy optimization (PPO) framework for three-dimensional rocket landing in a stochastic environment incorporating altitude-dependent aerodynamic drag, wind shear, temporally correlated turbulence, discrete gusts, uncertain propulsion and aerodynamic parameters, thrust-vector misalignment, engine lag, actuator-gain variation, noisy observations, and command latency. Learning is organised into far-range approach, soft descent, and integrated landing stages. Policy transfer, critic warm-up after reward changes, rehearsal of earlier tasks, and progressive expansion of weather severity are used to support learning across these coupled phases. Across 100,000 evaluation episodes, the policy achieves 96,656 successful landings, corresponding to a success rate of 96.656% and a 95% Wilson confidence interval of 96.543--96.766%. Success decreases from 98.84% under zero-disturbance conditions to 88.07% in the highest weather-severity band. Successful touchdowns exhibit small miss distances and low residual horizontal and vertical velocities. An exploratory logistic-regression analysis identifies mean wind speed as the strongest measured correlate of success, with initial distance, initial altitude, initial vertical velocity, and wind-shear exponent also associated with outcomes. The results quantify both the effectiveness of the learned policy over the evaluated disturbance distribution and the degradation that occurs under severe compound atmospheric conditions.</span> <span class="abstract-toggle" data-id="2610.38636">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.38636v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.38636v1) · [:material-content-copy: BibTeX](bibtex/2610.38636.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=reinforcement-learning" data-tag="reinforcement-learning">Reinforcement learning</a>
    { .paper-tags }

-   #### RainCheck: Harnessing Historical Weather Data with Machine Learning for Rainfall Forecasting in Bangladesh { #2610.38211 }

    *Shahran Rahman Alve, Muhammad Zawad Mahmud, Samiha Islam, Al Jubayer Pial* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.38211">Machine-learning rainfall forecasts are often evaluated by shuffling station records into random training and test sets. For rainfall, this lets a model see what happened at neighbouring stations on...</span><span class="abstract-full" id="full-2610.38211" hidden>Machine-learning rainfall forecasts are often evaluated by shuffling station records into random training and test sets. For rainfall, this lets a model see what happened at neighbouring stations on the very day it is asked to forecast, and it hides the fact that a seasonal lookup table already explains most of the variance. We quantify both effects on 618,650 station-days from 35 Bangladesh Meteorological Department stations (1948-2014). We compare random, temporal (train <=1999, test 2005-2014) and spatio-temporal (unseen stations and unseen years) evaluation, calendar-only versus antecedent-rainfall features, and six learners against climatology, persistence and SARIMA. Three findings stand out. First, a station-month climatology alone classifies high-rainfall months with 88.4% accuracy (AUC 0.950), and no learner improved on it by more than 0.8 points under temporal evaluation. Second, a calendar-only random forest reached a next-day wet-day AUC of 0.889 under a random split, higher than any model with real predictors under temporal evaluation, yet fell to 0.835, below climatology (0.859), on future years. Third, models with lagged rainfall add genuine but modest skill for next-day forecasts: a Brier skill score of 0.074 (95% CI 0.064-0.088) over climatology, rising to 0.091 when station and year inputs are removed and to 0.131 when other stations' same-day observations are added. A GRU sequence model matches but does not exceed gradient boosting, and skill disappears beyond a two-day lead. For monthly totals, no learner beat climatology significantly. We release a leakage-aware protocol and code so that future station-based studies report skill against climatology on held-out years.</span> <span class="abstract-toggle" data-id="2610.38211">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.38211v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.38211v1) · [:material-content-copy: BibTeX](bibtex/2610.38211.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=recurrent-networks" data-tag="recurrent-networks">Recurrent networks</a> <a class="md-tag" href="/explore/?t=classical-ml" data-tag="classical-ml">Classical ML</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=station-point" data-tag="station-point">Station / point</a> <a class="md-tag" href="/explore/?t=monthly" data-tag="monthly">Monthly</a>
    { .paper-tags }

-   #### Multi-Agent Reinforcement Learning for Global Climate Policy Cooperation { #2610.37870 }

    *Anne Moll-Elsborg, Oskar Bohn Lassen, Francisco Camara Pereira* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.37870">International climate policy requires coordination among countries whose mitigation decisions impose local costs but generate global benefits. Classical game-theoretic and integrated assessment...</span><span class="abstract-full" id="full-2610.37870" hidden>International climate policy requires coordination among countries whose mitigation decisions impose local costs but generate global benefits. Classical game-theoretic and integrated assessment approaches can study these incentives, but typically rely on prescribed behavioural assumptions or equilibrium concepts. Recent work has therefore explored Multi-Agent Reinforcement Learning as a way to model climate-policy cooperation as an emergent outcome of repeated strategic interaction. We extend an existing MARL climate framework with a multi-stage negotiation mechanism in which agents propose agreements, accept or reject commitments, and face penalties for non-compliance. This allows us to study how agreement design affects learned mitigation behaviour, compliance, and climate outcomes. In an idealized setting with four homogeneous agents, we find that non-binding agreements do little to overcome free-riding, while moderate non-compliance penalties improve cooperation and reduce warming. However, excessively strong penalties discourage ambitious commitments, revealing a trade-off between enforcement and participation. We position the framework as a methodological contribution for testing agreement design rather than a predictive model of real negotiations.</span> <span class="abstract-toggle" data-id="2610.37870">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.37870v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.37870v1) · [:material-content-copy: BibTeX](bibtex/2610.37870.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=reinforcement-learning" data-tag="reinforcement-learning">Reinforcement learning</a> <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a>
    { .paper-tags }

-   #### KPPax: preprocess differentiable kinetic ODE for machine learning in atmospheric chemistry { #2610.37655 }

    *Wenqing Peng, Michael Boy, Zhi-Song Liu* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.37655">Atmospheric chemical mechanisms are commonly distributed in the language of the Kinetic PreProcessor (KPP), which generates efficient Fortran or C code but does not fit into the differentiable...</span><span class="abstract-full" id="full-2610.37655" hidden>Atmospheric chemical mechanisms are commonly distributed in the language of the Kinetic PreProcessor (KPP), which generates efficient Fortran or C code but does not fit into the differentiable programming frameworks in which machine-learning models are developed. We present KPPax, a kinetic preprocessor that generates differentiable, just-in-time compiled, and vectorisable chemistry code in JAX from mechanisms written in the KPP language. KPPax evaluates the rate law with sparse operations in log space, which keeps it differentiable, and integrates the stiff system with the implicit solvers and the forward- and reverse-mode adjoints of the <code>diffrax</code> library, so that exact sensitivities are obtained without hand-derived adjoint code. We demonstrate two applications with the POLLU benchmark mechanism. In the first, the Fisher information matrix computed from an automatically differentiated Jacobian predicts, before any fit, which of the 25 rate coefficients a flow-tube-like design can identify, and noisy synthetic estimation confirms the predicted uncertainties of the nine identifiable rate coefficients to within about 30% while revealing clusters of fast reactions of which only ratios are constrained. In the second, a neural-network correction of an emission inventory is trained offline on per-scenario 4D-Var retrievals and then fine-tuned online through the chemical model, which reduces the error of the biogenic correction by a factor of 2.4--5 and recovers information on low-temperature emissions that the individual retrievals lose. KPPax makes KPP mechanisms directly usable for parameter estimation and hybrid physics--machine-learning modelling.</span> <span class="abstract-toggle" data-id="2610.37655">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.37655v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.37655v1) · [:material-content-copy: BibTeX](bibtex/2610.37655.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=physics-ml-hybrid" data-tag="physics-ml-hybrid">Physics–ML hybrid</a>
    { .paper-tags }

-   #### An Input Skip Can Reparameterize a VAE Without Improving the Generator { #2610.37456 }

    *Kevin Lee* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.37456">An input skip can reduce reconstruction error while leaving a model's distribution of new samples unchanged. For a Gaussian variational autoencoder (VAE) with constant input skip $0\leqα<1$, a model...</span><span class="abstract-full" id="full-2610.37456" hidden>An input skip can reduce reconstruction error while leaving a model's distribution of new samples unchanged. For a Gaussian variational autoencoder (VAE) with constant input skip $0\leqα<1$, a model without a skip can be written in skip form by multiplying its decoder output by $b=1-α$ and its noise variance by $b^2$. The distribution of observations and latent variables and its exact generative scores stay the same, while reconstruction mean squared error (MSE) is multiplied by $b^2$. Separate training can produce different generators because optimization depends on the parameter coordinates. In a Gaussian experiment with known density, changing skip and observation variance together reduces MSE but worsens energy score and the estimate of negative log density from importance sampling. On seven annual Deep-ocean Assessment and Reporting of Tsunamis (DART) records, energy score differences at the same observation variance have both signs on four original records and favor the skip on three records from 2018. Probabilistic principal component analysis (PPCA) has lower mean energy scores than every neural DART fit. With the encoder and decoder kept the same, estimating observation variance from training reconstruction error can also worsen energy score. Assessing generation requires more than reconstruction error.</span> <span class="abstract-toggle" data-id="2610.37456">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.37456v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.37456v1) · [:fontawesome-brands-github: Code](https://github.com/magnaprog/vae-skip-reparam) · [:material-content-copy: BibTeX](bibtex/2610.37456.bib){ .bibtex-link }
    { .paper-links }

-   #### Scale-aware and spatially-localised optimisation of kilometre-scale MLWP models { #2610.37325 }

    *James Chappell, Sophie Luise Arana, Jack Bowyer, Alejandro Coca-Castro, Evangeline Corcoran et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.37325">Preserving fine-scale structure without degrading spatial accuracy is a central challenge for kilometre-scale machine-learning weather prediction (MLWP). We introduce a loss based on the dual-tree...</span><span class="abstract-full" id="full-2610.37325" hidden>Preserving fine-scale structure without degrading spatial accuracy is a central challenge for kilometre-scale machine-learning weather prediction (MLWP). We introduce a loss based on the dual-tree complex wavelet transform that constrains atmospheric fields jointly by scale, location, and orientation. We evaluate its performance using an experimental joint global--regional FastNet configuration, which produces 1.5 km forecasts over the UK alongside $\sim$31 km global forecasts. Relative to both mean-squared error (MSE) and an FFT-based spectral objective, the wavelet loss substantially improves spectral fidelity. Compared with MSE training, it maintains similar pointwise performance, improves neighbourhood-based verification, and produces sharper precipitation fields with more realistic intensity distributions. These results demonstrate that spatially localised wavelet constraints can preserve multiscale atmospheric structure while largely avoiding the loss of pointwise accuracy associated with globally supported spectral objectives.</span> <span class="abstract-toggle" data-id="2610.37325">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.37325v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.37325v1) · [:material-content-copy: BibTeX](bibtex/2610.37325.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a> <a class="md-tag" href="/explore/?t=km-scale" data-tag="km-scale">Km-scale</a>
    { .paper-tags }

-   #### Noise Steering: Exploiting Wake Steering and Acoustic Directivity for Noise-Constrained Wind Farm Control via Reinforcement Learning { #2610.36888 }

    *Martín de Frutos, Baptiste Corban, Claire Bizon Monroc, Ana Bušić, Donatien Dubuc, Jiamin Zhud et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.36888">Existing noise mitigation strategies for wind farms usually consist in reducing turbines power output, without accounting for the wake interactions between them and the noise emission directivity....</span><span class="abstract-full" id="full-2610.36888" hidden>Existing noise mitigation strategies for wind farms usually consist in reducing turbines power output, without accounting for the wake interactions between them and the noise emission directivity. This leads to over-conservative policies and unexploited wind power. Yet, turbine yaw can simultaneously affect wake interactions and the direction of noise emission, offering an additional degree of freedom for farm-level noise control. We therefore propose Noise Steering, a model-free reinforcement learning (RL) strategy that jointly controls turbine yaw angles and power setpoints to maximise farm power while satisfying a local noise constraint in a defined region. To enable training at a reduced computational cost, a low-fidelity acoustic model is coupled with the steady-state FLORIS simulator. The RL agent effectively learns how to redirect wakes and noise directivity away from the protected region while redistributing power across the farm. On a six-turbine test farm, the policy satisfies the 40 dB noise constraint throughout the full range of turbulent winds, with an average power deficit of only 2% relative to greedy unimpeded control, and up to 7.7% at high wind speeds. In comparison, the industry-standard noise-reduced operation (NRO) baseline here generates an average power loss of 17%. When deployed in the higher-fidelity FAST.Farm simulator, the policy largely retains its performance under steady and turbulent inflow, despite differences in simulation fidelity. These results demonstrate the potential of combining yaw control and power derating through RL to improve the power-noise trade-off in wind farm operation.</span> <span class="abstract-toggle" data-id="2610.36888">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.36888v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.36888v1) · [:material-content-copy: BibTeX](bibtex/2610.36888.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=reinforcement-learning" data-tag="reinforcement-learning">Reinforcement learning</a> <a class="md-tag" href="/explore/?t=energy" data-tag="energy">Energy</a>
    { .paper-tags }

-   #### FASST: A Compact Stateless Atmosphere Model for Long-Term Ocean Coupling { #2610.36297 }

    *Pratham Aggarwal, Duncan Watson-Parris, William E. Chapman* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.36297">Coupled climate simulations are computationally expensive, largely because of the atmosphere. We introduce FASST, a coupled climate model that retains the prognostic POP2 ocean while replacing the...</span><span class="abstract-full" id="full-2610.36297" hidden>Coupled climate simulations are computationally expensive, largely because of the atmosphere. We introduce FASST, a coupled climate model that retains the prognostic POP2 ocean while replacing the atmosphere and sea ice with lightweight U-Net emulators. The atmospheric emulator is diagnostic, predicting surface fluxes from the evolving ocean-ice state every $6$ simulated hours, while the sea-ice emulator advances the ice state monthly. For $140$ years of unforced integration, FASST maintains a stable global-mean climate without SST restoring, with post-transient global-mean SST of $291.30 \pm 0.22\mathrm{K}$ compared with $291.25 \pm 0.10\mathrm{K}$ in CESM$2$ piControl. FASST also captures interannual Niño-$3.4$ variability, although with weaker amplitude than CESM ($0.62\mathrm{K}$ versus $1.02\mathrm{K}$). These results demonstrate that a stateless learned atmosphere can sustain stable, free-running coupled climate simulations over century-scale integrations.</span> <span class="abstract-toggle" data-id="2610.36297">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.36297v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.36297v1) · [:material-content-copy: BibTeX](bibtex/2610.36297.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a> <a class="md-tag" href="/explore/?t=efficiency" data-tag="efficiency">Efficiency</a> <a class="md-tag" href="/explore/?t=monthly" data-tag="monthly">Monthly</a>
    { .paper-tags }

-   #### Towards operational data-driven regional weather forecasting: extreme-event skill and near-real-time verification with HRRRCast { #2610.36186 }

    *Daniel Abdi, Isidora Jankov, Vanderlei Vargas, Paul Madden, Emily Carpenter, Travis Wilson et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.36186">Improving convective-scale skill in data-driven regional weather models requires better representing rare high-impact events, training at native resolution, broadening variable output, and...</span><span class="abstract-full" id="full-2610.36186" hidden>Improving convective-scale skill in data-driven regional weather models requires better representing rare high-impact events, training at native resolution, broadening variable output, and demonstrating performance in real-time settings. Here, we evaluate these challenges using HRRRCast, a data-driven convection-allowing emulator trained on the High-Resolution Rapid Refresh (HRRR) analysis data. HRRRCast combines native 3-km training, moderate sigmoid-weighted loss reweighting for reflectivity and precipitation, and broader system improvements including expansion of the training archive from 3 to 9 years (2016--2025), increased vertical resolution, expanded outputs, and architectural upgrades. The weighted loss improves high-threshold skill while preserving common-event behavior. Native-resolution training avoids aliasing from 6-km subsampling and substantially improves fractions skill score at convective scales, with gains that persist across pooling windows. HRRRCast has also been running in near-real-time mode for the past year. In seven months of independent observation-based verification, HRRRCast ensemble means improve several near-surface metrics relative to operational HRRR and maintain strong precipitation and reflectivity performance. Single-member reflectivity skill is strongest at 20--30 dBZ; at 40 dBZ, skill is near parity at 6-km pooling and better at pooling radii $\geq$30 km. Precipitation skill remains competitive through 1 in/h and extends to about 3 in/h in the best recent configuration. A focused comparison with HREF further shows useful low-cost probabilistic guidance for 3-h precipitation. At storm scale, HRRRCast exhibits a displacement--structure trade-off, with generally better storm placement than storm morphology.</span> <span class="abstract-toggle" data-id="2610.36186">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.36186v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.36186v1) · [:material-content-copy: BibTeX](bibtex/2610.36186.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=evaluation" data-tag="evaluation">Evaluation</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a>
    { .paper-tags }

-   #### GNN-VIEW: Visual Inspection and Explanation of Graph-based Earth-System Models { #2610.35693 }

    *Alessandro Lucatelli, Ferdinand Unterhuber, Katharina Krenzer, Mennatallah El-Assady, Benedikt Soja et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.35693">Machine learning Earth-system models are becoming increasingly important, and so is their development. Graph-based models, one of the most prominent approaches, allow developers to constrain the...</span><span class="abstract-full" id="full-2610.35693" hidden>Machine learning Earth-system models are becoming increasingly important, and so is their development. Graph-based models, one of the most prominent approaches, allow developers to constrain the learning process before training via graph connectivity. Yet, developing these models remains challenging: the graph is built algorithmically with little intuition about the final result beyond static plotting, which requires extra work and is often not detailed enough. Even after successful training, how the model arrives at forecasts remains obscure. We present GNN-VIEW, an interactive browser-based tool that supports the development of such systems. GNN-VIEW exposes the graph's information pathways, allowing model developers to debug the graph structure before training. After training, GNN-VIEW uses attribution-based explainability to reveal which input features are used by the trained model for a given forecast. This helps developers understand why a model fails. We showcase the usefulness of GNN-VIEW in the construction stage of a graph for an ocean forecasting model, and in explaining poorly predicted precipitation of a trained weather forecasting model during Hurricane Laura.</span> <span class="abstract-toggle" data-id="2610.35693">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.35693v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.35693v1) · [:material-content-copy: BibTeX](bibtex/2610.35693.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=graph-neural-networks" data-tag="graph-neural-networks">Graph neural networks</a> <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a>
    { .paper-tags }

-   #### Application of Artificial Intelligence in the Management of Climate Change-Related Health Issues in Africa { #2610.35294 }

    *Peter Kokol, Martha Kachweka, Helena Blazun Vosner, Samuel Chol Buol, Jernej Zavrsnik et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.35294">While climate change disproportionately impacts African public health, AI predictive modeling remains heavily constrained. Using a Synthetic Knowledge Synthesis approach and VOSviewer, we conducted a...</span><span class="abstract-full" id="full-2610.35294" hidden>While climate change disproportionately impacts African public health, AI predictive modeling remains heavily constrained. Using a Synthetic Knowledge Synthesis approach and VOSviewer, we conducted a bibliometric analysis of Scopus-indexed literature, comparing Global and African corpora. Network analysis reveals stark divergences: global research emphasizes complex, predictive models for planetary health, whereas African research focuses on localized, reactive solutions for immediate infectious disease triage. Severe structural barriers in Africa, including compute deficits, data silos, and energy instability, hinder advanced AI scaling. Ultimately, international funding must shift from siloed pilot projects toward building foundational digital infrastructure and localized AI frameworks.</span> <span class="abstract-toggle" data-id="2610.35294">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.35294v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.35294v1) · [:material-content-copy: BibTeX](bibtex/2610.35294.bib){ .bibtex-link }
    { .paper-links }

-   #### Machine Learning Applications in the Solar Interior and Atmosphere { #2610.34687 }

    *Vishal Upendran, Carlos J. Díaz Baso* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.34687">The solar interior and atmosphere constitute a strongly coupled, multiscale dynamical system defined by continuous temporal evolution. Predicting the state of this system requires modeling physical...</span><span class="abstract-full" id="full-2610.34687" hidden>The solar interior and atmosphere constitute a strongly coupled, multiscale dynamical system defined by continuous temporal evolution. Predicting the state of this system requires modeling physical interactions that span disparate spatial and temporal scales, constrained by heterogeneous and sparse data. This chapter formalizes the Sun within the mathematical framework of dynamical systems theory, defining a global state vector ($\vec{S}$) governed by thermodynamical, kinetic, and magnetic properties. Because solar observations represent projected, multi-modal, and partially observable measurements ($\vec{O}[\vec{S}]$) of this hidden state, pure physics-based deduction is fundamentally constrained. We present a unifying thesis: the primary role of machine learning in heliophysics is to act as a structured family of mathematical operators---specifically Inference ($\mathcal{I}$), Emulation ($\mathcal{E}$), Propagation ($\mathcal{P}$), and Control ($\mathcal{C}$)---that reconstruct, simulate, and couple this global state vector across disparate physical layers. By framing the vast, multi-disciplinary literature within this unified state-space taxonomy, we demonstrate how data-driven techniques bridge observational gaps to resolve hidden multiscale couplings and dramatically accelerate the simulation of non-linear processes, establishing a rigorous pathway to predict the global evolution of the Sun.</span> <span class="abstract-toggle" data-id="2610.34687">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.34687v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.34687v1) · [:material-content-copy: BibTeX](bibtex/2610.34687.bib){ .bibtex-link }
    { .paper-links }

-   #### An actively trained neural operator for near-explicit atmospheric chemistry { #2610.33771 }

    *Zhi-Song Liu, Wenqing Peng, Ammar Kheder, Risto Makkonen, Michael Boy* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.33771">Earth-system models simplify atmospheric chemistry into condensed mechanisms, because near-explicit mechanisms such as the Master Chemical Mechanism (MCM) are too costly to solve in every grid cell....</span><span class="abstract-full" id="full-2610.33771" hidden>Earth-system models simplify atmospheric chemistry into condensed mechanisms, because near-explicit mechanisms such as the Master Chemical Mechanism (MCM) are too costly to solve in every grid cell. Neural emulators could lift this limit, but most advance the chemistry in small time steps and cover only a few species or a reduced mechanism. Here we introduce FNO-MoE, a Fourier neural operator with a mixture of experts that predicts the full near-explicit chemical state over 30 minutes in a single call. The same network also guides its own training, using the disagreement among its experts to select the most informative box-model simulations. At equal cost, this active sampling yields a more accurate emulator than random or evenly spread sampling. Fine-tuning on longer trajectories keeps it stable over multi-hour runs in which models trained on short simulations diverge. The resulting emulator reproduces the evolution of thousands of species with high fidelity, remains accurate for unseen chemical compositions and runs orders of magnitude faster than the numerical solver. These results open a new direction for emulating complex atmospheric systems: neural operators that predict the full chemical state over long windows and choose their own training data. Training under changing meteorology is the next step towards bringing near-explicit chemistry into Earth-system models.</span> <span class="abstract-toggle" data-id="2610.33771">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.33771v2) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.33771v2) · [:material-content-copy: BibTeX](bibtex/2610.33771.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=neural-operators" data-tag="neural-operators">Neural operators</a> <a class="md-tag" href="/explore/?t=sub-hourly" data-tag="sub-hourly">Sub-hourly</a>
    { .paper-tags }

-   #### Nowcasting aerosol particle size distributions from PM2.5 with physics-regularized flow matching { #2610.33557 }

    *Zhi-Song Liu, Jenni Köykkä, Wenqing Peng, Ammar Kheder, Petri Clusius, Michael Boy* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.33557">The mass concentration of particles smaller than 2.5um (PM2.5) is cheap to measure and widely monitored, whereas the particle number size distribution (PSD) requires costly instruments and is...</span><span class="abstract-full" id="full-2610.33557" hidden>The mass concentration of particles smaller than 2.5um (PM2.5) is cheap to measure and widely monitored, whereas the particle number size distribution (PSD) requires costly instruments and is measured at only a few research stations. Yet many aerosol effects, such as vapor uptake, cloud droplet formation and deposition in the human respiratory tract, depend on the PSD rather than on mass. We explore whether recent generative deep learning models can nowcast the PSD: from the PM2.5 record and eight meteorological parameters over a 24h window, a model generates an ensemble of plausible hourly size distributions for the same window, reflecting that one mass can arise from many size distributions. Rather than proposing a single best model, we use two contrasting Finnish stations, the boreal forest site Hyytiälä and the urban site Kumpula, as a controlled testbed. On it, we benchmark six generative models against deterministic and climatological baselines, locally and after transfer between sites, and test physical constraints for neural network training. Flow matching performs best overall; its guided variant has the lowest continuous ranked probability score (CRPS) at both sites. A constraint based on a reduced aerosol general dynamic equation (GDE) lowers the CRPS by 5.5-11.5% at the training site and reduces errors after transfer between sites. Transfer remains harder than local estimation, identifying training on large multi-station data sets as the next step towards PSD estimates at unmonitored locations. This benchmark establishes physics-informed generative models as a promising route from routine air-quality monitoring to size-resolved aerosol information.</span> <span class="abstract-toggle" data-id="2610.33557">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.33557v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.33557v1) · [:material-content-copy: BibTeX](bibtex/2610.33557.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=physics-ml-hybrid" data-tag="physics-ml-hybrid">Physics–ML hybrid</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=hourly" data-tag="hourly">Hourly</a>
    { .paper-tags }

-   #### Climate downscaling with likelihood functional flow matching { #2610.33482 }

    *Zhi-Song Liu, Helmi Toropainen, Ammar Kheder, Michael Boy* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.33482">Decisions about heat, frost and flooding are made locally, yet reanalyses and climate models resolve tens of kilometres. Generative downscalers add kilometre-scale detail with uncertainty, but are...</span><span class="abstract-full" id="full-2610.33482" hidden>Decisions about heat, frost and flooding are made locally, yet reanalyses and climate models resolve tens of kilometres. Generative downscalers add kilometre-scale detail with uncertainty, but are tied to a grid, costly to sample and lack a tractable likelihood. Here we introduce CD-FFM, a functional flow-matching model that treats the high-resolution field as a continuous random field that one model can query at any location and resolution. Its velocity acts pointwise, so spatial variability comes from a noise prior fitted to the observed spectrum and every observation has an exact likelihood. Downscaling ERA5 to 1 km daily temperature over Fennoscandia, CD-FFM lowers the continuous ranked probability score by 27-31% against regression and 12-18% against a forty-times larger neural operator in each of three test years, and outperforms a CorrDiff-style diffusion model while sampling about fifty times faster. The same model transfers to precipitation and makes large kilometre-scale ensembles affordable.</span> <span class="abstract-toggle" data-id="2610.33482">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.33482v2) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.33482v2) · [:material-content-copy: BibTeX](bibtex/2610.33482.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=extremes" data-tag="extremes">Extremes</a> <a class="md-tag" href="/explore/?t=km-scale" data-tag="km-scale">Km-scale</a> <a class="md-tag" href="/explore/?t=daily" data-tag="daily">Daily</a>
    { .paper-tags }

-   #### Data Quality Audit and Bivariate Temperature-Humidity Analysis of a Long-Term LoRaWAN Network: Anomaly Detection, PCA, Clustering and Machine Learning { #2610.32438 }

    *Mamadou Sow* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.32438">Long-term environmental IoT/LPWAN deployments often aggregate heterogeneous data streams from diverse devices without systematic auditing of each variable's actual coverage prior to analysis-an...</span><span class="abstract-full" id="full-2610.32438" hidden>Long-term environmental IoT/LPWAN deployments often aggregate heterogeneous data streams from diverse devices without systematic auditing of each variable's actual coverage prior to analysis-an increased risk when multiple sensor types coexist within the same network. This work proposes a C++17 processing pipeline applied to the Tour Perret LoRaWAN dataset (421 937 frames, June 2021--June 2023, 5 devices). A systematic coverage audit integrated into the extraction phase reveals that CO2 is completely missing (0% coverage) and pressure is reported only by an isolated base module (Wyres, 4.24% of the corpus), invalidating the initial hypothesis of a joint four-variable analysis. The analysis is refocused on the Temperature--Humidity pair, jointly available across 205{,}005 frames (95.76%) from 4 Elsys EMS sensors. The pipeline sequentially performs anomaly detection (modified Z-score, Isolation Forest), bivariate Principal Component Analysis, clustering (K-means, DBSCAN), and a multilayer perceptron predicting humidity from temperature and time, whose residuals serve as a detector for slow drift. On real-world data: Temperature--Humidity correlation r=-0.662, two-component PCA (PC1 =83.1% of variance), Isolation Forest detects 2.12% multivariate anomalies while univariate Z-score detects none, calibrated DBSCAN isolates 0.17% noise across 13 clusters, and the MLP achieves $R^2=0.672$ (RMSE =10.54%) in cross-validation. This work demonstrates that per-variable coverage auditing prior to modeling is a necessary methodological step for heterogeneous multi-sensor IoT datasets.</span> <span class="abstract-toggle" data-id="2610.32438">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.32438v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.32438v1) · [:material-content-copy: BibTeX](bibtex/2610.32438.bib){ .bibtex-link }
    { .paper-links }

-   #### Covariance-Informed Ensemble Score Filtering: Embedding Localized Covariance Geometry into Score-Based Transport { #2610.32148 }

    *Siming Liang, Peter Jan van Leeuwen, Hristo G. Chipilski, Zezhong Zhang, Feng Bao, Guannan Zhang* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.32148">Background-error covariance models are central to atmospheric data assimilation because they spread partial observational information across space, variables, and dynamically coupled scales. Ensemble...</span><span class="abstract-full" id="full-2610.32148" hidden>Background-error covariance models are central to atmospheric data assimilation because they spread partial observational information across space, variables, and dynamically coupled scales. Ensemble and variational methods exploit this covariance geometry efficiently, but their standard analysis formulations are largely built around Gaussian or locally quadratic approximations. Score-based ensemble filtering provides a framework for nonlinear distributional transport, but existing Ensemble Score Filter (EnSF) formulations lack an intrinsic covariance-mediated pathway for propagating likelihood information from observed to unobserved state components. We introduce a covariance-informed ensemble score filter (CI-EnSF) that embeds localized forecast-error covariance geometry directly into the forward diffusion and reverse transport. The proposed forward diffusion is covariance-shaped in standardized state space and becomes isotropic under the corresponding control-variable transformation. This covariance structure continuously propagates observational information from observed to unobserved state components throughout the reverse transport analysis trajectory. Experiments with Lorenz-96 and surface quasi-geostrophic (SQG) dynamics show a clear regime dependence. For linear observations, a well-tuned LETKF is more accurate, while CI-EnSF maintains stable cycling and effective covariance-mediated information propagation. Under strongly nonlinear observations and under climatological initialization, CI-EnSF achieves lower analysis error and greater cycling robustness than Gaussian ensemble benchmarks such as the LETKF, with improvements extending across spatial scales in the nonlinear SQG experiment. These results suggest that the principal benefit of CI-EnSF arises from combining nonlinear distributional transport with the mature covariance geometry used in conventional data assimilation.</span> <span class="abstract-toggle" data-id="2610.32148">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.32148v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.32148v1) · [:material-content-copy: BibTeX](bibtex/2610.32148.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### Can We Trust Evaluations of AI Weather Forecasts for Extremes? How Evaluation Protocols Shape Model Rankings { #2610.31237 }

    *Tobias Biegert, Sam Allen, Zhongwei Zhang, Sebastian Lerch* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.31237">AI weather models perform competitively with traditional physics-based models at a fraction of the computational cost, and they are thus already used operationally. However, studies reach conflicting...</span><span class="abstract-full" id="full-2610.31237" hidden>AI weather models perform competitively with traditional physics-based models at a fraction of the computational cost, and they are thus already used operationally. However, studies reach conflicting conclusions about their ability to forecast extreme weather events, and comparisons across studies are complicated by differing evaluation protocols. We discuss the challenges that arise when comparing AI and physics-based weather forecasts for extreme events. Within a common experimental design, we find that model rankings are highly sensitive to the choice of evaluation metrics: the considered physics-based model tends to perform best when evaluation is restricted to observed extreme events, whereas AI models generally perform better when evaluated using weighted scoring functions. A contingency table analysis shows that AI models underpredict the frequency of extremes, so that their scores are generally dominated by misses, while the physics-based model forecasts extremes more frequently and its scores are dominated by false alarms. Different evaluation metrics penalise these errors differently. Further score decompositions confirm that the score differences arise primarily from miscalibration rather than discrimination ability. This sheds light on the contrasting results in the literature, and highlights the importance of the evaluation metrics used to evaluate AI models.</span> <span class="abstract-toggle" data-id="2610.31237">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.31237v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.31237v1) · [:material-content-copy: BibTeX](bibtex/2610.31237.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=extremes" data-tag="extremes">Extremes</a> <a class="md-tag" href="/explore/?t=evaluation" data-tag="evaluation">Evaluation</a>
    { .paper-tags }

-   #### Probabilistic Machine Learning for Short-Term Atmospheric Seeing Forecasting { #2610.29654 }

    *Mary-Joe Medlej, Rahul Srinivasan, Simon Prunet, Aziz Ziad, Christophe Giordano* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.29654">Accurate short-term forecasting of atmospheric seeing is of significant interest for astronomical observations and free-space optical communication systems, where rapid variations in optical...</span><span class="abstract-full" id="full-2610.29654" hidden>Accurate short-term forecasting of atmospheric seeing is of significant interest for astronomical observations and free-space optical communication systems, where rapid variations in optical turbulence can affect system performance and operational planning. In this work, we benchmark machine learning models for predicting atmospheric seeing up to two hours in advance, with a particular focus on probabilistic forecasting and uncertainty quantification. We compare two probabilistic approaches, Gaussian Processes (GPs) and a normalizing-flow- based model (FloTS), against deterministic recurrent neural network (RNN) and long short-term memory (LSTM) baselines. All models are trained and evaluated on seeing measurements from the Maunakea summit. We assess the models using both deterministic metrics, including the root mean square error and the Pearson correlation coefficient, and probabilistic metrics that evaluate the quality and calibration of predictive uncertainty estimates. The results show that deep learning approaches outperform standard recurrent networks for short-term forecasting, with LSTM and FloTS achieving the best overall predictive accuracy. While Gaussian Processes provide a useful probabilistic baseline, their predictive distributions are limited by the Gaussian assumption. FloTS combines competitive forecasting accuracy with flexible uncertainty quantification and produces the most informative probabilistic forecasts.</span> <span class="abstract-toggle" data-id="2610.29654">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.29654v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.29654v1) · [:material-content-copy: BibTeX](bibtex/2610.29654.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### Space weather dependence of decameter-scale ionospheric irregularities spatial distribution { #2610.29630 }

    *Oleg I. Berngardt, Ivan A. Lavygin* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.29630">This paper describes an approach to constructing nowcast and forecast predictive models for the occurrence of high-latitude decameter ionospheric irregularities based on SuperDARN and SECIRA radar...</span><span class="abstract-full" id="full-2610.29630" hidden>This paper describes an approach to constructing nowcast and forecast predictive models for the occurrence of high-latitude decameter ionospheric irregularities based on SuperDARN and SECIRA radar data and space weather parameters. The method relies on self-learning pattern recognition in the dynamics of space weather parameters, that influence the irregularities formation. The proposed approach is a data-driven, multiscale method in which the spatial distribution of irregularities is approximated using a neural network of an original architecture, inspired by graph neural networks.   The model was trained and tested using data from 11 radars during the period 2010-2025 (SuperDARN: CVW, ICE, ICW, INV, PGR, PYK, RKN, SAS, STO; SECIRA: EKB, MAGW). It is shown that the model achieves optimal performance when using geomagnetic coordinates MLT,MLAT. The optimal set of parameters for the best nowcast and forecast over studied set of parameters is a combination of geomagnetic and solar disturbance indices ($a_{p}$, $K_{p}$, $AE$, F10.7) from ground-based observations, and satellite-measured solar wind and interplanetary magnetic field parameters ($P_{sw}$, $E_{sw}$, $\widetilde{B_{z}}$, $T_{sw}$). Hourly mean values for all the parameters were used in the analysis. The results indicate that acceptable forecast quality is achieved with a lead time of up to 7-9 hours, with the highest quality reached for nowcast (zero lead time). It is shown that the parameters $\widetilde{B_{z}}$ and F10.7 are significant for nowcasting only and can be neglected for other lead times. It is shown that $AE$ and $T_{sw}$ are significant for mid-term forecast (\textgreater 2hours ahead) and can be neglected for short-term predictions.   Statistical relationships between the maximum latitude of the equatorial boundary of the irregularities and space weather parameters are also discussed.</span> <span class="abstract-toggle" data-id="2610.29630">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.29630v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.29630v1) · [:material-content-copy: BibTeX](bibtex/2610.29630.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=graph-neural-networks" data-tag="graph-neural-networks">Graph neural networks</a> <a class="md-tag" href="/explore/?t=hourly" data-tag="hourly">Hourly</a>
    { .paper-tags }

</div>

