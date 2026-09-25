---
title: 'Data Assimilation'
hide:
  - toc
---

<div class="listing-header" markdown>

# Data Assimilation

<p class="page-meta" markdown="span">74 papers · page 1 of 3 · <a href="../../bib/data-assimilation.bib" download>:material-download: BibTeX for this topic</a></p>

</div>

<div class="grid cards" markdown>

-   #### A score-based particle flow filter for non-Gaussian data assimilation in high-dimensional chaotic systems { #2608.22454 }

    *Zheqi Shen, Youmin Tang, Yuewei Fang* · Aug 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2608.22454">Current particle flow filters rely on Gaussian prior assumptions that fail to capture the non-Gaussian attractor structure of chaotic systems. This study proposes a Score-based Particle Flow Filter...</span><span class="abstract-full" id="full-2608.22454" hidden>Current particle flow filters rely on Gaussian prior assumptions that fail to capture the non-Gaussian attractor structure of chaotic systems. This study proposes a Score-based Particle Flow Filter (Score-PFF) that replaces the parametric prior gradient with a neural network-learned score function via denoising score matching. This enables flexible characterization of multimodal, skewed, and complex prior distributions in chaotic dynamics. Pure prior adjustment experiments demonstrate correct gradient directions toward the attractor (26.9%-29.0% error reduction over Gaussian priors). Under linear observations, Score-PFF significantly outperforms both Gaussian PFF and EAKF (Cohen's d = 1.03 and 1.40), preserving non-Gaussian structure that EAKF progressively Gaussianizes. Under nonlinear observation operators, Score-PFF maintains robust performance with up to 60% RMSE reduction in strongly non-Gaussian regimes. On the 1000-dimensional Lorenz-96 system, Score-PFF achieves 49.5% RMSE reduction over PFF while reducing per-assimilation cost by replacing SVD-based covariance inversion with neural network inference. Score-PFF establishes a computationally tractable, non-Gaussian data assimilation framework suitable for high-dimensional geophysical systems.</span> <span class="abstract-toggle" data-id="2608.22454">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2608.22454v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2608.22454v1) · [:material-content-copy: BibTeX](../../bibtex/2608.22454.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a>
    { .paper-tags }

-   #### Advanced Linear Algebra with Applications - Part I (Numerical linear algebra for PDEs, machine learning, and data assimilation) { #2608.21234 }

    *Victorita Dolean, Jemima Tabeart* · Aug 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2608.21234">These lecture notes form the first part of a master's-level course on advanced numerical linear algebra. Their aim is not only to present the classical algorithms, but to show why the subject has...</span><span class="abstract-full" id="full-2608.21234" hidden>These lecture notes form the first part of a master's-level course on advanced numerical linear algebra. Their aim is not only to present the classical algorithms, but to show why the subject has become considerably more central than it was a generation ago. Numerical linear algebra grew up alongside the numerical solution of partial differential equations, and for a long time that is where its large sparse systems came from. Ranking the nodes of a network, assimilating observations into a weather forecast, and fitting a model to a large noisy data set now lead to problems of the same kind: too large to factorise, structured, and accessible only through matrix-vector products. Strikingly few ideas are needed for all of them. Each chapter therefore develops a standard topic and then puts it to work outside its original setting. We treat norms, factorisations, conditioning and floating-point arithmetic; sparse matrices arising from finite differences, from graphs and from machine learning; stationary iterations and the smoothing property; the conjugate gradient and Lanczos methods, with spectral clustering and regularisation by early stopping; Arnoldi and GMRES, with PageRank and large least squares; and finally preconditioning, Schwarz domain decomposition and multigrid. We assume a first course in linear algebra. Every section closes with a summary of what should be retained and every chapter with exercises, several drawn from past examinations. Accompanying Python code reproduces the numerical illustrations.</span> <span class="abstract-toggle" data-id="2608.21234">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2608.21234v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2608.21234v1) · [:fontawesome-brands-github: Code](https://github.com/vicdolean/scicomp_examples) · [:material-content-copy: BibTeX](../../bibtex/2608.21234.bib){ .bibtex-link }
    { .paper-links }

-   #### Coupled multiscale paleoclimate reconstruction with four-dimensional variational data assimilation { #2608.19469 }

    *Zilu Meng, Gregory J. Hakim, Julien Emile-Geay, Tanaya Gondhalekar, Eric J. Steig* · Aug 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2608.19469">Paleoclimate archives extend climate knowledge beyond the instrumental era, registering different seasons, variables, time averages, and memory lengths. A longstanding problem is to integrate these...</span><span class="abstract-full" id="full-2608.19469" hidden>Paleoclimate archives extend climate knowledge beyond the instrumental era, registering different seasons, variables, time averages, and memory lengths. A longstanding problem is to integrate these heterogeneous sources of information within a unified methodology. Here we present a new data-assimilation framework, Last Millennium Reanalysis 4D-Var (LMR4D-Var), which reconstructs climate trajectories from these heterogeneous datasets while balancing errors in the model, observations, and initial conditions. We compare results using LMR4D-Var to assimilate proxies from PAGES2k, Temp12k, and borehole temperature profiles without treating them as instantaneous equivalents. Instrumental verification shows that LMR4D-Var achieves the highest skill compared with previous reconstructions. Borehole assimilation preserves skill against withheld annually resolved records, increases agreement between reconstructed 300--2000-m ocean heat content and independent estimates, and yields a cooler reconstructed Little Ice Age ocean. Results for Temp12k demonstrate assimilation of decadal-to-millennial records and the potential for Holocene and deeper-time applications with suitable emulators.</span> <span class="abstract-toggle" data-id="2608.19469">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2608.19469v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2608.19469v1) · [:material-content-copy: BibTeX](../../bibtex/2608.19469.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=benchmarks-datasets" data-tag="benchmarks-datasets">Benchmarks & datasets</a>
    { .paper-tags }

-   #### Benchmarking ConvLSTM for One-Day-Ahead IMDAA Rainfall-Field Prediction across Four Indian Cities { #2607.26581 }

    *Tanmay Ghosh, Shaurabh Anand, Rakesh Gomaji Nannewar, Nithin Nagaraj* · Jul 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2607.26581">Convolutional long short-term memory networks (ConvLSTMs) are widely used for precipitation forecasting, but most evidence for their performance comes from dense, high-frequency radar sequences. This...</span><span class="abstract-full" id="full-2607.26581" hidden>Convolutional long short-term memory networks (ConvLSTMs) are widely used for precipitation forecasting, but most evidence for their performance comes from dense, high-frequency radar sequences. This study tests whether convolutional recurrence improves one-day-ahead rainfall-field prediction on small daily reanalysis grids. Indian Monsoon Data Assimilation and Analysis (IMDAA) fields for June-September 1998-2020 were analysed for Bengaluru, Delhi, Kolkata and Mumbai. Ten naive, statistical, tree-based and neural approaches were compared using atmospheric-only and rainfall-history-plus-atmospheric inputs. Performance was assessed for complete fields, domain-mean rainfall, spatial anomalies and high-rainfall days.   ConvLSTM did not consistently outperform simpler alternatives. FC-LSTM produced the numerically lowest domain-mean rainfall error in Bengaluru, Kolkata and Mumbai, whereas persistence performed best in Delhi. ConvLSTM produced the numerically lowest spatial-anomaly error only in Mumbai, where rainfall fields showed greater short-term spatial continuity and rainfall-history inputs improved all three neural architectures. The difference between ConvLSTM and FC-LSTM was nevertheless small. Neural models underestimated rainfall magnitude and predicted too few threshold exceedances on high-rainfall days, while persistence achieved the highest detection performance in every city. Post-hoc analyses showed that the selected models were most sensitive to the latest input day, with broader recent-lag sensitivity in Mumbai. These findings show that gridded inputs alone do not justify ConvLSTM and that architecture choice should follow strong benchmarking across average, spatial and high-rainfall performance.</span> <span class="abstract-toggle" data-id="2607.26581">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2607.26581v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2607.26581v1) · [:material-content-copy: BibTeX](../../bibtex/2607.26581.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=recurrent-networks" data-tag="recurrent-networks">Recurrent networks</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=daily" data-tag="daily">Daily</a>
    { .paper-tags }

-   #### Sparse Sensor Placement for Reducing Forecast Errors in Ensemble Kalman Filtering { #2606.27267 }

    *Takumi Saito, Shunji Kotsuki* · Jun 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2606.27267">Designing efficient observation networks for reducing forecast errors is a fundamental challenge in numerical weather prediction. Data-driven sparse sensor placement (SSP) and ensemble-based data...</span><span class="abstract-full" id="full-2606.27267" hidden>Designing efficient observation networks for reducing forecast errors is a fundamental challenge in numerical weather prediction. Data-driven sparse sensor placement (SSP) and ensemble-based data assimilation via the Ensemble Kalman Filter (EnKF) have each addressed this challenge independently, yet their mathematical connections have not been systematically formalized. This study presents a unified theoretical framework integrating SSP and EnKF through optimal experimental design, providing new theoretical and algorithmic results. While conventional SSP methods aim to reduce analysis errors, this study extends the SSP to target forecast error reduction by using a tangent linear model approximated by an ensemble forecast. We derive the Fisher information matrices in the ensemble and model spaces for the EnKF, and clarify the mathematical interpretations of A-, D-, and E-optimality in terms of forecast error reduction. A-optimality in the model space minimizes the mean forecast error variance; D-optimality is ill-defined in the model space due to rank deficiency and is therefore formulated in the ensemble space, where it maximizes the Shannon information content of assimilated observations; and E-optimality in the model space minimizes the worst-case forecast error variance. We further propose a fast greedy algorithm for selecting observation locations under A-optimality in the model space, avoiding matrix inversion at each greedy step and substantially reducing computational cost. Numerical experiments using the Lorenz-96 model support these theoretical findings. Among the three optimality criteria, A-optimality in the model space most consistently reduces forecast spread and root-mean-square error, and yields stable incremental improvements consistent with post-assimilation observation impact diagnostics.</span> <span class="abstract-toggle" data-id="2606.27267">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2606.27267v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2606.27267v1) · [:material-content-copy: BibTeX](../../bibtex/2606.27267.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### Uncertainty quantification via conformal prediction in data assimilation { #2606.27001 }

    *Catherine George, Alireza Javanmardi, Tijana Janjić, Eyke Hüllermeier* · Jun 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2606.27001">Quantifying the evolution of uncertainty is critical to both probabilistic forecasting and data assimilation in numerical weather prediction. In this study, we investigate the applicability of...</span><span class="abstract-full" id="full-2606.27001" hidden>Quantifying the evolution of uncertainty is critical to both probabilistic forecasting and data assimilation in numerical weather prediction. In this study, we investigate the applicability of conformal prediction (CP), a recent machine learning (ML) method, to quantify uncertainty in a controlled, idealized setting. We use the one dimensional modified shallow water model, designed to mimic the convective process. CP provides a set of possible outcomes with a chosen confidence level. Here, we compare and evaluate the average empirical coverage, the average interval length, miss low, miss high and average interval score loss (AISL) for three variants of CP, namely a) Standard CP, b) Normalized CP and c) Conformalized Quantile Regression. We further compare these CP-based uncertainty estimates with traditional ensemble-based measures such as standard deviation intervals and ensemble spread. In addition, we investigate the integration of CP-derived uncertainty within the data assimilation cycle through CP perturbations. Our results highlight the strengths and limitations of each approach, providing insight into the effectiveness of CP to complement common ensemble-based uncertainty quantification in simplified atmospheric models.</span> <span class="abstract-toggle" data-id="2606.27001">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2606.27001v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2606.27001v1) · [:material-content-copy: BibTeX](../../bibtex/2606.27001.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### Using Distributional Regression Networks to Retrieve Cloud Properties from Solar Satellite Channels for Data Assimilation { #2606.21294 }

    *Stefano Franzoni, Christopher Bülte, Leonhard Scheck, Christian Keil, George C. Craig* · Jun 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2606.21294">Satellite observations in the solar spectrum (including visible and near-infrared channels) offer high-resolution information on clouds and atmospheric properties valuable for data assimilation....</span><span class="abstract-full" id="full-2606.21294" hidden>Satellite observations in the solar spectrum (including visible and near-infrared channels) offer high-resolution information on clouds and atmospheric properties valuable for data assimilation. While forward operators for a direct assimilation of solar images have become available recently and a first visible channel is already used operationally, their assimilation remains challenging due to strong non-linearities, ambiguities and high inter-channel correlations. This study addresses two central questions: what is the potential impact of assimilating multiple solar channels jointly, and can observed reflectances be transformed into physically meaningful, uncertainty-quantified variables better suited to assimilation than the raw reflectances themselves? As a proof of concept, we assess the joint information content of six solar channels from the Flexible Combined Imager (FCI) onboard Meteosat Third Generation and introduce a novel "Backward Operator" (BO) for probabilistic retrievals of cloud-related variables. The BO is implemented in a machine learning approach as a distributional regression network that is trained on synthetic images from a NWP regional model run and produces multivariate Gaussian estimates of total optical thickness, column cloud fraction, ice fraction, and effective radii of water and ice. The BO predictions are unbiased and well-calibrated, with realistic, situation-dependent and non-trivial covariance structures. The retrieved variables can be overall usefully constrained. Despite strong inter-channel correlations, combining multiple channels yields substantial performance improvements. As the BO does not require prior information, is consistent with an existing forward operator, and yields cloud variables more linearly related to the NWP model state, assimilating these variables could be a viable alternative to direct reflectance assimilation.</span> <span class="abstract-toggle" data-id="2606.21294">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2606.21294v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2606.21294v1) · [:material-content-copy: BibTeX](../../bibtex/2606.21294.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a>
    { .paper-tags }

-   #### Global kilometre-scale tropical cyclone inner-core vector winds from sparse scalar CYGNSS observations { #2605.18477 }

    *Xinhai Han, Xiaohui Li, Jingsong Yang, Zeyi Niu, Guoqi Han, Jiuke Wang, Wei Huang, Yunxia Zheng et al.* · May 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2605.18477">Tropical cyclone (TC) inner-core surface wind vectors underpin intensity forecasting and storm-surge prediction, yet direct observations remain scarce: routine aircraft reconnaissance is confined to...</span><span class="abstract-full" id="full-2605.18477" hidden>Tropical cyclone (TC) inner-core surface wind vectors underpin intensity forecasting and storm-surge prediction, yet direct observations remain scarce: routine aircraft reconnaissance is confined to the North Atlantic and Eastern Pacific and, even there, samples each storm only episodically. CYGNSS is the only satellite that penetrates heavy precipitation to measure inner-core surface winds, but delivers directionless scalar wind speeds and is assimilated by no operational analysis system. Here we show that the full 10 m vector wind field inside the TC inner core can be reconstructed globally at 1.5 km resolution from sparse CYGNSS scalar observations alone, by generalising score-based diffusion assimilation to a nonlinear observation operator and injecting three TC boundary-layer constraints; we further propose a CYGNSS-intrinsic Observation Coverage Sufficiency (OCS) criterion that flags reliable reconstructions without external references. Applied to 4,955 snapshots of 249 TCs across all six active basins (2020-2022), the reconstructions reduce systematic Vmax bias against IBTrACS best-track by ~79% and ~75% relative to ERA5 and CCMP. Independent Tail Doppler Radar validation (47 storms) yields a wind speed RMSE of 6.9 m/s on the 23 coverage-sufficient cases (7.5 m/s overall); ablation across the full sample shows that the physical constraints cut wind-direction RMSE by 60% without degrading speed accuracy. The framework further supports joint assimilation of heterogeneous observations: adding only 11 dropsonde vectors to CYGNSS for TC FIONA (2022) reduces the cross-eye profile RMSE by 42%, outlining a practical pathway for fusing CYGNSS with SFMR, SAR and scatterometer data. The result is a globally consistent, observation-anchored kilometre-scale description of TC inner-core vector winds across all six active basins, including those without routine aircraft reconnaissance.</span> <span class="abstract-toggle" data-id="2605.18477">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2605.18477v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2605.18477v1) · [:material-content-copy: BibTeX](../../bibtex/2605.18477.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a> <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a> <a class="md-tag" href="/explore/?t=km-scale" data-tag="km-scale">Km-scale</a>
    { .paper-tags }

-   #### ForcingDAS: Unified and Robust Data Assimilation via Diffusion Forcing { #2605.14285 }

    *Yixuan Jia, Siyi Chen, Yida Pan, Xiao Li, Lianghe Shi, Chanyong Jung, Haijie Yuan, Ismail Alkhouri et al.* · May 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2605.14285">Data assimilation (DA) estimates the state of an evolving dynamical system from noisy, partial observations, and is widely used in scientific simulation as well as weather and climate science. In...</span><span class="abstract-full" id="full-2605.14285" hidden>Data assimilation (DA) estimates the state of an evolving dynamical system from noisy, partial observations, and is widely used in scientific simulation as well as weather and climate science. In practice, filtering methods rely on frame-to-frame transition models. However, these models are fragile when observations are non-Markovian (when they form only a partial slice of a higher-dimensional latent state as in real-world weather data): they tend to accumulate errors over long horizons. At the same time, learned DA methods typically commit to a single regime, either filtering (nowcasting, real-time forecasting) or smoothing (retrospective reanalysis), which splits what should be a shared prior across application-specific pipelines. To address both issues, we introduce ForcingDAS, a unified and robust DA framework. Built on Diffusion Forcing with an independent noise level assigned to each frame, ForcingDAS learns a joint-trajectory prior instead of frame-to-frame transitions. This allows it to capture long-horizon temporal dependencies and reduce error accumulation. In addition, the same trained model spans the full filtering to smoothing spectrum at inference time. Specifically, nowcasting, fixed-lag smoothing, and batch reanalysis are selected through the inference schedule alone, without retraining. We evaluate ForcingDAS on 2D Navier-Stokes vorticity, precipitation nowcasting, and global atmospheric state estimation. Across all settings, a single model is competitive with or outperforms both learned and classical baselines that are specialized for individual regimes, with the largest gains observed on real-world weather benchmarks.</span> <span class="abstract-toggle" data-id="2605.14285">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2605.14285v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2605.14285v1) · [:material-content-copy: BibTeX](../../bibtex/2605.14285.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a>
    { .paper-tags }

-   #### Acceleration of horizontal numerical advection for atmospheric modeling through surrogate modeling with temporal coarse-graining { #2605.10956 }

    *Manho Park, Christopher V. Rackauckas, Christopher W. Tessum* · May 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2605.10956">Machine-learned surrogate modeling of advection may accelerate geoscientific models, but existing approaches have either achieved limited speedup or have sacrificed spatial resolution compared to the...</span><span class="abstract-full" id="full-2605.10956" hidden>Machine-learned surrogate modeling of advection may accelerate geoscientific models, but existing approaches have either achieved limited speedup or have sacrificed spatial resolution compared to the model they are trained to emulate. We developed a machine-learned solver that speeds up advection simulations without sacrificing spatial resolution through the use of temporal coarse-graining, where the model is trained to take larger integration steps than dictated by the Courant-Friedrich-Lewy (CFL) condition. Our solver framework includes a convolutional neural network that takes concentrations and CFL numbers as inputs and outputs mass flux. Our solvers emulate 10-day ground-level horizontal advection simulations with r$^2$ values against the baseline ranging from 0.60--0.98 with temporal coarsening factors of 4 to 32 times the baseline integration time step. Speed increases and accuracy decreases with increased coarsening, with $r^2 = 0.24$ in accuracy lost for every factor of 10 gained in speed, reaching a maximum 92$\times$ speedup while maintaining $r^2 = 0.60$. We deliberately trained our solvers only on January ground-level wind data to examine their ability to generalize across seasons and vertical heights. The 4$\times$-coarsened learned solver successfully reproduces simulations over 72 vertical levels. The 8$\times$--16$\times$ solvers (but not 32$\times$) emulate most vertical levels. The learned solvers also generalize well across seasons, except for instabilities in June and October. With additional fine-tuning, these learned solvers could be appropriate for operational use where trading accuracy for speed could be advantageous, such as in screening tools, in ensemble simulations, or with data assimilation.</span> <span class="abstract-toggle" data-id="2605.10956">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2605.10956v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2605.10956v1) · [:material-content-copy: BibTeX](../../bibtex/2605.10956.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a>
    { .paper-tags }

-   #### Earth-o1: A Grid-free Observation-native Atmospheric World Model { #2605.06337 }

    *Junchao Gong, Kaiyi Xu, Wangxu Wei, Siwei Tu, Jingyi Xu, Zili Liu, Hang Fan, Zhiwang Zhou, Tao Han et al.* · May 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2605.06337">Despite the unprecedented volume of multimodal data provided by modern Earth observation systems, our ability to model atmospheric dynamics remains constrained. Traditional modeling frameworks force...</span><span class="abstract-full" id="full-2605.06337" hidden>Despite the unprecedented volume of multimodal data provided by modern Earth observation systems, our ability to model atmospheric dynamics remains constrained. Traditional modeling frameworks force heterogeneous measurements into predefined spatial grids, inherently limiting the full exploitation of raw sensor data and creating severe computational bottlenecks. Here we present Earth-o1, an observation-native atmospheric world model that overcomes these structural limitations. Rather than relying on conventional atmospheric dynamical modeling systems or traditional data assimilation, Earth-o1 directly learns the continuous, three-dimensional physical evolution of the Earth system from ungridded observational data. By integrating diverse sensor inputs into a unified, grid-free dynamical field, the model autonomously advances the atmospheric state in space and time. We show that this fundamentally distinct paradigm enables direct, real-time forecasting and cross-sensor inference without the overhead of explicit numerical solvers. In hindcast evaluations, Earth-o1 achieves surface forecast skill comparable to the operational Integrated Forecasting System (IFS). These results establish that continuous, observation-driven world models -- a new class of fully observation-native geophysical simulators -- can match the fidelity of established physical frameworks, providing a scalable data-driven foundation for a digital twin of the Earth.</span> <span class="abstract-toggle" data-id="2605.06337">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2605.06337v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2605.06337v1) · [:material-content-copy: BibTeX](../../bibtex/2605.06337.bib){ .bibtex-link }
    { .paper-links }

-   #### The Physical Limit of Neural Hypoxia Detection in the Black Sea from Satellite Observations { #2604.25608 }

    *Victor Mangeleer, Luc Vandenbulcke, Marilaure Grégoire, Gilles Louppe* · Apr 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2604.25608">Coastal hypoxia (O_2 < 63 [mmol / m^3]) threatens ocean health worldwide. On continental shelves, summer stratification prevents bottom oxygen consumed by respiration from being renewed, making...</span><span class="abstract-full" id="full-2604.25608" hidden>Coastal hypoxia (O_2 < 63 [mmol / m^3]) threatens ocean health worldwide. On continental shelves, summer stratification prevents bottom oxygen consumed by respiration from being renewed, making monitoring essential to protect vulnerable ecosystems and reduce biodiversity loss. Although satellite observations are increasingly available, their potential to infer subsurface oxygen remains largely unexplored. This can be framed as a Bayesian inverse problem relating surface observations to the complete Black Sea states. Here, we solve it using a deep generative neural network trained on numerical model outputs, providing a tractable and computationally efficient approximation of the true posterior distribution of sea states. We find that accurate state estimation is limited to the mixed layer, because its homogeneity makes surface conditions representative of subsurface states. During summer, we detect 38% of all hypoxic events shelf-wide with a precision of 47%. Improving results will likely require longer assimilation windows or sub-surface observations.</span> <span class="abstract-toggle" data-id="2604.25608">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2604.25608v2) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2604.25608v2) · [:material-content-copy: BibTeX](../../bibtex/2604.25608.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a>
    { .paper-tags }

-   #### Uncertainty-Aware Spatiotemporal Super-Resolution Data Assimilation with Diffusion Models { #2604.21180 }

    *Aditya Sai Pranith Ayapilla, Kazuya Miyashita, Yuki Yasuda, Ryo Onishi* · Apr 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2604.21180">Data assimilation (DA) improves prediction of chaotic systems by combining model forecasts with sparse, noisy observations. Many DA methods are inherently probabilistic, but accurate probabilistic DA...</span><span class="abstract-full" id="full-2604.21180" hidden>Data assimilation (DA) improves prediction of chaotic systems by combining model forecasts with sparse, noisy observations. Many DA methods are inherently probabilistic, but accurate probabilistic DA is often computationally expensive because it requires repeated high-resolution (HR) forecasts and large ensembles. In this study, we develop DiffSRDA, a probabilistic spatiotemporal super-resolution data assimilation framework based on denoising diffusion models, and evaluate it on an idealized barotropic ocean jet instability testbed. DiffSRDA is trained offline to generate short HR analysis windows conditioned on (i) a time series of low-resolution (LR) forecast frames and (ii) sparse HR observations. Repeated reverse diffusion sampling then produces an ensemble of HR analyses, providing both point estimates and uncertainty information. Despite relying only on low-cost LR forecasts, DiffSRDA achieves reconstruction quality close to that of an Ensemble Kalman Filter (EnKF) driven by HR forecasts, while improving over deterministic CNN-based SRDA baselines. The sampled ensemble also yields physically meaningful uncertainty patterns, with spread concentrated in dynamically active regions similarly to EnKF. A key practical result is that accurate base DiffSRDA cycling does not require long reverse chains: most of the full-chain accuracy is retained with only a few reverse steps, making diffusion-based SRDA practical for repeated cycling. Finally, by exploiting the score-based structure of diffusion sampling, we demonstrate training-free observation-consistency guidance for deployment-time sensor-layout shifts, enabling improved use of changed observation configurations without retraining. Overall, diffusion models provide a practical, uncertainty-aware, and computationally efficient approach for spatiotemporal SRDA in chaotic fluid flows.</span> <span class="abstract-toggle" data-id="2604.21180">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2604.21180v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2604.21180v1) · [:material-content-copy: BibTeX](../../bibtex/2604.21180.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=efficiency" data-tag="efficiency">Efficiency</a>
    { .paper-tags }

-   #### Deep-Learned Observation Operators for Artificial Intelligence Weather Forecasting Models { #2604.00082 }

    *Kelsey Lieberman, Laura Slivinski, Matt Bender, Chris Miller, Josh DaRosa, Nick Krall et al.* · Apr 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2604.00082">Satellite observation operators play an essential role in atmospheric data assimilation by translating model state variables into observation space. Previous work has shown that deep-learned...</span><span class="abstract-full" id="full-2604.00082" hidden>Satellite observation operators play an essential role in atmospheric data assimilation by translating model state variables into observation space. Previous work has shown that deep-learned emulators can effectively predict the outputs of classic observation operators, like the Community Radiative Transfer Model (CRTM), with reduced inference time. This study expands previous work to show the potential for integrating observation operators into artificial intelligence (AI) weather forecasting models. Specifically, this study shows that (1) deep-learned models can effectively predict the innovations (or differences between the simulated and observed radiances) used by data assimilation models and (2) deep-learned observation models suffer only minor degradations in performance when the model state is represented with fewer vertical levels, as is commonly used by AI forecasting models. Experiments were performed using the Unified Forecast System (UFS) replay dataset, including Gridpoint Statistical Interpolation (GSI) observational data for the Advanced Technology Microwave Sounder (ATMS) sensor from 2022 and 2023. Code is available at https://github.com/mitre/deep-obs.</span> <span class="abstract-toggle" data-id="2604.00082">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2604.00082v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2604.00082v1) · [:fontawesome-brands-github: Code](https://github.com/mitre/deep-obs) · [:material-content-copy: BibTeX](../../bibtex/2604.00082.bib){ .bibtex-link }
    { .paper-links }

-   #### Self-Organizing Score-based Data Assimilation { #2603.28048 }

    *Yuma Yamaoka, Seiichi Uchida, Shoji Toyota* · Mar 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2603.28048">A state-space model is a statistical framework for inferring latent states from observed time-series data. However, inference with nonlinear and high-dimensional state-space models remains...</span><span class="abstract-full" id="full-2603.28048" hidden>A state-space model is a statistical framework for inferring latent states from observed time-series data. However, inference with nonlinear and high-dimensional state-space models remains challenging. To this end, an approach based on diffusion models-a powerful class of deep generative models-has been developed, known as Score-based Data Assimilation (SDA). However, SDA cannot be directly applied when the latent-state transition depends on unknown parameters that must be inferred jointly with the latent states. To overcome this limitation, we propose a framework that enables SDA to handle latent states with unknown parameters. A key feature of the proposed method is the incorporation of the self-organization technique, which has been used in classical state-space modeling for the joint estimation of latent states and parameters. By integrating this classical technique into modern SDA, our method enables joint inference of latent states and unknown parameters while maintaining the high training efficiency of SDA. The effectiveness of the proposed approach is validated through numerical experiments on dynamical systems arising in neuroscience and atmospheric science. In addition, its scalability is demonstrated using a high-dimensional Kolmogorov flow, with the data dimension on the order of several hundred thousand.</span> <span class="abstract-toggle" data-id="2603.28048">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2603.28048v2) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2603.28048v2) · [:material-content-copy: BibTeX](../../bibtex/2603.28048.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a>
    { .paper-tags }

-   #### Learning Data-driven Surrogate and Correction Models for Satellite Observations in Numerical Weather Prediction { #2603.22037 }

    *Gian Luca Buono, Stefanie Hollborn, Roland Potthast, Jörg Schäfer, Martin Simon* · Mar 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2603.22037">Satellite observations play a critical role in numerical weather prediction where they are assimilated through an observation operator that maps model states to radiances. In the traditional Ensemble...</span><span class="abstract-full" id="full-2603.22037" hidden>Satellite observations play a critical role in numerical weather prediction where they are assimilated through an observation operator that maps model states to radiances. In the traditional Ensemble Kalman Filter, these observations are used to update the state by weighting their associated errors against model uncertainties to produce an optimal estimate. This process requires radiative transfer simulations for passive, downward-viewing satellite radiometers operating in the visible, infrared, and microwave spectra. Typically, such simulations rely on numerically integrating physical laws via models like RTTOV. In this paper, we introduce two machine learning surrogate observation operators inspired by modern computer-vision architectures: First, a fully data-driven emulator of radiative transfer, and second, a hybrid incremental correction model that learns only the residual relative to RTTOV, thereby retaining established physics while enabling data-driven refinement in complex conditions such as cloud-affected situations. The residual formulation improves radiance accuracy (lower Root Mean Squared Error (RMSE) than the fully data-driven emulator and RTTOV) and adds only moderate computational costs to the assimilation step. Both models combine 3D convolutions for vertical profile encoding with a 2D U-Net operating on latitude-longitude grids, allowing joint learning of vertical structure, spatial correlations, and inter-channel dependencies. We further provide a theoretical justification for deploying the hybrid surrogate as an observation operator in data assimilation.</span> <span class="abstract-toggle" data-id="2603.22037">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2603.22037v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2603.22037v1) · [:material-content-copy: BibTeX](../../bibtex/2603.22037.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a>
    { .paper-tags }

-   #### Convergence Analysis of a Fully Discrete Observer For Data Assimilation of the Barotropic Euler Equations { #2603.10962 }

    *Aidan Chaumet, Jan Giesselmann* · Mar 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2603.10962">We study the convergence of a discrete Luenberger observer for the barotropic Euler equations in one dimension, for measurements of the velocity only. We use a mixed finite element method in space...</span><span class="abstract-full" id="full-2603.10962" hidden>We study the convergence of a discrete Luenberger observer for the barotropic Euler equations in one dimension, for measurements of the velocity only. We use a mixed finite element method in space and implicit Euler integration in time. We use a modified relative energy technique to show an error bound comparing the discrete observer to the original system's solution. The bound is the sum of three parts: an exponentially decaying part, proportional to the difference in initial value, a part proportional to the grid sizes in space and time and a part that is proportional to the size of the measurement errors as well as the nudging parameter. The proportionality constants of the second and third parts are independent of time and grid sizes. To the best of our knowledge, this provides the first error estimate for a discrete observer for a quasilinear hyperbolic system, and implies uniform-in-time accuracy of the discrete observer for long-time simulations.</span> <span class="abstract-toggle" data-id="2603.10962">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2603.10962v2) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2603.10962v2) · [:material-content-copy: BibTeX](../../bibtex/2603.10962.bib){ .bibtex-link }
    { .paper-links }

-   #### Accurate and Efficient Hybrid-Ensemble Atmospheric Data Assimilation in Latent Space with Uncertainty Quantification { #2603.04395 }

    *Hang Fan, Juan Nathaniel, Yi Xiao, Ce Bian, Fenghua Ling, Ben Fei, Lei Bai, Pierre Gentine* · Mar 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2603.04395">Data assimilation (DA) combines model forecasts and observations to estimate the optimal state of the atmosphere with its uncertainty, providing initial conditions for weather prediction and...</span><span class="abstract-full" id="full-2603.04395" hidden>Data assimilation (DA) combines model forecasts and observations to estimate the optimal state of the atmosphere with its uncertainty, providing initial conditions for weather prediction and reanalyses for climate research. Yet, existing traditional and machine-learning DA methods struggle to achieve accuracy, efficiency and uncertainty quantification simultaneously. Here, we propose HLOBA (Hybrid-Ensemble Latent Observation-Background Assimilation), a three-dimensional hybrid-ensemble DA method that operates in an atmospheric latent space learned via an autoencoder (AE). HLOBA maps both model forecasts and observations into a shared latent space via the AE encoder and an end-to-end Observation-to-Latent-space mapping network (O2Lnet), respectively, and fuses them through a Bayesian update with weights inferred from time-lagged ensemble forecasts. Both idealized and real-observation experiments demonstrate that HLOBA matches dynamically constrained four-dimensional DA methods in both analysis and forecast skill, while achieving end-to-end inference-level efficiency and theoretical flexibility applies to any forecasting model. Moreover, by exploiting the error decorrelation property of latent variables, HLOBA enables element-wise uncertainty estimates for its latent analysis and propagates them to model space via the decoder. Idealized experiments show that this uncertainty highlights large-error regions and captures their seasonal variability.</span> <span class="abstract-toggle" data-id="2603.04395">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2603.04395v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2603.04395v1) · [:material-content-copy: BibTeX](../../bibtex/2603.04395.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### Efficient Real-Time Adaptation of ROMs for Unsteady Flows Using Data Assimilation { #2602.23188 }

    *Ismaël Zighed, Andrea Nóvoa, Luca Magri, Taraneh Sayadi* · Feb 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2602.23188">We propose an efficient retraining strategy for a parameterized Reduced Order Model (ROM) that attains accuracy comparable to full retraining while requiring only a fraction of the computational time...</span><span class="abstract-full" id="full-2602.23188" hidden>We propose an efficient retraining strategy for a parameterized Reduced Order Model (ROM) that attains accuracy comparable to full retraining while requiring only a fraction of the computational time and relying solely on sparse observations of the full system. The architecture employs an encode-process-decode structure: a Variational Autoencoder (VAE) to perform dimensionality reduction, and a transformer network to evolve the latent states and model the dynamics. The ROM is parameterized by an external control variable, the Reynolds number in the Navier-Stokes setting, with the transformer exploiting attention mechanisms to capture both temporal dependencies and parameter effects. The probabilistic VAE enables stochastic sampling of trajectory ensembles, providing predictive means and uncertainty quantification through the first two moments. After initial training on a limited set of dynamical regimes, the model is adapted to out-of-sample parameter regions using only sparse data. Its probabilistic formulation naturally supports ensemble generation, which we employ within an ensemble Kalman filtering framework to assimilate data and reconstruct full-state trajectories from minimal observations. We further show that, for the dynamical system considered, the dominant source of error in out-of-sample forecasts stems from distortions of the latent manifold rather than changes in the latent dynamics. Consequently, retraining can be limited to the autoencoder, allowing for a lightweight, computationally efficient, real-time adaptation procedure with very sparse fine-tuning data.</span> <span class="abstract-toggle" data-id="2602.23188">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2602.23188v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2602.23188v1) · [:material-content-copy: BibTeX](../../bibtex/2602.23188.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=transformers" data-tag="transformers">Transformers</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=efficiency" data-tag="efficiency">Efficiency</a>
    { .paper-tags }

-   #### LEVDA: Latent Ensemble Variational Data Assimilation via Differentiable Dynamics { #2602.19406 }

    *Phillip Si, Peng Chen* · Feb 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2602.19406">Long-range geophysical forecasts are fundamentally limited by chaotic dynamics and numerical errors. While data assimilation can mitigate these issues, classical variational smoothers require...</span><span class="abstract-full" id="full-2602.19406" hidden>Long-range geophysical forecasts are fundamentally limited by chaotic dynamics and numerical errors. While data assimilation can mitigate these issues, classical variational smoothers require computationally expensive tangent-linear and adjoint models. Conversely, recent efficient latent filtering methods often enforce weak trajectory-level constraints and assume fixed observation grids. To bridge this gap, we propose Latent Ensemble Variational Data Assimilation (LEVDA), an ensemble-space variational smoother that operates in the low-dimensional latent space of a pretrained differentiable neural dynamics surrogate. By performing four-dimensional ensemble-variational (4DEnVar) optimization within an ensemble subspace, LEVDA jointly assimilates states and unknown parameters without the need for adjoint code or auxiliary observation-to-latent encoders. Leveraging the fully differentiable, continuous-in-time-and-space nature of the surrogate, LEVDA naturally accommodates highly irregular sampling at arbitrary spatiotemporal locations. Across three challenging geophysical benchmarks, LEVDA matches or outperforms state-of-the-art latent filtering baselines under severe observational sparsity while providing more reliable uncertainty quantification. Simultaneously, it achieves substantially improved assimilation accuracy and computational efficiency compared to full-state 4DEnVar.</span> <span class="abstract-toggle" data-id="2602.19406">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2602.19406v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2602.19406v1) · [:material-content-copy: BibTeX](../../bibtex/2602.19406.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### Preconditioned Adjoint Data Assimilation for Two-Dimensional Decaying Isotropic Turbulence { #2602.14016 }

    *Hongyi Ke, Zejian You, Qi Wang* · Feb 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2602.14016">Adjoint-based data assimilation for turbulent Navier-Stokes flows is fundamentally limited by the behavior of the adjoint dynamics: in backward time, adjoint fields exhibit exponential growth and...</span><span class="abstract-full" id="full-2602.14016" hidden>Adjoint-based data assimilation for turbulent Navier-Stokes flows is fundamentally limited by the behavior of the adjoint dynamics: in backward time, adjoint fields exhibit exponential growth and become increasingly dominated by small-scale structures, severely degrading reconstruction of the initial condition from sparse measurements. We demonstrate that the relative weighting of spectral components in the adjoint formulation can be systematically controlled by redefining the inner product under which the adjoint operator is defined. The inverse problem is formulated as a constrained minimization in which a cost functional measures the mismatch between model predictions and observations, and the adjoint equations provide the gradient with respect to the initial velocity field. Redefining the forward-adjoint duality through a Fourier-space weighting kernel preconditions the optimization and is mathematically equivalent to changing the representation of the control variable or, alternatively, introducing a smoothing operation on the governing dynamics. Specific kernel choices correspond to fractional integration or diffusion operators applied to the initial condition. Among these, exponential kernels provide effective regularization by suppressing high-wavenumber contributions while preserving large-scale coherence, leading to improved reconstruction across scales. A statistical analysis of an ensemble of adjoint fields from different turbulent realizations reveals scale-dependent backward growth rates, explaining the instability of the standard formulation and clarifying the mechanism by which the proposed preconditioning attenuates incoherent small-scale amplification.</span> <span class="abstract-toggle" data-id="2602.14016">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2602.14016v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2602.14016v1) · [:material-content-copy: BibTeX](../../bibtex/2602.14016.bib){ .bibtex-link }
    { .paper-links }

-   #### FlowDA: Accurate, Low-Latency Weather Data Assimilation via Flow Matching { #2602.06800 }

    *Ran Cheng, Lailai Zhu* · Feb 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2602.06800">Data assimilation (DA) is a fundamental component of modern weather prediction, yet it remains a major computational bottleneck in machine learning (ML)-based forecasting pipelines due to reliance on...</span><span class="abstract-full" id="full-2602.06800" hidden>Data assimilation (DA) is a fundamental component of modern weather prediction, yet it remains a major computational bottleneck in machine learning (ML)-based forecasting pipelines due to reliance on traditional variational methods. Recent generative ML-based DA methods offer a promising alternative but typically require many sampling steps and suffer from error accumulation under long-horizon auto-regressive rollouts with cycling assimilation. We propose FlowDA, a low-latency weather-scale generative DA framework based on flow matching. FlowDA conditions on observations through a SetConv-based embedding and fine-tunes the Aurora foundation model to deliver accurate, efficient, and robust analyses. Experiments across observation rates decreasing from $3.9\%$ to $0.1\%$ demonstrate superior performance of FlowDA over strong baselines with similar tunable-parameter size. FlowDA further shows robustness to observational noise and stable performance in long-horizon auto-regressive cycling DA. Overall, FlowDA points to an efficient and scalable direction for data-driven DA.</span> <span class="abstract-toggle" data-id="2602.06800">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2602.06800v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2602.06800v1) · [:material-content-copy: BibTeX](../../bibtex/2602.06800.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=foundation-models" data-tag="foundation-models">Foundation models</a>
    { .paper-tags }

-   #### On a system of equations arising in meteorology: Well-posedness and data assimilation { #2602.02328 }

    *Eduard Feireisl, Piotr Gwiazda, Agnieszka Świerczewska-Gwiazda* · Feb 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2602.02328">Data assimilation plays a crucial role in modern weather prediction, providing a systematic way to incorporate observational data into complex dynamical models. The paper addresses continuous data...</span><span class="abstract-full" id="full-2602.02328" hidden>Data assimilation plays a crucial role in modern weather prediction, providing a systematic way to incorporate observational data into complex dynamical models. The paper addresses continuous data assimilation for a model arising as a singular limit of the three-dimensional compressible Navier-Stokes-Fourier system with rotation driven by temperature gradient. The limit system preserves the essential physical mechanisms of the original model, while exhibiting a reduced, effectively two-and-a-half-dimensional structure. This simplified framework allows for a rigorous analytical study of the data assimilation process while maintaining a direct physical connection to the full compressible model. We establish well posedness of global-in-time solutions and a compact trajectory attractor, followed by the stability and convergence results for the nudging scheme applied to the limiting system. Finally, we demonstrate how these results can be combined with a relative entropy argument to extend the assimilation framework to the full three-dimensional compressible setting, thereby establishing a rigorous connection between the reduced and physically complete models.</span> <span class="abstract-toggle" data-id="2602.02328">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2602.02328v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2602.02328v1) · [:material-content-copy: BibTeX](../../bibtex/2602.02328.bib){ .bibtex-link }
    { .paper-links }

-   #### SENDAI: A Hierarchical Sparse-measurement, EfficieNt Data AssImilation Framework { #2601.21664 }

    *Xingyue Zhang, Yuxuan Bao, Mars Liyao Gao, J. Nathan Kutz* · Jan 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2601.21664">Bridging the gap between data-rich training regimes and observation-sparse deployment conditions remains a central challenge in spatiotemporal field reconstruction, particularly when target domains...</span><span class="abstract-full" id="full-2601.21664" hidden>Bridging the gap between data-rich training regimes and observation-sparse deployment conditions remains a central challenge in spatiotemporal field reconstruction, particularly when target domains exhibit distributional shifts, heterogeneous structure, and multi-scale dynamics absent from available training data. We present SENDAI, a hierarchical Sparse-measurement, EfficieNt Data AssImilation Framework that reconstructs full spatial states from hyper sparse sensor observations by combining simulation-derived priors with learned discrepancy corrections. We demonstrate the performance on satellite remote sensing, reconstructing MODIS (Moderate Resolution Imaging Spectroradiometer) derived vegetation index fields across six globally distributed sites. Using seasonal periods as a proxy for domain shift, the framework consistently outperforms established baselines that require substantially denser observations -- SENDAI achieves a maximum SSIM improvement of 185% over traditional baselines and a 36% improvement over recent high-frequency-based methods. These gains are particularly pronounced for landscapes with sharp boundaries and sub-seasonal dynamics; more importantly, the framework effectively preserves diagnostically relevant structures -- such as field topologies, land cover discontinuities, and spatial gradients. By yielding corrections that are more structurally and spectrally separable, the reconstructed fields are better suited for downstream inference of indirectly observed variables. The results therefore highlight a lightweight and operationally viable framework for sparse-measurement reconstruction that is applicable to physically grounded inference, resource-limited deployment, and real-time monitor and control.</span> <span class="abstract-toggle" data-id="2601.21664">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2601.21664v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2601.21664v1) · [:material-content-copy: BibTeX](../../bibtex/2601.21664.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a>
    { .paper-tags }

-   #### Cheap2Rich: A Multi-Fidelity Framework for Data Assimilation and System Identification of Multiscale Physics -- Rotating Detonation Engines { #2601.20295 }

    *Yuxuan Bao, Jan Zajac, Megan Powers, Venkat Raman, J. Nathan Kutz* · Jan 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2601.20295">Bridging the sim2real gap between computationally inexpensive models and complex physical systems remains a central challenge in machine learning applications to engineering problems, particularly in...</span><span class="abstract-full" id="full-2601.20295" hidden>Bridging the sim2real gap between computationally inexpensive models and complex physical systems remains a central challenge in machine learning applications to engineering problems, particularly in multi-scale settings where reduced-order models typically capture only dominant dynamics. In this work, we present Cheap2Rich, a multi-scale data assimilation framework that reconstructs high-fidelity state spaces from sparse sensor histories by combining a fast low-fidelity prior with learned, interpretable discrepancy corrections. We demonstrate the performance on rotating detonation engines (RDEs), a challenging class of systems that couple detonation-front propagation with injector-driven unsteadiness, mixing, and stiff chemistry across disparate scales. Our approach successfully reconstructs high-fidelity RDE states from sparse measurements while isolating physically meaningful discrepancy dynamics associated with injector-driven effects. The results highlight a general multi-fidelity framework for data assimilation and system identification in complex multi-scale systems, enabling rapid design exploration and real-time monitoring and control while providing interpretable discrepancy dynamics. Code for this project is is available at: github.com/kro0l1k/Cheap2Rich.</span> <span class="abstract-toggle" data-id="2601.20295">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2601.20295v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2601.20295v1) · [:material-content-copy: BibTeX](../../bibtex/2601.20295.bib){ .bibtex-link }
    { .paper-links }

-   #### GenDA: Generative Data Assimilation on Complex Urban Areas via Classifier-Free Diffusion Guidance { #2601.11440 }

    *Francisco Giral, Álvaro Manzano, Ignacio Gómez, Ricardo Vinuesa, Soledad Le Clainche* · Jan 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2601.11440">Urban wind flow reconstruction is essential for assessing air quality, heat dispersion, and pedestrian comfort, yet remains challenging when only sparse sensor data are available. We propose GenDA, a...</span><span class="abstract-full" id="full-2601.11440" hidden>Urban wind flow reconstruction is essential for assessing air quality, heat dispersion, and pedestrian comfort, yet remains challenging when only sparse sensor data are available. We propose GenDA, a generative data assimilation framework that reconstructs high-resolution wind fields on unstructured meshes from limited observations. The model employs a multiscale graph-based diffusion architecture trained on computational fluid dynamics (CFD) simulations and interprets classifier-free guidance as a learned posterior reconstruction mechanism: the unconditional branch learns a geometry-aware flow prior, while the sensor-conditioned branch injects observational constraints during sampling. This formulation enables obstacle-aware reconstruction and generalization to held-out mesh geometries, wind directions, and sensor configurations within the studied urban-flow setting, without retraining. We consider both sparse fixed sensors and trajectory-based observations using the same reconstruction procedure. When evaluated against supervised graph neural network (GNN) baselines and classical reduced-order data assimilation methods, GenDA reduces the relative root-mean-square error (RRMSE) by 25-57% and increases the structural similarity index (SSIM) by 23-33% across the tested meshes. Experiments are conducted on Reynolds-averaged Navier-Stokes (RANS) simulations of a real urban neighborhood in Bristol, United Kingdom, at a characteristic Reynolds number of $\mathrm{Re}\approx2\times10^{7}$, featuring complex building geometry and irregular terrain. The proposed framework provides a scalable path toward generative, geometry-aware data assimilation for environmental monitoring in complex domains.</span> <span class="abstract-toggle" data-id="2601.11440">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2601.11440v3) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2601.11440v3) · [:material-content-copy: BibTeX](../../bibtex/2601.11440.bib){ .bibtex-link }
    { .paper-links }

-   #### The Ensemble Schr{ö}dinger Bridge filter for Nonlinear Data Assimilation { #2512.18928 }

    *Feng Bao, Hui Sun* · Dec 2025
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2512.18928">This work puts forward a novel nonlinear optimal filter namely the Ensemble Schr{ö}dinger Bridge nonlinear filter. The proposed filter finds marriage of the standard prediction procedure and the...</span><span class="abstract-full" id="full-2512.18928" hidden>This work puts forward a novel nonlinear optimal filter namely the Ensemble Schr{ö}dinger Bridge nonlinear filter. The proposed filter finds marriage of the standard prediction procedure and the diffusion generative modeling for the analysis procedure to realize one filtering step. The designed approach finds no structural model error, and it is derivative free, training free and highly parallizable. Experimental results show that the designed algorithm performs well given highly nonlinear dynamics in (mildly) high dimension up to 40 or above under a chaotic environment. It also shows better performance than classical methods such as the ensemble Kalman filter and the Particle filter in numerous tests given different level of nonlinearity. Future work will focus on extending the proposed approach to practical meteorological applications and establishing a rigorous convergence analysis.</span> <span class="abstract-toggle" data-id="2512.18928">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2512.18928v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2512.18928v1) · [:material-content-copy: BibTeX](../../bibtex/2512.18928.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### A Neural-Network Model-Measurement-Based Observation Operator For Weather Radar Reflectivity Assimilation { #2512.18289 }

    *Marco Stefanelli, Žiga Zaplotnik, Gregor Skok* · Dec 2025
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2512.18289">In three-dimensional variational data assimilation (3DVar) for numerical weather prediction (NWP), the observation operator $\mathcal{H}$ plays a central role by mapping model state variables to an...</span><span class="abstract-full" id="full-2512.18289" hidden>In three-dimensional variational data assimilation (3DVar) for numerical weather prediction (NWP), the observation operator $\mathcal{H}$ plays a central role by mapping model state variables to an observation equivalent. For weather radar, however, specifying $\mathcal{H}$ is particularly challenging: reflectivity is a nonlinear, microphysics-dependent diagnostic quantity that only indirectly relates to the model's prognostic variables, making traditional parameterised radar operators complex, regime-dependent and difficult to tune. In this study, we propose a neural-network (NN)-based observation operator for radar reflectivity and apply it within a 3DVar framework. Using five years (2019-2023) of radar reflectivity data from the Lisca radar and 4.4 km-resolution short-range forecasts from ALADIN model over Slovenia, we train a convolutional encoder-decoder neural network to map model temperature, humidity, horizontal wind components and surface pressure fields to radar reflectivity. Across independent test cases spanning clear-sky, stratiform, and convective regimes, the NN-based operator accurately reproduces the spatial structure and intensity of observed reflectivity, relying primarily on the model state near the observation point. In the extreme precipitation case, which caused widespread floods in Slovenia on August 4, 2023, assimilating the full radar disc reduces the domain-averaged reflectivity root-mean-square error from 5.99 dBZ to 3.47 dBZ and improves the alignment between the analysed and observed convective bands. Embedded within 3DVar, the Jacobian of the NN observation operator allows radar reflectivity observations to inform model state variables, producing corresponding analysis increments. The proposed NN radar observation operator offers a flexible alternative to traditional parameterised radar operators for improving convective-storm forecasts.</span> <span class="abstract-toggle" data-id="2512.18289">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2512.18289v2) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2512.18289v2) · [:material-content-copy: BibTeX](../../bibtex/2512.18289.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=extremes" data-tag="extremes">Extremes</a> <a class="md-tag" href="/explore/?t=km-scale" data-tag="km-scale">Km-scale</a>
    { .paper-tags }

-   #### Continuous data assimilation for 2D stochastic Navier-Stokes equations { #2512.15184 }

    *Hakima Bessaih, Benedetta Ferrario, Oussama Landoulsi, Margherita Zanella* · Dec 2025
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2512.15184">Continuous data assimilation methods, such as the nudging algorithm introduced by Azouani, Olson, and Titi (AOT) [2], are known to be highly effective in deterministic settings for asymptotically...</span><span class="abstract-full" id="full-2512.15184" hidden>Continuous data assimilation methods, such as the nudging algorithm introduced by Azouani, Olson, and Titi (AOT) [2], are known to be highly effective in deterministic settings for asymptotically synchronizing approximate solutions with observed dynamics. In this work, we extend this framework to a stochastic regime by considering the two-dimensional incompressible Navier-Stokes equations subject to either additive or multiplicative noise. We establish sufficient conditions on the nudging parameter and the spatial observation scale that guarantee convergence of the nudged solution to the true stochastic flow.   In the case of multiplicative noise, convergence holds in expectation, with exponential or polynomial rates depending on the growth of the noise covariance. For additive noise, we obtain the exponential convergence both in expectation and pathwise. These results yield a stochastic generalization of the AOT theory, demonstrating how the interplay between random forcing, viscous dissipation and feedback control governs synchronization in stochastic fluid systems.</span> <span class="abstract-toggle" data-id="2512.15184">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2512.15184v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2512.15184v1) · [:material-content-copy: BibTeX](../../bibtex/2512.15184.bib){ .bibtex-link }
    { .paper-links }

-   #### Balancing Accuracy and Speed: A Multi-Fidelity Ensemble Kalman Filter with a Machine Learning Surrogate Model { #2512.12276 }

    *Jeffrey van der Voort, Martin Verlaan, Hanne Kekkonen* · Dec 2025
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2512.12276">Currently, more and more machine learning (ML) surrogates are being developed for computationally expensive physical models. In this work we investigate the use of a Multi-Fidelity Ensemble Kalman...</span><span class="abstract-full" id="full-2512.12276" hidden>Currently, more and more machine learning (ML) surrogates are being developed for computationally expensive physical models. In this work we investigate the use of a Multi-Fidelity Ensemble Kalman Filter (MF-EnKF) in which the low-fidelity model is such a machine learning surrogate model, instead of a traditional low-resolution or reduced-order model. The idea behind this is to use an ensemble of a few expensive full model runs, together with an ensemble of many cheap but less accurate ML model runs. In this way we hope to reach increased accuracy within the same computational budget. We investigate the performance by testing the approach on two common test problems, namely the Lorenz-2005 model and the Quasi-Geostrophic model. By keeping the original physical model in place, we obtain a higher accuracy than when we completely replace it by the ML model. Furthermore, the MF-EnKF reaches improved accuracy within the same computational budget. The ML surrogate has similar or improved accuracy compared to the low-resolution one, but it can provide a larger speed-up. Our method contributes to increasing the effective ensemble size in the EnKF, which improves the estimation of the initial condition and hence accuracy of the predictions in fields such as meteorology and oceanography.</span> <span class="abstract-toggle" data-id="2512.12276">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2512.12276v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2512.12276v1) · [:material-content-copy: BibTeX](../../bibtex/2512.12276.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

</div>

<nav class="pager" markdown="span">**1** [2](2.md) [3](3.md) [Older :material-arrow-right:](2.md){ .pager-step }</nav>

