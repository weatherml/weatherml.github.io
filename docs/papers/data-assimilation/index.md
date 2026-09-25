---
title: 'Data Assimilation'
hide:
  - toc
---

<div class="listing-header" markdown>

# Data Assimilation

<p class="page-meta" markdown="span">95 papers · page 1 of 4 · <a href="../../bib/data-assimilation.bib" download>:material-download: BibTeX for this topic</a></p>

</div>

<div class="grid cards" markdown>

-   #### Distilling deep optical flow stereo methods to retrieve dense three-dimensional wind fields { #2609.03100 }

    *Thomas J. Vandal, Dong L. Wu, James L. Carr, Derek J. Posselt, Elise Penn, Tristan Ballard et al.* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.03100">Geostationary atmospheric motion vectors (AMVs) provide the dense horizontal wind vectors (u,v) and heights ingested into data assimilation systems. Traditional AMVs track features using window-based...</span><span class="abstract-full" id="full-2609.03100" hidden>Geostationary atmospheric motion vectors (AMVs) provide the dense horizontal wind vectors (u,v) and heights ingested into data assimilation systems. Traditional AMVs track features using window-based cross-correlation and estimate heights via infrared brightness temperatures paired with numerical weather prediction (NWP) background states, creating a circular dependency that yields inaccurate heights, high computational cost, and sparse retrievals. Stereo winds from GEO-GEO and GEO-LEO geometrically resolve heights from parallax shifts across different poses, eliminating NWP dependence and improving accuracy, but they remain computationally heavy with limited coverage. In this work, we replace window-based tracking in stereo matching with deep optical flow for efficient, improved retrieval. Fine-tuning balances a self-supervised geometric residual loss with supervised radiosonde reconstruction. To eliminate multi-satellite overlap requirements, we distill the stereo teacher into a single-satellite student model. Chi-square and height uncertainties from the teacher are emulated by the student for quality assurance. The student generates winds across full-disk GEO imagery globally. Validation compares stereo and student models against radiosondes, operational AMVs, ERA5 reanalysis, and EarthCARE cloud profiles. Results through triple collocation show that stereo winds improve performance beyond operational AMVs for water vapor bands (6.2, 6.9, and 7.3 μm), wit degradation in the long-wave infrared (11.2 μm) band.</span> <span class="abstract-toggle" data-id="2609.03100">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.03100v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.03100v1) · [:material-content-copy: BibTeX](../../bibtex/2609.03100.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a>
    { .paper-tags }

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

-   #### Generative data assimilation highlights fronts as key regulators of ocean energy cascade { #2608.14955 }

    *Scott A. Martin, Georgy E. Manucharyan, Patrice Klein* · Aug 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2608.14955">Mesoscale eddies are fundamental to the ocean circulation, yet the extent to which submesoscale motions, a few kilometers across, influence mesoscale eddy energetics through a kinetic energy cascade...</span><span class="abstract-full" id="full-2608.14955" hidden>Mesoscale eddies are fundamental to the ocean circulation, yet the extent to which submesoscale motions, a few kilometers across, influence mesoscale eddy energetics through a kinetic energy cascade remains uncertain. High-resolution simulations predict that submesoscale fronts are key regulators of the cascade, transferring energy both downscale towards dissipation and upscale to sustain and shape the seasonality of mesoscale eddies. Testing these predictions has remained difficult because existing observations and state estimates cannot resolve submesoscale currents over sufficiently broad domains. Here we map the ocean's submesoscale energy cascade by combining multi-source satellite observations with a generative deep learning framework, reconstructing gap-free, kilometer-scale surface currents with physically plausible dynamics learned from simulations. Applying this to the eddy-rich Agulhas Current system, we find that submesoscales energize the mesoscale through an upscale energy cascade above 10 km, contributing to the seasonality of mesoscale eddies. Below 10 km, convergence at submesoscale fronts drives a downscale cascade towards dissipation. Both upscale and downscale pathways concentrate within fronts, where cross-scale transfer is up to an order of magnitude more efficient. Despite their limited extent, fronts account for a substantial fraction of the domain-integrated cascade, establishing them as key regulators of the cascade and targets for next-generation eddy parameterizations.</span> <span class="abstract-toggle" data-id="2608.14955">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2608.14955v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2608.14955v1) · [:material-content-copy: BibTeX](../../bibtex/2608.14955.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=km-scale" data-tag="km-scale">Km-scale</a>
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

-   #### ARCO-Mars: A Unified Cloud-Optimized Archive of Mars Atmosphere Reanalysis { #2606.21701 }

    *Ananyo Bhattacharya* · Jun 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2606.21701">Long-term records of the Martian atmosphere based on general circulation models and reanalysis of atmospheric state variables are important to understand the diurnal, seasonal, and climatological...</span><span class="abstract-full" id="full-2606.21701" hidden>Long-term records of the Martian atmosphere based on general circulation models and reanalysis of atmospheric state variables are important to understand the diurnal, seasonal, and climatological changes of the planet. Atmospheric dynamics of the Martian atmosphere are strongly influenced by the characterization of dust lifting, solar insolation, and spatial variations in topography. We present ARCO-Mars, a unified Analysis-Ready Cloud-Optimized dataset providing integrated access to three independent Mars atmospheric reanalysis products: EMARS, MACDA, and OpenMARS spanning over Mars Years 24-35. These reanalyses assimilate thermal infrared retrievals from the MGS/TES, ODY/THEMIS, and MRO/MCS instruments, providing both two and three-dimensional surface and atmospheric state variables, including temperature, winds, surface pressure, and dust optical depth. The dataset is stored in Zarr v3 format and hosted on HuggingFace, enabling efficient cloud-based access without requiring local storage of the full archive. We compare the state variables between the three reanalysis products to identify systematic differences, attributed to differences in data assimilation and general circulation models. ARCO-Mars provides a community resource for Mars atmospheric science, numerical weather prediction validation, and machine learning applications, including weather forecasting and data assimilation.</span> <span class="abstract-toggle" data-id="2606.21701">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2606.21701v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2606.21701v1) · [:material-content-copy: BibTeX](../../bibtex/2606.21701.bib){ .bibtex-link }
    { .paper-links }

-   #### Using Distributional Regression Networks to Retrieve Cloud Properties from Solar Satellite Channels for Data Assimilation { #2606.21294 }

    *Stefano Franzoni, Christopher Bülte, Leonhard Scheck, Christian Keil, George C. Craig* · Jun 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2606.21294">Satellite observations in the solar spectrum (including visible and near-infrared channels) offer high-resolution information on clouds and atmospheric properties valuable for data assimilation....</span><span class="abstract-full" id="full-2606.21294" hidden>Satellite observations in the solar spectrum (including visible and near-infrared channels) offer high-resolution information on clouds and atmospheric properties valuable for data assimilation. While forward operators for a direct assimilation of solar images have become available recently and a first visible channel is already used operationally, their assimilation remains challenging due to strong non-linearities, ambiguities and high inter-channel correlations. This study addresses two central questions: what is the potential impact of assimilating multiple solar channels jointly, and can observed reflectances be transformed into physically meaningful, uncertainty-quantified variables better suited to assimilation than the raw reflectances themselves? As a proof of concept, we assess the joint information content of six solar channels from the Flexible Combined Imager (FCI) onboard Meteosat Third Generation and introduce a novel "Backward Operator" (BO) for probabilistic retrievals of cloud-related variables. The BO is implemented in a machine learning approach as a distributional regression network that is trained on synthetic images from a NWP regional model run and produces multivariate Gaussian estimates of total optical thickness, column cloud fraction, ice fraction, and effective radii of water and ice. The BO predictions are unbiased and well-calibrated, with realistic, situation-dependent and non-trivial covariance structures. The retrieved variables can be overall usefully constrained. Despite strong inter-channel correlations, combining multiple channels yields substantial performance improvements. As the BO does not require prior information, is consistent with an existing forward operator, and yields cloud variables more linearly related to the NWP model state, assimilating these variables could be a viable alternative to direct reflectance assimilation.</span> <span class="abstract-toggle" data-id="2606.21294">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2606.21294v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2606.21294v1) · [:material-content-copy: BibTeX](../../bibtex/2606.21294.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a>
    { .paper-tags }

-   #### Towards a Foundation Model for the Martian Atmosphere { #2605.28851 }

    *Sujit Roy, Udayshankar Nair, Yuling Wu, Georgios Priftis, Liping Wang, Anastasia Georgiou et al.* · May 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2605.28851">The martian atmosphere hosts dynamical phenomena ranging from planet-encircling dust storms to mesoscale orographic clouds and nocturnal low-level jets. General circulation model show capability to...</span><span class="abstract-full" id="full-2605.28851" hidden>The martian atmosphere hosts dynamical phenomena ranging from planet-encircling dust storms to mesoscale orographic clouds and nocturnal low-level jets. General circulation model show capability to simulate these phenomena, but is computationally expensive at resolution needed to resolve mesoscale features. While assimilation of satellite remote sensing observation enable forecasting capabilities using such models, observation record is often sparse, short and fragmented across instrument generators. These constraints motivate the development of a data-driven foundation model for the Martian atmosphere.   Foundation models live in a complex design landscape. There is an interplay between the available data, the physics of the underlying processes and corresponding developments in AI. Even though the idea of a foundation model is to address multiple use cases in a data- and compute-efficient manner, it is important to have a clear picture what applications can sensibly addressed by a single model.   The purpose of this paper is to elucidate this design landscape. We discuss available data ranging from atmospheric retrievals to reanalysis datasets as well as existing physical models. Moreover, we identify a wide range of candidate downstream applications. Finally, we consider relevant recent developments in artificial intelligence (AI) that can be leveraged in this context. Here, we put a particular emphasis on AI models for atmospheric physics, data-driven approaches to data assimilation as well as methods to work in a limited data setting.</span> <span class="abstract-toggle" data-id="2605.28851">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2605.28851v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2605.28851v1) · [:material-content-copy: BibTeX](../../bibtex/2605.28851.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=foundation-models" data-tag="foundation-models">Foundation models</a>
    { .paper-tags }

-   #### Global kilometre-scale tropical cyclone inner-core vector winds from sparse scalar CYGNSS observations { #2605.18477 }

    *Xinhai Han, Xiaohui Li, Jingsong Yang, Zeyi Niu, Guoqi Han, Jiuke Wang, Wei Huang, Yunxia Zheng et al.* · May 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2605.18477">Tropical cyclone (TC) inner-core surface wind vectors underpin intensity forecasting and storm-surge prediction, yet direct observations remain scarce: routine aircraft reconnaissance is confined to...</span><span class="abstract-full" id="full-2605.18477" hidden>Tropical cyclone (TC) inner-core surface wind vectors underpin intensity forecasting and storm-surge prediction, yet direct observations remain scarce: routine aircraft reconnaissance is confined to the North Atlantic and Eastern Pacific and, even there, samples each storm only episodically. CYGNSS is the only satellite that penetrates heavy precipitation to measure inner-core surface winds, but delivers directionless scalar wind speeds and is assimilated by no operational analysis system. Here we show that the full 10 m vector wind field inside the TC inner core can be reconstructed globally at 1.5 km resolution from sparse CYGNSS scalar observations alone, by generalising score-based diffusion assimilation to a nonlinear observation operator and injecting three TC boundary-layer constraints; we further propose a CYGNSS-intrinsic Observation Coverage Sufficiency (OCS) criterion that flags reliable reconstructions without external references. Applied to 4,955 snapshots of 249 TCs across all six active basins (2020-2022), the reconstructions reduce systematic Vmax bias against IBTrACS best-track by ~79% and ~75% relative to ERA5 and CCMP. Independent Tail Doppler Radar validation (47 storms) yields a wind speed RMSE of 6.9 m/s on the 23 coverage-sufficient cases (7.5 m/s overall); ablation across the full sample shows that the physical constraints cut wind-direction RMSE by 60% without degrading speed accuracy. The framework further supports joint assimilation of heterogeneous observations: adding only 11 dropsonde vectors to CYGNSS for TC FIONA (2022) reduces the cross-eye profile RMSE by 42%, outlining a practical pathway for fusing CYGNSS with SFMR, SAR and scatterometer data. The result is a globally consistent, observation-anchored kilometre-scale description of TC inner-core vector winds across all six active basins, including those without routine aircraft reconnaissance.</span> <span class="abstract-toggle" data-id="2605.18477">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2605.18477v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2605.18477v1) · [:material-content-copy: BibTeX](../../bibtex/2605.18477.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a> <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a> <a class="md-tag" href="/explore/?t=km-scale" data-tag="km-scale">Km-scale</a>
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

-   #### Global Attention with Linear Complexity for Exascale Generative Data Assimilation in Earth System Prediction { #2604.16590 }

    *Xiao Wang, Zezhong Zhang, Isaac Lyngaas, Hong-Jun Yoon, Jong-Youl Choi, Siming Liang, Janet Wang et al.* · Apr 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2604.16590">Accurate weather and climate prediction relies on data assimilation (DA), which estimates the Earth system state by integrating observations with models. While exascale computing has significantly...</span><span class="abstract-full" id="full-2604.16590" hidden>Accurate weather and climate prediction relies on data assimilation (DA), which estimates the Earth system state by integrating observations with models. While exascale computing has significantly advanced earth simulation, scalable and accurate inference of the Earth system state remains a fundamental bottleneck, limiting uncertainty quantification and prediction of extreme events. We introduce a unified one-stage generative DA framework that reformulates assimilation as Bayesian posterior sampling, replacing the conventional forecast-update cycle with compute-dense, GPU-efficient inference. At the core is STORM, a novel spatiotemporal transformer with a global attention linear-complexity scaling algorithm that breaks the quadratic attention barrier. On 32,768 GPUs of the Frontier supercomputer, our method achieves 63% strong scaling efficiency and 1.6 ExaFLOP sustained performance. We further scale to 20 billion spatiotemporal tokens, enabling km-scale global modeling over 177k temporal frames, regimes previously unreachable, establishing a new paradigm for Earth system prediction.</span> <span class="abstract-toggle" data-id="2604.16590">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2604.16590v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2604.16590v1) · [:material-content-copy: BibTeX](../../bibtex/2604.16590.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=transformers" data-tag="transformers">Transformers</a> <a class="md-tag" href="/explore/?t=km-scale" data-tag="km-scale">Km-scale</a>
    { .paper-tags }

-   #### Calibration of a neural network ocean closure for improved mean state and variability { #2604.06398 }

    *Pavel Perezhogin, Alistair Adcroft, Laure Zanna* · Apr 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2604.06398">Global ocean models exhibit biases in the mean state and variability, particularly at coarse resolution, where mesoscale eddies are unresolved. To address these biases, parameterization coefficients...</span><span class="abstract-full" id="full-2604.06398" hidden>Global ocean models exhibit biases in the mean state and variability, particularly at coarse resolution, where mesoscale eddies are unresolved. To address these biases, parameterization coefficients are typically tuned ad hoc. Here, we formulate parameter tuning as a calibration problem using Ensemble Kalman Inversion (EKI). We optimize parameters of a neural network parameterization of mesoscale eddies in two idealized ocean models at coarse resolution. The calibrated parameterization reduces errors by factors of 1.7-3.3 in the time-averaged fluid interfaces and their variability compared to the unparameterized model, depending on the metric and configuration. The EKI method is robust to noise in time-averaged statistics arising from chaotic ocean dynamics. Furthermore, we propose an efficient calibration protocol that bypasses integration to statistical equilibrium by carefully choosing an initial condition. These results demonstrate that systematic calibration can substantially improve coarse-resolution ocean simulations and provide a practical pathway for reducing biases in global ocean models.</span> <span class="abstract-toggle" data-id="2604.06398">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2604.06398v2) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2604.06398v2) · [:material-content-copy: BibTeX](../../bibtex/2604.06398.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a>
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

</div>

<nav class="pager" markdown="span">**1** [2](2.md) [3](3.md) [4](4.md) [Older :material-arrow-right:](2.md){ .pager-step }</nav>

