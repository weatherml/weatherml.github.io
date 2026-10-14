---
hide:
  - navigation
  - toc
title: weatherml
---

A collection of papers on AI for weather forecasting, climate modelling and atmospheric science.

<p class="page-meta" markdown="span">1549 papers · updated 2026-10-14 · <a href="feed.xml">:material-rss: RSS</a> · <a href="all_papers.bib" download>:material-download: BibTeX</a> · <a href="https://github.com/weatherml/weatherml.github.io/issues/new?template=suggest-paper.yml">:material-plus: Suggest a paper</a></p>

## Browse by Topic

<div class="grid cards topics" markdown>

-   [Global Models](papers/global-models/index.md) <span class="topic-count">369</span>
-   [Regional Models](papers/regional-models/index.md) <span class="topic-count">65</span>
-   [Nowcasting](papers/nowcasting/index.md) <span class="topic-count">102</span>
-   [Downscaling](papers/downscaling/index.md) <span class="topic-count">102</span>
-   [Post-processing](papers/post-processing/index.md) <span class="topic-count">26</span>
-   [Data Assimilation](papers/data-assimilation/index.md) <span class="topic-count">79</span>
-   [Climate Modeling](papers/climate-modeling/index.md) <span class="topic-count">300</span>
-   [Hydrology](papers/hydrology/index.md) <span class="topic-count">38</span>
-   [Ocean & Sea Ice](papers/ocean-sea-ice/index.md) <span class="topic-count">89</span>
-   [Air Quality & Composition](papers/air-quality-composition/index.md) <span class="topic-count">71</span>
-   [Remote Sensing](papers/remote-sensing/index.md) <span class="topic-count">100</span>
-   [Other](papers/other/index.md) <span class="topic-count">208</span>

</div>

## Recent Additions

<div class="grid cards" markdown data-search-exclude>

-   #### Establishing baseline model performances for optical turbulence forecasting { #2610.18702 }

    *M. De Sepibus, E. Masciadri, C. Weinberger, A. Otarola, S. Miesske, C. Veillet, S. Ragland* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.18702">Accurate forecasting of optical turbulence (OT) is useful to optimise telescope scheduling and maximising observing efficiency, as the performance of adaptive optics systems strongly depends on...</span><span class="abstract-full" id="full-2610.18702" hidden>Accurate forecasting of optical turbulence (OT) is useful to optimise telescope scheduling and maximising observing efficiency, as the performance of adaptive optics systems strongly depends on turbulence conditions. While various numerical weather prediction and machine learning models are developed to predict these complex turbulence properties, properly evaluating their added value requires well-defined baseline reference models. This study establishes baseline performance thresholds for key astroclimatic parameters using two distinct models: one forecasting the parameter average over a specific timescale, and the other forecasting the temporal evolution over the same specific timescale. We apply these frameworks to extensive, multi-year datasets from the sites of the Very Large Telescope (VLT) and the Large Binocular Telescope (LBT) and discuss the possibility of extrapolating reference values of forecast performances for other sites based on their optical turbulence statistical characteristics. Our results show that baseline predictive accuracy varies according to the observing site, the specific parameter, the chosen timescale, and the forecasting approach. Therefore, an OT forecasting model's accuracy cannot be quantified in absolute terms, but relatively to the baseline.</span> <span class="abstract-toggle" data-id="2610.18702">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.18702v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.18702v1) · [:material-content-copy: BibTeX](bibtex/2610.18702.bib){ .bibtex-link }
    { .paper-links }

-   #### Extreme events and statistics in an AI emulator of a global storm-resolving model { #2610.17693 }

    *Anna Kwa, W. Andre Perkins, Troy Arcomano, Noah D. Brenowitz, Christopher S. Bretherton et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.17693">We present HiRO-ACEv2 (High Resolution Output Ai2 Climate Emulator), a two-stage framework for AI emulation of the 3~km resolution X-SHiELD global storm-resolving model (GSRM), and highlight its...</span><span class="abstract-full" id="full-2610.17693" hidden>We present HiRO-ACEv2 (High Resolution Output Ai2 Climate Emulator), a two-stage framework for AI emulation of the 3~km resolution X-SHiELD global storm-resolving model (GSRM), and highlight its ability to generate large ensembles of data for studying the characteristics of tropical cyclones and extreme events in high-resolution simulations at an inference computational cost over 300 times lower than that of traditional numerical GSRMs. HiRO-ACEv2 expands upon the precipitation-only HiRO-ACEv1 model to generate coherent precipitation, 10~m winds, and sea level pressure outputs, enabling a broader range of events to be simulated at high-resolution. Fine-resolution predictions are generated by running an autoregressive emulator of coarsened X-SHiELD at 100 km, whose multivariate outputs are regionally downscaled to 3~km resolution via a separately trained diffusion model. Particular importance is placed on capturing the statistics of extreme events during the diffusion downscaling step. Our ability to match the tails of distributions and generate realistic-looking extreme events in both the ‘perfect prediction’ case as well as in free-running emulator rollouts is improved via the use of multiple denoisers at inference time and several updates to the data sampling, noise sampling, and loss weighting during training. HiRO-ACEv2 closely matches the high-resolution reference distributions of precipitation and near-surface winds out to their 99.9999th percentile values, and broadly tracks the reference distribution of tropical cyclone intensities.</span> <span class="abstract-toggle" data-id="2610.17693">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.17693v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.17693v1) · [:material-content-copy: BibTeX](bibtex/2610.17693.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a> <a class="md-tag" href="/explore/?t=extremes" data-tag="extremes">Extremes</a> <a class="md-tag" href="/explore/?t=km-scale" data-tag="km-scale">Km-scale</a>
    { .paper-tags }

-   #### Accounting for model error in ensemble variational data assimilation through learned generative dynamics { #2610.17333 }

    *Marc Bocquet, Tobias S. Finn, Sibo Cheng* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.17333">Many approaches have recently been proposed to combine data assimilation with machine learning for improved forecasts with geophysical models. Here, we focus on combining advanced sequential data...</span><span class="abstract-full" id="full-2610.17333" hidden>Many approaches have recently been proposed to combine data assimilation with machine learning for improved forecasts with geophysical models. Here, we focus on combining advanced sequential data assimilation schemes with a generative forecasting model which accounts for complex model errors. This generative model provides a stochastic approximation of the conditional distribution of the successor state given the current state. The proposed variational formulation is a novel one-step weak-constraint 4D-Var scheme in which a learned conditional stochastic forecast replaces the prescribed additive Gaussian model error representation. We derive and implement this formulation as an ensemble variational algorithm, the generative iterative ensemble Kalman filter (IEnKF-G), which generalises the iterative ensemble Kalman filter with additive Gaussian model error (IEnKF-Q). We extend the method to a smoother with an assimilation window spanning multiple time steps and propose a simplified variant.   The IEnKF-G is evaluated on a stochastic variant of the Lorenz 1963 model to assess its robustness. It is then tested on a multi-scale variant of the one-dimensional Lorenz 1996 model, in which the resolved variables are affected by model error originating from the hidden small-scale variables. The algorithm is then compared to an ensemble Kalman filter that adaptively mitigates model and sampling errors, and to the IEnKF-Q. The IEnKF-G is generally more accurate in the tested configurations, with substantial gains under partial observations and significant model error. This demonstrates the potential of generative data-driven models when paired with classical data assimilation methods, particularly when model error is substantial.</span> <span class="abstract-toggle" data-id="2610.17333">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.17333v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.17333v1) · [:material-content-copy: BibTeX](bibtex/2610.17333.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### Statistical Downscaling of Daily Precipitation from ERA5 Atmospheric Fields to IMERG over West Africa { #2610.16816 }

    *David Sagna, Redouane Lguensat, Samo Diatta* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.16816">Global reanalyses are too coarse to resolve the convective systems that produce most West African monsoon rainfall. In this work, we use U-Net based architectures to downscale daily ERA5 fields...</span><span class="abstract-full" id="full-2610.16816" hidden>Global reanalyses are too coarse to resolve the convective systems that produce most West African monsoon rainfall. In this work, we use U-Net based architectures to downscale daily ERA5 fields (0.25°) to IMERG V07 satellite precipitation (0.1°) over West Africa for the June to October seasons of 2000 to 2024, using geopotential, temperature, humidity and wind at three pressure levels together with elevation and land fraction. Adding ERA5 precipitation as input slightly improves performance for this particular challenging region. We find that all the deterministic networks investigated for this work are too dry and lose almost all observed variance below a few hundred kilometers, which motivated our use of a GAN-based alternative. We also investigated permutation importance for the U-Net that ranks 500 hPa specific humidity first, consistent with the sensitivity of deep convection to mid-tropospheric moisture.</span> <span class="abstract-toggle" data-id="2610.16816">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.16816v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.16816v1) · [:material-content-copy: BibTeX](bibtex/2610.16816.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=gans" data-tag="gans">GANs</a> <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=daily" data-tag="daily">Daily</a>
    { .paper-tags }

-   #### Forecasting carbon-risk hotspots ahead of the extreme 2026 El Niño { #2610.16714 }

    *Xiaomeng Du, Runda Jiang, Philippe Ciais, Pierre Friedlingstein, Stephen Sitch, Mike O'Sullivan et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.16714">A potentially super El Niño is forecasted in 2026, raising the need to anticipate land carbon-cycle risks before its impacts unfold. We develop a prospective early-warning framework that uses dynamic...</span><span class="abstract-full" id="full-2610.16714" hidden>A potentially super El Niño is forecasted in 2026, raising the need to anticipate land carbon-cycle risks before its impacts unfold. We develop a prospective early-warning framework that uses dynamic global vegetation models driven by an ensemble of ECMWF seasonal climate forecasts and machine-learning burned area predictions. Forecasts indicate a $0.23 \pm 0.21$ PgC reduction in global net biome productivity in the second half of 2026, with hotspots of sink weakening in eastern South America and elevated fire risk in Southeast Asia.</span> <span class="abstract-toggle" data-id="2610.16714">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.16714v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.16714v1) · [:material-content-copy: BibTeX](bibtex/2610.16714.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=subseasonal-to-seasonal" data-tag="subseasonal-to-seasonal">Subseasonal to seasonal</a>
    { .paper-tags }

-   #### Identifying Arctic states vulnerable to rapid ice loss with a convolutional neural network { #2610.16584 }

    *Matthew Patterson, Ioana Colfescu, Nathan Kirk, Graeme MacGhilchrist* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.16584">This work investigates the incidence of rapid ice-loss events in the Arctic, in which a substantial fraction of September sea-ice is lost in only a few years. Such events are seen to occur in climate...</span><span class="abstract-full" id="full-2610.16584" hidden>This work investigates the incidence of rapid ice-loss events in the Arctic, in which a substantial fraction of September sea-ice is lost in only a few years. Such events are seen to occur in climate model simulations forced with increasing greenhouse gas concentrations, but not as yet in the relatively short observational record. We train a convolutional neural network (CNN) using six large climate model ensembles to diagnose vulnerability to rapid ice-loss events. The trained model is shown to separate higher and lower risk states in out-of-sample climate models via a leave-one-out cross-validation. The model emphasises the role of persistently thin sea-ice, remaining sea-ice extent and sustained high upper ocean heat content as predictors of rapid ice-loss. The CNN is applied to observational data over the satelite-era showing a signal for rapid ice-loss prior to a drop around 2007 and an increasing risk from around 2015 to present. Such rapid-ice loss events would likely have dramatic consequences for Arctic ecosystems, coastal communities and global shipping.</span> <span class="abstract-toggle" data-id="2610.16584">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.16584v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.16584v1) · [:material-content-copy: BibTeX](bibtex/2610.16584.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a>
    { .paper-tags }

-   #### Multivariate phase-space interpretation of ensemble forecasts: A comparison of physical and AI models { #2610.16422 }

    *Martin Bonte, Stéphane Vannitsem* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.16422">Ensemble forecasts from the ECMWF's Integrated Forecasting System (IFS) and the Artificial Intelligence Forecasting System (AIFS-ENS) are compared using a multi-dimensional evaluation framework....</span><span class="abstract-full" id="full-2610.16422" hidden>Ensemble forecasts from the ECMWF's Integrated Forecasting System (IFS) and the Artificial Intelligence Forecasting System (AIFS-ENS) are compared using a multi-dimensional evaluation framework. Within this framework, the uncertainty is interpreted as the linear subspace spanned by the ensemble members in phase space, from which various geometric properties are extracted. As revealed by the analysis, the error does not entirely lie in this subspace, but is reasonably well calibrated in all directions. Moreover, the framework allows to quantify that this subspace critically depends on the number of ensemble members, suggesting that the current setups only partially represent the uncertainty subspace. This phase-space interpretation furthermore allows a direct comparison of the error-representation patterns of AIFS-ENS and IFS by computing the principal angles between the subspaces: IFS and AIFS ensemble subspaces do not entirely overlap in phase space, even though they were shown to have similar abilities to represent the amplitude and the structure of the error.</span> <span class="abstract-toggle" data-id="2610.16422">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.16422v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.16422v1) · [:material-content-copy: BibTeX](bibtex/2610.16422.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### Sensitivity Fidelity Reveals Physical Dependencies Beyond Forecast Accuracy in Deep-Learning Earth-System Models { #2610.15540 }

    *Qi Chen, Peng Zhan, Hongxing Cui, Wenqing Cui, Ibrahim Hoteit* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.15540">Deep-learning models increasingly rival numerical simulation in Earth-system forecasting, yet forecast error alone cannot determine whether their learned input-output dependencies are physically...</span><span class="abstract-full" id="full-2610.15540" hidden>Deep-learning models increasingly rival numerical simulation in Earth-system forecasting, yet forecast error alone cannot determine whether their learned input-output dependencies are physically consistent. Here we introduce sensitivity fidelity as an additional criterion: alongside prediction error, model responses to input perturbations should be examined against a trusted reference or established physical knowledge. In an idealized Rayleigh-Bénard convection benchmark, four surrogate architectures achieved low forecast errors yet encoded markedly different sensitivities. Quantitative comparison with a matched adjoint distinguished the models far more sharply than forecast accuracy and showed that predictive skill and sensitivity fidelity need not coincide, while finite-perturbation experiments further linked these differences to the propagation of initial-condition uncertainty. We then applied this strategy to El Niño-Southern Oscillation prediction, where no adjoint reference exists and learned sensitivities showed event-wise robustness, validity under finite perturbations, and consistency with established precursors. These results identify gradient-based evaluation as a discriminating complement to forecast-error metrics, providing a practical test of whether data-driven Earth-system models learn physically credible dependencies.</span> <span class="abstract-toggle" data-id="2610.15540">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.15540v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.15540v1) · [:material-content-copy: BibTeX](bibtex/2610.15540.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=subseasonal-to-seasonal" data-tag="subseasonal-to-seasonal">Subseasonal to seasonal</a>
    { .paper-tags }

-   #### WeatherBridge: Forecast-Aware Multi-Source Input Adaptation for Pretrained Weather Forecasting Models { #2610.14863 }

    *Mutian Xi, Xinyu Su, Xiaohui Zhong, Hao Li* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.14863">Data-driven weather forecasting models are commonly trained on ERA5, yet operational deployment must use timely initial states from physics-based numerical weather prediction (NWP) systems whose...</span><span class="abstract-full" id="full-2610.14863" hidden>Data-driven weather forecasting models are commonly trained on ERA5, yet operational deployment must use timely initial states from physics-based numerical weather prediction (NWP) systems whose distributions differ substantially from ERA5. Direct initialization with these states can degrade autoregressive forecasts, motivating input adaptation that keeps the pretrained forecasting model unchanged. Here, we introduce WeatherBridge, a unified, source-conditioned adapter that maps heterogeneous operational NWP initial states to ERA5 reference states without updating the pretrained forecasting model. WeatherBridge predicts the structured residual corrections with a Swin-UNet and injects source embeddings across multiple spatial scales, allowing the model to capture shared atmospheric structure and source-specific discrepancies. Its training objective combines latitude-weighted ERA5 reconstruction with One-step Rollout Alignment (ORA), which backpropagates one-step forecast error through a frozen FuXi model to optimize the adapted states for both initial state fidelity and downstream forecast compatibility. We evaluate WeatherBridge using daily GFS, HRES, and CMA initializations from 1 January to 22 November 2025 across six variables and lead times up to 240 hours. Relative to raw source initialization of FuXi, WeatherBridge reduces the mean RMSE of FuXi over the 6- 240-hour rollout for 17 of 18 source-variable combinations, with a median reduction of 10.2% and a maximum reduction of 46.3%. The adapter trained with frozen FuXi also improves FengWu forecasts on GFS inputs without retraining either the adapter or FengWu. These results demonstrate that forecast-aware input adaptation can improve forecasts from heterogeneous operational NWP inputs and transfer across pretrained ERA5-based forecasting models.</span> <span class="abstract-toggle" data-id="2610.14863">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.14863v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.14863v1) · [:material-content-copy: BibTeX](bibtex/2610.14863.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=transformers" data-tag="transformers">Transformers</a> <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a> <a class="md-tag" href="/explore/?t=daily" data-tag="daily">Daily</a>
    { .paper-tags }

-   #### Learning native eddy-resolving ocean dynamics outperforms coarse emulation and downscaling { #2610.14645 }

    *Zilu Liu, Tao Zhang, Hailong Liu, Kai Xu, Qinghui Lin, Maoxue Yu, Jiangfeng Yu, Jingwei Xie et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.14645">Eddy-resolving ocean simulations are essential for representing mesoscale dynamics that regulate ocean circulation, energy transport, and climate variability. However, they are costly to generate,...</span><span class="abstract-full" id="full-2610.14645" hidden>Eddy-resolving ocean simulations are essential for representing mesoscale dynamics that regulate ocean circulation, energy transport, and climate variability. However, they are costly to generate, store, and share. Artificial intelligence (AI) may help, but it remains unclear whether high-resolution ocean fields can be faithfully represented by evolving ocean states at coarse resolution followed by spatial downscaling, or whether dynamical evolution needs to be learned directly from native high-resolution simulations. We show that directly learning dynamical evolution on a 0.1$^\circ$ grid provides a more faithful representation of mesoscale variability than evolving ocean states on a 1$^\circ$ grid followed by downscaling reconstruction. We developed a full-depth 1$^\circ$ emulator and a native 0.1$^\circ$ sea surface temperature (SST) emulator, both of which remained stable over long-term free-running simulations without significant drift, providing reliable baselines for this comparison. Direct learning at 0.1$^\circ$ preserves mesoscale energy, spectral characteristics, and frontal gradients, whereas the coarse-resolution evolution followed by downscaling approach recovers large-scale patterns but fails to reproduce key mesoscale signatures. We show that the learned mesoscale statistics remain consistent across diverse hardware platforms, supporting the reproducibility and portability of AI-based ocean simulations. These results suggest that downscaling of low-resolution large-scale states alone is insufficient to represent ocean mesoscale variability, supporting the need to learn dynamical evolution directly in a high-resolution state space. These findings highlight the continued importance of high-resolution physics ocean models as a foundation for AI-based ocean simulation, providing dynamically resolved training data for physically faithful high-resolution emulators.</span> <span class="abstract-toggle" data-id="2610.14645">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.14645v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.14645v1) · [:material-content-copy: BibTeX](bibtex/2610.14645.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=coarse" data-tag="coarse">Coarse (≥1°)</a>
    { .paper-tags }

-   #### A Gated Catchment Hydrology Model (GCHM) for Physically-Interpretable Rainfall-Evapotranspiration-Runoff Simulation Across Diverse Hydroclimatic Regimes { #2610.14493 }

    *M Jawad, HV Gupta, MA Farmani, A Behrangi, GY Niu* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.14493">Recent advances in machine learning have improved rainfall-runoff prediction, yet most data-driven models remain difficult to interpret and do not explicitly enforce physical conservation laws. We...</span><span class="abstract-full" id="full-2610.14493" hidden>Recent advances in machine learning have improved rainfall-runoff prediction, yet most data-driven models remain difficult to interpret and do not explicitly enforce physical conservation laws. We present the Gated Catchment Hydrology Model (GCHM), a mass-conserving architecture based on the Mass-Conserving Perceptron (MCP). GCHM uses five storage states representing canopy interception, snow, soil moisture, groundwater, and streamflow generation. Constrained gates govern precipitation partitioning, evapotranspiration, infiltration, seepage, runoff, and baseflow while learning context-dependent process relationships.   We evaluate GCHM across 50 CAMELS catchments spanning snowmelt-dominated, recent-rainfall, historical-rainfall, and arid regimes, using NOAA AORC meteorological forcing and remotely sensed vegetation and land-surface data. Performance is assessed using the Jawad-Kling-Gupta Efficiency (JKGE) metric, which accounts for stationary and non-stationary hydrologic behavior. We compare the full architecture with ablated variants and recurrent neural network benchmarks.   GCHM reproduces streamflow dynamics across diverse hydroclimatic regimes, including arid catchments, while maintaining interpretable states and fluxes. Ablation experiments show that simplified architectures may perform adequately under wet conditions but struggle to represent flow persistence and recession during dry periods, when groundwater storage sustains low flows. Learned gates recover hydrologically plausible relationships governing snowmelt, runoff, seepage, and evapotranspiration. GCHM combines competitive predictive performance with mass conservation and interpretability, providing a foundation for hydrologic modeling and counterfactual analysis.</span> <span class="abstract-toggle" data-id="2610.14493">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.14493v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.14493v1) · [:material-content-copy: BibTeX](bibtex/2610.14493.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=recurrent-networks" data-tag="recurrent-networks">Recurrent networks</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=interpretability" data-tag="interpretability">Interpretability</a>
    { .paper-tags }

-   #### Diffusion Precipitation Generation with Medium-range Weather Forecasting { #2610.12976 }

    *Guowen Li, Yang Liu, Mengxuan Chen, Jiashun Cheng, Haoyuan Liang, Shilei Cao, Yujie Wang et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.12976">Accurate daily precipitation forecasts are important for weather-sensitive decisions. Although recent Weather Foundation Models (WFMs) have achieved significant progress, they are trained on...</span><span class="abstract-full" id="full-2610.12976" hidden>Accurate daily precipitation forecasts are important for weather-sensitive decisions. Although recent Weather Foundation Models (WFMs) have achieved significant progress, they are trained on reanalysis data, which exhibit regional and intensity-dependent biases. Since retraining WFMs on observation-based targets is costly, we study Dynamics-Precipitation Decoupling, a plug-and-play framework that uses a separately trained generator to map non-precipitation forecasts from a frozen weather model to observation-aligned daily precipitation. Its key challenges are to capture heterogeneous rainfall distributions and align daily accumulations with atmospheric signals that evolve across space and time. To address them, we train a diffusion generator conditional on atmospheric states and satellite-gauge estimates of precipitation. A multi-scale recurrent encoder summarizes atmospheric evolution, while cross-attention aligns atmospheric and precipitation features during denoising. Extensive evaluation of various backbones, including Pangu-Weather, GraphCast, and HRES, on global and regional forecasting shows improved daily precipitation accuracy and event detection, with better representation of localized accumulation patterns. Global ensemble evaluation shows lower Brier scores than ENS and training-from-scratch GenCast at all reported lead times, with lower CRPS in most comparisons. Remarkably, a heavy-rainfall case study using 2,288 stations in China further supports improved daily accumulation patterns and event detection against ground observations.</span> <span class="abstract-toggle" data-id="2610.12976">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.12976v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.12976v1) · [:material-content-copy: BibTeX](bibtex/2610.12976.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a> <a class="md-tag" href="/explore/?t=daily" data-tag="daily">Daily</a>
    { .paper-tags }

-   #### Learning Kilometer-Scale Weather Prediction with Global-Regional Alignment { #2610.12401 }

    *Guowen Li, Yang Liu, Yujie Wang, Qiuyan Sun, Haoyuan Liang, Juepeng Zheng, Hong Cheng, Haohuan Fu* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.12401">Kilometer-scale regional weather forecasting is essential for local weather warnings and weather-sensitive decisions. Existing data-driven approaches often rely on numerical forecasts for large-scale...</span><span class="abstract-full" id="full-2610.12401" hidden>Kilometer-scale regional weather forecasting is essential for local weather warnings and weather-sensitive decisions. Existing data-driven approaches often rely on numerical forecasts for large-scale guidance or require additional training of global forecasting components. Pretrained global weather models offer an efficient source of large-scale forecasts, motivating their reuse to guide high-resolution regional prediction. However, this coupling requires aligning global and regional representations across different grids and integrating global guidance with local interactions to advance regional states. We propose ScaleCast, a regional forecasting framework that addresses these challenges through Global-Regional Alignment. Its Global-Regional Conversion module aligns joint global and regional representations with regional locations, while the Global-Regional Alignment and Dynamics block combines aligned guidance with regional neighborhood interactions. Experiments using ERA5 global analyses on a 0.25-degree grid and CERRA regional reanalysis at 5.5 km spacing demonstrate improved regional forecasts across surface and upper-air variables, with a single trained model supporting multiple global forecast drivers (i.e., Pangu-Weather, GraphCast, and HRES) without specific retraining. Fine-tuning on HRRR at 3 km spacing further demonstrates the framework's adaptability to a different regional domain and spatial resolution. Windstorm case studies show improved cyclone positioning and core-pressure estimates, while comparisons with HadISD station observations show closer agreement with local temperature and humidity changes.</span> <span class="abstract-toggle" data-id="2610.12401">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.12401v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.12401v1) · [:material-content-copy: BibTeX](bibtex/2610.12401.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a> <a class="md-tag" href="/explore/?t=km-scale" data-tag="km-scale">Km-scale</a> <a class="md-tag" href="/explore/?t=quarter-degree" data-tag="quarter-degree">0.25°</a>
    { .paper-tags }

-   #### Just Weather Scoring: Efficient End-to-end Nowcasting with Distributional Diffusion { #2610.12189 }

    *Jannik Wiese, Johannes Schusterbauer, Tommaso Martorella, Björn Ommer* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.12189">Generative diffusion models are well-suited for probabilistic precipitation nowcasting, but existing approaches often rely on separately trained compression or deterministic forecasting components...</span><span class="abstract-full" id="full-2610.12189" hidden>Generative diffusion models are well-suited for probabilistic precipitation nowcasting, but existing approaches often rely on separately trained compression or deterministic forecasting components and remain costly at inference due to iterative denoising. We introduce Just Weather Scoring (JWS), a single-stage, end-to-end diffusion model which addresses both issues by forecasting directly in radar space and enabling few-step generation. Radar-space modeling greatly simplifies training and inference and eliminates uncertainty arising from lossy compression. JWS combines Masked Asynchronous Diffusion, a timestep-sampling scheme that preserves clean context while adapting diffusion training to high-dimensional spatio-temporal data, with a simple scoring-rule objective that aligns training with probabilistic forecasting and unlocks few-step generation. On the SEVIR and MeteoNet benchmarks, JWS achieves state-of-the-art probabilistic forecasting performance at reduced training and inference cost. Even our smallest model remains competitive using substantially fewer parameters and more than 17x faster inference.</span> <span class="abstract-toggle" data-id="2610.12189">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.12189v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.12189v1) · [:material-content-copy: BibTeX](bibtex/2610.12189.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a>
    { .paper-tags }

-   #### legoESM: a modular, differentiable, multiscale, AI-ready Earth system model built with AI agents { #2610.11883 }

    *Pierre Gentine, Dhruv Balwada, Aytaç Paçal, Linnia Hawkins, Alistair Adcroft, Hang Fan et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.11883">Earth system models (ESMs) have grown tremendously in realism, yet key uncertainties persist in the climate response to greenhouse-gas forcing, particularly due to cloud radiative feedbacks. In...</span><span class="abstract-full" id="full-2610.11883" hidden>Earth system models (ESMs) have grown tremendously in realism, yet key uncertainties persist in the climate response to greenhouse-gas forcing, particularly due to cloud radiative feedbacks. In addition, their software architecture was not designed for accelerator hardware or modern artificial intelligence (AI). Here we present legoESM, a composable, differentiable, multiscale ESM written in JAX. It builds on decades of community-developed parameterizations and numerical methods, recast in a unified framework by AI coding agents under a human-specified scientific contract and verified through benchmarking. Dynamical cores, physics schemes, grids, complexity levels and components are swappable like building blocks, and can use conventional physics or machine-learned emulators. A single code base spans metre-scale large-eddy simulation to global simulations and weather to climate. End-to-end differentiability enables gradient-based calibration, variational data assimilation and online training. legoESM modular architecture enables systematic evaluation of diverse model variants to explore structural uncertainty and test hypotheses. legoESM produces realistic simulations across scales, reduces land-surface temperature bias through gradient-based calibration, and scales efficiently on GPUs to kilometer-scale simulations. It offers an open, community infrastructure for hypothesis testing, research and teaching in Earth sciences and a template for multiscale physical systems.</span> <span class="abstract-toggle" data-id="2610.11883">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.11883v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.11883v1) · [:material-content-copy: BibTeX](bibtex/2610.11883.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=km-scale" data-tag="km-scale">Km-scale</a>
    { .paper-tags }

-   #### A Physics-Constrained Implicit Profile Network for Continuous Reconstruction of Tropical Cyclone Near-Surface Wind Profiles { #2610.11405 }

    *Jian Ma, Yilin Yang, Robert Rogers, Jun A. Zhang, Jie Tang* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.11405">Near the ocean surface, tropical cyclone winds change rapidly with height, but direct measurements are limited because aircraft dropsondes provide only sparse and irregular observations. Continuous...</span><span class="abstract-full" id="full-2610.11405" hidden>Near the ocean surface, tropical cyclone winds change rapidly with height, but direct measurements are limited because aircraft dropsondes provide only sparse and irregular observations. Continuous wind profiles are important for understanding hurricane boundary-layer processes, improving storm-surge prediction, supporting offshore engineering, and assessing coastal hazards. In this study, we developed an artificial intelligence model that combines machine learning with physical principles to reconstruct continuous wind profiles from sparse observations. The model is designed to preserve the observed surface winds while generating realistic changes in wind speed and direction with height. Tests using more than two decades of NOAA hurricane observations show that the method accurately reproduces the vertical structure of tropical cyclone winds over a wide range of storm intensities. The framework can also extend satellite-derived surface wind measurements into three-dimensional near-surface wind fields, providing new opportunities for hurricane research, operational forecasting, and engineering applications.</span> <span class="abstract-toggle" data-id="2610.11405">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.11405v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.11405v1) · [:material-content-copy: BibTeX](bibtex/2610.11405.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=physics-ml-hybrid" data-tag="physics-ml-hybrid">Physics–ML hybrid</a> <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a>
    { .paper-tags }

-   #### A Graph Neural Network for Global Daily Fire Radiative Power Prediction at Medium-Range Lead Times { #2610.11022 }

    *Li Zhang, Jun Wang, Isidora Jankov, Yongxin Liu, Gonzalo A. Ferrada, Ravan Ahmadov, Ligia Bernardet et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.11022">Skillful prediction of biomass-burning activity several days in advance is important for air-quality forecasting and aerosol prediction. Two operational constraints motivate this work. First, the...</span><span class="abstract-full" id="full-2610.11022" hidden>Skillful prediction of biomass-burning activity several days in advance is important for air-quality forecasting and aerosol prediction. Two operational constraints motivate this work. First, the GBBEPx satellite fire radiative power (FRP) product used to initialize NOAA's GEFS-Aerosols is available with about a 1.5-day latency, so each forecast cycle relies on the most recently available, but already outdated, fire observations. Second, these fire inputs are then held fixed throughout the subsequent 5-day operational forecast, or 7 days in the GSL experimental system, effectively assuming no evolution in fire activity. We develop a data-driven model that predicts global FRP one to seven days ahead from the most recent available observations. The model adapts a spatiotemporal graph neural network using reanalysis meteorology, land-cover and vegetation information, recent fire history, and GBBEPx FRP as the training target. It is trained on 2020-2022 data and evaluated for 2023-2024. The model reproduces the global seasonal cycle and substantially outperforms persistence. At 0.1$^\circ$ resolution, mean squared error is reduced by 32% at one-day lead and 43% at seven days in 2023, and by 24% and 40% in 2024. At 1$^\circ$ resolution, the critical success index ranges from 0.32 to 0.60. Detection skill declines only modestly with lead time, whereas intensity skill degrades more rapidly. Large fires are detected reliably, but their radiative power is systematically underestimated. These results demonstrate useful predictability of fire activity several days ahead and identify intensity calibration and small-fire placement as the main remaining challenges before predicted FRP can support operational aerosol forecasts.</span> <span class="abstract-toggle" data-id="2610.11022">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.11022v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.11022v1) · [:material-content-copy: BibTeX](bibtex/2610.11022.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=graph-neural-networks" data-tag="graph-neural-networks">Graph neural networks</a> <a class="md-tag" href="/explore/?t=coarse" data-tag="coarse">Coarse (≥1°)</a> <a class="md-tag" href="/explore/?t=daily" data-tag="daily">Daily</a>
    { .paper-tags }

-   #### Low-rank tensor structure of precipitation and its application to satellite-reference merging { #2610.11000 }

    *Ryan Solgi, Rohan Shankar, Hugo A. Loaiciga* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.11000">The intermittent and variable nature of precipitation makes its accurate estimation over extended domains difficult, yet its spatiotemporal structure suggests that a low-rank representation may be...</span><span class="abstract-full" id="full-2610.11000" hidden>The intermittent and variable nature of precipitation makes its accurate estimation over extended domains difficult, yet its spatiotemporal structure suggests that a low-rank representation may be possible. This work represents daily precipitation over the contiguous United States (CONUS) as spatiotemporal tensors and applies CANDECOMP/PARAFAC factorization, showing that preserving the native spatial and temporal modes yields more accurate reconstruction than factorizing independent daily fields or unfolded space--time matrices. Building on this finding, this work presents TMerge, a tensor-based framework that integrates satellite precipitation with sparse reference observations through shared low-rank spatial and temporal factors. TMerge was applied to correct the IMERG Final Run product with climate prediction center reference observations over CONUS. During 2019-2022, TMerge increased correlation from 0.53 to 0.85 and reduced root-mean-square error and mean absolute error by 48.2% and 29.3%, respectively. TMerge consistently outperformed linear bias correction, quantile mapping, and neural networks across seasons, precipitation-intensity regimes, and regions. Improvements were spatially coherent and largest in coastal regions where IMERG errors were greatest. These results demonstrate that low-rank tensor structure parsimoniously approximates the dominant spatiotemporal variability of precipitation and provides a practical mechanism for improving satellite estimates under limited reference observations over extended domains.</span> <span class="abstract-toggle" data-id="2610.11000">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.11000v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.11000v1) · [:material-content-copy: BibTeX](bibtex/2610.11000.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a> <a class="md-tag" href="/explore/?t=daily" data-tag="daily">Daily</a>
    { .paper-tags }

-   #### Strategic Governance of AI Models in Earth Science { #2610.10560 }

    *Makoto Kelp, Amirhossein Arzani, Patricia Castellanos, Paul Griffiths, Ivan Higuera-Mendieta et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.10560">AI foundation models pretrained on weather and climate data are increasingly fine-tuned to Earth science tasks well beyond weather forecasting. Their development and adoption are outpacing the...</span><span class="abstract-full" id="full-2610.10560" hidden>AI foundation models pretrained on weather and climate data are increasingly fine-tuned to Earth science tasks well beyond weather forecasting. Their development and adoption are outpacing the scientific community's ability to evaluate them. These models are judged almost entirely by benchmark skill metrics, which measure how closely a forecast reproduces a reference product but not whether a model represents the physical processes governing the system it predicts. Forecast skill and physical reliability are therefore distinct properties. The distinction is most consequential under the nonstationary conditions of a changing climate for which these models were never trained. We identify five priorities for the physical evaluation of AI models in Earth science from task-specific emulators to foundation models, spanning training data, fine-tuning, behavioral testing, mechanistic interpretability, and output validation. We recommend three activities for the coming decade: 1) open AI-ready evaluation datasets, 2) a shared reporting standard for physics-based evaluation, and 3) a dedicated research program on the safety of these models.</span> <span class="abstract-toggle" data-id="2610.10560">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.10560v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.10560v1) · [:material-content-copy: BibTeX](bibtex/2610.10560.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=foundation-models" data-tag="foundation-models">Foundation models</a> <a class="md-tag" href="/explore/?t=evaluation" data-tag="evaluation">Evaluation</a> <a class="md-tag" href="/explore/?t=interpretability" data-tag="interpretability">Interpretability</a>
    { .paper-tags }

-   #### SciExam for ENSO: Can AI Agents Build Climate Models? { #2610.10513 }

    *Yinling Zhang, Langchen Liu, Dongbin Xiu, Xueyan Zou, Xu Kuang, Mengdi Wang, Shilong Liu* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.10513">Language-model agents are increasingly asked to carry out open-ended scientific research, yet their results are usually graded against a known answer, a rubric, or a language-model reviewer, none of...</span><span class="abstract-full" id="full-2610.10513" hidden>Language-model agents are increasingly asked to carry out open-ended scientific research, yet their results are usually graded against a known answer, a rubric, or a language-model reviewer, none of which can tell whether a new scientific model is valid. The AI Science Exam for El Nino-Southern Oscillation (SciExam for ENSO) is a benchmark in which agents build low-order stochastic models of ENSO, the dominant mode of interannual climate variability, from real observations. Within a six-hour budget, agents process the observations, write their own diagnostics, which are then frozen, and develop a model using only these diagnostics as feedback. Hidden graders then test whether the model reproduces ENSO's statistics, recovers unobserved variables, and forecasts held-out years, and score a published model in the same way. Across twelve agent systems, six produce models that score higher than the published model, mainly through better reconstruction and forecasting. The simplified forms of the stronger models are each compatible with one of the two competing explanations of ENSO's warm-cold asymmetry, an open debate that the task never mentions. Controlled runs of the top system under varied information suggest that its scores do not come from recalling the dated observational record and that the information it receives shapes how it builds its model. SciExam for ENSO can thus evaluate agent research where no answer is known, and the results suggest that agents can already build competitive models whose structures bear on questions that scientists still debate.</span> <span class="abstract-toggle" data-id="2610.10513">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.10513v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.10513v1) · [:fontawesome-brands-github: Code](https://github.com/ylzhang2447/SciExam-ENSO-code) · [:material-content-copy: BibTeX](bibtex/2610.10513.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=subseasonal-to-seasonal" data-tag="subseasonal-to-seasonal">Subseasonal to seasonal</a>
    { .paper-tags }

-   #### Conditional Flow Matching for Generation of 3D Multi-variable Instantaneous Urban Microclimate Fields { #2610.10430 }

    *Peng Liu, Shaoxiang Qin, Theodore Potsis, Lili Ji, Dingyang Geng, Liangzhu Leon Wang* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.10430">Rapid and accurate prediction of urban wind and temperature fields is important for urban microclimate design and climate adaptation. Large-eddy simulation (LES) effectively resolves these...</span><span class="abstract-full" id="full-2610.10430" hidden>Rapid and accurate prediction of urban wind and temperature fields is important for urban microclimate design and climate adaptation. Large-eddy simulation (LES) effectively resolves these instantaneous fields, but its application is limited in iterative design of urban microclimate applications due to high computational cost. Existing regressive data-driven models offers quick outputs, but they produce only deterministic point predictions that inherently fail to represent turbulent stochasticity. This paper adopts a novel generative framework of Conditional Flow Matching (CFM) that uses building geometry and mean flow as guidance to generate plausible three-dimensional instantaneous velocity and temperature fields for urban microclimate in seconds. To overcome the GPU memory bottleneck of pixel space 3D generation, the model operates in parallel on overlapping pixel space through a shared-noise initialization that preserves high spatial continuity of flow structure across the entire domain. Against reference LES data, the CFM surrogate can rapidly and accurately restore the first-order statistics with Normalized Root Mean Square Error (NRMSE) of 2.99% for wind and 1.77% for temperature, second-order turbulence metrics with NRMSE of 7.17% for wind and 8.84% for temperature, turbulent kinetic energy with NRMSE of 7%, probability density function and vertical profiles in representative locations. Wind engineering application of local gust prediction demonstrate that the speed and accuracy of CFM, supporting the use of generative AI for making turbulence-aware resilient urban design and climate adaptation more computationally feasible.</span> <span class="abstract-toggle" data-id="2610.10430">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.10430v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.10430v1) · [:material-content-copy: BibTeX](bibtex/2610.10430.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a>
    { .paper-tags }

-   #### WxFM-XL: Adapting Univariate Foundation Models to Multi-Station Weather Forecasting { #2610.10057 }

    *Xiao Wang, Changjian Chen, Zhuo Tang, Rongwen Li, Hongwu Liu, Kenli Li* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.10057">With the rise of univariate time series foundation models (e.g., Sundial, Timer), initial efforts have been made to extend them to multivariate settings. However, these models mainly focus on...</span><span class="abstract-full" id="full-2610.10057" hidden>With the rise of univariate time series foundation models (e.g., Sundial, Timer), initial efforts have been made to extend them to multivariate settings. However, these models mainly focus on modeling correlations among variables. When they are applied to multi-station weather forecasting, two important factors are often overlooked: (1) the spatial information of stations, and (2) different error priors of different stations relative to the foundation model. In this paper, we propose WxFM-XL, a model for adapting univariate time series foundation models to multi-station weather forecasting. WxFM-XL introduces a cross-station error correlation prior graph to capture stationwise error priors with respect to the foundation model. Building on this, we further propose a dynamic fusion mechanism that adaptively integrates a spatial correlation graph with the error correlation prior graph. Experiments on multiple datasets demonstrate that our model outperforms state of the art baselines.</span> <span class="abstract-toggle" data-id="2610.10057">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.10057v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.10057v1) · [:material-content-copy: BibTeX](bibtex/2610.10057.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=foundation-models" data-tag="foundation-models">Foundation models</a> <a class="md-tag" href="/explore/?t=station-point" data-tag="station-point">Station / point</a>
    { .paper-tags }

-   #### Learning joint probabilistic weather forecasts from station observations alone { #2610.09898 }

    *Chaeyeon Yi, Yun Am Seo* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.09898">Assessing compound weather risks requires forecasts representing dependence between variables. CLARA (Calibrated Advection-Routing Attention) learns joint Gaussian predictive distributions of five...</span><span class="abstract-full" id="full-2610.09898" hidden>Assessing compound weather risks requires forecasts representing dependence between variables. CLARA (Calibrated Advection-Routing Attention) learns joint Gaussian predictive distributions of five surface variables from station observations alone, without numerical weather prediction or reanalysis; the approximately 28,000-parameter model supports CPU training and prediction. Across six multi-year folds on 96 stations, its lead-mean energy score is 4.9% lower than that of a learned comparator with matched temporal inputs (4.7% with a similar parameter count) and 11-65% lower than those of statistical baselines. Holding marginal variances fixed, removing learned correlations worsens joint negative log-likelihood by 1.0-2.8 nats per station. A covariance-scale estimator, proved consistent under stated assumptions, improves short-lead calibration but over-corrects at long leads. Synthetic interventions show an attention-bias coefficient alone does not measure forecast influence. Retrained in ten regions on six continents, CLARA outperforms persistence in all 60 multi-year region-lead comparisons and a similarly sized learned model in 57 of 60.</span> <span class="abstract-toggle" data-id="2610.09898">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.09898v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.09898v1) · [:material-content-copy: BibTeX](bibtex/2610.09898.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=station-point" data-tag="station-point">Station / point</a>
    { .paper-tags }

-   #### Generative and deterministic deep learning models comparison for fine-scale precipitation retrievals from infrared brightness temperature { #2610.09859 }

    *Matthieu Meignin, Cécile Mallet, Nicolas Viltard* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.09859">Accurate precipitation estimation at fine spatial scales is critical for hydrology, agriculture, and climate studies. Infrared brightness temperatures from geostationary satellites offer excellent...</span><span class="abstract-full" id="full-2610.09859" hidden>Accurate precipitation estimation at fine spatial scales is critical for hydrology, agriculture, and climate studies. Infrared brightness temperatures from geostationary satellites offer excellent temporal coverage over continental-scale domains. However, because these measurements primarily characterize cloud-top properties rather than precipitation processes near the surface, their correlation with rainfall intensity remains limited, making quantitative precipitation estimation challenging. In this study, we conduct a systematic inter-comparison of state-of-the-art deep learning models for high-resolution precipitation retrieval from Meteosat Second Generation infrared brightness temperatures over metropolitan France. These models include deterministic U-Nets, transformer-based architectures, conditional GANs, and diffusion models. We construct a curated dataset spanning 2008--2023, combining M{é}t{é}o-France radar mosaics as reference with multi-channel infrared observations, and design preprocessing and sampling strategies to address the heavy-tailed, intermittent nature of rainfall. Our results show that deterministic models provide robust mean estimates and excel in pixel-wise accuracy, but systematically underestimate extreme precipitation. In contrast, generative models better capture the full precipitation distribution, including rare and heavy rainfall events, producing more realistic spatial structures at the cost of reduced pixel-wise fidelity. These results highlight a trade-off between pixel-wise accuracy and precipitation variability, showing that generative approaches are advantageous for extreme-event detection and probabilistic applications. This work establishes a reproducible framework for evaluating infrared- based precipitation retrieval methods and provides guidance for designing models that balance precision, variability, and extreme-event representation.</span> <span class="abstract-toggle" data-id="2610.09859">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.09859v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.09859v1) · [:material-content-copy: BibTeX](bibtex/2610.09859.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=gans" data-tag="gans">GANs</a> <a class="md-tag" href="/explore/?t=transformers" data-tag="transformers">Transformers</a> <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=benchmarks-datasets" data-tag="benchmarks-datasets">Benchmarks & datasets</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a>
    { .paper-tags }

</div>

