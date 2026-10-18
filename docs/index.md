---
hide:
  - navigation
  - toc
title: weatherml
---

A collection of papers on AI for weather forecasting, climate modelling and atmospheric science.

<p class="page-meta" markdown="span">1559 papers · updated 2026-10-18 · <a href="feed.xml">:material-rss: RSS</a> · <a href="all_papers.bib" download>:material-download: BibTeX</a> · <a href="https://github.com/weatherml/weatherml.github.io/issues/new?template=suggest-paper.yml">:material-plus: Suggest a paper</a></p>

## Browse by Topic

<div class="grid cards topics" markdown>

-   [Global Models](papers/global-models/index.md) <span class="topic-count">371</span>
-   [Regional Models](papers/regional-models/index.md) <span class="topic-count">65</span>
-   [Nowcasting](papers/nowcasting/index.md) <span class="topic-count">102</span>
-   [Downscaling](papers/downscaling/index.md) <span class="topic-count">103</span>
-   [Post-processing](papers/post-processing/index.md) <span class="topic-count">26</span>
-   [Data Assimilation](papers/data-assimilation/index.md) <span class="topic-count">79</span>
-   [Climate Modeling](papers/climate-modeling/index.md) <span class="topic-count">302</span>
-   [Hydrology](papers/hydrology/index.md) <span class="topic-count">39</span>
-   [Ocean & Sea Ice](papers/ocean-sea-ice/index.md) <span class="topic-count">89</span>
-   [Air Quality & Composition](papers/air-quality-composition/index.md) <span class="topic-count">72</span>
-   [Remote Sensing](papers/remote-sensing/index.md) <span class="topic-count">101</span>
-   [Other](papers/other/index.md) <span class="topic-count">210</span>

</div>

## Recent Additions

<div class="grid cards" markdown data-search-exclude>

-   #### Can AI Forecast the Extremes? Evaluating Probabilistic Weather Models on Tropical Cyclones { #2610.22077 }

    *Nathan Chalumeau, Xavier Renard, Remi Meynadier, Hugo Rakotoarimanga, Marcin Detyniecki et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.22077">Weather prediction is undergoing an important shift, with data-driven AI models outperforming leading numerical weather prediction systems on average at a fraction of the computational cost. Yet,...</span><span class="abstract-full" id="full-2610.22077" hidden>Weather prediction is undergoing an important shift, with data-driven AI models outperforming leading numerical weather prediction systems on average at a fraction of the computational cost. Yet, whether this success extends to extreme events remains contested. Earlier evaluations have exposed important shortcomings, but have largely focused on deterministic models and metrics, leaving the capabilities of the latest probabilistic models less understood. We conduct an extensive benchmark of tropical cyclone forecasting, comparing GenCast, FourCastNet 3, and FGN with ECMWF's operational physics-based ensemble, IFS-ENS. We develop a suite of probabilistic metrics and diagnostic analyses to assess cyclone tracking, intensity biases, uncertainty, and the wind-pressure relationship that characterizes storm structure. We find that, despite remaining limitations in intensity and ensemble spread, leading AI models substantially outperform IFS-ENS in cyclone tracking: FGN achieves better probabilistic location scores for 90% of evaluated storms in 2025. Further analyses reveal model-specific failure modes and suggest a lingering influence of pretraining data, pointing to opportunities for targeted improvement. Nonetheless, these results suggest that AI models are capable of capturing physical relationships that remain predictive well beyond typical weather conditions, motivating further investigation into how these capabilities emerge and how to strengthen them.</span> <span class="abstract-toggle" data-id="2610.22077">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.22077v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.22077v1) · [:material-content-copy: BibTeX](bibtex/2610.22077.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### Digital Typhoon Dataset V3: Benchmarking for Multi-spectral Extension and Analyzing Systematic Bias in Intensity Regression { #2610.21654 }

    *Asanobu Kitamoto, Victor Le Louarn, Noelia Falczuk, Ngoc Anh Tong, Anja Veleva* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.21654">This paper presents the Digital Typhoon Dataset V3, a new version of the longest typhoon satellite image dataset spanning nearly 50 years, designed to benchmark machine learning models on long-term...</span><span class="abstract-full" id="full-2610.21654" hidden>This paper presents the Digital Typhoon Dataset V3, a new version of the longest typhoon satellite image dataset spanning nearly 50 years, designed to benchmark machine learning models on long-term spatio-temporal data. The latest addition to Dataset V3 is multispectral satellite data, enabling multi-channel models. We investigate data-splitting strategies and present benchmarking results for the Northern and Southern Hemispheres. Experimental results show that, under the current dataset construction and experimental setting, multi-channel inputs do not clearly outperform single-channel inputs for pressure regression. Among the evaluated model architectures, image types, and spectral channels, ResNet-18, full-resolution images, and Channel 1 generally provide the best single-channel performance for pressure regression. Performance also differs between the WP and AU datasets by regression target, with lower RMSE on WP for pressure regression and on AU for wind regression. These results highlight the importance of considering dataset-specific and task-specific characteristics when interpreting benchmark performance. Additionally, we analyzed systematic bias in intensity regression, which leads to underestimation of intensity for strong typhoons. We tested imbalance-aware weighting strategies, including inverse-density weighting inspired by BalancedMSE, and showed that these approaches are not sufficient to remove the bias. The dataset is publicly available at http://agora.ex.nii.ac.jp/digital-typhoon/dataset/ and https://github.com/kitamoto-lab/digital-typhoon/.</span> <span class="abstract-toggle" data-id="2610.21654">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.21654v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.21654v1) · [:fontawesome-brands-github: Code](https://github.com/kitamoto-lab/digital-typhoon) · [:material-content-copy: BibTeX](bibtex/2610.21654.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a> <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a> <a class="md-tag" href="/explore/?t=benchmarks-datasets" data-tag="benchmarks-datasets">Benchmarks & datasets</a>
    { .paper-tags }

-   #### Energy Flow Shapes the Decomposition Dynamics of a Reactive Atmospheric Intermediate { #2610.21259 }

    *Mitradip Das, Cangtao Yin, M. Meuwly* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.21259">The unimolecular dynamics of Criegee intermediates depend strongly on their internal energy, which can differ substantially between nascent atmospheric species and laboratory-prepared reactants....</span><span class="abstract-full" id="full-2610.21259" hidden>The unimolecular dynamics of Criegee intermediates depend strongly on their internal energy, which can differ substantially between nascent atmospheric species and laboratory-prepared reactants. Here, the decomposition dynamics of <em>anti-</em>CH$_3$CHOO are investigated using molecular dynamics simulations on a neural network-based potential energy surface under thermal and mode-selective vibrational excitation. Simulations were performed at internal energies representative of those released during ozonolysis (chemical/thermal preparation, TP) or following vibrational excitation (VibEx) along the CH-stretching mode with two or three quanta, as can be done in laboratory studies. These excitation conditions strongly affect reaction yields and kinetics, with VibEx promoting passage across the initial cyclization barrier. For TP, 35 kcal/mol of internal energy are required to form molecular products, whereas for VibEx, 27 kcal/mol are sufficient. Although TP and VibEx ensembles access largely the same fragmentation pathways with comparable relative yields, the absolute product yields, characteristic timescales, and kinetic behavior differ substantially. Methyldioxirane is found to be a kinetic trap along the pathway. The resulting molecular-level description of <em>anti-</em>CH$_3$CHOO decomposition provides guidance for interpreting experiments and contributes to a broader understanding of energy-dependent Criegee-intermediate dynamics.</span> <span class="abstract-toggle" data-id="2610.21259">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.21259v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.21259v1) · [:material-content-copy: BibTeX](bibtex/2610.21259.bib){ .bibtex-link }
    { .paper-links }

-   #### Variable Masking for Climate Emulation in ACE { #2610.21012 }

    *Alexey Yermakov, Troy Arcomano, Jeremy McGibbon, Brian Henn, Oliver Watt-Meyer* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.21012">In the recent AI Model Intercomparison Project (AIMIP) benchmark, the ACE2.1-ERA5 emulator was shown to underestimate the true near-surface warming trend both during the training period and the...</span><span class="abstract-full" id="full-2610.21012" hidden>In the recent AI Model Intercomparison Project (AIMIP) benchmark, the ACE2.1-ERA5 emulator was shown to underestimate the true near-surface warming trend both during the training period and the out-of-sample extrapolation period. In this work, we present M-ACE2S (Masked-ACE2S), a modified version of the stochastic ACE2S model, which uses variable masking during training and reduces the reliance of M-ACE2S on any single input channel. We show that M-ACE2S improves long-term near-surface air temperature prediction. We also present a novel data normalization strategy called "global mean removal", which, combined with masking, further improves the near-surface temperature forecasts. We show that M-ACE2S outperforms the ACE2S baseline on in-sample data for ERA5 while improving training stability and generalizability of near-surface temperature to unseen forcing when paired with global mean removal.</span> <span class="abstract-toggle" data-id="2610.21012">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.21012v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.21012v1) · [:fontawesome-brands-github: Code](https://github.com/ai2cm/ace) · [:material-content-copy: BibTeX](bibtex/2610.21012.bib){ .bibtex-link }
    { .paper-links }

-   #### Forecast lead time and the timing of crew deployment for weather-driven power outages using AI weather models { #2610.20918 }

    *Mohammad Kian Golkar, Luciano Alves de Oliveira, Mohamad Alipour* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.20918">Utilities request mutual aid crews to restore power after severe storms, and requesting early secures more crews but relies on less accurate forecasts. We examined how forecast lead time affects the...</span><span class="abstract-full" id="full-2610.20918" hidden>Utilities request mutual aid crews to restore power after severe storms, and requesting early secures more crews but relies on less accurate forecasts. We examined how forecast lead time affects the value of outage predictions for crew deployment, using operational forecasts from two AI weather models, GraphCast and FourCastNet, to predict outage occurrence for counties across the contiguous United States 24 to 120 hours ahead. Forecasts were evaluated by the share of outage burden captured by the counties where crews would be staged. Captured burden declined with lead time: the loss between three and five days was significant in all analyses, whereas the loss within three days was significant only for outages that had not yet started. Early deployment with all requested crews outperformed late deployment with fewer, and we report the availability ratio at which the two break even. Antecedent land surface predictors computed at forecast valid time include weather from the forecast period, which inflates their skill by 42 to 53% and reverses their apparent advantage over forecast meteorology at long lead times. Computed correctly, they improved occurrence skill by 17.5% at 24 hours but did not significantly change captured burden, a gap that attribution analysis linked to their static component. Prediction intervals covered only 0.476 of tropical cyclone events against a nominal 0.90. Calibration by hazard class corrected most of this shortfall at an 11% cost in interval width, as a diagnostic only, because hazard class is known only after the event.</span> <span class="abstract-toggle" data-id="2610.20918">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.20918v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.20918v1) · [:material-content-copy: BibTeX](bibtex/2610.20918.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a>
    { .paper-tags }

-   #### Improved Precipitation Variability in the Community Atmosphere Model through Implementation of a Neural-Network Deep Convection Scheme { #2610.20405 }

    *Xavier J. Levine, Griffin Mooers, Jack W. Atkinson, Marat F. Khairoutdinov, Janni Yuval et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.20405">Conventional deep convection schemes remain a primary source of uncertainty in global climate models, yet replacing them with data-driven alternatives while maintaining physical consistency and...</span><span class="abstract-full" id="full-2610.20405" hidden>Conventional deep convection schemes remain a primary source of uncertainty in global climate models, yet replacing them with data-driven alternatives while maintaining physical consistency and numerical stability is an open challenge. We present a hybrid configuration of the Community Atmosphere Model version 6 (CAM6), in which a fully connected deep neural network replaces the Zhang-McFarlane deep convection scheme (ZM), while radiation, turbulence, and other cloud processes are left unchanged. The network is trained on output from a storm-resolving global model - the global System for Atmospheric Modeling (gSAM) - using a coarse-graining approach to derive subgrid-scale convective fluxes and tendencies. This data-driven parameterization is ported to CAM6 and found to run stably for more than a decade while conserving energy and moisture. It produces a broadly realistic climatology without retuning the remaining parameterizations. Relative to the default Zhang-McFarlane scheme, the neural network parameterization substantially improves some long-standing model deficiencies. In particular, the biases in light precipitation and extreme precipitation are markedly reduced, and the diurnal cycle of tropical land precipitation shifts from a spurious midday peak over many regions to the observed late-afternoon peak. These results support targeted ML-based convection schemes as a viable path toward reducing systematic model errors in next-generation climate models.</span> <span class="abstract-toggle" data-id="2610.20405">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.20405v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.20405v1) · [:material-content-copy: BibTeX](bibtex/2610.20405.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a> <a class="md-tag" href="/explore/?t=km-scale" data-tag="km-scale">Km-scale</a>
    { .paper-tags }

-   #### PrecipCorrDiff: A Deep Learning Framework for Generating 0.0625$^\circ$ High-Resolution IMD Gridded Precipitation Data { #2610.19994 }

    *Bipin Kumar, Rajib Chattopadhyay, Harsh P. Bachhav,  Jannu, Bhupendra Bahadur Singh, Ramesh Ragala et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.19994">High-resolution precipitation datasets are essential for weather forecasting, hydrological modeling, and climate risk assessment, yet observation-based gridded products often lack the spatial detail...</span><span class="abstract-full" id="full-2610.19994" hidden>High-resolution precipitation datasets are essential for weather forecasting, hydrological modeling, and climate risk assessment, yet observation-based gridded products often lack the spatial detail required for regional applications. Here we present PrecipCorrDiff, a diffusion-based downscaling framework built by modifying the original corrector diffusion based archetecture that incorporates a customized loss function and a multi-scale residual generation strategy to improve the precipitation especially the representation of extremes. PrecipCorrDiff generates 0.0625° resolution India Meteorological Department (IMD) precipitation fields from 0.25° gridded data while preserving precipitation statistics and reconstructing fine-scale spatial variability. Validation against independent IMD station observations, comparing the proposed framework with the SRGAN-based model and the original CorrDiff model, indicates that the proposed framework substantially improves the representation of spatial precipitation patterns, rainfall intensity distributions, and extreme rainfall events, while eliminating the systematic negative bias present in the previously developed SRGAN-based model. Across four climatologically distinct homogeneous Indian monsoon regions, PrecipCorrDiff reproduces observed precipitation characteristics with near-zero spatial bias, demonstrating the potential of diffusion-based downscaling for generating physically consistent, high-resolution precipitation datasets for hydrological, meteorological, and climate applications.</span> <span class="abstract-toggle" data-id="2610.19994">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.19994v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.19994v1) · [:material-content-copy: BibTeX](bibtex/2610.19994.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=quarter-degree" data-tag="quarter-degree">0.25°</a>
    { .paper-tags }

-   #### Generative modelling for multivariate spatio-temporal downscaling of monthly climate fields { #2610.19885 }

    *Yu Huang, Shangshang Yang, Sebastian Bathiany, Michael Aich, Philipp Hess, Niklas Boers* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.19885">Inferring climate fields at higher spatio-temporal resolutions from coarse representations is essential for accessing socioeconomic and environmental impacts. The downscaling of modelled climate...</span><span class="abstract-full" id="full-2610.19885" hidden>Inferring climate fields at higher spatio-temporal resolutions from coarse representations is essential for accessing socioeconomic and environmental impacts. The downscaling of modelled climate fields faces a trade-off between data storage constraints, simulation efficiency, and resolution requirements. While many methods for spatial downscaling have been proposed, joint spatio-temporal downscaling has received comparable little attention. When deriving, for example, daily fields from climate datasets that are available only as monthly averages at coarser spatial resolution, it is highly challenging to faithfully reproduce extreme event statistics and spatiotemporal consistency. To address this task, we introduce a generative machine learning approach, using a diffusion model to jointly downscale monthly temperature and precipitation fields into temporally consistent daily fields at higher spatial resolution. We demonstrate that our approach can generate high-dimensional multivariate climate fields at daily temporal and 0.25° spatial resolution from monthly input fields at 1° spatial resolution, exhibiting coherent temporal dynamics and physically plausible spatial structures. In particular, by incorporating a conditional guidance technique, we show that a pre-trained diffusion model can be directly leveraged to efficiently generate daily high-resolution fields conditioned on monthly inputs, without requiring additional fine-tuning or retraining. Our probabilistic approach thus provides an efficient and generalizable framework for spatio-temporal climate downscaling, enabling flexible climate fields generation for climate impact assessments and digital Earth simulator constructions.</span> <span class="abstract-toggle" data-id="2610.19885">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.19885v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.19885v1) · [:material-content-copy: BibTeX](bibtex/2610.19885.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=quarter-degree" data-tag="quarter-degree">0.25°</a> <a class="md-tag" href="/explore/?t=coarse" data-tag="coarse">Coarse (≥1°)</a> <a class="md-tag" href="/explore/?t=daily" data-tag="daily">Daily</a> <a class="md-tag" href="/explore/?t=monthly" data-tag="monthly">Monthly</a>
    { .paper-tags }

-   #### Unmasking Compensating Errors in Numerical Weather Models: A Process-Additive Neural Network Approach { #2610.19772 }

    *Yuiko Ichikawa, Edward Groot, Jack Atkinson* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.19772">This work introduces a Process Additive Neural Network (PANN) framework to disentangle compensating biases in atmospheric models, where errors in different physical processes can offset each other...</span><span class="abstract-full" id="full-2610.19772" hidden>This work introduces a Process Additive Neural Network (PANN) framework to disentangle compensating biases in atmospheric models, where errors in different physical processes can offset each other and obscure their origins. Using a single-column model, each PANN branch represents the tendency error associated with a separate physical process, and the summed branch outputs approximate the total error relative to a reference tendency. Once trained successfully, the branch outputs are expected to provide process-level error attributions and, when averaged, estimates of process-level bias. Initial experiments focus on cloud-related processes in OpenIFS SCM case studies, where cloud and convection tendencies are strongly coupled and compensation is difficult to diagnose using standard approaches. The aim is not to replace human interpretation of physical processes, but to extend it to regimes where model complexity, process interaction, and compensating errors make direct diagnosis difficult. By providing a physically structured form of machine-learning interpretability, PANN can help connect complex model behaviour with process-level explanations. This work supports both improved understanding of compensating errors and the development of more physically consistent weather and climate models.</span> <span class="abstract-toggle" data-id="2610.19772">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.19772v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.19772v1) · [:material-content-copy: BibTeX](bibtex/2610.19772.bib){ .bibtex-link }
    { .paper-links }

-   #### Collaborative Neural Networks significantly Improve Global Aerosol Retrievals From Multi-Angle Polarimeters { #2610.19217 }

    *Guangliang Fu, Sha Lu, Meng Gao, Jeroen Rietjens, Otto Hasekamp* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.19217">We present the RemoTAP-Hybrid algorithm for global aerosol products retrieved from Multi-Angle Polarimeters (MAP), and apply it to one-year aerosol retrievals from the SPEXone MAP instrument. For the...</span><span class="abstract-full" id="full-2610.19217" hidden>We present the RemoTAP-Hybrid algorithm for global aerosol products retrieved from Multi-Angle Polarimeters (MAP), and apply it to one-year aerosol retrievals from the SPEXone MAP instrument. For the first time, Multiple Collaborated Neural Networks (MCNN) are used to perform global aerosol MAP retrievals over land and ocean. The combined use of MCNN with a full physics inversion in RemoTAP-Hybrid, significantly improves accuracy and computational speed of SPEXone aerosol retrievals. We validate the retrieved Aerosol Optical Depth (AOD), Angstrom Exponent (AE), Single Scattering Albedo (SSA), Absorption AOD (AAOD), Fine mode AOD (FAOD), Coarse mode AOD (CAOD), and fine mode effective radius (reff_fine) with AERONET data. The validation suggests that aerosol retrievals can be further improved when extending the aerosol description to include a fine non-spherical mode. For AOD, the Root-Mean-Square Error (RMSE) is 0.051 (0.037) over land (ocean), while 78.3% (80.8%) of the retrievals are within the requirement formulated by the Global Climate Observing System (GCOS). The RMSE is 0.186 (0.207) for AE, 0.038 (0.044) for SSA, 0.023 (0.014) for AAOD, 0.047 (0.031) for FAOD, 0.043 (0.027) for CAOD, and 0.021 (0.026) for reff_fine over land (ocean). Compared to the current RemoTAP algorithm, RemoTAP-Hybrid shows improved agreement with AERONET for AOD, AE, and FAOD over land and ocean while the number of valid retrievals is increased by 4% (3%) over land (ocean) and the computational time is reduced by 43% (32%). Finally, we present the first global seasonal maps of aerosol optical and microphysical properties and chemical composition from SPEXone.</span> <span class="abstract-toggle" data-id="2610.19217">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.19217v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.19217v1) · [:material-content-copy: BibTeX](bibtex/2610.19217.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a>
    { .paper-tags }

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

</div>

