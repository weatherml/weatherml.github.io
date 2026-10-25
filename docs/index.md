---
hide:
  - navigation
  - toc
title: weatherml
---

A collection of papers on AI for weather forecasting, climate modelling and atmospheric science.

<p class="page-meta" markdown="span">1581 papers · updated 2026-10-25 · <a href="feed.xml">:material-rss: RSS</a> · <a href="all_papers.bib" download>:material-download: BibTeX</a> · <a href="https://github.com/weatherml/weatherml.github.io/issues/new?template=suggest-paper.yml">:material-plus: Suggest a paper</a></p>

## Browse by Topic

<div class="grid cards topics" markdown>

-   [Global Models](papers/global-models/index.md) <span class="topic-count">378</span>
-   [Regional Models](papers/regional-models/index.md) <span class="topic-count">66</span>
-   [Nowcasting](papers/nowcasting/index.md) <span class="topic-count">105</span>
-   [Downscaling](papers/downscaling/index.md) <span class="topic-count">106</span>
-   [Post-processing](papers/post-processing/index.md) <span class="topic-count">26</span>
-   [Data Assimilation](papers/data-assimilation/index.md) <span class="topic-count">79</span>
-   [Climate Modeling](papers/climate-modeling/index.md) <span class="topic-count">304</span>
-   [Hydrology](papers/hydrology/index.md) <span class="topic-count">39</span>
-   [Ocean & Sea Ice](papers/ocean-sea-ice/index.md) <span class="topic-count">89</span>
-   [Air Quality & Composition](papers/air-quality-composition/index.md) <span class="topic-count">74</span>
-   [Remote Sensing](papers/remote-sensing/index.md) <span class="topic-count">101</span>
-   [Other](papers/other/index.md) <span class="topic-count">214</span>

</div>

## Recent Additions

<div class="grid cards" markdown data-search-exclude>

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

-   #### Urban climate regressions require representative monitoring networks { #2610.29423 }

    *Martí Bosch, Aldo Brandi, Christoph Bachofen, Perrine Hamel, Roy P. Remme, Emma E. Ramsay et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.29423">Land use regression models relate intra-urban air temperature to spatial predictors such as building volume, tree canopy cover and topography, and are increasingly used to produce fine-scale...</span><span class="abstract-full" id="full-2610.29423" hidden>Land use regression models relate intra-urban air temperature to spatial predictors such as building volume, tree canopy cover and topography, and are increasingly used to produce fine-scale temperature maps. However, the accuracy of these estimates can be compromised by unrepresentative monitoring networks. Here we show that reference weather station networks, designed for regional rather than intra-urban monitoring, sample only a narrow portion of a city's spatial heterogeneity, missing both the densely built core and the cool, vegetated periphery that set the two ends of the urban temperature range. As a result, city-wide predictions underestimate temperature differences, in ways that cross-validation against the station measurements does not detect. Using Zurich, Switzerland, as a case study, we compare multiple linear regression, generalized additive models and random forests as up to 117 low-cost temperature sensors are added to the four reference weather stations. When trained on reference stations alone, all three models achieve comparable cross-validated performance. As low-cost sensors broaden the sampled spatial heterogeneity, multiple linear regression scores lower on both metrics, whereas the non-linear models remain at their reference-only level. Although a third of the low-cost sensors already recovers nearly the full predicted temperature range, which predictors are identified, and with which sign, is determined by the station sample rather than the model class. At full coverage, only multiple linear regression leaves significant residual spatial autocorrelation, suggesting a linear form unable to capture non-linear responses of air temperature to urban form. Drawing on landscape ecology research on focal site study design, we provide a transferable framework for diagnosing station network representativeness and improving the reliability of city-wide environmental maps.</span> <span class="abstract-toggle" data-id="2610.29423">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.29423v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.29423v1) · [:material-content-copy: BibTeX](bibtex/2610.29423.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=classical-ml" data-tag="classical-ml">Classical ML</a> <a class="md-tag" href="/explore/?t=station-point" data-tag="station-point">Station / point</a>
    { .paper-tags }

-   #### Data-driven and physics-guided hindcasting of measured wave spectra { #2610.29306 }

    *Sander Rikka* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.29306">Short-range hindcasts of measured wave spectra are evaluated using a recurrent model initialized by one hourly buoy spectrum and supplied with local wind, fetch and depth. Measurements from three...</span><span class="abstract-full" id="full-2610.29306" hidden>Short-range hindcasts of measured wave spectra are evaluated using a recurrent model initialized by one hourly buoy spectrum and supplied with local wind, fetch and depth. Measurements from three Baltic Sea stations are combined for training in 2017--2020, validation in 2021 and testing in 2022. ERA5 reanalysis supplies winds throughout the twelve-hour prediction interval. The comparison examines supervision of spectral variance, peak frequency and peak amplitude, followed by empirical wind-sea bounds and a local relaxation constraint. Variance and peak-frequency supervision gives the lowest spectral error within the data-driven comparison, whereas additional peak-amplitude supervision gives the lowest wave-height error. In the matched physical comparison, relaxation reduces height RMSE by 5.01% to \PhPtwoHs{}~cm. Extending this constraint through hours 13--24, without spectral targets at those hours, reduces primary spectral RMSE by 2.31% to \PhPthreeSpec{}~$\unitE$. Upper-bound constraints increase errors. Wind-condition comparisons show spectral gains from extended relaxation in increasing, decreasing and steady strong winds, with little change in steady weak winds. The results support modest, objective-dependent benefits from local physical guidance. Wave-height scores concern the retained frequency band, and operational forecasting and transfer to independent stations remain to be evaluated.</span> <span class="abstract-toggle" data-id="2610.29306">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.29306v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.29306v1) · [:material-content-copy: BibTeX](bibtex/2610.29306.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=physics-ml-hybrid" data-tag="physics-ml-hybrid">Physics–ML hybrid</a> <a class="md-tag" href="/explore/?t=station-point" data-tag="station-point">Station / point</a> <a class="md-tag" href="/explore/?t=hourly" data-tag="hourly">Hourly</a>
    { .paper-tags }

-   #### From Weather to Enhanced Weathering: Weather Foundation Models for Carbon Dioxide Removal at Operational Scale { #2610.28973 }

    *Sasankh Munukutla, Felipe Oviedo, Danrong Zhang, Kenji Takeda, Juan M Lavista Ferres* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.28973">Enhanced Rock Weathering (ERW) is a carbon removal solution with the potential to remove gigatons of CO$_2$ per year, but quantifying that removal today requires dense soil sampling and laboratory...</span><span class="abstract-full" id="full-2610.28973" hidden>Enhanced Rock Weathering (ERW) is a carbon removal solution with the potential to remove gigatons of CO$_2$ per year, but quantifying that removal today requires dense soil sampling and laboratory analysis. We evaluate five weather Foundation Models (FMs) as frozen encoders feeding lightweight decoders, under a leave-one-field-out protocol on 5,192 chronosequences from 264 fields across Brazil and the US. To our knowledge, this is the first holistic evaluation of weather FMs for ERW and the largest evaluation on operational ERW data. We find that across weather FMs, representations transfer to ERW: our best model, TerraNova-ERW, built on Aurora 1.5, positively orders measurements within fields it has never seen. Zero-shot prediction of a new field's mean removal is not yet feasible, and we attribute this to data scale: the same architecture excels on $\sim$$10^5$ natural-weathering sequences, anchoring the scale we expect ERW data must reach. Few-shot prediction, blending the model with a few samples per field, is supported today: TerraNova-ERW recovers over 60% of the highest-removal fields with two samples per field, roughly halves the samples needed to estimate a field's mean removal, and can cut sampling on our densest fields by 28-98%.</span> <span class="abstract-toggle" data-id="2610.28973">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.28973v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.28973v1) · [:material-content-copy: BibTeX](bibtex/2610.28973.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=foundation-models" data-tag="foundation-models">Foundation models</a>
    { .paper-tags }

-   #### Tropical Cyclone Attributes in Sparse Autoencoder Features of an AI Weather Model { #2610.28904 }

    *Jiapei Liu, Ruidong Li, Ting Sun, Guangheng Ni* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.28904">Artificial intelligence (AI) weather models now forecast weather events skillfully, but how they represent these events internally remains unknown. Sparse autoencoders (SAEs) provide sparse features...</span><span class="abstract-full" id="full-2610.28904" hidden>Artificial intelligence (AI) weather models now forecast weather events skillfully, but how they represent these events internally remains unknown. Sparse autoencoders (SAEs) provide sparse features for probing hidden states, but how information about multiple physical attributes of an event is organized across them remains unclear. Using tropical cyclones in GraphCast as a test case, we analyze 2,115 six-hourly timesteps from 205 cyclones, relating SAE activations to position, intensity, and pressure structure across five Layer-8 decompositions and 16 processor layers. Each attribute relates to multiple features; position and pressure-structure relations overlap in a smaller subset. This organization recurs across independent SAEs and layers without feature alignment. In deeper layers, the leading position feature aligns better with position and pressure structure six hours ahead than at present. These results suggest that the organization of physical-attribute relations provides a reproducible level of description for interpreting cyclone-related information in GraphCast.</span> <span class="abstract-toggle" data-id="2610.28904">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.28904v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.28904v1) · [:material-content-copy: BibTeX](bibtex/2610.28904.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a> <a class="md-tag" href="/explore/?t=6-hourly" data-tag="6-hourly">6-hourly</a>
    { .paper-tags }

-   #### GraphCast-Mamba: Learning Non-Markovian Corrections for Weather Forecasting { #2610.28512 }

    *Ilya Vilkoviskiy, Liang-Hong Mo, Suozhi Huang, Dmitry A. Abanin* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.28512">Unresolved atmospheric processes can introduce history dependence beyond the short input windows used by many weather forecasting models. We introduce GraphCast--Mamba, which augments a frozen,...</span><span class="abstract-full" id="full-2610.28512" hidden>Unresolved atmospheric processes can introduce history dependence beyond the short input windows used by many weather forecasting models. We introduce GraphCast--Mamba, which augments a frozen, pretrained $1^\circ$ GraphCast backbone with a trainable Mamba correction branch combining graph message passing and Mamba layers. The branch retains a temporal memory state across successive six-hour forecast steps, learning corrections informed by temporal context beyond GraphCast's two-frame input. Training is confined to the correction branch, using ERA5 data from 2015--2021. After 2,500 optimizer updates, including warm-up, the selected model achieves substantial improvements relative to the uncorrected GraphCast backbone. It reduces the GraphCast loss (normalized weighted mean-squared error; MSE) by 14.42% when aggregated over the full ten-day forecast and by 21.66% at day 10 (240 h after forecast initialization). At this final forecast lead time, root-mean-square error (RMSE) decreases by 7.09--14.60% across all 11 evaluated variable groups, spanning temperature, wind, pressure, geopotential, specific humidity, and precipitation. These results demonstrate broad improvements in ten-day weather forecasts through a short correction-training schedule while retaining the pretrained backbone.</span> <span class="abstract-toggle" data-id="2610.28512">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.28512v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.28512v1) · [:material-content-copy: BibTeX](bibtex/2610.28512.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=graph-neural-networks" data-tag="graph-neural-networks">Graph neural networks</a>
    { .paper-tags }

-   #### Observation Assimilation Transformer: Probabilistic Weather Analysis from Sparse Station Observations { #2610.28327 }

    *Christian Donner, Michele Cattaneo, Maxim Samarin, Francesco Zanetta, Icíar Lloréns Jover et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.28327">We present the Observation Assimilation Transformer (OAT), a geometry-aware neural process that corrects and refines a prior numerical weather prediction (NWP) field into a high-resolution posterior...</span><span class="abstract-full" id="full-2610.28327" hidden>We present the Observation Assimilation Transformer (OAT), a geometry-aware neural process that corrects and refines a prior numerical weather prediction (NWP) field into a high-resolution posterior field by assimilating sparse observations. OAT is an encoder-decoder transformer that considers a set of observed locations and their NWP prior to condition predictions at a set of arbitrary unobserved query locations. Its sparse attention mechanism makes it scalable to thousands of station observations and depends on relative positions between locations, never on the absolute coordinates. The model allows for partially missing observations, predictions at arbitrary continuous locations without grid projection, and outputs are parametric distributions. A station-randomized training algorithm is developed with continuous ranked probability score (CRPS) loss. We test OAT on two datasets - one at regional scale with hundreds of stations, and one at continental scale with thousands of stations. On these datasets OAT outperforms several baselines and shows well calibrated predictions. In an illustrative case study, OAT's downscaling capabilities are demonstrated by downscaling ERA5 (~31km) to COSMO (1km), where OAT achieves lower mean absolute error (MAE) and CRPS for temperature and precipitation on test stations compared to COSMO.</span> <span class="abstract-toggle" data-id="2610.28327">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.28327v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.28327v1) · [:material-content-copy: BibTeX](bibtex/2610.28327.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=transformers" data-tag="transformers">Transformers</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a> <a class="md-tag" href="/explore/?t=station-point" data-tag="station-point">Station / point</a>
    { .paper-tags }

-   #### Disentangling forced and internal components in observed climate evolution using spatiotemporal deep learning { #2610.28219 }

    *Parvathi Kooloth, Wenyu Zhou, L. Ruby Leung, Shang-Ping Xie, Hailong Wang, Qiang Fu* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.28219">Attribution of observed historical climate evolution to externally forced change and internal variability is fundamental to understanding climate change and evaluating climate models. Yet such...</span><span class="abstract-full" id="full-2610.28219" hidden>Attribution of observed historical climate evolution to externally forced change and internal variability is fundamental to understanding climate change and evaluating climate models. Yet such attribution remains challenging because the forced response is intertwined with internal variability in observations. Here, we develop a convolutional long short-term memory network (ConvLSTM) to disentangle these two components. Trained and evaluated using large-ensemble simulations from multiple climate models, ConvLSTM consistently outperforms existing model-based, statistical, and machine-learning approaches. Applying ConvLSTM to observations reveals an evolving spatial pattern of the forced response and substantial contributions from internal variability to recent multidecadal climate trends. For several key climate features whose attribution remains debated, ConvLSTM indicates that internal variability amplified the forced reversal of Atlantic interhemispheric temperature asymmetry, dominated the observed La Niña-like warming pattern over the equatorial Pacific and the associated precipitation decline over southwestern North America since 1980, and modulated Arctic amplification around its nearly constant threefold forced value. Comparison between the ConvLSTM-estimated and model-simulated forced responses reveals broadly consistent patterns but also important discrepancies, with climate models simulating stronger Northern Hemisphere warming, weaker subtropical high intensification, and smaller land precipitation declines. Overall, ConvLSTM provides a robust framework for attributing historical climate evolution and evaluating the fidelity of climate models</span> <span class="abstract-toggle" data-id="2610.28219">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.28219v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.28219v1) · [:material-content-copy: BibTeX](bibtex/2610.28219.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=recurrent-networks" data-tag="recurrent-networks">Recurrent networks</a>
    { .paper-tags }

-   #### TerraNostra-Bench: Benchmarking Spatial Downscaling and Temporal Rollout for Global 1-km Land-Use Reconstruction { #2610.27793 }

    *Amirpasha Mozaffari, Marina Castaño, Stefano Materia, Joan Vedrí, Jordi Varela-Agrelo et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.27793">Uncertainty in the terrestrial carbon cycle remains a major constraint in climate projections, partly due to imperfect land surface representations in Earth system models. We propose a data-driven...</span><span class="abstract-full" id="full-2610.27793" hidden>Uncertainty in the terrestrial carbon cycle remains a major constraint in climate projections, partly due to imperfect land surface representations in Earth system models. We propose a data-driven framework for high-resolution land-use reconstruction, treated as a semantic-segmentation task, that combines coarse-resolution scenario data and climate classification with static geophysical features to produce annual land-use maps at 1 km resolution. We introduce TerraNostra-Bench, a change-aware evaluation protocol that separates spatial downscaling from autoregressive temporal rollout and scores models on the rare land-use-change pixels rather than the dominant static background. Using it, we compare convolutional (U-Net) and transformer (Swin) backbones in autoregressive and non-autoregressive schemes against persistence, linear, and shallow convolutional baselines. The autoregressive prior is the most informative input yet accumulates error over long rollouts: autoregressive models lose 5.7-7.2 accuracy points over 13 years, against 2.2-2.8 for non-autoregressive models. This trade-off is governed by the autoregressive choice rather than the backbone. It can be tuned by an inference-time prior-trust threshold that requires no retraining and by the training-time weighting of reconstruction and change objectives; the two levers trace complementary frontiers, with calibration recovering up to 10 accuracy points while keeping change sensitivity above the persistence floor. Extrapolating beyond the training window, backward to 1955-1959 and forward to 2015-2019 under scenario forcing, the models retain most of their in-window skill and show the same trade-off, indicating it is intrinsic to the design. This work provides the methodological foundation and evaluation surface for a forthcoming temporally continuous land-use product.</span> <span class="abstract-toggle" data-id="2610.27793">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.27793v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.27793v1) · [:material-content-copy: BibTeX](bibtex/2610.27793.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=km-scale" data-tag="km-scale">Km-scale</a>
    { .paper-tags }

-   #### Wet-Biased Data Pipelines Mislead the Training and Evaluation of Generative Precipitation Nowcasting { #2610.26486 }

    *Christoph Metzl, Stefan Kapsreiter, Kianusch Vahid Yousefnia, Virginia Poli, Richard Müller et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.26486">Generative models for precipitation nowcasting are trained and evaluated predominantly on wet-biased datasets, which over-represent precipitation events to emphasize rare extremes. Selecting test...</span><span class="abstract-full" id="full-2610.26486" hidden>Generative models for precipitation nowcasting are trained and evaluated predominantly on wet-biased datasets, which over-represent precipitation events to emphasize rare extremes. Selecting test cases based on target observations is known to favor forecasts that match the selected distribution rather than the intended deployment distribution. We show analytically how this target-dependent sampling also shifts the conditional distribution learned by generative models. When training and test sampling match, even an ideal model learns precisely the distribution favored by evaluation, allowing the evaluation bias to conceal the training bias. Using a fixed satellite-to-radar diffusion model, we demonstrate that target-dependent test sampling can reverse the calibration assessment. The model appears under-predictive on a wet-biased test set but over-predictive on a deployment-representative test set. This result is replicated outside the model's training domain after a seven-year gap. Utilizing classifier-free guidance as an example, we demonstrate how the contradictory calibration assessment can misdirect downstream model development.</span> <span class="abstract-toggle" data-id="2610.26486">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.26486v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.26486v1) · [:material-content-copy: BibTeX](bibtex/2610.26486.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### Spatio-temporal precipitation downscaling at hourly resolution based on km-scale regional simulations { #2610.26145 }

    *Henry Addison, Elizabeth J. Kendon, Peter A. G. Watson* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.26145">Effectively preparing for changes in extreme flooding requires understanding present and future rainfall extremes with return periods up to $\sim$100-1000 years, including changes in intensity,...</span><span class="abstract-full" id="full-2610.26145" hidden>Effectively preparing for changes in extreme flooding requires understanding present and future rainfall extremes with return periods up to $\sim$100-1000 years, including changes in intensity, spatial structure at km-scale and sub-daily temporal variability. Conventional km-scale climate models give greatly increased realism compared to typical global climate models (GCMs) but are too computationally costly to allow enough simulations for robustly quantifying changes in extremes. This work presents an approach based on video diffusion to generate realistic precipitation at 8.8km-hourly resolution by spatially and temporally downscaling daily, GCM-scale atmospheric variables. The emulator is trained to produce similar output to a $4\times$-coarsening of the Met Office's 2.2km-resolution UK convection-permitting model. An application for such an emulator is to downscale large-ensemble GCM simulations to allow quantification of $\sim$100-1000-year return level rainfall, which in turn can be used to drive hydrological and flood models for improved projections of future changes in flood risk.</span> <span class="abstract-toggle" data-id="2610.26145">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.26145v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.26145v1) · [:material-content-copy: BibTeX](bibtex/2610.26145.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=extremes" data-tag="extremes">Extremes</a> <a class="md-tag" href="/explore/?t=km-scale" data-tag="km-scale">Km-scale</a> <a class="md-tag" href="/explore/?t=hourly" data-tag="hourly">Hourly</a> <a class="md-tag" href="/explore/?t=daily" data-tag="daily">Daily</a>
    { .paper-tags }

-   #### AirMPA: A Meteorology-to-Pollution Adapter for Global Air Quality Forecasting { #2610.25797 }

    *Yiheng Wang, Kai Zheng, Yuetan Lin, Fanglu Fan, Chenliang Tao, Guochao Chen, Hongliang Zhang, Hao Li* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.25797">Accurate real-time forecasting of atmospheric pollutants is essential for reducing exposure risks and improving public health. Traditional numerical forecasting systems suffer from incomplete...</span><span class="abstract-full" id="full-2610.25797" hidden>Accurate real-time forecasting of atmospheric pollutants is essential for reducing exposure risks and improving public health. Traditional numerical forecasting systems suffer from incomplete parameterizations, uncertain inputs, and high computational cost. Existing deep learning approaches improve efficiency, but often entangle meteorological and pollutant variables within unified forecasting frameworks, limiting flexibility and pollutant-specific representation learning. To address these challenges, we propose AirMPA, a decoupled autoregressive framework for meteorology-to-pollution forecasting. AirMPA predicts 13 atmospheric pollutants at 0.4° spatial resolution and 12-h temporal resolution using meteorological fields, emission priors, and historical pollutant concentrations. The model is built on an improved Swin-Transformer architecture with zonal release attention and dual-branch fusion, enabling more effective modeling of global transport continuity and heterogeneous pollutant dynamics. During training, AirMPA uses the fifth-generation atmospheric reanalysis from the European Centre for Medium-Range Weather Forecasts (ERA5) as meteorological forcing, while during inference it directly ingests forecast fields from external meteorological models, enabling stable autoregressive prediction up to 5 days ahead. Experiments show that AirMPA consistently outperforms Aurora and surpasses the Copernicus Atmosphere Monitoring Service (CAMS) in approximately 90% of pollutant-lead-time evaluation cases over the 5-day forecast horizon, while outperforming CAMS on all variables beyond 48 h. These results demonstrate the potential of decoupled meteorology-to-pollution forecasting for global air quality prediction.</span> <span class="abstract-toggle" data-id="2610.25797">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.25797v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.25797v1) · [:material-content-copy: BibTeX](bibtex/2610.25797.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=transformers" data-tag="transformers">Transformers</a>
    { .paper-tags }

-   #### Zephyrus-R1: Agentic Reinforcement Learning for Weather Reporting { #2610.25625 }

    *Fengfei Yu, Marshall Fisher, Isadora White, Sumanth Varambally, Yasaman Jafari, Shivam Yadav et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.25625">Reinforcement learning (RL) has emerged as a key method for training AI models in scientific applications, but RL remains difficult when outputs cannot easily be verified against a known ground-truth...</span><span class="abstract-full" id="full-2610.25625" hidden>Reinforcement learning (RL) has emerged as a key method for training AI models in scientific applications, but RL remains difficult when outputs cannot easily be verified against a known ground-truth answer. Natural-language weather forecast reports are one such domain: they translate atmospheric data into assessments of weather evolution, hazards, and uncertainty for both the general public and key decision-makers. Unlike numerical weather-analysis questions, they admit many valid outputs, which limits deterministic verification. To address this, we develop Zephyrus-R1, a practical framework for RL on scientific weather forecast reporting. We focus on three key components of the agentic RL pipeline: reward design, curriculum learning, and an analysis of human-expert alignment. To encourage both report legibility and factual accuracy, we design a reward function that combines a meteorological rubric with bidirectional claim entailment, both evaluated against National Weather Service reference reports. Our framework optimizes this reward in a simulated tool-calling environment, enabling vision-language agents to gather and analyze forecast evidence across regions and lead times before producing a report. We use annotations from three meteorologists to audit the claim labels produced by our LLM judges. Finally, a curriculum helps small models first develop tool use and scientific reasoning through RL on verifiable weather-analysis questions, then optimize the open-ended report reward. After training Qwen3.5-4B with Zephyrus-R1, we improve the mean across six judge-metric scores by 26% over the base model and achieve results competitive with frontier open-weight models. For Qwen3.8-27B, we improve the same mean by 13% over the base model, by 6.7% over Kimi-K3, and by 2.7% over GPT-5.6-Luna.</span> <span class="abstract-toggle" data-id="2610.25625">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.25625v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.25625v1) · [:material-content-copy: BibTeX](bibtex/2610.25625.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=llms-agents" data-tag="llms-agents">LLMs & agents</a> <a class="md-tag" href="/explore/?t=reinforcement-learning" data-tag="reinforcement-learning">Reinforcement learning</a>
    { .paper-tags }

-   #### Beyond the record: evaluating probabilistic AI weather forecasts deep in the tail { #2610.25545 }

    *Bastiaan Van Velthoven, Cas Decancq, Diego G. Miralles, Stijn Luca, Thomas Mortier* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.25545">Can probabilistic artificial intelligence (AI) weather forecasts retain skill beyond their training domain? We compare four state-of-the-art AI models (ArchesWeatherGen, U-Cast, WeatherNext2 and...</span><span class="abstract-full" id="full-2610.25545" hidden>Can probabilistic artificial intelligence (AI) weather forecasts retain skill beyond their training domain? We compare four state-of-the-art AI models (ArchesWeatherGen, U-Cast, WeatherNext2 and GenCast-mini) with the physics-based IFS-ENS over land in 2022, at 1.5° resolution and lead times of 2--10 days. Monthly ERA5 records from 1979--2021 define common historical boundaries, and we characterize new records set in 2022 by their depth, i.e., the exceedance margin relative to local historical variability. Two proper scores, record-threshold weighted CRPS and Brier score, assess the magnitude and probability of record exceedances across all eligible cases, retaining both missed extremes and false alarms. Verified against ERA5, AI models outperform IFS-ENS for heat and cold records at almost every lead and depth. For heat, the advantage strengthens with depth, partly because IFS-ENS over-forecasts deep records, and largely persists for deeper records under HRES-fc0 verification. For wind, gains erode for deeper records and longer leads. Better tail scores can nevertheless coexist with aggregate probability deficits or surpluses, and some improvements arise chiefly on non-exceedances. Increasing underprediction on progressively more extreme observed subsets can also arise through outcome selection, even for an ideal conditional-mean forecast. These findings establish that native probabilistic AI ensembles retain, and for heat increase, their advantage over IFS-ENS as thresholds move beyond local records, although all advantages narrow with lead time. They caution against generalizing the extrapolation failure reported for deterministic AI forecasts to probabilistic systems, and motivate common verification conventions that separate comparative tail skill, performance on realized extremes, and probabilistic calibration.</span> <span class="abstract-toggle" data-id="2610.25545">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.25545v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.25545v1) · [:material-content-copy: BibTeX](bibtex/2610.25545.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=evaluation" data-tag="evaluation">Evaluation</a> <a class="md-tag" href="/explore/?t=coarse" data-tag="coarse">Coarse (≥1°)</a> <a class="md-tag" href="/explore/?t=monthly" data-tag="monthly">Monthly</a>
    { .paper-tags }

-   #### UtoMe: Observation-Uncertainty-Guided Token Merging for Weather Foundation Models { #2610.25001 }

    *Yunbei Zhang, Janet Wang, Xi Xiao, Jihun Hamm, Xiao Wang* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.25001">Weather transformers process dense reanalysis fields whose reliability varies across variables, locations, and weather regimes. Token merging can reduce computation, but feature similarity alone does...</span><span class="abstract-full" id="full-2610.25001" hidden>Weather transformers process dense reanalysis fields whose reliability varies across variables, locations, and weather regimes. Token merging can reduce computation, but feature similarity alone does not account for this heterogeneous input uncertainty. We introduce UtoMe, an observation-uncertainty-guided token merging framework that combines dynamics saliency with ensemble data assimilation spread as a relative input-reliability proxy. The resulting importance score protects selected tokens as anchors and calibrates bipartite matching between the remaining tokens. Three variants use external uncertainty at both training and inference, during training only, or neither stage. In ERA5 72-hour forecasting at two spatial resolutions, UtoMe improves Overall error over train-time ToMe at matched token budgets. At $1.40625^\circ$ and 50% token reduction, UtoMe-E lowers Overall error from 67.32 to 53.00 at the same 0.55x encoder FLOPs. Small reductions remain close to the full-token baseline; larger reductions incur measurable accuracy losses. Component ablations and spatial uncertainty controls support combining dynamics with aligned uncertainty when allocating a weather transformer's token budget.</span> <span class="abstract-toggle" data-id="2610.25001">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.25001v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.25001v1) · [:material-content-copy: BibTeX](bibtex/2610.25001.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=transformers" data-tag="transformers">Transformers</a> <a class="md-tag" href="/explore/?t=foundation-models" data-tag="foundation-models">Foundation models</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### Spherical Physics-informed Neural Operator with Multi-scale Coupling for Meteorological Downscaling { #2610.24708 }

    *Yiqiang Ye, Yichi Wang, Haodi Zhang, Jiawei Wen, Jiahui Jiang, Zhaoyu Zhong, Jiangjian Yu et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.24708">Meteorological downscaling is crucial for high-resolution regional climate forecasting and disaster early warning. While neural operators have emerged as a promising paradigm for modeling complex...</span><span class="abstract-full" id="full-2610.24708" hidden>Meteorological downscaling is crucial for high-resolution regional climate forecasting and disaster early warning. While neural operators have emerged as a promising paradigm for modeling complex spatiotemporal mappings, existing frameworks often struggle with spherical manifold geometric distortions, inherent atmospheric multi-scale coupling mismatches, and lack of explicit atmospheric laws. We propose the Spherical Physics-informed Neural Operator, which utilizes a Spherical Laplacian Decomposition to partition atmospheric fields into hierarchical frequency components, maintaining exact point-wise correspondence across scales. To evaluate these representations at arbitrary locations, we introduce a localized spherical integral operator that approximates continuous kernel transforms via geometry-aware attention. Dynamical consistency is further enforced by embedding differentiable constraints into the learning process. Extensive experiments demonstrate that our framework attains superior accuracy and zero-shot generalization across various meteorological variables and unseen queries, representing a robust and interpretable solution for global-to-regional meteorological downscaling.</span> <span class="abstract-toggle" data-id="2610.24708">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.24708v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.24708v1) · [:material-content-copy: BibTeX](bibtex/2610.24708.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=neural-operators" data-tag="neural-operators">Neural operators</a> <a class="md-tag" href="/explore/?t=physics-ml-hybrid" data-tag="physics-ml-hybrid">Physics–ML hybrid</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a>
    { .paper-tags }

-   #### Interpretability of AI weather models via intermediate decoding { #2610.24110 }

    *Matthias Beylich, Kirsten I. Tempest, Tobias Selz, George C. Craig* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.24110">AI weather prediction models are often regarded as black boxes. Despite their impressive skill, there is little understanding of how the input data is processed during inference to produce the...</span><span class="abstract-full" id="full-2610.24110" hidden>AI weather prediction models are often regarded as black boxes. Despite their impressive skill, there is little understanding of how the input data is processed during inference to produce the forecast. This limitation arises from the use of high-dimensional latent representations in which individual dimensions are no longer directly interpretable in physical terms. To gain insight into how forecasts are constructed, we propose the method of intermediate decoding for AI weather models and apply it to GraphCast, Aurora and ArchesWeather. In this approach, latent states from intermediate depths are decoded back into meteorological fields. To do this, we use the decoder that has been trained to output the forecast at the end of processing. Since latent states at different depths do not share a consistent basis, we train affine transformations that align intermediate latent states with the decoder's expected input space. Applying this framework to the three models yields the following results. First, the skill of forecasts at intermediate stages of processing improves gradually through the processor layers. Second, bypassing all processor blocks shows that the encoder-decoder pair is not an identity mapping returning the input, but instead plays a significant role in the forecast construction. Third, by examining increments of individual processor blocks across model depth, we show that all three models progressively refine the forecast from coarse to finer spatial scales. Finally, we assess the physical consistency of the intermediate decoding forecasts and find that large scales remain physically balanced throughout processing, while small scales become more balanced with each additional processor block.</span> <span class="abstract-toggle" data-id="2610.24110">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.24110v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.24110v1) · [:material-content-copy: BibTeX](bibtex/2610.24110.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=interpretability" data-tag="interpretability">Interpretability</a>
    { .paper-tags }

-   #### Energy-Aware Earth System Science: A Community Roadmap { #2610.23216 }

    *Ioana Colfescu, Annalisa Bracco, Kara Moraw, Chen Lu, Michael Bareford, Tom Beucler et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.23216">Earth system models underpin the evidence that informs net-zero policy, yet their energy demands are rarely considered when judging model quality. Using archived output from 159 models spanning five...</span><span class="abstract-full" id="full-2610.23216" hidden>Earth system models underpin the evidence that informs net-zero policy, yet their energy demands are rarely considered when judging model quality. Using archived output from 159 models spanning five generations of the Coupled Model Intercomparison Project (CMIP), we show that model error has fallen at a constant rate of 13.5 per cent per doubling of compute since 1995. Halving CMIP6 error would require 28 times more compute. Improvements are uneven, with sea-level pressure and surface air temperature errors declining 3.5-fold and 2.7-fold since CMIP1, but precipitation error by only 7 per cent. Machine-learning weather models match or exceed operational forecast skill at a fraction of the runtime, but none publishes measured energy, and including training changes how they rank. We propose a skill-energy framework that combines an empirical Pareto front with a ranking score. These findings argue for standardised reporting of energy use and performance ahead of CMIP8.</span> <span class="abstract-toggle" data-id="2610.23216">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.23216v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.23216v1) · [:material-content-copy: BibTeX](bibtex/2610.23216.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=evaluation" data-tag="evaluation">Evaluation</a> <a class="md-tag" href="/explore/?t=energy" data-tag="energy">Energy</a>
    { .paper-tags }

-   #### Network-Wide Freeway Speed Nowcasting and Forecasting from Sparse Fixed Sensors with a Probe-Supervised Graph Convolutional Recurrent Network { #2610.22630 }

    *Ran Tu, Zihao Li, Arash Mirzaei* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.22630">Traffic agencies need current and near-future speeds on every link of their network, but their fixed sensors cover only part of it. We study joint nowcasting and next-hour forecasting of hourly...</span><span class="abstract-full" id="full-2610.22630" hidden>Traffic agencies need current and near-future speeds on every link of their network, but their fixed sensors cover only part of it. We study joint nowcasting and next-hour forecasting of hourly speeds on all 4{,}127 directional freeway links of the Dallas--Fort Worth region from sensors on 643 links (15.6%). Historical probe-vehicle data supervise every link in training; only sensor readings are used at inference. We call this setting probe-supervised sparse sensing. We propose GCRN+, a graph convolutional recurrent network that predicts deviations from a historical speed profile. It combines a parameter-free, direction-aware kernel interpolation of sensor anomalies, used as input features and as an output baseline, with a diffusion-convolutional GRU on the directed road graph, a low-rank adaptive graph and learnable link embeddings. It scales linearly with network size and trains on a CPU. Sensor readings are emulated from probe speeds, which isolates spatio-temporal inference from detector calibration. On four held-out test days, GCRN+ attains an RMSE of 2.21~mph for the current hour and 2.39~mph for the next hour on unsensored links, against 3.23~mph for the historical average. The best baselines that use the same interpolation features reach 2.53 and 2.75~mph. Adapted recent models without these features (STID, STAEformer, ImputeFormer, MoGERNN) reach at best 2.54 and 2.53~mph; this comparison is between complete pipelines rather than architectures. GCRN+ is best in every distance band except links not connected to any sensor through the road graph. Ablations within one recurrent backbone attribute most of the gain to the link embeddings and the adaptive graph, which can be trained because probe labels exist for every link. For congestion detection on unsensored links, recall rises from 77% (historical average) and 86% (best competitor) to 89% at aprecision of 0.94.</span> <span class="abstract-toggle" data-id="2610.22630">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.22630v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.22630v1) · [:material-content-copy: BibTeX](bibtex/2610.22630.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=hourly" data-tag="hourly">Hourly</a>
    { .paper-tags }

-   #### Atmospheric Chemistry-Climate Modeling with an AI Foundation Model { #2610.22555 }

    *James Weber, Makoto Kelp* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.22555">Atmospheric composition and chemistry-climate interactions are central to predicting future climate change. Yet simulating chemical and aerosol processes is typically the most expensive component of...</span><span class="abstract-full" id="full-2610.22555" hidden>Atmospheric composition and chemistry-climate interactions are central to predicting future climate change. Yet simulating chemical and aerosol processes is typically the most expensive component of an atmospheric model and is often highly simplified or prescribed entirely. AI weather forecasting foundation models have shown skill rivalling process-based models and are increasingly fine-tuned for new atmospheric tasks. Here, we present the first foundation model to capture interactions between emissions, atmospheric composition, and radiation. We fine-tune the Aurora Earth system foundation model on simulations from the chemistry-climate model UKCA. Our model reproduces daily meteorology, composition, aerosol, and radiation fields of the coupled system with 23 of 27 predicted fields achieving $R^2$ above 0.85 in both the troposphere and stratosphere. It reproduces ozone, CO, and temperature with annual mean biases below 4% and the downwelling shortwave and longwave fluxes to within 6%. Our model remains stable in autoregressive simulation over 50 years while running ~$100\times$ faster than UKCA. By training on paired emissions scenarios, our model learns responses to perturbations by capturing tropospheric ozone's response to changes in NO and CO emissions and surface radiation's response to aerosol. We further show that incorporating MLS satellite ozone retrievals into training corrects biases in the underlying UKCA fields, reducing zonal mean ozone error against collocated observations by 23% between 250 and 20 hPa. Our approach does not suggest process-based modeling can be replaced but demonstrates that a foundation model can be adapted to respond to forcings and learn from observations at a relatively small computational cost.</span> <span class="abstract-toggle" data-id="2610.22555">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.22555v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.22555v1) · [:material-content-copy: BibTeX](bibtex/2610.22555.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=foundation-models" data-tag="foundation-models">Foundation models</a> <a class="md-tag" href="/explore/?t=daily" data-tag="daily">Daily</a>
    { .paper-tags }

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

</div>

