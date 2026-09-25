---
title: 'Post-processing'
hide:
  - toc
---

<div class="listing-header" markdown>

# Post-processing

<p class="page-meta" markdown="span">25 papers · <a href="../../bib/post-processing.bib" download>:material-download: BibTeX for this topic</a></p>

</div>

<div class="grid cards" markdown>

-   #### ClimTip-GML: A global bias-corrected and downscaled dataset for assessing impacts of climate tipping events { #2609.23149 }

    *Philipp Hess, Sebastian Bathiany, Lucas Ferreira Correa, Laura C. Jackson, Casey R. Patrizio et al.* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.23149">Assessing the impacts of future climate scenarios including tipping events of major Earth system components such as the Amazon rainforest (ARF) or the Atlantic meridional overturning circulation...</span><span class="abstract-full" id="full-2609.23149" hidden>Assessing the impacts of future climate scenarios including tipping events of major Earth system components such as the Amazon rainforest (ARF) or the Atlantic meridional overturning circulation (AMOC), requires accurate and high-resolution simulations. Here, we present ClimTip-GML, the first globally bias-corrected and downscaled climate dataset for impact assessment of large-scale tipping scenarios, comprising eight key variables at 0.25° spatial resolution from three general circulation models (GCMs): CESM1-CAM5, HadGEM3-GC31-MM, and MPI-ESM1-2-HR. The dataset includes 100-year-long climate simulations with preindustrial and historical conditions, as well as scenarios at a +2°C warming level with and without tipping transitions of the AMOC or ARF. We apply generative machine learning (GML) techniques trained on reanalysis data to bias-correct and downscale the GCMs in a manner that is physically consistent across space, time, and all eight variables. Comprehensive validation shows substantially reduced biases, improved small-scale spatial variability, multivariate correlations, and consistent long-term climate responses to the external forcing and tipping events. The results hence permit substantially improved impact assessments of tipping transitions of the ARF and AMOC, directly informing mitigation and adaptation policies.</span> <span class="abstract-toggle" data-id="2609.23149">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.23149v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.23149v1) · [:material-content-copy: BibTeX](../../bibtex/2609.23149.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a> <a class="md-tag" href="/explore/?t=quarter-degree" data-tag="quarter-degree">0.25°</a>
    { .paper-tags }

-   #### Statistical versus machine learning-based spatial interpolation of post-processed ensemble weather forecasts { #2609.07512 }

    *Mária Lakatos* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.07512">Statistical post-processing improves ensemble weather forecasts, but generating calibrated predictions at locations without observations remains challenging. This study compares statistical and...</span><span class="abstract-full" id="full-2609.07512" hidden>Statistical post-processing improves ensemble weather forecasts, but generating calibrated predictions at locations without observations remains challenging. This study compares statistical and machine-learning-based methods for post-processing ECMWF 2-m temperature and 10-m wind speed forecasts at observed and unobserved stations in Germany. We consider EMOS-based approaches, distributional regression networks, Transformers, and graph neural networks under both limited and extended predictor settings. For temperature, we also investigate linear forecast combinations and propose an altitude-aware linear pool (ALP). The results show that post-processing improves upon the raw ensemble in most settings, but no single method performs best across all variables, station groups, and evaluation metrics. The proposed ALP provides a small but significant improvement over the standard linear pool at unobserved locations.</span> <span class="abstract-toggle" data-id="2609.07512">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.07512v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.07512v1) · [:material-content-copy: BibTeX](../../bibtex/2609.07512.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=transformers" data-tag="transformers">Transformers</a> <a class="md-tag" href="/explore/?t=graph-neural-networks" data-tag="graph-neural-networks">Graph neural networks</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### PCSDiff: Diffusion-Based Bias Correction and Super Resolution Toward Practical Operational Medium-Term Precipitation Forecast { #2609.06942 }

    *Yuze Sun, Shiyi Wang, Jiancheng Pan, Die Wang, Andreas F. Prein, Wentao Luo, Linhan Jiang, Jie Wu et al.* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.06942">Medium-range precipitation forecasts are impaired by persistent systematic biases, lead-time-dependent error accumulation, and coarse spatial resolution, restricting their reliability for...</span><span class="abstract-full" id="full-2609.06942" hidden>Medium-range precipitation forecasts are impaired by persistent systematic biases, lead-time-dependent error accumulation, and coarse spatial resolution, restricting their reliability for flood-drought risk assessment. Existing AI correction techniques lack dedicated modeling for multi-day dynamic bias evolution and proper meteorological constraints, often generating over-smoothed rainfall structures, and cannot meet operational deployment demands. This work introduces PCSDiff, a cascaded task-decoupled diffusion framework targeting 10-day precipitation bias correction and downscaling. To jointly counteract temporal error drifts and reconstruct physically plausible local precipitation details, PCSDiff integrates the Precipitation Intensity-aware Multi-branch Decoder (PIMD) module for dynamic multi-day error mitigation using synoptic-temporal features, followed by a two-phase conditional diffusion super-resolution module to restore fine-scale precipitation patterns. Evaluated against CMA-CRA observations over China after global-data training, PCSDiff cuts RMSE by 16.1% and lifts ACC by 13.9% relative to raw ECMWF forecasts at 3-10-day lead times, and consistently outperforms mainstream deep-learning baselines on both general and extreme-precipitation metrics. Benefiting from a streaming inference pipeline, our method achieves low-latency rolling forecasting for practical meteorological operations.</span> <span class="abstract-toggle" data-id="2609.06942">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.06942v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.06942v1) · [:material-content-copy: BibTeX](../../bibtex/2609.06942.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=extremes" data-tag="extremes">Extremes</a>
    { .paper-tags }

-   #### A Differentiable Framework for Global Circulation Model Precipitation Bias Correction { #2604.23045 }

    *Kamlesh Sawadekar, Seth McGinnis, Peijun Li, Kathryn Lawson, Chaopeng Shen* · Apr 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2604.23045">Systematic biases in General Circulation Model (GCM) outputs limit their direct applicability in regional planning, making bias correction a technically demanding but necessary step for both...</span><span class="abstract-full" id="full-2604.23045" hidden>Systematic biases in General Circulation Model (GCM) outputs limit their direct applicability in regional planning, making bias correction a technically demanding but necessary step for both short-term and long-term impact assessment. Correcting precipitation is particularly challenging due to its non-Gaussian distribution, intermittent nature, and heavy-tailed extremes. However, traditional statistical bias-correction methods have limited ability to learn systematic patterns from large datasets or generalize to new locations. While machine learning (ML) provides greater flexibility, it can produce unpredictable and difficult-to-interpret results, limiting generalization across GCMs and locations. In this study, we propose a differentiable bias-adjustment framework called dCLIMBA, that learns a spatiotemporally adaptive parametric bias-adjustment procedure, rather than corrected precipitation directly, between historical CMIP6 model outputs and a gridded observation-based dataset, Livneh. Results demonstrate that the proposed method corrects the magnitude and distribution of extreme precipitation with particularly strong performance in the upper tail. The quantile distribution of precipitation was well reproduced across diverse U.S. cities, and spatial patterns were comparable to those from the widely used LOCA2 statistical downscaling product. In addition, the framework showed partial future trend preservation and promising attenuation of marginal biases in unseen regions. This work presents a modular and efficient bias-correction approach. The differentiable approach provides an easy-to-use option for connecting atmospheric-model outputs to on-the-ground impacts.</span> <span class="abstract-toggle" data-id="2604.23045">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2604.23045v3) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2604.23045v3) · [:material-content-copy: BibTeX](../../bibtex/2604.23045.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a>
    { .paper-tags }

-   #### Improvements to the post-processing of weather forecasts using machine learning and feature selection { #2604.19340 }

    *Kazuma Iwase, Tomoyuki Takenawa* · Apr 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2604.19340">This study aims to develop and improve machine learning-based post-processing models for precipitation, temperature, and wind speed predictions using the Mesoscale Model (MSM) dataset provided by the...</span><span class="abstract-full" id="full-2604.19340" hidden>This study aims to develop and improve machine learning-based post-processing models for precipitation, temperature, and wind speed predictions using the Mesoscale Model (MSM) dataset provided by the Japan Meteorological Agency (JMA) for 18 locations across Japan, including plains, mountainous regions, and islands. By incorporating meteorological variables from grid points surrounding the target locations as input features and applying feature selection based on correlation analysis, we found that, in our experimental setting, the LightGBM-based models achieved lower RMSE than the specific neural-network baselines tested in this study, including a reproduced CNN baseline, and also generally achieved lower RMSE than both the raw MSM forecasts and the JMA post-processing product, MSM Guidance (MSMG), across many locations and forecast lead times. Because precipitation has a highly skewed distribution with many zero cases, we additionally examined Tweedie-based loss functions and event-weighted training strategies for precipitation forecasting. These improved event-oriented performance relative to the original LightGBM model, especially at higher rainfall thresholds, although the gains were site dependent and overall performance remained slightly below MSMG.</span> <span class="abstract-toggle" data-id="2604.19340">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2604.19340v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2604.19340v1) · [:material-content-copy: BibTeX](../../bibtex/2604.19340.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=classical-ml" data-tag="classical-ml">Classical ML</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a>
    { .paper-tags }

-   #### MP-MoE: Matrix Profile-Guided Mixture of Experts for Precipitation Forecasting { #2603.25046 }

    *Huyen Ngoc Tran, Dung Trung Tran, Hong Nguyen, Xuan Vu Phan, Nam-Phong Nguyen* · Mar 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2603.25046">Precipitation forecasting remains a persistent challenge in tropical regions like Vietnam, where complex topography and convective instability often limit the accuracy of Numerical Weather Prediction...</span><span class="abstract-full" id="full-2603.25046" hidden>Precipitation forecasting remains a persistent challenge in tropical regions like Vietnam, where complex topography and convective instability often limit the accuracy of Numerical Weather Prediction (NWP) models. While data-driven post-processing is widely used to mitigate these biases, most existing frameworks rely on point-wise objective functions, which suffer from the “double penalty” effect under minor temporal misalignments. In this work, we propose the Matrix Profile-guided Mixture of Experts (MP-MoE), a framework that integrates conventional intensity loss with a structural-aware Matrix Profile objective. By leveraging subsequence-level similarity rather than point-wise errors, the proposed loss facilitates more reliable expert selection and mitigates excessive penalization caused by phase shifts. We evaluate MP-MoE on rainfall datasets from two major river basins in Vietnam across multiple horizons, including 1-hour intensity and accumulated rainfall over 12, 24, and 48 hours. Experimental results demonstrate that MP-MoE outperforms raw NWP and baseline learning methods in terms of Mean Critical Success Index (CSI-M) for heavy rainfall events, while significantly reducing Dynamic Time Warping (DTW) values. These findings highlight the framework's efficacy in capturing peak rainfall intensities and preserving the morphological integrity of storm events.</span> <span class="abstract-toggle" data-id="2603.25046">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2603.25046v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2603.25046v1) · [:material-content-copy: BibTeX](../../bibtex/2603.25046.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a>
    { .paper-tags }

-   #### HURRI-GAN: A Novel Approach for Hurricane Bias-Correction Beyond Gauge Stations using Generative Adversarial Networks { #2603.06649 }

    *Noujoud Nadera, Hadi Majed, Stefanos Giaremis, Rola El Osta, Clint Dawson, Carola Kaiser et al.* · Mar 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2603.06649">The coastal regions of the eastern and southern United States are impacted by severe storm events, leading to significant loss of life and properties. Accurately forecasting storm surge and wind...</span><span class="abstract-full" id="full-2603.06649" hidden>The coastal regions of the eastern and southern United States are impacted by severe storm events, leading to significant loss of life and properties. Accurately forecasting storm surge and wind impacts from hurricanes is essential for mitigating some of the impacts, e.g., timely preparation of evacuations and other countermeasures. Physical simulation models like the ADCIRC hydrodynamics model, which run on high-performance computing resources, are sophisticated tools that produce increasingly accurate forecasts as the resolution of the computational meshes improves. However, a major drawback of these models is the significant time required to generate results at very high resolutions, which may not meet the near real-time demands of emergency responders. The presented work introduces HURRI-GAN, a novel AI-driven approach that augments the results produced by physical simulation models using time series generative adversarial networks (TimeGAN) to compensate for systemic errors of the physical model, thus reducing the necessary mesh size and runtime without loss in forecasting accuracy. We present first results in extrapolating model bias corrections for the spatial regions beyond the positions of the water level gauge stations. The presented results show that our methodology can accurately generate bias corrections at target locations spatially beyond gauge stations locations. The model's performance, as indicated by low root mean squared error (RMSE) values, highlights its capability to generate accurate extrapolated data. Applying the corrections generated by HURRI-GAN on the ADCIRC modeled water levels resulted in improving the overall prediction on the majority of the testing gauge stations.</span> <span class="abstract-toggle" data-id="2603.06649">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2603.06649v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2603.06649v1) · [:material-content-copy: BibTeX](../../bibtex/2603.06649.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=gans" data-tag="gans">GANs</a> <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a> <a class="md-tag" href="/explore/?t=station-point" data-tag="station-point">Station / point</a>
    { .paper-tags }

-   #### Ensemble-size-dependence of deep-learning post-processing methods that minimize an (un)fair score: motivating examples and a proof-of-concept solution { #2602.15830 }

    *Christopher David Roberts* · Feb 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2602.15830">Fair scores reward ensemble forecast members that behave like samples from the same distribution as the verifying observations. They are therefore an attractive choice as loss functions to train...</span><span class="abstract-full" id="full-2602.15830" hidden>Fair scores reward ensemble forecast members that behave like samples from the same distribution as the verifying observations. They are therefore an attractive choice as loss functions to train data-driven ensemble forecasts or post-processing methods when large training ensembles are either unavailable or computationally prohibitive. The adjusted continuous ranked probability score (aCRPS) is fair and unbiased with respect to ensemble size, provided forecast members are exchangeable and interpretable as conditionally independent draws from an underlying predictive distribution. However, distribution-aware post-processing methods that introduce structural dependency between members can violate this assumption, rendering aCRPS unfair. We demonstrate this effect using two approaches designed to minimize the expected aCRPS of a finite ensemble: (1) a linear member-by-member calibration, which couples members through a common dependency on the sample ensemble mean, and (2) a deep-learning method, which couples members via transformer self-attention across the ensemble dimension. In both cases, the results are sensitive to ensemble size and apparent gains in aCRPS can correspond to systematic unreliability characterized by over-dispersion. We introduce trajectory transformers as a proof-of-concept that ensemble-size independence can be achieved. This approach is an adaptation of the Post-processing Ensembles with Transformers (PoET) framework and applies self-attention over lead time while preserving the conditional independence required by aCRPS. When applied to weekly mean $T_{2m}$ forecasts from the ECMWF subseasonal forecasting system, this approach successfully reduces systematic model biases whilst also improving or maintaining forecast reliability regardless of the ensemble size used in training (3 vs 9 members) or real-time forecasts (9 vs 100 members).</span> <span class="abstract-toggle" data-id="2602.15830">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2602.15830v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2602.15830v1) · [:material-content-copy: BibTeX](../../bibtex/2602.15830.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=transformers" data-tag="transformers">Transformers</a> <a class="md-tag" href="/explore/?t=subseasonal-to-seasonal" data-tag="subseasonal-to-seasonal">Subseasonal to seasonal</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### Probabilistic Wind Power Forecasting with Tree-Based Machine Learning and Weather Ensembles { #2602.13010 }

    *Max Bruninx, Diederik van Binsbergen, Timothy Verstraeten, Ann Nowé, Jan Helsen* · Feb 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2602.13010">Accurate production forecasts are essential for the integration of renewable energy sources into the power grid. This paper illustrates how to obtain probabilistic forecasts of wind power generation...</span><span class="abstract-full" id="full-2602.13010" hidden>Accurate production forecasts are essential for the integration of renewable energy sources into the power grid. This paper illustrates how to obtain probabilistic forecasts of wind power generation using gradient boosting trees and an ensemble of weather forecasts. To this end, we perform a comparative analysis across three state-of-the-art probabilistic prediction methods-conformalized quantile regression, natural gradient boosting and conditional diffusion models-all of which can be combined with tree-based machine learning. The methods are validated using four years of data for all Belgian offshore wind farms. We benchmark the models against the power curve and a calibrated wake model as well as a probabilistic method using stochastic variational Gaussian process regression. The tree-based models significantly reduce the mean absolute error in comparison to the deterministic baselines. Additionally, all three methods outperform the Gaussian process baseline in probabilistic skill, while two out of the three also improve point forecast accuracy. The conditional diffusion model attains the best performance, with improvements of 5% in mean absolute error and 12% in continuous rank probability score compared to the probabilistic baseline. Last, the results indicate an average improvement in point forecast accuracy of 17% by using an ensemble of weather forecasts instead of a single provider.</span> <span class="abstract-toggle" data-id="2602.13010">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2602.13010v2) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2602.13010v2) · [:material-content-copy: BibTeX](../../bibtex/2602.13010.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=classical-ml" data-tag="classical-ml">Classical ML</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=energy" data-tag="energy">Energy</a>
    { .paper-tags }

-   #### STIPP: Space-time in situ postprocessing over the French Alps using proper scoring rules { #2601.02882 }

    *David Landry, Isabelle Gouttevin, Hugo Merizen, Claire Monteleoni, Anastase Charantonis* · Jan 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2601.02882">We propose Space-time in situ postprocessing (STIPP), a machine learning model that generates spatio-temporally consistent weather forecasts for a network of station locations. Gridded forecasts from...</span><span class="abstract-full" id="full-2601.02882" hidden>We propose Space-time in situ postprocessing (STIPP), a machine learning model that generates spatio-temporally consistent weather forecasts for a network of station locations. Gridded forecasts from classical numerical weather prediction or data-driven models often lack the necessary precision due to unresolved local effects. Typical statistical postprocessing methods correct these biases, but often degrade spatio-temporal correlation structures in doing so. Recent works based on generative modeling successfully improve spatial correlation structures but have to forecast every lead time independently. In contrast, STIPP makes joint spatio-temporal forecasts which have increased accuracy for surface temperature, wind, relative humidity and precipitation when compared to baseline methods. It makes hourly ensemble predictions given only a six-hourly deterministic forecast, blending the boundaries of postprocessing and temporal interpolation. By leveraging a multivariate proper scoring rule for training, STIPP contributes to ongoing work data-driven atmospheric models supervised only with distribution marginals.</span> <span class="abstract-toggle" data-id="2601.02882">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2601.02882v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2601.02882v1) · [:material-content-copy: BibTeX](../../bibtex/2601.02882.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=station-point" data-tag="station-point">Station / point</a> <a class="md-tag" href="/explore/?t=hourly" data-tag="hourly">Hourly</a> <a class="md-tag" href="/explore/?t=6-hourly" data-tag="6-hourly">6-hourly</a>
    { .paper-tags }

-   #### Are we misdiagnosing ensemble forecast reliability? On the insufficiency of Spread-Error and rank-based reliability metrics { #2512.02160 }

    *Arlan Dirkson, Mark Buehner* · Dec 2025
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2512.02160">It has been documented that Spread-Error equality and a flat rank histogram are necessary but insufficient for demonstrating ensemble forecast reliability. Nevertheless, these metrics are heavily...</span><span class="abstract-full" id="full-2512.02160" hidden>It has been documented that Spread-Error equality and a flat rank histogram are necessary but insufficient for demonstrating ensemble forecast reliability. Nevertheless, these metrics are heavily relied upon, both in the literature and at operational numerical weather prediction centers. In this study, we demonstrate theoretically why the Spread-Error relationship is necessary but insufficient for diagnosing reliability up to second order, even when mean bias is absent or accounted for. Assuming joint normality between ensemble members and the reference truth, we further show with idealized experiments that the same covariance structure responsible for this insufficiency also produces false diagnoses of reliability with the rank histogram and the reliability component of the continuous rank probability score. Under this structure and when the ensemble mean is meaningfully different from climatology, the truth lies among the least (most) extreme members when climatological variance is excessive (deficient) in each member. Importantly, this behavior is also shown to be plausible in operational ensemble weather forecasts. Combining these results with calibration principles from statistical postprocessing leads us to conclude that both perfect dispersion and underdispersion are ill-defined. When diagnostics are misinterpreted as indicating the latter, improper tuning can lead to further deterioration of forecast quality, even while improving Spread-Error and rank histogram behavior. To address these issues, we propose a new reliability diagnostic based on three easily computed statistics, motivated by the structure of the joint distribution of ensemble members and the reference truth up to second order. The diagnostic separates contributions to unreliability originating from climatology and predictability, enabling a more precise and robust characterization of ensemble behavior.</span> <span class="abstract-toggle" data-id="2512.02160">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2512.02160v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2512.02160v1) · [:material-content-copy: BibTeX](../../bibtex/2512.02160.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### COBASE: A new copula-based shuffling method for ensemble weather forecast postprocessing { #2510.25610 }

    *Maurits Flos, Bastien François, Irene Schicker, Kirien Whan, Elisa Perrone* · Oct 2025
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2510.25610">Weather predictions are often provided as ensembles generated by repeated runs of numerical weather prediction models. These forecasts typically exhibit bias and inaccurate dependence structures due...</span><span class="abstract-full" id="full-2510.25610" hidden>Weather predictions are often provided as ensembles generated by repeated runs of numerical weather prediction models. These forecasts typically exhibit bias and inaccurate dependence structures due to numerical and dispersion errors, requiring statistical postprocessing for improved precision. A common correction strategy is the two-step approach: first adjusting the univariate forecasts, then reconstructing the multivariate dependence. The second step is usually handled with nonparametric methods, which can underperform when historical data are limited. Parametric alternatives, such as the Gaussian Copula Approach (GCA), offer theoretical advantages but often produce poorly calibrated multivariate forecasts due to random sampling of the corrected univariate margins. In this work, we introduce COBASE, a novel copula-based postprocessing framework that preserves the flexibility of parametric modeling while mimicking the nonparametric techniques through a rank-shuffling mechanism. This design ensures calibrated margins and realistic dependence reconstruction. We evaluate COBASE on multi-site 2-meter temperature forecasts from the ALADIN-LAEF ensemble over Austria and on joint forecasts of temperature and dew point temperature from the ECMWF system in the Netherlands. Across all regions, COBASE variants consistently outperform traditional copula-based approaches, such as GCA, and achieve performance on par with state-of-the-art nonparametric methods like SimSchaake and ECC, with only minimal differences across settings. These results position COBASE as a competitive and robust alternative for multivariate ensemble postprocessing, offering a principled bridge between parametric and nonparametric dependence reconstruction.</span> <span class="abstract-toggle" data-id="2510.25610">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2510.25610v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2510.25610v1) · [:material-content-copy: BibTeX](../../bibtex/2510.25610.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### A Review of Neural Networks in Precipitation Prediction { #2510.22855 }

    *Yugong Zeng, Jiayuan Wang, Jonathan Wu* · Oct 2025
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2510.22855">Precipitation prediction has undergone a profound transformation. A notable limitation of traditional NWP is the need for extensive statistical post-processing. To address this challenge, neural...</span><span class="abstract-full" id="full-2510.22855" hidden>Precipitation prediction has undergone a profound transformation. A notable limitation of traditional NWP is the need for extensive statistical post-processing. To address this challenge, neural network-based approaches were developed. These approaches offer a framework that directly learns the mapping from atmospheric predictors to precipitation targets. Based on the technological development, this article first reviews the traditional precipitation forecasting methods and summarizes the development trends of precipitation forecasting based on neural networks. We then outline the training process, loss functions, and some datasets for precipitation prediction. In the main body of the article, we detail the basic artificial neural networks (ANNs), spatial feature extraction models, time feature extraction models, generative models, Transformer models, graph neural networks (GNNs), and emerging hybrid models. Finally, in the appendix, we supplement the commonly used evaluation metrics. This paper focuses on the advantages and disadvantages of various neural network models in precipitation forecasting applications, and also pays attention to the latest progress of neural network-based methods. Overall, neural networks have significantly improved the accuracy of short-term and medium-term precipitation forecasting, but still face challenges in representing extreme rainfall, handling imbalanced data, and ensuring physical consistency. The latest progress shows that future prediction systems will increasingly rely on the integration of multiple sources of data and hybrid physical-data-driven models to enhance their robustness and applicability. By compositing research covering multiple eras and paradigms, we not only depict the history of neural networks in precipitation prediction but also outline future directions in next generation forecasting systems.</span> <span class="abstract-toggle" data-id="2510.22855">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2510.22855v2) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2510.22855v2) · [:material-content-copy: BibTeX](../../bibtex/2510.22855.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=transformers" data-tag="transformers">Transformers</a> <a class="md-tag" href="/explore/?t=graph-neural-networks" data-tag="graph-neural-networks">Graph neural networks</a> <a class="md-tag" href="/explore/?t=physics-ml-hybrid" data-tag="physics-ml-hybrid">Physics–ML hybrid</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a>
    { .paper-tags }

-   #### A Composite-Loss Graph Neural Network for the Multivariate Post-Processing of Ensemble Weather Forecasts { #2509.02784 }

    *Mária Lakatos* · Sep 2025
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2509.02784">Ensemble forecasting systems have advanced meteorology by providing probabilistic estimates of future states. Nonetheless, systematic biases often persist, making statistical post-processing...</span><span class="abstract-full" id="full-2509.02784" hidden>Ensemble forecasting systems have advanced meteorology by providing probabilistic estimates of future states. Nonetheless, systematic biases often persist, making statistical post-processing essential. Traditional parametric post-processing techniques and machine learning-based methods can produce calibrated predictive distributions at specific locations and lead times, yet often struggle to capture dependencies across forecast dimensions. To address this, multivariate post-processing methods-such as ensemble copula coupling and the Schaake shuffle-are widely applied in a second step to restore realistic inter-variable or spatio-temporal dependencies. The aim of this study is the multivariate post-processing of ensemble forecasts using a graph neural network (dualGNN) trained with a composite loss function that combines the energy score (ES) and the variogram score (VS). The method is evaluated on two datasets: WRF-based solar irradiance forecasts over northern Chile and ECMWF visibility forecasts for Central Europe. The dualGNN consistently outperforms all empirical copula-based post-processed forecasts and shows significant improvements compared to graph neural networks trained solely on either the continuous ranked probability score or the ES, according to the evaluated multivariate verification metrics. Furthermore, for the WRF forecasts, the rank-order structure of the dualGNN forecasts captures valuable dependency information, enabling a more effective restoration of spatial relationships than either the raw numerical weather prediction ensemble or historical observational rank structures. Notably, incorporating VS into the loss function improved the univariate performance for both target variables compared to training on ES alone. Moreover, for the visibility forecasts, the ES-VS combination even outperformed the strongest calibrated reference in terms of univariate performance.</span> <span class="abstract-toggle" data-id="2509.02784">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2509.02784v2) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2509.02784v2) · [:material-content-copy: BibTeX](../../bibtex/2509.02784.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=graph-neural-networks" data-tag="graph-neural-networks">Graph neural networks</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=energy" data-tag="energy">Energy</a>
    { .paper-tags }

-   #### How to systematically develop an effective AI-based bias correction model? { #2504.15322 }

    *Xiao Zhou, Yuze Sun, Jie Wu, Xiaomeng Huang* · Apr 2025
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2504.15322">This study introduces ReSA-ConvLSTM, an artificial intelligence (AI) framework for systematic bias correction in numerical weather prediction (NWP). We propose three innovations by integrating...</span><span class="abstract-full" id="full-2504.15322" hidden>This study introduces ReSA-ConvLSTM, an artificial intelligence (AI) framework for systematic bias correction in numerical weather prediction (NWP). We propose three innovations by integrating dynamic climatological normalization, ConvLSTM with temporal causality constraints, and residual self-attention mechanisms. The model establishes a physics-aware nonlinear mapping between ECMWF forecasts and ERA5 reanalysis data. Using 41 years (1981-2021) of global atmospheric data, the framework reduces systematic biases in 2-m air temperature (T2m), 10-m winds (U10/V10), and sea-level pressure (SLP), achieving up to 20% RMSE reduction over 1-7 day forecasts compared to operational ECMWF outputs. The lightweight architecture (10.6M parameters) enables efficient generalization to multiple variables and downstream applications, reducing retraining time by 85% for cross-variable correction while improving ocean model skill through bias-corrected boundary conditions. The ablation experiments demonstrate that our innovations significantly improve the model's correction performance, suggesting that incorporating variable characteristics into the model helps enhance forecasting skills.</span> <span class="abstract-toggle" data-id="2504.15322">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2504.15322v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2504.15322v1) · [:material-content-copy: BibTeX](../../bibtex/2504.15322.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=transformers" data-tag="transformers">Transformers</a> <a class="md-tag" href="/explore/?t=recurrent-networks" data-tag="recurrent-networks">Recurrent networks</a>
    { .paper-tags }

-   #### Statistical post-processing yields accurate probabilistic forecasts from Artificial Intelligence weather models { #2504.12672 }

    *Belinda Trotta, Robert Johnson, Catherine de Burgh-Day, Debra Hudson, Esteban Abellan, James Canvin et al.* · Apr 2025
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2504.12672">Artificial Intelligence (AI) weather models are now reaching operational-grade performance for some variables, but like traditional Numerical Weather Prediction (NWP) models, they exhibit systematic...</span><span class="abstract-full" id="full-2504.12672" hidden>Artificial Intelligence (AI) weather models are now reaching operational-grade performance for some variables, but like traditional Numerical Weather Prediction (NWP) models, they exhibit systematic biases and reliability issues. We test the application of the Bureau of Meteorology's existing statistical post-processing system, IMPROVER, to ECMWF's deterministic Artificial Intelligence Forecasting System (AIFS), and compare results against post-processed outputs from the ECMWF HRES and ENS models. Without any modification to processing workflows, post-processing yields comparable accuracy improvements for AIFS as for traditional NWP forecasts, in both expected value and probabilistic outputs. We show that blending AIFS with NWP models improves overall forecast skill, even when AIFS alone is not the most accurate component. These findings show that statistical post-processing methods developed for NWP are directly applicable to AI models, enabling national meteorological centres to incorporate AI forecasts into existing workflows in a low-risk, incremental fashion.</span> <span class="abstract-toggle" data-id="2504.12672">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2504.12672v2) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2504.12672v2) · [:material-content-copy: BibTeX](../../bibtex/2504.12672.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### Graph Neural Networks for Enhancing Ensemble Forecasts of Extreme Rainfall { #2504.05471 }

    *Christopher Bülte, Sohir Maskey, Philipp Scholl, Jonas von Berg, Gitta Kutyniok* · Apr 2025
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2504.05471">Climate change is increasing the occurrence of extreme precipitation events, threatening infrastructure, agriculture, and public safety. Ensemble prediction systems provide probabilistic forecasts...</span><span class="abstract-full" id="full-2504.05471" hidden>Climate change is increasing the occurrence of extreme precipitation events, threatening infrastructure, agriculture, and public safety. Ensemble prediction systems provide probabilistic forecasts but exhibit biases and difficulties in capturing extreme weather. While post-processing techniques aim to enhance forecast accuracy, they rarely focus on precipitation, which exhibits complex spatial dependencies and tail behavior. Our novel framework leverages graph neural networks to post-process ensemble forecasts, specifically modeling the extremes of the underlying distribution. This allows to capture spatial dependencies and improves forecast accuracy for extreme events, thus leading to more reliable forecasts and mitigating risks of extreme precipitation and flooding.</span> <span class="abstract-toggle" data-id="2504.05471">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2504.05471v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2504.05471v1) · [:material-content-copy: BibTeX](../../bibtex/2504.05471.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=graph-neural-networks" data-tag="graph-neural-networks">Graph neural networks</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=extremes" data-tag="extremes">Extremes</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### Generating ensembles of spatially-coherent in-situ forecasts using flow matching { #2504.03463 }

    *David Landry, Claire Monteleoni, Anastase Charantonis* · Apr 2025
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2504.03463">We propose a machine-learning-based methodology for in-situ weather forecast postprocessing that is both spatially coherent and multivariate. Compared to previous work, our Flow MAtching...</span><span class="abstract-full" id="full-2504.03463" hidden>We propose a machine-learning-based methodology for in-situ weather forecast postprocessing that is both spatially coherent and multivariate. Compared to previous work, our Flow MAtching Postprocessing (FMAP) better represents the correlation structures of the observations distribution, while also improving marginal performance at the stations. FMAP generates forecasts that are not bound to what is already modeled by the underlying gridded prediction and can infer new correlation structures from data. The resulting model can generate an arbitrary number of forecasts from a limited number of numerical simulations, allowing for low-cost forecasting systems. A single training is sufficient to perform postprocessing at multiple lead times, in contrast with other methods which use multiple trained networks at generation time. This work details our methodology, including a spatial attention transformer backbone trained within a flow matching generative modeling framework. FMAP shows promising performance in experiments on the EUPPBench dataset, forecasting surface temperature and wind gust values at station locations in western Europe up to five-day lead times.</span> <span class="abstract-toggle" data-id="2504.03463">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2504.03463v2) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2504.03463v2) · [:material-content-copy: BibTeX](../../bibtex/2504.03463.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=transformers" data-tag="transformers">Transformers</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=station-point" data-tag="station-point">Station / point</a>
    { .paper-tags }

-   #### Improving Predictions of Convective Storm Wind Gusts through Statistical Post-Processing of Neural Weather Models { #2504.00128 }

    *Antoine Leclerc, Erwan Koch, Monika Feldmann, Daniele Nerini, Tom Beucler* · Apr 2025
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2504.00128">Issuing timely severe weather warnings helps mitigate potentially disastrous consequences. Recent advancements in Neural Weather Models (NWMs) offer a computationally inexpensive and fast approach...</span><span class="abstract-full" id="full-2504.00128" hidden>Issuing timely severe weather warnings helps mitigate potentially disastrous consequences. Recent advancements in Neural Weather Models (NWMs) offer a computationally inexpensive and fast approach for forecasting atmospheric environments on a 0.25° global grid. For thunderstorms, these environments can be empirically post-processed to predict wind gust distributions at specific locations. With the Pangu-Weather NWM, we apply a hierarchy of statistical and deep learning post-processing methods to forecast hourly wind gusts up to three days ahead. To ensure statistical robustness, we constrain our probabilistic forecasts using generalised extreme-value distributions across five regions in Switzerland. Using a convolutional neural network to post-process the predicted atmospheric environment's spatial patterns yields the best results, outperforming direct forecasting approaches across lead times and wind gust speeds. Our results confirm the added value of NWMs for extreme wind forecasting, especially for designing more responsive early-warning systems.</span> <span class="abstract-toggle" data-id="2504.00128">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2504.00128v3) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2504.00128v3) · [:material-content-copy: BibTeX](../../bibtex/2504.00128.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=extremes" data-tag="extremes">Extremes</a> <a class="md-tag" href="/explore/?t=quarter-degree" data-tag="quarter-degree">0.25°</a> <a class="md-tag" href="/explore/?t=hourly" data-tag="hourly">Hourly</a>
    { .paper-tags }

-   #### Self-attentive Transformer for Fast and Accurate Postprocessing of Temperature and Wind Speed Forecasts { #2412.13957 }

    *Aaron Van Poecke, Tobias Sebastian Finn, Ruoke Meng, Joris Van den Bergh, Geert Smet et al.* · Dec 2024
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2412.13957">Current postprocessing techniques often require separate models for each lead time and disregard possible inter-ensemble relationships by either correcting each member separately or by employing...</span><span class="abstract-full" id="full-2412.13957" hidden>Current postprocessing techniques often require separate models for each lead time and disregard possible inter-ensemble relationships by either correcting each member separately or by employing distributional approaches. In this work, we tackle these shortcomings with an innovative, fast and accurate Transformer which postprocesses each ensemble member individually while allowing information exchange across variables, spatial dimensions and lead times by means of multi-headed self-attention. Weather forecasts are postprocessed over 20 lead times simultaneously while including up to fifteen meteorological predictors. We use the EUPPBench dataset for training which contains ensemble predictions from the European Center for Medium-range Weather Forecasts' integrated forecasting system alongside corresponding observations. The work presented here is the first to postprocess the ten and one hundred-meter wind speed forecasts within this benchmark dataset, while also correcting two-meter temperature. Our approach significantly improves the original forecasts, as measured by the CRPS, with 16.5% for two-meter temperature, 10% for ten-meter wind speed and 9% for one hundred-meter wind speed, outperforming a classical member-by-member approach employed as a competitive benchmark. Furthermore, being up to six times faster, it fulfills the demand for rapid operational weather forecasts in various downstream applications, including renewable energy forecasting.</span> <span class="abstract-toggle" data-id="2412.13957">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2412.13957v2) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2412.13957v2) · [:material-content-copy: BibTeX](../../bibtex/2412.13957.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=transformers" data-tag="transformers">Transformers</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=energy" data-tag="energy">Energy</a>
    { .paper-tags }

-   #### Boosting weather forecast via generative superensemble { #2412.08377 }

    *Congyi Nai, Xi Chen, Shangshang Yang, Yuan Liang, Ziniu Xiao, Baoxiang Pan* · Dec 2024
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2412.08377">Accurate weather forecasting is essential for socioeconomic activities. While data-driven forecasting demonstrates superior predictive capabilities over traditional Numerical Weather Prediction (NWP)...</span><span class="abstract-full" id="full-2412.08377" hidden>Accurate weather forecasting is essential for socioeconomic activities. While data-driven forecasting demonstrates superior predictive capabilities over traditional Numerical Weather Prediction (NWP) with reduced computational demands, its deterministic nature and limited advantages over physics-based ensemble predictions restrict operational applications. We introduce the generative ensemble prediction system (GenEPS) framework to address these limitations by randomizing and mitigating both random errors and systematic biases. GenEPS provides a plug-and-play ensemble forecasting capability for deterministic models to eliminate random errors, while incorporating cross-model integration for cross-model ensembles to address systematic biases. The framework culminates in a super-ensemble approach utilizing all available data-driven models to further minimize systematic biases. GenEPS achieves an Anomaly Correlation Coefficient (ACC) of 0.679 for 500hPa geopotential (Z500), exceeding the ECMWF Ensemble Prediction System's (ENS) ACC of 0.646. Integration of the ECMWF ensemble mean further improves the ACC to 0.683. The framework also enhances extreme event representation and produces energy spectra more consistent with ERA5 reanalysis. GenEPS establishes a new paradigm in ensemble forecasting by enabling the integration of multiple data-driven models into a high-performing super-ensemble system.</span> <span class="abstract-toggle" data-id="2412.08377">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2412.08377v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2412.08377v1) · [:material-content-copy: BibTeX](../../bibtex/2412.08377.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### Self-Supervised Learning with Probabilistic Density Labeling for Rainfall Probability Estimation { #2412.05825 }

    *Junha Lee, Sojung An, Sujeong You, Namik Cho* · Dec 2024
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2412.05825">Numerical weather prediction (NWP) models are fundamental in meteorology for simulating and forecasting the behavior of various atmospheric variables. The accuracy of precipitation forecasts and the...</span><span class="abstract-full" id="full-2412.05825" hidden>Numerical weather prediction (NWP) models are fundamental in meteorology for simulating and forecasting the behavior of various atmospheric variables. The accuracy of precipitation forecasts and the acquisition of sufficient lead time are crucial for preventing hazardous weather events. However, the performance of NWP models is limited by the nonlinear and unpredictable patterns of extreme weather phenomena driven by temporal dynamics. In this regard, we propose a <strong>S</strong>elf-<strong>S</strong>upervised <strong>L</strong>earning with <strong>P</strong>robabilistic <strong>D</strong>ensity <strong>L</strong>abeling (SSLPDL) for estimating rainfall probability by post-processing NWP forecasts. Our post-processing method uses self-supervised learning (SSL) with masked modeling for reconstructing atmospheric physics variables, enabling the model to learn the dependency between variables. The pre-trained encoder is then utilized in transfer learning to a precipitation segmentation task. Furthermore, we introduce a straightforward labeling approach based on probability density to address the class imbalance in extreme weather phenomena like heavy rain events. Experimental results show that SSLPDL surpasses other precipitation forecasting models in regional precipitation post-processing and demonstrates competitive performance in extending forecast lead times. Our code is available at https://github.com/joonha425/SSLPDL</span> <span class="abstract-toggle" data-id="2412.05825">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2412.05825v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2412.05825v1) · [:fontawesome-brands-github: Code](https://github.com/joonha425/SSLPDL) · [:material-content-copy: BibTeX](../../bibtex/2412.05825.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=extremes" data-tag="extremes">Extremes</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### Machine learning models for daily rainfall forecasting in Northern Tropical Africa using tropical wave predictors { #2408.16349 }

    *Athul Rasheeda Satheesh, Peter Knippertz, Andreas H. Fink* · Aug 2024
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2408.16349">Numerical weather prediction (NWP) models often underperform compared to simpler climatology-based precipitation forecasts in northern tropical Africa, even after statistical postprocessing. AI-based...</span><span class="abstract-full" id="full-2408.16349" hidden>Numerical weather prediction (NWP) models often underperform compared to simpler climatology-based precipitation forecasts in northern tropical Africa, even after statistical postprocessing. AI-based forecasting models show promise but have avoided precipitation due to its complexity. Synoptic-scale forcings like African easterly waves and other tropical waves (TWs) are important for predictability in tropical Africa, yet their value for predicting daily rainfall remains unexplored. This study uses two machine-learning models--gamma regression and a convolutional neural network (CNN)--trained on TW predictors from satellite-based GPM IMERG data to predict daily rainfall during the July-September monsoon season. Predictor variables are derived from the local amplitude and phase information of seven TW from the target and up-and-downstream neighboring grids at 1-degree spatial resolution. The ML models are combined with Easy Uncertainty Quantification (EasyUQ) to generate calibrated probabilistic forecasts and are compared with three benchmarks: Extended Probabilistic Climatology (EPC15), ECMWF operational ensemble forecast (ENS), and a probabilistic forecast from the ENS control member using EasyUQ (CTRL EasyUQ). The study finds that downstream predictor variables offer the highest predictability, with downstream tropical depression (TD)-type wave-based predictors being most important. Other waves like mixed-Rossby gravity (MRG), Kelvin, and inertio-gravity waves also contribute significantly but show regional preferences. ENS forecasts exhibit poor skill due to miscalibration. CTRL EasyUQ shows improvement over ENS and marginal enhancement over EPC15. Both gamma regression and CNN forecasts significantly outperform benchmarks in tropical Africa. This study highlights the potential of ML models trained on TW-based predictors to improve daily precipitation forecasts in tropical Africa.</span> <span class="abstract-toggle" data-id="2408.16349">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2408.16349v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2408.16349v1) · [:material-content-copy: BibTeX](../../bibtex/2408.16349.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=coarse" data-tag="coarse">Coarse (≥1°)</a> <a class="md-tag" href="/explore/?t=daily" data-tag="daily">Daily</a>
    { .paper-tags }

-   #### Graph Neural Networks and Spatial Information Learning for Post-Processing Ensemble Weather Forecasts { #2407.11050 }

    *Moritz Feik, Sebastian Lerch, Jan Stühmer* · Jul 2024
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2407.11050">Ensemble forecasts from numerical weather prediction models show systematic errors that require correction via post-processing. While there has been substantial progress in flexible neural...</span><span class="abstract-full" id="full-2407.11050" hidden>Ensemble forecasts from numerical weather prediction models show systematic errors that require correction via post-processing. While there has been substantial progress in flexible neural network-based post-processing methods over the past years, most station-based approaches still treat every input data point separately which limits the capabilities for leveraging spatial structures in the forecast errors. In order to improve information sharing across locations, we propose a graph neural network architecture for ensemble post-processing, which represents the station locations as nodes on a graph and utilizes an attention mechanism to identify relevant predictive information from neighboring locations. In a case study on 2-m temperature forecasts over Europe, the graph neural network model shows substantial improvements over a highly competitive neural network-based post-processing method.</span> <span class="abstract-toggle" data-id="2407.11050">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2407.11050v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2407.11050v1) · [:material-content-copy: BibTeX](../../bibtex/2407.11050.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=graph-neural-networks" data-tag="graph-neural-networks">Graph neural networks</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a>
    { .paper-tags }

-   #### Improving ensemble extreme precipitation forecasts using generative artificial intelligence { #2407.04882 }

    *Yingkai Sha, Ryan A. Sobash, David John Gagne* · Jul 2024
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2407.04882">An ensemble post-processing method is developed to improve the probabilistic forecasts of extreme precipitation events across the conterminous United States (CONUS). The method combines a 3-D Vision...</span><span class="abstract-full" id="full-2407.04882" hidden>An ensemble post-processing method is developed to improve the probabilistic forecasts of extreme precipitation events across the conterminous United States (CONUS). The method combines a 3-D Vision Transformer (ViT) for bias correction with a Latent Diffusion Model (LDM), a generative Artificial Intelligence (AI) method, to post-process 6-hourly precipitation ensemble forecasts and produce an enlarged generative ensemble that contains spatiotemporally consistent precipitation trajectories. These trajectories are expected to improve the characterization of extreme precipitation events and offer skillful multi-day accumulated and 6-hourly precipitation guidance. The method is tested using the Global Ensemble Forecast System (GEFS) precipitation forecasts out to day 6 and is verified against the Climate-Calibrated Precipitation Analysis (CCPA) data. Verification results indicate that the method generated skillful ensemble members with improved Continuous Ranked Probabilistic Skill Scores (CRPSSs) and Brier Skill Scores (BSSs) over the raw operational GEFS and a multivariate statistical post-processing baseline. It showed skillful and reliable probabilities for events at extreme precipitation thresholds. Explainability studies were further conducted, which revealed the decision-making process of the method and confirmed its effectiveness on ensemble member generation. This work introduces a novel, generative-AI-based approach to address the limitation of small numerical ensembles and the need for larger ensembles to identify extreme precipitation events.</span> <span class="abstract-toggle" data-id="2407.04882">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2407.04882v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2407.04882v1) · [:material-content-copy: BibTeX](../../bibtex/2407.04882.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=transformers" data-tag="transformers">Transformers</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=extremes" data-tag="extremes">Extremes</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a> <a class="md-tag" href="/explore/?t=6-hourly" data-tag="6-hourly">6-hourly</a>
    { .paper-tags }

</div>

