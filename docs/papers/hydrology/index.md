---
title: 'Hydrology'
hide:
  - toc
---

<div class="listing-header" markdown>

# Hydrology

<p class="page-meta" markdown="span">37 papers · page 1 of 2 · <a href="../../bib/hydrology.bib" download>:material-download: BibTeX for this topic</a></p>

</div>

<div class="grid cards" markdown>

-   #### Rainfall Sensing via Mobile Communication Signals { #2608.16088 }

    *Zhongqin Wang, J. Andrew Zhang, Kai Wu, Y. Jay Guo* · Aug 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2608.16088">Rainfall monitoring is important for hydrological observation, disaster warning, and environmental sensing, but conventional rain gauges and weather radars suffer from sparse deployment and high...</span><span class="abstract-full" id="full-2608.16088" hidden>Rainfall monitoring is important for hydrological observation, disaster warning, and environmental sensing, but conventional rain gauges and weather radars suffer from sparse deployment and high infrastructure costs. This paper proposes PMN-RainSense, a rainfall sensing framework using sub-6-GHz mobile communication signals that supports practical single-antenna deployment. Unlike attenuation-based approaches, which are unreliable at sub-6 GHz because rain-induced attenuation over short mobile access links is only on the order of hundredths of a decibel, the proposed framework exploits fine-grained dynamics. A spectral-temporal channel state information (CSI) compensation method suppresses packet-wise timing and phase distortions while preserving sensing-relevant information. Rainfall-sensitive features are extracted from the delay-Doppler domain to mitigate environmental interference, with angle-domain filtering as an optional extension for multi-antenna receivers. Under bandwidth and antenna constraints, rainfall-correlated Doppler fluctuations serve as the dominant sensing signature, while Doppler-domain normalization improves robustness across links and deployments. Controlled WiFi experiments demonstrate rainfall-associated Doppler broadening and achieve a three-class classification accuracy of 95.48% using a random forest classifier. Long-Term Evolution (LTE) CSI measurements collected from cellular base stations over 11 carrier frequencies from 0.763 to 2.68 GHz yield a mean absolute error (MAE) of 0.25-0.27 mm/h for rainfall intensity estimation using a one-dimensional convolutional network.</span> <span class="abstract-toggle" data-id="2608.16088">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2608.16088v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2608.16088v1) · [:material-content-copy: BibTeX](../../bibtex/2608.16088.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a> <a class="md-tag" href="/explore/?t=classical-ml" data-tag="classical-ml">Classical ML</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a>
    { .paper-tags }

-   #### From Conceptual Hydrologic Models to Conceptually Interpretable Neural Networks: A Snow-Water Mass-Conserving-Perceptron Framework for Discovering Catchment-Scale Precipitation-Storage-Runoff Representations { #2607.26492 }

    *Yuan-Heng Wang, Hoshin V. Gupta* · Jul 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2607.26492">The Mass-Conserving Perceptron (MCP) establishes a modeling paradigm in which conceptual hydrologic models can be reformulated as physically constrained, conceptually interpretable neural networks....</span><span class="abstract-full" id="full-2607.26492" hidden>The Mass-Conserving Perceptron (MCP) establishes a modeling paradigm in which conceptual hydrologic models can be reformulated as physically constrained, conceptually interpretable neural networks. Here, we develop a snow-water MCP network framework and evaluate it across 513 CAMELS-US basins. We first recast a coupled two-state SOIL-MCP and SNOWMCP conceptual model as a mass-conserving neural network and show that the hydrologic-model and neural-network formulations achieve comparable predictive performance. We then examine cross-node state-information sharing within two-state HYDROMCP architectures and evaluate broader single-layer networks constructed from three types of interpretable MCP units with one to five states. Across CONUS, the median KGEss increases from 0.82 for one-state networks to 0.89 for two-state networks and 0.90 for five-state networks, suggesting diminishing aggregate gains beyond two states. Basin-specific MCP and LSTM selection yields the same median KGEss of 0.90, while the selected MCP networks use fewer parameters on average. Complementary AIC- and KGE-based selection identifies compact, basin-specific directed-graph representations that balance predictive accuracy and model complexity. These analyses provide an empirical basis for identifying the numbers, types, and interactions of states needed for hydrologic representation. Future studies should test joint training against multiple hydrologic responses, such as streamflow, snow water equivalent, and groundwater storage.</span> <span class="abstract-toggle" data-id="2607.26492">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2607.26492v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2607.26492v1) · [:material-content-copy: BibTeX](../../bibtex/2607.26492.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=recurrent-networks" data-tag="recurrent-networks">Recurrent networks</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=interpretability" data-tag="interpretability">Interpretability</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a>
    { .paper-tags }

-   #### Exploratory Analysis of Deep Learning Models for Forecasting Meteorological Parameters in the Agricultural Sector { #2607.10208 }

    *Piotr Sikora, Sotirios Kontogiannis* · Jul 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2607.10208">Accurate meteorological forecasting is essential for agricultural planning, irrigation management, and environmental decision support. This study conducts a comparative evaluation of recurrent and...</span><span class="abstract-full" id="full-2607.10208" hidden>Accurate meteorological forecasting is essential for agricultural planning, irrigation management, and environmental decision support. This study conducts a comparative evaluation of recurrent and hybrid deep learning architectures for multivariate forecasting of reference evapotranspiration ($ET_0$), vapour pressure deficit (VPD), wind speed, and the sine and cosine components of wind direction. The analysis utilizes 134,376 hourly observations from Ioannina, Greece, spanning January 2011 to April 2026, sourced from ERA5 via the OpenMeteo Historical Weather API. Single and multi-layer GRU and LSTM networks are compared with hybrid 1D-CNN-GRU and 1D-CNN-LSTM models for two forecasting tasks: a 24-hour next-day forecast and a 168-hour week-ahead forecast. Performance is evaluated using normalized root mean squared error, the coefficient of determination, and a composite Weighted Quotient Score (WQS). The most effective purely recurrent models are a 64-unit LSTM for the 24-hour horizon, with a WQS of 0.816755, and a 1024-unit GRU for the 168-hour horizon, with a WQS of 0.779465. The hybrid CNN-GRU models achieved the highest overall scores of 0.827535 and 0.782863 for the 24-hour and 168-hour horizons, but with additionally more number of units respectively to LSTM models, while the CNN-LSTM models yield nearly identical results with substantially fewer parameters. Compared to the corresponding recurrent baselines, the hybrid models improve WQS by 1.22--1.63% at 24 hours and by 0.44--0.45% at 168 hours, indicating that convolutional feature extraction is more beneficial for short-term forecasting.</span> <span class="abstract-toggle" data-id="2607.10208">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2607.10208v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2607.10208v1) · [:material-content-copy: BibTeX](../../bibtex/2607.10208.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a> <a class="md-tag" href="/explore/?t=recurrent-networks" data-tag="recurrent-networks">Recurrent networks</a> <a class="md-tag" href="/explore/?t=hourly" data-tag="hourly">Hourly</a>
    { .paper-tags }

-   #### A harmonised dataset for Earth system foundation models { #2607.03298 }

    *Carlos Rodriguez-Pardo, Massimo Tavoni* · Jul 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2607.03298">Foundation models for Earth systems have so far been trained primarily on physical climate and weather data, with limited representation of the human systems that both drive and respond to...</span><span class="abstract-full" id="full-2607.03298" hidden>Foundation models for Earth systems have so far been trained primarily on physical climate and weather data, with limited representation of the human systems that both drive and respond to environmental change. The lack of a unified global training resource that combines climate, land, ocean, cryosphere, infrastructure, hazards, and socioeconomic data on a common grid hinders progress toward truly multimodal Earth system foundation models. We present WorldTensor, a harmonised global dataset that aligns hundreds of environmental and socioeconomic variables to a standardised 0.25$^\circ$ spatial grid and annual temporal framework. WorldTensor integrates reanalysis products, remote sensing, emissions inventories, land use reconstructions, hydrological observations, infrastructure and hazard datasets, and socioeconomic indicators within a single representation designed for machine learning workflows. To build the dataset, we regridded inputs across heterogeneous native resolutions and projections, rasterised point and vector datasets into spatially meaningful gridded fields, and reconciled temporal coverages ranging from daily observations to sparse multiyear socioeconomic snapshots. All outputs are distributed as NetCDF files with standardised coordinates, variable metadata, and a common CF metadata convention. WorldTensor provides a reproducible resource for training and evaluating foundation models that learn coupled dynamics across environmental and human systems at planetary scale.</span> <span class="abstract-toggle" data-id="2607.03298">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2607.03298v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2607.03298v1) · [:material-content-copy: BibTeX](../../bibtex/2607.03298.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=foundation-models" data-tag="foundation-models">Foundation models</a> <a class="md-tag" href="/explore/?t=benchmarks-datasets" data-tag="benchmarks-datasets">Benchmarks & datasets</a> <a class="md-tag" href="/explore/?t=quarter-degree" data-tag="quarter-degree">0.25°</a> <a class="md-tag" href="/explore/?t=daily" data-tag="daily">Daily</a>
    { .paper-tags }

-   #### Deep Learning for Soil Moisture Estimation: Fusing Satellite Data with Optimally-Lagged Meteorological Features { #2606.21475 }

    *Adrian Canovas-Rodriguez, Aurora González Vidal, Antonio F. Skarmeta* · Jun 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2606.21475">Accurate soil moisture estimation in semi-arid agricultural regions requires integrating remote sensing and meteorological information while accounting for the delayed response of soil moisture to...</span><span class="abstract-full" id="full-2606.21475" hidden>Accurate soil moisture estimation in semi-arid agricultural regions requires integrating remote sensing and meteorological information while accounting for the delayed response of soil moisture to atmospheric forcing. This study introduces a Cross-Correlation Function (CCF) methodology to determine optimal temporal lags (0-30 days) between meteorological variables and soil moisture, as well as inter-depth lags (0-15 days) describing vertical moisture propagation from the surface (10 cm) to deeper layers (20-50 cm). The approach was validated across seven agricultural plots in southeastern Spain. Three deep learning architectures, each targeting a distinct prediction granularity, were evaluated under five feature configurations ranging from satellite-only to full satellite-meteorology-depth fusion: a CNN for per-pixel estimation within each plot, an LSTM for frame-level (daily plot-mean) prediction, and a CNN-LSTM hybrid operating on sliding windows with pooled multi-patch training. Models were assessed on held-out data to measure genuine generalisation. Meteorological variables improved performance over the satellite-only baseline, while subsurface depth information proved decisive across all architectures. The per-pixel CNN achieved the strongest single-patch result (R^2 = 0.877, RMSE = 2.28), with a seven-patch average R^2 of 0.535, representing an improvement of +1.00 over the satellite-only baseline. The pooled CNN-LSTM hybrid obtained the highest overall performance (R^2 = 0.930, CVRMSE = 8.0%). These results demonstrate that explicitly modelling atmospheric and vertical subsurface delays substantially improves soil moisture estimation for precision agriculture.</span> <span class="abstract-toggle" data-id="2606.21475">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2606.21475v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2606.21475v1) · [:material-content-copy: BibTeX](../../bibtex/2606.21475.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a> <a class="md-tag" href="/explore/?t=recurrent-networks" data-tag="recurrent-networks">Recurrent networks</a> <a class="md-tag" href="/explore/?t=daily" data-tag="daily">Daily</a>
    { .paper-tags }

-   #### Interpretable rainfall modelling reveals rapid reorganisation of Amazonian rainfall under vegetation loss { #2605.10948 }

    *Lilly Horvath-Makkos, Fayyaz Minhas* · May 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2605.10948">Understanding how vegetation loss alters rainfall remains a major challenge in climate and hydrological science, as deforestation modifies precipitation through heterogeneous, seasonal and nonlinear...</span><span class="abstract-full" id="full-2605.10948" hidden>Understanding how vegetation loss alters rainfall remains a major challenge in climate and hydrological science, as deforestation modifies precipitation through heterogeneous, seasonal and nonlinear land-atmosphere feedbacks. Existing models struggle to capture these dynamics: convection is parameterised at coarse scales, tipping behaviour is poorly constrained, and rainfall-deforestation analyses are limited to multi-decadal timescales. Therefore, many approaches resolve correlations rather than causal effects, limiting our ability to anticipate hydrological disruption. Using a neural-network model for hourly rainfall prediction, combined with pathway diagnostics and sensitivity analyses, we examine how vegetation perturbations reorganise rainfall across space, intensity regimes, and timescales under deforestation. We assess whether the model captures physically consistent dependencies linking vegetation, atmospheric state, and precipitation, and whether sustained canopy loss induces threshold behaviour. The model accurately predicts rainfall occurrence and intensity (Spearman = 0.84, F1 = 0.93, ROC-AUC = 0.98) and learns temporally ordered dependencies aligned with ecohydrological theory. Sensitivity analyses reveal rapid, asymmetric responses to vegetation loss: heavy rainfall (20-50 mm/h) declines by up to 7% under sustained deforestation, while light rainfall (0.1-1 mm/h) increases by 4%. Rainfall entropy rises by 1.3%, and dry-season intensity increases by 0.3-0.5% per 0.5% forest-cover loss, with strongest impacts in the north-western Amazon and Andean foothills. Threshold analysis reveals a sharp decline in precipitating area fraction after 2-3 months of sustained vegetation change in sensitive regions. These results demonstrate that data-driven approaches uncover process-relevant land-atmosphere coupling and highlight growing hydrological vulnerability in the Amazon.</span> <span class="abstract-toggle" data-id="2605.10948">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2605.10948v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2605.10948v1) · [:material-content-copy: BibTeX](../../bibtex/2605.10948.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=interpretability" data-tag="interpretability">Interpretability</a> <a class="md-tag" href="/explore/?t=hourly" data-tag="hourly">Hourly</a>
    { .paper-tags }

-   #### METBRA25Y: Brazil Surface Meteorology Archive with Harmonized Variables and Quality Control { #2605.08701 }

    *Matheus Lima Castro, William Dantas Vichete, Leopoldo Lusquino Filho* · May 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2605.08701">This data paper describes METBRA25Y, a harmonized archive of hourly surface meteorological observations from Brazil derived from public historical records of the Instituto Nacional de Meteorologia...</span><span class="abstract-full" id="full-2605.08701" hidden>This data paper describes METBRA25Y, a harmonized archive of hourly surface meteorological observations from Brazil derived from public historical records of the Instituto Nacional de Meteorologia (INMET). The dataset was designed to support reproducible environmental, climatological, hydrological, agricultural, urban-risk, and machine-learning studies that require station-level meteorological time series with standardized variable names and explicit quality-control metadata. The processing workflow ingests annual INMET archives, parses station metadata from raw file headers, normalizes heterogeneous Portuguese column names into a canonical schema, constructs hourly timestamps, consolidates observations by city and station, and exports compressed CSV files together with station manifests, per-station quality flags, daily precipitation aggregates, variable-level failure summaries, and missing-data audits. The quality-control protocol follows a two-stage strategy: first, physically implausible values are converted to missing values and flagged; second, temporal and cross-variable consistency checks generate diagnostic flags without necessarily overwriting the original measurements. The resulting package covers observations between 2000 and 2025, with stationspecific temporal coverage, and includes key meteorological variables such as precipitation, air temperature, dew point, relative humidity, atmospheric pressure, wind speed, wind gust, wind direction, and global solar radiation. Based on the summary files included in the current release snapshot, the archive contains 616 unique station codes across variable summaries, of which 605 have coordinates within a broad Brazil plausibility envelope. This paper documents the dataset provenance, file organization, harmonized schema, quality-control rules, technical validation outputs, limitations, and recommended usage practices.</span> <span class="abstract-toggle" data-id="2605.08701">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2605.08701v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2605.08701v1) · [:material-content-copy: BibTeX](../../bibtex/2605.08701.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=hourly" data-tag="hourly">Hourly</a> <a class="md-tag" href="/explore/?t=daily" data-tag="daily">Daily</a>
    { .paper-tags }

-   #### Observation-Guided Neural Surrogate Learning for Scientific Simulation Emulation: A Single-Gauge Flood-Inundation Proof of Concept { #2604.25890 }

    *Marzieh Alireza Mirhoseini* · Apr 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2604.25890">We present an observation-guided neural surrogate-learning framework for scientific simulation emulation, demonstrated on urban flood-inundation mapping. The framework combines LISFLOOD-FP...</span><span class="abstract-full" id="full-2604.25890" hidden>We present an observation-guided neural surrogate-learning framework for scientific simulation emulation, demonstrated on urban flood-inundation mapping. The framework combines LISFLOOD-FP hydrodynamic simulations with a real Gauge L stage record that is mapped to the simulation grid and converted to a datum-consistent local water-depth target before being used as single-site supervision. Focusing on a 256 x 256 crop around Gauge L in the Chicago metropolitan area, the method first constructs an ensemble-approximated Gaussian-process/local analogue surrogate (EnsCGP) to obtain a coarse flood-depth estimate and an uncertainty proxy. A U-Net-ASPP neural corrector then refines the coarse map using only simulation-derived and geospatial inputs: EnsCGP depth, the uncertainty proxy, rainfall, and spatial coordinates. The converted gauge-derived local depth is used only as a pointwise training target at the mapped gauge pixel; simulation-based losses are evaluated away from that pixel. Across temporally held-out events from 2013-2019, the emulator closely reproduces LISFLOOD-FP simulation targets outside the gauge-constrained pixel, with R^2 approximately 0.99 and mean absolute error below 0.01 m, and shows strong pointwise consistency with the converted Gauge L local depth target under the stated rolling-year protocol. We interpret these results as strong simulator-emulation agreement with pointwise observation-guided correction, not as independent validation of real-world inundation accuracy or as a complete operational flood-forecasting system.</span> <span class="abstract-toggle" data-id="2604.25890">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2604.25890v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2604.25890v1) · [:material-content-copy: BibTeX](../../bibtex/2604.25890.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a> <a class="md-tag" href="/explore/?t=extremes" data-tag="extremes">Extremes</a>
    { .paper-tags }

-   #### Process-Aware AI for Rainfall-Runoff Modeling: A Mass-Conserving Neural Framework with Hydrological Process Constraints { #2603.25093 }

    *Mohammad A. Farmani, Hoshin V. Gupta, Ali Behrangi, Muhammad Jawad, Sadaf Moghisi, Guo-Yue Niu* · Mar 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2603.25093">Machine learning models can achieve high predictive accuracy in hydrological applications but often lack physical interpretability. The Mass-Conserving Perceptron (MCP) provides a physics-aware...</span><span class="abstract-full" id="full-2603.25093" hidden>Machine learning models can achieve high predictive accuracy in hydrological applications but often lack physical interpretability. The Mass-Conserving Perceptron (MCP) provides a physics-aware artificial intelligence (AI) framework that enforces conservation principles while allowing hydrological process relationships to be learned from data. In this study, we investigate how progressively embedding physically meaningful representations of hydrological processes within a single MCP storage unit improves predictive skill and interpretability in rainfall-runoff modeling. Starting from a minimal MCP formulation, we sequentially introduce bounded soil storage, state-dependent conductivity, variable porosity, infiltration capacity, surface ponding, vertical drainage, and nonlinear water-table dynamics. The resulting hierarchy of process-aware MCP models is evaluated across 15 catchments spanning five hydroclimatic regions of the continental United States using daily streamflow prediction as the target. Results show that progressively augmenting the internal physical structure of the MCP unit generally improves predictive performance. The influence of these process representations is strongly hydroclimate dependent: vertical drainage substantially improves model skill in arid and snow-dominated basins but reduces performance in rainfall-dominated regions, while surface ponding has comparatively small effects. The best-performing MCP configurations approach the predictive skill of a Long Short-Term Memory benchmark while maintaining explicit physical interpretability. These results demonstrate that embedding hydrological process constraints within AI architectures provides a promising pathway toward interpretable and process-aware rainfall-runoff modeling.</span> <span class="abstract-toggle" data-id="2603.25093">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2603.25093v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2603.25093v1) · [:material-content-copy: BibTeX](../../bibtex/2603.25093.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=interpretability" data-tag="interpretability">Interpretability</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a> <a class="md-tag" href="/explore/?t=daily" data-tag="daily">Daily</a>
    { .paper-tags }

-   #### Climate Adaptation-Aware Flood Prediction for Coastal Cities Using Deep Learning { #2510.26017 }

    *Bilal Hassan, Areg Karapetyan, Aaron Chung Hin Chow, Samer Madanat* · Oct 2025
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2510.26017">Climate change and sea-level rise (SLR) pose escalating threats to coastal cities, intensifying the need for efficient and accurate methods to predict potential flood hazards. Traditional...</span><span class="abstract-full" id="full-2510.26017" hidden>Climate change and sea-level rise (SLR) pose escalating threats to coastal cities, intensifying the need for efficient and accurate methods to predict potential flood hazards. Traditional physics-based hydrodynamic simulators, although precise, are computationally expensive and impractical for city-scale coastal planning applications. Deep Learning (DL) techniques offer promising alternatives, however, they are often constrained by challenges such as data scarcity and high-dimensional output requirements. Leveraging a recently proposed vision-based, low-resource DL framework, we develop a novel, lightweight Convolutional Neural Network (CNN)-based model designed to predict coastal flooding under variable SLR projections and shoreline adaptation scenarios. Furthermore, we demonstrate the ability of the model to generalize across diverse geographical contexts by utilizing datasets from two distinct regions: Abu Dhabi and San Francisco. Our findings demonstrate that the proposed model significantly outperforms state-of-the-art methods, reducing the mean absolute error (MAE) in predicted flood depth maps on average by nearly 20%. These results highlight the potential of our approach to serve as a scalable and practical tool for coastal flood management, empowering decision-makers to develop effective mitigation strategies in response to the growing impacts of climate change. Project Page: https://caspiannet.github.io/</span> <span class="abstract-toggle" data-id="2510.26017">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2510.26017v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2510.26017v1) · [:material-content-copy: BibTeX](../../bibtex/2510.26017.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=extremes" data-tag="extremes">Extremes</a>
    { .paper-tags }

-   #### Progressive Scale Convolutional Network for Spatio-Temporal Downscaling of Soil Moisture: A Case Study Over the Tibetan Plateau { #2510.10244 }

    *Ziyu Zhou, Keyan Hu, Ling Zhang, Zhaohui Xue, Yutian Fang, Yusha Zheng* · Oct 2025
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2510.10244">Soil moisture (SM) plays a critical role in hydrological and meteorological processes. High-resolution SM can be obtained by combining coarse passive microwave data with fine-scale auxiliary...</span><span class="abstract-full" id="full-2510.10244" hidden>Soil moisture (SM) plays a critical role in hydrological and meteorological processes. High-resolution SM can be obtained by combining coarse passive microwave data with fine-scale auxiliary variables. However, the inversion of SM at the temporal scale is hindered by the incompleteness of surface auxiliary factors. To address this issue, first, we introduce validated high temporal resolution ERA5-Land variables into the downscaling process of the low-resolution SMAP SM product. Subsequently, we design a progressive scale convolutional network (PSCNet), at the core of which are two innovative components: a multi-frequency temporal fusion module (MFTF) for capturing temporal dynamics, and a bespoke squeeze-and-excitation (SE) block designed to preserve fine-grained spatial details. Using this approach, we obtained seamless SM products for the Tibetan Plateau (TP) from 2016 to 2018 at 10-km spatial and 3-hour temporal resolution. The experimental results on the TP demonstrated the following: 1) In the satellite product validation, the PSCNet exhibited comparable accuracy and lower error, with a mean R value of 0.881, outperforming other methods. 2) In the in-situ site validation, PSCNet consistently ranked among the top three models for the R metric across all sites, while also showing superior performance in overall error reduction. 3) In the temporal generalization validation, the feasibility of using high-temporal resolution ERA5-Land variables for downscaling was confirmed, as all methods maintained an average relative error within 6% for the R metric and 2% for the ubRMSE metric. 4) In the temporal dynamics and visualization validation, PSCNet demonstrated excellent temporal sensitivity and vivid spatial details. Overall, PSCNet provides a promising solution for spatio-temporal downscaling by effectively modeling the intricate spatio-temporal relationships in SM data.</span> <span class="abstract-toggle" data-id="2510.10244">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2510.10244v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2510.10244v1) · [:material-content-copy: BibTeX](../../bibtex/2510.10244.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a>
    { .paper-tags }

-   #### QDeepGR4J: Quantile-based ensemble of deep learning and GR4J hybrid rainfall-runoff models for extreme flow prediction with uncertainty quantification { #2510.05453 }

    *Arpit Kapoor, Rohitash Chandra* · Oct 2025
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2510.05453">Conceptual rainfall-runoff models aid hydrologists and climate scientists in modelling streamflow to inform water management practices. Recent advances in deep learning have unravelled the potential...</span><span class="abstract-full" id="full-2510.05453" hidden>Conceptual rainfall-runoff models aid hydrologists and climate scientists in modelling streamflow to inform water management practices. Recent advances in deep learning have unravelled the potential for combining hydrological models with deep learning models for better interpretability and improved predictive performance. In our previous work, we introduced DeepGR4J, which enhanced the GR4J conceptual rainfall-runoff model using a deep learning model to serve as a surrogate for the routing component. DeepGR4J had an improved rainfall-runoff prediction accuracy, particularly in arid catchments. Quantile regression models have been extensively used for quantifying uncertainty while aiding extreme value forecasting. In this paper, we extend DeepGR4J using a quantile regression-based ensemble learning framework to quantify uncertainty in streamflow prediction. We also leverage the uncertainty bounds to identify extreme flow events potentially leading to flooding. We further extend the model to multi-step streamflow predictions for uncertainty bounds. We design experiments for a detailed evaluation of the proposed framework using the CAMELS-Aus dataset. The results show that our proposed Quantile DeepGR4J framework improves the predictive accuracy and uncertainty interval quality (interval score) compared to baseline deep learning models. Furthermore, we carry out flood risk evaluation using Quantile DeepGR4J, and the results demonstrate its suitability as an early warning system.</span> <span class="abstract-toggle" data-id="2510.05453">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2510.05453v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2510.05453v1) · [:material-content-copy: BibTeX](../../bibtex/2510.05453.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=extremes" data-tag="extremes">Extremes</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=benchmarks-datasets" data-tag="benchmarks-datasets">Benchmarks & datasets</a>
    { .paper-tags }

-   #### Towards CONUS-Wide ML-Augmented Conceptually-Interpretable Modeling of Catchment-Scale Precipitation-Storage-Runoff Dynamics { #2510.02605 }

    *Yuan-Heng Wang, Yang Yang, Fabio Ciulla, Hoshin V. Gupta, Charuleka Varadharajan* · Oct 2025
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2510.02605">While many modern studies are dedicated to ML-based large-sample hydrologic modeling, these efforts have not necessarily translated into predictive improvements that are grounded in enhanced...</span><span class="abstract-full" id="full-2510.02605" hidden>While many modern studies are dedicated to ML-based large-sample hydrologic modeling, these efforts have not necessarily translated into predictive improvements that are grounded in enhanced physical-conceptual understanding. Here, we report on a CONUS-wide large-sample study (spanning diverse hydro-geo-climatic conditions) using ML-augmented physically-interpretable catchment-scale models of varying complexity based in the Mass-Conserving Perceptron (MCP). Results were evaluated using attribute masks such as snow regime, forest cover, and climate zone. Our results indicate the importance of selecting model architectures of appropriate model complexity based on how process dominance varies with hydrological regime. Benchmark comparisons show that physically-interpretable mass-conserving MCP-based models can achieve performance comparable to data-based models based in the Long Short-Term Memory network (LSTM) architecture. Overall, this study highlights the potential of a theory-informed, physically grounded approach to large-sample hydrology, with emphasis on mechanistic understanding and the development of parsimonious and interpretable model architectures, thereby laying the foundation for future models of everywhere that architecturally encode information about spatially- and temporally-varying process dominance.</span> <span class="abstract-toggle" data-id="2510.02605">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2510.02605v2) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2510.02605v2) · [:material-content-copy: BibTeX](../../bibtex/2510.02605.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=recurrent-networks" data-tag="recurrent-networks">Recurrent networks</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=interpretability" data-tag="interpretability">Interpretability</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a>
    { .paper-tags }

-   #### RainSeer: Fine-Grained Rainfall Reconstruction via Physics-Guided Modeling { #2510.02414 }

    *Lin Chen, Jun Chen, Minghui Qiu, Shuxin Zhong, Binghong Chen, Kaishun Wu* · Oct 2025
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2510.02414">Reconstructing high-resolution rainfall fields is essential for flood forecasting, hydrological modeling, and climate analysis. However, existing spatial interpolation methods-whether based on...</span><span class="abstract-full" id="full-2510.02414" hidden>Reconstructing high-resolution rainfall fields is essential for flood forecasting, hydrological modeling, and climate analysis. However, existing spatial interpolation methods-whether based on automatic weather station (AWS) measurements or enhanced with satellite/radar observations often over-smooth critical structures, failing to capture sharp transitions and localized extremes. We introduce RainSeer, a structure-aware reconstruction framework that reinterprets radar reflectivity as a physically grounded structural prior-capturing when, where, and how rain develops. This shift, however, introduces two fundamental challenges: (i) translating high-resolution volumetric radar fields into sparse point-wise rainfall observations, and (ii) bridging the physical disconnect between aloft hydro-meteors and ground-level precipitation. RainSeer addresses these through a physics-informed two-stage architecture: a Structure-to-Point Mapper performs spatial alignment by projecting mesoscale radar structures into localized ground-level rainfall, through a bidirectional mapping, and a Geo-Aware Rain Decoder captures the semantic transformation of hydro-meteors through descent, melting, and evaporation via a causal spatiotemporal attention mechanism. We evaluate RainSeer on two public datasets-RAIN-F (Korea, 2017-2019) and MeteoNet (France, 2016-2018)-and observe consistent improvements over state-of-the-art baselines, reducing MAE by over 13.31% and significantly enhancing structural fidelity in reconstructed rainfall fields.</span> <span class="abstract-toggle" data-id="2510.02414">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2510.02414v2) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2510.02414v2) · [:material-content-copy: BibTeX](../../bibtex/2510.02414.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=physics-ml-hybrid" data-tag="physics-ml-hybrid">Physics–ML hybrid</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a>
    { .paper-tags }

-   #### Projecting U.S. coastal storm surge risks and impacts with deep learning { #2506.13963 }

    *Julian R. Rice, Karthik Balaguru, Fadia Ticona Rollano, John Wilson, Brent Daniel, David Judi et al.* · Jun 2025
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2506.13963">Storm surge is one of the deadliest hazards posed by tropical cyclones (TCs), yet assessing its current and future risk is difficult due to the phenomenon's rarity and physical complexity. Recent...</span><span class="abstract-full" id="full-2506.13963" hidden>Storm surge is one of the deadliest hazards posed by tropical cyclones (TCs), yet assessing its current and future risk is difficult due to the phenomenon's rarity and physical complexity. Recent advances in artificial intelligence applications to natural hazard modeling suggest a new avenue for addressing this problem. We utilize a deep learning storm surge model to efficiently estimate coastal surge risk in the United States from 900,000 synthetic TC events, accounting for projected changes in TC behavior and sea levels. The derived historical 100-year surge (the event with a 1% yearly exceedance probability) agrees well with historical observations and other modeling techniques. When coupled with an inundation model, we find that heightened TC intensities and sea levels by the end of the century result in a 50% increase in population at risk. Key findings include markedly heightened risk in Florida, and critical thresholds identified in Georgia and South Carolina.</span> <span class="abstract-toggle" data-id="2506.13963">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2506.13963v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2506.13963v1) · [:material-content-copy: BibTeX](../../bibtex/2506.13963.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a>
    { .paper-tags }

-   #### Functional data decomposition reveals unexpectedly strong soil moisture-precipitation coupling over the Great Plains { #2506.13939 }

    *Yifu Gao, Runze Li, Efi Foufoula-Georgiou, Jasper A. Vrugt* · Jun 2025
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2506.13939">Soil moisture-precipitation coupling (SMPC) plays a critical role in Earth's water and energy cycles but remains difficult to quantify due to synoptic-scale variability and the complex interplay of...</span><span class="abstract-full" id="full-2506.13939" hidden>Soil moisture-precipitation coupling (SMPC) plays a critical role in Earth's water and energy cycles but remains difficult to quantify due to synoptic-scale variability and the complex interplay of land-atmosphere processes. Here, we apply high-dimensional model representation (HDMR) to functionally decompose the structural, correlative, and cooperative contributions of key land-atmosphere variables to precipitation. Benchmark tests confirm that HDMR overcomes limitations of commonly used correlation and regression approaches in isolating direct versus indirect effects. For example, analysis of gross primary productivity using a light-use-efficiency model shows that linear regression underestimates the temperature effect, while HDMR captures it accurately. Applying HDMR to CONUS404 reanalysis data reveals that morning soil moisture explains up to 40 percent of the variance in summertime afternoon precipitation over the Great Plains, more than double prior estimates. On days with afternoon rainfall (12-hour totals of 4.7-8.2 mm), first-order SM effects can boost precipitation by up to 8 mm under wet conditions, with an additional 3 mm from second-order interactions involving temperature and moisture. By capturing real-world co-variability and higher-order effects, HDMR provides a physically grounded, data-driven framework for diagnosing land-atmosphere coupling. These results underscore the need for more nuanced, interaction-aware data analysis methods in climate modeling and prediction.</span> <span class="abstract-toggle" data-id="2506.13939">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2506.13939v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2506.13939v1) · [:material-content-copy: BibTeX](../../bibtex/2506.13939.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a>
    { .paper-tags }

-   #### Towards NoahMP-AI: Enhancing Land Surface Model Prediction with Deep Learning { #2506.12919 }

    *Mahmoud Mbarak, Manmeet Singh, Naveen Sudharsan, Zong-Liang Yang* · Jun 2025
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2506.12919">Accurate soil moisture prediction during extreme events remains a critical challenge for earth system modeling, with profound implications for drought monitoring, flood forecasting, and climate...</span><span class="abstract-full" id="full-2506.12919" hidden>Accurate soil moisture prediction during extreme events remains a critical challenge for earth system modeling, with profound implications for drought monitoring, flood forecasting, and climate adaptation strategies. While land surface models (LSMs) provide physically-based predictions, they exhibit systematic biases during extreme conditions when their parameterizations operate outside calibrated ranges. Here we present NoahMP-AI, a physics-guided deep learning framework that addresses this challenge by leveraging the complete Noah-MP land surface model as a comprehensive physics-based feature generator while using machine learning to correct structural limitations against satellite observations. We employ a 3D U-Net architecture that processes Noah-MP outputs (soil moisture, latent heat flux, and sensible heat flux) to predict SMAP soil moisture across two contrasting extreme events: a prolonged drought (March-September 2022) and Hurricane Beryl (July 2024) over Texas. When comparing NoahMP-AI with NoahMP, our results demonstrate an increase in R-squared values from -0.7 to 0.5 during drought conditions, while maintaining physical consistency and spatial coherence. The framework's ability to preserve Noah-MP's physical relationships while learning observation-based corrections represents a significant advance in hybrid earth system modeling. This work establishes both a practical tool for operational forecasting and a benchmark for investigating the optimal integration of physics-based understanding with data-driven learning in environmental prediction systems.</span> <span class="abstract-toggle" data-id="2506.12919">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2506.12919v3) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2506.12919v3) · [:material-content-copy: BibTeX](../../bibtex/2506.12919.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a> <a class="md-tag" href="/explore/?t=physics-ml-hybrid" data-tag="physics-ml-hybrid">Physics–ML hybrid</a> <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a> <a class="md-tag" href="/explore/?t=extremes" data-tag="extremes">Extremes</a>
    { .paper-tags }

-   #### Spatially Resolved Meteorological and Ancillary Data in Central Europe for Rainfall Streamflow Modeling { #2506.03819 }

    *Marc Aurel Vischer, Noelia Otero, Jackie Ma* · Jun 2025
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2506.03819">We present a dataset for rainfall streamflow modeling that is fully spatially resolved with the aim of taking neural network-driven hydrological modeling beyond lumped catchments. To this end, we...</span><span class="abstract-full" id="full-2506.03819" hidden>We present a dataset for rainfall streamflow modeling that is fully spatially resolved with the aim of taking neural network-driven hydrological modeling beyond lumped catchments. To this end, we compiled data covering five river basins in central Europe: upper Danube, Elbe, Oder, Rhine, and Weser. The dataset contains meteorological forcings, as well as ancillary information on soil, rock, land cover, and orography. The data is harmonized to a regular 9km times 9km grid and contains daily values that span from October 1981 to September 2011. We also provide code to further combine our dataset with publicly available river discharge data for end-to-end rainfall streamflow modeling.</span> <span class="abstract-toggle" data-id="2506.03819">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2506.03819v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2506.03819v1) · [:material-content-copy: BibTeX](../../bibtex/2506.03819.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=benchmarks-datasets" data-tag="benchmarks-datasets">Benchmarks & datasets</a> <a class="md-tag" href="/explore/?t=daily" data-tag="daily">Daily</a>
    { .paper-tags }

-   #### A Multi-Tiered Bayesian Network Coastal Compound Flood Analysis Framework { #2505.15520 }

    *Ziyue Liu, Meredith L. Carr, Norberto C. Nadal-Caraballo, Luke A. Aucoin, Madison C. Yawn et al.* · May 2025
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2505.15520">Coastal compound floods (CCFs) are triggered by the interaction of multiple mechanisms, such as storm surges, storm rainfall, tides, and river flow. These events can bring significant damage to...</span><span class="abstract-full" id="full-2505.15520" hidden>Coastal compound floods (CCFs) are triggered by the interaction of multiple mechanisms, such as storm surges, storm rainfall, tides, and river flow. These events can bring significant damage to communities, and there is an increasing demand for accurate and efficient probabilistic analyses of CCFs to support risk assessments and decision-making. In this study, a multi-tiered Bayesian network (BN) CCF analysis framework is established. In this framework, conceptual designs of multiple tiers of BN models with varying complexities are developed for application with varying levels of data availability and resources. A case study is conducted in New Orleans, LA, with three tiers of BN models constructed to demonstrate this framework. In the Tier-1 BN model, storm surges and river flow are incorporated based on hydrodynamic simulations. A seasonality node is used to capture the dependence between concurrent river flow and tropical cyclone (TC) parameters. In the Tier-2 BN model, joint distribution models of TC parameters are built for separate TC intensity categories. TC-induced rainfall is modeled as input to hydraulic simulations. In the Tier-3 BN model, potential variations of meteorological conditions are incorporated by quantifying their effects on TC activity and coastal water level. Flood antecedent conditions are also incorporated to more completely represent the conditions contributing to flood severity. In this case study, a series of joint distribution, numerical, machine learning, and experimental models are used to compute conditional probability tables needed for the BNs. A series of probabilistic analyses is performed based on these BN models, including CCF hazard curve construction and CCF deaggregation. The results of the analysis demonstrate the promise of this framework in performing CCF hazard analysis under varying levels of resource availability and project needs.</span> <span class="abstract-toggle" data-id="2505.15520">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2505.15520v2) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2505.15520v2) · [:material-content-copy: BibTeX](../../bibtex/2505.15520.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a> <a class="md-tag" href="/explore/?t=extremes" data-tag="extremes">Extremes</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### Assessing wildfire susceptibility in Iran: Leveraging machine learning for geospatial analysis of climatic and anthropogenic factors { #2505.14122 }

    *Ehsan Masoudian, Ali Mirzaei, Hossein Bagheri* · May 2025
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2505.14122">This study investigates the multifaceted factors influencing wildfire risk in Iran, focusing on the interplay between climatic conditions and human activities. Utilizing advanced remote sensing,...</span><span class="abstract-full" id="full-2505.14122" hidden>This study investigates the multifaceted factors influencing wildfire risk in Iran, focusing on the interplay between climatic conditions and human activities. Utilizing advanced remote sensing, geospatial information system (GIS) processing techniques such as cloud computing, and machine learning algorithms, this research analyzed the impact of climatic parameters, topographic features, and human-related factors on wildfire susceptibility assessment and prediction in Iran. Multiple scenarios were developed for this purpose based on the data sampling strategy. The findings revealed that climatic elements such as soil moisture, temperature, and humidity significantly contribute to wildfire susceptibility, while human activities-particularly population density and proximity to powerlines-also played a crucial role. Furthermore, the seasonal impact of each parameter was separately assessed during warm and cold seasons. The results indicated that human-related factors, rather than climatic variables, had a more prominent influence during the seasonal analyses. This research provided new insights into wildfire dynamics in Iran by generating high-resolution wildfire susceptibility maps using advanced machine learning classifiers. The generated maps identified high risk areas, particularly in the central Zagros region, the northeastern Hyrcanian Forest, and the northern Arasbaran forest, highlighting the urgent need for effective fire management strategies.</span> <span class="abstract-toggle" data-id="2505.14122">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2505.14122v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2505.14122v1) · [:material-content-copy: BibTeX](../../bibtex/2505.14122.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=extremes" data-tag="extremes">Extremes</a>
    { .paper-tags }

-   #### A Physically Driven Long Short Term Memory Model for Estimating Snow Water Equivalent over the Continental United States { #2504.20129 }

    *Arun M. Saranathan, Mahmoud Saeedimoghaddam, Brandon Smith, Deepthi Raghunandan, Grey Nearing et al.* · Apr 2025
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2504.20129">Snow is an essential input for various land surface models. Seasonal snow estimates are available as snow water equivalent (SWE) from process-based reanalysis products or locally from in situ...</span><span class="abstract-full" id="full-2504.20129" hidden>Snow is an essential input for various land surface models. Seasonal snow estimates are available as snow water equivalent (SWE) from process-based reanalysis products or locally from in situ measurements. While the reanalysis products are computationally expensive and available at only fixed spatial and temporal resolutions, the in situ measurements are highly localized and sparse. To address these issues and enable the analysis of the effect of a large suite of physical, morphological, and geological conditions on the presence and amount of snow, we build a Long Short-Term Memory (LSTM) network, which is able to estimate the SWE based on time series input of the various physical/meteorological factors as well static spatial/morphological factors. Specifically, this model breaks down the SWE estimation into two separate tasks: (i) a classification task that indicates the presence/absence of snow on a specific day and (ii) a regression task that indicates the height of the SWE on a specific day in the case of snow presence. The model is trained using physical/in situ SWE measurements from the SNOw TELemetry (SNOTEL) snow pillows in the western United States. We will show that trained LSTM models have a classification accuracy of $\geq 93\%$ for the presence of snow and a coefficient of correlation of $\sim 0.9$ concerning their SWE estimates. We will also demonstrate that the models can generalize both spatially and temporally to previously unseen data.</span> <span class="abstract-toggle" data-id="2504.20129">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2504.20129v2) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2504.20129v2) · [:material-content-copy: BibTeX](../../bibtex/2504.20129.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=recurrent-networks" data-tag="recurrent-networks">Recurrent networks</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a> <a class="md-tag" href="/explore/?t=station-point" data-tag="station-point">Station / point</a>
    { .paper-tags }

-   #### Enhancing Deterministic Freezing Level Predictions in the Northern Sierra Nevada Through Deep Neural Networks { #2504.11560 }

    *Vesta Afzali Gorooh, Agniv Sengupta, Shawn Roj, Rachel Weihs, Brian Kawzenuk, Luca Delle Monache et al.* · Apr 2025
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2504.11560">Accurate prediction of the freezing level is essential for hydrometeorological forecasting systems, with direct implications for runoff generation and reservoir management. In this study, we develop...</span><span class="abstract-full" id="full-2504.11560" hidden>Accurate prediction of the freezing level is essential for hydrometeorological forecasting systems, with direct implications for runoff generation and reservoir management. In this study, we develop a deep learning based postprocessing framework using the Unet convolutional neural network architecture to refine the FZL forecasts from the West Weather Research and Forecasting West WRF model. The proposed framework leverages reforecast data from West WRF and FZL estimates from the California Nevada River Forecast Center to develop Unet models over the northern Sierra Nevada watersheds, such as the hydrologically critical Yuba Feather watershed. We introduce two Unet model variants, Unet_log and Unet_GMM, that employ specialized loss functions beyond the standard benchmarks to enhance forecast skill. Unet_log utilizes the cosine of Error, and Unet_MM uses Gaussian Mixture Model loss functions, to enhance FZL forecasts. Results show that Unet based postprocessing reduces centered root mean squared errors by up to 20% and increases forecast observation correlation by about 10% compared to raw WestWRF. Evaluation using the continuous ranked probability score for Unet_GMM further demonstrates consistent improvements across lead times. While performance fluctuates with forecast horizon, storm variability, and diurnal forcing, Unet_GMM and Unet_log consistently outperform the baseline. The models capture the spatiotemporal variability of the FZL across different elevations, mitigating biases from the West WRF model. This novel deep learning based postprocessing approach demonstrates a promising pathway for integrating machine learning into hydrometeorological forecasting and decision support within the Forecast Informed Reservoir Operations framework.</span> <span class="abstract-toggle" data-id="2504.11560">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2504.11560v2) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2504.11560v2) · [:material-content-copy: BibTeX](../../bibtex/2504.11560.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a>
    { .paper-tags }

-   #### A Spatiotemporal Radar-Based Precipitation Model for Water Level Prediction and Flood Forecasting { #2503.19943 }

    *Sakshi Dhankhar, Stefan Wittek, Hamidreza Eivazi, Andreas Rausch* · Mar 2025
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2503.19943">Study Region: Goslar and Göttingen, Lower Saxony, Germany. Study Focus: In July 2017, the cities of Goslar and Göttingen experienced severe flood events characterized by short warning time of only 20...</span><span class="abstract-full" id="full-2503.19943" hidden>Study Region: Goslar and Göttingen, Lower Saxony, Germany. Study Focus: In July 2017, the cities of Goslar and Göttingen experienced severe flood events characterized by short warning time of only 20 minutes, resulting in extensive regional flooding and significant damage. This highlights the critical need for a more reliable and timely flood forecasting system. This paper presents a comprehensive study on the impact of radar-based precipitation data on forecasting river water levels in Goslar. Additionally, the study examines how precipitation influences water level forecasts in Göttingen. The analysis integrates radar-derived spatiotemporal precipitation patterns with hydrological sensor data obtained from ground stations to evaluate the effectiveness of this approach in improving flood prediction capabilities. New Hydrological Insights for the Region: A key innovation in this paper is the use of residual-based modeling to address the non-linearity between precipitation images and water levels, leading to a Spatiotemporal Radar-based Precipitation Model with residuals (STRPMr). Unlike traditional hydrological models, our approach does not rely on upstream data, making it independent of additional hydrological inputs. This independence enhances its adaptability and allows for broader applicability in other regions with RADOLAN precipitation. The deep learning architecture integrates (2+1)D convolutional neural networks for spatial and temporal feature extraction with LSTM for timeseries forecasting. The results demonstrate the potential of the STRPMr for capturing extreme events and more accurate flood forecasting.</span> <span class="abstract-toggle" data-id="2503.19943">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2503.19943v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2503.19943v1) · [:material-content-copy: BibTeX](../../bibtex/2503.19943.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a> <a class="md-tag" href="/explore/?t=recurrent-networks" data-tag="recurrent-networks">Recurrent networks</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=extremes" data-tag="extremes">Extremes</a> <a class="md-tag" href="/explore/?t=sub-hourly" data-tag="sub-hourly">Sub-hourly</a>
    { .paper-tags }

-   #### Update hydrological states or meteorological forcings? Comparing data assimilation methods for differentiable hydrologic models { #2502.16444 }

    *Amirmoez Jamaat, Yalan Song, Farshid Rahmani, Jiangtao Liu, Kathryn Lawson, Chaopeng Shen* · Feb 2025
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2502.16444">Data assimilation (DA) enables hydrologic models to update their internal states using near-real-time observations for more accurate forecasts. With deep neural networks like long short-term memory...</span><span class="abstract-full" id="full-2502.16444" hidden>Data assimilation (DA) enables hydrologic models to update their internal states using near-real-time observations for more accurate forecasts. With deep neural networks like long short-term memory (LSTM), using either lagged observations as inputs (called "data integration") or variational DA has shown success in improving forecasts. However, it is unclear which methods are performant or optimal for physics-informed machine learning ("differentiable") models, which represent only a small amount of physically-meaningful states while using deep networks to supply parameters or missing processes. Here we developed variational DA methods for differentiable models, including optimizing adjusters for just precipitation data, just model internal hydrological states, or both. Our results demonstrated that differentiable streamflow models using the CAMELS dataset can benefit strongly and equivalently from variational DA as LSTM, with one-day lead time median Nash-Sutcliffe efficiency (NSE) elevated from 0.75 to 0.82. The resulting forecast matched or outperformed LSTM with DA in the eastern, northwestern, and central Great Plains regions of the conterminous United States. Both precipitation and state adjusters were needed to achieve these results, with the latter being substantially more effective on its own, and the former adding moderate benefits for high flows. Our DA framework does not need systematic training data and could serve as a practical DA scheme for whole river networks.</span> <span class="abstract-toggle" data-id="2502.16444">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2502.16444v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2502.16444v1) · [:material-content-copy: BibTeX](../../bibtex/2502.16444.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=recurrent-networks" data-tag="recurrent-networks">Recurrent networks</a> <a class="md-tag" href="/explore/?t=physics-ml-hybrid" data-tag="physics-ml-hybrid">Physics–ML hybrid</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a>
    { .paper-tags }

-   #### Refined climatologies of future precipitation over High Mountain Asia using probabilistic ensemble learning { #2501.15690 }

    *Kenza Tazi, Sun Woo P. Kim, Marc Girona-Mata, Richard E. Turner* · Jan 2025
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2501.15690">High Mountain Asia (HMA) holds the highest concentration of frozen water outside the polar regions, serving as a crucial water source for more than 1.9 billion people. Precipitation represents the...</span><span class="abstract-full" id="full-2501.15690" hidden>High Mountain Asia (HMA) holds the highest concentration of frozen water outside the polar regions, serving as a crucial water source for more than 1.9 billion people. Precipitation represents the largest source of uncertainty for future hydrological modelling in this area. In this study, we propose a probabilistic machine learning framework to combine monthly precipitation from 13 regional climate models developed under the Coordinated Regional Downscaling Experiment (CORDEX) over HMA via a mixture of experts (MoE). This approach accounts for seasonal and spatial biases within the models, enabling the prediction of more faithful precipitation distributions. The MoE is trained and validated against gridded historical precipitation data, yielding 32% improvement over an equally-weighted average and 254% improvement over choosing any single ensemble member. This approach is then used to generate precipitation projections for the near future (2036-2065) and far future (2066-2095) under RCP4.5 and RCP8.5 scenarios. Compared to previous estimates, the MoE projects wetter summers but drier winters over the western Himalayas and Karakoram and wetter winters over the Tibetan Plateau, Hengduan Shan, and South East Tibet.</span> <span class="abstract-toggle" data-id="2501.15690">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2501.15690v3) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2501.15690v3) · [:material-content-copy: BibTeX](../../bibtex/2501.15690.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a> <a class="md-tag" href="/explore/?t=monthly" data-tag="monthly">Monthly</a>
    { .paper-tags }

-   #### A Deep State Space Model for Rainfall-Runoff Simulations { #2501.14980 }

    *Yihan Wang, Lujun Zhang, Annan Yu, N. Benjamin Erichson, Tiantian Yang* · Jan 2025
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2501.14980">The classical way of studying the rainfall-runoff processes in the water cycle relies on conceptual or physically-based hydrologic models. Deep learning (DL) has recently emerged as an alternative...</span><span class="abstract-full" id="full-2501.14980" hidden>The classical way of studying the rainfall-runoff processes in the water cycle relies on conceptual or physically-based hydrologic models. Deep learning (DL) has recently emerged as an alternative and blossomed in hydrology community for rainfall-runoff simulations. However, the decades-old Long Short-Term Memory (LSTM) network remains the benchmark for this task, outperforming newer architectures like Transformers. In this work, we propose a State Space Model (SSM), specifically the Frequency Tuned Diagonal State Space Sequence (S4D-FT) model, for rainfall-runoff simulations. The proposed S4D-FT is benchmarked against the established LSTM and a physically-based Sacramento Soil Moisture Accounting model across 531 watersheds in the contiguous United States (CONUS). Results show that S4D-FT is able to outperform the LSTM model across diverse regions. Our pioneering introduction of the S4D-FT for rainfall-runoff simulations challenges the dominance of LSTM in the hydrology community and expands the arsenal of DL tools available for hydrological modeling.</span> <span class="abstract-toggle" data-id="2501.14980">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2501.14980v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2501.14980v1) · [:material-content-copy: BibTeX](../../bibtex/2501.14980.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=recurrent-networks" data-tag="recurrent-networks">Recurrent networks</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a>
    { .paper-tags }

-   #### AI-Driven Reinvention of Hydrological Modeling for Accurate Predictions and Interpretation to Transform Earth System Modeling { #2501.04733 }

    *Cuihui Xia, Lei Yue, Deliang Chen, Yuyang Li, Hongqiang Yang, Ancheng Xue, Zhiqiang Li, Qing He et al.* · Jan 2025
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2501.04733">Traditional equation-driven hydrological models often struggle to accurately predict streamflow in challenging regional Earth systems like the Tibetan Plateau, while hybrid and existing...</span><span class="abstract-full" id="full-2501.04733" hidden>Traditional equation-driven hydrological models often struggle to accurately predict streamflow in challenging regional Earth systems like the Tibetan Plateau, while hybrid and existing algorithm-driven models face difficulties in interpreting hydrological behaviors. This work introduces HydroTrace, an algorithm-driven, data-agnostic model that substantially outperforms these approaches, achieving a Nash-Sutcliffe Efficiency of 98% and demonstrating strong generalization on unseen data. Moreover, HydroTrace leverages advanced attention mechanisms to capture spatial-temporal variations and feature-specific impacts, enabling the quantification and spatial resolution of streamflow partitioning as well as the interpretation of hydrological behaviors such as glacier-snow-streamflow interactions and monsoon dynamics. Additionally, a large language model (LLM)-based application allows users to easily understand and apply HydroTrace's insights for practical purposes. These advancements position HydroTrace as a transformative tool in hydrological and broader Earth system modeling, offering enhanced prediction accuracy and interpretability.</span> <span class="abstract-toggle" data-id="2501.04733">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2501.04733v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2501.04733v1) · [:material-content-copy: BibTeX](../../bibtex/2501.04733.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=llms-agents" data-tag="llms-agents">LLMs & agents</a>
    { .paper-tags }

-   #### Graph Learning-based Regional Heavy Rainfall Prediction Using Low-Cost Rain Gauges { #2412.16842 }

    *Edwin Salcedo* · Dec 2024
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2412.16842">Accurate and timely prediction of heavy rainfall events is crucial for effective flood risk management and disaster preparedness. By monitoring, analysing, and evaluating rainfall data at a local...</span><span class="abstract-full" id="full-2412.16842" hidden>Accurate and timely prediction of heavy rainfall events is crucial for effective flood risk management and disaster preparedness. By monitoring, analysing, and evaluating rainfall data at a local level, it is not only possible to take effective actions to prevent any severe climate variation but also to improve the planning of surface and underground hydrological resources. However, developing countries often lack the weather stations to collect data continuously due to the high cost of installation and maintenance. In light of this, the contribution of the present paper is twofold: first, we propose a low-cost IoT system for automatic recording, monitoring, and prediction of rainfall in rural regions. Second, we propose a novel approach to regional heavy rainfall prediction by implementing graph neural networks (GNNs), which are particularly well-suited for capturing the complex spatial dependencies inherent in rainfall patterns. The proposed approach was tested using a historical dataset spanning 72 months, with daily measurements, and experimental results demonstrated the effectiveness of the proposed method in predicting heavy rainfall events, making this approach particularly attractive for regions with limited resources or where traditional weather radar or station coverage is sparse.</span> <span class="abstract-toggle" data-id="2412.16842">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2412.16842v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2412.16842v1) · [:material-content-copy: BibTeX](../../bibtex/2412.16842.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=graph-neural-networks" data-tag="graph-neural-networks">Graph neural networks</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=station-point" data-tag="station-point">Station / point</a> <a class="md-tag" href="/explore/?t=daily" data-tag="daily">Daily</a>
    { .paper-tags }

-   #### A Physics-Constrained Neural Differential Equation Framework for Data-Driven Snowpack Simulation { #2412.06819 }

    *Andrew Charbonneau, Katherine Deck, Tapio Schneider* · Dec 2024
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2412.06819">This paper presents a physics-constrained neural differential equation framework for parameterization, and employs it to model the time evolution of seasonal snow depth given hydrometeorological...</span><span class="abstract-full" id="full-2412.06819" hidden>This paper presents a physics-constrained neural differential equation framework for parameterization, and employs it to model the time evolution of seasonal snow depth given hydrometeorological forcings. When trained on data from multiple SNOTEL sites, the parameterization predicts daily snow depth with under 9% median error and Nash Sutcliffe Efficiencies over 0.94 across a wide variety of snow climates. The parameterization also generalizes to new sites not seen during training, which is not often true for calibrated snow models. Requiring the parameterization to predict snow water equivalent in addition to snow depth only increases error to ~12%. The structure of the approach guarantees the satisfaction of physical constraints, enables these constraints during model training, and allows modeling at different temporal resolutions without additional retraining of the parameterization. These benefits hold potential in climate modeling, and could extend to other dynamical systems with physical constraints.</span> <span class="abstract-toggle" data-id="2412.06819">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2412.06819v3) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2412.06819v3) · [:material-content-copy: BibTeX](../../bibtex/2412.06819.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=physics-ml-hybrid" data-tag="physics-ml-hybrid">Physics–ML hybrid</a> <a class="md-tag" href="/explore/?t=daily" data-tag="daily">Daily</a>
    { .paper-tags }

-   #### Using Machine Learning to Discover Parsimonious and Physically-Interpretable Representations of Catchment-Scale Rainfall-Runoff Dynamics { #2412.04845 }

    *Yuan-Heng Wang, Hoshin V. Gupta* · Dec 2024
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2412.04845">Due largely to challenges associated with physical interpretability of machine learning (ML) methods, and because model interpretability is key to credibility in management applications, many...</span><span class="abstract-full" id="full-2412.04845" hidden>Due largely to challenges associated with physical interpretability of machine learning (ML) methods, and because model interpretability is key to credibility in management applications, many scientists and practitioners are hesitant to discard traditional physical-conceptual (PC) modeling approaches despite their poorer predictive performance. Here, we examine how to develop parsimonious minimally-optimal representations that can facilitate better insight regarding system functioning. The term minimally-optimal indicates that the desired outcome can be achieved with the smallest possible effort and resources, while parsimony is widely held to support understanding. Accordingly, we suggest that ML-based modeling should use computational units that are inherently physically-interpretable, and explore how generic network architectures comprised of Mass-Conserving-Perceptron can be used to model dynamical systems in a physically-interpretable manner.   In the context of spatially-lumped catchment-scale modeling, we find that both physical interpretability and good predictive performance can be achieved using a distributed-state network with context-dependent gating and information sharing across nodes. The distributed-state mechanism ensures a sufficient number of temporally-evolving properties of system storage while information-sharing ensures proper synchronization of such properties. The results indicate that MCP-based ML models with only a few layers (up to two) and relativity few physical flow pathways (up to three) can play a significant role in ML-based streamflow modelling.</span> <span class="abstract-toggle" data-id="2412.04845">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2412.04845v5) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2412.04845v5) · [:material-content-copy: BibTeX](../../bibtex/2412.04845.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=interpretability" data-tag="interpretability">Interpretability</a>
    { .paper-tags }

</div>

<nav class="pager" markdown="span">**1** [2](2.md) [Older :material-arrow-right:](2.md){ .pager-step }</nav>

