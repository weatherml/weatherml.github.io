---
title: 'Nowcasting'
hide:
  - toc
---

<div class="listing-header" markdown>

# Nowcasting

<p class="page-meta" markdown="span">97 papers · page 1 of 4 · <a href="../../bib/nowcasting.bib" download>:material-download: BibTeX for this topic</a></p>

</div>

<div class="grid cards" markdown>

-   #### IRENE: A Convolutional GRU Ensemble Model for Radar Precipitation Nowcasting over Italy { #2609.17175 }

    *Alessandro Camilletti, Gabriele Franch, Elena Tomasi, Marco Cristoforetti* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.17175">We present IRENE (Italian Radar Ensemble Nowcasting Experiment), a deep learning model for probabilistic short-range precipitation nowcasting over the Italian domain at 1 km spatial and 5 min...</span><span class="abstract-full" id="full-2609.17175" hidden>We present IRENE (Italian Radar Ensemble Nowcasting Experiment), a deep learning model for probabilistic short-range precipitation nowcasting over the Italian domain at 1 km spatial and 5 min temporal resolution. IRENE adopts an encoder--forecaster architecture built on multi-scale Convolutional Gated Recurrent Units (ConvGRUs), trained on the national radar composite produced by the Italian Civil Protection Department (DPC). An importance-sampling scheme focuses training on precipitation-relevant events, while the almost-fair Continuous Ranked Probability Score (afCRPS) is adopted as the primary probabilistic loss function. Two additional training configurations are proposed: an adversarial (GAN) variant, IRENE-GAN, designed to improve the spatial sharpness of the generated forecasts, and a spectrally constrained variant, IRENE-GAN-RAPSD, in which the adversarial objective is complemented by an explicit penalty on the radially averaged power spectral density. The three configurations are evaluated against the stochastic extrapolation method STEPS and the pre-trained deep learning model DGMR. All IRENE configurations attain a lower Continuous Ranked Probability Score than both benchmarks at every lead time and rank histograms closer to uniformity, indicating better probabilistic skill and ensemble calibration. In terms of ensemble-mean mean absolute error the advantage is confined to the first 90 min, beyond which the strongly damped DGMR fields and, to a lesser extent, STEPS become competitive. Spectral analysis shows that the adversarial training removes the progressive loss of small-scale variance exhibited by IRENE, at the cost of an excess of fine-scale power at long lead times that the spectral penalty only partially controls.</span> <span class="abstract-toggle" data-id="2609.17175">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.17175v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.17175v1) · [:material-content-copy: BibTeX](../../bibtex/2609.17175.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=gans" data-tag="gans">GANs</a> <a class="md-tag" href="/explore/?t=recurrent-networks" data-tag="recurrent-networks">Recurrent networks</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a> <a class="md-tag" href="/explore/?t=sub-hourly" data-tag="sub-hourly">Sub-hourly</a>
    { .paper-tags }

-   #### From Nowcasting to Forecasting: Adapting a Reanalysis-Trained { #2609.03763 }

    *Mikko Partio, Leila Hieta, Ossi Laine* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.03763">Accurate cloud-cover forecasts are important for temperature prediction, radiation forecasting, and solar-power operations. Short-range forecasting methods can preserve observed cloud placement...</span><span class="abstract-full" id="full-2609.03763" hidden>Accurate cloud-cover forecasts are important for temperature prediction, radiation forecasting, and solar-power operations. Short-range forecasting methods can preserve observed cloud placement during the first forecast hours, but their skill decreases when cloud fields evolve through formation, dissipation and deformation. Longer lead times require accounting for atmospheric evolution, but operational numerical weather prediction (NWP) forecasts may not accurately represent the satellite-observed cloud state at initialization. We develop CloudCast v2, a machine-learning model for 12-hour cloud-cover forecasting from observation-based initial conditions. The model is first trained on the Copernicus European Regional Reanalysis (Ridal2024) to learn cloud-evolution dynamics, and is then adapted to satellite-derived cloud fields using conditional flow matching (Lipman2023), a generative method that transforms noise into cloud-cover forecasts conditioned on the observed initial cloud fields and NWP inputs. CloudCast v2 reduces mean absolute error by 10% relative to its predecessor, CloudCast v1 (Partio2025), over the 1-12 h range. It also overtakes CloudCast v1 in fractions skill score, a neighborhood-based measure of spatial agreement, after approximately 3-6 h, depending on the cloudiness category. These results show that observation-initialized machine-learning forecasts can extend beyond the usual 1-3-hour nowcasting range while retaining spatial detail from satellite cloud fields.</span> <span class="abstract-toggle" data-id="2609.03763">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.03763v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.03763v1) · [:material-content-copy: BibTeX](../../bibtex/2609.03763.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a>
    { .paper-tags }

-   #### GenONet: A Generative operator Network for High-Resolution Precipitation Nowcasting { #2609.00544 }

    *Mohammad Kian Golkar, Luciano Alves de Oliveira, Mohammad Khanjani* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.00544">High-resolution precipitation nowcasting is critical for reducing the impacts of severe weather but remains difficult because of rapid storm evolution. Deep learning models have shown great promise...</span><span class="abstract-full" id="full-2609.00544" hidden>High-resolution precipitation nowcasting is critical for reducing the impacts of severe weather but remains difficult because of rapid storm evolution. Deep learning models have shown great promise for this task, but their predictive skill often deteriorates over longer forecast horizons. This leads to increasingly blurry forecasts that fail to capture the complex, non-linear evolution of storm systems. In order to address these limitations, we introduce Spatio-Temporal U-DeepONet (GenONet), a novel architecture for long-range precipitation forecasting up to 3 hours, specifically designed to produce sharp and physically consistent results. GenONet's architecture pioneers the use of a Deep Operator Network (DeepONet) as a generator within a Generative Adversarial Network (GAN) framework for this task. The DeepONet learns the continuous-time dynamics of precipitation, ensuring stability over long forecast horizons. Adversial training against a spatio-temporal discriminator compels the model to produce sharp, coherent forecasts, while a physics-informed loss regularizer, derived from the Moisture Conservation Equation, improves physical plausibility in our ablation setting. Quantitative evaluations show that our model achieves consistently higher scores on most of the metrics, especially for highintensity events and at longer lead times. Qualitatively, GenONet produces structurally coherent forecasts that maintain their integrity, whereas baseline models degrade into indistinct patterns. Finally, an ablation study confirms the benefit of this physics-informed loss, highlighting the strength of combining operator learning with adversarial training.</span> <span class="abstract-toggle" data-id="2609.00544">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.00544v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.00544v1) · [:material-content-copy: BibTeX](../../bibtex/2609.00544.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=gans" data-tag="gans">GANs</a> <a class="md-tag" href="/explore/?t=neural-operators" data-tag="neural-operators">Neural operators</a> <a class="md-tag" href="/explore/?t=physics-ml-hybrid" data-tag="physics-ml-hybrid">Physics–ML hybrid</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a>
    { .paper-tags }

-   #### GOES-East full-disk AI nowcasting of cloud evolution in observation space { #2608.20540 }

    *Dhamma Kimpara, Omid Bagheri, Ivette Hernandez Banos, Byoung-Joo Jung, Chris Snyder* · Aug 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2608.20540">Clouds affect aviation, solar energy, remote sensing, and storm prediction, yet they remain among the hardest atmospheric features to forecast, particularly at convective scales. Because clouds are...</span><span class="abstract-full" id="full-2608.20540" hidden>Clouds affect aviation, solar energy, remote sensing, and storm prediction, yet they remain among the hardest atmospheric features to forecast, particularly at convective scales. Because clouds are shaped by processes spanning a wide range of space and time scales, numerical weather prediction, extrapolation methods, and existing machine learning (ML) approaches are each limited by some combination of accuracy, domain size, and temporal resolution. We present DOP+, an ML approach for clouds that forecasts GOES-East full-disk infrared brightness temperatures by extending direct observation prediction (DOP) with conditioning on meteorological fields. The domain covers tropical, midlatitude, and marine regimes across $\sim 10^8 ~ km^2$, roughly a fifth of Earth's surface. DOP+ forecasts cloud evolution at 10-minute resolution and outperforms persistence, synoptic-scale NWP, and a pure DOP baseline across 0-6 h lead times in fractions skill score and mean absolute error skill score. Convective structure is retained out to 2-3 h. DOP+ thus achieves a state-of-the-art combination of accuracy, temporal resolution, and spatial coverage. Our work lays the foundation for fully global cloud nowcasting at convective timescales.</span> <span class="abstract-toggle" data-id="2608.20540">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2608.20540v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2608.20540v1) · [:material-content-copy: BibTeX](../../bibtex/2608.20540.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=energy" data-tag="energy">Energy</a> <a class="md-tag" href="/explore/?t=sub-hourly" data-tag="sub-hourly">Sub-hourly</a>
    { .paper-tags }

-   #### Meteorology-driven Causal Nowcasting of Fugitive Landfill Emissions Enables Proactive Public Health Response { #2608.14254 }

    *Timothy C. Pearce, David J. T. Smith, Alec Dobney, Alessia Freddo* · Aug 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2608.14254">Fugitive emissions from waste sites increasingly expose communities to toxic and odorous gases, yet public-health responses remain largely retrospective, with episodes investigated only after...</span><span class="abstract-full" id="full-2608.14254" hidden>Fugitive emissions from waste sites increasingly expose communities to toxic and odorous gases, yet public-health responses remain largely retrospective, with episodes investigated only after residents have been exposed. Here we show that the meteorological drivers of elevated hydrogen sulphide (HS) at a long-monitored European landfill, and the timescales over which they act, can be identified directly from routine monitoring data. We introduce CAIRN (Causal-Anchored Inference for Receptor Nowcasting), a machine-learning framework whose internal memory is matched to these measured timescales: a fast component tracking hour-scale wind-borne transport and a slow component tracking multi-hour weather changes. Trained to predict gas measurements, CAIRN operates using only routine weather variables and the calendar, without hand-engineered features. Its behaviour is consistent with the identified transport mechanisms, and the framework transfers unchanged to a second monitoring station and to co-emitted methane. Combining four such nowcasters produces a site-level, tiered alert aligned with WHO odour guidance that closely reproduces the alert generated by a direct sensor network and tracks an independent record of community odour complaints. Weather-driven nowcasting can therefore estimate community impact as an emission episode unfolds, providing public-health authorities with a validated, graded trigger for intervention and enabling exposure to be reduced during events rather than after them.</span> <span class="abstract-toggle" data-id="2608.14254">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2608.14254v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2608.14254v1) · [:material-content-copy: BibTeX](../../bibtex/2608.14254.bib){ .bibtex-link }
    { .paper-links }

-   #### Real-Time Climate Risk Assessment for Supply Chain Resilience: A Data-Driven Nowcasting Framework for Colombian Agriculture { #2608.09846 }

    *Hernan J. Silva-Sosa* · Aug 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2608.09846">This paper presents a methodological framework for real-time climate risk assessment using data-driven nowcasting techniques to enhance supply chain resilience in Colombian agricultural contexts....</span><span class="abstract-full" id="full-2608.09846" hidden>This paper presents a methodological framework for real-time climate risk assessment using data-driven nowcasting techniques to enhance supply chain resilience in Colombian agricultural contexts. Climate variability in Colombia, characterized by irregular rainfall, temperature fluctuations, and recurrent extreme events, has a direct impact on agricultural production and logistics, particularly for time sensitive crops. The proposed approach integrates short term climate forecasting based on historical meteorological observations with supply chain risk modeling to establish a conceptual early warning system architecture. A prototype implementation developed in a controlled computational environment demonstrates the feasibility of the framework using historical meteorological and agricultural time series derived from official statistics and reanalysis products, without reliance on satellite imagery or computer vision components. The methodology addresses the integration of climate nowcasting with supply chain decision making through explicit risk mapping, threshold-based categorization, and stakeholder-oriented risk signals. Results from synthetic and historical data experiments indicate that short term precipitation nowcasts can be translated into actionable risk indicators for agricultural supply chains, supporting anticipatory decisions related to inventory, sourcing, and transport.</span> <span class="abstract-toggle" data-id="2608.09846">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2608.09846v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2608.09846v1) · [:material-content-copy: BibTeX](../../bibtex/2608.09846.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a>
    { .paper-tags }

-   #### FreCast: Refining Radar Echo Intensity via Phase-Preserving Amplitude Residual Diffusion for Precipitation Nowcasting { #2608.08436 }

    *Heping Fang, Zihuai Yin, Kaicheng Mao, Peiguang Zhang, Peng Yang* · Aug 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2608.08436">Precipitation nowcasting predicts the spatiotemporal evolution of future radar echoes from historical radar echo sequences, thereby estimating the occurrence, development, and movement of...</span><span class="abstract-full" id="full-2608.08436" hidden>Precipitation nowcasting predicts the spatiotemporal evolution of future radar echoes from historical radar echo sequences, thereby estimating the occurrence, development, and movement of precipitation over the near term. In recent years, deep learning has become an important approach to precipitation nowcasting. Although state-of-the-art models can generally capture the overall spatial distribution of future precipitation, their predictions still exhibit substantial biases in radar echo intensity at individual locations. This observation motivates a more targeted strategy for reducing forecast errors. Instead of regenerating an entire radar echo sequence without spatial constraints, the predicted precipitation structure can be used to guide the refinement of echo intensities at individual locations. This structure-guided refinement directly targets echo intensity biases. Accordingly, we propose FreCast, a two-stage framework for radar echo prediction. The first stage generates an initial forecast of future radar echoes. The second stage uses the spatial structure of the initial forecast as a constraint to further correct intensity biases at individual locations in the first-stage prediction. Experiments on three datasets demonstrate that FreCast achieves consistent improvements across forecast skill metrics. Qualitative results further show that FreCast better preserves rainband continuity and intense precipitation structures at longer lead times.</span> <span class="abstract-toggle" data-id="2608.08436">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2608.08436v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2608.08436v1) · [:material-content-copy: BibTeX](../../bibtex/2608.08436.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a>
    { .paper-tags }

-   #### Dense-Cast: A lightweight ensemble of deep learning architectures for precipitation nowcasting { #2608.06082 }

    *Gourav Jyoti Kalita, Hidam Kumarjit Singh* · Aug 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2608.06082">Proper short-term forecasting of precipitation is crucial in disaster management and preparedness. Nonetheless, the variability and nonlinearity of precipitation make short-term forecasting...</span><span class="abstract-full" id="full-2608.06082" hidden>Proper short-term forecasting of precipitation is crucial in disaster management and preparedness. Nonetheless, the variability and nonlinearity of precipitation make short-term forecasting challenging for meteorologists. Moreover, capturing temporal dependencies in spatiotemporal data is a challenge in precipitation nowcasting. In this article, we introduce a lightweight deep learning model for half-hourly precipitation nowcasting. This model has been designed by incorporating the DenseNet architecture, residual connections, and transformer encoders for effective precipitation nowcasting with reduced model parameters. The North-Eastern region of India has been selected as the area of interest for our study. The region receives the highest precipitation during the months of June-September due to the monsoon season. The proposed model takes the previous five time-steps of half-hourly precipitation as inputs and predicts the precipitation in the next two half-hours. The GPM IMERG precipitation dataset with a 30-minute cadence has been used in this study for training and testing the model. The proposed architecture achieves best MAE of 0.235 millimetres, RMSE of 0.735 millimetres, and KGE score of 0.816 at an interval of 30 minutes.</span> <span class="abstract-toggle" data-id="2608.06082">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2608.06082v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2608.06082v1) · [:material-content-copy: BibTeX](../../bibtex/2608.06082.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=transformers" data-tag="transformers">Transformers</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=efficiency" data-tag="efficiency">Efficiency</a> <a class="md-tag" href="/explore/?t=sub-hourly" data-tag="sub-hourly">Sub-hourly</a>
    { .paper-tags }

-   #### Physics-Based Deep Spatiotemporal Hyperlocal Radar Nowcasting with a Multi-Variable U-Net for High-Resolution Precipitation Forecasting { #2607.16080 }

    *Akshay Sunil, Muhammed Rashid, Raja Sekhar Sivaraju, Sushma Nair, Subimal Ghosh* · Jul 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2607.16080">Precipitation nowcasting over the immediate 10-90 min period is important for flood management and real-time decision-making in urban regions. Conventional short-range forecasting with...</span><span class="abstract-full" id="full-2607.16080" hidden>Precipitation nowcasting over the immediate 10-90 min period is important for flood management and real-time decision-making in urban regions. Conventional short-range forecasting with high-resolution numerical weather prediction requires frequent data assimilation, model initialization, and spin-up, introducing computational latency. Machine learning provides an alternative by learning storm evolution directly from high-frequency observations and producing forecasts quickly after training. This is particularly relevant for Mumbai, India, where monsoon convection, land-sea interactions, and localized intense rainfall make short-term prediction difficult. Here, we develop a compact radar-only nowcasting framework that combines multi-elevation reflectivity, Doppler radial velocity, and radial-velocity-gradient proxy features within an encoder-decoder U-Net. Using the most recent radar volume scan, the model predicts 12 future composite reflectivity fields at 7.5-min intervals up to 90 min lead time. The derived velocity magnitude, divergence-like, directional-shear, and vorticity-like channels represent kinematic signatures associated with convergence and boundary interactions without requiring full wind-field retrieval. A high-reflectivity attention module improves sensitivity to convective cores, and physics-guided attribution examines whether the learned sensitivities are meteorologically meaningful. The model is trained using Mumbai Doppler radar observations from May to August 2023 and evaluated on temporally independent events. At 90 min lead time, Critical Success Index values are 0.437, 0.332, and 0.193 for $\geq$10, $\geq$20, and $\geq$30 dBZ thresholds, respectively. Compared with persistence, the model gives lower RMSE and higher spatial correlation at longer lead times. Once trained, it runs on a standard computer, generating nowcasts within seconds for real-time use.</span> <span class="abstract-toggle" data-id="2607.16080">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2607.16080v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2607.16080v1) · [:material-content-copy: BibTeX](../../bibtex/2607.16080.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a> <a class="md-tag" href="/explore/?t=physics-ml-hybrid" data-tag="physics-ml-hybrid">Physics–ML hybrid</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=sub-hourly" data-tag="sub-hourly">Sub-hourly</a>
    { .paper-tags }

-   #### Pointwise is Pointless? A Multimodal Ablation Study for Precipitation Nowcasting with Graph Neural Networks { #2606.18436 }

    *Ophélia Miralles, Máté Mile, Christoffer Artturi, Thomas Nipen, Ivar Seierstad* · Jun 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2606.18436">Sparse point observations are increasingly available for precipitation nowcasting, but it is unclear how much they improve dense radar-field forecasts. We partially address this question with a...</span><span class="abstract-full" id="full-2606.18436" hidden>Sparse point observations are increasingly available for precipitation nowcasting, but it is unclear how much they improve dense radar-field forecasts. We partially address this question with a multimodal graph neural network nowcasting system over the Nordic radar domain. The model predicts rain rate every five minutes up to two hours ahead and is trained with different combinations of radar history, MEPS numerical weather prediction, Netatmo surface observations, MSG satellite channels, stochastic noise, and CRPS-based ensemble losses. The study is designed as an ablation of operationally relevant information sources and training objectives. We compare radar-only, NWP-informed, station-informed, satellite-informed, noise-augmented, and CRPS-based configurations using complementary diagnostics on the radar grid, at station locations, for rain onset, and through oracle, displacement, and amplitude scores. The results show that each source improves a different part of the forecast problem. MEPS stabilises radar-only extrapolation, Netatmo observations improve local station and onset diagnostics, and satellite predictors reduce some station-level biases but may activate rain too early when used deterministically. CRPS-based configurations provide the most consistent radar-grid gains, while the combined satellite and CRPS setup gives the best overall oracle/DAS score. These results do not support the conclusion that point observations are uninformative for nowcasting, but they show that local observational skill and spatially coherent radar-field skill are distinct targets. The practical implication is that sparse observations can provide useful local constraints, but their benefit for radar-like fields depends on the training loss, uncertainty representation, and how observation support is encoded in the model.</span> <span class="abstract-toggle" data-id="2606.18436">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2606.18436v2) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2606.18436v2) · [:material-content-copy: BibTeX](../../bibtex/2606.18436.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=graph-neural-networks" data-tag="graph-neural-networks">Graph neural networks</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### When the Past Matters: FlashBack Memory for Precipitation Nowcasting { #2606.16342 }

    *Yuhao Du, Boxiao Huang, Chengrong Wu, Jiankai Zhang* · Jun 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2606.16342">Accurate precipitation nowcasting is crucial for disaster mitigation and socio-economic planning, yet existing methods often struggle with false alarms, missed events, and long range dependency...</span><span class="abstract-full" id="full-2606.16342" hidden>Accurate precipitation nowcasting is crucial for disaster mitigation and socio-economic planning, yet existing methods often struggle with false alarms, missed events, and long range dependency modeling at high spatiotemporal resolution. To address these challenges, we propose FlashBack Memory (FB), a module that dynamically retrieves key historical states and integrates them via an adaptive fusion gate, enhancing the spatiotemporal representation capability of recurrent-based models. We incorporate FB into PredRNN, PredRNNpp, MIM, MotionRNN, and PredRNN-V2, and evaluate on CIKM2017, Shanghai2020, and SEVIR datasets. Experimental results demonstrate that FB significantly improves MSE, MAE, SSIM, and CSI metrics, particularly for high-intensity rainfall and long-sequence predictions, while reducing false alarms and missed events and enhancing temporal consistency and spatial localization. The proposed method provides a general and efficient memory enhancement mechanism, improving the overall performance of recurrent-based precipitation nowcasting models.</span> <span class="abstract-toggle" data-id="2606.16342">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2606.16342v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2606.16342v1) · [:material-content-copy: BibTeX](../../bibtex/2606.16342.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a>
    { .paper-tags }

-   #### Temporal Context Conditioning for Seasonality-Aware Precipitation Nowcasting of High-Intensity Rainfall { #2606.09959 }

    *Gijs van Nieuwkoop, Siamak Mehrkanoon* · Jun 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2606.09959">Precipitation nowcasting is increasingly being approached with deep learning models that learn directly from recent radar observations. Although such models can efficiently capture short-term...</span><span class="abstract-full" id="full-2606.09959" hidden>Precipitation nowcasting is increasingly being approached with deep learning models that learn directly from recent radar observations. Although such models can efficiently capture short-term precipitation motion, they often lack broader contextual information about the meteorological conditions under which rainfall develops. This paper investigates whether lightweight temporal context can improve radar-based nowcasting, particularly for high-intensity rainfall. We propose the Time-Aware Small-Attention U-Net (TA-SmaAt-UNet), which extends the core SmaAt-UNet model with temporal conditioning layers that use cyclical encodings of time-of-day and time-of-year to modulate intermediate feature representations. Experiments on KNMI radar precipitation data show that temporal conditioning is most beneficial for rare, high-intensity precipitation events, while also improving the representation of seasonal variability and predicted rainfall-intensity distributions. A layer conductance analysis further indicates that the added temporal conditioning layers are actively used by the model despite their small parameter cost. These findings suggest that simple, physically motivated temporal context can improve the realism and reliability of deep learning-based precipitation nowcasts. The implementation of our models and training setup is available on GitHub.</span> <span class="abstract-toggle" data-id="2606.09959">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2606.09959v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2606.09959v1) · [:fontawesome-brands-github: Code](https://github.com/gijsvn/TA-SmaAt-UNet) · [:material-content-copy: BibTeX](../../bibtex/2606.09959.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a>
    { .paper-tags }

-   #### Learning to Solve Generative ODEs Beyond the Linear Span { #2606.08672 }

    *Sihyeon Kim, Seunghun Lee, Vikas Singh, Hyunwoo J. Kim* · Jun 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2606.08672">Diffusion and flow generative models sample by integrating a learned ODE, but high quality still requires many sequential model evaluations. Solver learning reduces this cost by adapting scalar...</span><span class="abstract-full" id="full-2606.08672" hidden>Diffusion and flow generative models sample by integrating a learned ODE, but high quality still requires many sequential model evaluations. Solver learning reduces this cost by adapting scalar coefficients, timesteps, or both, while keeping the backbone model fixed. In this work, we identify a structural bottleneck in this update family: each step remains span-limited. Since the scalar-coefficient update lies in the span of buffered velocity evaluations, it can fit only the in-span component while leaving any out-of-span residual unreachable by scalar recombination alone. We propose SpanLift, a lightweight neural solver that augments scalar-coefficient updates with a spatial residual operator. SpanLift keeps a fixed base solver as an in-span prior and learns a spatial residual operator over the state and velocity buffer. The operator is trained by endpoint teacher matching, preserves the pretrained backbone, and adds no model NFEs. Empirically, the learned correction transfers across base solvers and is predominantly out-of-span. Across pixel-space diffusion, latent flow matching, and precipitation nowcasting, SpanLift achieves state-of-the-art few-step sampling. With only 3 NFE, it improves CIFAR-10 FID from 8.16 to 5.69 and ImageNet FID from 17.37 to 11.83.</span> <span class="abstract-toggle" data-id="2606.08672">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2606.08672v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2606.08672v1) · [:material-content-copy: BibTeX](../../bibtex/2606.08672.bib){ .bibtex-link }
    { .paper-links }

-   #### Learning to Refine: Spectral-Decoupled Iterative Refinement Framework for Precipitation Nowcasting { #2606.02661 }

    *Yunlong Zhou, Chen Zhao, Danyang Peng, Fanfan Ji, Xiao-Tong Yuan* · Jun 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2606.02661">Accurate precipitation nowcasting is vital for disaster mitigation, but deep learning methods face a key trade-off: regression models produce over-smoothed, spectrally decaying predictions that blur...</span><span class="abstract-full" id="full-2606.02661" hidden>Accurate precipitation nowcasting is vital for disaster mitigation, but deep learning methods face a key trade-off: regression models produce over-smoothed, spectrally decaying predictions that blur convective details and violate turbulence power laws; diffusion models generate realistic yet unanchored hallucinations lacking physical grounding. We propose Spectral-Decoupled Iterative Refinement (SDIR), a deterministic framework that reformulates nowcasting as progressive frequency-decoupled refinement. SDIR first extracts a stable low-frequency synoptic skeleton, then iteratively refines high-frequency textures under physical constraints, eliminating both blurring and hallucinations. It features a dual-path design: the Synoptic Frequency-Guided Former (SFG-Former) with Scale-Adaptive Transformers for global structure, and the Fourier Residual Refiner (FR-Refiner) with Scale-Conditioned Fourier Neural Operators for fine residuals. A Physically Consistent Power Spectral Density (PCPSD) loss with dynamic masking enforces a turbulence-consistent spectral distribution. Experiments on three benchmarks show SDIR significantly outperforms SOTA methods in spatial accuracy while achieving spectral fidelity competitive with diffusion-based methods, enabling reliable high-resolution operational nowcasting. Code link: https://github.com/RuntimeWarning/SDIR.</span> <span class="abstract-toggle" data-id="2606.02661">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2606.02661v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2606.02661v1) · [:fontawesome-brands-github: Code](https://github.com/RuntimeWarning/SDIR) · [:material-content-copy: BibTeX](../../bibtex/2606.02661.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=transformers" data-tag="transformers">Transformers</a> <a class="md-tag" href="/explore/?t=neural-operators" data-tag="neural-operators">Neural operators</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a>
    { .paper-tags }

-   #### Probabilistic Precipitation Nowcasting with Rectified Flow Transformers { #2605.31204 }

    *Johannes Schusterbauer, Jannik Wiese, Nick Stracke, Timy Phan, Björn Ommer* · May 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2605.31204">Accurate weather forecasts are essential across various domains and are safety-critical in extreme weather conditions. Compared to simulation-based forecasting, data-driven approaches show greater...</span><span class="abstract-full" id="full-2605.31204" hidden>Accurate weather forecasts are essential across various domains and are safety-critical in extreme weather conditions. Compared to simulation-based forecasting, data-driven approaches show greater efficiency, enabling short-term, high-resolution nowcasting. In particular, diffusion models proved effective in weather nowcasting due to their strong probabilistic foundation. However, existing methods rely on deterministic compression to reduce the complexity of high-dimensional weather data, limiting their ability to capture uncertainty in the decoding process. In this work, we introduce $\textbf{FREUD}$, a $\textbf{Fr}$ame-wise $\textbf{E}$ncoder and $\textbf{U}$nited $\textbf{D}$ecoder model based on rectified flow transformers for efficient compression of spatio-temporal weather data. Frame-wise encoding enables continuous forecast updates, while the unified video decoder ensures temporal consistency. Our uncertainty-preserving first stage allows us to capture aleatoric uncertainty via ensembling, which is particularly beneficial for extreme weather events with high decoding variability. We achieve state-of-the-art performance in precipitation nowcasting with a compact latent-space rectified flow transformer on the SEVIR benchmark and show further performance gains by model and test-time scaling. Code available here: https://github.com/CompVis/weather-rf</span> <span class="abstract-toggle" data-id="2605.31204">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2605.31204v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2605.31204v1) · [:fontawesome-brands-github: Code](https://github.com/CompVis/weather-rf) · [:material-content-copy: BibTeX](../../bibtex/2605.31204.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=transformers" data-tag="transformers">Transformers</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=extremes" data-tag="extremes">Extremes</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### Beyond MSE: Improving Precipitation Nowcasting with Multi-Quantile Regression { #2605.30122 }

    *Gijs van Nieuwkoop, Siamak Mehrkanoon* · May 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2605.30122">Deep-learning precipitation nowcasting models are often optimized using pointwise losses such as mean squared error or mean absolute error, which can lead to overly smooth forecasts and poor...</span><span class="abstract-full" id="full-2605.30122" hidden>Deep-learning precipitation nowcasting models are often optimized using pointwise losses such as mean squared error or mean absolute error, which can lead to overly smooth forecasts and poor representation of heavy rainfall. This study investigates whether the predictive performance of an established deterministic nowcasting architecture can be improved by reformulating training as a multi-quantile regression problem. Using SmaAt-UNet as a core model, we compare MSE, MAE, and multi-quantile pinball-loss training on radar precipitation nowcasting over the Netherlands. The results show that multi-quantile training improves the central deterministic forecast, decreasing test-set MSE by 8.6% compared to a model trained using MSE, while also producing upper-quantile outputs that are useful for risk-sensitive prediction of heavy precipitation. These findings suggest that quantile regression provides a simple alternative to standard pointwise losses without requiring a new architecture or generative sampling procedure. The implementation of our models and training setup is available on GitHub.</span> <span class="abstract-toggle" data-id="2605.30122">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2605.30122v2) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2605.30122v2) · [:fontawesome-brands-github: Code](https://github.com/gijsvn/Multi-Quantile-Precipitation-Nowcasting) · [:material-content-copy: BibTeX](../../bibtex/2605.30122.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a>
    { .paper-tags }

-   #### Visibility nowcasting in South Korea: a machine learning approach to class imbalance and distribution shift { #2605.21507 }

    *Bong Gyun Shin, Chan Sik Lee, Hyesun Suh* · May 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2605.21507">Atmospheric visibility is a critical variable for transportation safety and air quality management, however, accurate prediction remains challenging due to the complex interactions between...</span><span class="abstract-full" id="full-2605.21507" hidden>Atmospheric visibility is a critical variable for transportation safety and air quality management, however, accurate prediction remains challenging due to the complex interactions between meteorological conditions and air pollutants, as well as the rarity of low-visibility events. This study introduces a machine learning framework to nowcast visibility in six major South Korean cities. To handle the imbalance in the 2018-2020 training data, we applied the Synthetic Minority Over-sampling Technique with Nominal and Continuous (SMOTENC) and Conditional Tabular Generative Adversarial Network (CTGAN). An ensemble approach combining machine learning and deep learning models was then used and evaluated on a 2021 test dataset. The results revealed a marked decline in predictive performance in the test set compared to the cross-validation phase. This degradation was attributed to a distributional shift between training and testing periods, which was quantitatively confirmed by measuring the Wasserstein distance of the most influential feature identified by SHAP analysis. In general, this study presents a methodology that aims to simultaneously address the dual challenges of data imbalance and temporal distributional shifts, and emphasizes the necessity of accounting for evolving external environmental factors when implementing nowcasting models on time-series data.</span> <span class="abstract-toggle" data-id="2605.21507">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2605.21507v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2605.21507v1) · [:material-content-copy: BibTeX](../../bibtex/2605.21507.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=gans" data-tag="gans">GANs</a>
    { .paper-tags }

-   #### MambaRain: Multi-Scale Mamba-Attention Framework for 0-3 Hour Precipitation Nowcasting { #2605.14606 }

    *Chunlei Shi, Cui Wu, Xiang Xu, Hao Li, Ni Fan, Xue Han, Yongchao Feng, Yufeng Zhu, Boyu Liu et al.* · May 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2605.14606">Accurate precipitation nowcasting over extended horizons (0-3 hours) is essential for disaster mitigation and operational decision-making, yet remains a critical challenge in the field. Existing...</span><span class="abstract-full" id="full-2605.14606" hidden>Accurate precipitation nowcasting over extended horizons (0-3 hours) is essential for disaster mitigation and operational decision-making, yet remains a critical challenge in the field. Existing deterministic approaches are predominantly constrained to shorter prediction windows (0-2 hours), exhibiting severe performance degradation beyond 90 minutes owing to their inherent difficulty in capturing long-range spatiotemporal dependencies from radar-derived observations. To address these fundamental limitations, we propose MambaRain, a novel multi-scale encoder-decoder architecture that synergistically integrates Mamba's linear-complexity long-range temporal modeling with self-attention mechanisms for explicit spatial correlation capture. The core innovation lies in a hybrid design paradigm wherein Mamba blocks leverage selective state space mechanisms to model global temporal dynamics across extended sequences with computational efficiency, while self-attention modules explicitly characterize spatial correlations within precipitation fields - a capability inherently absent in Mamba's sequential processing paradigm. This complementary synergy enables comprehensive spatiotemporal representation learning, effectively extending the viable forecasting horizon to 2-3 hours with substantial accuracy improvements. Furthermore, we introduce a spectral loss formulation to mitigate blurring artifacts characteristic of chaotic precipitation systems, thereby preserving fine-scale motion details critical for nowcasting accuracy. Experimental validation demonstrates that MambaRain substantially outperforms existing deterministic methodologies in 0-3 hour nowcasting tasks, with particularly pronounced performance gains in the challenging 2-3 hour prediction range.</span> <span class="abstract-toggle" data-id="2605.14606">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2605.14606v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2605.14606v1) · [:material-content-copy: BibTeX](../../bibtex/2605.14606.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=transformers" data-tag="transformers">Transformers</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=sub-hourly" data-tag="sub-hourly">Sub-hourly</a>
    { .paper-tags }

-   #### VMU-Diff: A Coarse-to-fine Multi-source Data Fusion Framework for Precipitation Nowcasting { #2605.14597 }

    *Chunlei Shi, Hao Li, Yufeng Zhu, Boyu Liu, Yongchao Feng, Zengliang Zang, Hongbin Wang, Yanlan Yang et al.* · May 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2605.14597">Precipitation nowcasting is a vital spatio-temporal prediction task for meteorological applications but faces challenges due to the chaotic property of precipitation systems. Existing methods...</span><span class="abstract-full" id="full-2605.14597" hidden>Precipitation nowcasting is a vital spatio-temporal prediction task for meteorological applications but faces challenges due to the chaotic property of precipitation systems. Existing methods predominantly rely on single-source radar data to build either deterministic or probabilistic models for extrapolation. However, the single deterministic model suffers from blurring due to MSE convergence. The single probabilistic model, typically represented by diffusion models, can generate fine details but suffers from spurious artifacts that compromise accuracy and computational inefficiency. To address these challenges, this paper proposes a novel coarse-to-fine Vision Mamba Unet and residual Diffusion (VMU-Diff) based precipitation nowcasting framework. It realizes precipitation nowcasting through a two-stage process, i.e., a deterministic model-based coarse stage to predict global motion trends and a probabilistic model-based fine stage to generate fine prediction details. In the coarse prediction stage, rather than single-source radar data, both radar and multi-band satellite data are taken as input. A spatial-temporal attention block and several Vision mamba state-space blocks realize multi-source data fusion, and predict the future echo global dynamics. The fine-grained stage is realized by a spatio-temporal refine generator based on residual conditional diffusion models. It first obtains spatio-temporal residual features based on coarse prediction and ground truth, and further reconstructs the residual via conditional Mamba state-space module. Experiments on Jiangsu SWAN datasets demonstrate the improvements of our method over state-of-the-art methods, particularly in short-term forecasts.</span> <span class="abstract-toggle" data-id="2605.14597">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2605.14597v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2605.14597v1) · [:material-content-copy: BibTeX](../../bibtex/2605.14597.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### Spatiotemporal downscaling and nowcasting of urban land surface temperatures with deep neural networks { #2605.13566 }

    *Solomiia Kurchaba, Angela Meyer* · May 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2605.13566">Land Surface Temperature (LST) is a key variable for various applications, such as urban climate and ecology studies. Yet, existing satellite-derived LST products provide either high spatial or high...</span><span class="abstract-full" id="full-2605.13566" hidden>Land Surface Temperature (LST) is a key variable for various applications, such as urban climate and ecology studies. Yet, existing satellite-derived LST products provide either high spatial or high temporal resolution, resulting in a fundamental trade-off between the two. To address this trade-off, we combine observations from a geostationary and a polar orbiting satellite and provide LST fields at high spatial and high temporal resolution (1 km at 15-min intervals). We demonstrate their application for intraday forecasting of LSTs. To estimate LST fields at high spatiotemporal resolution, a U-Net model is trained to map LST fields from SEVIRI/MSG (3 km and 15 min resolution) to LST fields from Terra/Aqua MODIS (1 km, 4 overpasses per day) that are collocated in space and time. The presented model has been trained on LSTs across large European cities with a population exceeding 1 million inhabitants, and achieves an RMSE = $1.92$°C and near-zero bias MBE = $0.01$°C on the hold-out test set. As a second step, we present an LST nowcasting model based on ConvLSTM architecture, trained across downscaled LST fields with forecast lead times of 15 to 75 minutes. The nowcasting model outperforms a persistence and a Climatological Rolling Median benchmarks, with RMSEs of $0.57$ to $1.15$°C for the considered lead times and biases ranging from $-0.1$ to $0.14$°C. An additional validation conducted against independent MODIS overpasses confirms robust performance. Our LST forecast model at high spatiotemporal resolution is directly applicable to operational satellite-based LST monitoring.</span> <span class="abstract-toggle" data-id="2605.13566">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2605.13566v2) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2605.13566v2) · [:material-content-copy: BibTeX](../../bibtex/2605.13566.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a> <a class="md-tag" href="/explore/?t=recurrent-networks" data-tag="recurrent-networks">Recurrent networks</a> <a class="md-tag" href="/explore/?t=km-scale" data-tag="km-scale">Km-scale</a> <a class="md-tag" href="/explore/?t=sub-hourly" data-tag="sub-hourly">Sub-hourly</a>
    { .paper-tags }

-   #### McCast: Memory-Guided Latent Drift Correction for Long-Horizon Precipitation Nowcasting { #2605.13197 }

    *Penghui Wen, Yu Luo, Lintao Wang, Mengwei He, Patrick Filippi, Thomas Francis Bishop, Zhiyong Wang* · May 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2605.13197">Existing precipitation nowcasting methods typically adopt an autoregressive formulation, where future states are predicted from previous outputs. However, such an approach accumulates errors over...</span><span class="abstract-full" id="full-2605.13197" hidden>Existing precipitation nowcasting methods typically adopt an autoregressive formulation, where future states are predicted from previous outputs. However, such an approach accumulates errors over long rollouts, causing forecasts to drift away from physically plausible evolution trajectories. Although various studies have attempted to alleviate this problem by improving step-wise prediction accuracy, they largely neglect the global temporal evolution of meteorological systems and lack mechanisms to actively correct drift during rollouts. To address this issue, we propose McCast, a memory-guided latent drift correction method for precipitation nowcasting. Rather than treating memory as an unordered dictionary of latent states for passive conditioning, McCast leverages temporally organized memory to actively correct autoregressive latent evolution. Specifically, McCast introduces a Drift-Corrective Memory Bank (DCBank) that explicitly estimates the temporally consistent drift corrections to calibrate the divergent trajectory. DCBank performs drift correction in two stages: a Corrective Latent Extractor first predicts an initial correction from the current prediction and a reference latent state, and a Correction-Aware Memory Retrieval module then refines the initial correction using temporally organized historical memory. By explicitly correcting latent evolution, instead of improving step-wise prediction accuracy only, McCast produces more temporally coherent and reliable long-horizon forecasts. Experiments on two widely used benchmarks, SEVIR and MeteoNet, show that McCast achieves state-of-the-art performance, particularly in challenging long-horizon forecasting scenarios.</span> <span class="abstract-toggle" data-id="2605.13197">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2605.13197v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2605.13197v1) · [:material-content-copy: BibTeX](../../bibtex/2605.13197.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a>
    { .paper-tags }

-   #### Stable Attention Response for Reliable Precipitation Nowcasting { #2605.13181 }

    *Penghui Wen, Zexin Hu, Sen Zhang, Patrick Filippi, Xiaogang Zhu, Allen Benter, Thomas Bishop et al.* · May 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2605.13181">Precipitation nowcasting remains challenging due to the highly localized, rapidly evolving, and heterogeneous nature of atmospheric dynamics. Although recent methods increasingly adopt...</span><span class="abstract-full" id="full-2605.13181" hidden>Precipitation nowcasting remains challenging due to the highly localized, rapidly evolving, and heterogeneous nature of atmospheric dynamics. Although recent methods increasingly adopt attention-based architectures in both unimodal and multimodal settings, they mainly emphasize stronger representation learning and prediction capacity, while paying less attention to the stability of attention responses across samples. In this work, we show that cross-sample instability of attention-response energy is an important and previously underexplored source of forecasting unreliability. Empirically, inaccurate forecasts are associated with larger attention-response energy variance across heads and layers. Theoretically, we show that cross-sample variability can propagate through self-attention, and enlarge a lower bound on prediction error. Based on this insight, we propose HARECast, a Head-wise Attention Response Energy-regulated framework for precipitation nowcasting. HARECast explicitly models head-wise attention-response energy and stabilizes it through a group-wise regularization objective that reduces cross-sample fluctuations. The proposed formulation is generic and applicable to both unimodal and multimodal nowcasting architectures. We instantiate HARECast in a standard forecasting pipeline with reconstruction branches and a diffusion-based predictor, and evaluate it on commonly used benchmarks--SEVIR and MeteoNet. Experimental results demonstrate that HARECast achieves state-of-the-art performance.</span> <span class="abstract-toggle" data-id="2605.13181">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2605.13181v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2605.13181v1) · [:material-content-copy: BibTeX](../../bibtex/2605.13181.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=transformers" data-tag="transformers">Transformers</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a>
    { .paper-tags }

-   #### PixelFlowCast: Latent-Free Precipitation Nowcasting via Pixel Mean Flows { #2605.10046 }

    *Yufeng Zhu, Chunlei Shi, Yongchao Feng, Dan Niu* · May 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2605.10046">Precipitation nowcasting aims to forecast short-term radar echo sequences for extreme weather warning, where both prediction fidelity and inference efficiency are critical for real-world deployment....</span><span class="abstract-full" id="full-2605.10046" hidden>Precipitation nowcasting aims to forecast short-term radar echo sequences for extreme weather warning, where both prediction fidelity and inference efficiency are critical for real-world deployment. However, diffusion-based models, despite their strong generative capability, suffer from slow inference due to multi-step sampling trajectories, limiting their practical usability. Conditional Flow Matching (CFM) improves efficiency via straightened trajectories, but relies on latent space compression, which inevitably discards high-frequency physical details and degrades fine-grained prediction quality. To address these limitations, we propose PixelFlowCast, a two-stage probabilistic forecasting framework that achieves both high-efficiency and high-fidelity prediction without latent compression. Specifically, in the first stage, a deterministic model first produces coarse forecasts to capture global evolution trends. In the subsequent stage, the proposed KANCondNet extracts deep spatiotemporal evolution features to provide accurate conditional guidance. Based on this, a latent-free, few-step Pixel Mean Flows (PMF) predictor employs an $x$-prediction mechanism to generate high-quality predictions, effectively preserving fine-grained structures while maintaining fast inference. Experiments on the publicly available SEVIR dataset demonstrate that PixelFlowCast outperforms existing mainstream methods in both prediction accuracy and inference efficiency, particularly for long sequence forecasting, highlighting its strong potential for real-world operational deployment.</span> <span class="abstract-toggle" data-id="2605.10046">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2605.10046v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2605.10046v1) · [:material-content-copy: BibTeX](../../bibtex/2605.10046.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a>
    { .paper-tags }

-   #### IMPA-Net: Meteorology-Aware Multi-Scale Attention and Dynamic Loss for Extreme Convective Radar Nowcasting { #2604.24224 }

    *Haofei Cui, Guangxin He, Juanzhen Sun, Jingjia Luo, Haonan Chen, Xiaoran Zhuang, Mingxuan Chen et al.* · Apr 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2604.24224">Short-range prediction of convective precipitation from weather radar observations is essential for severe weather warnings. However, deep learning models trained with pixel-wise error metrics tend...</span><span class="abstract-full" id="full-2604.24224" hidden>Short-range prediction of convective precipitation from weather radar observations is essential for severe weather warnings. However, deep learning models trained with pixel-wise error metrics tend to produce overly smooth forecasts that suppress intense echoes critical for hazard detection. This issue is exacerbated by insufficient multi-scale feature interaction and suboptimal fusion of heterogeneous geophysical inputs. We propose IMPA-Net (Integrated Multi-scale Predictive Attention Network), a deterministic 0-2 hour nowcasting framework that addresses these limitations through meteorologically-informed designs at the input, architecture, and loss function levels. A parameter-free Spatial Mixer reorganizes heterogeneous input channels at the mesoscale-$γ$ neighborhood (~2 km) via deterministic channel permutation, providing a structured cross-field prior. An integrated multi-scale predictive attention module serves as the spatiotemporal translator, capturing dynamics from mesoscale-$β$ to mesoscale-$γ$ scales. A Meteorologically-Aware Dynamic Loss employs three-level asymmetric weighting -- adapting across training epochs, storm intensity, and forecast lead time -- to counteract regression-to-the-mean. Evaluated against seven baselines on a multi-source radar dataset over eastern China, IMPA-Net raises the Heidke Skill Score at $\geq$45 dBZ from 0.049 (SimVP baseline) to 0.143 under matched settings. Relative to pySTEPS, it provides a better trade-off between severe-event detection and false-alarm control. Spectral analysis confirms preserved energy across mesoscale bands where competing methods show progressive smoothing. These improvements are shown within a single domain and convective regime; generalizability to other orographic and climatic regions remains to be tested.</span> <span class="abstract-toggle" data-id="2604.24224">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2604.24224v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2604.24224v1) · [:material-content-copy: BibTeX](../../bibtex/2604.24224.bib){ .bibtex-link }
    { .paper-links }

-   #### M3R: Localized Rainfall Nowcasting with Meteorology-Informed MultiModal Attention { #2604.15377 }

    *Sanjeev Panta, Rhett M Morvant, Xu Yuan, Li Chen, Nian-Feng Tzeng* · Apr 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2604.15377">Accurate and timely rainfall nowcasting is crucial for disaster mitigation and water resource management. Despite recent advances in deep learning, precipitation prediction remains challenging due to...</span><span class="abstract-full" id="full-2604.15377" hidden>Accurate and timely rainfall nowcasting is crucial for disaster mitigation and water resource management. Despite recent advances in deep learning, precipitation prediction remains challenging due to limitations in effectively leveraging diverse multimedia data sources. We introduce M3R, a Meteorology-informed MultiModal attention-based architecture for direct Rainfall prediction that synergistically combines visual NEXRAD radar imagery with numerical Personal Weather Station (PWS) measurements, using a comprehensive pipeline for temporal alignment of heterogeneous meteorological data. With specialized multimodal attention mechanisms, M3R novelly leverages weather station time series as queries to selectively attend to spatial radar features, enabling focused extraction of precipitation signatures. Experimental results for three spatial areas of 100 km * 100 km centered at NEXRAD radar stations demonstrate that M3R outperforms existing approaches, achieving substantial improvements in accuracy, efficiency, and precipitation detection capabilities. Our work establishes new benchmarks for multimedia-based precipitation nowcasting and provides practical tools for operational weather prediction systems. The source code is available at https://github.com/Sanjeev97/M3Rain</span> <span class="abstract-toggle" data-id="2604.15377">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2604.15377v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2604.15377v1) · [:fontawesome-brands-github: Code](https://github.com/Sanjeev97/M3Rain) · [:material-content-copy: BibTeX](../../bibtex/2604.15377.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=station-point" data-tag="station-point">Station / point</a>
    { .paper-tags }

-   #### A Diffusion-Contrastive Graph Neural Network with Virtual Nodes for Wind Nowcasting in Unobserved Regions { #2604.10328 }

    *Jie Shi, Siamak Mehrkanoon* · Apr 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2604.10328">Accurate weather nowcasting remains one of the central challenges in atmospheric science, with critical implications for climate resilience, energy security, and disaster preparedness. Since it is...</span><span class="abstract-full" id="full-2604.10328" hidden>Accurate weather nowcasting remains one of the central challenges in atmospheric science, with critical implications for climate resilience, energy security, and disaster preparedness. Since it is not feasible to deploy observation stations everywhere, some regions lack dense observational networks, resulting in unreliable short-term wind predictions across those unobserved areas. Here we present a deep graph self-supervised framework that extends nowcasting capability into such unobserved regions without requiring new sensors. Our approach introduces "virtual nodes" into a diffusion and contrastive-based graph neural network, enabling the model to learn wind condition (i.e., speed, direction and gusts) in places with no direct measurements. Using high-temporal resolution weather station data across the Netherlands, we demonstrate that this approach reduces nowcast mean absolute error (MAE) of wind speed, gusts, and direction in unobserved regions by more than 30% - 46% compared with interpolation and regression methods. By enabling localized nowcasts where no measurements exist, this method opens new pathways for renewable energy integration, agricultural planning, and early-warning systems in data-sparse regions.</span> <span class="abstract-toggle" data-id="2604.10328">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2604.10328v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2604.10328v1) · [:material-content-copy: BibTeX](../../bibtex/2604.10328.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=graph-neural-networks" data-tag="graph-neural-networks">Graph neural networks</a> <a class="md-tag" href="/explore/?t=energy" data-tag="energy">Energy</a>
    { .paper-tags }

-   #### MAG-Net: Physics-Aware Multi-Modal Fusion of Geostationary Satellite and Radar for Severe Convective Precipitation Nowcasting { #2604.02818 }

    *Dandan Chen, Yaqiang Wang, Anyuan Xiong, Enda Zhu* · Apr 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2604.02818">Radar-based convective precipitation nowcasting suffers from rapid performance degradation beyond 30 minutes due to missing thermodynamic variables. Existing deep learning models also face blurring...</span><span class="abstract-full" id="full-2604.02818" hidden>Radar-based convective precipitation nowcasting suffers from rapid performance degradation beyond 30 minutes due to missing thermodynamic variables. Existing deep learning models also face blurring effects, training instability, and limited interpretability. To address this, we propose MAG-Net, a Physics-Aware Multi-modal Attention-guided Generator Network. It integrates radar dynamics with selected geostationary satellite channels (IR 10.8, WV 7.1, BTD) to incorporate thermodynamic and microphysical precursors. MAG-Net features a Dual-Stream Encoder for heterogeneous modalities and a Symmetric Dual-Head Decoder optimizing reflectivity regression and event probability via an uncertainty-weighted multi-task strategy. Furthermore, an inference-time Gradient-Preserving Fusion (GPF) strategy combines probabilistic constraints with regression details for better high-frequency texture retention. Experiments on a large-scale dataset (2018-2023) over southeastern China show MAG-Net outperforms deterministic (e.g., CPrecNet) and generative (e.g., DGMR) baselines. Specifically, it improves CSI40 by 0.083 (0.172 to 0.255) over CPrecNet, enhancing intense convective echo detection. Finally, Integrated Gradients (IG) analysis reveals the model's reliance on satellite inputs increases with forecast lead time and convective intensity, confirming that satellite data captures critical precursors for severe weather prediction.</span> <span class="abstract-toggle" data-id="2604.02818">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2604.02818v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2604.02818v1) · [:material-content-copy: BibTeX](../../bibtex/2604.02818.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=sub-hourly" data-tag="sub-hourly">Sub-hourly</a>
    { .paper-tags }

-   #### SmaAT-QMix-UNet: A Parameter-Efficient Vector-Quantized UNet for Precipitation Nowcasting { #2603.21879 }

    *Nikolas Stavrou, Siamak Mehrkanoon* · Mar 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2603.21879">Weather forecasting supports critical socioeconomic activities and complements environmental protection, yet operational Numerical Weather Prediction (NWP) systems remain computationally intensive,...</span><span class="abstract-full" id="full-2603.21879" hidden>Weather forecasting supports critical socioeconomic activities and complements environmental protection, yet operational Numerical Weather Prediction (NWP) systems remain computationally intensive, thus being inefficient for certain applications. Meanwhile, recent advances in deep data-driven models have demonstrated promising results in nowcasting tasks. This paper presents SmaAT-QMix-UNet, an enhanced variant of SmaAT-UNet that introduces two key innovations: a vector quantization (VQ) bottleneck at the encoder-decoder bridge, and mixed kernel depth-wise convolutions (MixConv) replacing selected encoder and decoder blocks. These enhancements both reduce the model's size and improve its nowcasting performance. We train and evaluate SmaAT-QMix-UNet on a Dutch radar precipitation dataset (2016-2019), predicting precipitation 30 minutes ahead. Three configurations are benchmarked: using only VQ, only MixConv, and the full SmaAT-QMix-UNet. Grad-CAM saliency maps highlight the regions influencing each nowcast, while a UMAP embedding of the codewords illustrates how the VQ layer clusters encoder outputs. The source code for SmaAT-QMix-UNet is publicly available on GitHub: https://github.com/nstavr04/MasterThesisSnellius.</span> <span class="abstract-toggle" data-id="2603.21879">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2603.21879v2) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2603.21879v2) · [:fontawesome-brands-github: Code](https://github.com/nstavr04/MasterThesisSnellius) · [:material-content-copy: BibTeX](../../bibtex/2603.21879.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=sub-hourly" data-tag="sub-hourly">Sub-hourly</a>
    { .paper-tags }

-   #### Extending Precipitation Nowcasting Horizons via Spectral Fusion of Radar Observations and Foundation Model Priors { #2603.21768 }

    *Yuze Qin, Qingyong Li, Zhiqing Guo, Wen Wang, Yan Liu, Yangli-ao Geng* · Mar 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2603.21768">Precipitation nowcasting is critical for disaster mitigation and aviation safety. However, radar-only models frequently suffer from a lack of large-scale atmospheric context, leading to performance...</span><span class="abstract-full" id="full-2603.21768" hidden>Precipitation nowcasting is critical for disaster mitigation and aviation safety. However, radar-only models frequently suffer from a lack of large-scale atmospheric context, leading to performance degradation at longer lead times. While integrating meteorological variables predicted by weather foundation models offers a potential remedy, existing architectures fail to reconcile the profound representational heterogeneities between radar imagery and meteorological data. To bridge this gap, we propose PW-FouCast, a novel frequency-domain fusion framework that leverages Pangu-Weather forecasts as spectral priors within a Fourier-based backbone. Our architecture introduces three key innovations: (i) Pangu-Weather-guided Frequency Modulation to align spectral magnitudes and phases with meteorological priors; (ii) Frequency Memory to correct phase discrepancies and preserve temporal evolution; and (iii) Inverted Frequency Attention to reconstruct high-frequency details typically lost in spectral filtering. Extensive experiments on the SEVIR and MeteoNet benchmarks demonstrate that PW-FouCast achieves state-of-the-art performance, effectively extending the reliable forecast horizon while maintaining structural fidelity. Our code is available at https://github.com/Onemissed/PW-FouCast.</span> <span class="abstract-toggle" data-id="2603.21768">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2603.21768v3) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2603.21768v3) · [:fontawesome-brands-github: Code](https://github.com/Onemissed/PW-FouCast) · [:material-content-copy: BibTeX](../../bibtex/2603.21768.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=foundation-models" data-tag="foundation-models">Foundation models</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a>
    { .paper-tags }

-   #### PA-Net: Precipitation-Adaptive Mixture-of-Experts for Long-Tail Rainfall Nowcasting { #2603.13818 }

    *Xinyu Xiao, Sen Lei, Eryun Liu, Shiming Xiang, Hao Li, Cheng Yuan, Yuan Qi, Qizhao Jin* · Mar 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2603.13818">Precipitation nowcasting is vital for flood warning, agricultural management, and emergency response, yet two bottlenecks persist: the prohibitive cost of modeling million-scale spatiotemporal tokens...</span><span class="abstract-full" id="full-2603.13818" hidden>Precipitation nowcasting is vital for flood warning, agricultural management, and emergency response, yet two bottlenecks persist: the prohibitive cost of modeling million-scale spatiotemporal tokens from multi-variate atmospheric fields, and the extreme long-tailed rainfall distribution where heavy-to-torrential events -- those of greatest societal impact -- constitute fewer than 0.1% of all samples. We propose the Precipitation-Adaptive Network (PA-Net), a Transformer framework whose computational budget is explicitly governed by rainfall intensity. Its core component, Precipitation-Adaptive MoE (PA-MoE), dynamically scales the number of activated experts per token according to local precipitation magnitude, channeling richer representational capacity toward the rare yet critical heavy-rainfall tail. A Dual-Axis Compressed Latent Attention mechanism factorizes spatiotemporal attention with convolutional reduction to manage massive context lengths, while an intensity-aware training protocol progressively amplifies learning signals from extreme-rainfall samples. Experiment on ERA5 demonstrate consistent improvements over state-of-the-art baselines, with particularly significant gains in heavy-rain and rainstorm regimes.</span> <span class="abstract-toggle" data-id="2603.13818">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2603.13818v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2603.13818v1) · [:material-content-copy: BibTeX](../../bibtex/2603.13818.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=transformers" data-tag="transformers">Transformers</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a>
    { .paper-tags }

</div>

<nav class="pager" markdown="span">**1** [2](2.md) [3](3.md) [4](4.md) [Older :material-arrow-right:](2.md){ .pager-step }</nav>

