---
title: 'Downscaling'
hide:
  - toc
---

<div class="listing-header" markdown>

# Downscaling

<p class="page-meta" markdown="span">95 papers · page 1 of 4 · <a href="../../bib/downscaling.bib" download>:material-download: BibTeX for this topic</a></p>

</div>

<div class="grid cards" markdown>

-   #### Diffusion-Based Super-Resolution of Adriatic Sea Oceanographic Fields { #2609.22574 }

    *Rajat Srivastava, Muhammad Sarmad, Emanuele Mele, Massimo Cafaro, Marco Pulimeno, Italo Epicoco* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.22574">High-resolution oceanographic fields are critical for resolving mesoscale and sub-mesoscale coastal dynamics, yet their generation remains constrained by both computational cost and observational...</span><span class="abstract-full" id="full-2609.22574" hidden>High-resolution oceanographic fields are critical for resolving mesoscale and sub-mesoscale coastal dynamics, yet their generation remains constrained by both computational cost and observational sparsity. We present OcDiffSR, a conditional denoising diffusion probabilistic model (DDPM) for oceanographic super-resolution that reconstructs high-resolution sea-surface fields from coarse-resolution reanalysis inputs. The model is trained on ten years (2011-2020) of paired low-resolution (GLORYS12V1, 1/12) and high-resolution (Mediterranean Sea Physics Reanalysis, Med MFC, 1/24) data, and evaluated on an independent test year (2009) over the Adriatic Sea. OcDiffSR employs a conditional U-Net augmented with multi-scale low-resolution encoders, cross-attention bottleneck layers, and sinusoidal seasonal embeddings via Feature-wise Linear Modulation (FiLM), enabling joint super-resolution of sea-surface temperature (SST), salinity (SSS), and horizontal velocity components with visually coherent circulation patterns. Benchmarked against bilinear interpolation and the state-of-the-art residual diffusion model CorrDiff, OcDiffSR achieves substantially lower reconstruction errors for scalar fields (RMSESST=0.477 C, RMSESSS=0.346 psu), near-unity Pearson correlation (PCC >= 0.999), and high structural similarity (SSIM >= 0.964). For dynamical vector fields, OcDiffSR outperforms both baselines in absolute error and spatial coherence, though moderate correlation (PCC = 0.64) reflects the intrinsic stochasticity of oceanic velocity fields. Daily and monthly evaluations confirm temporal robustness across all seasons. These results establish OcDiffSR as a reliable framework for high-fidelity oceanographic downscaling and reanalysis enhancement, producing fields that are visually consistent with known ocean dynamics.</span> <span class="abstract-toggle" data-id="2609.22574">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.22574v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.22574v1) · [:material-content-copy: BibTeX](../../bibtex/2609.22574.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a> <a class="md-tag" href="/explore/?t=daily" data-tag="daily">Daily</a> <a class="md-tag" href="/explore/?t=monthly" data-tag="monthly">Monthly</a>
    { .paper-tags }

-   #### Steering Diffusion Priors with Sparse Observations for High-Resolution Temperature Downscaling { #2609.09247 }

    *Anirudh Avireddy, Manmeet Singh, Shivanshi Singh, Ayush Raj, Saptarishi Dhanuka et al.* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.09247">Local heatwave hazard depends on fine-scale air temperature, but ground stations are sparse and reanalysis products such as ERA5 cannot resolve the terrain and land-surface contrasts that shape real...</span><span class="abstract-full" id="full-2609.09247" hidden>Local heatwave hazard depends on fine-scale air temperature, but ground stations are sparse and reanalysis products such as ERA5 cannot resolve the terrain and land-surface contrasts that shape real heat exposure. We present a conditional diffusion emulator for high-resolution 2-m temperature downscaling, conditioned on static geography, a training climatology, exact-time ERA5 temperature, and solar and temporal features, guided at inference by score-based data assimilation (SDA): a differentiable Gaussian observation likelihood steers the diffusion score toward sparse revealed temperature observations without any retraining. On a controlled 32-case synthetic-grid protocol over AORC, guidance improves hidden-cell reconstruction over both ERA5 and a strong observation-proximal nearest-neighbor baseline once observation density reaches 1% (RMSE 0.318 vs.\ 0.431~K, winning all 32 cases), while sparser regimes still favor direct interpolation. We further map the full guidance-strength landscape across three observation densities, showing that the optimal strength shifts systematically with density and that over-guiding causes sharp, predictable degradation -- giving a concrete operating recipe rather than a single untuned setting. The resulting fields are intended as a temperature layer for downstream heatwave-hazard products such as threshold exceedance and cumulative heat-burden. The present evidence is a controlled synthetic-grid validation; station-network and held-out-year evaluations are the next steps toward deployment.</span> <span class="abstract-toggle" data-id="2609.09247">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.09247v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.09247v1) · [:material-content-copy: BibTeX](../../bibtex/2609.09247.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a>
    { .paper-tags }

-   #### Python-Fortran Hybrid Programming to Fuse AI and Physical Models: Examples of AI-LDA in climate and weather models (Hf2pMDA_v1.0) { #2608.29532 }

    *Xianrui Zhu, Zikuan Lin, Shaoqing Zhang, Zebin Lu, Songhua Wu, Xiangyun Hou, Zhisheng Xiao et al.* · Aug 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2608.29532">AI provides an unprecedented opportunity for advancing physics numerical modeling including data assimilation, which is a highly efficient and critically-important tool for advancing our...</span><span class="abstract-full" id="full-2608.29532" hidden>AI provides an unprecedented opportunity for advancing physics numerical modeling including data assimilation, which is a highly efficient and critically-important tool for advancing our understanding on Earth system and its applications. At the same time, deep incorporation of AI and physical modeling can make great driving to advance AI by injecting it rich physics from long time physics-based modeling development. However, since such physics models are conventionally coded in Fortran and AI algorithms usually are conveniently designed in Python, difficulties exist to directly incorporate AI algorithms into physics models, vice versa. Here, based on the F2PY protocol, we have developed a procedure that implements an infrastructure which conveniently conducts Hf2pMDA to form a program entity so that AI algorithms and physical models can invoke mutually. As examples, within Hf2pMDA, a climate coupled data assimilation (CDA) system is naturally upgraded to a strongly CDA (SCDA) system, and a 1 km high-resolution weather DA system is conveniently implemented within a multi-layer downscaling model that has multiscale DA in different nesting layers. In the climate SCDA system, a coupled general circulation model (CGCM) and a multiscale filtering algorithm is integrated by a Python main controller (PMC) that calls Fortran CGCM components and Weakly-CDA modules as well as a data-trained SCDA algorithm by latent space autoencoder in Python. In the high-resolution weather DA system, the downscaled model consisting of traditional Fortran DA modules in all mother domains and Python AE DA algorithm in the central child domain is integrated by a PMC that organizes these components. With convenient realization of deep incorporation of any AI algorithm and physics model, the Hf2pMDA has a great potential to make progress on both AI and scientific modeling.</span> <span class="abstract-toggle" data-id="2608.29532">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2608.29532v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2608.29532v1) · [:material-content-copy: BibTeX](../../bibtex/2608.29532.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=km-scale" data-tag="km-scale">Km-scale</a>
    { .paper-tags }

-   #### Precipitation Downscaling Using Foundation Model-Conditioned Diffusion { #2608.25858 }

    *Victor Nascimento Ribeiro, Jorge Guevara, Jorge Sebastian Moraga, Chris Lucas, Natalie Lord et al.* · Aug 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2608.25858">High-resolution precipitation fields are essential for hydrological impact assessment, yet global climate model outputs are too coarse and biased for direct use. AI-based statistical downscaling with...</span><span class="abstract-full" id="full-2608.25858" hidden>High-resolution precipitation fields are essential for hydrological impact assessment, yet global climate model outputs are too coarse and biased for direct use. AI-based statistical downscaling with diffusion models offers a promising approach, but the mechanism by which large-scale atmospheric predictors condition generation remains largely unexplored. We investigate three conditioning strategies for a denoising diffusion probabilistic model applied to daily precipitation downscaling: channel concatenation of upsampled coarse predictors, cross-attention conditioning with a learned convolutional encoder, and cross-attention conditioning with the frozen encoder of the pretrained Prithvi WxC weather foundation model. All strategies are evaluated against an unconditioned baseline under identical conditions using probabilistic, distributional, spectral, and extreme-event metrics for the Colorado River Basin. Concatenation conditioning achieves the lowest point-wise CRPS and MSE, but tends to produce over-smoothed fields that suppress high-intensity events. In contrast, cross-attention conditioning provides substantially better distributional realism and modest improvements in spectral fidelity. Improvements are greatest for extremes: the Prithvi-WxC conditioned model retains over half of >100mm/day events, although estimates are uncertain due to limited samples. When trained on the full dataset, the learned convolutional model performs similarly to the foundation model-conditioned approach while requiring lower computational resources. However, the Prithvi-WxC-conditioned model achieves comparable performance with only five years of training data. These results indicate that cross-attention conditioning offers advantages over simple concatenation for probabilistic precipitation downscaling, and that pre-trained foundation model representations may offer benefits in data-limited settings.</span> <span class="abstract-toggle" data-id="2608.25858">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2608.25858v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2608.25858v1) · [:material-content-copy: BibTeX](../../bibtex/2608.25858.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a> <a class="md-tag" href="/explore/?t=foundation-models" data-tag="foundation-models">Foundation models</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a> <a class="md-tag" href="/explore/?t=daily" data-tag="daily">Daily</a>
    { .paper-tags }

-   #### Earth observation embeddings are effective sub-grid descriptors for probabilistic weather downscaling { #2608.12271 }

    *Pedro Sousa, Will Tebbutt, Sadiq Jaffer, Robin Young, Anil Madhavapeddy, Richard E. Turner* · Aug 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2608.12271">Global weather reanalyses and forecasts resolve the evolving atmospheric state on coarse grids, but site-specific applications require predictions at arbitrary locations where near-surface conditions...</span><span class="abstract-full" id="full-2608.12271" hidden>Global weather reanalyses and forecasts resolve the evolving atmospheric state on coarse grids, but site-specific applications require predictions at arbitrary locations where near-surface conditions also depend on unresolved terrain and land-surface properties. Existing probabilistic downscalers address this gap using hand-crafted topographic descriptors. We ask instead whether Earth observation foundation models can provide transferable sub-grid surface representations for probabilistic weather downscaling.   We augment a convolutional conditional neural process that downscales coarse ERA5 reanalysis fields at ~25 km resolution with a learned local surface descriptor, obtained by compressing a patch of TESSERA embeddings at 10 m resolution. Although these embeddings summarise surface conditions over annual timescales, they improve downscaling of instantaneous 2 m temperature and 10 m wind speed by encoding persistent surface properties that capture a location's departure from the coarse-grid atmospheric state. Across five climatically diverse regions, the embedding improves point and probabilistic skill at stations held out in both space and time, overall improving CRPS skill by 11.5% for 2 m temperature and 6.2% for 10 m wind speed. We further analyse how its contribution differs by variable, finding that topography explains more of temperature's sub-grid structure, while TESSERA provides additional surface information for wind speed.   These improvements persist when the coarse input is changed from ERA5 to forecasts from the Aurora AI forecasting model, and when predicting at newly deployed stations with no regional history. To our knowledge, this is the first evidence that long-timescale Earth-observation embeddings can support short-timescale weather downscaling where sub-grid departures are systematically structured by persistent surface properties.</span> <span class="abstract-toggle" data-id="2608.12271">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2608.12271v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2608.12271v1) · [:material-content-copy: BibTeX](../../bibtex/2608.12271.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=foundation-models" data-tag="foundation-models">Foundation models</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a> <a class="md-tag" href="/explore/?t=station-point" data-tag="station-point">Station / point</a> <a class="md-tag" href="/explore/?t=quarter-degree" data-tag="quarter-degree">0.25°</a>
    { .paper-tags }

-   #### Deep Learning-Based Statistical Downscaling of Sea Surface Temperature Using a Residual Corrective Neural Network { #2608.10022 }

    *Onkar Jadhav, Tim French, Ivica Janekovic, Nicole L. Jones, Matthew Rayson* · Aug 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2608.10022">The large-scale oceanic and atmospheric forecasts provided by global climate models typically lack sufficient resolution to accurately capture the response of the coastal ocean to atmospheric forcing...</span><span class="abstract-full" id="full-2608.10022" hidden>The large-scale oceanic and atmospheric forecasts provided by global climate models typically lack sufficient resolution to accurately capture the response of the coastal ocean to atmospheric forcing and coastal circulation that drive fine-scale SST variability. Dynamical downscaling is computationally prohibitive, when applied to extensive coastlines, predictive ensembles, or long time periods. Therefore, this work presents a statistical downscaling of sea surface temperature (SST) from the seasonal coupled ocean-atmosphere forecast system (ACCESS-S2) using machine learning techniques. This study proposes a novel deep learning framework that uses a U-Net to generate an initial high-resolution SST estimate, which is subsequently refined using a residual corrective approach. The target SST fields are derived from the Regional Ocean Modeling System (ROMS). This two step approach called Residual Corrective Neural Network (RCNN) progressively refines initial U-Net predictions by incorporating dynamically scaled residuals at each step, enabling accurate capture of broad patterns and fine-grained features such as eddies and fronts. We also introduce a custom loss-assisted RCNN variant to improve performance during extreme events, which may be absent from training data due to climate-driven shifts in SST extremes. The framework efficiently downscales SST along the west coast of Australia. A 2011 marine heatwave case study shows that the RCNN improves ACCESS-S2 SST predictions by increasing horizontal resolution from 25 km to 2 km, enabling identification of fine-scale anomalies unresolved in the ACCESS-S2 dataset. This balance between computational efficiency and accuracy supports applications in coastal impact assessment and marine ecosystem studies.</span> <span class="abstract-toggle" data-id="2608.10022">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2608.10022v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2608.10022v1) · [:material-content-copy: BibTeX](../../bibtex/2608.10022.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a> <a class="md-tag" href="/explore/?t=extremes" data-tag="extremes">Extremes</a> <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a> <a class="md-tag" href="/explore/?t=km-scale" data-tag="km-scale">Km-scale</a> <a class="md-tag" href="/explore/?t=quarter-degree" data-tag="quarter-degree">0.25°</a>
    { .paper-tags }

-   #### Temporal Bridges for Spatial Resolution: Enhancing Climate Data Super-Resolution with Bidirectional Alignment { #2608.05981 }

    *Yichen Zhang, Yixiong Xiao, Congxi Xiao, Jingbo Zhou* · Aug 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2608.05981">High-resolution climate data is crucial for meteorological predictions and for informing decision support across diverse domains. However, the acquisition of such high-resolution climate information...</span><span class="abstract-full" id="full-2608.05981" hidden>High-resolution climate data is crucial for meteorological predictions and for informing decision support across diverse domains. However, the acquisition of such high-resolution climate information is often prohibitively costly, necessitating the development of data-driven meteorological prediction models. These models aim to generate fine-grained climate data from low-resolution inputs, a process termed climate data super-resolution (SR). Nevertheless, recent advancements in deep learning for climate data SR have primarily focused on leveraging single-frame spatial information, largely neglecting the temporal correlations between different time frames that could enhance SR outcomes. Furthermore, climate data are inherently stochastic and noisy, rendering widely used temporal alignment methods, such as optical flow models, ineffective in this context. Consequently, the development of a framework tailored for climate data SR that effectively captures implicit temporal correlations remains an unresolved challenge. To this end, we propose a novel Temporal-Enhanced framework with bidirectional temporal alignment. In essence, our framework establishes a temporal bridge to enhance spatial resolution in climate data SR through bidirectional alignment, leading to improved SR performance. Within this framework, Paired Latent Mapping achieves spatial alignment and noise reduction by unifying latent spaces. Then a Bidirectional Temporal Alignment captures temporal correlations by training forward and backward networks on consecutive latent frames. Temporal Enhanced Super-resolution then optimizes the entire framework for climate data SR. Experiments on large-scale real-world datasets demonstrated the superior performance of our framework.</span> <span class="abstract-toggle" data-id="2608.05981">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2608.05981v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2608.05981v1) · [:material-content-copy: BibTeX](../../bibtex/2608.05981.bib){ .bibtex-link }
    { .paper-links }

-   #### Transferable Dual-Stream Representations for Mesoscale-Preserving Sea Surface Temperature Downscaling { #2608.04230 }

    *Parth Doshi, Priyanka Aravindan, Vaishnav Vaidheeswaran, Md Mahbub Alam, Gabriel Spadon* · Aug 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2608.04230">Deep learning models for scientific spatio-temporal downscaling often minimize reconstruction error while failing to preserve physically meaningful multi-scale structure. For sea surface temperature...</span><span class="abstract-full" id="full-2608.04230" hidden>Deep learning models for scientific spatio-temporal downscaling often minimize reconstruction error while failing to preserve physically meaningful multi-scale structure. For sea surface temperature prediction, this can yield outputs that are numerically plausible yet overly smooth, missing mesoscale variability critical to regional ocean dynamics. Existing methods often focus on pixel-wise objectives or single-context conditioning, which limits their ability to preserve spectral fidelity and generalize across regions. To address this, we propose EddyFlow, a representation learning framework for kilometer-scale sea surface temperature downscaling that balances predictive accuracy, scale-dependent structure, and regional generalization. EddyFlow is trained on the Gulf of St.~Lawrence and evaluated in zero-shot and few-shot settings on the Bay of Fundy and the Gulf of Mexico. EddyFlow demonstrates that physics-informed representation learning reduces zero-shot RMSE by 21%, achieves up to 85.6% skill relative to persistence on unseen domains, and maintains near-ideal spectral fidelity with a PSD ratio of $\approx 1.00$.</span> <span class="abstract-toggle" data-id="2608.04230">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2608.04230v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2608.04230v1) · [:material-content-copy: BibTeX](../../bibtex/2608.04230.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=physics-ml-hybrid" data-tag="physics-ml-hybrid">Physics–ML hybrid</a> <a class="md-tag" href="/explore/?t=km-scale" data-tag="km-scale">Km-scale</a>
    { .paper-tags }

-   #### Physics-Informed Super-Resolution of Atmospheric Data { #2607.18877 }

    *Chang Xu, Gencer Sumbul, Hugo Porta, Manon Béchaz, Sebastian Schemm, Devis Tuia* · Jul 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2607.18877">In the context of global warming, extreme events have become more frequent and intense, making their trustworthy detection and forecasting more important than ever. Yet, atmospheric observations lack...</span><span class="abstract-full" id="full-2607.18877" hidden>In the context of global warming, extreme events have become more frequent and intense, making their trustworthy detection and forecasting more important than ever. Yet, atmospheric observations lack sufficient spatial resolution, motivating atmospheric data downscaling as a way to reconstruct high-resolution data from coarse observations. This task is now being formulated as a super-resolution (SR) problem with machine learning methods featuring high efficiency. Nevertheless, it remains unclear whether the super-resolved atmospheric data still satisfies fundamental physics governing the Earth system, raising concerns about their trustworthiness in climate-related applications. In this work, we address this challenge by constraining SR models to respect hydrostatic primitive equations that represent multivariate atmospheric physics. First, we propose a Physics-Informed Super-Resolution (PISR) method involving multi-scale physics-informed objectives based on primitive equations. PISR favors the SR outputs to respect these equations and therefore naturally encodes inter-variable relationships. In addition, we propose a metric called Normalized Physical Consistency (NPC) derived from said primitive equations to measure the physical consistency of super-resolved data. Experiments on ERA5, CERRA, and COSMO demonstrate that PISR enhances the reconstruction fidelity by improving physical consistency, SR accuracy, and downstream detection of extreme events, as demonstrated by case studies in heatwaves and extreme winds.</span> <span class="abstract-toggle" data-id="2607.18877">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2607.18877v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2607.18877v1) · [:material-content-copy: BibTeX](../../bibtex/2607.18877.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=physics-ml-hybrid" data-tag="physics-ml-hybrid">Physics–ML hybrid</a> <a class="md-tag" href="/explore/?t=extremes" data-tag="extremes">Extremes</a>
    { .paper-tags }

-   #### Super-Resolution of Radar/Raingauge-Analyzed Precipitation Using Gaussian Process Regression with a Steering Kernel { #2607.07290 }

    *Shoichi Akami, Tsuyoshi T. Sekiyama, Mizuo Kajino* · Jul 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2607.07290">Super-resolution estimates a high-resolution image from a low-resolution image and has been used for downscaling and resolution enhancement of observations in meteorology. Super-resolution Gaussian...</span><span class="abstract-full" id="full-2607.07290" hidden>Super-resolution estimates a high-resolution image from a low-resolution image and has been used for downscaling and resolution enhancement of observations in meteorology. Super-resolution Gaussian process regression with a steering kernel (SRGP-SK) generates more accurate high-resolution images than super-resolution Gaussian process regression, but it has not yet been applied in meteorology. In this study, we applied SRGP-SK to radar/raingauge-analyzed precipitation for a convective case and a stratiform case, and evaluated the results using the structural similarity index (SSIM) and the radially averaged power spectral density (PSD). SRGP-SK achieved the SSIM comparable to that of bicubic interpolation while reconstructing finer precipitation structures: it reconstructed variations down to a wavelength of 6 km, whereas bicubic interpolation reconstructed variations only down to 8 km. We further compared several kernel functions and found that the kernel optimal for SSIM differed from that optimal for the geometric mean PSD ratio, reflecting the different properties that the two measures quantify. This is the first study to demonstrate the usefulness of SRGP-SK in meteorology, and it represents a step toward super-resolution with physical interpretability.</span> <span class="abstract-toggle" data-id="2607.07290">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2607.07290v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2607.07290v1) · [:material-content-copy: BibTeX](../../bibtex/2607.07290.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=classical-ml" data-tag="classical-ml">Classical ML</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a>
    { .paper-tags }

-   #### Domain-Adaptive Climate Downscaling Under Temporal Distribution Shift { #2607.05645 }

    *Shuochen Wang, Nishant Yadav, Auroop R. Ganguly* · Jul 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2607.05645">Deep-learning-based climate downscaling aims to learn relationships from historical low-resolution (LR) and high-resolution (HR) climate data to generate HR climate projections. However, this setting...</span><span class="abstract-full" id="full-2607.05645" hidden>Deep-learning-based climate downscaling aims to learn relationships from historical low-resolution (LR) and high-resolution (HR) climate data to generate HR climate projections. However, this setting faces a temporal out-of-distribution (OOD) challenge: models trained on historical data are commonly applied to future projections whose distributions may differ substantially from the training period. This study investigates temporal OOD shift for daily temperature downscaling over the Continental United States using paired LR-HR model simulations. We propose a temporal domain-adaptive downscaling framework that combines supervised HR reconstruction on historical data with domain alignment between historical and future climate distributions. Experiments across future validation periods show that the proposed domain-adaptive model consistently outperforms statistical and deep-learning-based bias-correction methods, with the largest gains occurring when the temporal distribution shift is strongest. Spatial analyses indicate stronger improvements over high-elevation and topographically complex regions, along with higher spatiotemporal correlation with the HR target. The extreme analysis shows that domain adaptation also reduces upper-tail temperature bias relative to the non-adaptive model. These results demonstrate that temporal domain adaptation can improve the robustness of HR climate projections under non-stationary climate conditions.</span> <span class="abstract-toggle" data-id="2607.05645">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2607.05645v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2607.05645v1) · [:material-content-copy: BibTeX](../../bibtex/2607.05645.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a> <a class="md-tag" href="/explore/?t=daily" data-tag="daily">Daily</a>
    { .paper-tags }

-   #### Exploring Convolutional Neural Processes for Weather Downscaling { #2607.04190 }

    *Francisco Passos* · Jul 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2607.04190">Global reanalysis products such as ERA5-Land provide spatially complete weather fields but at resolutions too coarse for local applications, particularly in mountainous regions where temperature can...</span><span class="abstract-full" id="full-2607.04190" hidden>Global reanalysis products such as ERA5-Land provide spatially complete weather fields but at resolutions too coarse for local applications, particularly in mountainous regions where temperature can vary by several degrees over short distances. This project investigates Convolutional Conditional Neural Processes (ConvCNPs) for statistical downscaling of daily maximum temperature from the ~11km resolution ERA5-Land grid to ~1km resolution over Switzerland, building upon the architecture of Vaughan et al. (2022) and adapting it to the topographically complex Swiss domain with high-resolution elevation features from the swisstopo DHM25. The best model, trained on ten years of data (2014-2023) with five-fold temporal cross-validation, achieves a mean absolute error of 1.31 Celsius and a CRPS-based skill score of 0.524 relative to bilinear interpolation, reducing the expected prediction error by more than half. An ablation study reveals that the elevation MLP is the indispensable component - without it, the model diverges entirely - while explicit seasonal features and Topographic Position Index provide secondary benefits. Under sparse on-grid input the model degrades gracefully, maintaining positive skill down to approximately 10% of the input grid; however, zero-shot deployment on off-grid station observations does not achieve positive skill at any density tested. All configurations exhibit severely overconfident uncertainty estimates, a structural limitation of the Gaussian likelihood training objective. These results demonstrate that ConvCNPs are a viable and effective approach to climate downscaling in complex terrain, and identify uncertainty calibration and native support for non-gridded input as the key challenges for operational deployment.</span> <span class="abstract-toggle" data-id="2607.04190">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2607.04190v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2607.04190v1) · [:material-content-copy: BibTeX](../../bibtex/2607.04190.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a> <a class="md-tag" href="/explore/?t=km-scale" data-tag="km-scale">Km-scale</a> <a class="md-tag" href="/explore/?t=daily" data-tag="daily">Daily</a>
    { .paper-tags }

-   #### CORDEX-ML-Bench: A Benchmark for Data-Driven Regional Climate Downscaling -Experiment Design and Overview { #2606.29172 }

    *Neelesh Rampal, José González-Abad, Henry Addison, Jorge Baño-Medina, Maria Laura Bettolli et al.* · Jun 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2606.29172">Machine learning (ML) has emerged as a cost-effective approach to complement dynamical downscaling for producing high-resolution regional climate projections. However, the absence of standardised...</span><span class="abstract-full" id="full-2606.29172" hidden>Machine learning (ML) has emerged as a cost-effective approach to complement dynamical downscaling for producing high-resolution regional climate projections. However, the absence of standardised training and evaluation protocols, applied consistently across multiple domains, continues to hinder meaningful model intercomparison. We introduce CORDEX-ML-Bench, a benchmark aligned with CORDEX, which constitutes the first phase of a community initiative to advance data-driven downscaling toward operational readiness, and complement future dynamical downscaling efforts under CMIP7. The framework targets downscaled daily maximum temperature and precipitation to ~10 km resolution (20x increase) across three pilot regions; European Alps, New Zealand, and Southern Africa. Using a perfect-model experimental design, we evaluate 40 ML configurations developed independently, spanning traditional ML, convolutional U-Nets, vision transformers, graph neural networks, and generative models based on diffusion, flow matching, and generative adversarial networks. Models are trained under two experimental periods, an empirical-statistical downscaling pseudo-reality (historical period only) and Emulator (historical and future periods) -and are evaluated against a core set of metrics developed specifically for assessing downscaling skill. Generative models consistently outperform deterministic approaches for precipitation, better capturing fine-scale variability and extremes. For temperature, the generative advantage narrows and deterministic architectures remain competitive. Models trained solely on the historical period systematically underestimate future climate-change signals while those additionally trained on a future period perform better. These findings raise concerns about historically trained models widely used in an operational setting, underscoring the need for rigorous extrapolation testing.</span> <span class="abstract-toggle" data-id="2606.29172">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2606.29172v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2606.29172v1) · [:material-content-copy: BibTeX](../../bibtex/2606.29172.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a> <a class="md-tag" href="/explore/?t=daily" data-tag="daily">Daily</a>
    { .paper-tags }

-   #### Temporal Coverage over Density: Parsimonious Training-Set Design for ML Climate Downscaling { #2606.07898 }

    *Karandeep Singh, Stefan Rahimi, Chad W. Thackeray, Stephen Cropper, Alex Hall* · Jun 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2606.07898">High-resolution regional climate simulations provide critical information for climate impacts assessments but remain computationally expensive, motivating the development of machine-learning...</span><span class="abstract-full" id="full-2606.07898" hidden>High-resolution regional climate simulations provide critical information for climate impacts assessments but remain computationally expensive, motivating the development of machine-learning downscalers and emulators. A key challenge is determining how limited high-resolution simulations should be distributed across a changing climate trajectory to capture both forced climate response and internal variability. Using the CESM2 Large Ensemble over the western United States, we compare three training-year selection strategies under fixed data budgets: a contiguous block of historical years, years drawn from both the beginning and end of the simulation period, and years distributed throughout the full climate trajectory. Including both historical and future years consistently outperforms training on historical years alone, demonstrating the importance of exposing downscaling models to climate states outside the historical record and highlighting limitations of stationarity assumptions common in statistical downscaling. Training on years distributed throughout the full climate trajectory performs best overall, indicating that broad sampling of internal variability provides additional information beyond exposure to the forced climate response alone. Models trained on temporally distributed subsets more successfully reproduce variability in unseen ensemble members while retaining strong performance across a wide range of climate diagnostics. Even when trained on only one-tenth of the available high-resolution years, temporally distributed models remain highly competitive with full-data training. These results suggest that, under fixed computational budgets, broad sampling of climate states is more valuable than temporal continuity when allocating scarce high-resolution simulations. The findings provide practical guidance for regional climate downscaling and large-ensemble projection workflows.</span> <span class="abstract-toggle" data-id="2606.07898">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2606.07898v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2606.07898v1) · [:material-content-copy: BibTeX](../../bibtex/2606.07898.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a>
    { .paper-tags }

-   #### Flow Matching for Convective-Scale Precipitation Downscaling { #2606.00281 }

    *Tom Wetherell* · Jun 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2606.00281">Generative machine learning is an increasingly important complement to dynamical downscaling for producing high-resolution precipitation projections, with diffusion models currently the leading...</span><span class="abstract-full" id="full-2606.00281" hidden>Generative machine learning is an increasingly important complement to dynamical downscaling for producing high-resolution precipitation projections, with diffusion models currently the leading approach. Flow matching is a related generative framework that has recently achieved strong results across image, video and other domains, and shown early promise for downscaling. We train a flow matching model to map daily precipitation from 8 km to 2 km over a convective-scale domain centred on Singapore, and benchmark it against CPMGEM, a score-based diffusion model. Flow matching achieves consistently better spatial skill: higher fractions skill score at every precipitation threshold and neighbourhood scale tested, and tighter structure and amplitude components of the SAL score with comparable location skill. However, flow matching underestimates the upper tail of the precipitation distribution, resulting in a dry bias in the climatological mean. These results suggest that flow matching is a competitive generative framework for convective-scale precipitation downscaling, particularly well suited to capturing spatial structure.</span> <span class="abstract-toggle" data-id="2606.00281">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2606.00281v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2606.00281v1) · [:material-content-copy: BibTeX](../../bibtex/2606.00281.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=daily" data-tag="daily">Daily</a>
    { .paper-tags }

-   #### Precipitation diffusion downscaling and application to out-of-distribution simulations with and without stratospheric aerosol injection { #2605.23776 }

    *Cameron Dong, James W. Hurrell, Elizabeth A. Barnes* · May 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2605.23776">Stratospheric aerosol injection (SAI), a possible climate engineering strategy where reflective particles are injected into the stratosphere, has been explored to mitigate global warming and its...</span><span class="abstract-full" id="full-2605.23776" hidden>Stratospheric aerosol injection (SAI), a possible climate engineering strategy where reflective particles are injected into the stratosphere, has been explored to mitigate global warming and its associated risks, such as the intensification of extreme precipitation events. However, current Earth system models (ESMs) often used to simulate SAI and other climate change scenarios are too coarse to properly assess such risks. Traditional statistical downscaling methods, used to project higher resolution impacts, may be biased and unrealistic. To address this, we train a deep learning diffusion downscaler to generate 0.25° contiguous United States (CONUS) daily precipitation using historical and future climate simulations from the Mesoscale Atmosphere-Ocean Interaction in Seasonal-to-Decadal Climate Prediction (MESACLIP) project, then apply the diffusion downscaler to out-of-distribution CESM2 simulations with and without SAI. The diffusion model generates realistic downscaled precipitation using either MESACLIP or CESM2 inputs. It also faithfully recreates the climate change projections of extreme precipitation in MESACLIP. Diffusion-downscaled projections of the future CESM2 SAI scenarios suggest that SAI could nearly cut in half the CONUS-average increase in yearly max precipitation, compared to the non-SAI scenario. However, there is considerable regional variation and internal variability, with SAI modeled to only slightly reduce increases in extreme precipitation frequency in the Mid Atlantic and the Pacific Northwest, but mitigating most intensification in other regions. Future application of diffusion downscaling to a wider variety of SAI scenarios would provide valuable insight into how proposed SAI strategies may affect precipitation variability on fine spatial scales for regional impact assessments.</span> <span class="abstract-toggle" data-id="2605.23776">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2605.23776v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2605.23776v1) · [:material-content-copy: BibTeX](../../bibtex/2605.23776.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=extremes" data-tag="extremes">Extremes</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a> <a class="md-tag" href="/explore/?t=daily" data-tag="daily">Daily</a>
    { .paper-tags }

-   #### Hybrid Quantum-Classical Corrective Diffusion Modeling for Meteorological Downscaling { #2605.23403 }

    *Rui Wang, Edoardo Pasetto, Amer Delilbasic, Morris Riedel, Kristel Michielsen, Gabriele Cavallaro* · May 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2605.23403">Statistical downscaling is a crucial component of the weather modeling field, where high-resolution outputs must be reconstructed from coarse-resolution inputs with the full cost of dynamical...</span><span class="abstract-full" id="full-2605.23403" hidden>Statistical downscaling is a crucial component of the weather modeling field, where high-resolution outputs must be reconstructed from coarse-resolution inputs with the full cost of dynamical refinement. In this work, we investigate a hybrid quantum-classical corrective diffusion model for probabilistic statistical downscaling of weather fields. The proposed model inserts variational quantum circuit layers into the most compressed bottleneck of the diffusion UNet while leaving the regression branch fully classical. This placement tests whether quantum circuits can act as compact nonlinear feature maps for latent-channel mixing. We evaluate intra-channel and cross-channel ansätze on 10m wind components. On the 2020 validation set, the hybrid models remain stable, preserve the large-scale spatial organization of the generated wind fields, and improve both MAE and CRPS relative to a classical corrective diffusion model in several configurations. Structural diagnostics further show that the hybrid variants preserve kinetic-energy spectra and windspeed distributions similar to its classical counterpart while producing controlled changes in tail behavior, extreme-windspeed localization, and joint wind field components structure. Backend studies on the 2020 validation set show negligible impact from simulated device noise at the tested circuit scale, whereas real-hardware deployment remains limited by qubit availability and execution fidelity. The 2021 out-of-distribution test shows that these in-distribution gains do not transfer uniformly under temporal shift, revealing a generalization gap that motivates future mitigation through stabilization and regularization. These results show that bottleneck-level quantum hybridization can make a nontrivial contribution to weather statistical downscaling, while also highlighting that circuit scale and hardware deployment remain key limiting factors.</span> <span class="abstract-toggle" data-id="2605.23403">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2605.23403v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2605.23403v1) · [:material-content-copy: BibTeX](../../bibtex/2605.23403.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a> <a class="md-tag" href="/explore/?t=physics-ml-hybrid" data-tag="physics-ml-hybrid">Physics–ML hybrid</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### Longwang: Zero-Shot Global Spatiotemporal Precipitation Downscaling with a Latent Generative Prior { #2605.17603 }

    *Yue Wang, Daniele Visioni* · May 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2605.17603">High-resolution precipitation information is essential for climate impact assessment, yet global climate models remain too coarse to resolve key small-scale processes. Existing machine learning...</span><span class="abstract-full" id="full-2605.17603" hidden>High-resolution precipitation information is essential for climate impact assessment, yet global climate models remain too coarse to resolve key small-scale processes. Existing machine learning downscaling methods often require paired low- and high-resolution data for supervised learning, are tied to fixed regions or scale factors during inference, and can be computationally expensive to train and run in physical space. Here we introduce Longwang, a zero-shot latent generative framework for global spatiotemporal precipitation downscaling. Longwang learns a context-conditioned latent generative prior and combines it with a physically informed observation operator through posterior sampling, enabling daily O(10 km) precipitation fields to be generated from monthly O(100 km) inputs. On ERA5 reanalysis, Longwang outperforms standard posterior sampling with an unconditional generative prior in reconstructing fine-scale spatial patterns, preserving temporal coherence, and recovering extreme precipitation intensities. The framework further generalizes to historical climate simulations and future climate projections under substantial distribution shift.</span> <span class="abstract-toggle" data-id="2605.17603">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2605.17603v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2605.17603v1) · [:material-content-copy: BibTeX](../../bibtex/2605.17603.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a> <a class="md-tag" href="/explore/?t=daily" data-tag="daily">Daily</a> <a class="md-tag" href="/explore/?t=monthly" data-tag="monthly">Monthly</a>
    { .paper-tags }

-   #### Generative climate downscaling enables high-resolution compound risk assessment by preserving multivariate dependencies { #2605.11531 }

    *Takuro Kutsuna, Noriko N. Ishizaki, Norihiro Oyama, Hiroaki Yoshida* · May 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2605.11531">Physics-based climate projections using general circulation models are essential for assessing future risks, but their coarse resolution limits regional decision-making. Statistical downscaling can...</span><span class="abstract-full" id="full-2605.11531" hidden>Physics-based climate projections using general circulation models are essential for assessing future risks, but their coarse resolution limits regional decision-making. Statistical downscaling can efficiently add detail, yet many methods treat variables independently, degrading inter-variable relationships that govern compound hazards such as heat stress, drought, and wildfire. Here we show that a diffusion-based multivariate generative framework, combined with bias correction, recovers degraded inter-variable correlations even under a 50$\times$ increase in linear resolution. When applied to five meteorological variables over Japan, the framework reduces inter-variable correlation errors by more than fourfold relative to existing baselines while improving both univariate and spatial accuracy, leading to more accurate detection of severe drought. These results demonstrate that multivariate generative downscaling improves the reliability of compound risk assessment under large resolution gaps.</span> <span class="abstract-toggle" data-id="2605.11531">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2605.11531v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2605.11531v1) · [:material-content-copy: BibTeX](../../bibtex/2605.11531.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a>
    { .paper-tags }

-   #### PODiff: Latent Diffusion in Proper Orthogonal Decomposition Space for Scientific Super-Resolution { #2605.03399 }

    *Onkar Jadhav, Tim French, Matthew Rayson, Nicole L. Jones* · May 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2605.03399">Probabilistic super-resolution of high-dimensional spatial fields using diffusion models is often computationally prohibitive due to the cost of operating directly in pixel space. We propose PODiff,...</span><span class="abstract-full" id="full-2605.03399" hidden>Probabilistic super-resolution of high-dimensional spatial fields using diffusion models is often computationally prohibitive due to the cost of operating directly in pixel space. We propose PODiff, a structured conditional generative framework that performs diffusion in a fixed, variance-ordered Proper Orthogonal Decomposition (POD) coefficient space, exploiting the orthogonality of POD modes to impose an interpretable, variance-ordered latent geometry. This design enables efficient ensemble generation, preserves dominant spatial structure, and yields spatially interpretable, well-calibrated uncertainty at substantially lower computational cost. We evaluate PODiff on sea surface temperature downscaling over the West Australian coast and on a controlled advection-diffusion benchmark. PODiff achieves reconstruction accuracy comparable to pixel-space diffusion while requiring significantly less memory and producing more reliable uncertainty estimates than deterministic and Monte Carlo Dropout baselines.</span> <span class="abstract-toggle" data-id="2605.03399">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2605.03399v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2605.03399v1) · [:material-content-copy: BibTeX](../../bibtex/2605.03399.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=interpretability" data-tag="interpretability">Interpretability</a>
    { .paper-tags }

-   #### Conditional Flow Matching for Probabilistic Downscaling of Maximum 3-day Snowfall in Alaska { #2604.25172 }

    *Douglas Brinkerhoff, Elizabeth Fischer* · Apr 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2604.25172">Precipitation in complex terrain is governed by orographic processes operating at scales of a few kilometers, yet climate models typically run at resolutions of 50--100~km where this topographic...</span><span class="abstract-full" id="full-2604.25172" hidden>Precipitation in complex terrain is governed by orographic processes operating at scales of a few kilometers, yet climate models typically run at resolutions of 50--100~km where this topographic detail is absent. Dynamical downscaling with high-resolution regional models such as WRF can resolve these processes, but the computational cost -- months of wall-clock time per scenario -- precludes the large ensembles needed for uncertainty quantification. We present WxFlow, a conditional generative model based on flow matching that learns to map coarse-resolution climate model output and high-resolution topography to calibrated probabilistic ensembles of fine-scale precipitation fields. Applied to 4~km WRF simulations of maximum 3-day snowfall over southeast Alaska, WxFlow achieves 87.8% improvement in spectral fidelity and dramatically lower Continuous Ranked Probability Scores relative to conventional lapse-rate-corrected bicubic downscaling, while generating 50-member ensembles in seconds on a laptop. Ensemble spread is spatially coherent and governed by topography, reflecting physically plausible uncertainty structure. All code is available at https://github.com/glide-ism/wrf-flow.</span> <span class="abstract-toggle" data-id="2604.25172">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2604.25172v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2604.25172v1) · [:fontawesome-brands-github: Code](https://github.com/glide-ism/wrf-flow) · [:material-content-copy: BibTeX](../../bibtex/2604.25172.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### Generative 3D Gaussian Splatting for Arbitrary-ResolutionAtmospheric Downscaling and Forecasting { #2604.07928 }

    *Tao Hana, Zhibin Wen, Zhenghao Chen, Fenghua Lin, Junyu Gao, Song Guo, Lei Bai* · Apr 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2604.07928">While AI-based numerical weather prediction (NWP) enables rapid forecasting, generating high-resolution outputs remains computationally demanding due to limited multi-scale adaptability and...</span><span class="abstract-full" id="full-2604.07928" hidden>While AI-based numerical weather prediction (NWP) enables rapid forecasting, generating high-resolution outputs remains computationally demanding due to limited multi-scale adaptability and inefficient data representations. We propose the 3D Gaussian splatting-based scale-aware vision transformer (GSSA-ViT), a novel framework for arbitrary-resolution forecasting and flexible downscaling of high-dimensional atmospheric fields. Specifically, latitude-longitude grid points are treated as centers of 3D Gaussians. A generative 3D Gaussian prediction scheme is introduced to estimate key parameters, including covariance, attributes, and opacity, for unseen samples, improving generalization and mitigating overfitting. In addition, a scale-aware attention module is designed to capture cross-scale dependencies, enabling the model to effectively integrate information across varying downscaling ratios and support continuous resolution adaptation. To our knowledge, this is the first NWP approach that combines generative 3D Gaussian modeling with scale-aware attention for unified multi-scale prediction. Experiments on ERA5 show that the proposed method accurately forecasts 87 atmospheric variables at arbitrary resolutions, while evaluations on ERA5 and CMIP6 demonstrate its superior performance in downscaling tasks. The proposed framework provides an efficient and scalable solution for high-resolution, multi-scale atmospheric prediction and downscaling. Code is available at: https://github.com/binbin2xs/weather-GS.</span> <span class="abstract-toggle" data-id="2604.07928">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2604.07928v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2604.07928v1) · [:fontawesome-brands-github: Code](https://github.com/binbin2xs/weather-GS) · [:material-content-copy: BibTeX](../../bibtex/2604.07928.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=transformers" data-tag="transformers">Transformers</a>
    { .paper-tags }

-   #### Physics-Constrained Adaptive Flow Matching for Climate Downscaling { #2604.03459 }

    *Kevin Debeire, Aytaç Paçal, Pierre Gentine, Luis Medrano-Navarro, Nils Thuerey, Veronika Eyring* · Apr 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2604.03459">Regional climate information at kilometer scales is essential for assessing the impacts of climate change, but generating it with global climate models is too expensive due to their high...</span><span class="abstract-full" id="full-2604.03459" hidden>Regional climate information at kilometer scales is essential for assessing the impacts of climate change, but generating it with global climate models is too expensive due to their high computational costs. Machine learning models offer a fast alternative, yet they often violate basic physical laws and degrade when applied to climates outside of their training distribution. We present Physics-Constrained Adaptive Flow Matching (PC-AFM), a generative downscaling model that addresses both problems. Building on the Adaptive Flow Matching (AFM) model of Fotiadis et al. (2025) as our baseline, we add soft conservation constraints that keep the downscaled output consistent with the large-scale input for precipitation and humidity, and use gradient surgery via the ConFIG algorithm to prevent these constraints from interfering with the generative objective. We train the model on Central Europe climate data, evaluate it on a 10-time downscaling task (63km to 6.3km) over six variables (near-surface temperature, precipitation, specific humidity, surface pressure, and horizontal wind components) across a comprehensive set of metrics including bias, ensemble skill scores, power spectra, and conservation error, and test the generalization on two held-out climate regions. Within the training distribution, PC-AFM reduces conservation errors and improves ensemble calibration while matching the baseline on standard skill metrics. Outside the training distribution, where unconstrained models develop large systematic errors by extrapolating learned statistics, PC-AFM halves precipitation wet bias, reduces conservation error and improves extreme-quantile accuracy, all without any information about the target climate at inference time. These results indicate that physical consistency is a practical requirement for deploying generative downscaling models in real-world applications.</span> <span class="abstract-toggle" data-id="2604.03459">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2604.03459v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2604.03459v1) · [:material-content-copy: BibTeX](../../bibtex/2604.03459.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=physics-ml-hybrid" data-tag="physics-ml-hybrid">Physics–ML hybrid</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a>
    { .paper-tags }

-   #### Downscaling weather forecasts from Low- to High-Resolution with Diffusion Models { #2604.03303 }

    *Joffrey Dumont Le Brazidec, Simon Lang, Martin Leutbecher, Baudouin Raoult, Gert Mertes et al.* · Apr 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2604.03303">We introduce a probabilistic diffusion-based method for global atmospheric downscaling implemented within the Anemoi framework. The approach transforms low-resolution ensemble forecasts into...</span><span class="abstract-full" id="full-2604.03303" hidden>We introduce a probabilistic diffusion-based method for global atmospheric downscaling implemented within the Anemoi framework. The approach transforms low-resolution ensemble forecasts into high-resolution ensembles by learning the conditional distribution of finer-scale residuals, defined as the difference between the high-resolution fields and the interpolated low-resolution inputs. The system is trained on reforecast pairs from ECMWF IFS, using coarse fields at 100 km to reconstruct fine-scale variability at 30 km resolution. The bulk of the training focuses on recovering small-scale structures, while fine-tuning in high-noise regimes enables the generation of extremes. Evaluation against the medium-range IFS ensemble target shows that the model increases probabilistic skill (FCRPS) for surface variables, reproduces target power spectra at small scales, captures physically consistent multivariate relationships such as wind-pressure coupling, and generates extreme values consistent with those of the target ensemble in tropical cyclones.</span> <span class="abstract-toggle" data-id="2604.03303">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2604.03303v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2604.03303v1) · [:material-content-copy: BibTeX](../../bibtex/2604.03303.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a> <a class="md-tag" href="/explore/?t=quarter-degree" data-tag="quarter-degree">0.25°</a>
    { .paper-tags }

-   #### IPSL-AID: Generative Diffusion Models for Climate Downscaling from Global to Regional Scales { #2604.03275 }

    *Kishanthan Kingston, Olivier Boucher, Freddy Bouchet, Pierre Chapel, Rosemary Eade et al.* · Apr 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2604.03275">Effective adaptation and mitigation strategies for climate change require high-resolution projections to inform strategic decision-making. Conventional global climate models, which typically operate...</span><span class="abstract-full" id="full-2604.03275" hidden>Effective adaptation and mitigation strategies for climate change require high-resolution projections to inform strategic decision-making. Conventional global climate models, which typically operate at resolutions of 150 to 200 kilometers, lack the capacity to represent essential regional processes. IPSL-AID is a global to regional downscaling tool based on a denoising diffusion probabilistic model designed to address this limitation. Trained on ERA5 reanalysis data, it generates 0.25 degree resolution fields for temperature, wind, and precipitation using coarse inputs and their spatiotemporal context. It also models probability distributions of fine-scale features to produce plausible scenarios for uncertainty quantification. The model accurately reconstructs statistical distributions, including extreme events, power spectra, and spatial structures. This work highlights the potential of generative diffusion models for efficient climate downscaling with uncertainty</span> <span class="abstract-toggle" data-id="2604.03275">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2604.03275v2) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2604.03275v2) · [:material-content-copy: BibTeX](../../bibtex/2604.03275.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a> <a class="md-tag" href="/explore/?t=quarter-degree" data-tag="quarter-degree">0.25°</a>
    { .paper-tags }

-   #### 30-meter Land Surface Temperature from Landsat via Progressive Self-Training Downscaling { #2603.29478 }

    *Huanfeng Shen, Chan Li, Menghui Jiang, Penghai Wu, Guanhao Zhang, Tian Xie* · Mar 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2603.29478">Land surface temperature (LST) is a critical parameter for characterizing surface energy balance and hydrothermal processes. While Landsat provides invaluable LST observations at medium spatial...</span><span class="abstract-full" id="full-2603.29478" hidden>Land surface temperature (LST) is a critical parameter for characterizing surface energy balance and hydrothermal processes. While Landsat provides invaluable LST observations at medium spatial resolution for over 40 years, its native spatial resolution of thermal bands (e.g., 100 m) remains insufficient compared to its 30 m optical bands, failing to meet the demands of fine-scale studies. To address this issues, this study proposes a progressive self-training framework for downscaling Landsat LST to 30 m without relying on fine-scale ground truth, while maintaining minimal data dependence. The framework progressively optimizes a cross-modal fusion network to refine thermal details in a coarse-to-fine manner, characterized by one pre-training and two fine-tuning stages. Spatial validation against SDGSAT-1 30 m LST and temporal validation using in situ measurements confirm its reliability and accuracy, with both station-averaged MAE and RMSE outperforming the official cubic product by approximately 0.4 K. Further performance comparison experiments demonstrate that the proposed framework consistently reconstructs coherent fine-scale thermal patterns while preserving spatial heterogeneity. Multi spatial resolution evaluations and ablation studies verify the effectiveness of the proposed strategy and network design. Overall, the framework provides a stable pathway for enhancing the spatial resolution of Landsat LST, providing fine-resolution data support for fine-scale surface process studies and localized environmental monitoring.</span> <span class="abstract-toggle" data-id="2603.29478">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2603.29478v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2603.29478v1) · [:material-content-copy: BibTeX](../../bibtex/2603.29478.bib){ .bibtex-link }
    { .paper-links }

-   #### Climate Downscaling with Stochastic Interpolants (CDSI) { #2603.03838 }

    *Erik Larsson, Ramon Fuentes-Franco, Mikhail Ivanov, Fredrik Lindsten* · Mar 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2603.03838">Global climate projections rely on computationally demanding Earth System Models (ESMs), which are typically limited to coarse spatial resolutions due to their high cost. To obtain high-resolution...</span><span class="abstract-full" id="full-2603.03838" hidden>Global climate projections rely on computationally demanding Earth System Models (ESMs), which are typically limited to coarse spatial resolutions due to their high cost. To obtain high-resolution projections for regions of interest, it is common to use Regional Climate Models (RCMs), which are driven by data produced by ESMs as boundary conditions. While more efficient than running ESMs at fine resolution, RCMs remain expensive and restrict the size of ensemble simulations. Inspired by recent advances in probabilistic machine learning for weather and climate, we introduce a data-driven climate downscaling method based on stochastic interpolants. Our approach efficiently transforms coarse ESM output into high-resolution regional climate projections at a fraction of the computational cost of traditional RCMs. Through extensive validation, we demonstrate that our method generates accurate regional ensembles, enabling both improved uncertainty quantification and broader use of high-resolution climate information.</span> <span class="abstract-toggle" data-id="2603.03838">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2603.03838v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2603.03838v1) · [:material-content-copy: BibTeX](../../bibtex/2603.03838.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a>
    { .paper-tags }

-   #### Physics Encoded Spatial and Temporal Generative Adversarial Network for Tropical Cyclone Image Super-resolution { #2602.17277 }

    *Ruoyi Zhang, Jiawei Yuan, Lujia Ye, Runling Yu, Liling Zhao* · Feb 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2602.17277">High-resolution satellite imagery is indispensable for tracking the genesis, intensification, and trajectory of tropical cyclones (TCs). However, existing deep learning-based super-resolution (SR)...</span><span class="abstract-full" id="full-2602.17277" hidden>High-resolution satellite imagery is indispensable for tracking the genesis, intensification, and trajectory of tropical cyclones (TCs). However, existing deep learning-based super-resolution (SR) methods often treat satellite image sequences as generic videos, neglecting the underlying atmospheric physical laws governing cloud motion. To address this, we propose a Physics Encoded Spatial and Temporal Generative Adversarial Network (PESTGAN) for TC image super-resolution. Specifically, we design a disentangled generator architecture incorporating a PhyCell module, which approximates the vorticity equation via constrained convolutions and encodes the resulting approximate physical dynamics as implicit latent representations to separate physical dynamics from visual textures. Furthermore, a dual-discriminator framework is introduced, employing a temporal discriminator to enforce motion consistency alongside spatial realism. Experiments on the Digital Typhoon dataset for 4$\times$ upscaling demonstrate that PESTGAN establishes a better performance in structural fidelity and perceptual quality. While maintaining competitive pixel-wise accuracy compared to existing approaches, our method significantly excels in reconstructing meteorologically plausible cloud structures with superior physical fidelity.</span> <span class="abstract-toggle" data-id="2602.17277">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2602.17277v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2602.17277v1) · [:material-content-copy: BibTeX](../../bibtex/2602.17277.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=gans" data-tag="gans">GANs</a> <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a>
    { .paper-tags }

-   #### MAUNet-Light: A Concise MAUNet Architecture for Bias Correction and Downscaling of Precipitation Estimates { #2602.12980 }

    *Sumanta Chandra Mishra Sharma, Adway Mitra, Auroop Ratan Ganguly* · Feb 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2602.12980">Satellite-derived data products and climate model simulations of geophysical variables like precipitation, often exhibit systematic biases compared to in-situ measurements. Bias correction and...</span><span class="abstract-full" id="full-2602.12980" hidden>Satellite-derived data products and climate model simulations of geophysical variables like precipitation, often exhibit systematic biases compared to in-situ measurements. Bias correction and spatial downscaling are fundamental components to develop operational weather forecast systems, as they seek to improve the consistency between coarse-resolution climate model simulations or satellite-based estimates and ground-based observations. In recent years, deep learning-based models have been increasingly replaced traditional statistical methods to generate high-resolution, bias free projections of climate variables. For example, Max-Average U-Net (MAUNet) architecture has been demonstrated for its ability to downscale precipitation estimates. The versatility and adaptability of these neural models make them highly effective across a range of applications, though this often come at the cost of high computational and memory requirements. The aim of this research is to develop light-weight neural network architectures for both bias correction and downscaling of precipitation, for which the teacher-student based learning paradigm is explored. This research demonstrates the adaptability of MAUNet to the task of bias correction, and further introduces a compact, lightweight neural network architecture termed MAUNet-Light.The proposed MAUNet-Light model is developed by transferring knowledge from the trained MAUNet, and it is designed to perform both downscaling and bias correction with reduced computational requirements without any significant loss in accuracy compared to state-of-the-art.</span> <span class="abstract-toggle" data-id="2602.12980">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2602.12980v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2602.12980v1) · [:material-content-copy: BibTeX](../../bibtex/2602.12980.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a>
    { .paper-tags }

-   #### Universal Diffusion-Based Probabilistic Downscaling { #2602.11893 }

    *Roberto Molinaro, Niall Siegenheim, Henry Martin, Mark Frey, Niels Poulsen, Philipp Seitz et al.* · Feb 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2602.11893">We introduce a universal diffusion-based downscaling framework that lifts deterministic low-resolution weather forecasts into probabilistic high-resolution predictions without any model-specific...</span><span class="abstract-full" id="full-2602.11893" hidden>We introduce a universal diffusion-based downscaling framework that lifts deterministic low-resolution weather forecasts into probabilistic high-resolution predictions without any model-specific fine-tuning. A single conditional diffusion model is trained on paired coarse-resolution inputs (~25 km resolution) and high-resolution regional reanalysis targets (~5 km resolution), and is applied in a fully zero-shot manner to deterministic forecasts from heterogeneous upstream weather models. Focusing on near-surface variables, we evaluate probabilistic forecasts against independent in situ station observations over lead times up to 90 h. Across a diverse set of AI-based and numerical weather prediction (NWP) systems, the ensemble mean of the downscaled forecasts consistently improves upon each model's own raw deterministic forecast, and substantially larger gains are observed in probabilistic skill as measured by CRPS. These results demonstrate that diffusion-based downscaling provides a scalable, model-agnostic probabilistic interface for enhancing spatial resolution and uncertainty representation in operational weather forecasting pipelines.</span> <span class="abstract-toggle" data-id="2602.11893">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2602.11893v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2602.11893v1) · [:material-content-copy: BibTeX](../../bibtex/2602.11893.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a> <a class="md-tag" href="/explore/?t=station-point" data-tag="station-point">Station / point</a> <a class="md-tag" href="/explore/?t=km-scale" data-tag="km-scale">Km-scale</a> <a class="md-tag" href="/explore/?t=quarter-degree" data-tag="quarter-degree">0.25°</a>
    { .paper-tags }

</div>

<nav class="pager" markdown="span">**1** [2](2.md) [3](3.md) [4](4.md) [Older :material-arrow-right:](2.md){ .pager-step }</nav>

