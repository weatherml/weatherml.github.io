---
title: 'Downscaling'
hide:
  - toc
---

<div class="listing-header" markdown>

# Downscaling

<p class="page-meta" markdown="span">43 papers · page 1 of 2 · <a href="../../bib/downscaling.bib" download>:material-download: BibTeX for this topic</a></p>

</div>

<div class="grid cards" markdown>

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

-   #### Deep Learning-Based Statistical Downscaling of Sea Surface Temperature Using a Residual Corrective Neural Network { #2608.10022 }

    *Onkar Jadhav, Tim French, Ivica Janekovic, Nicole L. Jones, Matthew Rayson* · Aug 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2608.10022">The large-scale oceanic and atmospheric forecasts provided by global climate models typically lack sufficient resolution to accurately capture the response of the coastal ocean to atmospheric forcing...</span><span class="abstract-full" id="full-2608.10022" hidden>The large-scale oceanic and atmospheric forecasts provided by global climate models typically lack sufficient resolution to accurately capture the response of the coastal ocean to atmospheric forcing and coastal circulation that drive fine-scale SST variability. Dynamical downscaling is computationally prohibitive, when applied to extensive coastlines, predictive ensembles, or long time periods. Therefore, this work presents a statistical downscaling of sea surface temperature (SST) from the seasonal coupled ocean-atmosphere forecast system (ACCESS-S2) using machine learning techniques. This study proposes a novel deep learning framework that uses a U-Net to generate an initial high-resolution SST estimate, which is subsequently refined using a residual corrective approach. The target SST fields are derived from the Regional Ocean Modeling System (ROMS). This two step approach called Residual Corrective Neural Network (RCNN) progressively refines initial U-Net predictions by incorporating dynamically scaled residuals at each step, enabling accurate capture of broad patterns and fine-grained features such as eddies and fronts. We also introduce a custom loss-assisted RCNN variant to improve performance during extreme events, which may be absent from training data due to climate-driven shifts in SST extremes. The framework efficiently downscales SST along the west coast of Australia. A 2011 marine heatwave case study shows that the RCNN improves ACCESS-S2 SST predictions by increasing horizontal resolution from 25 km to 2 km, enabling identification of fine-scale anomalies unresolved in the ACCESS-S2 dataset. This balance between computational efficiency and accuracy supports applications in coastal impact assessment and marine ecosystem studies.</span> <span class="abstract-toggle" data-id="2608.10022">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2608.10022v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2608.10022v1) · [:material-content-copy: BibTeX](../../bibtex/2608.10022.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a> <a class="md-tag" href="/explore/?t=extremes" data-tag="extremes">Extremes</a> <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a> <a class="md-tag" href="/explore/?t=km-scale" data-tag="km-scale">Km-scale</a> <a class="md-tag" href="/explore/?t=quarter-degree" data-tag="quarter-degree">0.25°</a>
    { .paper-tags }

-   #### Transferable Dual-Stream Representations for Mesoscale-Preserving Sea Surface Temperature Downscaling { #2608.04230 }

    *Parth Doshi, Priyanka Aravindan, Vaishnav Vaidheeswaran, Md Mahbub Alam, Gabriel Spadon* · Aug 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2608.04230">Deep learning models for scientific spatio-temporal downscaling often minimize reconstruction error while failing to preserve physically meaningful multi-scale structure. For sea surface temperature...</span><span class="abstract-full" id="full-2608.04230" hidden>Deep learning models for scientific spatio-temporal downscaling often minimize reconstruction error while failing to preserve physically meaningful multi-scale structure. For sea surface temperature prediction, this can yield outputs that are numerically plausible yet overly smooth, missing mesoscale variability critical to regional ocean dynamics. Existing methods often focus on pixel-wise objectives or single-context conditioning, which limits their ability to preserve spectral fidelity and generalize across regions. To address this, we propose EddyFlow, a representation learning framework for kilometer-scale sea surface temperature downscaling that balances predictive accuracy, scale-dependent structure, and regional generalization. EddyFlow is trained on the Gulf of St.~Lawrence and evaluated in zero-shot and few-shot settings on the Bay of Fundy and the Gulf of Mexico. EddyFlow demonstrates that physics-informed representation learning reduces zero-shot RMSE by 21%, achieves up to 85.6% skill relative to persistence on unseen domains, and maintains near-ideal spectral fidelity with a PSD ratio of $\approx 1.00$.</span> <span class="abstract-toggle" data-id="2608.04230">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2608.04230v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2608.04230v1) · [:material-content-copy: BibTeX](../../bibtex/2608.04230.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=physics-ml-hybrid" data-tag="physics-ml-hybrid">Physics–ML hybrid</a> <a class="md-tag" href="/explore/?t=km-scale" data-tag="km-scale">Km-scale</a>
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

-   #### Hybrid Quantum-Classical Corrective Diffusion Modeling for Meteorological Downscaling { #2605.23403 }

    *Rui Wang, Edoardo Pasetto, Amer Delilbasic, Morris Riedel, Kristel Michielsen, Gabriele Cavallaro* · May 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2605.23403">Statistical downscaling is a crucial component of the weather modeling field, where high-resolution outputs must be reconstructed from coarse-resolution inputs with the full cost of dynamical...</span><span class="abstract-full" id="full-2605.23403" hidden>Statistical downscaling is a crucial component of the weather modeling field, where high-resolution outputs must be reconstructed from coarse-resolution inputs with the full cost of dynamical refinement. In this work, we investigate a hybrid quantum-classical corrective diffusion model for probabilistic statistical downscaling of weather fields. The proposed model inserts variational quantum circuit layers into the most compressed bottleneck of the diffusion UNet while leaving the regression branch fully classical. This placement tests whether quantum circuits can act as compact nonlinear feature maps for latent-channel mixing. We evaluate intra-channel and cross-channel ansätze on 10m wind components. On the 2020 validation set, the hybrid models remain stable, preserve the large-scale spatial organization of the generated wind fields, and improve both MAE and CRPS relative to a classical corrective diffusion model in several configurations. Structural diagnostics further show that the hybrid variants preserve kinetic-energy spectra and windspeed distributions similar to its classical counterpart while producing controlled changes in tail behavior, extreme-windspeed localization, and joint wind field components structure. Backend studies on the 2020 validation set show negligible impact from simulated device noise at the tested circuit scale, whereas real-hardware deployment remains limited by qubit availability and execution fidelity. The 2021 out-of-distribution test shows that these in-distribution gains do not transfer uniformly under temporal shift, revealing a generalization gap that motivates future mitigation through stabilization and regularization. These results show that bottleneck-level quantum hybridization can make a nontrivial contribution to weather statistical downscaling, while also highlighting that circuit scale and hardware deployment remain key limiting factors.</span> <span class="abstract-toggle" data-id="2605.23403">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2605.23403v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2605.23403v1) · [:material-content-copy: BibTeX](../../bibtex/2605.23403.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a> <a class="md-tag" href="/explore/?t=physics-ml-hybrid" data-tag="physics-ml-hybrid">Physics–ML hybrid</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
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

-   #### A Differentiable Framework for Global Circulation Model Precipitation Bias Correction { #2604.23045 }

    *Kamlesh Sawadekar, Seth McGinnis, Peijun Li, Kathryn Lawson, Chaopeng Shen* · Apr 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2604.23045">Systematic biases in General Circulation Model (GCM) outputs limit their direct applicability in regional planning, making bias correction a technically demanding but necessary step for both...</span><span class="abstract-full" id="full-2604.23045" hidden>Systematic biases in General Circulation Model (GCM) outputs limit their direct applicability in regional planning, making bias correction a technically demanding but necessary step for both short-term and long-term impact assessment. Correcting precipitation is particularly challenging due to its non-Gaussian distribution, intermittent nature, and heavy-tailed extremes. However, traditional statistical bias-correction methods have limited ability to learn systematic patterns from large datasets or generalize to new locations. While machine learning (ML) provides greater flexibility, it can produce unpredictable and difficult-to-interpret results, limiting generalization across GCMs and locations. In this study, we propose a differentiable bias-adjustment framework called dCLIMBA, that learns a spatiotemporally adaptive parametric bias-adjustment procedure, rather than corrected precipitation directly, between historical CMIP6 model outputs and a gridded observation-based dataset, Livneh. Results demonstrate that the proposed method corrects the magnitude and distribution of extreme precipitation with particularly strong performance in the upper tail. The quantile distribution of precipitation was well reproduced across diverse U.S. cities, and spatial patterns were comparable to those from the widely used LOCA2 statistical downscaling product. In addition, the framework showed partial future trend preservation and promising attenuation of marginal biases in unseen regions. This work presents a modular and efficient bias-correction approach. The differentiable approach provides an easy-to-use option for connecting atmospheric-model outputs to on-the-ground impacts.</span> <span class="abstract-toggle" data-id="2604.23045">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2604.23045v3) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2604.23045v3) · [:material-content-copy: BibTeX](../../bibtex/2604.23045.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a>
    { .paper-tags }

-   #### Generative 3D Gaussian Splatting for Arbitrary-ResolutionAtmospheric Downscaling and Forecasting { #2604.07928 }

    *Tao Hana, Zhibin Wen, Zhenghao Chen, Fenghua Lin, Junyu Gao, Song Guo, Lei Bai* · Apr 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2604.07928">While AI-based numerical weather prediction (NWP) enables rapid forecasting, generating high-resolution outputs remains computationally demanding due to limited multi-scale adaptability and...</span><span class="abstract-full" id="full-2604.07928" hidden>While AI-based numerical weather prediction (NWP) enables rapid forecasting, generating high-resolution outputs remains computationally demanding due to limited multi-scale adaptability and inefficient data representations. We propose the 3D Gaussian splatting-based scale-aware vision transformer (GSSA-ViT), a novel framework for arbitrary-resolution forecasting and flexible downscaling of high-dimensional atmospheric fields. Specifically, latitude-longitude grid points are treated as centers of 3D Gaussians. A generative 3D Gaussian prediction scheme is introduced to estimate key parameters, including covariance, attributes, and opacity, for unseen samples, improving generalization and mitigating overfitting. In addition, a scale-aware attention module is designed to capture cross-scale dependencies, enabling the model to effectively integrate information across varying downscaling ratios and support continuous resolution adaptation. To our knowledge, this is the first NWP approach that combines generative 3D Gaussian modeling with scale-aware attention for unified multi-scale prediction. Experiments on ERA5 show that the proposed method accurately forecasts 87 atmospheric variables at arbitrary resolutions, while evaluations on ERA5 and CMIP6 demonstrate its superior performance in downscaling tasks. The proposed framework provides an efficient and scalable solution for high-resolution, multi-scale atmospheric prediction and downscaling. Code is available at: https://github.com/binbin2xs/weather-GS.</span> <span class="abstract-toggle" data-id="2604.07928">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2604.07928v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2604.07928v1) · [:fontawesome-brands-github: Code](https://github.com/binbin2xs/weather-GS) · [:material-content-copy: BibTeX](../../bibtex/2604.07928.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=transformers" data-tag="transformers">Transformers</a>
    { .paper-tags }

-   #### Downscaling weather forecasts from Low- to High-Resolution with Diffusion Models { #2604.03303 }

    *Joffrey Dumont Le Brazidec, Simon Lang, Martin Leutbecher, Baudouin Raoult, Gert Mertes et al.* · Apr 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2604.03303">We introduce a probabilistic diffusion-based method for global atmospheric downscaling implemented within the Anemoi framework. The approach transforms low-resolution ensemble forecasts into...</span><span class="abstract-full" id="full-2604.03303" hidden>We introduce a probabilistic diffusion-based method for global atmospheric downscaling implemented within the Anemoi framework. The approach transforms low-resolution ensemble forecasts into high-resolution ensembles by learning the conditional distribution of finer-scale residuals, defined as the difference between the high-resolution fields and the interpolated low-resolution inputs. The system is trained on reforecast pairs from ECMWF IFS, using coarse fields at 100 km to reconstruct fine-scale variability at 30 km resolution. The bulk of the training focuses on recovering small-scale structures, while fine-tuning in high-noise regimes enables the generation of extremes. Evaluation against the medium-range IFS ensemble target shows that the model increases probabilistic skill (FCRPS) for surface variables, reproduces target power spectra at small scales, captures physically consistent multivariate relationships such as wind-pressure coupling, and generates extreme values consistent with those of the target ensemble in tropical cyclones.</span> <span class="abstract-toggle" data-id="2604.03303">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2604.03303v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2604.03303v1) · [:material-content-copy: BibTeX](../../bibtex/2604.03303.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a> <a class="md-tag" href="/explore/?t=quarter-degree" data-tag="quarter-degree">0.25°</a>
    { .paper-tags }

-   #### 30-meter Land Surface Temperature from Landsat via Progressive Self-Training Downscaling { #2603.29478 }

    *Huanfeng Shen, Chan Li, Menghui Jiang, Penghai Wu, Guanhao Zhang, Tian Xie* · Mar 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2603.29478">Land surface temperature (LST) is a critical parameter for characterizing surface energy balance and hydrothermal processes. While Landsat provides invaluable LST observations at medium spatial...</span><span class="abstract-full" id="full-2603.29478" hidden>Land surface temperature (LST) is a critical parameter for characterizing surface energy balance and hydrothermal processes. While Landsat provides invaluable LST observations at medium spatial resolution for over 40 years, its native spatial resolution of thermal bands (e.g., 100 m) remains insufficient compared to its 30 m optical bands, failing to meet the demands of fine-scale studies. To address this issues, this study proposes a progressive self-training framework for downscaling Landsat LST to 30 m without relying on fine-scale ground truth, while maintaining minimal data dependence. The framework progressively optimizes a cross-modal fusion network to refine thermal details in a coarse-to-fine manner, characterized by one pre-training and two fine-tuning stages. Spatial validation against SDGSAT-1 30 m LST and temporal validation using in situ measurements confirm its reliability and accuracy, with both station-averaged MAE and RMSE outperforming the official cubic product by approximately 0.4 K. Further performance comparison experiments demonstrate that the proposed framework consistently reconstructs coherent fine-scale thermal patterns while preserving spatial heterogeneity. Multi spatial resolution evaluations and ablation studies verify the effectiveness of the proposed strategy and network design. Overall, the framework provides a stable pathway for enhancing the spatial resolution of Landsat LST, providing fine-resolution data support for fine-scale surface process studies and localized environmental monitoring.</span> <span class="abstract-toggle" data-id="2603.29478">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2603.29478v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2603.29478v1) · [:material-content-copy: BibTeX](../../bibtex/2603.29478.bib){ .bibtex-link }
    { .paper-links }

-   #### Downscaling land surface temperature data using edge detection and block-diagonal Gaussian process regression { #2602.02813 }

    *Sanjit Dandapanthula, Margaret Johnson, Madeleine Pascolini-Campbell, Glynn Hulley, Mikael Kuusela* · Feb 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2602.02813">Accurate and high-resolution estimation of land surface temperature (LST) is crucial in estimating evapotranspiration, a measure of plant water use and a central quantity in agricultural...</span><span class="abstract-full" id="full-2602.02813" hidden>Accurate and high-resolution estimation of land surface temperature (LST) is crucial in estimating evapotranspiration, a measure of plant water use and a central quantity in agricultural applications. In this work, we develop a novel statistical method for downscaling LST data obtained from NASA's ECOSTRESS mission, using high-resolution data from the Landsat 8 mission as a proxy for modeling agricultural field structure. Using the Landsat data, we identify the boundaries of agricultural fields through edge detection techniques, allowing us to capture the inherent block structure present in the spatial domain. We propose a block-diagonal Gaussian process (BDGP) model that captures the spatial structure of the agricultural fields, leverages independence of LST across fields for computational tractability, and accounts for the change of support present in ECOSTRESS observations. We use the resulting BDGP model to perform Gaussian process regression and obtain high-resolution estimates of LST from ECOSTRESS data, along with uncertainty quantification. Our results demonstrate the practicality of the proposed method in producing reliable high-resolution LST estimates, with potential applications in agriculture, urban planning, and climate studies.</span> <span class="abstract-toggle" data-id="2602.02813">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2602.02813v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2602.02813v1) · [:material-content-copy: BibTeX](../../bibtex/2602.02813.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=classical-ml" data-tag="classical-ml">Classical ML</a>
    { .paper-tags }

-   #### Zero-Shot Statistical Downscaling via Diffusion Posterior Sampling { #2601.21760 }

    *Ruian Tie, Wenbo Xiong, Zhengyu Shi, Xinyu Su, Chenyu jiang, Libo Wu, Hao Li* · Jan 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2601.21760">Conventional supervised climate downscaling struggles to generalize to Global Climate Models (GCMs) due to the lack of paired training data and inherent domain gaps relative to reanalysis. Meanwhile,...</span><span class="abstract-full" id="full-2601.21760" hidden>Conventional supervised climate downscaling struggles to generalize to Global Climate Models (GCMs) due to the lack of paired training data and inherent domain gaps relative to reanalysis. Meanwhile, current zero-shot methods suffer from physical inconsistencies and vanishing gradient issues under large scaling factors. We propose Zero-Shot Statistical Downscaling (ZSSD), a zero-shot framework that performs statistical downscaling without paired data during training. ZSSD leverages a Physics-Consistent Climate Prior learned from reanalysis data, conditioned on geophysical boundaries and temporal information to enforce physical validity. Furthermore, to enable robust inference across varying GCMs, we introduce Unified Coordinate Guidance. This strategy addresses the vanishing gradient problem in vanilla DPS and ensures consistency with large-scale fields. Results show that ZSSD significantly outperforms existing zero-shot baselines in 99th percentile errors and successfully reconstructs complex weather events, such as tropical cyclones, across heterogeneous GCMs.</span> <span class="abstract-toggle" data-id="2601.21760">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2601.21760v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2601.21760v1) · [:material-content-copy: BibTeX](../../bibtex/2601.21760.bib){ .bibtex-link }
    { .paper-links }

-   #### Time-aware UNet and super-resolution deep residual networks for spatial downscaling { #2512.13753 }

    *Mika Sipilä, Sabrina Maggio, Sandra De Iaco, Klaus Nordhausen, Monica Palma, Sara Taskinen* · Dec 2025
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2512.13753">Satellite data of atmospheric pollutants are often available only at coarse spatial resolution, limiting their applicability in local-scale environmental analysis and decision-making. Spatial...</span><span class="abstract-full" id="full-2512.13753" hidden>Satellite data of atmospheric pollutants are often available only at coarse spatial resolution, limiting their applicability in local-scale environmental analysis and decision-making. Spatial downscaling methods aim to transform the coarse satellite data into high-resolution fields. In this work, two widely used deep learning architectures, the super-resolution deep residual network (SRDRN) and the encoder-decoder-based UNet, are considered for spatial downscaling of tropospheric ozone. Both methods are extended with a lightweight temporal module, which encodes observation time using either sinusoidal or radial basis function (RBF) encoding, and fuses the temporal features with the spatial representations in the networks. The proposed time-aware extensions are evaluated against their baseline counterparts in a case study on ozone downscaling over Italy. The results suggest that, while only slightly increasing computational complexity, the temporal modules significantly improve downscaling performance and convergence speed.</span> <span class="abstract-toggle" data-id="2512.13753">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2512.13753v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2512.13753v1) · [:material-content-copy: BibTeX](../../bibtex/2512.13753.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a>
    { .paper-tags }

-   #### China Regional 3km Downscaling Based on Residual Corrective Diffusion Model { #2512.05377 }

    *Honglu Sun, Hao Jing, Zhixiang Dai, Sa Xiao, Wei Xue, Jian Sun, Qifeng Lu* · Dec 2025
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2512.05377">A fundamental challenge in numerical weather prediction is to efficiently produce high-resolution forecasts. A common solution is applying downscaling methods, which include dynamical downscaling and...</span><span class="abstract-full" id="full-2512.05377" hidden>A fundamental challenge in numerical weather prediction is to efficiently produce high-resolution forecasts. A common solution is applying downscaling methods, which include dynamical downscaling and statistical downscaling, to the outputs of global models. This work focuses on statistical downscaling, which establishes statistical relationships between low-resolution and high-resolution historical data using statistical models. Deep learning has emerged as a powerful tool for this task, giving rise to various high-performance super-resolution models, which can be directly applied for downscaling, such as diffusion models and Generative Adversarial Networks. This work relies on a diffusion-based downscaling framework named CorrDiff. In contrast to the original work of CorrDiff, the region considered in this work is nearly 40 times larger, and we not only consider surface variables as in the original work, but also encounter high-level variables (six pressure levels) as target downscaling variables. In addition, a global residual connection is added to improve accuracy. In order to generate the 3km forecasts for the China region, we apply our trained models to the 25km global grid forecasts of CMA-GFS, an operational global model of the China Meteorological Administration (CMA), and SFF, a data-driven deep learning-based weather model developed from Spherical Fourier Neural Operators (SFNO). CMA-MESO, a high-resolution regional model, is chosen as the baseline model. The experimental results demonstrate that the forecasts downscaled by our method generally outperform the direct forecasts of CMA-MESO in terms of MAE for the target variables. Our forecasts of radar composite reflectivity show that CorrDiff, as a generative model, can generate fine-scale details that lead to more realistic predictions compared to the corresponding deterministic regression models.</span> <span class="abstract-toggle" data-id="2512.05377">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2512.05377v4) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2512.05377v4) · [:material-content-copy: BibTeX](../../bibtex/2512.05377.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=neural-operators" data-tag="neural-operators">Neural operators</a> <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a> <a class="md-tag" href="/explore/?t=quarter-degree" data-tag="quarter-degree">0.25°</a>
    { .paper-tags }

-   #### Spatiotemporal Satellite Image Downscaling with Transfer Encoders and Autoregressive Generative Models { #2512.05139 }

    *Yang Xiang, Jingwen Zhong, Yige Yan, Petros Koutrakis, Eric Garshick, Meredith Franklin* · Dec 2025
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2512.05139">We present a transfer-learning generative downscaling framework to reconstruct fine resolution satellite images from coarse scale inputs. Our approach combines a lightweight U-Net transfer encoder...</span><span class="abstract-full" id="full-2512.05139" hidden>We present a transfer-learning generative downscaling framework to reconstruct fine resolution satellite images from coarse scale inputs. Our approach combines a lightweight U-Net transfer encoder with a diffusion-based generative model. The simpler U-Net is first pretrained on a long time series of coarse resolution data to learn spatiotemporal representations; its encoder is then frozen and transferred to a larger downscaling model as physically meaningful latent features. Our application uses NASA's MERRA-2 reanalysis as the low resolution source domain (50 km) and the GEOS-5 Nature Run (G5NR) as the high resolution target (7 km). Our study area included a large area in Asia, which was made computationally tractable by splitting into two subregions and four seasons. We conducted domain similarity analysis using Wasserstein distances confirmed minimal distributional shift between MERRA-2 and G5NR, validating the safety of parameter frozen transfer. Across seasonal regional splits, our model achieved excellent performance (R2 = 0.65 to 0.94), outperforming comparison models including deterministic U-Nets, variational autoencoders, and prior transfer learning baselines. Out of data evaluations using semivariograms, ACF/PACF, and lag-based RMSE/R2 demonstrated that the predicted downscaled images preserved physically consistent spatial variability and temporal autocorrelation, enabling stable autoregressive reconstruction beyond the G5NR record. These results show that transfer enhanced diffusion models provide a robust and physically coherent solution for downscaling a long time series of coarse resolution images with limited training periods. This advancement has significant implications for improving environmental exposure assessment and long term environmental monitoring.</span> <span class="abstract-toggle" data-id="2512.05139">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2512.05139v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2512.05139v1) · [:material-content-copy: BibTeX](../../bibtex/2512.05139.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a> <a class="md-tag" href="/explore/?t=foundation-models" data-tag="foundation-models">Foundation models</a>
    { .paper-tags }

-   #### On Global Applicability and Location Transferability of Generative Deep Learning Models for Precipitation Downscaling { #2512.01400 }

    *Paula Harder, Christian Lessig, Matthew Chantry, Francis Pelletier, David Rolnick* · Dec 2025
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2512.01400">Deep learning offers promising capabilities for the statistical downscaling of climate and weather forecasts, with generative approaches showing particular success in capturing fine-scale...</span><span class="abstract-full" id="full-2512.01400" hidden>Deep learning offers promising capabilities for the statistical downscaling of climate and weather forecasts, with generative approaches showing particular success in capturing fine-scale precipitation patterns. However, most existing models are region-specific, and their ability to generalize to unseen geographic areas remains largely unexplored. In this study, we evaluate the generalization performance of generative downscaling models across diverse regions. Using a global framework, we employ ERA5 reanalysis data as predictors and IMERG precipitation estimates at $0.1^\circ$ resolution as targets. A hierarchical location-based data split enables a systematic assessment of model performance across 15 regions around the world.</span> <span class="abstract-toggle" data-id="2512.01400">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2512.01400v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2512.01400v1) · [:material-content-copy: BibTeX](../../bibtex/2512.01400.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a>
    { .paper-tags }

-   #### Climate Downscaling of Tropical Cyclone Intensity using Deep Learning { #2511.05392 }

    *Minh-Khanh Luong, Chanh Kieu* · Nov 2025
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2511.05392">Traditional methods for enhancing tropical cyclone (TC) intensity from climate model outputs or projections have primarily relied on either dynamical or statistical downscaling. With recent advances...</span><span class="abstract-full" id="full-2511.05392" hidden>Traditional methods for enhancing tropical cyclone (TC) intensity from climate model outputs or projections have primarily relied on either dynamical or statistical downscaling. With recent advances in deep learning (DL) techniques, a natural question is whether DL can provide an alternative approach for improving TC intensity estimation from climate data. Using a common DL architecture based on convolutional neural networks (CNN) and selecting a set of key environmental features, we show that both TC intensity and structure can be effectively downscaled from climate reanalysis data as compared to common vortex detection methods, even when applied to coarse-resolution (0.5-degree) data. Our results thus highlight that TC intensity and structure are governed not only by its internal dynamics but also by local environments during TC development, for which DL models can learn and capture beyond the potential intensity framework. The performance of our DL model depends on several factors such as data sampling strategy, season, or the stage of TC development, with root-mean-square errors ranging from 3-9 ms$^{-1}$ for maximum 10 m wind and 10-20 hPa for minimum central pressure. Although these errors are better than direct vortex detection methods, their wide ranges also suggest that 0.5-degree resolution climate data may contain limited TC information for DL models to learn from, regardless of model optimizations or architectures. Possible improvements and challenges in addressing the lack of fine-scale TC information in coarse resolution climate reanalysis datasets will be discussed.</span> <span class="abstract-toggle" data-id="2511.05392">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2511.05392v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2511.05392v1) · [:material-content-copy: BibTeX](../../bibtex/2511.05392.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a>
    { .paper-tags }

-   #### A PDE-Informed Latent Diffusion Model for 2-m Temperature Downscaling { #2510.23866 }

    *Paul Rosu, Muchang Bahng, Erick Jiang, Rico Zhu, Vahid Tarokh* · Oct 2025
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2510.23866">This work presents a physics-conditioned latent diffusion model tailored for dynamical downscaling of atmospheric data, with a focus on reconstructing high-resolution 2-m temperature fields. Building...</span><span class="abstract-full" id="full-2510.23866" hidden>This work presents a physics-conditioned latent diffusion model tailored for dynamical downscaling of atmospheric data, with a focus on reconstructing high-resolution 2-m temperature fields. Building upon a pre-existing diffusion architecture and employing a residual formulation against a reference UNet, we integrate a partial differential equation (PDE) loss term into the model's training objective. The PDE loss is computed in the full resolution (pixel) space by decoding the latent representation and is designed to enforce physical consistency through a finite-difference approximation of an effective advection-diffusion balance. Empirical observations indicate that conventional diffusion training already yields low PDE residuals, and we investigate how fine-tuning with this additional loss further regularizes the model and enhances the physical plausibility of the generated fields. The entirety of our codebase is available on Github, for future reference and development.</span> <span class="abstract-toggle" data-id="2510.23866">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2510.23866v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2510.23866v1) · [:material-content-copy: BibTeX](../../bibtex/2510.23866.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a>
    { .paper-tags }

-   #### Sparse Local Implicit Image Function for sub-km Weather Downscaling { #2510.20228 }

    *Yago del Valle Inclan Redondo, Enrique Arriaga-Varela, Dmitry Lyamzin, Pablo Cervantes et al.* · Oct 2025
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2510.20228">We introduce SpLIIF to generate implicit neural representations and enable arbitrary downscaling of weather variables. We train a model from sparse weather stations and topography over Japan and...</span><span class="abstract-full" id="full-2510.20228" hidden>We introduce SpLIIF to generate implicit neural representations and enable arbitrary downscaling of weather variables. We train a model from sparse weather stations and topography over Japan and evaluate in- and out-of-distribution accuracy predicting temperature and wind, comparing it to both an interpolation baseline and CorrDiff. We find the model to be up to 50% better than both CorrDiff and the baseline at downscaling temperature, and around 10-20% better for wind.</span> <span class="abstract-toggle" data-id="2510.20228">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2510.20228v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2510.20228v1) · [:material-content-copy: BibTeX](../../bibtex/2510.20228.bib){ .bibtex-link }
    { .paper-links }

-   #### Assessing the Geographic Generalization and Physical Consistency of Generative Models for Climate Downscaling { #2510.13722 }

    *Carlo Saccardi, Maximilian Pierzyna, Haitz Sáez de Ocáriz Borde, Simone Monaco, Cristian Meo et al.* · Oct 2025
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2510.13722">Kilometer-scale weather data is crucial for real-world applications but remains computationally intensive to produce using traditional weather simulations. An emerging solution is to use deep...</span><span class="abstract-full" id="full-2510.13722" hidden>Kilometer-scale weather data is crucial for real-world applications but remains computationally intensive to produce using traditional weather simulations. An emerging solution is to use deep learning models, which offer a faster alternative for climate downscaling. However, their reliability is still in question, as they are often evaluated using standard machine learning metrics rather than insights from atmospheric and weather physics. This paper benchmarks recent state-of-the-art deep learning models and introduces physics-inspired diagnostics to evaluate their performance and reliability, with a particular focus on geographic generalization and physical consistency. Our experiments show that, despite the seemingly strong performance of models such as CorrDiff, when trained on a limited set of European geographies (e.g., central Europe), they struggle to generalize to other regions such as Iberia, Morocco in the south, or Scandinavia in the north. They also fail to accurately capture second-order variables such as divergence and vorticity derived from predicted velocity fields. These deficiencies appear even in in-distribution geographies, indicating challenges in producing physically consistent predictions. We propose a simple initial solution: introducing a power spectral density loss function that empirically improves geographic generalization by encouraging the reconstruction of small-scale physical structures. The code for reproducing the experimental results can be found at https://github.com/CarloSaccardi/PSD-Downscaling</span> <span class="abstract-toggle" data-id="2510.13722">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2510.13722v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2510.13722v1) · [:fontawesome-brands-github: Code](https://github.com/CarloSaccardi/PSD-Downscaling) · [:material-content-copy: BibTeX](../../bibtex/2510.13722.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=km-scale" data-tag="km-scale">Km-scale</a>
    { .paper-tags }

-   #### Km-scale dynamical downscaling through conformalized latent diffusion models { #2510.13301 }

    *Alessandro Brusaferri, Andrea Ballarino* · Oct 2025
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2510.13301">Dynamical downscaling is crucial for deriving high-resolution meteorological fields from coarse-scale simulations, enabling detailed analysis for critical applications such as weather forecasting and...</span><span class="abstract-full" id="full-2510.13301" hidden>Dynamical downscaling is crucial for deriving high-resolution meteorological fields from coarse-scale simulations, enabling detailed analysis for critical applications such as weather forecasting and renewable energy modeling. Generative Diffusion models (DMs) have recently emerged as powerful data-driven tools for this task, offering reconstruction fidelity and more scalable sampling supporting uncertainty quantification. However, DMs lack finite-sample guarantees against overconfident predictions, resulting in miscalibrated grid-point-level uncertainty estimates hindering their reliability in operational contexts. In this work, we tackle this issue by augmenting the downscaling pipeline with a conformal prediction framework. Specifically, the DM's samples are post-processed to derive conditional quantile estimates, incorporated into a conformalized quantile regression procedure targeting locally adaptive prediction intervals with finite-sample marginal validity. The proposed approach is evaluated on ERA5 reanalysis data over Italy, downscaled to a 2-km grid. Results demonstrate grid-point-level uncertainty estimates with markedly improved coverage and stable probabilistic scores relative to the DM baseline, highlighting the potential of conformalized generative models for more trustworthy probabilistic downscaling to high-resolution meteorological fields.</span> <span class="abstract-toggle" data-id="2510.13301">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2510.13301v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2510.13301v1) · [:material-content-copy: BibTeX](../../bibtex/2510.13301.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a> <a class="md-tag" href="/explore/?t=km-scale" data-tag="km-scale">Km-scale</a>
    { .paper-tags }

-   #### CERA: A Framework for Improved Generalization of Machine Learning Models to Changed Climates { #2509.00010 }

    *Shuchang Liu, Paul A. O'Gorman* · Sep 2025
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2509.00010">Robust generalization under climate change remains a major challenge for machine learning applications in climate science. Most existing approaches struggle to extrapolate beyond the climate they...</span><span class="abstract-full" id="full-2509.00010" hidden>Robust generalization under climate change remains a major challenge for machine learning applications in climate science. Most existing approaches struggle to extrapolate beyond the climate they were trained on, leading to a strong dependence on training data from model simulations of warm climates. Use of climate-invariant inputs improves generalization but requires challenging manual feature engineering. Here, we present CERA (Climate-invariant Encoding through Representation Alignment), a machine learning framework consisting of an autoencoder with explicit latent-space alignment, followed by a predictor for downstream process estimation. We test CERA on the problem of parameterizing moist-physics processes. Without training on labeled data from a +4K climate, CERA leverages labeled control-climate data and unlabeled warmer-climate inputs to improve generalization to the warmer climate, outperforming both raw-input and physically informed baselines in predicting key moisture and energy tendencies. It captures not only the vertical and meridional structures of the moisture tendencies, but also shifts in the intensity distribution of precipitation including extremes. Ablation experiments show that latent alignment improves both accuracy and the robustness across random seeds used in training. While some reduced skill remains in the boundary layer, the framework offers a data-driven alternative to manual feature engineering of climate invariant inputs. Beyond parameterizations used in hybrid ML-physics systems, the approach holds promise for other climate applications such as statistical downscaling.</span> <span class="abstract-toggle" data-id="2509.00010">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2509.00010v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2509.00010v1) · [:material-content-copy: BibTeX](../../bibtex/2509.00010.bib){ .bibtex-link }
    { .paper-links }

</div>

<nav class="pager" markdown="span">**1** [2](2.md) [Older :material-arrow-right:](2.md){ .pager-step }</nav>

