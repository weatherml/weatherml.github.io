---
title: 'Ocean & Sea Ice'
hide:
  - toc
---

<div class="listing-header" markdown>

# Ocean & Sea Ice

<p class="page-meta" markdown="span">83 papers · page 1 of 3 · <a href="../../bib/ocean-sea-ice.bib" download>:material-download: BibTeX for this topic</a></p>

</div>

<div class="grid cards" markdown>

-   #### How well is surface ocean carbon represented in observations and ocean models? { #2609.00133 }

    *Viviana Acquaviva, Romina Wild, Alessandro Laio, Amanda R. Fay, Thea H. Heimdal, Galen A. McKinley* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.00133">We introduce a general framework for quantifying the information content and representation quality of complex geophysical datasets based on the intrinsic dimension and differentiable information...</span><span class="abstract-full" id="full-2609.00133" hidden>We introduce a general framework for quantifying the information content and representation quality of complex geophysical datasets based on the intrinsic dimension and differentiable information imbalance of data manifolds. We use it to derive and compare optimal representations of surface ocean carbon in the SOCAT database of observations and in global ocean biogeochemistry models (GOBMs) and to assess the robustness of the information we can extract from existing data. We find that within the most widely used feature set, the complexity of the data space of SOCAT observations is not fully captured by GOBMs, but the ranking and relative importance of variables learned through GOBMs are substantially correct. We observe that the learned representation of ocean carbon is less accurate in some regions, including the Southern Ocean, but doesn't appear to have evolved significantly over the last two decades. Finally, we show how the optimal representations can be used to improve the skill of distance-based machine learning models and demonstrate it for ocean carbon, and we propose two new metrics to compare models and observations that can be used to build more accurate weighted ensembles of estimates.</span> <span class="abstract-toggle" data-id="2609.00133">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.00133v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.00133v1) · [:material-content-copy: BibTeX](../../bibtex/2609.00133.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=benchmarks-datasets" data-tag="benchmarks-datasets">Benchmarks & datasets</a> <a class="md-tag" href="/explore/?t=evaluation" data-tag="evaluation">Evaluation</a>
    { .paper-tags }

-   #### Decadal wave reconstruction in the Mediterranean Sea with graph neural networks { #2608.16449 }

    *Federica Benassi, Lorenzo Mentaschi, Salvatore Causio, Daniel Holmberg, Ivan Federico, Nadia Pinardi* · Aug 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2608.16449">Accurate simulation and prediction of ocean waves are essential for coastal risk management and climate studies. Deep learning has shown promising results for wave modeling, but most approaches still...</span><span class="abstract-full" id="full-2608.16449" hidden>Accurate simulation and prediction of ocean waves are essential for coastal risk management and climate studies. Deep learning has shown promising results for wave modeling, but most approaches still operate on regular grids and on forecasting time scales, and do not generalize to unstructured discretization or to long time horizons. Here we present WaveGraph, a model based on Graph Neural Networks (GNNs) that emulates basin-scale wave dynamics directly on unstructured meshes with high resolution along the coasts (up to 2-3 km). Trained on bias-corrected simulation data over the Mediterranean Sea, WaveGraph uses a multiscale architecture combining the unstructured model mesh with a uniform graph, allowing simultaneous representation of local coastal interactions and large-scale wave dynamics. The model reconstructs the evolution of significant wave height, mean period, and mean direction, and is applied autoregressively for a continuous 17-year period without reinitialization or drift. Validation against buoy and satellite observations shows skill comparable to the input data set, and ablation experiments indicate that wind forcing drives most of the long-term stability while wave history improves swell-driven and basin-scale dynamics. These results show that GNNs can provide stable and efficient emulators of spectral wave models on unstructured domains, enabling decadal wave reconstructions.</span> <span class="abstract-toggle" data-id="2608.16449">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2608.16449v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2608.16449v1) · [:material-content-copy: BibTeX](../../bibtex/2608.16449.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=graph-neural-networks" data-tag="graph-neural-networks">Graph neural networks</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a> <a class="md-tag" href="/explore/?t=km-scale" data-tag="km-scale">Km-scale</a>
    { .paper-tags }

-   #### DLESyM-Ocean: A Deep Learning Probabilistic Global Model for Simulating Present-Day Upper Ocean and Sea Ice { #2608.11545 }

    *Zachary I Espinosa, Nathaniel Cresswell-Clay, William Yik, Cecilia M. Bitz et al.* · Aug 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2608.11545">While AI has shown remarkable promise in atmospheric and meteorological forecasting, accurately simulating other components of the Earth system with AI remains an active frontier. We present...</span><span class="abstract-full" id="full-2608.11545" hidden>While AI has shown remarkable promise in atmospheric and meteorological forecasting, accurately simulating other components of the Earth system with AI remains an active frontier. We present DLESyM-Ocean, a Deep Learning Earth System Model that simulates global present-day sea ice and upper ocean conditions. Unlike conventional probabilistic models optimized via diffusion objectives or losses such as continuous-ranked probability score, DLESyM-Ocean is trained using a patch energy score loss. When driven by atmospheric forcing, DLESyM-Ocean produces a well-calibrated, spatially coherent, and skillful ensemble of sea ice and upper ocean conditions with minimal bias relative to reanalysis products. DLESyM-Ocean is stable when autoregressively run for multi-year simulations and produces a climatology and variability with minimal bias compared with reanalysis. We evaluate case studies including a recent sea ice extreme, a severe marine heatwave, the 2023 El Niño transition, and the 2023 spike in global mean temperature. In all of these case studies, DLESyM-Ocean produces realistic surface and subsurface trajectories and ample ensemble diversity in response to common atmospheric forcing, suggestive of learned autoregressive ocean dynamics. When coupled with other Earth system components, such as the atmosphere, the computational efficiency of DLESyM-Ocean makes it a promising tool for subseasonal to seasonal forecasting.</span> <span class="abstract-toggle" data-id="2608.11545">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2608.11545v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2608.11545v1) · [:material-content-copy: BibTeX](../../bibtex/2608.11545.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a>
    { .paper-tags }

-   #### Stochastic Emulation of a Fully Coupled Preindustrial E3SMv3 Simulation { #2608.10277 }

    *Elynn Wu, James P. C. Duncan, Troy Arcomano, Jeremy McGibbon, Oliver Watt-Meyer et al.* · Aug 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2608.10277">We present a stochastic coupled emulator of E3SM version 3, built on the SamudrACE framework, which couples an atmosphere emulator (ACE2) with a full-depth ocean emulator (Samudra). We replace the...</span><span class="abstract-full" id="full-2608.10277" hidden>We present a stochastic coupled emulator of E3SM version 3, built on the SamudrACE framework, which couples an atmosphere emulator (ACE2) with a full-depth ocean emulator (Samudra). We replace the deterministic atmosphere emulator with its stochastic counterpart, ACE2S, and fine-tune the coupled system with a probabilistic objective, so that the atmosphere acts as a source of internal variability for the ocean. Trained on 105 years of a pre-industrial control simulation and evaluated on an independent 400 years, the emulator reproduces E3SMv3's mean climate state with biases much smaller than existing model-to-observation differences. Relative to a deterministic baseline, stochastic training maintains internal variability across timescales, most notably in the ENSO power spectrum, eddy-rich SST anomalies, and sea ice variability in the marginal ice zone. The emulator captures daily precipitation accurately up to the 99.99th percentile, but underestimates the rarest tropical extremes. These results show that stochastic coupled emulators can reproduce long-timescale variability with high fidelity, while extrapolation to unseen extremes remains a key challenge.</span> <span class="abstract-toggle" data-id="2608.10277">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2608.10277v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2608.10277v1) · [:material-content-copy: BibTeX](../../bibtex/2608.10277.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=daily" data-tag="daily">Daily</a>
    { .paper-tags }

-   #### FESOM2-JAX v1.0: a differentiable shadow of the ocean-sea-ice model FESOM2, cast onto GPUs { #2608.01546 }

    *Nikolay V. Koldunov, Sergey Danilov, Suvarchal Cheedela, Dmitry Sidorenko, Sebastian Beyer et al.* · Aug 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2608.01546">We present FESOM2-JAX, a Python re-implementation of the Finite-volumE Sea ice-Ocean Model (FESOM2) in JAX. The model retains the unstructured-mesh, cell-vertex finite-volume formulation of the...</span><span class="abstract-full" id="full-2608.01546" hidden>We present FESOM2-JAX, a Python re-implementation of the Finite-volumE Sea ice-Ocean Model (FESOM2) in JAX. The model retains the unstructured-mesh, cell-vertex finite-volume formulation of the original, runs unchanged from a laptop CPU to 256 GPUs, and is end-to-end differentiable. FESOM2-JAX is a code shadow of the Fortran model: a projection onto the Python ecosystem, translated with large language models and verified kernel by kernel against the original. It is built to lower the barrier to experimentation, from new numerics and parameterizations to gradient-based calibration and hybrid physics-machine-learning components, while remaining close enough to the original so that what is developed in the shadow can be transferred back. In a 1958-2019 hindcast at 1$^{\circ}$ equivalent resolution with identical physics and forcing, the mean states of the JAX and Fortran versions differ from each other by two orders of magnitude less than either differs from observations, and the two runs agree for six decades in global temperature, salinity, heat content, and sea ice. The complete 1$^{\circ}$ configuration fits on a single GPU, a node of four GH200 superchips integrates $\sim$113 simulated years per wall-clock day, and meshes of up to 7.4 million surface vertices ($\sim$5 km) scale to 128 GPUs. What limits the model is communication rather than arithmetic. What the shadow adds to the original is the gradient: a single reverse-mode pass through the full time loop returns the sensitivity of a model diagnostic to a parameter at every mesh vertex, verified against finite differences. To our knowledge, FESOM2-JAX is the first global ocean-sea-ice model of CMIP-class complexity written natively in a differentiable framework, and the first on an unstructured mesh.</span> <span class="abstract-toggle" data-id="2608.01546">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2608.01546v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2608.01546v1) · [:material-content-copy: BibTeX](../../bibtex/2608.01546.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=physics-ml-hybrid" data-tag="physics-ml-hybrid">Physics–ML hybrid</a> <a class="md-tag" href="/explore/?t=llms-agents" data-tag="llms-agents">LLMs & agents</a> <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a> <a class="md-tag" href="/explore/?t=coarse" data-tag="coarse">Coarse (≥1°)</a>
    { .paper-tags }

-   #### Memory compression and physical state augmentation favor different AMOC prediction tasks { #2607.28468 }

    *Mauricio Herrera-Marín* · Jul 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2607.28468">The Atlantic Meridional Overturning Circulation is monitored and emulated through reduced indices, but such projections discard thermohaline structure and may require either explicit physical state...</span><span class="abstract-full" id="full-2607.28468" hidden>The Atlantic Meridional Overturning Circulation is monitored and emulated through reduced indices, but such projections discard thermohaline structure and may require either explicit physical state or memory of the observed index. We compare these strategies in 30 branch-consistent CMIP6 trajectories from eight model families using leave-one-family-out validation. Salinity, temperature and density information improves direct 20-year forecasts, whereas compact scalar memory is top-ranked at every recursive horizon and yields the lowest case-averaged Brier score. A matched ablation confirms that feedback from memory improves long-horizon prediction. Physical state and recent trends also predict future ocean-state changes beyond the emissions pathway, most robustly at five years. NorESM under SSP5--8.5 identifies a forcing-dependent limit of scalar compression, while MIROC shows negative long-horizon transfer. A resolvent analysis explains why stable memory components do not guarantee stability of the complete learned model. Physical augmentation and memory compression therefore serve different AMOC prediction tasks.</span> <span class="abstract-toggle" data-id="2607.28468">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2607.28468v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2607.28468v1) · [:material-content-copy: BibTeX](../../bibtex/2607.28468.bib){ .bibtex-link }
    { .paper-links }

-   #### Locally stationary Argo ocean heat content estimates: Modeling, validation and uncertainty quantification { #2606.31957 }

    *Thea Sukianto, Mikael Kuusela, Donata Giglio, Anirban Mondal, Pulong Ma, Douglas W. Nychka* · Jun 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2606.31957">Argo profiling floats measure seawater temperature and salinity in the upper 2000 meters of the ocean. These floats are uniquely capable of measuring the global Ocean Heat Content (OHC), a quantity...</span><span class="abstract-full" id="full-2606.31957" hidden>Argo profiling floats measure seawater temperature and salinity in the upper 2000 meters of the ocean. These floats are uniquely capable of measuring the global Ocean Heat Content (OHC), a quantity that is of central importance for understanding Earth Energy Imbalance. Yet, producing Argo-based OHC estimates with reliable uncertainties is statistically challenging due to the complex structure and large size of the Argo dataset. Here we present an end-to-end mapping and uncertainty quantification framework for Argo-based OHC estimation using state-of-the-art methods from spatio-temporal statistics. The framework is based on modeling vertically integrated Argo temperature profiles as a locally stationary Gaussian process defined over space and time. This enables us to produce computationally tractable OHC anomaly maps based on data-driven decorrelation scales estimated from the Argo observations. Our modeling choices are validated using statistical cross-validation, which demonstrates the importance of including a climatological time trend in the mean field and accounting for time in the covariance function. We quantify the uncertainty of these maps using local conditional simulation ensembles, a novel approach that leads to principled spatially and temporally correlated uncertainty quantification. A new paired cross-validation technique is presented to validate these uncertainties. The mapping framework is implemented in an open-source codebase that is designed to be modular, reproducible and extensible. To demonstrate the mapping and uncertainty quantification capabilities of this approach, we present new Argo OHC maps with uncertainties for 2004-2022 and report on various downstream climatological estimates and their uncertainties.</span> <span class="abstract-toggle" data-id="2606.31957">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2606.31957v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2606.31957v1) · [:material-content-copy: BibTeX](../../bibtex/2606.31957.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=classical-ml" data-tag="classical-ml">Classical ML</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a>
    { .paper-tags }

-   #### Disentangling the effects of sea surface temperature and CO$_2$ in global machine learned weather-climate emulators { #2606.07928 }

    *Spencer K. Clark, Troy Arcomano, James P. C. Duncan, Brian Henn, Anna Kwa, Jeremy McGibbon et al.* · Jun 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2606.07928">While previous versions of the Ai2 Climate Emulator (ACE) have been trained with CO$_2$ as a forcing, they are only accurate within a narrow range of scenarios, for example climate over the last 80...</span><span class="abstract-full" id="full-2606.07928" hidden>While previous versions of the Ai2 Climate Emulator (ACE) have been trained with CO$_2$ as a forcing, they are only accurate within a narrow range of scenarios, for example climate over the last 80 years forced by observed sea surface temperature (SST), sea ice, and CO$_2$ (AMIP), or equilibrium or near-equilibrium climates with CO$_2$ concentrations ranging from 1x to 4x that of the present day. Attempting to simulate climate forced by AMIP SST perturbed by +4 K or the response to an abrupt quadrupling of CO$_2$, results in unphysical behavior. We attribute this to these models being trained on datasets where the SST and CO$_2$ are correlated, limiting their ability to accurately learn their separate effects. In this study we introduce a new class of "random-CO$_2$" reference simulations where the SST and CO$_2$ are prescribed to vary independently. Trained on a balance of AMIP, equilibrium-climate, and random-CO$_2$ data, and including a total energy conservation constraint for improved interpretability, we present a more data-efficient model that not only accurately emulates its reference model in scenarios in which previous models excelled, but also scenarios like AMIP +4 K and slab-ocean-coupled abrupt 4xCO$_2$ where they did not. Limitations are that it has simplified or prescribed representations of other Earth system components like the ocean, land, and sea ice; does not expose other known climate drivers as forcings; and relies solely on physics-based model output for training data, inheriting the biases relative to observations thereof. Each of these represent opportunities for future work.</span> <span class="abstract-toggle" data-id="2606.07928">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2606.07928v2) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2606.07928v2) · [:material-content-copy: BibTeX](../../bibtex/2606.07928.bib){ .bibtex-link }
    { .paper-links }

-   #### Samudra 2: Scaling Ocean Emulators across Resolutions { #2606.02610 }

    *Yuan Yuan, Jesse Rusak, Alexander Merose, Adam Subel, Pavel Perezhogin, Alistair Adcroft et al.* · Jun 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2606.02610">Ocean general circulation models (OGCMs) are essential to climate science but computationally expensive, limiting ensemble size and forcing scenarios. Neural emulators promise orders-of-magnitude...</span><span class="abstract-full" id="full-2606.02610" hidden>Ocean general circulation models (OGCMs) are essential to climate science but computationally expensive, limiting ensemble size and forcing scenarios. Neural emulators promise orders-of-magnitude speedups, yet existing ocean emulators have not combined fine spatial resolution with multi-year autoregressive rollouts. Samudra, the first autoregressive neural ocean emulator to produce multi-decade global rollouts, is limited to $1^\circ$ resolution and exhibits two long-horizon failure modes: <em>variance collapse</em>, the loss of temporal variability, and <em>imprinting artifacts</em>, in which velocity patterns leak into deep-ocean fields. We present Samudra 2, which introduces a wider U-Net backbone with modified ConvNeXt-style blocks and a reduced block-internal expansion factor, together with a dynamic loss that reweights output channels according to their prediction errors, strengthening gradients for slow-evolving deep-ocean fields. At $1^\circ$, Samudra 2 increases upper-ocean global-mean temperature $R^2$ from 0.56 to 0.87 and reduces deep-ocean temperature error by roughly sevenfold. The same architecture scales to $1/2^\circ$ and $1/4^\circ$ over approximately 8-year autoregressive rollouts, recovering mesoscale eddies and sharp western boundary currents. Running on a single GPU, Samudra 2 enables larger ensembles for sea-level projections, ocean heat uptake, and climate variability studies. All artifacts are publicly available: project page, code, checkpoints, documentation.</span> <span class="abstract-toggle" data-id="2606.02610">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2606.02610v2) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2606.02610v2) · [:material-content-copy: BibTeX](../../bibtex/2606.02610.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=coarse" data-tag="coarse">Coarse (≥1°)</a>
    { .paper-tags }

-   #### Njord: A Probabilistic Graph Neural Network for Ensemble Ocean Forecasting { #2605.15470 }

    *Daniel Holmberg, Joel Oskarsson, Erik Wikingsson, Fredrik Lindsten, Teemu Roos* · May 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2605.15470">Ocean dynamics are inherently chaotic, yet existing machine learning ocean models produce only deterministic forecasts. We introduce Njord, a probabilistic data-driven model for ocean forecasting,...</span><span class="abstract-full" id="full-2605.15470" hidden>Ocean dynamics are inherently chaotic, yet existing machine learning ocean models produce only deterministic forecasts. We introduce Njord, a probabilistic data-driven model for ocean forecasting, applicable to both global and regional domains. Njord combines a deep latent variable framework with a graph neural network architecture, enabling sampling each forecast step in a single forward pass. We apply Njord globally at 0.25° resolution and regionally to the Baltic Sea at 2 km resolution. To scale to these large ocean grids we introduce K-means cluster meshes that adapt to irregular sea surface geometry. Experiments demonstrate strong performance on both domains compared to deterministic machine learning baselines, while also providing uncertainty estimates from the sampled ensemble forecasts. On the global OceanBench benchmark, Njord achieves the lowest errors on average across upper-ocean variables when evaluated against real-world observations, with the largest improvements in surface temperature prediction.</span> <span class="abstract-toggle" data-id="2605.15470">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2605.15470v2) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2605.15470v2) · [:material-content-copy: BibTeX](../../bibtex/2605.15470.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=graph-neural-networks" data-tag="graph-neural-networks">Graph neural networks</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a> <a class="md-tag" href="/explore/?t=km-scale" data-tag="km-scale">Km-scale</a> <a class="md-tag" href="/explore/?t=quarter-degree" data-tag="quarter-degree">0.25°</a>
    { .paper-tags }

-   #### Prediction and Predictability of the Wet-Season Rainfall over Southeast India { #2605.01326 }

    *Harini S, Devabrat Sharma, Yogenraj Patil, Gaurav Chopra, Shruti Tandon, B. N. Goswami, R. I. Sujith* · May 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2605.01326">The challenge in predicting sub-regional climate within the Indian monsoon region is exacerbated by its increasing variability in a warming world. While exploring the seasonal predictability of...</span><span class="abstract-full" id="full-2605.01326" hidden>The challenge in predicting sub-regional climate within the Indian monsoon region is exacerbated by its increasing variability in a warming world. While exploring the seasonal predictability of rainfall over the state of Tamil Nadu in southeast India, we identify an overall increase in the monthly rainfall and its variability in recent years due to an increase in surface temperature, water vapour and moisture convergence. We attribute the increasing excess rainfall to a long-term reduction in convective inhibition. We further find an increasing trend in the length of the rainy season due to an earlier onset and a delayed withdrawal of the large-scale monsoon over the southeastern and southwestern regions of southern peninsular India, respectively. Further, the simultaneous (0- month lead) predictability of the primary wet-season (October-December, OND) rainfall over Tamil Nadu is dominated by sea surface temperature (SST) anomalies in the North Indian Ocean. However, a global tropical SST climate network reveals a high potential predictability and potential to realize significant forecast skill at a lead time of up to 10 months. The long-lead predictability arises from SST and rainfall interactions across the tropical Indo-Pacific and equatorial Atlantic regions. Our findings provide a robust data-driven methodology for skillful seasonal rainfall prediction over Tamil Nadu, despite the increasing rainfall variability.</span> <span class="abstract-toggle" data-id="2605.01326">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2605.01326v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2605.01326v1) · [:material-content-copy: BibTeX](../../bibtex/2605.01326.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a> <a class="md-tag" href="/explore/?t=monthly" data-tag="monthly">Monthly</a>
    { .paper-tags }

-   #### An Adaptive Spatiotemporal Clustering Framework for 3D Ocean Subsurface Temperature Reconstruction { #2605.00860 }

    *Ming Shan Loo, Wengen Li, Xudong Jiang, Hailiang Cheng, Zhifei Zhang, Jihong Guan, Yichao Zhang* · May 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2605.00860">The reconstruction of ocean subsurface temperature (OST) using satellite remote sensing data holds significant scientific value for advancing the understanding of ocean dynamics and climate...</span><span class="abstract-full" id="full-2605.00860" hidden>The reconstruction of ocean subsurface temperature (OST) using satellite remote sensing data holds significant scientific value for advancing the understanding of ocean dynamics and climate variability. However, the scarcity of subsurface observations, combined with the high degree of nonlinearity and spatiotemporal heterogeneity in subsurface processes, poses substantial challenges to the accuracy and generalization capability of traditional reconstruction methods. To address these limitations, this study proposes an adaptive framework that could capture both vertical structural dependencies and temporal variation patterns of OST via spatio-temporal clustering. By incorporating this framework with various deep learning models, e.g., dual-path convolutional neural networks (DP-CNN), Attention U-Net, and Vision Transformer (ViT), the OST field can be accurately reconstructed at a global scale only using surface observations, i.e., sea surface temperature (SST), sea surface salinity (SSS), sea surface height (SSH), and sea surface wind (SSW). Experimental results demonstrate that multiple deep learning methods using the proposed framework largely outperform their original counterparts, yielding improvements in RMSE ranging from 12.4% to 27.2%. This study provides a reliable solution for subsurface temperature reconstruction, offering important implications for meteorological modeling and climate change assessment.</span> <span class="abstract-toggle" data-id="2605.00860">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2605.00860v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2605.00860v1) · [:material-content-copy: BibTeX](../../bibtex/2605.00860.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=transformers" data-tag="transformers">Transformers</a> <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a> <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a>
    { .paper-tags }

-   #### Optimal sensor placement for the reconstruction of ocean states using differentiable Gumbel-Softmax sampling operator { #2604.22511 }

    *Oscar Chapron, Ronan Fablet, Yann Stéphan* · Apr 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2604.22511">Accurately reconstructing and forecasting ocean fields from sparse observations is critical for both operational and scientific purposes. Optimizing sensor placement to maximize reconstruction skill...</span><span class="abstract-full" id="full-2604.22511" hidden>Accurately reconstructing and forecasting ocean fields from sparse observations is critical for both operational and scientific purposes. Optimizing sensor placement to maximize reconstruction skill remains challenging due to evolving ocean dynamics and practical deployment constraints. Traditional approaches, such as Empirical Orthogonal Functions, greedy search, or Gaussian processes, either assume static observation networks or scale poorly in high-resolution and non-stationary regimes.   We introduce a differentiable adaptive sensor placement framework based on a Gumbel-Softmax sampling operator. Given an ensemble of forecasts or simulations, the method jointly optimizes a probabilistic sampling mask and the reconstruction mapping (e.g., Optimal Interpolation correlation lengths) under strict observation budgets. Numerical experiments are conducted for Sea Surface Height reconstruction in a Gulf Stream region through Observing-System Simulation Experiments using a state-of-the-art high-resolution ocean simulation.   With a sensor budget of only 0.1% (fewer than 100 point-wise observations on a 14°x14° domain) the optimized sampling reduces the reconstruction RMSE by more than half (0.0908 m versus 0.1750 m) and increases explained variance by about 20% (93.1% versus 74.4%) compared with a uniform random strategy. The method remains robust when trained on noisy ensembles with significant spatial displacement (up to 1°), demonstrating practical applicability under forecast uncertainty.   Overall, the framework provides a scalable, budget-aware approach to designing observation networks. Beyond improved skill, it yields interpretable sampling patterns that consistently target energetic regions such as eddies and fronts, offering a transferable tool for adaptive sensing in geophysical systems.</span> <span class="abstract-toggle" data-id="2604.22511">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2604.22511v2) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2604.22511v2) · [:material-content-copy: BibTeX](../../bibtex/2604.22511.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### Comparing Ocean Forecasts Driven with Machine Learning-based and Physics-based Atmospheric Forcings { #2604.07861 }

    *Xiaobing Zhou, Frank Colberg, Debra Hudson, Yonghong Yin, Griffith Young, Christopher Bladwell et al.* · Apr 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2604.07861">Operational ocean forecasting systems conventionally employ dynamical ocean models driven by atmospheric forcing derived from numerical weather prediction (NWP) models. Recent advancements in...</span><span class="abstract-full" id="full-2604.07861" hidden>Operational ocean forecasting systems conventionally employ dynamical ocean models driven by atmospheric forcing derived from numerical weather prediction (NWP) models. Recent advancements in artificial intelligence and machine learning (ML) have led to the development of ML-based atmospheric weather models, which have competitive, if not better, medium range forecast accuracy compared to traditional NWP systems. This study evaluates the impact of ML-based atmospheric forcing on ocean forecast skill through two sets of 10-day forecasts using the UK Met Office GOSI9 configuration of the NEMO dynamical ocean model. Both experiments share identical ocean initial conditions; but differ in atmospheric forcing: one uses ECMWF's ML-based AIFS model, while the other uses the Australian Bureau of Meteorology's physics-based NWP model, ACCESS-G3. Forecasts were initialized on the first day of each month over the period 2023-2024. The quality of the atmospheric forcing was assessed by comparing AIFS and ACCESS-G3 forecast skill against both ECMWF reanalysis v5 (ERA5) and ACCESS-G3 analyses. Results indicate that AIFS consistently outperforms ACCESS-G3, either from the initial forecast time or after the first few days. Oceanic forecast skill was evaluated against both the GOSI9 reanalysis and observations, focusing on key surface variables including sea surface temperature, salinity, sea level, and ocean currents. The ocean forecasts forced with AIFS atmospheric data exhibit comparable or enhanced predictive skill compared to those forced with ACCESS-G3 data. These findings underscore the potential of ML-based atmospheric models to replace traditional NWP forcing in operational ocean forecasting systems, offering improved accuracy and computational efficiency.</span> <span class="abstract-toggle" data-id="2604.07861">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2604.07861v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2604.07861v1) · [:material-content-copy: BibTeX](../../bibtex/2604.07861.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=evaluation" data-tag="evaluation">Evaluation</a>
    { .paper-tags }

-   #### What's in the latent space? Exploring coupled tropical Pacific variability within a Multi-branch $β$-Variational Autoencoder { #2604.07137 }

    *Emily F. Wisinski, Maria J. Molina, Kyle J. C. Hall, Hannah Bao, Salil Mahajan, Nan Rosenbloom et al.* · Apr 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2604.07137">What is encoded in the latent space of a multi-branch $β$-variational autoencoder ($β$-VAE) trained on coupled tropical Pacific climate fields? To answer this question, we assess the reconstruction...</span><span class="abstract-full" id="full-2604.07137" hidden>What is encoded in the latent space of a multi-branch $β$-variational autoencoder ($β$-VAE) trained on coupled tropical Pacific climate fields? To answer this question, we assess the reconstruction skill and physical interpretability of the latent space of a multi-branch $β$-VAE trained on sea surface temperature, ocean heat content, and outgoing longwave radiation across the tropical Pacific from a 500-year preindustrial control simulation. The model generalizes well, with only modest degradation from training to test performance, and preserves the dominant basin-scale structure of all three fields. Latent-space diagnostics show that variability is organized unevenly across dimensions: sea surface temperature is concentrated in a smaller subset of latent dimensions, whereas ocean heat content and outgoing longwave radiation are more broadly distributed across multiple dimensions. Comparisons with conventional tropical Pacific diagnostics further show that several latent dimensions align with known El Niño and La Niña variability, while others capture related coupled ocean-atmosphere variability on decadal or longer timescales. Sensitivity experiments and latent traversals identify dimensions associated with eastern-Pacific-like, central-Pacific-like, coastal, subsurface-dominant, and atmosphere-dominant variability. Together, these results show that the multi-branch $β$-variational autoencoder yields a skillful and physically informative reduced representation of coupled tropical Pacific variability.</span> <span class="abstract-toggle" data-id="2604.07137">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2604.07137v2) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2604.07137v2) · [:material-content-copy: BibTeX](../../bibtex/2604.07137.bib){ .bibtex-link }
    { .paper-links }

-   #### Calibration of a neural network ocean closure for improved mean state and variability { #2604.06398 }

    *Pavel Perezhogin, Alistair Adcroft, Laure Zanna* · Apr 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2604.06398">Global ocean models exhibit biases in the mean state and variability, particularly at coarse resolution, where mesoscale eddies are unresolved. To address these biases, parameterization coefficients...</span><span class="abstract-full" id="full-2604.06398" hidden>Global ocean models exhibit biases in the mean state and variability, particularly at coarse resolution, where mesoscale eddies are unresolved. To address these biases, parameterization coefficients are typically tuned ad hoc. Here, we formulate parameter tuning as a calibration problem using Ensemble Kalman Inversion (EKI). We optimize parameters of a neural network parameterization of mesoscale eddies in two idealized ocean models at coarse resolution. The calibrated parameterization reduces errors by factors of 1.7-3.3 in the time-averaged fluid interfaces and their variability compared to the unparameterized model, depending on the metric and configuration. The EKI method is robust to noise in time-averaged statistics arising from chaotic ocean dynamics. Furthermore, we propose an efficient calibration protocol that bypasses integration to statistical equilibrium by carefully choosing an initial condition. These results demonstrate that systematic calibration can substantially improve coarse-resolution ocean simulations and provide a practical pathway for reducing biases in global ocean models.</span> <span class="abstract-toggle" data-id="2604.06398">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2604.06398v2) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2604.06398v2) · [:material-content-copy: BibTeX](../../bibtex/2604.06398.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a>
    { .paper-tags }

-   #### Impact of geophysical fields on Deep Learning-based Lagrangian drift simulations { #2604.03292 }

    *Daria Botvynko, Carlos Granero-Belinchon, Simon Van Gennip, Abdesslam Benzinou, Ronan Fablet* · Apr 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2604.03292">We assess the influence of different Eulerian geophysical input fields on Lagrangian drift simulations using DriftNet, a learning-based method designed to simulate Lagrangian drift on the sea...</span><span class="abstract-full" id="full-2604.03292" hidden>We assess the influence of different Eulerian geophysical input fields on Lagrangian drift simulations using DriftNet, a learning-based method designed to simulate Lagrangian drift on the sea surface. Two experiments are conducted: a fully numerical experiment (Benchmark B1) and a real-world drifters-based experiment (Benchmark B2). Both experiments are performed in two regions with different ocean dynamics: North East Pacific and Gulf Stream regions. The performance of DrifNet is evaluated with three different metrics: separation distance between simulated and ground-truth trajectories, the normalized cumulative Lagrangian separation and the autocorrelation of Lagrangian velocities. In both regions, results from B1 show that combining assimilated sea surface currents (SSC) with fully observed sea surface height (SSH) leads to greatest improvement in trajectory simulation. This configuration reduces separation distance by over 50% and significantly decreases normalized cumulative Lagrangian separation and metrics related to velocities autocorrelation functions compared to the baseline using SSC alone. On the other hand, the inclusion of sea surface temperature (SST) either alone or in combination with SSC generally degrades performance. In B2, using satellite-derived SSH, Ekman and winds velocities improves surface drifters trajectories simulation, particularly in the North East Pacific. While the satellite-derived SST in combination with reanalysis-based SSC configuration leads to better trajectories simulation in the Gulf Stream. Overall, we highlight the added value of combining multiple geophysical fields to improve Lagrangian drift simulation on both numerical and real-world experiments.</span> <span class="abstract-toggle" data-id="2604.03292">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2604.03292v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2604.03292v1) · [:material-content-copy: BibTeX](../../bibtex/2604.03292.bib){ .bibtex-link }
    { .paper-links }

-   #### High-resolution probabilistic estimation of three-dimensional regional ocean dynamics from sparse surface observations { #2604.02850 }

    *Niloofar Asefi, Tianning Wu, Ruoying He, Ashesh Chattopadhyay* · Apr 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2604.02850">The ocean interior regulates Earth's climate but remains sparsely observed due to limited in situ measurements, while satellite observations are restricted to the surface. We present a depth-aware...</span><span class="abstract-full" id="full-2604.02850" hidden>The ocean interior regulates Earth's climate but remains sparsely observed due to limited in situ measurements, while satellite observations are restricted to the surface. We present a depth-aware generative framework for reconstructing high-resolution three-dimensional ocean states from extremely sparse surface data. Our approach employs a conditional denoising diffusion probabilistic model (DDPM) trained on sea surface height and temperature observations with up to 99.9 percent sparsity, without reliance on a background dynamical model. By incorporating continuous depth embeddings, the model learns a unified vertical representation of the ocean states and generalizes to previously unseen depths. Applied to the Gulf of Mexico, the framework accurately reconstructs subsurface temperature, salinity, and velocity fields across multiple depths. Evaluations using statistical metrics, spectral analysis, and heat transport diagnostics demonstrate recovery of both large-scale circulation and multiscale variability. These results establish generative diffusion models as a scalable approach for probabilistic ocean reconstruction in data-limited regimes, with implications for climate monitoring and forecasting.</span> <span class="abstract-toggle" data-id="2604.02850">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2604.02850v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2604.02850v1) · [:material-content-copy: BibTeX](../../bibtex/2604.02850.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### CNN-based forecasting of early winter NAO using sea surface temperature { #2603.16312 }

    *Elena Provenzano, Guillaume Gastineau, Carlos Mejia, Didier Swingedouw, Sylvie Thiria* · Mar 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2603.16312">The North Atlantic Oscillation (NAO) is the dominant mode of atmospheric variability over the North Atlantic sector, influencing temperature and precipitation across Europe. While the NAO's impact on...</span><span class="abstract-full" id="full-2603.16312" hidden>The North Atlantic Oscillation (NAO) is the dominant mode of atmospheric variability over the North Atlantic sector, influencing temperature and precipitation across Europe. While the NAO's impact on North Atlantic sea surface temperatures (SSTs) is well understood, the NAO can also be driven by SST anomalies. However, this NAO response to SST anomalies is believed to be weak and nonlinear. Former studies highlight that during early winter (November-December), El Nino Southern Oscillation (ENSO) events modulate the NAO, with El Nino (La Nina) events being linked to positive (negative) NAO phases, and an opposite effect observed in late winter (January-February). Indian Ocean SSTs and the North Atlantic Horseshoe SST anomaly have also been suggested as contributors to early winter NAO variability. However, climate models often struggle to capture these SST-NAO teleconnections, particularly in early winter. To address this, a statistical framework based on convolutional neural networks (CNNs) is developed to predict the early winter NAO using observed SST fields one-, two-, and three-month before. A linear model serves as a benchmark, and both models are trained on ERA5 reanalysis data from 1940 to 2023. A sensitivity analysis is used to interpret the CNN's decision-making process, revealing that it focuses on regions such as the tropical Pacific and North Atlantic, confirming results from previous works. The CNN outperforms the linear model, highlighting the value of capturing nonlinear SST-NAO relationships. Prediction skill appears to be linked to ENSO, with strong ENSO events associated with greater skill in forecasting the NAO than neutral events. These findings underscore the potential of deep learning to build medium-range NAO prediction.</span> <span class="abstract-toggle" data-id="2603.16312">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2603.16312v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2603.16312v1) · [:material-content-copy: BibTeX](../../bibtex/2603.16312.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a> <a class="md-tag" href="/explore/?t=subseasonal-to-seasonal" data-tag="subseasonal-to-seasonal">Subseasonal to seasonal</a>
    { .paper-tags }

-   #### Probabilistic reconstruction of global sea surface temperature using generative diffusion models { #2603.16272 }

    *Haijie Li, Ya Wang, Kai Yang, Gang Huang, Xiangao Xia, Ziming Chen, Weichen Tao, Chenglin Lyu et al.* · Mar 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2603.16272">Accurate reconstruction of global Sea surface temperature (SST), which dominates the air-sea coupling and global climate variability, underpins climate monitoring and prediction. Existing SST...</span><span class="abstract-full" id="full-2603.16272" hidden>Accurate reconstruction of global Sea surface temperature (SST), which dominates the air-sea coupling and global climate variability, underpins climate monitoring and prediction. Existing SST reconstruction products primarily provide one deterministic field derived from heterogeneous satellite data and in situ observations, limiting their ability to represent observation uncertainty and to support probabilistic forecasting. Here, we introduce Satellite and in situ Adaptive Guided Estimation (SAGE), a diffusion-based uncertainty-aware generative framework for probabilistic SST reconstruction. SAGE learns a physically consistent prior from historical SST data and performs observation-conditioned posterior sampling without requiring satellite or in situ data during training, enabling flexible state inference from heterogeneous observations. Through a progressive data-fusion strategy, observations from two FengYun-3D polar-orbiting satellites constrain basin-scale structures, while sparse in situ measurements serve to refine local anomalies and extremes. The resulting ensemble SST fields well capture observational uncertainty and scale-dependent variability. Validation against independent in situ observations shows that SAGE substantially reduces reconstruction errors compared with widely used operational products. When used to initialize forecasting systems, SAGE-generated SST fields substantially reduce 10-day SST forecast errors relative to current operational analyses. At the climate scale, SAGE-driven forecasts of the 2023-2024 El Nino event show added value in capturing its onset and intensity evolution compared to conventional approaches. Our results demonstrate that SAGE represents a step toward a new paradigm for ocean state estimation and climate prediction.</span> <span class="abstract-toggle" data-id="2603.16272">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2603.16272v2) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2603.16272v2) · [:material-content-copy: BibTeX](../../bibtex/2603.16272.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a> <a class="md-tag" href="/explore/?t=station-point" data-tag="station-point">Station / point</a>
    { .paper-tags }

-   #### FloeNet: A mass-conserving global sea ice emulator that generalizes across climates { #2603.12449 }

    *William Gregory, Mitchell Bushuk, James Duncan, Elynn Wu, Adam Subel, Spencer K. Clark, Bill Hurlin et al.* · Mar 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2603.12449">We introduce FloeNet, a machine-learning emulator trained on the Geophysical Fluid Dynamics Laboratory global sea ice model, SIS2. FloeNet is a mass-conserving model, emulating 6-hour mass and area...</span><span class="abstract-full" id="full-2603.12449" hidden>We introduce FloeNet, a machine-learning emulator trained on the Geophysical Fluid Dynamics Laboratory global sea ice model, SIS2. FloeNet is a mass-conserving model, emulating 6-hour mass and area budget tendencies related to sea ice and snow-on-sea-ice growth, melt, and advection. We train FloeNet using simulated data from a reanalysis-forced ice-ocean simulation and test its ability to generalize to pre-industrial control and 1% CO2 climates. FloeNet outperforms a non-conservative model at reproducing sea ice and snow-on-sea-ice mean state, trends, and inter-annual variability, with volume anomaly correlations above 0.96 in the Antarctic and 0.76 in the Arctic, across all forcings. FloeNet also produces the correct thermodynamic vs dynamic response to forcing, enabling physical interpretability of emulator output. Finally, we show that FloeNet outputs high-fidelity coupling-related variables, including ice-surface skin temperature, ice-to-ocean salt flux, and melting energy fluxes. We hypothesize that FloeNet will improve polar climate processes within existing atmosphere and ocean emulators.</span> <span class="abstract-toggle" data-id="2603.12449">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2603.12449v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2603.12449v1) · [:material-content-copy: BibTeX](../../bibtex/2603.12449.bib){ .bibtex-link }
    { .paper-links }

-   #### Reduced-Order Surrogates for Forced Flexible Mesh Coastal-Ocean Models { #2602.05416 }

    *Freja Høgholm Petersen, Jesper Sandvig Mariegaard, Rocco Palmitessa, Allan P. Engsig-Karup* · Feb 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2602.05416">While proper orthogonal decomposition (POD)-based surrogates are widely explored for hydrodynamic applications, the use of Koopman autoencoders for real-world coastal-ocean modelling remains...</span><span class="abstract-full" id="full-2602.05416" hidden>While proper orthogonal decomposition (POD)-based surrogates are widely explored for hydrodynamic applications, the use of Koopman autoencoders for real-world coastal-ocean modelling remains relatively limited. This paper introduces a flexible Koopman autoencoder formulation that incorporates meteorological forcings and boundary conditions, and systematically compares its performance against POD-based surrogates. The Koopman autoencoder employs a learned linear temporal operator in latent space, enabling eigenvalue regularization to promote temporal stability. This strategy is evaluated alongside temporal unrolling techniques for achieving stable and accurate long-term predictions. The models are assessed on three test cases spanning distinct dynamical regimes, with prediction horizons up to one year at 30-minute temporal resolution. Across all cases, the reduced order surrogates with temporal unrolling achieve high accuracy with relative root-mean-squared-errors of 0.0068-0.14 and $R^2$-values of 0.61-0.995, where prediction errors are largest for current velocities, and smallest for water surface elevations. In two of the three cases, the Koopman Autoencoder have higher accuracy than the POD-based surrogates. Comparing to in-situ observations, the surrogate yields -0.64% to 12% increase in water surface elevation prediction error when compared to prediction errors of the physics-based model. These error levels, corresponding to a few centimeters, are acceptable for many practical applications, while inference speed-ups of 300-1400x enables workflows such as ensemble forecasting and long climate simulations for coastal-ocean modelling.</span> <span class="abstract-toggle" data-id="2602.05416">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2602.05416v2) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2602.05416v2) · [:material-content-copy: BibTeX](../../bibtex/2602.05416.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=sub-hourly" data-tag="sub-hourly">Sub-hourly</a>
    { .paper-tags }

-   #### Large-Ensemble Simulations Reveal Links Between Atmospheric Blocking Frequency and Sea Surface Temperature Variability { #2602.05083 }

    *Zilu Meng, Gregory J. Hakim, Wenchang Yang, Gabriel A. Vecchi* · Feb 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2602.05083">Atmospheric blocking events drive persistent weather extremes in midlatitudes, but isolating the influence of sea surface temperature (SST) from chaotic internal atmospheric variability on these...</span><span class="abstract-full" id="full-2602.05083" hidden>Atmospheric blocking events drive persistent weather extremes in midlatitudes, but isolating the influence of sea surface temperature (SST) from chaotic internal atmospheric variability on these events remains a challenge. We address this challenge using century-long (1900-2010), large-ensemble simulations with two computationally efficient deep-learning general circulation models. We find these models skillfully reproduce the observed blocking climatology, matching or exceeding the performance of a traditional high-resolution model and representative CMIP6 models. Averaging the large ensembles filters internal atmospheric noise to isolate the SST-forced component of blocking variability, yielding substantially higher correlations with reanalysis than for individual ensemble members. We identify robust teleconnections linking Greenland blocking frequency to North Atlantic SST and El Niño-like patterns. Furthermore, SST-forced trends in blocking frequency show a consistent decline in winter over Greenland, and an increase over Europe. These results demonstrate that SST variability exerts a significant and physically interpretable influence on blocking frequency and establishes large ensembles from deep learning models as a powerful tool for separating forced SST signals from internal noise.</span> <span class="abstract-toggle" data-id="2602.05083">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2602.05083v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2602.05083v1) · [:material-content-copy: BibTeX](../../bibtex/2602.05083.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=subseasonal-to-seasonal" data-tag="subseasonal-to-seasonal">Subseasonal to seasonal</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a>
    { .paper-tags }

-   #### Hybrid physics-data-driven modeling for sea ice thermodynamics and transfer learning { #2601.23190 }

    *Giovanni De Cillis, Alberto Carrassi, Julien Brajard, Laurent Bertino, Matteo Broccoli et al.* · Jan 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2601.23190">This study explores a physics-data driven hybrid approach for sea-ice column physics models, in which a machine learning (ML) component acts as a state-dependent parameterization of forecast errors....</span><span class="abstract-full" id="full-2601.23190" hidden>This study explores a physics-data driven hybrid approach for sea-ice column physics models, in which a machine learning (ML) component acts as a state-dependent parameterization of forecast errors. We examine how perturbations in snow thermodynamics and sea-ice radiative properties affect forecast errors, and train dedicated neural networks (NNs) for each model configuration. The performance of the hybrid models is evaluated for long lead-time forecasts and compared against a benchmark system based on climatological forecast-error estimates. The NN-based hybrids prove to be stable, robust to initial condition and atmospheric forcing errors, and consistently outperform their climatology-based counterpart. To derive guiding principles for efficiently handling possible physical model updates, we perform transfer learning experiments to test whether pretrained NNs optimized for one model configuration can be successfully adapted to another. Results indicate that direct evaluation of pretrained networks on the target task provides useful insights into their adaptability, recommending transfer learning whenever performance exceeds a trivial baseline. Finally, a feature-importance analysis shows that atmospheric forcing inputs have negligible influence on NN predictive skill, while ice-layer enthalpies play a key role in achieving satisfactory performance.</span> <span class="abstract-toggle" data-id="2601.23190">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2601.23190v2) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2601.23190v2) · [:material-content-copy: BibTeX](../../bibtex/2601.23190.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=physics-ml-hybrid" data-tag="physics-ml-hybrid">Physics–ML hybrid</a>
    { .paper-tags }

-   #### Rapid estimation of global sea surface temperatures from sparse streaming in situ observations { #2601.21913 }

    *Cassidy All, Kevin Ho, Maya Magnuski, Christopher Nicolaides, Louisa B. Ebby, Mohammad Farazmand* · Jan 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2601.21913">Reconstructing high-resolution sea surface temperatures (SST) from staggered SST measurements is essential for weather forecasting and climate projections. However, when SST measurements are sparse,...</span><span class="abstract-full" id="full-2601.21913" hidden>Reconstructing high-resolution sea surface temperatures (SST) from staggered SST measurements is essential for weather forecasting and climate projections. However, when SST measurements are sparse, the resulting inferred SST fields are rather inaccurate. Here, we demonstrate the ability of Sparse Discrete Empirical Interpolation Method (S-DEIM) to reconstruct the high-resolution SST field from sparse in situ observations, without using a model. The S-DEIM estimate consists of two terms, one computed from instantaneous in situ observations using empirical interpolation, and the other learned from the historical time series of observations using recurrent neural networks (RNNs). We train the RNNs using the National Oceanic and Atmospheric Administration's weekly high-resolution SST dataset spanning the years 1989-2021 which constitutes the training data. Subsequently, we examine the performance of S-DEIM on the test data, comprising January 2022 to January 2023. For this test data, S-DEIM infers the high-resolution SST from 100 in situ observations, constituting only 0.2% of the high-resolution spatial grid. We show that the resulting S-DEIM reconstructions are about 40% more accurate than earlier empirical interpolation methods, such as DEIM and Q-DEIM. Furthermore, 91% of S-DEIM estimates fall within $\pm 1^\circ$C of the true SST. We also demonstrate that S-DEIM is robust with respect to sensor placement: even when the sensors are distributed randomly, S-DEIM reconstruction error deteriorates only by 1-2%. S-DEIM is also computationally efficient. Training the RNN, which is performed only once offline, takes approximately one minute. Once trained, the S-DEIM reconstructions are computed in less than a second. As such, S-DEIM can be used for rapid SST reconstruction from sparse streaming observational data in real time.</span> <span class="abstract-toggle" data-id="2601.21913">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2601.21913v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2601.21913v1) · [:material-content-copy: BibTeX](../../bibtex/2601.21913.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=recurrent-networks" data-tag="recurrent-networks">Recurrent networks</a> <a class="md-tag" href="/explore/?t=station-point" data-tag="station-point">Station / point</a>
    { .paper-tags }

-   #### Estimation of temperature and precipitation uncertainties using quantile neural networks { #2601.17243 }

    *Andrew Brettin, Laure Zanna* · Jan 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2601.17243">Extreme events pose significant risks and are challenging to predict. Assessing climate hazards requires placing quantitative constraints on geophysical fields under observable but fluctuating...</span><span class="abstract-full" id="full-2601.17243" hidden>Extreme events pose significant risks and are challenging to predict. Assessing climate hazards requires placing quantitative constraints on geophysical fields under observable but fluctuating conditions. We propose a framework for estimating uncertainties -- a ReLU-bias loss quantile neural network (RBLQNN) -- with two novel modifications to the loss function to enforce uniform quantile accuracy and reduce degenerate predicted probability distributions. We evaluate the RBLQNN against other probabilistic baselines on a suite of datasets: synthetic datasets, observed daily temperature maxima from 1,501 NOAA Global Surface Summary of the Day (GSOD) weather stations, and altimetry-observed precipitation from the Tropical Rainfall Measuring Mission (TRMM). On synthetic datasets, the RBLQNN accurately predicts conditional distributions where more restrictive methods like linear quantile regression (LQR) or mean-variance estimation (MVE) neural networks fail, mitigates shortcomings of some other quantile neural networks, and converges stably under a range of hyperparameters. When applied to daily temperature maxima, the RBLQNN reveals that temperature distributions are relatively well described by Gaussian statistics, though nonlinear dependencies on local sea level pressure and geopotential heights appear important. For precipitation statistics, the RBLQNN strongly outperforms both LQR and MVE baselines, demonstrating its capacity to capture highly nonlinear and non-Gaussian conditional distributions. The RBLQNN's performance across varied datasets demonstrates it is a flexible and general approach for constraining uncertainties in geophysical quantities with nonlinear or non-Gaussian conditional dependencies.</span> <span class="abstract-toggle" data-id="2601.17243">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2601.17243v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2601.17243v1) · [:material-content-copy: BibTeX](../../bibtex/2601.17243.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=daily" data-tag="daily">Daily</a>
    { .paper-tags }

-   #### Extending SST Anomaly Forecasts Through Simultaneous Decomposition of Seasonal and PDO Modes { #2601.01864 }

    *Rameshan Kallummal* · Jan 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2601.01864">We present a new approach to forecasting North Pacific Sea Surface Temperatures (SST) by recognizing that interannual variability primarily reflects amplitude changes in four dominant seasonal...</span><span class="abstract-full" id="full-2601.01864" hidden>We present a new approach to forecasting North Pacific Sea Surface Temperatures (SST) by recognizing that interannual variability primarily reflects amplitude changes in four dominant seasonal cycles. Our multivariate linear model simultaneously captures these amplitude-modulated seasonal cycles along with the Pacific Decadal Oscillation (PDO), which naturally emerges as an intrinsic feature of the system rather than a separate phenomenon. Using sixteen-dimensional regression based on four spatially distributed time series per variable, the model delivers unprecedented forecast accuracy for both interannual amplitude modulations and PDO evolution, maintaining skill beyond 36 months -- a substantial improvement over current operational and research forecasts, including machine learning methods. Predictions initialized in 2024 project that the PDO will remain in its negative phase through late 2026, implying reduced likelihood of severe marine heatwaves in the eastern North Pacific during this period. These findings have direct implications for regional climate impacts, including storm tracks, precipitation patterns, and marine ecosystem health. By treating seasonal and interannual variability as coupled rather than independent processes, this framework advances our understanding of North Pacific climate dynamics and provides a powerful tool for stakeholders managing climate-sensitive resources and planning adaptation strategies in regions strongly influenced by North Pacific conditions.</span> <span class="abstract-toggle" data-id="2601.01864">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2601.01864v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2601.01864v1) · [:material-content-copy: BibTeX](../../bibtex/2601.01864.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a>
    { .paper-tags }

-   #### Neural ocean forecasting from sparse satellite-derived observations: a case-study for SSH dynamics and altimetry data { #2512.22152 }

    *Daria Botvynko, Pierre Haslée, Lucile Gaultier, Bertrand Chapron, Clement de Boyer Montégut et al.* · Dec 2025
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2512.22152">We present an end-to-end deep learning framework for short-term forecasting of global sea surface dynamics based on sparse satellite altimetry data. Building on two state-of-the-art architectures:...</span><span class="abstract-full" id="full-2512.22152" hidden>We present an end-to-end deep learning framework for short-term forecasting of global sea surface dynamics based on sparse satellite altimetry data. Building on two state-of-the-art architectures: U-Net and 4DVarNet, originally developed for image segmentation and spatiotemporal interpolation respectively, we adapt the models to forecast the sea level anomaly and sea surface currents over a 7-day horizon using sequences of sparse nadir altimeters observations. The model is trained on data from the GLORYS12 operational ocean reanalysis, with synthetic nadir sampling patterns applied to simulate realistic observational coverage. The forecasting task is formulated as a sequence-to-sequence mapping, with the input comprising partial sea level anomaly (SLA) snapshots and the target being the corresponding future full-field SLA maps. We evaluate model performance using (i) normalized root mean squared error (nRMSE), (ii) averaged effective resolution, (iii) percentage of correctly predicted velocities magnitudes and angles, and benchmark results against the operational Mercator Ocean forecast product. Results show that end-to-end neural forecasts outperform the baseline across all lead times, with particularly notable improvements in high variability regions. Our framework is developed within the OceanBench benchmarking initiative, promoting reproducibility and standardized evaluation in ocean machine learning. These results demonstrate the feasibility and potential of end-to-end neural forecasting models for operational oceanography, even in data-sparse conditions.</span> <span class="abstract-toggle" data-id="2512.22152">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2512.22152v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2512.22152v1) · [:material-content-copy: BibTeX](../../bibtex/2512.22152.bib){ .bibtex-link }
    { .paper-links }

-   #### Lazy Diffusion: Mitigating spectral collapse in generative diffusion-based stable autoregressive emulation of turbulent flows { #2512.09572 }

    *Anish Sambamurthy, Ashesh Chattopadhyay* · Dec 2025
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2512.09572">Turbulent flows posses broadband, power-law spectra in which multiscale interactions couple high-wavenumber fluctuations to large-scale dynamics. Although diffusion-based generative models offer a...</span><span class="abstract-full" id="full-2512.09572" hidden>Turbulent flows posses broadband, power-law spectra in which multiscale interactions couple high-wavenumber fluctuations to large-scale dynamics. Although diffusion-based generative models offer a principled probabilistic forecasting framework, we show that standard DDPMs induce a fundamental <em>spectral collapse</em>: a Fourier-space analysis of the forward SDE reveals a closed-form, mode-wise signal-to-noise ratio (SNR) that decays monotonically in wavenumber, $|k|$ for spectra $S(k)\!\propto\!|k|^{-λ}$, rendering high-wavenumber modes indistinguishable from noise and producing an intrinsic spectral bias. We reinterpret the noise schedule as a spectral regularizer and introduce power-law schedules $β(τ)\!\propto\!τ^γ$ that preserve fine-scale structure deeper into diffusion time, along with <em>Lazy Diffusion</em>, a one-step distillation method that leverages the learned score geometry to bypass long reverse-time trajectories and prevent high-$k$ degradation. Applied to high-Reynolds-number 2D Kolmogorov turbulence and $1/12^\circ$ Gulf of Mexico ocean reanalysis, these methods resolve spectral collapse, stabilize long-horizon autoregression, and restore physically realistic inertial-range scaling. Together, they show that naïve Gaussian scheduling is structurally incompatible with power-law physics and that physics-aware diffusion processes can yield accurate, efficient, and fully probabilistic surrogates for multiscale dynamical systems.</span> <span class="abstract-toggle" data-id="2512.09572">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2512.09572v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2512.09572v1) · [:material-content-copy: BibTeX](../../bibtex/2512.09572.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### CLIMATEAGENT: Multi-Agent Orchestration for Complex Climate Data Science Workflows { #2511.20109 }

    *Hyeonjae Kim, Chenyue Li, Wen Deng, Mengxi Jin, Wen Huang, Mengqian Lu, Binhang Yuan* · Nov 2025
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2511.20109">Climate science demands automated workflows to transform comprehensive questions into data-driven statements across massive, heterogeneous datasets. However, generic LLM agents and static scripting...</span><span class="abstract-full" id="full-2511.20109" hidden>Climate science demands automated workflows to transform comprehensive questions into data-driven statements across massive, heterogeneous datasets. However, generic LLM agents and static scripting pipelines lack climate-specific context and flexibility, thus, perform poorly in practice. We present ClimateAgent, an autonomous multi-agent framework that orchestrates end-to-end climate data analytic workflows. ClimateAgent decomposes user questions into executable sub-tasks coordinated by an Orchestrate-Agent and a Plan-Agent; acquires data via specialized Data-Agents that dynamically introspect APIs to synthesize robust download scripts; and completes analysis and reporting with a Coding-Agent that generates Python code, visualizations, and a final report with a built-in self-correction loop. To enable systematic evaluation, we introduce Climate-Agent-Bench-85, a benchmark of 85 real-world tasks spanning atmospheric rivers, drought, extreme precipitation, heat waves, sea surface temperature, and tropical cyclones. On Climate-Agent-Bench-85, ClimateAgent achieves 100% task completion and a report quality score of 8.32, outperforming GitHub-Copilot (6.27) and a GPT-5 baseline (3.26). These results demonstrate that our multi-agent orchestration with dynamic API awareness and self-correcting execution substantially advances reliable, end-to-end automation for climate science analytic tasks.</span> <span class="abstract-toggle" data-id="2511.20109">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2511.20109v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2511.20109v1) · [:material-content-copy: BibTeX](../../bibtex/2511.20109.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=llms-agents" data-tag="llms-agents">LLMs & agents</a> <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a> <a class="md-tag" href="/explore/?t=extremes" data-tag="extremes">Extremes</a>
    { .paper-tags }

</div>

<nav class="pager" markdown="span">**1** [2](2.md) [3](3.md) [Older :material-arrow-right:](2.md){ .pager-step }</nav>

