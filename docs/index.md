---
hide:
  - navigation
  - toc
title: weatherml
---

A collection of papers on AI for weather forecasting, climate modelling and atmospheric science.

<p class="page-meta" markdown="span">1595 papers · updated 2026-10-28 · <a href="feed.xml">:material-rss: RSS</a> · <a href="all_papers.bib" download>:material-download: BibTeX</a> · <a href="https://github.com/weatherml/weatherml.github.io/issues/new?template=suggest-paper.yml">:material-plus: Suggest a paper</a></p>

## Browse by Topic

<div class="grid cards topics" markdown>

-   [Global Models](papers/global-models/index.md) <span class="topic-count">380</span>
-   [Regional Models](papers/regional-models/index.md) <span class="topic-count">67</span>
-   [Nowcasting](papers/nowcasting/index.md) <span class="topic-count">106</span>
-   [Downscaling](papers/downscaling/index.md) <span class="topic-count">107</span>
-   [Post-processing](papers/post-processing/index.md) <span class="topic-count">26</span>
-   [Data Assimilation](papers/data-assimilation/index.md) <span class="topic-count">80</span>
-   [Climate Modeling](papers/climate-modeling/index.md) <span class="topic-count">306</span>
-   [Hydrology](papers/hydrology/index.md) <span class="topic-count">39</span>
-   [Ocean & Sea Ice](papers/ocean-sea-ice/index.md) <span class="topic-count">89</span>
-   [Air Quality & Composition](papers/air-quality-composition/index.md) <span class="topic-count">78</span>
-   [Remote Sensing](papers/remote-sensing/index.md) <span class="topic-count">101</span>
-   [Other](papers/other/index.md) <span class="topic-count">216</span>

</div>

## Recent Additions

<div class="grid cards" markdown data-search-exclude>

-   #### KPPax: preprocess differentiable kinetic ODE for machine learning in atmospheric chemistry { #2610.37655 }

    *Wenqing Peng, Michael Boy, Zhi-Song Liu* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.37655">Atmospheric chemical mechanisms are commonly distributed in the language of the Kinetic PreProcessor (KPP), which generates efficient Fortran or C code but does not fit into the differentiable...</span><span class="abstract-full" id="full-2610.37655" hidden>Atmospheric chemical mechanisms are commonly distributed in the language of the Kinetic PreProcessor (KPP), which generates efficient Fortran or C code but does not fit into the differentiable programming frameworks in which machine-learning models are developed. We present KPPax, a kinetic preprocessor that generates differentiable, just-in-time compiled, and vectorisable chemistry code in JAX from mechanisms written in the KPP language. KPPax evaluates the rate law with sparse operations in log space, which keeps it differentiable, and integrates the stiff system with the implicit solvers and the forward- and reverse-mode adjoints of the <code>diffrax</code> library, so that exact sensitivities are obtained without hand-derived adjoint code. We demonstrate two applications with the POLLU benchmark mechanism. In the first, the Fisher information matrix computed from an automatically differentiated Jacobian predicts, before any fit, which of the 25 rate coefficients a flow-tube-like design can identify, and noisy synthetic estimation confirms the predicted uncertainties of the nine identifiable rate coefficients to within about 30% while revealing clusters of fast reactions of which only ratios are constrained. In the second, a neural-network correction of an emission inventory is trained offline on per-scenario 4D-Var retrievals and then fine-tuned online through the chemical model, which reduces the error of the biogenic correction by a factor of 2.4--5 and recovers information on low-temperature emissions that the individual retrievals lose. KPPax makes KPP mechanisms directly usable for parameter estimation and hybrid physics--machine-learning modelling.</span> <span class="abstract-toggle" data-id="2610.37655">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.37655v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.37655v1) · [:material-content-copy: BibTeX](bibtex/2610.37655.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=physics-ml-hybrid" data-tag="physics-ml-hybrid">Physics–ML hybrid</a>
    { .paper-tags }

-   #### An Input Skip Can Reparameterize a VAE Without Improving the Generator { #2610.37456 }

    *Kevin Lee* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.37456">An input skip can reduce reconstruction error while leaving a model's distribution of new samples unchanged. For a Gaussian variational autoencoder (VAE) with constant input skip $0\leqα<1$, a model...</span><span class="abstract-full" id="full-2610.37456" hidden>An input skip can reduce reconstruction error while leaving a model's distribution of new samples unchanged. For a Gaussian variational autoencoder (VAE) with constant input skip $0\leqα<1$, a model without a skip can be written in skip form by multiplying its decoder output by $b=1-α$ and its noise variance by $b^2$. The distribution of observations and latent variables and its exact generative scores stay the same, while reconstruction mean squared error (MSE) is multiplied by $b^2$. Separate training can produce different generators because optimization depends on the parameter coordinates. In a Gaussian experiment with known density, changing skip and observation variance together reduces MSE but worsens energy score and the estimate of negative log density from importance sampling. On seven annual Deep-ocean Assessment and Reporting of Tsunamis (DART) records, energy score differences at the same observation variance have both signs on four original records and favor the skip on three records from 2018. Probabilistic principal component analysis (PPCA) has lower mean energy scores than every neural DART fit. With the encoder and decoder kept the same, estimating observation variance from training reconstruction error can also worsen energy score. Assessing generation requires more than reconstruction error.</span> <span class="abstract-toggle" data-id="2610.37456">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.37456v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.37456v1) · [:fontawesome-brands-github: Code](https://github.com/magnaprog/vae-skip-reparam) · [:material-content-copy: BibTeX](bibtex/2610.37456.bib){ .bibtex-link }
    { .paper-links }

-   #### Scale-aware and spatially-localised optimisation of kilometre-scale MLWP models { #2610.37325 }

    *James Chappell, Sophie Luise Arana, Jack Bowyer, Alejandro Coca-Castro, Evangeline Corcoran et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.37325">Preserving fine-scale structure without degrading spatial accuracy is a central challenge for kilometre-scale machine-learning weather prediction (MLWP). We introduce a loss based on the dual-tree...</span><span class="abstract-full" id="full-2610.37325" hidden>Preserving fine-scale structure without degrading spatial accuracy is a central challenge for kilometre-scale machine-learning weather prediction (MLWP). We introduce a loss based on the dual-tree complex wavelet transform that constrains atmospheric fields jointly by scale, location, and orientation. We evaluate its performance using an experimental joint global--regional FastNet configuration, which produces 1.5 km forecasts over the UK alongside $\sim$31 km global forecasts. Relative to both mean-squared error (MSE) and an FFT-based spectral objective, the wavelet loss substantially improves spectral fidelity. Compared with MSE training, it maintains similar pointwise performance, improves neighbourhood-based verification, and produces sharper precipitation fields with more realistic intensity distributions. These results demonstrate that spatially localised wavelet constraints can preserve multiscale atmospheric structure while largely avoiding the loss of pointwise accuracy associated with globally supported spectral objectives.</span> <span class="abstract-toggle" data-id="2610.37325">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.37325v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.37325v1) · [:material-content-copy: BibTeX](bibtex/2610.37325.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a> <a class="md-tag" href="/explore/?t=km-scale" data-tag="km-scale">Km-scale</a>
    { .paper-tags }

-   #### Noise Steering: Exploiting Wake Steering and Acoustic Directivity for Noise-Constrained Wind Farm Control via Reinforcement Learning { #2610.36888 }

    *Martín de Frutos, Baptiste Corban, Claire Bizon Monroc, Ana Bušić, Donatien Dubuc, Jiamin Zhud et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.36888">Existing noise mitigation strategies for wind farms usually consist in reducing turbines power output, without accounting for the wake interactions between them and the noise emission directivity....</span><span class="abstract-full" id="full-2610.36888" hidden>Existing noise mitigation strategies for wind farms usually consist in reducing turbines power output, without accounting for the wake interactions between them and the noise emission directivity. This leads to over-conservative policies and unexploited wind power. Yet, turbine yaw can simultaneously affect wake interactions and the direction of noise emission, offering an additional degree of freedom for farm-level noise control. We therefore propose Noise Steering, a model-free reinforcement learning (RL) strategy that jointly controls turbine yaw angles and power setpoints to maximise farm power while satisfying a local noise constraint in a defined region. To enable training at a reduced computational cost, a low-fidelity acoustic model is coupled with the steady-state FLORIS simulator. The RL agent effectively learns how to redirect wakes and noise directivity away from the protected region while redistributing power across the farm. On a six-turbine test farm, the policy satisfies the 40 dB noise constraint throughout the full range of turbulent winds, with an average power deficit of only 2% relative to greedy unimpeded control, and up to 7.7% at high wind speeds. In comparison, the industry-standard noise-reduced operation (NRO) baseline here generates an average power loss of 17%. When deployed in the higher-fidelity FAST.Farm simulator, the policy largely retains its performance under steady and turbulent inflow, despite differences in simulation fidelity. These results demonstrate the potential of combining yaw control and power derating through RL to improve the power-noise trade-off in wind farm operation.</span> <span class="abstract-toggle" data-id="2610.36888">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.36888v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.36888v1) · [:material-content-copy: BibTeX](bibtex/2610.36888.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=reinforcement-learning" data-tag="reinforcement-learning">Reinforcement learning</a> <a class="md-tag" href="/explore/?t=energy" data-tag="energy">Energy</a>
    { .paper-tags }

-   #### FASST: A Compact Stateless Atmosphere Model for Long-Term Ocean Coupling { #2610.36297 }

    *Pratham Aggarwal, Duncan Watson-Parris, William E. Chapman* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.36297">Coupled climate simulations are computationally expensive, largely because of the atmosphere. We introduce FASST, a coupled climate model that retains the prognostic POP2 ocean while replacing the...</span><span class="abstract-full" id="full-2610.36297" hidden>Coupled climate simulations are computationally expensive, largely because of the atmosphere. We introduce FASST, a coupled climate model that retains the prognostic POP2 ocean while replacing the atmosphere and sea ice with lightweight U-Net emulators. The atmospheric emulator is diagnostic, predicting surface fluxes from the evolving ocean-ice state every $6$ simulated hours, while the sea-ice emulator advances the ice state monthly. For $140$ years of unforced integration, FASST maintains a stable global-mean climate without SST restoring, with post-transient global-mean SST of $291.30 \pm 0.22\mathrm{K}$ compared with $291.25 \pm 0.10\mathrm{K}$ in CESM$2$ piControl. FASST also captures interannual Niño-$3.4$ variability, although with weaker amplitude than CESM ($0.62\mathrm{K}$ versus $1.02\mathrm{K}$). These results demonstrate that a stateless learned atmosphere can sustain stable, free-running coupled climate simulations over century-scale integrations.</span> <span class="abstract-toggle" data-id="2610.36297">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.36297v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.36297v1) · [:material-content-copy: BibTeX](bibtex/2610.36297.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a> <a class="md-tag" href="/explore/?t=efficiency" data-tag="efficiency">Efficiency</a> <a class="md-tag" href="/explore/?t=monthly" data-tag="monthly">Monthly</a>
    { .paper-tags }

-   #### Towards operational data-driven regional weather forecasting: extreme-event skill and near-real-time verification with HRRRCast { #2610.36186 }

    *Daniel Abdi, Isidora Jankov, Vanderlei Vargas, Paul Madden, Emily Carpenter, Travis Wilson et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.36186">Improving convective-scale skill in data-driven regional weather models requires better representing rare high-impact events, training at native resolution, broadening variable output, and...</span><span class="abstract-full" id="full-2610.36186" hidden>Improving convective-scale skill in data-driven regional weather models requires better representing rare high-impact events, training at native resolution, broadening variable output, and demonstrating performance in real-time settings. Here, we evaluate these challenges using HRRRCast, a data-driven convection-allowing emulator trained on the High-Resolution Rapid Refresh (HRRR) analysis data. HRRRCast combines native 3-km training, moderate sigmoid-weighted loss reweighting for reflectivity and precipitation, and broader system improvements including expansion of the training archive from 3 to 9 years (2016--2025), increased vertical resolution, expanded outputs, and architectural upgrades. The weighted loss improves high-threshold skill while preserving common-event behavior. Native-resolution training avoids aliasing from 6-km subsampling and substantially improves fractions skill score at convective scales, with gains that persist across pooling windows. HRRRCast has also been running in near-real-time mode for the past year. In seven months of independent observation-based verification, HRRRCast ensemble means improve several near-surface metrics relative to operational HRRR and maintain strong precipitation and reflectivity performance. Single-member reflectivity skill is strongest at 20--30 dBZ; at 40 dBZ, skill is near parity at 6-km pooling and better at pooling radii $\geq$30 km. Precipitation skill remains competitive through 1 in/h and extends to about 3 in/h in the best recent configuration. A focused comparison with HREF further shows useful low-cost probabilistic guidance for 3-h precipitation. At storm scale, HRRRCast exhibits a displacement--structure trade-off, with generally better storm placement than storm morphology.</span> <span class="abstract-toggle" data-id="2610.36186">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.36186v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.36186v1) · [:material-content-copy: BibTeX](bibtex/2610.36186.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=evaluation" data-tag="evaluation">Evaluation</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a>
    { .paper-tags }

-   #### GNN-VIEW: Visual Inspection and Explanation of Graph-based Earth-System Models { #2610.35693 }

    *Alessandro Lucatelli, Ferdinand Unterhuber, Katharina Krenzer, Mennatallah El-Assady, Benedikt Soja et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.35693">Machine learning Earth-system models are becoming increasingly important, and so is their development. Graph-based models, one of the most prominent approaches, allow developers to constrain the...</span><span class="abstract-full" id="full-2610.35693" hidden>Machine learning Earth-system models are becoming increasingly important, and so is their development. Graph-based models, one of the most prominent approaches, allow developers to constrain the learning process before training via graph connectivity. Yet, developing these models remains challenging: the graph is built algorithmically with little intuition about the final result beyond static plotting, which requires extra work and is often not detailed enough. Even after successful training, how the model arrives at forecasts remains obscure. We present GNN-VIEW, an interactive browser-based tool that supports the development of such systems. GNN-VIEW exposes the graph's information pathways, allowing model developers to debug the graph structure before training. After training, GNN-VIEW uses attribution-based explainability to reveal which input features are used by the trained model for a given forecast. This helps developers understand why a model fails. We showcase the usefulness of GNN-VIEW in the construction stage of a graph for an ocean forecasting model, and in explaining poorly predicted precipitation of a trained weather forecasting model during Hurricane Laura.</span> <span class="abstract-toggle" data-id="2610.35693">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.35693v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.35693v1) · [:material-content-copy: BibTeX](bibtex/2610.35693.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=graph-neural-networks" data-tag="graph-neural-networks">Graph neural networks</a> <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a>
    { .paper-tags }

-   #### Application of Artificial Intelligence in the Management of Climate Change-Related Health Issues in Africa { #2610.35294 }

    *Peter Kokol, Martha Kachweka, Helena Blazun Vosner, Samuel Chol Buol, Jernej Zavrsnik et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.35294">While climate change disproportionately impacts African public health, AI predictive modeling remains heavily constrained. Using a Synthetic Knowledge Synthesis approach and VOSviewer, we conducted a...</span><span class="abstract-full" id="full-2610.35294" hidden>While climate change disproportionately impacts African public health, AI predictive modeling remains heavily constrained. Using a Synthetic Knowledge Synthesis approach and VOSviewer, we conducted a bibliometric analysis of Scopus-indexed literature, comparing Global and African corpora. Network analysis reveals stark divergences: global research emphasizes complex, predictive models for planetary health, whereas African research focuses on localized, reactive solutions for immediate infectious disease triage. Severe structural barriers in Africa, including compute deficits, data silos, and energy instability, hinder advanced AI scaling. Ultimately, international funding must shift from siloed pilot projects toward building foundational digital infrastructure and localized AI frameworks.</span> <span class="abstract-toggle" data-id="2610.35294">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.35294v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.35294v1) · [:material-content-copy: BibTeX](bibtex/2610.35294.bib){ .bibtex-link }
    { .paper-links }

-   #### Machine Learning Applications in the Solar Interior and Atmosphere { #2610.34687 }

    *Vishal Upendran, Carlos J. Díaz Baso* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.34687">The solar interior and atmosphere constitute a strongly coupled, multiscale dynamical system defined by continuous temporal evolution. Predicting the state of this system requires modeling physical...</span><span class="abstract-full" id="full-2610.34687" hidden>The solar interior and atmosphere constitute a strongly coupled, multiscale dynamical system defined by continuous temporal evolution. Predicting the state of this system requires modeling physical interactions that span disparate spatial and temporal scales, constrained by heterogeneous and sparse data. This chapter formalizes the Sun within the mathematical framework of dynamical systems theory, defining a global state vector ($\vec{S}$) governed by thermodynamical, kinetic, and magnetic properties. Because solar observations represent projected, multi-modal, and partially observable measurements ($\vec{O}[\vec{S}]$) of this hidden state, pure physics-based deduction is fundamentally constrained. We present a unifying thesis: the primary role of machine learning in heliophysics is to act as a structured family of mathematical operators---specifically Inference ($\mathcal{I}$), Emulation ($\mathcal{E}$), Propagation ($\mathcal{P}$), and Control ($\mathcal{C}$)---that reconstruct, simulate, and couple this global state vector across disparate physical layers. By framing the vast, multi-disciplinary literature within this unified state-space taxonomy, we demonstrate how data-driven techniques bridge observational gaps to resolve hidden multiscale couplings and dramatically accelerate the simulation of non-linear processes, establishing a rigorous pathway to predict the global evolution of the Sun.</span> <span class="abstract-toggle" data-id="2610.34687">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.34687v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.34687v1) · [:material-content-copy: BibTeX](bibtex/2610.34687.bib){ .bibtex-link }
    { .paper-links }

-   #### An actively trained neural operator for near-explicit atmospheric chemistry { #2610.33771 }

    *Zhi-Song Liu, Wenqing Peng, Ammar Kheder, Risto Makkonen, Michael Boy* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.33771">Earth-system models simplify atmospheric chemistry into condensed mechanisms, because near-explicit mechanisms such as the Master Chemical Mechanism (MCM) are too costly to solve in every grid cell....</span><span class="abstract-full" id="full-2610.33771" hidden>Earth-system models simplify atmospheric chemistry into condensed mechanisms, because near-explicit mechanisms such as the Master Chemical Mechanism (MCM) are too costly to solve in every grid cell. Neural emulators could lift this limit, but most advance the chemistry in small time steps and cover only a few species or a reduced mechanism. Here we introduce FNO-MoE, a Fourier neural operator with a mixture of experts that predicts the full near-explicit chemical state over 30 minutes in a single call. The same network also guides its own training, using the disagreement among its experts to select the most informative box-model simulations. At equal cost, this active sampling yields a more accurate emulator than random or evenly spread sampling. Fine-tuning on longer trajectories keeps it stable over multi-hour runs in which models trained on short simulations diverge. The resulting emulator reproduces the evolution of thousands of species with high fidelity, remains accurate for unseen chemical compositions and runs orders of magnitude faster than the numerical solver. These results open a new direction for emulating complex atmospheric systems: neural operators that predict the full chemical state over long windows and choose their own training data. Training under changing meteorology is the next step towards bringing near-explicit chemistry into Earth-system models.</span> <span class="abstract-toggle" data-id="2610.33771">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.33771v2) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.33771v2) · [:material-content-copy: BibTeX](bibtex/2610.33771.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=neural-operators" data-tag="neural-operators">Neural operators</a> <a class="md-tag" href="/explore/?t=sub-hourly" data-tag="sub-hourly">Sub-hourly</a>
    { .paper-tags }

-   #### Nowcasting aerosol particle size distributions from PM2.5 with physics-regularized flow matching { #2610.33557 }

    *Zhi-Song Liu, Jenni Köykkä, Wenqing Peng, Ammar Kheder, Petri Clusius, Michael Boy* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.33557">The mass concentration of particles smaller than 2.5um (PM2.5) is cheap to measure and widely monitored, whereas the particle number size distribution (PSD) requires costly instruments and is...</span><span class="abstract-full" id="full-2610.33557" hidden>The mass concentration of particles smaller than 2.5um (PM2.5) is cheap to measure and widely monitored, whereas the particle number size distribution (PSD) requires costly instruments and is measured at only a few research stations. Yet many aerosol effects, such as vapor uptake, cloud droplet formation and deposition in the human respiratory tract, depend on the PSD rather than on mass. We explore whether recent generative deep learning models can nowcast the PSD: from the PM2.5 record and eight meteorological parameters over a 24h window, a model generates an ensemble of plausible hourly size distributions for the same window, reflecting that one mass can arise from many size distributions. Rather than proposing a single best model, we use two contrasting Finnish stations, the boreal forest site Hyytiälä and the urban site Kumpula, as a controlled testbed. On it, we benchmark six generative models against deterministic and climatological baselines, locally and after transfer between sites, and test physical constraints for neural network training. Flow matching performs best overall; its guided variant has the lowest continuous ranked probability score (CRPS) at both sites. A constraint based on a reduced aerosol general dynamic equation (GDE) lowers the CRPS by 5.5-11.5% at the training site and reduces errors after transfer between sites. Transfer remains harder than local estimation, identifying training on large multi-station data sets as the next step towards PSD estimates at unmonitored locations. This benchmark establishes physics-informed generative models as a promising route from routine air-quality monitoring to size-resolved aerosol information.</span> <span class="abstract-toggle" data-id="2610.33557">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.33557v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.33557v1) · [:material-content-copy: BibTeX](bibtex/2610.33557.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=physics-ml-hybrid" data-tag="physics-ml-hybrid">Physics–ML hybrid</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=hourly" data-tag="hourly">Hourly</a>
    { .paper-tags }

-   #### Climate downscaling with likelihood functional flow matching { #2610.33482 }

    *Zhi-Song Liu, Helmi Toropainen, Ammar Kheder, Michael Boy* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.33482">Decisions about heat, frost and flooding are made locally, yet reanalyses and climate models resolve tens of kilometres. Generative downscalers add kilometre-scale detail with uncertainty, but are...</span><span class="abstract-full" id="full-2610.33482" hidden>Decisions about heat, frost and flooding are made locally, yet reanalyses and climate models resolve tens of kilometres. Generative downscalers add kilometre-scale detail with uncertainty, but are tied to a grid, costly to sample and lack a tractable likelihood. Here we introduce CD-FFM, a functional flow-matching model that treats the high-resolution field as a continuous random field that one model can query at any location and resolution. Its velocity acts pointwise, so spatial variability comes from a noise prior fitted to the observed spectrum and every observation has an exact likelihood. Downscaling ERA5 to 1 km daily temperature over Fennoscandia, CD-FFM lowers the continuous ranked probability score by 27-31% against regression and 12-18% against a forty-times larger neural operator in each of three test years, and outperforms a CorrDiff-style diffusion model while sampling about fifty times faster. The same model transfers to precipitation and makes large kilometre-scale ensembles affordable.</span> <span class="abstract-toggle" data-id="2610.33482">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.33482v2) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.33482v2) · [:material-content-copy: BibTeX](bibtex/2610.33482.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=extremes" data-tag="extremes">Extremes</a> <a class="md-tag" href="/explore/?t=km-scale" data-tag="km-scale">Km-scale</a> <a class="md-tag" href="/explore/?t=daily" data-tag="daily">Daily</a>
    { .paper-tags }

-   #### Data Quality Audit and Bivariate Temperature-Humidity Analysis of a Long-Term LoRaWAN Network: Anomaly Detection, PCA, Clustering and Machine Learning { #2610.32438 }

    *Mamadou Sow* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.32438">Long-term environmental IoT/LPWAN deployments often aggregate heterogeneous data streams from diverse devices without systematic auditing of each variable's actual coverage prior to analysis-an...</span><span class="abstract-full" id="full-2610.32438" hidden>Long-term environmental IoT/LPWAN deployments often aggregate heterogeneous data streams from diverse devices without systematic auditing of each variable's actual coverage prior to analysis-an increased risk when multiple sensor types coexist within the same network. This work proposes a C++17 processing pipeline applied to the Tour Perret LoRaWAN dataset (421 937 frames, June 2021--June 2023, 5 devices). A systematic coverage audit integrated into the extraction phase reveals that CO2 is completely missing (0% coverage) and pressure is reported only by an isolated base module (Wyres, 4.24% of the corpus), invalidating the initial hypothesis of a joint four-variable analysis. The analysis is refocused on the Temperature--Humidity pair, jointly available across 205{,}005 frames (95.76%) from 4 Elsys EMS sensors. The pipeline sequentially performs anomaly detection (modified Z-score, Isolation Forest), bivariate Principal Component Analysis, clustering (K-means, DBSCAN), and a multilayer perceptron predicting humidity from temperature and time, whose residuals serve as a detector for slow drift. On real-world data: Temperature--Humidity correlation r=-0.662, two-component PCA (PC1 =83.1% of variance), Isolation Forest detects 2.12% multivariate anomalies while univariate Z-score detects none, calibrated DBSCAN isolates 0.17% noise across 13 clusters, and the MLP achieves $R^2=0.672$ (RMSE =10.54%) in cross-validation. This work demonstrates that per-variable coverage auditing prior to modeling is a necessary methodological step for heterogeneous multi-sensor IoT datasets.</span> <span class="abstract-toggle" data-id="2610.32438">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.32438v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.32438v1) · [:material-content-copy: BibTeX](bibtex/2610.32438.bib){ .bibtex-link }
    { .paper-links }

-   #### Covariance-Informed Ensemble Score Filtering: Embedding Localized Covariance Geometry into Score-Based Transport { #2610.32148 }

    *Siming Liang, Peter Jan van Leeuwen, Hristo G. Chipilski, Zezhong Zhang, Feng Bao, Guannan Zhang* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.32148">Background-error covariance models are central to atmospheric data assimilation because they spread partial observational information across space, variables, and dynamically coupled scales. Ensemble...</span><span class="abstract-full" id="full-2610.32148" hidden>Background-error covariance models are central to atmospheric data assimilation because they spread partial observational information across space, variables, and dynamically coupled scales. Ensemble and variational methods exploit this covariance geometry efficiently, but their standard analysis formulations are largely built around Gaussian or locally quadratic approximations. Score-based ensemble filtering provides a framework for nonlinear distributional transport, but existing Ensemble Score Filter (EnSF) formulations lack an intrinsic covariance-mediated pathway for propagating likelihood information from observed to unobserved state components. We introduce a covariance-informed ensemble score filter (CI-EnSF) that embeds localized forecast-error covariance geometry directly into the forward diffusion and reverse transport. The proposed forward diffusion is covariance-shaped in standardized state space and becomes isotropic under the corresponding control-variable transformation. This covariance structure continuously propagates observational information from observed to unobserved state components throughout the reverse transport analysis trajectory. Experiments with Lorenz-96 and surface quasi-geostrophic (SQG) dynamics show a clear regime dependence. For linear observations, a well-tuned LETKF is more accurate, while CI-EnSF maintains stable cycling and effective covariance-mediated information propagation. Under strongly nonlinear observations and under climatological initialization, CI-EnSF achieves lower analysis error and greater cycling robustness than Gaussian ensemble benchmarks such as the LETKF, with improvements extending across spatial scales in the nonlinear SQG experiment. These results suggest that the principal benefit of CI-EnSF arises from combining nonlinear distributional transport with the mature covariance geometry used in conventional data assimilation.</span> <span class="abstract-toggle" data-id="2610.32148">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.32148v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.32148v1) · [:material-content-copy: BibTeX](bibtex/2610.32148.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

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

</div>

