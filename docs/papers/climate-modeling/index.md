---
title: 'Climate Modeling'
hide:
  - toc
---

<div class="listing-header" markdown>

# Climate Modeling

<p class="page-meta" markdown="span">315 papers · page 1 of 11 · <a href="../../bib/climate-modeling.bib" download>:material-download: BibTeX for this topic</a></p>

</div>

<div class="grid cards" markdown>

-   #### Reconstructing Boundary Layer Wind Profiles with Machine Learning: A Comparative Study on IGRA Radiosonde Data { #2611.13359 }

    *Fedor Buzaev, Dmitry Efremenko, Daria Pugacheva, Fedor Ratnikov, Egor Bugaev, Vladimir Gordin et al.* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.13359">Wind profiles in the atmospheric boundary layer (ABL) drive short-range forecasting and wind-energy siting, yet sparse radiosonde coverage forces practitioners to interpolate them from a handful of...</span><span class="abstract-full" id="full-2611.13359" hidden>Wind profiles in the atmospheric boundary layer (ABL) drive short-range forecasting and wind-energy siting, yet sparse radiosonde coverage forces practitioners to interpolate them from a handful of physical predictors. We benchmark modern tabular learners against analytical parameterizations on 158,097 IGRA soundings. CatBoost, TabM, and FT-Transformer reduce RMSE to 1.3-1.6 m/s, against approximately 3 m/s for power-law and Monin-Obukhov fits; once the predictor set is fixed the three architectures differ by less than 0.05 m/s, so accuracy is driven by predictors rather than architecture. A cohort analysis of within-bin profile dispersion places an empirical floor of approximately 1.77 m/s on any point-wise predictor that uses standard radiosonde features, consistent with prior eddy-viscosity inversions. A one-dimensional Fourier Neural Operator applied as a post-hoc corrector to frozen tabular predictions reduces near-surface RMSE on U.S. stations from 1.5 to 1.2 m/s, while leaving upper-layer error unchanged. Further gains should come from new predictors rather than larger models.</span> <span class="abstract-toggle" data-id="2611.13359">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.13359v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.13359v1) · [:material-content-copy: BibTeX](../../bibtex/2611.13359.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=neural-operators" data-tag="neural-operators">Neural operators</a>
    { .paper-tags }

-   #### MarsNet: Learning the Martian Atmosphere from Observations Alone { #2611.12787 }

    *Sebastian Mengwall, J. Michael Battalio* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.12787">We present MarsNet, a machine-learned model of the Martian atmosphere trained only on observations. Existing reanalyses assimilate spacecraft retrievals into general circulation models (GCMs), so...</span><span class="abstract-full" id="full-2611.12787" hidden>We present MarsNet, a machine-learned model of the Martian atmosphere trained only on observations. Existing reanalyses assimilate spacecraft retrievals into general circulation models (GCMs), so where observations are sparse their fields largely reflect the model. We therefore use extant reanalyses only for evaluation. MarsNet alternates a two-hour forecast with an update that assimilates the observations of each window, and both steps are learned jointly. Because the Martian record is too sparse for an unrestricted network, learning is confined within a fixed physical structure: the state is in physical units, the forecast uses physical tendencies of fixed form and learned strength, and the update is a bounded Kalman update. MarsNet is trained on Mars Years (MY) 24--31 and evaluated on independent observations. On three seasons of MGS radio-occultation temperatures assimilated by no dataset, MarsNet matches or improves on the three reanalyses in the lowest three scale heights, with a near-surface root-mean-square error (RMSE) of 4.2 K against 4.4--5.4 K, but is less accurate aloft. The model's surface pressures are the most accurate of the four products (RMSE 10 Pa against Curiosity in MY32, compared with 20--46 Pa). On withheld Mars Climate Sounder profiles of MY32, its temperatures beat an observed climatology and persistence, including in ten-sol forecasts without new observations, and its stated uncertainty is close to calibrated. Much of the temperature skill is present before training and appears to come from the structure and tables fitted to the observations, while training mainly improves the surface pressures, at some cost to dust. The model cycles stably through nine Mars years, and the influence of each observation can be traced. Its speed makes experiments on the observing record practical, and it offers an independent reference for GCMs and reanalyses.</span> <span class="abstract-toggle" data-id="2611.12787">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.12787v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.12787v1) · [:material-content-copy: BibTeX](../../bibtex/2611.12787.bib){ .bibtex-link }
    { .paper-links }

-   #### Perils and opportunities of AI for public climate services { #2611.12106 }

    *Francesca Larosa, Sandro Calmanti, Matteo De Felice, Marcello Petitta* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.12106">The rapid integration of artificial intelligence (AI), particularly generative and machine-learning systems, in climate services development and provision is transforming how climate information is...</span><span class="abstract-full" id="full-2611.12106" hidden>The rapid integration of artificial intelligence (AI), particularly generative and machine-learning systems, in climate services development and provision is transforming how climate information is produced, accessed, and interpreted. AI-enabled climate services offer significant opportunities for process automation, optimisation, tailoring, and the translation of complex climate data into actionable knowledge for diverse users. However, the growing reliance on privately controlled algorithms, data infrastructures, and computing facilities raises critical concerns related to governance, transparency, equity and economic sustainability, among others. While the technical architecture of climate services increasingly relies on advanced machine learning and large-scale climate models, their legitimacy as public services may be weakened in the absence of arrangements that prioritise public value and long-lasting societal benefit over profit maximisation. The tension between the public mandate of climate services and the private nature of AI infrastructure challenges traditional notions of openness. Using a consensus-based elicitation method coupled with the PESTLE framework, we analyse the political, economic, social, technological, legal, and environmental risks and opportunities shaping the co-production value chain of AI-enabled climate services. We draw policy-relevant conclusions for data, model and infrastructure supporting the next generation of upstream and downstream climate services as critical applications for society. Our work presents potential and plausible challenges and benefits for a pan-European, decentralised public climate service built on sovereign AI infrastructure and open governance principles.</span> <span class="abstract-toggle" data-id="2611.12106">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.12106v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.12106v1) · [:material-content-copy: BibTeX](../../bibtex/2611.12106.bib){ .bibtex-link }
    { .paper-links }

-   #### Rescene-ESM: an Earth system model built around a frozen AI weather model { #2611.11066 }

    *Minjong Cheon, Kaoru Kakinuma* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.11066">Earth system models underpin climate projections, but their cost limits how many scenarios, ensemble members and impact pathways can be explored. Machine-learning weather models forecast as...</span><span class="abstract-full" id="full-2611.11066" hidden>Earth system models underpin climate projections, but their cost limits how many scenarios, ensemble members and impact pathways can be explored. Machine-learning weather models forecast as skillfully as physics-based models at a fraction of the cost, yet coupled to an ocean they have produced weak El Nino variability or have learned it from climate-model output, and none carries the climate through rivers to the people exposed to it. Here we present Rescene-ESM, an Earth system model whose coupler, YAHO, yokes a frozen 20.8-million-parameter AI weather model to physically formulated ocean, sea-ice, land, river, carbon and energy components constrained by observations. Adding the couplings one at a time shows what each contributes: over a climatological ocean the AI atmosphere varies from year to year by a third of the observed amount over the oceans, and an interactive mixed layer, an equatorial wave ocean, the residual of the observed sea surface temperature (SST) equation as stochastic forcing and the land's moisture flux raise this step by step and bring El Nino and its teleconnections, leaving the mean climate almost unchanged. The full model simulates about 1,000 years per day on one graphics processing unit, holds the observed climate for a thousand years without drift, reproduces the observed 1979-2022 global warming, carries forced change with a physically based structure around the frozen network (transient climate response 1.60 K), routes runoff to 212 river gauges with a median seasonal-cycle correlation of 0.79 and, driven by emissions, separates the heatwave exposure of an SSP1 and an SSP3 world by a factor of 4.7 in 2081-2100, three-quarters of it through the warmer climate.</span> <span class="abstract-toggle" data-id="2611.11066">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.11066v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.11066v1) · [:material-content-copy: BibTeX](../../bibtex/2611.11066.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=subseasonal-to-seasonal" data-tag="subseasonal-to-seasonal">Subseasonal to seasonal</a>
    { .paper-tags }

-   #### Guiding future global temperature distributions with simulations and observations { #2611.10882 }

    *Francesco Immorlano, Elijah Tavares, Felix Draxler, Juan Nathaniel, Violette Launeau et al.* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.10882">Assessing climate risks requires projections of daily temperature distributions, including rare and spatially widespread extremes. Climate simulations are costly and biased, while correcting them...</span><span class="abstract-full" id="full-2611.10882" hidden>Assessing climate risks requires projections of daily temperature distributions, including rare and spatially widespread extremes. Climate simulations are costly and biased, while correcting them toward observations risks suppressing inter-model differences. We train a single machine-learning emulator, a conditional diffusion model, on seventeen CMIP6 models and ERA5 reanalysis to generate spatially coherent daily temperature fields conditioned on model identity, date, and CO2-equivalent concentration. At sampling time, diffusion guidance reduces model biases by steering each emulated climate-model distribution toward observations, with its strength selected through perfect-model experiments using withheld simulated futures. Observational influence decreases with projection horizon and forcing. Across seventeen synthetic targets, guidance more than halves median biases in the 1st and 99th percentiles of daily temperature, improves probabilistic scores and reliability, and reduces median annual global-mean temperature RMSE by 32-57% across three emission scenarios. In a detailed benchmark, guidance also outperforms established bias corrections on daily-field probabilistic scores. Efficient sampling generates large ensembles without further climate simulations. For example, our framework corrects CMIP6's roughly threefold historical overestimate of extreme-heat coverage, yet projects that concurrent extremes will affect nearly half of global land on a typical day in the 2080s under SSP3-7.0.</span> <span class="abstract-toggle" data-id="2611.10882">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.10882v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.10882v1) · [:material-content-copy: BibTeX](../../bibtex/2611.10882.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=daily" data-tag="daily">Daily</a>
    { .paper-tags }

-   #### Evaluation of a Multi-Link Machine Learning Approach for Rainfall Detection Using Satellite Microwave Links { #2611.09156 }

    *Mostafa Ftouni, Margherita Spalla, Matteo Colli, Andrea Randazzo, Christian Gianoglio* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.09156">The impacts of climate change have increased the need for reliable real-time rainfall monitoring systems. This study investigates rainfall detection using a network of opportunistic sensors based on...</span><span class="abstract-full" id="full-2611.09156" hidden>The impacts of climate change have increased the need for reliable real-time rainfall monitoring systems. This study investigates rainfall detection using a network of opportunistic sensors based on satellite microwave links (SMLs) combined with machine learning (ML) classifiers. Two models, a random forest (RF) and a convolutional neural network (CNN), were trained on aggregated SML data, including information from neighboring links. The results demonstrate that leveraging information from multiple links substantially enhances the detection of both wet and dry periods compared to single-link approaches. Moreover, the RF model consistently provides a more stable and reliable trade-off than CNN, improving rainfall detection and reducing misclassifications overall, with particularly strong performance during harsh weather conditions. The study spanned up to 32 rainy days between 2021 and 2024.</span> <span class="abstract-toggle" data-id="2611.09156">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.09156v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.09156v1) · [:material-content-copy: BibTeX](../../bibtex/2611.09156.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a> <a class="md-tag" href="/explore/?t=classical-ml" data-tag="classical-ml">Classical ML</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a>
    { .paper-tags }

-   #### Evaluating the prediction of subgrid eddy kinetic energy from a machine learning model { #2611.06539 }

    *Thomas Wilder, Ranjini Swaminathan, David Storkey, Till Kuhlbrodt* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.06539">Global climate model resolution is still too coarse to fully resolve mesoscale processes, therefore subgrid closures are required to approximate the feedback of eddies onto the resolved flow. With...</span><span class="abstract-full" id="full-2611.06539" hidden>Global climate model resolution is still too coarse to fully resolve mesoscale processes, therefore subgrid closures are required to approximate the feedback of eddies onto the resolved flow. With high-resolution datasets now readily available, there has been a rapid growth in the use of machine learning to develop new eddy closures that could help reduce model biases and uncertainty. In this work we employ an offline learning approach with a convolutional neural network to predict subgrid eddy kinetic energy ($KE_e$), laying the initial groundwork for a machine-learned scale-aware GEOMETRIC eddy closure for the UK's Earth System Model. We address the question of whether global parameterizations are needed or if regional closures are sufficient by evaluating a number of neural network models through region and feature ablation experiments. ML parameterization models generalize to unseen regions when their training data share similar feature and target distributions with those regions. Models that fail to generalize incur distributional shift, or employ different filtering scales for the true eddy kinetic energy field. When holding out features, there is improvement among the generalizable ML models, with some feature spaces leading to improvements in some regions but degrading performance in others. Generally, power spectral density of $KE_e$ tend to diverge at small scales, though some feature spaces are able to minimize this divergence. Overall, the best performing model trained on all regions achieves time-mean and anomaly R2 scores of R2$\approx 0.95$ and R2$\approx 0.8 - 0.9$, respectively.</span> <span class="abstract-toggle" data-id="2611.06539">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.06539v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.06539v1) · [:material-content-copy: BibTeX](../../bibtex/2611.06539.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a> <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a>
    { .paper-tags }

-   #### Can LLM Agents Improve Imperfect Earth System Model Parameterizations? { #2611.03543 }

    *Kara D. Lamb* · Nov 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2611.03543">Earth system models (ESMs) rely on parameterizations to represent unresolved processes such as convection, cloud microphysics, and turbulence. However, translating advances in physical understanding...</span><span class="abstract-full" id="full-2611.03543" hidden>Earth system models (ESMs) rely on parameterizations to represent unresolved processes such as convection, cloud microphysics, and turbulence. However, translating advances in physical understanding into stable, validated parameterizations and integrating them into ESMs often takes years. Agentic AI could accelerate this process and enable more systematic exploration of design choices. Using warm-rain collision-coalescence as a test case, we evaluate agents' ability to synthesize scientific literature, reimplement published bulk parameterizations, and propose new functional forms as differentiable JAX code. We compare conventional parameterizations, data-driven emulators, and agent-generated closures against high-fidelity superdroplet simulations, repeating unattended experiments across ten language models. With expert guidance or iterative feedback, agents produced executable, differentiable, mass-conserving code, although unattended performance depended strongly on the scheme and language model. Iterative feedback lowered median rollout error for eight of ten models. Retuned physical schemes matched or outperformed many learned parameterizations at substantially lower computational cost, but two of 100 unattended agent-generated closures achieved slightly lower rain-mass error than the most accurate retuned scheme on held-out simulations.</span> <span class="abstract-toggle" data-id="2611.03543">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2611.03543v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2611.03543v1) · [:material-content-copy: BibTeX](../../bibtex/2611.03543.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=llms-agents" data-tag="llms-agents">LLMs & agents</a>
    { .paper-tags }

-   #### Multi-Agent Reinforcement Learning for Global Climate Policy Cooperation { #2610.37870 }

    *Anne Moll-Elsborg, Oskar Bohn Lassen, Francisco Camara Pereira* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.37870">International climate policy requires coordination among countries whose mitigation decisions impose local costs but generate global benefits. Classical game-theoretic and integrated assessment...</span><span class="abstract-full" id="full-2610.37870" hidden>International climate policy requires coordination among countries whose mitigation decisions impose local costs but generate global benefits. Classical game-theoretic and integrated assessment approaches can study these incentives, but typically rely on prescribed behavioural assumptions or equilibrium concepts. Recent work has therefore explored Multi-Agent Reinforcement Learning as a way to model climate-policy cooperation as an emergent outcome of repeated strategic interaction. We extend an existing MARL climate framework with a multi-stage negotiation mechanism in which agents propose agreements, accept or reject commitments, and face penalties for non-compliance. This allows us to study how agreement design affects learned mitigation behaviour, compliance, and climate outcomes. In an idealized setting with four homogeneous agents, we find that non-binding agreements do little to overcome free-riding, while moderate non-compliance penalties improve cooperation and reduce warming. However, excessively strong penalties discourage ambitious commitments, revealing a trade-off between enforcement and participation. We position the framework as a methodological contribution for testing agreement design rather than a predictive model of real negotiations.</span> <span class="abstract-toggle" data-id="2610.37870">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.37870v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.37870v1) · [:material-content-copy: BibTeX](../../bibtex/2610.37870.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=reinforcement-learning" data-tag="reinforcement-learning">Reinforcement learning</a> <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a>
    { .paper-tags }

-   #### FASST: A Compact Stateless Atmosphere Model for Long-Term Ocean Coupling { #2610.36297 }

    *Pratham Aggarwal, Duncan Watson-Parris, William E. Chapman* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.36297">Coupled climate simulations are computationally expensive, largely because of the atmosphere. We introduce FASST, a coupled climate model that retains the prognostic POP2 ocean while replacing the...</span><span class="abstract-full" id="full-2610.36297" hidden>Coupled climate simulations are computationally expensive, largely because of the atmosphere. We introduce FASST, a coupled climate model that retains the prognostic POP2 ocean while replacing the atmosphere and sea ice with lightweight U-Net emulators. The atmospheric emulator is diagnostic, predicting surface fluxes from the evolving ocean-ice state every $6$ simulated hours, while the sea-ice emulator advances the ice state monthly. For $140$ years of unforced integration, FASST maintains a stable global-mean climate without SST restoring, with post-transient global-mean SST of $291.30 \pm 0.22\mathrm{K}$ compared with $291.25 \pm 0.10\mathrm{K}$ in CESM$2$ piControl. FASST also captures interannual Niño-$3.4$ variability, although with weaker amplitude than CESM ($0.62\mathrm{K}$ versus $1.02\mathrm{K}$). These results demonstrate that a stateless learned atmosphere can sustain stable, free-running coupled climate simulations over century-scale integrations.</span> <span class="abstract-toggle" data-id="2610.36297">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.36297v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.36297v1) · [:material-content-copy: BibTeX](../../bibtex/2610.36297.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a> <a class="md-tag" href="/explore/?t=efficiency" data-tag="efficiency">Efficiency</a> <a class="md-tag" href="/explore/?t=monthly" data-tag="monthly">Monthly</a>
    { .paper-tags }

-   #### Application of Artificial Intelligence in the Management of Climate Change-Related Health Issues in Africa { #2610.35294 }

    *Peter Kokol, Martha Kachweka, Helena Blazun Vosner, Samuel Chol Buol, Jernej Zavrsnik et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.35294">While climate change disproportionately impacts African public health, AI predictive modeling remains heavily constrained. Using a Synthetic Knowledge Synthesis approach and VOSviewer, we conducted a...</span><span class="abstract-full" id="full-2610.35294" hidden>While climate change disproportionately impacts African public health, AI predictive modeling remains heavily constrained. Using a Synthetic Knowledge Synthesis approach and VOSviewer, we conducted a bibliometric analysis of Scopus-indexed literature, comparing Global and African corpora. Network analysis reveals stark divergences: global research emphasizes complex, predictive models for planetary health, whereas African research focuses on localized, reactive solutions for immediate infectious disease triage. Severe structural barriers in Africa, including compute deficits, data silos, and energy instability, hinder advanced AI scaling. Ultimately, international funding must shift from siloed pilot projects toward building foundational digital infrastructure and localized AI frameworks.</span> <span class="abstract-toggle" data-id="2610.35294">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.35294v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.35294v1) · [:material-content-copy: BibTeX](../../bibtex/2610.35294.bib){ .bibtex-link }
    { .paper-links }

-   #### Disentangling forced and internal components in observed climate evolution using spatiotemporal deep learning { #2610.28219 }

    *Parvathi Kooloth, Wenyu Zhou, L. Ruby Leung, Shang-Ping Xie, Hailong Wang, Qiang Fu* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.28219">Attribution of observed historical climate evolution to externally forced change and internal variability is fundamental to understanding climate change and evaluating climate models. Yet such...</span><span class="abstract-full" id="full-2610.28219" hidden>Attribution of observed historical climate evolution to externally forced change and internal variability is fundamental to understanding climate change and evaluating climate models. Yet such attribution remains challenging because the forced response is intertwined with internal variability in observations. Here, we develop a convolutional long short-term memory network (ConvLSTM) to disentangle these two components. Trained and evaluated using large-ensemble simulations from multiple climate models, ConvLSTM consistently outperforms existing model-based, statistical, and machine-learning approaches. Applying ConvLSTM to observations reveals an evolving spatial pattern of the forced response and substantial contributions from internal variability to recent multidecadal climate trends. For several key climate features whose attribution remains debated, ConvLSTM indicates that internal variability amplified the forced reversal of Atlantic interhemispheric temperature asymmetry, dominated the observed La Niña-like warming pattern over the equatorial Pacific and the associated precipitation decline over southwestern North America since 1980, and modulated Arctic amplification around its nearly constant threefold forced value. Comparison between the ConvLSTM-estimated and model-simulated forced responses reveals broadly consistent patterns but also important discrepancies, with climate models simulating stronger Northern Hemisphere warming, weaker subtropical high intensification, and smaller land precipitation declines. Overall, ConvLSTM provides a robust framework for attributing historical climate evolution and evaluating the fidelity of climate models</span> <span class="abstract-toggle" data-id="2610.28219">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.28219v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.28219v1) · [:material-content-copy: BibTeX](../../bibtex/2610.28219.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=recurrent-networks" data-tag="recurrent-networks">Recurrent networks</a>
    { .paper-tags }

-   #### Energy-Aware Earth System Science: A Community Roadmap { #2610.23216 }

    *Ioana Colfescu, Annalisa Bracco, Kara Moraw, Chen Lu, Michael Bareford, Tom Beucler et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.23216">Earth system models underpin the evidence that informs net-zero policy, yet their energy demands are rarely considered when judging model quality. Using archived output from 159 models spanning five...</span><span class="abstract-full" id="full-2610.23216" hidden>Earth system models underpin the evidence that informs net-zero policy, yet their energy demands are rarely considered when judging model quality. Using archived output from 159 models spanning five generations of the Coupled Model Intercomparison Project (CMIP), we show that model error has fallen at a constant rate of 13.5 per cent per doubling of compute since 1995. Halving CMIP6 error would require 28 times more compute. Improvements are uneven, with sea-level pressure and surface air temperature errors declining 3.5-fold and 2.7-fold since CMIP1, but precipitation error by only 7 per cent. Machine-learning weather models match or exceed operational forecast skill at a fraction of the runtime, but none publishes measured energy, and including training changes how they rank. We propose a skill-energy framework that combines an empirical Pareto front with a ranking score. These findings argue for standardised reporting of energy use and performance ahead of CMIP8.</span> <span class="abstract-toggle" data-id="2610.23216">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.23216v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.23216v1) · [:material-content-copy: BibTeX](../../bibtex/2610.23216.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=evaluation" data-tag="evaluation">Evaluation</a> <a class="md-tag" href="/explore/?t=energy" data-tag="energy">Energy</a>
    { .paper-tags }

-   #### Improved Precipitation Variability in the Community Atmosphere Model through Implementation of a Neural-Network Deep Convection Scheme { #2610.20405 }

    *Xavier J. Levine, Griffin Mooers, Jack W. Atkinson, Marat F. Khairoutdinov, Janni Yuval et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.20405">Conventional deep convection schemes remain a primary source of uncertainty in global climate models, yet replacing them with data-driven alternatives while maintaining physical consistency and...</span><span class="abstract-full" id="full-2610.20405" hidden>Conventional deep convection schemes remain a primary source of uncertainty in global climate models, yet replacing them with data-driven alternatives while maintaining physical consistency and numerical stability is an open challenge. We present a hybrid configuration of the Community Atmosphere Model version 6 (CAM6), in which a fully connected deep neural network replaces the Zhang-McFarlane deep convection scheme (ZM), while radiation, turbulence, and other cloud processes are left unchanged. The network is trained on output from a storm-resolving global model - the global System for Atmospheric Modeling (gSAM) - using a coarse-graining approach to derive subgrid-scale convective fluxes and tendencies. This data-driven parameterization is ported to CAM6 and found to run stably for more than a decade while conserving energy and moisture. It produces a broadly realistic climatology without retuning the remaining parameterizations. Relative to the default Zhang-McFarlane scheme, the neural network parameterization substantially improves some long-standing model deficiencies. In particular, the biases in light precipitation and extreme precipitation are markedly reduced, and the diurnal cycle of tropical land precipitation shifts from a spurious midday peak over many regions to the observed late-afternoon peak. These results support targeted ML-based convection schemes as a viable path toward reducing systematic model errors in next-generation climate models.</span> <span class="abstract-toggle" data-id="2610.20405">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.20405v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.20405v1) · [:material-content-copy: BibTeX](../../bibtex/2610.20405.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a> <a class="md-tag" href="/explore/?t=km-scale" data-tag="km-scale">Km-scale</a>
    { .paper-tags }

-   #### Unmasking Compensating Errors in Numerical Weather Models: A Process-Additive Neural Network Approach { #2610.19772 }

    *Yuiko Ichikawa, Edward Groot, Jack Atkinson* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.19772">This work introduces a Process Additive Neural Network (PANN) framework to disentangle compensating biases in atmospheric models, where errors in different physical processes can offset each other...</span><span class="abstract-full" id="full-2610.19772" hidden>This work introduces a Process Additive Neural Network (PANN) framework to disentangle compensating biases in atmospheric models, where errors in different physical processes can offset each other and obscure their origins. Using a single-column model, each PANN branch represents the tendency error associated with a separate physical process, and the summed branch outputs approximate the total error relative to a reference tendency. Once trained successfully, the branch outputs are expected to provide process-level error attributions and, when averaged, estimates of process-level bias. Initial experiments focus on cloud-related processes in OpenIFS SCM case studies, where cloud and convection tendencies are strongly coupled and compensation is difficult to diagnose using standard approaches. The aim is not to replace human interpretation of physical processes, but to extend it to regimes where model complexity, process interaction, and compensating errors make direct diagnosis difficult. By providing a physically structured form of machine-learning interpretability, PANN can help connect complex model behaviour with process-level explanations. This work supports both improved understanding of compensating errors and the development of more physically consistent weather and climate models.</span> <span class="abstract-toggle" data-id="2610.19772">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.19772v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.19772v1) · [:material-content-copy: BibTeX](../../bibtex/2610.19772.bib){ .bibtex-link }
    { .paper-links }

-   #### Identifying Arctic states vulnerable to rapid ice loss with a convolutional neural network { #2610.16584 }

    *Matthew Patterson, Ioana Colfescu, Nathan Kirk, Graeme MacGhilchrist* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.16584">This work investigates the incidence of rapid ice-loss events in the Arctic, in which a substantial fraction of September sea-ice is lost in only a few years. Such events are seen to occur in climate...</span><span class="abstract-full" id="full-2610.16584" hidden>This work investigates the incidence of rapid ice-loss events in the Arctic, in which a substantial fraction of September sea-ice is lost in only a few years. Such events are seen to occur in climate model simulations forced with increasing greenhouse gas concentrations, but not as yet in the relatively short observational record. We train a convolutional neural network (CNN) using six large climate model ensembles to diagnose vulnerability to rapid ice-loss events. The trained model is shown to separate higher and lower risk states in out-of-sample climate models via a leave-one-out cross-validation. The model emphasises the role of persistently thin sea-ice, remaining sea-ice extent and sustained high upper ocean heat content as predictors of rapid ice-loss. The CNN is applied to observational data over the satelite-era showing a signal for rapid ice-loss prior to a drop around 2007 and an increasing risk from around 2015 to present. Such rapid-ice loss events would likely have dramatic consequences for Arctic ecosystems, coastal communities and global shipping.</span> <span class="abstract-toggle" data-id="2610.16584">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.16584v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.16584v1) · [:material-content-copy: BibTeX](../../bibtex/2610.16584.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a>
    { .paper-tags }

-   #### SciExam for ENSO: Can AI Agents Build Climate Models? { #2610.10513 }

    *Yinling Zhang, Langchen Liu, Dongbin Xiu, Xueyan Zou, Xu Kuang, Mengdi Wang, Shilong Liu* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.10513">Language-model agents are increasingly asked to carry out open-ended scientific research, yet their results are usually graded against a known answer, a rubric, or a language-model reviewer, none of...</span><span class="abstract-full" id="full-2610.10513" hidden>Language-model agents are increasingly asked to carry out open-ended scientific research, yet their results are usually graded against a known answer, a rubric, or a language-model reviewer, none of which can tell whether a new scientific model is valid. The AI Science Exam for El Nino-Southern Oscillation (SciExam for ENSO) is a benchmark in which agents build low-order stochastic models of ENSO, the dominant mode of interannual climate variability, from real observations. Within a six-hour budget, agents process the observations, write their own diagnostics, which are then frozen, and develop a model using only these diagnostics as feedback. Hidden graders then test whether the model reproduces ENSO's statistics, recovers unobserved variables, and forecasts held-out years, and score a published model in the same way. Across twelve agent systems, six produce models that score higher than the published model, mainly through better reconstruction and forecasting. The simplified forms of the stronger models are each compatible with one of the two competing explanations of ENSO's warm-cold asymmetry, an open debate that the task never mentions. Controlled runs of the top system under varied information suggest that its scores do not come from recalling the dated observational record and that the information it receives shapes how it builds its model. SciExam for ENSO can thus evaluate agent research where no answer is known, and the results suggest that agents can already build competitive models whose structures bear on questions that scientists still debate.</span> <span class="abstract-toggle" data-id="2610.10513">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.10513v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.10513v1) · [:fontawesome-brands-github: Code](https://github.com/ylzhang2447/SciExam-ENSO-code) · [:material-content-copy: BibTeX](../../bibtex/2610.10513.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=subseasonal-to-seasonal" data-tag="subseasonal-to-seasonal">Subseasonal to seasonal</a>
    { .paper-tags }

-   #### Artificial intelligence pathways from weather to climate { #2610.09770 }

    *Tom Beucler, J. David Neelin, Hui Su, Shivanshi Asthana, Chris Bretherton, Will Chapman et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.09770">Deep learning has made rapid advances in weather forecasting: autoregressive models trained on atmospheric reanalyses now rival dynamical models across nowcasting, medium-range, and...</span><span class="abstract-full" id="full-2610.09770" hidden>Deep learning has made rapid advances in weather forecasting: autoregressive models trained on atmospheric reanalyses now rival dynamical models across nowcasting, medium-range, and subseasonal-to-seasonal lead times, producing well-calibrated ensemble forecasts at reduced cost. We review these advances and consider their extension to climate horizons, where the challenge shifts from initial-condition skill to producing reliable statistical responses under altered forcings. AI-powered climate prediction systems must produce credible forced responses to drivers (e.g., greenhouse gases, land-use change) typically outside the observed record. We propose two minimum requirements for AI in climate modeling: (i) external forcing agents must enter explicitly enough to support interventions in which they vary independently; and (ii) robustness must be stress-tested in out-of-distribution regimes, including extremes and counterfactual trajectories. Using leading AI autoregressive emulators and hybrid physics-AI models, we identify development and coupling challenges. Comparing the reported throughput of these models with that of GPU-ported dynamical models highlights how AI can reduce time-to-solution by advancing only the target variables at the required resolution and using longer time steps, rather than integrating a full high-frequency, multivariate state. Diverse AI downscaling strategies can partially substitute for explicit fine-scale resolution, paving the way toward inexpensive local hazard assessment across prediction horizons.</span> <span class="abstract-toggle" data-id="2610.09770">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.09770v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.09770v1) · [:material-content-copy: BibTeX](../../bibtex/2610.09770.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=physics-ml-hybrid" data-tag="physics-ml-hybrid">Physics–ML hybrid</a>
    { .paper-tags }

-   #### EC-EarthFlow: Probabilistic emulation of daily transient global climate model simulations with flow matching { #2610.09715 }

    *Kirien Whan, Nikolaj T. Mücke, Karin van der Wiel* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.09715">We introduce EC-EarthFlow, a generative flow matching model that emulates simulations from the physical climate model EC-Earth3. The model is trained on transient simulations from EC-Earth3...</span><span class="abstract-full" id="full-2610.09715" hidden>We introduce EC-EarthFlow, a generative flow matching model that emulates simulations from the physical climate model EC-Earth3. The model is trained on transient simulations from EC-Earth3 (1950-2166, SSP2-4.5) to predict the day ahead temperature field from the previous days temperature as well as annual mean temperature. Predictions are made auto-regressively with rollout periods of between a month and an extended season. Using only this variable of interest, we are able to reproduce the daily variability, spatial patterns, annual cycle and long-term trend from EC-Earth3 at a substantially lower computational cost than the physical model. We demonstrate that EC-EarthFlow is stable for long inference periods, and that it can learn the physical relationships as simulated in EC-Earth3.</span> <span class="abstract-toggle" data-id="2610.09715">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.09715v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.09715v1) · [:material-content-copy: BibTeX](../../bibtex/2610.09715.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a> <a class="md-tag" href="/explore/?t=daily" data-tag="daily">Daily</a>
    { .paper-tags }

-   #### Skillful Data-Driven Subseasonal Soil Moisture Forecasting: Prospects and Limits for Flash Drought Prediction { #2610.07060 }

    *Noelia Otero, Atahan Özer, Miguel-Ángel Fernández-Torres, Jackie Ma* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.07060">Despite substantial progress in short-to-medium-range weather forecasting, predicting high-impact events such as flash droughts remains a key challenge for both early warning operations and...</span><span class="abstract-full" id="full-2610.07060" hidden>Despite substantial progress in short-to-medium-range weather forecasting, predicting high-impact events such as flash droughts remains a key challenge for both early warning operations and physically-based subseasonal-to-seasonal (S2S) prediction systems. Here we demonstrate that, for S2S soil-moisture forecasting over Europe, forecast skill depends as much on how the prediction problem is formulated as on the forecasting model itself. Using a Vision Transformer-based architecture with dual-pathway temporal and spatial attention, we show that residual learning is essential to outperform persistence. This advantage is realized only when forecasting root-zone soil moisture in physical units rather than standardized anomalies, revealing that the target representation itself constrains predictability. A probabilistic extension via quantile-head fine-tuning further provides well-calibrated predictive distributions. Benchmarked against deep-learning and operational ECMWF S2S baselines over 2021-2022, our model achieves the highest deterministic and probabilistic skill at all lead times and reliably detects anomalously dry root-zone states (below the 20th percentile). Yet flash drought onset, defined by multi-pentad intensification criteria, remains a fundamental challenge shared across all current S2S systems. These findings advance data-driven S2S soil-moisture forecasting while highlighting the remaining challenge of predicting rapid drought development.</span> <span class="abstract-toggle" data-id="2610.07060">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.07060v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.07060v1) · [:material-content-copy: BibTeX](../../bibtex/2610.07060.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=extremes" data-tag="extremes">Extremes</a> <a class="md-tag" href="/explore/?t=subseasonal-to-seasonal" data-tag="subseasonal-to-seasonal">Subseasonal to seasonal</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=regional" data-tag="regional">Regional</a>
    { .paper-tags }

-   #### ClimateBench v2.0: Probabilistic Climate Model Benchmarking { #2610.04558 }

    *Duncan Watson-Parris, Willa Tobin, Aytaç Paçal, Manuel Schlund, V. Balaji, Kevin Bowman et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.04558">We present ClimateBench v2, a standardized protocol for evaluating climate models on diagnostics expected to be informative for their skill in projecting mid-century regional temperature and...</span><span class="abstract-full" id="full-2610.04558" hidden>We present ClimateBench v2, a standardized protocol for evaluating climate models on diagnostics expected to be informative for their skill in projecting mid-century regional temperature and precipitation changes. The protocol is designed to evaluate any physics-based, data-driven, or hybrid climate model on equal footing using a common set of observational and out-of-distribution tests. We define three tiers of evaluation. Tier I establishes physical credibility through entry-ticket tests of energy conservation, coupled (co-)variability, and basic forced responses. Tier II scores models against post-2015 observations of surface temperature, precipitation, radiative fluxes, sea ice, and key modes of variability using fair CRPS as the primary probabilistic score, complemented by distributional and ensemble-consistency diagnostics. Tier III tests out-of-distribution generalization through paleoclimate simulations spanning the Last Interglacial, Last Glacial Maximum, and Mid-Holocene, and through perfect-model experiments in which data-driven models must predict the future climate of existing Earth system models from historical data alone. We reserve all observational data after 2015 for testing, and submissions must include multiple ensemble members to enable probabilistic evaluation. This reservation exploits a new opportunity provided by the decade of observations accumulated since the end of the CMIP6 historical experiment, which constitutes an out-of-sample record of forced climate change (and internal variability) for the current generation of models, and we quantify, in an idealized setting, the information it carries about mid-century warming. We provide the evaluation code, observational reference datasets, and perfect-model training data as an open benchmark to drive measurable progress in climate projection across all modeling approaches.</span> <span class="abstract-toggle" data-id="2610.04558">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.04558v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.04558v1) · [:material-content-copy: BibTeX](../../bibtex/2610.04558.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### AEGIS: Differentiable Mars Climate Model with Neural Closures { #2610.04081 }

    *Sameera S Kashyap, Victor Cruz, Angel Yepez, Razvan Marinescu* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.04081">General circulation models (GCMs) are the primary tool for simulating planetary atmospheres. They play a vital role in understanding Mars's atmosphere, as forecasting its unique weather is...</span><span class="abstract-full" id="full-2610.04081" hidden>General circulation models (GCMs) are the primary tool for simulating planetary atmospheres. They play a vital role in understanding Mars's atmosphere, as forecasting its unique weather is mission-critical for operations such as entry, descent, and landing. Mars poses unusual challenges for these models, as observations are sparse compared to Earth. In addition, a thin \co{} atmosphere alongside a radiatively active dust cycle creates a volatile atmosphere with large diurnal temperature swings and no true terrestrial analog for validation. Existing Mars GCMs, including the LMD PCM, the NASA Ames Mars GCM, and PlanetWRF, are mature and physically detailed but are implemented in legacy Fortran with finite-difference or finite-volume solvers, and they do not expose gradients for calibration or machine learning. Here we present AEGIS, a modular differentiable Mars climate model that couples Mars's unique atmospheric physics to the Dinosaur dynamical core, with interfaces for neural closures. We showcase stable ten-Mars-year simulations that reproduce the seasonal \co{} cycle while conserving the total \co{} inventory, capture realistic large-scale surface-temperature structure, and produce surface pressure that follows Mars Orbiter Laser Altimeter (MOLA) topography. Gradients through coupled trajectories agree with finite differences and support physical calibration and neural training. We compare with conventional GCMs, highlighting the framework's computational efficiency and differentiability.</span> <span class="abstract-toggle" data-id="2610.04081">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.04081v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.04081v1) · [:material-content-copy: BibTeX](../../bibtex/2610.04081.bib){ .bibtex-link }
    { .paper-links }

-   #### S2S-JEPA: Predicting the Predictable at Subseasonal-to-Seasonal Timescales { #2610.03106 }

    *Chenyu Dong, Gianmarco Mengaldo* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.03106">The subseasonal-to-seasonal (S2S) timescale, roughly from two weeks to two months ahead, is a critical forecast window for sectors such as agriculture, energy, and water management. Yet, it is widely...</span><span class="abstract-full" id="full-2610.03106" hidden>The subseasonal-to-seasonal (S2S) timescale, roughly from two weeks to two months ahead, is a critical forecast window for sectors such as agriculture, energy, and water management. Yet, it is widely known as the ‘predictability desert’. Recent AI weather models excel up to two weeks ahead but deteriorate beyond, largely because they are trained to predict fine-scale details that are neither predictable nor essential at S2S timescales. We argue that a more physically grounded objective is to forecast only the slowly varying components that remain predictable. Computer vision reached the same conclusion with the Joint-Embedding Predictive Architecture (JEPA), which predicts in latent space, discarding unpredictable details. In this work, we introduce S2S-JEPA, which brings the JEPA paradigm to S2S forecasting. It is tailored to this task through design elements from state-of-the-art AI weather models. S2S-JEPA achieves comparable skill to the gold-standard ECMWF physics-based ensemble and surpasses it on multiple metrics at weeks 5 to 6.</span> <span class="abstract-toggle" data-id="2610.03106">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.03106v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.03106v1) · [:material-content-copy: BibTeX](../../bibtex/2610.03106.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=subseasonal-to-seasonal" data-tag="subseasonal-to-seasonal">Subseasonal to seasonal</a>
    { .paper-tags }

-   #### Safe Greenhouse Climate Control Using Lagrangian-Constrained PPO with Kolmogorov-Arnold Networks { #2609.34966 }

    *Hangzun Liu, Yuling Fan, Fang Tian, Zhilong Bie, Zaiwen Feng, Yongliang Qiao* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.34966">Greenhouse climate control balances economic return with maintaining temperature, humidity and CO2 within crop-adapted growth ranges. Conventional reinforcement learning (RL) greenhouse controllers...</span><span class="abstract-full" id="full-2609.34966" hidden>Greenhouse climate control balances economic return with maintaining temperature, humidity and CO2 within crop-adapted growth ranges. Conventional reinforcement learning (RL) greenhouse controllers use fixed reward penalties to limit climate constraint violations, yet such heuristic penalties cannot explicitly constrain long-term cumulative violations. Poorly tuned weights either lead to overly conservative policies and lower yields, or fail to suppress persistent climate deviations that harm photosynthesis and induce crop diseases. To address this issue, we formulate greenhouse climate regulation as a Constrained Markov Decision Process (CMDP) and use a Lagrangian safe RL framework RCPO-PPO to separate economic optimization and cumulative safety constraints, enabling adaptive penalty adjustment without manual tuning. To handle strong nonlinear, time-varying coupling between greenhouse microclimate and crop growth, Kolmogorov-Arnold Networks (KANs) replace Multi-Layer Perceptrons (MLPs) as policy and value approximators for improved nonlinear representation. Sinusoidal cyclic time features are embedded in observations to capture diurnal environmental periodicity. Simulations use a classic winter lettuce greenhouse model driven by 40-day real weather disturbances. Compared with vanilla penalty-based PPO, our method cuts cumulative climate violations by 18.65% and raises lettuce economic profit by 2.91%, keeping violations stable near the safety threshold. This decoupled CMDP optimization with KAN-based policy representation mitigates long-term climate risks and boosts planting profits, offering a constraint-aware control strategy for precision greenhouse cultivation.</span> <span class="abstract-toggle" data-id="2609.34966">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.34966v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.34966v1) · [:material-content-copy: BibTeX](../../bibtex/2609.34966.bib){ .bibtex-link }
    { .paper-links }

-   #### Learning Hierarchical Causal Representations of the Effects of Forcings on Temperature in Climate Models { #2609.30995 }

    *Shan Zhao, Ilija Trajkovic, Julia Kaltenborn, Yaniv Gurwicz, Peer Nowack, David Rolnick et al.* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.30995">Machine learning (ML) emulators provide a fast and cost-effective method to simulate climate change scenarios after being trained on Earth System Models projections. However, the black-box nature of...</span><span class="abstract-full" id="full-2609.30995" hidden>Machine learning (ML) emulators provide a fast and cost-effective method to simulate climate change scenarios after being trained on Earth System Models projections. However, the black-box nature of those data-driven approaches limit the usability and trustworthiness of their outputs and in particular their use as causal attribution tools. Here, we develop a hierarchical causal representation learning framework applied to sea surface temperature fields from a state-of-the-art global climate model. As a key advance over previous work, our framework explicitly models both atmospheric dynamical interactions arising from internal climate variability and forced responses due to changes in atmospheric greenhouse gas and aerosol concentrations. When trained on future climate change scenarios, our method accurately predicts the long-term global mean and regional temperature evolution and shows physically realistic responses to perturbations in greenhouse gas and aerosol concentrations when evaluated on unseen scenarios. Our results underline the potential of causal representation learning frameworks for advancing climate model emulation.</span> <span class="abstract-toggle" data-id="2609.30995">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.30995v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.30995v1) · [:material-content-copy: BibTeX](../../bibtex/2609.30995.bib){ .bibtex-link }
    { .paper-links }

-   #### Understanding Perturbed Parameter Ensemble Sensitivities Using A Contrastive Learning Approach { #2609.30420 }

    *Da Fan, David John Gagne, Gregory S Elsaesser, Brian Medeiros, Addisu G Semie, Qingyuan Yang et al.* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.30420">Perturbed parameter ensembles (PPEs) reveal how physics parameters affect climate simulations, but interpreting parameter sensitivities across multivariate, spatially structured outputs remains...</span><span class="abstract-full" id="full-2609.30420" hidden>Perturbed parameter ensembles (PPEs) reveal how physics parameters affect climate simulations, but interpreting parameter sensitivities across multivariate, spatially structured outputs remains challenging, particularly when calibrating models against observations. We develop an explainable contrastive learning model that maps 5 monthly cloud and radiation fields into a shared representation space. We train the model on the fields of two 100-member Community Atmosphere Model version 6 (CAM6) PPEs, spanning 34 parameters, that only differ in the warm rain microphysics scheme: KK2000, the default bulk microphysics scheme, and TAU-ML, a neural network emulator of a bin microphysics scheme. The learned representations separates two PPEs with over 94% linear classification accuracy while preserving the seasonal variability and ensemble spread due to parameter perturbations. In the shared representation space, the representations of satellite observations occupy the same low-dimensional manifold as the PPEs but are displaced from them most strongly during boreal spring and autumn. TAU-ML PPE has a lower distance to observations compared to KK2000 in the representation space. Integrated Gradients attributions highlights the contributions in subtropical low-cloud regions, Northern and Southern Hemisphere storm track regions, and tropical convection regions to differences between PPEs and observations. Regional attributions correlate most strongly with parameters associated with cloud microphysics, boundary layer turbulence, and deep convection. These results demonstrate that explainable representations of climate fields can attribute model differences to specific variables, regions, seasons, and physical parameters.</span> <span class="abstract-toggle" data-id="2609.30420">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.30420v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.30420v1) · [:material-content-copy: BibTeX](../../bibtex/2609.30420.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=interpretability" data-tag="interpretability">Interpretability</a> <a class="md-tag" href="/explore/?t=monthly" data-tag="monthly">Monthly</a>
    { .paper-tags }

-   #### Analysis of trade-offs in urban heat mitigation using a Bayesian Optimization framework for an urban canopy layer model { #2609.25953 }

    *Rebekka Walter, Johanna Gelhaus, David Anton, Henning Wessles, Stephan Weber* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.25953">To mitigate the challenges of climate change and intensifying heat stress in urban areas, local adaptation strategies are discussed and introduced in cities worldwide. To understand processes and...</span><span class="abstract-full" id="full-2609.25953" hidden>To mitigate the challenges of climate change and intensifying heat stress in urban areas, local adaptation strategies are discussed and introduced in cities worldwide. To understand processes and potential trade-offs of these strategies a Bayesian optimization and surrogate modeling framework was employed to investigate urban parameter ranges of heat mitigation strategies with focus on three thermal metrics: daytime air temperature, Universal Thermal Climate Index (UTCI), and nighttime air temperature. Based on an urban street canyon configuration, it was shown that heat mitigation measures that reduce daytime air temperature and UTCI are often associated with higher nighttime temperatures. This results in a curved Pareto front that reflects the trade-off between daytime and nighttime thermal comfort. Under identical forcing conditions, different urban configurations, varying in geometry, vegetation, and surface characteristics, are shown to alter peak canyon air temperature by up to 5.2~$^\circ$C during the day and 2.6~$^\circ$C at night, while UTCI varies by up to 7.9~$^\circ$C, demonstrating that favorable urban configurations can substantially mitigate microclimatic heat stress. These findings suggest that combining multi-objective Bayesian optimization and surrogate modeling can help bridge the gap between computationally intensive climate simulations and practical decision-making in urban planning. An interactive visualization tool was developed to explore these trade-offs, making the often opposing relationships between urban parameters and the three thermal metrics directly accessible to planners.</span> <span class="abstract-toggle" data-id="2609.25953">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.25953v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.25953v1) · [:material-content-copy: BibTeX](../../bibtex/2609.25953.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=global" data-tag="global">Global</a>
    { .paper-tags }

-   #### Learning Prognostic Variables for AI Convective Parameterizations via Symbolic Distillation { #2609.24882 }

    *Jurij Schönfeld, Tom Beucler, Julien Savre, Steven Sherwood, Veronika Eyring* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.24882">Hybrid AI-physics climate modeling aims to improve coarse (~100km-resolution) Earth system models by learning to parameterize subgrid processes from high-fidelity data. However, this so far mostly...</span><span class="abstract-full" id="full-2609.24882" hidden>Hybrid AI-physics climate modeling aims to improve coarse (~100km-resolution) Earth system models by learning to parameterize subgrid processes from high-fidelity data. However, this so far mostly involves local-in-time, diagnostic parameterizations, in which the subgrid state depends only on the current coarse state with no memory of previous states, which is unrealistic for processes such as convection that have intrinsic persistence. To address this, we enhance local-in-time parameterizations by learning prognostic variables that compactly carry important, additional past information where no explicit sub-grid information is available. First we compress past information into a low-dimensional latent space using an autoencoder, which then informs a neural network trained to parameterize targeted subgrid-scale processes. We then replace the autoencoder with symbolic equations that govern the time evolution of the latent variables, yielding additional prognostic memory variables that can be integrated alongside the resolved atmospheric state. We evaluate this approach on two systems: the Lorenz-96 model (online) and surface precipitation from high-resolution atmospheric simulations (offline). A forced multivariate linear ordinary differential equation recovers most of the added value achieved by the autoencoder-based approach in both experiments. Benchmarked against diagnostic parameterizations without memory, our memory-informed approach improves climate statistics and temporal structure, including a realistic diurnal cycle of tropical land precipitation.</span> <span class="abstract-toggle" data-id="2609.24882">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.24882v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.24882v1) · [:material-content-copy: BibTeX](../../bibtex/2609.24882.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a>
    { .paper-tags }

-   #### Climate Variability Modulates the Impact of Price Spikes on Food Insecurity { #2609.24394 }

    *Jordi Cerdà-Bautista, Vasileios Sitokonstantinou, Homer Durand, Gherardo Varando, Michele Ronco et al.* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.24394">Climate variability influences whether a market disruption escalates into a food crisis, yet broad climate patterns like El Niño, tracked months before they alter hydro-climatic conditions, are still...</span><span class="abstract-full" id="full-2609.24394" hidden>Climate variability influences whether a market disruption escalates into a food crisis, yet broad climate patterns like El Niño, tracked months before they alter hydro-climatic conditions, are still not incorporated as an early-warning component in food-security responses. We address this gap by introducing sensitivity regimes, a stratification of regions by the direction and strength of their vegetation response to the El Niño Southern Oscillation, and using them to estimate how food price spikes affect acute food insecurity across sub-Saharan Africa. Integrating remote sensing, socioeconomic data, and causal machine learning, we find that in regions where ENSO systematically suppresses vegetation, a price spike raises the share of the population at acute risk by 5.4 percentage points in the following month. In regions where vegetation is unaffected by or positively linked to ENSO, the estimated effect is smaller (around 2 percentage points) and statistically insignificant. These results demonstrate that climate context is critical for understanding food security vulnerabilities. Sensitivity regimes can be combined with operational price-spike triggers to stage anticipatory action: the ENSO state flags vulnerable regions months ahead, and a pre-positioned response in those regions to a price spike would avert the largest jump in acute food insecurity.</span> <span class="abstract-toggle" data-id="2609.24394">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.24394v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.24394v1) · [:material-content-copy: BibTeX](../../bibtex/2609.24394.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=subseasonal-to-seasonal" data-tag="subseasonal-to-seasonal">Subseasonal to seasonal</a>
    { .paper-tags }

-   #### A more predictable Madden-Julian Oscillation index derived from Koopman spectral analysis { #2609.19435 }

    *Claire Valva, Edwin P. Gerber* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.19435">The Madden-Julian oscillation (MJO) is a major source of subseasonal-to-seasonal (S2S) predictability. The MJO is commonly defined and tracked with indices such as the Real-time Multivariate MJO...</span><span class="abstract-full" id="full-2609.19435" hidden>The Madden-Julian oscillation (MJO) is a major source of subseasonal-to-seasonal (S2S) predictability. The MJO is commonly defined and tracked with indices such as the Real-time Multivariate MJO (RMM) index. Although the RMM provides a useful description of the MJO, its evolution can be noisy and difficult to predict. We define an MJO index using a data-driven approximation of the Koopman operator. The Koopman index captures similar tropical circulation and convection patterns to the RMM but evolves more smoothly and predictably. Skillful prediction extends to 46 days for the Koopman index compared to 11 days for the RMM under the same prediction framework. While this new approach does not recover the RMM as well as operational S2S models, which provide skillful forecasts up to 35 days, the Koopman index could complement existing MJO diagnostics in evaluating and developing extended-range forecast systems.</span> <span class="abstract-toggle" data-id="2609.19435">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.19435v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.19435v1) · [:material-content-copy: BibTeX](../../bibtex/2609.19435.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=subseasonal-to-seasonal" data-tag="subseasonal-to-seasonal">Subseasonal to seasonal</a>
    { .paper-tags }

</div>

<nav class="pager" markdown="span">**1** [2](2.md) [3](3.md) [4](4.md) [5](5.md) [6](6.md) [7](7.md) [8](8.md) [9](9.md) [10](10.md) [11](11.md) [Older :material-arrow-right:](2.md){ .pager-step }</nav>

