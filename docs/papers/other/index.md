---
title: 'Other'
hide:
  - toc
---

<div class="listing-header" markdown>

# Other

<p class="page-meta" markdown="span">216 papers · page 1 of 8 · <a href="../../bib/other.bib" download>:material-download: BibTeX for this topic</a></p>

</div>

<div class="grid cards" markdown>

-   #### An Input Skip Can Reparameterize a VAE Without Improving the Generator { #2610.37456 }

    *Kevin Lee* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.37456">An input skip can reduce reconstruction error while leaving a model's distribution of new samples unchanged. For a Gaussian variational autoencoder (VAE) with constant input skip $0\leqα<1$, a model...</span><span class="abstract-full" id="full-2610.37456" hidden>An input skip can reduce reconstruction error while leaving a model's distribution of new samples unchanged. For a Gaussian variational autoencoder (VAE) with constant input skip $0\leqα<1$, a model without a skip can be written in skip form by multiplying its decoder output by $b=1-α$ and its noise variance by $b^2$. The distribution of observations and latent variables and its exact generative scores stay the same, while reconstruction mean squared error (MSE) is multiplied by $b^2$. Separate training can produce different generators because optimization depends on the parameter coordinates. In a Gaussian experiment with known density, changing skip and observation variance together reduces MSE but worsens energy score and the estimate of negative log density from importance sampling. On seven annual Deep-ocean Assessment and Reporting of Tsunamis (DART) records, energy score differences at the same observation variance have both signs on four original records and favor the skip on three records from 2018. Probabilistic principal component analysis (PPCA) has lower mean energy scores than every neural DART fit. With the encoder and decoder kept the same, estimating observation variance from training reconstruction error can also worsen energy score. Assessing generation requires more than reconstruction error.</span> <span class="abstract-toggle" data-id="2610.37456">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.37456v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.37456v1) · [:fontawesome-brands-github: Code](https://github.com/magnaprog/vae-skip-reparam) · [:material-content-copy: BibTeX](../../bibtex/2610.37456.bib){ .bibtex-link }
    { .paper-links }

-   #### Machine Learning Applications in the Solar Interior and Atmosphere { #2610.34687 }

    *Vishal Upendran, Carlos J. Díaz Baso* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.34687">The solar interior and atmosphere constitute a strongly coupled, multiscale dynamical system defined by continuous temporal evolution. Predicting the state of this system requires modeling physical...</span><span class="abstract-full" id="full-2610.34687" hidden>The solar interior and atmosphere constitute a strongly coupled, multiscale dynamical system defined by continuous temporal evolution. Predicting the state of this system requires modeling physical interactions that span disparate spatial and temporal scales, constrained by heterogeneous and sparse data. This chapter formalizes the Sun within the mathematical framework of dynamical systems theory, defining a global state vector ($\vec{S}$) governed by thermodynamical, kinetic, and magnetic properties. Because solar observations represent projected, multi-modal, and partially observable measurements ($\vec{O}[\vec{S}]$) of this hidden state, pure physics-based deduction is fundamentally constrained. We present a unifying thesis: the primary role of machine learning in heliophysics is to act as a structured family of mathematical operators---specifically Inference ($\mathcal{I}$), Emulation ($\mathcal{E}$), Propagation ($\mathcal{P}$), and Control ($\mathcal{C}$)---that reconstruct, simulate, and couple this global state vector across disparate physical layers. By framing the vast, multi-disciplinary literature within this unified state-space taxonomy, we demonstrate how data-driven techniques bridge observational gaps to resolve hidden multiscale couplings and dramatically accelerate the simulation of non-linear processes, establishing a rigorous pathway to predict the global evolution of the Sun.</span> <span class="abstract-toggle" data-id="2610.34687">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.34687v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.34687v1) · [:material-content-copy: BibTeX](../../bibtex/2610.34687.bib){ .bibtex-link }
    { .paper-links }

-   #### Probabilistic Machine Learning for Short-Term Atmospheric Seeing Forecasting { #2610.29654 }

    *Mary-Joe Medlej, Rahul Srinivasan, Simon Prunet, Aziz Ziad, Christophe Giordano* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.29654">Accurate short-term forecasting of atmospheric seeing is of significant interest for astronomical observations and free-space optical communication systems, where rapid variations in optical...</span><span class="abstract-full" id="full-2610.29654" hidden>Accurate short-term forecasting of atmospheric seeing is of significant interest for astronomical observations and free-space optical communication systems, where rapid variations in optical turbulence can affect system performance and operational planning. In this work, we benchmark machine learning models for predicting atmospheric seeing up to two hours in advance, with a particular focus on probabilistic forecasting and uncertainty quantification. We compare two probabilistic approaches, Gaussian Processes (GPs) and a normalizing-flow- based model (FloTS), against deterministic recurrent neural network (RNN) and long short-term memory (LSTM) baselines. All models are trained and evaluated on seeing measurements from the Maunakea summit. We assess the models using both deterministic metrics, including the root mean square error and the Pearson correlation coefficient, and probabilistic metrics that evaluate the quality and calibration of predictive uncertainty estimates. The results show that deep learning approaches outperform standard recurrent networks for short-term forecasting, with LSTM and FloTS achieving the best overall predictive accuracy. While Gaussian Processes provide a useful probabilistic baseline, their predictive distributions are limited by the Gaussian assumption. FloTS combines competitive forecasting accuracy with flexible uncertainty quantification and produces the most informative probabilistic forecasts.</span> <span class="abstract-toggle" data-id="2610.29654">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.29654v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.29654v1) · [:material-content-copy: BibTeX](../../bibtex/2610.29654.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### Urban climate regressions require representative monitoring networks { #2610.29423 }

    *Martí Bosch, Aldo Brandi, Christoph Bachofen, Perrine Hamel, Roy P. Remme, Emma E. Ramsay et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.29423">Land use regression models relate intra-urban air temperature to spatial predictors such as building volume, tree canopy cover and topography, and are increasingly used to produce fine-scale...</span><span class="abstract-full" id="full-2610.29423" hidden>Land use regression models relate intra-urban air temperature to spatial predictors such as building volume, tree canopy cover and topography, and are increasingly used to produce fine-scale temperature maps. However, the accuracy of these estimates can be compromised by unrepresentative monitoring networks. Here we show that reference weather station networks, designed for regional rather than intra-urban monitoring, sample only a narrow portion of a city's spatial heterogeneity, missing both the densely built core and the cool, vegetated periphery that set the two ends of the urban temperature range. As a result, city-wide predictions underestimate temperature differences, in ways that cross-validation against the station measurements does not detect. Using Zurich, Switzerland, as a case study, we compare multiple linear regression, generalized additive models and random forests as up to 117 low-cost temperature sensors are added to the four reference weather stations. When trained on reference stations alone, all three models achieve comparable cross-validated performance. As low-cost sensors broaden the sampled spatial heterogeneity, multiple linear regression scores lower on both metrics, whereas the non-linear models remain at their reference-only level. Although a third of the low-cost sensors already recovers nearly the full predicted temperature range, which predictors are identified, and with which sign, is determined by the station sample rather than the model class. At full coverage, only multiple linear regression leaves significant residual spatial autocorrelation, suggesting a linear form unable to capture non-linear responses of air temperature to urban form. Drawing on landscape ecology research on focal site study design, we provide a transferable framework for diagnosing station network representativeness and improving the reliability of city-wide environmental maps.</span> <span class="abstract-toggle" data-id="2610.29423">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.29423v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.29423v1) · [:material-content-copy: BibTeX](../../bibtex/2610.29423.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=classical-ml" data-tag="classical-ml">Classical ML</a> <a class="md-tag" href="/explore/?t=station-point" data-tag="station-point">Station / point</a>
    { .paper-tags }

-   #### Data-driven and physics-guided hindcasting of measured wave spectra { #2610.29306 }

    *Sander Rikka* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.29306">Short-range hindcasts of measured wave spectra are evaluated using a recurrent model initialized by one hourly buoy spectrum and supplied with local wind, fetch and depth. Measurements from three...</span><span class="abstract-full" id="full-2610.29306" hidden>Short-range hindcasts of measured wave spectra are evaluated using a recurrent model initialized by one hourly buoy spectrum and supplied with local wind, fetch and depth. Measurements from three Baltic Sea stations are combined for training in 2017--2020, validation in 2021 and testing in 2022. ERA5 reanalysis supplies winds throughout the twelve-hour prediction interval. The comparison examines supervision of spectral variance, peak frequency and peak amplitude, followed by empirical wind-sea bounds and a local relaxation constraint. Variance and peak-frequency supervision gives the lowest spectral error within the data-driven comparison, whereas additional peak-amplitude supervision gives the lowest wave-height error. In the matched physical comparison, relaxation reduces height RMSE by 5.01% to \PhPtwoHs{}~cm. Extending this constraint through hours 13--24, without spectral targets at those hours, reduces primary spectral RMSE by 2.31% to \PhPthreeSpec{}~$\unitE$. Upper-bound constraints increase errors. Wind-condition comparisons show spectral gains from extended relaxation in increasing, decreasing and steady strong winds, with little change in steady weak winds. The results support modest, objective-dependent benefits from local physical guidance. Wave-height scores concern the retained frequency band, and operational forecasting and transfer to independent stations remain to be evaluated.</span> <span class="abstract-toggle" data-id="2610.29306">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.29306v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.29306v1) · [:material-content-copy: BibTeX](../../bibtex/2610.29306.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=physics-ml-hybrid" data-tag="physics-ml-hybrid">Physics–ML hybrid</a> <a class="md-tag" href="/explore/?t=station-point" data-tag="station-point">Station / point</a> <a class="md-tag" href="/explore/?t=hourly" data-tag="hourly">Hourly</a>
    { .paper-tags }

-   #### Zephyrus-R1: Agentic Reinforcement Learning for Weather Reporting { #2610.25625 }

    *Fengfei Yu, Marshall Fisher, Isadora White, Sumanth Varambally, Yasaman Jafari, Shivam Yadav et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.25625">Reinforcement learning (RL) has emerged as a key method for training AI models in scientific applications, but RL remains difficult when outputs cannot easily be verified against a known ground-truth...</span><span class="abstract-full" id="full-2610.25625" hidden>Reinforcement learning (RL) has emerged as a key method for training AI models in scientific applications, but RL remains difficult when outputs cannot easily be verified against a known ground-truth answer. Natural-language weather forecast reports are one such domain: they translate atmospheric data into assessments of weather evolution, hazards, and uncertainty for both the general public and key decision-makers. Unlike numerical weather-analysis questions, they admit many valid outputs, which limits deterministic verification. To address this, we develop Zephyrus-R1, a practical framework for RL on scientific weather forecast reporting. We focus on three key components of the agentic RL pipeline: reward design, curriculum learning, and an analysis of human-expert alignment. To encourage both report legibility and factual accuracy, we design a reward function that combines a meteorological rubric with bidirectional claim entailment, both evaluated against National Weather Service reference reports. Our framework optimizes this reward in a simulated tool-calling environment, enabling vision-language agents to gather and analyze forecast evidence across regions and lead times before producing a report. We use annotations from three meteorologists to audit the claim labels produced by our LLM judges. Finally, a curriculum helps small models first develop tool use and scientific reasoning through RL on verifiable weather-analysis questions, then optimize the open-ended report reward. After training Qwen3.5-4B with Zephyrus-R1, we improve the mean across six judge-metric scores by 26% over the base model and achieve results competitive with frontier open-weight models. For Qwen3.8-27B, we improve the same mean by 13% over the base model, by 6.7% over Kimi-K3, and by 2.7% over GPT-5.6-Luna.</span> <span class="abstract-toggle" data-id="2610.25625">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.25625v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.25625v1) · [:material-content-copy: BibTeX](../../bibtex/2610.25625.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=llms-agents" data-tag="llms-agents">LLMs & agents</a> <a class="md-tag" href="/explore/?t=reinforcement-learning" data-tag="reinforcement-learning">Reinforcement learning</a>
    { .paper-tags }

-   #### Energy Flow Shapes the Decomposition Dynamics of a Reactive Atmospheric Intermediate { #2610.21259 }

    *Mitradip Das, Cangtao Yin, M. Meuwly* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.21259">The unimolecular dynamics of Criegee intermediates depend strongly on their internal energy, which can differ substantially between nascent atmospheric species and laboratory-prepared reactants....</span><span class="abstract-full" id="full-2610.21259" hidden>The unimolecular dynamics of Criegee intermediates depend strongly on their internal energy, which can differ substantially between nascent atmospheric species and laboratory-prepared reactants. Here, the decomposition dynamics of <em>anti-</em>CH$_3$CHOO are investigated using molecular dynamics simulations on a neural network-based potential energy surface under thermal and mode-selective vibrational excitation. Simulations were performed at internal energies representative of those released during ozonolysis (chemical/thermal preparation, TP) or following vibrational excitation (VibEx) along the CH-stretching mode with two or three quanta, as can be done in laboratory studies. These excitation conditions strongly affect reaction yields and kinetics, with VibEx promoting passage across the initial cyclization barrier. For TP, 35 kcal/mol of internal energy are required to form molecular products, whereas for VibEx, 27 kcal/mol are sufficient. Although TP and VibEx ensembles access largely the same fragmentation pathways with comparable relative yields, the absolute product yields, characteristic timescales, and kinetic behavior differ substantially. Methyldioxirane is found to be a kinetic trap along the pathway. The resulting molecular-level description of <em>anti-</em>CH$_3$CHOO decomposition provides guidance for interpreting experiments and contributes to a broader understanding of energy-dependent Criegee-intermediate dynamics.</span> <span class="abstract-toggle" data-id="2610.21259">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.21259v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.21259v1) · [:material-content-copy: BibTeX](../../bibtex/2610.21259.bib){ .bibtex-link }
    { .paper-links }

-   #### Variable Masking for Climate Emulation in ACE { #2610.21012 }

    *Alexey Yermakov, Troy Arcomano, Jeremy McGibbon, Brian Henn, Oliver Watt-Meyer* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.21012">In the recent AI Model Intercomparison Project (AIMIP) benchmark, the ACE2.1-ERA5 emulator was shown to underestimate the true near-surface warming trend both during the training period and the...</span><span class="abstract-full" id="full-2610.21012" hidden>In the recent AI Model Intercomparison Project (AIMIP) benchmark, the ACE2.1-ERA5 emulator was shown to underestimate the true near-surface warming trend both during the training period and the out-of-sample extrapolation period. In this work, we present M-ACE2S (Masked-ACE2S), a modified version of the stochastic ACE2S model, which uses variable masking during training and reduces the reliance of M-ACE2S on any single input channel. We show that M-ACE2S improves long-term near-surface air temperature prediction. We also present a novel data normalization strategy called "global mean removal", which, combined with masking, further improves the near-surface temperature forecasts. We show that M-ACE2S outperforms the ACE2S baseline on in-sample data for ERA5 while improving training stability and generalizability of near-surface temperature to unseen forcing when paired with global mean removal.</span> <span class="abstract-toggle" data-id="2610.21012">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.21012v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.21012v1) · [:fontawesome-brands-github: Code](https://github.com/ai2cm/ace) · [:material-content-copy: BibTeX](../../bibtex/2610.21012.bib){ .bibtex-link }
    { .paper-links }

-   #### Forecasting carbon-risk hotspots ahead of the extreme 2026 El Niño { #2610.16714 }

    *Xiaomeng Du, Runda Jiang, Philippe Ciais, Pierre Friedlingstein, Stephen Sitch, Mike O'Sullivan et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.16714">A potentially super El Niño is forecasted in 2026, raising the need to anticipate land carbon-cycle risks before its impacts unfold. We develop a prospective early-warning framework that uses dynamic...</span><span class="abstract-full" id="full-2610.16714" hidden>A potentially super El Niño is forecasted in 2026, raising the need to anticipate land carbon-cycle risks before its impacts unfold. We develop a prospective early-warning framework that uses dynamic global vegetation models driven by an ensemble of ECMWF seasonal climate forecasts and machine-learning burned area predictions. Forecasts indicate a $0.23 \pm 0.21$ PgC reduction in global net biome productivity in the second half of 2026, with hotspots of sink weakening in eastern South America and elevated fire risk in Southeast Asia.</span> <span class="abstract-toggle" data-id="2610.16714">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.16714v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.16714v1) · [:material-content-copy: BibTeX](../../bibtex/2610.16714.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=subseasonal-to-seasonal" data-tag="subseasonal-to-seasonal">Subseasonal to seasonal</a>
    { .paper-tags }

-   #### Sensitivity Fidelity Reveals Physical Dependencies Beyond Forecast Accuracy in Deep-Learning Earth-System Models { #2610.15540 }

    *Qi Chen, Peng Zhan, Hongxing Cui, Wenqing Cui, Ibrahim Hoteit* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.15540">Deep-learning models increasingly rival numerical simulation in Earth-system forecasting, yet forecast error alone cannot determine whether their learned input-output dependencies are physically...</span><span class="abstract-full" id="full-2610.15540" hidden>Deep-learning models increasingly rival numerical simulation in Earth-system forecasting, yet forecast error alone cannot determine whether their learned input-output dependencies are physically consistent. Here we introduce sensitivity fidelity as an additional criterion: alongside prediction error, model responses to input perturbations should be examined against a trusted reference or established physical knowledge. In an idealized Rayleigh-Bénard convection benchmark, four surrogate architectures achieved low forecast errors yet encoded markedly different sensitivities. Quantitative comparison with a matched adjoint distinguished the models far more sharply than forecast accuracy and showed that predictive skill and sensitivity fidelity need not coincide, while finite-perturbation experiments further linked these differences to the propagation of initial-condition uncertainty. We then applied this strategy to El Niño-Southern Oscillation prediction, where no adjoint reference exists and learned sensitivities showed event-wise robustness, validity under finite perturbations, and consistency with established precursors. These results identify gradient-based evaluation as a discriminating complement to forecast-error metrics, providing a practical test of whether data-driven Earth-system models learn physically credible dependencies.</span> <span class="abstract-toggle" data-id="2610.15540">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.15540v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.15540v1) · [:material-content-copy: BibTeX](../../bibtex/2610.15540.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=subseasonal-to-seasonal" data-tag="subseasonal-to-seasonal">Subseasonal to seasonal</a>
    { .paper-tags }

-   #### A Physics-Constrained Implicit Profile Network for Continuous Reconstruction of Tropical Cyclone Near-Surface Wind Profiles { #2610.11405 }

    *Jian Ma, Yilin Yang, Robert Rogers, Jun A. Zhang, Jie Tang* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.11405">Near the ocean surface, tropical cyclone winds change rapidly with height, but direct measurements are limited because aircraft dropsondes provide only sparse and irregular observations. Continuous...</span><span class="abstract-full" id="full-2610.11405" hidden>Near the ocean surface, tropical cyclone winds change rapidly with height, but direct measurements are limited because aircraft dropsondes provide only sparse and irregular observations. Continuous wind profiles are important for understanding hurricane boundary-layer processes, improving storm-surge prediction, supporting offshore engineering, and assessing coastal hazards. In this study, we developed an artificial intelligence model that combines machine learning with physical principles to reconstruct continuous wind profiles from sparse observations. The model is designed to preserve the observed surface winds while generating realistic changes in wind speed and direction with height. Tests using more than two decades of NOAA hurricane observations show that the method accurately reproduces the vertical structure of tropical cyclone winds over a wide range of storm intensities. The framework can also extend satellite-derived surface wind measurements into three-dimensional near-surface wind fields, providing new opportunities for hurricane research, operational forecasting, and engineering applications.</span> <span class="abstract-toggle" data-id="2610.11405">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.11405v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.11405v1) · [:material-content-copy: BibTeX](../../bibtex/2610.11405.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=physics-ml-hybrid" data-tag="physics-ml-hybrid">Physics–ML hybrid</a> <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a>
    { .paper-tags }

-   #### Conditional Flow Matching for Generation of 3D Multi-variable Instantaneous Urban Microclimate Fields { #2610.10430 }

    *Peng Liu, Shaoxiang Qin, Theodore Potsis, Lili Ji, Dingyang Geng, Liangzhu Leon Wang* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.10430">Rapid and accurate prediction of urban wind and temperature fields is important for urban microclimate design and climate adaptation. Large-eddy simulation (LES) effectively resolves these...</span><span class="abstract-full" id="full-2610.10430" hidden>Rapid and accurate prediction of urban wind and temperature fields is important for urban microclimate design and climate adaptation. Large-eddy simulation (LES) effectively resolves these instantaneous fields, but its application is limited in iterative design of urban microclimate applications due to high computational cost. Existing regressive data-driven models offers quick outputs, but they produce only deterministic point predictions that inherently fail to represent turbulent stochasticity. This paper adopts a novel generative framework of Conditional Flow Matching (CFM) that uses building geometry and mean flow as guidance to generate plausible three-dimensional instantaneous velocity and temperature fields for urban microclimate in seconds. To overcome the GPU memory bottleneck of pixel space 3D generation, the model operates in parallel on overlapping pixel space through a shared-noise initialization that preserves high spatial continuity of flow structure across the entire domain. Against reference LES data, the CFM surrogate can rapidly and accurately restore the first-order statistics with Normalized Root Mean Square Error (NRMSE) of 2.99% for wind and 1.77% for temperature, second-order turbulence metrics with NRMSE of 7.17% for wind and 8.84% for temperature, turbulent kinetic energy with NRMSE of 7%, probability density function and vertical profiles in representative locations. Wind engineering application of local gust prediction demonstrate that the speed and accuracy of CFM, supporting the use of generative AI for making turbulence-aware resilient urban design and climate adaptation more computationally feasible.</span> <span class="abstract-toggle" data-id="2610.10430">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.10430v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.10430v1) · [:material-content-copy: BibTeX](../../bibtex/2610.10430.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a>
    { .paper-tags }

-   #### SoftSEEPS improves ML-based precipitation forecasting { #2610.09752 }

    *Jost Arndt, Utku Isil, Noelia Otero, Rodrigo Almeida, Wojciech Samek, Jackie Ma* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.09752">In this paper we have developed a differentiable approximation of the well-known SEEPS score, which we name SoftSEEPS. This allows the training of a Machine Learning model to forecast precipitation...</span><span class="abstract-full" id="full-2610.09752" hidden>In this paper we have developed a differentiable approximation of the well-known SEEPS score, which we name SoftSEEPS. This allows the training of a Machine Learning model to forecast precipitation directly. We test SoftSEEPS on the IMERG dataset (0.1 degree resolution) by training a decoder for precipitation on the latent space of a pre-trained low-resolution forecasting model. Combining SoftSEEPS and RMSE in a joint objective is possible with marginal trade-offs in either metric.</span> <span class="abstract-toggle" data-id="2610.09752">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.09752v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.09752v1) · [:material-content-copy: BibTeX](../../bibtex/2610.09752.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a>
    { .paper-tags }

-   #### Beyond the Doppler Dilemma: Improved Fast Weather Radar Unambiguous Doppler Velocity Spectrum Reconstruction from Sparse Aperiodic Sweeps { #2610.09172 }

    *Tworit Dash, S. A. K. Syed Mohamed, Oleg Krasnov, Alexander Yarovoy* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.09172">The problem of fast Doppler counter-aliasing for short-dwell weather-radar measurements is addressed. The proposed approach realizes a log-periodic slow-time design by changing only the waiting time...</span><span class="abstract-full" id="full-2610.09172" hidden>The problem of fast Doppler counter-aliasing for short-dwell weather-radar measurements is addressed. The proposed approach realizes a log-periodic slow-time design by changing only the waiting time between otherwise unchanged frequency-modulated continuous-wave (FMCW) chirps. The fast-time waveform, bandwidth, and range-processing chain are preserved while the acquisition changes the Doppler ambiguity structure. We formulate a novel moment-space ambiguity function for distributed weather spectra. It measures statistical ambiguity jointly in mean Doppler velocity and spectral width and expresses the competing sidelobes through their expected relative likelihood. A complex Gaussian process (CGP) maximum-likelihood estimator reconstructs the spectrum from the measured aperiodic samples. A deterministic basin-selection and optimization strategy accelerates the same CGP likelihood without defining a different statistical estimator. Real precipitation measurements establish central-branch stability over an extended search domain. Using 64 slow-time sweeps, we compare three acquisitions and their associated estimators. Periodic data are processed by the discrete Fourier transform (DFT) and parametric spectral estimation (PSE), log-periodic data by the nonuniform discrete Fourier transform (NUDFT) and CGP, and staggered data by velocity-difference and CGP estimators. The differences become more pronounced when the dwell is reduced to 32 sweeps. The results show that aperiodic time sampling enables extended-domain branch identification without degrading spectral-width estimation.</span> <span class="abstract-toggle" data-id="2610.09172">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.09172v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.09172v1) · [:material-content-copy: BibTeX](../../bibtex/2610.09172.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=classical-ml" data-tag="classical-ml">Classical ML</a>
    { .paper-tags }

-   #### AI Emulation of Stochastic Sudden Stratospheric Warming with Interpretable Latent Structure { #2610.02069 }

    *C. Daniel Boscu, Daniel Hernandez, Fabio Alvarez Ventura, Justin Finkel, Ashesh Chattopadhyay et al.* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.02069">Rare weather regime transitions pose a challenge for data-driven modeling due to class imbalance. In this study, we develop a probabilistic deep learning emulator for a prototypical system with...</span><span class="abstract-full" id="full-2610.02069" hidden>Rare weather regime transitions pose a challenge for data-driven modeling due to class imbalance. In this study, we develop a probabilistic deep learning emulator for a prototypical system with regime transitions, the stochastic Holton--Mass model of stratospheric variability, and analyze the structure of its learned latent space. The Holton--Mass model exhibits two metastable regimes, a strong and a weak polar vortex, maintained by nonlinear wave--mean flow interactions, with weak stochastic forcing intermittently triggering rare transitions between these regimes that qualitatively represent SSW events. We employ a ResNet-inspired Conditional Variational Autoencoder with six-layer encoder and decoder layers and explicit current-state conditioning to model the distribution of the system's state at the next time step (one day). The emulator accurately reproduces short-term dynamics, steady-state probability distributions, regime persistence statistics, rare transition rates, the transition committor function, and the transition expected lead time of the physical model. Beyond emulation fidelity, we interrogate the learned latent representation to understand how the model internalizes the underlying metastable structure of the dynamics. Principal Component Analysis of the 32-dimensional latent space reveals a clear and unsupervised separation into four physically interpretable clusters corresponding to strong versus weak vortex regimes and stable versus transition-prone configurations. Such emergent regime separation in latent space is hard to identify for deep generative models applied to high-dimensional stochastic systems. Our results show that carefully designed probabilistic emulators can uncover physically meaningful manifolds governing extreme-event dynamics, potentially aiding the development of improved operational advanced warning systems.</span> <span class="abstract-toggle" data-id="2610.02069">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.02069v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.02069v1) · [:material-content-copy: BibTeX](../../bibtex/2610.02069.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=cnn-u-net" data-tag="cnn-u-net">CNN / U-Net</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=interpretability" data-tag="interpretability">Interpretability</a>
    { .paper-tags }

-   #### Unsupervised Domain Adaptation for Enhanced Radiometer Image Precipitation Estimation using Conditional Flow Matching { #2610.01890 }

    *Victor Enescu, Assaad Zeghina, Matthieu Meignin, Nicolas Viltard, Cécile Mallet* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.01890">Deep generative networks have recently achieved unprecedented performance in precise image and video editing using sophisticated textual prompts. However, the effectiveness of such models heavily...</span><span class="abstract-full" id="full-2610.01890" hidden>Deep generative networks have recently achieved unprecedented performance in precise image and video editing using sophisticated textual prompts. However, the effectiveness of such models heavily depends on access to very large supervised and annotated image datasets, which can be very difficult to obtain. This is particularly true for satellite instruments, which very rarely overlap with labelled data, and suffer from domain shifts in the rare occasions they do. In this paper, we investigate the potential of flow matching models for unsupervised domain adaptation of satellite radiometer images. Our main contribution is a novel unsupervised method that achieves precise domain alignment by leveraging parts of the deterministic ordinary differential equations in flow matching models, conditioned on different satellite instruments. A key strength of our approach is its ability to preserve essential information while adapting across any domains since the perturbations are in theory bijective. Extensive experiments conducted on the GPM-Core constellation show the benefit of our conditional domain adaptation, particularly in improving rain precipitation estimation from radiometer imagery.</span> <span class="abstract-toggle" data-id="2610.01890">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.01890v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.01890v1) · [:material-content-copy: BibTeX](../../bibtex/2610.01890.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=diffusion-flow-matching" data-tag="diffusion-flow-matching">Diffusion & flow matching</a> <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a>
    { .paper-tags }

-   #### Explaining El Niño Forecasts with the Average Gradient Outer Product { #2610.01095 }

    *Yuan Hui, Dorian S. Abbot, Robert J. Webber* · Oct 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2610.01095">An important and unresolved problem in the physical sciences is explaining the predictions made by neural networks. Several explainable artificial intelligence (XAI) methods have been proposed to...</span><span class="abstract-full" id="full-2610.01095" hidden>An important and unresolved problem in the physical sciences is explaining the predictions made by neural networks. Several explainable artificial intelligence (XAI) methods have been proposed to address this problem, including gradient XAI, Integrated Gradients, and GradientSHAP. We evaluate the baseline XAI methods according to four scores: sensitivity (XAI patterns strongly affect predictions), attribution (XAI patterns reproduce the change in prediction relative to a baseline), robustness (XAI patterns remain stable for nearby inputs), and coherence (XAI patterns are spatially smooth). We also introduce a new method, average gradient outer product (AGOP) XAI, that uses global gradient information to identify an important direction for a specific input. We apply XAI to neural network predictions of the El Niño-Southern Oscillation (ENSO) based on data from the Zebiak-Cane model.   AGOP XAI achieves the highest attribution, robustness, and coherence scores in the architecture and lead-time comparisons reported here. Its sensitivity is surpassed by gradient XAI, which is maximally sensitive by definition. Beyond diagnosing neural-network behavior, AGOP XAI can generate candidate hypotheses about physical mechanisms. The method highlights an equatorial thermocline-depth signal consistent with recharge oscillator physics, together with a southeastern-Pacific lobe that may be specific to the Zebiak-Cane model. Finally, we test the physical relevance of AGOP using optimized perturbations that move the Zebiak-Cane model along AGOP explanation coordinates. Such perturbations can suppress the selected extreme events or, from a near-neutral ensemble, generate strong El Niño or La Niña events 10 months later.</span> <span class="abstract-toggle" data-id="2610.01095">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2610.01095v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2610.01095v1) · [:fontawesome-brands-github: Code](https://github.com/rjwebber/agop-xai) · [:material-content-copy: BibTeX](../../bibtex/2610.01095.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=subseasonal-to-seasonal" data-tag="subseasonal-to-seasonal">Subseasonal to seasonal</a> <a class="md-tag" href="/explore/?t=interpretability" data-tag="interpretability">Interpretability</a>
    { .paper-tags }

-   #### A library for differentiable signal processing and machine learning on the sphere { #2609.39737 }

    *Thorsten Kurth, Max Rietmann, Mauro Bisson, Andrea Paris, Alberto Carpentieri, Jean Kossaifi et al.* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.39737">The two-dimensional sphere embedded in three-dimensional Euclidean space S2, plays a central role in a variety of scientific and engineering domains, including geophysics, planetary science, geodesy,...</span><span class="abstract-full" id="full-2609.39737" hidden>The two-dimensional sphere embedded in three-dimensional Euclidean space S2, plays a central role in a variety of scientific and engineering domains, including geophysics, planetary science, geodesy, atmospheric physics, quantum chemistry, cosmology, and virtual reality, among many others. As machine learning increasingly permeates these fields, the demand grows for robust tools that process and model functions on the sphere, while respecting the inherent topological and symmetry properties of the domain. We present torch-harmonics, a comprehensive library that offers efficient, differentiable implementations of advanced signal processing and machine learning (ML) methods for spherical data. These include the spherical harmonic transform (SHT), the spherical analogue of the Fourier transform, vector spherical harmonics, discrete-continuous and spectral convolutions, as well as both global and neighborhood spherical attention mechanisms. Beyond traditional representations, torch-harmonics provides the building blocks for state-of-the-art spherical ML architectures such as spherical transformers in order to enable scalable, rotationally-aware learning and inference in modern scientific and engineering applications.</span> <span class="abstract-toggle" data-id="2609.39737">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.39737v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.39737v1) · [:material-content-copy: BibTeX](../../bibtex/2609.39737.bib){ .bibtex-link }
    { .paper-links }

-   #### Methodological Changes to the Attention ResUNet Hourly Precipitation Postprocessor { #2609.38609 }

    *Thomas M. Hamill* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.38609">This note is a technical companion to a previously published preprint describing an Attention Residual U-Net that postprocesses deterministic forecasts from The Weather Company's Global and Regional...</span><span class="abstract-full" id="full-2609.38609" hidden>This note is a technical companion to a previously published preprint describing an Attention Residual U-Net that postprocesses deterministic forecasts from The Weather Company's Global and Regional Atmospheric Forecast (GRAF) model into probabilistic hourly precipitation forecasts. It documents what has changed in that method since publication. Feature-wise Linear Modulation conditioning on calendar season and forecast lead time is used to produce a single trained model for each season, replacing 192 separately trained per-month, per-lead checkpoints. Lead time is extended from 48 to 72 h. Two new input channels are used, per-pixel local solar hour and a static, monthly-varying precipitation climatology. During verification, the climatological reference against which the Brier Skill Score is computed now has an added diurnal dimension, on top of the monthly resolution it already had. Brier Skill Score and reliability are compared between the new vs. the previous training. Forecasts generated with the new training show a modest, consistent improvement of the current training over the original.</span> <span class="abstract-toggle" data-id="2609.38609">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.38609v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.38609v1) · [:material-content-copy: BibTeX](../../bibtex/2609.38609.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=precipitation" data-tag="precipitation">Precipitation</a> <a class="md-tag" href="/explore/?t=hourly" data-tag="hourly">Hourly</a> <a class="md-tag" href="/explore/?t=monthly" data-tag="monthly">Monthly</a>
    { .paper-tags }

-   #### Predicting Delayed Train Trajectories on the Dutch Railway Network: Explainable AI Evaluation of Topological, Operational and Weather Features with Tree Based Ensemble Methods { #2609.34692 }

    *Jia Long Bao, Ali Mohammed Mansoor Alsahag, Seyed Sahand Mohammadi Ziabari* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.34692">The reliable prediction of passenger train delays is a critical component of railway management. While contemporary research frequently attempts to maximize absolute accuracy by deploying opaque deep...</span><span class="abstract-full" id="full-2609.34692" hidden>The reliable prediction of passenger train delays is a critical component of railway management. While contemporary research frequently attempts to maximize absolute accuracy by deploying opaque deep learning architectures, the underlying data mechanics driving longitudinal predictive decay remain underexplored. Consequently, this study provides an explainable temporal robustness analysis of network-wide railway delay prediction. Focusing on the Dutch railway network, this research utilizes interpretable tree-based ensembles to integrate granular topological, environmental, and operational features. The overarching finding establishes that while feature-rich tree-based models improve simultaneous (within-month) prediction, predictive performance systematically degrades when evaluated across non-simultaneous (future) months. Furthermore, multi-horizon SHAP and dispersion analyses explicitly link this degradation to environmental feature volatility and instability within the statistical target definition. Ultimately, this thesis demonstrates that richer feature sets alone are insufficient to resolve long-term forecasting constraints, underscoring the necessity to transition toward dynamic, season-aware architectures anchored by absolute operational boundaries.</span> <span class="abstract-toggle" data-id="2609.34692">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.34692v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.34692v1) · [:material-content-copy: BibTeX](../../bibtex/2609.34692.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a> <a class="md-tag" href="/explore/?t=interpretability" data-tag="interpretability">Interpretability</a>
    { .paper-tags }

-   #### StatD2GAN: When Calibration Masks Generator Quality in Held-Out Evaluation of Synthetic Weather Sequences { #2609.33761 }

    *Mustafa Ozaytac, Ozge Karadag Atas* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.33761">Generative models for multivariate weather series are routinely evaluated with pooled distributional metrics computed after marginal calibration. We show this practice can invalidate architectural...</span><span class="abstract-full" id="full-2609.33761" hidden>Generative models for multivariate weather series are routinely evaluated with pooled distributional metrics computed after marginal calibration. We show this practice can invalidate architectural conclusions, and rebuild the evaluation of StatD2GAN, a three-discriminator GAN with evolutionary weight adaptation, around a held-out protocol: the final two calendar years of each dataset are held out behind a 168 hour embargo, calibration is fitted on the training block only, and all metrics are computed on the held-out block. Evidence comes from 25 matched (location, seed) pairs across five Koppen-Geiger climates, tested with Wilcoxon signed-rank tests under Holm correction. Four results follow. First, isotonic calibration drives the Kolmogorov-Smirnov distance to within 2% of a per-location noise-and-shift floor for every architecture tested, including a deliberately weak RCGAN baseline, so calibrated marginal metrics cannot discriminate between architectures. Second, the sorted-representation discriminator is the only component whose removal significantly degrades cross-variable dependence (Kendall tau MAE +0.080, Holm p = 0.009), with a regime-dependent effect: near zero in Ankara, above 115% in Dubai and Yakutsk. A rank-transformed variant isolates the mechanism as quantile supervision of the marginals rather than copula matching. Third, physical constraint violations are injected by calibration, not the generator; projection removes them at negligible cost (deltaKS <= 0.003). Fourth, pooled metrics conceal a collapse of between-sequence weekly-mean variability, a proxy for seasonal and regime diversity, in TimeGAN that only sequence-level statistics expose. We recommend floor-referenced marginal evaluation, matched-pair testing, and sequence-level variance decomposition as minimum requirements for calibrated generative pipelines.</span> <span class="abstract-toggle" data-id="2609.33761">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.33761v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.33761v1) · [:material-content-copy: BibTeX](../../bibtex/2609.33761.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=gans" data-tag="gans">GANs</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### Mechanism-Aware Ensemble Conditioning for Data-Limited Emulation of Extreme Events { #2609.30746 }

    *Isabella S. Thiel, Juan Bello-Rivas, Yannis G. Kevrekidis, Themistoklis P. Sapsis* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.30746">Extreme events in chaotic systems are difficult to learn from short trajectories because they are controlled by transient finite-time instability rather than by frequently observed bulk dynamics. We...</span><span class="abstract-full" id="full-2609.30746" hidden>Extreme events in chaotic systems are difficult to learn from short trajectories because they are controlled by transient finite-time instability rather than by frequently observed bulk dynamics. We propose a mechanism-aware conditioning plug-in framework that turns a nudged coarse ensemble into a non-intrusive sensor of local instability geometry. In the small-noise regime, the ensemble covariance aggregates the same finite-time deformation kernels that govern local instability, providing a Jacobian-free proxy for the local amplification structure around a synchronized coarse trajectory. A small FiLM module injects statistics of this ensemble geometry into an otherwise unchanged backbone while leaving the coarse simulator unchanged. We demonstrate this interface in two distinct pipelines: a Transformer-style residual-attention corrector for a controlled low-dimensional chaotic system and a probabilistic recurrent STORN corrector for topographic two-layer quasi-geostrophic (QG) flow. In the low-dimensional benchmark, ensemble covariance directions co-activate with OTD modes and FiLM conditioning improves 99th-percentile exceedance-frequency errors over an identical no-context Transformer baseline. In QG, a fixed ensemble-conditioned FiLM-STORN model trained on only \(50\) time units substantially improves long-horizon rare-event statistics in the data-limited regime, including density-tail errors, exceedance frequencies, and spatial exceedance-area distributions relative to an unconditioned STORN trained on the same data; on averaged high-threshold exceedance diagnostics, it also outperforms the baseline STORN trained with $20$ times more high-resolution data. These results show that local instability geometry is not merely interpretable post hoc, but an actionable conditioning signal for data-efficient rare-event emulation.</span> <span class="abstract-toggle" data-id="2609.30746">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.30746v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.30746v1) · [:material-content-copy: BibTeX](../../bibtex/2609.30746.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=transformers" data-tag="transformers">Transformers</a> <a class="md-tag" href="/explore/?t=extremes" data-tag="extremes">Extremes</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### PISCES: Physics-Informed Solar-wind Convolutional autoEncoder for Space-weather Anomaly Detection and Early Warning { #2609.28022 }

    *Kevin Lee, Alison J. March* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.28022">Space weather early warning depends on detecting solar wind transients in in-situ measurements at the first Sun-Earth Lagrange point (L1), before they reach Earth. Fixed thresholds can miss combined...</span><span class="abstract-full" id="full-2609.28022" hidden>Space weather early warning depends on detecting solar wind transients in in-situ measurements at the first Sun-Earth Lagrange point (L1), before they reach Earth. Fixed thresholds can miss combined magnetic and plasma structure, and many learning methods provide a single anomaly score. We present the Physics-Informed Solar-wind Convolutional autoEncoder for Space-weather (PISCES), a convolutional autoencoder trained without catalog labels on OMNI solar wind measurements under physics constraints. Its loss includes magnetic field consistency, an empirical relation between temperature and velocity, the Parker spiral angle, and penalties on changes between consecutive one-minute samples in derived quantities calculated from the reconstruction. At inference, PISCES separates the anomaly score into magnetic and plasma reconstruction errors, physics relations, and residual corrections, and reports the magnitude of each contribution. Attenuation of the skip connections, selected on validation data, improves average precision for the trained models, while the untrained scores remain nearly the same. The trained models also give a more consistent ordering of these physical contributions. After smoothing with a trailing median, the alarms can precede independently observed sudden commencements, including positive sudden impulses.</span> <span class="abstract-toggle" data-id="2609.28022">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.28022v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.28022v1) · [:fontawesome-brands-github: Code](https://github.com/magnaprog/PISCES) · [:material-content-copy: BibTeX](../../bibtex/2609.28022.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=physics-ml-hybrid" data-tag="physics-ml-hybrid">Physics–ML hybrid</a>
    { .paper-tags }

-   #### Sparse-Observation Atmospheric Thermal Forecasting with Physics-Informed Neural Networks for Climate-Aware Digital Twins { #2609.27290 }

    *Tannaz Goodarzvand Chegini, Elyas Shivanian, Behzad Karimi, Faraz Dadgostari* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.27290">Short-horizon forecasts of atmospheric temperature are needed to support climate-aware digital-twin systems, but such forecasts must be produced where thermal observations are incomplete. This study...</span><span class="abstract-full" id="full-2609.27290" hidden>Short-horizon forecasts of atmospheric temperature are needed to support climate-aware digital-twin systems, but such forecasts must be produced where thermal observations are incomplete. This study evaluates a physics-informed neural network for potential-temperature forecasting, constrained by a pressure-coordinate thermodynamic advection-source equation and a diabatic-source closure fit from the preceding 12-hour period and frozen before future-time training. Using hourly ERA5 reanalysis at three pressure levels, the model is evaluated as a conditional hindcast at lead times of one, two and three hours against persistence, local-trend, and two matched neural-network baselines, one of which receives the same future meteorological forcing as the PINN, helping distinguish the physical constraint from access to future forcing. In an Oklahoma development case, mean RMSE improvement over the strongest baseline grew from 8.1% at one hour to 23.8% at three hours; under an observation-density sweep down to 5% of candidate locations, this 3-hour advantage remained 14.6--16.9%, with no evidence that lower density improves performance. Under a fixed protocol transferred to an Alabama heat event with three virtual-observation layouts, three-hour improvement ranged 19.7-24.4% with consistent origin-level wins. A parallel Montana stress test, in which fixed pressure levels intersected complex terrain, produced a three-hour degradation of roughly 17.5%, identifying a terrain-related applicability limit of the formulation. Together, these results indicate that the physics constraint's benefit grows with forecast horizon, persists under severe observation sparsity, and transfers across regions, but is bounded by the validity of a fixed vertical-coordinate representation over complex terrain, evidence relevant to physics-constrained components of climate-aware forecasting and digital-twin systems.</span> <span class="abstract-toggle" data-id="2609.27290">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.27290v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.27290v1) · [:material-content-copy: BibTeX](../../bibtex/2609.27290.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=physics-ml-hybrid" data-tag="physics-ml-hybrid">Physics–ML hybrid</a> <a class="md-tag" href="/explore/?t=hourly" data-tag="hourly">Hourly</a>
    { .paper-tags }

-   #### Inference of Unknown Dynamical Components Using Next Generation Reservoir Computing: From Chaotic Systems to Climate Data { #2609.24754 }

    *Jule Budnick, Andrew Keane, Serhiy Yanchuk* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.24754">We investigate next generation reservoir computing (NGRC) as a data-driven approach for inferring unseen components of dynamical systems. We compare NGRC with traditional reservoir computing (RC)...</span><span class="abstract-full" id="full-2609.24754" hidden>We investigate next generation reservoir computing (NGRC) as a data-driven approach for inferring unseen components of dynamical systems. We compare NGRC with traditional reservoir computing (RC) using the Lorenz and Rössler system, where two unknown components are inferred from one given component. For both systems, NGRC achieves accurate results while requiring fewer training data and less computational time than RC. We identified an inverse proportional behavior between the number of time-delayed steps needed for NGRC and the temporal resolution, indicating that the physical time span covered by the delay interval is an important factor in determining the required number of delayed steps. Finally, we apply NGRC to the observational climate data of ENSO (El Niño--Southern Oscillation) and infer one observable from the remaining variables. Despite the noise and complexity of the real-world data, the NGRC shows promising results. Our findings demonstrate the potential of NGRC for efficient inference of unseen components in both controlled dynamical systems and real-world data.</span> <span class="abstract-toggle" data-id="2609.24754">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.24754v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.24754v1) · [:material-content-copy: BibTeX](../../bibtex/2609.24754.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=subseasonal-to-seasonal" data-tag="subseasonal-to-seasonal">Subseasonal to seasonal</a>
    { .paper-tags }

-   #### Predictability-Guided Multiscale Probabilistic Forecasting of Wind Direction under Extreme Shear { #2609.16707 }

    *Hailong Shu* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.16707">Accurate multi-horizon wind direction forecasting is critical for turbine yaw control and grid security. Rapid directional shear (turning $\ge 90^\circ$) challenges models via non-Euclidean geometry...</span><span class="abstract-full" id="full-2609.16707" hidden>Accurate multi-horizon wind direction forecasting is critical for turbine yaw control and grid security. Rapid directional shear (turning $\ge 90^\circ$) challenges models via non-Euclidean geometry on $S^1$, multiscale dynamics, and regime-dependent uncertainty. Conventional discrete models and foundation models suffer from mid-frequency phase lag and turning misalignments. We show that directional predictability decays at disparate rates across frequency subbands, rendering monolithic mechanisms suboptimal. We propose a predictability-guided paradigm: slow synoptic drift $\to$ deterministic regression; intermediate turning $\to$ continuous latent differential flows; unresolved turbulence $\to$ conditional residual diffusion; followed by causal recalibration. On a 10,000-sequence multi-year benchmark, our framework maintains calm-weather accuracy (Test MCE $38.48^\circ$) while reducing extreme-turning error (Case 1 MCE $60.69^\circ$ vs $70.42^\circ$ for zero-shot foundation models). The circular CRPS reaches $22.36^\circ$, with 93.88% coverage at nominal 95% (91.01% out-of-distribution). Density estimation further reveals near-antipodal bimodal structure under severe shear (13.39%--15.43% tail mass $\ge 135^\circ$), exposing a geometric bound where single-center calibration under-covers (81.56%), motivating multimodal circular manifold learning.</span> <span class="abstract-toggle" data-id="2609.16707">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.16707v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.16707v1) · [:material-content-copy: BibTeX](../../bibtex/2609.16707.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=foundation-models" data-tag="foundation-models">Foundation models</a> <a class="md-tag" href="/explore/?t=uncertainty-ensembles" data-tag="uncertainty-ensembles">Uncertainty & ensembles</a>
    { .paper-tags }

-   #### 4D Parallelism Unlocks Exascale Bayesian Neural Networks for High-Fidelity Atmospheric Modeling { #2609.12815 }

    *Deifilia Kieckhefen, Juan Pedro Gutiérrez Hermosillo Muriedas, Lars Helge Heyen, Mathis Bode et al.* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.12815">We present BEAST, the first-ever Bayesian Swin Transformer for atmospheric forecasting on 0.25$^\circ$ global resolution able to accurately quantify both aleatoric and epistemic uncertainty. To...</span><span class="abstract-full" id="full-2609.12815" hidden>We present BEAST, the first-ever Bayesian Swin Transformer for atmospheric forecasting on 0.25$^\circ$ global resolution able to accurately quantify both aleatoric and epistemic uncertainty. To overcome the associated computational bottlenecks, we devise an orthogonal 4D-parallelization scheme that introduces a unique domain-tensor-parallelism strategy and a novel uncertainty parallel method, enabling us to fully leverage GPU capacity and efficiently scale model training. For a 2.4-billion-parameter model, we achieve a peak performance of 3.96 EFLOP/s on 20,480 NVIDIA GH200 GPUs on the JUPITER supercomputer. We train BEAST as a 700-million-parameter model with 96 random weight samples on 384 nodes on 40 years of data for nearly one million gradient updates. This model achieves predictive skill scores competitive with state-of-the-art probabilistic atmospheric AI models and numerical models, and can predict extreme events with exceptional skill, while generating large ensembles 3 to 4 times faster than the current-best AI model. Our contribution unlocks the potential of high-fidelity uncertainty quantification in atmospheric AI models, heralding a new era for AI-based models in climate and Earth system sciences.</span> <span class="abstract-toggle" data-id="2609.12815">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.12815v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.12815v1) · [:material-content-copy: BibTeX](../../bibtex/2609.12815.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=transformers" data-tag="transformers">Transformers</a> <a class="md-tag" href="/explore/?t=quarter-degree" data-tag="quarter-degree">0.25°</a>
    { .paper-tags }

-   #### Climate-ModernBERT: Revisiting Corpus Composition for Domain-Adaptive Continued Pretraining { #2609.07798 }

    *Yongan Yu, Shantam Raj, Jingwei Ni, Ario Saeid Vaghefi, Dominik Stammbach, Markus Leippold* · Sep 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2609.07798">Natural Language Processing (NLP) in the climate domain requires models to process heterogeneous text sources, including scientific literature, policy disclosures, and synthetic reports. However, how...</span><span class="abstract-full" id="full-2609.07798" hidden>Natural Language Processing (NLP) in the climate domain requires models to process heterogeneous text sources, including scientific literature, policy disclosures, and synthetic reports. However, how to effectively combine diverse domain corpora during continued pretraining (CPT) remains underexplored. We introduce Climate-ModernBERT, a family of climate-adapted encoder models obtained through continued pretraining of ModernBERT-Base on three climate corpora: academic climate text, climate-filtered web data, and synthetic climate documents. We systematically compare joint continued pretraining on corpus mixtures with parameter-space merging of independently specialized checkpoints. Across nine climate NLP benchmarks, our best model achieves 76.3 average F_1, improving significantly over a vanilla ModernBERT baseline by 2.8 points. Within the climate NLP setting, the results show that academic climate corpora provide the strongest adaptation signal among the evaluated sources, while parameter-space merging improves over joint multi-source training and better preserves complementary information from heterogeneous climate corpora. We release all Climate-ModernBERT variants and training checkpoints to support future research in climate NLP and domain-adaptive pretraining.</span> <span class="abstract-toggle" data-id="2609.07798">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2609.07798v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2609.07798v1) · [:material-content-copy: BibTeX](../../bibtex/2609.07798.bib){ .bibtex-link }
    { .paper-links }

-   #### When Does Forecast-Error Energy Grow Logistically in Geophysical Turbulence? { #2608.26492 }

    *Malaquias Peña* · Aug 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2608.26492">Coarse-graining can yield a simple macroscopic growth curve in a bounded chaotic system even when constituent scales follow different clocks. The distinction matters as reduced-order and generative...</span><span class="abstract-full" id="full-2608.26492" hidden>Coarse-graining can yield a simple macroscopic growth curve in a bounded chaotic system even when constituent scales follow different clocks. The distinction matters as reduced-order and generative models compress multiscale forecast uncertainty into learned coordinates. We ask when forecast-error energy admits a logistic law. From the exact twin-error budget and correlated and decorrelated spectra, we derive two scalar limits: an invariant decorrelation amplitude, logistic only when contributing scales share one shape and one clock, and a self-similar upscale error front whose law depends on spectral slope and front speed. With local-strain scaling, the front predicts exponential error-energy growth for the canonical barotropic-vorticity spectrum and linear growth for the surface-quasigeostrophic spectrum. Stationary forced surface-quasigeostrophic twins test the logistic admission conditions. A response-blind partition of 16 trajectories gives cluster-mean logistic root-mean-square deviations 0.080 and 0.093, although every trajectory has resolved clock heterogeneity. An exact averaging identity shows how signed shape and clock corrections cancel, producing a nearly logistic aggregate while constituent scales retain distinct clocks. Mechanism identification therefore requires more than goodness of fit: independent shape, clock, and residual tests are required. These admission conditions provide physics-based guardrails for compact representations of chaotic systems and generative forecast ensembles.</span> <span class="abstract-toggle" data-id="2608.26492">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2608.26492v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2608.26492v1) · [:material-content-copy: BibTeX](../../bibtex/2608.26492.bib){ .bibtex-link }
    { .paper-links }

-   #### Frequency-aware forecasting for short-term typhoon gust prediction { #2608.25604 }

    *Xuefei Wang, Tingyi Liu, Heng Zhang, Shengjun Zhang* · Aug 2026
    { .paper-meta }

    <span class="abstract-snippet" id="snip-2608.25604">Accurate gust forecasting under typhoon conditions remains challenging due to the highly non-stationary and multi-scale characteristics of extreme wind fluctuations. Existing deep learning models...</span><span class="abstract-full" id="full-2608.25604" hidden>Accurate gust forecasting under typhoon conditions remains challenging due to the highly non-stationary and multi-scale characteristics of extreme wind fluctuations. Existing deep learning models often struggle to simultaneously capture long-term trends and rapid local variations, resulting in degraded performance during extreme events. We propose WDANet, a frequency-aware forecasting framework that integrates stationary wavelet decomposition, a Feature-wise Linear Modulation (FiLM) strategy, and a dual-branch encoder-decoder architecture, enabling separate modeling of trend and fluctuation components. Taking the offshore regions of the Western Pacific in China as an example, we conduct fine-grid wind gust prediction research. The results demonstrate that WDANet shows advantages for short lead times under the experimental setting across a 24-h forecasting horizon and achieves higher prediction accuracy than ECMWF-HRES within the first 6 h. During extreme wind events, WDANet more accurately captures gust peaks and attains the best RMSE and MAE performance. These results highlight its potential for offshore wind power operation, disaster warning, and risk mitigation.</span> <span class="abstract-toggle" data-id="2608.25604">more</span>

    [:material-file-document-outline: arXiv](https://arxiv.org/abs/2608.25604v1) · [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2608.25604v1) · [:material-content-copy: BibTeX](../../bibtex/2608.25604.bib){ .bibtex-link }
    { .paper-links }

    <a class="md-tag" href="/explore/?t=tropical-cyclones" data-tag="tropical-cyclones">Tropical cyclones</a> <a class="md-tag" href="/explore/?t=extremes" data-tag="extremes">Extremes</a> <a class="md-tag" href="/explore/?t=energy" data-tag="energy">Energy</a>
    { .paper-tags }

</div>

<nav class="pager" markdown="span">**1** [2](2.md) [3](3.md) [4](4.md) [5](5.md) [6](6.md) [7](7.md) [8](8.md) [Older :material-arrow-right:](2.md){ .pager-step }</nav>

