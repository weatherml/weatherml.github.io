---
hide:
  - navigation
---

## Global Models (23)

<div class="grid cards" markdown>

-   #### El Nino Prediction Based on Weather Forecast and Geographical Time-series Data

    ---

    *Viet Trinh, Ha-Vy Luu, Quoc-Khiem Nguyen-Pham, Hung Tong, Thanh-Huyen Tran, Hoai-Nam Nguyen Dang* · 2026

    <span class="abstract-snippet" id="snip-2604.04998">This paper proposes a novel framework for enhancing the prediction accuracy and lead time of El Niño events, crucial for mitigating their global climatic, economic, and societal impacts. Traditional...</span><span class="abstract-full" id="full-2604.04998" hidden>This paper proposes a novel framework for enhancing the prediction accuracy and lead time of El Niño events, crucial for mitigating their global climatic, economic, and societal impacts. Traditional prediction models often rely on oceanic and atmospheric indices, which may lack the granularity or dynamic interplay captured by comprehensive meteorological and geographical datasets. Our framework integrates real-time global weather forecast data with anomalies, subsurface ocean heat content, and atmospheric pressure across various temporal and spatial resolutions. Leveraging a hybrid deep learning architecture that combines a Convolutional Neural Network (CNN) for spatial feature extraction and a Long Short-Term Memory (LSTM) network for temporal dependency modeling, the framework aims to identify complex precursors and evolving patterns of El Niño events.</span> <span class="abstract-toggle" data-id="2604.04998">more</span>

    [:material-file-document: 2604.04998](https://arxiv.org/abs/2604.04998v1) · [:material-content-copy: BibTeX](bibtex/2604.04998.bib){ .bibtex-link }

    <span class="md-tag">CNN</span> <span class="md-tag">recurrent</span>

-   #### Super-Resolving Coarse-Resolution Weather Forecasts With Flow Matching

    ---

    *Aymeric Delefosse, Anastase Charantonis, Dominique Béréziat* · 2026

    <span class="abstract-snippet" id="snip-2604.00897">Machine learning-based weather forecasting models now surpass state-of-the-art numerical weather prediction systems, but training and operating these models at high spatial resolution remains...</span><span class="abstract-full" id="full-2604.00897" hidden>Machine learning-based weather forecasting models now surpass state-of-the-art numerical weather prediction systems, but training and operating these models at high spatial resolution remains computationally expensive. We present a modular framework that decouples forecasting from spatial resolution by applying learned generative super-resolution as a post-processing step to coarse-resolution forecast trajectories. We formulate super-resolution as a stochastic inverse problem, using a residual formulation to preserve large-scale structure while reconstructing unresolved variability. The model is trained with flow matching exclusively on reanalysis data and is applied to global medium-range forecasts. We evaluate (i) design consistency by re-coarsening super-resolved forecasts and comparing them to the original coarse trajectories, and (ii) high-resolution forecast quality using standard ensemble verification metrics and spectral diagnostics. Results show that super-resolution preserves large-scale structure and variance after re-coarsening, introduces physically consistent small-scale variability, and achieves competitive probabilistic forecast skill at 0.25° resolution relative to an operational ensemble baseline, while requiring only a modest additional training cost compared with end-to-end high-resolution forecasting.</span> <span class="abstract-toggle" data-id="2604.00897">more</span>

    [:material-file-document: 2604.00897](https://arxiv.org/abs/2604.00897v1) · [:material-content-copy: BibTeX](bibtex/2604.00897.bib){ .bibtex-link }

    <span class="md-tag">diffusion</span> <span class="md-tag">probabilistic</span>

-   #### Improving Ensemble Forecasts of Abnormally Deflecting Tropical Cyclones with Fused Atmosphere-Ocean-Terrain Data

    ---

    *Qixiang Li, Yuan Zhou, Shuwei Huo, Chong Wang, Xiaofeng Li* · 2026

    <span class="abstract-snippet" id="snip-2603.29200">Deep learning-based tropical cyclone (TC) forecasting methods have demonstrated significant potential and application advantages, as they feature much lower computational cost and faster operation...</span><span class="abstract-full" id="full-2603.29200" hidden>Deep learning-based tropical cyclone (TC) forecasting methods have demonstrated significant potential and application advantages, as they feature much lower computational cost and faster operation speed than numerical weather prediction models. However, existing deep learning methods still have key limitations: they can only process a single type of sequential trajectory data or homogeneous meteorological variables, and fail to achieve accurate forecasting of abnormal deflected TCs. To address these challenges, we present two groundbreaking contributions. First, we have constructed a multimodal and multi-source dataset named AOT-TCs for TC forecasting in the Northwest Pacific basin. As the first dataset of its kind, it innovatively integrates heterogeneous variables from the atmosphere, ocean, and land, thus obtaining a comprehensive and information-rich meteorological dataset. Second, based on the AOT-TCs dataset, we propose a forecasting model that can handle both normal and abnormally deflected TCs. This is the first TC forecasting model to adopt an explicit atmosphere-ocean-terrain coupling architecture, enabling it to effectively capture complex interactions across physical domains. Extensive experiments on all TC cases in the Northwest Pacific from 2017 to 2024 show that our model achieves state-of-the-art performance in TC forecasting: it not only significantly improves the forecasting accuracy of normal TCs but also breaks through the technical bottleneck in forecasting abnormally deflected TCs.</span> <span class="abstract-toggle" data-id="2603.29200">more</span>

    [:material-file-document: 2603.29200](https://arxiv.org/abs/2603.29200v2) · [:material-content-copy: BibTeX](bibtex/2603.29200.bib){ .bibtex-link }

-   #### Skillful Kilometer-Scale Regional Weather Forecasting via Global and Regional Coupling

    ---

    *Weiqi Chen, Wenwei Wang, Qilong Yuan, Lefei Shen, Bingqing Peng, Jiawei Chen, Bo Wu, Liang Sun* · 2026

    <span class="abstract-snippet" id="snip-2603.28173">Data-driven weather models have advanced global medium-range forecasting, yet high-resolution regional prediction remains challenging due to unresolved multiscale interactions between large-scale...</span><span class="abstract-full" id="full-2603.28173" hidden>Data-driven weather models have advanced global medium-range forecasting, yet high-resolution regional prediction remains challenging due to unresolved multiscale interactions between large-scale dynamics and small-scale processes such as terrain-induced circulations and coastal effects. This paper presents a global-regional coupling framework for kilometer-scale regional weather forecasting that synergistically couples a pretrained Transformer-based global model with a high-resolution regional network via a novel bidirectional coupling module, ScaleMixer. ScaleMixer dynamically identifies meteorologically critical regions through adaptive key-position sampling and enables cross-scale feature interaction through dedicated attention mechanisms. The framework produces forecasts at $0.05^\circ$ ($\sim 5 \mathrm{km}$ ) and 1-hour resolution over China, significantly outperforming operational NWP and AI baselines on both gridded reanalysis data and real-time weather station observations. It exhibits exceptional skill in capturing fine-grained phenomena such as orographic wind patterns and Foehn warming, demonstrating effective global-scale coherence with high-resolution fidelity. The code is available at https://anonymous.4open.science/r/ScaleMixer-6B66.</span> <span class="abstract-toggle" data-id="2603.28173">more</span>

    [:material-file-document: 2603.28173](https://arxiv.org/abs/2603.28173v1) · [:material-content-copy: BibTeX](bibtex/2603.28173.bib){ .bibtex-link }

    <span class="md-tag">transformer</span>

-   #### FuXiWeather2: Learning accurate atmospheric state estimation for operational global weather forecasting

    ---

    *Xiaoze Xu, Xiuyu Sun, Songling Zhu, Xiaohui Zhong, Yuanqing Huang, Zijian Zhu, Jun Liu, Hao Li* · 2026

    <span class="abstract-snippet" id="snip-2603.15358">Numerical weather prediction has long been constrained by the computational bottlenecks inherent in data assimilation and numerical modeling. While machine learning has accelerated forecasting,...</span><span class="abstract-full" id="full-2603.15358" hidden>Numerical weather prediction has long been constrained by the computational bottlenecks inherent in data assimilation and numerical modeling. While machine learning has accelerated forecasting, existing models largely serve as "emulators of reanalysis products," thereby retaining their systematic biases and operational latencies. Here, we present FuXiWeather2, a unified end-to-end neural framework for assimilation and forecasting. We align training objectives directly with a combination of real-world observations and reanalysis data, enabling the framework to effectively rectify inherent errors within reanalysis products. To address the distribution shift between NWP-derived background inputs during training and self-generated backgrounds during deployment, we introduce a recursive unrolling training method to enhance the precision and stability of analysis generation. Furthermore, our model is trained on a hybrid dataset of raw and simulated observations to mitigate the impact of observational distribution inconsistency. FuXiWeather2 generates high-resolution ($0.25^{\circ}$) global analysis fields and 10-day forecasts within minutes. The analysis fields surpass the NCEP-GFS across most variables and demonstrate superior accuracy over both ERA5 and the ECMWF-HRES system in lower-tropospheric and surface variables. These high-quality analysis fields drive deterministic forecasts that exceed the skill of the HRES system in 91\% of evaluated metrics. Additionally, its outstanding performance in typhoon track prediction underscores its practical value for rapid response to extreme weather events. The FuXiWeather2 analysis dataset is available at https://doi.org/10.5281/zenodo.18872728.</span> <span class="abstract-toggle" data-id="2603.15358">more</span>

    [:material-file-document: 2603.15358](https://arxiv.org/abs/2603.15358v1) · [:material-content-copy: BibTeX](bibtex/2603.15358.bib){ .bibtex-link }

-   #### AGCD: Agent-Guided Cross-Modal Decoding for Weather Forecasting

    ---

    *Jing Wu, Yang Liu, Lin Zhang, Junbo Zeng, Jiabin Wang, Zi Ye, Guowen Li, Shilei Cao, Jiashun Cheng et al.* · 2026

    <span class="abstract-snippet" id="snip-2603.15260">Accurate weather forecasting is more than grid-wise regression: it must preserve coherent synoptic structures and physical consistency of meteorological fields, especially under autoregressive...</span><span class="abstract-full" id="full-2603.15260" hidden>Accurate weather forecasting is more than grid-wise regression: it must preserve coherent synoptic structures and physical consistency of meteorological fields, especially under autoregressive rollouts where small one-step errors can amplify into structural bias. Existing physics-priors approaches typically impose global, once-for-all constraints via architectures, regularization, or NWP coupling, offering limited state-adaptive and sample-specific controllability at deployment. To bridge this gap, we propose Agent-Guided Cross-modal Decoding (AGCD), a plug-and-play decoding-time prior-injection paradigm that derives state-conditioned physics-priors from the current multivariate atmosphere and injects them into forecasters in a controllable and reusable way. Specifically, We design a multi-agent meteorological narration pipeline to generate state-conditioned physics-priors, utilizing MLLMs to extract various meteorological elements effectively. To effectively apply the priors, AGCD further introduce cross-modal region interaction decoding that performs region-aware multi-scale tokenization and efficient physics-priors injection to refine visual features without changing the backbone interface. Experiments on WeatherBench demonstrate consistent gains for 6-hour forecasting across two resolutions (5.625 degree and 1.40625 degree) and diverse backbones (generic and weather-specialized), including strictly causal 48-hour autoregressive rollouts that reduce early-stage error accumulation and improve long-horizon stability.</span> <span class="abstract-toggle" data-id="2603.15260">more</span>

    [:material-file-document: 2603.15260](https://arxiv.org/abs/2603.15260v1) · [:material-content-copy: BibTeX](bibtex/2603.15260.bib){ .bibtex-link }

-   #### On Using Medium-Range Ensemble Forecasts for Storm Transposition of Synoptic-Scale Systems in Probable Maximum Precipitation Estimation

    ---

    *Mathieu Mure-Ravaud* · 2026

    <span class="abstract-snippet" id="snip-2602.19233">Most methods for estimating probable maximum precipitation (PMP) -- the greatest depth of precipitation that is physically possible over a given area and duration -- rely on storm transposition (ST),...</span><span class="abstract-full" id="full-2602.19233" hidden>Most methods for estimating probable maximum precipitation (PMP) -- the greatest depth of precipitation that is physically possible over a given area and duration -- rely on storm transposition (ST), the process of transporting a storm, either historically observed or simulated, from its original location to a target basin. Existing ST approaches, whether classical or physically based, involve assumptions and manipulations that can introduce inconsistencies, leaving the physical validity of the transposed storm uncertain. In this study, the internal variability leveraging (IVL) approach is used to transpose an atmospheric river cluster that affected the U.S. West Coast during 20-29 October 2021. Steering the storm toward the target basin and determining its transposition region are achieved by considering an ensemble of plausible storm evolutions and trajectories obtained from archived ECMWF medium-range forecasts. The Willamette River and Nass River watersheds, located approximately 6 deg N, 2 deg W and 16 deg N, 8 deg W, respectively, from the area most affected by the observed precipitation, were selected as target basins. For each basin, the IVL realization yielding the largest 24-h basin-average precipitation depth was identified, and the initial and boundary condition shifting method was subsequently applied to further enhance its impact, producing 24-h precipitation depths of 119 mm for the Willamette and 98 mm for the Nass.</span> <span class="abstract-toggle" data-id="2602.19233">more</span>

    [:material-file-document: 2602.19233](https://arxiv.org/abs/2602.19233v1) · [:material-content-copy: BibTeX](bibtex/2602.19233.bib){ .bibtex-link }

-   #### Universal Diffusion-Based Probabilistic Downscaling

    ---

    *Roberto Molinaro, Niall Siegenheim, Henry Martin, Mark Frey, Niels Poulsen, Philipp Seitz et al.* · 2026

    <span class="abstract-snippet" id="snip-2602.11893">We introduce a universal diffusion-based downscaling framework that lifts deterministic low-resolution weather forecasts into probabilistic high-resolution predictions without any model-specific...</span><span class="abstract-full" id="full-2602.11893" hidden>We introduce a universal diffusion-based downscaling framework that lifts deterministic low-resolution weather forecasts into probabilistic high-resolution predictions without any model-specific fine-tuning. A single conditional diffusion model is trained on paired coarse-resolution inputs (~25 km resolution) and high-resolution regional reanalysis targets (~5 km resolution), and is applied in a fully zero-shot manner to deterministic forecasts from heterogeneous upstream weather models. Focusing on near-surface variables, we evaluate probabilistic forecasts against independent in situ station observations over lead times up to 90 h. Across a diverse set of AI-based and numerical weather prediction (NWP) systems, the ensemble mean of the downscaled forecasts consistently improves upon each model's own raw deterministic forecast, and substantially larger gains are observed in probabilistic skill as measured by CRPS. These results demonstrate that diffusion-based downscaling provides a scalable, model-agnostic probabilistic interface for enhancing spatial resolution and uncertainty representation in operational weather forecasting pipelines.</span> <span class="abstract-toggle" data-id="2602.11893">more</span>

    [:material-file-document: 2602.11893](https://arxiv.org/abs/2602.11893v1) · [:material-content-copy: BibTeX](bibtex/2602.11893.bib){ .bibtex-link }

    <span class="md-tag">diffusion</span> <span class="md-tag">probabilistic</span>

-   #### Learning to Advect: A Neural Semi-Lagrangian Architecture for Weather Forecasting

    ---

    *Carlos A. Pereira, Stéphane Gaudreault, Valentin Dallerit, Christopher Subich, Shoyon Panday et al.* · 2026

    <span class="abstract-snippet" id="snip-2601.21151">Recent machine-learning approaches to weather forecasting often employ a monolithic architecture, where distinct physical mechanisms (advection, transport), diffusion-like mixing, thermodynamic...</span><span class="abstract-full" id="full-2601.21151" hidden>Recent machine-learning approaches to weather forecasting often employ a monolithic architecture, where distinct physical mechanisms (advection, transport), diffusion-like mixing, thermodynamic processes, and forcing are represented implicitly within a single large network. This representation is particularly problematic for advection, where long-range transport must be treated with expensive global interaction mechanisms or through deep, stacked convolutional layers. To mitigate this, we present PARADIS, a physics-inspired global weather prediction model that imposes inductive biases on network behavior through a functional decomposition into advection, diffusion, and reaction blocks acting on latent variables. We implement advection through a Neural Semi-Lagrangian operator that performs trajectory-based transport via differentiable interpolation on the sphere, enabling end-to-end learning of both the latent modes to be transported and their characteristic trajectories. Diffusion-like processes are modeled through depthwise-separable spatial mixing, while local source terms and vertical interactions are modeled via pointwise channel interactions, enabling operator-level physical structure. PARADIS provides state-of-the-art forecast skill at a fraction of the training cost. On ERA5-based benchmarks, the 1 degree PARADIS model, with a total training cost of less than a GPU month, meets or exceeds the performance of 0.25 degree traditional and machine-learning baselines, including the ECMWF HRES forecast and DeepMind's GraphCast.</span> <span class="abstract-toggle" data-id="2601.21151">more</span>

    [:material-file-document: 2601.21151](https://arxiv.org/abs/2601.21151v1) · [:material-content-copy: BibTeX](bibtex/2601.21151.bib){ .bibtex-link }

-   #### Demystifying Data-Driven Probabilistic Medium-Range Weather Forecasting

    ---

    *Jean Kossaifi, Nikola Kovachki, Morteza Mardani, Daniel Leibovici, Suman Ravuri, Ira Shokar et al.* · 2026

    <span class="abstract-snippet" id="snip-2601.18111">The recent revolution in data-driven methods for weather forecasting has lead to a fragmented landscape of complex, bespoke architectures and training strategies, obscuring the fundamental drivers of...</span><span class="abstract-full" id="full-2601.18111" hidden>The recent revolution in data-driven methods for weather forecasting has lead to a fragmented landscape of complex, bespoke architectures and training strategies, obscuring the fundamental drivers of forecast accuracy. Here, we demonstrate that state-of-the-art probabilistic skill requires neither intricate architectural constraints nor specialized training heuristics. We introduce a scalable framework for learning multi-scale atmospheric dynamics by combining a directly downsampled latent space with a history-conditioned local projector that resolves high-resolution physics. We find that our framework design is robust to the choice of probabilistic estimator, seamlessly supporting stochastic interpolants, diffusion models, and CRPS-based ensemble training. Validated against the Integrated Forecasting System and the deep learning probabilistic model GenCast, our framework achieves statistically significant improvements on most of the variables. These results suggest scaling a general-purpose model is sufficient for state-of-the-art medium-range prediction, eliminating the need for tailored training recipes and proving effective across the full spectrum of probabilistic frameworks.</span> <span class="abstract-toggle" data-id="2601.18111">more</span>

    [:material-file-document: 2601.18111](https://arxiv.org/abs/2601.18111v1) · [:material-content-copy: BibTeX](bibtex/2601.18111.bib){ .bibtex-link }

    <span class="md-tag">diffusion</span> <span class="md-tag">probabilistic</span>

-   #### Searth Transformer: A Transformer Architecture Incorporating Earth's Geospheric Physical Priors for Global Mid-Range Weather Forecasting

    ---

    *Tianye Li, Qi Liu, Hao Li, Lei Chen, Wencong Cheng, Fei Zheng, Xiangao Xia, Ya Wang, Gang Huang et al.* · 2026

    <span class="abstract-snippet" id="snip-2601.09467">Accurate global medium-range weather forecasting is fundamental to Earth system science. Most existing Transformer-based forecasting models adopt vision-centric architectures that neglect the Earth's...</span><span class="abstract-full" id="full-2601.09467" hidden>Accurate global medium-range weather forecasting is fundamental to Earth system science. Most existing Transformer-based forecasting models adopt vision-centric architectures that neglect the Earth's spherical geometry and zonal periodicity. In addition, conventional autoregressive training is computationally expensive and limits forecast horizons due to error accumulation. To address these challenges, we propose the Shifted Earth Transformer (Searth Transformer), a physics-informed architecture that incorporates zonal periodicity and meridional boundaries into window-based self-attention for physically consistent global information exchange. We further introduce a Relay Autoregressive (RAR) fine-tuning strategy that enables learning long-range atmospheric evolution under constrained memory and computational budgets. Based on these methods, we develop YanTian, a global medium-range weather forecasting model. YanTian achieves higher accuracy than the high-resolution forecast of the European Centre for Medium-Range Weather Forecasts and performs competitively with state-of-the-art AI models at one-degree resolution, while requiring roughly 200 times lower computational cost than standard autoregressive fine-tuning. Furthermore, YanTian attains a longer skillful forecast lead time for Z500 (10.3 days) than HRES (9 days). Beyond weather forecasting, this work establishes a robust algorithmic foundation for predictive modeling of complex global-scale geophysical circulation systems, offering new pathways for Earth system science.</span> <span class="abstract-toggle" data-id="2601.09467">more</span>

    [:material-file-document: 2601.09467](https://arxiv.org/abs/2601.09467v1) · [:material-content-copy: BibTeX](bibtex/2601.09467.bib){ .bibtex-link }

    <span class="md-tag">transformer</span> <span class="md-tag">physics-informed</span>

-   #### Efficient Parameter Calibration of Numerical Weather Prediction Models via Evolutionary Sequential Transfer Optimization

    ---

    *Heping Fang, Peng Yang* · 2026

    <span class="abstract-snippet" id="snip-2601.08663">The configuration of physical parameterization schemes in Numerical Weather Prediction (NWP) models plays a critical role in determining the accuracy of the forecast. However, existing parameter...</span><span class="abstract-full" id="full-2601.08663" hidden>The configuration of physical parameterization schemes in Numerical Weather Prediction (NWP) models plays a critical role in determining the accuracy of the forecast. However, existing parameter calibration methods typically treat each calibration task as an isolated optimization problem. This approach suffers from prohibitive computational costs and necessitates performing iterative searches from scratch for each task, leading to low efficiency in sequential calibration scenarios. To address this issue, we propose the SEquential Evolutionary Transfer Optimization (SEETO) algorithm driven by the representations of the meteorological state. First, to accurately measure the physical similarity between calibration tasks, a meteorological state representation extractor is introduced to map high-dimensional meteorological fields into latent representations. Second, given the similarity in the latent space, a bi-level adaptive knowledge transfer mechanism is designed. At the solution level, superior populations from similar historical tasks are reused to achieve a "warm start" for optimization. At the model level, an ensemble surrogate model based on source task data is constructed to assist the search, employing an adaptive weighting mechanism to dynamically balance the contributions of source domain knowledge and target domain data. Extensive experiments across 10 distinct calibration tasks, which span varying source-target similarities, highlight SEETO's superior efficiency. Under a strict budget of 20 expensive evaluations, SEETO achieves a 6% average improvement in Hypervolume (HV) over two state-of-the-art baselines. Notably, to match SEETO's performance at this stage, the comparison algorithms would require an average of 64% and 28% additional evaluations, respectively. This presents a new paradigm for the efficient and accurate automated calibration of NWP model parameters.</span> <span class="abstract-toggle" data-id="2601.08663">more</span>

    [:material-file-document: 2601.08663](https://arxiv.org/abs/2601.08663v1) · [:material-content-copy: BibTeX](bibtex/2601.08663.bib){ .bibtex-link }

-   #### Evaluating Weather Forecasts from a Decision Maker's Perspective

    ---

    *Kornelius Raeth, Nicole Ludwig* · 2025

    <span class="abstract-snippet" id="snip-2512.14779">Standard weather forecast evaluations focus on the forecaster's perspective and on a statistical assessment comparing forecasts and observations. In practice, however, forecasts are used to make...</span><span class="abstract-full" id="full-2512.14779" hidden>Standard weather forecast evaluations focus on the forecaster's perspective and on a statistical assessment comparing forecasts and observations. In practice, however, forecasts are used to make decisions, so it seems natural to take the decision-maker's perspective and quantify the value of a forecast by its ability to improve decision-making. Decision calibration provides a novel framework for evaluating forecast performance at the decision level rather than the forecast level. We evaluate decision calibration to compare Machine Learning and classical numerical weather prediction models on various weather-dependent decision tasks. We find that model performance at the forecast level does not reliably translate to performance in downstream decision-making: some performance differences only become apparent at the decision level, and model rankings can change among different decision tasks. Our results confirm that typical forecast evaluations are insufficient for selecting the optimal forecast model for a specific decision task.</span> <span class="abstract-toggle" data-id="2512.14779">more</span>

    [:material-file-document: 2512.14779](https://arxiv.org/abs/2512.14779v1) · [:material-content-copy: BibTeX](bibtex/2512.14779.bib){ .bibtex-link }

-   #### Bridging Artificial Intelligence and Data Assimilation: The Data-driven Ensemble Forecasting System ClimaX-LETKF

    ---

    *Akira Takeshima, Kenta Shiraishi, Atsushi Okazaki, Tadashi Tsuyuki, Shunji Kotsuki* · 2025

    <span class="abstract-snippet" id="snip-2512.14444">While machine learning-based weather prediction (MLWP) has achieved significant advancements, research on assimilating real observations or ensemble forecasts within MLWP models remains limited. We...</span><span class="abstract-full" id="full-2512.14444" hidden>While machine learning-based weather prediction (MLWP) has achieved significant advancements, research on assimilating real observations or ensemble forecasts within MLWP models remains limited. We introduce ClimaX-LETKF, the first purely data-driven ML-based ensemble weather forecasting system. It operates stably over multiple years, independently of numerical weather prediction (NWP) models, by assimilating the NCEP ADP Global Upper Air and Surface Weather Observations. The system demonstrates greater stability and accuracy with relaxation to prior perturbation (RTPP) than with relaxation to prior spread (RTPS), while NWP models tend to be more stable with RTPS. RTPP replaces an analysis perturbation with a weighted blend of analysis and background perturbations, whereas RTPS simply rescales the analysis perturbation. Our experiments reveal that MLWP models are less capable of restoring the atmospheric field to its attractor than NWP models. This work provides valuable insights for enhancing MLWP ensemble forecasting systems and represents a substantial step toward their practical applications.</span> <span class="abstract-toggle" data-id="2512.14444">more</span>

    [:material-file-document: 2512.14444](https://arxiv.org/abs/2512.14444v1) · [:material-content-copy: BibTeX](bibtex/2512.14444.bib){ .bibtex-link }

-   #### Observation-driven correction of numerical weather prediction for marine winds

    ---

    *Matteo Peduto, Qidong Yang, Jonathan Giezendanner, Devis Tuia, Sherrie Wang* · 2025

    <span class="abstract-snippet" id="snip-2512.03606">Accurate marine wind forecasts are essential for safe navigation, ship routing, and energy operations, yet they remain challenging because observations over the ocean are sparse, heterogeneous, and...</span><span class="abstract-full" id="full-2512.03606" hidden>Accurate marine wind forecasts are essential for safe navigation, ship routing, and energy operations, yet they remain challenging because observations over the ocean are sparse, heterogeneous, and temporally variable. We reformulate wind forecasting as observation-informed correction of a global numerical weather prediction (NWP) model. Rather than forecasting winds directly, we learn local correction patterns by assimilating the latest in-situ observations to adjust the Global Forecast System (GFS) output. We propose a transformer-based deep learning architecture that (i) handles irregular and time-varying observation sets through masking and set-based attention mechanisms, (ii) conditions predictions on recent observation-forecast pairs via cross-attention, and (iii) employs cyclical time embeddings and coordinate-aware location representations to enable single-pass inference at arbitrary spatial coordinates. We evaluate our model over the Atlantic Ocean using observations from the International Comprehensive Ocean-Atmosphere Data Set (ICOADS) as reference. The model reduces GFS 10-meter wind RMSE at all lead times up to 48 hours, achieving 45% improvement at 1-hour lead time and 13% improvement at 48-hour lead time. Spatial analyses reveal the most persistent improvements along coastlines and shipping routes, where observations are most abundant. The tokenized architecture naturally accommodates heterogeneous observing platforms (ships, buoys, tide gauges, and coastal stations) and produces both site-specific predictions and basin-scale gridded products in a single forward pass. These results demonstrate a practical, low-latency post-processing approach that complements NWP by learning to correct systematic forecast errors.</span> <span class="abstract-toggle" data-id="2512.03606">more</span>

    [:material-file-document: 2512.03606](https://arxiv.org/abs/2512.03606v1) · [:material-content-copy: BibTeX](bibtex/2512.03606.bib){ .bibtex-link }

    <span class="md-tag">transformer</span>

-   #### COBASE: A new copula-based shuffling method for ensemble weather forecast postprocessing

    ---

    *Maurits Flos, Bastien François, Irene Schicker, Kirien Whan, Elisa Perrone* · 2025

    <span class="abstract-snippet" id="snip-2510.25610">Weather predictions are often provided as ensembles generated by repeated runs of numerical weather prediction models. These forecasts typically exhibit bias and inaccurate dependence structures due...</span><span class="abstract-full" id="full-2510.25610" hidden>Weather predictions are often provided as ensembles generated by repeated runs of numerical weather prediction models. These forecasts typically exhibit bias and inaccurate dependence structures due to numerical and dispersion errors, requiring statistical postprocessing for improved precision. A common correction strategy is the two-step approach: first adjusting the univariate forecasts, then reconstructing the multivariate dependence. The second step is usually handled with nonparametric methods, which can underperform when historical data are limited. Parametric alternatives, such as the Gaussian Copula Approach (GCA), offer theoretical advantages but often produce poorly calibrated multivariate forecasts due to random sampling of the corrected univariate margins. In this work, we introduce COBASE, a novel copula-based postprocessing framework that preserves the flexibility of parametric modeling while mimicking the nonparametric techniques through a rank-shuffling mechanism. This design ensures calibrated margins and realistic dependence reconstruction. We evaluate COBASE on multi-site 2-meter temperature forecasts from the ALADIN-LAEF ensemble over Austria and on joint forecasts of temperature and dew point temperature from the ECMWF system in the Netherlands. Across all regions, COBASE variants consistently outperform traditional copula-based approaches, such as GCA, and achieve performance on par with state-of-the-art nonparametric methods like SimSchaake and ECC, with only minimal differences across settings. These results position COBASE as a competitive and robust alternative for multivariate ensemble postprocessing, offering a principled bridge between parametric and nonparametric dependence reconstruction.</span> <span class="abstract-toggle" data-id="2510.25610">more</span>

    [:material-file-document: 2510.25610](https://arxiv.org/abs/2510.25610v1) · [:material-content-copy: BibTeX](bibtex/2510.25610.bib){ .bibtex-link }

-   #### Revealing the Potential of Learnable Perturbation Ensemble Forecast Model for Tropical Cyclone Prediction

    ---

    *Jun Liu, Tao Zhou, Jiarui Li, Xiaohui Zhong, Peng Zhang, Jie Feng, Lei Chen, Hao Li* · 2025

    <span class="abstract-snippet" id="snip-2510.23794">Tropical cyclones (TCs) are highly destructive and inherently uncertain weather systems. Ensemble forecasting helps quantify these uncertainties, yet traditional systems are constrained by high...</span><span class="abstract-full" id="full-2510.23794" hidden>Tropical cyclones (TCs) are highly destructive and inherently uncertain weather systems. Ensemble forecasting helps quantify these uncertainties, yet traditional systems are constrained by high computational costs and limited capability to fully represent atmospheric nonlinearity. FuXi-ENS introduces a learnable perturbation scheme for ensemble generation, representing a novel AI-based forecasting paradigm. Here, we systematically compare FuXi-ENS with ECMWF-ENS using all 90 global TCs in 2018, examining their performance in TC-related physical variables, track and intensity forecasts, and the associated dynamical and thermodynamical fields. FuXi-ENS demonstrates clear advantages in predicting TC-related physical variables, and achieves more accurate track forecasts with reduced ensemble spread, though it still underestimates intensity relative to observations. Further dynamical and thermodynamical analyses reveal that FuXi-ENS better captures large-scale circulation, with moisture turbulent energy more tightly concentrated around the TC warm core, whereas ECMWF-ENS exhibits a more dispersed distribution. These findings highlight the potential of learnable perturbations to improve TC forecasting skill and provide valuable insights for advancing AI-based ensemble prediction of extreme weather events that have significant societal impacts.</span> <span class="abstract-toggle" data-id="2510.23794">more</span>

    [:material-file-document: 2510.23794](https://arxiv.org/abs/2510.23794v1) · [:material-content-copy: BibTeX](bibtex/2510.23794.bib){ .bibtex-link }

-   #### Mesh Interpolation Graph Network for Dynamic and Spatially Irregular Global Weather Forecasting

    ---

    *Zinan Zheng, Yang Liu, Jia Li* · 2025

    <span class="abstract-snippet" id="snip-2509.20911">Graph neural networks have shown promising results in weather forecasting, which is critical for human activity such as agriculture planning and extreme weather preparation. However, most studies...</span><span class="abstract-full" id="full-2509.20911" hidden>Graph neural networks have shown promising results in weather forecasting, which is critical for human activity such as agriculture planning and extreme weather preparation. However, most studies focus on finite and local areas for training, overlooking the influence of broader areas and limiting their ability to generalize effectively. Thus, in this work, we study global weather forecasting that is irregularly distributed and dynamically varying in practice, requiring the model to generalize to unobserved locations. To address such challenges, we propose a general Mesh Interpolation Graph Network (MIGN) that models the irregular weather station forecasting, consisting of two key designs: (1) learning spatially irregular data with regular mesh interpolation network to align the data; (2) leveraging parametric spherical harmonics location embedding to further enhance spatial generalization ability. Extensive experiments on an up-to-date observation dataset show that MIGN significantly outperforms existing data-driven models. Besides, we show that MIGN has spatial generalization ability, and is capable of generalizing to previous unseen stations.</span> <span class="abstract-toggle" data-id="2509.20911">more</span>

    [:material-file-document: 2509.20911](https://arxiv.org/abs/2509.20911v1) · [:material-content-copy: BibTeX](bibtex/2509.20911.bib){ .bibtex-link }

    <span class="md-tag">GNN</span>

-   #### An update to ECMWF's machine-learned weather forecast model AIFS

    ---

    *Gabriel Moldovan, Ewan Pinnington, Ana Prieto Nemesio, Simon Lang, Zied Ben Bouallègue et al.* · 2025

    <span class="abstract-snippet" id="snip-2509.18994">We present an update to ECMWF's machine-learned weather forecasting model AIFS Single with several key improvements. The model now incorporates physical consistency constraints through bounding...</span><span class="abstract-full" id="full-2509.18994" hidden>We present an update to ECMWF's machine-learned weather forecasting model AIFS Single with several key improvements. The model now incorporates physical consistency constraints through bounding layers, an updated training schedule, and an expanded set of variables. The physical constraints substantially improve precipitation forecasts and the new variables show a high level of skill. Upper-air headline scores also show improvement over the previous AIFS version. The AIFS has been fully operational at ECMWF since the 25th of February 2025.</span> <span class="abstract-toggle" data-id="2509.18994">more</span>

    [:material-file-document: 2509.18994](https://arxiv.org/abs/2509.18994v1) · [:material-content-copy: BibTeX](bibtex/2509.18994.bib){ .bibtex-link }

-   #### Training-Free Data Assimilation with GenCast

    ---

    *Thomas Savary, François Rozet, Gilles Louppe* · 2025

    <span class="abstract-snippet" id="snip-2509.18811">Data assimilation is widely used in many disciplines such as meteorology, oceanography, and robotics to estimate the state of a dynamical system from noisy observations. In this work, we propose a...</span><span class="abstract-full" id="full-2509.18811" hidden>Data assimilation is widely used in many disciplines such as meteorology, oceanography, and robotics to estimate the state of a dynamical system from noisy observations. In this work, we propose a lightweight and general method to perform data assimilation using diffusion models pre-trained for emulating dynamical systems. Our method builds on particle filters, a class of data assimilation algorithms, and does not require any further training. As a guiding example throughout this work, we illustrate our methodology on GenCast, a diffusion-based model that generates global ensemble weather forecasts.</span> <span class="abstract-toggle" data-id="2509.18811">more</span>

    [:material-file-document: 2509.18811](https://arxiv.org/abs/2509.18811v1) · [:material-content-copy: BibTeX](bibtex/2509.18811.bib){ .bibtex-link }

    <span class="md-tag">diffusion</span> <span class="md-tag">foundation-model</span>

-   #### GraphCast: Learning skillful medium-range global weather forecasting

    ---

    *Remi Lam, Alvaro Sanchez-Gonzalez, Matthew Willson, Peter Wirnsberger, Meire Fortunato, Ferran Alet et al.* · 2022

    <span class="abstract-snippet" id="snip-2212.12794">Global medium-range weather forecasting is critical to decision-making across many social and economic domains. Traditional numerical weather prediction uses increased compute resources to improve...</span><span class="abstract-full" id="full-2212.12794" hidden>Global medium-range weather forecasting is critical to decision-making across many social and economic domains. Traditional numerical weather prediction uses increased compute resources to improve forecast accuracy, but cannot directly use historical weather data to improve the underlying model. We introduce a machine learning-based method called "GraphCast", which can be trained directly from reanalysis data. It predicts hundreds of weather variables, over 10 days at 0.25 degree resolution globally, in under one minute. We show that GraphCast significantly outperforms the most accurate operational deterministic systems on 90% of 1380 verification targets, and its forecasts support better severe event prediction, including tropical cyclones, atmospheric rivers, and extreme temperatures. GraphCast is a key advance in accurate and efficient weather forecasting, and helps realize the promise of machine learning for modeling complex dynamical systems.</span> <span class="abstract-toggle" data-id="2212.12794">more</span>

    [:material-file-document: 2212.12794](https://arxiv.org/abs/2212.12794v2) · [:fontawesome-brands-github:](https://github.com/deepmind/graphcast) · [:material-content-copy: BibTeX](bibtex/2212.12794.bib){ .bibtex-link }

-   #### Pangu-Weather: A 3D High-Resolution Model for Fast and Accurate Global Weather Forecast

    ---

    *Kaifeng Bi, Lingxi Xie, Hengheng Zhang, Xin Chen, Xiaotao Gu, Qi Tian* · 2022

    <span class="abstract-snippet" id="snip-2211.02556">In this paper, we present Pangu-Weather, a deep learning based system for fast and accurate global weather forecast. For this purpose, we establish a data-driven environment by downloading $43$ years...</span><span class="abstract-full" id="full-2211.02556" hidden>In this paper, we present Pangu-Weather, a deep learning based system for fast and accurate global weather forecast. For this purpose, we establish a data-driven environment by downloading $43$ years of hourly global weather data from the 5th generation of ECMWF reanalysis (ERA5) data and train a few deep neural networks with about $256$ million parameters in total. The spatial resolution of forecast is $0.25^\circ\times0.25^\circ$, comparable to the ECMWF Integrated Forecast Systems (IFS). More importantly, for the first time, an AI-based method outperforms state-of-the-art numerical weather prediction (NWP) methods in terms of accuracy (latitude-weighted RMSE and ACC) of all factors (e.g., geopotential, specific humidity, wind speed, temperature, etc.) and in all time ranges (from one hour to one week). There are two key strategies to improve the prediction accuracy: (i) designing a 3D Earth Specific Transformer (3DEST) architecture that formulates the height (pressure level) information into cubic data, and (ii) applying a hierarchical temporal aggregation algorithm to alleviate cumulative forecast errors. In deterministic forecast, Pangu-Weather shows great advantages for short to medium-range forecast (i.e., forecast time ranges from one hour to one week). Pangu-Weather supports a wide range of downstream forecast scenarios, including extreme weather forecast (e.g., tropical cyclone tracking) and large-member ensemble forecast in real-time. Pangu-Weather not only ends the debate on whether AI-based methods can surpass conventional NWP methods, but also reveals novel directions for improving deep learning weather forecast systems.</span> <span class="abstract-toggle" data-id="2211.02556">more</span>

    [:material-file-document: 2211.02556](https://arxiv.org/abs/2211.02556v1) · [:material-content-copy: BibTeX](bibtex/2211.02556.bib){ .bibtex-link }

    <span class="md-tag">transformer</span>

-   #### FourCastNet: A Global Data-driven High-resolution Weather Model using Adaptive Fourier Neural Operators

    ---

    *Jaideep Pathak, Shashank Subramanian, Peter Harrington, Sanjeev Raja, Ashesh Chattopadhyay et al.* · 2022

    <span class="abstract-snippet" id="snip-2202.11214">FourCastNet, short for Fourier Forecasting Neural Network, is a global data-driven weather forecasting model that provides accurate short to medium-range global predictions at $0.25^{\circ}$...</span><span class="abstract-full" id="full-2202.11214" hidden>FourCastNet, short for Fourier Forecasting Neural Network, is a global data-driven weather forecasting model that provides accurate short to medium-range global predictions at $0.25^{\circ}$ resolution. FourCastNet accurately forecasts high-resolution, fast-timescale variables such as the surface wind speed, precipitation, and atmospheric water vapor. It has important implications for planning wind energy resources, predicting extreme weather events such as tropical cyclones, extra-tropical cyclones, and atmospheric rivers. FourCastNet matches the forecasting accuracy of the ECMWF Integrated Forecasting System (IFS), a state-of-the-art Numerical Weather Prediction (NWP) model, at short lead times for large-scale variables, while outperforming IFS for variables with complex fine-scale structure, including precipitation. FourCastNet generates a week-long forecast in less than 2 seconds, orders of magnitude faster than IFS. The speed of FourCastNet enables the creation of rapid and inexpensive large-ensemble forecasts with thousands of ensemble-members for improving probabilistic forecasting. We discuss how data-driven deep learning models such as FourCastNet are a valuable addition to the meteorology toolkit to aid and augment NWP models.</span> <span class="abstract-toggle" data-id="2202.11214">more</span>

    [:material-file-document: 2202.11214](https://arxiv.org/abs/2202.11214v1) · [:material-content-copy: BibTeX](bibtex/2202.11214.bib){ .bibtex-link }

    <span class="md-tag">operator-learning</span> <span class="md-tag">probabilistic</span>

</div>

## Nowcasting (27)

<div class="grid cards" markdown>

-   #### Pointwise is Pointless? A Multimodal Ablation Study for Precipitation Nowcasting with Graph Neural Networks

    ---

    *Ophélia Miralles, Máté Mile, Christoffer Artturi, Thomas Nipen, Ivar Seierstad* · 2026

    <span class="abstract-snippet" id="snip-2606.18436">Sparse point observations are increasingly available for precipitation nowcasting, but it is unclear how much they improve dense radar-field forecasts. We partially address this question with a...</span><span class="abstract-full" id="full-2606.18436" hidden>Sparse point observations are increasingly available for precipitation nowcasting, but it is unclear how much they improve dense radar-field forecasts. We partially address this question with a multimodal graph neural network nowcasting system over the Nordic radar domain. The model predicts rain rate every five minutes up to two hours ahead and is trained with different combinations of radar history, MEPS numerical weather prediction, Netatmo surface observations, MSG satellite channels, stochastic noise, and CRPS-based ensemble losses. The study is designed as an ablation of operationally relevant information sources and training objectives. We compare radar-only, NWP-informed, station-informed, satellite-informed, noise-augmented, and CRPS-based configurations using complementary diagnostics on the radar grid, at station locations, for rain onset, and through oracle, displacement, and amplitude scores. The results show that each source improves a different part of the forecast problem. MEPS stabilises radar-only extrapolation, Netatmo observations improve local station and onset diagnostics, and satellite predictors reduce some station-level biases but may activate rain too early when used deterministically. CRPS-based configurations provide the most consistent radar-grid gains, while the combined satellite and CRPS setup gives the best overall oracle/DAS score. These results do not support the conclusion that point observations are uninformative for nowcasting, but they show that local observational skill and spatially coherent radar-field skill are distinct targets. The practical implication is that sparse observations can provide useful local constraints, but their benefit for radar-like fields depends on the training loss, uncertainty representation, and how observation support is encoded in the model.</span> <span class="abstract-toggle" data-id="2606.18436">more</span>

    [:material-file-document: 2606.18436](https://arxiv.org/abs/2606.18436v2) · [:material-content-copy: BibTeX](bibtex/2606.18436.bib){ .bibtex-link }

    <span class="md-tag">GNN</span>

-   #### When the Past Matters: FlashBack Memory for Precipitation Nowcasting

    ---

    *Yuhao Du, Boxiao Huang, Chengrong Wu, Jiankai Zhang* · 2026

    <span class="abstract-snippet" id="snip-2606.16342">Accurate precipitation nowcasting is crucial for disaster mitigation and socio-economic planning, yet existing methods often struggle with false alarms, missed events, and long range dependency...</span><span class="abstract-full" id="full-2606.16342" hidden>Accurate precipitation nowcasting is crucial for disaster mitigation and socio-economic planning, yet existing methods often struggle with false alarms, missed events, and long range dependency modeling at high spatiotemporal resolution. To address these challenges, we propose FlashBack Memory (FB), a module that dynamically retrieves key historical states and integrates them via an adaptive fusion gate, enhancing the spatiotemporal representation capability of recurrent-based models. We incorporate FB into PredRNN, PredRNNpp, MIM, MotionRNN, and PredRNN-V2, and evaluate on CIKM2017, Shanghai2020, and SEVIR datasets. Experimental results demonstrate that FB significantly improves MSE, MAE, SSIM, and CSI metrics, particularly for high-intensity rainfall and long-sequence predictions, while reducing false alarms and missed events and enhancing temporal consistency and spatial localization. The proposed method provides a general and efficient memory enhancement mechanism, improving the overall performance of recurrent-based precipitation nowcasting models.</span> <span class="abstract-toggle" data-id="2606.16342">more</span>

    [:material-file-document: 2606.16342](https://arxiv.org/abs/2606.16342v1) · [:material-content-copy: BibTeX](bibtex/2606.16342.bib){ .bibtex-link }

-   #### Temporal Context Conditioning for Seasonality-Aware Precipitation Nowcasting of High-Intensity Rainfall

    ---

    *Gijs van Nieuwkoop, Siamak Mehrkanoon* · 2026

    <span class="abstract-snippet" id="snip-2606.09959">Precipitation nowcasting is increasingly being approached with deep learning models that learn directly from recent radar observations. Although such models can efficiently capture short-term...</span><span class="abstract-full" id="full-2606.09959" hidden>Precipitation nowcasting is increasingly being approached with deep learning models that learn directly from recent radar observations. Although such models can efficiently capture short-term precipitation motion, they often lack broader contextual information about the meteorological conditions under which rainfall develops. This paper investigates whether lightweight temporal context can improve radar-based nowcasting, particularly for high-intensity rainfall. We propose the Time-Aware Small-Attention U-Net (TA-SmaAt-UNet), which extends the core SmaAt-UNet model with temporal conditioning layers that use cyclical encodings of time-of-day and time-of-year to modulate intermediate feature representations. Experiments on KNMI radar precipitation data show that temporal conditioning is most beneficial for rare, high-intensity precipitation events, while also improving the representation of seasonal variability and predicted rainfall-intensity distributions. A layer conductance analysis further indicates that the added temporal conditioning layers are actively used by the model despite their small parameter cost. These findings suggest that simple, physically motivated temporal context can improve the realism and reliability of deep learning-based precipitation nowcasts. The implementation of our models and training setup is available on \href{https://github.com/gijsvn/TA-SmaAt-UNet}{GitHub}.</span> <span class="abstract-toggle" data-id="2606.09959">more</span>

    [:material-file-document: 2606.09959](https://arxiv.org/abs/2606.09959v1) · [:fontawesome-brands-github:](https://github.com/gijsvn/TA-SmaAt-UNet) · [:material-content-copy: BibTeX](bibtex/2606.09959.bib){ .bibtex-link }

    <span class="md-tag">CNN</span>

-   #### Learning to Solve Generative ODEs Beyond the Linear Span

    ---

    *Sihyeon Kim, Seunghun Lee, Vikas Singh, Hyunwoo J. Kim* · 2026

    <span class="abstract-snippet" id="snip-2606.08672">Diffusion and flow generative models sample by integrating a learned ODE, but high quality still requires many sequential model evaluations. Solver learning reduces this cost by adapting scalar...</span><span class="abstract-full" id="full-2606.08672" hidden>Diffusion and flow generative models sample by integrating a learned ODE, but high quality still requires many sequential model evaluations. Solver learning reduces this cost by adapting scalar coefficients, timesteps, or both, while keeping the backbone model fixed. In this work, we identify a structural bottleneck in this update family: each step remains span-limited. Since the scalar-coefficient update lies in the span of buffered velocity evaluations, it can fit only the in-span component while leaving any out-of-span residual unreachable by scalar recombination alone. We propose SpanLift, a lightweight neural solver that augments scalar-coefficient updates with a spatial residual operator. SpanLift keeps a fixed base solver as an in-span prior and learns a spatial residual operator over the state and velocity buffer. The operator is trained by endpoint teacher matching, preserves the pretrained backbone, and adds no model NFEs. Empirically, the learned correction transfers across base solvers and is predominantly out-of-span. Across pixel-space diffusion, latent flow matching, and precipitation nowcasting, SpanLift achieves state-of-the-art few-step sampling. With only 3 NFE, it improves CIFAR-10 FID from 8.16 to 5.69 and ImageNet FID from 17.37 to 11.83.</span> <span class="abstract-toggle" data-id="2606.08672">more</span>

    [:material-file-document: 2606.08672](https://arxiv.org/abs/2606.08672v1) · [:material-content-copy: BibTeX](bibtex/2606.08672.bib){ .bibtex-link }

    <span class="md-tag">diffusion</span>

-   #### Learning to Refine: Spectral-Decoupled Iterative Refinement Framework for Precipitation Nowcasting

    ---

    *Yunlong Zhou, Chen Zhao, Danyang Peng, Fanfan Ji, Xiao-Tong Yuan* · 2026

    <span class="abstract-snippet" id="snip-2606.02661">Accurate precipitation nowcasting is vital for disaster mitigation, but deep learning methods face a key trade-off: regression models produce over-smoothed, spectrally decaying predictions that blur...</span><span class="abstract-full" id="full-2606.02661" hidden>Accurate precipitation nowcasting is vital for disaster mitigation, but deep learning methods face a key trade-off: regression models produce over-smoothed, spectrally decaying predictions that blur convective details and violate turbulence power laws; diffusion models generate realistic yet unanchored hallucinations lacking physical grounding. We propose Spectral-Decoupled Iterative Refinement (SDIR), a deterministic framework that reformulates nowcasting as progressive frequency-decoupled refinement. SDIR first extracts a stable low-frequency synoptic skeleton, then iteratively refines high-frequency textures under physical constraints, eliminating both blurring and hallucinations. It features a dual-path design: the Synoptic Frequency-Guided Former (SFG-Former) with Scale-Adaptive Transformers for global structure, and the Fourier Residual Refiner (FR-Refiner) with Scale-Conditioned Fourier Neural Operators for fine residuals. A Physically Consistent Power Spectral Density (PCPSD) loss with dynamic masking enforces a turbulence-consistent spectral distribution. Experiments on three benchmarks show SDIR significantly outperforms SOTA methods in spatial accuracy while achieving spectral fidelity competitive with diffusion-based methods, enabling reliable high-resolution operational nowcasting. Code link: https://github.com/RuntimeWarning/SDIR.</span> <span class="abstract-toggle" data-id="2606.02661">more</span>

    [:material-file-document: 2606.02661](https://arxiv.org/abs/2606.02661v1) · [:fontawesome-brands-github:](https://github.com/RuntimeWarning/SDIR) · [:material-content-copy: BibTeX](bibtex/2606.02661.bib){ .bibtex-link }

    <span class="md-tag">transformer</span> <span class="md-tag">diffusion</span> <span class="md-tag">operator-learning</span>

-   #### Probabilistic Precipitation Nowcasting with Rectified Flow Transformers

    ---

    *Johannes Schusterbauer, Jannik Wiese, Nick Stracke, Timy Phan, Björn Ommer* · 2026

    <span class="abstract-snippet" id="snip-2605.31204">Accurate weather forecasts are essential across various domains and are safety-critical in extreme weather conditions. Compared to simulation-based forecasting, data-driven approaches show greater...</span><span class="abstract-full" id="full-2605.31204" hidden>Accurate weather forecasts are essential across various domains and are safety-critical in extreme weather conditions. Compared to simulation-based forecasting, data-driven approaches show greater efficiency, enabling short-term, high-resolution nowcasting. In particular, diffusion models proved effective in weather nowcasting due to their strong probabilistic foundation. However, existing methods rely on deterministic compression to reduce the complexity of high-dimensional weather data, limiting their ability to capture uncertainty in the decoding process. In this work, we introduce $\textbf{FREUD}$, a $\textbf{Fr}$ame-wise $\textbf{E}$ncoder and $\textbf{U}$nited $\textbf{D}$ecoder model based on rectified flow transformers for efficient compression of spatio-temporal weather data. Frame-wise encoding enables continuous forecast updates, while the unified video decoder ensures temporal consistency. Our uncertainty-preserving first stage allows us to capture aleatoric uncertainty via ensembling, which is particularly beneficial for extreme weather events with high decoding variability. We achieve state-of-the-art performance in precipitation nowcasting with a compact latent-space rectified flow transformer on the SEVIR benchmark and show further performance gains by model and test-time scaling. Code available here: https://github.com/CompVis/weather-rf</span> <span class="abstract-toggle" data-id="2605.31204">more</span>

    [:material-file-document: 2605.31204](https://arxiv.org/abs/2605.31204v1) · [:fontawesome-brands-github:](https://github.com/CompVis/weather-rf) · [:material-content-copy: BibTeX](bibtex/2605.31204.bib){ .bibtex-link }

    <span class="md-tag">transformer</span> <span class="md-tag">diffusion</span> <span class="md-tag">probabilistic</span>

-   #### Beyond MSE: Improving Precipitation Nowcasting with Multi-Quantile Regression

    ---

    *Gijs van Nieuwkoop, Siamak Mehrkanoon* · 2026

    <span class="abstract-snippet" id="snip-2605.30122">Deep-learning precipitation nowcasting models are often optimized using pointwise losses such as mean squared error or mean absolute error, which can lead to overly smooth forecasts and poor...</span><span class="abstract-full" id="full-2605.30122" hidden>Deep-learning precipitation nowcasting models are often optimized using pointwise losses such as mean squared error or mean absolute error, which can lead to overly smooth forecasts and poor representation of heavy rainfall. This study investigates whether the predictive performance of an established deterministic nowcasting architecture can be improved by reformulating training as a multi-quantile regression problem. Using SmaAt-UNet as a core model, we compare MSE, MAE, and multi-quantile pinball-loss training on radar precipitation nowcasting over the Netherlands. The results show that multi-quantile training improves the central deterministic forecast, decreasing test-set MSE by 8.6\% compared to a model trained using MSE, while also producing upper-quantile outputs that are useful for risk-sensitive prediction of heavy precipitation. These findings suggest that quantile regression provides a simple alternative to standard pointwise losses without requiring a new architecture or generative sampling procedure. The implementation of our models and training setup is available on \href{https://github.com/gijsvn/Multi-Quantile-Precipitation-Nowcasting}{GitHub}.</span> <span class="abstract-toggle" data-id="2605.30122">more</span>

    [:material-file-document: 2605.30122](https://arxiv.org/abs/2605.30122v2) · [:fontawesome-brands-github:](https://github.com/gijsvn/Multi-Quantile-Precipitation-Nowcasting) · [:material-content-copy: BibTeX](bibtex/2605.30122.bib){ .bibtex-link }

    <span class="md-tag">CNN</span>

-   #### Beyond MSE: Improving Precipitation Nowcasting with Multi-Quantile Regression

    ---

    *Gijs van Nieuwkoop, Siamak Mehrkanoon* · 2026

    <span class="abstract-snippet" id="snip-2605.30122">Deep-learning precipitation nowcasting models are often optimized using pointwise losses such as mean squared error or mean absolute error, which can lead to overly smooth forecasts and poor...</span><span class="abstract-full" id="full-2605.30122" hidden>Deep-learning precipitation nowcasting models are often optimized using pointwise losses such as mean squared error or mean absolute error, which can lead to overly smooth forecasts and poor representation of heavy rainfall. This study investigates whether the predictive performance of an established deterministic nowcasting architecture can be improved by reformulating training as a multi-quantile regression problem. Using SmaAt-UNet as a core model, we compare MSE, MAE, and multi-quantile pinball-loss training on radar precipitation nowcasting over the Netherlands. The results show that multi-quantile training improves the central deterministic forecast, decreasing test-set MSE by 8.6\% compared to a model trained using MSE, while also producing upper-quantile outputs that are useful for risk-sensitive prediction of heavy precipitation. These findings suggest that quantile regression provides a simple alternative to standard pointwise losses without requiring a new architecture or generative sampling procedure. The implementation of our models and training setup is available on \href{https://github.com/gijsvn/Multi-Quantile-Precipitation-Nowcasting}{GitHub}.</span> <span class="abstract-toggle" data-id="2605.30122">more</span>

    [:material-file-document: 2605.30122](https://arxiv.org/abs/2605.30122v1) · [:fontawesome-brands-github:](https://github.com/gijsvn/Multi-Quantile-Precipitation-Nowcasting) · [:material-content-copy: BibTeX](bibtex/2605.30122.bib){ .bibtex-link }

    <span class="md-tag">CNN</span>

-   #### MambaRain: Multi-Scale Mamba-Attention Framework for 0-3 Hour Precipitation Nowcasting

    ---

    *Chunlei Shi, Cui Wu, Xiang Xu, Hao Li, Ni Fan, Xue Han, Yongchao Feng, Yufeng Zhu, Boyu Liu et al.* · 2026

    <span class="abstract-snippet" id="snip-2605.14606">Accurate precipitation nowcasting over extended horizons (0-3 hours) is essential for disaster mitigation and operational decision-making, yet remains a critical challenge in the field. Existing...</span><span class="abstract-full" id="full-2605.14606" hidden>Accurate precipitation nowcasting over extended horizons (0-3 hours) is essential for disaster mitigation and operational decision-making, yet remains a critical challenge in the field. Existing deterministic approaches are predominantly constrained to shorter prediction windows (0-2 hours), exhibiting severe performance degradation beyond 90 minutes owing to their inherent difficulty in capturing long-range spatiotemporal dependencies from radar-derived observations. To address these fundamental limitations, we propose MambaRain, a novel multi-scale encoder-decoder architecture that synergistically integrates Mamba's linear-complexity long-range temporal modeling with self-attention mechanisms for explicit spatial correlation capture. The core innovation lies in a hybrid design paradigm wherein Mamba blocks leverage selective state space mechanisms to model global temporal dynamics across extended sequences with computational efficiency, while self-attention modules explicitly characterize spatial correlations within precipitation fields - a capability inherently absent in Mamba's sequential processing paradigm. This complementary synergy enables comprehensive spatiotemporal representation learning, effectively extending the viable forecasting horizon to 2-3 hours with substantial accuracy improvements. Furthermore, we introduce a spectral loss formulation to mitigate blurring artifacts characteristic of chaotic precipitation systems, thereby preserving fine-scale motion details critical for nowcasting accuracy. Experimental validation demonstrates that MambaRain substantially outperforms existing deterministic methodologies in 0-3 hour nowcasting tasks, with particularly pronounced performance gains in the challenging 2-3 hour prediction range.</span> <span class="abstract-toggle" data-id="2605.14606">more</span>

    [:material-file-document: 2605.14606](https://arxiv.org/abs/2605.14606v1) · [:material-content-copy: BibTeX](bibtex/2605.14606.bib){ .bibtex-link }

    <span class="md-tag">transformer</span>

-   #### VMU-Diff: A Coarse-to-fine Multi-source Data Fusion Framework for Precipitation Nowcasting

    ---

    *Chunlei Shi, Hao Li, Yufeng Zhu, Boyu Liu, Yongchao Feng, Zengliang Zang, Hongbin Wang, Yanlan Yang et al.* · 2026

    <span class="abstract-snippet" id="snip-2605.14597">Precipitation nowcasting is a vital spatio-temporal prediction task for meteorological applications but faces challenges due to the chaotic property of precipitation systems. Existing methods...</span><span class="abstract-full" id="full-2605.14597" hidden>Precipitation nowcasting is a vital spatio-temporal prediction task for meteorological applications but faces challenges due to the chaotic property of precipitation systems. Existing methods predominantly rely on single-source radar data to build either deterministic or probabilistic models for extrapolation. However, the single deterministic model suffers from blurring due to MSE convergence. The single probabilistic model, typically represented by diffusion models, can generate fine details but suffers from spurious artifacts that compromise accuracy and computational inefficiency. To address these challenges, this paper proposes a novel coarse-to-fine Vision Mamba Unet and residual Diffusion (VMU-Diff) based precipitation nowcasting framework. It realizes precipitation nowcasting through a two-stage process, i.e., a deterministic model-based coarse stage to predict global motion trends and a probabilistic model-based fine stage to generate fine prediction details. In the coarse prediction stage, rather than single-source radar data, both radar and multi-band satellite data are taken as input. A spatial-temporal attention block and several Vision mamba state-space blocks realize multi-source data fusion, and predict the future echo global dynamics. The fine-grained stage is realized by a spatio-temporal refine generator based on residual conditional diffusion models. It first obtains spatio-temporal residual features based on coarse prediction and ground truth, and further reconstructs the residual via conditional Mamba state-space module. Experiments on Jiangsu SWAN datasets demonstrate the improvements of our method over state-of-the-art methods, particularly in short-term forecasts.</span> <span class="abstract-toggle" data-id="2605.14597">more</span>

    [:material-file-document: 2605.14597](https://arxiv.org/abs/2605.14597v1) · [:material-content-copy: BibTeX](bibtex/2605.14597.bib){ .bibtex-link }

    <span class="md-tag">diffusion</span> <span class="md-tag">CNN</span> <span class="md-tag">probabilistic</span>

-   #### ForcingDAS: Unified and Robust Data Assimilation via Diffusion Forcing

    ---

    *Yixuan Jia, Siyi Chen, Yida Pan, Xiao Li, Lianghe Shi, Chanyong Jung, Haijie Yuan, Ismail Alkhouri et al.* · 2026

    <span class="abstract-snippet" id="snip-2605.14285">Data assimilation (DA) estimates the state of an evolving dynamical system from noisy, partial observations, and is widely used in scientific simulation as well as weather and climate science. In...</span><span class="abstract-full" id="full-2605.14285" hidden>Data assimilation (DA) estimates the state of an evolving dynamical system from noisy, partial observations, and is widely used in scientific simulation as well as weather and climate science. In practice, filtering methods rely on frame-to-frame transition models. However, these models are fragile when observations are non-Markovian (when they form only a partial slice of a higher-dimensional latent state as in real-world weather data): they tend to accumulate errors over long horizons. At the same time, learned DA methods typically commit to a single regime, either filtering (nowcasting, real-time forecasting) or smoothing (retrospective reanalysis), which splits what should be a shared prior across application-specific pipelines. To address both issues, we introduce ForcingDAS, a unified and robust DA framework. Built on Diffusion Forcing with an independent noise level assigned to each frame, ForcingDAS learns a joint-trajectory prior instead of frame-to-frame transitions. This allows it to capture long-horizon temporal dependencies and reduce error accumulation. In addition, the same trained model spans the full filtering to smoothing spectrum at inference time. Specifically, nowcasting, fixed-lag smoothing, and batch reanalysis are selected through the inference schedule alone, without retraining. We evaluate ForcingDAS on 2D Navier-Stokes vorticity, precipitation nowcasting, and global atmospheric state estimation. Across all settings, a single model is competitive with or outperforms both learned and classical baselines that are specialized for individual regimes, with the largest gains observed on real-world weather benchmarks.</span> <span class="abstract-toggle" data-id="2605.14285">more</span>

    [:material-file-document: 2605.14285](https://arxiv.org/abs/2605.14285v1) · [:material-content-copy: BibTeX](bibtex/2605.14285.bib){ .bibtex-link }

-   #### McCast: Memory-Guided Latent Drift Correction for Long-Horizon Precipitation Nowcasting

    ---

    *Penghui Wen, Yu Luo, Lintao Wang, Mengwei He, Patrick Filippi, Thomas Francis Bishop, Zhiyong Wang* · 2026

    <span class="abstract-snippet" id="snip-2605.13197">Existing precipitation nowcasting methods typically adopt an autoregressive formulation, where future states are predicted from previous outputs. However, such an approach accumulates errors over...</span><span class="abstract-full" id="full-2605.13197" hidden>Existing precipitation nowcasting methods typically adopt an autoregressive formulation, where future states are predicted from previous outputs. However, such an approach accumulates errors over long rollouts, causing forecasts to drift away from physically plausible evolution trajectories. Although various studies have attempted to alleviate this problem by improving step-wise prediction accuracy, they largely neglect the global temporal evolution of meteorological systems and lack mechanisms to actively correct drift during rollouts. To address this issue, we propose McCast, a memory-guided latent drift correction method for precipitation nowcasting. Rather than treating memory as an unordered dictionary of latent states for passive conditioning, McCast leverages temporally organized memory to actively correct autoregressive latent evolution. Specifically, McCast introduces a Drift-Corrective Memory Bank (DCBank) that explicitly estimates the temporally consistent drift corrections to calibrate the divergent trajectory. DCBank performs drift correction in two stages: a Corrective Latent Extractor first predicts an initial correction from the current prediction and a reference latent state, and a Correction-Aware Memory Retrieval module then refines the initial correction using temporally organized historical memory. By explicitly correcting latent evolution, instead of improving step-wise prediction accuracy only, McCast produces more temporally coherent and reliable long-horizon forecasts. Experiments on two widely used benchmarks, SEVIR and MeteoNet, show that McCast achieves state-of-the-art performance, particularly in challenging long-horizon forecasting scenarios.</span> <span class="abstract-toggle" data-id="2605.13197">more</span>

    [:material-file-document: 2605.13197](https://arxiv.org/abs/2605.13197v1) · [:material-content-copy: BibTeX](bibtex/2605.13197.bib){ .bibtex-link }

-   #### Stable Attention Response for Reliable Precipitation Nowcasting

    ---

    *Penghui Wen, Zexin Hu, Sen Zhang, Patrick Filippi, Xiaogang Zhu, Allen Benter, Thomas Bishop et al.* · 2026

    <span class="abstract-snippet" id="snip-2605.13181">Precipitation nowcasting remains challenging due to the highly localized, rapidly evolving, and heterogeneous nature of atmospheric dynamics. Although recent methods increasingly adopt...</span><span class="abstract-full" id="full-2605.13181" hidden>Precipitation nowcasting remains challenging due to the highly localized, rapidly evolving, and heterogeneous nature of atmospheric dynamics. Although recent methods increasingly adopt attention-based architectures in both unimodal and multimodal settings, they mainly emphasize stronger representation learning and prediction capacity, while paying less attention to the stability of attention responses across samples. In this work, we show that cross-sample instability of attention-response energy is an important and previously underexplored source of forecasting unreliability. Empirically, inaccurate forecasts are associated with larger attention-response energy variance across heads and layers. Theoretically, we show that cross-sample variability can propagate through self-attention, and enlarge a lower bound on prediction error. Based on this insight, we propose HARECast, a Head-wise Attention Response Energy-regulated framework for precipitation nowcasting. HARECast explicitly models head-wise attention-response energy and stabilizes it through a group-wise regularization objective that reduces cross-sample fluctuations. The proposed formulation is generic and applicable to both unimodal and multimodal nowcasting architectures. We instantiate HARECast in a standard forecasting pipeline with reconstruction branches and a diffusion-based predictor, and evaluate it on commonly used benchmarks--SEVIR and MeteoNet. Experimental results demonstrate that HARECast achieves state-of-the-art performance.</span> <span class="abstract-toggle" data-id="2605.13181">more</span>

    [:material-file-document: 2605.13181](https://arxiv.org/abs/2605.13181v1) · [:material-content-copy: BibTeX](bibtex/2605.13181.bib){ .bibtex-link }

    <span class="md-tag">transformer</span>

-   #### PixelFlowCast: Latent-Free Precipitation Nowcasting via Pixel Mean Flows

    ---

    *Yufeng Zhu, Chunlei Shi, Yongchao Feng, Dan Niu* · 2026

    <span class="abstract-snippet" id="snip-2605.10046">Precipitation nowcasting aims to forecast short-term radar echo sequences for extreme weather warning, where both prediction fidelity and inference efficiency are critical for real-world deployment....</span><span class="abstract-full" id="full-2605.10046" hidden>Precipitation nowcasting aims to forecast short-term radar echo sequences for extreme weather warning, where both prediction fidelity and inference efficiency are critical for real-world deployment. However, diffusion-based models, despite their strong generative capability, suffer from slow inference due to multi-step sampling trajectories, limiting their practical usability. Conditional Flow Matching (CFM) improves efficiency via straightened trajectories, but relies on latent space compression, which inevitably discards high-frequency physical details and degrades fine-grained prediction quality. To address these limitations, we propose PixelFlowCast, a two-stage probabilistic forecasting framework that achieves both high-efficiency and high-fidelity prediction without latent compression. Specifically, in the first stage, a deterministic model first produces coarse forecasts to capture global evolution trends. In the subsequent stage, the proposed KANCondNet extracts deep spatiotemporal evolution features to provide accurate conditional guidance. Based on this, a latent-free, few-step Pixel Mean Flows (PMF) predictor employs an $x$-prediction mechanism to generate high-quality predictions, effectively preserving fine-grained structures while maintaining fast inference. Experiments on the publicly available SEVIR dataset demonstrate that PixelFlowCast outperforms existing mainstream methods in both prediction accuracy and inference efficiency, particularly for long sequence forecasting, highlighting its strong potential for real-world operational deployment.</span> <span class="abstract-toggle" data-id="2605.10046">more</span>

    [:material-file-document: 2605.10046](https://arxiv.org/abs/2605.10046v1) · [:material-content-copy: BibTeX](bibtex/2605.10046.bib){ .bibtex-link }

    <span class="md-tag">diffusion</span> <span class="md-tag">probabilistic</span>

-   #### M3R: Localized Rainfall Nowcasting with Meteorology-Informed MultiModal Attention

    ---

    *Sanjeev Panta, Rhett M Morvant, Xu Yuan, Li Chen, Nian-Feng Tzeng* · 2026

    <span class="abstract-snippet" id="snip-2604.15377">Accurate and timely rainfall nowcasting is crucial for disaster mitigation and water resource management. Despite recent advances in deep learning, precipitation prediction remains challenging due to...</span><span class="abstract-full" id="full-2604.15377" hidden>Accurate and timely rainfall nowcasting is crucial for disaster mitigation and water resource management. Despite recent advances in deep learning, precipitation prediction remains challenging due to limitations in effectively leveraging diverse multimedia data sources. We introduce M3R, a Meteorology-informed MultiModal attention-based architecture for direct Rainfall prediction that synergistically combines visual NEXRAD radar imagery with numerical Personal Weather Station (PWS) measurements, using a comprehensive pipeline for temporal alignment of heterogeneous meteorological data. With specialized multimodal attention mechanisms, M3R novelly leverages weather station time series as queries to selectively attend to spatial radar features, enabling focused extraction of precipitation signatures. Experimental results for three spatial areas of 100 km * 100 km centered at NEXRAD radar stations demonstrate that M3R outperforms existing approaches, achieving substantial improvements in accuracy, efficiency, and precipitation detection capabilities. Our work establishes new benchmarks for multimedia-based precipitation nowcasting and provides practical tools for operational weather prediction systems. The source code is available at https://github.com/Sanjeev97/M3Rain</span> <span class="abstract-toggle" data-id="2604.15377">more</span>

    [:material-file-document: 2604.15377](https://arxiv.org/abs/2604.15377v1) · [:fontawesome-brands-github:](https://github.com/Sanjeev97/M3Rain) · [:material-content-copy: BibTeX](bibtex/2604.15377.bib){ .bibtex-link }

    <span class="md-tag">transformer</span>

-   #### MAD-SmaAt-GNet: A Multimodal Advection-Guided Neural Network for Precipitation Nowcasting

    ---

    *Samuel van Wonderen, Siamak Mehrkanoon* · 2026

    <span class="abstract-snippet" id="snip-2603.04461">Precipitation nowcasting (short-term forecasting) is still often performed using numerical solvers for physical equations, which are computationally expensive and make limited use of the large...</span><span class="abstract-full" id="full-2603.04461" hidden>Precipitation nowcasting (short-term forecasting) is still often performed using numerical solvers for physical equations, which are computationally expensive and make limited use of the large volumes of available weather data. Deep learning models have shown strong potential for precipitation nowcasting, offering both accuracy and computational efficiency. Among these models, convolutional neural networks (CNNs) are particularly effective for image-to-image prediction tasks. The SmaAt-UNet is a lightweight CNN based architecture that has demonstrated strong performance for precipitation nowcasting. This paper introduces the Multimodal Advection-Guided Small Attention GNet (MAD-SmaAt-GNet), which extends the core SmaAt-UNet by (i) incorporating an additional encoder to learn from multiple weather variables and (ii) integrating a physics-based advection component to ensure physically consistent predictions. We show that each extension individually improves rainfall forecasts and that their combination yields further gains. MAD-SmaAt-GNet reduces the mean squared error (MSE) by 8.9% compared with the baseline SmaAt-UNet for four-step precipitation forecasting up to four hours ahead. Additionally, experiments indicate that multimodal inputs are particularly beneficial for short lead times, while the advection-based component enhances performance across both short and long forecasting horizons.</span> <span class="abstract-toggle" data-id="2603.04461">more</span>

    [:material-file-document: 2603.04461](https://arxiv.org/abs/2603.04461v1) · [:material-content-copy: BibTeX](bibtex/2603.04461.bib){ .bibtex-link }

    <span class="md-tag">CNN</span> <span class="md-tag">physics-informed</span>

-   #### WADEPre: A Wavelet-based Decomposition Model for Extreme Precipitation Nowcasting with Multi-Scale Learning

    ---

    *Baitian Liu, Haiping Zhang, Huiling Yuan, Dongjing Wang, Ying Li, Feng Chen, Hao Wu* · 2026

    <span class="abstract-snippet" id="snip-2602.02096">The heavy-tailed nature of precipitation intensity impedes precise precipitation nowcasting. Standard models that optimize pixel-wise losses are prone to regression-to-the-mean bias, which blurs...</span><span class="abstract-full" id="full-2602.02096" hidden>The heavy-tailed nature of precipitation intensity impedes precise precipitation nowcasting. Standard models that optimize pixel-wise losses are prone to regression-to-the-mean bias, which blurs extreme values. Existing Fourier-based methods also lack the spatial localization needed to resolve transient convective cells. To overcome these intrinsic limitations, we propose WADEPre, a wavelet-based decomposition model for extreme precipitation that transitions the modeling into the wavelet domain. By leveraging the Discrete Wavelet Transform for explicit decomposition, WADEPre employs a dual-branch architecture: an Approximation Network to model stable, low-frequency advection, isolating deterministic trends from statistical bias, and a spatially localized Detail Network to capture high-frequency stochastic convection, resolving transient singularities and preserving sharp boundaries. A subsequent Refiner module then dynamically reconstructs these decoupled multi-scale components into the final high-fidelity forecast. To address optimization instability, we introduce a multi-scale curriculum learning strategy that progressively shifts supervision from coarse scales to fine-grained details. Extensive experiments on the SEVIR and Shanghai Radar datasets demonstrate that WADEPre achieves state-of-the-art performance, yielding significant improvements in capturing extreme thresholds and maintaining structural fidelity. Our code is available at https://github.com/sonderlau/WADEPre.</span> <span class="abstract-toggle" data-id="2602.02096">more</span>

    [:material-file-document: 2602.02096](https://arxiv.org/abs/2602.02096v1) · [:fontawesome-brands-github:](https://github.com/sonderlau/WADEPre) · [:material-content-copy: BibTeX](bibtex/2602.02096.bib){ .bibtex-link }

-   #### StormDiT: A generative AI model bridges the 2-6 hour 'gray zone' in precipitation nowcasting

    ---

    *Haofei Sun, Yunfan Yang, Wei Han, Wei Huang, Huaguan Chen, Zhiqiu Gao, Zeting Li, Zhaoyang Huo et al.* · 2026

    <span class="abstract-snippet" id="snip-2601.20342">Accurate short-term warnings for extreme precipitation are critical for global disaster mitigation but are hindered by a persistent predictability barrier at the 2-6 hour horizon -- the "nowcasting...</span><span class="abstract-full" id="full-2601.20342" hidden>Accurate short-term warnings for extreme precipitation are critical for global disaster mitigation but are hindered by a persistent predictability barrier at the 2-6 hour horizon -- the "nowcasting gray zone." In this window, traditional observation-based extrapolation fails due to error accumulation, while numerical weather prediction is computationally too slow to resolve storm-scale dynamics. Recent generative AI approaches attempt to bridge this gap by decomposing precipitation into separate deterministic advection and stochastic diffusion components. However, this decomposition can sever fundamental causal links between entangled atmospheric processes, such as the dynamic initiation of convection triggered by boundary advection. Here we present StormDiT, a unified generative model that treats weather evolution as a holistic spatiotemporal problem, learning the coupled physics of the gray zone without human-imposed structural priors. Trained on a massive dataset of 7,720 precipitation events from China, our model achieves a breakthrough in long-horizon stability. On a heavy-rainfall test set, it maintains skillful prediction for strong convection ($\ge$ 35 dBZ) with a Critical Success Index (CSI) near 0.2 across the full 6-hour forecast at 6-minute resolution. Crucially, the model exhibits superior probabilistic calibration, accurately quantifying operational risks. On the public SEVIR benchmark, our unified paradigm more than doubles the state-of-the-art 1-hour performance for heavy rain and establishes the first robust baseline for 3-hour forecasting. Furthermore, interpretability analysis reveals that the model attends to non-local physical precursors, such as outflow boundaries, explicitly validating its emergent understanding of convective organization.</span> <span class="abstract-toggle" data-id="2601.20342">more</span>

    [:material-file-document: 2601.20342](https://arxiv.org/abs/2601.20342v1) · [:material-content-copy: BibTeX](bibtex/2601.20342.bib){ .bibtex-link }

    <span class="md-tag">probabilistic</span>

-   #### STLDM: Spatio-Temporal Latent Diffusion Model for Precipitation Nowcasting

    ---

    *Shi Quan Foo, Chi-Ho Wong, Zhihan Gao, Dit-Yan Yeung, Ka-Hing Wong, Wai-Kin Wong* · 2025

    <span class="abstract-snippet" id="snip-2512.21118">Precipitation nowcasting is a critical spatio-temporal prediction task for society to prevent severe damage owing to extreme weather events. Despite the advances in this field, the complex and...</span><span class="abstract-full" id="full-2512.21118" hidden>Precipitation nowcasting is a critical spatio-temporal prediction task for society to prevent severe damage owing to extreme weather events. Despite the advances in this field, the complex and stochastic nature of this task still poses challenges to existing approaches. Specifically, deterministic models tend to produce blurry predictions while generative models often struggle with poor accuracy. In this paper, we present a simple yet effective model architecture termed STLDM, a diffusion-based model that learns the latent representation from end to end alongside both the Variational Autoencoder and the conditioning network. STLDM decomposes this task into two stages: a deterministic forecasting stage handled by the conditioning network, and an enhancement stage performed by the latent diffusion model. Experimental results on multiple radar datasets demonstrate that STLDM achieves superior performance compared to the state of the art, while also improving inference efficiency. The code is available in https://github.com/sqfoo/stldm_official.</span> <span class="abstract-toggle" data-id="2512.21118">more</span>

    [:material-file-document: 2512.21118](https://arxiv.org/abs/2512.21118v1) · [:fontawesome-brands-github:](https://github.com/sqfoo/stldm_official) · [:material-content-copy: BibTeX](bibtex/2512.21118.bib){ .bibtex-link }

    <span class="md-tag">diffusion</span> <span class="md-tag">variational</span>

-   #### PIANO: Physics-informed Dual Neural Operator for Precipitation Nowcasting

    ---

    *Seokhyun Chin, Junghwan Park, Woojin Cho* · 2025

    <span class="abstract-snippet" id="snip-2512.01062">Precipitation nowcasting, key for early warning of disasters, currently relies on computationally expensive and restrictive methods that limit access to many countries. To overcome this challenge, we...</span><span class="abstract-full" id="full-2512.01062" hidden>Precipitation nowcasting, key for early warning of disasters, currently relies on computationally expensive and restrictive methods that limit access to many countries. To overcome this challenge, we propose precipitation nowcasting using satellite imagery with physics constraints for improved accuracy and physical consistency. We use a novel physics-informed dual neural operator (PIANO) structure to enforce the fundamental equation of advection-diffusion during training to predict satellite imagery using a PINN loss. Then, we use a generative model to convert satellite images to radar images, which are used for precipitation nowcasting. Compared to baseline models, our proposed model shows a notable improvement in moderate (4mm/h) precipitation event prediction alongside short-term heavy (8mm/h) precipitation event prediction. It also demonstrates low seasonal variability in predictions, indicating robustness for generalization. This study suggests the potential of the PIANO and serves as a good baseline for physics-informed precipitation nowcasting.</span> <span class="abstract-toggle" data-id="2512.01062">more</span>

    [:material-file-document: 2512.01062](https://arxiv.org/abs/2512.01062v1) · [:material-content-copy: BibTeX](bibtex/2512.01062.bib){ .bibtex-link }

    <span class="md-tag">physics-informed</span> <span class="md-tag">operator-learning</span>

-   #### FlowCast: Advancing Precipitation Nowcasting with Conditional Flow Matching

    ---

    *Bernardo Perrone Ribeiro, Jana Faganeli Pucer* · 2025

    <span class="abstract-snippet" id="snip-2511.09731">Radar-based precipitation nowcasting, the task of forecasting short-term precipitation fields from previous radar images, is a critical problem for flood risk management and decision-making. While...</span><span class="abstract-full" id="full-2511.09731" hidden>Radar-based precipitation nowcasting, the task of forecasting short-term precipitation fields from previous radar images, is a critical problem for flood risk management and decision-making. While deep learning has substantially advanced this field, two challenges remain fundamental: the uncertainty of atmospheric dynamics and the efficient modeling of high-dimensional data. Diffusion models have shown strong promise by producing sharp, reliable forecasts, but their iterative sampling process is computationally prohibitive for time-critical applications. We introduce FlowCast, the first model to apply Conditional Flow Matching (CFM) to precipitation nowcasting. Unlike diffusion, CFM learns a direct noise-to-data mapping, enabling rapid, high-fidelity sample generation with drastically fewer function evaluations. Our experiments demonstrate that FlowCast establishes a new state-of-the-art in predictive accuracy. A direct comparison further reveals the CFM objective is both more accurate and significantly more efficient than a diffusion objective on the same architecture, maintaining high performance with significantly fewer sampling steps. This work positions CFM as a powerful and practical alternative for high-dimensional spatiotemporal forecasting.</span> <span class="abstract-toggle" data-id="2511.09731">more</span>

    [:material-file-document: 2511.09731](https://arxiv.org/abs/2511.09731v1) · [:material-content-copy: BibTeX](bibtex/2511.09731.bib){ .bibtex-link }

    <span class="md-tag">diffusion</span>

-   #### Nowcast3D: Reliable precipitation nowcasting via gray-box learning

    ---

    *Huaguan Chen, Wei Han, Haofei Sun, Ning Lin, Xingtao Song, Yunfan Yang, Jie Tian, Yang Liu et al.* · 2025

    <span class="abstract-snippet" id="snip-2511.04659">Extreme precipitation nowcasting demands high spatiotemporal fidelity and extended lead times, yet existing approaches remain limited. Numerical Weather Prediction (NWP) and its deep-learning...</span><span class="abstract-full" id="full-2511.04659" hidden>Extreme precipitation nowcasting demands high spatiotemporal fidelity and extended lead times, yet existing approaches remain limited. Numerical Weather Prediction (NWP) and its deep-learning emulations are too slow and coarse for rapidly evolving convection, while extrapolation and purely data-driven models suffer from error accumulation and excessive smoothing. Hybrid 2D radar-based methods discard crucial vertical information, preventing accurate reconstruction of height-dependent dynamics. We introduce a gray-box, fully three-dimensional nowcasting framework that directly processes volumetric radar reflectivity and couples physically constrained neural operators with datadriven learning. The model learns vertically varying 3D advection fields under a conservative advection operator, parameterizes spatially varying diffusion, and introduces a Brownian-motion--inspired stochastic term to represent unresolved motions. A residual branch captures small-scale convective initiation and microphysical variability, while a diffusion-based stochastic module estimates uncertainty. The framework achieves more accurate forecasts up to three-hour lead time across precipitation regimes and ranked first in 57\% of cases in a blind evaluation by 160 meteorologists. By restoring full 3D dynamics with physical consistency, it offers a scalable and robust pathway for skillful and reliable nowcasting of extreme precipitation.</span> <span class="abstract-toggle" data-id="2511.04659">more</span>

    [:material-file-document: 2511.04659](https://arxiv.org/abs/2511.04659v1) · [:material-content-copy: BibTeX](bibtex/2511.04659.bib){ .bibtex-link }

    <span class="md-tag">operator-learning</span>

-   #### RainDiff: End-to-end Precipitation Nowcasting Via Token-wise Attention Diffusion

    ---

    *Thao Nguyen, Jiaqi Ma, Fahad Shahbaz Khan, Souhaib Ben Taieb, Salman Khan* · 2025

    <span class="abstract-snippet" id="snip-2510.14962">Precipitation nowcasting, predicting future radar echo sequences from current observations, is a critical yet challenging task due to the inherently chaotic and tightly coupled spatio-temporal...</span><span class="abstract-full" id="full-2510.14962" hidden>Precipitation nowcasting, predicting future radar echo sequences from current observations, is a critical yet challenging task due to the inherently chaotic and tightly coupled spatio-temporal dynamics of the atmosphere. While recent advances in diffusion-based models attempt to capture both large-scale motion and fine-grained stochastic variability, they often suffer from scalability issues: latent-space approaches require a separately trained autoencoder, adding complexity and limiting generalization, while pixel-space approaches are computationally intensive and often omit attention mechanisms, reducing their ability to model long-range spatio-temporal dependencies. To address these limitations, we propose a Token-wise Attention integrated into not only the U-Net diffusion model but also the spatio-temporal encoder that dynamically captures multi-scale spatial interactions and temporal evolution. Unlike prior approaches, our method natively integrates attention into the architecture without incurring the high resource cost typical of pixel-space diffusion, thereby eliminating the need for separate latent modules. Our extensive experiments and visual evaluations across diverse datasets demonstrate that the proposed method significantly outperforms state-of-the-art approaches, yielding superior local fidelity, generalization, and robustness in complex precipitation forecasting scenarios.</span> <span class="abstract-toggle" data-id="2510.14962">more</span>

    [:material-file-document: 2510.14962](https://arxiv.org/abs/2510.14962v1) · [:material-content-copy: BibTeX](bibtex/2510.14962.bib){ .bibtex-link }

    <span class="md-tag">transformer</span> <span class="md-tag">diffusion</span> <span class="md-tag">CNN</span>

-   #### SimCast: Enhancing Precipitation Nowcasting with Short-to-Long Term Knowledge Distillation

    ---

    *Yifang Yin, Shengkai Chen, Yiyao Li, Lu Wang, Ruibing Jin, Wei Cui, Shili Xiang* · 2025

    <span class="abstract-snippet" id="snip-2510.07953">Precipitation nowcasting predicts future radar sequences based on current observations, which is a highly challenging task driven by the inherent complexity of the Earth system. Accurate nowcasting...</span><span class="abstract-full" id="full-2510.07953" hidden>Precipitation nowcasting predicts future radar sequences based on current observations, which is a highly challenging task driven by the inherent complexity of the Earth system. Accurate nowcasting is of utmost importance for addressing various societal needs, including disaster management, agriculture, transportation, and energy optimization. As a complementary to existing non-autoregressive nowcasting approaches, we investigate the impact of prediction horizons on nowcasting models and propose SimCast, a novel training pipeline featuring a short-to-long term knowledge distillation technique coupled with a weighted MSE loss to prioritize heavy rainfall regions. Improved nowcasting predictions can be obtained without introducing additional overhead during inference. As SimCast generates deterministic predictions, we further integrate it into a diffusion-based framework named CasCast, leveraging the strengths from probabilistic models to overcome limitations such as blurriness and distribution shift in deterministic outputs. Extensive experimental results on three benchmark datasets validate the effectiveness of the proposed framework, achieving mean CSI scores of 0.452 on SEVIR, 0.474 on HKO-7, and 0.361 on MeteoNet, which outperforms existing approaches by a significant margin.</span> <span class="abstract-toggle" data-id="2510.07953">more</span>

    [:material-file-document: 2510.07953](https://arxiv.org/abs/2510.07953v1) · [:material-content-copy: BibTeX](bibtex/2510.07953.bib){ .bibtex-link }

    <span class="md-tag">probabilistic</span>

-   #### Probability calibration for precipitation nowcasting

    ---

    *Lauri Kurki, Yaniel Cabrera, Samu Karanko* · 2025

    <span class="abstract-snippet" id="snip-2510.00594">Reliable precipitation nowcasting is critical for weather-sensitive decision-making, yet neural weather models (NWMs) can produce poorly calibrated probabilistic forecasts. Standard calibration...</span><span class="abstract-full" id="full-2510.00594" hidden>Reliable precipitation nowcasting is critical for weather-sensitive decision-making, yet neural weather models (NWMs) can produce poorly calibrated probabilistic forecasts. Standard calibration metrics such as the expected calibration error (ECE) fail to capture miscalibration across precipitation thresholds. We introduce the expected thresholded calibration error (ETCE), a new metric that better captures miscalibration in ordered classes like precipitation amounts. We extend post-processing techniques from computer vision to the forecasting domain. Our results show that selective scaling with lead time conditioning reduces model miscalibration without reducing the forecast quality.</span> <span class="abstract-toggle" data-id="2510.00594">more</span>

    [:material-file-document: 2510.00594](https://arxiv.org/abs/2510.00594v1) · [:material-content-copy: BibTeX](bibtex/2510.00594.bib){ .bibtex-link }

    <span class="md-tag">probabilistic</span>

-   #### Skilful Precipitation Nowcasting Using NowcastNet

    ---

    *Ajitabh Kumar* · 2023

    <span class="abstract-snippet" id="snip-2311.17961">Designing early warning system for precipitation requires accurate short-term forecasting system. Climate change has led to an increase in frequency of extreme weather events, and hence such systems...</span><span class="abstract-full" id="full-2311.17961" hidden>Designing early warning system for precipitation requires accurate short-term forecasting system. Climate change has led to an increase in frequency of extreme weather events, and hence such systems can prevent disasters and loss of life. Managing such events remain a challenge for both public and private institutions. Precipitation nowcasting can help relevant institutions to better prepare for such events as they impact agriculture, transport, public health and safety, etc. Physics-based numerical weather prediction (NWP) is unable to perform well for nowcasting because of large computational turn-around time. Deep-learning based models on the other hand are able to give predictions within seconds. We use recently proposed NowcastNet, a physics-conditioned deep generative network, to forecast precipitation for different regions of Europe using satellite images. Both spatial and temporal transfer learning is done by forecasting for the unseen regions and year. Model makes realistic predictions and is able to outperform baseline for such a prediction task.</span> <span class="abstract-toggle" data-id="2311.17961">more</span>

    [:material-file-document: 2311.17961](https://arxiv.org/abs/2311.17961v2) · [:material-content-copy: BibTeX](bibtex/2311.17961.bib){ .bibtex-link }

    <span class="md-tag">physics-informed</span>

-   #### MetNet: A Neural Weather Model for Precipitation Forecasting

    ---

    *Casper Kaae Sønderby, Lasse Espeholt, Jonathan Heek, Mostafa Dehghani, Avital Oliver, Tim Salimans et al.* · 2020

    <span class="abstract-snippet" id="snip-2003.12140">Weather forecasting is a long standing scientific challenge with direct social and economic impact. The task is suitable for deep neural networks due to vast amounts of continuously collected data...</span><span class="abstract-full" id="full-2003.12140" hidden>Weather forecasting is a long standing scientific challenge with direct social and economic impact. The task is suitable for deep neural networks due to vast amounts of continuously collected data and a rich spatial and temporal structure that presents long range dependencies. We introduce MetNet, a neural network that forecasts precipitation up to 8 hours into the future at the high spatial resolution of 1 km$^2$ and at the temporal resolution of 2 minutes with a latency in the order of seconds. MetNet takes as input radar and satellite data and forecast lead time and produces a probabilistic precipitation map. The architecture uses axial self-attention to aggregate the global context from a large input patch corresponding to a million square kilometers. We evaluate the performance of MetNet at various precipitation thresholds and find that MetNet outperforms Numerical Weather Prediction at forecasts of up to 7 to 8 hours on the scale of the continental United States.</span> <span class="abstract-toggle" data-id="2003.12140">more</span>

    [:material-file-document: 2003.12140](https://arxiv.org/abs/2003.12140v2) · [:material-content-copy: BibTeX](bibtex/2003.12140.bib){ .bibtex-link }

    <span class="md-tag">transformer</span> <span class="md-tag">probabilistic</span>

</div>

## Downscaling (14)

<div class="grid cards" markdown>

-   #### CORDEX-ML-Bench: A Benchmark for Data-Driven Regional Climate Downscaling -Experiment Design and Overview

    ---

    *Neelesh Rampal, José González-Abad, Henry Addison, Jorge Baño-Medina, Maria Laura Bettolli et al.* · 2026

    <span class="abstract-snippet" id="snip-2606.29172">Machine learning (ML) has emerged as a cost-effective approach to complement dynamical downscaling for producing high-resolution regional climate projections. However, the absence of standardised...</span><span class="abstract-full" id="full-2606.29172" hidden>Machine learning (ML) has emerged as a cost-effective approach to complement dynamical downscaling for producing high-resolution regional climate projections. However, the absence of standardised training and evaluation protocols, applied consistently across multiple domains, continues to hinder meaningful model intercomparison. We introduce CORDEX-ML-Bench, a benchmark aligned with CORDEX, which constitutes the first phase of a community initiative to advance data-driven downscaling toward operational readiness, and complement future dynamical downscaling efforts under CMIP7. The framework targets downscaled daily maximum temperature and precipitation to ~10 km resolution (20x increase) across three pilot regions; European Alps, New Zealand, and Southern Africa. Using a perfect-model experimental design, we evaluate 40 ML configurations developed independently, spanning traditional ML, convolutional U-Nets, vision transformers, graph neural networks, and generative models based on diffusion, flow matching, and generative adversarial networks. Models are trained under two experimental periods, an empirical-statistical downscaling pseudo-reality (historical period only) and Emulator (historical and future periods) -and are evaluated against a core set of metrics developed specifically for assessing downscaling skill. Generative models consistently outperform deterministic approaches for precipitation, better capturing fine-scale variability and extremes. For temperature, the generative advantage narrows and deterministic architectures remain competitive. Models trained solely on the historical period systematically underestimate future climate-change signals while those additionally trained on a future period perform better. These findings raise concerns about historically trained models widely used in an operational setting, underscoring the need for rigorous extrapolation testing.</span> <span class="abstract-toggle" data-id="2606.29172">more</span>

    [:material-file-document: 2606.29172](https://arxiv.org/abs/2606.29172v1) · [:material-content-copy: BibTeX](bibtex/2606.29172.bib){ .bibtex-link }

    <span class="md-tag">transformer</span> <span class="md-tag">diffusion</span> <span class="md-tag">GAN</span> <span class="md-tag">CNN</span> <span class="md-tag">GNN</span>

-   #### Temporal Coverage over Density: Parsimonious Training-Set Design for ML Climate Downscaling

    ---

    *Karandeep Singh, Stefan Rahimi, Chad W. Thackeray, Stephen Cropper, Alex Hall* · 2026

    <span class="abstract-snippet" id="snip-2606.07898">High-resolution regional climate simulations provide critical information for climate impacts assessments but remain computationally expensive, motivating the development of machine-learning...</span><span class="abstract-full" id="full-2606.07898" hidden>High-resolution regional climate simulations provide critical information for climate impacts assessments but remain computationally expensive, motivating the development of machine-learning downscalers and emulators. A key challenge is determining how limited high-resolution simulations should be distributed across a changing climate trajectory to capture both forced climate response and internal variability. Using the CESM2 Large Ensemble over the western United States, we compare three training-year selection strategies under fixed data budgets: a contiguous block of historical years, years drawn from both the beginning and end of the simulation period, and years distributed throughout the full climate trajectory. Including both historical and future years consistently outperforms training on historical years alone, demonstrating the importance of exposing downscaling models to climate states outside the historical record and highlighting limitations of stationarity assumptions common in statistical downscaling. Training on years distributed throughout the full climate trajectory performs best overall, indicating that broad sampling of internal variability provides additional information beyond exposure to the forced climate response alone. Models trained on temporally distributed subsets more successfully reproduce variability in unseen ensemble members while retaining strong performance across a wide range of climate diagnostics. Even when trained on only one-tenth of the available high-resolution years, temporally distributed models remain highly competitive with full-data training. These results suggest that, under fixed computational budgets, broad sampling of climate states is more valuable than temporal continuity when allocating scarce high-resolution simulations. The findings provide practical guidance for regional climate downscaling and large-ensemble projection workflows.</span> <span class="abstract-toggle" data-id="2606.07898">more</span>

    [:material-file-document: 2606.07898](https://arxiv.org/abs/2606.07898v1) · [:material-content-copy: BibTeX](bibtex/2606.07898.bib){ .bibtex-link }

-   #### Generative climate downscaling enables high-resolution compound risk assessment by preserving multivariate dependencies

    ---

    *Takuro Kutsuna, Noriko N. Ishizaki, Norihiro Oyama, Hiroaki Yoshida* · 2026

    <span class="abstract-snippet" id="snip-2605.11531">Physics-based climate projections using general circulation models are essential for assessing future risks, but their coarse resolution limits regional decision-making. Statistical downscaling can...</span><span class="abstract-full" id="full-2605.11531" hidden>Physics-based climate projections using general circulation models are essential for assessing future risks, but their coarse resolution limits regional decision-making. Statistical downscaling can efficiently add detail, yet many methods treat variables independently, degrading inter-variable relationships that govern compound hazards such as heat stress, drought, and wildfire. Here we show that a diffusion-based multivariate generative framework, combined with bias correction, recovers degraded inter-variable correlations even under a 50$\times$ increase in linear resolution. When applied to five meteorological variables over Japan, the framework reduces inter-variable correlation errors by more than fourfold relative to existing baselines while improving both univariate and spatial accuracy, leading to more accurate detection of severe drought. These results demonstrate that multivariate generative downscaling improves the reliability of compound risk assessment under large resolution gaps.</span> <span class="abstract-toggle" data-id="2605.11531">more</span>

    [:material-file-document: 2605.11531](https://arxiv.org/abs/2605.11531v1) · [:material-content-copy: BibTeX](bibtex/2605.11531.bib){ .bibtex-link }

    <span class="md-tag">physics-informed</span>

-   #### Generative 3D Gaussian Splatting for Arbitrary-ResolutionAtmospheric Downscaling and Forecasting

    ---

    *Tao Hana, Zhibin Wen, Zhenghao Chen, Fenghua Lin, Junyu Gao, Song Guo, Lei Bai* · 2026

    <span class="abstract-snippet" id="snip-2604.07928">While AI-based numerical weather prediction (NWP) enables rapid forecasting, generating high-resolution outputs remains computationally demanding due to limited multi-scale adaptability and...</span><span class="abstract-full" id="full-2604.07928" hidden>While AI-based numerical weather prediction (NWP) enables rapid forecasting, generating high-resolution outputs remains computationally demanding due to limited multi-scale adaptability and inefficient data representations. We propose the 3D Gaussian splatting-based scale-aware vision transformer (GSSA-ViT), a novel framework for arbitrary-resolution forecasting and flexible downscaling of high-dimensional atmospheric fields. Specifically, latitude-longitude grid points are treated as centers of 3D Gaussians. A generative 3D Gaussian prediction scheme is introduced to estimate key parameters, including covariance, attributes, and opacity, for unseen samples, improving generalization and mitigating overfitting. In addition, a scale-aware attention module is designed to capture cross-scale dependencies, enabling the model to effectively integrate information across varying downscaling ratios and support continuous resolution adaptation. To our knowledge, this is the first NWP approach that combines generative 3D Gaussian modeling with scale-aware attention for unified multi-scale prediction. Experiments on ERA5 show that the proposed method accurately forecasts 87 atmospheric variables at arbitrary resolutions, while evaluations on ERA5 and CMIP6 demonstrate its superior performance in downscaling tasks. The proposed framework provides an efficient and scalable solution for high-resolution, multi-scale atmospheric prediction and downscaling. Code is available at: https://github.com/binbin2xs/weather-GS.</span> <span class="abstract-toggle" data-id="2604.07928">more</span>

    [:material-file-document: 2604.07928](https://arxiv.org/abs/2604.07928v1) · [:fontawesome-brands-github:](https://github.com/binbin2xs/weather-GS) · [:material-content-copy: BibTeX](bibtex/2604.07928.bib){ .bibtex-link }

    <span class="md-tag">transformer</span>

-   #### 30-meter Land Surface Temperature from Landsat via Progressive Self-Training Downscaling

    ---

    *Huanfeng Shen, Chan Li, Menghui Jiang, Penghai Wu, Guanhao Zhang, Tian Xie* · 2026

    <span class="abstract-snippet" id="snip-2603.29478">Land surface temperature (LST) is a critical parameter for characterizing surface energy balance and hydrothermal processes. While Landsat provides invaluable LST observations at medium spatial...</span><span class="abstract-full" id="full-2603.29478" hidden>Land surface temperature (LST) is a critical parameter for characterizing surface energy balance and hydrothermal processes. While Landsat provides invaluable LST observations at medium spatial resolution for over 40 years, its native spatial resolution of thermal bands (e.g., 100 m) remains insufficient compared to its 30 m optical bands, failing to meet the demands of fine-scale studies. To address this issues, this study proposes a progressive self-training framework for downscaling Landsat LST to 30 m without relying on fine-scale ground truth, while maintaining minimal data dependence. The framework progressively optimizes a cross-modal fusion network to refine thermal details in a coarse-to-fine manner, characterized by one pre-training and two fine-tuning stages. Spatial validation against SDGSAT-1 30 m LST and temporal validation using in situ measurements confirm its reliability and accuracy, with both station-averaged MAE and RMSE outperforming the official cubic product by approximately 0.4 K. Further performance comparison experiments demonstrate that the proposed framework consistently reconstructs coherent fine-scale thermal patterns while preserving spatial heterogeneity. Multi spatial resolution evaluations and ablation studies verify the effectiveness of the proposed strategy and network design. Overall, the framework provides a stable pathway for enhancing the spatial resolution of Landsat LST, providing fine-resolution data support for fine-scale surface process studies and localized environmental monitoring.</span> <span class="abstract-toggle" data-id="2603.29478">more</span>

    [:material-file-document: 2603.29478](https://arxiv.org/abs/2603.29478v1) · [:material-content-copy: BibTeX](bibtex/2603.29478.bib){ .bibtex-link }

-   #### Downscaling land surface temperature data using edge detection and block-diagonal Gaussian process regression

    ---

    *Sanjit Dandapanthula, Margaret Johnson, Madeleine Pascolini-Campbell, Glynn Hulley, Mikael Kuusela* · 2026

    <span class="abstract-snippet" id="snip-2602.02813">Accurate and high-resolution estimation of land surface temperature (LST) is crucial in estimating evapotranspiration, a measure of plant water use and a central quantity in agricultural...</span><span class="abstract-full" id="full-2602.02813" hidden>Accurate and high-resolution estimation of land surface temperature (LST) is crucial in estimating evapotranspiration, a measure of plant water use and a central quantity in agricultural applications. In this work, we develop a novel statistical method for downscaling LST data obtained from NASA's ECOSTRESS mission, using high-resolution data from the Landsat 8 mission as a proxy for modeling agricultural field structure. Using the Landsat data, we identify the boundaries of agricultural fields through edge detection techniques, allowing us to capture the inherent block structure present in the spatial domain. We propose a block-diagonal Gaussian process (BDGP) model that captures the spatial structure of the agricultural fields, leverages independence of LST across fields for computational tractability, and accounts for the change of support present in ECOSTRESS observations. We use the resulting BDGP model to perform Gaussian process regression and obtain high-resolution estimates of LST from ECOSTRESS data, along with uncertainty quantification. Our results demonstrate the practicality of the proposed method in producing reliable high-resolution LST estimates, with potential applications in agriculture, urban planning, and climate studies.</span> <span class="abstract-toggle" data-id="2602.02813">more</span>

    [:material-file-document: 2602.02813](https://arxiv.org/abs/2602.02813v1) · [:material-content-copy: BibTeX](bibtex/2602.02813.bib){ .bibtex-link }

    <span class="md-tag">probabilistic</span>

-   #### Zero-Shot Statistical Downscaling via Diffusion Posterior Sampling

    ---

    *Ruian Tie, Wenbo Xiong, Zhengyu Shi, Xinyu Su, Chenyu jiang, Libo Wu, Hao Li* · 2026

    <span class="abstract-snippet" id="snip-2601.21760">Conventional supervised climate downscaling struggles to generalize to Global Climate Models (GCMs) due to the lack of paired training data and inherent domain gaps relative to reanalysis. Meanwhile,...</span><span class="abstract-full" id="full-2601.21760" hidden>Conventional supervised climate downscaling struggles to generalize to Global Climate Models (GCMs) due to the lack of paired training data and inherent domain gaps relative to reanalysis. Meanwhile, current zero-shot methods suffer from physical inconsistencies and vanishing gradient issues under large scaling factors. We propose Zero-Shot Statistical Downscaling (ZSSD), a zero-shot framework that performs statistical downscaling without paired data during training. ZSSD leverages a Physics-Consistent Climate Prior learned from reanalysis data, conditioned on geophysical boundaries and temporal information to enforce physical validity. Furthermore, to enable robust inference across varying GCMs, we introduce Unified Coordinate Guidance. This strategy addresses the vanishing gradient problem in vanilla DPS and ensures consistency with large-scale fields. Results show that ZSSD significantly outperforms existing zero-shot baselines in 99th percentile errors and successfully reconstructs complex weather events, such as tropical cyclones, across heterogeneous GCMs.</span> <span class="abstract-toggle" data-id="2601.21760">more</span>

    [:material-file-document: 2601.21760](https://arxiv.org/abs/2601.21760v1) · [:material-content-copy: BibTeX](bibtex/2601.21760.bib){ .bibtex-link }

-   #### Time-aware UNet and super-resolution deep residual networks for spatial downscaling

    ---

    *Mika Sipilä, Sabrina Maggio, Sandra De Iaco, Klaus Nordhausen, Monica Palma, Sara Taskinen* · 2025

    <span class="abstract-snippet" id="snip-2512.13753">Satellite data of atmospheric pollutants are often available only at coarse spatial resolution, limiting their applicability in local-scale environmental analysis and decision-making. Spatial...</span><span class="abstract-full" id="full-2512.13753" hidden>Satellite data of atmospheric pollutants are often available only at coarse spatial resolution, limiting their applicability in local-scale environmental analysis and decision-making. Spatial downscaling methods aim to transform the coarse satellite data into high-resolution fields. In this work, two widely used deep learning architectures, the super-resolution deep residual network (SRDRN) and the encoder-decoder-based UNet, are considered for spatial downscaling of tropospheric ozone. Both methods are extended with a lightweight temporal module, which encodes observation time using either sinusoidal or radial basis function (RBF) encoding, and fuses the temporal features with the spatial representations in the networks. The proposed time-aware extensions are evaluated against their baseline counterparts in a case study on ozone downscaling over Italy. The results suggest that, while only slightly increasing computational complexity, the temporal modules significantly improve downscaling performance and convergence speed.</span> <span class="abstract-toggle" data-id="2512.13753">more</span>

    [:material-file-document: 2512.13753](https://arxiv.org/abs/2512.13753v1) · [:material-content-copy: BibTeX](bibtex/2512.13753.bib){ .bibtex-link }

    <span class="md-tag">CNN</span>

-   #### On Global Applicability and Location Transferability of Generative Deep Learning Models for Precipitation Downscaling

    ---

    *Paula Harder, Christian Lessig, Matthew Chantry, Francis Pelletier, David Rolnick* · 2025

    <span class="abstract-snippet" id="snip-2512.01400">Deep learning offers promising capabilities for the statistical downscaling of climate and weather forecasts, with generative approaches showing particular success in capturing fine-scale...</span><span class="abstract-full" id="full-2512.01400" hidden>Deep learning offers promising capabilities for the statistical downscaling of climate and weather forecasts, with generative approaches showing particular success in capturing fine-scale precipitation patterns. However, most existing models are region-specific, and their ability to generalize to unseen geographic areas remains largely unexplored. In this study, we evaluate the generalization performance of generative downscaling models across diverse regions. Using a global framework, we employ ERA5 reanalysis data as predictors and IMERG precipitation estimates at $0.1^\circ$ resolution as targets. A hierarchical location-based data split enables a systematic assessment of model performance across 15 regions around the world.</span> <span class="abstract-toggle" data-id="2512.01400">more</span>

    [:material-file-document: 2512.01400](https://arxiv.org/abs/2512.01400v1) · [:material-content-copy: BibTeX](bibtex/2512.01400.bib){ .bibtex-link }

-   #### A PDE-Informed Latent Diffusion Model for 2-m Temperature Downscaling

    ---

    *Paul Rosu, Muchang Bahng, Erick Jiang, Rico Zhu, Vahid Tarokh* · 2025

    <span class="abstract-snippet" id="snip-2510.23866">This work presents a physics-conditioned latent diffusion model tailored for dynamical downscaling of atmospheric data, with a focus on reconstructing high-resolution 2-m temperature fields. Building...</span><span class="abstract-full" id="full-2510.23866" hidden>This work presents a physics-conditioned latent diffusion model tailored for dynamical downscaling of atmospheric data, with a focus on reconstructing high-resolution 2-m temperature fields. Building upon a pre-existing diffusion architecture and employing a residual formulation against a reference UNet, we integrate a partial differential equation (PDE) loss term into the model's training objective. The PDE loss is computed in the full resolution (pixel) space by decoding the latent representation and is designed to enforce physical consistency through a finite-difference approximation of an effective advection-diffusion balance. Empirical observations indicate that conventional diffusion training already yields low PDE residuals, and we investigate how fine-tuning with this additional loss further regularizes the model and enhances the physical plausibility of the generated fields. The entirety of our codebase is available on Github, for future reference and development.</span> <span class="abstract-toggle" data-id="2510.23866">more</span>

    [:material-file-document: 2510.23866](https://arxiv.org/abs/2510.23866v1) · [:material-content-copy: BibTeX](bibtex/2510.23866.bib){ .bibtex-link }

    <span class="md-tag">diffusion</span> <span class="md-tag">CNN</span>

-   #### Sparse Local Implicit Image Function for sub-km Weather Downscaling

    ---

    *Yago del Valle Inclan Redondo, Enrique Arriaga-Varela, Dmitry Lyamzin, Pablo Cervantes et al.* · 2025

    <span class="abstract-snippet" id="snip-2510.20228">We introduce SpLIIF to generate implicit neural representations and enable arbitrary downscaling of weather variables. We train a model from sparse weather stations and topography over Japan and...</span><span class="abstract-full" id="full-2510.20228" hidden>We introduce SpLIIF to generate implicit neural representations and enable arbitrary downscaling of weather variables. We train a model from sparse weather stations and topography over Japan and evaluate in- and out-of-distribution accuracy predicting temperature and wind, comparing it to both an interpolation baseline and CorrDiff. We find the model to be up to 50% better than both CorrDiff and the baseline at downscaling temperature, and around 10-20% better for wind.</span> <span class="abstract-toggle" data-id="2510.20228">more</span>

    [:material-file-document: 2510.20228](https://arxiv.org/abs/2510.20228v1) · [:material-content-copy: BibTeX](bibtex/2510.20228.bib){ .bibtex-link }

-   #### Assessing the Geographic Generalization and Physical Consistency of Generative Models for Climate Downscaling

    ---

    *Carlo Saccardi, Maximilian Pierzyna, Haitz Sáez de Ocáriz Borde, Simone Monaco, Cristian Meo et al.* · 2025

    <span class="abstract-snippet" id="snip-2510.13722">Kilometer-scale weather data is crucial for real-world applications but remains computationally intensive to produce using traditional weather simulations. An emerging solution is to use deep...</span><span class="abstract-full" id="full-2510.13722" hidden>Kilometer-scale weather data is crucial for real-world applications but remains computationally intensive to produce using traditional weather simulations. An emerging solution is to use deep learning models, which offer a faster alternative for climate downscaling. However, their reliability is still in question, as they are often evaluated using standard machine learning metrics rather than insights from atmospheric and weather physics. This paper benchmarks recent state-of-the-art deep learning models and introduces physics-inspired diagnostics to evaluate their performance and reliability, with a particular focus on geographic generalization and physical consistency. Our experiments show that, despite the seemingly strong performance of models such as CorrDiff, when trained on a limited set of European geographies (e.g., central Europe), they struggle to generalize to other regions such as Iberia, Morocco in the south, or Scandinavia in the north. They also fail to accurately capture second-order variables such as divergence and vorticity derived from predicted velocity fields. These deficiencies appear even in in-distribution geographies, indicating challenges in producing physically consistent predictions. We propose a simple initial solution: introducing a power spectral density loss function that empirically improves geographic generalization by encouraging the reconstruction of small-scale physical structures. The code for reproducing the experimental results can be found at https://github.com/CarloSaccardi/PSD-Downscaling</span> <span class="abstract-toggle" data-id="2510.13722">more</span>

    [:material-file-document: 2510.13722](https://arxiv.org/abs/2510.13722v1) · [:fontawesome-brands-github:](https://github.com/CarloSaccardi/PSD-Downscaling) · [:material-content-copy: BibTeX](bibtex/2510.13722.bib){ .bibtex-link }

-   #### Km-scale dynamical downscaling through conformalized latent diffusion models

    ---

    *Alessandro Brusaferri, Andrea Ballarino* · 2025

    <span class="abstract-snippet" id="snip-2510.13301">Dynamical downscaling is crucial for deriving high-resolution meteorological fields from coarse-scale simulations, enabling detailed analysis for critical applications such as weather forecasting and...</span><span class="abstract-full" id="full-2510.13301" hidden>Dynamical downscaling is crucial for deriving high-resolution meteorological fields from coarse-scale simulations, enabling detailed analysis for critical applications such as weather forecasting and renewable energy modeling. Generative Diffusion models (DMs) have recently emerged as powerful data-driven tools for this task, offering reconstruction fidelity and more scalable sampling supporting uncertainty quantification. However, DMs lack finite-sample guarantees against overconfident predictions, resulting in miscalibrated grid-point-level uncertainty estimates hindering their reliability in operational contexts. In this work, we tackle this issue by augmenting the downscaling pipeline with a conformal prediction framework. Specifically, the DM's samples are post-processed to derive conditional quantile estimates, incorporated into a conformalized quantile regression procedure targeting locally adaptive prediction intervals with finite-sample marginal validity. The proposed approach is evaluated on ERA5 reanalysis data over Italy, downscaled to a 2-km grid. Results demonstrate grid-point-level uncertainty estimates with markedly improved coverage and stable probabilistic scores relative to the DM baseline, highlighting the potential of conformalized generative models for more trustworthy probabilistic downscaling to high-resolution meteorological fields.</span> <span class="abstract-toggle" data-id="2510.13301">more</span>

    [:material-file-document: 2510.13301](https://arxiv.org/abs/2510.13301v1) · [:material-content-copy: BibTeX](bibtex/2510.13301.bib){ .bibtex-link }

    <span class="md-tag">diffusion</span> <span class="md-tag">probabilistic</span>

-   #### MambaDS: Near-Surface Meteorological Field Downscaling with Topography Constrained Selective State Space Modeling

    ---

    *Zili Liu, Hao Chen, Lei Bai, Wenyuan Li, Wanli Ouyang, Zhengxia Zou, Zhenwei Shi* · 2024

    <span class="abstract-snippet" id="snip-2408.10854">In an era of frequent extreme weather and global warming, obtaining precise, fine-grained near-surface weather forecasts is increasingly essential for human activities. Downscaling (DS), a crucial...</span><span class="abstract-full" id="full-2408.10854" hidden>In an era of frequent extreme weather and global warming, obtaining precise, fine-grained near-surface weather forecasts is increasingly essential for human activities. Downscaling (DS), a crucial task in meteorological forecasting, enables the reconstruction of high-resolution meteorological states for target regions from global-scale forecast results. Previous downscaling methods, inspired by CNN and Transformer-based super-resolution models, lacked tailored designs for meteorology and encountered structural limitations. Notably, they failed to efficiently integrate topography, a crucial prior in the downscaling process. In this paper, we address these limitations by pioneering the selective state space model into the meteorological field downscaling and propose a novel model called MambaDS. This model enhances the utilization of multivariable correlations and topography information, unique challenges in the downscaling process while retaining the advantages of Mamba in long-range dependency modeling and linear computational complexity. Through extensive experiments in both China mainland and the continental United States (CONUS), we validated that our proposed MambaDS achieves state-of-the-art results in three different types of meteorological field downscaling settings. We will release the code subsequently.</span> <span class="abstract-toggle" data-id="2408.10854">more</span>

    [:material-file-document: 2408.10854](https://arxiv.org/abs/2408.10854v1) · [:material-content-copy: BibTeX](bibtex/2408.10854.bib){ .bibtex-link }

    <span class="md-tag">transformer</span> <span class="md-tag">CNN</span>

</div>

## Data Assimilation (26)

<div class="grid cards" markdown>

-   #### Deep-Learned Observation Operators for Artificial Intelligence Weather Forecasting Models

    ---

    *Kelsey Lieberman, Laura Slivinski, Matt Bender, Chris Miller, Josh DaRosa, Nick Krall et al.* · 2026

    <span class="abstract-snippet" id="snip-2604.00082">Satellite observation operators play an essential role in atmospheric data assimilation by translating model state variables into observation space. Previous work has shown that deep-learned...</span><span class="abstract-full" id="full-2604.00082" hidden>Satellite observation operators play an essential role in atmospheric data assimilation by translating model state variables into observation space. Previous work has shown that deep-learned emulators can effectively predict the outputs of classic observation operators, like the Community Radiative Transfer Model (CRTM), with reduced inference time. This study expands previous work to show the potential for integrating observation operators into artificial intelligence (AI) weather forecasting models. Specifically, this study shows that (1) deep-learned models can effectively predict the innovations (or differences between the simulated and observed radiances) used by data assimilation models and (2) deep-learned observation models suffer only minor degradations in performance when the model state is represented with fewer vertical levels, as is commonly used by AI forecasting models. Experiments were performed using the Unified Forecast System (UFS) replay dataset, including Gridpoint Statistical Interpolation (GSI) observational data for the Advanced Technology Microwave Sounder (ATMS) sensor from 2022 and 2023. Code is available at https://github.com/mitre/deep-obs.</span> <span class="abstract-toggle" data-id="2604.00082">more</span>

    [:material-file-document: 2604.00082](https://arxiv.org/abs/2604.00082v1) · [:fontawesome-brands-github:](https://github.com/mitre/deep-obs) · [:material-content-copy: BibTeX](bibtex/2604.00082.bib){ .bibtex-link }

-   #### Self-Organizing Score-based Data Assimilation

    ---

    *Yuma Yamaoka, Seiichi Uchida, Shoji Toyota* · 2026

    <span class="abstract-snippet" id="snip-2603.28048">A state-space model is a statistical framework for inferring latent states from observed time-series data. However, inference with nonlinear and high-dimensional state-space models remains...</span><span class="abstract-full" id="full-2603.28048" hidden>A state-space model is a statistical framework for inferring latent states from observed time-series data. However, inference with nonlinear and high-dimensional state-space models remains challenging. To this end, an approach based on diffusion models-a powerful class of deep generative models-has been developed, known as Score-based Data Assimilation (SDA). However, SDA cannot be directly applied when the latent-state transition depends on unknown parameters that must be inferred jointly with the latent states. To overcome this limitation, we propose a framework that enables SDA to handle latent states with unknown parameters. A key feature of the proposed method is the incorporation of the self-organization technique, which has been used in classical state-space modeling for the joint estimation of latent states and parameters. By integrating this classical technique into modern SDA, our method enables joint inference of latent states and unknown parameters while maintaining the high training efficiency of SDA. The effectiveness of the proposed approach is validated through numerical experiments on dynamical systems arising in neuroscience and atmospheric science. In addition, its scalability is demonstrated using a high-dimensional Kolmogorov flow, with the data dimension on the order of several hundred thousand.</span> <span class="abstract-toggle" data-id="2603.28048">more</span>

    [:material-file-document: 2603.28048](https://arxiv.org/abs/2603.28048v2) · [:material-content-copy: BibTeX](bibtex/2603.28048.bib){ .bibtex-link }

    <span class="md-tag">diffusion</span>

-   #### Convergence Analysis of a Fully Discrete Observer For Data Assimilation of the Barotropic Euler Equations

    ---

    *Aidan Chaumet, Jan Giesselmann* · 2026

    <span class="abstract-snippet" id="snip-2603.10962">We study the convergence of a discrete Luenberger observer for the barotropic Euler equations in one dimension, for measurements of the velocity only. We use a mixed finite element method in space...</span><span class="abstract-full" id="full-2603.10962" hidden>We study the convergence of a discrete Luenberger observer for the barotropic Euler equations in one dimension, for measurements of the velocity only. We use a mixed finite element method in space and implicit Euler integration in time. We use a modified relative energy technique to show an error bound comparing the discrete observer to the original system's solution. The bound is the sum of three parts: an exponentially decaying part, proportional to the difference in initial value, a part proportional to the grid sizes in space and time and a part that is proportional to the size of the measurement errors as well as the nudging parameter. The proportionality constants of the second and third parts are independent of time and grid sizes. To the best of our knowledge, this provides the first error estimate for a discrete observer for a quasilinear hyperbolic system, and implies uniform-in-time accuracy of the discrete observer for long-time simulations.</span> <span class="abstract-toggle" data-id="2603.10962">more</span>

    [:material-file-document: 2603.10962](https://arxiv.org/abs/2603.10962v2) · [:material-content-copy: BibTeX](bibtex/2603.10962.bib){ .bibtex-link }

-   #### Accurate and Efficient Hybrid-Ensemble Atmospheric Data Assimilation in Latent Space with Uncertainty Quantification

    ---

    *Hang Fan, Juan Nathaniel, Yi Xiao, Ce Bian, Fenghua Ling, Ben Fei, Lei Bai, Pierre Gentine* · 2026

    <span class="abstract-snippet" id="snip-2603.04395">Data assimilation (DA) combines model forecasts and observations to estimate the optimal state of the atmosphere with its uncertainty, providing initial conditions for weather prediction and...</span><span class="abstract-full" id="full-2603.04395" hidden>Data assimilation (DA) combines model forecasts and observations to estimate the optimal state of the atmosphere with its uncertainty, providing initial conditions for weather prediction and reanalyses for climate research. Yet, existing traditional and machine-learning DA methods struggle to achieve accuracy, efficiency and uncertainty quantification simultaneously. Here, we propose HLOBA (Hybrid-Ensemble Latent Observation-Background Assimilation), a three-dimensional hybrid-ensemble DA method that operates in an atmospheric latent space learned via an autoencoder (AE). HLOBA maps both model forecasts and observations into a shared latent space via the AE encoder and an end-to-end Observation-to-Latent-space mapping network (O2Lnet), respectively, and fuses them through a Bayesian update with weights inferred from time-lagged ensemble forecasts. Both idealized and real-observation experiments demonstrate that HLOBA matches dynamically constrained four-dimensional DA methods in both analysis and forecast skill, while achieving end-to-end inference-level efficiency and theoretical flexibility applies to any forecasting model. Moreover, by exploiting the error decorrelation property of latent variables, HLOBA enables element-wise uncertainty estimates for its latent analysis and propagates them to model space via the decoder. Idealized experiments show that this uncertainty highlights large-error regions and captures their seasonal variability.</span> <span class="abstract-toggle" data-id="2603.04395">more</span>

    [:material-file-document: 2603.04395](https://arxiv.org/abs/2603.04395v1) · [:material-content-copy: BibTeX](bibtex/2603.04395.bib){ .bibtex-link }

    <span class="md-tag">probabilistic</span>

-   #### Efficient Real-Time Adaptation of ROMs for Unsteady Flows Using Data Assimilation

    ---

    *Ismaël Zighed, Andrea Nóvoa, Luca Magri, Taraneh Sayadi* · 2026

    <span class="abstract-snippet" id="snip-2602.23188">We propose an efficient retraining strategy for a parameterized Reduced Order Model (ROM) that attains accuracy comparable to full retraining while requiring only a fraction of the computational time...</span><span class="abstract-full" id="full-2602.23188" hidden>We propose an efficient retraining strategy for a parameterized Reduced Order Model (ROM) that attains accuracy comparable to full retraining while requiring only a fraction of the computational time and relying solely on sparse observations of the full system. The architecture employs an encode-process-decode structure: a Variational Autoencoder (VAE) to perform dimensionality reduction, and a transformer network to evolve the latent states and model the dynamics. The ROM is parameterized by an external control variable, the Reynolds number in the Navier-Stokes setting, with the transformer exploiting attention mechanisms to capture both temporal dependencies and parameter effects. The probabilistic VAE enables stochastic sampling of trajectory ensembles, providing predictive means and uncertainty quantification through the first two moments. After initial training on a limited set of dynamical regimes, the model is adapted to out-of-sample parameter regions using only sparse data. Its probabilistic formulation naturally supports ensemble generation, which we employ within an ensemble Kalman filtering framework to assimilate data and reconstruct full-state trajectories from minimal observations. We further show that, for the dynamical system considered, the dominant source of error in out-of-sample forecasts stems from distortions of the latent manifold rather than changes in the latent dynamics. Consequently, retraining can be limited to the autoencoder, allowing for a lightweight, computationally efficient, real-time adaptation procedure with very sparse fine-tuning data.</span> <span class="abstract-toggle" data-id="2602.23188">more</span>

    [:material-file-document: 2602.23188](https://arxiv.org/abs/2602.23188v1) · [:material-content-copy: BibTeX](bibtex/2602.23188.bib){ .bibtex-link }

    <span class="md-tag">transformer</span> <span class="md-tag">variational</span> <span class="md-tag">probabilistic</span>

-   #### LEVDA: Latent Ensemble Variational Data Assimilation via Differentiable Dynamics

    ---

    *Phillip Si, Peng Chen* · 2026

    <span class="abstract-snippet" id="snip-2602.19406">Long-range geophysical forecasts are fundamentally limited by chaotic dynamics and numerical errors. While data assimilation can mitigate these issues, classical variational smoothers require...</span><span class="abstract-full" id="full-2602.19406" hidden>Long-range geophysical forecasts are fundamentally limited by chaotic dynamics and numerical errors. While data assimilation can mitigate these issues, classical variational smoothers require computationally expensive tangent-linear and adjoint models. Conversely, recent efficient latent filtering methods often enforce weak trajectory-level constraints and assume fixed observation grids. To bridge this gap, we propose Latent Ensemble Variational Data Assimilation (LEVDA), an ensemble-space variational smoother that operates in the low-dimensional latent space of a pretrained differentiable neural dynamics surrogate. By performing four-dimensional ensemble-variational (4DEnVar) optimization within an ensemble subspace, LEVDA jointly assimilates states and unknown parameters without the need for adjoint code or auxiliary observation-to-latent encoders. Leveraging the fully differentiable, continuous-in-time-and-space nature of the surrogate, LEVDA naturally accommodates highly irregular sampling at arbitrary spatiotemporal locations. Across three challenging geophysical benchmarks, LEVDA matches or outperforms state-of-the-art latent filtering baselines under severe observational sparsity while providing more reliable uncertainty quantification. Simultaneously, it achieves substantially improved assimilation accuracy and computational efficiency compared to full-state 4DEnVar.</span> <span class="abstract-toggle" data-id="2602.19406">more</span>

    [:material-file-document: 2602.19406](https://arxiv.org/abs/2602.19406v1) · [:material-content-copy: BibTeX](bibtex/2602.19406.bib){ .bibtex-link }

    <span class="md-tag">probabilistic</span>

-   #### Preconditioned Adjoint Data Assimilation for Two-Dimensional Decaying Isotropic Turbulence

    ---

    *Hongyi Ke, Zejian You, Qi Wang* · 2026

    <span class="abstract-snippet" id="snip-2602.14016">Adjoint-based data assimilation for turbulent Navier-Stokes flows is fundamentally limited by the behavior of the adjoint dynamics: in backward time, adjoint fields exhibit exponential growth and...</span><span class="abstract-full" id="full-2602.14016" hidden>Adjoint-based data assimilation for turbulent Navier-Stokes flows is fundamentally limited by the behavior of the adjoint dynamics: in backward time, adjoint fields exhibit exponential growth and become increasingly dominated by small-scale structures, severely degrading reconstruction of the initial condition from sparse measurements. We demonstrate that the relative weighting of spectral components in the adjoint formulation can be systematically controlled by redefining the inner product under which the adjoint operator is defined. The inverse problem is formulated as a constrained minimization in which a cost functional measures the mismatch between model predictions and observations, and the adjoint equations provide the gradient with respect to the initial velocity field. Redefining the forward-adjoint duality through a Fourier-space weighting kernel preconditions the optimization and is mathematically equivalent to changing the representation of the control variable or, alternatively, introducing a smoothing operation on the governing dynamics. Specific kernel choices correspond to fractional integration or diffusion operators applied to the initial condition. Among these, exponential kernels provide effective regularization by suppressing high-wavenumber contributions while preserving large-scale coherence, leading to improved reconstruction across scales. A statistical analysis of an ensemble of adjoint fields from different turbulent realizations reveals scale-dependent backward growth rates, explaining the instability of the standard formulation and clarifying the mechanism by which the proposed preconditioning attenuates incoherent small-scale amplification.</span> <span class="abstract-toggle" data-id="2602.14016">more</span>

    [:material-file-document: 2602.14016](https://arxiv.org/abs/2602.14016v1) · [:material-content-copy: BibTeX](bibtex/2602.14016.bib){ .bibtex-link }

-   #### On a system of equations arising in meteorology: Well-posedness and data assimilation

    ---

    *Eduard Feireisl, Piotr Gwiazda, Agnieszka Świerczewska-Gwiazda* · 2026

    <span class="abstract-snippet" id="snip-2602.02328">Data assimilation plays a crucial role in modern weather prediction, providing a systematic way to incorporate observational data into complex dynamical models. The paper addresses continuous data...</span><span class="abstract-full" id="full-2602.02328" hidden>Data assimilation plays a crucial role in modern weather prediction, providing a systematic way to incorporate observational data into complex dynamical models. The paper addresses continuous data assimilation for a model arising as a singular limit of the three-dimensional compressible Navier-Stokes-Fourier system with rotation driven by temperature gradient. The limit system preserves the essential physical mechanisms of the original model, while exhibiting a reduced, effectively two-and-a-half-dimensional structure. This simplified framework allows for a rigorous analytical study of the data assimilation process while maintaining a direct physical connection to the full compressible model. We establish well posedness of global-in-time solutions and a compact trajectory attractor, followed by the stability and convergence results for the nudging scheme applied to the limiting system. Finally, we demonstrate how these results can be combined with a relative entropy argument to extend the assimilation framework to the full three-dimensional compressible setting, thereby establishing a rigorous connection between the reduced and physically complete models.</span> <span class="abstract-toggle" data-id="2602.02328">more</span>

    [:material-file-document: 2602.02328](https://arxiv.org/abs/2602.02328v1) · [:material-content-copy: BibTeX](bibtex/2602.02328.bib){ .bibtex-link }

-   #### SENDAI: A Hierarchical Sparse-measurement, EfficieNt Data AssImilation Framework

    ---

    *Xingyue Zhang, Yuxuan Bao, Mars Liyao Gao, J. Nathan Kutz* · 2026

    <span class="abstract-snippet" id="snip-2601.21664">Bridging the gap between data-rich training regimes and observation-sparse deployment conditions remains a central challenge in spatiotemporal field reconstruction, particularly when target domains...</span><span class="abstract-full" id="full-2601.21664" hidden>Bridging the gap between data-rich training regimes and observation-sparse deployment conditions remains a central challenge in spatiotemporal field reconstruction, particularly when target domains exhibit distributional shifts, heterogeneous structure, and multi-scale dynamics absent from available training data. We present SENDAI, a hierarchical Sparse-measurement, EfficieNt Data AssImilation Framework that reconstructs full spatial states from hyper sparse sensor observations by combining simulation-derived priors with learned discrepancy corrections. We demonstrate the performance on satellite remote sensing, reconstructing MODIS (Moderate Resolution Imaging Spectroradiometer) derived vegetation index fields across six globally distributed sites. Using seasonal periods as a proxy for domain shift, the framework consistently outperforms established baselines that require substantially denser observations -- SENDAI achieves a maximum SSIM improvement of 185% over traditional baselines and a 36% improvement over recent high-frequency-based methods. These gains are particularly pronounced for landscapes with sharp boundaries and sub-seasonal dynamics; more importantly, the framework effectively preserves diagnostically relevant structures -- such as field topologies, land cover discontinuities, and spatial gradients. By yielding corrections that are more structurally and spectrally separable, the reconstructed fields are better suited for downstream inference of indirectly observed variables. The results therefore highlight a lightweight and operationally viable framework for sparse-measurement reconstruction that is applicable to physically grounded inference, resource-limited deployment, and real-time monitor and control.</span> <span class="abstract-toggle" data-id="2601.21664">more</span>

    [:material-file-document: 2601.21664](https://arxiv.org/abs/2601.21664v1) · [:material-content-copy: BibTeX](bibtex/2601.21664.bib){ .bibtex-link }

-   #### Cheap2Rich: A Multi-Fidelity Framework for Data Assimilation and System Identification of Multiscale Physics -- Rotating Detonation Engines

    ---

    *Yuxuan Bao, Jan Zajac, Megan Powers, Venkat Raman, J. Nathan Kutz* · 2026

    <span class="abstract-snippet" id="snip-2601.20295">Bridging the sim2real gap between computationally inexpensive models and complex physical systems remains a central challenge in machine learning applications to engineering problems, particularly in...</span><span class="abstract-full" id="full-2601.20295" hidden>Bridging the sim2real gap between computationally inexpensive models and complex physical systems remains a central challenge in machine learning applications to engineering problems, particularly in multi-scale settings where reduced-order models typically capture only dominant dynamics. In this work, we present Cheap2Rich, a multi-scale data assimilation framework that reconstructs high-fidelity state spaces from sparse sensor histories by combining a fast low-fidelity prior with learned, interpretable discrepancy corrections. We demonstrate the performance on rotating detonation engines (RDEs), a challenging class of systems that couple detonation-front propagation with injector-driven unsteadiness, mixing, and stiff chemistry across disparate scales. Our approach successfully reconstructs high-fidelity RDE states from sparse measurements while isolating physically meaningful discrepancy dynamics associated with injector-driven effects. The results highlight a general multi-fidelity framework for data assimilation and system identification in complex multi-scale systems, enabling rapid design exploration and real-time monitoring and control while providing interpretable discrepancy dynamics. Code for this project is is available at: github.com/kro0l1k/Cheap2Rich.</span> <span class="abstract-toggle" data-id="2601.20295">more</span>

    [:material-file-document: 2601.20295](https://arxiv.org/abs/2601.20295v1) · [:material-content-copy: BibTeX](bibtex/2601.20295.bib){ .bibtex-link }

-   #### The Ensemble Schr{ö}dinger Bridge filter for Nonlinear Data Assimilation

    ---

    *Feng Bao, Hui Sun* · 2025

    <span class="abstract-snippet" id="snip-2512.18928">This work puts forward a novel nonlinear optimal filter namely the Ensemble Schr{ö}dinger Bridge nonlinear filter. The proposed filter finds marriage of the standard prediction procedure and the...</span><span class="abstract-full" id="full-2512.18928" hidden>This work puts forward a novel nonlinear optimal filter namely the Ensemble Schr{ö}dinger Bridge nonlinear filter. The proposed filter finds marriage of the standard prediction procedure and the diffusion generative modeling for the analysis procedure to realize one filtering step. The designed approach finds no structural model error, and it is derivative free, training free and highly parallizable. Experimental results show that the designed algorithm performs well given highly nonlinear dynamics in (mildly) high dimension up to 40 or above under a chaotic environment. It also shows better performance than classical methods such as the ensemble Kalman filter and the Particle filter in numerous tests given different level of nonlinearity. Future work will focus on extending the proposed approach to practical meteorological applications and establishing a rigorous convergence analysis.</span> <span class="abstract-toggle" data-id="2512.18928">more</span>

    [:material-file-document: 2512.18928](https://arxiv.org/abs/2512.18928v1) · [:material-content-copy: BibTeX](bibtex/2512.18928.bib){ .bibtex-link }

-   #### Continuous data assimilation for 2D stochastic Navier-Stokes equations

    ---

    *Hakima Bessaih, Benedetta Ferrario, Oussama Landoulsi, Margherita Zanella* · 2025

    <span class="abstract-snippet" id="snip-2512.15184">Continuous data assimilation methods, such as the nudging algorithm introduced by Azouani, Olson, and Titi (AOT) [2], are known to be highly effective in deterministic settings for asymptotically...</span><span class="abstract-full" id="full-2512.15184" hidden>Continuous data assimilation methods, such as the nudging algorithm introduced by Azouani, Olson, and Titi (AOT) [2], are known to be highly effective in deterministic settings for asymptotically synchronizing approximate solutions with observed dynamics. In this work, we extend this framework to a stochastic regime by considering the two-dimensional incompressible Navier-Stokes equations subject to either additive or multiplicative noise. We establish sufficient conditions on the nudging parameter and the spatial observation scale that guarantee convergence of the nudged solution to the true stochastic flow.   In the case of multiplicative noise, convergence holds in expectation, with exponential or polynomial rates depending on the growth of the noise covariance. For additive noise, we obtain the exponential convergence both in expectation and pathwise. These results yield a stochastic generalization of the AOT theory, demonstrating how the interplay between random forcing, viscous dissipation and feedback control governs synchronization in stochastic fluid systems.</span> <span class="abstract-toggle" data-id="2512.15184">more</span>

    [:material-file-document: 2512.15184](https://arxiv.org/abs/2512.15184v1) · [:material-content-copy: BibTeX](bibtex/2512.15184.bib){ .bibtex-link }

-   #### Predicting CME Arrivals with Heliospheric Imagers from L5: A Data Assimilation Approach

    ---

    *Tanja Amerstorfer, Justin Le Louëdec, David Barnes, Maike Bauer, Jackie A. Davies et al.* · 2025

    <span class="abstract-snippet" id="snip-2512.09738">The Solar TErrestrial RElations Observatory (STEREO) mission has laid a foundation for advancing real-time space weather forecasting by enabling the evaluation of heliospheric imager (HI) data for...</span><span class="abstract-full" id="full-2512.09738" hidden>The Solar TErrestrial RElations Observatory (STEREO) mission has laid a foundation for advancing real-time space weather forecasting by enabling the evaluation of heliospheric imager (HI) data for predicting coronal mass ejection (CME) arrivals at Earth. This study employs the ELEvoHI model to assess how incorporating STEREO/HI data from the Lagrange 5 (L5) perspective can enhance prediction accuracy for CME arrival times and speeds. Our investigation, preparing for the upcoming ESA Vigil mission, explores whether the progressive incorporation of HI data in real-time enhances forecasting accuracy. The role of human tracking variability is evaluated by comparing predictions based on observations by three different scientists, highlighting the influence of manual biases on forecasting outcomes. Furthermore, the study examines the efficacy of deriving CME propagation directions using HI-specific methods versus coronagraph-based techniques, emphasising the trade-offs in prediction accuracy. Our results demonstrate the potential of HI data to significantly improve operational space weather forecasting when integrated with other observational platforms, especially when HI data from beyond 35° elongation are used. These findings pave the way for optimising real-time prediction methodologies, providing valuable groundwork for the forthcoming Vigil mission and enhancing preparedness for CME-driven space weather events.</span> <span class="abstract-toggle" data-id="2512.09738">more</span>

    [:material-file-document: 2512.09738](https://arxiv.org/abs/2512.09738v1) · [:material-content-copy: BibTeX](bibtex/2512.09738.bib){ .bibtex-link }

-   #### Data assimilation and discrepancy modeling with shallow recurrent decoders

    ---

    *Yuxuan Bao, J. Nathan Kutz* · 2025

    <span class="abstract-snippet" id="snip-2512.01170">The requirements of modern sensing are rapidly evolving, driven by increasing demands for data efficiency, real-time processing, and deployment under limited sensing coverage. Complex physical...</span><span class="abstract-full" id="full-2512.01170" hidden>The requirements of modern sensing are rapidly evolving, driven by increasing demands for data efficiency, real-time processing, and deployment under limited sensing coverage. Complex physical systems are often characterized through the integration of a limited number of point sensors in combination with scientific computations which approximate the dominant, full-state dynamics. Simulation models, however, inevitably neglect small-scale or hidden processes, are sensitive to perturbations, or oversimplify parameter correlations, leading to reconstructions that often diverge from the reality measured by sensors. This creates a critical need for data assimilation, the process of integrating observational data with predictive simulation models to produce coherent and accurate estimates of the full state of complex physical systems. We propose a machine learning framework for Data Assimilation with a SHallow REcurrent Decoder (DA-SHRED) which bridges the simulation-to-real (SIM2REAL) gap between computational modeling and experimental sensor data. For real-world physics systems modeling high-dimensional spatiotemporal fields, where the full state cannot be directly observed and must be inferred from sparse sensor measurements, we leverage the latent space learned from a reduced simulation model via SHRED, and update these latent variables using real sensor data to accurately reconstruct the full system state. Furthermore, our algorithm incorporates a sparse identification of nonlinear dynamics based regression model in the latent space to identify functionals corresponding to missing dynamics in the simulation model. We demonstrate that DA-SHRED successfully closes the SIM2REAL gap and additionally recovers missing dynamics in highly complex systems, demonstrating that the combination of efficient temporal encoding and physics-informed correction enables robust data assimilation.</span> <span class="abstract-toggle" data-id="2512.01170">more</span>

    [:material-file-document: 2512.01170](https://arxiv.org/abs/2512.01170v1) · [:material-content-copy: BibTeX](bibtex/2512.01170.bib){ .bibtex-link }

    <span class="md-tag">physics-informed</span>

-   #### Towards Streaming Prediction of Oscillatory Flows: A Data Assimilation and Machine Learning Approach

    ---

    *Miguel M. Valero, Marcello Meldi* · 2025

    <span class="abstract-snippet" id="snip-2511.15758">Data-driven methods have demonstrated strong predictive capabilities in fluid mechanics, yet most current applications still focus on simplified configurations, often characterised by statistical...</span><span class="abstract-full" id="full-2511.15758" hidden>Data-driven methods have demonstrated strong predictive capabilities in fluid mechanics, yet most current applications still focus on simplified configurations, often characterised by statistical stationarity or limited temporal variability. This work proposes a methodology that combines Data Assimilation (DA) and Machine Learning (ML) to predict flow configurations that exhibit cyclic behaviour over time. Starting from limited, sparse high-fidelity measurements and a low-fidelity numerical model, the DA approach performs data fusion to obtain complete and accurate flow state estimations in time. This complete dataset is used to train multiple ML tools, which are applied across different phases of the flow cycle to augment the model's predictions when high-fidelity data might not be available for the DA application. The methodology is applied to the analysis of an oscillating cylinder in a laminar regime using a sliding-window approach, in which separate models are trained for specific flow conditions to ensure each model specialises in flow dynamics representative of a phase of the oscillation period. This phase-resolved learning enables the efficient capture of transient features that would be challenging for a single global model. The results highlight the potential of this method to study complex flow configurations with oscillatory features in which neither the flow nor the cycle is known a priori, in particular by exploiting real-time training and updates, as is commonly done in digital twins, which require continuous model correction and adaptation.</span> <span class="abstract-toggle" data-id="2511.15758">more</span>

    [:material-file-document: 2511.15758](https://arxiv.org/abs/2511.15758v1) · [:material-content-copy: BibTeX](bibtex/2511.15758.bib){ .bibtex-link }

-   #### Exploring Ultra Rapid Data Assimilation Based on Ensemble Transform Kalman Filter with the Lorenz 96 Model

    ---

    *Fumitoshi Kawasaki, Atsushi Okazaki, Kenta Kurosawa, Shunji Kotsuki* · 2025

    <span class="abstract-snippet" id="snip-2511.12620">To explore the effectiveness of ultra-rapid data assimilation (URDA) for numerical weather prediction (NWP), this study investigates the properties of URDA in nonlinear models and proposes technical...</span><span class="abstract-full" id="full-2511.12620" hidden>To explore the effectiveness of ultra-rapid data assimilation (URDA) for numerical weather prediction (NWP), this study investigates the properties of URDA in nonlinear models and proposes technical treatments to enhance its performance. URDA rapidly updates preemptive forecasts derived from observations without integrating a dynamical model each time additional observations become available. First, we analytically demonstrate that the preemptive forecast obtained by URDA in nonlinear models is approximately equivalent to the forecast integrated from the analysis. Furthermore, numerical experiments are conducted with the 40-variable Lorenz 96 model. The results show that URDA in nonlinear models tends to exhibit deterioration of forecast accuracy and collapse of ensemble spread when preemptive forecasts are repeatedly updated or when the forecasts are extended over longer periods. Furthermore, the roles of inflation and localization, both essential technical treatments in NWP, are examined in the context of URDA. It is shown that although inflation and localization are essential to URDA, conventional inflation techniques are not suitable for it. Therefore, this study proposes new technical treatments for URDA, namely relaxation to baseline perturbations (RTBP) and relaxation to baseline forecast (RTBF). Applying RTBP and RTBF mitigates the difficulties associated with URDA and yields preemptive forecasts with higher accuracy than the baseline forecast. Consequently, URDA, particularly when combined with RTBP and RTBF, would stand as a step toward practical application in NWP.</span> <span class="abstract-toggle" data-id="2511.12620">more</span>

    [:material-file-document: 2511.12620](https://arxiv.org/abs/2511.12620v1) · [:material-content-copy: BibTeX](bibtex/2511.12620.bib){ .bibtex-link }

-   #### DAMBench: A Multi-Modal Benchmark for Deep Learning-based Atmospheric Data Assimilation

    ---

    *Hao Wang, Zixuan Weng, Jindong Han, Wei Fan, Hao Liu* · 2025

    <span class="abstract-snippet" id="snip-2511.01468">Data Assimilation is a cornerstone of atmospheric system modeling, tasked with reconstructing system states by integrating sparse, noisy observations with prior estimation. While traditional...</span><span class="abstract-full" id="full-2511.01468" hidden>Data Assimilation is a cornerstone of atmospheric system modeling, tasked with reconstructing system states by integrating sparse, noisy observations with prior estimation. While traditional approaches like variational and ensemble Kalman filtering have proven effective, recent advances in deep learning offer more scalable, efficient, and flexible alternatives better suited for complex, real-world data assimilation involving large-scale and multi-modal observations. However, existing deep learning-based DA research suffers from two critical limitations: (1) reliance on oversimplified scenarios with synthetically perturbed observations, and (2) the absence of standardized benchmarks for fair model comparison. To address these gaps, in this work, we introduce DAMBench, the first large-scale multi-modal benchmark designed to evaluate data-driven DA models under realistic atmospheric conditions. DAMBench integrates high-quality background states from state-of-the-art forecasting systems and real-world multi-modal observations (i.e., real-world weather stations and satellite imagery). All data are resampled to a common grid and temporally aligned to support systematic training, validation, and testing. We provide unified evaluation protocols and benchmark representative data assimilation approaches, including latent generative models and neural process frameworks. Additionally, we propose a lightweight multi-modal plugin to demonstrate how integrating realistic observations can enhance even simple baselines. Through comprehensive experiments, DAMBench establishes a rigorous foundation for future research, promoting reproducibility, fair comparison, and extensibility to real-world multi-modal scenarios. Our dataset and code are publicly available at https://github.com/figerhaowang/DAMBench.</span> <span class="abstract-toggle" data-id="2511.01468">more</span>

    [:material-file-document: 2511.01468](https://arxiv.org/abs/2511.01468v1) · [:fontawesome-brands-github:](https://github.com/figerhaowang/DAMBench) · [:material-content-copy: BibTeX](bibtex/2511.01468.bib){ .bibtex-link }

-   #### Interpolated Discrepancy Data Assimilation for PDEs with Sparse Observations

    ---

    *Tong Wu, Humberto Godinez, Vitaliy Gyrya, James M. Hyman* · 2025

    <span class="abstract-snippet" id="snip-2510.24944">Sparse sensor networks in weather and ocean modeling observe only a small fraction of the system state, which destabilizes standard nudging-based data assimilation. We introduce Interpolated...</span><span class="abstract-full" id="full-2510.24944" hidden>Sparse sensor networks in weather and ocean modeling observe only a small fraction of the system state, which destabilizes standard nudging-based data assimilation. We introduce Interpolated Discrepancy Data Assimilation (IDDA), which modifies how discrepancies enter the governing equations. Rather than adding observations as a forcing term alone, IDDA also adjusts the nonlinear operator using interpolated observational information. This structural change suppresses error amplification when nonlinear effects dominate. We prove exponential convergence under explicit conditions linking error decay to observation spacing, nudging strength, and diffusion coefficient. The key requirement establishes bounds on nudging strength relative to observation spacing and diffusion, giving practitioners a clear operating window. When observations resolve the relevant scales, error decays at a user-specified rate. Critically, the error bound scales with the square of observation spacing rather than through hard-to-estimate nonlinear growth rates. We validate IDDA on Burgers flow, Kuramoto-Sivashinsky dynamics, and two-dimensional Navier-Stokes turbulence. Across these tests, IDDA reaches target accuracy faster than standard interpolated nudging, remains stable in chaotic regimes, avoids non-monotone transients, and requires minimal parameter tuning. Because IDDA uses standard explicit time integration, it fits readily into existing simulation pipelines without specialized solvers. These properties make IDDA a practical upgrade for operational systems constrained by sparse sensor coverage.</span> <span class="abstract-toggle" data-id="2510.24944">more</span>

    [:material-file-document: 2510.24944](https://arxiv.org/abs/2510.24944v1) · [:material-content-copy: BibTeX](bibtex/2510.24944.bib){ .bibtex-link }

-   #### LO-SDA: Latent Optimization for Score-based Atmospheric Data Assimilation

    ---

    *Jing-An Sun, Hang Fan, Junchao Gong, Ben Fei, Kun Chen, Fenghua Ling, Wenlong Zhang, Wanghan Xu et al.* · 2025

    <span class="abstract-snippet" id="snip-2510.22562">Data assimilation (DA) plays a pivotal role in numerical weather prediction by systematically integrating sparse observations with model forecasts to estimate optimal atmospheric initial condition...</span><span class="abstract-full" id="full-2510.22562" hidden>Data assimilation (DA) plays a pivotal role in numerical weather prediction by systematically integrating sparse observations with model forecasts to estimate optimal atmospheric initial condition for forthcoming forecasts. Traditional Bayesian DA methods adopt a Gaussian background prior as a practical compromise for the curse of dimensionality in atmospheric systems, that simplifies the nonlinear nature of atmospheric dynamics and can result in biased estimates. To address this limitation, we propose a novel generative DA method, LO-SDA. First, a variational autoencoder is trained to learn compact latent representations that disentangle complex atmospheric correlations. Within this latent space, a background-conditioned diffusion model is employed to directly learn the conditional distribution from data, thereby generalizing and removing assumptions in the Gaussian prior in traditional DA methods. Most importantly, we introduce latent optimization during the reverse process of the diffusion model to ensure strict consistency between the generated states and sparse observations. Idealized experiments demonstrate that LO-SDA not only outperforms score-based DA methods based on diffusion posterior sampling but also surpasses traditional DA approaches. To our knowledge, this is the first time that a diffusion-based DA method demonstrates the potential to outperform traditional approaches on high-dimensional global atmospheric systems. These findings suggest that long-standing reliance on Gaussian priors-a foundational assumption in operational atmospheric DA-may no longer be necessary in light of advances in generative modeling.</span> <span class="abstract-toggle" data-id="2510.22562">more</span>

    [:material-file-document: 2510.22562](https://arxiv.org/abs/2510.22562v1) · [:material-content-copy: BibTeX](bibtex/2510.22562.bib){ .bibtex-link }

    <span class="md-tag">diffusion</span> <span class="md-tag">variational</span> <span class="md-tag">probabilistic</span>

-   #### Continuous data assimilation applied to the Rayleigh-Benard problem for compressible fluid flows

    ---

    *Eduard Feireisl, Wladimir Neves* · 2025

    <span class="abstract-snippet" id="snip-2510.20316">We apply a continuous data assimilation method to the Navier-Stokes-Fourier system governing the evolution of a compressible, rotating and thermally driven fluid. A rigorous proof of the tracking...</span><span class="abstract-full" id="full-2510.20316" hidden>We apply a continuous data assimilation method to the Navier-Stokes-Fourier system governing the evolution of a compressible, rotating and thermally driven fluid. A rigorous proof of the tracking property is given in the asymptotic regime of low Mach and high Rossby and Froude numbers. Large data in the framework of weak solutions are considered.</span> <span class="abstract-toggle" data-id="2510.20316">more</span>

    [:material-file-document: 2510.20316](https://arxiv.org/abs/2510.20316v1) · [:material-content-copy: BibTeX](bibtex/2510.20316.bib){ .bibtex-link }

-   #### Non-intrusive structural-preserving sequential data assimilation

    ---

    *Lizuo Liu, Tongtong Li, Anne Gelb* · 2025

    <span class="abstract-snippet" id="snip-2510.19701">Data assimilation (DA) methods combine model predictions with observational data to improve state estimation in dynamical systems, inspiring their increasingly prominent role in geophysical and...</span><span class="abstract-full" id="full-2510.19701" hidden>Data assimilation (DA) methods combine model predictions with observational data to improve state estimation in dynamical systems, inspiring their increasingly prominent role in geophysical and climate applications. Classical DA methods assume that the governing equations modeling the dynamics are known, which is unlikely for most real world applications. Machine learning (ML) provides a flexible alternative by learning surrogate models directly from data, but standard ML methods struggle in noisy and data-scarce environments, where meaningful extrapolation requires incorporating physical constraints. Recent advances in structure-preserving ML architectures, such as the development of the entropy-stable conservative flux form network (ESCFN), highlight the critical role of physical structure in improving learning stability and accuracy for unknown systems of conservation laws. Structural information has also been shown to improve DA performance. Gradient-based measures of spatial variability, in particular, can help refine ensemble updates in discontinuous systems. Motivated by both of these recent innovations, this investigation proposes a new non-intrusive, structure-preserving sequential data assimilation (NSSDA) framework that leverages structure at both the forecast and analysis stages. We use the ESCFN to construct a surrogate model to preserve physical laws during forecasting, and a structurally informed ensemble transform Kalman filter (SETKF) to embed local statistical structure into the assimilation step. Our method operates in a highly constrained environment, using only a single noisy trajectory for both training and assimilation. Numerical experiments where the unknown dynamics correspond respectively to the shallow water and Euler equations demonstrate significantly improved predictive accuracy.</span> <span class="abstract-toggle" data-id="2510.19701">more</span>

    [:material-file-document: 2510.19701](https://arxiv.org/abs/2510.19701v1) · [:material-content-copy: BibTeX](bibtex/2510.19701.bib){ .bibtex-link }

-   #### Incorporating Multivariate Consistency in ML-Based Weather Forecasting with Latent-space Constraints

    ---

    *Hang Fan, Yi Xiao, Yongquan Qu, Fenghua Ling, Ben Fei, Lei Bai, Pierre Gentine* · 2025

    <span class="abstract-snippet" id="snip-2510.04006">Data-driven machine learning (ML) models have recently shown promise in surpassing traditional physics-based approaches for weather forecasting, leading to a so-called second revolution in weather...</span><span class="abstract-full" id="full-2510.04006" hidden>Data-driven machine learning (ML) models have recently shown promise in surpassing traditional physics-based approaches for weather forecasting, leading to a so-called second revolution in weather forecasting. However, most ML-based forecast models treat reanalysis as the truth and are trained under variable-specific loss weighting, ignoring their physical coupling and spatial structure. Over long time horizons, the forecasts become blurry and physically unrealistic under rollout training. To address this, we reinterpret model training as a weak-constraint four-dimensional variational data assimilation (WC-4DVar) problem, treating reanalysis data as imperfect observations. This allows the loss function to incorporate reanalysis error covariance and capture multivariate dependencies. In practice, we compute the loss in a latent space learned by an autoencoder (AE), where the reanalysis error covariance becomes approximately diagonal, thus avoiding the need to explicitly model it in the high-dimensional model space. We show that rollout training with latent-space constraints improves long-term forecast skill and better preserves fine-scale structures and physical realism compared to training with model-space loss. Finally, we extend this framework to accommodate heterogeneous data sources, enabling the forecast model to be trained jointly on reanalysis and multi-source observations within a unified theoretical formulation.</span> <span class="abstract-toggle" data-id="2510.04006">more</span>

    [:material-file-document: 2510.04006](https://arxiv.org/abs/2510.04006v1) · [:material-content-copy: BibTeX](bibtex/2510.04006.bib){ .bibtex-link }

    <span class="md-tag">physics-informed</span>

-   #### On the joint observability of flow fields and particle properties from Lagrangian trajectories: evidence from neural data assimilation

    ---

    *Ke Zhou, Samuel J. Grauer* · 2025

    <span class="abstract-snippet" id="snip-2510.00479">We numerically investigate the joint observability of flow states and unknown particle properties from Lagrangian particle tracking (LPT) data. LPT offers time-resolved, volumetric measurements of...</span><span class="abstract-full" id="full-2510.00479" hidden>We numerically investigate the joint observability of flow states and unknown particle properties from Lagrangian particle tracking (LPT) data. LPT offers time-resolved, volumetric measurements of particle trajectories, but experimental tracks are spatially sparse, potentially noisy, and may be further complicated by inertial transport, raising the question of whether both Eulerian fields and particle characteristics can be reliably inferred. To address this, we develop a data assimilation framework that couples an Eulerian flow representation with Lagrangian particle models, enabling the simultaneous inference of carrier fields and particle properties under the governing equations of disperse multiphase flow. Using this approach, we establish empirical existence proofs of joint observability across three representative regimes. In a turbulent boundary layer with noisy tracer tracks (St to 0), flow states and true particle positions are jointly observable. In homogeneous isotropic turbulence seeded with inertial particles (St ~ 1-5), we demonstrate simultaneous recovery of flow states and particle diameters, showing the feasibility of implicit particle characterization. In a compressible, shock-dominated flow, we report the first joint reconstructions of velocity, pressure, density, and inertial particle properties (diameter and density), highlighting both the potential and certain limits of observability in supersonic regimes. Systematic sensitivity studies further reveal how seeding density, noise level, and Stokes number govern reconstruction accuracy, yielding practical guidelines for experimental design. Taken together, these results show that the scope of LPT could be broadened to multiphase and high-speed flows, in which tracer and measurement fidelity are limited.</span> <span class="abstract-toggle" data-id="2510.00479">more</span>

    [:material-file-document: 2510.00479](https://arxiv.org/abs/2510.00479v1) · [:material-content-copy: BibTeX](bibtex/2510.00479.bib){ .bibtex-link }

-   #### Comparing Data Assimilation and Likelihood-Based Inference on Latent State Estimation in Agent-Based Models

    ---

    *Blas Kolic, Corrado Monti, Gianmarco De Francisci Morales, Marco Pangallo* · 2025

    <span class="abstract-snippet" id="snip-2509.17625">In this paper, we present the first systematic comparison of Data Assimilation (DA) and Likelihood-Based Inference (LBI) in the context of Agent-Based Models (ABMs). These models generate observable...</span><span class="abstract-full" id="full-2509.17625" hidden>In this paper, we present the first systematic comparison of Data Assimilation (DA) and Likelihood-Based Inference (LBI) in the context of Agent-Based Models (ABMs). These models generate observable time series driven by evolving, partially-latent microstates. Latent states need to be estimated to align simulations with real-world data -- a task traditionally addressed by DA, especially in continuous and equation-based models such as those used in weather forecasting. However, the nature of ABMs poses challenges for standard DA methods. Solving such issues requires adaptation of previous DA techniques, or ad-hoc alternatives such as LBI. DA approximates the likelihood in a model-agnostic way, making it broadly applicable but potentially less precise. In contrast, LBI provides more accurate state estimation by directly leveraging the model's likelihood, but at the cost of requiring a hand-crafted, model-specific likelihood function, which may be complex or infeasible to derive. We compare the two methods on the Bounded-Confidence Model, a well-known opinion dynamics ABM, where agents are affected only by others holding sufficiently similar opinions. We find that LBI better recovers latent agent-level opinions, even under model mis-specification, leading to improved individual-level forecasts. At the aggregate level, however, both methods perform comparably, and DA remains competitive across levels of aggregation under certain parameter settings. Our findings suggest that DA is well-suited for aggregate predictions, while LBI is preferable for agent-level inference.</span> <span class="abstract-toggle" data-id="2509.17625">more</span>

    [:material-file-document: 2509.17625](https://arxiv.org/abs/2509.17625v1) · [:material-content-copy: BibTeX](bibtex/2509.17625.bib){ .bibtex-link }

-   #### Lagrangian-Eulerian Multiscale Data Assimilation in Physical Domain based on Conditional Gaussian Nonlinear System

    ---

    *Hyeonggeun Yun, Quanling Deng* · 2025

    <span class="abstract-snippet" id="snip-2509.14586">This research aims to further investigate the process of Lagrangian-Eulerian Multiscale Data Assimilation (LEMDA) by replacing the Fourier space with the physical domain. Such change in the...</span><span class="abstract-full" id="full-2509.14586" hidden>This research aims to further investigate the process of Lagrangian-Eulerian Multiscale Data Assimilation (LEMDA) by replacing the Fourier space with the physical domain. Such change in the perspective of domain introduces the advantages of being able to deal in non-periodic system and more intuitive representation of localised phenomena or time-dependent problems. The context of the domain for this paper was set as sea ice floe trajectories to recover the ocean eddies in the Arctic regions, which led the model to be derived from two-layer Quasi geostrophic (QG) model. The numerical solution to this model utilises the Conditional Gaussian Nonlinear System (CGNS) to accommodate the inherent non-linearity in analytical and continuous manner. The normalised root mean square error (RMSE) and pattern correlation (Corr) are used to evaluate the performance of the posterior mean of the model. The results corroborate the effectiveness of exploiting the two-layer QG model in physical domain. Nonetheless, the paper still discusses opportunities of improvement, such as deploying neural network (NN) to accelerate the recovery of local particle of Lagrangian DA for the fine scale.</span> <span class="abstract-toggle" data-id="2509.14586">more</span>

    [:material-file-document: 2509.14586](https://arxiv.org/abs/2509.14586v1) · [:material-content-copy: BibTeX](bibtex/2509.14586.bib){ .bibtex-link }

-   #### FengWu-4DVar: Coupling the Data-driven Weather Forecasting Model with 4D Variational Assimilation

    ---

    *Yi Xiao, Lei Bai, Wei Xue, Kang Chen, Tao Han, Wanli Ouyang* · 2023

    <span class="abstract-snippet" id="snip-2312.12455">Weather forecasting is a crucial yet highly challenging task. With the maturity of Artificial Intelligence (AI), the emergence of data-driven weather forecasting models has opened up a new paradigm...</span><span class="abstract-full" id="full-2312.12455" hidden>Weather forecasting is a crucial yet highly challenging task. With the maturity of Artificial Intelligence (AI), the emergence of data-driven weather forecasting models has opened up a new paradigm for the development of weather forecasting systems. Despite the significant successes that have been achieved (e.g., surpassing advanced traditional physical models for global medium-range forecasting), existing data-driven weather forecasting models still rely on the analysis fields generated by the traditional assimilation and forecasting system, which hampers the significance of data-driven weather forecasting models regarding both computational cost and forecasting accuracy. In this work, we explore the possibility of coupling the data-driven weather forecasting model with data assimilation by integrating the global AI weather forecasting model, FengWu, with one of the most popular assimilation algorithms, Four-Dimensional Variational (4DVar) assimilation, and develop an AI-based cyclic weather forecasting system, FengWu-4DVar. FengWu-4DVar can incorporate observational data into the data-driven weather forecasting model and consider the temporal evolution of atmospheric dynamics to obtain accurate analysis fields for making predictions in a cycling manner without the help of physical models. Owning to the auto-differentiation ability of deep learning models, FengWu-4DVar eliminates the need of developing the cumbersome adjoint model, which is usually required in the traditional implementation of the 4DVar algorithm. Experiments on the simulated observational dataset demonstrate that FengWu-4DVar is capable of generating reasonable analysis fields for making accurate and efficient iterative predictions.</span> <span class="abstract-toggle" data-id="2312.12455">more</span>

    [:material-file-document: 2312.12455](https://arxiv.org/abs/2312.12455v2) · [:material-content-copy: BibTeX](bibtex/2312.12455.bib){ .bibtex-link }

</div>

## Ensembles (9)

<div class="grid cards" markdown>

-   #### Distillation and Interpretability of Ensemble Forecasts of ENSO Phase using Entropic Learning

    ---

    *Michael Groom, Davide Bassetti, Illia Horenko, Terence J. O'Kane* · 2026

    <span class="abstract-snippet" id="snip-2602.16857">This paper introduces a distillation framework for an ensemble of entropy-optimal Sparse Probabilistic Approximation (eSPA) models, trained exclusively on satellite-era observational and reanalysis...</span><span class="abstract-full" id="full-2602.16857" hidden>This paper introduces a distillation framework for an ensemble of entropy-optimal Sparse Probabilistic Approximation (eSPA) models, trained exclusively on satellite-era observational and reanalysis data to predict ENSO phase up to 24 months in advance. While eSPA ensembles yield state-of-the-art forecast skill, they are harder to interpret than individual eSPA models. We show how to compress the ensemble into a compact set of "distilled" models by aggregating the structure of only those ensemble members that make correct predictions. This process yields a single, diagnostically tractable model for each forecast lead time that preserves forecast performance while also enabling diagnostics that are impractical to implement on the full ensemble.   An analysis of the regime persistence of the distilled model "superclusters", as well as cross-lead clustering consistency, shows that the discretised system accurately captures the spatiotemporal dynamics of ENSO. By considering the effective dimension of the feature importance vectors, the complexity of the input space required for correct ENSO phase prediction is shown to peak when forecasts must cross the boreal spring predictability barrier. Spatial importance maps derived from the feature importance vectors are introduced to identify where predictive information resides in each field and are shown to include known physical precursors at certain lead times. Case studies of key events are also presented, showing how fields reconstructed from distilled model centroids trace the evolution from extratropical and inter-basin precursors to the mature ENSO state. Overall, the distillation framework enables a rigorous investigation of long-range ENSO predictability that complements real-time data-driven operational forecasts.</span> <span class="abstract-toggle" data-id="2602.16857">more</span>

    [:material-file-document: 2602.16857](https://arxiv.org/abs/2602.16857v1) · [:material-content-copy: BibTeX](bibtex/2602.16857.bib){ .bibtex-link }

    <span class="md-tag">probabilistic</span>

-   #### PuYun-LDM: A Latent Diffusion Model for High-Resolution Ensemble Weather Forecasts

    ---

    *Lianjun Wu, Shengchen Zhu, Yuxuan Liu, Liuyu Kai, Xiaoduan Feng, Duomin Wang, Wenshuo Liu et al.* · 2026

    <span class="abstract-snippet" id="snip-2602.11807">Latent diffusion models (LDMs) suffer from limited diffusability in high-resolution (<=0.25°) ensemble weather forecasting, where diffusability characterizes how easily a latent data distribution can...</span><span class="abstract-full" id="full-2602.11807" hidden>Latent diffusion models (LDMs) suffer from limited diffusability in high-resolution (<=0.25°) ensemble weather forecasting, where diffusability characterizes how easily a latent data distribution can be modeled by a diffusion process. Unlike natural image fields, meteorological fields lack task-agnostic foundation models and explicit semantic structures, making VFM-based regularization inapplicable. Moreover, existing frequency-based approaches impose identical spectral regularization across channels under a homogeneity assumption, which leads to uneven regularization strength under the inter-variable spectral heterogeneity in multivariate meteorological data. To address these challenges, we propose a 3D Masked AutoEncoder (3D-MAE) that encodes weather-state evolution features as an additional conditioning for the diffusion model, together with a Variable-Aware Masked Frequency Modeling (VA-MFM) strategy that adaptively selects thresholds based on the spectral energy distribution of each variable. Together, we propose PuYun-LDM, which enhances latent diffusability and achieves superior performance to ENS at short lead times while remaining comparable to ENS at longer horizons. PuYun-LDM generates a 15-day global forecast with a 6-hour temporal resolution in five minutes on a single NVIDIA H200 GPU, while ensemble forecasts can be efficiently produced in parallel.</span> <span class="abstract-toggle" data-id="2602.11807">more</span>

    [:material-file-document: 2602.11807](https://arxiv.org/abs/2602.11807v1) · [:material-content-copy: BibTeX](bibtex/2602.11807.bib){ .bibtex-link }

    <span class="md-tag">diffusion</span> <span class="md-tag">foundation-model</span>

-   #### Are we misdiagnosing ensemble forecast reliability? On the insufficiency of Spread-Error and rank-based reliability metrics

    ---

    *Arlan Dirkson, Mark Buehner* · 2025

    <span class="abstract-snippet" id="snip-2512.02160">It has been documented that Spread-Error equality and a flat rank histogram are necessary but insufficient for demonstrating ensemble forecast reliability. Nevertheless, these metrics are heavily...</span><span class="abstract-full" id="full-2512.02160" hidden>It has been documented that Spread-Error equality and a flat rank histogram are necessary but insufficient for demonstrating ensemble forecast reliability. Nevertheless, these metrics are heavily relied upon, both in the literature and at operational numerical weather prediction centers. In this study, we demonstrate theoretically why the Spread-Error relationship is necessary but insufficient for diagnosing reliability up to second order, even when mean bias is absent or accounted for. Assuming joint normality between ensemble members and the reference truth, we further show with idealized experiments that the same covariance structure responsible for this insufficiency also produces false diagnoses of reliability with the rank histogram and the reliability component of the continuous rank probability score. Under this structure and when the ensemble mean is meaningfully different from climatology, the truth lies among the least (most) extreme members when climatological variance is excessive (deficient) in each member. Importantly, this behavior is also shown to be plausible in operational ensemble weather forecasts. Combining these results with calibration principles from statistical postprocessing leads us to conclude that both perfect dispersion and underdispersion are ill-defined. When diagnostics are misinterpreted as indicating the latter, improper tuning can lead to further deterioration of forecast quality, even while improving Spread-Error and rank histogram behavior. To address these issues, we propose a new reliability diagnostic based on three easily computed statistics, motivated by the structure of the joint distribution of ensemble members and the reference truth up to second order. The diagnostic separates contributions to unreliability originating from climatology and predictability, enabling a more precise and robust characterization of ensemble behavior.</span> <span class="abstract-toggle" data-id="2512.02160">more</span>

    [:material-file-document: 2512.02160](https://arxiv.org/abs/2512.02160v1) · [:material-content-copy: BibTeX](bibtex/2512.02160.bib){ .bibtex-link }

-   #### Bridging the Gap Between Bayesian Deep Learning and Ensemble Weather Forecasts

    ---

    *Xinlei Xiong, Wenbo Hu, Shuxun Zhou, Kaifeng Bi, Lingxi Xie, Ying Liu, Richang Hong, Qi Tian* · 2025

    <span class="abstract-snippet" id="snip-2511.14218">Weather forecasting is fundamentally challenged by the chaotic nature of the atmosphere, necessitating probabilistic approaches to quantify uncertainty. While traditional ensemble prediction (EPS)...</span><span class="abstract-full" id="full-2511.14218" hidden>Weather forecasting is fundamentally challenged by the chaotic nature of the atmosphere, necessitating probabilistic approaches to quantify uncertainty. While traditional ensemble prediction (EPS) addresses this through computationally intensive simulations, recent advances in Bayesian Deep Learning (BDL) offer a promising but often disconnected alternative. We bridge these paradigms through a unified hybrid Bayesian Deep Learning framework for ensemble weather forecasting that explicitly decomposes predictive uncertainty into epistemic and aleatoric components, learned via variational inference and a physics-informed stochastic perturbation scheme modeling flow-dependent atmospheric dynamics, respectively. We further establish a unified theoretical framework that rigorously connects BDL and EPS, providing formal theorems that decompose total predictive uncertainty into epistemic and aleatoric components under the hybrid BDL framework. We validate our framework on the large-scale 40-year ERA5 reanalysis dataset (1979-2019) with 0.25° spatial resolution. Experimental results show that our method not only improves forecast accuracy and yields better-calibrated uncertainty quantification but also achieves superior computational efficiency compared to state-of-the-art probabilistic diffusion models. We commit to making our code open-source upon acceptance of this paper.</span> <span class="abstract-toggle" data-id="2511.14218">more</span>

    [:material-file-document: 2511.14218](https://arxiv.org/abs/2511.14218v1) · [:material-content-copy: BibTeX](bibtex/2511.14218.bib){ .bibtex-link }

    <span class="md-tag">diffusion</span> <span class="md-tag">physics-informed</span> <span class="md-tag">variational</span> <span class="md-tag">probabilistic</span>

-   #### The Role of Deep Mesoscale Eddies in Ensemble Forecast Performance

    ---

    *Justin Cooke, Kathleen Donohue, Clark D Rowley, Prasad G Thoppil, D Randolph Watts* · 2025

    <span class="abstract-snippet" id="snip-2511.09747">Present forecasting efforts rely on assimilation of observational data captured in the upper ocean (< 1000 m depth). These observations constrain the upper ocean and minimally influence the deep...</span><span class="abstract-full" id="full-2511.09747" hidden>Present forecasting efforts rely on assimilation of observational data captured in the upper ocean (< 1000 m depth). These observations constrain the upper ocean and minimally influence the deep ocean. Nevertheless, development of the full water column circulation critically depends upon the dynamical interactions between upper and deep fields. Forecasts demonstrate that the initialization of the deep field is influential for the development and evolution of the surface in the forecast. Deep initial conditions that better agree with observations have lower upper ocean uncertainty as the forecast progresses. Here, best and worst ensemble members in two 92-day forecasts are identified and contrasted in order to determine how the deep ocean differs between these groups. The forecasts cover the duration of the Loop Current Eddy Thor separation event, which coincides with available deep observations in the Gulf. Model member performance is assessed by comparing surface variables against verifying analysis and satellite altimeter data during the forecast time-period. Deep cyclonic and anticyclonic features are reviewed, and compared against deep observations, indicating subtle differences in locations of deep eddies at relevant times. These results highlight both the importance of deep circulation in the dynamics of the Loop Current system and more broadly motivate efforts to assimilate deep observations to better constrain the deep initial fields and improve surface predictions.</span> <span class="abstract-toggle" data-id="2511.09747">more</span>

    [:material-file-document: 2511.09747](https://arxiv.org/abs/2511.09747v1) · [:material-content-copy: BibTeX](bibtex/2511.09747.bib){ .bibtex-link }

-   #### Swift: An Autoregressive Consistency Model for Efficient Weather Forecasting

    ---

    *Jason Stock, Troy Arcomano, Rao Kotamarthi* · 2025

    <span class="abstract-snippet" id="snip-2509.25631">Diffusion models offer a physically grounded framework for probabilistic weather forecasting, but their typical reliance on slow, iterative solvers during inference makes them impractical for...</span><span class="abstract-full" id="full-2509.25631" hidden>Diffusion models offer a physically grounded framework for probabilistic weather forecasting, but their typical reliance on slow, iterative solvers during inference makes them impractical for subseasonal-to-seasonal (S2S) applications where long lead-times and domain-driven calibration are essential. To address this, we introduce Swift, a single-step consistency model that, for the first time, enables autoregressive finetuning of a probability flow model with a continuous ranked probability score (CRPS) objective. This eliminates the need for multi-model ensembling or parameter perturbations. Results show that Swift produces skillful 6-hourly forecasts that remain stable for up to 75 days, running $39\times$ faster than state-of-the-art diffusion baselines while achieving forecast skill competitive with the numerical-based, operational IFS ENS. This marks a step toward efficient and reliable ensemble forecasting from medium-range to seasonal-scales.</span> <span class="abstract-toggle" data-id="2509.25631">more</span>

    [:material-file-document: 2509.25631](https://arxiv.org/abs/2509.25631v1) · [:material-content-copy: BibTeX](bibtex/2509.25631.bib){ .bibtex-link }

    <span class="md-tag">diffusion</span> <span class="md-tag">probabilistic</span>

-   #### Data-Efficient Ensemble Weather Forecasting with Diffusion Models

    ---

    *Kevin Valencia, Ziyang Liu, Justin Cui* · 2025

    <span class="abstract-snippet" id="snip-2509.11047">Although numerical weather forecasting methods have dominated the field, recent advances in deep learning methods, such as diffusion models, have shown promise in ensemble weather forecasting....</span><span class="abstract-full" id="full-2509.11047" hidden>Although numerical weather forecasting methods have dominated the field, recent advances in deep learning methods, such as diffusion models, have shown promise in ensemble weather forecasting. However, such models are typically autoregressive and are thus computationally expensive. This is a challenge in climate science, where data can be limited, costly, or difficult to work with. In this work, we explore the impact of curated data selection on these autoregressive diffusion models. We evaluate several data sampling strategies and show that a simple time stratified sampling approach achieves performance similar to or better than full-data training. Notably, it outperforms the full-data model on certain metrics and performs only slightly worse on others while using only 20% of the training data. Our results demonstrate the feasibility of data-efficient diffusion training, especially for weather forecasting, and motivates future work on adaptive or model-aware sampling methods that go beyond random or purely temporal sampling.</span> <span class="abstract-toggle" data-id="2509.11047">more</span>

    [:material-file-document: 2509.11047](https://arxiv.org/abs/2509.11047v1) · [:material-content-copy: BibTeX](bibtex/2509.11047.bib){ .bibtex-link }

    <span class="md-tag">diffusion</span>

-   #### GenCast: Diffusion-based ensemble forecasting for medium-range weather

    ---

    *Ilan Price, Alvaro Sanchez-Gonzalez, Ferran Alet, Tom R. Andersson, Andrew El-Kadi, Dominic Masters et al.* · 2023

    <span class="abstract-snippet" id="snip-2312.15796">Weather forecasts are fundamentally uncertain, so predicting the range of probable weather scenarios is crucial for important decisions, from warning the public about hazardous weather, to planning...</span><span class="abstract-full" id="full-2312.15796" hidden>Weather forecasts are fundamentally uncertain, so predicting the range of probable weather scenarios is crucial for important decisions, from warning the public about hazardous weather, to planning renewable energy use. Here, we introduce GenCast, a probabilistic weather model with greater skill and speed than the top operational medium-range weather forecast in the world, the European Centre for Medium-Range Forecasts (ECMWF)'s ensemble forecast, ENS. Unlike traditional approaches, which are based on numerical weather prediction (NWP), GenCast is a machine learning weather prediction (MLWP) method, trained on decades of reanalysis data. GenCast generates an ensemble of stochastic 15-day global forecasts, at 12-hour steps and 0.25 degree latitude-longitude resolution, for over 80 surface and atmospheric variables, in 8 minutes. It has greater skill than ENS on 97.4% of 1320 targets we evaluated, and better predicts extreme weather, tropical cyclones, and wind power production. This work helps open the next chapter in operational weather forecasting, where critical weather-dependent decisions are made with greater accuracy and efficiency.</span> <span class="abstract-toggle" data-id="2312.15796">more</span>

    [:material-file-document: 2312.15796](https://arxiv.org/abs/2312.15796v2) · [:material-content-copy: BibTeX](bibtex/2312.15796.bib){ .bibtex-link }

    <span class="md-tag">probabilistic</span>

-   #### SwinVRNN: A Data-Driven Ensemble Forecasting Model via Learned Distribution Perturbation

    ---

    *Yuan Hu, Lei Chen, Zhibin Wang, Hao Li* · 2022

    <span class="abstract-snippet" id="snip-2205.13158">Data-driven approaches for medium-range weather forecasting are recently shown extraordinarily promising for ensemble forecasting for their fast inference speed compared to traditional numerical...</span><span class="abstract-full" id="full-2205.13158" hidden>Data-driven approaches for medium-range weather forecasting are recently shown extraordinarily promising for ensemble forecasting for their fast inference speed compared to traditional numerical weather prediction (NWP) models, but their forecast accuracy can hardly match the state-of-the-art operational ECMWF Integrated Forecasting System (IFS) model. Previous data-driven attempts achieve ensemble forecast using some simple perturbation methods, like initial condition perturbation and Monte Carlo dropout. However, they mostly suffer unsatisfactory ensemble performance, which is arguably attributed to the sub-optimal ways of applying perturbation. We propose a Swin Transformer-based Variational Recurrent Neural Network (SwinVRNN), which is a stochastic weather forecasting model combining a SwinRNN predictor with a perturbation module. SwinRNN is designed as a Swin Transformer-based recurrent neural network, which predicts future states deterministically. Furthermore, to model the stochasticity in prediction, we design a perturbation module following the Variational Auto-Encoder paradigm to learn multivariate Gaussian distributions of a time-variant stochastic latent variable from data. Ensemble forecasting can be easily achieved by perturbing the model features leveraging noise sampled from the learned distribution. We also compare four categories of perturbation methods for ensemble forecasting, i.e. fixed distribution perturbation, learned distribution perturbation, MC dropout, and multi model ensemble. Comparisons on WeatherBench dataset show the learned distribution perturbation method using our SwinVRNN model achieves superior forecast accuracy and reasonable ensemble spread due to joint optimization of the two targets. More notably, SwinVRNN surpasses operational IFS on surface variables of 2-m temperature and 6-hourly total precipitation at all lead times up to five days.</span> <span class="abstract-toggle" data-id="2205.13158">more</span>

    [:material-file-document: 2205.13158](https://arxiv.org/abs/2205.13158v1) · [:material-content-copy: BibTeX](bibtex/2205.13158.bib){ .bibtex-link }

    <span class="md-tag">transformer</span> <span class="md-tag">recurrent</span>

</div>

## Climate Modeling (27)

<div class="grid cards" markdown>

-   #### Capturing Aleatoric Uncertainty in Climate Models

    ---

    *Cornelia Gruber, Henri Funk, Magdalena Mittermeier, Helmut Küchenhoff, Göran Kauermann* · 2026

    <span class="abstract-snippet" id="snip-2604.15067">Internal climate variability arises from the climate system's inherently chaotic dynamics. Quantifying it is essential for climate science, as it enables risk-based decision-making and differentiates...</span><span class="abstract-full" id="full-2604.15067" hidden>Internal climate variability arises from the climate system's inherently chaotic dynamics. Quantifying it is essential for climate science, as it enables risk-based decision-making and differentiates between externally forced change and internal fluctuations. In statistical terms, natural variability corresponds to aleatoric uncertainty, i.e., irreducible stochastic variability. Despite this close conceptual alignment, the link between internal climate variability and aleatoric uncertainty has not yet been formalized. We establish a theoretical link by showing that member-to-member differences in single-model large ensembles provide a direct representation of aleatoric uncertainty. To quantify the spatio-temporal structure of aleatoric uncertainty, we employ generalized additive models. The proposed framework is validated through comparison with ERA5-Land reanalysis data, demonstrating that ensemble-derived estimates reproduce key spatial and temporal patterns of real-world variability. Applied to the water balance over the Iberian Peninsula, our approach reveals coherent variability structures and pronounced regional heterogeneity. We find a decline in variability in drought-prone regions and seasons, a pattern that strengthens under +3 °C global warming, implying an increased risk of persistent summer drought conditions. Beyond this application, the framework is climate-model agnostic and transferable to other variables and spatial scales, providing a statistical basis for quantifying internal climate variability as aleatoric uncertainty.</span> <span class="abstract-toggle" data-id="2604.15067">more</span>

    [:material-file-document: 2604.15067](https://arxiv.org/abs/2604.15067v1) · [:material-content-copy: BibTeX](bibtex/2604.15067.bib){ .bibtex-link }

-   #### Climate Downscaling with Stochastic Interpolants (CDSI)

    ---

    *Erik Larsson, Ramon Fuentes-Franco, Mikhail Ivanov, Fredrik Lindsten* · 2026

    <span class="abstract-snippet" id="snip-2603.03838">Global climate projections rely on computationally demanding Earth System Models (ESMs), which are typically limited to coarse spatial resolutions due to their high cost. To obtain high-resolution...</span><span class="abstract-full" id="full-2603.03838" hidden>Global climate projections rely on computationally demanding Earth System Models (ESMs), which are typically limited to coarse spatial resolutions due to their high cost. To obtain high-resolution projections for regions of interest, it is common to use Regional Climate Models (RCMs), which are driven by data produced by ESMs as boundary conditions. While more efficient than running ESMs at fine resolution, RCMs remain expensive and restrict the size of ensemble simulations. Inspired by recent advances in probabilistic machine learning for weather and climate, we introduce a data-driven climate downscaling method based on stochastic interpolants. Our approach efficiently transforms coarse ESM output into high-resolution regional climate projections at a fraction of the computational cost of traditional RCMs. Through extensive validation, we demonstrate that our method generates accurate regional ensembles, enabling both improved uncertainty quantification and broader use of high-resolution climate information.</span> <span class="abstract-toggle" data-id="2603.03838">more</span>

    [:material-file-document: 2603.03838](https://arxiv.org/abs/2603.03838v1) · [:material-content-copy: BibTeX](bibtex/2603.03838.bib){ .bibtex-link }

    <span class="md-tag">probabilistic</span>

-   #### The Effect of Planetary Rotation Period on Clouds in a Global Climate Model with a Bin Microphysics Scheme

    ---

    *Huanzhou Yang, Eric T. Wolf, Cheng-Cheng Liu, Yunqian Zhu, Owen B. Toon, Dorian S. Abbot* · 2026

    <span class="abstract-snippet" id="snip-2603.03767">Clouds are the largest source of uncertainty in climate simulations. For exoplanets, cloud simulation is particularly challenging because of the lack of observational data to tune parameterized cloud...</span><span class="abstract-full" id="full-2603.03767" hidden>Clouds are the largest source of uncertainty in climate simulations. For exoplanets, cloud simulation is particularly challenging because of the lack of observational data to tune parameterized cloud models. Here we apply Community Aerosol and Radiation Model for Atmospheres (CARMA), a size-resolved bin cloud microphysics model, to the atmospheric global climate model Community Atmosphere Model (CAM6) and simulate exoplanets with a range of planetary rotation rates. CARMA produces fewer liquid clouds than the native CAM6 parameterized cloud microphysics scheme (Morrison-Gettelman two-moment microphysics, MG), more ice clouds, and a significantly different ice cloud size distribution. Overall, this leads to a decrease in the magnitude of the net CRE by 4-10 $W/m^2$, which is unlikely to change the determination of habitability from a climate perspective in most cases. The difference in ice cloud size distribution is likely to strongly affect transmission spectral retrievals. Our work confirms that the MG parameterized cloud microphysics scheme can produce reasonable climate simulation when extrapolated to some exoplanet contexts and highlights the value of resolved cloud microphysics for evaluating parameterized schemes and for interpreting observations.</span> <span class="abstract-toggle" data-id="2603.03767">more</span>

    [:material-file-document: 2603.03767](https://arxiv.org/abs/2603.03767v1) · [:material-content-copy: BibTeX](bibtex/2603.03767.bib){ .bibtex-link }

-   #### Near-surface Extreme Wind Events and Their Responses to Climate Forcings in a Hierarchy of Global Climate Models

    ---

    *G. Zhang, M. Rao, I. Simpson, K. A. Reed, B. Medeiros, H. -H. Chou, T. Shaw* · 2026

    <span class="abstract-snippet" id="snip-2603.03483">Near-surface extreme winds profoundly affect human society, yet process-based understanding of their changes under climate forcings remains limited. This study systematically investigates the...</span><span class="abstract-full" id="full-2603.03483" hidden>Near-surface extreme winds profoundly affect human society, yet process-based understanding of their changes under climate forcings remains limited. This study systematically investigates the responses of high (HWE) and low (LWE) wind extremes (10-meter) to climate forcings using a hierarchy of climate model experiments from multiple general circulation models that participated in the Cloud Feedback Model Intercomparison Project. We analyze idealized atmosphere-only aquaplanet (Aqua) simulations and more realistic land-atmosphere (AMIP) simulations to identify robust responses to climate forcings and trace the sources of structural uncertainty. In Aqua simulations, tropical LWE changes exhibit large inter-model spread, which can be traced to dynamically distinct representations of low-pressure systems between models. In contrast, extratropical HWE intensify robustly with surface warming, linked to the strengthening of high-latitude extratropical cyclones. The AMIP simulations confirm the robust intensification of extratropical HWE. The more realistic boundary conditions in AMIP simulations act as a constraint, reducing inter-model spread in tropical zonal means compared to Aqua simulations. A comparison of uniform and patterned 4-K warming experiments suggests that the global magnitude of warming, rather than the specific warming pattern, dominates the large-scale responses of wind extremes. However, regional projections of extreme wind changes, especially over land, remain highly uncertain due to divergences in model physics. Case studies reveal that major disagreements in HWE changes can stem from fundamental differences in representing the type and seasonality of extreme-producing weather systems. Our results underscore that reducing uncertainty in regional wind projections requires constraining the physical representation of weather systems in climate models.</span> <span class="abstract-toggle" data-id="2603.03483">more</span>

    [:material-file-document: 2603.03483](https://arxiv.org/abs/2603.03483v1) · [:material-content-copy: BibTeX](bibtex/2603.03483.bib){ .bibtex-link }

-   #### PICASO 4.0: Clouds and Photochemistry in Climate Models of Brown Dwarfs and Exoplanets

    ---

    *James Mang, Natasha E. Batalha, Caroline V. Morley, Nicholas F. Wogan, Sagnick Mukherjee et al.* · 2026

    <span class="abstract-snippet" id="snip-2602.22468">We present a major update to the open-source atmospheric modeling package \texttt{PICASO}, designed for simulating the thermal structure and spectra of hydrogen-rich atmospheres of brown dwarfs and...</span><span class="abstract-full" id="full-2602.22468" hidden>We present a major update to the open-source atmospheric modeling package \texttt{PICASO}, designed for simulating the thermal structure and spectra of hydrogen-rich atmospheres of brown dwarfs and exoplanets. This release, \texttt{PICASO 4.0}, expands upon the existing radiative-convective equilibrium model framework by incorporating several new capabilities. Key additions include the integration of \texttt{Virga} for self-consistent cloud modeling, new flexible treatments for rainout and cold trapping of volatile species, and support for photochemistry. We also introduce a parameterized energy injection scheme to simulate additional external or internal heating processes. These features are motivated by lessons from recent JWST observations that reveal the prevalence of non-equilibrium chemistry and clouds. We benchmark the new functionalities against previously published results in the literature, including the Sonora Diamondback grid, energy injected atmospheres, patchy cloud models, and other photochemical models of WASP-39b. \texttt{PICASO} continues to be actively developed as an open-source package aimed at enabling reproducible, community-driven atmospheric modeling of all substellar objects.</span> <span class="abstract-toggle" data-id="2602.22468">more</span>

    [:material-file-document: 2602.22468](https://arxiv.org/abs/2602.22468v1) · [:material-content-copy: BibTeX](bibtex/2602.22468.bib){ .bibtex-link }

-   #### Decision-oriented benchmarking to transform AI weather forecast access: Application to the Indian monsoon

    ---

    *Rajat Masiwal, Colin Aitken, Adam Marchakitus, Mayank Gupta, Katherine Kowal, Hamid A. Pahlavan et al.* · 2026

    <span class="abstract-snippet" id="snip-2602.03767">Artificial intelligence weather prediction (AIWP) models now often outperform traditional physics-based models on common metrics while requiring orders-of-magnitude less computing resources and time....</span><span class="abstract-full" id="full-2602.03767" hidden>Artificial intelligence weather prediction (AIWP) models now often outperform traditional physics-based models on common metrics while requiring orders-of-magnitude less computing resources and time. Open-access AIWP models thus hold promise as transformational tools for helping low- and middle-income populations make decisions in the face of high-impact weather shocks. Yet, current approaches to evaluating AIWP models focus mainly on aggregated meteorological metrics without considering local stakeholders' needs in decision-oriented, operational frameworks. Here, we introduce such a framework that connects meteorology, AI, and social sciences. As an example, we apply it to the 150-year-old problem of Indian monsoon forecasting, focusing on benefits to rain-fed agriculture, which is highly susceptible to climate change. AIWP models skillfully predict an agriculturally relevant onset index at regional scales weeks in advance when evaluated out-of-sample using deterministic and probabilistic metrics. This framework informed a government-led effort in 2025 to send 38 million Indian farmers AI-based monsoon onset forecasts, which captured an unusual weeks-long pause in monsoon progression. This decision-oriented benchmarking framework provides a key component of a blueprint for harnessing the power of AIWP models to help large vulnerable populations adapt to weather shocks in the face of climate variability and change.</span> <span class="abstract-toggle" data-id="2602.03767">more</span>

    [:material-file-document: 2602.03767](https://arxiv.org/abs/2602.03767v1) · [:material-content-copy: BibTeX](bibtex/2602.03767.bib){ .bibtex-link }

    <span class="md-tag">physics-informed</span> <span class="md-tag">probabilistic</span>

-   #### Making Tunable Parameters State-Dependent in Weather and Climate Models with Reinforcement Learning

    ---

    *Pritthijit Nath, Sebastian Schemm, Henry Moss, Peter Haynes, Emily Shuckburgh, Mark J. Webb* · 2026

    <span class="abstract-snippet" id="snip-2601.04268">Weather and climate models rely on parametrisations to represent unresolved sub-grid processes. Traditional schemes rely on fixed coefficients that are weakly constrained and tuned offline,...</span><span class="abstract-full" id="full-2601.04268" hidden>Weather and climate models rely on parametrisations to represent unresolved sub-grid processes. Traditional schemes rely on fixed coefficients that are weakly constrained and tuned offline, contributing to persistent biases that limit their ability to adapt to the underlying physics. This study presents a framework that learns components of parametrisation schemes online as a function of the evolving model state using reinforcement learning (RL) and evaluates the resulting RL-driven parameter updates across a hierarchy of idealised testbeds spanning a simple climate bias correction (SCBC), a radiative-convective equilibrium (RCE), and a zonal mean energy balance model (EBM) with both single-agent and federated multi-agent settings. Across nine RL algorithms, Truncated Quantile Critics (TQC), Deep Deterministic Policy Gradient (DDPG), and Twin Delayed DDPG (TD3) achieved the highest skill and the most stable convergence across configurations, with performance assessed against a static baseline using area-weighted RMSE, temperature profile and pressure-level diagnostics. For the EBM, single-agent RL outperformed static parameter tuning with the strongest gains in tropical and mid-latitude bands, while federated RL on multi-agent setups enabled geographically specialised control and faster convergence, with a six-agent DDPG configuration using frequent aggregation yielding the lowest area-weighted RMSE across the tropics and mid-latitudes. The learnt corrections were also physically meaningful as agents modulated EBM radiative parameters to reduce meridional biases, adjusted RCE lapse rates to match vertical temperature errors, and stabilised SCBC heating increments to limit drift. Overall, results highlight RL to deliver skilful state-dependent, and regime-aware parametrisations, offering a scalable pathway for online learning within numerical models.</span> <span class="abstract-toggle" data-id="2601.04268">more</span>

    [:material-file-document: 2601.04268](https://arxiv.org/abs/2601.04268v1) · [:material-content-copy: BibTeX](bibtex/2601.04268.bib){ .bibtex-link }

    <span class="md-tag">reinforcement-learning</span>

-   #### Quantum Bayesian Optimization for the Automatic Tuning of Lorenz-96 as a Surrogate Climate Model

    ---

    *Paul J. Christiansen, Daniel Ohl de Mello, Cedric Brügmann, Steffen Hien, Felix Herbort et al.* · 2025

    <span class="abstract-snippet" id="snip-2512.20437">In this work, we propose a hybrid quantum-inspired heuristic for automatically tuning the Lorenz-96 model -- a simple proxy to describe atmospheric dynamics, yet exhibiting chaotic behavior. Building...</span><span class="abstract-full" id="full-2512.20437" hidden>In this work, we propose a hybrid quantum-inspired heuristic for automatically tuning the Lorenz-96 model -- a simple proxy to describe atmospheric dynamics, yet exhibiting chaotic behavior. Building on the history matching framework by Lguensat et al. (2023), we fully automate the tuning process with a new convergence criterion and propose replacing classical Gaussian process emulators with quantum counterparts. We benchmark three quantum kernel architectures, distinguished by their quantum feature map circuits. A dimensionality argument implies, in principle, an increased expressivity of the quantum kernels over their classical competitors. For each kernel type, we perform an extensive hyperparameter optimization of our tuning algorithm. We confirm the validity of a quantum-inspired approach based on statevector simulation by numerically demonstrating the superiority of two studied quantum kernels over the canonical classical RBF kernel. Finally, we discuss the pathway towards real quantum hardware, mainly driven by a transition to shot-based simulations and evaluating quantum kernels via randomized measurements, which can mitigate the effect of gate errors. The very low qubit requirements and moderate circuit depths, together with a minimal number of trainable circuit parameters, make our method particularly NISQ-friendly.</span> <span class="abstract-toggle" data-id="2512.20437">more</span>

    [:material-file-document: 2512.20437](https://arxiv.org/abs/2512.20437v1) · [:material-content-copy: BibTeX](bibtex/2512.20437.bib){ .bibtex-link }

    <span class="md-tag">probabilistic</span>

-   #### Quantum Machine Learning for Climate Modelling

    ---

    *Mierk Schwabe, Lorenzo Pastori, Valentina Sarandrea, Veronika Eyring* · 2025

    <span class="abstract-snippet" id="snip-2512.14208">Quantum machine learning (QML) is making rapid progress, and QML-based models hold the promise of quantum advantages such as potentially higher expressivity and generalizability than their classical...</span><span class="abstract-full" id="full-2512.14208" hidden>Quantum machine learning (QML) is making rapid progress, and QML-based models hold the promise of quantum advantages such as potentially higher expressivity and generalizability than their classical counterparts. Here, we present work on using a quantum neural net (QNN) to develop a parameterization of cloud cover for an Earth system model (ESM). ESMs are needed for predicting and projecting climate change, and can be improved in hybrid models incorporating both traditional physics-based components as well as machine learning (ML) models. We show that a QNN can predict cloud cover with a performance similar to a classical NN with the same number of free parameters and significantly better than the traditional scheme. We also analyse the learning capability of the QNN in comparison to the classical NN and show that, at least for our example, QNNs learn more consistent relationships than classical NNs.</span> <span class="abstract-toggle" data-id="2512.14208">more</span>

    [:material-file-document: 2512.14208](https://arxiv.org/abs/2512.14208v1) · [:material-content-copy: BibTeX](bibtex/2512.14208.bib){ .bibtex-link }

    <span class="md-tag">physics-informed</span>

-   #### An intercomparison of generative machine learning methods for downscaling precipitation at fine spatial scales

    ---

    *Bryn Ward-Leikis, Neelesh Rampal, Yun Sing Koh, Peter B. Gibson, Hong-Yang Liu, Vassili Kitsios et al.* · 2025

    <span class="abstract-snippet" id="snip-2512.13987">Machine learning (ML) offers a computationally efficient approach for generating large ensembles of high-resolution climate projections, but deterministic ML methods often smooth fine-scale...</span><span class="abstract-full" id="full-2512.13987" hidden>Machine learning (ML) offers a computationally efficient approach for generating large ensembles of high-resolution climate projections, but deterministic ML methods often smooth fine-scale structures and underestimate extremes. While stochastic generative models show promise for predicting fine-scale weather and extremes, few studies have compared their performance under present-day and future climates. This study compares a previously developed conditional Generative Adversarial Network (cGAN) with an intensity constraint against different configurations of diffusion models for downscaling daily precipitation from a regional climate model (RCM) over Aotearoa New Zealand. Model skill is comprehensively assessed across spatial structure, distributional metrics, means, extremes, and their respective climate change signals. Both generative approaches outperform the deterministic baseline across most metrics and exhibit similar overall skill. Diffusion models better predict the fine-scale spatial structure of precipitation and the length of dry spells, but underestimate climate change signals for extreme precipitation compared to the ground truth RCMs. In contrast, cGANs achieve comparable skill for most metrics while better predicting the overall precipitation distribution and climate change responses for extremes at a fraction of the computational cost. These results demonstrate that while diffusion models can readily generate predictions with greater visual "realism", they do not necessarily better preserve climate change responses compared to cGANs with intensity constraints. At present, incorporating constraints into diffusion models remains challenging compared to cGANs, but may represent an opportunity to further improve skill for predicting climate change responses.</span> <span class="abstract-toggle" data-id="2512.13987">more</span>

    [:material-file-document: 2512.13987](https://arxiv.org/abs/2512.13987v1) · [:material-content-copy: BibTeX](bibtex/2512.13987.bib){ .bibtex-link }

    <span class="md-tag">diffusion</span> <span class="md-tag">GAN</span>

-   #### Bridging CORDEX and CMIP6: Machine Learning Downscaling for Wind and Solar Energy Droughts in Central Europe

    ---

    *Nina Effenberger, Maxim Samarin, Maybritt Schillinger, Reto Knutti* · 2025

    <span class="abstract-snippet" id="snip-2512.07429">Reliable regional climate information is essential for assessing the impacts of climate change and for planning in sectors such as renewable energy; yet, producing high-resolution projections through...</span><span class="abstract-full" id="full-2512.07429" hidden>Reliable regional climate information is essential for assessing the impacts of climate change and for planning in sectors such as renewable energy; yet, producing high-resolution projections through coordinated initiatives like CORDEX that run multiple physical regional climate models is both computationally demanding and difficult to organize. Machine learning emulators that learn the mapping between global and regional climate fields offer a promising way to address these limitations. Here we introduce the application of such an emulator: trained on CMIP5 and CORDEX simulations, it reproduces regional climate model data with sufficient accuracy. When applied to CMIP6 simulations not seen during training, it also produces realistic results, indicating stable performance. Using CORDEX data, CMIP5 and CMIP6 simulations, as well as regional data generated by two machine learning models, we analyze the co-occurrence of low wind speed and low solar radiation and find indications that the number of such energy drought days is likely to decrease in the future. Our results highlight that downscaling with machine learning emulators provides an efficient complement to efforts such as CORDEX, supplying the higher-resolution information required for impact assessments.</span> <span class="abstract-toggle" data-id="2512.07429">more</span>

    [:material-file-document: 2512.07429](https://arxiv.org/abs/2512.07429v1) · [:material-content-copy: BibTeX](bibtex/2512.07429.bib){ .bibtex-link }

-   #### EcoCast: A Spatio-Temporal Model for Continual Biodiversity and Climate Risk Forecasting

    ---

    *Hammed A. Akande, Abdulrauf A. Gidado* · 2025

    <span class="abstract-snippet" id="snip-2512.02260">Increasing climate change and habitat loss are driving unprecedented shifts in species distributions. Conservation professionals urgently need timely, high-resolution predictions of biodiversity...</span><span class="abstract-full" id="full-2512.02260" hidden>Increasing climate change and habitat loss are driving unprecedented shifts in species distributions. Conservation professionals urgently need timely, high-resolution predictions of biodiversity risks, especially in ecologically diverse regions like Africa. We propose EcoCast, a spatio-temporal model designed for continual biodiversity and climate risk forecasting. Utilizing multisource satellite imagery, climate data, and citizen science occurrence records, EcoCast predicts near-term (monthly to seasonal) shifts in species distributions through sequence-based transformers that model spatio-temporal environmental dependencies. The architecture is designed with support for continual learning to enable future operational deployment with new data streams. Our pilot study in Africa shows promising improvements in forecasting distributions of selected bird species compared to a Random Forest baseline, highlighting EcoCast's potential to inform targeted conservation policies. By demonstrating an end-to-end pipeline from multi-modal data ingestion to operational forecasting, EcoCast bridges the gap between cutting-edge machine learning and biodiversity management, ultimately guiding data-driven strategies for climate resilience and ecosystem conservation throughout Africa.</span> <span class="abstract-toggle" data-id="2512.02260">more</span>

    [:material-file-document: 2512.02260](https://arxiv.org/abs/2512.02260v1) · [:material-content-copy: BibTeX](bibtex/2512.02260.bib){ .bibtex-link }

    <span class="md-tag">transformer</span>

-   #### Efficient Regional Storm Surge Surrogate Model Training Strategy Under Evolving Landscape and Climate Scenarios

    ---

    *Ziyue Liu, Mohammad Ahmadi Gharehtoragh, Brenna Kari Losch, David R. Johnson* · 2025

    <span class="abstract-snippet" id="snip-2511.07269">Coastal communities can be exposed to risk from catastrophic storm-induced coastal hazards, causing major global losses each year. Recent advances in computational power have enabled the integration...</span><span class="abstract-full" id="full-2511.07269" hidden>Coastal communities can be exposed to risk from catastrophic storm-induced coastal hazards, causing major global losses each year. Recent advances in computational power have enabled the integration of machine learning (ML) into coastal hazard modeling, particularly for storm surge prediction. Given the potential variation in future climate and landscape conditions, efficient predictive models that can incorporate multiple future scenarios are needed. Existing studies built a framework for training ML models using storm surge simulation data under different potential future climate and landscape scenarios. However, storm surge simulation data under designed future scenarios require computationally expensive numerical simulations of synthetic storm suites over extensive geospatial grids. As the number of designed scenarios increases, the computational cost associated with both numerical simulation and ML training increases rapidly. This study introduces a cost-effective reduction strategy that incorporates new scenario data while minimizing computational burden. The approach reduces training data across three dimensions: (1) grid points, (2) input features, and (3) storm suite size. Reducing the storm suite size is especially effective in cutting simulation costs. Model performance was evaluated using different ML algorithms, showing consistent effectiveness. When trained on 5,000 of 80,000 grid points, 10 of 12 features, and 40 of 90 storms, the model achieved an R=0.93, comparable to that of models trained on the full dataset, with substantially lower computational expense.</span> <span class="abstract-toggle" data-id="2511.07269">more</span>

    [:material-file-document: 2511.07269](https://arxiv.org/abs/2511.07269v1) · [:material-content-copy: BibTeX](bibtex/2511.07269.bib){ .bibtex-link }

-   #### Deep Learning-Driven Downscaling for Climate Risk Assessment of Projected Temperature Extremes in the Nordic Region

    ---

    *Parthiban Loganathan, Elias Zea, Ricardo Vinuesa, Evelyn Otero* · 2025

    <span class="abstract-snippet" id="snip-2511.03770">Rapid changes and increasing climatic variability across the widely varied Koppen-Geiger regions of northern Europe generate significant needs for adaptation. Regional planning needs high-resolution...</span><span class="abstract-full" id="full-2511.03770" hidden>Rapid changes and increasing climatic variability across the widely varied Koppen-Geiger regions of northern Europe generate significant needs for adaptation. Regional planning needs high-resolution projected temperatures. This work presents an integrative downscaling framework that incorporates Vision Transformer (ViT), Convolutional Long Short-Term Memory (ConvLSTM), and Geospatial Spatiotemporal Transformer with Attention and Imbalance-Aware Network (GeoStaNet) models. The framework is evaluated with a multicriteria decision system, Deep Learning-TOPSIS (DL-TOPSIS), for ten strategically chosen meteorological stations encompassing the temperate oceanic (Cfb), subpolar oceanic (Cfc), warm-summer continental (Dfb), and subarctic (Dfc) climate regions. Norwegian Earth System Model (NorESM2-LM) Coupled Model Intercomparison Project Phase 6 (CMIP6) outputs were bias-corrected during the 1951-2014 period and subsequently validated against earlier observations of day-to-day temperature metrics and diurnal range statistics. The ViT showed improved performance (Root Mean Squared Error (RMSE): 1.01 degrees C; R^2: 0.92), allowing for production of credible downscaled projections. Under the SSP5-8.5 scenario, the Dfc and Dfb climate zones are projected to warm by 4.8 degrees C and 3.9 degrees C, respectively, by 2100, with expansion in the diurnal temperature range by more than 1.5 degrees C. The Time of Emergence signal first appears in subarctic winter seasons (Dfc: approximately 2032), signifying an urgent need for adaptation measures. The presented framework offers station-based, high-resolution estimates of uncertainties and extremes, with direct uses for adaptation policy over high-latitude regions with fast environmental change.</span> <span class="abstract-toggle" data-id="2511.03770">more</span>

    [:material-file-document: 2511.03770](https://arxiv.org/abs/2511.03770v1) · [:material-content-copy: BibTeX](bibtex/2511.03770.bib){ .bibtex-link }

    <span class="md-tag">transformer</span> <span class="md-tag">recurrent</span>

-   #### A Probabilistic U-Net Approach to Downscaling Climate Simulations

    ---

    *Maryam Alipourhajiagha, Pierre-Louis Lemaire, Youssef Diouane, Julie Carreau* · 2025

    <span class="abstract-snippet" id="snip-2511.03197">Climate models are limited by heavy computational costs, often producing outputs at coarse spatial resolutions, while many climate change impact studies require finer scales. Statistical downscaling...</span><span class="abstract-full" id="full-2511.03197" hidden>Climate models are limited by heavy computational costs, often producing outputs at coarse spatial resolutions, while many climate change impact studies require finer scales. Statistical downscaling bridges this gap, and we adapt the probabilistic U-Net for this task, combining a deterministic U-Net backbone with a variational latent space to capture aleatoric uncertainty. We evaluate four training objectives, afCRPS and WMSE-MS-SSIM with three settings for downscaling precipitation and temperature from $16\times$ coarser resolution. Our main finding is that WMSE-MS-SSIM performs well for extremes under certain settings, whereas afCRPS better captures spatial variability across scales.</span> <span class="abstract-toggle" data-id="2511.03197">more</span>

    [:material-file-document: 2511.03197](https://arxiv.org/abs/2511.03197v1) · [:material-content-copy: BibTeX](bibtex/2511.03197.bib){ .bibtex-link }

    <span class="md-tag">CNN</span> <span class="md-tag">probabilistic</span>

-   #### Intercomparison of a High-Resolution Regional Climate Model Ensemble for Catchment-Scale Water Cycle Processes under Human Influence

    ---

    *J. L. Roque, F. Da Silva Lopes, J. A. Giles, B. D. Gutknecht, B. Schalge, Y. Zhang, M. Ferro et al.* · 2025

    <span class="abstract-snippet" id="snip-2511.02799">Understanding regional hydroclimatic variability and its drivers is essential for anticipating the impacts of climate change on water resources and sustainability. Yet, considerable uncertainty...</span><span class="abstract-full" id="full-2511.02799" hidden>Understanding regional hydroclimatic variability and its drivers is essential for anticipating the impacts of climate change on water resources and sustainability. Yet, considerable uncertainty remains in the simulation of the coupled land atmosphere water and energy cycles, largely due to structural model limitations, simplified process representations, and insufficient spatial resolution. Within the framework of the Collaborative Research Center 1502 DETECT, this study presents a coordinated intercomparison of regional climate model simulations designed for water cycle process analysis over Europe. We analyze the performance of simulations using the ICON and TSMP1 model systems and covering the period from 1990 to 2020, comparing against reference datasets (E-OBS, GPCC, and GLEAM). We focus on 2 m air temperature, precipitation and evapotranspiration over four representative basins, the Ebro, Po, Rhine, and Tisa, within the EURO CORDEX domain.   Our analysis reveals systematic cold biases across all basins and seasons, with ICON generally outperforming TSMP1. Precipitation biases exhibit substantial spread, particularly in summer, reflecting the persistent challenge of accurately simulating precipitation. ICON tends to underestimate evapotranspiration, while TSMP1 performs better some seasons. Sensitivity experiments further indicate that the inclusion of irrigation improves simulation performance in the Po basin, which is intensively irrigated, and that higher-resolution sea surface temperature forcing data improves the overall precipitation representation. This baseline evaluation provides a first assessment of the DETECT multimodel ensemble and highlights key structural differences influencing model skill across hydroclimatic regimes.</span> <span class="abstract-toggle" data-id="2511.02799">more</span>

    [:material-file-document: 2511.02799](https://arxiv.org/abs/2511.02799v1) · [:material-content-copy: BibTeX](bibtex/2511.02799.bib){ .bibtex-link }

-   #### A Stochastic Parameterization of Non-Orographic Gravity Waves Induced Mixing for Mars Planetary Climate Model

    ---

    *Jiandong Liu, Ehouarn Millour, François Forget, François Lott, Jean-Yves Chaufray* · 2025

    <span class="abstract-snippet" id="snip-2510.20410">This paper presents a formalism of mixing induced by non-orographic gravity waves (GWs) to integrate with the stochastic GWs scheme in the Mars Planetary Climate Model. We derive the formalism of GWs...</span><span class="abstract-full" id="full-2510.20410" hidden>This paper presents a formalism of mixing induced by non-orographic gravity waves (GWs) to integrate with the stochastic GWs scheme in the Mars Planetary Climate Model. We derive the formalism of GWs and their mixing under the same assumptions, integrating the two schemes within a unified framework. Specifically, a surface-to-exosphere parameterization of GW-induced turbulence has been derived in terms of the eddy diffusion coefficient. Simulations show that the coefficient is on the order of 1E4 to 1E9 cm2 s-1 and a turbopause is at altitudes of 70 to 140 km, varying with seasons. The triggered mixing has minor effects on model temperatures, yet it substantially impacts upper atmospheric abundances. Simulations are consistent with observations from the Mars Climate Sounder and the Neutral Gas and Ion Mass Spectrometer. Mixing enhances the tracer transports in the middle and upper atmosphere, governing the dynamics of these regions. The scheme reveals how non-orographic GW-induced turbulence can regulate upper atmospheric processes, such as tracer escape.</span> <span class="abstract-toggle" data-id="2510.20410">more</span>

    [:material-file-document: 2510.20410](https://arxiv.org/abs/2510.20410v1) · [:material-content-copy: BibTeX](bibtex/2510.20410.bib){ .bibtex-link }

-   #### OmniCast: A Masked Latent Diffusion Model for Weather Forecasting Across Time Scales

    ---

    *Tung Nguyen, Tuan Pham, Troy Arcomano, Veerabhadra Kotamarthi, Ian Foster, Sandeep Madireddy et al.* · 2025

    <span class="abstract-snippet" id="snip-2510.18707">Accurate weather forecasting across time scales is critical for anticipating and mitigating the impacts of climate change. Recent data-driven methods based on deep learning have achieved significant...</span><span class="abstract-full" id="full-2510.18707" hidden>Accurate weather forecasting across time scales is critical for anticipating and mitigating the impacts of climate change. Recent data-driven methods based on deep learning have achieved significant success in the medium range, but struggle at longer subseasonal-to-seasonal (S2S) horizons due to error accumulation in their autoregressive approach. In this work, we propose OmniCast, a scalable and skillful probabilistic model that unifies weather forecasting across timescales. OmniCast consists of two components: a VAE model that encodes raw weather data into a continuous, lower-dimensional latent space, and a diffusion-based transformer model that generates a sequence of future latent tokens given the initial conditioning tokens. During training, we mask random future tokens and train the transformer to estimate their distribution given conditioning and visible tokens using a per-token diffusion head. During inference, the transformer generates the full sequence of future tokens by iteratively unmasking random subsets of tokens. This joint sampling across space and time mitigates compounding errors from autoregressive approaches. The low-dimensional latent space enables modeling long sequences of future latent states, allowing the transformer to learn weather dynamics beyond initial conditions. OmniCast performs competitively with leading probabilistic methods at the medium-range timescale while being 10x to 20x faster, and achieves state-of-the-art performance at the subseasonal-to-seasonal scale across accuracy, physics-based, and probabilistic metrics. Furthermore, we demonstrate that OmniCast can generate stable rollouts up to 100 years ahead. Code and model checkpoints are available at https://github.com/tung-nd/omnicast.</span> <span class="abstract-toggle" data-id="2510.18707">more</span>

    [:material-file-document: 2510.18707](https://arxiv.org/abs/2510.18707v1) · [:fontawesome-brands-github:](https://github.com/tung-nd/omnicast) · [:material-content-copy: BibTeX](bibtex/2510.18707.bib){ .bibtex-link }

    <span class="md-tag">transformer</span> <span class="md-tag">diffusion</span> <span class="md-tag">physics-informed</span> <span class="md-tag">variational</span> <span class="md-tag">probabilistic</span>

-   #### Specification and Verification for Climate Modeling: Formalization Leading to Impactful Tooling

    ---

    *Alper Altuntas, Allison H. Baker, John Baugh, Ganesh Gopalakrishnan, Stephen F. Siegel* · 2025

    <span class="abstract-snippet" id="snip-2510.13425">Earth System Models (ESMs) are critical for understanding past climates and projecting future scenarios. However, the complexity of these models, which include large code bases, a wide community of...</span><span class="abstract-full" id="full-2510.13425" hidden>Earth System Models (ESMs) are critical for understanding past climates and projecting future scenarios. However, the complexity of these models, which include large code bases, a wide community of developers, and diverse computational platforms, poses significant challenges for software quality assurance. The increasing adoption of GPUs and heterogeneous architectures further complicates verification efforts. Traditional verification methods often rely on bitwise reproducibility, which is not always feasible, particularly under new compilers or hardware. Manual expert evaluation, on the other hand, is subjective and time-consuming. Formal methods offer a mathematically rigorous alternative, yet their application in ESM development has been limited due to the lack of climate model-specific representations and tools. Here, we advocate for the broader adoption of formal methods in climate modeling. In particular, we identify key aspects of ESMs that are well suited to formal specification and introduce abstraction approaches for a tailored framework. To demonstrate this approach, we present a case study using CIVL model checker to formally verify a bug fix in an ocean mixing parameterization scheme. Our goal is to develop accessible, domain-specific formal tools that enhance model confidence and support more efficient and reliable ESM development.</span> <span class="abstract-toggle" data-id="2510.13425">more</span>

    [:material-file-document: 2510.13425](https://arxiv.org/abs/2510.13425v1) · [:material-content-copy: BibTeX](bibtex/2510.13425.bib){ .bibtex-link }

-   #### Beyond the Training Data: Confidence-Guided Mixing of Parameterizations in a Hybrid AI-Climate Model

    ---

    *Helge Heuer, Tom Beucler, Mierk Schwabe, Julien Savre, Manuel Schlund, Veronika Eyring* · 2025

    <span class="abstract-snippet" id="snip-2510.08107">Persistent systematic errors in Earth system models (ESMs) arise from difficulties in representing the full diversity of subgrid, multiscale atmospheric convection and turbulence. Machine learning...</span><span class="abstract-full" id="full-2510.08107" hidden>Persistent systematic errors in Earth system models (ESMs) arise from difficulties in representing the full diversity of subgrid, multiscale atmospheric convection and turbulence. Machine learning (ML) parameterizations trained on short high-resolution simulations show strong potential to reduce these errors. However, stable long-term atmospheric simulations with hybrid (physics + ML) ESMs remain difficult, as neural networks (NNs) trained offline often destabilize online runs. Training convection parameterizations directly on coarse-grained data is challenging, notably because scales cannot be cleanly separated. This issue is mitigated using data from superparameterized simulations, which provide clearer scale separation. Yet, transferring a parameterization from one ESM to another remains difficult due to distribution shifts that induce large inference errors. Here, we present a proof-of-concept where a ClimSim-trained, physics-informed NN convection parameterization is successfully transferred to ICON-A. The scheme is (a) trained on adjusted ClimSim data with subtracted radiative tendencies, and (b) integrated into ICON-A. The NN parameterization predicts its own error, enabling mixing with a conventional convection scheme when confidence is low, thus making the hybrid AI-physics model tunable with respect to observations and reanalysis through mixing parameters. This improves process understanding by constraining convective tendencies across column water vapor, lower-tropospheric stability, and geographical conditions, yielding interpretable regime behavior. In AMIP-style setups, several hybrid configurations outperform the default convection scheme (e.g., improved precipitation statistics). With additive input noise during training, both hybrid and pure-ML schemes lead to stable simulations and remain physically consistent for at least 20 years.</span> <span class="abstract-toggle" data-id="2510.08107">more</span>

    [:material-file-document: 2510.08107](https://arxiv.org/abs/2510.08107v1) · [:material-content-copy: BibTeX](bibtex/2510.08107.bib){ .bibtex-link }

    <span class="md-tag">physics-informed</span>

-   #### Climate Model Tuning with Online Synchronization-Based Parameter Estimation

    ---

    *Jordan Seneca, Suzanne Bintanja, Frank M. Selten* · 2025

    <span class="abstract-snippet" id="snip-2510.06180">In climate science, the tuning of climate models is a computationally intensive problem due to the combination of the high-dimensionality of the system state and long integration times. Here we...</span><span class="abstract-full" id="full-2510.06180" hidden>In climate science, the tuning of climate models is a computationally intensive problem due to the combination of the high-dimensionality of the system state and long integration times. Here we demonstrate the potential of a parameter estimation algorithm which makes use of synchronization to tune a global atmospheric model at modest computational costs. We first use it to directly optimize internal model parameters. We then apply the algorithm to the weights of each member of a supermodel ensemble to optimize the overall predictions. In both cases, the algorithm is able to find parameters which result in reduced errors in the climatology of the model. Finally, we introduce a novel approach which combines both methods called adaptive supermodeling, where the internal parameters of the members of a supermodel are tuned simultaneously with the model weights such that the supermodel predictions are optimized. For a case designed to challenge the two previous methods, adaptive supermodeling achieves a performance similar to a perfect model.</span> <span class="abstract-toggle" data-id="2510.06180">more</span>

    [:material-file-document: 2510.06180](https://arxiv.org/abs/2510.06180v1) · [:material-content-copy: BibTeX](bibtex/2510.06180.bib){ .bibtex-link }

-   #### EnScale: Temporally-consistent multivariate generative downscaling via proper scoring rules

    ---

    *Maybritt Schillinger, Maxim Samarin, Xinwei Shen, Reto Knutti, Nicolai Meinshausen* · 2025

    <span class="abstract-snippet" id="snip-2509.26258">The practical use of future climate projections from global circulation models (GCMs) is often limited by their coarse spatial resolution, requiring downscaling to generate high-resolution data....</span><span class="abstract-full" id="full-2509.26258" hidden>The practical use of future climate projections from global circulation models (GCMs) is often limited by their coarse spatial resolution, requiring downscaling to generate high-resolution data. Regional climate models (RCMs) provide this refinement, but are computationally expensive. To address this issue, machine learning models can learn the downscaling function, mapping coarse GCM outputs to high-resolution fields. Among these, generative approaches aim to capture the full conditional distribution of RCM data given coarse-scale GCM data, which is characterized by large variability and thus challenging to model accurately. We introduce EnScale, a generative machine learning framework that emulates the full GCM-to-RCM map by training on multiple pairs of GCM and corresponding RCM data. It first adjusts large-scale mismatches between GCM and coarsened RCM data, followed by a super-resolution step to generate high-resolution fields. Both steps employ generative models optimized with the energy score, a proper scoring rule. Compared to state-of-the-art ML downscaling approaches, our setup reduces computational cost by about one order of magnitude. EnScale jointly emulates multiple variables -- temperature, precipitation, solar radiation, and wind -- spatially consistent over an area in Central Europe. In addition, we propose a variant EnScale-t that enables temporally consistent downscaling. We establish a comprehensive evaluation framework across various categories including calibration, spatial structure, extremes, and multivariate dependencies. Comparison with diverse benchmarks demonstrates EnScale's strong performance and computational efficiency. EnScale offers a promising approach for accurate and temporally consistent RCM emulation.</span> <span class="abstract-toggle" data-id="2509.26258">more</span>

    [:material-file-document: 2509.26258](https://arxiv.org/abs/2509.26258v1) · [:material-content-copy: BibTeX](bibtex/2509.26258.bib){ .bibtex-link }

-   #### The Open-Source Photochem Code: A General Chemical and Climate Model for Interpreting (Exo)Planet Observations

    ---

    *Nicholas F. Wogan, Natasha E. Batalha, Kevin Zahnle, Joshua Krissansen-Totton, David C. Catling et al.* · 2025

    <span class="abstract-snippet" id="snip-2509.25578">With the launch of the James Webb Space Telescope, we are firmly in the era of exoplanet atmosphere characterization. Understanding exoplanet spectra requires atmospheric chemical and climate models...</span><span class="abstract-full" id="full-2509.25578" hidden>With the launch of the James Webb Space Telescope, we are firmly in the era of exoplanet atmosphere characterization. Understanding exoplanet spectra requires atmospheric chemical and climate models that span the diversity of planetary atmospheres. Here, we present a more general chemical and climate model of planetary atmospheres. Specifically, we introduce the open-source, one-dimensional photochemical and climate code Photochem, and benchmark the model against the observed compositions and climates of Venus, Earth, Mars, Jupiter and Titan with a single set of kinetics, thermodynamics and opacities. We also model the chemistry of the hot Jupiter exoplanet WASP-39b. All simulations are open-source and reproducible. To first order, Photochem broadly reproduces the gas-phase chemistry and pressure-temperature profiles of all six planets. The largest model-data discrepancies are found in Venus's sulfur chemistry, motivating future experimental work on sulfur kinetics and spacecraft missions to Venus. We also find that clouds and hazes are important for the energy balance of Venus, Earth, Mars and Titan, and that accurately predicting aerosols with Photochem is challenging. Finally, we benchmark Photochem against the popular VULCAN and HELIOS photochemistry and climate models, finding excellent agreement for the same inputs; we also find that Photochem simulates atmospheres 2 to 100 time more efficiently. These results show that Photochem provides a comparatively general description of atmospheric chemistry and physics that can be leveraged to study Solar System worlds or interpret telescope observations of exoplanets.</span> <span class="abstract-toggle" data-id="2509.25578">more</span>

    [:material-file-document: 2509.25578](https://arxiv.org/abs/2509.25578v1) · [:material-content-copy: BibTeX](bibtex/2509.25578.bib){ .bibtex-link }

-   #### Disrespect Others, Respect the Climate? Applying Social Dynamics with Inequality to Forest Climate Models

    ---

    *Luke Wisniewski, Thomas Zdyrski, Feng Fu* · 2025

    <span class="abstract-snippet" id="snip-2509.17252">Understanding the role of human behavior in shaping environmental outcomes is crucial for addressing global challenges such as climate change. Environmental systems are influenced not only by natural...</span><span class="abstract-full" id="full-2509.17252" hidden>Understanding the role of human behavior in shaping environmental outcomes is crucial for addressing global challenges such as climate change. Environmental systems are influenced not only by natural factors like temperature, but also by human decisions regarding mitigation efforts, which are often based on forecasts or predictions about future environmental conditions. Over time, different outcomes can emerge, including scenarios where the environment deteriorates despite efforts to mitigate, or where successful mitigation leads to environmental resilience. Additionally, fluctuations in the level of human participation in mitigation can occur, reflecting shifts in collective behavior. In this study, we consider a variety of human mitigation decisions, in addition to the feedback loop that is created by changes in human behavior because of environmental changes. While these outcomes are based on simplified models, they offer important insights into the dynamics of human decision-making and the factors that influence effective action in the context of environmental sustainability. This study aims to examine key social dynamics influencing society's response to a worsening climate. While others conclude that homophily prompts greater warming unconditionally, this model finds that homophily can prevent catastrophic effects given a poor initial environmental state. Assuming that poor countries have the resources to do so, a consensus in that class group to defect from the strategy of the rich group (who are generally incentivized to continue ``business as usual'') can frequently prevent the vegetation proportion from converging to 0.</span> <span class="abstract-toggle" data-id="2509.17252">more</span>

    [:material-file-document: 2509.17252](https://arxiv.org/abs/2509.17252v1) · [:material-content-copy: BibTeX](bibtex/2509.17252.bib){ .bibtex-link }

-   #### SamudrACE: Fast and Accurate Coupled Climate Modeling with 3D Ocean and Atmosphere Emulators

    ---

    *James P. C. Duncan, Elynn Wu, Surya Dheeshjith, Adam Subel, Troy Arcomano, Spencer K. Clark et al.* · 2025

    <span class="abstract-snippet" id="snip-2509.12490">Traditional numerical global climate models simulate the full Earth system by exchanging boundary conditions between separate simulators of the atmosphere, ocean, sea ice, land surface, and other...</span><span class="abstract-full" id="full-2509.12490" hidden>Traditional numerical global climate models simulate the full Earth system by exchanging boundary conditions between separate simulators of the atmosphere, ocean, sea ice, land surface, and other geophysical processes. This paradigm allows for distributed development of individual components within a common framework, unified by a coupler that handles translation between realms via spatial or temporal alignment and flux exchange. Following a similar approach adapted for machine learning-based emulators, we present SamudrACE: a coupled global climate model emulator which produces centuries-long simulations at 1-degree horizontal, 6-hourly atmospheric, and 5-daily oceanic resolution, with 145 2D fields spanning 8 atmospheric and 19 oceanic vertical levels, plus sea ice, surface, and top-of-atmosphere variables. SamudrACE is highly stable and has low climate biases comparable to those of its components with prescribed boundary forcing, with realistic variability in coupled climate phenomena such as ENSO that is not possible to simulate in uncoupled mode.</span> <span class="abstract-toggle" data-id="2509.12490">more</span>

    [:material-file-document: 2509.12490](https://arxiv.org/abs/2509.12490v1) · [:material-content-copy: BibTeX](bibtex/2509.12490.bib){ .bibtex-link }

-   #### Do machine learning climate models work in changing climate dynamics?

    ---

    *Maria Conchita Agana Navarro, Geng Li, Theo Wolf, María Pérez-Ortiz* · 2025

    <span class="abstract-snippet" id="snip-2509.12147">Climate change is accelerating the frequency and severity of unprecedented events, deviating from established patterns. Predicting these out-of-distribution (OOD) events is critical for assessing...</span><span class="abstract-full" id="full-2509.12147" hidden>Climate change is accelerating the frequency and severity of unprecedented events, deviating from established patterns. Predicting these out-of-distribution (OOD) events is critical for assessing risks and guiding climate adaptation. While machine learning (ML) models have shown promise in providing precise, high-speed climate predictions, their ability to generalize under distribution shifts remains a significant limitation that has been underexplored in climate contexts. This research systematically evaluates state-of-the-art ML-based climate models in diverse OOD scenarios by adapting established OOD evaluation methodologies to climate data. Experiments on large-scale datasets reveal notable performance variability across scenarios, shedding light on the strengths and limitations of current models. These findings underscore the importance of robust evaluation frameworks and provide actionable insights to guide the reliable application of ML for climate risk forecasting.</span> <span class="abstract-toggle" data-id="2509.12147">more</span>

    [:material-file-document: 2509.12147](https://arxiv.org/abs/2509.12147v1) · [:material-content-copy: BibTeX](bibtex/2509.12147.bib){ .bibtex-link }

-   #### Using machine learning to downscale coarse-resolution environmental variables for understanding the spatial frequency of convective storms

    ---

    *Hungjui Yu, Lander Ver Hoef, Kristen L. Rasmussen, Imme Ebert-Uphoff* · 2025

    <span class="abstract-snippet" id="snip-2509.08802">Global climate models (GCMs), typically run at ~100-km resolution, capture large-scale environmental conditions but cannot resolve convection and cloud processes at kilometer scales....</span><span class="abstract-full" id="full-2509.08802" hidden>Global climate models (GCMs), typically run at ~100-km resolution, capture large-scale environmental conditions but cannot resolve convection and cloud processes at kilometer scales. Convection-permitting models offer higher-resolution simulations that explicitly simulate convection but are computationally expensive and impractical for large ensemble runs. This study explores machine learning (ML) as a bridge between these approaches. We train simple, pixel-based neural networks to predict convective storm frequency from environmental variables produced by a regional convection-permitting model. The ML models achieve promising results, with structural similarity index measure (SSIM) values exceeding 0.8, capturing the diurnal cycle and orographic convection without explicit temporal or spatial coordinates as input. Model performance declines when fewer input features are used or specific regions are excluded, underscoring the role of diverse physical mechanisms in convective activity. These findings highlight ML potential as a computationally efficient tool for representing convection and as a means of scientific discovery, offering insights into convective processes. Unlike convolutional neural networks, which depend on spatial structure and grid size, the pixel-based model treats each grid point independently, enabling value-to-value prediction without spatial context. This design enhances adaptability to resolution changes and supports generalization to unseen environmental regimes, making it particularly suited for linking environmental conditions to convective features and for application across diverse model grids or climate scenarios.</span> <span class="abstract-toggle" data-id="2509.08802">more</span>

    [:material-file-document: 2509.08802](https://arxiv.org/abs/2509.08802v1) · [:material-content-copy: BibTeX](bibtex/2509.08802.bib){ .bibtex-link }

    <span class="md-tag">CNN</span>

</div>

## Extreme Weather (2)

<div class="grid cards" markdown>

-   #### Forecasting threshold exceedance of atmospheric variables at a specific location

    ---

    *Roberta Baggio, Jean-François Muzy* · 2026

    <span class="abstract-snippet" id="snip-2605.31079">This study compares two methodological approaches for predicting, at a given site, threshold exceedances of atmospheric variables such as temperature and wind speed: (i) direct probabilistic methods,...</span><span class="abstract-full" id="full-2605.31079" hidden>This study compares two methodological approaches for predicting, at a given site, threshold exceedances of atmospheric variables such as temperature and wind speed: (i) direct probabilistic methods, which treat exceedance as a binary classification problem, and (ii) full distribution probabilistic methods, which model the complete conditional probability law of the target variable. Using theoretical analysis and numerical simulations on a toy model, alongside real-world data from the MeteoNet dataset (2016--2018) for southeastern France, we demonstrate that the full distribution approach consistently outperforms the direct method for rare, extreme events. This advantage arises because the full distribution approach effectively learns the parameters of the conditional distribution from moderate and mild intensity events, thereby achieving better calibration and discrimination in the tails. We find that the specific parametric shape of the chosen distribution plays a secondary role compared to accurately capturing predictable shifts in its bulk properties (i.e., mean and variance). This empirical indistinguishability is also informative about the physical mechanics driving atmospheric extremes, suggesting that extreme exceedances are primarily driven by significant conditional displacements of the entire distribution rather than by unpredictable, fat-tailed anomalies within a static climatology. Our results are validated for both strong surface wind speeds and intense hourly rainfall, with performance evaluated using proper scoring rules (Brier score, logarithmic score) and deterministic skill scores (Peirce Skill Score, CSI, HSS). These findings highlight the critical importance of modeling the full probability distribution for rare-event forecasting and provide practical guidance for improving extreme weather prediction in operational meteorology.</span> <span class="abstract-toggle" data-id="2605.31079">more</span>

    [:material-file-document: 2605.31079](https://arxiv.org/abs/2605.31079v1) · [:material-content-copy: BibTeX](bibtex/2605.31079.bib){ .bibtex-link }

    <span class="md-tag">probabilistic</span>

-   #### FlowCast-ODE: Continuous Hourly Weather Forecasting with Dynamic Flow Matching and ODE Integration

    ---

    *Shuangshuang He, Yuanting Zhang, Hongli Liang, Qingye Meng, Xingyuan Yuan* · 2025

    <span class="abstract-snippet" id="snip-2509.14775">Accurate hourly weather forecasting is critical for numerous applications. Recent deep learning models have demonstrated strong capability on 6-hour intervals, yet achieving accurate and stable...</span><span class="abstract-full" id="full-2509.14775" hidden>Accurate hourly weather forecasting is critical for numerous applications. Recent deep learning models have demonstrated strong capability on 6-hour intervals, yet achieving accurate and stable hourly predictions remains a critical challenge. This is primarily due to the rapid accumulation of errors in autoregressive rollouts and temporal discontinuities within the ERA5 data's 12-hour assimilation cycle. To address these issues, we propose FlowCast-ODE, a framework that models atmospheric state evolution as a continuous flow. FlowCast-ODE learns the conditional flow path directly from the previous state, an approach that aligns more naturally with physical dynamic systems and enables efficient computation. A coarse-to-fine strategy is introduced to train the model on 6-hour data using dynamic flow matching and then refined on hourly data that incorporates an Ordinary Differential Equation (ODE) solver to achieve temporally coherent forecasts. In addition, a lightweight low-rank AdaLN-Zero modulation mechanism is proposed and reduces model size by 15% without compromising accuracy. Experiments demonstrate that FlowCast-ODE outperforms strong baselines, yielding lower root mean square error (RMSE) and better energy conservation, which reduces blurring and preserves more fine-scale spatial details. It also shows comparable performance to the state-of-the-art model in forecasting extreme events like typhoons. Furthermore, the model alleviates temporal discontinuities associated with assimilation cycle transitions.</span> <span class="abstract-toggle" data-id="2509.14775">more</span>

    [:material-file-document: 2509.14775](https://arxiv.org/abs/2509.14775v1) · [:material-content-copy: BibTeX](bibtex/2509.14775.bib){ .bibtex-link }

    <span class="md-tag">diffusion</span>

</div>

## Other (9)

<div class="grid cards" markdown>

-   #### Splitting horizontal and vertical polynomial order in a compatible finite element discretisation for numerical weather prediction

    ---

    *Daniel Witt, Thomas Bendall, Jemma Shipton* · 2026

    <span class="abstract-snippet" id="snip-2603.16571">The accurate and efficient representation of atmospheric dynamics remains a central challenge in numerical weather prediction. A particular difficulty arises from the strong anisotropy of the...</span><span class="abstract-full" id="full-2603.16571" hidden>The accurate and efficient representation of atmospheric dynamics remains a central challenge in numerical weather prediction. A particular difficulty arises from the strong anisotropy of the atmosphere, in which horizontal and vertical motions occur on very different length scales, motivating numerical discretisations that can reflect this structure. In this study, we introduce a compatible finite element discretisation of the compressible Boussinesq and compressible Euler equations in which the horizontal and vertical polynomial orders are treated independently.   The split-order discretisation is constructed using a tensor-product framework that preserves the discrete de Rham complex and associated mimetic properties. Its wave-propagation characteristics are examined through a discrete dispersion analysis that extends previous analyses to configurations with differing horizontal and vertical polynomial orders. The results show that increasing horizontal order improves the representation of gravity waves at low and intermediate wavenumbers, while increasing vertical order can degrade dispersion accuracy near the grid scale and introduce spectral gaps.   A series of idealised numerical experiments, including gravity-wave propagation, advective transport, mountain-wave flow, and a global baroclinic-wave test, is used to assess the scheme's accuracy and convergence properties. These experiments demonstrate that increasing the polynomial order in the dominant direction of motion improves convergence, and that increasing the horizontal order yields the greatest gain in accuracy under typical atmospheric conditions. The results indicate that split-order compatible finite element discretisations provide a viable alternative for controlling accuracy and numerical behaviour in atmospheric dynamical cores.</span> <span class="abstract-toggle" data-id="2603.16571">more</span>

    [:material-file-document: 2603.16571](https://arxiv.org/abs/2603.16571v1) · [:material-content-copy: BibTeX](bibtex/2603.16571.bib){ .bibtex-link }

-   #### Exploring Novel Data Storage Approaches for Large-Scale Numerical Weather Prediction

    ---

    *Nicolau Manubens Gil* · 2026

    <span class="abstract-snippet" id="snip-2602.17610">Driven by scientific and industry ambition, HPC and AI applications such as operational Numerical Weather Prediction (NWP) require processing and storing ever-increasing data volumes as fast as...</span><span class="abstract-full" id="full-2602.17610" hidden>Driven by scientific and industry ambition, HPC and AI applications such as operational Numerical Weather Prediction (NWP) require processing and storing ever-increasing data volumes as fast as possible. Whilst POSIX distributed file systems and NVMe SSDs are currently a common HPC storage configuration providing I/O to applications, new storage solutions have proliferated or gained traction over the last decade with potential to address performance limitations POSIX file systems manifest at scale for certain I/O workloads.   This work has primarily aimed to assess the suitability and performance of two object storage systems -namely DAOS and Ceph- for the ECMWF's operational NWP as well as for HPC and AI applications in general. New software-level adapters have been developed which enable the ECMWF's NWP to leverage these systems, and extensive I/O benchmarking has been conducted on a few computer systems, comparing the performance delivered by the evaluated object stores to that of equivalent Lustre file system deployments on the same hardware. Challenges of porting to object storage and its benefits with respect to the traditional POSIX I/O approach have been discussed and, where possible, domain-agnostic performance analysis has been conducted, leading to insight also of relevance to I/O practitioners and the broader HPC community.   DAOS and Ceph have both demonstrated excellent performance, but DAOS stood out relative to Ceph and Lustre, providing superior scalability and flexibility for applications to perform I/O at scale as desired. This sets a promising outlook for DAOS and object storage, which might see greater adoption at HPC centres in the years to come, although not necessarily implying a shift away from POSIX-like I/O.</span> <span class="abstract-toggle" data-id="2602.17610">more</span>

    [:material-file-document: 2602.17610](https://arxiv.org/abs/2602.17610v1) · [:material-content-copy: BibTeX](bibtex/2602.17610.bib){ .bibtex-link }

-   #### EMFormer: Efficient Multi-Scale Transformer for Accumulative Context Weather Forecasting

    ---

    *Hao Chen, Tao Han, Jie Zhang, Song Guo, Fenghua Ling, Lei Bai* · 2026

    <span class="abstract-snippet" id="snip-2602.01194">Long-term weather forecasting is critical for socioeconomic planning and disaster preparedness. While recent approaches employ finetuning to extend prediction horizons, they remain constrained by the...</span><span class="abstract-full" id="full-2602.01194" hidden>Long-term weather forecasting is critical for socioeconomic planning and disaster preparedness. While recent approaches employ finetuning to extend prediction horizons, they remain constrained by the issues of catastrophic forgetting, error accumulation, and high training overhead. To address these limitations, we present a novel pipeline across pretraining, finetuning and forecasting to enhance long-context modeling while reducing computational overhead. First, we introduce an Efficient Multi-scale Transformer (EMFormer) to extract multi-scale features through a single convolution in both training and inference. Based on the new architecture, we further employ an accumulative context finetuning to improve temporal consistency without degrading short-term accuracy. Additionally, we propose a composite loss that dynamically balances different terms via a sinusoidal weighting, thereby adaptively guiding the optimization trajectory throughout pretraining and finetuning. Experiments show that our approach achieves strong performance in weather forecasting and extreme event prediction, substantially improving long-term forecast accuracy. Moreover, EMFormer demonstrates strong generalization on vision benchmarks (ImageNet-1K and ADE20K) while delivering a 5.69x speedup over conventional multi-scale modules.</span> <span class="abstract-toggle" data-id="2602.01194">more</span>

    [:material-file-document: 2602.01194](https://arxiv.org/abs/2602.01194v1) · [:material-content-copy: BibTeX](bibtex/2602.01194.bib){ .bibtex-link }

    <span class="md-tag">transformer</span>

-   #### Hybrid SARIMA LSTM Model for Local Weather Forecasting: A Residual Learning Approach for Data Driven Meteorological Prediction

    ---

    *Shreyas Rajeev, Karthik Mudenahalli Ashoka, Amit Mallappa Tiparaddi* · 2026

    <span class="abstract-snippet" id="snip-2601.07951">Accurately forecasting long-term atmospheric variables remains a defining challenge in meteorological science due to the chaotic nature of atmospheric systems. Temperature data represents a complex...</span><span class="abstract-full" id="full-2601.07951" hidden>Accurately forecasting long-term atmospheric variables remains a defining challenge in meteorological science due to the chaotic nature of atmospheric systems. Temperature data represents a complex superposition of deterministic cyclical climate forces and stochastic, short-term fluctuations. While planetary mechanics drive predictable seasonal periodicities, rapid meteorological changes such as thermal variations, pressure anomalies, and humidity shifts introduce nonlinear volatilities that defy simple extrapolation. Historically, the Seasonal Autoregressive Integrated Moving Average (SARIMA) model has been the standard for modeling historical weather data, prized for capturing linear seasonal trends. However, SARIMA operates under strict assumptions of stationarity, failing to capture abrupt, nonlinear transitions. This leads to systematic residual errors, manifesting as the under-prediction of sudden spikes or the over-smoothing of declines. Conversely, Deep Learning paradigms, specifically Long Short-Term Memory (LSTM) networks, demonstrate exceptional efficacy in handling intricate time-series data. By utilizing memory gates, LSTMs learn complex nonlinear dependencies. Yet, LSTMs face instability in open-loop forecasting; without ground truth feedback, minor deviations compound recursively, causing divergence. To resolve these limitations, we propose a Hybrid SARIMA-LSTM architecture. This framework employs a residual-learning strategy to decompose temperature into a predictable climate component and a nonlinear weather component. The SARIMA unit models the robust, long-term seasonal trend, while the LSTM is trained exclusively on the residuals the nonlinear errors SARIMA fails to capture. By fusing statistical stability with neural plasticity, this hybrid approach minimizes error propagation and enhances long-horizon accuracy.</span> <span class="abstract-toggle" data-id="2601.07951">more</span>

    [:material-file-document: 2601.07951](https://arxiv.org/abs/2601.07951v1) · [:material-content-copy: BibTeX](bibtex/2601.07951.bib){ .bibtex-link }

    <span class="md-tag">recurrent</span>

-   #### The promising potential of vision language models for the generation of textual weather forecasts

    ---

    *Edward C. C. Steele, Dinesh Mane, Emilio Monti, Luis Orus, Rebecca Chantrill-Cheyette et al.* · 2025

    <span class="abstract-snippet" id="snip-2512.03623">Despite the promising capability of multimodal foundation models, their application to the generation of meteorological products and services remains nascent. To accelerate aspiration and adoption,...</span><span class="abstract-full" id="full-2512.03623" hidden>Despite the promising capability of multimodal foundation models, their application to the generation of meteorological products and services remains nascent. To accelerate aspiration and adoption, we explore the novel use of a vision language model for writing the iconic Shipping Forecast text directly from video-encoded gridded weather data. These early results demonstrate promising scalable technological opportunities for enhancing production efficiency and service innovation within the weather enterprise and beyond.</span> <span class="abstract-toggle" data-id="2512.03623">more</span>

    [:material-file-document: 2512.03623](https://arxiv.org/abs/2512.03623v1) · [:material-content-copy: BibTeX](bibtex/2512.03623.bib){ .bibtex-link }

    <span class="md-tag">foundation-model</span>

-   #### Causal Feature Selection for Weather-Driven Residential Load Forecasting

    ---

    *Elise Zhang, François Mirallès, Stéphane Dellacherie, Di Wu, Benoit Boulet* · 2025

    <span class="abstract-snippet" id="snip-2511.20508">Weather is a dominant external driver of residential electricity demand, but adding many meteorological covariates can inflate model complexity and may even impair accuracy. Selecting appropriate...</span><span class="abstract-full" id="full-2511.20508" hidden>Weather is a dominant external driver of residential electricity demand, but adding many meteorological covariates can inflate model complexity and may even impair accuracy. Selecting appropriate exogenous features is non-trivial and calls for a principled selection framework, given the direct operational implications for day-to-day planning and reliability. This work investigates whether causal feature selection can retain the most informative weather drivers while improving parsimony and robustness for short-term load forecasting. We present a case study on Southern Ontario with two open-source datasets: (i) IESO hourly electricity consumption by Forward Sortation Areas; (ii) ERA5 weather reanalysis data. We compare different feature selection regimes (no feature selection, non-causal selection, PCMCI-causal selection) on city-level forecasting with three different time series forecasting models: GRU, TCN, PatchTST. In the feature analysis, non-causal selection prioritizes radiation and moisture variables that show correlational dependence, whereas PCMCI-causal selection emphasizes more direct thermal drivers and prunes the indirect covariates. We detail the evaluation pipeline and report diagnostics on prediction accuracy and extreme-weather robustness, positioning causal feature selection as a practical complement to modern forecasters when integrating weather into residential load forecasting.</span> <span class="abstract-toggle" data-id="2511.20508">more</span>

    [:material-file-document: 2511.20508](https://arxiv.org/abs/2511.20508v1) · [:material-content-copy: BibTeX](bibtex/2511.20508.bib){ .bibtex-link }

-   #### Weather Maps as Tokens: Transformers for Renewable Energy Forecasting

    ---

    *Federico Battini* · 2025

    <span class="abstract-snippet" id="snip-2511.13935">Accurate renewable energy forecasting is essential to reduce dependence on fossil fuels and enabling grid decarbonization. However, current approaches fail to effectively integrate the rich spatial...</span><span class="abstract-full" id="full-2511.13935" hidden>Accurate renewable energy forecasting is essential to reduce dependence on fossil fuels and enabling grid decarbonization. However, current approaches fail to effectively integrate the rich spatial context of weather patterns with their temporal evolution. This work introduces a novel approach that treats weather maps as tokens in transformer sequences to predict renewable energy. Hourly weather maps are encoded as spatial tokens using a lightweight convolutional neural network, and then processed by a transformer to capture temporal dynamics across a 45-hour forecast horizon. Despite disadvantages in input initialization, evaluation against ENTSO-E operational forecasts shows a reduction in RMSE of about 60% and 20% for wind and solar respectively. A live dashboard showing daily forecasts is available at: https://www.sardiniaforecast.ifabfoundation.it.</span> <span class="abstract-toggle" data-id="2511.13935">more</span>

    [:material-file-document: 2511.13935](https://arxiv.org/abs/2511.13935v2) · [:material-content-copy: BibTeX](bibtex/2511.13935.bib){ .bibtex-link }

    <span class="md-tag">transformer</span> <span class="md-tag">CNN</span>

-   #### Road Surface Condition Detection with Machine Learning using New York State Department of Transportation Camera Images and Weather Forecast Data

    ---

    *Carly Sutter, Kara J. Sulia, Nick P. Bassill, Christopher D. Wirz, Christopher D. Thorncroft et al.* · 2025

    <span class="abstract-snippet" id="snip-2510.06440">The New York State Department of Transportation (NYSDOT) has a network of roadside traffic cameras that are used by both the NYSDOT and the public to observe road conditions. The NYSDOT evaluates...</span><span class="abstract-full" id="full-2510.06440" hidden>The New York State Department of Transportation (NYSDOT) has a network of roadside traffic cameras that are used by both the NYSDOT and the public to observe road conditions. The NYSDOT evaluates road conditions by driving on roads and observing live cameras, tasks which are labor-intensive but necessary for making critical operational decisions during winter weather events. However, machine learning models can provide additional support for the NYSDOT by automatically classifying current road conditions across the state. In this study, convolutional neural networks and random forests are trained on camera images and weather data to predict road surface conditions. Models are trained on a hand-labeled dataset of ~22,000 camera images, each classified by human labelers into one of six road surface conditions: severe snow, snow, wet, dry, poor visibility, or obstructed. Model generalizability is prioritized to meet the operational needs of the NYSDOT decision makers, and the weather-related road surface condition model in this study achieves an accuracy of 81.5% on completely unseen cameras.</span> <span class="abstract-toggle" data-id="2510.06440">more</span>

    [:material-file-document: 2510.06440](https://arxiv.org/abs/2510.06440v1) · [:material-content-copy: BibTeX](bibtex/2510.06440.bib){ .bibtex-link }

    <span class="md-tag">CNN</span>

-   #### Graph-based Neural Space Weather Forecasting

    ---

    *Daniel Holmberg, Ivan Zaitsev, Markku Alho, Ioanna Bouri, Fanni Franssila, Haewon Jeong et al.* · 2025

    <span class="abstract-snippet" id="snip-2509.19605">Accurate space weather forecasting is crucial for protecting our increasingly digital infrastructure. Hybrid-Vlasov models, like Vlasiator, offer physical realism beyond that of current operational...</span><span class="abstract-full" id="full-2509.19605" hidden>Accurate space weather forecasting is crucial for protecting our increasingly digital infrastructure. Hybrid-Vlasov models, like Vlasiator, offer physical realism beyond that of current operational systems, but are too computationally expensive for real-time use. We introduce a graph-based neural emulator trained on Vlasiator data to autoregressively predict near-Earth space conditions driven by an upstream solar wind. We show how to achieve both fast deterministic forecasts and, by using a generative model, produce ensembles to capture forecast uncertainty. This work demonstrates that machine learning offers a way to add uncertainty quantification capability to existing space weather prediction systems, and make hybrid-Vlasov simulation tractable for operational use.</span> <span class="abstract-toggle" data-id="2509.19605">more</span>

    [:material-file-document: 2509.19605](https://arxiv.org/abs/2509.19605v1) · [:material-content-copy: BibTeX](bibtex/2509.19605.bib){ .bibtex-link }

    <span class="md-tag">probabilistic</span>

</div>

