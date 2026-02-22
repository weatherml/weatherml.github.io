# New

### Making Tunable Parameters State-Dependent in Weather and Climate Models with Reinforcement Learning

**Authors:** Pritthijit Nath, Sebastian Schemm, Henry Moss, Peter Haynes, Emily Shuckburgh, Mark J. Webb

**Year:** 2026

**Abstract:**
> Weather and climate models rely on parametrisations to represent unresolved sub-grid processes. Traditional schemes rely on fixed coefficients that are weakly constrained and tuned offline, contributing to persistent biases that limit their ability to adapt to the underlying physics. This study presents a framework that learns components of parametrisation schemes online as a function of the evolving model state using reinforcement learning (RL) and evaluates the resulting RL-driven parameter updates across a hierarchy of idealised testbeds spanning a simple climate bias correction (SCBC), a radiative-convective equilibrium (RCE), and a zonal mean energy balance model (EBM) with both single-agent and federated multi-agent settings. Across nine RL algorithms, Truncated Quantile Critics (TQC), Deep Deterministic Policy Gradient (DDPG), and Twin Delayed DDPG (TD3) achieved the highest skill and the most stable convergence across configurations, with performance assessed against a static baseline using area-weighted RMSE, temperature profile and pressure-level diagnostics. For the EBM, single-agent RL outperformed static parameter tuning with the strongest gains in tropical and mid-latitude bands, while federated RL on multi-agent setups enabled geographically specialised control and faster convergence, with a six-agent DDPG configuration using frequent aggregation yielding the lowest area-weighted RMSE across the tropics and mid-latitudes. The learnt corrections were also physically meaningful as agents modulated EBM radiative parameters to reduce meridional biases, adjusted RCE lapse rates to match vertical temperature errors, and stabilised SCBC heating increments to limit drift. Overall, results highlight RL to deliver skilful state-dependent, and regime-aware parametrisations, offering a scalable pathway for online learning within numerical models.

[**arXiv:2601.04268v1**](https://arxiv.org/abs/2601.04268v1)

**Tags:** ``

---

### Searth Transformer: A Transformer Architecture Incorporating Earth's Geospheric Physical Priors for Global Mid-Range Weather Forecasting

**Authors:** Tianye Li, Qi Liu, Hao Li, Lei Chen, Wencong Cheng, Fei Zheng, Xiangao Xia, Ya Wang, Gang Huang, Weiwei Wang, Xuan Tong, Ziqing Zu, Yi Fang, Shenming Fu, Jiang Jiang, Haochen Li, Mingxing Li, Jiangjiang Xia

**Year:** 2026

**Abstract:**
> Accurate global medium-range weather forecasting is fundamental to Earth system science. Most existing Transformer-based forecasting models adopt vision-centric architectures that neglect the Earth's spherical geometry and zonal periodicity. In addition, conventional autoregressive training is computationally expensive and limits forecast horizons due to error accumulation. To address these challenges, we propose the Shifted Earth Transformer (Searth Transformer), a physics-informed architecture that incorporates zonal periodicity and meridional boundaries into window-based self-attention for physically consistent global information exchange. We further introduce a Relay Autoregressive (RAR) fine-tuning strategy that enables learning long-range atmospheric evolution under constrained memory and computational budgets. Based on these methods, we develop YanTian, a global medium-range weather forecasting model. YanTian achieves higher accuracy than the high-resolution forecast of the European Centre for Medium-Range Weather Forecasts and performs competitively with state-of-the-art AI models at one-degree resolution, while requiring roughly 200 times lower computational cost than standard autoregressive fine-tuning. Furthermore, YanTian attains a longer skillful forecast lead time for Z500 (10.3 days) than HRES (9 days). Beyond weather forecasting, this work establishes a robust algorithmic foundation for predictive modeling of complex global-scale geophysical circulation systems, offering new pathways for Earth system science.

[**arXiv:2601.09467v1**](https://arxiv.org/abs/2601.09467v1)

**Tags:** ``

---

### Detail Loss in Super-Resolution Models Based on the Laplacian Pyramid and Repeated Upscaling and Downscaling Process

**Authors:** Sangjun Han, Youngmi Hur

**Year:** 2026

**Abstract:**
> With advances in artificial intelligence, image processing has gained significant interest. Image super-resolution is a vital technology closely related to real-world applications, as it enhances the quality of existing images. Since enhancing fine details is crucial for the super-resolution task, pixels that contribute to high-frequency information should be emphasized. This paper proposes two methods to enhance high-frequency details in super-resolution images: a Laplacian pyramid-based detail loss and a repeated upscaling and downscaling process. Total loss with our detail loss guides a model by separately generating and controlling super-resolution and detail images. This approach allows the model to focus more effectively on high-frequency components, resulting in improved super-resolution images. Additionally, repeated upscaling and downscaling amplify the effectiveness of the detail loss by extracting diverse information from multiple low-resolution features. We conduct two types of experiments. First, we design a CNN-based model incorporating our methods. This model achieves state-of-the-art results, surpassing all currently available CNN-based and even some attention-based models. Second, we apply our methods to existing attention-based models on a small scale. In all our experiments, attention-based models adding our detail loss show improvements compared to the originals. These results demonstrate our approaches effectively enhance super-resolution images across different model structures.

[**arXiv:2601.09410v1**](https://arxiv.org/abs/2601.09410v1)

**Tags:** ``

---

### Efficient Parameter Calibration of Numerical Weather Prediction Models via Evolutionary Sequential Transfer Optimization

**Authors:** Heping Fang, Peng Yang

**Year:** 2026

**Abstract:**
> The configuration of physical parameterization schemes in Numerical Weather Prediction (NWP) models plays a critical role in determining the accuracy of the forecast. However, existing parameter calibration methods typically treat each calibration task as an isolated optimization problem. This approach suffers from prohibitive computational costs and necessitates performing iterative searches from scratch for each task, leading to low efficiency in sequential calibration scenarios. To address this issue, we propose the SEquential Evolutionary Transfer Optimization (SEETO) algorithm driven by the representations of the meteorological state. First, to accurately measure the physical similarity between calibration tasks, a meteorological state representation extractor is introduced to map high-dimensional meteorological fields into latent representations. Second, given the similarity in the latent space, a bi-level adaptive knowledge transfer mechanism is designed. At the solution level, superior populations from similar historical tasks are reused to achieve a "warm start" for optimization. At the model level, an ensemble surrogate model based on source task data is constructed to assist the search, employing an adaptive weighting mechanism to dynamically balance the contributions of source domain knowledge and target domain data. Extensive experiments across 10 distinct calibration tasks, which span varying source-target similarities, highlight SEETO's superior efficiency. Under a strict budget of 20 expensive evaluations, SEETO achieves a 6% average improvement in Hypervolume (HV) over two state-of-the-art baselines. Notably, to match SEETO's performance at this stage, the comparison algorithms would require an average of 64% and 28% additional evaluations, respectively. This presents a new paradigm for the efficient and accurate automated calibration of NWP model parameters.

[**arXiv:2601.08663v1**](https://arxiv.org/abs/2601.08663v1)

**Tags:** ``

---

### Hybrid SARIMA LSTM Model for Local Weather Forecasting: A Residual Learning Approach for Data Driven Meteorological Prediction

**Authors:** Shreyas Rajeev, Karthik Mudenahalli Ashoka, Amit Mallappa Tiparaddi

**Year:** 2026

**Abstract:**
> Accurately forecasting long-term atmospheric variables remains a defining challenge in meteorological science due to the chaotic nature of atmospheric systems. Temperature data represents a complex superposition of deterministic cyclical climate forces and stochastic, short-term fluctuations. While planetary mechanics drive predictable seasonal periodicities, rapid meteorological changes such as thermal variations, pressure anomalies, and humidity shifts introduce nonlinear volatilities that defy simple extrapolation. Historically, the Seasonal Autoregressive Integrated Moving Average (SARIMA) model has been the standard for modeling historical weather data, prized for capturing linear seasonal trends. However, SARIMA operates under strict assumptions of stationarity, failing to capture abrupt, nonlinear transitions. This leads to systematic residual errors, manifesting as the under-prediction of sudden spikes or the over-smoothing of declines. Conversely, Deep Learning paradigms, specifically Long Short-Term Memory (LSTM) networks, demonstrate exceptional efficacy in handling intricate time-series data. By utilizing memory gates, LSTMs learn complex nonlinear dependencies. Yet, LSTMs face instability in open-loop forecasting; without ground truth feedback, minor deviations compound recursively, causing divergence. To resolve these limitations, we propose a Hybrid SARIMA-LSTM architecture. This framework employs a residual-learning strategy to decompose temperature into a predictable climate component and a nonlinear weather component. The SARIMA unit models the robust, long-term seasonal trend, while the LSTM is trained exclusively on the residuals the nonlinear errors SARIMA fails to capture. By fusing statistical stability with neural plasticity, this hybrid approach minimizes error propagation and enhances long-horizon accuracy.

[**arXiv:2601.07951v1**](https://arxiv.org/abs/2601.07951v1)

**Tags:** ``

---

### A Relaxed Direct-insertion Downscaling Method For Discrete-in-time Data Assimilation

**Authors:** Emine Celik, Eric Olson

**Year:** 2026

**Abstract:**
> This paper improves the spectrally-filtered direct-insertion downscaling method for discrete-in-time data assimilation by introducing a relaxation parameter that overcomes a constraint on the observation frequency. Numerical simulations demonstrate that taking the relaxation parameter proportional to the time between observations allows one to vary the observation frequency over a wide range while maintaining convergence of the approximating solution to the reference solution. Under the same assumptions we analytically prove that taking the observation frequency to infinity results in the continuous-in-time nudging method.

[**arXiv:2601.07025v1**](https://arxiv.org/abs/2601.07025v1)

**Tags:** ``

---

### Zero-Shot Statistical Downscaling via Diffusion Posterior Sampling

**Authors:** Ruian Tie, Wenbo Xiong, Zhengyu Shi, Xinyu Su, Chenyu jiang, Libo Wu, Hao Li

**Year:** 2026

**Abstract:**
> Conventional supervised climate downscaling struggles to generalize to Global Climate Models (GCMs) due to the lack of paired training data and inherent domain gaps relative to reanalysis. Meanwhile, current zero-shot methods suffer from physical inconsistencies and vanishing gradient issues under large scaling factors. We propose Zero-Shot Statistical Downscaling (ZSSD), a zero-shot framework that performs statistical downscaling without paired data during training. ZSSD leverages a Physics-Consistent Climate Prior learned from reanalysis data, conditioned on geophysical boundaries and temporal information to enforce physical validity. Furthermore, to enable robust inference across varying GCMs, we introduce Unified Coordinate Guidance. This strategy addresses the vanishing gradient problem in vanilla DPS and ensures consistency with large-scale fields. Results show that ZSSD significantly outperforms existing zero-shot baselines in 99th percentile errors and successfully reconstructs complex weather events, such as tropical cyclones, across heterogeneous GCMs.

[**arXiv:2601.21760v1**](https://arxiv.org/abs/2601.21760v1)

**Tags:** ``

---

### SENDAI: A Hierarchical Sparse-measurement, EfficieNt Data AssImilation Framework

**Authors:** Xingyue Zhang, Yuxuan Bao, Mars Liyao Gao, J. Nathan Kutz

**Year:** 2026

**Abstract:**
> Bridging the gap between data-rich training regimes and observation-sparse deployment conditions remains a central challenge in spatiotemporal field reconstruction, particularly when target domains exhibit distributional shifts, heterogeneous structure, and multi-scale dynamics absent from available training data. We present SENDAI, a hierarchical Sparse-measurement, EfficieNt Data AssImilation Framework that reconstructs full spatial states from hyper sparse sensor observations by combining simulation-derived priors with learned discrepancy corrections. We demonstrate the performance on satellite remote sensing, reconstructing MODIS (Moderate Resolution Imaging Spectroradiometer) derived vegetation index fields across six globally distributed sites. Using seasonal periods as a proxy for domain shift, the framework consistently outperforms established baselines that require substantially denser observations -- SENDAI achieves a maximum SSIM improvement of 185% over traditional baselines and a 36% improvement over recent high-frequency-based methods. These gains are particularly pronounced for landscapes with sharp boundaries and sub-seasonal dynamics; more importantly, the framework effectively preserves diagnostically relevant structures -- such as field topologies, land cover discontinuities, and spatial gradients. By yielding corrections that are more structurally and spectrally separable, the reconstructed fields are better suited for downstream inference of indirectly observed variables. The results therefore highlight a lightweight and operationally viable framework for sparse-measurement reconstruction that is applicable to physically grounded inference, resource-limited deployment, and real-time monitor and control.

[**arXiv:2601.21664v1**](https://arxiv.org/abs/2601.21664v1)

**Tags:** ``

---

### Learning to Advect: A Neural Semi-Lagrangian Architecture for Weather Forecasting

**Authors:** Carlos A. Pereira, Stéphane Gaudreault, Valentin Dallerit, Christopher Subich, Shoyon Panday, Siqi Wei, Sasa Zhang, Siddharth Rout, Eldad Haber, Raymond J. Spiteri, David Millard, Emilia Diaconescu

**Year:** 2026

**Abstract:**
> Recent machine-learning approaches to weather forecasting often employ a monolithic architecture, where distinct physical mechanisms (advection, transport), diffusion-like mixing, thermodynamic processes, and forcing are represented implicitly within a single large network. This representation is particularly problematic for advection, where long-range transport must be treated with expensive global interaction mechanisms or through deep, stacked convolutional layers. To mitigate this, we present PARADIS, a physics-inspired global weather prediction model that imposes inductive biases on network behavior through a functional decomposition into advection, diffusion, and reaction blocks acting on latent variables. We implement advection through a Neural Semi-Lagrangian operator that performs trajectory-based transport via differentiable interpolation on the sphere, enabling end-to-end learning of both the latent modes to be transported and their characteristic trajectories. Diffusion-like processes are modeled through depthwise-separable spatial mixing, while local source terms and vertical interactions are modeled via pointwise channel interactions, enabling operator-level physical structure. PARADIS provides state-of-the-art forecast skill at a fraction of the training cost. On ERA5-based benchmarks, the 1 degree PARADIS model, with a total training cost of less than a GPU month, meets or exceeds the performance of 0.25 degree traditional and machine-learning baselines, including the ECMWF HRES forecast and DeepMind's GraphCast.

[**arXiv:2601.21151v1**](https://arxiv.org/abs/2601.21151v1)

**Tags:** ``

---

### StormDiT: A generative AI model bridges the 2-6 hour 'gray zone' in precipitation nowcasting

**Authors:** Haofei Sun, Yunfan Yang, Wei Han, Wei Huang, Huaguan Chen, Zhiqiu Gao, Zeting Li, Zhaoyang Huo, Zeyi Niu

**Year:** 2026

**Abstract:**
> Accurate short-term warnings for extreme precipitation are critical for global disaster mitigation but are hindered by a persistent predictability barrier at the 2-6 hour horizon -- the "nowcasting gray zone." In this window, traditional observation-based extrapolation fails due to error accumulation, while numerical weather prediction is computationally too slow to resolve storm-scale dynamics. Recent generative AI approaches attempt to bridge this gap by decomposing precipitation into separate deterministic advection and stochastic diffusion components. However, this decomposition can sever fundamental causal links between entangled atmospheric processes, such as the dynamic initiation of convection triggered by boundary advection. Here we present StormDiT, a unified generative model that treats weather evolution as a holistic spatiotemporal problem, learning the coupled physics of the gray zone without human-imposed structural priors. Trained on a massive dataset of 7,720 precipitation events from China, our model achieves a breakthrough in long-horizon stability. On a heavy-rainfall test set, it maintains skillful prediction for strong convection ($\ge$ 35 dBZ) with a Critical Success Index (CSI) near 0.2 across the full 6-hour forecast at 6-minute resolution. Crucially, the model exhibits superior probabilistic calibration, accurately quantifying operational risks. On the public SEVIR benchmark, our unified paradigm more than doubles the state-of-the-art 1-hour performance for heavy rain and establishes the first robust baseline for 3-hour forecasting. Furthermore, interpretability analysis reveals that the model attends to non-local physical precursors, such as outflow boundaries, explicitly validating its emergent understanding of convective organization.

[**arXiv:2601.20342v1**](https://arxiv.org/abs/2601.20342v1)

**Tags:** ``

---

### Cheap2Rich: A Multi-Fidelity Framework for Data Assimilation and System Identification of Multiscale Physics -- Rotating Detonation Engines

**Authors:** Yuxuan Bao, Jan Zajac, Megan Powers, Venkat Raman, J. Nathan Kutz

**Year:** 2026

**Abstract:**
> Bridging the sim2real gap between computationally inexpensive models and complex physical systems remains a central challenge in machine learning applications to engineering problems, particularly in multi-scale settings where reduced-order models typically capture only dominant dynamics. In this work, we present Cheap2Rich, a multi-scale data assimilation framework that reconstructs high-fidelity state spaces from sparse sensor histories by combining a fast low-fidelity prior with learned, interpretable discrepancy corrections. We demonstrate the performance on rotating detonation engines (RDEs), a challenging class of systems that couple detonation-front propagation with injector-driven unsteadiness, mixing, and stiff chemistry across disparate scales. Our approach successfully reconstructs high-fidelity RDE states from sparse measurements while isolating physically meaningful discrepancy dynamics associated with injector-driven effects. The results highlight a general multi-fidelity framework for data assimilation and system identification in complex multi-scale systems, enabling rapid design exploration and real-time monitoring and control while providing interpretable discrepancy dynamics. Code for this project is is available at: github.com/kro0l1k/Cheap2Rich.

[**arXiv:2601.20295v1**](https://arxiv.org/abs/2601.20295v1)

**Tags:** ``

---

### Evolving beyond collapse: An adaptive particle batch smoother for cryospheric data assimilation

**Authors:** Kristoffer Aalstad, Esteban Alonso-González, Norbert Pirk, Sebastian Westermann, Clarissa Willmes, Ruitang Yang

**Year:** 2026

**Abstract:**
> We present a new adaptive particle-based data assimilation scheme for cryospheric applications that leverages promising developments in importance sampling. The proposed approach seeks to combine some of the advantages of two widely used classes of schemes: particle methods and iterative ensemble Kalman methods. Specifically, it extends the PBS that is commonly used in cryospheric data assimilation, with the AMIS algorithm. This adaptive formulation transforms the PBS into an iterative scheme with improved resilience against ensemble collapse and the ability to implement early-stopping strategies. As such, computational cost is automatically adapted to the complexity of the problem at hand, even down to the grid-cell and water year level in distributed multiyear simulations. In homage to the schemes that it builds on, we coin this new algorithm the Adaptive Particle Batch Smoother (AdaPBS) and we test it across a range of scenarios. First, we conducted an intercomparison of some of the most commonly used cryospheric data assimilation algorithms using MCMC simulation as a costly gold-standard benchmark in a simplified temperature index model assimilating snow depth observations. We further evaluated AdaPBS by assimilating snow depth observations from the ESMSnowMIP project at 6 different sites spanning 3 continents, using an ensemble of simulations generated with the more complex FSM2. Our results demonstrate that AdaPBS is a robust and reliable tool, outperforming or at least matching the performance of other commonly used algorithms and successfully handling complex cases with dense observational datasets. All experiments were carried out using the open-source MuSA toolbox, which now includes AdaPBS and MCMC among the growing list of available cryospheric data assimilation methods.

[**arXiv:2601.20049v1**](https://arxiv.org/abs/2601.20049v1)

**Tags:** ``

---

### Demystifying Data-Driven Probabilistic Medium-Range Weather Forecasting

**Authors:** Jean Kossaifi, Nikola Kovachki, Morteza Mardani, Daniel Leibovici, Suman Ravuri, Ira Shokar, Edoardo Calvello, Mohammad Shoaib Abbas, Peter Harrington, Ashay Subramaniam, Noah Brenowitz, Boris Bonev, Wonmin Byeon, Karsten Kreis, Dale Durran, Arash Vahdat, Mike Pritchard, Jan Kautz

**Year:** 2026

**Abstract:**
> The recent revolution in data-driven methods for weather forecasting has lead to a fragmented landscape of complex, bespoke architectures and training strategies, obscuring the fundamental drivers of forecast accuracy. Here, we demonstrate that state-of-the-art probabilistic skill requires neither intricate architectural constraints nor specialized training heuristics. We introduce a scalable framework for learning multi-scale atmospheric dynamics by combining a directly downsampled latent space with a history-conditioned local projector that resolves high-resolution physics. We find that our framework design is robust to the choice of probabilistic estimator, seamlessly supporting stochastic interpolants, diffusion models, and CRPS-based ensemble training. Validated against the Integrated Forecasting System and the deep learning probabilistic model GenCast, our framework achieves statistically significant improvements on most of the variables. These results suggest scaling a general-purpose model is sufficient for state-of-the-art medium-range prediction, eliminating the need for tailored training recipes and proving effective across the full spectrum of probabilistic frameworks.

[**arXiv:2601.18111v1**](https://arxiv.org/abs/2601.18111v1)

**Tags:** ``

---

### Decision-oriented benchmarking to transform AI weather forecast access: Application to the Indian monsoon

**Authors:** Rajat Masiwal, Colin Aitken, Adam Marchakitus, Mayank Gupta, Katherine Kowal, Hamid A. Pahlavan, Tyler Yang, Y. Qiang Sun, Michael Kremer, Amir Jina, William R. Boos, Pedram Hassanzadeh

**Year:** 2026

**Abstract:**
> Artificial intelligence weather prediction (AIWP) models now often outperform traditional physics-based models on common metrics while requiring orders-of-magnitude less computing resources and time. Open-access AIWP models thus hold promise as transformational tools for helping low- and middle-income populations make decisions in the face of high-impact weather shocks. Yet, current approaches to evaluating AIWP models focus mainly on aggregated meteorological metrics without considering local stakeholders' needs in decision-oriented, operational frameworks. Here, we introduce such a framework that connects meteorology, AI, and social sciences. As an example, we apply it to the 150-year-old problem of Indian monsoon forecasting, focusing on benefits to rain-fed agriculture, which is highly susceptible to climate change. AIWP models skillfully predict an agriculturally relevant onset index at regional scales weeks in advance when evaluated out-of-sample using deterministic and probabilistic metrics. This framework informed a government-led effort in 2025 to send 38 million Indian farmers AI-based monsoon onset forecasts, which captured an unusual weeks-long pause in monsoon progression. This decision-oriented benchmarking framework provides a key component of a blueprint for harnessing the power of AIWP models to help large vulnerable populations adapt to weather shocks in the face of climate variability and change.

[**arXiv:2602.03767v1**](https://arxiv.org/abs/2602.03767v1)

**Tags:** ``

---

### Downscaling land surface temperature data using edge detection and block-diagonal Gaussian process regression

**Authors:** Sanjit Dandapanthula, Margaret Johnson, Madeleine Pascolini-Campbell, Glynn Hulley, Mikael Kuusela

**Year:** 2026

**Abstract:**
> Accurate and high-resolution estimation of land surface temperature (LST) is crucial in estimating evapotranspiration, a measure of plant water use and a central quantity in agricultural applications. In this work, we develop a novel statistical method for downscaling LST data obtained from NASA's ECOSTRESS mission, using high-resolution data from the Landsat 8 mission as a proxy for modeling agricultural field structure. Using the Landsat data, we identify the boundaries of agricultural fields through edge detection techniques, allowing us to capture the inherent block structure present in the spatial domain. We propose a block-diagonal Gaussian process (BDGP) model that captures the spatial structure of the agricultural fields, leverages independence of LST across fields for computational tractability, and accounts for the change of support present in ECOSTRESS observations. We use the resulting BDGP model to perform Gaussian process regression and obtain high-resolution estimates of LST from ECOSTRESS data, along with uncertainty quantification. Our results demonstrate the practicality of the proposed method in producing reliable high-resolution LST estimates, with potential applications in agriculture, urban planning, and climate studies.

[**arXiv:2602.02813v1**](https://arxiv.org/abs/2602.02813v1)

**Tags:** ``

---

### On a system of equations arising in meteorology: Well-posedness and data assimilation

**Authors:** Eduard Feireisl, Piotr Gwiazda, Agnieszka Świerczewska-Gwiazda

**Year:** 2026

**Abstract:**
> Data assimilation plays a crucial role in modern weather prediction, providing a systematic way to incorporate observational data into complex dynamical models. The paper addresses continuous data assimilation for a model arising as a singular limit of the three-dimensional compressible Navier-Stokes-Fourier system with rotation driven by temperature gradient. The limit system preserves the essential physical mechanisms of the original model, while exhibiting a reduced, effectively two-and-a-half-dimensional structure. This simplified framework allows for a rigorous analytical study of the data assimilation process while maintaining a direct physical connection to the full compressible model. We establish well posedness of global-in-time solutions and a compact trajectory attractor, followed by the stability and convergence results for the nudging scheme applied to the limiting system. Finally, we demonstrate how these results can be combined with a relative entropy argument to extend the assimilation framework to the full three-dimensional compressible setting, thereby establishing a rigorous connection between the reduced and physically complete models.

[**arXiv:2602.02328v1**](https://arxiv.org/abs/2602.02328v1)

**Tags:** ``

---

### WADEPre: A Wavelet-based Decomposition Model for Extreme Precipitation Nowcasting with Multi-Scale Learning

**Authors:** Baitian Liu, Haiping Zhang, Huiling Yuan, Dongjing Wang, Ying Li, Feng Chen, Hao Wu

**Year:** 2026

**Abstract:**
> The heavy-tailed nature of precipitation intensity impedes precise precipitation nowcasting. Standard models that optimize pixel-wise losses are prone to regression-to-the-mean bias, which blurs extreme values. Existing Fourier-based methods also lack the spatial localization needed to resolve transient convective cells. To overcome these intrinsic limitations, we propose WADEPre, a wavelet-based decomposition model for extreme precipitation that transitions the modeling into the wavelet domain. By leveraging the Discrete Wavelet Transform for explicit decomposition, WADEPre employs a dual-branch architecture: an Approximation Network to model stable, low-frequency advection, isolating deterministic trends from statistical bias, and a spatially localized Detail Network to capture high-frequency stochastic convection, resolving transient singularities and preserving sharp boundaries. A subsequent Refiner module then dynamically reconstructs these decoupled multi-scale components into the final high-fidelity forecast. To address optimization instability, we introduce a multi-scale curriculum learning strategy that progressively shifts supervision from coarse scales to fine-grained details. Extensive experiments on the SEVIR and Shanghai Radar datasets demonstrate that WADEPre achieves state-of-the-art performance, yielding significant improvements in capturing extreme thresholds and maintaining structural fidelity. Our code is available at https://github.com/sonderlau/WADEPre.

[**arXiv:2602.02096v1**](https://arxiv.org/abs/2602.02096v1)

**Tags:** ``

---

### EMFormer: Efficient Multi-Scale Transformer for Accumulative Context Weather Forecasting

**Authors:** Hao Chen, Tao Han, Jie Zhang, Song Guo, Fenghua Ling, Lei Bai

**Year:** 2026

**Abstract:**
> Long-term weather forecasting is critical for socioeconomic planning and disaster preparedness. While recent approaches employ finetuning to extend prediction horizons, they remain constrained by the issues of catastrophic forgetting, error accumulation, and high training overhead. To address these limitations, we present a novel pipeline across pretraining, finetuning and forecasting to enhance long-context modeling while reducing computational overhead. First, we introduce an Efficient Multi-scale Transformer (EMFormer) to extract multi-scale features through a single convolution in both training and inference. Based on the new architecture, we further employ an accumulative context finetuning to improve temporal consistency without degrading short-term accuracy. Additionally, we propose a composite loss that dynamically balances different terms via a sinusoidal weighting, thereby adaptively guiding the optimization trajectory throughout pretraining and finetuning. Experiments show that our approach achieves strong performance in weather forecasting and extreme event prediction, substantially improving long-term forecast accuracy. Moreover, EMFormer demonstrates strong generalization on vision benchmarks (ImageNet-1K and ADE20K) while delivering a 5.69x speedup over conventional multi-scale modules.

[**arXiv:2602.01194v1**](https://arxiv.org/abs/2602.01194v1)

**Tags:** ``

---

### Universal Diffusion-Based Probabilistic Downscaling

**Authors:** Roberto Molinaro, Niall Siegenheim, Henry Martin, Mark Frey, Niels Poulsen, Philipp Seitz, Marvin Vincent Gabler

**Year:** 2026

**Abstract:**
> We introduce a universal diffusion-based downscaling framework that lifts deterministic low-resolution weather forecasts into probabilistic high-resolution predictions without any model-specific fine-tuning. A single conditional diffusion model is trained on paired coarse-resolution inputs (~25 km resolution) and high-resolution regional reanalysis targets (~5 km resolution), and is applied in a fully zero-shot manner to deterministic forecasts from heterogeneous upstream weather models. Focusing on near-surface variables, we evaluate probabilistic forecasts against independent in situ station observations over lead times up to 90 h. Across a diverse set of AI-based and numerical weather prediction (NWP) systems, the ensemble mean of the downscaled forecasts consistently improves upon each model's own raw deterministic forecast, and substantially larger gains are observed in probabilistic skill as measured by CRPS. These results demonstrate that diffusion-based downscaling provides a scalable, model-agnostic probabilistic interface for enhancing spatial resolution and uncertainty representation in operational weather forecasting pipelines.

[**arXiv:2602.11893v1**](https://arxiv.org/abs/2602.11893v1)

**Tags:** ``

---

### PuYun-LDM: A Latent Diffusion Model for High-Resolution Ensemble Weather Forecasts

**Authors:** Lianjun Wu, Shengchen Zhu, Yuxuan Liu, Liuyu Kai, Xiaoduan Feng, Duomin Wang, Wenshuo Liu, Jingxuan Zhang, Kelvin Li, Bin Wang

**Year:** 2026

**Abstract:**
> Latent diffusion models (LDMs) suffer from limited diffusability in high-resolution (<=0.25°) ensemble weather forecasting, where diffusability characterizes how easily a latent data distribution can be modeled by a diffusion process. Unlike natural image fields, meteorological fields lack task-agnostic foundation models and explicit semantic structures, making VFM-based regularization inapplicable. Moreover, existing frequency-based approaches impose identical spectral regularization across channels under a homogeneity assumption, which leads to uneven regularization strength under the inter-variable spectral heterogeneity in multivariate meteorological data. To address these challenges, we propose a 3D Masked AutoEncoder (3D-MAE) that encodes weather-state evolution features as an additional conditioning for the diffusion model, together with a Variable-Aware Masked Frequency Modeling (VA-MFM) strategy that adaptively selects thresholds based on the spectral energy distribution of each variable. Together, we propose PuYun-LDM, which enhances latent diffusability and achieves superior performance to ENS at short lead times while remaining comparable to ENS at longer horizons. PuYun-LDM generates a 15-day global forecast with a 6-hour temporal resolution in five minutes on a single NVIDIA H200 GPU, while ensemble forecasts can be efficiently produced in parallel.

[**arXiv:2602.11807v1**](https://arxiv.org/abs/2602.11807v1)

**Tags:** ``

---

### Regularized Ensemble Forecasting for Learning Weights from Historical and Current Forecasts

**Authors:** Han Su, Xiaojia Guo, Xiaoke Zhang

**Year:** 2026

**Abstract:**
> Combining forecasts from multiple experts often yields more accurate results than relying on a single expert. In this paper, we introduce a novel regularized ensemble method that extends the traditional linear opinion pool by leveraging both current forecasts and historical performances to set the weights. Unlike existing approaches that rely only on either the current forecasts or past accuracy, our method accounts for both sources simultaneously. It learns weights by minimizing the variance of the combined forecast (or its transformed version) while incorporating a regularization term informed by historical performances. We also show that this approach has a Bayesian interpretation. Different distributional assumptions within this Bayesian framework yield different functional forms for the variance component and the regularization term, adapting the method to various scenarios. In empirical studies on Walmart sales and macroeconomic forecasting, our ensemble outperforms leading benchmark models both when experts' full forecasting histories are available and when experts enter and exit over time, resulting in incomplete historical records. Throughout, we provide illustrative examples that show how the optimal weights are determined and, based on the empirical results, we discuss where the framework's strengths lie and when experts' past versus current forecasts are more informative.

[**arXiv:2602.11379v1**](https://arxiv.org/abs/2602.11379v1)

**Tags:** ``

---

### Data assimilation via model reference adaptation for linear and nonlinear dynamical systems

**Authors:** Benedikt Kaltenbach, Christian Aarset, Tram Thi Ngoc Nguyen

**Year:** 2026

**Abstract:**
> We address data assimilation for linear and nonlinear dynamical systems via the so-called \emph{model reference adaptive system}. Continuing our theoretical developments in \cite{Tram_Kaltenbacher_2021}, we deliver the first practical implementation of this approach for online parameter identification with time series data. Our semi-implicit scheme couples a modified state equation with a parameter evolution law that is driven by model-data residuals. We demonstrate four benchmark problems of increasing complexity: the Darcy flow, the Fisher-KPP equation, a nonlinear potential equation and finally, an Allen-Cahn type equation. Across all cases, explicit model reference adaptive system construction, verified assumptions and numerically stable reconstructions underline our proposed method as a reliable, versatile tool for data assimilation and real-time inversion.

[**arXiv:2602.10920v1**](https://arxiv.org/abs/2602.10920v1)

**Tags:** ``

---

### Exploring Novel Data Storage Approaches for Large-Scale Numerical Weather Prediction

**Authors:** Nicolau Manubens Gil

**Year:** 2026

**Abstract:**
> Driven by scientific and industry ambition, HPC and AI applications such as operational Numerical Weather Prediction (NWP) require processing and storing ever-increasing data volumes as fast as possible. Whilst POSIX distributed file systems and NVMe SSDs are currently a common HPC storage configuration providing I/O to applications, new storage solutions have proliferated or gained traction over the last decade with potential to address performance limitations POSIX file systems manifest at scale for certain I/O workloads.   This work has primarily aimed to assess the suitability and performance of two object storage systems -namely DAOS and Ceph- for the ECMWF's operational NWP as well as for HPC and AI applications in general. New software-level adapters have been developed which enable the ECMWF's NWP to leverage these systems, and extensive I/O benchmarking has been conducted on a few computer systems, comparing the performance delivered by the evaluated object stores to that of equivalent Lustre file system deployments on the same hardware. Challenges of porting to object storage and its benefits with respect to the traditional POSIX I/O approach have been discussed and, where possible, domain-agnostic performance analysis has been conducted, leading to insight also of relevance to I/O practitioners and the broader HPC community.   DAOS and Ceph have both demonstrated excellent performance, but DAOS stood out relative to Ceph and Lustre, providing superior scalability and flexibility for applications to perform I/O at scale as desired. This sets a promising outlook for DAOS and object storage, which might see greater adoption at HPC centres in the years to come, although not necessarily implying a shift away from POSIX-like I/O.

[**arXiv:2602.17610v1**](https://arxiv.org/abs/2602.17610v1)

**Tags:** ``

---

### Preconditioned Adjoint Data Assimilation for Two-Dimensional Decaying Isotropic Turbulence

**Authors:** Hongyi Ke, Zejian You, Qi Wang

**Year:** 2026

**Abstract:**
> Adjoint-based data assimilation for turbulent Navier-Stokes flows is fundamentally limited by the behavior of the adjoint dynamics: in backward time, adjoint fields exhibit exponential growth and become increasingly dominated by small-scale structures, severely degrading reconstruction of the initial condition from sparse measurements. We demonstrate that the relative weighting of spectral components in the adjoint formulation can be systematically controlled by redefining the inner product under which the adjoint operator is defined. The inverse problem is formulated as a constrained minimization in which a cost functional measures the mismatch between model predictions and observations, and the adjoint equations provide the gradient with respect to the initial velocity field. Redefining the forward-adjoint duality through a Fourier-space weighting kernel preconditions the optimization and is mathematically equivalent to changing the representation of the control variable or, alternatively, introducing a smoothing operation on the governing dynamics. Specific kernel choices correspond to fractional integration or diffusion operators applied to the initial condition. Among these, exponential kernels provide effective regularization by suppressing high-wavenumber contributions while preserving large-scale coherence, leading to improved reconstruction across scales. A statistical analysis of an ensemble of adjoint fields from different turbulent realizations reveals scale-dependent backward growth rates, explaining the instability of the standard formulation and clarifying the mechanism by which the proposed preconditioning attenuates incoherent small-scale amplification.

[**arXiv:2602.14016v1**](https://arxiv.org/abs/2602.14016v1)

**Tags:** ``

---

### Distillation and Interpretability of Ensemble Forecasts of ENSO Phase using Entropic Learning

**Authors:** Michael Groom, Davide Bassetti, Illia Horenko, Terence J. O'Kane

**Year:** 2026

**Abstract:**
> This paper introduces a distillation framework for an ensemble of entropy-optimal Sparse Probabilistic Approximation (eSPA) models, trained exclusively on satellite-era observational and reanalysis data to predict ENSO phase up to 24 months in advance. While eSPA ensembles yield state-of-the-art forecast skill, they are harder to interpret than individual eSPA models. We show how to compress the ensemble into a compact set of "distilled" models by aggregating the structure of only those ensemble members that make correct predictions. This process yields a single, diagnostically tractable model for each forecast lead time that preserves forecast performance while also enabling diagnostics that are impractical to implement on the full ensemble.   An analysis of the regime persistence of the distilled model "superclusters", as well as cross-lead clustering consistency, shows that the discretised system accurately captures the spatiotemporal dynamics of ENSO. By considering the effective dimension of the feature importance vectors, the complexity of the input space required for correct ENSO phase prediction is shown to peak when forecasts must cross the boreal spring predictability barrier. Spatial importance maps derived from the feature importance vectors are introduced to identify where predictive information resides in each field and are shown to include known physical precursors at certain lead times. Case studies of key events are also presented, showing how fields reconstructed from distilled model centroids trace the evolution from extratropical and inter-basin precursors to the mature ENSO state. Overall, the distillation framework enables a rigorous investigation of long-range ENSO predictability that complements real-time data-driven operational forecasts.

[**arXiv:2602.16857v1**](https://arxiv.org/abs/2602.16857v1)

**Tags:** ``

---

### Using machine learning to downscale coarse-resolution environmental variables for understanding the spatial frequency of convective storms

**Authors:** Hungjui Yu, Lander Ver Hoef, Kristen L. Rasmussen, Imme Ebert-Uphoff

**Year:** 2025

**Abstract:**
> Global climate models (GCMs), typically run at ~100-km resolution, capture large-scale environmental conditions but cannot resolve convection and cloud processes at kilometer scales. Convection-permitting models offer higher-resolution simulations that explicitly simulate convection but are computationally expensive and impractical for large ensemble runs. This study explores machine learning (ML) as a bridge between these approaches. We train simple, pixel-based neural networks to predict convective storm frequency from environmental variables produced by a regional convection-permitting model. The ML models achieve promising results, with structural similarity index measure (SSIM) values exceeding 0.8, capturing the diurnal cycle and orographic convection without explicit temporal or spatial coordinates as input. Model performance declines when fewer input features are used or specific regions are excluded, underscoring the role of diverse physical mechanisms in convective activity. These findings highlight ML potential as a computationally efficient tool for representing convection and as a means of scientific discovery, offering insights into convective processes. Unlike convolutional neural networks, which depend on spatial structure and grid size, the pixel-based model treats each grid point independently, enabling value-to-value prediction without spatial context. This design enhances adaptability to resolution changes and supports generalization to unseen environmental regimes, making it particularly suited for linking environmental conditions to convective features and for application across diverse model grids or climate scenarios.

[**arXiv:2509.08802v1**](https://arxiv.org/abs/2509.08802v1)

**Tags:** ``

---

### Nuclear Data Adjustment for Nonlinear Applications in the OECD/NEA WPNCS SG14 Benchmark -- A Bayesian Inverse UQ-based Approach for Data Assimilation

**Authors:** Christopher Brady, Xu Wu

**Year:** 2025

**Abstract:**
> The Organization for Economic Cooperation and Development (OECD) Working Party on Nuclear Criticality Safety (WPNCS) proposed a benchmark exercise to assess the performance of current nuclear data adjustment techniques applied to nonlinear applications and experiments with low correlation to applications. This work introduces Bayesian Inverse Uncertainty Quantification (IUQ) as a method for nuclear data adjustments in this benchmark, and compares IUQ to the more traditional methods of Generalized Linear Least Squares (GLLS) and Monte Carlo Bayes (MOCABA). Posterior predictions from IUQ showed agreement with GLLS and MOCABA for linear applications. When comparing GLLS, MOCABA, and IUQ posterior predictions to computed model responses using adjusted parameters, we observe that GLLS predictions fail to replicate computed response distributions for nonlinear applications, while MOCABA shows near agreement, and IUQ uses computed model responses directly. We also discuss observations on why experiments with low correlation to applications can be informative to nuclear data adjustments and identify some properties useful in selecting experiments for inclusion in nuclear data adjustment. Performance in this benchmark indicates potential for Bayesian IUQ in nuclear data adjustments.

[**arXiv:2509.07790v1**](https://arxiv.org/abs/2509.07790v1)

**Tags:** ``

---

### FlowCast-ODE: Continuous Hourly Weather Forecasting with Dynamic Flow Matching and ODE Integration

**Authors:** Shuangshuang He, Yuanting Zhang, Hongli Liang, Qingye Meng, Xingyuan Yuan

**Year:** 2025

**Abstract:**
> Accurate hourly weather forecasting is critical for numerous applications. Recent deep learning models have demonstrated strong capability on 6-hour intervals, yet achieving accurate and stable hourly predictions remains a critical challenge. This is primarily due to the rapid accumulation of errors in autoregressive rollouts and temporal discontinuities within the ERA5 data's 12-hour assimilation cycle. To address these issues, we propose FlowCast-ODE, a framework that models atmospheric state evolution as a continuous flow. FlowCast-ODE learns the conditional flow path directly from the previous state, an approach that aligns more naturally with physical dynamic systems and enables efficient computation. A coarse-to-fine strategy is introduced to train the model on 6-hour data using dynamic flow matching and then refined on hourly data that incorporates an Ordinary Differential Equation (ODE) solver to achieve temporally coherent forecasts. In addition, a lightweight low-rank AdaLN-Zero modulation mechanism is proposed and reduces model size by 15% without compromising accuracy. Experiments demonstrate that FlowCast-ODE outperforms strong baselines, yielding lower root mean square error (RMSE) and better energy conservation, which reduces blurring and preserves more fine-scale spatial details. It also shows comparable performance to the state-of-the-art model in forecasting extreme events like typhoons. Furthermore, the model alleviates temporal discontinuities associated with assimilation cycle transitions.

[**arXiv:2509.14775v1**](https://arxiv.org/abs/2509.14775v1)

**Tags:** ``

---

### Lagrangian-Eulerian Multiscale Data Assimilation in Physical Domain based on Conditional Gaussian Nonlinear System

**Authors:** Hyeonggeun Yun, Quanling Deng

**Year:** 2025

**Abstract:**
> This research aims to further investigate the process of Lagrangian-Eulerian Multiscale Data Assimilation (LEMDA) by replacing the Fourier space with the physical domain. Such change in the perspective of domain introduces the advantages of being able to deal in non-periodic system and more intuitive representation of localised phenomena or time-dependent problems. The context of the domain for this paper was set as sea ice floe trajectories to recover the ocean eddies in the Arctic regions, which led the model to be derived from two-layer Quasi geostrophic (QG) model. The numerical solution to this model utilises the Conditional Gaussian Nonlinear System (CGNS) to accommodate the inherent non-linearity in analytical and continuous manner. The normalised root mean square error (RMSE) and pattern correlation (Corr) are used to evaluate the performance of the posterior mean of the model. The results corroborate the effectiveness of exploiting the two-layer QG model in physical domain. Nonetheless, the paper still discusses opportunities of improvement, such as deploying neural network (NN) to accelerate the recovery of local particle of Lagrangian DA for the fine scale.

[**arXiv:2509.14586v1**](https://arxiv.org/abs/2509.14586v1)

**Tags:** ``

---

### Variational data assimilation for the wave equation in heterogeneous media: Numerical investigation of stability

**Authors:** Erik Burman, Janosch Preuss, Tim van Beeck

**Year:** 2025

**Abstract:**
> In recent years, several numerical methods for solving the unique continuation problem for the wave equation in a homogeneous medium with given data on the lateral boundary of the space-time cylinder have been proposed. This problem enjoys Lipschitz stability if the geometric control condition is fulfilled, which allows devising optimally convergent numerical methods. In this article, we investigate whether these results carry over to the case in which the medium exhibits a jump discontinuity. Our numerical experiments suggest a positive answer. However, we also observe that the presence of discontinuities in the medium renders the computations far more demanding than in the homogeneous case.

[**arXiv:2509.13108v1**](https://arxiv.org/abs/2509.13108v1)

**Tags:** ``

---

### SamudrACE: Fast and Accurate Coupled Climate Modeling with 3D Ocean and Atmosphere Emulators

**Authors:** James P. C. Duncan, Elynn Wu, Surya Dheeshjith, Adam Subel, Troy Arcomano, Spencer K. Clark, Brian Henn, Anna Kwa, Jeremy McGibbon, W. Andre Perkins, William Gregory, Carlos Fernandez-Granda, Julius Busecke, Oliver Watt-Meyer, William J. Hurlin, Alistair Adcroft, Laure Zanna, Christopher Bretherton

**Year:** 2025

**Abstract:**
> Traditional numerical global climate models simulate the full Earth system by exchanging boundary conditions between separate simulators of the atmosphere, ocean, sea ice, land surface, and other geophysical processes. This paradigm allows for distributed development of individual components within a common framework, unified by a coupler that handles translation between realms via spatial or temporal alignment and flux exchange. Following a similar approach adapted for machine learning-based emulators, we present SamudrACE: a coupled global climate model emulator which produces centuries-long simulations at 1-degree horizontal, 6-hourly atmospheric, and 5-daily oceanic resolution, with 145 2D fields spanning 8 atmospheric and 19 oceanic vertical levels, plus sea ice, surface, and top-of-atmosphere variables. SamudrACE is highly stable and has low climate biases comparable to those of its components with prescribed boundary forcing, with realistic variability in coupled climate phenomena such as ENSO that is not possible to simulate in uncoupled mode.

[**arXiv:2509.12490v1**](https://arxiv.org/abs/2509.12490v1)

**Tags:** ``

---

### Do machine learning climate models work in changing climate dynamics?

**Authors:** Maria Conchita Agana Navarro, Geng Li, Theo Wolf, María Pérez-Ortiz

**Year:** 2025

**Abstract:**
> Climate change is accelerating the frequency and severity of unprecedented events, deviating from established patterns. Predicting these out-of-distribution (OOD) events is critical for assessing risks and guiding climate adaptation. While machine learning (ML) models have shown promise in providing precise, high-speed climate predictions, their ability to generalize under distribution shifts remains a significant limitation that has been underexplored in climate contexts. This research systematically evaluates state-of-the-art ML-based climate models in diverse OOD scenarios by adapting established OOD evaluation methodologies to climate data. Experiments on large-scale datasets reveal notable performance variability across scenarios, shedding light on the strengths and limitations of current models. These findings underscore the importance of robust evaluation frameworks and provide actionable insights to guide the reliable application of ML for climate risk forecasting.

[**arXiv:2509.12147v1**](https://arxiv.org/abs/2509.12147v1)

**Tags:** ``

---

### Data-Efficient Ensemble Weather Forecasting with Diffusion Models

**Authors:** Kevin Valencia, Ziyang Liu, Justin Cui

**Year:** 2025

**Abstract:**
> Although numerical weather forecasting methods have dominated the field, recent advances in deep learning methods, such as diffusion models, have shown promise in ensemble weather forecasting. However, such models are typically autoregressive and are thus computationally expensive. This is a challenge in climate science, where data can be limited, costly, or difficult to work with. In this work, we explore the impact of curated data selection on these autoregressive diffusion models. We evaluate several data sampling strategies and show that a simple time stratified sampling approach achieves performance similar to or better than full-data training. Notably, it outperforms the full-data model on certain metrics and performs only slightly worse on others while using only 20% of the training data. Our results demonstrate the feasibility of data-efficient diffusion training, especially for weather forecasting, and motivates future work on adaptive or model-aware sampling methods that go beyond random or purely temporal sampling.

[**arXiv:2509.11047v1**](https://arxiv.org/abs/2509.11047v1)

**Tags:** ``

---

### Mesh Interpolation Graph Network for Dynamic and Spatially Irregular Global Weather Forecasting

**Authors:** Zinan Zheng, Yang Liu, Jia Li

**Year:** 2025

**Abstract:**
> Graph neural networks have shown promising results in weather forecasting, which is critical for human activity such as agriculture planning and extreme weather preparation. However, most studies focus on finite and local areas for training, overlooking the influence of broader areas and limiting their ability to generalize effectively. Thus, in this work, we study global weather forecasting that is irregularly distributed and dynamically varying in practice, requiring the model to generalize to unobserved locations. To address such challenges, we propose a general Mesh Interpolation Graph Network (MIGN) that models the irregular weather station forecasting, consisting of two key designs: (1) learning spatially irregular data with regular mesh interpolation network to align the data; (2) leveraging parametric spherical harmonics location embedding to further enhance spatial generalization ability. Extensive experiments on an up-to-date observation dataset show that MIGN significantly outperforms existing data-driven models. Besides, we show that MIGN has spatial generalization ability, and is capable of generalizing to previous unseen stations.

[**arXiv:2509.20911v1**](https://arxiv.org/abs/2509.20911v1)

**Tags:** ``

---

### Graph-based Neural Space Weather Forecasting

**Authors:** Daniel Holmberg, Ivan Zaitsev, Markku Alho, Ioanna Bouri, Fanni Franssila, Haewon Jeong, Minna Palmroth, Teemu Roos

**Year:** 2025

**Abstract:**
> Accurate space weather forecasting is crucial for protecting our increasingly digital infrastructure. Hybrid-Vlasov models, like Vlasiator, offer physical realism beyond that of current operational systems, but are too computationally expensive for real-time use. We introduce a graph-based neural emulator trained on Vlasiator data to autoregressively predict near-Earth space conditions driven by an upstream solar wind. We show how to achieve both fast deterministic forecasts and, by using a generative model, produce ensembles to capture forecast uncertainty. This work demonstrates that machine learning offers a way to add uncertainty quantification capability to existing space weather prediction systems, and make hybrid-Vlasov simulation tractable for operational use.

[**arXiv:2509.19605v1**](https://arxiv.org/abs/2509.19605v1)

**Tags:** ``

---

### Physics-Informed Field Inversion for Sparse Data Assimilation

**Authors:** Levent Ugur, Beckett Y. Zhou

**Year:** 2025

**Abstract:**
> Data-driven methods keep increasing their popularity in engineering applications, given the developments in data analysis techniques. Some of these approaches, such as Field Inversion Machine Learning (FIML), suggest correcting low-fidelity models by leveraging available observations of the problem. However, the solely data-driven field inversion stage of the method generally requires dense observations that limit the usage of sparse data. In this study, we propose a physical loss term addition to the field inversion stage of the FIML technique similar to the physics-informed machine learning applications. This addition embeds the complex physics of the problem into the low-fidelity model, which allows for obtaining dense gradient information for every correction parameter and acts as an adaptive regularization term improving inversion accuracy. The proposed Physics-Informed Field Inversion approach is tested using three different examples and highlights that incorporating physical loss can enhance the reconstruction performance for limited data cases, such as sparse, truncated, and noisy observations. Additionally, this modification enables us to obtain accurate posterior correction parameter distribution with limited realizations, making it data-efficient. The increase in the computational cost caused by the physical loss calculation is at an acceptable level given the relaxed grid and numerical scheme requirements.

[**arXiv:2509.19160v1**](https://arxiv.org/abs/2509.19160v1)

**Tags:** ``

---

### An update to ECMWF's machine-learned weather forecast model AIFS

**Authors:** Gabriel Moldovan, Ewan Pinnington, Ana Prieto Nemesio, Simon Lang, Zied Ben Bouallègue, Jesper Dramsch, Mihai Alexe, Mario Santa Cruz, Sara Hahner, Harrison Cook, Helen Theissen, Mariana Clare, Cathal O'Brien, Jan Polster, Linus Magnusson, Gert Mertes, Florian Pinault, Baudouin Raoult, Patricia de Rosnay, Richard Forbes, Matthew Chantry

**Year:** 2025

**Abstract:**
> We present an update to ECMWF's machine-learned weather forecasting model AIFS Single with several key improvements. The model now incorporates physical consistency constraints through bounding layers, an updated training schedule, and an expanded set of variables. The physical constraints substantially improve precipitation forecasts and the new variables show a high level of skill. Upper-air headline scores also show improvement over the previous AIFS version. The AIFS has been fully operational at ECMWF since the 25th of February 2025.

[**arXiv:2509.18994v1**](https://arxiv.org/abs/2509.18994v1)

**Tags:** ``

---

### Training-Free Data Assimilation with GenCast

**Authors:** Thomas Savary, François Rozet, Gilles Louppe

**Year:** 2025

**Abstract:**
> Data assimilation is widely used in many disciplines such as meteorology, oceanography, and robotics to estimate the state of a dynamical system from noisy observations. In this work, we propose a lightweight and general method to perform data assimilation using diffusion models pre-trained for emulating dynamical systems. Our method builds on particle filters, a class of data assimilation algorithms, and does not require any further training. As a guiding example throughout this work, we illustrate our methodology on GenCast, a diffusion-based model that generates global ensemble weather forecasts.

[**arXiv:2509.18811v1**](https://arxiv.org/abs/2509.18811v1)

**Tags:** ``

---

### Comparing Data Assimilation and Likelihood-Based Inference on Latent State Estimation in Agent-Based Models

**Authors:** Blas Kolic, Corrado Monti, Gianmarco De Francisci Morales, Marco Pangallo

**Year:** 2025

**Abstract:**
> In this paper, we present the first systematic comparison of Data Assimilation (DA) and Likelihood-Based Inference (LBI) in the context of Agent-Based Models (ABMs). These models generate observable time series driven by evolving, partially-latent microstates. Latent states need to be estimated to align simulations with real-world data -- a task traditionally addressed by DA, especially in continuous and equation-based models such as those used in weather forecasting. However, the nature of ABMs poses challenges for standard DA methods. Solving such issues requires adaptation of previous DA techniques, or ad-hoc alternatives such as LBI. DA approximates the likelihood in a model-agnostic way, making it broadly applicable but potentially less precise. In contrast, LBI provides more accurate state estimation by directly leveraging the model's likelihood, but at the cost of requiring a hand-crafted, model-specific likelihood function, which may be complex or infeasible to derive. We compare the two methods on the Bounded-Confidence Model, a well-known opinion dynamics ABM, where agents are affected only by others holding sufficiently similar opinions. We find that LBI better recovers latent agent-level opinions, even under model mis-specification, leading to improved individual-level forecasts. At the aggregate level, however, both methods perform comparably, and DA remains competitive across levels of aggregation under certain parameter settings. Our findings suggest that DA is well-suited for aggregate predictions, while LBI is preferable for agent-level inference.

[**arXiv:2509.17625v1**](https://arxiv.org/abs/2509.17625v1)

**Tags:** ``

---

### Disrespect Others, Respect the Climate? Applying Social Dynamics with Inequality to Forest Climate Models

**Authors:** Luke Wisniewski, Thomas Zdyrski, Feng Fu

**Year:** 2025

**Abstract:**
> Understanding the role of human behavior in shaping environmental outcomes is crucial for addressing global challenges such as climate change. Environmental systems are influenced not only by natural factors like temperature, but also by human decisions regarding mitigation efforts, which are often based on forecasts or predictions about future environmental conditions. Over time, different outcomes can emerge, including scenarios where the environment deteriorates despite efforts to mitigate, or where successful mitigation leads to environmental resilience. Additionally, fluctuations in the level of human participation in mitigation can occur, reflecting shifts in collective behavior. In this study, we consider a variety of human mitigation decisions, in addition to the feedback loop that is created by changes in human behavior because of environmental changes. While these outcomes are based on simplified models, they offer important insights into the dynamics of human decision-making and the factors that influence effective action in the context of environmental sustainability. This study aims to examine key social dynamics influencing society's response to a worsening climate. While others conclude that homophily prompts greater warming unconditionally, this model finds that homophily can prevent catastrophic effects given a poor initial environmental state. Assuming that poor countries have the resources to do so, a consensus in that class group to defect from the strategy of the rich group (who are generally incentivized to continue ``business as usual'') can frequently prevent the vegetation proportion from converging to 0.

[**arXiv:2509.17252v1**](https://arxiv.org/abs/2509.17252v1)

**Tags:** ``

---

### Numerical Reconstruction of Coefficients in Elliptic Equations Using Continuous Data Assimilation

**Authors:** Peiran Zhang

**Year:** 2025

**Abstract:**
> We consider the numerical reconstruction of the spatially dependent conductivity coefficient and the source term in elliptic partial differential equations of in a two-dimensional convex polygonal domain, with the homogeneous Dirichlet boundary condition and given interior observation of the solution. Using data assimilation, some approximated gradients of our error functional are derived to update the reconstructed coefficients, and new $L^2$ error estimates for such minimization formulations are given for the spatially discretized reconstructions. Numerical examples are provided to show the effectiveness of the method and demonstrate the error estimates. The numerical results also show that the reconstruction is very robust for the error in certain inputted coefficients.

[**arXiv:2509.16954v1**](https://arxiv.org/abs/2509.16954v1)

**Tags:** ``

---

### Integrating AI and Ensemble Forecasting: Explainable Materials Planning with Scorecards and Trend Insights for a Large-Scale Manufacturer

**Authors:** Saravanan Venkatachalam

**Year:** 2025

**Abstract:**
> This paper presents a practical architecture for after-sales demand forecasting and monitoring that unifies a revenue- and cluster-aware ensemble of statistical, machine-learning, and deep-learning models with a role-driven analytics layer for scorecards and trend diagnostics. The framework ingests exogenous signals (installed base, pricing, macro indicators, life cycle, seasonality) and treats COVID-19 as a distinct regime, producing country-part forecasts with calibrated intervals. A Pareto-aware segmentation forecasts high-revenue items individually and pools the long tail via clusters, while horizon-aware ensembling aligns weights with business-relevant losses (e.g., WMAPE). Beyond forecasts, a performance scorecard delivers decision-focused insights: accuracy within tolerance thresholds by revenue share and count, bias decomposition (over- vs under-forecast), geographic and product-family hotspots, and ranked root causes tied to high-impact part-country pairs. A trend module tracks trajectories of MAPE/WMAPE and bias across recent months, flags entities that are improving or deteriorating, detects change points aligned with known regimes, and attributes movements to lifecycle and seasonal factors. LLMs are embedded in the analytics layer to generate role-aware narratives and enforce reporting contracts. They standardize business definitions, automate quality checks and reconciliations, and translate quantitative results into concise, explainable summaries for planners and executives. The system exposes a reproducible workflow -- request specification, model execution, database-backed artifacts, and AI-generated narratives -- so planners can move from "How accurate are we now?" to "Where is accuracy heading and which levers should we pull?", closing the loop between forecasting, monitoring, and inventory decisions across more than 90 countries and about 6,000 parts.

[**arXiv:2510.01006v1**](https://arxiv.org/abs/2510.01006v1)

**Tags:** ``

---

### Probability calibration for precipitation nowcasting

**Authors:** Lauri Kurki, Yaniel Cabrera, Samu Karanko

**Year:** 2025

**Abstract:**
> Reliable precipitation nowcasting is critical for weather-sensitive decision-making, yet neural weather models (NWMs) can produce poorly calibrated probabilistic forecasts. Standard calibration metrics such as the expected calibration error (ECE) fail to capture miscalibration across precipitation thresholds. We introduce the expected thresholded calibration error (ETCE), a new metric that better captures miscalibration in ordered classes like precipitation amounts. We extend post-processing techniques from computer vision to the forecasting domain. Our results show that selective scaling with lead time conditioning reduces model miscalibration without reducing the forecast quality.

[**arXiv:2510.00594v1**](https://arxiv.org/abs/2510.00594v1)

**Tags:** ``

---

### On the joint observability of flow fields and particle properties from Lagrangian trajectories: evidence from neural data assimilation

**Authors:** Ke Zhou, Samuel J. Grauer

**Year:** 2025

**Abstract:**
> We numerically investigate the joint observability of flow states and unknown particle properties from Lagrangian particle tracking (LPT) data. LPT offers time-resolved, volumetric measurements of particle trajectories, but experimental tracks are spatially sparse, potentially noisy, and may be further complicated by inertial transport, raising the question of whether both Eulerian fields and particle characteristics can be reliably inferred. To address this, we develop a data assimilation framework that couples an Eulerian flow representation with Lagrangian particle models, enabling the simultaneous inference of carrier fields and particle properties under the governing equations of disperse multiphase flow. Using this approach, we establish empirical existence proofs of joint observability across three representative regimes. In a turbulent boundary layer with noisy tracer tracks (St to 0), flow states and true particle positions are jointly observable. In homogeneous isotropic turbulence seeded with inertial particles (St ~ 1-5), we demonstrate simultaneous recovery of flow states and particle diameters, showing the feasibility of implicit particle characterization. In a compressible, shock-dominated flow, we report the first joint reconstructions of velocity, pressure, density, and inertial particle properties (diameter and density), highlighting both the potential and certain limits of observability in supersonic regimes. Systematic sensitivity studies further reveal how seeding density, noise level, and Stokes number govern reconstruction accuracy, yielding practical guidelines for experimental design. Taken together, these results show that the scope of LPT could be broadened to multiphase and high-speed flows, in which tracer and measurement fidelity are limited.

[**arXiv:2510.00479v1**](https://arxiv.org/abs/2510.00479v1)

**Tags:** ``

---

### EnScale: Temporally-consistent multivariate generative downscaling via proper scoring rules

**Authors:** Maybritt Schillinger, Maxim Samarin, Xinwei Shen, Reto Knutti, Nicolai Meinshausen

**Year:** 2025

**Abstract:**
> The practical use of future climate projections from global circulation models (GCMs) is often limited by their coarse spatial resolution, requiring downscaling to generate high-resolution data. Regional climate models (RCMs) provide this refinement, but are computationally expensive. To address this issue, machine learning models can learn the downscaling function, mapping coarse GCM outputs to high-resolution fields. Among these, generative approaches aim to capture the full conditional distribution of RCM data given coarse-scale GCM data, which is characterized by large variability and thus challenging to model accurately. We introduce EnScale, a generative machine learning framework that emulates the full GCM-to-RCM map by training on multiple pairs of GCM and corresponding RCM data. It first adjusts large-scale mismatches between GCM and coarsened RCM data, followed by a super-resolution step to generate high-resolution fields. Both steps employ generative models optimized with the energy score, a proper scoring rule. Compared to state-of-the-art ML downscaling approaches, our setup reduces computational cost by about one order of magnitude. EnScale jointly emulates multiple variables -- temperature, precipitation, solar radiation, and wind -- spatially consistent over an area in Central Europe. In addition, we propose a variant EnScale-t that enables temporally consistent downscaling. We establish a comprehensive evaluation framework across various categories including calibration, spatial structure, extremes, and multivariate dependencies. Comparison with diverse benchmarks demonstrates EnScale's strong performance and computational efficiency. EnScale offers a promising approach for accurate and temporally consistent RCM emulation.

[**arXiv:2509.26258v1**](https://arxiv.org/abs/2509.26258v1)

**Tags:** ``

---

### Swift: An Autoregressive Consistency Model for Efficient Weather Forecasting

**Authors:** Jason Stock, Troy Arcomano, Rao Kotamarthi

**Year:** 2025

**Abstract:**
> Diffusion models offer a physically grounded framework for probabilistic weather forecasting, but their typical reliance on slow, iterative solvers during inference makes them impractical for subseasonal-to-seasonal (S2S) applications where long lead-times and domain-driven calibration are essential. To address this, we introduce Swift, a single-step consistency model that, for the first time, enables autoregressive finetuning of a probability flow model with a continuous ranked probability score (CRPS) objective. This eliminates the need for multi-model ensembling or parameter perturbations. Results show that Swift produces skillful 6-hourly forecasts that remain stable for up to 75 days, running $39\times$ faster than state-of-the-art diffusion baselines while achieving forecast skill competitive with the numerical-based, operational IFS ENS. This marks a step toward efficient and reliable ensemble forecasting from medium-range to seasonal-scales.

[**arXiv:2509.25631v1**](https://arxiv.org/abs/2509.25631v1)

**Tags:** ``

---

### The Open-Source Photochem Code: A General Chemical and Climate Model for Interpreting (Exo)Planet Observations

**Authors:** Nicholas F. Wogan, Natasha E. Batalha, Kevin Zahnle, Joshua Krissansen-Totton, David C. Catling, Eric T. Wolf, Tyler D. Robinson, Victoria Meadows, Giada Arney, Shawn Domagal-Goldman

**Year:** 2025

**Abstract:**
> With the launch of the James Webb Space Telescope, we are firmly in the era of exoplanet atmosphere characterization. Understanding exoplanet spectra requires atmospheric chemical and climate models that span the diversity of planetary atmospheres. Here, we present a more general chemical and climate model of planetary atmospheres. Specifically, we introduce the open-source, one-dimensional photochemical and climate code Photochem, and benchmark the model against the observed compositions and climates of Venus, Earth, Mars, Jupiter and Titan with a single set of kinetics, thermodynamics and opacities. We also model the chemistry of the hot Jupiter exoplanet WASP-39b. All simulations are open-source and reproducible. To first order, Photochem broadly reproduces the gas-phase chemistry and pressure-temperature profiles of all six planets. The largest model-data discrepancies are found in Venus's sulfur chemistry, motivating future experimental work on sulfur kinetics and spacecraft missions to Venus. We also find that clouds and hazes are important for the energy balance of Venus, Earth, Mars and Titan, and that accurately predicting aerosols with Photochem is challenging. Finally, we benchmark Photochem against the popular VULCAN and HELIOS photochemistry and climate models, finding excellent agreement for the same inputs; we also find that Photochem simulates atmospheres 2 to 100 time more efficiently. These results show that Photochem provides a comparatively general description of atmospheric chemistry and physics that can be leveraged to study Solar System worlds or interpret telescope observations of exoplanets.

[**arXiv:2509.25578v1**](https://arxiv.org/abs/2509.25578v1)

**Tags:** ``

---

### Beyond the Training Data: Confidence-Guided Mixing of Parameterizations in a Hybrid AI-Climate Model

**Authors:** Helge Heuer, Tom Beucler, Mierk Schwabe, Julien Savre, Manuel Schlund, Veronika Eyring

**Year:** 2025

**Abstract:**
> Persistent systematic errors in Earth system models (ESMs) arise from difficulties in representing the full diversity of subgrid, multiscale atmospheric convection and turbulence. Machine learning (ML) parameterizations trained on short high-resolution simulations show strong potential to reduce these errors. However, stable long-term atmospheric simulations with hybrid (physics + ML) ESMs remain difficult, as neural networks (NNs) trained offline often destabilize online runs. Training convection parameterizations directly on coarse-grained data is challenging, notably because scales cannot be cleanly separated. This issue is mitigated using data from superparameterized simulations, which provide clearer scale separation. Yet, transferring a parameterization from one ESM to another remains difficult due to distribution shifts that induce large inference errors. Here, we present a proof-of-concept where a ClimSim-trained, physics-informed NN convection parameterization is successfully transferred to ICON-A. The scheme is (a) trained on adjusted ClimSim data with subtracted radiative tendencies, and (b) integrated into ICON-A. The NN parameterization predicts its own error, enabling mixing with a conventional convection scheme when confidence is low, thus making the hybrid AI-physics model tunable with respect to observations and reanalysis through mixing parameters. This improves process understanding by constraining convective tendencies across column water vapor, lower-tropospheric stability, and geographical conditions, yielding interpretable regime behavior. In AMIP-style setups, several hybrid configurations outperform the default convection scheme (e.g., improved precipitation statistics). With additive input noise during training, both hybrid and pure-ML schemes lead to stable simulations and remain physically consistent for at least 20 years.

[**arXiv:2510.08107v1**](https://arxiv.org/abs/2510.08107v1)

**Tags:** ``

---

### SimCast: Enhancing Precipitation Nowcasting with Short-to-Long Term Knowledge Distillation

**Authors:** Yifang Yin, Shengkai Chen, Yiyao Li, Lu Wang, Ruibing Jin, Wei Cui, Shili Xiang

**Year:** 2025

**Abstract:**
> Precipitation nowcasting predicts future radar sequences based on current observations, which is a highly challenging task driven by the inherent complexity of the Earth system. Accurate nowcasting is of utmost importance for addressing various societal needs, including disaster management, agriculture, transportation, and energy optimization. As a complementary to existing non-autoregressive nowcasting approaches, we investigate the impact of prediction horizons on nowcasting models and propose SimCast, a novel training pipeline featuring a short-to-long term knowledge distillation technique coupled with a weighted MSE loss to prioritize heavy rainfall regions. Improved nowcasting predictions can be obtained without introducing additional overhead during inference. As SimCast generates deterministic predictions, we further integrate it into a diffusion-based framework named CasCast, leveraging the strengths from probabilistic models to overcome limitations such as blurriness and distribution shift in deterministic outputs. Extensive experimental results on three benchmark datasets validate the effectiveness of the proposed framework, achieving mean CSI scores of 0.452 on SEVIR, 0.474 on HKO-7, and 0.361 on MeteoNet, which outperforms existing approaches by a significant margin.

[**arXiv:2510.07953v1**](https://arxiv.org/abs/2510.07953v1)

**Tags:** ``

---

### Control-Augmented Autoregressive Diffusion for Data Assimilation

**Authors:** Prakhar Srivastava, Farrin Marouf Sofian, Francesco Immorlano, Kushagra Pandey, Stephan Mandt

**Year:** 2025

**Abstract:**
> Despite recent advances in test-time scaling and finetuning of diffusion models, guidance in Auto-Regressive Diffusion Models (ARDMs) remains underexplored. We introduce an amortized framework that augments pretrained ARDMs with a lightweight controller network, trained offline by previewing future ARDM rollouts and learning stepwise controls that anticipate upcoming observations under a terminal cost objective. We evaluate this framework in the context of data assimilation (DA) for chaotic spatiotemporal partial differential equations (PDEs), a setting where existing methods are often computationally prohibitive and prone to forecast drift under sparse observations. Our approach reduces DA inference to a single forward rollout with on-the-fly corrections, avoiding expensive adjoint computations and/or optimizations during inference. We demonstrate that our method consistently outperforms four state-of-the-art baselines in stability, accuracy, and physical fidelity across two canonical PDEs and six observation regimes. We will release code and checkpoints publicly.

[**arXiv:2510.06637v1**](https://arxiv.org/abs/2510.06637v1)

**Tags:** ``

---

### Road Surface Condition Detection with Machine Learning using New York State Department of Transportation Camera Images and Weather Forecast Data

**Authors:** Carly Sutter, Kara J. Sulia, Nick P. Bassill, Christopher D. Wirz, Christopher D. Thorncroft, Jay C. Rothenberger, Vanessa Przybylo, Mariana G. Cains, Jacob Radford, David Aaron Evans

**Year:** 2025

**Abstract:**
> The New York State Department of Transportation (NYSDOT) has a network of roadside traffic cameras that are used by both the NYSDOT and the public to observe road conditions. The NYSDOT evaluates road conditions by driving on roads and observing live cameras, tasks which are labor-intensive but necessary for making critical operational decisions during winter weather events. However, machine learning models can provide additional support for the NYSDOT by automatically classifying current road conditions across the state. In this study, convolutional neural networks and random forests are trained on camera images and weather data to predict road surface conditions. Models are trained on a hand-labeled dataset of ~22,000 camera images, each classified by human labelers into one of six road surface conditions: severe snow, snow, wet, dry, poor visibility, or obstructed. Model generalizability is prioritized to meet the operational needs of the NYSDOT decision makers, and the weather-related road surface condition model in this study achieves an accuracy of 81.5% on completely unseen cameras.

[**arXiv:2510.06440v1**](https://arxiv.org/abs/2510.06440v1)

**Tags:** ``

---

### Structurally informed data assimilation in two dimensions

**Authors:** Tongtong Li, Anne Gelb, Yoonsang Lee

**Year:** 2025

**Abstract:**
> Accurate data assimilation (DA) for systems with piecewise-smooth or discontinuous state variables remains a significant challenge, as conventional covariance-based ensemble Kalman filter approaches often fail to effectively balance observations and model information near sharp features. In this paper we develop a structurally informed DA framework using ensemble transform Kalman filtering (ETKF). Our approach introduces gradient-based weighting matrices constructed from finite difference statistics of the forecast ensemble, thereby allowing the assimilation process to dynamically adjust the influence of observations and prior estimates according to local roughness. The design is intentionally flexible so that it can be suitably refined for sparse data environments. Numerical experiments demonstrate that our new structurally informed data assimilation framework consistently yields greater accuracy when compared to more conventional approaches.

[**arXiv:2510.06369v1**](https://arxiv.org/abs/2510.06369v1)

**Tags:** ``

---

### Climate Model Tuning with Online Synchronization-Based Parameter Estimation

**Authors:** Jordan Seneca, Suzanne Bintanja, Frank M. Selten

**Year:** 2025

**Abstract:**
> In climate science, the tuning of climate models is a computationally intensive problem due to the combination of the high-dimensionality of the system state and long integration times. Here we demonstrate the potential of a parameter estimation algorithm which makes use of synchronization to tune a global atmospheric model at modest computational costs. We first use it to directly optimize internal model parameters. We then apply the algorithm to the weights of each member of a supermodel ensemble to optimize the overall predictions. In both cases, the algorithm is able to find parameters which result in reduced errors in the climatology of the model. Finally, we introduce a novel approach which combines both methods called adaptive supermodeling, where the internal parameters of the members of a supermodel are tuned simultaneously with the model weights such that the supermodel predictions are optimized. For a case designed to challenge the two previous methods, adaptive supermodeling achieves a performance similar to a perfect model.

[**arXiv:2510.06180v1**](https://arxiv.org/abs/2510.06180v1)

**Tags:** ``

---

### Incorporating Multivariate Consistency in ML-Based Weather Forecasting with Latent-space Constraints

**Authors:** Hang Fan, Yi Xiao, Yongquan Qu, Fenghua Ling, Ben Fei, Lei Bai, Pierre Gentine

**Year:** 2025

**Abstract:**
> Data-driven machine learning (ML) models have recently shown promise in surpassing traditional physics-based approaches for weather forecasting, leading to a so-called second revolution in weather forecasting. However, most ML-based forecast models treat reanalysis as the truth and are trained under variable-specific loss weighting, ignoring their physical coupling and spatial structure. Over long time horizons, the forecasts become blurry and physically unrealistic under rollout training. To address this, we reinterpret model training as a weak-constraint four-dimensional variational data assimilation (WC-4DVar) problem, treating reanalysis data as imperfect observations. This allows the loss function to incorporate reanalysis error covariance and capture multivariate dependencies. In practice, we compute the loss in a latent space learned by an autoencoder (AE), where the reanalysis error covariance becomes approximately diagonal, thus avoiding the need to explicitly model it in the high-dimensional model space. We show that rollout training with latent-space constraints improves long-term forecast skill and better preserves fine-scale structures and physical realism compared to training with model-space loss. Finally, we extend this framework to accommodate heterogeneous data sources, enabling the forecast model to be trained jointly on reanalysis and multi-source observations within a unified theoretical formulation.

[**arXiv:2510.04006v1**](https://arxiv.org/abs/2510.04006v1)

**Tags:** ``

---

### RainDiff: End-to-end Precipitation Nowcasting Via Token-wise Attention Diffusion

**Authors:** Thao Nguyen, Jiaqi Ma, Fahad Shahbaz Khan, Souhaib Ben Taieb, Salman Khan

**Year:** 2025

**Abstract:**
> Precipitation nowcasting, predicting future radar echo sequences from current observations, is a critical yet challenging task due to the inherently chaotic and tightly coupled spatio-temporal dynamics of the atmosphere. While recent advances in diffusion-based models attempt to capture both large-scale motion and fine-grained stochastic variability, they often suffer from scalability issues: latent-space approaches require a separately trained autoencoder, adding complexity and limiting generalization, while pixel-space approaches are computationally intensive and often omit attention mechanisms, reducing their ability to model long-range spatio-temporal dependencies. To address these limitations, we propose a Token-wise Attention integrated into not only the U-Net diffusion model but also the spatio-temporal encoder that dynamically captures multi-scale spatial interactions and temporal evolution. Unlike prior approaches, our method natively integrates attention into the architecture without incurring the high resource cost typical of pixel-space diffusion, thereby eliminating the need for separate latent modules. Our extensive experiments and visual evaluations across diverse datasets demonstrate that the proposed method significantly outperforms state-of-the-art approaches, yielding superior local fidelity, generalization, and robustness in complex precipitation forecasting scenarios.

[**arXiv:2510.14962v1**](https://arxiv.org/abs/2510.14962v1)

**Tags:** ``

---

### Water wave reconstruction of full hydrodynamic models via data assimilation

**Authors:** Liwen Yan, Linyuan Che, Jing Li

**Year:** 2025

**Abstract:**
> A strategy for reconstructing the water wave field using a data assimilation method is proposed in the present study. Special treatments are introduced to address the ensemble diversity and the discontinuous free surface with hydrodynamic constraints when implementing the EnKF approach. Additionally, the POD method is employed for dimensionality reduction, but from an ensemble point of view. The main purpose of this study is to achieve satisfactory consistency between the water waves computed by the numerical solver, particularly by the VOF method, and those observed in the laboratory wave flume within the test section of interest. To validate the proposed framework, three representative conditions are tested: regular waves, irregular waves, and plunging waves. The effects of observation noise, modal truncation, and other factors are also examined. From a practical perspective, this work provides a promising way to realize the coupling between experiments and numerical simulations, and establishes a prototype of a ``digital twin wave tank''.

[**arXiv:2510.14356v1**](https://arxiv.org/abs/2510.14356v1)

**Tags:** ``

---

### Assessing the Geographic Generalization and Physical Consistency of Generative Models for Climate Downscaling

**Authors:** Carlo Saccardi, Maximilian Pierzyna, Haitz Sáez de Ocáriz Borde, Simone Monaco, Cristian Meo, Pietro Liò, Rudolf Saathof, Geethu Joseph, Justin Dauwels

**Year:** 2025

**Abstract:**
> Kilometer-scale weather data is crucial for real-world applications but remains computationally intensive to produce using traditional weather simulations. An emerging solution is to use deep learning models, which offer a faster alternative for climate downscaling. However, their reliability is still in question, as they are often evaluated using standard machine learning metrics rather than insights from atmospheric and weather physics. This paper benchmarks recent state-of-the-art deep learning models and introduces physics-inspired diagnostics to evaluate their performance and reliability, with a particular focus on geographic generalization and physical consistency. Our experiments show that, despite the seemingly strong performance of models such as CorrDiff, when trained on a limited set of European geographies (e.g., central Europe), they struggle to generalize to other regions such as Iberia, Morocco in the south, or Scandinavia in the north. They also fail to accurately capture second-order variables such as divergence and vorticity derived from predicted velocity fields. These deficiencies appear even in in-distribution geographies, indicating challenges in producing physically consistent predictions. We propose a simple initial solution: introducing a power spectral density loss function that empirically improves geographic generalization by encouraging the reconstruction of small-scale physical structures. The code for reproducing the experimental results can be found at https://github.com/CarloSaccardi/PSD-Downscaling

[**arXiv:2510.13722v1**](https://arxiv.org/abs/2510.13722v1)

**Tags:** ``

---

### Specification and Verification for Climate Modeling: Formalization Leading to Impactful Tooling

**Authors:** Alper Altuntas, Allison H. Baker, John Baugh, Ganesh Gopalakrishnan, Stephen F. Siegel

**Year:** 2025

**Abstract:**
> Earth System Models (ESMs) are critical for understanding past climates and projecting future scenarios. However, the complexity of these models, which include large code bases, a wide community of developers, and diverse computational platforms, poses significant challenges for software quality assurance. The increasing adoption of GPUs and heterogeneous architectures further complicates verification efforts. Traditional verification methods often rely on bitwise reproducibility, which is not always feasible, particularly under new compilers or hardware. Manual expert evaluation, on the other hand, is subjective and time-consuming. Formal methods offer a mathematically rigorous alternative, yet their application in ESM development has been limited due to the lack of climate model-specific representations and tools. Here, we advocate for the broader adoption of formal methods in climate modeling. In particular, we identify key aspects of ESMs that are well suited to formal specification and introduce abstraction approaches for a tailored framework. To demonstrate this approach, we present a case study using CIVL model checker to formally verify a bug fix in an ocean mixing parameterization scheme. Our goal is to develop accessible, domain-specific formal tools that enhance model confidence and support more efficient and reliable ESM development.

[**arXiv:2510.13425v1**](https://arxiv.org/abs/2510.13425v1)

**Tags:** ``

---

### Km-scale dynamical downscaling through conformalized latent diffusion models

**Authors:** Alessandro Brusaferri, Andrea Ballarino

**Year:** 2025

**Abstract:**
> Dynamical downscaling is crucial for deriving high-resolution meteorological fields from coarse-scale simulations, enabling detailed analysis for critical applications such as weather forecasting and renewable energy modeling. Generative Diffusion models (DMs) have recently emerged as powerful data-driven tools for this task, offering reconstruction fidelity and more scalable sampling supporting uncertainty quantification. However, DMs lack finite-sample guarantees against overconfident predictions, resulting in miscalibrated grid-point-level uncertainty estimates hindering their reliability in operational contexts. In this work, we tackle this issue by augmenting the downscaling pipeline with a conformal prediction framework. Specifically, the DM's samples are post-processed to derive conditional quantile estimates, incorporated into a conformalized quantile regression procedure targeting locally adaptive prediction intervals with finite-sample marginal validity. The proposed approach is evaluated on ERA5 reanalysis data over Italy, downscaled to a 2-km grid. Results demonstrate grid-point-level uncertainty estimates with markedly improved coverage and stable probabilistic scores relative to the DM baseline, highlighting the potential of conformalized generative models for more trustworthy probabilistic downscaling to high-resolution meteorological fields.

[**arXiv:2510.13301v1**](https://arxiv.org/abs/2510.13301v1)

**Tags:** ``

---

### A Stochastic Parameterization of Non-Orographic Gravity Waves Induced Mixing for Mars Planetary Climate Model

**Authors:** Jiandong Liu, Ehouarn Millour, François Forget, François Lott, Jean-Yves Chaufray

**Year:** 2025

**Abstract:**
> This paper presents a formalism of mixing induced by non-orographic gravity waves (GWs) to integrate with the stochastic GWs scheme in the Mars Planetary Climate Model. We derive the formalism of GWs and their mixing under the same assumptions, integrating the two schemes within a unified framework. Specifically, a surface-to-exosphere parameterization of GW-induced turbulence has been derived in terms of the eddy diffusion coefficient. Simulations show that the coefficient is on the order of 1E4 to 1E9 cm2 s-1 and a turbopause is at altitudes of 70 to 140 km, varying with seasons. The triggered mixing has minor effects on model temperatures, yet it substantially impacts upper atmospheric abundances. Simulations are consistent with observations from the Mars Climate Sounder and the Neutral Gas and Ion Mass Spectrometer. Mixing enhances the tracer transports in the middle and upper atmosphere, governing the dynamics of these regions. The scheme reveals how non-orographic GW-induced turbulence can regulate upper atmospheric processes, such as tracer escape.

[**arXiv:2510.20410v1**](https://arxiv.org/abs/2510.20410v1)

**Tags:** ``

---

### Continuous data assimilation applied to the Rayleigh-Benard problem for compressible fluid flows

**Authors:** Eduard Feireisl, Wladimir Neves

**Year:** 2025

**Abstract:**
> We apply a continuous data assimilation method to the Navier-Stokes-Fourier system governing the evolution of a compressible, rotating and thermally driven fluid. A rigorous proof of the tracking property is given in the asymptotic regime of low Mach and high Rossby and Froude numbers. Large data in the framework of weak solutions are considered.

[**arXiv:2510.20316v1**](https://arxiv.org/abs/2510.20316v1)

**Tags:** ``

---

### Sparse Local Implicit Image Function for sub-km Weather Downscaling

**Authors:** Yago del Valle Inclan Redondo, Enrique Arriaga-Varela, Dmitry Lyamzin, Pablo Cervantes, Tiago Ramalho

**Year:** 2025

**Abstract:**
> We introduce SpLIIF to generate implicit neural representations and enable arbitrary downscaling of weather variables. We train a model from sparse weather stations and topography over Japan and evaluate in- and out-of-distribution accuracy predicting temperature and wind, comparing it to both an interpolation baseline and CorrDiff. We find the model to be up to 50% better than both CorrDiff and the baseline at downscaling temperature, and around 10-20% better for wind.

[**arXiv:2510.20228v1**](https://arxiv.org/abs/2510.20228v1)

**Tags:** ``

---

### IEnSF: Iterative Ensemble Score Filter for Reducing Error in Posterior Score Estimation in Nonlinear Data Assimilation

**Authors:** Zezhong Zhang, Feng Bao, Guannan Zhang

**Year:** 2025

**Abstract:**
> The Ensemble Score Filter (EnSF) has emerged as a promising approach to leverage score-based diffusion models for solving high-dimensional and nonlinear data assimilation problems. While initial applications of EnSF to the Lorenz-96 model and the quasi-geostrophic system showed potential, the current method employs a heuristic weighted sum to combine the prior and the likelihood score functions. This introduces a structural error into the estimation of the posterior score function in the nonlinear setting. This work addresses this challenge by developing an iterative ensemble score filter (IEnSF) that applies an iterative algorithm as an outer loop around the reverse-time stochastic differential equation solver. When the state dynamics or the observation operator is nonlinear, the iterative algorithm can gradually reduce the posterior score estimation error by improving the accuracy of approximating the conditional expectation of the likelihood score function. The number of iterations required depends on the distance between the prior and posterior distributions and the nonlinearity of the observation operator. Numerical experiments demonstrate that the IEnSF algorithm substantially reduces the error in posterior score estimation in the nonlinear setting and thus improves the accuracy of tracking high-dimensional dynamical systems.

[**arXiv:2510.20159v1**](https://arxiv.org/abs/2510.20159v1)

**Tags:** ``

---

### Non-intrusive structural-preserving sequential data assimilation

**Authors:** Lizuo Liu, Tongtong Li, Anne Gelb

**Year:** 2025

**Abstract:**
> Data assimilation (DA) methods combine model predictions with observational data to improve state estimation in dynamical systems, inspiring their increasingly prominent role in geophysical and climate applications. Classical DA methods assume that the governing equations modeling the dynamics are known, which is unlikely for most real world applications. Machine learning (ML) provides a flexible alternative by learning surrogate models directly from data, but standard ML methods struggle in noisy and data-scarce environments, where meaningful extrapolation requires incorporating physical constraints. Recent advances in structure-preserving ML architectures, such as the development of the entropy-stable conservative flux form network (ESCFN), highlight the critical role of physical structure in improving learning stability and accuracy for unknown systems of conservation laws. Structural information has also been shown to improve DA performance. Gradient-based measures of spatial variability, in particular, can help refine ensemble updates in discontinuous systems. Motivated by both of these recent innovations, this investigation proposes a new non-intrusive, structure-preserving sequential data assimilation (NSSDA) framework that leverages structure at both the forecast and analysis stages. We use the ESCFN to construct a surrogate model to preserve physical laws during forecasting, and a structurally informed ensemble transform Kalman filter (SETKF) to embed local statistical structure into the assimilation step. Our method operates in a highly constrained environment, using only a single noisy trajectory for both training and assimilation. Numerical experiments where the unknown dynamics correspond respectively to the shallow water and Euler equations demonstrate significantly improved predictive accuracy.

[**arXiv:2510.19701v1**](https://arxiv.org/abs/2510.19701v1)

**Tags:** ``

---

### OmniCast: A Masked Latent Diffusion Model for Weather Forecasting Across Time Scales

**Authors:** Tung Nguyen, Tuan Pham, Troy Arcomano, Veerabhadra Kotamarthi, Ian Foster, Sandeep Madireddy, Aditya Grover

**Year:** 2025

**Abstract:**
> Accurate weather forecasting across time scales is critical for anticipating and mitigating the impacts of climate change. Recent data-driven methods based on deep learning have achieved significant success in the medium range, but struggle at longer subseasonal-to-seasonal (S2S) horizons due to error accumulation in their autoregressive approach. In this work, we propose OmniCast, a scalable and skillful probabilistic model that unifies weather forecasting across timescales. OmniCast consists of two components: a VAE model that encodes raw weather data into a continuous, lower-dimensional latent space, and a diffusion-based transformer model that generates a sequence of future latent tokens given the initial conditioning tokens. During training, we mask random future tokens and train the transformer to estimate their distribution given conditioning and visible tokens using a per-token diffusion head. During inference, the transformer generates the full sequence of future tokens by iteratively unmasking random subsets of tokens. This joint sampling across space and time mitigates compounding errors from autoregressive approaches. The low-dimensional latent space enables modeling long sequences of future latent states, allowing the transformer to learn weather dynamics beyond initial conditions. OmniCast performs competitively with leading probabilistic methods at the medium-range timescale while being 10x to 20x faster, and achieves state-of-the-art performance at the subseasonal-to-seasonal scale across accuracy, physics-based, and probabilistic metrics. Furthermore, we demonstrate that OmniCast can generate stable rollouts up to 100 years ahead. Code and model checkpoints are available at https://github.com/tung-nd/omnicast.

[**arXiv:2510.18707v1**](https://arxiv.org/abs/2510.18707v1)

**Tags:** ``

---

### Uncertainty-aware data assimilation through variational inference

**Authors:** Anthony Frion, David S Greenberg

**Year:** 2025

**Abstract:**
> Data assimilation, consisting in the combination of a dynamical model with a set of noisy and incomplete observations in order to infer the state of a system over time, involves uncertainty in most settings. Building upon an existing deterministic machine learning approach, we propose a variational inference-based extension in which the predicted state follows a multivariate Gaussian distribution. Using the chaotic Lorenz-96 dynamics as a testing ground, we show that our new model enables to obtain nearly perfectly calibrated predictions, and can be integrated in a wider variational data assimilation pipeline in order to achieve greater benefit from increasing lengths of data assimilation windows. Our code is available at https://github.com/anthony-frion/Stochastic_CODA.

[**arXiv:2510.17268v1**](https://arxiv.org/abs/2510.17268v1)

**Tags:** ``

---

### COBASE: A new copula-based shuffling method for ensemble weather forecast postprocessing

**Authors:** Maurits Flos, Bastien François, Irene Schicker, Kirien Whan, Elisa Perrone

**Year:** 2025

**Abstract:**
> Weather predictions are often provided as ensembles generated by repeated runs of numerical weather prediction models. These forecasts typically exhibit bias and inaccurate dependence structures due to numerical and dispersion errors, requiring statistical postprocessing for improved precision. A common correction strategy is the two-step approach: first adjusting the univariate forecasts, then reconstructing the multivariate dependence. The second step is usually handled with nonparametric methods, which can underperform when historical data are limited. Parametric alternatives, such as the Gaussian Copula Approach (GCA), offer theoretical advantages but often produce poorly calibrated multivariate forecasts due to random sampling of the corrected univariate margins. In this work, we introduce COBASE, a novel copula-based postprocessing framework that preserves the flexibility of parametric modeling while mimicking the nonparametric techniques through a rank-shuffling mechanism. This design ensures calibrated margins and realistic dependence reconstruction. We evaluate COBASE on multi-site 2-meter temperature forecasts from the ALADIN-LAEF ensemble over Austria and on joint forecasts of temperature and dew point temperature from the ECMWF system in the Netherlands. Across all regions, COBASE variants consistently outperform traditional copula-based approaches, such as GCA, and achieve performance on par with state-of-the-art nonparametric methods like SimSchaake and ECC, with only minimal differences across settings. These results position COBASE as a competitive and robust alternative for multivariate ensemble postprocessing, offering a principled bridge between parametric and nonparametric dependence reconstruction.

[**arXiv:2510.25610v1**](https://arxiv.org/abs/2510.25610v1)

**Tags:** ``

---

### Interpolated Discrepancy Data Assimilation for PDEs with Sparse Observations

**Authors:** Tong Wu, Humberto Godinez, Vitaliy Gyrya, James M. Hyman

**Year:** 2025

**Abstract:**
> Sparse sensor networks in weather and ocean modeling observe only a small fraction of the system state, which destabilizes standard nudging-based data assimilation. We introduce Interpolated Discrepancy Data Assimilation (IDDA), which modifies how discrepancies enter the governing equations. Rather than adding observations as a forcing term alone, IDDA also adjusts the nonlinear operator using interpolated observational information. This structural change suppresses error amplification when nonlinear effects dominate. We prove exponential convergence under explicit conditions linking error decay to observation spacing, nudging strength, and diffusion coefficient. The key requirement establishes bounds on nudging strength relative to observation spacing and diffusion, giving practitioners a clear operating window. When observations resolve the relevant scales, error decays at a user-specified rate. Critically, the error bound scales with the square of observation spacing rather than through hard-to-estimate nonlinear growth rates. We validate IDDA on Burgers flow, Kuramoto-Sivashinsky dynamics, and two-dimensional Navier-Stokes turbulence. Across these tests, IDDA reaches target accuracy faster than standard interpolated nudging, remains stable in chaotic regimes, avoids non-monotone transients, and requires minimal parameter tuning. Because IDDA uses standard explicit time integration, it fits readily into existing simulation pipelines without specialized solvers. These properties make IDDA a practical upgrade for operational systems constrained by sparse sensor coverage.

[**arXiv:2510.24944v1**](https://arxiv.org/abs/2510.24944v1)

**Tags:** ``

---

### High-Quality and Large-Scale Image Downscaling for Modern Display Devices

**Authors:** Suvrojit Mitra, G B Kevin Arjun, Sanjay Ghosh

**Year:** 2025

**Abstract:**
> In modern display technology and visualization tools, downscaling images is one of the most important activities. This procedure aims to maintain both visual authenticity and structural integrity while reducing the dimensions of an image at a large scale to fit the dimension of the display devices. In this study, we proposed a new technique for downscaling images that uses co-occurrence learning to maintain structural and perceptual information while reducing resolution. The technique uses the input image to create a data-driven co-occurrence profile that captures the frequency of intensity correlations in nearby neighborhoods. A refined filtering process is guided by this profile, which acts as a content-adaptive range kernel. The contribution of each input pixel is based on how closely it resembles pair-wise intensity values with it's neighbors. We validate our proposed technique on four datasets: DIV2K, BSD100, Urban100, and RealSR to show its effective downscaling capacity. Our technique could obtain up to 39.22 dB PSNR on the DIV2K dataset and PIQE up to 26.35 on the same dataset when downscaling by 8x and 16x, respectively. Numerous experimental findings attest to the ability of the suggested picture downscaling method to outperform more contemporary approaches in terms of both visual quality and performance measures. Unlike most existing methods, which did not focus on the large-scale image resizing scenario, we achieve high-quality downscaled images without texture loss or edge blurring. Our method, LSID (large scale image downscaling), successfully preserves high-frequency structures like edges, textures, and repeating patterns by focusing on statistically consistent pixels while reducing aliasing and blurring artifacts that are typical of traditional downscaling techniques.

[**arXiv:2510.24334v1**](https://arxiv.org/abs/2510.24334v1)

**Tags:** ``

---

### A PDE-Informed Latent Diffusion Model for 2-m Temperature Downscaling

**Authors:** Paul Rosu, Muchang Bahng, Erick Jiang, Rico Zhu, Vahid Tarokh

**Year:** 2025

**Abstract:**
> This work presents a physics-conditioned latent diffusion model tailored for dynamical downscaling of atmospheric data, with a focus on reconstructing high-resolution 2-m temperature fields. Building upon a pre-existing diffusion architecture and employing a residual formulation against a reference UNet, we integrate a partial differential equation (PDE) loss term into the model's training objective. The PDE loss is computed in the full resolution (pixel) space by decoding the latent representation and is designed to enforce physical consistency through a finite-difference approximation of an effective advection-diffusion balance. Empirical observations indicate that conventional diffusion training already yields low PDE residuals, and we investigate how fine-tuning with this additional loss further regularizes the model and enhances the physical plausibility of the generated fields. The entirety of our codebase is available on Github, for future reference and development.

[**arXiv:2510.23866v1**](https://arxiv.org/abs/2510.23866v1)

**Tags:** ``

---

### Revealing the Potential of Learnable Perturbation Ensemble Forecast Model for Tropical Cyclone Prediction

**Authors:** Jun Liu, Tao Zhou, Jiarui Li, Xiaohui Zhong, Peng Zhang, Jie Feng, Lei Chen, Hao Li

**Year:** 2025

**Abstract:**
> Tropical cyclones (TCs) are highly destructive and inherently uncertain weather systems. Ensemble forecasting helps quantify these uncertainties, yet traditional systems are constrained by high computational costs and limited capability to fully represent atmospheric nonlinearity. FuXi-ENS introduces a learnable perturbation scheme for ensemble generation, representing a novel AI-based forecasting paradigm. Here, we systematically compare FuXi-ENS with ECMWF-ENS using all 90 global TCs in 2018, examining their performance in TC-related physical variables, track and intensity forecasts, and the associated dynamical and thermodynamical fields. FuXi-ENS demonstrates clear advantages in predicting TC-related physical variables, and achieves more accurate track forecasts with reduced ensemble spread, though it still underestimates intensity relative to observations. Further dynamical and thermodynamical analyses reveal that FuXi-ENS better captures large-scale circulation, with moisture turbulent energy more tightly concentrated around the TC warm core, whereas ECMWF-ENS exhibits a more dispersed distribution. These findings highlight the potential of learnable perturbations to improve TC forecasting skill and provide valuable insights for advancing AI-based ensemble prediction of extreme weather events that have significant societal impacts.

[**arXiv:2510.23794v1**](https://arxiv.org/abs/2510.23794v1)

**Tags:** ``

---

### LO-SDA: Latent Optimization for Score-based Atmospheric Data Assimilation

**Authors:** Jing-An Sun, Hang Fan, Junchao Gong, Ben Fei, Kun Chen, Fenghua Ling, Wenlong Zhang, Wanghan Xu, Li Yan, Pierre Gentine, Lei Bai

**Year:** 2025

**Abstract:**
> Data assimilation (DA) plays a pivotal role in numerical weather prediction by systematically integrating sparse observations with model forecasts to estimate optimal atmospheric initial condition for forthcoming forecasts. Traditional Bayesian DA methods adopt a Gaussian background prior as a practical compromise for the curse of dimensionality in atmospheric systems, that simplifies the nonlinear nature of atmospheric dynamics and can result in biased estimates. To address this limitation, we propose a novel generative DA method, LO-SDA. First, a variational autoencoder is trained to learn compact latent representations that disentangle complex atmospheric correlations. Within this latent space, a background-conditioned diffusion model is employed to directly learn the conditional distribution from data, thereby generalizing and removing assumptions in the Gaussian prior in traditional DA methods. Most importantly, we introduce latent optimization during the reverse process of the diffusion model to ensure strict consistency between the generated states and sparse observations. Idealized experiments demonstrate that LO-SDA not only outperforms score-based DA methods based on diffusion posterior sampling but also surpasses traditional DA approaches. To our knowledge, this is the first time that a diffusion-based DA method demonstrates the potential to outperform traditional approaches on high-dimensional global atmospheric systems. These findings suggest that long-standing reliance on Gaussian priors-a foundational assumption in operational atmospheric DA-may no longer be necessary in light of advances in generative modeling.

[**arXiv:2510.22562v1**](https://arxiv.org/abs/2510.22562v1)

**Tags:** ``

---

### Structure Aware Image Downscaling

**Authors:** G B Kevin Arjun, Suvrojit Mitra, Sanjay Ghosh

**Year:** 2025

**Abstract:**
> Image downscaling is one of the key operations in recent display technology and visualization tools. By this process, the dimension of an image is reduced, aiming to preserve structural integrity and visual fidelity. In this paper, we propose a new image downscaling method which is built on the core ideas of image filtering and edge detection. In particular, we present a structure-informed downscaling algorithm that maintains fine details through edge-aware processing. The proposed method comprises three steps: (i) edge map computation, (ii) edge-guided interpolation, and (iii) texture enhancement. To faithfully retain the strong structures in an image, we first compute the edge maps by applying an efficient edge detection operator. This is followed by an edge-guided interpolation to preserve fine details after resizing. Finally, we fuse local texture enriched component of the original image to the interpolated one to restore high-frequency information. By integrating edge information with adaptive filtering, our approach effectively minimizes artifacts while retaining crucial image features. To demonstrate the effective downscaling capability of our proposed method, we validate on four datasets: DIV2K, BSD100, Urban100, and RealSR. For downscaling by 4x, our method could achieve as high as 39.07 dB PSNR on the DIV2K dataset and 38.71 dB on the RealSR dataset. Extensive experimental results confirm that the proposed image downscaling method is capable of achieving superior performance in terms of both visual quality and performance metrics with reference to recent methods. Most importantly, the downscaled images by our method do not suffer from edge blurring and texture loss, unlike many existing ones.

[**arXiv:2510.22551v1**](https://arxiv.org/abs/2510.22551v1)

**Tags:** ``

---

### Nowcast3D: Reliable precipitation nowcasting via gray-box learning

**Authors:** Huaguan Chen, Wei Han, Haofei Sun, Ning Lin, Xingtao Song, Yunfan Yang, Jie Tian, Yang Liu, Ji-Rong Wen, Xiaoye Zhang, Xueshun Shen, Hao Sun

**Year:** 2025

**Abstract:**
> Extreme precipitation nowcasting demands high spatiotemporal fidelity and extended lead times, yet existing approaches remain limited. Numerical Weather Prediction (NWP) and its deep-learning emulations are too slow and coarse for rapidly evolving convection, while extrapolation and purely data-driven models suffer from error accumulation and excessive smoothing. Hybrid 2D radar-based methods discard crucial vertical information, preventing accurate reconstruction of height-dependent dynamics. We introduce a gray-box, fully three-dimensional nowcasting framework that directly processes volumetric radar reflectivity and couples physically constrained neural operators with datadriven learning. The model learns vertically varying 3D advection fields under a conservative advection operator, parameterizes spatially varying diffusion, and introduces a Brownian-motion--inspired stochastic term to represent unresolved motions. A residual branch captures small-scale convective initiation and microphysical variability, while a diffusion-based stochastic module estimates uncertainty. The framework achieves more accurate forecasts up to three-hour lead time across precipitation regimes and ranked first in 57\% of cases in a blind evaluation by 160 meteorologists. By restoring full 3D dynamics with physical consistency, it offers a scalable and robust pathway for skillful and reliable nowcasting of extreme precipitation.

[**arXiv:2511.04659v1**](https://arxiv.org/abs/2511.04659v1)

**Tags:** ``

---

### Deep Learning-Driven Downscaling for Climate Risk Assessment of Projected Temperature Extremes in the Nordic Region

**Authors:** Parthiban Loganathan, Elias Zea, Ricardo Vinuesa, Evelyn Otero

**Year:** 2025

**Abstract:**
> Rapid changes and increasing climatic variability across the widely varied Koppen-Geiger regions of northern Europe generate significant needs for adaptation. Regional planning needs high-resolution projected temperatures. This work presents an integrative downscaling framework that incorporates Vision Transformer (ViT), Convolutional Long Short-Term Memory (ConvLSTM), and Geospatial Spatiotemporal Transformer with Attention and Imbalance-Aware Network (GeoStaNet) models. The framework is evaluated with a multicriteria decision system, Deep Learning-TOPSIS (DL-TOPSIS), for ten strategically chosen meteorological stations encompassing the temperate oceanic (Cfb), subpolar oceanic (Cfc), warm-summer continental (Dfb), and subarctic (Dfc) climate regions. Norwegian Earth System Model (NorESM2-LM) Coupled Model Intercomparison Project Phase 6 (CMIP6) outputs were bias-corrected during the 1951-2014 period and subsequently validated against earlier observations of day-to-day temperature metrics and diurnal range statistics. The ViT showed improved performance (Root Mean Squared Error (RMSE): 1.01 degrees C; R^2: 0.92), allowing for production of credible downscaled projections. Under the SSP5-8.5 scenario, the Dfc and Dfb climate zones are projected to warm by 4.8 degrees C and 3.9 degrees C, respectively, by 2100, with expansion in the diurnal temperature range by more than 1.5 degrees C. The Time of Emergence signal first appears in subarctic winter seasons (Dfc: approximately 2032), signifying an urgent need for adaptation measures. The presented framework offers station-based, high-resolution estimates of uncertainties and extremes, with direct uses for adaptation policy over high-latitude regions with fast environmental change.

[**arXiv:2511.03770v1**](https://arxiv.org/abs/2511.03770v1)

**Tags:** ``

---

### A Probabilistic U-Net Approach to Downscaling Climate Simulations

**Authors:** Maryam Alipourhajiagha, Pierre-Louis Lemaire, Youssef Diouane, Julie Carreau

**Year:** 2025

**Abstract:**
> Climate models are limited by heavy computational costs, often producing outputs at coarse spatial resolutions, while many climate change impact studies require finer scales. Statistical downscaling bridges this gap, and we adapt the probabilistic U-Net for this task, combining a deterministic U-Net backbone with a variational latent space to capture aleatoric uncertainty. We evaluate four training objectives, afCRPS and WMSE-MS-SSIM with three settings for downscaling precipitation and temperature from $16\times$ coarser resolution. Our main finding is that WMSE-MS-SSIM performs well for extremes under certain settings, whereas afCRPS better captures spatial variability across scales.

[**arXiv:2511.03197v1**](https://arxiv.org/abs/2511.03197v1)

**Tags:** ``

---

### Intercomparison of a High-Resolution Regional Climate Model Ensemble for Catchment-Scale Water Cycle Processes under Human Influence

**Authors:** J. L. Roque, F. Da Silva Lopes, J. A. Giles, B. D. Gutknecht, B. Schalge, Y. Zhang, M. Ferro, P. Friederichs, K. Goergen, S. Poll, A. Valmassoi

**Year:** 2025

**Abstract:**
> Understanding regional hydroclimatic variability and its drivers is essential for anticipating the impacts of climate change on water resources and sustainability. Yet, considerable uncertainty remains in the simulation of the coupled land atmosphere water and energy cycles, largely due to structural model limitations, simplified process representations, and insufficient spatial resolution. Within the framework of the Collaborative Research Center 1502 DETECT, this study presents a coordinated intercomparison of regional climate model simulations designed for water cycle process analysis over Europe. We analyze the performance of simulations using the ICON and TSMP1 model systems and covering the period from 1990 to 2020, comparing against reference datasets (E-OBS, GPCC, and GLEAM). We focus on 2 m air temperature, precipitation and evapotranspiration over four representative basins, the Ebro, Po, Rhine, and Tisa, within the EURO CORDEX domain.   Our analysis reveals systematic cold biases across all basins and seasons, with ICON generally outperforming TSMP1. Precipitation biases exhibit substantial spread, particularly in summer, reflecting the persistent challenge of accurately simulating precipitation. ICON tends to underestimate evapotranspiration, while TSMP1 performs better some seasons. Sensitivity experiments further indicate that the inclusion of irrigation improves simulation performance in the Po basin, which is intensively irrigated, and that higher-resolution sea surface temperature forcing data improves the overall precipitation representation. This baseline evaluation provides a first assessment of the DETECT multimodel ensemble and highlights key structural differences influencing model skill across hydroclimatic regimes.

[**arXiv:2511.02799v1**](https://arxiv.org/abs/2511.02799v1)

**Tags:** ``

---

### Reinforcement learning based data assimilation for unknown state model

**Authors:** Ziyi Wang, Lijian Jiang

**Year:** 2025

**Abstract:**
> Data assimilation (DA) has increasingly emerged as a critical tool for state estimation   across a wide range of applications. It is signiffcantly challenging when the governing equations of the underlying dynamics are unknown. To this end, various machine learning approaches have been employed to construct a surrogate state transition   model in a supervised learning framework, which relies on pre-computed training   datasets. However, it is often infeasible to obtain noise-free ground-truth state sequences in practice. To address this challenge, we propose a novel method that integrates reinforcement learning with ensemble-based Bayesian ffltering methods, enabling   the learning of surrogate state transition model for unknown dynamics directly from noisy observations, without using true state trajectories. Speciffcally, we treat the process for computing maximum likelihood estimation of surrogate model parameters   as a sequential decision-making problem, which can be formulated as a discretetime   Markov decision process (MDP). Under this formulation, learning the surrogate transition model is equivalent to ffnding an optimal policy of the MDP, which can be effectively addressed using reinforcement learning techniques. Once the model is trained offfine, state estimation can be performed in the online stage using ffltering methods based on the learned dynamics. The proposed framework accommodates a wide range of observation scenarios, including nonlinear and partially observed measurement   models. A few numerical examples demonstrate that the proposed method achieves superior accuracy and robustness in high-dimensional settings.

[**arXiv:2511.02286v1**](https://arxiv.org/abs/2511.02286v1)

**Tags:** ``

---

### Learned Adaptive Kernels for High-Fidelity Image Downscaling

**Authors:** Piyush Narhari Pise, Sanjay Ghosh

**Year:** 2025

**Abstract:**
> Image downscaling is a fundamental operation in image processing, crucial for adapting high-resolution content to various display and storage constraints. While classic methods often introduce blurring or aliasing, recent learning-based approaches offer improved adaptivity. However, achieving maximal fidelity against ground-truth low-resolution (LR) images, particularly by accounting for channel-specific characteristics, remains an open challenge. This paper introduces ADK-Net (Adaptive Downscaling Kernel Network), a novel deep convolutional neural network framework for high-fidelity supervised image downscaling. ADK-Net explicitly addresses channel interdependencies by learning to predict spatially-varying, adaptive resampling kernels independently for each pixel and uniquely for each color channel (RGB). The architecture employs a hierarchical design featuring a ResNet-based feature extractor and parallel channel-specific kernel generators, themselves composed of ResNet-based trunk and branch sub-modules, enabling fine-grained kernel prediction. Trained end-to-end using an L1 reconstruction loss against ground-truth LR data, ADK-Net effectively learns the target downscaling transformation. Extensive quantitative and qualitative experiments on standard benchmarks, including the RealSR dataset, demonstrate that ADK-Net establishes a new state-of-the-art in supervised image downscaling, yielding significant improvements in PSNR and SSIM metrics compared to existing learning-based and traditional methods.

[**arXiv:2511.01620v1**](https://arxiv.org/abs/2511.01620v1)

**Tags:** ``

---

### DAMBench: A Multi-Modal Benchmark for Deep Learning-based Atmospheric Data Assimilation

**Authors:** Hao Wang, Zixuan Weng, Jindong Han, Wei Fan, Hao Liu

**Year:** 2025

**Abstract:**
> Data Assimilation is a cornerstone of atmospheric system modeling, tasked with reconstructing system states by integrating sparse, noisy observations with prior estimation. While traditional approaches like variational and ensemble Kalman filtering have proven effective, recent advances in deep learning offer more scalable, efficient, and flexible alternatives better suited for complex, real-world data assimilation involving large-scale and multi-modal observations. However, existing deep learning-based DA research suffers from two critical limitations: (1) reliance on oversimplified scenarios with synthetically perturbed observations, and (2) the absence of standardized benchmarks for fair model comparison. To address these gaps, in this work, we introduce DAMBench, the first large-scale multi-modal benchmark designed to evaluate data-driven DA models under realistic atmospheric conditions. DAMBench integrates high-quality background states from state-of-the-art forecasting systems and real-world multi-modal observations (i.e., real-world weather stations and satellite imagery). All data are resampled to a common grid and temporally aligned to support systematic training, validation, and testing. We provide unified evaluation protocols and benchmark representative data assimilation approaches, including latent generative models and neural process frameworks. Additionally, we propose a lightweight multi-modal plugin to demonstrate how integrating realistic observations can enhance even simple baselines. Through comprehensive experiments, DAMBench establishes a rigorous foundation for future research, promoting reproducibility, fair comparison, and extensibility to real-world multi-modal scenarios. Our dataset and code are publicly available at https://github.com/figerhaowang/DAMBench.

[**arXiv:2511.01468v1**](https://arxiv.org/abs/2511.01468v1)

**Tags:** ``

---

### The Role of Deep Mesoscale Eddies in Ensemble Forecast Performance

**Authors:** Justin Cooke, Kathleen Donohue, Clark D Rowley, Prasad G Thoppil, D Randolph Watts

**Year:** 2025

**Abstract:**
> Present forecasting efforts rely on assimilation of observational data captured in the upper ocean (< 1000 m depth). These observations constrain the upper ocean and minimally influence the deep ocean. Nevertheless, development of the full water column circulation critically depends upon the dynamical interactions between upper and deep fields. Forecasts demonstrate that the initialization of the deep field is influential for the development and evolution of the surface in the forecast. Deep initial conditions that better agree with observations have lower upper ocean uncertainty as the forecast progresses. Here, best and worst ensemble members in two 92-day forecasts are identified and contrasted in order to determine how the deep ocean differs between these groups. The forecasts cover the duration of the Loop Current Eddy Thor separation event, which coincides with available deep observations in the Gulf. Model member performance is assessed by comparing surface variables against verifying analysis and satellite altimeter data during the forecast time-period. Deep cyclonic and anticyclonic features are reviewed, and compared against deep observations, indicating subtle differences in locations of deep eddies at relevant times. These results highlight both the importance of deep circulation in the dynamics of the Loop Current system and more broadly motivate efforts to assimilate deep observations to better constrain the deep initial fields and improve surface predictions.

[**arXiv:2511.09747v1**](https://arxiv.org/abs/2511.09747v1)

**Tags:** ``

---

### FlowCast: Advancing Precipitation Nowcasting with Conditional Flow Matching

**Authors:** Bernardo Perrone Ribeiro, Jana Faganeli Pucer

**Year:** 2025

**Abstract:**
> Radar-based precipitation nowcasting, the task of forecasting short-term precipitation fields from previous radar images, is a critical problem for flood risk management and decision-making. While deep learning has substantially advanced this field, two challenges remain fundamental: the uncertainty of atmospheric dynamics and the efficient modeling of high-dimensional data. Diffusion models have shown strong promise by producing sharp, reliable forecasts, but their iterative sampling process is computationally prohibitive for time-critical applications. We introduce FlowCast, the first model to apply Conditional Flow Matching (CFM) to precipitation nowcasting. Unlike diffusion, CFM learns a direct noise-to-data mapping, enabling rapid, high-fidelity sample generation with drastically fewer function evaluations. Our experiments demonstrate that FlowCast establishes a new state-of-the-art in predictive accuracy. A direct comparison further reveals the CFM objective is both more accurate and significantly more efficient than a diffusion objective on the same architecture, maintaining high performance with significantly fewer sampling steps. This work positions CFM as a powerful and practical alternative for high-dimensional spatiotemporal forecasting.

[**arXiv:2511.09731v1**](https://arxiv.org/abs/2511.09731v1)

**Tags:** ``

---

### Physics-based localization methodology for Data Assimilation by Ensemble Kalman Filter

**Authors:** Sarp Er, Marcello Meldi

**Year:** 2025

**Abstract:**
> A physics-based methodology for the determination of the localization function for the Ensemble Kalman Filter (EnKF) is proposed. The spatial features of such function evolve dynamically over time according to the relevant instantaneous flow features of the ensemble members with the objective, to reduce the computational cost of the Data Assimilation (DA) procedure when applied with solvers for Computational Fluid Dynamics (CFD). The validation of the methodology has been carried out by the analysis of two test cases exhibiting different features. This permits to investigate different physical features, tailored for each test case, which affect the localization function. The flow over a two-dimensional square cylinder at $Re=150$ is the first case investigated. It has been shown that the proposed localization procedure leads to a more cost-effective DA process by reducing the size of the assimilated regions while keeping the same level of accuracy. The capabilities of the methodology are further demonstrated by the investigation of the turbulent flow around a three-dimensional circular cylinder for $Re=3900$. Again, the methodology exhibits an excellent trade off in terms of accuracy versus computational requirements.

[**arXiv:2511.08845v1**](https://arxiv.org/abs/2511.08845v1)

**Tags:** ``

---

### Recovering the Parameter $α$ in the Simplified Bardina Model through Continuous Data Assimilation

**Authors:** Débora A. F. Albanez, Maicon José Benvenutti, Jing Tian

**Year:** 2025

**Abstract:**
> In this study, we develop a continuous data assimilation algorithm to recover the parameter $α$ in the simplified Bardina model. Our method utilizes the observations of finitely many Fourier modes by using a nudging framework that involves recursive parameter updates. We provide a rigorous convergence analysis, showing that the approximated parameter approaches the true value under suitable conditions.

[**arXiv:2511.08421v1**](https://arxiv.org/abs/2511.08421v1)

**Tags:** ``

---

### Efficient Regional Storm Surge Surrogate Model Training Strategy Under Evolving Landscape and Climate Scenarios

**Authors:** Ziyue Liu, Mohammad Ahmadi Gharehtoragh, Brenna Kari Losch, David R. Johnson

**Year:** 2025

**Abstract:**
> Coastal communities can be exposed to risk from catastrophic storm-induced coastal hazards, causing major global losses each year. Recent advances in computational power have enabled the integration of machine learning (ML) into coastal hazard modeling, particularly for storm surge prediction. Given the potential variation in future climate and landscape conditions, efficient predictive models that can incorporate multiple future scenarios are needed. Existing studies built a framework for training ML models using storm surge simulation data under different potential future climate and landscape scenarios. However, storm surge simulation data under designed future scenarios require computationally expensive numerical simulations of synthetic storm suites over extensive geospatial grids. As the number of designed scenarios increases, the computational cost associated with both numerical simulation and ML training increases rapidly. This study introduces a cost-effective reduction strategy that incorporates new scenario data while minimizing computational burden. The approach reduces training data across three dimensions: (1) grid points, (2) input features, and (3) storm suite size. Reducing the storm suite size is especially effective in cutting simulation costs. Model performance was evaluated using different ML algorithms, showing consistent effectiveness. When trained on 5,000 of 80,000 grid points, 10 of 12 features, and 40 of 90 storms, the model achieved an R=0.93, comparable to that of models trained on the full dataset, with substantially lower computational expense.

[**arXiv:2511.07269v1**](https://arxiv.org/abs/2511.07269v1)

**Tags:** ``

---

### Multi-layer Stack Ensembles for Time Series Forecasting

**Authors:** Nathanael Bosch, Oleksandr Shchur, Nick Erickson, Michael Bohlke-Schneider, Caner Türkmen

**Year:** 2025

**Abstract:**
> Ensembling is a powerful technique for improving the accuracy of machine learning models, with methods like stacking achieving strong results in tabular tasks. In time series forecasting, however, ensemble methods remain underutilized, with simple linear combinations still considered state-of-the-art. In this paper, we systematically explore ensembling strategies for time series forecasting. We evaluate 33 ensemble models -- both existing and novel -- across 50 real-world datasets. Our results show that stacking consistently improves accuracy, though no single stacker performs best across all tasks. To address this, we propose a multi-layer stacking framework for time series forecasting, an approach that combines the strengths of different stacker models. We demonstrate that this method consistently provides superior accuracy across diverse forecasting scenarios. Our findings highlight the potential of stacking-based methods to improve AutoML systems for time series forecasting.

[**arXiv:2511.15350v1**](https://arxiv.org/abs/2511.15350v1)

**Tags:** ``

---

### Towards Streaming Prediction of Oscillatory Flows: A Data Assimilation and Machine Learning Approach

**Authors:** Miguel M. Valero, Marcello Meldi

**Year:** 2025

**Abstract:**
> Data-driven methods have demonstrated strong predictive capabilities in fluid mechanics, yet most current applications still focus on simplified configurations, often characterised by statistical stationarity or limited temporal variability. This work proposes a methodology that combines Data Assimilation (DA) and Machine Learning (ML) to predict flow configurations that exhibit cyclic behaviour over time. Starting from limited, sparse high-fidelity measurements and a low-fidelity numerical model, the DA approach performs data fusion to obtain complete and accurate flow state estimations in time. This complete dataset is used to train multiple ML tools, which are applied across different phases of the flow cycle to augment the model's predictions when high-fidelity data might not be available for the DA application. The methodology is applied to the analysis of an oscillating cylinder in a laminar regime using a sliding-window approach, in which separate models are trained for specific flow conditions to ensure each model specialises in flow dynamics representative of a phase of the oscillation period. This phase-resolved learning enables the efficient capture of transient features that would be challenging for a single global model. The results highlight the potential of this method to study complex flow configurations with oscillatory features in which neither the flow nor the cycle is known a priori, in particular by exploiting real-time training and updates, as is commonly done in digital twins, which require continuous model correction and adaptation.

[**arXiv:2511.15758v1**](https://arxiv.org/abs/2511.15758v1)

**Tags:** ``

---

### Bridging the Gap Between Bayesian Deep Learning and Ensemble Weather Forecasts

**Authors:** Xinlei Xiong, Wenbo Hu, Shuxun Zhou, Kaifeng Bi, Lingxi Xie, Ying Liu, Richang Hong, Qi Tian

**Year:** 2025

**Abstract:**
> Weather forecasting is fundamentally challenged by the chaotic nature of the atmosphere, necessitating probabilistic approaches to quantify uncertainty. While traditional ensemble prediction (EPS) addresses this through computationally intensive simulations, recent advances in Bayesian Deep Learning (BDL) offer a promising but often disconnected alternative. We bridge these paradigms through a unified hybrid Bayesian Deep Learning framework for ensemble weather forecasting that explicitly decomposes predictive uncertainty into epistemic and aleatoric components, learned via variational inference and a physics-informed stochastic perturbation scheme modeling flow-dependent atmospheric dynamics, respectively. We further establish a unified theoretical framework that rigorously connects BDL and EPS, providing formal theorems that decompose total predictive uncertainty into epistemic and aleatoric components under the hybrid BDL framework. We validate our framework on the large-scale 40-year ERA5 reanalysis dataset (1979-2019) with 0.25° spatial resolution. Experimental results show that our method not only improves forecast accuracy and yields better-calibrated uncertainty quantification but also achieves superior computational efficiency compared to state-of-the-art probabilistic diffusion models. We commit to making our code open-source upon acceptance of this paper.

[**arXiv:2511.14218v1**](https://arxiv.org/abs/2511.14218v1)

**Tags:** ``

---

### Weather Maps as Tokens: Transformers for Renewable Energy Forecasting

**Authors:** Federico Battini

**Year:** 2025

**Abstract:**
> Accurate renewable energy forecasting is essential to reduce dependence on fossil fuels and enabling grid decarbonization. However, current approaches fail to effectively integrate the rich spatial context of weather patterns with their temporal evolution. This work introduces a novel approach that treats weather maps as tokens in transformer sequences to predict renewable energy. Hourly weather maps are encoded as spatial tokens using a lightweight convolutional neural network, and then processed by a transformer to capture temporal dynamics across a 45-hour forecast horizon. Despite disadvantages in input initialization, evaluation against ENTSO-E operational forecasts shows a reduction in RMSE of about 60% and 20% for wind and solar respectively. A live dashboard showing daily forecasts is available at: https://www.sardiniaforecast.ifabfoundation.it.

[**arXiv:2511.13935v2**](https://arxiv.org/abs/2511.13935v2)

**Tags:** ``

---

### Exploring Ultra Rapid Data Assimilation Based on Ensemble Transform Kalman Filter with the Lorenz 96 Model

**Authors:** Fumitoshi Kawasaki, Atsushi Okazaki, Kenta Kurosawa, Shunji Kotsuki

**Year:** 2025

**Abstract:**
> To explore the effectiveness of ultra-rapid data assimilation (URDA) for numerical weather prediction (NWP), this study investigates the properties of URDA in nonlinear models and proposes technical treatments to enhance its performance. URDA rapidly updates preemptive forecasts derived from observations without integrating a dynamical model each time additional observations become available. First, we analytically demonstrate that the preemptive forecast obtained by URDA in nonlinear models is approximately equivalent to the forecast integrated from the analysis. Furthermore, numerical experiments are conducted with the 40-variable Lorenz 96 model. The results show that URDA in nonlinear models tends to exhibit deterioration of forecast accuracy and collapse of ensemble spread when preemptive forecasts are repeatedly updated or when the forecasts are extended over longer periods. Furthermore, the roles of inflation and localization, both essential technical treatments in NWP, are examined in the context of URDA. It is shown that although inflation and localization are essential to URDA, conventional inflation techniques are not suitable for it. Therefore, this study proposes new technical treatments for URDA, namely relaxation to baseline perturbations (RTBP) and relaxation to baseline forecast (RTBF). Applying RTBP and RTBF mitigates the difficulties associated with URDA and yields preemptive forecasts with higher accuracy than the baseline forecast. Consequently, URDA, particularly when combined with RTBP and RTBF, would stand as a step toward practical application in NWP.

[**arXiv:2511.12620v1**](https://arxiv.org/abs/2511.12620v1)

**Tags:** ``

---

### Causal Feature Selection for Weather-Driven Residential Load Forecasting

**Authors:** Elise Zhang, François Mirallès, Stéphane Dellacherie, Di Wu, Benoit Boulet

**Year:** 2025

**Abstract:**
> Weather is a dominant external driver of residential electricity demand, but adding many meteorological covariates can inflate model complexity and may even impair accuracy. Selecting appropriate exogenous features is non-trivial and calls for a principled selection framework, given the direct operational implications for day-to-day planning and reliability. This work investigates whether causal feature selection can retain the most informative weather drivers while improving parsimony and robustness for short-term load forecasting. We present a case study on Southern Ontario with two open-source datasets: (i) IESO hourly electricity consumption by Forward Sortation Areas; (ii) ERA5 weather reanalysis data. We compare different feature selection regimes (no feature selection, non-causal selection, PCMCI-causal selection) on city-level forecasting with three different time series forecasting models: GRU, TCN, PatchTST. In the feature analysis, non-causal selection prioritizes radiation and moisture variables that show correlational dependence, whereas PCMCI-causal selection emphasizes more direct thermal drivers and prunes the indirect covariates. We detail the evaluation pipeline and report diagnostics on prediction accuracy and extreme-weather robustness, positioning causal feature selection as a practical complement to modern forecasters when integrating weather into residential load forecasting.

[**arXiv:2511.20508v1**](https://arxiv.org/abs/2511.20508v1)

**Tags:** ``

---

### Cyclical Temporal Encoding and Hybrid Deep Ensembles for Multistep Energy Forecasting

**Authors:** Salim Khazem, Houssam Kanso

**Year:** 2025

**Abstract:**
> Accurate electricity consumption forecasting is essential for demand management and smart grid operations. This paper introduces a unified deep learning framework that integrates cyclical temporal encoding with hybrid LSTM-CNN architectures to enhance multistep energy forecasting. We systematically transform calendar-based attributes using sine cosine encodings to preserve periodic structure and evaluate their predictive relevance through correlation analysis. To exploit both long-term seasonal effects and short-term local patterns, we employ an ensemble model composed of an LSTM, a CNN, and a meta-learner of MLP regressors specialized for each forecast horizon. Using a one year national consumption dataset, we conduct an extensive experimental study including ablation analyses with and without cyclical encodings and calendar features and comparisons with established baselines from the literature. Results demonstrate consistent improvements across all seven forecast horizons, with our hybrid model achieving lower RMSE and MAE than individual architectures and prior methods. These findings confirm the benefit of combining cyclical temporal representations with complementary deep learning structures. To our knowledge, this is the first work to jointly evaluate temporal encodings, calendar-based features, and hybrid ensemble architectures within a unified short-term energy forecasting framework.

[**arXiv:2512.03656v1**](https://arxiv.org/abs/2512.03656v1)

**Tags:** ``

---

### The promising potential of vision language models for the generation of textual weather forecasts

**Authors:** Edward C. C. Steele, Dinesh Mane, Emilio Monti, Luis Orus, Rebecca Chantrill-Cheyette, Matthew Couch, Kirstine I. Dale, Simon Eaton, Govindarajan Rangarajan, Amir Majlesi, Steven Ramsdale, Michael Sharpe, Craig Smith, Jonathan Smith, Rebecca Yates, Holly Ellis, Charles Ewen

**Year:** 2025

**Abstract:**
> Despite the promising capability of multimodal foundation models, their application to the generation of meteorological products and services remains nascent. To accelerate aspiration and adoption, we explore the novel use of a vision language model for writing the iconic Shipping Forecast text directly from video-encoded gridded weather data. These early results demonstrate promising scalable technological opportunities for enhancing production efficiency and service innovation within the weather enterprise and beyond.

[**arXiv:2512.03623v1**](https://arxiv.org/abs/2512.03623v1)

**Tags:** ``

---

### Observation-driven correction of numerical weather prediction for marine winds

**Authors:** Matteo Peduto, Qidong Yang, Jonathan Giezendanner, Devis Tuia, Sherrie Wang

**Year:** 2025

**Abstract:**
> Accurate marine wind forecasts are essential for safe navigation, ship routing, and energy operations, yet they remain challenging because observations over the ocean are sparse, heterogeneous, and temporally variable. We reformulate wind forecasting as observation-informed correction of a global numerical weather prediction (NWP) model. Rather than forecasting winds directly, we learn local correction patterns by assimilating the latest in-situ observations to adjust the Global Forecast System (GFS) output. We propose a transformer-based deep learning architecture that (i) handles irregular and time-varying observation sets through masking and set-based attention mechanisms, (ii) conditions predictions on recent observation-forecast pairs via cross-attention, and (iii) employs cyclical time embeddings and coordinate-aware location representations to enable single-pass inference at arbitrary spatial coordinates. We evaluate our model over the Atlantic Ocean using observations from the International Comprehensive Ocean-Atmosphere Data Set (ICOADS) as reference. The model reduces GFS 10-meter wind RMSE at all lead times up to 48 hours, achieving 45% improvement at 1-hour lead time and 13% improvement at 48-hour lead time. Spatial analyses reveal the most persistent improvements along coastlines and shipping routes, where observations are most abundant. The tokenized architecture naturally accommodates heterogeneous observing platforms (ships, buoys, tide gauges, and coastal stations) and produces both site-specific predictions and basin-scale gridded products in a single forward pass. These results demonstrate a practical, low-latency post-processing approach that complements NWP by learning to correct systematic forecast errors.

[**arXiv:2512.03606v1**](https://arxiv.org/abs/2512.03606v1)

**Tags:** ``

---

### EcoCast: A Spatio-Temporal Model for Continual Biodiversity and Climate Risk Forecasting

**Authors:** Hammed A. Akande, Abdulrauf A. Gidado

**Year:** 2025

**Abstract:**
> Increasing climate change and habitat loss are driving unprecedented shifts in species distributions. Conservation professionals urgently need timely, high-resolution predictions of biodiversity risks, especially in ecologically diverse regions like Africa. We propose EcoCast, a spatio-temporal model designed for continual biodiversity and climate risk forecasting. Utilizing multisource satellite imagery, climate data, and citizen science occurrence records, EcoCast predicts near-term (monthly to seasonal) shifts in species distributions through sequence-based transformers that model spatio-temporal environmental dependencies. The architecture is designed with support for continual learning to enable future operational deployment with new data streams. Our pilot study in Africa shows promising improvements in forecasting distributions of selected bird species compared to a Random Forest baseline, highlighting EcoCast's potential to inform targeted conservation policies. By demonstrating an end-to-end pipeline from multi-modal data ingestion to operational forecasting, EcoCast bridges the gap between cutting-edge machine learning and biodiversity management, ultimately guiding data-driven strategies for climate resilience and ecosystem conservation throughout Africa.

[**arXiv:2512.02260v1**](https://arxiv.org/abs/2512.02260v1)

**Tags:** ``

---

### Are we misdiagnosing ensemble forecast reliability? On the insufficiency of Spread-Error and rank-based reliability metrics

**Authors:** Arlan Dirkson, Mark Buehner

**Year:** 2025

**Abstract:**
> It has been documented that Spread-Error equality and a flat rank histogram are necessary but insufficient for demonstrating ensemble forecast reliability. Nevertheless, these metrics are heavily relied upon, both in the literature and at operational numerical weather prediction centers. In this study, we demonstrate theoretically why the Spread-Error relationship is necessary but insufficient for diagnosing reliability up to second order, even when mean bias is absent or accounted for. Assuming joint normality between ensemble members and the reference truth, we further show with idealized experiments that the same covariance structure responsible for this insufficiency also produces false diagnoses of reliability with the rank histogram and the reliability component of the continuous rank probability score. Under this structure and when the ensemble mean is meaningfully different from climatology, the truth lies among the least (most) extreme members when climatological variance is excessive (deficient) in each member. Importantly, this behavior is also shown to be plausible in operational ensemble weather forecasts. Combining these results with calibration principles from statistical postprocessing leads us to conclude that both perfect dispersion and underdispersion are ill-defined. When diagnostics are misinterpreted as indicating the latter, improper tuning can lead to further deterioration of forecast quality, even while improving Spread-Error and rank histogram behavior. To address these issues, we propose a new reliability diagnostic based on three easily computed statistics, motivated by the structure of the joint distribution of ensemble members and the reference truth up to second order. The diagnostic separates contributions to unreliability originating from climatology and predictability, enabling a more precise and robust characterization of ensemble behavior.

[**arXiv:2512.02160v1**](https://arxiv.org/abs/2512.02160v1)

**Tags:** ``

---

### On Global Applicability and Location Transferability of Generative Deep Learning Models for Precipitation Downscaling

**Authors:** Paula Harder, Christian Lessig, Matthew Chantry, Francis Pelletier, David Rolnick

**Year:** 2025

**Abstract:**
> Deep learning offers promising capabilities for the statistical downscaling of climate and weather forecasts, with generative approaches showing particular success in capturing fine-scale precipitation patterns. However, most existing models are region-specific, and their ability to generalize to unseen geographic areas remains largely unexplored. In this study, we evaluate the generalization performance of generative downscaling models across diverse regions. Using a global framework, we employ ERA5 reanalysis data as predictors and IMERG precipitation estimates at $0.1^\circ$ resolution as targets. A hierarchical location-based data split enables a systematic assessment of model performance across 15 regions around the world.

[**arXiv:2512.01400v1**](https://arxiv.org/abs/2512.01400v1)

**Tags:** ``

---

### Data assimilation and discrepancy modeling with shallow recurrent decoders

**Authors:** Yuxuan Bao, J. Nathan Kutz

**Year:** 2025

**Abstract:**
> The requirements of modern sensing are rapidly evolving, driven by increasing demands for data efficiency, real-time processing, and deployment under limited sensing coverage. Complex physical systems are often characterized through the integration of a limited number of point sensors in combination with scientific computations which approximate the dominant, full-state dynamics. Simulation models, however, inevitably neglect small-scale or hidden processes, are sensitive to perturbations, or oversimplify parameter correlations, leading to reconstructions that often diverge from the reality measured by sensors. This creates a critical need for data assimilation, the process of integrating observational data with predictive simulation models to produce coherent and accurate estimates of the full state of complex physical systems. We propose a machine learning framework for Data Assimilation with a SHallow REcurrent Decoder (DA-SHRED) which bridges the simulation-to-real (SIM2REAL) gap between computational modeling and experimental sensor data. For real-world physics systems modeling high-dimensional spatiotemporal fields, where the full state cannot be directly observed and must be inferred from sparse sensor measurements, we leverage the latent space learned from a reduced simulation model via SHRED, and update these latent variables using real sensor data to accurately reconstruct the full system state. Furthermore, our algorithm incorporates a sparse identification of nonlinear dynamics based regression model in the latent space to identify functionals corresponding to missing dynamics in the simulation model. We demonstrate that DA-SHRED successfully closes the SIM2REAL gap and additionally recovers missing dynamics in highly complex systems, demonstrating that the combination of efficient temporal encoding and physics-informed correction enables robust data assimilation.

[**arXiv:2512.01170v1**](https://arxiv.org/abs/2512.01170v1)

**Tags:** ``

---

### PIANO: Physics-informed Dual Neural Operator for Precipitation Nowcasting

**Authors:** Seokhyun Chin, Junghwan Park, Woojin Cho

**Year:** 2025

**Abstract:**
> Precipitation nowcasting, key for early warning of disasters, currently relies on computationally expensive and restrictive methods that limit access to many countries. To overcome this challenge, we propose precipitation nowcasting using satellite imagery with physics constraints for improved accuracy and physical consistency. We use a novel physics-informed dual neural operator (PIANO) structure to enforce the fundamental equation of advection-diffusion during training to predict satellite imagery using a PINN loss. Then, we use a generative model to convert satellite images to radar images, which are used for precipitation nowcasting. Compared to baseline models, our proposed model shows a notable improvement in moderate (4mm/h) precipitation event prediction alongside short-term heavy (8mm/h) precipitation event prediction. It also demonstrates low seasonal variability in predictions, indicating robustness for generalization. This study suggests the potential of the PIANO and serves as a good baseline for physics-informed precipitation nowcasting.

[**arXiv:2512.01062v1**](https://arxiv.org/abs/2512.01062v1)

**Tags:** ``

---

### Predicting CME Arrivals with Heliospheric Imagers from L5: A Data Assimilation Approach

**Authors:** Tanja Amerstorfer, Justin Le Louëdec, David Barnes, Maike Bauer, Jackie A. Davies, Satabdwa Majumdar, Eva Weiler, Christian Möstl

**Year:** 2025

**Abstract:**
> The Solar TErrestrial RElations Observatory (STEREO) mission has laid a foundation for advancing real-time space weather forecasting by enabling the evaluation of heliospheric imager (HI) data for predicting coronal mass ejection (CME) arrivals at Earth. This study employs the ELEvoHI model to assess how incorporating STEREO/HI data from the Lagrange 5 (L5) perspective can enhance prediction accuracy for CME arrival times and speeds. Our investigation, preparing for the upcoming ESA Vigil mission, explores whether the progressive incorporation of HI data in real-time enhances forecasting accuracy. The role of human tracking variability is evaluated by comparing predictions based on observations by three different scientists, highlighting the influence of manual biases on forecasting outcomes. Furthermore, the study examines the efficacy of deriving CME propagation directions using HI-specific methods versus coronagraph-based techniques, emphasising the trade-offs in prediction accuracy. Our results demonstrate the potential of HI data to significantly improve operational space weather forecasting when integrated with other observational platforms, especially when HI data from beyond 35° elongation are used. These findings pave the way for optimising real-time prediction methodologies, providing valuable groundwork for the forthcoming Vigil mission and enhancing preparedness for CME-driven space weather events.

[**arXiv:2512.09738v1**](https://arxiv.org/abs/2512.09738v1)

**Tags:** ``

---

### Bridging CORDEX and CMIP6: Machine Learning Downscaling for Wind and Solar Energy Droughts in Central Europe

**Authors:** Nina Effenberger, Maxim Samarin, Maybritt Schillinger, Reto Knutti

**Year:** 2025

**Abstract:**
> Reliable regional climate information is essential for assessing the impacts of climate change and for planning in sectors such as renewable energy; yet, producing high-resolution projections through coordinated initiatives like CORDEX that run multiple physical regional climate models is both computationally demanding and difficult to organize. Machine learning emulators that learn the mapping between global and regional climate fields offer a promising way to address these limitations. Here we introduce the application of such an emulator: trained on CMIP5 and CORDEX simulations, it reproduces regional climate model data with sufficient accuracy. When applied to CMIP6 simulations not seen during training, it also produces realistic results, indicating stable performance. Using CORDEX data, CMIP5 and CMIP6 simulations, as well as regional data generated by two machine learning models, we analyze the co-occurrence of low wind speed and low solar radiation and find indications that the number of such energy drought days is likely to decrease in the future. Our results highlight that downscaling with machine learning emulators provides an efficient complement to efforts such as CORDEX, supplying the higher-resolution information required for impact assessments.

[**arXiv:2512.07429v1**](https://arxiv.org/abs/2512.07429v1)

**Tags:** ``

---

### Latent-space variational data assimilation in two-dimensional turbulence

**Authors:** Andrew Cleary, Qi Wang, Tamer A. Zaki

**Year:** 2025

**Abstract:**
> Starting from limited measurements of a turbulent flow, data assimilation (DA) attempts to estimate all the spatio-temporal scales of motion. Success is dependent on whether the system is observable from the measurements, or how much of the initial turbulent field is encoded in the available measurements. Adjoint-variational DA minimises the discrepancy between the true and estimated measurements by optimising the initial velocity or vorticity field (the `state space'). Here we propose to instead optimise in a lower-dimensional latent space which is learned by implicit rank minimising autoencoders. Assimilating in latent space, rather than state space, redefines the observability of the measurements and identifies the physically meaningful perturbation directions which matter most for accurate prediction of the flow evolution. When observing coarse-grained measurements of two-dimensional Kolmogorov flow at moderate Reynolds numbers, the proposed latent-space DA approach estimates the full turbulent state with a relative error improvement of two orders of magnitude over the standard state-space DA approach. The small scales of the estimated turbulent field are predicted more faithfully with latent-space DA, greatly reducing erroneous small-scale velocities typically introduced by state-space DA. These findings demonstrate that the observability of the system from available data can be greatly improved when turbulent measurements are assimilated in the right space, or coordinates.

[**arXiv:2512.15470v1**](https://arxiv.org/abs/2512.15470v1)

**Tags:** ``

---

### Continuous data assimilation for 2D stochastic Navier-Stokes equations

**Authors:** Hakima Bessaih, Benedetta Ferrario, Oussama Landoulsi, Margherita Zanella

**Year:** 2025

**Abstract:**
> Continuous data assimilation methods, such as the nudging algorithm introduced by Azouani, Olson, and Titi (AOT) [2], are known to be highly effective in deterministic settings for asymptotically synchronizing approximate solutions with observed dynamics. In this work, we extend this framework to a stochastic regime by considering the two-dimensional incompressible Navier-Stokes equations subject to either additive or multiplicative noise. We establish sufficient conditions on the nudging parameter and the spatial observation scale that guarantee convergence of the nudged solution to the true stochastic flow.   In the case of multiplicative noise, convergence holds in expectation, with exponential or polynomial rates depending on the growth of the noise covariance. For additive noise, we obtain the exponential convergence both in expectation and pathwise. These results yield a stochastic generalization of the AOT theory, demonstrating how the interplay between random forcing, viscous dissipation and feedback control governs synchronization in stochastic fluid systems.

[**arXiv:2512.15184v1**](https://arxiv.org/abs/2512.15184v1)

**Tags:** ``

---

### Bridging Artificial Intelligence and Data Assimilation: The Data-driven Ensemble Forecasting System ClimaX-LETKF

**Authors:** Akira Takeshima, Kenta Shiraishi, Atsushi Okazaki, Tadashi Tsuyuki, Shunji Kotsuki

**Year:** 2025

**Abstract:**
> While machine learning-based weather prediction (MLWP) has achieved significant advancements, research on assimilating real observations or ensemble forecasts within MLWP models remains limited. We introduce ClimaX-LETKF, the first purely data-driven ML-based ensemble weather forecasting system. It operates stably over multiple years, independently of numerical weather prediction (NWP) models, by assimilating the NCEP ADP Global Upper Air and Surface Weather Observations. The system demonstrates greater stability and accuracy with relaxation to prior perturbation (RTPP) than with relaxation to prior spread (RTPS), while NWP models tend to be more stable with RTPS. RTPP replaces an analysis perturbation with a weighted blend of analysis and background perturbations, whereas RTPS simply rescales the analysis perturbation. Our experiments reveal that MLWP models are less capable of restoring the atmospheric field to its attractor than NWP models. This work provides valuable insights for enhancing MLWP ensemble forecasting systems and represents a substantial step toward their practical applications.

[**arXiv:2512.14444v1**](https://arxiv.org/abs/2512.14444v1)

**Tags:** ``

---

### Evaluating Weather Forecasts from a Decision Maker's Perspective

**Authors:** Kornelius Raeth, Nicole Ludwig

**Year:** 2025

**Abstract:**
> Standard weather forecast evaluations focus on the forecaster's perspective and on a statistical assessment comparing forecasts and observations. In practice, however, forecasts are used to make decisions, so it seems natural to take the decision-maker's perspective and quantify the value of a forecast by its ability to improve decision-making. Decision calibration provides a novel framework for evaluating forecast performance at the decision level rather than the forecast level. We evaluate decision calibration to compare Machine Learning and classical numerical weather prediction models on various weather-dependent decision tasks. We find that model performance at the forecast level does not reliably translate to performance in downstream decision-making: some performance differences only become apparent at the decision level, and model rankings can change among different decision tasks. Our results confirm that typical forecast evaluations are insufficient for selecting the optimal forecast model for a specific decision task.

[**arXiv:2512.14779v1**](https://arxiv.org/abs/2512.14779v1)

**Tags:** ``

---

### Quantum Machine Learning for Climate Modelling

**Authors:** Mierk Schwabe, Lorenzo Pastori, Valentina Sarandrea, Veronika Eyring

**Year:** 2025

**Abstract:**
> Quantum machine learning (QML) is making rapid progress, and QML-based models hold the promise of quantum advantages such as potentially higher expressivity and generalizability than their classical counterparts. Here, we present work on using a quantum neural net (QNN) to develop a parameterization of cloud cover for an Earth system model (ESM). ESMs are needed for predicting and projecting climate change, and can be improved in hybrid models incorporating both traditional physics-based components as well as machine learning (ML) models. We show that a QNN can predict cloud cover with a performance similar to a classical NN with the same number of free parameters and significantly better than the traditional scheme. We also analyse the learning capability of the QNN in comparison to the classical NN and show that, at least for our example, QNNs learn more consistent relationships than classical NNs.

[**arXiv:2512.14208v1**](https://arxiv.org/abs/2512.14208v1)

**Tags:** ``

---

### An intercomparison of generative machine learning methods for downscaling precipitation at fine spatial scales

**Authors:** Bryn Ward-Leikis, Neelesh Rampal, Yun Sing Koh, Peter B. Gibson, Hong-Yang Liu, Vassili Kitsios, Tristan Meyers, Jeff Adie, Yang Juntao, Steven C. Sherwood

**Year:** 2025

**Abstract:**
> Machine learning (ML) offers a computationally efficient approach for generating large ensembles of high-resolution climate projections, but deterministic ML methods often smooth fine-scale structures and underestimate extremes. While stochastic generative models show promise for predicting fine-scale weather and extremes, few studies have compared their performance under present-day and future climates. This study compares a previously developed conditional Generative Adversarial Network (cGAN) with an intensity constraint against different configurations of diffusion models for downscaling daily precipitation from a regional climate model (RCM) over Aotearoa New Zealand. Model skill is comprehensively assessed across spatial structure, distributional metrics, means, extremes, and their respective climate change signals. Both generative approaches outperform the deterministic baseline across most metrics and exhibit similar overall skill. Diffusion models better predict the fine-scale spatial structure of precipitation and the length of dry spells, but underestimate climate change signals for extreme precipitation compared to the ground truth RCMs. In contrast, cGANs achieve comparable skill for most metrics while better predicting the overall precipitation distribution and climate change responses for extremes at a fraction of the computational cost. These results demonstrate that while diffusion models can readily generate predictions with greater visual "realism", they do not necessarily better preserve climate change responses compared to cGANs with intensity constraints. At present, incorporating constraints into diffusion models remains challenging compared to cGANs, but may represent an opportunity to further improve skill for predicting climate change responses.

[**arXiv:2512.13987v1**](https://arxiv.org/abs/2512.13987v1)

**Tags:** ``

---

### Time-aware UNet and super-resolution deep residual networks for spatial downscaling

**Authors:** Mika Sipilä, Sabrina Maggio, Sandra De Iaco, Klaus Nordhausen, Monica Palma, Sara Taskinen

**Year:** 2025

**Abstract:**
> Satellite data of atmospheric pollutants are often available only at coarse spatial resolution, limiting their applicability in local-scale environmental analysis and decision-making. Spatial downscaling methods aim to transform the coarse satellite data into high-resolution fields. In this work, two widely used deep learning architectures, the super-resolution deep residual network (SRDRN) and the encoder-decoder-based UNet, are considered for spatial downscaling of tropospheric ozone. Both methods are extended with a lightweight temporal module, which encodes observation time using either sinusoidal or radial basis function (RBF) encoding, and fuses the temporal features with the spatial representations in the networks. The proposed time-aware extensions are evaluated against their baseline counterparts in a case study on ozone downscaling over Italy. The results suggest that, while only slightly increasing computational complexity, the temporal modules significantly improve downscaling performance and convergence speed.

[**arXiv:2512.13753v1**](https://arxiv.org/abs/2512.13753v1)

**Tags:** ``

---

### STLDM: Spatio-Temporal Latent Diffusion Model for Precipitation Nowcasting

**Authors:** Shi Quan Foo, Chi-Ho Wong, Zhihan Gao, Dit-Yan Yeung, Ka-Hing Wong, Wai-Kin Wong

**Year:** 2025

**Abstract:**
> Precipitation nowcasting is a critical spatio-temporal prediction task for society to prevent severe damage owing to extreme weather events. Despite the advances in this field, the complex and stochastic nature of this task still poses challenges to existing approaches. Specifically, deterministic models tend to produce blurry predictions while generative models often struggle with poor accuracy. In this paper, we present a simple yet effective model architecture termed STLDM, a diffusion-based model that learns the latent representation from end to end alongside both the Variational Autoencoder and the conditioning network. STLDM decomposes this task into two stages: a deterministic forecasting stage handled by the conditioning network, and an enhancement stage performed by the latent diffusion model. Experimental results on multiple radar datasets demonstrate that STLDM achieves superior performance compared to the state of the art, while also improving inference efficiency. The code is available in https://github.com/sqfoo/stldm_official.

[**arXiv:2512.21118v1**](https://arxiv.org/abs/2512.21118v1)

**Tags:** ``

---

### Quantum Bayesian Optimization for the Automatic Tuning of Lorenz-96 as a Surrogate Climate Model

**Authors:** Paul J. Christiansen, Daniel Ohl de Mello, Cedric Brügmann, Steffen Hien, Felix Herbort, Martin Kiffner, Lorenzo Pastori, Veronika Eyring, Mierk Schwabe

**Year:** 2025

**Abstract:**
> In this work, we propose a hybrid quantum-inspired heuristic for automatically tuning the Lorenz-96 model -- a simple proxy to describe atmospheric dynamics, yet exhibiting chaotic behavior. Building on the history matching framework by Lguensat et al. (2023), we fully automate the tuning process with a new convergence criterion and propose replacing classical Gaussian process emulators with quantum counterparts. We benchmark three quantum kernel architectures, distinguished by their quantum feature map circuits. A dimensionality argument implies, in principle, an increased expressivity of the quantum kernels over their classical competitors. For each kernel type, we perform an extensive hyperparameter optimization of our tuning algorithm. We confirm the validity of a quantum-inspired approach based on statevector simulation by numerically demonstrating the superiority of two studied quantum kernels over the canonical classical RBF kernel. Finally, we discuss the pathway towards real quantum hardware, mainly driven by a transition to shot-based simulations and evaluating quantum kernels via randomized measurements, which can mitigate the effect of gate errors. The very low qubit requirements and moderate circuit depths, together with a minimal number of trainable circuit parameters, make our method particularly NISQ-friendly.

[**arXiv:2512.20437v1**](https://arxiv.org/abs/2512.20437v1)

**Tags:** ``

---

### The Ensemble Schr{ö}dinger Bridge filter for Nonlinear Data Assimilation

**Authors:** Feng Bao, Hui Sun

**Year:** 2025

**Abstract:**
> This work puts forward a novel nonlinear optimal filter namely the Ensemble Schr{ö}dinger Bridge nonlinear filter. The proposed filter finds marriage of the standard prediction procedure and the diffusion generative modeling for the analysis procedure to realize one filtering step. The designed approach finds no structural model error, and it is derivative free, training free and highly parallizable. Experimental results show that the designed algorithm performs well given highly nonlinear dynamics in (mildly) high dimension up to 40 or above under a chaotic environment. It also shows better performance than classical methods such as the ensemble Kalman filter and the Particle filter in numerous tests given different level of nonlinearity. Future work will focus on extending the proposed approach to practical meteorological applications and establishing a rigorous convergence analysis.

[**arXiv:2512.18928v1**](https://arxiv.org/abs/2512.18928v1)

**Tags:** ``

---

