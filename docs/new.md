# New

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

