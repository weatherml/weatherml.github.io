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

