# Global Models

### GraphCast: Learning skillful medium-range global weather forecasting

**Authors:** Remi Lam, Alvaro Sanchez-Gonzalez, Matthew Willson, Peter Wirnsberger, Meire Fortunato, Ferran Alet, Suman Ravuri, Timo Ewalds, Zach Eaton-Rosen, Weihua Hu, et al.

**Year:** 2023

**Abstract:**
> We describe GraphCast, a machine learning-based method for medium-range weather forecasting. It predicts hundreds of weather variables for the next 10 days at 0.25 degree resolution globally. GraphCast's predictions are more accurate than the most accurate operational deterministic system, HRES, on 90% of 1380 verification targets, and its forecasts can be produced in under one minute. GraphCast is a significant milestone in machine learning for weather forecasting and a testament to the power of deep learning for modeling complex dynamical systems.

[**arXiv:2212.12794**](https://arxiv.org/abs/2212.12794)

**Tags:** `GNN`

---

### Pangu-Weather: Accurate medium-range global weather forecasting with 3D neural networks

**Authors:** Kaifeng Bi, Lingxi Xie, Hengheng Zhang, Xin Chen, Xiaotao Gu, and Qi Tian

**Year:** 2023

**Abstract:**
> We present Pangu-Weather, a deep learning model for medium-range global weather forecasting. Pangu-Weather is based on a 3D U-Net architecture that processes weather data on a spherical grid. We show that Pangu-Weather can produce forecasts that are more accurate than those from the operational IFS model, especially for temperature and geopotential height. Pangu-Weather is also significantly faster than IFS, producing a 10-day forecast in just a few minutes. Pangu-Weather is a powerful new tool for weather forecasting that has the potential to improve our ability to predict a wide range of weather phenomena.

[**arXiv:2301.03748**](https://arxiv.org/abs/2301.03748)

**Tags:** `3D U-Net`

---

### FourCastNet: A Global Data-driven High-resolution Weather Model using Adaptive Fourier Neural Operators

**Authors:** Jaideep Pathak, Shashank Subramanian, Peter Harrington, Sanjeev Raja, Ashesh Chattopadhyay, Morteza Mardani, Thorsten Kurth, David Hall, Zongyi Li, Kamyar Azizzadenesheli, Pedram Hassanzadeh, Karthik Kashinath, Animashree Anandkumar

**Year:** 2022

**Abstract:**
> FourCastNet, short for Fourier Forecasting Neural Network, is a global data-driven weather forecasting model that provides accurate short to medium-range global predictions at 0.25◦ resolution. FourCastNet accurately forecasts high-resolution, fast-timescale variables such as the surface wind speed, precipitation, and atmospheric water vapor. It has important implications for planning wind energy resources, predicting extreme weather events such as tropical cyclones, extra-tropical cyclones, and atmospheric rivers. FourCastNet matches the forecasting accuracy of the ECMWF Integrated Forecasting System (IFS), a state-of-the-art Numerical Weather Prediction (NWP) model, at short lead times for large-scale variables, while outperforming IFS for variables with complex fine-scale structure, including precipitation. FourCastNet generates a week-long forecast in less than 2 seconds, orders of magnitude faster than IFS. The speed of FourCastNet enables the creation of rapid and inexpensive large-ensemble forecasts with thousands of ensemble-members for improving probabilistic forecasting. We discuss how data-driven deep learning models such as FourCastNet are a valuable addition to the meteorology toolkit to aid and augment NWP models.

[**arXiv:2202.11214**](https://arxiv.org/abs/2202.11214)

**Tags:** `Transformer`, `Fourier Neural Operator`

---
