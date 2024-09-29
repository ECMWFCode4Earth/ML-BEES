# ML-BEES (Machine Learning-based Emulation of the Earth's Surface)
<p align="center">
  <img src="https://github.com/ECMWFCode4Earth/ML-BEES/assets/55485922/77261329-0553-4688-b674-3292d60e0a53" alt="ml-bees_logo" width="50" />
</p>
## Major contribution

By developing and evaluating four types of state-of-art machine learning emulators (XGBoost, MLP, UniMP (GNN), and MAMBA), our project demonstrated that machine learning (ML) models can effectively and efficiently emulate land surface models in terms of spatial-temporal variability, uncertainty, and physical consistency. Among the models tested, XGBoost delivered the best overall performance across all 17 target variables, while the MLP model, though slightly less accurate, was the most efficient, providing inference six times faster than XGBoost. These ML models can serve as reliable alternatives to physically-based land surface models for computationally intensive experiments, with the choice of model depending on the specific task.

<p align="center">
  <img src="https://github.com/ECMWFCode4Earth/ML-BEES/blob/main/media/ailand_video_4_unimp.gif" alt="UniMP" width="45%" />
  <img src="https://github.com/ECMWFCode4Earth/ML-BEES/blob/main/media/ecland_video_4_unimp.gif" alt="ECLand" width="45%" />
</p>
<p align="center">
  <b>UniMP</b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <b>ECLand</b>
</p>

## Challenge description

Machine Learning (ML) is becoming increasingly important for numerical weather prediction (NWP), and ML-based models have reached similar or improved forecast scores than state-of-the-art physical models. ECMWF has intensified its activities in the application of ML models for atmospheric forecasting and developed the Artificial Intelligence/Integrated Forecasting System (AIFS). To harness the potential of ML for land modelling and data assimilation activities at ECMWF, a first ML emulator prototype has been developed (Pinnington et al. AMS Annual Meeting 2024). The ML model was trained on the "offline" ECMWF Land Surface Modelling System (ECLand) using a preselected ML training database. The current prototype is based on the information of model increments without introducing further temporal constraints and provides a cheap alternative to physical models. 

So far, a qualitative comparison between ECLand-based and emulated fields has been performed on a subset of sites, which revealed that the time series of land variables match well in terms of dynamic range and general trend behaviour. However, more targeted evaluation is required to assess the performance of the land emulator prototype. In this project, we seek to validate the prototype emulator thoroughly, to understand the model's capabilities in approximating the ECLand model output, and also compare its results to in-situ observations by developing an evaluation framework for LSM emulators. Furtheron, we seek to improve upon the identified shortcomings in the validation and extend the current prototype’s capabilities using state-of-the-art techniques and ML model structures. 
