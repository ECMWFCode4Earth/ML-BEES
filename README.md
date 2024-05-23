# ML-BEES
Evaluating and improving the performance of ECMWF’s current land surface Machine Learning model prototype.

![ml-bees_logo](https://github.com/ECMWFCode4Earth/ML-BEES/assets/55485922/77261329-0553-4688-b674-3292d60e0a53)

## Challenge description

Machine Learning (ML) is becoming increasingly important for numerical weather prediction (NWP), and ML-based models have reached similar or improved forecast scores than state-of-the-art physical models. ECMWF has intensified its activities in the application of ML models for atmospheric forecasting and developed the Artificial Intelligence/Integrated Forecasting System (AIFS). To harness the potential of ML for land modelling and data assimilation activities at ECMWF, a first ML emulator prototype has been developed (Pinnington et al. AMS Annual Meeting 2024). The ML model was trained on the "offline" ECMWF Land Surface Modelling System (ECLand) using a preselected ML training database. The current prototype is based on the information of model increments without introducing further temporal constraints and provides a cheap alternative to physical models. 

So far, a qualitative comparison between ECLand-based and emulated fields has been performed on a subset of sites, which revealed that the time series of land variables match well in terms of dynamic range and general trend behaviour. However, more targeted evaluation is required to assess the performance of the land emulator prototype. In this project, we seek to validate the prototype emulator thoroughly, to understand the model's capabilities in approximating the ECLand model output, and also compare its results to in-situ observations by developing an evaluation framework for LSM emulators. Furtheron, we seek to improve upon the identified shortcomings in the validation and extend the current prototype’s capabilities using state-of-the-art techniques and ML model structures. 
