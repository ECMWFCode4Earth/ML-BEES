# ML-BEES
Evaluating and improving the performance of ECMWF’s current land surface Machine Learning model prototype.


## Challenge description

Machine Learning (ML) is becoming increasingly important for numerical weather prediction (NWP), and ML-based models have reached similar or improved forecast scores than state-of-the-art physical models. ECMWF has intensified its activities in the application of ML models for atmospheric forecasting and developed the Artificial Intelligence/Integrated Forecasting System (AIFS). To harness the potential of ML for land modelling and data assimilation activities at ECMWF, a first ML emulator prototype has been developed (Pinnington et al. AMS Annual Meeting 2024). The ML model was trained on the "offline" ECMWF Land Surface Modelling System (ECLand) using a preselected ML training database. The current prototype is based on the information of model increments without introducing further temporal constraints and provides a cheap alternative to physical models. It opens up many application possibilities such as the optimization of model parameters and the generation of cost-effective ensembles and land surface initial conditions for NWP.

So far, a qualitative comparison between ECLand-based and emulated fields has been performed on a subset of sites, which revealed that the time series of land variables match well in terms of dynamic range and general trend behaviour. However, more targeted evaluation is required to assess the performance of the land emulator prototype. The aim is to understand the model's capabilities in reproducing the ECLand spatial and temporal patterns and its performance evaluated against in-situ observations.
