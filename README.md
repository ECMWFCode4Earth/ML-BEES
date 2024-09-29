<p align="center">
  <img src="https://github.com/ECMWFCode4Earth/ML-BEES/assets/55485922/77261329-0553-4688-b674-3292d60e0a53" alt="ml-bees_logo" width="300" />
</p>

<h1 align="left" style="margin-bottom: 0;">ML-BEES</h1>
<p align="left" style="margin-top: 0;">Developing Machine Learning-based Emulation of the Earth's Surface</p>

<div style="display: flex; align-items: center; justify-content: center;">
  <div style="text-align: left; margin-right: 20px;">
    <p><b>Layer 1 Soil moisture inference from UniMP</b></p>
    <img src="https://github.com/ECMWFCode4Earth/ML-BEES/blob/main/media/ailand_video_4_unimp.gif" alt="UniMP" width="400px">
  </div>
  <div style="text-align: left;">
    <p><b>Layer 1 Soil moisture simulation from ECLand</b></p>
    <img src="https://github.com/ECMWFCode4Earth/ML-BEES/blob/main/media/ecland_video_4_unimp.gif" alt="ECLand" width="400px">
  </div>
</div>

## About the project

Machine Learning (ML) is becoming increasingly important for numerical weather prediction (NWP), and ML-based models have reached similar or improved forecast scores than state-of-the-art physical models. ECMWF has intensified its activities in the application of ML models for atmospheric forecasting and developed the Artificial Intelligence/Integrated Forecasting System (AIFS). To harness the potential of ML for land modelling and data assimilation activities at ECMWF, a first ML emulator prototype has been developed (Pinnington et al. AMS Annual Meeting 2024). The ML model was trained on the "offline" ECMWF Land Surface Modelling System (ECLand) using a preselected ML training database. The current prototype is based on the information of model increments without introducing further temporal constraints and provides a cheap alternative to physical models. 

By developing and evaluating four types of state-of-art machine learning emulators (XGBoost, MLP, UniMP (GNN), and MAMBA), our project demonstrated that machine learning (ML) models can effectively and efficiently emulate land surface models in terms of spatial-temporal variability, uncertainty, and physical consistency. Among the models tested, XGBoost delivered the best overall performance across all 17 target variables, while the MLP model, though slightly less accurate, was the most efficient, providing inference six times faster than XGBoost. These ML models can serve as reliable alternatives to physically-based land surface models for computationally intensive experiments, with the choice of model depending on the specific task.

## Project structure

ML-BEES/
├── AR5_region_mask/         # .shp for masking AR5 regions for spatial condition analysis
├── ML-BEES-eval/            # functions and Jupyter notebooks for ML emulator evaluation; more details please refers to readme.md
├── ML-BEES-train/           # Source code for ML models and utility functions for data loading and inference
├── media/                   # Images and videos used in README or documentation
├── ai_land_original/        # Example scripts developed by Pinnington et al.
├── README.md                
├── LICENSE                  # License for the project
└── .gitignore               # Files to ignore in git

## Documentation

For more details about our project workflow and analysis, please find our documentations in the following slides and project report.

- [Download Project Presentation (PDF)](https://drive.google.com/file/d/1Yu7L-Ikw_flcHfVMN6kOAIoPAaS1gTN2/view?usp=sharing)
- [Download Project Report (PDF)](link_to_your_pdf_file)

## Authors

Participants:

- Till Fohrmann
- Johannes Leonhardt
- Hakam Shams
- Yikui Zhang (https://www.linkedin.com/in/yikui-zhang-0254721a1/)

## Acknowledgments

Big thanks for all of our mentors from European Centre for Medium-Range Weather Forecasts (ECMWF) in Reading and Bonn, as well as the supports and funding from ECMWF Code For Earth 2024 Event!