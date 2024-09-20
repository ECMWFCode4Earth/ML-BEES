# ML-BEES Evaluation Readme
This directory contains all scripts used in the evaluation of the ECLand emulators. 

## Configuration
For many scripts, it is important to set up the *config.yaml* first.
- *Data storage paths*: Absolute paths to the location of the European and global ECLand data sets. The global data was not used in the coding phase.
- *Inference data paths*: Paths to the outputs (evaluation period) of each AILand version to be evaluated.
- *Evaluation results data path*: Set the storage paths for the evaluation outputs.
- *Prognostic and diagnostic variables*: Informs the scripts about the variables used. What is considered prognostic and diagnostic is not important for the evaluation, this is just to match the model development set-up. Note that the earliest emulators do not feature all variables. The scripts are written in such a way that only available variables are processed.

## Eval_Utilities
This custom module is used to wrap frequently used functions and provide some convenience in the execution of the evaluation. The metrics are part of *spatial_temporal_metrics.py*. 
As the name suggests, *visualization.py* contains the plotting routines used in both the auomatic and in-depth evaluation. 
The test of soil moisture sensitivity to field capacity (part of *physical_consistency.ipynb*) requires to re-run parts of the inference. For this, we require the model definitions contained in the folder *model*, which is linking to the AILand configurations in the developement branch. The other requirement is the modified copy of *EclandPointDataset.py* from the original AILand prototype, which is also contained in the module.

## Tests
To ensure the functionality of the base metrics, some unit tests were implemented. Due to time constraints, this testing is far from complete. What is available is included in the *tests* folder.

## Automatic Evaluation
The automatic evaluation uses the mentioned configuration file. Apart from that, some further adjustments can be made in the script *run_workflow.py* itself. These include the domain, evaluation time span, variables, metrics and switches for the three parts of the evaluation: computation of the metrics, creation of the visualizations and the scoreboard.
Outputs will be stored to the paths set in the configuration file. The only output, which is saved to this directory itself is the scoreboard markdown file.

## Other Notebooks
To help navigate this directory, we give a short account of other evaluation scripts included in this directory in alphabetical order:
- *consistency.ipynb*: This is used for exploring the output data. We looked at whether the data actually matches with the computed scores to rule out mistakes. We also looked at exceptional grid points and whether they have something in common.
- *data_extract.ipynb*: ???
- *explore_insitu_data.ipynb*: Used to get a feeling for the format and coverage of in-situ data.
- *model_scores.ipynb*: Here we tested how to implement the ILAMB scoring system and how to generate a scoreboard from it. The code was developed for the automatic evaluation and integrated into *run_workflow.py*. We kept the notebook, because it explains a few things about the process.
- *observations.ipynb*: The script for actually computing the comparisons against observations.
- *physical_consistency.ipynb*: This contains the sensitivity tests regarding soil moisture and field capacity as well as the water balance calculations.
- *spatial_condition.ipynb*: The notebook contains the evaluation related to spatial features such as soil types.
- *spatial_variability_new.ipynb*: ???
- *temporal_variability.ipynb*: Here we do the power spectra analysis and error analysis conditioned on time and season.
- *uncertainty.ipynb*: This notebook contains the uncertainty analysis.

## Other
*preds_original_clim.pckl* is used in the soil moisture vs. field capacity analysis for MLP. It contains output using the un-modified static features and was created to not have to re-run the un-modified inference in the notebook.

*scoreboard.md* is the scoreboard output from the *run_workflow.py* script.