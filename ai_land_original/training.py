import logging
import os

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from data_module import EcDataset, NonLinRegDataModule

# from data_module_cat import EcDataset, NonLinRegDataModule
# from model import NonLinearRegression
from model_simple import NonLinearRegression
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import CSVLogger, MLFlowLogger
from torch import tensor

# from pytorch_lightning.plugins.environments import SLURMEnvironment
from train_callbacks import PlotCallback

# import signal


# from pytorch_lightning.utilities.distributed import rank_zero_only
# from torch.distributed import init_process_group, destroy_process_group

logging.basicConfig(level=logging.INFO)


PATH_NAME = os.path.dirname(os.path.abspath(__file__))
with open(f"{PATH_NAME}/config.yaml") as stream:
    try:
        CONFIG = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


if __name__ == "__main__":
    # Set device
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)

    data_module = NonLinRegDataModule()
    dataset = EcDataset()

    if CONFIG["logging"]["logger"] == "csv":
        logger = CSVLogger(
            CONFIG["logging"]["location"], name="testing"
        )  # Change 'logs' to the directory where you want to save the logs
    elif CONFIG["logging"]["logger"] == "mlflow":
        logger = MLFlowLogger(
            experiment_name=CONFIG["logging"]["project"],
            run_name=CONFIG["logging"]["name"],
            tracking_uri=CONFIG["logging"]["uri"],  # "file:./mlruns",
        )
        logging.info(f"logging to {logger.run_id}")
    else:
        logger = None

    # checkpoint_callback = ModelCheckpoint(monitor="val_R2", mode="max")
    # checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")
    checkpoint_callback = ModelCheckpoint(monitor="val_step_loss", mode="min")

    # Setting a small validation dataset for plotting during training
    logging.info("Opening dataset for plotting...")
    test_ds = EcDataset(
        start_yr=2022,
        end_yr=2022,
        # x_idxs=(500, 500 + 1),
        # path="/ec/res4/hpcperm/daep/ecland_i6aj_o400_2010_2022_6hr_subset.zarr",
        x_idxs=(9973, 9973 + 1),
        # path="/ec/res4/scratch/daep/ec_training_db_out_O200/ecland_i8ki_2010_2022_6h.zarr"
        path="/ec/res4/hpcperm/daep/ec_land_training_db/ecland_i8ki_o200_2010_2022_6h.zarr",
    )
    logging.info("Setting plot callback...")
    plot_callback = PlotCallback(
        plot_frequency=CONFIG["logging"]["plot_freq"],
        dataset=test_ds,
        device=device,
        # device="cpu",
        logger=logger,
    )

    # print("Opening dataset for plotting...")
    # dataset_plot = EcDataset("2022", "2022", (8239, 8240))
    # print("Setting plot callback...")
    # plot_callback = PlotCallback(plot_frequency=1, dataset=dataset_plot)

    std = dataset.y_prog_stdevs.cpu().numpy()
    # ds_mean = np.nanmean(dataset.ds_ecland.firstdiff_means[slice(*dataset.x_idxs), dataset.targ_index], axis=0) / (std + 1e-5)
    # ds_std =dataset.ds_ecland.firstdiff_stdevs[slice(*dataset.x_idxs), dataset.targ_index] / (std + 1e-5)
    # ds_std = np.nanstd(dataset.ds_ecland.firstdiff_stdevs[slice(*dataset.x_idxs), dataset.targ_index], axis=0) / (std + 1e-5)
    # ds_mean = dataset.ds_ecland.firstdiff_means[dataset.targ_index] / std  # (std + 1e-5)
    # ds_std = dataset.ds_ecland.firstdiff_stdevs[dataset.targ_index] / std  # (std + 1e-5)
    # ds_mean = dataset.ds_ecland.data_1stdiff_means[dataset.targ_index] / std
    # ds_std = dataset.ds_ecland.data_1stdiff_stdevs[dataset.targ_index] / std
    ds_mean = dataset.ds_ecland.data_1stdiff_means[dataset.targ_index] / (std + 1e-5)
    ds_std = dataset.ds_ecland.data_1stdiff_stdevs[dataset.targ_index] / (std + 1e-5)

    # train
    logging.info("Setting model params...")
    input_clim_dim = dataset.x_static_scaled.shape[-1]
    input_met_dim = len(dataset.dynamic_feat_lst)
    input_state_dim = len(dataset.targ_lst)
    output_dim = len(dataset.targ_lst)  # Number of output targets
    output_diag_dim = len(dataset.targ_diag_lst)
    hidden_dim = CONFIG["hidden_dim"]  # Number of hidden units
    model_pyt = NonLinearRegression(
        input_clim_dim,
        input_met_dim,
        input_state_dim,
        hidden_dim,
        output_dim,
        output_diag_dim,
        mu_norm=ds_mean,
        std_norm=ds_std,
        dataset=dataset,
    )

    torch.set_float32_matmul_precision("high")

    logging.info("Setting Trainer...")
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, RichProgressBar(), plot_callback],
        # callbacks=[checkpoint_callback, RichProgressBar()],
        max_epochs=CONFIG["max_epochs"],
        # plugins=[SLURMEnvironment(requeue_signal=signal.SIGUSR1)],
        strategy=CONFIG["strategy"],
        devices=CONFIG["devices"],
        # barebones=True,
    )

    logging.info("Training...")
    trainer.logger.log_hyperparams(CONFIG)
    trainer.fit(model_pyt, data_module)

    logging.info("Saving model...")
    model_pyt.eval()
    torch.save(model_pyt.state_dict(), CONFIG["model_path"])
