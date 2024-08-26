# ------------------------------------------------------------------
# Testing an MLP model on observations
# Script for testing and validating on EC-Land dataset
# Testing is done on the normalized data
# ------------------------------------------------------------------

import os
import logging
import yaml
from dataset.EclandObsPointDataset import EcObsDataset
from model.MLP_Obs import MLP_Obs
from torch.utils.tensorboard import SummaryWriter
import torch
import argparse
import numpy as np
from utils import utils
import time
from tqdm import tqdm

# ------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)
torch.set_float32_matmul_precision("high")
torch.cuda.empty_cache()


# ------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default=r'configs/config.yaml',
                        type=str, help='configuration file for training')
    args = parser.parse_args()
    return args


# ------------------------------------------------------------------


def main(config):

    # get logger
    logger = utils.get_logger(config, phase='test')

    # fix random seed
    utils.log_string(logger, "fix random seed...")
    utils.fix_seed(config["random_seed"])

    # initialize the dataset class and dataloader
    utils.log_string(logger, "loading validation dataset...")

    val_dataset = EcObsDataset(
        start_year=config["validation_start"],
        end_year=config["validation_end"],
        x_slice_indices=config["x_slice_indices"],
        root=config["file_path"],
        root_sm=config["smap_file"],
        root_temp=config["modis_temp_file"],
        use_time_var_lai=config["use_time_var_lai"],
        root_lail=config["lail_file"],
        root_laih=config["laih_file"],
        roll_out=1,
        clim_features=config["clim_feats"],
        dynamic_features=config["dynamic_feats"],
        target_prog_features=config["targets_prog"],
        target_diag_features=config["targets_diag"],
        is_add_lat_lon=config["is_add_lat_lon"],
        is_norm=config["is_norm"],
        point_dropout=0.
    )

    utils.log_string(logger, "loading testing dataset...")

    test_dataset = EcObsDataset(
        start_year=config["test_start"],
        end_year=config["test_end"],
        x_slice_indices=config["x_slice_indices"],
        root=config["file_path"],
        root_sm=config["smap_file"],
        root_temp=config["modis_temp_file"],
        use_time_var_lai=config["use_time_var_lai"],
        root_lail=config["lail_file"],
        root_laih=config["laih_file"],
        roll_out=1,
        clim_features=config["clim_feats"],
        dynamic_features=config["dynamic_feats"],
        target_prog_features=config["targets_prog"],
        target_diag_features=config["targets_diag"],
        is_add_lat_lon=config["is_add_lat_lon"],
        is_norm=config["is_norm"],
        point_dropout=0.
    )

    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=config["batch_size"],
                                                 drop_last=False,
                                                 shuffle=False,
                                                 pin_memory=config["pin_memory"],
                                                 num_workers=config["num_workers"],
                                                 persistent_workers=config["persistent_workers"]
                                                 )

    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=config["batch_size"],
                                                  drop_last=False,
                                                  shuffle=False,
                                                  pin_memory=config["pin_memory"],
                                                  num_workers=config["num_workers"],
                                                  persistent_workers=config["persistent_workers"]
                                                  )

    utils.log_string(logger, "# evaluation samples: %d" % len(val_dataset))
    utils.log_string(logger, "# testing samples: %d" % len(test_dataset))

    # get statistic to normalize the output delta x
    logging.info("prepare mean and std for delta x...")
    y_prog_inc_mean = val_dataset.y_prog_inc_mean
    y_prog_inc_std = val_dataset.y_prog_inc_std

    # get indices for swvl1 and skt
    swvl1_idx = test_dataset.target_prog_features.index('swvl1')
    skt_idx = test_dataset.target_diag_features.index('skt')

    # get models
    utils.log_string(logger, "\nbuild the model ...")

    if config["devices"] != -1:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config["devices"])
        device = 'cuda'
    else:
        device = 'cpu'

    model = MLP_Obs(in_static=val_dataset.n_static,
                    in_dynamic=val_dataset.n_dynamic,
                    in_prog=val_dataset.n_prog,
                    out_prog=val_dataset.n_prog,
                    out_diag=val_dataset.n_diag,
                    hidden_dim=config["hidden_dim"],
                    rollout=config["roll_out"],
                    dropout=config["dropout"],
                    mu_norm=y_prog_inc_mean,
                    std_norm=y_prog_inc_std,
                    swvl1_idx=swvl1_idx,
                    skt_idx=skt_idx,
                    pretrained=config["pretrained"]
                    )

    utils.log_string(logger, "model parameters ...")
    utils.log_string(logger, "all parameters: %d" % utils.count_parameters(model))

    model.to(device)

    # testing loop
    utils.log_string(logger, 'testing on EC-Land dataset...')

    # initialize the evaluation class
    eval_val = utils.evaluator(logger, 'Validation', val_dataset.target_prog_features, val_dataset.target_diag_features)
    eval_test = utils.evaluator(logger, 'Testing', test_dataset.target_prog_features, test_dataset.target_diag_features)
    eval_obs_val = utils.evaluator_obs(logger, 'Validation', ['swvl1'], ['skt'])
    eval_obs_test = utils.evaluator_obs(logger, 'Testing', ['swvl1'], ['skt'])

    time.sleep(1)

    with torch.no_grad():
        model.eval()

        # validation
        pred_prog_inc_all, pred_diag_all, data_prognostic_inc_all, data_diagnostic_all = [], [], [], []
        data_prognostic_obs_inc_all, data_diagnostic_obs_all = [], []

        for i, (data_dynamic, data_prognostic, data_prognostic_inc, data_diagnostic,
                data_static, data_sm_obs_inc, data_temp_obs, data_time) in tqdm(enumerate(val_dataloader),
                                                                                total=len(val_dataloader),
                                                                                smoothing=0.9,
                                                                                postfix="  validation"):

            pred_prog_inc, pred_diag, _, _ = model(data_static.to(device),
                                                   data_dynamic.to(device),
                                                   data_prognostic.to(device),
                                                   data_time.to(device),
                                                   )

            #pred_prog_inc = val_dataset.transform(pred_prog_inc.cpu().numpy(), y_prog_inc_mean, y_prog_inc_std)

            pred_prog_inc_all.append(pred_prog_inc.cpu().numpy())
            pred_diag_all.append(pred_diag.cpu().numpy())
            data_prognostic_inc_all.append(data_prognostic_inc.numpy())
            data_diagnostic_all.append(data_diagnostic.numpy())

            data_prognostic_obs_inc_all.append(data_sm_obs_inc.numpy())
            data_diagnostic_obs_all.append(data_temp_obs.numpy())

        pred_prog_inc_all = np.concatenate(pred_prog_inc_all, axis=0)
        pred_diag_all = np.concatenate(pred_diag_all, axis=0)
        data_prognostic_inc_all = np.concatenate(data_prognostic_inc_all, axis=0)
        data_diagnostic_all = np.concatenate(data_diagnostic_all, axis=0)

        data_prognostic_obs_inc_all = np.concatenate(data_prognostic_obs_inc_all, axis=0)
        data_diagnostic_obs_all = np.concatenate(data_diagnostic_obs_all, axis=0)

        eval_val(pred_prog_inc_all, data_prognostic_inc_all, pred_diag_all, data_diagnostic_all)
        eval_val.get_results()

        eval_obs_val(pred_prog_inc_all[:, :, :, swvl1_idx:swvl1_idx+1],
                     data_prognostic_obs_inc_all,
                     pred_diag_all[:, :, :, skt_idx:skt_idx+1],
                     data_diagnostic_obs_all)
        eval_obs_val.get_results()

        # testing

        pred_prog_inc_all, pred_diag_all, data_prognostic_inc_all, data_diagnostic_all = [], [], [], []
        data_prognostic_obs_inc_all, data_diagnostic_obs_all = [], []

        for i, (data_dynamic, data_prognostic, data_prognostic_inc, data_diagnostic,
                data_static, data_time) in tqdm(enumerate(test_dataloader),
                                                total=len(test_dataloader),
                                                smoothing=0.9,
                                                postfix="  testing"):

            pred_prog_inc, pred_diag, _, _ = model(data_static.to(device),
                                                   data_dynamic.to(device),
                                                   data_prognostic.to(device),
                                                   data_time.to(device),
                                                   )

            #pred_prog_inc = val_dataset.transform(pred_prog_inc.cpu().numpy(), y_prog_inc_mean, y_prog_inc_std)

            pred_prog_inc_all.append(pred_prog_inc.cpu().numpy())
            pred_diag_all.append(pred_diag.cpu().numpy())
            data_prognostic_inc_all.append(data_prognostic_inc.numpy())
            data_diagnostic_all.append(data_diagnostic.numpy())

            data_prognostic_obs_inc_all.append(data_sm_obs_inc.numpy())
            data_diagnostic_obs_all.append(data_temp_obs.numpy())

        pred_prog_inc_all = np.concatenate(pred_prog_inc_all, axis=0)
        pred_diag_all = np.concatenate(pred_diag_all, axis=0)
        data_prognostic_inc_all = np.concatenate(data_prognostic_inc_all, axis=0)
        data_diagnostic_all = np.concatenate(data_diagnostic_all, axis=0)

        data_prognostic_obs_inc_all = np.concatenate(data_prognostic_obs_inc_all, axis=0)
        data_diagnostic_obs_all = np.concatenate(data_diagnostic_obs_all, axis=0)

        eval_test(pred_prog_inc_all, data_prognostic_inc_all, pred_diag_all, data_diagnostic_all)
        eval_test.get_results()

        eval_obs_test(pred_prog_inc_all[:, :, :, swvl1_idx:swvl1_idx+1],
                      data_prognostic_obs_inc_all,
                      pred_diag_all[:, :, :, skt_idx:skt_idx+1],
                      data_diagnostic_obs_all)
        eval_obs_test.get_results()


if __name__ == '__main__':

    args = parse_args()
    config_file = args.config_file

    # read config arguments
    with open(config_file) as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    main(config)

