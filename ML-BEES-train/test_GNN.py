# ------------------------------------------------------------------
# Script for training and validating on EC-Land dataset
# ------------------------------------------------------------------

import sys
import os
#sys.path.append(os.path.join(os.path.dirname(__file__), "model"))
import logging
import yaml
from dataset.EclandGraphDataset import EcDataset
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
from utils import utils
import time
from tqdm import tqdm
import importlib

# ------------------------------------------------------------------

#logging.basicConfig(level=logging.INFO)
np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)
torch.set_float32_matmul_precision("high")
torch.cuda.empty_cache()

# ------------------------------------------------------------------

def import_class(name):
    module = importlib.import_module("model." + name)
    return getattr(module, name)


def main(config):

    # get logger
    logger = utils.get_logger(config, phase='test')

    # fix random seed
    utils.log_string(logger, "fix random seed...")
    utils.fix_seed(config["random_seed"])

    # dataloader
    utils.log_string(logger, "loading validation dataset...")

    val_dataset = EcDataset(
        start_year=config["validation_start"],
        end_year=config["validation_end"],
        x_slice_indices=config["x_slice_indices"],
        root=config["file_path"],
        clim_features=config["clim_feats"],
        dynamic_features=config["dynamic_feats"],
        target_prog_features=config["targets_prog"],
        target_diag_features=config["targets_diag"],
        is_add_lat_lon=config["is_add_lat_lon"],
        is_norm=config["is_norm"],
        graph_type = config["graph_type"],
        d_graph = config["d_graph"],
        max_num_points = config["max_num_points"],
        k_graph = config["k_graph"]
    )

    utils.log_string(logger, "loading testing dataset...")

    test_dataset = EcDataset(
        start_year=config["test_start"],
        end_year=config["test_end"],
        x_slice_indices=config["x_slice_indices"],
        root=config["file_path"],
        clim_features=config["clim_feats"],
        dynamic_features=config["dynamic_feats"],
        target_prog_features=config["targets_prog"],
        target_diag_features=config["targets_diag"],
        is_add_lat_lon=config["is_add_lat_lon"],
        is_norm=config["is_norm"],
        graph_type=config["graph_type"],
        d_graph=config["d_graph"],
        max_num_points=config["max_num_points"],
        k_graph=config["k_graph"]
    )

    val_dataloader = DataLoader(val_dataset,
                                batch_size=config["batch_size"],
                                drop_last=False,
                                shuffle=False,
                                pin_memory=config["pin_memory"],
                                num_workers=config["num_workers"],
                                persistent_workers=config["persistent_workers"]
                                )

    test_dataloader = DataLoader(test_dataset,
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

    # get models
    utils.log_string(logger, "\nbuild the model ...")

    if config["devices"] != -1:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config["devices"])
        device = 'cuda'
    else:
        device = 'cpu'

    model = import_class(config["model"])(in_static=val_dataset.n_static,
                                          in_dynamic=val_dataset.n_dynamic,
                                          in_prog=val_dataset.n_prog,
                                          out_prog=val_dataset.n_prog,
                                          out_diag=val_dataset.n_diag,
                                          hidden_dim=config["hidden_dim"],
                                          dropout=config["dropout"],
                                          mu_norm=y_prog_inc_mean,
                                          std_norm=y_prog_inc_std,
                                          pretrained=config["pretrained"]
                                          )

    utils.log_string(logger, "model parameters ...")
    utils.log_string(logger, "all parameters: %d" % utils.count_parameters(model))

    model.to(device)

    # get edge_indices for GNN
    #edge_index = torch.from_numpy(val_dataset.edge_index).to(device)

    # testing loop
    utils.log_string(logger, 'testing on EC-Land dataset...')

    eval_val = utils.evaluator(logger, 'Validation', val_dataset.target_prog_features, val_dataset.target_diag_features)
    eval_test = utils.evaluator(logger, 'Testing', test_dataset.target_prog_features, test_dataset.target_diag_features)

    time.sleep(1)

    with torch.no_grad():
        model.eval()

        # validation

        pred_prog_inc_all, pred_diag_all, data_prognostic_inc_all, data_diagnostic_all = [], [], [], []

        for i, batch in tqdm(enumerate(val_dataloader),
                             total=len(val_dataloader),
                             smoothing=0.9,
                             postfix="  validation"
                             ):

            (data_dynamic, data_prognostic, data_prognostic_inc, data_diagnostic,
             data_static, data_time, edge_index) = (batch.data_dynamic, batch.data_prognostic, batch.data_prognostic_inc,
                                                    batch.data_diagnostic, batch.data_static, batch.data_time, batch.edge_index)

            pred_prog_inc, pred_diag, _, _ = model(data_static.to(device),
                                                   data_dynamic.to(device),
                                                   data_prognostic.to(device),
                                                   data_time.to(device),
                                                   edge_index.to(device),
                                                   None,
                                                   None
                                                   )

            #pred_prog_inc = val_dataset.transform(pred_prog_inc.cpu().numpy(), y_prog_inc_mean, y_prog_inc_std)

            pred_prog_inc_all.append(pred_prog_inc.cpu().unsqueeze(1).unsqueeze(1).numpy())
            pred_diag_all.append(pred_diag.cpu().unsqueeze(1).unsqueeze(1).numpy())
            data_prognostic_inc_all.append(data_prognostic_inc.unsqueeze(1).unsqueeze(1).numpy())
            data_diagnostic_all.append(data_diagnostic.unsqueeze(1).unsqueeze(1).numpy())

        pred_prog_inc_all = np.concatenate(pred_prog_inc_all, axis=0)

        pred_diag_all = np.concatenate(pred_diag_all, axis=0)
        data_prognostic_inc_all = np.concatenate(data_prognostic_inc_all, axis=0)
        data_diagnostic_all = np.concatenate(data_diagnostic_all, axis=0)

        eval_val(pred_prog_inc_all, data_prognostic_inc_all, pred_diag_all, data_diagnostic_all)
        eval_val.get_results()

        # testing

        pred_prog_inc_all, pred_diag_all, data_prognostic_inc_all, data_diagnostic_all = [], [], [], []

        for i, batch in tqdm(enumerate(test_dataloader),
                             total=len(test_dataloader),
                             smoothing=0.9,
                             postfix="  testing"
                             ):

            (data_dynamic, data_prognostic, data_prognostic_inc, data_diagnostic,
             data_static, data_time, edge_index) = (batch.data_dynamic, batch.data_prognostic, batch.data_prognostic_inc,
                                                    batch.data_diagnostic, batch.data_static, batch.data_time, batch.edge_index)

            pred_prog_inc, pred_diag, _, _ = model(data_static.to(device),
                                                   data_dynamic.to(device),
                                                   data_prognostic.to(device),
                                                   data_time.to(device),
                                                   edge_index.to(device),
                                                   None,
                                                   None
                                                   )

            #pred_prog_inc = val_dataset.transform(pred_prog_inc.cpu().numpy(), y_prog_inc_mean, y_prog_inc_std)

            pred_prog_inc_all.append(pred_prog_inc.cpu().unsqueeze(1).unsqueeze(1).numpy())
            pred_diag_all.append(pred_diag.cpu().unsqueeze(1).unsqueeze(1).numpy())
            data_prognostic_inc_all.append(data_prognostic_inc.unsqueeze(1).unsqueeze(1).numpy())
            data_diagnostic_all.append(data_diagnostic.unsqueeze(1).unsqueeze(1).numpy())

        pred_prog_inc_all = np.concatenate(pred_prog_inc_all, axis=0)
        pred_diag_all = np.concatenate(pred_diag_all, axis=0)
        data_prognostic_inc_all = np.concatenate(data_prognostic_inc_all, axis=0)
        data_diagnostic_all = np.concatenate(data_diagnostic_all, axis=0)

        eval_test(pred_prog_inc_all, data_prognostic_inc_all, pred_diag_all, data_diagnostic_all)
        eval_test.get_results()




if __name__ == '__main__':

    # TODO add different config files
    # read config arguments
    with open(r'configs/config.yaml') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    main(config)

