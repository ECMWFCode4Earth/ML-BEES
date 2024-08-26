# ------------------------------------------------------------------
# Training an MLP model on observations
# Script for training and validating on EC-Land dataset
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
    logger = utils.get_logger(config)

    # save config file
    utils.save_config(config)

    # get tensorboard writer
    writer = SummaryWriter(os.path.join(config["logging"]["location"], config["logging"]["name"]))

    # fix random seed
    utils.log_string(logger, "fix random seed...")
    utils.fix_seed(config["random_seed"])

    # initialize the dataset class and dataloader
    utils.log_string(logger, "loading training dataset...")

    train_dataset = EcObsDataset(
        start_year=config["training_start"],
        end_year=config["training_end"],
        x_slice_indices=config["x_slice_indices"],
        root=config["file_path"],
        root_sm=config["smap_file"],
        root_temp=config["modis_temp_file"],
        use_time_var_lai=config["use_time_var_lai"],
        root_lail=config["lail_file"],
        root_laih=config["laih_file"],
        roll_out=config["roll_out"],
        clim_features=config["clim_feats"],
        dynamic_features=config["dynamic_feats"],
        target_prog_features=config["targets_prog"],
        target_diag_features=config["targets_diag"],
        is_add_lat_lon=config["is_add_lat_lon"],
        is_norm=config["is_norm"],
        point_dropout=config["point_dropout"]
        )

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

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=config["batch_size"],
                                                   shuffle=True,
                                                   pin_memory=config["pin_memory"],
                                                   num_workers=config["num_workers"],
                                                   persistent_workers=config["persistent_workers"]
                                                   )

    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=config["batch_size"],
                                                 drop_last=False,
                                                 shuffle=False,
                                                 pin_memory=config["pin_memory"],
                                                 num_workers=config["num_workers"],
                                                 persistent_workers=config["persistent_workers"]
                                                 )

    utils.log_string(logger, "# training samples: %d" % len(train_dataset))
    utils.log_string(logger, "# evaluation samples: %d" % len(val_dataset))

    # get statistic to normalize the output delta x
    logging.info("prepare mean and std for delta x...")
    y_prog_inc_mean = train_dataset.y_prog_inc_mean
    y_prog_inc_std = train_dataset.y_prog_inc_std

    # get indices for swvl1 and skt
    swvl1_idx = train_dataset.target_prog_features.index('swvl1')
    skt_idx = train_dataset.target_diag_features.index('skt')

    # get models
    utils.log_string(logger, "\nbuild the model ...")

    if config["devices"] != -1:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config["devices"])
        device = 'cuda'
    else:
        device = 'cpu'

    model = MLP_Obs(in_static=train_dataset.n_static,
                    in_dynamic=train_dataset.n_dynamic,
                    in_prog=train_dataset.n_prog,
                    out_prog=train_dataset.n_prog,
                    out_diag=train_dataset.n_diag,
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

    # get optimizer
    utils.log_string(logger, "get optimizer and learning rate scheduler...")
    optimizer = utils.get_optimizer([x for x in model.parameters() if x.requires_grad], config)
    lr_scheduler = utils.get_learning_scheduler(optimizer, config)

    model.to(device)

    # training loop
    utils.log_string(logger, 'training on EC-Land dataset...')

    # initialize the evaluation class
    eval_train = utils.evaluator(logger, 'Training', train_dataset.target_prog_features, train_dataset.target_diag_features)
    eval_val = utils.evaluator(logger, 'Validation', val_dataset.target_prog_features, val_dataset.target_diag_features)

    eval_obs_train = utils.evaluator_obs(logger, 'Training', ['swvl1'], ['skt'])
    eval_obs_val = utils.evaluator_obs(logger, 'Validation', ['swvl1'], ['skt'])

    time.sleep(1)

    # initialize the best values
    best_loss_train = np.inf
    best_loss_val = np.inf

    for epoch in range(config["n_epochs"]):
        utils.log_string(logger, '################# Epoch (%s/%s) #################' % (epoch + 1, config["n_epochs"]))

        # train
        model.train()
        loss_train = 0

        time.sleep(1)

        for i, (data_dynamic, data_prognostic, data_prognostic_inc, data_diagnostic,
                data_static, data_sm_obs_inc, data_temp_obs, data_time) in tqdm(enumerate(train_dataloader),
                                                                                total=len(train_dataloader),
                                                                                smoothing=0.9,
                                                                                postfix="  training"):

            optimizer.zero_grad(set_to_none=True)

            pred_prog_inc, pred_diag, loss_prog, loss_diag = model(data_static.to(device),
                                                                   data_dynamic.to(device),
                                                                   data_prognostic.to(device),
                                                                   data_time.to(device),
                                                                   data_prognostic_inc.to(device),
                                                                   data_diagnostic.to(device),
                                                                   data_sm_obs_inc.to(device),
                                                                   data_temp_obs.to(device)
                                                                   )

            loss = loss_prog + loss_diag
            loss.backward()
            optimizer.step()

            loss_train += loss.item()

            eval_train(pred_prog_inc.detach().cpu().numpy(),
                       data_prognostic_inc.cpu().numpy(),
                       pred_diag.detach().cpu().numpy(),
                       data_diagnostic.cpu().numpy()
                       )

            eval_obs_train(pred_prog_inc[:, :, :, swvl1_idx:swvl1_idx+1].detach().cpu().numpy(),
                           data_sm_obs_inc.cpu().numpy(),
                           pred_diag[:, :, :, skt_idx:skt_idx+1].detach().cpu().numpy(),
                           data_temp_obs.cpu().numpy()
                           )

        mean_loss_train = loss_train / float(len(train_dataset))
        eval_train.get_results()
        eval_obs_train.get_results()

        utils.log_string(logger, 'Training mean loss      : %.6f' % mean_loss_train)
        utils.log_string(logger, 'Training best mean loss : %.6f' % best_loss_train)

        if mean_loss_train <= best_loss_train:
            best_loss_train = mean_loss_train

        #utils.save_model(model, optimizer, epoch, logger, config, 'train')

        # validation
        with torch.no_grad():
            optimizer.zero_grad(set_to_none=True)

            model.eval()
            loss_val = 0

            time.sleep(1)

            for i, (data_dynamic, data_prognostic, data_prognostic_inc, data_diagnostic,
                    data_static, data_sm_obs_inc, data_temp_obs, data_time) in tqdm(enumerate(val_dataloader),
                                                                                    total=len(val_dataloader),
                                                                                    smoothing=0.9,
                                                                                    postfix="  validation"):

                pred_prog_inc, pred_diag, loss_prog, loss_diag = model(data_static.to(device),
                                                                       data_dynamic.to(device),
                                                                       data_prognostic.to(device),
                                                                       data_time.to(device),
                                                                       data_prognostic_inc.to(device),
                                                                       data_diagnostic.to(device),
                                                                       data_sm_obs_inc.to(device),
                                                                       data_temp_obs.to(device)
                                                                       )

                loss = loss_prog + loss_diag
                loss_val += loss.item()

                eval_val(pred_prog_inc.cpu().numpy(),
                         data_prognostic_inc.cpu().numpy(),
                         pred_diag.cpu().numpy(),
                         data_diagnostic.cpu().numpy()
                         )

                eval_obs_val(pred_prog_inc[:, :, :, swvl1_idx:swvl1_idx + 1].cpu().numpy(),
                             data_sm_obs_inc.cpu().numpy(),
                             pred_diag[:, :, :, skt_idx:skt_idx + 1].cpu().numpy(),
                             data_temp_obs.cpu().numpy()
                             )

            mean_loss_val = loss_val / float(len(val_dataloader))
            eval_val.get_results()
            eval_obs_val.get_results()

            utils.log_string(logger, 'Validation mean loss      : %.6f' % mean_loss_val)
            utils.log_string(logger, 'Validation best mean loss : %.6f' % best_loss_val)

            if mean_loss_val <= best_loss_val:
                best_loss_val = mean_loss_val
                utils.save_model(model, optimizer, epoch, logger, config, 'loss')

        writer.add_scalars("loss", {'train': mean_loss_train, 'val': mean_loss_val}, epoch + 1)

        eval_train.reset()
        eval_val.reset()

        lr_scheduler.step_update(epoch)


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


