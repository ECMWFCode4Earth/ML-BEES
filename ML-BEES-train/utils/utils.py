# ------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------

import torch
import numpy as np
import random
import os
import datetime
import logging
import yaml
from timm.scheduler.step_lr import StepLRScheduler
from timm.scheduler.cosine_lr import CosineLRScheduler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)

# ------------------------------------------------------------------

def log_string(logger, str):
    logger.info(str)
    #print(str)


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_dir(dir):
    os.makedirs(dir, exist_ok=True)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_logger(config, phase='train'):
    # Set Logger and create Directories

    if config["logging"]["name"] is None or len(config["logging"]["name"]) == 0:
        config["logging"]["name"] = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    dir_log = os.path.join(config["logging"]["location"], config["logging"]["name"])
    make_dir(dir_log)

    if phase == 'train':
        checkpoints_dir = os.path.join(dir_log, 'model_checkpoints/')
        make_dir(checkpoints_dir)

    if phase == 'train':
        logger = logging.getLogger("Trainer")
    else:
        logger = logging.getLogger("Logger")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/log_file.txt' % dir_log)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def save_config(config):

    dir_log = os.path.join(config["logging"]["location"], config["logging"]["name"])
    with open(os.path.join(dir_log, "config.yaml"), 'w') as file:
        yaml.dump(config, file)

def get_optimizer(optim_groups, config):

    optim = config["optimizer"]

    # TODO add more optimizers

    if optim == 'Adam':
        optimizer = torch.optim.Adam(optim_groups, betas=(config["beta1"], config["beta2"]), lr=config["lr"], weight_decay=config["weight_decay"])
    elif optim == 'AdamW':
        optimizer = torch.optim.AdamW(optim_groups, betas=(config["beta1"], config["beta2"]), lr=config["lr"], weight_decay=config["weight_decay"])
    else:
        raise ValueError('Unexpected optimizer {} supported optimizers are Adam and AdamW'.format(config["optimizer"]))

    return optimizer


def get_learning_scheduler(optimizer, config):
    lr_scheduler = config["lr_scheduler"]

    if lr_scheduler == 'step':
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=config["lr_decay_step"],
            decay_rate=config["lr_decay_rate"],
            warmup_lr_init=config["lr_warmup"],
            warmup_t=config["lr_warmup_epochs"],
            t_in_epochs=True,
        )

    elif lr_scheduler == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=config["n_epochs"],
            cycle_mul=1.,
            lr_min=config["lr_min"],
            warmup_lr_init=config["lr_warmup"],
            warmup_t=config["lr_warmup_epochs"],
            cycle_limit=1,
            t_in_epochs=False,
            warmup_prefix=False
        )

    else:
        raise ValueError('Unexpected scheduler {}, supported scheduler is step or cosine'.format(config["lr_scheduler"]))

    return lr_scheduler


def save_model(model, optimizer, epoch, logger, config, metric):

    dir_log = os.path.join(config["logging"]["location"], config["logging"]["name"])
    checkpoints_dir = os.path.join(dir_log, 'model_checkpoints/')
    if metric == 'loss':
        path = os.path.join(checkpoints_dir, 'best_loss_model.pth')
        log_string(logger, 'saving model to %s' % path)

    elif metric == 'train':
        path = os.path.join(checkpoints_dir, 'train_model_epoch_{}.pth'.format(epoch))
    else:
        raise ValueError('Unexpected metric {}, supported metric is loss or train'.format(metric))

    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }

    torch.save(state, path)

def calc_R2(pred, target):
    target_hat = np.mean(target)
    residuals_sum = np.sum((target - pred) ** 2)
    total_sum = np.sum((target - target_hat) ** 2)
    R2 = 1 - (residuals_sum / total_sum)
    return R2


def r2_score_multi(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Calculated the r-squared score between 2 arrays of values

    Args:
        y_pred (np.ndarray): Predicted values
        y_true (np.ndarray): Target values

    Returns:
         (float) r-squared score
    """
    return r2_score(y_pred.flatten(), y_true.flatten(), force_finite=False)

class evaluator():
    def __init__(self, logger, mode, target_prog, target_diag):

        self.mode = mode
        self.logger = logger

        self.target_prog = target_prog
        self.target_diag = target_diag
        self.n_classes_prog = len(target_prog)
        self.n_classes_diag = len(target_diag)

        self.seen_iter = 0
        self.prog_mae, self.prog_rmse, self.prog_r2 = 0, 0, 0
        self.diag_mae, self.diag_rmse, self.diag_r2 = 0, 0, 0
        self.prog_diag_mae, self.prog_diag_rmse, self.prog_diag_r2 = 0, 0, 0

        self.classes_prog_mae = [0 for _ in range(self.n_classes_prog)]
        self.classes_prog_rmse = [0 for _ in range(self.n_classes_prog)]
        self.classes_prog_r2 = [0 for _ in range(self.n_classes_prog)]
        self.classes_diag_mae = [0 for _ in range(self.n_classes_diag)]
        self.classes_diag_r2 = [0 for _ in range(self.n_classes_diag)]
        self.classes_diag_rmse = [0 for _ in range(self.n_classes_diag)]


    def get_results(self):

        self.prog_mae = self.prog_mae / float(self.seen_iter)
        self.prog_rmse = self.prog_rmse / float(self.seen_iter)
        self.prog_r2 = self.prog_r2 / float(self.seen_iter)
        self.diag_mae = self.diag_mae / float(self.seen_iter)
        self.diag_rmse = self.diag_rmse / float(self.seen_iter)
        self.diag_r2 = self.diag_r2 / float(self.seen_iter)

        self.prog_diag_mae = self.prog_diag_mae / float(self.seen_iter)
        self.prog_diag_rmse = self.prog_diag_rmse / float(self.seen_iter)
        self.prog_diag_r2 = self.prog_diag_r2 / float(self.seen_iter)

        for label in range(self.n_classes_prog):
            self.classes_prog_mae[label] = self.classes_prog_mae[label] / float(self.seen_iter)
            self.classes_prog_rmse[label] = self.classes_prog_rmse[label] / float(self.seen_iter)
            self.classes_prog_r2[label] = self.classes_prog_r2[label] / float(self.seen_iter)

        for label in range(self.n_classes_diag):
            self.classes_diag_mae[label] = self.classes_diag_mae[label] / float(self.seen_iter)
            self.classes_diag_rmse[label] = self.classes_diag_rmse[label] / float(self.seen_iter)
            self.classes_diag_r2[label] = self.classes_diag_r2[label] / float(self.seen_iter)

        message = '-----------------   %s   -----------------\n' % self.mode

        for label in range(self.n_classes_prog):
            message += 'class prog - %s   MAE: %.4f, RMSE: %.4f, R2: %.4f \n' % (
                self.target_prog[label] + ' ' * (7 - len(self.target_prog[label])),
                self.classes_prog_mae[label],
                self.classes_prog_rmse[label],
                self.classes_prog_r2[label])

        message += '\n'

        for label in range(self.n_classes_diag):
            message += 'class diag - %s   MAE: %.4f, RMSE: %.4f, R2: %.4f \n' % (
                self.target_diag[label] + ' ' * (7 - len(self.target_diag[label])),
                self.classes_diag_mae[label],
                self.classes_diag_rmse[label],
                self.classes_diag_r2[label])

        message += '\n'

        if self.n_classes_prog > 1:

            message += 'class prog             MAE: %.4f, RMSE: %.4f, R2: %.4f \n' % (
                self.prog_mae,
                self.prog_rmse,
                self.prog_r2)

        if self.n_classes_diag > 1:

            message += 'class diag             MAE: %.4f, RMSE: %.4f, R2: %.4f \n' % (
                self.diag_mae,
                self.diag_rmse,
                self.diag_r2)

        message += 'class prog + diag      MAE: %.4f, RMSE: %.4f, R2: %.4f \n' % (
            self.prog_diag_mae,
            self.prog_diag_rmse,
            self.prog_diag_r2)

        log_string(self.logger, message)

    def reset(self):

        self.seen_iter = 0
        self.prog_mae, self.prog_rmse, self.prog_r2 = 0, 0, 0
        self.diag_mae, self.diag_rmse, self.diag_r2 = 0, 0, 0
        self.prog_diag_mae, self.prog_diag_rmse, self.prog_diag_r2 = 0, 0, 0

        self.classes_prog_mae = [0 for _ in range(self.n_classes_prog)]
        self.classes_prog_rmse = [0 for _ in range(self.n_classes_prog)]
        self.classes_prog_r2 = [0 for _ in range(self.n_classes_prog)]
        self.classes_diag_mae = [0 for _ in range(self.n_classes_diag)]
        self.classes_diag_r2 = [0 for _ in range(self.n_classes_diag)]
        self.classes_diag_rmse = [0 for _ in range(self.n_classes_diag)]

    def __call__(self, pred_prog, target_prog, pred_diag, target_diag):

        self.seen_iter += 1

        if self.n_classes_prog > 1:
            self.prog_mae += np.mean(np.abs(pred_prog - target_prog))
            self.prog_rmse += np.sqrt(np.mean((pred_prog - target_prog) ** 2))
            self.prog_r2 += r2_score_multi(pred_prog, target_prog)

        if self.n_classes_diag > 1:
            self.diag_mae += np.mean(np.abs(pred_diag - target_diag))
            self.diag_rmse += np.sqrt(np.mean((pred_diag - target_diag) ** 2))
            self.diag_r2 += r2_score_multi(pred_diag, target_diag)

        pred_prog_diag = np.concatenate((pred_prog, pred_diag), axis=-1).flatten()
        target_prog_diag = np.concatenate((target_prog, target_diag), axis=-1).flatten()

        #self.prog_diag_mae += np.mean(np.abs(pred_prog_diag - target_prog_diag))
        #self.prog_diag_rmse += np.sqrt(np.mean((pred_prog_diag - target_prog_diag) ** 2))
        self.prog_diag_mae += mean_absolute_error(pred_prog_diag, target_prog_diag)
        self.prog_diag_rmse += np.sqrt(mean_squared_error(pred_prog_diag, target_prog_diag))
        self.prog_diag_r2 += r2_score(pred_prog_diag, target_prog_diag)

        for label in range(self.n_classes_prog):
            self.classes_prog_mae[label] += np.mean(np.abs(pred_prog[:, :, :, label] - target_prog[:, :, :, label]))
            self.classes_prog_rmse[label] += np.sqrt(np.mean((pred_prog[:, :, :, label] - target_prog[:, :, :, label]) ** 2))
            self.classes_prog_r2[label] += r2_score_multi(pred_prog[:, :, :, label], target_prog[:, :, :, label])

        for label in range(self.n_classes_diag):
            self.classes_diag_mae[label] += np.mean(np.abs(pred_diag[:, :, :, label] - target_diag[:, :, :, label]))
            self.classes_diag_rmse[label] += np.sqrt(np.mean((pred_diag[:, :, :, label] - target_diag[:, :, :, label]) ** 2))
            self.classes_diag_r2[label] += r2_score_multi(pred_diag[:, :, :, label], target_diag[:, :, :, label])


class evaluator_obs():
    def __init__(self, logger, mode, target_prog, target_diag):

        self.mode = mode
        self.logger = logger

        self.target_prog = target_prog
        self.target_diag = target_diag
        self.n_classes_prog = len(target_prog)
        self.n_classes_diag = len(target_diag)

        self.seen_iter = 0
        self.prog_mae, self.prog_rmse, self.prog_r2 = 0, 0, 0
        self.diag_mae, self.diag_rmse, self.diag_r2 = 0, 0, 0

        self.classes_prog_mae = [0 for _ in range(self.n_classes_prog)]
        self.classes_prog_rmse = [0 for _ in range(self.n_classes_prog)]
        self.classes_prog_r2 = [0 for _ in range(self.n_classes_prog)]
        self.classes_diag_mae = [0 for _ in range(self.n_classes_diag)]
        self.classes_diag_r2 = [0 for _ in range(self.n_classes_diag)]
        self.classes_diag_rmse = [0 for _ in range(self.n_classes_diag)]


    def get_results(self):

        self.prog_mae = self.prog_mae / float(self.seen_iter)
        self.prog_rmse = self.prog_rmse / float(self.seen_iter)
        self.prog_r2 = self.prog_r2 / float(self.seen_iter)
        self.diag_mae = self.diag_mae / float(self.seen_iter)
        self.diag_rmse = self.diag_rmse / float(self.seen_iter)
        self.diag_r2 = self.diag_r2 / float(self.seen_iter)

        for label in range(self.n_classes_prog):
            self.classes_prog_mae[label] = self.classes_prog_mae[label] / float(self.seen_iter)
            self.classes_prog_rmse[label] = self.classes_prog_rmse[label] / float(self.seen_iter)
            self.classes_prog_r2[label] = self.classes_prog_r2[label] / float(self.seen_iter)

        for label in range(self.n_classes_diag):
            self.classes_diag_mae[label] = self.classes_diag_mae[label] / float(self.seen_iter)
            self.classes_diag_rmse[label] = self.classes_diag_rmse[label] / float(self.seen_iter)
            self.classes_diag_r2[label] = self.classes_diag_r2[label] / float(self.seen_iter)

        message = '-----------------   %s   -----------------\n' % self.mode

        for label in range(self.n_classes_prog):
            message += 'class prog obs - %s   MAE: %.4f, RMSE: %.4f, R2: %.4f \n' % (
                self.target_prog[label] + ' ' * (7 - len(self.target_prog[label])),
                self.classes_prog_mae[label],
                self.classes_prog_rmse[label],
                self.classes_prog_r2[label])

        message += '\n'

        for label in range(self.n_classes_diag):
            message += 'class diag obs - %s   MAE: %.4f, RMSE: %.4f, R2: %.4f \n' % (
                self.target_diag[label] + ' ' * (7 - len(self.target_diag[label])),
                self.classes_diag_mae[label],
                self.classes_diag_rmse[label],
                self.classes_diag_r2[label])

        message += '\n'

        if self.n_classes_prog > 1:

            message += 'class prog obs             MAE: %.4f, RMSE: %.4f, R2: %.4f \n' % (
                self.prog_mae,
                self.prog_rmse,
                self.prog_r2)
        if self.n_classes_diag > 1:
            message += 'class diag obs             MAE: %.4f, RMSE: %.4f, R2: %.4f' % (
                self.diag_mae,
                self.diag_rmse,
                self.diag_r2)

        log_string(self.logger, message)

    def reset(self):

        self.seen_iter = 0
        self.prog_mae, self.prog_rmse, self.prog_r2 = 0, 0, 0
        self.diag_mae, self.diag_rmse, self.diag_r2 = 0, 0, 0
        self.prog_diag_mae, self.prog_diag_rmse, self.prog_diag_r2 = 0, 0, 0

        self.classes_prog_mae = [0 for _ in range(self.n_classes_prog)]
        self.classes_prog_rmse = [0 for _ in range(self.n_classes_prog)]
        self.classes_prog_r2 = [0 for _ in range(self.n_classes_prog)]
        self.classes_diag_mae = [0 for _ in range(self.n_classes_diag)]
        self.classes_diag_r2 = [0 for _ in range(self.n_classes_diag)]
        self.classes_diag_rmse = [0 for _ in range(self.n_classes_diag)]

    def __call__(self, pred_prog, target_prog, pred_diag, target_diag):

        pred_prog_valid, target_prog_valid = [], []
        for c in range(self.n_classes_prog):
            indices = ~np.isnan(target_prog[:, :, :, c])
            target_prog_valid.append(target_prog[:, :, :, c][indices][:, None])
            pred_prog_valid.append(pred_prog[:, :, :, c][indices][:, None])

        target_prog_valid = np.concatenate(target_prog_valid, axis=-1)
        pred_prog_valid = np.concatenate(pred_prog_valid, axis=-1)

        pred_diag_valid, target_diag_valid = [], []
        for c in range(self.n_classes_diag):
            indices = ~np.isnan(target_diag[:, :, :, c])
            target_diag_valid.append(target_diag[:, :, :, c][indices][:, None])
            pred_diag_valid.append(pred_diag[:, :, :, c][indices][:, None])

        target_diag_valid = np.concatenate(target_diag_valid, axis=-1)
        pred_diag_valid = np.concatenate(pred_diag_valid, axis=-1)

        if len(target_prog_valid) > 0 and len(target_diag_valid) > 0:
            if self.n_classes_prog > 1:
                self.prog_mae += np.mean(np.abs(pred_prog_valid - target_prog_valid))
                self.prog_rmse += np.sqrt(np.mean((pred_prog_valid - target_prog_valid) ** 2))
                self.prog_r2 += r2_score_multi(pred_prog_valid, target_prog_valid)
            if self.n_classes_diag > 1:
                self.diag_mae += np.mean(np.abs(pred_diag_valid - target_diag_valid))
                self.diag_rmse += np.sqrt(np.mean((pred_diag_valid - target_diag_valid) ** 2))
                self.diag_r2 += r2_score_multi(pred_diag_valid, target_diag_valid)

            for label in range(self.n_classes_prog):
                self.classes_prog_mae[label] += np.mean(np.abs(pred_prog_valid[:, label] - target_prog_valid[:, label]))
                self.classes_prog_rmse[label] += np.sqrt(np.mean((pred_prog_valid[:, label] - target_prog_valid[:, label]) ** 2))
                self.classes_prog_r2[label] += r2_score_multi(pred_prog_valid[:, label], target_prog_valid[:, label])

            for label in range(self.n_classes_diag):
                self.classes_diag_mae[label] += np.mean(np.abs(pred_diag_valid[:, label] - target_diag_valid[:, label]))
                self.classes_diag_rmse[label] += np.sqrt(np.mean((pred_diag_valid[:, label] - target_diag_valid[:, label]) ** 2))
                self.classes_diag_r2[label] += r2_score_multi(pred_diag_valid[:, label], target_diag_valid[:, label])

            self.seen_iter += 1

