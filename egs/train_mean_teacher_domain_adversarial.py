import argparse
import os
import torch
from torch import nn
import pytorch_lightning as pl
import scipy
from collections import OrderedDict
import pandas as pd
from pathlib import Path
from SED.baseline_tools.evaluation_measures import compute_sed_eval_metrics
import yaml
from pytorch_lightning.logging import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
from copy import deepcopy
from local.data_utils.strongsetsimple import StrongSetSimpleUnlabeled, StrongSetSimpleLabeled
from local.data_utils.weakset import WeakSetUnlabeled, WeakSetLabeled
from local.data_utils.unlabeledset import UnlabeledSet
from local.data_utils.multidataset import MultiDataset
from local.data_utils.concatdataset import MultiStreamBatchSampler, ConcatDataset
from local.data import data_init
from local.data_utils.label_hashtable import label_hashtable
from local.data_utils.valset import ValSet
from SED.baseline_tools.utilities import ramps
import numpy as np
from SED.baseline_tools.utilities.ManyHotEncoder import ManyHotEncoder
from SED.baseline_tools.models.TCN import TCNAdv

parser = argparse.ArgumentParser("end to end separation without retraining")
parser.add_argument("conf_file")
parser.add_argument("log_dir")
parser.add_argument("gpus")


class DomainAdversarialNet(nn.Module):

    def __init__(self, in_feats=64):
        super(DomainAdversarialNet, self).__init__()
        self.net = TCNAdv(in_feats, 1, 4, 2, 64, 128, 3)

    def forward(self, x):

        return self.net(x)


class FixMatch(pl.LightningModule):

    def __init__(self, hparams):
        super(FixMatch, self).__init__()
        self.config = hparams # avoid pytorch-lightning hparams logging

        bce = nn.BCELoss(reduction="none")
        bce_logits = nn.BCEWithLogitsLoss(reduction="none")
        mse = nn.MSELoss(reduction="none")

        ### defining the losses ###
        from SED.losses.weightedBCE import WeightedBCE
        from SED.losses.lovasz import sed_lovasz
        self.weighted_strong = lambda x, y: bce_logits(x, y).mean(dim=[d for d in range(len(x.shape)) if d!= 0]) #lambda x, y: sed_lovasz(x, y) #
        self.weighted_weak = lambda x, y: bce(x, y).mean(dim=[d for d in range(len(x.shape)) if d!= 0]) #WeightedBCE(weights=[4.17, 6.56, 3.66, 2.53, 3.18, 7.38,  8.25, 5.38, 1, 6.49], base_loss="bce")
        self.strong_criterion = lambda x, y: bce_logits(x, y).mean(dim=[d for d in range(len(x.shape)) if d!= 0])
        self.weak_criterion = lambda x, y: bce(x, y).mean(dim=[d for d in range(len(x.shape)) if d!= 0])
        self.consistency_criterion = lambda x, y: mse(x, y).mean(dim=[d for d in range(len(x.shape)) if d!= 0])

        self.adversarial_criterion = lambda x, y: bce(x, y).mean()

        ############################
        # network CRNN init
        ############################

        n_layers = 7
        crnn_kwargs = {"n_in_channel": 1, "nclass": self.config["data"]["n_classes"], "attention": True, "n_RNN_cell": 128,
                       "n_layers_RNN": 2,
                       "activation": "glu", "rnn_type": "BGRU",
                       "dropout": 0.5,
                       "kernel_size": n_layers * [3], "padding": n_layers * [1], "stride": n_layers * [1],
                       "nb_filters": [16, 32, 64, 128, 128, 128, 128],
                       "pooling": [[2, 2], [2, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2]]}
          # 2 * 2
        from SED.baseline_tools.baseline import SEDBaseline
        self.student = SEDBaseline(self.config, False, adversarial=True, use_init=False, use_pcen=False) # student here is simply ema of teacher
        self.teacher = SEDBaseline(self.config, False, adversarial=True, use_init=False, use_pcen=False)

        self.adversarial_net = DomainAdversarialNet(128) # at same branch as RNN

        train_synth, weak, unlabeled, unlabeled_others, rirs, backgrounds, validation = data_init(self.config)

        ##### defining dataloaders #####
        self.encoder = ManyHotEncoder(list(label_hashtable.keys()), self.config["feats"]["max_len"] // self.config["net"]["pool_factor"])

        #synth_labeled = StrongSetSimpleLabeled(train_synth, self.config, encoder=self.encoder, time_augment=False,
                                             #  backgrounds=backgrounds, rirs=rirs)

        synth_unlabeled = StrongSetSimpleUnlabeled(train_synth, self.config, self.encoder, time_augment=True,
                                                   backgrounds=backgrounds, rirs=rirs, as_labelled=True)

        #weak_labeled_data = WeakSetLabeled(weak, self.config, self.encoder, time_augment=False)
        weak_unlabeled = WeakSetUnlabeled(weak, self.config, self.encoder, backgrounds=backgrounds, rirs=rirs, as_labelled=True, time_augment=False)

        unlabeled_in_domain = UnlabeledSet(unlabeled, self.config, backgrounds=backgrounds,
                                           rirs=rirs, time_augment=False)  # WeakSetLabeled(weak, confs, encoder = encoder, time_augment=True)
        #unlabeled_others = UnlabeledSet(unlabeled_others, self.config, backgrounds=backgrounds, rirs=rirs)

        tot_data = ConcatDataset([unlabeled_in_domain, weak_unlabeled, synth_unlabeled])

        self.train_set = tot_data # training
        self.val_set = ValSet(validation, self.config, self.encoder)

        ## vars for schedules and metrics
        self.buffer_dataframe_valid_student = pd.DataFrame()
        self.buffer_dataframe_valid_teacher = pd.DataFrame()

        #self.th = 0.95
        self.tot_step = 1
        self.consistency_weight = 0

        self.adv_buffer = None
        self.target_domain_buffer = None

    def forward(self, *args, **kwargs):
        pass

    def adjust_learning_rate(self, lr, optimizer):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def update_ema(self, alpha, global_step):
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_params, params in zip(self.student.parameters(), self.teacher.parameters()):
            ema_params.data.mul_(alpha).add_(1 - alpha, params.data)

    def apply_median(self, predictions):

        device = predictions.device
        predictions = predictions.cpu().detach().numpy()
        for batch in range(predictions.shape[0]):
            predictions[batch, ...] = scipy.ndimage.filters.median_filter(predictions[batch, ...], (self.config["training"]["median_window"], 1))

        return torch.from_numpy(predictions).float().to(device)

    def training_step(self, batch, batch_idx, optimizer_idx):

        if optimizer_idx == 0:

            ## standard mean teacher code + maximize loss of domain classifier

            self.tot_step += 1

            if self.current_epoch == 0 and batch_idx == 0:
                # init lr
                tot_steps = len(self.train_dataloader()) * self.config["training"]["n_epochs_lr_schedule"] // \
                            self.config["training"]["accumulate_batches"]
                rampup = ramps.exp_rampup(0, tot_steps)
                lr_main = self.config["opt"]["lr_main"] * rampup
                self.consistency_weight = self.config["training"]["const_max"] * ramps.exp_rampup(0, tot_steps)
                self.adjust_learning_rate(lr_main, self.opt1)
                lr_adv = self.config["opt"]["lr_adv"] * rampup
                self.adjust_learning_rate(lr_adv, self.opt2)

            for p in self.student.parameters():  # we don't update the ema model directly
                p.detach()

            # if we use accumulation we must take care of how is updated the ema model.

            weak_augm_feats, strong_augm_feats, strong, weak, strong_mask, weak_mask = batch
            strong_mask = strong_mask.squeeze(-1)
            weak_mask = weak_mask.squeeze(-1)
            #unlabel_mask = ~((strong_mask + weak_mask).bool())

            # we get domain labels from masks we have weak domain synthetic domain and unlabel domain
            # strong is synth domain label 0 (weak and unlabel same domain) --> strong mask is my labels easy !!


            strong_weak_augm_teach, weak_weak_augm_teach, logits_weak_augm_teach, adv = self.teacher(
                weak_augm_feats.unsqueeze(1))  # weak augmented predictions

            adv_logits = self.adversarial_net(adv)
            self.adv_buffer = adv
            self.target_domain_buffer = strong_mask.float()

            loss_adv = -1*self.adversarial_criterion(torch.sigmoid(adv_logits), self.target_domain_buffer)*self.config["training"]["lambda_adv"] # maximize

            # labeled loss
            loss = loss_adv # ADD to lightining hooks.py

            strong_loss_teacher = 0
            if strong_mask.any():  # we have strong examples
                strong_loss_teacher = (
                    self.weighted_strong(logits_weak_augm_teach[strong_mask], strong[strong_mask])).sum()
                loss += strong_loss_teacher / torch.sum(strong_mask)

            weak_loss_teacher = 0
            if weak_mask.any():  # we have weak examples
                weak_loss_teacher = (self.weighted_weak(weak_weak_augm_teach[weak_mask], weak[weak_mask])).sum()
                loss += weak_loss_teacher / torch.sum(weak_mask)

            strong_strong_augm, weak_strong_augm, logits_strong_augm, _ = self.student(
                strong_augm_feats.unsqueeze(1))  # strong augmented prediction
            strong_strong_augm = strong_strong_augm.detach()
            weak_strong_augm = weak_strong_augm.detach()

            consistency_strong = self.consistency_criterion(strong_weak_augm_teach, strong_strong_augm).mean()
            consistency_weak = self.consistency_criterion(weak_weak_augm_teach, weak_strong_augm).mean()

            consistency_tot = self.consistency_weight * (consistency_strong + consistency_weak)
            loss += consistency_tot

            # consistency loss
            if self.tot_step % self.config["training"]["accumulate_batches"] == 0:
                real_step = self.tot_step // self.config["training"]["accumulate_batches"]
                self.update_ema(0.999, real_step + 1)
                tot_steps = len(self.train_dataloader()) * self.config["training"]["n_epochs_lr_schedule"] // \
                            self.config["training"]["accumulate_batches"]

                rampup = ramps.exp_rampup(real_step, tot_steps)
                lr_main = self.config["opt"]["lr_main"] * rampup
                self.consistency_weight = self.config["training"]["const_max"] * ramps.exp_rampup(real_step, tot_steps)
                self.adjust_learning_rate(lr_main, self.opt1)
                lr_adv = self.config["opt"]["lr_adv"] * rampup
                self.adjust_learning_rate(lr_adv, self.opt2)

            tqdm_dict = {'loss_train': loss}  ### NOTE before the first accumulate steps loss will be NaN don't panic

            tensorboard_logs = {'train_batch_loss': loss,
                                'strong_loss_teacher': strong_loss_teacher,
                                'weak_loss_teacher': weak_loss_teacher,
                                'lr': self.opt1.param_groups[-1]["lr"],
                                'consistency_strong': consistency_strong,
                                'consistency_weak': consistency_weak,
                                'consistency_tot': consistency_tot,
                                'consistency_weight': self.consistency_weight,
                                'loss_adv': loss_adv
                                }

            output = OrderedDict({
                'loss': loss,
                'progress_bar': tqdm_dict,
                'log': tensorboard_logs
            })
            return output

        elif optimizer_idx == 1: # we train the domain classifier adversarially


            adv_logits = self.adversarial_net(self.adv_buffer.detach())
            loss_adv = self.adversarial_criterion(torch.sigmoid(adv_logits), self.target_domain_buffer)*self.config["training"]["lambda_adv"] # minimize

            # labeled loss
            loss = loss_adv  # ADD to lightining hooks.py

            tqdm_dict = {'loss_train': loss}  ### NOTE before the first accumulate steps loss will be NaN don't panic

            tensorboard_logs = {'train_batch_loss': loss,
                                'loss_adv': loss_adv
                                }

            output = OrderedDict({
                'loss': loss,
                'progress_bar': tqdm_dict,
                'log': tensorboard_logs
            })
            return output

        else:
            raise NotImplementedError

    def validation_step(self, batch, batch_indx):

        mixture, strong, weak, filenames = batch

        student_strong, student_weak, student_logits, _ = self.student(mixture.unsqueeze(1))
        teacher_strong, teacher_weak, teacher_logits, _ = self.teacher(mixture.unsqueeze(1))

        loss_student = self.strong_criterion(student_logits, strong).mean()
        loss_teacher = self.strong_criterion(teacher_logits, strong).mean()

        for j in range(student_strong.shape[0]): # over batches
            pred = student_strong[j].cpu().detach().numpy()
            pred = pred > 0.5
            pred = scipy.ndimage.filters.median_filter(pred, (self.config["net"]["median_window"], 1))
            pred = self.encoder.decode_strong(pred)
            pred = pd.DataFrame(pred, columns=["event_label", "onset", "offset"])
            pred["filename"] = Path(filenames[j]).stem + ".wav"
            self.buffer_dataframe_valid_student = self.buffer_dataframe_valid_student.append(pred)

        for j in range(teacher_strong.shape[0]): # over batches
            pred = teacher_strong[j].cpu().detach().numpy()
            pred = pred > 0.5
            pred = scipy.ndimage.filters.median_filter(pred, (self.config["net"]["median_window"], 1))
            pred = self.encoder.decode_strong(pred)
            pred = pd.DataFrame(pred, columns=["event_label", "onset", "offset"])
            pred["filename"] = Path(filenames[j]).stem + ".wav"
            self.buffer_dataframe_valid_teacher = self.buffer_dataframe_valid_teacher.append(pred)

        tqdm_dict = {'val_loss': loss_student}
        output = OrderedDict({
            'val_loss': loss_student,
            'progress_bar': tqdm_dict,
            "val_loss_teacher": loss_teacher
        })

        return output

    def validation_end(self, outputs):

        avg_loss_student = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_loss_teacher = torch.stack([x['val_loss_teacher'] for x in outputs]).mean()

        with open(os.path.join(self.config["data"]["metadata_root"], "validation", "validation.tsv"), "r") as f:
            gt = pd.read_csv(f, sep="\t")

        time_pooling = confs["net"]["pool_factor"]
        sample_rate = confs["data"]["sample_rate"]
        hop_size = confs["feats"]["hop_size"]
        self.buffer_dataframe_valid_teacher.loc[:, "onset"] = self.buffer_dataframe_valid_teacher.onset * time_pooling / (sample_rate / hop_size)
        self.buffer_dataframe_valid_teacher.loc[:, "offset"] = self.buffer_dataframe_valid_teacher.offset * time_pooling / (sample_rate / hop_size)
        self.buffer_dataframe_valid_teacher = self.buffer_dataframe_valid_teacher.reset_index(drop=True)

        self.buffer_dataframe_valid_student.loc[:, "onset"] = self.buffer_dataframe_valid_student.onset * time_pooling / (sample_rate / hop_size)
        self.buffer_dataframe_valid_student.loc[:, "offset"] = self.buffer_dataframe_valid_student.offset * time_pooling / (sample_rate / hop_size)
        self.buffer_dataframe_valid_student = self.buffer_dataframe_valid_student.reset_index(drop=True)

        save_dir = os.path.join(self.config["log_dir"], "metrics")
        os.makedirs(save_dir, exist_ok=True)

        event_teacher, segment_teacher = compute_sed_eval_metrics(self.buffer_dataframe_valid_teacher, gt)

        with open(os.path.join(save_dir, "event_teacher_{}.txt".format(self.current_epoch)), "w") as f:
            f.write(str(event_teacher))

        with open(os.path.join(save_dir, "segment_teacher_{}.txt".format(self.current_epoch)), "w") as f:
            f.write(str(segment_teacher))

        event_student, segment_student = compute_sed_eval_metrics(self.buffer_dataframe_valid_student, gt)

        with open(os.path.join(save_dir, "event_student_{}.txt".format(self.current_epoch)), "w") as f:
            f.write(str(event_student))

        with open(os.path.join(save_dir, "segment_student_{}.txt".format(self.current_epoch)), "w") as f:
            f.write(str(segment_student))


        tqdm_dict = {'val_loss_student': avg_loss_student, "val_loss_teacher": avg_loss_teacher,
                     "val_f_measure_student": event_student.results()["class_wise_average"]['f_measure']['f_measure'],
                     "val_f_measure_teacher": event_teacher.results()["class_wise_average"]['f_measure']['f_measure']}
        tensorboard_logs = {'val_loss_student': avg_loss_student, "val_loss_teacher": avg_loss_teacher
                            }

        output = OrderedDict({
            'val_loss': avg_loss_student,
            'progress_bar': tqdm_dict,
            'log': tensorboard_logs
        })

        self.buffer_dataframe_valid_student = pd.DataFrame() # free the buffers
        self.buffer_dataframe_valid_teacher = pd.DataFrame()

        return output

    def backward(self, trainer, loss, optimizer, optimizer_idx):
        if loss != 0:
            loss.backward()

    def configure_optimizers(self):

        self.opt1 = torch.optim.Adam(self.teacher.parameters(), self.config["opt"]["lr_main"], betas=(0.9, 0.999), weight_decay=self.config["opt"]["weight_decay"])
        self.opt2 = torch.optim.Adam(self.adversarial_net.parameters(), self.config["opt"]["lr_adv"], betas=(0.9, 0.999), weight_decay=self.config["opt"]["weight_decay"])

        return [self.opt1, self.opt2], []

    @pl.data_loader
    def train_dataloader(self):

        bsz = self.config["training"]["batch_size"]
        dataloader = DataLoader(self.train_set, batch_sampler=MultiStreamBatchSampler(self.train_set, batch_sizes=[bsz//2, bsz//4, bsz//4]),
                         num_workers=self.config["training"]["num_workers"])

        return dataloader

    @pl.data_loader
    def val_dataloader(self):

        dataloader = DataLoader(self.val_set, batch_size=self.config["training"]["batch_size"],
                                shuffle=False, num_workers=self.config["training"]["num_workers"], drop_last=False)
        return dataloader


if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.conf_file, "r") as f:
        confs = yaml.load(f)

    # test if compatible with lightning
    confs.update(args.__dict__)
    sed = FixMatch(confs)

    checkpoint_dir = os.path.join(confs["log_dir"], 'checkpoints/')
    checkpoint = ModelCheckpoint(checkpoint_dir, monitor='val_loss',
                                 mode='min',  verbose=True, save_top_k=confs["training"]["save_top_k"])

    #early_stop_callback = EarlyStopping(
     #   monitor='val_loss',
      #  patience=confs["training"]["patience"],
       # verbose=True,
        #mode='min'
    #)

    with open(os.path.join(confs["log_dir"], "confs.yml"), "w") as f:
        yaml.dump(confs, f)

    logger = TensorBoardLogger(os.path.dirname(confs["log_dir"]), confs["log_dir"].split("/")[-1])

    trainer = pl.Trainer(max_nb_epochs=confs["training"]["n_epochs"], gpus=confs["gpus"], checkpoint_callback=checkpoint,
                         accumulate_grad_batches=confs["training"]["accumulate_batches"],
                         logger = logger,
                         gradient_clip=bool(confs["training"]["gradient_clip"]),
                         gradient_clip_val=confs["training"]["gradient_clip"], check_val_every_n_epoch=5)
    trainer.fit(sed)