import torch
import torch.nn as nn
import numpy as np

import pytorch_lightning as pl
import wandb

from models.pose_hrnet_modded_in_notebook import PoseHighResolutionNet

class SegmentationNetModule(pl.LightningModule):
    def __init__(self, config, wandb_run, learning_rate=1e-3):
    #def __init__(self, pose_hrnet, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters("learning_rate")
        self.config = config

        if (self.config.model['BACKBONE'] == "hrnet"):
            self.pose_net = PoseHighResolutionNet(num_key_points=self.config.segmentation_net_module['NUM_KEY_POINTS'],
                                                    num_image_channels=self.config.segmentation_net_module['NUM_IMG_CHANNELS'])
        #self.pose_hrnet = pose_hrnet
        print("Pose HRNet is on device " + str(next(self.pose_net.parameters()).get_device()))     # testing line
        print("Is Pose HRNet on GPU? " + str(next(self.pose_net.parameters()).is_cuda))            # testing line
        self.pose_net.to(device='cpu', dtype=torch.float32)                          # added recently and may fix a lot
        # *** IF the above line causes an error because you do not have CUDA, then just comment it out and the model should run, albeit on the CPU ***
        print("Pose HRNet is on device " + str(next(self.pose_net.parameters()).get_device()))     # testing line
        print("Is Pose HRNet on GPU? " + str(next(self.pose_net.parameters()).is_cuda))            # testing line
        self.wandb_run = wandb_run
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        #print(self.pose_hrnet.get_device())

    # TODO: Add couple forwarding functions to choose in config
    def forward(self, x):
        """This performs a forward pass on the dataset

        Args:
            x (this_type): This is a tensor containing the information yaya

        Returns:
            the forward pass of the dataset: using a certain type of input
        """
        return self.pose_net(x)

    # TODO: Add other types of optimizers
    def configure_optimizers(self):
        #optimizer = torch.optim.Adam(self.parameters, lr=1e-3)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        training_batch, training_batch_labels = train_batch['image'], train_batch['label']
        x = training_batch
        print("Training batch is on device " + str(x.get_device()))         # testing line
        training_output = self.pose_net(x)
        loss = self.loss_fn(training_output, training_batch_labels)
        #self.log('exp_train/loss', loss, on_step=True)
        #self.wandb_run.log('train/loss', loss, on_step=True)
        self.wandb_run.log({'train/loss': loss})
        #self.log(name="train/loss", value=loss)
        return loss

    def validation_step(self, validation_batch, batch_idx):
        val_batch, val_batch_labels = validation_batch['image'], validation_batch['label']
        x = val_batch
        print("Validation batch is on device " + str(x.get_device()))       # testing line
        val_output = self.pose_net(x)
        loss = self.loss_fn(val_output, val_batch_labels)
        #self.log('validation/loss', loss)
        #self.wandb_run.log('validation/loss', loss, on_step=True)
        self.wandb_run.log({'validation/loss': loss})
        #self.log('validation/loss', loss)
        image = wandb.Image(val_output, caption='Validation output')
        self.wandb_run.log({'val_output': image})
        return loss

    def test_step(self, test_batch, batch_idx):
        test_batch, test_batch_labels = test_batch['image'], test_batch['label']
        x = test_batch
        test_output = self.pose_net(x)
        loss = self.loss_fn(test_output, test_batch_labels)
        #self.log('test/loss', loss)
        #self.wandb_run.log('test/loss', loss, on_step=True)
        #self.wandb_run.log({'test/loss': loss})
        #self.on_test_batch_end(self, outputs=test_output, batch=test_batch, batch_idx=batch_idx)
        #self.on_test_batch_end(outputs=test_output, batch=test_batch, batch_idx=batch_idx, dataloader_idx=0)
        return loss
    
