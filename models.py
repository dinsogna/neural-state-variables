
import os
import torch
import shutil
import numpy as np
from torch import nn
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models
from torch.autograd.functional import jacobian
from collections import OrderedDict
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from dataset import NeuralPhysDataset
from schedulers import CyclicLambdaScheduler
from model_utils import (EncoderDecoder,
                         EncoderDecoder64x1x1,
                         RefineDoublePendulumModel,
                         RefineSinglePendulumModel,
                         RefineCircularMotionModel,
                         RefineModelReLU,
                         RefineSwingStickNonMagneticModel,
                         RefineAirDancerModel,
                         RefineLavaLampModel,
                         RefineFireModel,
                         RefineElasticPendulumModel,
                         RefineReactionDiffusionModel)


def mkdir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

class VisDynamicsModel(pl.LightningModule):

    def __init__(self,
                 lr: float=1e-4,
                 seed: int=1,
                 if_cuda: bool=True,
                 if_test: bool=False,
                 gamma: float=0.5,
                 log_dir: str='logs',
                 train_batch: int=512,
                 val_batch: int=256,
                 test_batch: int=256,
                 num_workers: int=8,
                 model_name: str='encoder-decoder-64',
                 data_filepath: str='data',
                 dataset: str='single_pendulum',
                 lr_schedule: list=[20, 50, 100],
                 lambda_schedule: list=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,]) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.kwargs = {'num_workers': self.hparams.num_workers, 'pin_memory': True} if self.hparams.if_cuda else {}
        # create visualization saving folder if testing
        self.pred_log_dir = os.path.join(self.hparams.log_dir, 'predictions')
        self.var_log_dir = os.path.join(self.hparams.log_dir, 'variables')
        if not self.hparams.if_test:
            mkdir(self.pred_log_dir)
            mkdir(self.var_log_dir)
        # ex1
        self.latent_lambda_scheduler = CyclicLambdaScheduler(step_size=512, max_lda=40.0)
        # s1
        # self.refine_lambda_scheduler = CyclicLambdaScheduler(max_lda=1e-5)
        # s2
        self.refine_lambda_scheduler = CyclicLambdaScheduler(max_lda=1e-4)

        self.__build_model()
        
        #DOM: store training losses / valuable data here
        # self.train_loss_logs = {"loss":[], "recon": [], "reg":[]}
        # self.loss_log_dir = os.path.join("./loss_logs", "_", self.hparams.seed)
        # self.train_loss_log = []
        # self.val_loss_logs = dict()
        # self.test_loss_logs = dict()

    def __build_model(self):
        if self.hparams.model_name == 'encoder-decoder':
            self.model = EncoderDecoder(in_channels=3)
        if self.hparams.model_name == 'encoder-decoder-64':
            self.model = EncoderDecoder64x1x1(in_channels=3)
        if self.hparams.model_name == 'refine-64' and self.hparams.dataset == 'single_pendulum':
            self.model = RefineSinglePendulumModel(in_channels=64)
        if self.hparams.model_name == 'refine-64' and self.hparams.dataset == 'double_pendulum':
            self.model = RefineDoublePendulumModel(in_channels=64)
        if self.hparams.model_name == 'refine-64' and self.hparams.dataset == 'circular_motion':
            self.model = RefineCircularMotionModel(in_channels=64)
        if self.hparams.model_name == 'refine-64' and self.hparams.dataset == 'swingstick_non_magnetic':
            self.model = RefineSwingStickNonMagneticModel(in_channels=64)
        if self.hparams.model_name == 'refine-64' and self.hparams.dataset == 'air_dancer':
            self.model = RefineAirDancerModel(in_channels=64)
        if self.hparams.model_name == 'refine-64' and self.hparams.dataset == 'lava_lamp':
            self.model = RefineLavaLampModel(in_channels=64)
        if self.hparams.model_name == 'refine-64' and self.hparams.dataset == 'fire':
            self.model = RefineFireModel(in_channels=64)
        if self.hparams.model_name == 'refine-64' and self.hparams.dataset == 'elastic_pendulum':
            self.model = RefineElasticPendulumModel(in_channels=64)
        if self.hparams.model_name == 'refine-64' and self.hparams.dataset == 'reaction_diffusion':
            self.model = RefineReactionDiffusionModel(in_channels=64)
        if 'refine' in self.hparams.model_name and self.hparams.if_test:
            self.high_dim_model = EncoderDecoder64x1x1(in_channels=3)
    
    # def latent_lambda_scheduler(self, epoch, training=False):
        # lda = 0.8 * np.exp(-epoch / 20)
        # return lda

        # s1
        # if epoch < 5:
        #     return 0.0
        # if epoch < 10:
        #     return 1e-3
        # if epoch < 15:
        #     return 1e-2
        # if epoch < 20:
        #     return 5e-2
        # return 0.1

        # s2
        # if epoch < 5:
        #     return 0.0
        # if epoch < 10:
        #     return 1e-2
        # if epoch < 15:
        #     return 0.1
        # if epoch < 20:
        #     return 0.5
        # return 1.0

        # s3
        # if epoch < 5:
        #     return 0.0
        # if epoch < 10:
        #     return 1e-1
        # if epoch < 15:
        #     return 1.0
        # if epoch < 20:
        #     return 5.0
        # return 10.0

        # s4
        # if epoch < 5:
        #     return 0.0
        # if epoch < 10:
        #     return 1.0
        # if epoch < 15:
        #     return 8.0
        # if epoch < 20:
        #     return 32.0
        # return 64.0

        # s5
        # if not training:
        #     return 40.0
        # if epoch < 5:
        #     return 0.0
        # if epoch < 10:
        #     return 1.0
        # if epoch < 15:
        #     return 5.0
        # if epoch < 20:
        #     return 20.0
        # return 40.0
    
    # def refine_lambda_scheduler(self, epoch, training=False):
        # s0
        # return 0.0

        # s1
        # if epoch < 20:
        #     return 0.0
        # if epoch < 30:
        #     return 1e-5
        # if epoch < 40
        #     return 5e-5
        # if epoch < 50
        #     return 1e-4

        # s2
        # stage = epoch // 10
        # if stage < 2:
        #     return 0.0
        # if stage == 9:
        #     return 1e-3
        # return 1e-5 * (2 ** (stage - 2))

        # s3
        # if not training:
        #     return 1e-3
        # it = epoch % 25
        # if it < 5:
        #     return 0.0
        # if it < 10:
        #     return 1e-5
        # if it < 15:
        #     return 1e-4
        # if it < 20:
        #     return 5e-4
        # return 1e-3

        # s4
        # if not training:
        #     return 5e-4
        # it = epoch % 25
        # if it < 5:
        #     return 0.0
        # if it < 10:
        #     return 5e-6
        # if it < 15:
        #     return 5e-5
        # if it < 20:
        #     return 2e-4
        # return 5e-4

        # s5
        # if not training:
        #     return 1e-4
        # it = epoch % 25
        # if it < 5:
        #     return 0.0
        # if it < 10:
        #     return 1e-6
        # if it < 15:
        #     return 1e-5
        # if it < 20:
        #     return 5e-5
        # return 1e-4

        # s6
        # if not training:
        #     return 1e-5
        # it = epoch % 25
        # if it < 5:
        #     return 0.0
        # if it < 10:
        #     return 1e-7
        # if it < 15:
        #     return 1e-6
        # if it < 20:
        #     return 5e-6
        # return 1e-5
    
    # Loss = lambda*recon_loss + (1-lambda)*smoothing_loss
    def loss_func(self, data, output, latent, target, training=False, all_losses=False): 
        recon_loss = self.recon_loss_func(output, target)
        if 'refine' in self.hparams.model_name:
            # lda = self.refine_lambda_scheduler(self.trainer.current_epoch, training=training)
            lda = self.refine_lambda_scheduler.select_lambda(training)
            reg_loss = self.refine_reg_loss_func(data)
        else:
            lda = self.latent_lambda_scheduler.select_lambda(training)
            reg_loss = self.latent_reg_loss_func(output, latent, target)            
        loss = recon_loss + lda * reg_loss
        if all_losses:
            return loss, recon_loss, reg_loss
        return loss

    # MSE loss function
    def recon_loss_func(self, output, target):
        mse_loss = nn.MSELoss(reduction='none')
        output, target = output.squeeze(), target.squeeze()
        loss = torch.sum(mse_loss(output, target), dim=1)
        loss = torch.mean(loss)
        return loss
    
    def refine_reg_loss_func(self, data):

        def _sum_encoder(data):
            assert data.requires_grad
            output, latent = self.train_forward(data)
            _sum_latent = latent.sum(dim=0)
            assert _sum_latent.requires_grad
            return _sum_latent
        
        with torch.enable_grad():
            data.requires_grad_()
            J = jacobian(_sum_encoder, data, create_graph=True).permute(1,0,2)
        F_norm = torch.linalg.norm(J, dim=[1,2])
        reg_loss = torch.mean(torch.square(F_norm))
        return reg_loss
    
    def latent_reg_loss_func(self, output, latent, target):
        mse_loss = nn.MSELoss(reduction='none')
        next_output, next_latent = self.train_forward(target)
        latent_loss = torch.sum(mse_loss(latent, next_latent), dim=1)
        latent_norm = torch.linalg.norm(latent, dim=1)
        next_latent_norm = torch.linalg.norm(next_latent, dim=1)
        prod_norm = torch.mul(latent_norm, next_latent_norm)
        scaled_latent_loss = torch.div(latent_loss, prod_norm)
        scaled_latent_loss = torch.mean(scaled_latent_loss)
        return scaled_latent_loss

    def train_forward(self, x):
        if self.hparams.model_name == 'encoder-decoder' or 'refine' in self.hparams.model_name:
            output, latent = self.model(x)
        if self.hparams.model_name == 'encoder-decoder-64':
            output, latent = self.model(x, x, False)
        return output, latent

    def training_step(self, batch, batch_idx):
        data, target, filepath = batch
        output, latent = self.train_forward(data)
            
        # Recon + Smoothing loss
        train_loss, train_recon_loss, train_reg_loss = self.loss_func(data, output, latent, target, training=True, all_losses=True)
        self.log('train_loss', train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_reconstruction_loss', train_recon_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_regularization_loss', train_reg_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return train_loss

    def validation_step(self, batch, batch_idx):
        data, target, filepath = batch
        output, latent = self.train_forward(data)

        val_loss, val_recon_loss, val_reg_loss = self.loss_func(data, output, latent, target, all_losses=True)
        self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_reconstruction_loss', val_recon_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_regularization_loss', val_reg_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        
        if self.hparams.model_name == 'encoder-decoder' or self.hparams.model_name == 'encoder-decoder-64':
            data, target, filepath = batch
            if self.hparams.model_name == 'encoder-decoder':
                output, latent = self.model(data)
            if self.hparams.model_name == 'encoder-decoder-64':
                output, latent = self.model(data, data, False)
            test_loss = self.loss_func(data, output, latent, target)
            self.log('test_loss', test_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            # save the output images and latent vectors
            self.all_filepaths.extend(filepath)
            for idx in range(data.shape[0]):
                comparison = torch.cat([data[idx,:, :, :128].unsqueeze(0),
                                        data[idx,:, :, 128:].unsqueeze(0),
                                        target[idx, :, :, :128].unsqueeze(0),
                                        target[idx, :, :, 128:].unsqueeze(0),
                                        output[idx, :, :, :128].unsqueeze(0),
                                        output[idx, :, :, 128:].unsqueeze(0)])
                save_image(comparison.cpu(), os.path.join(self.pred_log_dir, filepath[idx]), nrow=1)
                latent_tmp = latent[idx].view(1, -1)[0]
                latent_tmp = latent_tmp.cpu().detach().numpy()
                self.all_latents.append(latent_tmp)

        if 'refine' in self.hparams.model_name:
            data, target, filepath = batch
            _, latent = self.high_dim_model(data, data, False)
            latent = latent.squeeze(-1).squeeze(-1)
            latent_reconstructed, latent_latent = self.model(latent)
            output, _ = self.high_dim_model(data, latent_reconstructed.unsqueeze(2).unsqueeze(3), True)
            # calculate losses
            pixel_reconstruction_loss = self.recon_loss_func(output, target)
            self.log('pixel_reconstruction_loss', pixel_reconstruction_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            # save the output images and latent vectors
            self.all_filepaths.extend(filepath)
            for idx in range(data.shape[0]):
                comparison = torch.cat([data[idx, :, :, :128].unsqueeze(0),
                                        data[idx, :, :, 128:].unsqueeze(0),
                                        target[idx, :, :, :128].unsqueeze(0),
                                        target[idx, :, :, 128:].unsqueeze(0),
                                        output[idx, :, :, :128].unsqueeze(0),
                                        output[idx, :, :, 128:].unsqueeze(0)])
                save_image(comparison.cpu(), os.path.join(self.pred_log_dir, filepath[idx]), nrow=1)
                latent_tmp = latent[idx].view(1, -1)[0]
                latent_tmp = latent_tmp.cpu().detach().numpy()
                self.all_latents.append(latent_tmp)
                # save latent_latent: the latent vector in the refine network
                latent_latent_tmp = latent_latent[idx].view(1, -1)[0]
                latent_latent_tmp = latent_latent_tmp.cpu().detach().numpy()
                self.all_refine_latents.append(latent_latent_tmp)
                # save latent_reconstructed: the latent vector reconstructed by the entire refine network
                latent_reconstructed_tmp = latent_reconstructed[idx].view(1, -1)[0]
                latent_reconstructed_tmp = latent_reconstructed_tmp.cpu().detach().numpy()
                self.all_reconstructed_latents.append(latent_reconstructed_tmp)


    def test_save(self):
        if self.hparams.model_name == 'encoder-decoder' or self.hparams.model_name == 'encoder-decoder-64':
            np.save(os.path.join(self.var_log_dir, 'ids.npy'), self.all_filepaths)
            np.save(os.path.join(self.var_log_dir, 'latent.npy'), self.all_latents)
        if 'refine' in self.hparams.model_name:
            np.save(os.path.join(self.var_log_dir, 'ids.npy'), self.all_filepaths)
            np.save(os.path.join(self.var_log_dir, 'latent.npy'), self.all_latents)
            np.save(os.path.join(self.var_log_dir, 'refine_latent.npy'), self.all_refine_latents)
            np.save(os.path.join(self.var_log_dir, 'reconstructed_latent.npy'), self.all_reconstructed_latents)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.hparams.lr_schedule, gamma=self.hparams.gamma)
        return [optimizer], [scheduler]
    
    def paths_to_tuple(self, paths):
        new_paths = []
        for i in range(len(paths)):
            tmp = paths[i].split('.')[0].split('_')
            new_paths.append((int(tmp[0]), int(tmp[1])))
        return new_paths

    def setup(self, stage=None):

        if stage == 'fit':
            # for the training of the refine network, we need to have the latent data as the dataset
            if 'refine' in self.hparams.model_name:
                high_dim_var_log_dir = self.var_log_dir.replace('refine', 'encoder-decoder')
                train_data = torch.FloatTensor(np.load(os.path.join(high_dim_var_log_dir+'_train', 'latent.npy')))
                train_target = torch.FloatTensor(np.load(os.path.join(high_dim_var_log_dir+'_train', 'latent.npy')))
                val_data = torch.FloatTensor(np.load(os.path.join(high_dim_var_log_dir+'_val', 'latent.npy')))
                val_target = torch.FloatTensor(np.load(os.path.join(high_dim_var_log_dir+'_val', 'latent.npy')))
                train_filepaths = list(np.load(os.path.join(high_dim_var_log_dir+'_train', 'ids.npy')))
                val_filepaths = list(np.load(os.path.join(high_dim_var_log_dir+'_val', 'ids.npy')))
                # convert the file strings into tuple so that we can use TensorDataset to load everything together
                train_filepaths = torch.Tensor(self.paths_to_tuple(train_filepaths))
                val_filepaths = torch.Tensor(self.paths_to_tuple(val_filepaths))
                self.train_dataset = torch.utils.data.TensorDataset(train_data, train_target, train_filepaths)
                self.val_dataset = torch.utils.data.TensorDataset(val_data, val_target, val_filepaths)
            else:
                self.train_dataset = NeuralPhysDataset(data_filepath=self.hparams.data_filepath,
                                                       flag='train',
                                                       seed=self.hparams.seed,
                                                       object_name=self.hparams.dataset)
                self.val_dataset = NeuralPhysDataset(data_filepath=self.hparams.data_filepath,
                                                     flag='val',
                                                     seed=self.hparams.seed,
                                                     object_name=self.hparams.dataset)

        if stage == 'test':
            self.test_dataset = NeuralPhysDataset(data_filepath=self.hparams.data_filepath,
                                                  flag='test',
                                                  seed=self.hparams.seed,
                                                  object_name=self.hparams.dataset)
            
            # initialize lists for saving variables and latents during testing
            self.all_filepaths = []
            self.all_latents = []
            self.all_refine_latents = []
            self.all_reconstructed_latents = []

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                   batch_size=self.hparams.train_batch,
                                                   shuffle=True,
                                                   **self.kwargs)
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(dataset=self.val_dataset,
                                                 batch_size=self.hparams.val_batch,
                                                 shuffle=False,
                                                 **self.kwargs)
        return val_loader

    def test_dataloader(self):
        test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                  batch_size=self.hparams.test_batch,
                                                  shuffle=False,
                                                  **self.kwargs)
        return test_loader
