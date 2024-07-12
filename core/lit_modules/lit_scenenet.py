

from typing import Any, List, Tuple
import torch

import sys
from torchmetrics import MetricCollection

sys.path.insert(0, '..')
sys.path.insert(1, '../..')
sys.path.insert(2, '../../..')
from core.models.GENEONets.SCENE_Net import SceneNet_multiclass, SceneNet_multiclass_CNN
from core.lit_modules.lit_model_wrappers import LitWrapperModel
from core.criterions.elastic_net_reg import ElasticNetRegularization
from utils.my_utils import pointcloud_to_wandb


class LitSceneNet_multiclass(LitWrapperModel):

    def __init__(self, 
                geneo_num:dict,
                num_observers:int,
                kernel_size:Tuple[int],
                hidden_dims:Tuple[int],
                num_classes:int, 
                ignore_index:int=-1,
                criterion:torch.nn.Module=None, 
                optimizer:str=None, 
                learning_rate=1e-2, 
                metric_initializer=None,
                **kwargs
            ):
    
        model = SceneNet_multiclass(geneo_num, num_observers, kernel_size, hidden_dims, num_classes)
        super().__init__(model, criterion, optimizer, learning_rate, None)

        
        if metric_initializer is not None:
            self.train_metrics:MetricCollection = metric_initializer(num_classes=num_classes, ignore_index=ignore_index)
            self.val_metrics:MetricCollection = metric_initializer(num_classes=num_classes, ignore_index=ignore_index)
            self.test_metrics:MetricCollection = metric_initializer(num_classes=num_classes, ignore_index=ignore_index)

        self.save_hyperparameters()
        self.logged_batch = False
        self.gradient_check = False
        self.elastic_reg = ElasticNetRegularization(alpha=0.001, l1_ratio=0.5)

    def prediction(self, model_output: torch.Tensor) -> torch.Tensor:
        return self.model.prediction(model_output)
    
    def get_model_architecture_hyperparameters():
        return ['geneo_num', 'num_observers','kernel_size', 'hidden_dims', 'num_classes']
    
    def forward(self, x, pts_locs):
        return self.model(x, pts_locs)
    

    def evaluate(self, batch, stage=None, metric=None, prog_bar=True, logger=True):
        x, y, pts_locs = batch
        out = self.model(x, pts_locs)
        y = y.long()
        loss = self.criterion(out, y) + self.elastic_reg(self.model.get_cvx_coefficients().parameters())
        preds = self.prediction(out)

        # print shapes
        # print(f'x: {x.shape}, y: {y.shape}, pts_locs: {pts_locs.shape}, preds: {preds.shape}')

        # pts_locs = pts_locs[0].detach().cpu().numpy()

        # preds = torch.squeeze(preds[0]).detach().cpu().numpy() # 1st batch sample
        # preds = preds / np.max(preds) # to plot the different classes in different colors
        # preds = np.column_stack((pts_locs, preds))
        # Vox.plot_voxelgrid(preds, color_mode='ranges', plot=True)

        # x = torch.squeeze(x[0]).detach().cpu().numpy()
        # x = Vox.plot_voxelgrid(x, color_mode='ranges', plot=True)

    
        if stage:
            on_step = True
            self.log(f"{stage}_loss", loss, on_epoch=True, on_step=on_step, prog_bar=prog_bar, logger=logger)

            if metric:
                for metric_name, metric_val in metric.items():
                    met = metric_val(preds, y)
                    if isinstance(met, torch.Tensor):
                        met = torch.mean(met[met > 0]) #if a metric is zero for a class, it is not included in the mean
                    self.log(f"{stage}_{metric_name}", met, on_epoch=True, on_step=on_step, prog_bar=True, logger=True)

        return loss, preds, y
    

    def _log_pointcloud_wandb(self, pcd, input=None, gt=None, prefix='run'):
        point_clouds = pointcloud_to_wandb(pcd, input, gt)
        self.logger.experiment.log({f'{prefix}_point_cloud': point_clouds})  

    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int) -> None:
        super().on_train_batch_end(outputs, batch, batch_idx)
        self.model.maintain_convexity()

    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()
        # cvx_coeffs = self.model.get_cvx_coefficients()
        # print(f'\n\n{"="*10} cvx coefficients {"="*10}')
        # for name, coeff in cvx_coeffs.items():
        #     for i in range(coeff.shape[0]):
        #         if torch.any(coeff[i] < 0) or torch.any(coeff[i] > 0.5):
        #             print(f'\t{name}_obs{i}:\n\t {coeff[i]}')

        self.logged_batch = False

    # def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
    #     print(f'\n{"="*10} Model Values & Gradients {"="*10}')
    #     cvx_coeffs = self.model.get_cvx_coefficients()
    #     geneo_params = self.model.get_geneo_params()
    #     for name, cvx in cvx_coeffs.items():
    #         print(f'\t{name} -- cvx: {cvx}  --grad: {cvx.grad}')
    #     # for name, param in geneo_params.items():
    #     #     print(f'\t{name} -- value: {param} --grad: {param.grad}')
    #     return super().on_before_optimizer_step(optimizer)
    
    # def on_validation_batch_end(self, outputs, batch: Any, batch_idx: int) -> None:
    #     # logging batch point clouds to wandb
    #     if self.trainer.current_epoch % 10 == 0 and not self.logged_batch: 
    #         x, y, pt_locs = batch 
    #         preds = outputs["preds"]

    #         pt_locs = pt_locs[0].detach().cpu().numpy()

    #         preds = torch.squeeze(preds[0]).detach().cpu().numpy() # 1st batch sample
    #         preds = preds / np.max(preds) # to plot the different classes in different colors
    #         preds = np.column_stack((pt_locs, preds))

    #         x = torch.squeeze(x[0]).detach().cpu().numpy()
    #         x = Vox.plot_voxelgrid(x, color_mode='ranges', plot=False)
            
    #         y = torch.squeeze(y[0]).detach().cpu().numpy() # 1st batch sample
    #         y = y / np.max(y) # to plot the different classes in different colors
    #         y = np.column_stack((pt_locs, y))

    #         self._log_pointcloud_wandb(preds, x, y, prefix=f'val_{self.trainer.global_step}')
    #         self.logged_batch = True


    def get_geneo_params(self):
        return self.model.get_geneo_params()

    def get_cvx_coefficients(self):
        return self.model.get_cvx_coefficients()
    


class LitSceneNet_multiclass_CNN(LitWrapperModel):


    def __init__(self, 
                geneo_num:dict,
                num_observers:int,
                kernel_size:Tuple[int],
                hidden_dims:Tuple[int],
                num_classes:int,
                cnn_out_channels:int=32,
                cnn_kernel_size:int=3, 
                ignore_index:int=-1,
                criterion:torch.nn.Module=None, 
                optimizer:str=None, 
                learning_rate=1e-2, 
                metric_initializer=None,
                **kwargs
            ):
        

        model = SceneNet_multiclass_CNN(geneo_num, num_observers, kernel_size, hidden_dims, num_classes, cnn_out_channels=cnn_out_channels, cnn_kernel_size=cnn_kernel_size)

        super().__init__(model, criterion, optimizer, learning_rate, None)
        
        if metric_initializer is not None:
            self.train_metrics:MetricCollection = metric_initializer(num_classes=num_classes, ignore_index=ignore_index)
            self.val_metrics:MetricCollection = metric_initializer(num_classes=num_classes, ignore_index=ignore_index)
            self.test_metrics:MetricCollection = metric_initializer(num_classes=num_classes, ignore_index=ignore_index)

        self.save_hyperparameters()
        self.logged_batch = False
        self.gradient_check = False
        self.elastic_reg = ElasticNetRegularization(alpha=0.001, l1_ratio=0.5)

    def prediction(self, model_output: torch.Tensor) -> torch.Tensor:
        return self.model.prediction(model_output)
    
    def get_model_architecture_hyperparameters():
        return ['geneo_num', 'num_observers','kernel_size', 'hidden_dims', 'num_classes']
    
    def forward(self, x, pts_locs):
        return self.model(x, pts_locs)
    

    def evaluate(self, batch, stage=None, metric=None, prog_bar=True, logger=True):
        x, y, pts_locs = batch
        out = self.model(x, pts_locs)
        y = y.long()
        loss = self.criterion(out, y) + self.elastic_reg(self.model.get_cvx_coefficients().parameters())
        preds = self.prediction(out)

        # print shapes
        # print(f'x: {x.shape}, y: {y.shape}, pts_locs: {pts_locs.shape}, preds: {preds.shape}')

        # pts_locs = pts_locs[0].detach().cpu().numpy()

        # preds = torch.squeeze(preds[0]).detach().cpu().numpy() # 1st batch sample
        # preds = preds / np.max(preds) # to plot the different classes in different colors
        # preds = np.column_stack((pts_locs, preds))
        # Vox.plot_voxelgrid(preds, color_mode='ranges', plot=True)

        # x = torch.squeeze(x[0]).detach().cpu().numpy()
        # x = Vox.plot_voxelgrid(x, color_mode='ranges', plot=True)

        if stage:
            on_step = True
            self.log(f"{stage}_loss", loss, on_epoch=True, on_step=on_step, prog_bar=prog_bar, logger=logger)

            if metric:
                for metric_name, metric_val in metric.items():
                    met = metric_val(preds, y)
                    if isinstance(met, torch.Tensor):
                        met = torch.mean(met[met > 0]) #if a metric is zero for a class, it is not included in the mean
                    self.log(f"{stage}_{metric_name}", met, on_epoch=True, on_step=on_step, prog_bar=True, logger=True)

        return loss, preds, y
    

    def _log_pointcloud_wandb(self, pcd, input=None, gt=None, prefix='run'):
        point_clouds = pointcloud_to_wandb(pcd, input, gt)
        self.logger.experiment.log({f'{prefix}_point_cloud': point_clouds})  

    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int) -> None:
        super().on_train_batch_end(outputs, batch, batch_idx)
        self.model.maintain_convexity()

    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()
        # cvx_coeffs = self.model.get_cvx_coefficients()
        # print(f'\n\n{"="*10} cvx coefficients {"="*10}')
        # for name, coeff in cvx_coeffs.items():
        #     for i in range(coeff.shape[0]):
        #         if torch.any(coeff[i] < 0) or torch.any(coeff[i] > 0.5):
        #             print(f'\t{name}_obs{i}:\n\t {coeff[i]}')

        self.logged_batch = False

    # def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
    #     print(f'\n{"="*10} Model Values & Gradients {"="*10}')
    #     cvx_coeffs = self.model.get_cvx_coefficients()
    #     geneo_params = self.model.get_geneo_params()
    #     for name, cvx in cvx_coeffs.items():
    #         print(f'\t{name} -- cvx: {cvx}  --grad: {cvx.grad}')
    #     # for name, param in geneo_params.items():
    #     #     print(f'\t{name} -- value: {param} --grad: {param.grad}')
    #     return super().on_before_optimizer_step(optimizer)
    
    # def on_validation_batch_end(self, outputs, batch: Any, batch_idx: int) -> None:
    #     # logging batch point clouds to wandb
    #     if self.trainer.current_epoch % 10 == 0 and not self.logged_batch: 
    #         x, y, pt_locs = batch 
    #         preds = outputs["preds"]

    #         pt_locs = pt_locs[0].detach().cpu().numpy()

    #         preds = torch.squeeze(preds[0]).detach().cpu().numpy() # 1st batch sample
    #         preds = preds / np.max(preds) # to plot the different classes in different colors
    #         preds = np.column_stack((pt_locs, preds))

    #         x = torch.squeeze(x[0]).detach().cpu().numpy()
    #         x = Vox.plot_voxelgrid(x, color_mode='ranges', plot=False)
            
    #         y = torch.squeeze(y[0]).detach().cpu().numpy() # 1st batch sample
    #         y = y / np.max(y) # to plot the different classes in different colors
    #         y = np.column_stack((pt_locs, y))

    #         self._log_pointcloud_wandb(preds, x, y, prefix=f'val_{self.trainer.global_step}')
    #         self.logged_batch = True


    def get_geneo_params(self):
        return self.model.get_geneo_params()

    def get_cvx_coefficients(self):
        return self.model.get_cvx_coefficients()