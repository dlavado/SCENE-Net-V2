


from typing import Any, List, Tuple
import torch
from torch import nn
import pytorch_lightning as pl
from torchmetrics import MetricCollection


class LitWrapperModel(pl.LightningModule):
    """
    Generic Pytorch Lightning wrapper for Pytorch models that defines the logic for training, validation,testing and prediciton. 
    It also defines the logic for logging metrics and losses.    
    
    Parameters
    ----------

    `model` - torch.nn.Module:
        The model to be wrapped.
    
    `criterion` - torch.nn.Module:
        The loss function to be used

    `optimizer` - str:
        The Pytorch optimizer to be used for training.
        Note: str must be \in {'Adam', 'SGD', 'RMSprop'}

    `metric_initilizer` - function:
        A function that returns a TorchMetric object. The metric object must have a reset() and update() method.
        The reset() method is called at the end of each epoch and the update() method is called at the end of each step.
    """

    def __init__(self, model:torch.nn.Module, criterion:torch.nn.Module, optimizer_name:str, learning_rate=1e-2, metric_initializer=None, **kwargs):
        super().__init__()
        self.model = model
        self.criterion = criterion

        self.save_hyperparameters()


        if metric_initializer is not None:
            self.train_metrics:MetricCollection = metric_initializer()
            self.val_metrics:MetricCollection = metric_initializer()
            self.test_metrics:MetricCollection = metric_initializer()
        else:
            self.train_metrics = None
            self.val_metrics = None
            self.test_metrics = None
    
    def forward(self, x):
        return self.model(x)
    
    def prediction(self, model_output:torch.Tensor) -> torch.Tensor:
        return model_output

    def evaluate(self, batch, stage=None, metric=None, prog_bar=True, logger=True):
        x, y = batch
        out = self(x)
        
        loss = self.criterion(out, y)
        preds = self.prediction(out)

        if stage:
            on_step = stage == "train" 
            self.log(f"{stage}_loss", loss, on_epoch=True, on_step=on_step, prog_bar=prog_bar, logger=logger)

            if metric:
                for metric_name, metric_val in metric.items():
                    met = metric_val(preds, y)
                    if isinstance(met, torch.Tensor):
                        met = met.mean()
                    self.log(f"{stage}_{metric_name}", met, on_epoch=True, on_step=on_step, prog_bar=True, logger=True)

        return loss, preds, y  

    def training_step(self, batch, batch_idx):
        loss, preds, y = self.evaluate(batch, "train", self.train_metrics)
        state = {"loss": loss, "preds": preds}
       
        return state  

    # def on_before_backward(self, loss: torch.Tensor) -> None:
    #     print(f'\n{"="*10} Model Values & Gradients {"="*10}')
    #     print(f'L1/L2 Norms of the gradient: {torch.norm(loss, 1)}, {torch.norm(loss, 2)}')
    #     # for name, param in self.model.named_parameters():
    #     #     if 'geneo' in name:
    #     #         print(f'\t{name} -- value: {param.data.item():.5f} grad: {param.grad}')
    #     return super().on_before_backward(loss)                
    
    def on_train_epoch_end(self) -> None:
        if self.train_metrics is not None:
            self._epoch_end_metric_logging(self.train_metrics, 'train', print_metrics=False)

    
    def validation_step(self, batch, batch_idx):
        loss, preds, y = self.evaluate(batch, "val", self.val_metrics)
        state = {"val_loss": loss, "preds": preds}

        return state
    
    def on_validation_epoch_end(self) -> None: 
        if self.val_metrics is not None: # On epoch metric logging
           self._epoch_end_metric_logging(self.val_metrics, 'val', print_metrics=True)
    
    def test_step(self, batch, batch_idx):
        loss, preds, y = self.evaluate(batch, "test", self.test_metrics)
        state = {"test_loss": loss,
                 "preds": preds,
                }
     
        return state

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        x, _ = batch
        pred = self(x)
        pred = self.prediction(pred)

        return pred
    
    def on_test_epoch_end(self) -> None:
        if self.test_metrics is not None: # On epoch metric logging
            self._epoch_end_metric_logging(self.test_metrics, 'test', True)

    def get_model(self):
        return self.model
    
    def set_criteria(self, criterion):
        self.criterion = criterion
    
    def _epoch_end_metric_logging(self, metrics, prefix, print_metrics=False):
        metric_res = metrics.compute()
        if print_metrics:
            print(f'{"="*10} {prefix} metrics {"="*10}')
        for metric_name, metric_val in metric_res.items():
            if print_metrics:
                # if metric is per class
                if isinstance(metric_val, torch.Tensor) and metric_val.ndim > 0: 
                    print(f'\t{prefix}_{metric_name}: {metric_val}; mean: {metric_val[metric_val > 0].mean():.4f}') # class 0 is noise
                else:
                    print(f'\t{prefix}_{metric_name}: {metric_val}')

        metrics.reset()

    def configure_optimizers(self):
        return self._resolve_optimizer(self.hparams.optimizer_name)
    
    def _check_model_gradients(self):
        print(f'\n{"="*10} Model Values & Gradients {"="*10}')
        for name, param in self.model.named_parameters():
            print(f'\t{name} -- value: {param.data.item():.5f} grad: {param.grad}')

    def _resolve_optimizer(self, optimizer_name:str):
        optimizer_name = optimizer_name.lower()
        if  optimizer_name == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)
        elif optimizer_name == 'sgd':
            return torch.optim.SGD(self.model.parameters(), lr=self.hparams.learning_rate)
        elif optimizer_name == 'rmsprop':
            return torch.optim.RMSprop(self.model.parameters(), lr=self.hparams.learning_rate)
        elif optimizer_name == 'lbfgs':
            return torch.optim.LBFGS(self.model.parameters(), lr=self.hparams.learning_rate, max_iter=20)
        
        raise NotImplementedError(f'Optimizer {self.hparams.optimizer_name} not implemented')
    
