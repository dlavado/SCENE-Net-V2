



import torch
import sys
sys.path.insert(0, '..')
sys.path.insert(1, '../..')


from core.models.cnn import CNN_Baseline
from core.lit_modules.lit_model_wrappers import LitWrapperModel

class LitCNNBaseline(LitWrapperModel):

    def __init__(self, 
                in_channels: int,
                out_channels: int,
                kernel_size: int,
                num_groups: int = 1,
                padding: int = 1,
                MLP_hidden_dims: list = [],
                num_classes: int = 10,  
                ignore_index:int=-1,
                criterion:torch.nn.Module=None, 
                optimizer:str=None, 
                learning_rate=1e-2, 
                metric_initializer=None,
            ):

        model = CNN_Baseline(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            num_groups=num_groups,
            padding=padding,
            MLP_hidden_dims=MLP_hidden_dims,
            num_classes=num_classes
        )

        super().__init__(model, criterion, optimizer, learning_rate, None)
        
        if metric_initializer is not None:
            self.train_metrics = metric_initializer(num_classes=num_classes, ignore_index=ignore_index)
            self.val_metrics = metric_initializer(num_classes=num_classes, ignore_index=ignore_index)
            self.test_metrics = metric_initializer(num_classes=num_classes, ignore_index=ignore_index)

        self.save_hyperparameters()
        self.logged_batch = False

    def prediction(self, model_output: torch.Tensor) -> torch.Tensor:
        return self.model.prediction(model_output)
    
    def forward(self, x, pts_locs):
        return self.model(x, pts_locs)
    

    def evaluate(self, batch, stage=None, metric=None, prog_bar=True, logger=True):
        x, y, pts_locs = batch
        out = self.model(x, pts_locs)
        y = y.long()
        loss = self.criterion(out, y)
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