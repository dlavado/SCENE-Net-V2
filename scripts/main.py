
from datetime import datetime
from pprint import pprint
from typing import List
import warnings
import numpy as np
import sys
import os
import yaml
import ast

# Vanilla PyTorch
import torch
from torchvision.transforms import Compose
import torch.nn.functional as F
import torchinfo


# PyTorch Lightning
import pytorch_lightning as pl
import  pytorch_lightning.callbacks as  pl_callbacks
from pytorch_lightning.callbacks import BatchSizeFinder

# Weights & Biases
import wandb
from pytorch_lightning.loggers import WandbLogger


# Our code
sys.path.insert(0, '..')
sys.path.insert(1, '../..')

from utils.constants import *
import utils.my_utils as su
import utils.pointcloud_processing as eda


from core.datasets.torch_transforms import Farthest_Point_Sampling
import core.lit_modules.lit_callbacks as lit_callbacks
from core.lit_modules.lit_scenenet import LitSceneNet_multiclass
from core.lit_modules.lit_ts40k import LitTS40K_FULL, LitTS40K_FULL_Preprocessed
from core.criterions.geneo_loss import GENEO_Loss
from core.datasets.torch_transforms import *

#####################################################################
# PARSER
#####################################################################

def replace_variables(string):
    """
    Replace variables marked with '$' in a string with their corresponding values from the local scope.

    Args:
    - string: Input string containing variables marked with '$'

    Returns:
    - Updated string with replaced variables
    """
    import re

    pattern = r'\${(\w+)}'
    matches = re.finditer(pattern, string)

    for match in matches:
        variable = match.group(1)
        value = locals().get(variable)
        if value is None:
            value = globals().get(variable)

        if value is not None:
            string = string.replace(match.group(), str(value))
        else:
            raise ValueError(f"Variable '{variable}' not found.")

    return string


#####################################################################
# INIT CALLBACKS
#####################################################################

def init_callbacks(ckpt_dir):
    # Call back definition
    callbacks = []
    model_ckpts: List[pl_callbacks.ModelCheckpoint] = []

    ckpt_metrics = [str(met) for met in su.init_metrics()]

    for metric in ckpt_metrics:
        model_ckpts.append(
            lit_callbacks.callback_model_checkpoint(
                dirpath=ckpt_dir,
                filename=f"{metric}",
                monitor=f"val_{metric}",
                mode="max",
                save_top_k=1,
                save_last=False,
                every_n_epochs=wandb.config.checkpoint_every_n_epochs,
                every_n_train_steps=wandb.config.checkpoint_every_n_steps,
                verbose=False,
            )
        )


    model_ckpts.append( # train loss checkpoint
        lit_callbacks.callback_model_checkpoint(
            dirpath=ckpt_dir, #None for default logger dir
            filename=f"val_loss",
            monitor=f"val_loss",
            mode="min",
            every_n_epochs=wandb.config.checkpoint_every_n_epochs,
            every_n_train_steps=wandb.config.checkpoint_every_n_steps,
            verbose=False,
        )
    )

    callbacks.extend(model_ckpts)

    if wandb.config.auto_scale_batch_size:
        batch_finder = BatchSizeFinder(mode='power')
        callbacks.append(batch_finder)

    # early_stop_callback = EarlyStopping(monitor=wandb.config.early_stop_metric, 
    #                                     min_delta=0.00, 
    #                                     patience=10, 
    #                                     verbose=False, 
    #                                     mode="max")

    # callbacks.append(early_stop_callback)

    return callbacks


#####################################################################
# INIT MODELS
#####################################################################

def init_scenenet(criterion):

    geneo_config = {
        'cy'   : wandb.config.cylinder_geneo,
        'arrow': wandb.config.arrow_geneo,
        'neg'  : wandb.config.neg_sphere_geneo,
        'disk' : wandb.config.disk_geneo,
        'cone' : wandb.config.cone_geneo,
        'ellip': wandb.config.ellipsoid_geneo, 
    }

    hidden_dims = ast.literal_eval(wandb.config.hidden_dims)         
    num_classes = wandb.config.num_classes

    model = LitSceneNet_multiclass(geneo_num=geneo_config,
                                    num_observers=ast.literal_eval(wandb.config.num_observers),
                                    kernel_size=ast.literal_eval(wandb.config.kernel_size),
                                    hidden_dims=hidden_dims,
                                    num_classes=num_classes,
                                    ignore_index=wandb.config.ignore_index,
                                    criterion=criterion,
                                    optimizer=wandb.config.optimizer,
                                    learning_rate=wandb.config.learning_rate,
                                    metric_initializer=su.init_metrics,
                                )
        
    return model


def init_cnn(criterion):
    from core.lit_modules.lit_cnn import LitCNNBaseline
    from core.lit_modules.lit_scenenet import LitSceneNet_multiclass_CNN

    if wandb.config.model == 'cnn_scenenet':
        geneo_config = {
            'cy'   : wandb.config.cylinder_geneo,
            'arrow': wandb.config.arrow_geneo,
            'neg'  : wandb.config.neg_sphere_geneo,
            'disk' : wandb.config.disk_geneo,
            'cone' : wandb.config.cone_geneo,
            'ellip': wandb.config.ellipsoid_geneo, 
        }

        model = LitSceneNet_multiclass_CNN(
                    geneo_num=geneo_config,
                    num_observers=ast.literal_eval(wandb.config.num_observers),
                    kernel_size=ast.literal_eval(wandb.config.kernel_size),
                    hidden_dims=ast.literal_eval(wandb.config.hidden_dims),
                    num_classes=wandb.config.num_classes,
                    cnn_out_channels=wandb.config.out_channels,
                    cnn_kernel_size=wandb.config.cnn_kernel_size,
                    ignore_index=wandb.config.ignore_index,
                    criterion=criterion,
                    optimizer=wandb.config.optimizer,
                    learning_rate=wandb.config.learning_rate,
                    metric_initializer=su.init_metrics,
                )
        return model

    # Model definition
    model = LitCNNBaseline(
                    in_channels=wandb.config.in_channels,
                    out_channels=wandb.config.out_channels,
                    kernel_size=wandb.config.cnn_kernel_size,
                    num_groups=wandb.config.num_groups,
                    padding=wandb.config.padding,
                    MLP_hidden_dims=ast.literal_eval(wandb.config.hidden_dims),
                    num_classes=wandb.config.num_classes,
                    ignore_index=wandb.config.ignore_index,
                    criterion=criterion,
                    optimizer=wandb.config.optimizer,
                    learning_rate=wandb.config.learning_rate,
                    metric_initializer=su.init_metrics,
                )
    return model



def init_GENEO_loss(model, base_criterion=None):
    criterion_params = {}

    if 'tversky' in wandb.config.criterion.lower():
        criterion_params = {
            'tversky_alpha': wandb.config.tversky_alpha,
            'tversky_beta': wandb.config.tversky_beta,
            'tversky_smooth': wandb.config.tversky_smooth,
            'focal_gamma': wandb.config.focal_gamma,
        }

    if 'focal' in wandb.config.criterion.lower():
        criterion_params['focal_gamma'] = wandb.config.focal_gamma


    if base_criterion is None:
        criterion_class = su.resolve_criterion(wandb.config.criterion)
        base_criterion = criterion_class(criterion, **criterion_params)
    
    if wandb.config.geneo_criterion:
        criterion = GENEO_Loss(base_criterion, 
                                model.get_geneo_params(),
                                model.get_cvx_coefficients(),
                                convex_weight=wandb.config.convex_weight,
                            )  
    else:
        criterion = base_criterion

    model.criterion = criterion # assign criterion to model
    

#####################################################################
# INIT DATASETS
#####################################################################
# fd654c61852c40948c264d606c81f59a9dddcc67

def init_ts40k(data_path, preprocessed=False):

    sample_types = ['tower_radius']
    
    if preprocessed:
        if wandb.config.model == 'unet':
            # vxg_size = ast.literal_eval(wandb.config.voxel_grid_size) # default is 64^3
            # vox_size = ast.literal_eval(wandb.config.voxel_size) # only use vox_size after training or with batch_size = 1
            # voxel_method = Voxelization_withPCD if wandb.config.model == 'scenenet' else Voxelization
            # transform = voxel_method(keep_labels='all', vxg_size=vxg_size, vox_size=vox_size)
            data_path = TS40K_FULL_PREPROCESSED_VOXELIZED_PATH

            transform = Compose([
                           Ignore_Label(0) # turn noise class into ignore
                        ])

        elif 'scenenet' in wandb.config.model or wandb.config.model == 'cnn':
            vxg_size = ast.literal_eval(wandb.config.voxel_grid_size) # default is 64^3
            vox_size = ast.literal_eval(wandb.config.voxel_size) # only use vox_size after training or with batch_size = 1
            transform = Compose([
                            # Ignore_Label(0), # turn noise class into ignore,
                            Voxelization_withPCD(keep_labels='all', vxg_size=vxg_size, vox_size=vox_size)
                        ])
        else:
            transform = None

        return LitTS40K_FULL_Preprocessed(
                            data_path,
                            wandb.config.batch_size,
                            sample_types=sample_types,
                            transform=transform,
                            transform_test=transform,
                            num_workers=wandb.config.num_workers,
                            val_split=wandb.config.val_split,
                            load_into_memory=wandb.config.load_into_memory,
        )

    if wandb.config.model == 'scenenet' or wandb.config.model == 'unet':
        vxg_size = ast.literal_eval(wandb.config.voxel_grid_size) # default is 64^3
        vox_size = ast.literal_eval(wandb.config.voxel_size) # only use vox_size after training or with batch_size = 1

        voxel_method = Voxelization_withPCD if wandb.config.model == 'scenenet' else Voxelization

        composed = Compose([
                            Farthest_Point_Sampling(wandb.config.fps_points),
                            voxel_method(keep_labels='all', vxg_size=vxg_size, vox_size=vox_size)
                        ])
    else:
        composed = Compose([
                            Normalize_PCD(),
                            Farthest_Point_Sampling(wandb.config.fps_points),
                            To(torch.float32),
                        ])
    
    data_module = LitTS40K_FULL(
                           data_path,
                           wandb.config.batch_size,
                           sample_types=sample_types,
                           task='sem_seg',
                           transform=composed,
                           transform_test=None,
                           num_workers=wandb.config.num_workers,
                           val_split=wandb.config.val_split,
                           load_into_memory=wandb.config.load_into_memory,
                        )
    
    return data_module




def init_model(model_name, criterion) -> pl.LightningModule:
    if model_name == 'scenenet':
        # test_MulticlassJaccardIndex: tensor([0.0000, 0.6459, 0.3951, 0.3087, 0.0633, 0.7802], device='cuda:0'); mean: 0.4387
        return init_scenenet(criterion)
    elif 'cnn' in model_name:
        return init_cnn(criterion)
    else:
        raise ValueError(f"Model {model_name} not supported.")
    

def resume_from_checkpoint(ckpt_path, model:pl.LightningModule, class_weights=None):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint {ckpt_path} does not exist.")
    
    checkpoint = torch.load(ckpt_path)
    # print(f"{checkpoint.keys()}")
    print(f"Loading model from checkpoint {ckpt_path}...\n\n")
    if wandb.config.class_weights and 'pointnet' not in wandb.config.model and 'scenenet' not in wandb.config.model:
        checkpoint['state_dict']['criterion.weight'] = class_weights
    model.load_state_dict(checkpoint['state_dict'])
    print(f"Model loaded from checkpoint {ckpt_path}")
    
    # model_class = model.__class__
    
    # print(f"Resuming from checkpoint {ckpt_path}")
    # model = model_class.load_from_checkpoint(ckpt_path,
    #                                    criterion=criterion,
    #                                    optimizer=wandb.config.optimizer,
    #                                    learning_rate=wandb.config.learning_rate,
    #                                    metric_initilizer=su.init_metrics
    #                                 )
    return model



def init_criterion(class_weights=None):
    
    print("Loss function: ", wandb.config.criterion)
    print(f"{'='*5}> Class weights: {class_weights}")
    criterion = torch.nn.CrossEntropyLoss(ignore_index=wandb.config.ignore_index,
                                          weight=class_weights) # default criterion; idx zero is noise
    return criterion



def main():
    # ------------------------
    # 0 INIT CALLBACKS
    # ------------------------

    if not wandb.config.resume_from_checkpoint:
        ckpt_dir = os.path.join(wandb.run.dir, "checkpoints") 
    else:
        ckpt_dir = wandb.config.checkpoint_dir

    ckpt_dir = replace_variables(ckpt_dir)
    ckpt_path = os.path.join(ckpt_dir, wandb.config.resume_checkpoint_name + '.ckpt')

    callbacks = init_callbacks(ckpt_dir)


    # ------------------------
    # 1 INIT BASE CRITERION
    # ------------------------
    if wandb.config.class_weights:
        alpha, epsilon = 3, 0.1
        class_densities = torch.tensor([0.0702, 0.3287, 0.4226, 0.1495, 0.0046, 0.0244], dtype=torch.float32)
        class_weights = torch.max(1 - alpha*class_densities, torch.full_like(class_densities, epsilon))
        # class_weights = 1 / class_densities
        class_weights[0] = 0.0 # ignore noise class
        # class_weights = class_weights / class_weights.mean()
    else:
        class_weights = None

    criterion = init_criterion(class_weights)

    # ------------------------
    # 2 INIT MODEL
    # ------------------------

    model = init_model(wandb.config.model, criterion)
    # torchinfo.summary(model, input_size=(wandb.config.batch_size, 1, 64, 64, 64))
    
    if wandb.config.resume_from_checkpoint:
        ckpt_path = replace_variables(ckpt_path)
        model = resume_from_checkpoint(ckpt_path, model, class_weights)
    

    if wandb.config.get('geneo_criterion', False):
        init_GENEO_loss(model, base_criterion=criterion)

    # ------------------------
    # 4 INIT DATA MODULE
    # ------------------------

    dataset_name = wandb.config.dataset       
    if dataset_name == 'ts40k':
        data_path = TS40K_FULL_PATH
        if wandb.config.preprocessed:
            data_path = TS40K_FULL_PREPROCESSED_PATH
            if idis_mode:
                data_path = TS40K_FULL_PREPROCESSED_IDIS_PATH
            elif smote_mode:
                data_path = TS40K_FULL_PREPROCESSED_SMOTE_PATH
        data_module = init_ts40k(data_path, wandb.config.preprocessed)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")
    
    wandb.config.update({'data_path': data_path}, allow_val_change=True) # override data path

    print(f"\n=== Data Module {dataset_name.upper()} initialized. ===\n")
    print(f"{data_module}")
    print(data_path)
    
    # ------------------------
    # 5 INIT TRAINER
    # ------------------------

    # WandbLogger
    wandb_logger = WandbLogger(project=f"{project_name}",
                               log_model=True, 
                               name=wandb.run.name, 
                               config=wandb.config
                            )
    
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=callbacks,
        detect_anomaly=False,
        max_epochs=wandb.config.max_epochs,
        accelerator=wandb.config.accelerator,
        devices=wandb.config.devices,
        num_nodes=wandb.config.num_nodes,
        strategy=wandb.config.strategy,
        profiler=wandb.config.profiler if wandb.config.profiler else None,
        precision=wandb.config.precision,
        enable_model_summary=True,
        enable_checkpointing=True,
        enable_progress_bar=True,
        accumulate_grad_batches = wandb.config.accumulate_grad_batches,
    )

    if not prediction_mode:

        trainer.fit(model, data_module)

        print(f"{'='*20} Model ckpt scores {'='*20}")

        for ckpt in trainer.callbacks:
            if isinstance(ckpt, pl_callbacks.ModelCheckpoint):
                print(f"{ckpt.monitor} checkpoint : score {ckpt.best_model_score}")
            
            
    # ------------------------
    # 6 TEST
    # ------------------------

    if not os.path.exists(ckpt_path):
        print(f"Checkpoint {ckpt_path} does not exist. Using last checkpoint.")
        ckpt_path = None

    if wandb.config.save_onnx:
        print("Saving ONNX model...")
        onnx_file_path = os.path.join(ckpt_dir, f"{project_name}.onnx")
        input_sample = next(iter(data_module.test_dataloader()))
        model.to_onnx(onnx_file_path, input_sample, export_params=True)
        wandb_logger.log({"onnx_model": wandb.File(onnx_file_path)})

    
    
    test_results = trainer.test(model,
                                datamodule=data_module,
                                ckpt_path='best' if not prediction_mode else None,
                            )

if __name__ == '__main__':

    # --------------------------------
    su.fix_randomness()
    warnings.filterwarnings("ignore")
    torch.set_float32_matmul_precision('high')
    torch.autograd.set_detect_anomaly(True)
    # --------------------------------


    # is cuda available
    print(f"{'='*50} CUDA available: {torch.cuda.is_available()} {'='*50}")
    # get device specs
    print(f"{'='*3}> Device specs: {torch.cuda.get_device_properties(0)}")

    main_parser = su.main_arg_parser().parse_args()
    
    model_name = main_parser.model
    dataset_name = main_parser.dataset
    project_name = f"TS40K_SoA"

    prediction_mode = main_parser.predict
    idis_mode = main_parser.idis
    smote_mode = main_parser.smote

    # config_path = get_experiment_config_path(model_name, dataset_name)
    experiment_path = get_experiment_path(model_name, dataset_name)

    os.environ["WANDB_DIR"] = os.path.abspath(os.path.join(experiment_path, 'wandb'))
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # raw_config = yaml.safe_load(open(config_path))
    # pprint(raw_config)

    print(f"\n\n{'='*50}")
    print("Entering main method...") 

    if main_parser.wandb_sweep: 
        #sweep mode
        print("wandb sweep.")
        wandb.init(project=project_name, 
                dir = experiment_path,
                name = f'{project_name}_{model_name}_{datetime.now().strftime("%Y%m%d-%H%M%S")}',
        )
    else:
        # default mode
        sweep_config = os.path.join(experiment_path, 'defaults_config.yml')

        print("wandb init.")

        wandb.init(project=project_name, 
                dir = experiment_path,
                name = f'{model_name}_{dataset_name}_{datetime.now().strftime("%Y%m%d-%H%M%S")}',
                config=sweep_config,
                mode=main_parser.wandb_mode,
        )        

    main()
    

    




