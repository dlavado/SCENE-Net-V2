

from pathlib import Path
import sys
from typing import Tuple, Union

sys.path.insert(0, '..')
import numpy as np
import pandas as pd
import utils.pointcloud_processing as eda
import laspy as lp
import torch
import os
import webcolors

ROOT_PROJECT = str(Path(os.getcwd()).parent.absolute())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#ROOT_PROJECT = "/home/didi/VSCode/lidar_thesis"

DATA_DIR = os.path.join(ROOT_PROJECT, "/dataset")



###############################################################
#                    Plotting Functions                       #
###############################################################

def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c/255 - requested_colour[0]) ** 2
        gd = (g_c/255 - requested_colour[1]) ** 2
        bd = (b_c/255 - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def get_colour_name(requested_colour):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return actual_name, closest_name


def plot_voxelgrid(grid:Union[np.ndarray, torch.Tensor], color_mode='density', title='VoxelGrid', visual=False, plot=True, **kwargs):
    """
    Plots voxel-grid.\\
    Color of the voxels depends on the values of each voxel;

    Parameters
    ----------
    `grid` - np.3Darray:
        voxel-grid with len(grid.shape) == 3

    `color_mode` - str:
        How to color the voxel-grid;
        color_mode \in ['density', 'ranges']

            `density` mode - colors the points as [-1, 0[ - blue; ~0 white; ]0, 1] - red

            `ranges` mode - selects colors for specific ranges of values according to the 'jet' colormap
    
    `title` - str:
        Title for the visualization window of the voxelgrid

    `visual` - bool:
        If True, it shows a legend for the point cloud visualization

    `plot` - bool:
        If True, it plots the voxelgrid; if False, it returns the voxelgrid colored accordingly.


    Returns
    -------
    if plot is True:
        None
    else:
        colored pointcloud in (x, y, z, r, g, b) format
    """

    if isinstance(grid, torch.Tensor):
        grid = grid.cpu().numpy()

    z, x, y = grid.nonzero()

    xyz = np.empty((len(z), 4))
    idx = 0
    for i, j, k in zip(x, y, z):
        if np.abs(grid[k, i, j]) < 0.01: # if close to zero, do not include
            continue
        xyz[idx] = [int(i), int(j), int(k), grid[k, i, j]]
        idx += 1

    if len(xyz) == 0:
        return
    
    uq_classes = np.unique(xyz[:, -1])
    # print(f"Unique classes: {uq_classes}")
    class_colors = np.empty((len(uq_classes), 3))

    if color_mode == 'density': # colored according to 'coolwarm' scheme
        for i, c in enumerate(uq_classes):
        # [-1, 0[ - blue; ~0 white; ]0, 1] - red
            if c < 0:
                class_colors[i] = [1+c, 1+c, 1]
            else:
                class_colors[i] = [1, 1-c, 1-c]
    # meant to be used only when `grid` contains values \in [0, 1]
    elif color_mode == 'ranges': #colored according to the ranges of values in `grid`
        import matplotlib.cm as cm
        r = 10
        step = (1 / r) / 2
        lin = np.linspace(0, 1, r) 
        # color for each range
        color_ranges = cm.jet(lin) # shape = (r, 4); 4 = (r, g, b, a)
        color_ranges[0] = [1, 1, 1, 0] # [0, 0.111] -> force color white 

        xyz = xyz[xyz[:, -1] > lin[1]] # voxels with the color white are eliminated for a better visual
        #xyz = np.delete(xyz, np.arange(0, len(xyz))[xyz[:, -1] < lin[1]], axis=0) # voxels with the color white are eliminated for a better visual
        uq_classes = np.unique(xyz[:, -1])
    
        # idx in `color_ranges` for each `uq_classes`
        color_idxs = np.argmin(np.abs(np.expand_dims(uq_classes, -1) - lin - step), axis=-1) # len == len(uq_classes)

        class_colors = np.empty((len(uq_classes), 3))
        for i, c in enumerate(uq_classes): 
            class_colors[i] = color_ranges[color_idxs[i]][:-1]

        if visual:
            print('Ranges Colors:')
            for i in range(r-1):
                print(f"[{lin[i]:.3f}, {lin[i+1]:.3f}[ : {get_colour_name(color_ranges[i][:-1])[1]}")
    else:
        ValueError(f"color_mode must be in ['coolwarm', 'ranges']; got {color_mode}")
    
    # class_colors = class_colors * (class_colors > 0) # remove negative color values

    colors = np.array([class_colors[np.where(uq_classes == c)[0][0]] for c in xyz[:, -1]])
    pcd = eda.np_to_ply(xyz[:, :-1])
    xyz_colors = eda.color_pointcloud(pcd, xyz[:, -1], colors=colors)

    if not plot:
        return np.concatenate((xyz[:, :-1], xyz_colors*255), axis=1) # 255 to convert color to int8

    eda.visualize_ply([pcd], window_name=title) # visualize the point cloud




###############################################################
#                  Voxelization Functions                     #
###############################################################


def voxelize_sample(xyz, labels, keep_labels, voxelgrid_dims=(64, 64, 64), voxel_dims=None):
    """
    Voxelizes the point cloud xyz and applies a histogram function on each voxel as a density function.

    Parameters
    ----------
    `xyz` - numpy array:
        point cloud in (N, 3) format.

    `labels` - 1d numpy array:
        point labels in (1, N) format.

    `keep_labels` - int or list:
        labels to be kept in the voxelization process.

    `voxegrid_dims` - tuple int:
        Dimensions of the voxel grid to be applied to the point cloud

    `voxel_dims` - tuple int:
        Dimensions of the voxels that compose the voxel_grid that will encase the point cloud
        if voxel_dims is not None, it overrides voxelgrid_dims;
    
    Returns
    -------
    `in` - np.ndarray with voxel_dims shape    
        voxelized data with histogram density functions
    
    `gt` - np.ndarray with voxel_dims shape
        voxelized labels with histogram density functions
    
    `point_locations` - np.ndarray withg shape (N, 3)
        point locations in the voxel grid
    """
    to_tensor = False
    if isinstance(xyz, torch.Tensor):
        xyz = xyz.numpy()
        to_tensor = True
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()

    crop_tower_ply = eda.np_to_ply(xyz)
    pynt, id = eda.voxelize_ply(crop_tower_ply, voxelgrid_dims=voxelgrid_dims, voxel_dims=voxel_dims, plot=False)
    grid = pynt.structures[id]
    grid_shape = grid.x_y_z

    inp = np.zeros((grid_shape[-1], grid_shape[0], grid_shape[1]))
    
    gt = np.copy(inp)

    voxs = pd.DataFrame(data = {
                            "z": grid.voxel_z, 
                            "x": grid.voxel_x, 
                            "y": grid.voxel_y,
                            "points": np.ones_like(grid.voxel_x), 
                            "labels": labels
                           }
                        )
    
    groups = voxs.groupby(['z', 'x', 'y'])

    point_locations = np.column_stack((grid.voxel_z, grid.voxel_x, grid.voxel_y))

    def voxel_label(x):
        group = np.array(x)
        keep = group[np.isin(group, keep_labels)]
        if len(keep) == 0:
            return 0.0
        label, count = np.unique(keep, return_counts=True)
        label = label[np.argmax(count)] # performs a mode operation
        return label

    aggs = groups.agg({'labels': voxel_label, 'points': 'count'})

    for zxy, row in aggs.iterrows():
        inp[zxy] = 1.0 if row['points'] > 0 else 0.0

        gt[zxy] = eda.DICT_NEW_LABELS[row['labels']] # convert EDP labels to semantic labels

    if to_tensor:
        inp = torch.from_numpy(inp).unsqueeze(0)
        gt = torch.from_numpy(gt).unsqueeze(0)
        point_locations = torch.from_numpy(point_locations)
    
    return inp, gt, point_locations




def voxelize_input_pcd(xyz, labels, keep_labels='all', voxelgrid_dims=(64, 64, 64), voxel_dims=None):
    """
    Voxelizes the point cloud xyz and applies a histogram function on each voxel as a density function.

    Parameters
    ----------
    `xyz` - numpy array:
        point cloud in (N, 3) format.

    `labels` - 1d numpy array:
        point labels in (1, N) format.

    `keep_labels` - str or list:
        labels to be kept in the voxelization process.

    `voxegrid_dims` - tuple int:
        Dimensions of the voxel grid to be applied to the point cloud

    `voxel_dims` - tuple int:
        Dimensions of the voxels that compose the voxel_grid that will encase the point cloud
        if voxel_dims is not None, it overrides voxelgrid_dims;
    
    Returns
    -------
    `in` - np.ndarray with voxel_dims shape    
        voxelized data with histogram density functions
    
    `gt` - np.ndarray with shape (1, N) and semantic labels
    
    `point_locations` - np.ndarray withg shape (N, 3)
        point locations in the voxel grid
    """
    to_tensor = False
    req_grad = False
    to_cuda = False
    if isinstance(xyz, torch.Tensor):
        if xyz.requires_grad:
            xyz.data = xyz.detach()
            req_grad = True
        if xyz.is_cuda:
            xyz = xyz.cpu()
            to_cuda = True
        xyz = xyz.numpy()
        to_tensor = True
    if isinstance(labels, torch.Tensor):
        if labels.requires_grad:
            labels.data = labels.detach()
        if labels.is_cuda:
            labels = labels.cpu()
        labels = labels.numpy()

    crop_tower_ply = eda.np_to_ply(xyz)
    pynt, id = eda.voxelize_ply(crop_tower_ply, voxelgrid_dims=voxelgrid_dims, voxel_dims=voxel_dims, plot=False)
    grid = pynt.structures[id]
    grid_shape = grid.x_y_z

    inp = np.zeros((grid_shape[-1], grid_shape[0], grid_shape[1]))

    voxs = pd.DataFrame(data = {
                            "z": grid.voxel_z, 
                            "x": grid.voxel_x, 
                            "y": grid.voxel_y,
                            "points": np.ones_like(grid.voxel_x), 
                           }
                        )
    
    groups = voxs.groupby(['z', 'x', 'y'])

    point_locations = np.column_stack((grid.voxel_z, grid.voxel_x, grid.voxel_y))

    aggs = groups.agg({'points': 'count'})

    for zxy, row in aggs.iterrows():
        inp[zxy] = 1.0 if row['points'] > 0 else 0.0

    if keep_labels != 'all':
        def change_label(x):
            return x if x in keep_labels else 0
        labels = np.vectorize(change_label)(labels)

    if to_tensor:
        inp = torch.from_numpy(inp).unsqueeze(0)
        labels = torch.from_numpy(labels).to(torch.long)
        point_locations = torch.from_numpy(point_locations)

    if req_grad:
        inp.requires_grad = True
        labels.requires_grad = True
        point_locations.requires_grad = True
    
    if to_cuda:
        inp = inp.cuda()
        labels = labels.cuda()
        point_locations = point_locations.cuda()
    
    return inp, labels, point_locations




def torch_voxelize_input_pcd(xyz:torch.Tensor, labels:torch.Tensor, keep_labels='all', voxelgrid_dims=(64, 64, 64), voxel_dims=None):
    """
    Voxelizes the point cloud xyz and applies a histogram function on each voxel as a density function.

    Parameters
    ----------
    `xyz` - torch.Tensor:
        point cloud in (N, 3) format.

    `labels` - torch.Tensor:
        point labels in (1, N) format.

    `keep_labels` - str or list:
        labels to be kept in the voxelization process.

    `voxelgrid_dims` - tuple int:
        Dimensions of the voxel grid to be applied to the point cloud

    `voxel_dims` - tuple int:
        Dimensions of the voxels that compose the voxel_grid that will encase the point cloud
        if voxel_dims is not None, it overrides voxelgrid_dims;

    Returns
    -------
    `inp` - torch.Tensor with voxel_dims shape    
        voxelized data with histogram density functions

    `gt` - torch.Tensor with shape (1, N) and semantic labels

    `point_locations` - torch.Tensor with shape (N, 3)
        point locations in the voxel grid
    """

    # switch z axis with x axis in xyz input
    xyz = torch.concatenate([xyz[:, [2]], xyz[:, :2]], dim=1)

    # Calculate voxel size
    if voxel_dims is not None:
        voxel_size = torch.tensor(voxel_dims, device=xyz.device)
        voxelgrid_dims = torch.floor((xyz.max(0)[0] - xyz.min(0)[0]) / voxel_size).to(torch.long) + 1 #+1 to make it so that the voxel_indices fall within
    elif voxelgrid_dims is not None:
        voxelgrid_dims = torch.tensor(voxelgrid_dims, device=xyz.device)
        voxel_size = (xyz.max(0)[0] - xyz.min(0)[0]) / (voxelgrid_dims - 1.1) # so that the voxel_size calculated fit the voxel indices
    else:
        raise ValueError("Either voxel_dims or voxelgrid_dims must be specified")

    # Calculate voxel indices for each point
    voxel_indices = ((xyz - xyz.min(0)[0]) / voxel_size).to(torch.long)

    # Calculate voxel indices for each point, so voxel_indices is (N, 3), where each value in \in [0, voxelgrid_dims - 1]
    # voxel_indices = torch.clamp(((xyz - xyz.min(0)[0]) / voxel_size), torch.zeros_like(voxelgrid_dims), voxelgrid_dims - 1).to(torch.long)

    grid_shape = voxelgrid_dims.long()

    # Compute the indices for each point
    z_indices, x_indices, y_indices = voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]

    # Compute the number of points in each voxel
    voxel_grid = torch.zeros(grid_shape.tolist(), dtype=torch.float32, device=xyz.device)
    voxel_grid[z_indices, x_indices, y_indices] += 1

    # Filter labels if specified
    if keep_labels != 'all':
        mask = torch.tensor([label in keep_labels for label in labels], dtype=torch.bool)
        labels = labels[mask]

    return voxel_grid, labels, voxel_indices



def torch_voxelize_pcd_gt(xyz:torch.Tensor, labels:torch.Tensor, keep_labels='all', voxelgrid_dims=(64, 64, 64), voxel_dims=None):
    """
    Voxelizes the point cloud xyz and applies a histogram function on each voxel as a density function.

    Parameters
    ----------
    `xyz` - torch.Tensor:
        point cloud in (N, 3) format.

    `labels` - torch.Tensor:
        point labels in (1, N) format.

    `keep_labels` - str or list:
        labels to be kept in the voxelization process.

    `voxelgrid_dims` - tuple int:
        Dimensions of the voxel grid to be applied to the point cloud

    `voxel_dims` - tuple int:
        Dimensions of the voxels that compose the voxel_grid that will encase the point cloud
        if voxel_dims is not None, it overrides voxelgrid_dims;

    Returns
    -------
    `inp` - torch.Tensor with voxel_dims shape    
        voxelized data with histogram density functions

    `gt` - torch.Tensor with voxel_dims shape
        voxelized labels with semantic labels (empty spaces are labeled with -1)
    """

    # switch z axis with x axis in xyz input
    xyz = torch.concatenate([xyz[:, [2]], xyz[:, :2]], dim=1)

    # Calculate voxel size
    if voxel_dims is not None:
        voxel_size = torch.tensor(voxel_dims, device=xyz.device)
        voxelgrid_dims = torch.floor((xyz.max(0)[0] - xyz.min(0)[0]) / voxel_size).to(torch.long) + 1 #+1 to make it so that the voxel_indices fall within
    elif voxelgrid_dims is not None:
        voxelgrid_dims = torch.tensor(voxelgrid_dims, device=xyz.device)
        voxel_size = (xyz.max(0)[0] - xyz.min(0)[0]) / (voxelgrid_dims - 1.1) # so that the voxel_size calculated fit the voxel indices
    else:
        raise ValueError("Either voxel_dims or voxelgrid_dims must be specified")

    # Calculate voxel indices for each point
    voxel_indices = ((xyz - xyz.min(0)[0]) / voxel_size).to(torch.long) # (N, 3)
    grid_shape = voxelgrid_dims.long()

    # Compute the indices for each point
    z_indices, x_indices, y_indices = voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]

    # Compute the number of points in each voxel
    voxel_grid = torch.zeros(grid_shape.tolist(), dtype=torch.float32, device=xyz.device)
    voxel_grid[z_indices, x_indices, y_indices] += 1

    # Filter labels if specified
    if keep_labels != 'all':
        mask = torch.tensor([label in keep_labels for label in labels], dtype=torch.bool)
        labels = labels[mask]

    # Compute the labels for each voxel
    labels_grid = torch.full(grid_shape.tolist(), -1, dtype=torch.long, device=xyz.device) # empty voxels are labeled with -1
    # a voxel takes the label os the most frequent label in the voxel
    for z, x, y in zip(z_indices, x_indices, y_indices): # for each point
        if labels_grid[z, x, y] == -1:
            voxel_labels = labels[(z_indices == z) & (x_indices == x) & (y_indices == y)]  # select labels within the voxel
        if voxel_labels.numel() > 0:  # if there are labels within the voxel
            mode_label = voxel_labels.mode()[0]  # calculate mode
            labels_grid[z, x, y] = mode_label  # assign mode to the voxel
        else:
            labels_grid[z, x, y] = -1  # assign -1 if the voxel is empty
    
    # print(torch.unique(labels_grid))

    # xyzl = torch_voxel_to_pointcloud(labels_grid)
    # xyzl = xyzl[xyzl[:, -1] != -1].numpy()
    # print(xyzl.shape)
    # print(np.unique(xyzl[:, -1]))
    # eda.plot_pointcloud(xyzl[:, :3], xyzl[:, -1], window_name='Original Point Cloud', use_preset_colors=False)

    # plot_voxelgrid(labels_grid / torch.max(labels_grid), color_mode='ranges', title='Voxelized Point Cloud')
    

    return voxel_grid, labels_grid



def vxg_to_xyz(vxg:torch.Tensor, origin = None, voxel_size = None) -> None:
    """
    Converts voxel-grid to a raw point cloud.\\
    The selected voxels to represent the raw point cloud have label == 1.0\n

    Parameters
    ----------
    `vxg` - torch.Tensor:
        voxel-grid to be transformed with shape (64, 64, 64) for instance

    `origin` - np.ndarray:
        (3,) numpy array that encodes the origin of the voxel-grid

    `voxel_size` - np.ndarray:
        (3,) numpy array that encodes the voxel size of the voxel-grid

    Returns
    -------
    `points` - np.ndarray:
        (N, 4) numpy array that encodes the raw pcd.
    """
    # point_cloud_np = np.asarray([voxel_volume.origin + pt.grid_index*voxel_volume.voxel_size for pt in voxel_volume.get_voxels()])

    shape = vxg.shape
    origin = np.array([0, 0, 0]) if origin is None else origin
    voxel_size = np.array([1, 1, 1]) if voxel_size is None else voxel_size
    grid_indexes = np.indices(shape).reshape(3, -1).T

    points = origin + grid_indexes * voxel_size

    labels = np.array([vxg[tuple(index)] for index in grid_indexes])

    return np.concatenate((points, labels.reshape(-1, 1)), axis=1)


def voxel_to_pointcloud(voxelgrid:np.ndarray, point_locations:np.ndarray):
    """
    Converts a voxelgrid to a pointcloud given the point locations inside the voxelgrid.
    """

    if point_locations is None:
        # transform voxel grid to point cloud
        xyz = np.argwhere(voxelgrid > -1)
        xyz = np.concatenate((xyz, voxelgrid[voxelgrid > -1].reshape(-1, 1)), axis=1)
        return xyz
    
    voxel_values = np.array([voxelgrid[tuple(point)] for point in point_locations])

    return np.concatenate((point_locations, voxel_values.reshape(-1, 1)), axis=1)


def torch_voxel_to_pointcloud(voxelgrid:torch.Tensor, point_locations:torch.Tensor=None):

    if point_locations is None:
        # transform voxel grid to point cloud
        xyz = torch.argwhere(voxelgrid > -1)
        xyz = torch.cat((xyz, voxelgrid[voxelgrid > -1].view(-1, 1)), dim=1) # (N, 4)
        return xyz

    voxel_values = torch.tensor([voxelgrid[tuple(point)] for point in point_locations])
    return torch.cat((point_locations, voxel_values.reshape(-1, 1)), dim=1)


def vox_to_pts(vox:torch.Tensor, pt_loc:torch.Tensor, include_locs:bool=False) -> torch.Tensor:
    """

    Transforms a voxelgrid tensor into a pointcloud tensor

    Parameters
    ----------

    `vox`: torch.Tensor
        Voxelgrid tensor of shape (batch, 1, z, x, y)
    
    `pt_loc`: torch.Tensor
        Pointcloud locations tensor of shape (batch, P, 3) where P is the max number of points in the input batch (padded with zeros)

    `include_locs`: bool
        If True, the output tensor will include the point locations in the voxelgrid. 
        If False, the output tensor will only include the point values in the voxelgrid.

    Returns
    -------
    `pt`: torch.Tensor
        Pointcloud tensor of shape (batch, P, 3+C | C) where P is the max number of points in the input batch (padded with zeros)

    """

    batch_size = vox.shape[0]

    #Gather the pointcloud values from the voxelgrid tensor
    pt_cloud = []
    for i in range(batch_size):
        pt_cloud.append(vox[i, :, pt_loc[i, :, 0].to(torch.int32), pt_loc[i, :, 1].to(torch.int32), pt_loc[i, :, 2].to(torch.int32)]) # (C, P, 1)
    
    pt_cloud = torch.stack(pt_cloud, dim=0) # (batch, C, P)

    # print(pt_loc.shape)
    # print(pt_cloud.shape)
   
    pt_cloud = pt_cloud.permute(0, 2, 1) # (batch, P, C)

    if include_locs:
        # Concatenate pt_loc and pt_cloud tensors
        return torch.cat((pt_loc, pt_cloud), dim=2)  # (batch, P, 3+C)

    return pt_cloud



if __name__ == "__main__":
    
    
    pass

