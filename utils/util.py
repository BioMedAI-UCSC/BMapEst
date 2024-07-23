import math
import torch

import matplotlib.pyplot as plt
import numpy as np

def gaussian_kernel(l=5, sig=1., out_chan=4):
    """
    Creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    kernel = torch.from_numpy(kernel / np.sum(kernel)).unsqueeze(0)
    kernel = kernel.unsqueeze(0).repeat(out_chan, 1, 1, 1).type(torch.float32)
    return kernel

def sigmoid(trajectory: torch.Tensor, nyquist: torch.Tensor) -> torch.Tensor:
    """Differentiable approximation of the sinc voxel dephasing function.

    The true dephasing function of a sinc-shaped voxel (in real space) is a
    box - function, with the FFT conform size [-nyquist, nyquist[. This is not
    differentiable, so we approximate the edges with a narrow sigmod at
    ±(nyquist + 0.5). The difference is neglegible at usual nyquist freqs.
    """
    return torch.prod(torch.sigmoid(
        (nyquist - trajectory.abs() + 0.5) * 100
    ), dim=1)

def PSNR(gt, pred, reduction=True):
    if reduction:
        mse = torch.mean((gt-pred)**2)
        max_pixel = 1
        psnr = 20 * torch.log10(max_pixel/torch.sqrt(mse))
        return psnr
    else:
        batch_size = gt.shape[0]
        mse = ((gt-pred)**2).reshape(batch_size, -1).mean(1, keepdim=True)
        max_pixel = 1
        psnr = 20 * torch.log10(max_pixel/torch.sqrt(mse))
        return psnr.squeeze()


def DICE(gt: torch.Tensor, pred: torch.Tensor, reduction=True):
    if reduction:
        return ((2. * gt.logical_and(pred))).sum() / (gt.sum() + pred.sum() + 1e-6)
    else:
        batchsize = gt.shape[0]
        return ((
            2 * gt.reshape(batchsize, -1).logical_and(
                pred.reshape(batchsize, -1)
            ).sum(1, keepdim=True)
        ) / (gt.reshape(batchsize, -1).sum(1, keepdim=True) + pred.reshape(
            batchsize, -1
        ).sum(1, keepdim=True) + 1e-6)).squeeze()

def draw_single_image(
    image, clim=(0, 1), image_name=None, save_at=None, figsize=(16, 16),
    axis=True
):
    """
        Method to draw and save a single image.
        Parameter
        ---------
            image: np.array (height, width, channels)
                Image to be saved
            clim: Tuple(float, float)
                Color map limits
            image_name: str
                Image title
            save_at: str
                Location to store the plot
        Return
        ------
            None
    """
    fig = plt.figure(figsize=figsize)
    axes1 = fig.add_subplot(121)
    if not isinstance(clim, type(None)):
        axes1.imshow(image, clim=clim)
    else:
        axes1.imshow(image, cmap='gray')
    if not isinstance(image_name, type(None)):
        axes1.set_title(image_name)
    if not axis:
        plt.axis('off')
    if not isinstance(save_at, type(None)):
        plt.savefig(save_at, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def draw(
    ground_truth, predicted, clim=(0, 1), gt_name="None",
    pred_name="None", save_at=None, figsize=(16, 16)
):
    """
        Method to draw the ground truth and the predicted probability maps
        side-by-side.
        Parameter
        ---------
            ground_truth: np.array (height, width, channels)
                Ground truth probability map
            predicted: np.array (height, width, channels)
                Optimized probability map array
            clim: Tuple(float, float)
                Color map limits
            gt_name: str
                Title for ground truth map
            pred_name: str
                Title for optimized probability map
            save_at: str
                Location to store the plot
        Return
        ------
            None
    """
    fig = plt.figure(figsize=figsize)
    axes1 = fig.add_subplot(121)
    axes2 = fig.add_subplot(122)
    if not isinstance(clim, type(None)):
        axes1.imshow(ground_truth, clim=clim)
    else:
        axes1.imshow(ground_truth, cmap='gray')
    axes1.set_title(gt_name)
    if not isinstance(clim, type(None)):
        axes2.imshow(predicted, clim=clim)
    else:
        axes2.imshow(predicted, cmap='gray')
    axes2.set_title(pred_name)

    if not isinstance(save_at, type(None)):
        plt.savefig(save_at, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def log_args(args, logger):
    """
        Method to log args
        Parameter
        ---------
            args: argparse.ArgumentParser
                Command Line argument parser
            logger: logging
                Python Logger
        Return
        ------
            None
    """
    logger.info("-----------------------")
    logger.info("Command Line Arguments:")
    logger.info("-----------------------")
    for arg in vars(args):
        logger.info(f"{arg} = {getattr(args, arg)}")
    logger.info("-----------------------")
    logger.info("-----------------------")

def generate_prob_map_init_mask(
        prob_map_shape, shape = "ellipse"
    ):
    """
        Method to generate a mask of a given shape

        Parameters
        ----------
            prob_map_shape: (height, width, depth)
            shape: ellipse, circle

        Return
        ------
            mask: (height, width, 1)

    """
    center_height = prob_map_shape[0] // 2
    center_width = prob_map_shape[1] // 2
    mask = torch.zeros(prob_map_shape[:2])
    major_axis_length = (0.8 * prob_map_shape[1]) / 2
    minor_axis_length = (0.75 * prob_map_shape[0]) / 2
    if shape=="ellipse":
        def ellipse_point_checker(point: tuple):
            x_val = point[0]
            y_val = point[1]
            p = (
                math.pow((x_val - center_width), 2) /
                    math.pow(major_axis_length, 2) +
                math.pow((y_val - center_height), 2) /
                    math.pow(minor_axis_length, 2)
            )
            return p<=1
        for y in range(prob_map_shape[0]):
            for x in range(prob_map_shape[1]):
                is_present = int(ellipse_point_checker((x, y)))
                if is_present==0:
                    mask[y, x] = 0.001
                else:
                    mask[y, x] = 0.99
    elif shape=="identity":
        mask = torch.ones(prob_map_shape[:2])
    else:
        raise NotImplementedError(f"{shape} is not implemented yet!")
    return torch.unsqueeze(mask, dim=-1)