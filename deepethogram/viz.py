from collections import OrderedDict
import itertools
import logging
import os
import warnings
from typing import Union

import cv2
import h5py
import matplotlib
import numpy as np
# import tifffile as TIFF
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable, inset_locator
import torch

from deepethogram.flow_generator.utils import flow_to_rgb_polar
# from deepethogram.metrics import load_threshold_data
from deepethogram.utils import tensor_to_np

log = logging.getLogger(__name__)
# override warning level for matplotlib, which outputs a million debugging statements
logging.getLogger('matplotlib').setLevel(logging.WARNING)


def imshow_with_colorbar(image: np.ndarray,
                         ax_handle,
                         fig_handle: matplotlib.figure.Figure,
                         clim: tuple = None,
                         cmap: str = None,
                         interpolation: str = None,
                         symmetric: bool = False,
                         func: str = 'imshow',
                         **kwargs) -> matplotlib.colorbar.Colorbar:
    """ Show an image in a matplotlib figure with a colorbar *with the same height as the axis!!*

    Without this function, matplotlib color bars can be taller than the axis which is ugly.

    Parameters
    ----------
    image: np.ndarray. 2-dimensional
        Image to be shown
    ax_handle: matplotlib.axes._subplots.AxesSubplot
        Axis on which to plot
    fig_handle: matplotlib.figure.Figure
        matplotlib figure. Used in colorbar creation
    clim: tuple
        (vmin, vmax). Sets the lower and upper limit in the color scale
    cmap: str
        One of matplotlib's colormaps. https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html
    interpolation: str
        Interpolation for showing the image. From plt docs:
        Supported values are 'none', 'antialiased', 'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36',
        'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian',
        'bessel', 'mitchell', 'sinc', 'lanczos'
    symmetric: bool
        If True, sets clim to be symmetric around zero. Useful for things like z-scored inputs, optic flows, etc.
    func: str
        Which function to use. Default: imshow. Pcolormesh is more useful if you have real-valued inputs

    Returns
    -------
    cbar: matplotlib.colorbar.Colorbar
        handle to colorbar. useful for subsequently adding labels, etc.

    References
    -------
    .. [1]: https://stackoverflow.com/questions/32462881/add-colorbar-to-existing-axis/39938019#39938019
    .. [2]: https://matplotlib.org/3.3.0/api/_as_gen/matplotlib.pyplot.imshow.html
    """
    assert isinstance(ax_handle, matplotlib.axes.SubplotBase)
    # if we get a vector, change into a row
    if image.ndim == 1:
        image = image[np.newaxis, :]
    
    if symmetric:
        cmap = 'bwr'
    divider = make_axes_locatable(ax_handle)
    if func == 'imshow':
        im = ax_handle.imshow(image, interpolation=interpolation, cmap=cmap, **kwargs)
    elif func == 'pcolor' or func == 'pcolormesh':
        im = ax_handle.pcolormesh(image, cmap=cmap, **kwargs)
    if symmetric:
        limit = np.max(np.abs((image.min(), image.max())))
        im.set_clim(-limit, limit)
    if clim is not None:
        im.set_clim(clim[0], clim[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig_handle.colorbar(im, cax=cax)
    return cbar


def stack_image_list(image_list: list, num_cols: int = 4) -> np.ndarray:
    """ Stacks a list of images into one image with a certain number of columns. Used for viewing many images at once

    Parameters
    ----------
    image_list: list
        List of numpy arrays. Can have different heights / widths, but should have same depth
    num_cols: int
        number of columns in the output

    Returns
    -------
    stack: np.ndarray
        stacked image

    Examples
    -------
    # make a list of 10 images
    image_list = [np.random.uniform(size=(256,256)) for _ in range(10)]
    # result will be a 3 x 4 grid, with the bottom-right two elements being zeros
    stack = stack_image_list(image_list, num_cols=4)
    print(stack.shape)
    # (768, 1024)
    """
    sequence_length = len(image_list)
    num_rows = int(np.ceil(sequence_length / num_cols))
    row_shape = None
    rows = []
    for i in range(num_rows):
        start = i * num_cols
        end = min(start + num_cols, len(image_list))
        row = np.hstack(image_list[start:end])
        if row_shape is None:
            row_shape = row.shape
        if row.shape[1] < row_shape[1]:
            padval = row_shape[1] - row.shape[1]
            if row.ndim == 2:
                pad_width = ((0, 0), (0, padval))
            elif row.ndim == 3:
                pad_width = ((0, 0), (0, padval), (0, 0))
            else:
                raise ValueError('input with weird shape: {}'.format(row.shape))
            row = np.pad(row, pad_width)
        rows.append(row)
    stack = np.vstack(rows)
    return stack


def plot_flow(rgb, ax, show_scale=True, height=30, maxval: float = 1.0, interpolation='nearest', inset_label=False):
    """ Plots an optic flow in polar coordinates, with an inset colorbar """
    ax.imshow(rgb, interpolation=interpolation)
    if show_scale:
        x = np.linspace(-1, 1, 100)
        y = np.linspace(1, -1, 100)
        xv, yv = np.meshgrid(x, y)
        flow_colorbar = flow_to_rgb_polar(np.dstack((xv, yv)), maxval=1)
        # flow_colorbar = colorize_flow(np.dstack((xv, yv)), maxval=1)
        aspect = ax.get_data_ratio()
        width = int(height * aspect)
        # https://stackoverflow.com/questions/53204267
        inset = inset_locator.inset_axes(ax, width=str(width) + '%',
                                         height=str(height) + '%',
                                         loc=1)
        # axes_class=get_projection_class('polar'))
        inset.imshow(flow_colorbar)
        inset.invert_yaxis()
        if inset_label:
            inset.set_xticks([0, 50, 100])
            inset.set_yticks([0, 50, 100])
            inset.set_xticklabels([-maxval, 0, maxval])
            inset.set_yticklabels([-maxval, 0, maxval])
            ax.grid([0, 50, 100])
        else:
            inset.set_xticklabels([])
            inset.set_yticklabels([])
    else:
        inset = None
    return inset


def visualize_images_and_flows(downsampled_t0, flows_reshaped, sequence_length: int = 10, fig=None,
                               max_flow: float = 5.0, height=15, batch_ind: int = None):
    """ Plot a list of images and optic flows """
    plt.style.use('ggplot')
    if fig is None:
        fig = plt.figure(figsize=(16, 12))

    axes = fig.subplots(2, 1)

    N, C, H, W = downsampled_t0[0].shape
    batch_size = N // sequence_length
    if batch_ind is None:
        batch_ind = np.random.choice(batch_size)

    inds = range(batch_ind*sequence_length, batch_ind*sequence_length + sequence_length)
    images = downsampled_t0[0][inds].detach().cpu().numpy().astype(np.float32)

    # N is actually N * T
    image_list = [i.transpose(1,2,0) for i in images]
    # image_list = [images[i, ...].transpose(1, 2, 0) for i in range(batch_ind * sequence_length,
    #                                                                batch_ind * sequence_length + sequence_length)]
    stack = stack_image_list(image_list)
    minimum, mean, maximum = stack.min(), stack.mean(), stack.max()
    stack = (stack * 255).clip(min=0, max=255).astype(np.uint8)

    ax = axes[0]
    ax.imshow(stack, interpolation='nearest')
    ax.set_title('min: {:.4f} mean: {:.4f} max: {:.4f}'.format(minimum, mean, maximum))
    ax.grid(False)
    ax.axis('off')

    ax = axes[1]
    flows = flows_reshaped[0][inds].detach().cpu().numpy().astype(np.float32)
    flow_list = [i.transpose(1,2,0) for i in flows]
    # flow_list = [flows[i, ...].transpose(1, 2, 0).astype(np.float32) for i in range(batch_ind * sequence_length,
    #                                                                                 batch_ind * sequence_length + sequence_length)]
    stack = stack_image_list(flow_list)
    minimum, mean, maximum = stack.min(), stack.mean(), stack.max()
    stack = flow_to_rgb_polar(stack, maxval=max_flow)
    plot_flow(stack.clip(min=0, max=255).astype(np.uint8), ax, maxval=max_flow, inset_label=True, height=height)

    ax.set_title('min: {:.4f} mean: {:.4f} max: {:.4f}'.format(minimum, mean, maximum))
    ax.grid(False)
    ax.axis('off')

    fig.suptitle('Images and flows. Batch element: {}'.format(batch_ind))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.tight_layout()
    fig.subplots_adjust(top=0.9)
    # plt.show()


def visualize_multiresolution(downsampled_t0, estimated_t0, flows_reshaped, sequence_length: int = 10,
                              max_flow: float = 5.0, height=15, batch_ind: int = None, fig=None,
                              sequence_ind: int = None):
    """ visualize images, optic flows, and reconstructions at multiple resolutions at which the loss is actually
    applied. useful for seeing what the loss function actually sees, and debugging multi-resolution issues
    """
    plt.style.use('ggplot')
    if fig is None:
        fig = plt.figure(figsize=(16, 12))

    # N is actually N * T
    N, C, H, W = downsampled_t0[0].shape
    batch_size = N // sequence_length
    if batch_ind is None:
        batch_ind = np.random.choice(batch_size)
    if sequence_ind is None:
        sequence_ind = np.random.choice(sequence_length)

    # inds = range(batch_ind * sequence_length, batch_ind * sequence_length + sequence_length)

    N_resolutions = len(downsampled_t0)

    axes = fig.subplots(4, N_resolutions)

    images = downsampled_t0[0].detach().cpu().numpy().astype(np.float32)

    index = batch_ind * sequence_length + sequence_ind
    t0 = [downsampled_t0[i][index].detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32) for i in
          range(N_resolutions)]

    for i, image in enumerate(t0):
        ax = axes[0, i]
        if i == 0:
            ax.set_ylabel('T0', fontsize=18)
        minimum, mean, maximum = image.min(), image.mean(), image.max()
        image = (image * 255).clip(min=0, max=255).astype(np.uint8)
        ax.imshow(image, interpolation='nearest')
        ax.set_title('min: {:.4f} mean: {:.4f} max: {:.4f}'.format(minimum, mean, maximum),
                     fontsize=8)

    t1 = [estimated_t0[i][index].detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32) for i in
          range(N_resolutions)]
    for i, image in enumerate(t1):
        ax = axes[1, i]
        minimum, mean, maximum = image.min(), image.mean(), image.max()
        image = (image * 255).clip(min=0, max=255).astype(np.uint8)
        ax.imshow(image, interpolation='nearest')
        ax.set_title('min: {:.4f} mean: {:.4f} max: {:.4f}'.format(minimum, mean, maximum),
                     fontsize=8)
        if i == 0:
            ax.set_ylabel('T1', fontsize=18)

    flows = [flows_reshaped[i][index].detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32) for i in
             range(N_resolutions)]
    for i, image in enumerate(flows):
        ax = axes[2, i]
        minimum, mean, maximum = image.min(), image.mean(), image.max()
        flow_im = flow_to_rgb_polar(image, maxval=max_flow)
        plot_flow(flow_im, ax, maxval=max_flow)
        ax.set_title('min: {:.4f} mean: {:.4f} max: {:.4f}'.format(minimum, mean, maximum),
                     fontsize=8)
        if i == 0:
            ax.set_ylabel('Flow', fontsize=18)

    L1s = [np.sum(np.abs(t0[i] - t1[i]), axis=2) for i in range(N_resolutions)]
    for i, image in enumerate(L1s):
        ax = axes[3, i]
        minimum, mean, maximum = image.min(), image.mean(), image.max()
        ax.imshow(image, interpolation='nearest')
        ax.set_title('min: {:.4f} mean: {:.4f} max: {:.4f}'.format(minimum, mean, maximum),
                     fontsize=8)
        if i == 0:
            ax.set_ylabel('L1', fontsize=18)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.tight_layout()


def tensor_to_list(images: torch.Tensor, batch_ind: int, channels: int = 3) -> list:
    if images.ndim == 4:
        N, C, H, W = images.shape
        sequence_length = C // channels
        image_list = [images[batch_ind, i * channels:i * channels + channels, ...].transpose(1, 2, 0)
                      for i in range(sequence_length)]
    elif images.ndim == 5:
        N, C, T, H, W = images.shape
        image_list = [images[batch_ind, :, i, ...].transpose(1, 2, 0) for i in range(T)]
    else:
        raise ValueError('weird shape of input: {}'.format(images.shape))
    return image_list


def predictions_labels_string(pred, label, class_names=None):
    if class_names is None:
        class_names = [i for i in range(len(pred))]
    inds = np.argsort(pred)[::-1]
    string = 'label: '
    if label.ndim > 0:
        for i in range(len(label)):
            if label[i] == 1:
                string += '{} '.format(class_names[i])
        string += '\n'
    else:
        string += '{}'.format(label)
    for i in range(10):
        if i >= len(inds):
            break
        ind = inds[i]
        string += '{}: {:.3f} '.format(class_names[ind], pred[ind])
        if (i % 5) == 4:
            string += '\n'
    return string


def visualize_hidden(images, flows, predictions, labels, class_names: list = None, batch_ind: int = None,
                     max_flow: float = 5.0, height: float = 15.0, fig=None, normalizer=None):
    """ Visualize inputs and outputs of a hidden two stream model """
    # import pdb; pdb.set_trace()
    plt.style.use('ggplot')
    if fig is None:
        fig = plt.figure(figsize=(16, 12))

    axes = fig.subplots(2, 1)

    # images = downsampled_t0[0].detach().cpu().numpy()
    # if normalizer is not None:
    #     images = normalizer.denormalize(images)
    batch_size = images.shape[0]
    if batch_ind is None:
        batch_ind = np.random.choice(batch_size)

    images = images.detach().cpu().numpy()
    image_list = tensor_to_list(images, batch_ind)
    del images

    stack = stack_image_list(image_list)
    minimum, mean, maximum = stack.min(), stack.mean(), stack.max()
    stack = (stack * 255).clip(min=0, max=255).astype(np.uint8)

    ax = axes[0]
    ax.imshow(stack, interpolation='nearest')
    ax.set_title('min: {:.4f} mean: {:.4f} max: {:.4f}'.format(minimum, mean, maximum), fontsize=8)
    ax.grid(False)
    ax.axis('off')

    ax = axes[1]
    flows = flows.detach().cpu().numpy()
    flow_list = tensor_to_list(flows, batch_ind, 2)
    stack = stack_image_list(flow_list)
    minimum, mean, maximum = stack.min(), stack.mean(), stack.max()
    stack = flow_to_rgb_polar(stack, maxval=max_flow)
    plot_flow(stack, ax, maxval=max_flow, inset_label=True, height=height)

    #     inset.set_xticklabels([-max_flow, 0, max_flow])
    #     inset.set_yticklabels([-max_flow, 0, max_flow])
    ax.set_title('min: {:.4f} mean: {:.4f} max: {:.4f}'.format(minimum, mean, maximum), fontsize=8)
    ax.grid(False)
    ax.axis('off')

    pred = predictions[batch_ind].detach().cpu().numpy()
    label = labels[batch_ind].detach().cpu().numpy()
    string = predictions_labels_string(pred, label, class_names)

    fig.suptitle(string)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.tight_layout()
    fig.subplots_adjust(top=0.9)

    # print_top_largest_variables(locals())
    del stack, pred, label


def to_uint8(im: np.ndarray) -> np.ndarray:
    """ helper function for converting from [0,1] float to [0, 255] uint8 """
    return (im.copy() * 255).clip(min=0, max=255).astype(np.uint8)


def visualize_batch_unsupervised(downsampled_t0, estimated_t0, flows_reshaped, batch_ind=0, sequence_ind: int = 0,
                                 fig=None, sequence_length: int = 10):
    """ Visualize t0, t1, optic flow, reconstruction, and the L1 between t0 and estimated t0 """
    plt.style.use('ggplot')
    if fig is None:
        fig = plt.figure(figsize=(16, 12))

    axes = fig.subplots(3, 2)

    index = batch_ind * sequence_length + sequence_ind

    ax = axes[0, 0]
    t0 = downsampled_t0[0][index].detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32)
    ax.imshow(to_uint8(t0), interpolation='nearest')
    ax.set_title('min: {:.4f} max: {:.4f}'.format(t0.min(), t0.max()))

    ax = axes[0, 1]
    t1 = downsampled_t0[0][index + 1].detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32)
    ax.imshow(to_uint8(t1), interpolation='nearest')
    ax.set_title('min: {:.4f} max: {:.4f}'.format(t1.min(), t1.max()))

    ax = axes[1, 0]
    flow = flows_reshaped[0][index].detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32)
    imshow_with_colorbar(flow[..., 0], ax, fig, symmetric=True, interpolation='nearest')

    ax = axes[1, 1]
    imshow_with_colorbar(flow[..., 1], ax, fig, symmetric=True, interpolation='nearest')

    ax = axes[2, 0]
    est = estimated_t0[0][index].detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32)
    ax.imshow(to_uint8(est), interpolation='nearest')

    ax = axes[2, 1]
    L1 = np.abs(est - t0.astype(np.float32)).sum(axis=2)
    imshow_with_colorbar(L1, ax, fig, interpolation='nearest')
    # pdb.set_trace()
    ax.set_title('L1')
    plt.tight_layout()


def visualize_batch_spatial(images, predictions, labels, fig=None, class_names=None, num_cols: int=4):
    """ visualize spatial stream of hidden two stream model """

    plt.style.use('ggplot')
    if fig is None:
        fig = plt.figure(figsize=(16, 12))

    batch_size = images.shape[0]
    num_rows = int(min(np.ceil(batch_size / num_cols), 6))

    total_images = min(num_rows*num_cols, batch_size)

    # only use the first total_images elements in the batch, to try to reduce RAM usage
    images = images[:total_images].detach().cpu().numpy()
    predictions = predictions[:total_images].detach().cpu().numpy()
    labels = labels[:total_images].detach().cpu().numpy()

    images = images.clip(min=0, max=1)



    axes = fig.subplots(num_rows, num_cols)
    cnt = 0
    if num_rows == 1:
        axes = axes[np.newaxis, ...]
    for i in range(num_rows):
        for j in range(num_cols):
            ax = axes[i, j]
            if cnt >= batch_size:
                ax.remove()
                cnt += 1
                continue
            pred = predictions[cnt]
            label = labels[cnt]
            string = predictions_labels_string(pred, label, class_names)
            string = '{:03d}: '.format(cnt) + string

            # spatial stream should almost always be one single image
            image = tensor_to_list(images, cnt)[0]

            ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])

            ax.set_title(string, size=8)
            cnt += 1
    fig.suptitle('Spatial stream')
    plt.tight_layout()
    del images, predictions, labels

def visualize_batch_sequence(sequence, outputs, labels, N_in_batch=None, fig=None):
    """ Visualize an input sequence, probabilities, and the true labels """
    if fig is None:
        fig = plt.figure(figsize=(16, 12))

    if N_in_batch is None:
        N_in_batch = np.random.choice(outputs.shape[0])

    sequence = tensor_to_np(sequence[N_in_batch])
    outputs = tensor_to_np(outputs[N_in_batch])
    labels = tensor_to_np(labels[N_in_batch])

    # import pdb; pdb.set_trace()

    axes = fig.subplots(4, 1)

    ax = axes[0]
    # seq = sequence[N_in_batch]
    aspect_ratio = outputs

    # tmp = outputs[N_in_batch]
    # seq = cv2.resize(sequence[N_in_batch], (tmp.shape[1]*10,tmp.shape[0]*10), interpolation=cv2.INTER_NEAREST)
    # seq = cv2.imresize(sequence[N_in_batch], )
    imshow_with_colorbar(sequence, ax, fig, interpolation='nearest',
                         symmetric=False, func='pcolor', cmap='viridis')
    ax.invert_yaxis()
    ax.set_ylabel('inputs')

    ax = axes[1]
    imshow_with_colorbar(outputs, ax, fig, interpolation='nearest', symmetric=False, cmap='Reds',
                         func='pcolor', clim=[0, 1])
    ax.invert_yaxis()
    ax.set_ylabel('P')

    ax = axes[2]
    imshow_with_colorbar(labels, ax, fig, interpolation='nearest', cmap='Reds', func='pcolor')
    ax.invert_yaxis()
    ax.set_ylabel('Labels')

    ax = axes[3]
    dumb_loss = np.abs(outputs - labels)
    imshow_with_colorbar(dumb_loss, ax, fig, interpolation='nearest', cmap='Reds', func='pcolor', clim=[0, 1])
    ax.set_title('L1 between outputs and labels (not true loss)')
    ax.invert_yaxis()
    plt.tight_layout()
    del sequence, outputs, labels


def fig_to_img(fig_handle: matplotlib.figure.Figure) -> np.ndarray:
    """ Convenience function for returning the RGB values of a matplotlib figure """
    # should do nothing if already drawn
    fig_handle.canvas.draw()
    # from stack overflow
    data = np.array(fig_handle.canvas.renderer._renderer)[:, :, 0:3]
    return data


# def image_list_to_tiff_stack(images, tiff_fname):
#     """ Write a list of images to a tiff stack using tifffile """
#     # WRITE ALL TO TIFF!
#     height = images[0].shape[0]
#     width = images[0].shape[1]
#     channels = images[0].shape[2]
#     N = len(images)
#     fig_mat = np.empty([N, height, width, channels], dtype='uint8')
#     for i in range(N):
#         img = images[i]
#         if img.shape != fig_mat.shape[1:2]:
#             img = cv2.resize(img, (fig_mat.shape[2], fig_mat.shape[1]), interpolation=cv2.INTER_LINEAR)
#         img = np.uint8(img)

#         fig_mat[i, :, :, :] = img
#     TIFF.imsave(tiff_fname, fig_mat, photometric='rgb', compress=0, metadata={'axes': 'TYXC'})


def plot_histogram(array, ax, bins='auto', width_factor=0.9, rotation=30):
    """ Helper function for plotting a histogram """
    if type(array) != np.ndarray:
        array = np.array(array)

    hist, bin_edges = np.histogram(array, bins=bins, density=False)
    hist = hist / np.sum(hist)
    center = (bin_edges[:-1] + bin_edges[1:]) / 2
    width = width = np.diff(center)[0] * width_factor
    ax.bar(center, hist, width=width)

    # from http://blog.quizzicol.com/2016/09/13/rotate-x-axis-tick-labels-in-matplotlib-subplots/
    for tick in ax.get_xticklabels():
        tick.set_rotation(rotation)
    med = np.median(array)
    ylims = ax.get_ylim()

    leg_str = 'median: %0.4f' % (med)
    lineh = ax.plot(np.array([med, med]), np.array([ylims[0], ylims[1]]),
                    color='k', linestyle='dashed', lw=3, label=leg_str)
    ax.set_ylabel('P')
    ax.legend()


def errorfill(x, y, yerr, color=None, alpha_fill=0.3, ax=None, label=None):
    """ Convenience function for plotting a shaded error bar """
    ax = ax if ax is not None else plt.gca()
    #     if color is None:
    #         color = ax._get_lines.color_cycle.next()
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    ax.plot(x, y, color=color, label=label)
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)


def plot_curve(x, ys, ax, xlabel: str = None, class_names=None, colors=None):
    """ Plots a set of curves. Will add a scatter to the maximum of each curve with text indicating location """
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if x is None:
        x = np.arange(ys.shape[0])
    if ys.ndim > 1:
        K = ys.shape[1]
    else:
        K = 1
        ys = ys[..., np.newaxis]
    if class_names is None:
        class_names = [str(i) for i in range(K)]
    for i in range(K):
        ax.plot(x, ys[:, i], label=class_names[i])
        index = np.argmax(ys[:, i])
        max_acc = ys[index, i]
        scatter_x, scatter_y = remove_nan_or_inf(x[index]), remove_nan_or_inf(max_acc)
        ax.scatter(scatter_x, scatter_y)
        text = '{:.2f}, {:.2f}'.format(x[index], max_acc)
        text_x = x[index] + np.random.randn() / 20
        text_y = max_acc + np.random.randn() / 20
        text_x, text_y = remove_nan_or_inf(text_x), remove_nan_or_inf(text_y)
        if i < len(colors):
            color = colors[i]
        else:
            color = colors[-1]
        ax.text(text_x, text_y, text, color=color)
        ax.set_ylim([0, 1])
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    ax.legend()


def thresholds_by_epoch_figure(epoch_summaries, class_names=None, fig=None):
    plt.style.use('ggplot')

    if fig is None:
        fig = plt.figure(figsize=(14, 14))

    ax = fig.add_subplot(2, 3, 1)
    split = 'train'

    keys = ['accuracy', 'accuracy_valid_bg']
    arr = np.vstack(([epoch_summaries[split][key] for key in keys])).T
    plot_curve(None, arr, ax, class_names=keys)
    ax.set_ylabel('Train')

    ax = fig.add_subplot(2, 3, 2)
    keys = ['f1_by_class', 'f1_by_class_valid_bg', 'f1_overall', 'f1_overall_valid_bg']
    arr = np.vstack(([epoch_summaries[split][key] for key in keys])).T
    plot_curve(None, arr, ax, class_names=keys)
    # ax.set_ylabel('Train')

    ax = fig.add_subplot(2, 3, 3)
    keys = ['auroc', 'auroc_by_class', 'mAP', 'mAP_by_class']
    arr = np.vstack(([epoch_summaries[split][key] for key in keys])).T
    plot_curve(None, arr, ax, class_names=keys)

    ax = fig.add_subplot(2, 3, 4)
    split = 'val'
    keys = ['accuracy', 'accuracy_valid_bg']
    arr = np.vstack(([epoch_summaries[split][key] for key in keys])).T
    plot_curve(None, arr, ax, class_names=keys)

    ax.set_ylabel('Validation')

    ax = fig.add_subplot(2, 3, 5)
    keys = ['f1_by_class', 'f1_by_class_valid_bg', 'f1_overall', 'f1_overall_valid_bg']
    arr = np.vstack(([epoch_summaries[split][key] for key in keys])).T
    plot_curve(None, arr, ax, class_names=keys)

    ax = fig.add_subplot(2, 3, 6)
    keys = ['auroc', 'auroc_by_class', 'mAP', 'mAP_by_class']
    arr = np.vstack(([epoch_summaries[split][key] for key in keys])).T
    plot_curve(None, arr, ax, class_names=keys)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.tight_layout()


# TWEAKED FROM SCIKIT-LEARN
def plot_confusion_matrix(cm, classes, ax, fig,
                          normalize=False,
                          title='Confusion matrix',
                          cmap='Blues', colorbar=True, fontsize=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-7)
        # print("Normalized confusion matrix")
    else:
        # print('Confusion matrix, without normalization')
        pass

    # print(cm)
    if colorbar:
        cbar = imshow_with_colorbar(cm, ax, fig, interpolation='nearest', cmap=cmap)

    else:
        ax.imshow(cm, cmap=cmap)
    # ax.set_title(title)
    tick_marks = np.arange(0, len(classes))
    ax.set_xticks(tick_marks)
    ax.tick_params(axis='x', rotation=45)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    fmt = '.2f' if normalize else 'd'
    if not normalize:
        cm = cm.astype(int)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        j, i = remove_nan_or_inf(j), remove_nan_or_inf(i)
        element = cm[i,j]
        if element < 1e-2:
            element = 0
            fmt = 'd'
        else:
            fmt = '.2f' if normalize else 'd'
        text = format(element, fmt)
        if text.startswith('0.'):
            text = text[1:]
        ax.text(j, i, text,
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=fontsize)
    ax.set_xlim([-0.5, len(classes) - 0.5])
    ax.set_ylim([len(classes) - 0.5, -0.5])
    plt.tight_layout()
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')


def remove_nan_or_inf(value: Union[int, float]):
    """ removes nans or infs. can happen in edge cases for plotting """
    if np.isnan(value) or np.isinf(value):
        return 0
    return value


# def plot_metrics(logger_file, fig):
#     """ plot all metrics in a Metrics hdf5 file. see deepethogram.metrics """
#     splits = ['train', 'val']
#     num_cols = 2
#
#     with h5py.File(logger_file, 'r') as f:
#         for split in splits:
#             keys = list(f[split].keys())
#             # all metrics files will have loss and time
#             num_custom_vars = len(keys) - 2
#             if 'confusion' in keys:
#                 num_custom_vars -= 1
#     num_rows = int(np.ceil(num_custom_vars / num_cols)) + 1
#
#     forbidden = ['loss', 'time', 'confusion']
#
#     shape = (num_rows, num_cols)
#     with h5py.File(logger_file, 'r') as f:
#         ax = fig.add_subplot(num_rows, num_cols, 1)
#         plot_metric(f, ax, 'loss', legend=True)
#         ax = fig.add_subplot(num_rows, num_cols, 2)
#         plot_metric(f, ax, 'time')
#         cnt = 3
#         for key in keys:
#             if key in forbidden:
#                 continue
#             ax = fig.add_subplot(num_rows, num_cols, cnt)
#             cnt += 1
#             plot_metric(f, ax, key)
#         keys = f.attrs.keys()
#         args = {}
#         for key in keys:
#             args[key] = f.attrs[key]
#     # title = 'Project {}: model:{} \nNotes: {}'.format(args['name'], args['model'], args['notes'])
#     # fig.suptitle(title, size=18)
#     plt.tight_layout()
#     fig.subplots_adjust(top=0.9)


def plot_confusion_from_logger(logger_file, fig, class_names=None, epoch=None):
    """ Plots train and validation confusion matrices from a Metrics file """
    with h5py.File(logger_file, 'r') as f:
        best_epoch = np.argmax(f['val/' + f.attrs['key_metric']][:])
        if epoch is None:
            epoch = best_epoch
        if epoch == 'last':
            epoch = -1
        splits = list(f.keys())
        if 'train' in splits:
            cm_train = f['train/confusion'][epoch, ...].astype(np.int64)
        else:
            cm_train = np.array([np.nan])
        if 'val' in splits:
            cm_val = f['val/confusion'][epoch, ...].astype(np.int64)
        else:
            cm_val = np.array([np.nan])
    if class_names is None:
        class_names = np.arange(cm_train.shape[0])
    ax0 = fig.add_subplot(221)
    plot_confusion_matrix(cm_train, class_names, ax0, fig)
    ax0.set_title('Train')
    ax1 = fig.add_subplot(222)
    plot_confusion_matrix(cm_train, class_names, ax1, fig,
                          normalize=True)

    ax0 = fig.add_subplot(223)
    plot_confusion_matrix(cm_val, class_names, ax0, fig)
    ax0.set_title('Val')
    ax1 = fig.add_subplot(224)
    plot_confusion_matrix(cm_val, class_names, ax1, fig,
                          normalize=True)
    fig.suptitle('Confusion matrices at epoch: %d' % (epoch))
    plt.subplots_adjust(top=0.8)
    plt.tight_layout()


def make_precision_recall_figure(logger_file, fig=None, splits=['train', 'val']):
    """ Plots precision vs recall """
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if fig is None:
        fig = plt.figure(figsize=(14, 7))

    for i, split in enumerate(splits):
        ap_by_class = load_logger_data(logger_file, 'mAP_by_class', split)
        precision = load_logger_data(logger_file, 'precision', split, is_threshold=True)
        recall = load_logger_data(logger_file, 'recall', split, is_threshold=True)

        ax = fig.add_subplot(1, len(splits), i + 1)
        # precision, recall = train_metrics['precision'], train_metrics['recall']

        K = precision.shape[1]
        for j in range(K):
            color = colors[j % len(colors)]
            x = recall[:, j]
            y = precision[:, j]
            # there's a bug in how this is computed
            au_prc = ap_by_class[j]
            string = '{}: {:.4f}'.format(j, au_prc)
            ax.plot(x, y, color=color, label=string)
            ax.set_aspect('equal', 'box')
        ax.legend()
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(split)

    fig.suptitle('Precision vs recall. Legend: Average Precision\nNote: curves are approximated with only ' +
                 '101 thresholds. Legend is exact')
    plt.tight_layout()
    return fig


def add_text_to_line(xs, ys, ax, color):
    if len(xs) == 1 or len(ys) == 1:
        return
    x, y = xs[-1], ys[-1]
    if np.isinf(x) or np.isnan(x) or np.isinf(y) or np.isnan(y):
        return
    # x, y = remove_nan_or_inf(x), remove_nan_or_inf(y)
    ax.text(x, y, '{:.4f}'.format(y), color=color)


def plot_metric(data: Union[dict, OrderedDict], name, ax, legend: bool = False, plot_args: dict = None,
                color_inds: list = None):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # data = {'train': train, 'val': val}
    for i, (split, array) in enumerate(data.items()):
        xs = np.arange(len(array))
        # use modulos to make the colors cycle if there are more items than there are colors
        if color_inds is not None:
            color_ind = color_inds[i] % len(colors)
        else:
            color_ind = i % len(colors)
        color = colors[color_ind]
        if plot_args is not None and split in plot_args.keys():
            ax.plot(xs, array, label=split, **plot_args[split], color=color)
        else:
            ax.plot(xs, array, label=split, color=color)
        add_text_to_line(xs, array, ax, color)

    ax.set_xlim([-0.5, len(xs) - 0.5])
    ax.set_ylabel(name)
    ax.set_xlabel('Epochs')
    ax.set_title(name)
    if legend:
        ax.legend()


def make_learning_curves_figure_multilabel_classification(logger_file, fig=None):
    def get_data_from_file(f, name):
        data = OrderedDict(train=f[f'train/{name}_overall'][:],
                           train_class_mean=f[f'train/{name}_class_mean'][:],
                           val=f[f'val/{name}_overall'][:],
                           val_class_mean=f[f'val/{name}_class_mean'][:])
        return data

    with h5py.File(logger_file, 'r') as f:
        plt.style.use('seaborn')
        if fig is None:
            fig = plt.figure(figsize=(12, 12))

        # loss and learning rate
        ax = fig.add_subplot(4, 2, 1)
        data = OrderedDict(train=f['train/loss'][:],
                           val=f['val/loss'][:])
        # import pdb; pdb.set_trace()
        plot_metric(data, 'loss', ax)
        ax2 = ax.twinx()
        ax2.plot(f['train/lr'][:], 'k', label='LR', alpha=0.5)
        ax2.set_ylabel('learning rate')
        ax2.grid(False)

        ax = fig.add_subplot(4, 2, 2)
        data = OrderedDict(train=f['train/data_loss'][:],
                           val=f['val/data_loss'][:])
        # import pdb; pdb.set_trace()
        plot_metric(data, 'data_loss', ax)
        
        ax = fig.add_subplot(4, 2, 3)
        data = OrderedDict(train=f['train/reg_loss'][:],
                           val=f['val/reg_loss'][:])
        # import pdb; pdb.set_trace()
        plot_metric(data, 'reg_loss', ax)

        # FPS
        ax = fig.add_subplot(4, 2, 4)
        try:
            data = OrderedDict(train=f['train/fps'][:],
                               val=f['val/fps'][:],
                               speedtest=f['speedtest/fps'][:])
        except Exception:
            # likely don't have speedtest, not too important
            data = OrderedDict(train=f['train/fps'][:],
                               val=f['val/fps'][:])

        plot_metric(data, 'FPS', ax, legend=True)
        ax.semilogy()

        # accuracy
        ax = fig.add_subplot(4, 2, 5)
        data = OrderedDict(train=f['train/accuracy_overall'][:],
                           val=f['val/accuracy_overall'][:])

        plot_metric(data, 'accuracy', ax)

        # F1 score!
        ax = fig.add_subplot(4, 2, 6)
        data = OrderedDict(train=f['train/f1_overall'][:],
                           train_class_mean=f['train/f1_class_mean'][:],
                           train_class_mean_nobg=f['train/f1_class_mean_nobg'][:],
                           val=f['val/f1_overall'][:],
                           val_class_mean=f['val/f1_class_mean'][:], 
                           val_class_mean_nobg=f['val/f1_class_mean_nobg'][:])
        # we'll reuse these for the following figures
        plot_args = {'train_class_mean': {'linestyle': '--'},
                     'train_class_mean_nobg': {'linestyle': 'dotted'},
                     'val_class_mean': {'linestyle': '--'},
                     'val_class_mean_nobg': {'linestyle': 'dotted'},}
        color_inds = [0, 0, 0, 1, 1, 1]
        # data = get_data_from_file(f, 'f1')
        plot_metric(data, 'F1', ax, True, plot_args, color_inds)

        # AUROC
        plot_args = {'train_class_mean': {'linestyle': '--'},
                     'val_class_mean': {'linestyle': '--'}}
        color_inds = [0, 0, 1, 1]
        ax = fig.add_subplot(4, 2, 7)
        data = get_data_from_file(f, 'auroc')
        plot_metric(data, 'AUROC', ax, False, plot_args, color_inds)

        # Average precision
        ax = fig.add_subplot(4, 2, 8)
        data = get_data_from_file(f, 'mAP')
        plot_metric(data, 'Average Precision', ax, False, plot_args, color_inds)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.tight_layout()
    return fig


def plot_multilabel_by_class(logger_file):
    def load_data(f, name):
        data = {'train': f[f'train/{name}_by_class'][:],
                'val': f[f'val/{name}_by_class'][:]}
        return data

    with h5py.File(logger_file, 'r') as f:

        def plot_row(row, name, legend: bool = False, title: bool = False):
            data = load_data(f, name)

            for i, split in enumerate(['train', 'val']):
                array = data[split]
                ax = row[i]
                # loop over classes
                class_data = OrderedDict()
                for j in range(array.shape[1]):
                    class_data[j] = array[:, j]
                plot_metric(class_data, name, ax, legend and i == 0)
                ax.set_xlabel('')
                if title:
                    ax.set_title(split)
                else:
                    ax.set_title('')

        fig, axes = plt.subplots(4, 2, figsize=(8, 12))

        row = axes[0]
        plot_row(row, 'accuracy', True, True)

        row = axes[1]
        plot_row(row, 'f1')

        row = axes[2]
        plot_row(row, 'auroc')

        row = axes[3]
        plot_row(row, 'mAP')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plt.tight_layout()
    return fig


def load_logger_data(logger_file, name, split, is_threshold: bool = False, epoch: int = -1):
    if is_threshold:
        key = f'{split}/metrics_by_threshold/{name}'
    else:
        key = f'{split}/{name}'
    with h5py.File(logger_file, 'r') as f:
        data = f[key][epoch, ...]

    return data


def make_thresholds_figure(logger_file, split, fig=None, class_names=None):
    plt.style.use('seaborn')

    if fig is None:
        fig = plt.figure(figsize=(12, 12))
    #     axes = axes.flatten()

    x = load_logger_data(logger_file, 'thresholds', split, True)

    for i, metric in enumerate(['accuracy', 'f1', 'precision', 'recall', 'informedness']):
        ax = fig.add_subplot(3, 2, i + 1)
        y = load_logger_data(logger_file, metric, split, True)
        plot_curve(x, y, ax, class_names)
        ax.set_title(f'{metric} by class')

    plt.tight_layout()
    return fig


def make_roc_figure(logger_file, fig=None, splits=['train', 'val']):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if fig is None:
        fig = plt.figure(figsize=(14, 7))

    for i, split in enumerate(splits):

        auroc_by_class = load_logger_data(logger_file, 'auroc_by_class', split)
        tpr = load_logger_data(logger_file, 'tpr', split, is_threshold=True)
        fpr = load_logger_data(logger_file, 'fpr', split, is_threshold=True)

        ax = fig.add_subplot(1, len(splits), i + 1)

        K = tpr.shape[1]
        for j in range(K):
            color = colors[j] if j < len(colors) else colors[-1]
            auroc = auroc_by_class[j]
            string = '{}: {:4f}'.format(j, auroc)
            ax.plot(fpr[:, j], tpr[:, j], color=color, label=string)
        ax.legend()
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.set_title(split)
    fig.suptitle('ROC Curves. Curves are approximate because only 101 thresholds were used. AUC values are precise')
    plt.tight_layout()
    return fig


def visualize_binary_confusion(logger_file, fig=None, splits=['train', 'val']):
    """ Visualizes binary confusion matrices """
    if fig is None:
        fig = plt.figure(figsize=(14, 14))

    cms = load_logger_data(logger_file, 'binary_confusion', 'train')
    # if there's more than 3 dimensions, it could be [epochs, classes, 2, 2]
    # take the last one
    if cms.ndim > 3:
        cms = cms[-1, ...]
    K = cms.shape[0]

    num_rows = len(splits)*2
    num_cols = K
    ind = 1

    # print(cms.shape)

    def plot_cms_in_row(cms, ylabel, normalize: bool = False):
        nonlocal ind
        for j in range(num_cols):
            ax = fig.add_subplot(num_rows, num_cols, ind)
            cm = cms[j, ...]
            # print(cm.shape)
            plot_confusion_matrix(cms[j, ...], range(cm.shape[0]),
                                  ax, fig, colorbar=False, normalize=normalize)
            if j == 0:
                ax.set_ylabel(ylabel)
                ax.set_xlabel('')
            else:
                ax.set_ylabel('')
                ax.set_xlabel('')
            ind += 1

    for split in splits:
        cms = load_logger_data(logger_file, 'binary_confusion', split)
        plot_cms_in_row(cms, split)
        plot_cms_in_row(cms, f'{split}\nNormalized', normalize=True)

    plt.tight_layout()
    return fig


def visualize_logger_multilabel_classification(logger_file):
    """ makes a bunch of figures from a Metrics hdf5 file """
    plt.style.use('seaborn')
    fig = make_learning_curves_figure_multilabel_classification(logger_file)
    save_figure(fig, 'learning_curves', False, 0)

    fig = plot_multilabel_by_class(logger_file)
    save_figure(fig, 'learning_curves_by_class', False, 1)

    fig = make_thresholds_figure(logger_file, 'train')
    save_figure(fig, 'thresholds_this_epoch_train', False, 2)

    fig = make_thresholds_figure(logger_file, 'val')
    save_figure(fig, 'thresholds_this_epoch_val', False, 3)

    fig = visualize_binary_confusion(logger_file)
    save_figure(fig, 'binary_confusion', False, 4)

    fig = make_roc_figure(logger_file)
    save_figure(fig, 'ROC', False, 5)

    fig = make_precision_recall_figure(logger_file)
    save_figure(fig, 'precision_recall', False, 6)

    try:
        splits = ['train', 'val', 'test']
        fig = make_thresholds_figure(logger_file, 'test')
        save_figure(fig, 'thresholds_this_epoch_test', False, 7)

        fig = visualize_binary_confusion(logger_file, splits=splits)
        save_figure(fig, 'binary_confusion_with_test', False, 8)

        fig = make_roc_figure(logger_file, splits=splits)
        save_figure(fig, 'ROC_with_test', False, 9)

        fig = make_precision_recall_figure(logger_file, splits=['train', 'val', 'test'])
        save_figure(fig, 'precision_recall_withtest', False, 10)

    except Exception as e:
        # no test set yet
        log.debug('error in test set viz: {}'.format(e))
        # pass
    plt.close('all')


def make_learning_curves_figure_opticalflow(logger_file, fig=None):
    if fig is None:
        fig = plt.figure(figsize=(12, 12))

    def get_data(h5py_obj, name):
        data = OrderedDict(train=h5py_obj[f'train/{name}'][:],
                           val=h5py_obj[f'val/{name}'][:])
        return data

    ax = fig.add_subplot(4, 2, 1)
    with h5py.File(logger_file, 'r') as f:
        data = get_data(f, 'loss')
        plot_metric(data, 'loss', ax)

        ax2 = ax.twinx()

        ax2.plot(f['train/lr'][:], 'k', label='LR', alpha=0.5)
        ax2.set_ylabel('learning rate')
        ax2.grid(False)

        keys = list(f['train'].keys())

        plot_ind = 2
        for metric in ['fps','reg_loss', 'SSIM', 'L1', 'smoothness', 'sparsity']:
            if metric in keys:
                ax = fig.add_subplot(4, 2, plot_ind)
                data = get_data(f, metric)
                plot_metric(data, metric, ax, legend=metric == 'fps')

                plot_ind += 1

    plt.tight_layout()
    return fig


def visualize_logger_optical_flow(logger_file):
    """ makes a bunch of figures from a Metrics hdf5 file """
    plt.style.use('seaborn')
    fig = make_learning_curves_figure_opticalflow(logger_file)
    save_figure(fig, 'learning_curves', False, 0)


hues = [212, 4, 121, 36, 55, 276, 237, 299, 186]
hues = np.array(hues) / 360 * 180
saturation = .85 * 255
value = .95 * 255
start = [0, 0, value]

gray_value = 102


class Mapper:
    """ Applies a custom colormap to a K x T matrix. Used in the GUI to visualize probabilities and labels """

    def __init__(self, colormap='deepethogram'):
        if colormap == 'deepethogram':
            self.init_deepethogram()
        else:
            try:
                self.cmap = plt.get_cmap(colormap)
            except ValueError:
                raise ('Colormap not in matplotlib''s defaults! {}'.format(colormap))

    def init_deepethogram(self):
        gray_LUT = make_LUT([0, 0, value], [0, 0, gray_value])
        LUTs = []
        for hue in hues:
            LUTs.append(make_LUT([hue, 0, value], [hue, saturation, value]))
        self.gray_LUT = gray_LUT
        self.LUTs = LUTs
        self.cmap = self.apply_cmaps

    def apply_cmaps(self, array: Union[np.ndarray, int, float]) -> np.ndarray:
        # assume columns are timepoints, rpws are behaviors
        if type(array) == int or type(array) == float:
            # use the 0th LUT by default
            return apply_cmap(array, self.LUTs[0])
        elif array.shape[0] == 1 and len(array.shape) == 1:
            return apply_cmap(array[0], self.LUTs[0])

        # print('array shape apply cmaps: {}'.format(array.shape))
        K, T = array.shape
        ims = []
        for k in range(K):
            if k == 0:
                # print('gray')
                ims.append(apply_cmap(array[k, :], self.gray_LUT))
            else:
                # print('not gray')
                ims.append(apply_cmap(array[k, :], self.LUTs[k % len(self.LUTs)]))
        # print('im shape: {}'.format(ims[0].shape))

        # mapped = np.ascontiguousarray(np.vstack(ims).swapaxes(1,0))
        mapped = np.vstack(ims)
        # import pdb
        # pdb.set_trace()
        # print('output of apply cmaps: {}'.format(mapped))
        return mapped

    def __call__(self, array: Union[np.ndarray, int, float]) -> np.ndarray:
        return self.cmap(array)


def make_LUT(start_hsv: Union[tuple, list, np.ndarray], end_hsv: Union[tuple, list, np.ndarray]) -> np.ndarray:
    if type(start_hsv) != np.ndarray:
        start_hsv = np.array(start_hsv).astype(np.uint8)
    if type(end_hsv) != np.ndarray:
        end_hsv = np.array(end_hsv).astype(np.uint8)

    # interpolate in HSV space; if they have two different hues, will result in very weird colormap
    interped = np.stack([np.linspace(start_hsv[i], end_hsv[i], 256) for i in range(3)]).T
    # make it a 1 row, many column, 3 channel image for opencv's LUT function
    interped = np.round(interped).astype(np.uint8)[None, ...]
    rgb = cv2.cvtColor(interped, cv2.COLOR_HSV2RGB)
    return rgb


def apply_cmap(array: Union[np.ndarray, int, float], LUT: np.ndarray) -> np.ndarray:
    single_input = False
    if type(array) == int:
        assert array >= 0 and array <= 255
        array = np.array([array]).astype(np.uint8)
        single_input = True
    elif type(array) == float:
        array = np.array([array]).astype(float)
        single_input = True
    if array.dtype != np.uint8:
        if array.min() >= 0 and array.max() <= 1:
            # assume we want to map [0,1] -> [0, 255]
            array = (array * 255).clip(min=0, max=255).astype(np.uint8)
        elif array.min() >= 0 and array.max() <= 255:
            array = array.astype(np.uint8)
        else:
            raise ValueError('Float arrays must be in the range of either [0, 1] or [0, 255], not [{},{}]'.format(
                array.min(), array.max()))

    if LUT.dtype != np.uint8:
        raise ValueError('LUT must be uint8, not {}'.format(LUT.dtype))
    if len(array.shape) < 2:
        array = np.vstack([array, array, array]).T[None, ...]
    elif array.shape[1] != 3:
        array = np.dstack([array, array, array])

    # add an opacity channel, change from integer to float (from opencv convention to matplotlib cmap convention)
    mapped = cv2.LUT(array, LUT).astype(np.float64) / 255
    if single_input:
        mapped = tuple(np.concatenate([mapped.squeeze(), [1]]))
    else:
        mapped = np.concatenate((mapped, np.ones_like(mapped[..., 0:1])), axis=2)
    return mapped


def plot_ethogram(ethogram: np.ndarray, mapper, start_index: Union[int, float],
                  ax, classes: list = None, rotation: int = 15, ylabel: str = None):
    """ Visualizes a K x T ethogram using some mapper """
    # assume inputs is T x K
    im = mapper(ethogram.T)
    im_h = ax.imshow(im, aspect='auto', interpolation='nearest')
    xticks = ax.get_xticks()
    new_ticks = [i + start_index for i in xticks]
    ax.set_xticklabels([str(int(i)) for i in new_ticks])
    ax.set_yticks(np.arange(0, ethogram.shape[1]))
    if classes is not None:
        ax.set_yticklabels(classes, rotation=rotation, fontdict={'fontsize': 12})
    ax.set_ylabel(ylabel)
    return im_h


def make_ethogram_movie(outfile: Union[str, bytes, os.PathLike],
                        ethogram: np.ndarray,
                        mapper,
                        frames: list,
                        start: int,
                        classes: list,
                        width: int = 100,
                        fps: float = 30):
    """ Makes a movie out of an ethogram. Can be very slow due to matplotlib's animations """
    if mapper is None:
        mapper = Mapper()

    fig = plt.figure(figsize=(10, 12))
    # camera = Camera(fig)

    # ethogram_keys = list(ethogram.keys())
    # ethograms = list(ethogram.values())
    # n_ethograms = len(ethograms)
    gs = fig.add_gridspec(3, 1)
    ax0 = fig.add_subplot(gs[0:2])
    ax1 = fig.add_subplot(gs[2])

    starts = np.arange(0, ethogram.shape[0], width)

    if type(classes) != np.ndarray:
        classes = np.array(classes)

    framenum = 0

    im_h = ax0.imshow(frames[0])
    etho_h = plot_ethogram(ethogram[starts[0]:starts[0] + width, :],
                           mapper, start + framenum, ax1, classes)
    ylim = ax1.get_ylim()
    x = (0, 1, 1, 0, 0)
    y = (ylim[0], ylim[0], ylim[1], ylim[1], ylim[0])
    plot_h = ax1.plot(x, y, color='k', lw=0.5)[0]
    title_h = ax0.set_title('{:,}: {}'.format(start, classes[np.where(ethogram[0])[0]].tolist()))
    plt.tight_layout()

    # etho_h = plot_ethogram(ethogram[starts[0]:starts[0] + width, :],
    #                        mapper, start + framenum, ax1, classes)

    def init():
        return [im_h, etho_h, plot_h, title_h]

    def animate(i):
        # print(i)
        im_h.set_data(frames[i])
        x0 = i - starts[i // width] - 0.5
        x1 = x0 + 1
        x = (x0, x1, x1, x0, x0)
        # print(x)
        if (i % width) == 0:
            etho_h = plot_ethogram(ethogram[starts[i // width]:starts[i // width] + width, :],
                                   mapper, start + i, ax1, classes)
            # no idea why plot ethogram doesn't change this
            xticks = ax1.get_xticks()
            new_ticks = xticks + starts[i // width] + start
            ax1.set_xticklabels([str(int(i)) for i in new_ticks])

        else:
            etho_h = [i for i in ax1.get_children() if type(i) == matplotlib.image.AxesImage][0]
        plot_h.set_xdata(x)

        title_h.set_text('{:,}: {}'.format(start + i, classes[np.where(ethogram[i])[0]].tolist()))
        return [im_h, etho_h, plot_h, title_h]

    anim = FuncAnimation(fig, animate, init_func=init,
                         frames=len(frames), interval=int(1000 / fps), blit=True)
    print('Rendering animation, may take a few minutes...')

    if outfile is None:
        out = anim.to_jshtml()
    else:
        anim.save(outfile, fps=fps)# , extra_args=['-vcodec', 'libx264'])
        out = None
    # have to use this ugly return syntax so that we can close the figure after saving
    plt.close(fig)
    return out


def make_ethogram_movie_with_predictions(outfile: Union[str, bytes, os.PathLike],
                                         ethogram: np.ndarray,
                                         predictions: np.ndarray,
                                         mapper,
                                         frames: list,
                                         start: int,
                                         classes: list,
                                         width: int = 100,
                                         fps: float = 30):
    """ Makes a movie with movie, then ethogram, then model predictions """
    
    if mapper is None:
        mapper = Mapper()
    fig = plt.figure(figsize=(6, 8))
    # camera = Camera(fig)

    gs = fig.add_gridspec(4, 1)
    axes = []
    axes.append(fig.add_subplot(gs[0:2]))
    axes.append(fig.add_subplot(gs[2:3]))
    axes.append(fig.add_subplot(gs[3:]))

    # ax1 = fig.add_subplot(gs[2])
    starts = np.arange(0, ethogram.shape[0], width)

    if type(classes) != np.ndarray:
        classes = np.array(classes)

    framenum = 0

    # values_to_return = []

    im_h = axes[0].imshow(frames[0])

    ax = axes[1]
    im_h1 = plot_ethogram(ethogram[starts[0]:starts[0] + width, :],
                          mapper, start + framenum, ax, classes, ylabel='Labels')
    x = (0, 1, 1, 0, 0)
    ylim = ax.get_ylim()
    y = (ylim[0], ylim[0], ylim[1], ylim[1], ylim[0])
    plot_h1 = ax.plot(x, y, color='k', lw=0.5)[0]

    ax = axes[2]
    im_h2 = plot_ethogram(predictions[starts[0]:starts[0] + width, :],
                          mapper, start + framenum, ax, classes, ylabel='Predictions')
    ylim = ax.get_ylim()
    y = (ylim[0], ylim[0], ylim[1], ylim[1], ylim[0])
    plot_h2 = ax.plot(x, y, color='k', lw=0.5)[0]

    title_h = axes[0].set_title('{:,}'.format(start))

    plt.tight_layout()

    # etho_h = plot_ethogram(ethogram[starts[0]:starts[0] + width, :],
    #                        mapper, start + framenum, ax1, classes)

    def init():
        return [im_h, im_h1, im_h2, plot_h1, plot_h2, title_h]

    def animate(i):
        # values_to_return = []
        # print(i)
        im_h.set_data(frames[i])
        x0 = i - starts[i // width] - 0.5
        x1 = x0 + 1
        x = (x0, x1, x1, x0, x0)
        # print(x)
        if (i % width) == 0:
            im_h1 = plot_ethogram(ethogram[starts[i // width]:starts[i // width] + width, :],
                                  mapper, start + i, axes[1], classes, ylabel='Labels')
            # no idea why plot ethogram doesn't change this
            xticks = axes[1].get_xticks()
            new_ticks = xticks + starts[i // width] + start
            axes[1].set_xticklabels([str(int(i)) for i in new_ticks])

            im_h2 = plot_ethogram(predictions[starts[i // width]:starts[i // width] + width, :],
                                  mapper, start + i, axes[2], classes, ylabel='Predictions')
            # no idea why plot ethogram doesn't change this
            xticks = axes[2].get_xticks()
            new_ticks = xticks + starts[i // width] + start
            axes[2].set_xticklabels([str(int(i)) for i in new_ticks])

        else:
            im_h1 = [i for i in axes[1].get_children() if type(i) == matplotlib.image.AxesImage][0]
            im_h2 = [i for i in axes[2].get_children() if type(i) == matplotlib.image.AxesImage][0]
        plot_h1.set_xdata(x)
        plot_h2.set_xdata(x)

        title_h.set_text('{:,}'.format(start + i))
        return [im_h, im_h1, im_h2, plot_h1, plot_h2, title_h]

    anim = FuncAnimation(fig, animate, init_func=init,
                         frames=len(frames), interval=int(1000 / fps), blit=True)
    print('Rendering animation, may take a few minutes...')

    if outfile is None:
        out = anim.to_jshtml()
    else:
        anim.save(outfile, fps=fps, extra_args=['-vcodec', 'libx264'])
        out = None
    # have to use this ugly return syntax so that we can close the figure after saving
    plt.close(fig)
    return out


def make_figure_filename(name, is_example, num, split='train', overwrite:bool=True):
    basedir = os.path.join(os.getcwd(), 'figures')
    if is_example:
        basedir = os.path.join(basedir, 'examples', split)
    if not os.path.isdir(basedir):
        os.makedirs(basedir)
    fname = os.path.join(basedir, '{:02d}_{}.png'.format(num, name))
    if overwrite:
        return fname
    cnt = 0
    while os.path.isfile(fname):
        fname = os.path.join(basedir, '{:02d}_{}_{}.png'.format(num, name, cnt))
        cnt += 1
    return fname


def save_figure(figure, name, is_example, num, split='train', overwrite:bool=True):
    fname = make_figure_filename(name, is_example, num, split, overwrite)
    figure.savefig(fname)
    plt.close(figure)
    del figure
