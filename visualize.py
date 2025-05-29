import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

# Function copied from implementation in https://github.com/Waphil/ASE_MFG_simulation/blob/main/visualization/basic_plot.py

def basic_multiline_plot(x_arr, y_arr, label_list, ax=None, figsize=None, colors=None, linestyles=None, alphas=None,
                         title=None, x_label=None, y_label=None, grid_kwargs=None, ticklabel_kwargs=None,
                         is_use_scalar_formatter=False, x_tick_major_spacing=None, y_tick_major_spacing=None,
                         x_tick_minor_spacing=None, y_tick_minor_spacing=None,
                         x_scale=None, y_scale=None, x_lim=None, y_lim=None, labels_fontsize=None,
                         legend_title=None, legend_kwargs=None, is_show=True):
    # Allow ax to be passed, which means users can plot this content in a subplot somewhere.
    # If no ax is given, create it.
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        is_created_new_axis = True
    else:
        is_created_new_axis = False

    if colors is None:
        colors = plt.cm.jet(np.linspace(0, 1, y_arr.shape[0])) # Jet = default color map
    if linestyles is None:
        linestyles = ["-" for label in label_list]
    if alphas is None:
        alphas = [1. for label in label_list]

    if title is None:
        title = ""
    if legend_title is None:
        legend_title = ""
    #if legend_kwargs is None:
    #    legend_kwargs = {}

    ax.set_title(title)
    for i, label in enumerate(label_list):
        if np.ndim(x_arr) > 1:
            x_arr_plot = x_arr[i]
        else:
            x_arr_plot = x_arr
        ax.plot(x_arr_plot, y_arr[i], color=colors[i], linestyle=linestyles[i], alpha=alphas[i], label=label)

    if x_label is not None:
        ax.set_xlabel(x_label, fontsize=labels_fontsize)
    if y_label is not None:
        ax.set_ylabel(y_label, fontsize=labels_fontsize)
    if x_scale is not None:
        ax.set_xscale(x_scale)
    if y_scale is not None:
        ax.set_yscale(y_scale)
    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    if grid_kwargs is not None:
        ax.grid(**grid_kwargs)
    if is_use_scalar_formatter:
        for axis in [ax.xaxis, ax.yaxis]:
            axis.set_major_formatter(plticker.ScalarFormatter())
            axis.set_minor_formatter(plticker.ScalarFormatter())
    if ticklabel_kwargs is not None:
        ax.ticklabel_format(**ticklabel_kwargs)
    if x_tick_major_spacing is not None:
        loc = plticker.MultipleLocator(base=x_tick_major_spacing)
        ax.xaxis.set_major_locator(loc)
    if y_tick_major_spacing is not None:
        loc = plticker.MultipleLocator(base=y_tick_major_spacing)
        ax.yaxis.set_major_locator(loc)
    if x_tick_minor_spacing is not None:
        loc = plticker.MultipleLocator(base=x_tick_minor_spacing)
        ax.xaxis.set_minor_locator(loc)
    if y_tick_minor_spacing is not None:
        loc = plticker.MultipleLocator(base=y_tick_minor_spacing)
        ax.yaxis.set_minor_locator(loc)

    if legend_kwargs is not None:
        ax.legend(title=legend_title, **legend_kwargs)

    if is_created_new_axis:
        plt.tight_layout()

    if is_show:
        plt.show()

    return ax
