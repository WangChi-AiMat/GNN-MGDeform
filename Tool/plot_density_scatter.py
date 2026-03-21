import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import seaborn as sns
from scipy import stats
import os
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

def plot_density_scatter(y_true, y_pred, save_dir):

    plot_params = {
        "fig_size": (8, 8),
        "dpi": 500,
        "aspect_ratio": 1,

        "font_family": "Arial",
        "font_size": 8,
        "xlim": (-0.05, 1),
        "ylim": (-0.05, 1),

        "axis_label_enable": False,
        "xlabel": r'True Value',
        "ylabel": r'Predicted Value',
        "xlabel_fontsize": 14,
        "ylabel_fontsize": 14,

        "axis_line_width": 1.7,
        "aspect_equal": True,

        "tick_enable": True,
        "tick_label_enable": True,
        "tick_pad": 5,
        "major_tick_length": 10,
        "major_tick_width": 1.7,
        "minor_tick_enable": False,
        "minor_tick_length": 5,
        "minor_tick_width": 1,
        "tick_labelsize": 25,
        "tick_direction": "out",

        "major_tick_interval": 0.2,
        "minor_tick_interval": 0.05,

        "cmap": 'RdBu',
        "alpha": 0.8,
        "s": 20,
        "edgecolors": 'none',

        "colorbar_enable": True,
        "colorbar_width": "4%",
        "colorbar_padding": 0.3,
        "colorbar_label": "Density",
        "colorbar_tick_size": 25,
        "colorbar_tick_length": 6,
        "colorbar_tick_width": 1.5,
        "colorbar_tick_enable": True,

        "file_name": "density_scatter.png",

        "diagonal_line_width": 1.5,
    }

    os.makedirs(save_dir, exist_ok=True)
    file_name = plot_params.get("file_name", "density_scatter.png")

    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)

    try:
        kernel = stats.gaussian_kde([y_true, y_pred])
        density = kernel(np.vstack([y_true, y_pred]))
        print("KDE")
    except Exception as e:
        raise RuntimeError(f"{str(e)}")

    times_path = r"C:\Windows\Fonts\Arial.TTF"
    prop = fm.FontProperties(fname=times_path)

    fig_size = plot_params.get("fig_size", (6, 6))
    aspect_ratio = plot_params.get("aspect_ratio", 1.0)
    dpi = plot_params.get("dpi", 500)
    fig_width = fig_size[0] * aspect_ratio
    fig_height = fig_size[1]
    plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    ax = plt.gca()

    sc = ax.scatter(
        y_true, y_pred,
        c=density,
        cmap=plot_params.get("cmap", 'Reds'),
        alpha=plot_params.get("alpha", 0.9),
        s=plot_params.get("s", 30),
        edgecolors=plot_params.get("edgecolors", 'none')
    )

    ax.plot([-0.05, 1], [-0.05, 1], 'k--',
            linewidth=plot_params.get("diagonal_line_width", 1.2),
            alpha=0.8)

    if plot_params.get("axis_label_enable", True):
        ax.set_xlabel(plot_params.get("xlabel", 'True Value'),
                      fontsize=plot_params.get("xlabel_fontsize", 14),
                      fontproperties=prop,
                      fontweight='bold')
        ax.set_ylabel(plot_params.get("ylabel", 'Predicted Value'),
                      fontsize=plot_params.get("ylabel_fontsize", 14),
                      fontproperties=prop,
                      fontweight='bold')

    if plot_params.get("xlim", None):
        ax.set_xlim(plot_params["xlim"])
    if plot_params.get("ylim", None):
        ax.set_ylim(plot_params["ylim"])

    if plot_params.get("aspect_equal", True):
        ax.set_aspect('equal', adjustable='box')

    for spine in ax.spines.values():
        spine.set_linewidth(plot_params.get("axis_line_width", 1.5))

    if plot_params.get("tick_enable", True):
        major_interval = plot_params.get("major_tick_interval", 0.2)
        minor_interval = plot_params.get("minor_tick_interval", 0.05)

        ax.xaxis.set_major_locator(MultipleLocator(major_interval))
        ax.yaxis.set_major_locator(MultipleLocator(major_interval))

        if plot_params.get("minor_tick_enable", False):
            ax.minorticks_on()
            ax.xaxis.set_minor_locator(MultipleLocator(minor_interval))
            ax.yaxis.set_minor_locator(MultipleLocator(minor_interval))
            ax.xaxis.set_minor_formatter(FormatStrFormatter('%0.2f'))
            ax.yaxis.set_minor_formatter(FormatStrFormatter('%0.2f'))

        ax.tick_params(
            axis='both',
            which='major',
            direction=plot_params.get("tick_direction", "in"),
            length=plot_params.get("major_tick_length", 6),
            width=plot_params.get("major_tick_width", 1.2),
            pad=plot_params.get("tick_pad", 5)
        )
        ax.tick_params(
            axis='both',
            which='minor',
            direction=plot_params.get("tick_direction", "in"),
            length=plot_params.get("minor_tick_length", 3),
            width=plot_params.get("minor_tick_width", 0.8)
        )

        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(prop)
            label.set_fontsize(plot_params.get("tick_labelsize", 12))

    if plot_params.get("colorbar_enable", True):
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(
            "right",
            size=plot_params.get("colorbar_width", "3%"),
            pad=plot_params.get("colorbar_padding", 0.1)
        )
        cbar = plt.colorbar(sc, cax=cax)
        cbar.set_label(plot_params.get("colorbar_label", "Density"),
                       fontsize=12, fontweight='bold',
                       fontproperties=prop)

        if plot_params.get("colorbar_tick_enable", True):
            for tick in cbar.ax.get_yticklabels():
                tick.set_fontproperties(prop)
                tick.set_fontsize(plot_params.get("colorbar_tick_size", 10))
            cbar.ax.tick_params(
                direction='in',
                length=plot_params.get("colorbar_tick_length", 4),
                width=plot_params.get("colorbar_tick_width", 1.0)
            )

        cbar.outline.set_linewidth(1.2)

    plt.tight_layout()
    if not file_name.lower().endswith('.png'):
        file_name += '.png'
    save_path = os.path.join(save_dir, file_name)
    plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
    plt.close()
    print(f"{save_path}")
