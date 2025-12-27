from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.metrics import calculate_cohesion_sim, calculate_polarization_sim, \
    calculate_elongation_sim
from utils.utils import load_simulation_results, load_simulation_results_matlab


def calculate_all_metrics(data):
    pos_s = data['pos_s']
    vel_s = data['vel_s']

    cohesion_all = np.ravel(calculate_cohesion_sim(pos_s))
    polarization_all = np.ravel(calculate_polarization_sim(vel_s))
    elongation_all = np.ravel(calculate_elongation_sim(pos_s, vel_s))

    return {
        'cohesion': np.array(cohesion_all),
        'polarization': np.array(polarization_all),
        'elongation': np.array(elongation_all)
    }


def compute_pdf(data, bins):
    data_flat = data.flatten()
    data_flat = data_flat[~np.isnan(data_flat)]

    counts, edges = np.histogram(data_flat, bins=bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2

    return centers, counts


def plot_metric_pdf(data, data_matlab=None, bins=None, xlabel='', color='#2365C4',
                    filename=None, bin_width=None):
    data_clean = data.flatten()
    data_clean = data_clean[~np.isnan(data_clean)]

    if bins is None:
        if bin_width is None:
            bin_width = 0.2
        data_min = np.min(data_clean)
        data_max = np.max(data_clean)
        bins = np.arange(data_min, data_max + bin_width, bin_width)

    centers, pdf = compute_pdf(data, bins)

    xlim = [np.min(data_clean) - 0.3, np.max(data_clean) + 0.3]
    max_pdf = np.max(pdf)
    ylim = [0, max_pdf * 1.15]

    if data_matlab is not None:
        data_matlab_clean = data_matlab.flatten()
        data_matlab_clean = data_matlab_clean[~np.isnan(data_matlab_clean)]

        centers_matlab, pdf_matlab = compute_pdf(data_matlab, bins)

        xlim_matlab = [np.min(data_matlab_clean) - 0.3, np.max(data_matlab_clean) + 0.3]
        max_pdf_matlab = np.max(pdf_matlab)
        ylim_matlab = [0, max_pdf_matlab * 1.15]

        # Update limits to include both datasets
        xlim[0] = min(xlim[0], xlim_matlab[0])
        xlim[1] = max(xlim[1], xlim_matlab[1])
        ylim[1] = max(ylim[1], ylim_matlab[1])

    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.plot(centers, pdf, '-', color=color, linewidth=2.5, label='Python')

    # Plot MATLAB data with dashed line
    if data_matlab is not None:
        ax.plot(centers_matlab, pdf_matlab, '--', color='#E84A5F',
                linewidth=2.5, label='MATLAB')

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel('PDF', fontsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Add legend if MATLAB data is present
    if data_matlab is not None:
        ax.legend(fontsize=12)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()

    return fig, ax

def load_data(filename):
    if not Path(filename).exists():
        print(f"Error: {filename} not found")
        exit(1)

    if filename.endswith('.npz'):
        results = load_simulation_results(filename)
    elif filename.endswith('.mat'):
        results = load_simulation_results_matlab(filename)
    else:
        print(f"Error: Unknown file format for {filename}")
        exit(1)

    return results


if __name__ == "__main__":
    filename = "../data/vivek_14_300_370.npz"
    filename_matlab = "../data/fixed_hm_n_14.mat"

    data = load_data(filename)
    metrics = calculate_all_metrics(data)

    data_matlab = load_data(filename_matlab)
    metrics_matlab = calculate_all_metrics(data_matlab)

    plot_metric_pdf(
        data=metrics['polarization'],
        data_matlab=metrics_matlab['polarization'],
        bins=np.linspace(0, 1, 26),
        xlabel='Group Polarization',
        color='#2365C4',
        filename='polarization_pdf.png'
    )

    plot_metric_pdf(
        data=metrics['cohesion'],
        data_matlab=metrics_matlab['cohesion'],
        bin_width=0.2,
        xlabel='Cohesion (m)',
        color='#964B00',
        filename='cohesion_pdf.png'
    )

    plot_metric_pdf(
        data=metrics['elongation'],
        data_matlab=metrics_matlab['elongation'],
        bin_width=0.2,
        xlabel='Elongation',
        color='#A020F0',
        filename='elongation_pdf.png'
    )
