"""Compare lambda maps by variance across multiple models."""
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pathlib import Path
import sys

PERMUTATION_EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = PERMUTATION_EXP_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data_gen.lowfield_dataset import LowfieldMRIData
from src.networks.cdl_fista import CDLFISTA2D
from src.networks.unet import UNet
from src.operators.mr_ops import MRIOperator

# Set font to serif (Times-like) with LaTeX-style rendering
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Times']
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

# Configuration
DATA_PATH = "data/testing/"
IMAGE_IDX = 63
N_CHANNELS = 10  # Top channels by variance to display
MODEL_NAMES = ["v1_32_11x11", "v2_32_11x11", "v3_32_11x11"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Model configurations (using the current versioning: v1, v2, v3)
model_configs = {
    "v1_32_11x11": {
        "checkpoint": "models/single_dict/v1_32_11x11.pt",
        "n_dict_filters": 64,
        "dict_filter_size": (11, 11),
        "n_iterations": 64,
        "lambda_mode": "lambda_cnn_map",
        "version": "version1",

        "n_ch_in": 2,
        "n_ch_out": 32,     # K/2
        "n_unet_filters": 128,
        "n_enc_stages": 3,
        "n_convs_per_stage": 2,
        "kernel_size": (3, 3),
        "pooling_kernel_size": (2, 2),
        "res_connection": False,
    },
    "v2_32_11x11": {
        "checkpoint": "models/single_dict/v2_32_11x11.pt",
        "n_dict_filters": 64,
        "dict_filter_size": (11, 11),
        "n_iterations": 64,
        "lambda_mode": "lambda_cnn_map",
        "version": "version2",

        "n_ch_in": 64,
        "n_ch_out": 32,
        "n_unet_filters": 64,
        "n_enc_stages": 3,
        "n_convs_per_stage": 2,
        "kernel_size": (3, 3),
        "pooling_kernel_size": (2, 2),
        "res_connection": False,
    },
    "v3_32_11x11": {
        "checkpoint": "models/single_dict/v3_32_11x11.pt",
        "n_dict_filters": 64,
        "dict_filter_size": (11, 11),
        "n_iterations": 64,
        "lambda_mode": "lambda_cnn_map",
        "version": "version3",

        "n_ch_in": 2,
        "n_ch_out": 1,
        "n_unet_filters": 32,
        "n_enc_stages": 2,
        "n_convs_per_stage": 2,
        "kernel_size": (3, 3),
        "pooling_kernel_size": (2, 2),
        "res_connection": False,
    },
}


def build_model(config, device) -> CDLFISTA2D:
    """Build the CDL-FISTA model from config and load weights from checkpoint."""
    cnn = UNet(
        dim=2,
        n_ch_in=config["n_ch_in"],
        n_ch_out=config["n_ch_out"],
        n_filters=config["n_unet_filters"],
        n_enc_stages=config["n_enc_stages"],
        n_convs_per_stage=config["n_convs_per_stage"],
        kernel_size=config["kernel_size"],
        pooling_kernel_size=config["pooling_kernel_size"],
        res_connection=config["res_connection"],
    ).to(device)

    model = CDLFISTA2D(
        mr_operator=MRIOperator(dims=(-2, -1)),
        n_dict_filters=config["n_dict_filters"],
        dict_filter_size=config["dict_filter_size"],
        n_iterations=config["n_iterations"],
        lambda_mode=config["lambda_mode"],
        version=config["version"],
        cnn_block_reg_parameter=cnn,
        reg_parameter_bounds=(0.0, 10.0),
        phase="inference",
    ).to(device)

    model.load_state_dict(torch.load(config["checkpoint"], map_location=device, weights_only=True))
    model.eval()
    return model


def load_and_infer(model_name, x0, kdata, mask):
    """Load model, run inference, return lambda map and filters."""
    print(f"  Processing {model_name}...")

    config = model_configs[model_name]

    model = build_model(config, device)

    with torch.no_grad():
        x_est, sparse_code, lambda_map = model(x0, kdata, mask)

    # Extract first 32 channels, rotate 180 degrees, and move to CPU
    lmap = torch.rot90(lambda_map[0, :32], k=2, dims=(-2, -1)).cpu()
    filters = torch.rot90(model.d_filter.cpu().squeeze(1), k=2, dims=(-2, -1))

    return lmap, filters


def compare_maps_by_variance(models_data, n=6):
    """
    Compare lambda maps across models, sorted by variance.

    Args:
        models_data: dict of {name: (lambda_map, filters)}
        n: number of top channels by variance to show
    """
    # Map model names to version labels
    name_to_label = {
        'v1_32_11x11': 'Version 1',
        'v2_32_11x11': 'Version 2',
        'v3_32_11x11': 'Version 3'
    }

    names = list(models_data.keys())
    # Reduce figure width per column so columns sit close together; lower height to tighten rows
    fig, axes = plt.subplots(len(names), n, figsize=(1.72 * n, 1.7 * len(names)))

    # Handle single model case
    if len(names) == 1:
        axes = axes.reshape(1, -1)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    for row, name in enumerate(names):
        lmap, filters = models_data[name]

        # Sort channels by variance of lambda maps
        variances = torch.var(lmap, dim=(1, 2))
        variance_order = torch.argsort(variances, descending=True)

        top_idx = variance_order[:n]

        print(f"\n{name}:")
        print(f"  Top {n} indices by variance: {top_idx.tolist()}")
        print(f"  Lambda range: [{lmap.min():.4f}, {lmap.max():.4f}]")

        # Plot top n channels
        for i in range(n):
            idx = top_idx[i].item()

            # Show lambda map (no title anymore)
            axes[row, i].imshow(lmap[idx], cmap='inferno', clim=[0, 10])

            # Add channel label as text overlay at top right
            axes[row, i].text(0.95, 0.95, f'filt {idx}', transform=axes[row, i].transAxes,
                    fontsize=14, color='black',
                    ha='right', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                        edgecolor='none', alpha=0.7))

            # Inset: dictionary filter
            axins = inset_axes(axes[row, i], width="35%", height="35%",
                               loc='upper left', borderpad=0)
            axins.imshow(filters[idx], cmap='viridis')
            axins.set_xticks([])
            axins.set_yticks([])

        # Label each row with version name
        version_label = name_to_label.get(name, name.upper())
        axes[row, 0].set_ylabel(version_label, fontsize=14, fontweight='bold',
                    rotation=90, labelpad=6, va='center')

    # Put title clearly above the plots and give a little top margin
    # plt.suptitle(f'Top {n} Lambda Maps by Variance',
    #              fontsize=18, fontweight='bold', y=0.995)
    # Tight layout with tiny white lines between maps
    plt.tight_layout(pad=0.06, rect=[0, 0, 1, 1])
    plt.subplots_adjust(wspace=0.01, hspace=0.01, left=0.04, right=0.995, bottom=0.03)
    plt.savefig(f'lambda_maps_{N_CHANNELS}.pdf', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    # Load dataset and get single sample
    print(f"Loading image {IMAGE_IDX} from {DATA_PATH}...")
    dataset = LowfieldMRIData(
        path=DATA_PATH, n_images=696, im_shape=(320, 320),
        cutoff_range_y=(80, 80), cutoff_range_x=(80, 80),
        dims=(-2, -1), noise_vars=0.2, train=False, seed=42
    )

    kdata, mask, x0, x_true = [t.unsqueeze(0).to(device) for t in dataset[IMAGE_IDX]]

    # Run inference for all models
    print("\nRunning inference on models...")
    models_data = {}
    for name in MODEL_NAMES:
        lmap, filters = load_and_infer(name, x0, kdata, mask)
        models_data[name] = (lmap, filters)

    # Compare lambda maps
    print("\nGenerating comparison plot...")
    compare_maps_by_variance(models_data, n=N_CHANNELS)


if __name__ == "__main__":
    main()