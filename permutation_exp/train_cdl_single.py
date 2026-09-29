# %%
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
import sys

PERMUTATION_EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = PERMUTATION_EXP_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from src.networks.cdl_fista import CDLFISTA2D
from src.networks.unet import UNet
from src.operators.mr_ops import MRIOperator
from src.data_gen.lowfield_dataset import LowfieldMRIData

from dataclasses import dataclass

@dataclass
class TrainConfig:
    """Configuration for CDL-FISTA building & training."""
    
    # Training hyperparameters
    epochs: int = 48
    batch_size: int = 1
    lr_unet: float = 1e-4
    lr_scalar: float = 1e-1
    weight_decay_unet: float = 1e-5
    
    # Data - use small subsets for testing
    n_train_images: int = 4875
    n_val_images: int = 1393

    # Model architecture
    K: int = 64                             # Number of dict filters
    dict_filter_size: tuple = (11, 11)
    n_iterations: int = 64
    lambda_mode: str = "lambda_cnn_map"
    version: str = "version3"
    reg_parameter_bounds: tuple = (0.0, 10.0)
    pretrained_dict_path: str = "path/to/dicionary_filter_set/tensor.pt"
    freeze_dictionary: bool = True
    
    # UNet architecture
    n_ch_in: int = 2
    n_ch_out: int = 1
    n_unet_filters: int = 32
    n_enc_stages: int = 2
    n_convs_per_stage: int = 2
    kernel_size: tuple = (3, 3)
    pooling_kernel_size: tuple = (2, 2)
    res_connection: bool = False
    

def setup_data(config: TrainConfig):
    train_data = LowfieldMRIData(
        path = "data/training/",
        n_images = config.n_train_images,
        im_shape = (320, 320),
        cutoff_range_y = (40, 120),
        cutoff_range_x = (40, 120),
        dims = (-2, -1),
        noise_vars = (0.15, 0.25),
        train = True
    )
    
    val_data = LowfieldMRIData(
        path = "data/validation/",
        n_images = config.n_val_images,
        im_shape = (320, 320),
        cutoff_range_y = (80, 80),
        cutoff_range_x = (80, 80),
        dims = (-2, -1),
        noise_vars = 0.2,
        train = False,
        seed = 42
    )
    
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False)
    
    return train_loader, val_loader


def build_model(config: TrainConfig, device):
    """ Builds the CDL-FISTA model with UNet for lambda map prediction. """

    # Construct the UNet, which is a deep CNN block used to predict lambda reg. parameter maps from inital input images!
    unet_1 = UNet(
        dim = 2,                                # 2D convolutions (vs 1D or 3D)
        n_ch_in = config.n_ch_in,               # Input: 2 channels (real + imag parts of complex MRI)
        n_ch_out = config.n_ch_out,             # Output: 32 lambda maps, later duplicated to match n_dict_filters
        n_filters = config.n_unet_filters,
        n_enc_stages = config.n_enc_stages,
        n_convs_per_stage = config.n_convs_per_stage,
        kernel_size = config.kernel_size,
        pooling_kernel_size = config.pooling_kernel_size,
        res_connection = config.res_connection,
    ).to(device)

    # Model setup
    model = CDLFISTA2D(
        mr_operator = MRIOperator(dims=(-2, -1)),
        n_dict_filters = config.K,
        dict_filter_size = config.dict_filter_size,
        n_iterations = config.n_iterations,
        lambda_mode = config.lambda_mode,
        version = config.version,
        cnn_block_reg_parameter = unet_1,
        reg_parameter_bounds = config.reg_parameter_bounds,
    ).to(device)

    return model


def load_pretrained_dictionary(model, config: TrainConfig):
    """Load and validate pretrained dictionary filters."""
    
    # Standard fixed dictionary loading
    pretrained_dict = torch.load(config.pretrained_dict_path, map_location='cpu')
    expected_shape = (config.K//2, 1, config.dict_filter_size[0], config.dict_filter_size[1])
        
    if pretrained_dict.shape != expected_shape:
        raise ValueError(f"Dictionary shape mismatch: got {pretrained_dict.shape}, expected {expected_shape}")
        
    model.d_filter.data.copy_(pretrained_dict)
    print(f"* Loaded pretrained dictionary from: {config.pretrained_dict_path}")


def setup_optimizer(model, config: TrainConfig):
    """Configure optimizer with parameter groups and different learning rates."""
    param_groups = [
        {
            'params': model.cnn_block_reg_parameter.parameters(),
            'lr': config.lr_unet,
            'weight_decay': config.weight_decay_unet,
            'name': 'unet'
        },
        {
            'params': [model.lambda_reg_raw, model.lambda_reg_high_pass_raw],
            'lr': config.lr_scalar,
            'weight_decay': 0.0,
            'name': 'scalars'
        }
    ]
    return torch.optim.Adam(param_groups)



def train_model(config: TrainConfig):
    """Main training loop."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = build_model(config, device)

    # Verify model is on GPU
    print(f"* Model on device: {next(model.parameters()).device}")
    print(f"* d_filter on device: {model.d_filter.device}")
    print(f"* UNet first layer on device: {next(model.cnn_block_reg_parameter.parameters()).device}")
    
    load_pretrained_dictionary(model, config)
 
    if config.freeze_dictionary:
        model.d_filter.requires_grad = False
        print("* Dictionary filters frozen (no gradient updates)")
    
    optimizer = setup_optimizer(model, config)
    train_loader, val_loader = setup_data(config)
    torch.cuda.empty_cache()

    # Training loop
    for epoch in range(config.epochs):
        model.train()

        for batch_idx, (kdata, mask, x0, xtrue) in enumerate(train_loader):
            kdata, mask, x0, xtrue = [x.to(device) for x in (kdata, mask, x0, xtrue)]

            optimizer.zero_grad()
            xrecon = model(x0, kdata, mask)      # Forward pass: internally calls CDLFISTA2D.forward() -> get_lambda_cnn() -> UNet estimates 32 maps -> duplicated to 64

            if epoch == 0 and batch_idx == 0 and torch.cuda.is_available():
                print(f"* Allocated GPU memory post first forward: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                print(f"* Reserved GPU memory (doesn't change): {torch.cuda.memory_reserved() / 1e9:.2f} GB")
                torch.cuda.reset_peak_memory_stats()

            loss = torch.nn.functional.mse_loss(torch.view_as_real(xrecon), torch.view_as_real(xtrue))
            loss.backward()

            optimizer.step()
            
        model.eval()
        train_eval_loss = 0.0
        with torch.no_grad():
            for batch_idx, (kdata, mask, x0, xtrue) in enumerate(train_loader):
                kdata, mask, x0, xtrue = [x.to(device) for x in (kdata, mask, x0, xtrue)]
                
                xrecon = model(x0, kdata, mask)
                train_eval_loss += torch.nn.functional.mse_loss(torch.view_as_real(xrecon), torch.view_as_real(xtrue)).item()
        
        train_eval_loss /= len(train_loader)

        # on validation set
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (kdata, mask, x0, xtrue) in enumerate(val_loader):
                kdata, mask, x0, xtrue = [x.to(device) for x in (kdata, mask, x0, xtrue)]
                
                xrecon = model(x0, kdata, mask)
                val_loss += torch.nn.functional.mse_loss(torch.view_as_real(xrecon), torch.view_as_real(xtrue)).item()

        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch}/{config.epochs} - Train: {train_eval_loss:.6f}, Val: {val_loss:.6f}")

    # final save
    torch.save(model.state_dict(), "test.pt")
    print("Training finished. Model saved to ...")


if __name__ == "__main__":
    train_model(TrainConfig())