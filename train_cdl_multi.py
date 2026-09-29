# %%
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import wandb
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple
import random

from src.networks.cdl_fista import CDLFISTA2D
from src.networks.unet import UNet
from src.operators.mr_ops import MRIOperator
from src.data_gen.lowfield_dataset import LowfieldMRIData


@dataclass
class DictConfig:
    """Configuration and loaded tensor for a single dictionary."""
    name: str
    K: int
    filter_size: Tuple[int, int]
    tensor: torch.Tensor
    
    def __repr__(self):
        return self.name


@dataclass
class TrainConfig:
    """Configuration for multi-dictionary CDL-FISTA training."""
    
    # Training hyperparameters
    epochs: int = 128
    batch_size: int = 1
    lr_unet: float = 1e-4
    lr_scalar: float = 1e-1
    weight_decay_unet: float = 1e-5
    
    # Data
    n_train_images: int = 4875    # 4875 is the full training set size, reduce for quick tests
    n_val_images: int = 1393      # 1393 is the full validation set size, reduce for quick tests

    # Multi-dictionary setup
    train_dicts_path: str = "data/training/dictionary_filters"
    val_dicts_path: str = "data/validation/dictionary_filters"
    
    # Model architecture
    n_iterations: int = 64     # 64 unrolled fista iters, reduce for quick test
    lambda_mode: str = "lambda_cnn_map"
    version: str = "version3"
    reg_parameter_bounds: tuple = (0.0, 10.0)
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
    
    # W&B logging
    use_wandb: bool = True
    wandb_entity: str = "your_wandb_entity"  # <-- replace with your W&B entity
    wandb_project: str = "your_wandb_project"  # <-- replace with your W&B project name

    # Checkpointing
    checkpoint_every: int = 1  # Save every N epochs
    checkpoint_dir: str = "output/training_checkpoints/epochs"


def build_model(config: TrainConfig, dict_config: DictConfig, device):
    unet = UNet(
        dim=2,
        n_ch_in=config.n_ch_in,
        n_ch_out=config.n_ch_out,
        n_filters=config.n_unet_filters,
        n_enc_stages=config.n_enc_stages,
        n_convs_per_stage=config.n_convs_per_stage,
        kernel_size=config.kernel_size,
        pooling_kernel_size=config.pooling_kernel_size,
        res_connection=config.res_connection,
    ).to(device)
    
    # Create ONE model instance
    model = CDLFISTA2D(
        mr_operator=MRIOperator(dims=(-2, -1)),
        n_dict_filters=dict_config.K,
        dict_filter_size=dict_config.filter_size,
        n_iterations=config.n_iterations,
        lambda_mode=config.lambda_mode,
        version=config.version,
        cnn_block_reg_parameter=unet,
        reg_parameter_bounds=config.reg_parameter_bounds,
    ).to(device)
    
    # Freeze dictionary parameter
    if config.freeze_dictionary:
        model.d_filter.requires_grad = False
    
    return model


def setup_data(config: TrainConfig):
    train_data = LowfieldMRIData(
        path="data/training/",
        n_images=config.n_train_images,
        im_shape=(320, 320),
        cutoff_range_y=(40, 120),
        cutoff_range_x=(40, 120),
        dims=(-2, -1),
        noise_vars=(0.15, 0.3),
        train=True
    )
    
    val_data = LowfieldMRIData(
        path="data/validation/",
        n_images=config.n_val_images,
        im_shape=(320, 320),
        cutoff_range_y=(80, 80),
        cutoff_range_x=(80, 80),
        dims=(-2, -1),
        noise_vars=0.2,
        train=False,
        seed=42
    )
    
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False)
    
    return train_loader, val_loader


def load_dictionaries(base_path: str) -> List[DictConfig]:
    """Load dictionary tensors and their parsed config directly from filenames."""
    base_path = Path(base_path)
    if not base_path.exists():
        raise ValueError(f"Dictionary base path does not exist: {base_path}")
    
    dicts = []
    for dict_file in sorted(base_path.glob("*.pt")):
        try:
            name = dict_file.stem
            parts = name.split('_')
            
            n_filters = int(parts[0])
            ky, kx = map(int, parts[1].split('x'))
            K = 2 * n_filters
            
            # Load to CPU first to prevent device OOM, move to GPU only when needed
            tensor = torch.load(dict_file, map_location='cpu')
            expected_shape = (K // 2, 1, ky, kx)
            
            if tensor.shape != expected_shape:
                print(f"Warning: Match fail for {name}, got {tensor.shape}, expected {expected_shape}")
                continue
                
            dicts.append(DictConfig(name=name, K=K, filter_size=(ky, kx), tensor=tensor))
        except Exception as e:
            print(f"Warning: Could not parse dictionary file {dict_file.name}: {e}")
    
    if not dicts:
        raise ValueError(f"No valid dictionary files found in {base_path}")
    
    print(f"* Loaded {len(dicts)} dictionary configurations/tensors:")
    for d in dicts[:5]:  # Show first 5
        print(f"  - {d.name}")
    if len(dicts) > 5:
        print(f"  ... and {len(dicts) - 5} more")
    
    return dicts


def update_dictionary(model, dict_config: DictConfig, dictionary_tensor, device):
    """Dynamically update model attributes to match the dictionary architecture."""
    # Update model attributes
    model.n_dict_filters = dict_config.K
    model.dict_filter_size = dict_config.filter_size
    
    # Update padding based on new filter size
    model.padding = tuple(k // 2 for k in dict_config.filter_size)
    model.pad = tuple(p for p in model.padding for _ in range(2))  # duplicate each
    model.crop = tuple(-p for p in model.padding for _ in range(2))  # negate duplicates
    
    # Resize d_filter parameter if needed
    expected_shape = (dict_config.K // 2, 1, dict_config.filter_size[0], dict_config.filter_size[1])
    if model.d_filter.shape != expected_shape:
        # Create new parameter with correct shape
        model.d_filter = torch.nn.Parameter(
            torch.zeros(expected_shape, device=device),
            requires_grad=model.d_filter.requires_grad
        )
    
    # Copy dictionary data IN PLACE operation
    model.d_filter.data.copy_(dictionary_tensor)


def setup_optimizer(model, config: TrainConfig):
    """Configure optimizer - only UNet and scalar parameters are trainable."""
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


def setup_wandb(config: TrainConfig, n_dicts: int = None):
    """Initialize W&B run with metadata."""
    if not config.use_wandb:
        return None
    
    wandb_config = {
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "lr_unet": config.lr_unet,
        "lr_scalar": config.lr_scalar,
        "weight_decay_unet": config.weight_decay_unet,
        "optimizer": "Adam",
        "n_iterations": config.n_iterations,
        "lambda_mode": config.lambda_mode,
        "version": config.version,
        "freeze_dictionary": config.freeze_dictionary,
        "reg_parameter_bounds": str(config.reg_parameter_bounds),
        "n_unet_filters": config.n_unet_filters,
        "n_enc_stages": config.n_enc_stages,
        "n_convs_per_stage": config.n_convs_per_stage,
        "unet_kernel_size": str(config.kernel_size),
        "res_connection": config.res_connection,
    }

    run = wandb.init(
        entity=config.wandb_entity,
        project=config.wandb_project,
        config=wandb_config
    )
    return run


def save_checkpoint(model, epoch, avg_val_loss, config: TrainConfig):
    """Save model checkpoint."""
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    checkpoint_path = checkpoint_dir / f"epoch{epoch:03d}_loss{avg_val_loss:.4f}.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"  Saved: {checkpoint_path.name}")


#################
# Main Training #
#################
def train_multi_dict(config: TrainConfig):
    """Main training loop with single model and dynamic dictionary swapping."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load training dictionaries - tensors are loaded on CPU
    train_dicts = load_dictionaries(config.train_dicts_path)
    val_dicts = load_dictionaries(config.val_dicts_path)

    model = build_model(config, train_dicts[0], device)
    
    train_loader, val_loader = setup_data(config)
    optimizer = setup_optimizer(model, config)
    run = setup_wandb(config, n_dicts=len(train_dicts))
    
    print(f"* Starting multi-dictionary training with {len(train_dicts)} train dicts, {len(val_dicts)} val dicts")
    
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(config.epochs):
        model.train()
        
        for batch_idx, (kdata, mask, x0, xtrue) in enumerate(train_loader):
            kdata, mask, x0, xtrue = [x.to(device) for x in (kdata, mask, x0, xtrue)]
            
            # Sample a random dictionary from training set and dynamically update model architecture
            dict_obj = random.choice(train_dicts)
            update_dictionary(model, dict_obj, dict_obj.tensor.to(device), device)
            
            optimizer.zero_grad()
            
            xrecon = model(x0, kdata, mask)

            loss = torch.nn.functional.mse_loss(torch.view_as_real(xrecon), torch.view_as_real(xtrue))
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f"  Epoch {epoch}, Batch {batch_idx}, Dict: {dict_obj.name}, Loss: {loss.item():.6f}")
        
        # VALIDATION PHASE
        model.eval()
        all_val_losses = []
        
        with torch.no_grad():
            for dict_idx, dict_obj in enumerate(val_dicts):
                val_loss = 0.0
                val_count = 0
                # Load validation dictionary and update model architecture
                update_dictionary(model, dict_obj, dict_obj.tensor.to(device), device)
                
                for batch_idx, (kdata, mask, x0, xtrue) in enumerate(val_loader):
                    kdata, mask, x0, xtrue = [x.to(device) for x in (kdata, mask, x0, xtrue)]
                    
                    xrecon = model(x0, kdata, mask)
                    loss = torch.nn.functional.mse_loss(torch.view_as_real(xrecon), torch.view_as_real(xtrue)).item()
                    val_loss += loss
                    val_count += 1
                    
                val_loss /= val_count               # for entire val set for current (single) dict
                all_val_losses.append(val_loss)     # append for averaging over val dicts later
                
                print(f"Epoch {epoch}/{config.epochs} - Val ({dict_obj.name}): {val_loss:.6f}")

        # Average validation loss across val dictionaries
        avg_val_loss = sum(all_val_losses) / len(all_val_losses)
        print(f"Epoch {epoch}/{config.epochs} - Avg Val Loss: {avg_val_loss:.6f}")
        
        # wandb bundled logging over epoch
        if run:
            per_dict_logs = {f"val_loss/{d.name}": l for d, l in zip(val_dicts, all_val_losses)}
            wandb.log({
                **per_dict_logs,
                "val_loss/average": avg_val_loss,
                "beta_effective": model.lambda_reg_high_pass.item(),
                "epoch": epoch,
            }, step=epoch)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = Path(config.checkpoint_dir).parent / "best_model.pt"  # Save in output/checkpoints/16_32_64/
            best_path.parent.mkdir(exist_ok=True, parents=True)  # Ensure parent exists
            torch.save(model.state_dict(), best_path)
            print(f"  New best model saved: {best_path.name}")
        
        # Save regular checkpoint every epoch
        if (epoch + 1) % config.checkpoint_every == 0:
            save_checkpoint(model, epoch, avg_val_loss, config)
    
    # Save final model
    final_path = Path(config.checkpoint_dir).parent / "final_model.pt"  # Save in output/checkpoints/16_32_64/
    final_path.parent.mkdir(exist_ok=True, parents=True)
    # torch.save(model.state_dict(), final_path)
    print(f"Training finished. Final model saved to {final_path}")
    
    if run:
        run.finish()

if __name__ == "__main__":
    train_multi_dict(TrainConfig())
