import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path

from ..operators.mr_ops import MRIOperator
from ..utils.noise_funcs import add_gaussian_noise


class LowfieldMRIData(Dataset):
    """
    Dataset class for low-field MRI data generation and augmentation.
    
    This dataset loads ground truth MRI images, converts them to k-space, and generates
    training/testing samples by applying masks and noise to simulate low-field MRI acquisition.
    
    """
    DEFAULT_FACTOR = 10

    def __init__(
        self,
        path,
        n_images=10,
        im_shape=(320, 320),
        cutoff_range_y=None,
        cutoff_range_x=None,
        dims=(-2, -1),
        noise_vars=(0.15, 0.3),
        train=True,
        seed=None,
        factor=10,
    ):

        fname = Path(path) / "xtrue.pt"

        self.im_shape = im_shape
        self.xtrue = torch.load(fname)[:n_images, ...]
        scale = torch.amax(self.xtrue.abs(), dim=(-2, -1), keepdim=True)
        self.xtrue /= scale
        self.xtrue *= factor

        self.kdata_true = torch.fft.fftn(self.xtrue, dim=(-2, -1), norm="ortho")
        self.dims = dims
        self.op = MRIOperator(dims)
        self.train = train
        self.noise_vars = noise_vars
        self.cutoff_range_x = cutoff_range_x
        self.cutoff_range_y = cutoff_range_y

        if not train and seed is not None:
            self.seed = 2025
        else:
            self.seed = None

    def __len__(self):
        return self.xtrue.shape[0]

    def _generate_mask(self, cy, cx, fftshift=True):
        """ Internal helper: build central rectangular pass-band mask. """
        mask = torch.zeros(self.im_shape, dtype=torch.int8)
        if cy != 0 and cx != 0:
            mask[cy:-cy, cx:-cx] = 1
        elif cy != 0 and cx == 0:
            mask[cy:-cy, ...] = 1
        elif cy == 0 and cx != 0:
            mask[..., cx:-cx] = 1
        if fftshift:
            mask = torch.fft.ifftshift(mask, dim=self.dims)
        return mask

    def __getitem__(self, idx):
        """
        Generate a single MRI data sample with masked k-space and noise.
        
        In training mode, randomly selects mask cutoffs and noise variance for augmentation.
        In test mode, uses fixed parameters for reproducibility.
        
        Args:
            idx (int): Index of the sample to retrieve.
        
        Returns:
            tuple: A tuple containing four torch.Tensor elements:
                - kdata (torch.Tensor): Noisy, masked k-space data. Complex tensor of shape (H, W).
                - mask (torch.Tensor): Binary mask indicating which k-space frequencies are retained.
                    Integer tensor of shape (H, W) with values 0 or 1.
                - x0 (torch.Tensor): Zero-filled reconstruction obtained by applying inverse FFT
                    to the masked noisy k-space. Complex tensor of shape (H, W). This is the
                    initial degraded image used as input to reconstruction models.
                - xtrue (torch.Tensor): Ground truth MRI image in image domain. Complex tensor
                    of shape (H, W).
        
        Note:
            All returned tensors have the batch dimension removed (squeezed) and are 2D.
        """

        xtrue = self.xtrue[idx, ...].unsqueeze(0)
        kdata = self.kdata_true[idx, ...].unsqueeze(0)

        if self.train:
            # cuttoffs being passed to _generate_mask() should be at max half of the image size!
            # otherwise the mask might be all zeroes!

            # randomly pick cutoffs for the masks
            if self.cutoff_range_y[0] != self.cutoff_range_y[1]:
                cutoff_y = torch.randint(
                    self.cutoff_range_y[0], self.cutoff_range_y[1], (1,)
                ).item()
            else:
                cutoff_y = self.cutoff_range_y[0]
            if self.cutoff_range_x[0] != self.cutoff_range_x[1]:
                cutoff_x = torch.randint(
                    self.cutoff_range_x[0], self.cutoff_range_x[1], (1,)
                ).item()
            else:
                cutoff_x = self.cutoff_range_x[0]

            # randomly choose noise variance
            noise_var = self.noise_vars[0] + torch.rand(1) * (
                self.noise_vars[1] - self.noise_vars[0]
            )

            seed = torch.randint(0, 1000, (1,)).item()
        else:
            # Controlled random generation for testing
            if self.seed is None:
                raise ValueError(
                    "Random seed must be set in test mode for reproducibility."
                )
            seed = self.seed

            if self.cutoff_range_y[0] != self.cutoff_range_y[1] or self.cutoff_range_x[0] != self.cutoff_range_x[1]:
                raise ValueError("A specific cutoff must be chosen in test-mode, no range.")

            if type(self.noise_vars) is not float:
                raise ValueError(
                    f"Choose a unique noise-variance for testing.\n\
                        Currently, the interval {self.noise_vars} is specified."
                )

            cutoff_y, cutoff_x = (
                self.cutoff_range_y[0],
                self.cutoff_range_x[0],
            )  
            noise_var = self.noise_vars

        # The same amount of the image is masked from both sides, thus one value for y and x suffices!
        mask = self._generate_mask(cutoff_y, cutoff_x).unsqueeze(0)

        # Adds gaussian noise and applies mask
        kdata, _ = add_gaussian_noise(
            kdata,
            mask,
            mean_dim=self.dims,
            noise_var=noise_var,
            seed=seed,
        )

        x0 = self.op.apply_adjoint(kdata, mask)

        x0 = x0.squeeze(0)
        mask = mask.squeeze(0)
        kdata = kdata.squeeze(0)
        xtrue = xtrue.squeeze(0)

        return kdata, mask, x0, xtrue

