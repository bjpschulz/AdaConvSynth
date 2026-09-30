import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SingleCoilDC(nn.Module):
    def __init__(self, norm="ortho"):
        super().__init__()

        self.norm = norm

    def forward(self, xreg, kdata, mask, reg_param):
        kreg = torch.fft.fftn(xreg, dim=(-2, -1), norm=self.norm)

        if mask is not None:
            kest = (
                mask * (reg_param / (1.0 + reg_param)) * kreg
                + (1.0 / (1.0 + reg_param)) * kdata
                + (1 - mask) * kreg
            )

            x = torch.fft.ifftn(kest, dim=(-2, -1), norm=self.norm)
        else:
            xu = torch.fft.ifftn(kdata, dim=(-2, -1), norm=self.norm)
            x = (1 / (1 + reg_param)) * (xu + reg_param * xreg)

        return x


class MoDL(nn.Module):
    """
    TODO: write description
    """

    def __init__(
        self,
        mr_operator,
        cnn_block,
        lambda_mode="lambda_scalar",
        cnn_block_reg_parameter=None,
        n_iterations=10,
        normalize_input=False,
    ):
        super().__init__()

        self.mr_operator = mr_operator
        self.lambda_mode = lambda_mode
        self.cnn_block = cnn_block
        self.cnn_block_reg_parameter = cnn_block_reg_parameter
        self.n_iterations = n_iterations
        self.dc = SingleCoilDC()
        self.normalize_input = normalize_input

        # regularization parameter prior to activation
        self.lambda_reg_raw = nn.Parameter(
            torch.tensor(
                2.0,
            ),
            requires_grad=True,
        )

    @property
    def lambda_reg(self):
        return F.softplus(self.lambda_reg_raw, beta=1.5)

    def apply_cnn(self, x, normalize_input=False):

        # x is a 2D complex-valued image
        # -> transform to real, apply nn, transform back

        x = torch.view_as_real(x)
        nb, ny, nx, nc = x.shape
        x = rearrange(x, "b y x c -> b c y x")

        # normalize or not:
        if normalize_input:
            (mu, std) = (
                torch.mean(x, dim=tuple(range(2, x.ndim)), keepdim=True),
                torch.std(x, dim=tuple(range(2, x.ndim)), keepdim=True),
            )
            x = (x - mu) / std
        x = self.cnn_block(x)
        if normalize_input:
            x = x * std + mu
        x = rearrange(x, "b c y x -> b y x c", b=nb, y=ny, x=nx, c=nc)
        x = torch.view_as_complex(x.contiguous())
        return x

    def get_lambda_cnn(self, x):
        """Estimate the scalar lambda from the image."""

        # convert to two channels
        x = torch.view_as_real(x).permute(0, 3, 1, 2)

        lambda_cnn = self.lambda_reg * F.softplus(
            self.cnn_block_reg_parameter(x), beta=1.5
        )

        return lambda_cnn.unsqueeze(1)

    def forward(self, x, kdata, mask):

        if self.lambda_mode == "lambda_cnn_scalar":
            lambda_reg = self.get_lambda_cnn(x)
        elif self.lambda_mode == "lambda_scalar":
            lambda_reg = self.lambda_reg

        if self.n_iterations == 0:
            x = self.apply_cnn(x, normalize_input=self.normalize_input)
        else:
            for _ in range(self.n_iterations):
                # obtain CNN-prior
                xnn = self.apply_cnn(x, normalize_input=self.normalize_input)

                # data-consistency step
                # x = self.dc(xnn, kdata, mask, self.lambda_reg)
                x = self.dc(xnn, kdata, mask, lambda_reg)

        return x


class MoDLBlock(nn.Module):
    """CNN block as described in the MoDL paper (Conv+BN+ReLU, last layer no ReLU).
    Forward returns residual output: net(x) + x, when shapes are compatible.
    """

    def __init__(
        self,
        n_layers: int = 5,
        in_channels: int = 2,
        out_channels: int = 2,
        num_filters: int = 64,
        kernel_size: int = 3,
        padding: int = 1,
        bias: bool = False,
    ):
        super().__init__()

        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")

        layers = []

        if n_layers == 1:
            # Single layer: in_channels -> out_channels, no ReLU (per "last layer no ReLU")
            layers.append(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size, padding=padding, bias=bias
                )
            )
            layers.append(nn.BatchNorm2d(out_channels))
        else:
            # Layer 1: in_channels -> num_filters (with ReLU)
            layers.append(
                nn.Conv2d(
                    in_channels, num_filters, kernel_size, padding=padding, bias=bias
                )
            )
            layers.append(nn.BatchNorm2d(num_filters))
            layers.append(nn.ReLU(inplace=True))

            # Layers 2..N-1: num_filters -> num_filters (with ReLU)
            for _ in range(n_layers - 2):
                layers.append(
                    nn.Conv2d(
                        num_filters,
                        num_filters,
                        kernel_size,
                        padding=padding,
                        bias=bias,
                    )
                )
                layers.append(nn.BatchNorm2d(num_filters))
                layers.append(nn.ReLU(inplace=True))

            # Layer N: num_filters -> out_channels (NO ReLU)
            layers.append(
                nn.Conv2d(
                    num_filters, out_channels, kernel_size, padding=padding, bias=bias
                )
            )
            layers.append(nn.BatchNorm2d(out_channels))

        self.net = nn.Sequential(*layers)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)

        # Residual add is only valid if shapes match.
        # If you always want residual learning, set out_channels == in_channels.
        if y.shape != x.shape:
            raise RuntimeError(
                f"Residual add requires net(x) and x to have the same shape, "
                f"but got net(x): {tuple(y.shape)} vs x: {tuple(x.shape)}. "
                f"Set out_channels=in_channels (and keep spatial sizes unchanged) to use y + x."
            )

        return y + x
