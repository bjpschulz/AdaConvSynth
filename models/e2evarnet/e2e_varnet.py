import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class End2EndVarNet(nn.Module):
    """
    TODO: write description
    """

    def __init__(
        self,
        mr_operator,
        cnn_block,
        n_iterations=12,
        normalize_input=False,
    ):
        super().__init__()

        self.mr_operator = mr_operator

        self.cnn_block = cnn_block
        self.n_iterations = n_iterations
        self.normalize_input = normalize_input

        # regularization parameter prior to activation
        self.lambda_reg_raw = torch.nn.Parameter(
            -4 * torch.ones(n_iterations, requires_grad=True), requires_grad=True
        )

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

        kdata_measured = kdata.clone()

        for iteration in range(self.n_iterations):
            step_size = torch.nn.functional.softplus(
                self.lambda_reg_raw[iteration], beta=1.5
            )

            xnn = self.apply_cnn(x)
            knn = torch.fft.fftn(xnn, dim=(-2, -1), norm="ortho")

            kdata = kdata - step_size * mask * (kdata - kdata_measured) + knn

            x = torch.fft.ifftn(kdata, dim=(-2, -1), norm="ortho")

        return x
