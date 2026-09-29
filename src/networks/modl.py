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
    A network that implements the MoDL (Model-Based Deep Learning) approach
    for MRI reconstruction as described in Aggarwal et al., "MoDL: Model-Based
    Deep Learning Architecture for Inverse Problems", TMI 2019.

    The network implements the following iterations:
    Let x0 = A^H y be the zero-filled reconstruction, where A is the
    forward operator and y the k-space data.
    For k=0, ..., n_iteratons - 1 do:
        1. x_nn = CNN(x_k) (denoising step using a CNN)
        2. x_{k+1} = DC(x_nn, y, lambda) (data-consistency step)

    where DC is a data-consistency step that solves the optimization problem
        x_{k+1} = argmin_x || A x - y ||_2^2 + lambda || x - x_nn ||_2^2
    in closed form.

    Additionally, we make it possible to estimate the scalar regularization parameter
    with a small CNN from the input image.
    """

    def __init__(
        self,
        mr_operator,
        cnn_block,
        lambda_mode="lambda_scalar",
        cnn_block_reg_parameter=None,
        n_iterations=6,
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
                -2.0,
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
            # estimate the scalar value from an input image such that the lambda is
            # adapted to the considered image (i.e. to the noise level at hand)
            lambda_reg = self.get_lambda_cnn(x)
        elif self.lambda_mode == "lambda_scalar":
            # only use one global scalar; this might yield suboptimal results
            # if the noise level varies between images
            lambda_reg = self.lambda_reg

        if self.n_iterations == 0:
            x = self.apply_cnn(x, normalize_input=self.normalize_input)
        else:
            for _ in range(self.n_iterations):
                print(_)
                # obtain CNN-prior
                xnn = self.apply_cnn(x, normalize_input=self.normalize_input)

                # data-consistency step
                # x = self.dc(xnn, kdata, mask, self.lambda_reg)
                x = self.dc(xnn, kdata, mask, lambda_reg)

        return x
