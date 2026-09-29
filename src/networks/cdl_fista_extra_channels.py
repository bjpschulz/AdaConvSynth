import torch
import torch.nn as nn
import torch.nn.functional as F

from ..operators.grad_ops import GradOperators
from ..utils.linalg import power_iteration, conj_grad
from einops import rearrange

from math import sqrt


def project_conv_dico(d_filter):
    """
    project filters onto unit-sphere

    Return None since in-place operation

    """
    with torch.no_grad():
        n_filters = d_filter.shape[0]
        for kf in range(n_filters):
            d_filter[kf, ...].div_(         # in-place division
                torch.norm(d_filter[kf, ...].flatten(), p=2, keepdim=True)
            )


class SoftShrinkAct(nn.Module):
    """
    soft shrinkage operator; lambda can be a trainable parameter
    """

    def __init__(self):
        super(SoftShrinkAct, self).__init__()

    def forward(self, x, threshold):
        # the soft-thresholding can be expressed as
        # S_t(x) = ReLU(x-t) - ReLU(-x -t)
        return F.relu(x - threshold) - F.relu(-x - threshold)


class ApproxSoftShrinkAct(nn.Module):
    def __init__(self, b=0.001):
        super(ApproxSoftShrinkAct, self).__init__()

        self.b = b

    def approx_softshrink(self, x, lambd):
        return x + torch.tensor(1.0 / 2) * (
            torch.sqrt(torch.pow(x - lambd, 2) + self.b)
            - torch.sqrt(torch.pow(x + lambd, 2) + self.b)
        )

    def forward(self, x, threshold):
        return self.approx_softshrink(x, threshold)


def softplus(x, beta=1.5):
    """Constrain x to be in (bound,infty)."""
    return -(1 / beta) * torch.nn.functional.logsigmoid(-beta * x)


def softplus_inverse(x, beta=1.5):
    """Inverse of `softplus_transformation`."""
    return x + torch.log(-torch.expm1(-beta * x)) / beta


def sigmoid(x, beta=1.0):
    """Constraint x to be in the range given by 'bounds'."""
    return F.sigmoid(beta * x)


def sigmoid_inverse(x, beta: 1.0):
    """Constraint x to be in the range given by 'bounds'."""
    return torch.logit(x) / beta


class CDLFISTA2D(nn.Module):
    def __init__(
        self,
        mr_operator,
        n_dict_filters,                          # removed default, since always required
        dict_filter_size=(11, 11),
        n_iterations=64,
        cnn_block_reg_parameter=None,
        reg_parameter_bounds=(None, None),
        high_pass_filtering=True,
        lambda_mode="lambda_scalar",        # Options: "lambda_scalar", "lambda_cnn_scalar", "lambda_cnn_map"
        phase="training",
        version="version3",
    ):
        super(CDLFISTA2D, self).__init__()

        self.mr_operator = mr_operator
        self.dict_filter_size = dict_filter_size
        self.n_dict_filters = n_dict_filters
        self.n_iterations = n_iterations

        self.lambda_mode = lambda_mode

    
        if lambda_mode in ["lambda_cnn_scalar", "lambda_cnn_map"]:
            self.cnn_block_reg_parameter = cnn_block_reg_parameter  # UNet gets set here


        # initialize HALFED d_filter randomly
        filter_shape = (self.n_dict_filters // 2, 1) + self.dict_filter_size
        d_filter_init = 0.001 * torch.randn(filter_shape)
        # False currently here, frozen by default!
        self.d_filter = nn.Parameter(d_filter_init, requires_grad=False)     # This is of shape (n_dict_filters / 2, 1, ky, kx), .data attribute later used for loading
        
        # normalization for later maybe

        # paddings -- for What ???
        self.padding = tuple(k // 2 for k in self.dict_filter_size)
        self.pad = tuple(p for p in self.padding for _ in range(2))  # duplicate each
        self.crop = tuple(-p for p in self.padding for _ in range(2))  # negate duplicates
        # self.crop = -self.pad ?

        # regularization parameter
        self.lambda_reg_raw = nn.Parameter(
            torch.tensor([-0.0], requires_grad=True), requires_grad=True
        )

        self.beta_softplus = 1.5

        self.conv_op_transposed = F.conv_transpose2d
        self.conv_op = F.conv2d

        # self.soft_thresholding = SoftShrinkAct()
        self.soft_thresholding = ApproxSoftShrinkAct()

        # lower bound for regularization parameter;
        # should theoretically guarantee that no overfitting occurs
        self.reg_parameter_lower_bound = (
            torch.tensor(reg_parameter_bounds[0])
            if reg_parameter_bounds[0] is not None
            else None
        )

        self.reg_parameter_upper_bound = (
            torch.tensor(reg_parameter_bounds[1])
            if reg_parameter_bounds[0] is not None
            else None
        )

        self.high_pass_filtering = high_pass_filtering
        if high_pass_filtering:
            self.lambda_reg_high_pass_raw = torch.nn.Parameter(         # This is beta
                torch.tensor(20.0), requires_grad=True
            )

            self.grad_ops = GradOperators(2)

        self.phase = phase          # "training" or "inference"

        # version for construction of the unet for lambda map estimation
        # unet_1 corresponds to a 2-to-K channels UNet
        # unet_2 corresponds to 2-to-2 channels, where each feature map is processed independently
        # unet_3 corresponds to K-t-K channels
        # with the same 2-to-K UNet
        self.version = version

    @property
    def lambda_reg_high_pass(self):
        # return F.softplus(self.lambda_reg_raw, beta=self.beta_softplus)
        return softplus(self.lambda_reg_high_pass_raw, beta=self.beta_softplus)

    @property
    def lambda_reg(self):
        # return F.softplus(self.lambda_reg_raw, beta=self.beta_softplus)
        return softplus(self.lambda_reg_raw, beta=self.beta_softplus)

    def high_pass_filter(self, x):
        # the low filter component is given as the solution of
        # low_comp=argmin_l 1/2*||l - x||_2^2 + lmbda*||G l||_2^2
        # see https://sporco.readthedocs.io/en/latest/modules/sporco.signal.html#sporco.signal.tikhonov_filter
        # i.e. solve a linear problem, then subtract low-pass component to get xhigh = x - xlow

        operator = lambda x: x + self.lambda_reg_high_pass * self.grad_ops.apply_GHG(x)
        rhs = x
        xlow = conj_grad(operator, rhs, x, niter=16)

        # subtract low-pass component
        xhigh = x - xlow
        return xhigh, xlow


    def conv_dico(self, sparse_code):
        """Synthesis - Apply convolutional dictionary to sparse codes.
        
        D (the convolutional dictionary) SYNTHESIZES a 2-channel image (real/imag)
        from K sparse maps s via grouped transposed convolutions.

        Concatenation MUST happen here to apply same filter to real and imag part consistently during training.
        If done during init, learning of full d_filter would diverge for real and imag part.
        """
        # prepare filter,
        full_cd_filters = torch.cat(2 * [self.d_filter], dim=0)

        sparse_code_padded = F.pad(sparse_code, self.pad, mode="circular")
        reconstructed = self.conv_op_transposed(
            sparse_code_padded, 
            full_cd_filters, 
            groups=2, 
            padding=self.padding
        )
        reconstructed = F.pad(reconstructed, self.crop)
        
        return reconstructed

    def conv_dico_adjoint(self, input):
        """Apply adjoint of convolutional dictionary.

        D^H = D^T (the adjoint is the transpose, since real) ANALYZES a 2-channel image
        into K feature maps via grouped standard convolutions (correlations).
        
        groups = 2:
        Specifies that the input channels are split into 2 groups of 1 channel each.
        This means both our real and imag channels get their own "set of filters",
        being just 1 kernel matrix here.
        """
        # prepare filter
        full_cd_filters = torch.cat(2 * [self.d_filter], dim=0)

        input_padded = F.pad(input, self.pad, mode="circular")
        sparse_codes = self.conv_op(
            input_padded, 
            full_cd_filters, 
            groups=2,
            padding=self.padding
        )
        sparse_codes = F.pad(sparse_codes, self.crop)
        
        return sparse_codes

    def ista(self, input, reg_parameter, niter=4):
        """ISTA MODULE,

        approximately solves the problem
            min_s 1/2 || Ds - x ||_2^2 + lambda * ||s||_1
        with ISTA for a fixed number of iterations.

        Returns:
           approximation of the solution of the problem

        Convergence is only linear for ISTA, thus, we better use FISTA instead.
        However, this is included for completeness.
        """

        # x is a complex-valued 3d image with shape (b,z,y,x)
        input = rearrange(torch.view_as_real(input), "b y x ri -> b ri y x")        # <-- What is input?

        # get shape
        nb, _, ny, nx = input.shape
        input_type = input.dtype
        input_device = input.device

        # initialize sparse code
        sparse_code = torch.zeros(nb, self.n_dict_filters, ny, nx, dtype=input_type).to(    # <-- why self.n_dict_filters
            input_device
        )

        # define operator whose opnorm needs to be estimated
        operator = lambda s: self.conv_dico_adjoint(self.conv_dico(s))      # <-- is this good practice, even though lambda functions are specifically for not needing to give a name?
                                                                            # <-- just used for power iteration

        # initial valze for power iteration (can be small, since conv op)
        initial_value = torch.randn(nb, self.n_dict_filters, 8, 8, dtype=input_type).to(
            input_device
        )
        dico_op_norm = power_iteration(operator, initial_value, niter=36)

        for _ in range(niter):
            Ds = self.conv_dico(sparse_code)

            residual = Ds - input
            gradient = self.conv_dico_adjoint(residual)

            step_size = 0.9 * (1.0 / dico_op_norm)              # <-- power iteration being used here
            threshold = reg_parameter / dico_op_norm
            sparse_code = self.soft_thresholding(
                sparse_code - step_size * gradient, threshold
            )

        return sparse_code

    def fista(
        self,
        operator,
        operator_adjoint,
        data,
        reg_parameter,
        niter,
        initial_value=None,
        convergent_iterates=True,
    ):
        """FISTA MODULE,

        approximately solves the problem
            min_s 1/2 || operator s - data ||_2^2 + lambda * ||s||_1
        with FISTA for a fixed number of iterations.

        Returns:
           approximation of the solution of the problem
        """

        # get shape (data will be complex-valued)
        nb, ny, nx = data.shape
        input_type = torch.float32  # data type for input of conv dicos
        data_device = data.device

        # initial values for the iterations
        sparse_code = torch.zeros(nb, self.n_dict_filters, ny, nx, dtype=input_type).to(
            data_device
        )
        zeta = torch.zeros(nb, self.n_dict_filters, ny, nx, dtype=input_type).to(data_device)
        beta = torch.tensor(1.0).to(data_device)

        # initial value for power iteration (can be small, since conv op)
        if initial_value is None:
            initial_value = torch.randn(nb, self.n_dict_filters, ny, nx, dtype=input_type).to(
                data_device
            )

        gram_operator = lambda s: operator_adjoint(operator(s))
        op_norm = power_iteration(gram_operator, initial_value, niter=16)

        for iteration in range(niter):

            conv_dico_sparse_code = operator(zeta)

            residual = conv_dico_sparse_code - data
            gradient = operator_adjoint(residual)

            step_size = 0.9 * (1.0 / op_norm)
            threshold = reg_parameter / op_norm             # <-- essential! each fista iter uses current reg_parameter / lambda_reg in soft-thresholding

            # soft thresholding
            sparse_code_new = self.soft_thresholding(
                zeta - step_size * gradient, threshold
            )

            if convergent_iterates:
                # THIS CORRESPONDS TO THE UPDATE WHICH, IN CONTRAST TO THE STANDARD
                # UPDATE OF BECK ET AL (WHICH ONLY ENSURES CONVERGENCE OF THE FUNCTIONAL VALUES)
                # ALSO ENSURE CONVERGENCE OF THE ITERATES;
                # A. Chambolle and C. H. Dossal, “On the convergence of the iterates of
                # ”FISTA”,” J. Optim. Theory Appl., vol. 166, no. 3, p. 25, 2015.
                zeta = sparse_code_new + ((iteration + 1) - 1.0) / (
                    (iteration + 1) + 2
                ) * (sparse_code_new - sparse_code)
            else:

                # THIS CORRESPONDS TO THE STANDARD FISTA UPDATE IN
                # A. Beck and M. Teboulle, “A fast iterative shrinkage-thresholding
                # algorithm for linear inverse problems,” SIAM J. Imaging Sci., vol. 2,
                # no. 1, pp. 183–202, 2009.
                # update beta
                beta_new = (1.0 + torch.sqrt(1 + 4 * beta**2)) / 2.0

                # update sparse codes
                zeta = sparse_code_new + (beta - 1.0) / beta_new * (
                    sparse_code_new - sparse_code
                )
                beta = beta_new

            sparse_code = sparse_code_new

        return sparse_code

    def sparse_approximation(
        self, operator, operator_adjoint, kdata, reg_parameter, niter=64, initial_value=None
    ):
        """ Is this even needed? It just calls fista and uses the attribute!
        """

        if niter is None:
            niter = self.n_iterations

        # unrolled ista/fista for getting sparse code
        s = self.fista(
            operator, operator_adjoint, kdata, reg_parameter, self.n_iterations, initial_value=initial_value
        )

        return s

    def mr_operator_conv_dico(self, sparse_code, mask):
        """The composition of mr forward operator and dictionary, i.e. A = BD, thus
        a mapping from sparse codes to k-space."""

        dico_sparse_code = self.conv_dico(sparse_code)

        x = torch.view_as_complex(
            rearrange(dico_sparse_code, "b ri y x  -> b y x ri").contiguous()
        )

        return self.mr_operator.apply_forward(x, mask)

    def mr_operator_conv_dico_adjoint(self, kdata, mask):

        x = self.mr_operator.apply_adjoint(kdata, mask)

        x = rearrange(torch.view_as_real(x), "b y x ri -> b ri y x")

        return self.conv_dico_adjoint(x)

    def get_lambda_cnn(self, x, residual_connection=False):
        """ Estimate the set of paramter maps from an input image x0.
        
        This means the network should be a 2-to-K U-Net for estimating maps.
        The λ maps are predicted during the cnns/unets forward pass here, BUT learned during backpropagation.
        """

        # convert to two channels
        x = torch.view_as_real(x).permute(0, 3, 1, 2)

        # estimate lambda reg parameter (single value or map)
        # self.cnn_block_reg_parameter is the UNet object
        # lambda_cnn is a torch.Tensor (the UNet's output)
        lambda_cnn = self.cnn_block_reg_parameter(x)            # <-- this is getting it already, forward
        # Shape: (batch, 32, y, x) - 32 parameter maps

        if residual_connection is True:
            xmu = x.mean(1, keepdim=True)
            lambda_cnn = lambda_cnn + torch.concat(lambda_cnn.shape[1] * [xmu], dim=1)

        if self.lambda_mode == "lambda_cnn_scalar":             # <-- are we even interested in this?
            # means the output is a single scalar value
            # e.g., lambda_cnn.shape = [4,1] would be 4 values for a mini-batch size of 4
            lambda_cnn = lambda_cnn.unsqueeze(-1).unsqueeze(-1)

        elif self.lambda_mode == "lambda_cnn_map":
            # Dupe to 64 channels, apply bounds
            lambda_cnn = self.stack_lambda_apply_bounds(lambda_cnn)

        return lambda_cnn

    def get_lambda_cnn_v2(self, x, residual_connection=False):
        """Estimate lambda-map from adjoint of dictionary applied to input image.
        UNet is K-to-K/2 channels, which is then duped.
        This K channels go into the sparse approximation later."""

        # reshape complex-valued image to real/image representation and apply D^T;
        # will have shape (b, 2 * n_dict_filters, y, x)
        sparse_codes_like = self.conv_dico_adjoint(
            rearrange(torch.view_as_real(x), "b y x ri -> b ri y x") # batch real/imag y x
        )   # view_as_real makes it [1,320,320,2] --> permute --> [1,2,320,320] --> conv_dico_adjoint --> [1,64,320,320]

        # must map e.g. 128 --> 64
        lambda_cnn = self.cnn_block_reg_parameter(sparse_codes_like)
        lambda_cnn = self.stack_lambda_apply_bounds(lambda_cnn)

        return lambda_cnn

    def get_lambda_cnn_v3(self, x, residual_connection=False):
        """Estimate lambda-map from adjoint of dictionary applied to input image and reshaping.
        The UNet is 2-to-1 channels.
        Then the output is stacked to K/2 - K/2 be used for real and imaginary part."""

        # reshape complex-valued image to real/image representation and apply D^T;
        # view_as_real makes it --> [1,320,320,2]
        # permute --> [1,2,320,320]
        # conv_dico_adjoint uses group convolution! --> [1,64,320,320]
        # That means different single kernel matrix as filter applied to group seperated real and imag channels.
        # Used when channels have DISTINCT meaning/properties, thus different filters needed to learn these.
        sparse_codes = self.conv_dico_adjoint(
            rearrange(torch.view_as_real(x), "b y x ri -> b ri y x")
        )
        x_as_real = rearrange(torch.view_as_real(x), "b y x ri -> b ri y x")

        # for unet_21, the different feature maps are processed independently;
        # therefore the corresponding for real and imaginary part of each feature map
        # are stacked to a 2-channel image and the batch dimension is expanded accordingly.
        # here, a 2-to-1 UNet is applied to each of the feature maps
        # print(sparse_codes.shape, sparse_codes.dtype)

        nb, _, ny, nx = sparse_codes.shape
        n_dict_filters = int(self.n_dict_filters / 2)

        # group real and imag parts of the feature maps
        input_list = [
            torch.stack(
                [x_as_real[:, 0, ...], x_as_real[:, 1, ...], sparse_codes[:, k, ...], sparse_codes[:, n_dict_filters + k, ...]],
                dim=1,
            )
            for k in range(n_dict_filters)
        ]

        # bring the filters dimension to the batch dimension
        # shape (b * n_dict_filters, 2, y, x). i.e. (b*64, 2, 320, 320) for 64 filters ????????
        input = torch.concat(input_list, dim=0)

        # estimate the lambda maps with a 2-channel input and a 1-channel output CNN
        # interpretation: the SAME CNN is applied to the real and imaginary part of
        # of the feature map
        lambda_cnn = self.cnn_block_reg_parameter(input)

        # print(lambda_cnn.shape, lambda_cnn.dtype)

        if residual_connection:
            lambda_cnn = lambda_cnn + input

        # bring back the batch dimension to the filters
        lambda_cnn = rearrange(
            lambda_cnn,
            "(b n_dict_filters) 1 y x -> b n_dict_filters y x",
            b=nb,
            n_dict_filters=n_dict_filters,
            y=ny,
            x=nx,
        )

        # print(lambda_cnn.shape, lambda_cnn.dtype)
        
        # stack the lambda cnn map to be used for real and imaginary part, apply bounds
        # This finally makes it of shape (b*64, 2, 320, 320) for 64 filters  ??????? CHECK THIS
        lambda_cnn = self.stack_lambda_apply_bounds(lambda_cnn)

        return lambda_cnn

    def stack_lambda_apply_bounds(self, lambda_cnn):
        # stack the lambda cnn map to be used for real and imaginary part
        lambda_cnn = torch.concat(2 * [lambda_cnn], dim=1)

        # print(lambda_cnn.shape, lambda_cnn.dtype)

        if (
            self.reg_parameter_lower_bound is not None
            and self.reg_parameter_upper_bound is None
        ):  # [a, \infty], a>0
            lambda_cnn = self.reg_parameter_lower_bound + softplus(
                lambda_cnn, beta=self.beta_softplus
            )
        elif (
            self.reg_parameter_lower_bound is None
            and self.reg_parameter_upper_bound is not None
        ):  # [0, a]
            lambda_cnn = self.reg_parameter_upper_bound * sigmoid(lambda_cnn, beta=1.0)
            # print("[0, a]")
        elif (
            self.reg_parameter_lower_bound is not None
            and self.reg_parameter_upper_bound is not None
        ):  # [a, b]
            lambda_cnn = self.reg_parameter_lower_bound + (
                self.reg_parameter_upper_bound - self.reg_parameter_lower_bound
            ) * softplus(lambda_cnn, beta=self.beta_softplus)
        
        return lambda_cnn
        
    def forward(self, x0, kdata, mask, s_start = None, reg_parameter = None):

        if self.high_pass_filtering:
            # Splits x0 into low-frequency (xlow) and high-frequency (xhigh) components
            xhigh, xlow = self.high_pass_filter(x0)
            # y' = y - A xlow   (7)
            # Subtracts low-frequency k-space from kdata
            # From this point forward, FISTA optimizes only the high-frequency reconstruction
            kdata = kdata - self.mr_operator.apply_forward(xlow, mask)

        if reg_parameter is None:
            if self.lambda_mode == "lambda_scalar":
                lambda_reg = self.lambda_reg
            elif self.lambda_mode == "lambda_cnn_scalar":       # <-- even needed?
                lambda_reg = self.get_lambda_cnn(x)
            elif self.lambda_mode == "lambda_cnn_map":
                if self.version == "version1":
                    lambda_reg = self.get_lambda_cnn(x)         # x is x0, zero-filled adjoint reconstruction
                elif self.version == "version2":
                    lambda_reg = self.get_lambda_cnn_v2(xhigh)
                elif self.version == "version3":
                    lambda_reg = self.get_lambda_cnn_v3(x0)
        else:
            lambda_reg = reg_parameter
        # print(
            # f"lambda_highpass{round(self.lambda_reg_high_pass.item(),2)}",
            # f"lambda_reg{round(self.lambda_reg.item(),2)}",
        # )

        # print(
        #     f"lambda_cnn_min{round(lambda_reg.min().item(),2)}",
        #     f"lambda_cnn_min{round(lambda_reg.max().item(),2)}",
        # )
        operator = lambda s: self.mr_operator_conv_dico(s, mask)
        operator_adjoint = lambda k: self.mr_operator_conv_dico_adjoint(k, mask)

        # sparse coding
        s = self.sparse_approximation(
            operator, operator_adjoint, kdata, lambda_reg, self.n_iterations, initial_value=s_start
        )

        # apply convolutional dictionary to sparse code to get image estimate
        Ds = self.conv_dico(s)

        # x is a complex-valued 2d image with shape (b,y,x)
        x = torch.view_as_complex(rearrange(Ds, "b ri y x  -> b y x ri").contiguous())

        if self.high_pass_filtering:
            x = x + xlow

        if self.phase == "training":
            return x
        elif self.phase == "inference":
            return (
                x,
                s,
                lambda_reg
            )  # return sparse codes as well during inference to be able to analyze them

# %%
