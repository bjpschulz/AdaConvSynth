import torch
import torch.nn as nn
import numpy as np

import torch.nn.functional as F


def pad_circular_nd(x, pad, dim):
    """
    function for circular padding
    """

    if isinstance(dim, int):
        dim = [dim]

    for d in dim:
        if d >= len(x.shape):
            raise IndexError(f"dim {d} out of range")

        idx = tuple(slice(0, None if s != d else pad, 1) for s in range(len(x.shape)))
        x = torch.cat([x, x[idx]], dim=d)

        idx = tuple(
            slice(None if s != d else -2 * pad, None if s != d else -pad, 1)
            for s in range(len(x.shape))
        )

        x = torch.cat([x[idx], x], dim=d)
        pass

    return x


def create_grad_kernel(ndims, diff_types="fwd"):
    """
    function for creating finite-differences kernels
    """

    if diff_types == "fwd":
        diff_vect = np.array([0, 1, -1])

    elif diff_types == "bckwd":
        diff_vect = np.array([1, -1, 0])

    elif diff_types == "symm":
        diff_vect = np.array([1, 0, -1])

    if ndims == 1:
        dx = np.zeros((1, 3))
        dx = diff_vect

        filters_list = [dx]
        grad_kernel = torch.zeros(1, 1, 3)
        for kf in range(ndims):
            h = torch.tensor(filters_list[kf])
            grad_kernel[kf, 0, ...] = h

    elif ndims == 2:
        dx = np.zeros((3, 3))
        dy = np.zeros((3, 3))

        dx[1, :] = diff_vect
        dy[:, 1] = diff_vect

        filters_list = [dx, dy]
        grad_kernel = torch.zeros(2, 1, 3, 3)
        for kf in range(ndims):
            h = torch.tensor(filters_list[kf])
            grad_kernel[kf, 0, ...] = h

    if ndims == 3:
        d0 = np.zeros((3, 3, 3))
        d1 = np.zeros((3, 3, 3))
        d2 = np.zeros((3, 3, 3))

        d0[1, 1, :] = diff_vect  # x direction
        d1[1, :, 1] = diff_vect  # y direction
        d2[:, 1, 1] = diff_vect  # t direction

        filters_list = [d0, d1, d2]

        grad_kernel = torch.zeros(3, 1, 3, 3, 3)
        for kf in range(ndims):

            h = torch.tensor(filters_list[kf])
            grad_kernel[kf, 0, ...] = h

    return grad_kernel


class GradOperators(nn.Module):
    """
    module which contains  the opeartions
    G, G^H and G^H G

    the input is always going to be a tensor of one of the shapes

    (1,Nx,Ny,Nt) with dtype either torch.complex or torch.real

    N.B. dim=1 is currently not supported because of the padding F.pad,
    could be easily hand-crafted, though


    """

    def __init__(self, dim):

        super().__init__()

        self.dim = dim
        self.register_buffer("grad_kernel", create_grad_kernel(dim))

        self.npad = 1
        self.pad = tuple([self.npad for k in range(2 * self.dim)])

        if self.dim == 1:
            self.conv_op = F.conv1d
            self.conv_op_transpose = F.conv_transpose1d
        elif self.dim == 2:
            self.conv_op = F.conv2d
            self.conv_op_transpose = F.conv_transpose2d
        elif self.dim == 3:
            self.conv_op = F.conv3d
            self.conv_op_transpose = F.conv_transpose3d

    def apply_G(self, x):

        dtype = x.dtype
        if dtype in [torch.complex32, torch.complex64, torch.complex128]:

            if self.dim == 1:
                mb, Nx = x.shape
                x = torch.view_as_real(x).permute(0, 2, 1)

            if self.dim == 2:
                mb, Nx, Ny = x.shape
                x = torch.view_as_real(x).permute(0, 3, 1, 2)

            elif self.dim == 3:
                mb, Nx, Ny, Nt = x.shape
                x = torch.view_as_real(x).permute(0, 4, 1, 2, 3)

        else:
            x = x.unsqueeze(1)

        n_ch = x.shape[1]

        if n_ch == 2:
            grad_kernel = torch.cat(2 * [self.grad_kernel], dim=0).to(x.device)
            groups = 2

        elif n_ch == 1:
            grad_kernel = self.grad_kernel.to(x.device)
            groups = 1

        # circular padding
        Gx = self.conv_op(
            F.pad(x, self.pad, mode="circular"),
            grad_kernel,
            bias=None,
            padding=1,  # zeropadding cause already circularly padded
            groups=groups,
        )

        npad = self.npad
        if self.dim == 1:
            Gx = Gx[..., npad:-npad]
        elif self.dim == 2:
            Gx = Gx[..., npad:-npad, npad:-npad]  # crop
        elif self.dim == 3:
            Gx = Gx[..., npad:-npad, npad:-npad, npad:-npad]  # crop

        if dtype in [torch.complex32, torch.complex64, torch.complex128]:
            Gx = torch.stack([Gx[:, : self.dim, ...], Gx[:, self.dim :, ...]], dim=-1)
            Gx = torch.view_as_complex(Gx)

        return Gx

    def apply_GH(self, z):

        dtype = z.dtype
        if dtype in [torch.complex32, torch.complex64, torch.complex128]:

            z = torch.concat([z.real, z.imag], dim=1)

        n_ch = z.shape[1]

        if n_ch == 2 * self.dim:
            groups = 2
            grad_kernel = torch.cat(2 * [self.grad_kernel], dim=0).to(z.device)
        elif n_ch == self.dim:
            groups = 1
            grad_kernel = self.grad_kernel.to(z.device)

        GHz = self.conv_op_transpose(
            F.pad(z, self.pad, mode="circular"),
            grad_kernel,
            bias=None,
            padding=1,  # zeropadding cause already circularly padded
            groups=groups,
        )

        npad = self.npad
        if self.dim == 1:
            GHz = GHz[..., npad:-npad]  # crop
        elif self.dim == 2:
            GHz = GHz[..., npad:-npad, npad:-npad]  # crop
        elif self.dim == 3:
            GHz = GHz[..., npad:-npad, npad:-npad, npad:-npad]  # crop

        if dtype in [torch.complex32, torch.complex64, torch.complex128]:

            if self.dim == 1:
                GHz = torch.view_as_complex(GHz.permute(0, 2, 1).contiguous())
            elif self.dim == 2:
                GHz = torch.view_as_complex(GHz.permute(0, 2, 3, 1).contiguous())
            elif self.dim == 3:
                GHz = torch.view_as_complex(GHz.permute(0, 2, 3, 4, 1).contiguous())

        return GHz.squeeze(1)

    def apply_GHG(self, x):

        return self.apply_GH(self.apply_G(x))


class SymGradOperators(nn.Module):
    """
    module which contains the opeartions
    SymG, (SymG)^H and (SymG)^H SymG

    for the symmetrized gradient defined according to:

            https://onlinelibrary.wiley.com/doi/epdf/10.1002/mrm.26352

            page 153 (i.e. 11), Appendix

    the input is always going to be a tensor of one of the shapes

    (1,Nx,Ny,Nt) with dtype either torch.complex or torch.real

    N.B:currently olnly works for 3D

    """

    def __init__(self, dim=3, mu_init=(1.0, 4.0)):

        super(SymGradOperators, self).__init__()

        self.dim = dim
        self.grad_kernel = create_grad_kernel(dim, diff_types="bckwd")

        self.npad = 1
        self.pad = tuple([self.npad for k in range(2 * self.dim)])

        self.conv_op = F.conv3d
        self.conv_op_transpose = F.conv_transpose3d

        self.kernel_x = self.grad_kernel[0].unsqueeze(0)
        self.kernel_y = self.grad_kernel[1].unsqueeze(0)
        self.kernel_t = self.grad_kernel[2].unsqueeze(0)

        self.mu_xy = torch.nn.Parameter(
            torch.log(torch.tensor(mu_init[0])), requires_grad=True
        )
        self.mu_t = torch.nn.Parameter(
            torch.log(torch.tensor(mu_init[1])), requires_grad=True
        )

    def apply_SymG(self, w):
        """
        w.shape = [1,3,Nx,Ny,Nt] = [w1,w2,w3] with each w.shape = [1,Nx,Ny,Nt]
        has to return
        SymG = [\nabla_x w_1,
                    \nabla_y w_2,
                        \nabla_t w_3,
                ]
        """
        mb, _, Nx, Ny, Nt = w.shape
        device = w.device

        w_Re = w.real
        w_Im = w.imag

        SymGw_Re = torch.zeros(mb, 6, Nx, Ny, Nt).to(device)
        SymGw_Im = torch.zeros(mb, 6, Nx, Ny, Nt).to(device)

        npad = self.npad

        kernel_x, kernel_y, kernel_t = (
            self.kernel_x.to(device),
            self.kernel_y.to(device),
            self.kernel_t.to(device),
        )
        kernels_list = [kernel_x, kernel_y, kernel_t]

        mu_list = [self.mu_xy, self.mu_xy, self.mu_t]

        # first part
        for kk in range(3):

            mu = mu_list[kk]
            SymGw_Re[:, kk, ...] = (
                torch.exp(mu)
                * self.conv_op(
                    F.pad(w_Re[:, kk, ...].unsqueeze(0), self.pad, mode="circular"),
                    kernels_list[kk],
                    bias=None,
                    padding=1,  # zeropadding cause already circularly padded
                    groups=1,
                )[..., npad:-npad, npad:-npad, npad:-npad]
            )

            SymGw_Im[:, kk, ...] = (
                torch.exp(mu)
                * self.conv_op(
                    F.pad(w_Im[:, kk, ...].unsqueeze(0), self.pad, mode="circular"),
                    kernels_list[kk],
                    bias=None,
                    padding=1,  # zeropadding cause already circularly padded
                    groups=1,
                )[..., npad:-npad, npad:-npad, npad:-npad]
            )

        # second part
        inp_Re_list = [
            (w_Re[:, 0, ...], w_Re[:, 1, ...]),
            (w_Re[:, 0, ...], w_Re[:, 2, ...]),
            (w_Re[:, 1, ...], w_Re[:, 2, ...]),
        ]
        inp_Im_list = [
            (w_Im[:, 0, ...], w_Im[:, 1, ...]),
            (w_Im[:, 0, ...], w_Im[:, 2, ...]),
            (w_Im[:, 1, ...], w_Im[:, 2, ...]),
        ]

        kernels_list_part2 = [
            (kernel_y, kernel_x),
            (kernel_t, kernel_x),
            (kernel_t, kernel_y),
        ]

        mu_list_part2 = [
            (self.mu_xy, self.mu_xy),
            (self.mu_t, self.mu_xy),
            (self.mu_t, self.mu_xy),
        ]

        counter = 3
        for kk in range(3):

            mu1, mu2 = mu_list_part2[kk]
            SymGw_Re[:, counter, ...] = (
                torch.exp(mu1)
                * 0.5
                * self.conv_op(
                    F.pad(inp_Re_list[kk][0].unsqueeze(0), self.pad, mode="circular"),
                    kernels_list_part2[kk][0],
                    bias=None,
                    padding=1,  # zeropadding cause already circularly padded
                    groups=1,
                )[..., npad:-npad, npad:-npad, npad:-npad]
                + torch.exp(mu2)
                * 0.5
                * self.conv_op(
                    F.pad(inp_Re_list[kk][1].unsqueeze(0), self.pad, mode="circular"),
                    kernels_list_part2[kk][1],
                    bias=None,
                    padding=1,  # zeropadding cause already circularly padded
                    groups=1,
                )[..., npad:-npad, npad:-npad, npad:-npad]
            )

            SymGw_Im[:, counter, ...] = (
                torch.exp(mu1)
                * 0.5
                * self.conv_op(
                    F.pad(inp_Im_list[kk][0].unsqueeze(0), self.pad, mode="circular"),
                    kernels_list_part2[kk][0],
                    bias=None,
                    padding=1,  # zeropadding cause already circularly padded
                    groups=1,
                )[..., npad:-npad, npad:-npad, npad:-npad]
                + torch.exp(mu2)
                * 0.5
                * self.conv_op(
                    F.pad(inp_Im_list[kk][1].unsqueeze(0), self.pad, mode="circular"),
                    kernels_list_part2[kk][1],
                    bias=None,
                    padding=1,  # zeropadding cause already circularly padded
                    groups=1,
                )[..., npad:-npad, npad:-npad, npad:-npad]
            )

            counter += 1

        SymGw = SymGw_Re + 1j * SymGw_Im
        return SymGw

    def apply_SymGH(self, z):

        mb, _, Nx, Ny, Nt = z.shape
        device = z.device

        z_Re = z.real
        z_Im = z.imag

        SymGHz_Re = torch.zeros(mb, 3, Nx, Ny, Nt).to(device)
        SymGHz_Im = torch.zeros(mb, 3, Nx, Ny, Nt).to(device)

        npad = self.npad

        kernel_x, kernel_y, kernel_t = (
            self.kernel_x.to(device),
            self.kernel_y.to(device),
            self.kernel_t.to(device),
        )
        kernels_list = [kernel_x, kernel_y, kernel_t]

        inp_Re_list = [
            (z_Re[:, 0, ...], z_Re[:, 3, ...], z_Re[:, 4, ...]),
            (z_Re[:, 1, ...], z_Re[:, 3, ...], z_Re[:, 5, ...]),
            (z_Re[:, 2, ...], z_Re[:, 4, ...], z_Re[:, 5, ...]),
        ]

        inp_Im_list = [
            (z_Im[:, 0, ...], z_Im[:, 3, ...], z_Im[:, 4, ...]),
            (z_Im[:, 1, ...], z_Im[:, 3, ...], z_Im[:, 5, ...]),
            (z_Im[:, 2, ...], z_Im[:, 4, ...], z_Im[:, 5, ...]),
        ]

        kernels_list = [
            (kernel_x, kernel_y, kernel_t),
            (kernel_y, kernel_x, kernel_t),
            (kernel_t, kernel_x, kernel_y),
        ]

        mu_list = [
            (self.mu_xy, self.mu_xy, self.mu_t),
            (self.mu_xy, self.mu_xy, self.mu_t),
            (self.mu_t, self.mu_xy, self.mu_xy),
        ]

        for kk in range(3):

            mu1, mu2, mu3 = mu_list[kk]
            SymGHz_Re[:, kk, ...] = (
                torch.exp(mu1)
                * self.conv_op_transpose(
                    F.pad(inp_Re_list[kk][0].unsqueeze(0), self.pad, mode="circular"),
                    kernels_list[kk][0],
                    bias=None,
                    padding=1,  # zeropadding cause already circularly padded
                    groups=1,
                )[..., npad:-npad, npad:-npad, npad:-npad]
                + torch.exp(mu2)
                * 0.5
                * self.conv_op_transpose(
                    F.pad(inp_Re_list[kk][1].unsqueeze(0), self.pad, mode="circular"),
                    kernels_list[kk][1],
                    bias=None,
                    padding=1,  # zeropadding cause already circularly padded
                    groups=1,
                )[..., npad:-npad, npad:-npad, npad:-npad]
                + torch.exp(mu3)
                * 0.5
                * self.conv_op_transpose(
                    F.pad(inp_Re_list[kk][2].unsqueeze(0), self.pad, mode="circular"),
                    kernels_list[kk][2],
                    bias=None,
                    padding=1,  # zeropadding cause already circularly padded
                    groups=1,
                )[..., npad:-npad, npad:-npad, npad:-npad]
            )

            SymGHz_Im[:, kk, ...] = (
                torch.exp(mu1)
                * self.conv_op_transpose(
                    F.pad(inp_Im_list[kk][0].unsqueeze(0), self.pad, mode="circular"),
                    kernels_list[kk][0],
                    bias=None,
                    padding=1,  # zeropadding cause already circularly padded
                    groups=1,
                )[..., npad:-npad, npad:-npad, npad:-npad]
                + torch.exp(mu2)
                * 0.5
                * self.conv_op_transpose(
                    F.pad(inp_Im_list[kk][1].unsqueeze(0), self.pad, mode="circular"),
                    kernels_list[kk][1],
                    bias=None,
                    padding=1,  # zeropadding cause already circularly padded
                    groups=1,
                )[..., npad:-npad, npad:-npad, npad:-npad]
                + torch.exp(mu3)
                * 0.5
                * self.conv_op_transpose(
                    F.pad(inp_Im_list[kk][2].unsqueeze(0), self.pad, mode="circular"),
                    kernels_list[kk][2],
                    bias=None,
                    padding=1,  # zeropadding cause already circularly padded
                    groups=1,
                )[..., npad:-npad, npad:-npad, npad:-npad]
            )

        SymGHz = SymGHz_Re + 1j * SymGHz_Im
        return SymGHz

    def apply_SymGHSymG(self, w):

        return self.apply_SymGH(self.apply_SymG(w))
