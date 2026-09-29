import torch
import torch.nn as nn
import torch.nn.functional as F


class MRIOperator(nn.Module):
    def __init__(self, dims):
        super().__init__()

        self.dims = dims

    def apply_mask(self, kdata, mask):
        """Apply mask in k-space domain.

        Args:
            kdata (torch.tensor): k-space data
            mask (torch.tensor): binary mask (fft-shifted)

        Returns:
            torch.tensor: masked k-space data
        """

        return mask * kdata         # A = S * F (this is just sampling mask S on a previously FFTed image)

    def apply_mask_adjoint(self, kdata, mask):
        """Apply adjoint of mask in k-space domain; the operation is the same
        as the forward, since the multiplication with the mask corresponds to a
        multiplication with a diagonal operator.

        A* = F* * S* (this is just S* on a previously FFTed image, but S* = S)
        Since you can't really unmask, the adjoint zeros out the masked values, the same way the forward does.
        Just with different input: kdata instead of image.

        Args:
            kdata (torch.tensor): k-space data
            mask (torch.tensor): binary mask (fft-shifted)

        Returns:
            torch.tensor: adjoint mask operation in k-space domain
        """

        return self.apply_mask(kdata, mask)

    def apply_forward(self, image, mask):
        """Apply forward operator.

        Args:
            image (torch.tensor): k-space data
            mask (torch.tensor): binary mask (fft-shifted)

        Returns:
            torch.tensor: adjoint operation k-space data
        """
        return self.apply_mask(torch.fft.fftn(image, dim=self.dims, norm="ortho"), mask)        # A = S * F (this is all of A, first FFT then mask)

    def apply_adjoint(self, kdata, mask):
        """Apply adjoint operator. In finite dimensions, this is simply the CONJUGATE TRANSPOSE of the forward operator.
        
        A* = F* * S* (this is all of A*, first adjoint mask then adjoint FFT)
           = F^-1 * S (since FFT operator with norm="ortho" is unitary and S* = S for real binary masks)

        Args:
            kdata (torch.tensor): k-space data
            mask (torch.tensor): binary mask (fft-shifted)

        Returns:
            torch.tensor: adjoint operation k-space data
        """
        return torch.fft.ifftn(
            self.apply_mask_adjoint(kdata, mask), dim=self.dims, norm="ortho"
        )