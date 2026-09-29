import torch
import torch.nn as nn
import torch.nn.functional as F


from math import prod


class ImageToScalarNet(nn.Module):
    def __init__(
        self, image_size=(320, 320), input_channels=2, n_conv_layers=3, n_filters=16
    ):
        super(ImageToScalarNet, self).__init__()

        input_channels_list = [input_channels]
        output_channels_list = [n_filters]
        input_channels_list.extend([n_filters * 2**k for k in range(n_conv_layers - 1)])
        output_channels_list.extend(
            [2 * input_channels_list[k + 1] for k in range(n_conv_layers - 1)]
        )

        conv_layers = []
        for n_ch_in, n_ch_out in zip(input_channels_list, output_channels_list):
            conv_layers.extend(
                [
                    nn.Conv2d(n_ch_in, n_ch_out, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                ]
            )
        # CNN feature extractor
        self.conv_layers = nn.Sequential(*conv_layers)

        # Compute the flattened size after convolutions
        conv_output_size_y, conv_output_size_x = (
            torch.tensor(image_size) / 2**n_conv_layers
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(
                int(conv_output_size_y.item())
                * int(conv_output_size_x.item())
                * output_channels_list[-1],
                128,
            ),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def _get_conv_output_size(self, x):
        x = self.conv_layers(x)
        return x.numel() // x.shape[0]

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_layers(x)  # .squeeze(1)
        return x
