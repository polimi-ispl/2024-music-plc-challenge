import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from loss import MultiResolutionSTFTLoss
from metrics import mse, sdr


class DilatedResBlock(nn.Module):
    def __init__(self, input_channel: int, output_channel: int, kernel_size: int, alpha: float = 0.2):
        super().__init__()

        self.in_conv = nn.Conv1d(input_channel, output_channel, kernel_size, padding=kernel_size // 2)

        self.conv_1 = nn.Sequential(
            nn.Conv1d(input_channel, output_channel, kernel_size, padding=2 * (kernel_size - 1) // 2, dilation=2),
            nn.BatchNorm1d(output_channel),
            nn.LeakyReLU(alpha, inplace=True)
        )

        self.conv_2 = nn.Sequential(
            nn.Conv1d(output_channel, output_channel, kernel_size, padding=4 * (kernel_size - 1) // 2, dilation=4),
            nn.BatchNorm1d(output_channel),
            nn.LeakyReLU(alpha, inplace=True)
        )

    def forward(self, inputs):
        skip = self.in_conv(inputs)
        x = self.conv_1(inputs)
        x = self.conv_2(x)
        x = x + skip
        return x


class DownSample(nn.Module):
    def __init__(self, factor: int = 2):
        super().__init__()
        self.downsample = nn.MaxPool1d(factor, factor)

    def forward(self, inputs):
        return self.downsample(inputs)


class UpSample(nn.Module):
    def __init__(self, factor: int = 2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=factor, mode='linear', align_corners=True)

    def forward(self, inputs):
        return self.upsample(inputs)


class GLUBlock(nn.Module):
    def __init__(self, n_channels: int, dilation_rate: int):
        super().__init__()

        self.in_conv = nn.Sequential(
            nn.Conv1d(n_channels, n_channels // 2, kernel_size=1, dilation=1),
            nn.BatchNorm1d(n_channels // 2)
        )

        self.padding = nn.ConstantPad1d((int(dilation_rate * 10), 0), value=0.)

        self.conv_left = nn.Sequential(
            nn.PReLU(),
            self.padding,
            nn.Conv1d(n_channels // 2, n_channels // 2, kernel_size=11, dilation=dilation_rate),
            nn.BatchNorm1d(n_channels // 2)
        )

        self.conv_right = nn.Sequential(
            nn.PReLU(),
            self.padding,
            nn.Conv1d(n_channels // 2, n_channels // 2, kernel_size=11, dilation=dilation_rate),
            nn.BatchNorm1d(n_channels // 2),

        )

        self.out_conv = nn.Sequential(
            nn.Conv1d(n_channels // 2, n_channels, kernel_size=1, dilation=1),
            nn.BatchNorm1d(n_channels)
        )

        self.out_activ = nn.PReLU()

    def forward(self, inputs):
        x = self.in_conv(inputs)
        xl = self.conv_left(x)
        xr = self.conv_right(x)
        x = xl * torch.sigmoid(xr)
        x = self.out_conv(x)
        x = self.out_activ(x + inputs)
        return x


class Generator(nn.Module):
    def __init__(self, channels: int = 1, lite: bool = True):
        super().__init__()

        dim = 8 if lite else 16

        self.body = nn.Sequential(
            DilatedResBlock(channels, dim, 11),
            DownSample(),
            DilatedResBlock(dim, 2 * dim, 11),
            DownSample(),
            DilatedResBlock(2 * dim, 4 * dim, 11),
            DownSample(),
            DilatedResBlock(4 * dim, 8 * dim, 11),
            DownSample(),
            GLUBlock(dilation_rate=1, n_channels=8 * dim),
            GLUBlock(dilation_rate=2, n_channels=8 * dim),
            GLUBlock(dilation_rate=4, n_channels=8 * dim),
            GLUBlock(dilation_rate=8, n_channels=8 * dim),
            GLUBlock(dilation_rate=16, n_channels=8 * dim),
            GLUBlock(dilation_rate=32, n_channels=8 * dim),
            UpSample(),
            DilatedResBlock(8 * dim, 8 * dim, 7),
            UpSample(),
            DilatedResBlock(8 * dim, 4 * dim, 7),
            UpSample(),
            DilatedResBlock(4 * dim, 2 * dim, 7),
            UpSample(),
            DilatedResBlock(2 * dim, dim, 7)
        )

        self.last_conv = nn.Sequential(
            nn.ConvTranspose1d(dim, 1, 1),
            nn.Tanh(),
        )

    def forward(self, inputs):
        x = self.body(inputs)
        x = self.last_conv(x)
        return x


class HybridModel(pl.LightningModule):
    def __init__(self, channels: int, lite: bool, packet_dim: int, extra_pred_dim: int):
        super().__init__()
        self.kwargs = {'num_workers': 8, 'pin_memory': True} if torch.cuda.is_available() else {}
        self.pred_dim = packet_dim + extra_pred_dim
        self.generator = Generator(channels=channels, lite=lite)
        self.stft_loss = MultiResolutionSTFTLoss()
        self.lambda_t = 100.
        self.lambda_s = 1.

    def configure_optimizers(self):
        optimizer_g = torch.optim.RAdam(self.generator.parameters(), lr=1e-4, betas=(0.5, 0.9))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_g, patience=20)
        return {"optimizer": optimizer_g, "lr_scheduler": scheduler, "monitor": "packet_val_mse"}

    def forward(self, x):
        return self.generator(x)

    def training_step(self, batch, batch_idx):
        true, past, ar_data = batch

        pred = self.forward(past) + ar_data

        temporal_loss = F.l1_loss(pred, true)

        sc_loss, log_loss = self.stft_loss(y_pred=pred, y_true=true)
        spectral_loss = 0.5 * (sc_loss + log_loss)

        tot_loss = self.lambda_t * temporal_loss + self.lambda_s * spectral_loss

        self.log('tot_loss', tot_loss, prog_bar=True)
        self.log('temporal_loss', temporal_loss, prog_bar=True)
        self.log('spectral_loss', spectral_loss, prog_bar=True)
        self.log('sc_loss', sc_loss, prog_bar=False)
        self.log('log_mag_loss', log_loss, prog_bar=False)

        return tot_loss

    def validation_step(self, batch, batch_idx):
        true, past, ar_data = batch

        pred = self.forward(past) + ar_data

        val_mse = mse(y_pred=pred, y_true=true)
        packet_val_mse = mse(y_pred=pred[..., -self.pred_dim:], y_true=true[..., -self.pred_dim:])

        self.log('val_mse', val_mse)
        self.log('packet_val_mse', packet_val_mse)

        val_sdr = sdr(y_pred=pred, y_true=true)
        packet_val_sdr = sdr(y_pred=pred[..., -self.pred_dim:], y_true=true[..., -self.pred_dim:])

        self.log('val_sdr', val_sdr)
        self.log('packet_val_sdr', packet_val_sdr)

        return packet_val_mse

    def test_step(self, batch, batch_idx):
        true, past, ar_data = batch

        pred = self.forward(past) + ar_data

        test_mse = mse(y_pred=pred, y_true=true)
        packet_test_mse = mse(y_pred=pred[..., -self.pred_dim:], y_true=true[..., -self.pred_dim:])

        self.log('test_mse', test_mse)
        self.log('packet_test_mse', packet_test_mse)

        test_sdr = sdr(y_pred=pred, y_true=true)
        packet_test_sdr = sdr(y_pred=pred[..., -self.pred_dim:], y_true=true[..., -self.pred_dim:])

        self.log('test_sdr', test_sdr)
        self.log('packet_test_sdr', packet_test_sdr)

        return packet_test_mse
