import torch
import numpy as np
import pytorch_lightning as pl
from copy import deepcopy
from ar_model import ARModel
from nn_model import HybridModel


class PARCnet:
    def __init__(self,
                 model_checkpoint: str,
                 packet_dim: int,
                 extra_pred_dim: int,
                 ar_order: int,
                 ar_diagonal_load: float,
                 ar_context_dim: int,
                 nn_context_dim: int,
                 nn_fade_dim: int,
                 device: str,
                 lite: bool,
                 ):

        # Store arguments
        self.packet_dim = packet_dim
        self.extra_dim = extra_pred_dim
        self.device = device

        # Define the prediction length, including the extra length
        self.pred_dim = packet_dim + extra_pred_dim

        # Define the AR and neural network contexts in sample
        self.ar_context_dim = ar_context_dim * packet_dim
        self.nn_context_dim = nn_context_dim * packet_dim

        # Define fade-in modulation vector (neural network contribution only)
        self.nn_fade_dim = nn_fade_dim
        self.nn_fade = np.linspace(0., 1., nn_fade_dim)

        # Define fade-in and fade-out modulation vectors
        self.fade_in = np.linspace(0., 1., extra_pred_dim)
        self.fade_out = np.linspace(1., 0., extra_pred_dim)

        # Instantiate the linear predictor
        self.ar_model = ARModel(ar_order, ar_diagonal_load)

        # Load the pretrained neural network
        print(f'Loading model checkpoint: {model_checkpoint.stem}.ckpt')

        self.neural_net = HybridModel.load_from_checkpoint(
            model_checkpoint,
            packet_dim=packet_dim,
            extra_pred_dim=extra_pred_dim,
            channels=1,
            lite=lite,
        ).to(self.device)

    def __call__(self, input_signal: np.ndarray, trace: np.ndarray, **kwargs) -> np.ndarray:
        # Neural estimator in eval mode
        self.neural_net.eval()

        # Instantiate the output signal
        output_signal = deepcopy(input_signal)
        output_signal = np.pad(output_signal, (0, self.extra_dim))

        # Initialize a flag keeping track of consecutive packet losses
        is_burst = False

        for i, is_lost in enumerate(trace):
            if is_lost:
                # Start index of the ith packet
                idx = i * self.packet_dim

                # AR model prediction
                ar_context = output_signal[max(0, idx - self.ar_context_dim):idx]
                ar_context = np.pad(ar_context, (self.ar_context_dim - len(ar_context), 0))
                ar_pred = self.ar_model.predict(valid=ar_context, steps=self.pred_dim)

                # NN model context
                nn_context = output_signal[max(0, idx - self.nn_context_dim): idx]
                nn_context = np.pad(nn_context, (self.nn_context_dim - len(nn_context), self.pred_dim))
                nn_context = torch.Tensor(nn_context[None, None, ...]).to(self.device)

                # NN model inference
                with torch.no_grad():
                    nn_pred = self.neural_net(nn_context)
                    nn_pred = nn_pred[..., -self.pred_dim:]
                    nn_pred = nn_pred.squeeze().cpu().numpy()

                # Apply fade-in to the neural network contribution (inbound fade-in)
                nn_pred[:self.nn_fade_dim] *= self.nn_fade

                # Combine the two predictions
                prediction = ar_pred + nn_pred

                # Cross-fade the compound prediction (outbound fade-out)
                prediction[-self.extra_dim:] *= self.fade_out

                if is_burst:
                    # Cross-fade the prediction in case of consecutive packet losses (inbound fade-in)
                    prediction[:self.extra_dim] *= self.fade_in

                # Cross-fade the output signal (outbound fade-in)
                output_signal[idx + self.packet_dim:idx + self.pred_dim] *= self.fade_in

                # Conceal lost packet
                output_signal[idx: idx + self.pred_dim] += prediction

                # Keep track of consecutive packet losses
                is_burst = True

            else:
                # Reset burst loss indicator
                is_burst = False

        # Remove extra samples at the end of the enhanced signal
        output_signal = output_signal[:len(input_signal)]

        return output_signal
