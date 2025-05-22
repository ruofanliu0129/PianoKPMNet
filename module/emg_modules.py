import torch
from torch import nn
from torch.nn.functional import interpolate


POSE_SAMPLE_RATE = 60

class BaseEmgModule(nn.Module):
    """
    Follows the PoseModule construction from
    https://arxiv.org/abs/2412.02725.
    """

    def __init__(
        self,
        network: nn.Module,
        out_channels: int = 20,
    ):
        super().__init__()
        self.network = network
        self.out_channels = out_channels
        self.left_context = network.left_context
        self.right_context = network.right_context

    def forward(
        self, batch: dict[str, torch.Tensor], provide_initial_pos: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # all input should be (B, C, T)
        emg = batch["emg"]
        model_input = batch["model_input"]
        temb = batch["temb"]

        # Get initial position
        self.left_context = 0
        self.right_context = 0
        initial_pos = emg[..., self.left_context]
        if not provide_initial_pos:
            initial_pos = torch.zeros_like(initial_pos)

        # Generate prediction
        pred = self._predict_emg(model_input, initial_pos, temb)

        start = self.left_context
        stop = None if self.right_context == 0 else -self.right_context
        emg = emg[..., slice(start, stop)]

        return pred, emg
    
    def inference(
        self, batch: dict[str, torch.Tensor], provide_initial_pos: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, emg_channels = batch['model_input'].shape[0], 6
        # all input should be (B, C, T)
        model_input = batch["model_input"]
        temb = batch["temb"]

        # Get initial position
        self.left_context = 0
        self.right_context = 0
        initial_pos = torch.zeros(batch_size, emg_channels, device=model_input.device)

        if not provide_initial_pos:
            initial_pos = torch.zeros_like(initial_pos)

        # Generate prediction
        pred = self._predict_emg(model_input, initial_pos, temb)

        return pred

    def _predict_emg(self, emg: torch.Tensor, initial_pos: torch.Tensor):
        raise NotImplementedError

    def align_predictions(self, pred: torch.Tensor, n_time: int):
        """Temporally resamples predictions to match the length of targets."""
        return interpolate(pred, size=n_time, mode="linear")

    def align_mask(self, mask: torch.Tensor, n_time: int):
        """Temporally resample mask to match the length of targets."""
        mask = mask[:, None].to(torch.float32)
        aligned = interpolate(mask, size=n_time, mode="nearest")
        return aligned.squeeze(1).to(torch.bool)   
    


class EmgModule(BaseEmgModule):
    """
    Predicts EMG sequences, the initial pos is set to 0
    """

    def __init__(
        self,
        network: nn.Module,
        decoder: nn.Module,
        state_condition: bool = True,
        predict_vel: bool = False,
        rollout_freq: int = 1024,
    ):
        super().__init__(network)
        self.decoder = decoder
        self.state_condition = state_condition
        self.predict_vel = predict_vel
        self.rollout_freq = rollout_freq

    def _predict_emg(self, model_input: torch.Tensor, initial_pos: torch.Tensor, temb: torch.Tensor):

        features = self.network(model_input)  # BCT
        preds = [initial_pos] #[(B,emg_channels)]

        # Resample features to rollout frequency
        #------------------------only use when input pose-------------------
        self.left_context = 0
        self.right_context = 0
        seconds = (
            model_input.shape[-1] - self.left_context - self.right_context
        ) / POSE_SAMPLE_RATE
        n_time = round(seconds * self.rollout_freq)
        features = interpolate(features, n_time, mode="linear", align_corners=True)
        # ----------------------------------------------------------------------

        for t in range(features.shape[-1]):
            # Prepare decoder inputs
            inputs = features[:, :, t]
            if self.state_condition:
                inputs = torch.concat([inputs, preds[-1]], dim=-1)
            
            # Predict EMG
            pred = self.decoder(inputs)
            preds.append(pred)

        # Remove first pred, because it is the initial_pos (not a network prediction)
        return torch.stack(preds[1:], dim=-1)
    


class EmgModule_multimodal(BaseEmgModule):
    """
    Predicts EMG sequences, the initial pos is set to 0
    """

    def __init__(
        self,
        network: nn.Module,
        decoder: nn.Module,
        state_condition: bool = True,
        predict_vel: bool = False,
        rollout_freq: int = 1024,
    ):
        super().__init__(network)
        self.decoder = decoder
        self.state_condition = state_condition
        self.predict_vel = predict_vel
        self.rollout_freq = rollout_freq

    def _predict_emg(self, model_input: torch.Tensor, initial_pos: torch.Tensor, temb: torch.Tensor):
        features = self.network(model_input, temb)  # BCT

        preds = [initial_pos] #[(B,emg_channels)]

        # Resample features to rollout frequency
        #------------------------only use when input pose-------------------
        self.left_context = 0
        self.right_context = 0
        seconds = (
            model_input.shape[-1] - self.left_context - self.right_context
        ) / POSE_SAMPLE_RATE
        n_time = round(seconds * self.rollout_freq)
        features = interpolate(features, n_time, mode="linear", align_corners=True)
        # ----------------------------------------------------------------------


        for t in range(features.shape[-1]):
            # Prepare decoder inputs
            inputs = features[:, :, t]
            if self.state_condition:
                inputs = torch.concat([inputs, preds[-1]], dim=-1)
            
            # Predict EMG
            pred = self.decoder(inputs)
            if self.predict_vel:
                pred = pred + preds[-1]
            preds.append(pred)

        # Remove first pred, because it is the initial_pos (not a network prediction)
        return torch.stack(preds[1:], dim=-1)
    