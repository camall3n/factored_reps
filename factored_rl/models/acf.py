from typing import Tuple

import torch
from torch import Tensor
from torch.nn import Linear, ModuleList

from factored_rl import configs
from factored_rl.models.nnutils import Module, one_hot
from factored_rl.models import MLP, losses
from factored_rl.models.residualcnn import ResidualEncoderNet
from factored_rl.models.ae import BaseModel


class ACFModel(BaseModel):
    def __init__(self, input_shape: Tuple, n_actions: int, cfg: configs.Config):
        super().__init__(input_shape, n_actions, cfg)
        self.encoder = ResidualEncoderNet(
            input_shape=self.input_shape,
            cnn_blocks=cfg.model.residualcnn.cnn_blocks,
            depth=cfg.model.residualcnn.depth,
            min_resolution=cfg.model.residualcnn.min_resolution,
            cnn_activation=cfg.model.residualcnn.cnn_activation,
            mlp_layers=cfg.model.residualcnn.mlp_layers + [cfg.model.n_latent_dims],
            mlp_activation=cfg.model.residualcnn.mlp_activation,
        )

    def process_configs(self):
        super().process_configs()
