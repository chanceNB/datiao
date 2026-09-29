from pathlib import Path
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator


class TCNConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')
    baseline_id: Literal['r1-pen-tcn-v1'] = 'r1-pen-tcn-v1'
    baseline_version: Literal['1.0.0'] = '1.0.0'
    seed: Literal[20260929] = 20260929
    device: Literal['cpu'] = 'cpu'
    batch_size: Literal[32] = 32
    max_epochs: Literal[200] = 200
    learning_rate: Literal[0.001] = .001
    weight_decay: Literal[0.0001] = .0001
    early_stopping_patience: Literal[20] = 20
    min_delta: Literal[0.00001] = .00001
    gradient_clip_norm: Literal[1.0] = 1.
    threshold: Literal[0.5] = .5
    hidden_channels: tuple[int, ...] = (32,32)
    kernel_size: Literal[2] = 2
    dilations: tuple[int, ...] = (1,2)
    convs_per_block: Literal[2] = 2
    dropout: Literal[0.1] = .1

    @model_validator(mode='after')
    def fixed_architecture(self):
        if self.hidden_channels != (32,32) or self.dilations != (1,2):
            raise ValueError('architecture is frozen')
        return self


def load_tcn_config(path: str | Path) -> TCNConfig:
    return TCNConfig.model_validate(json.loads(Path(path).read_text(encoding='utf-8')))
