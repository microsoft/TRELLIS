from typing import *
import torch
import torch.nn as nn
from .. import models


class Pipeline:
    """
    A base class for pipelines.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
    ):
        if models is None:
            return
        self.models = models
        for model in self.models.values():
            model.eval()

    @staticmethod
    def from_pretrained(path: str) -> "Pipeline":
        """
        Load a pretrained model.
        """
        import os
        import json
        print("==========================Loading pipeline from:", path)
        print(f"{path}/pipeline.json")
        is_local = os.path.exists(f"{path}/pipeline.json")

        if is_local:
            print("Loading from local")
            config_file = f"{path}/pipeline.json"
        else:
            from huggingface_hub import hf_hub_download
            proxies = {
                "http": "http://127.0.0.1:52260",
                "https": "http://127.0.0.1:52260",
            }
            print("Loading from hub")
            config_file = hf_hub_download(path, "pipeline.json",proxies=proxies)

        with open(config_file, 'r') as f:
            args = json.load(f)['args']

        _models = {
            k: models.from_pretrained(f"{path}/{v}")
            for k, v in args['models'].items()
        }

        new_pipeline = Pipeline(_models)
        new_pipeline._pretrained_args = args
        return new_pipeline

    @property
    def device(self) -> torch.device:
        for model in self.models.values():
            if hasattr(model, 'device'):
                return model.device
        for model in self.models.values():
            if hasattr(model, 'parameters'):
                return next(model.parameters()).device
        raise RuntimeError("No device found.")

    def to(self, device: torch.device) -> None:
        for model in self.models.values():
            model.to(device)

    def cuda(self) -> None:
        self.to(torch.device("cuda"))

    def cpu(self) -> None:
        self.to(torch.device("cpu"))
