"""OpenPI data transforms for OpenArm diagnostic data; no training defaults."""
import dataclasses

from openpi.training import config as openpi_config
from openpi import transforms

from openarm_pi_adapter import OpenArmInputs, OpenArmOutputs


@dataclasses.dataclass(frozen=True)
class OpenArmDataConfig(openpi_config.DataConfigFactory):
    """Use the same observation/action mapping for training and inference."""

    def create(self, assets_dirs, model_config):
        repack = transforms.Group(inputs=[transforms.RepackTransform({
            "observation/table_image": "table_image",
            "observation/front_image": "front_image",
            "observation/state": "state",
            "actions": "actions",
            "prompt": "prompt",
        })])
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack,
            data_transforms=transforms.Group(inputs=[OpenArmInputs()], outputs=[OpenArmOutputs()]),
            model_transforms=openpi_config.ModelTransformFactory()(model_config),
        )
