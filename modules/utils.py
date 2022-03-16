"""Module with utils for whole project"""


from pathlib import Path
from typing import Any, Dict, Union

import yaml


def load_yaml(yaml_path: Union[str, Path]) -> Dict[str, Any]:
    yaml_path = Path(yaml_path)

    with yaml_path.open(mode='r') as file:
        result = yaml.safe_load(file)

    return result
