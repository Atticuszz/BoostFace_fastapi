from pathlib import Path

import yaml


def load_config(config_path: Path) -> dict:
    # utf-8
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)
