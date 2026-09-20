"""Session-wide tiny synthetic project: small box, short span, fast to build."""
import shutil
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oceanembed.config import load_config  # noqa: E402


@pytest.fixture(scope="session")
def cfg(tmp_path_factory):
    root = tmp_path_factory.mktemp("proj")
    src = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "mvp.yaml").read_text())
    src["region"].update(lon_min=60.0, lon_max=62.0, lat_min=10.0, lat_max=12.0)
    src["time"].update(start="2021-01-01", end="2021-04-30",
                       train_end="2021-02-28", val_end="2021-03-31")
    src["train"].update(epochs=2, batch_size=512, max_train_samples=20000)
    (root / "configs").mkdir()
    cfg_path = root / "configs" / "mvp.yaml"
    cfg_path.write_text(yaml.safe_dump(src))
    return load_config(cfg_path)


@pytest.fixture(scope="session")
def processed(cfg):
    from oceanembed.data.download import output_path, synthesise
    from oceanembed.data.preprocess import build

    raw = output_path(cfg)
    raw.parent.mkdir(parents=True, exist_ok=True)
    synthesise(cfg, raw)
    return build(cfg)
