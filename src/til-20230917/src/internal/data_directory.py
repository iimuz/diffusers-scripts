"""データフォルダの設定を管理するモジュール."""
from pathlib import Path


def get_data_dir() -> Path:
    """dataフォルダの場所を絶対パスで返す.

    Returns
    -------
    Path
        `/path/to/project/root/data`
    """
    filepath = Path(__file__)
    project_root = filepath.parents[2]
    data_dir = project_root / "data"
    absolute_data_dir = data_dir.resolve()

    return absolute_data_dir


def get_interim_dir() -> Path:
    """data/interimフォルダの場所を絶対パスで返す.

    Returns
    -------
    Path
        `/path/to/project/root/data/interim`
    """
    data_dir = get_data_dir()
    interim_dir = data_dir / "interim"
    absolute_interim_diir = interim_dir.resolve()

    return absolute_interim_diir


def get_processed_dir() -> Path:
    """data/processedフォルダの場所を絶対パスで返す.

    Returns
    -------
    Path
        `/path/to/project/root/data/processed`
    """
    data_dir = get_data_dir()
    processed_dir = data_dir / "processed"
    absolute_processed_diir = processed_dir.resolve()

    return absolute_processed_diir
