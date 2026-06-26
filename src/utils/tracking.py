from pathlib import Path

from ruamel.yaml import YAML


def get_dvc_hash_from_data_path(data_path_str: str) -> str:
    """
    Extracts the unique MD5 data hash from a DVC tracking file.
    Uses ruamel.yaml syntax to comply with your local environment rules.
    """
    data_path = Path(data_path_str)
    dvc_pointer_path = data_path.with_suffix(data_path.suffix + ".dvc")

    if not dvc_pointer_path.exists():
        print(f"Warning: DVC file not found at {dvc_pointer_path}")
        return "untracked"

    try:
        # 2. Initialize the parser as demanded by the error message
        yaml_parser = YAML(typ="safe", pure=True)

        with open(dvc_pointer_path, "r") as f:
            # 3. Load using the instantiated parser object
            dvc_meta = yaml_parser.load(f)

        # Look for 'md5' at the root level first (No-download S3 imports)
        if "md5" in dvc_meta and dvc_meta["md5"]:
            return dvc_meta["md5"]

        # Fallback for standard local DVC file structures
        if "outs" in dvc_meta and dvc_meta["outs"]:
            if "md5" in dvc_meta["outs"][0]:
                return dvc_meta["outs"][0]["md5"]

        print(f"Warning: No MD5 string value found inside {dvc_pointer_path}")
        return "hash_not_found"

    except Exception as e:
        print(f"Error parsing DVC file {dvc_pointer_path}: {e}")
        return "error_reading_hash"
