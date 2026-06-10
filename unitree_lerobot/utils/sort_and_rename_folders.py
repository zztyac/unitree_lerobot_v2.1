"""
Script to validate, sort, and rename Unitree JSON episode folders.

python unitree_lerobot/utils/sort_and_rename_folders.py --data_dir $HOME/datasets/g1_grabcube_double_hand

运行删除并排序：

python unitree_lerobot/utils/sort_and_rename_folders.py \
--data-dir /mnt/data/zty/json_data/ccw_tmp

只检查不删除：

python unitree_lerobot/utils/sort_and_rename_folders.py \
--data-dir /mnt/data/zty/json_data/ccw_tmp \
--check-only
"""

import json
import os
import shutil
import tyro
import uuid
from pathlib import Path


def _get_sorted_episode_folders(data_dir: Path) -> list[Path]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Data path is not a directory: {data_dir}")

    folders = sorted([path for path in data_dir.iterdir() if path.is_dir()], key=lambda path: path.name)
    if not folders:
        raise FileNotFoundError(f"No episode folders found under {data_dir}")

    return folders


def _format_json_decode_error(json_path: Path, exc: json.JSONDecodeError) -> str:
    return f"{json_path}: {exc.msg} (line {exc.lineno}, column {exc.colno}, char {exc.pos})"


def check_json_data_validity(
    data_dir: Path,
    *,
    json_file: str = "data.json",
    allow_empty: bool = False,
    delete_invalid: bool = False,
) -> list[Path]:
    """Check each episode folder before any rename operation."""

    folders = _get_sorted_episode_folders(data_dir)
    valid_folders = []
    errors = []

    for folder in folders:
        json_path = folder / json_file
        if not json_path.exists():
            errors.append(f"{folder}: missing {json_file}")
            continue
        if not json_path.is_file():
            errors.append(f"{json_path}: not a regular file")
            continue

        try:
            with json_path.open(encoding="utf-8") as jsonf:
                episode_data = json.load(jsonf)
        except json.JSONDecodeError as exc:
            errors.append(_format_json_decode_error(json_path, exc))
            continue

        if not isinstance(episode_data, dict):
            errors.append(f"{json_path}: top-level JSON must be an object")
            continue

        frames = episode_data.get("data")
        if not isinstance(frames, list):
            errors.append(f"{json_path}: 'data' must be a list")
            continue
        if not allow_empty and not frames:
            errors.append(f"{json_path}: 'data' is empty")
            continue

        for frame_index, frame in enumerate(frames):
            if not isinstance(frame, dict):
                errors.append(f"{json_path}: data[{frame_index}] must be an object")
                break

            missing_keys = [key for key in ("states", "actions", "colors") if key not in frame]
            if missing_keys:
                errors.append(f"{json_path}: data[{frame_index}] missing keys: {', '.join(missing_keys)}")
                break
        else:
            valid_folders.append(folder)

    if errors:
        max_errors = 50
        details = "\n".join(f"  - {error}" for error in errors[:max_errors])
        if len(errors) > max_errors:
            details += f"\n  - ... and {len(errors) - max_errors} more errors"

        if not delete_invalid:
            raise ValueError(f"JSON data check failed for {len(errors)} episode folders:\n{details}")

        print(f"Deleting {len(errors)} invalid episode folders:\n{details}")
        invalid_folders = [folder for folder in folders if folder not in valid_folders]
        for folder in invalid_folders:
            shutil.rmtree(folder)

        if not valid_folders:
            raise ValueError(f"No valid episode folders left under {data_dir}")

    print(f"JSON data check passed: {len(valid_folders)} valid episode folders.")
    return valid_folders


def sort_and_rename_folders(
    data_dir: Path,
    *,
    json_file: str = "data.json",
    allow_empty: bool = False,
    check_only: bool = False,
) -> None:
    folders = check_json_data_validity(
        data_dir,
        json_file=json_file,
        allow_empty=allow_empty,
        delete_invalid=not check_only,
    )
    if check_only:
        print("Check only mode: folders were not renamed.")
        return

    for index in range(len(folders)):
        target_path = data_dir / f"episode_{index:04d}"
        if target_path.exists() and target_path not in folders:
            raise FileExistsError(f"Cannot rename folders because target path already exists: {target_path}")

    temp_mapping = {}

    # First, rename all folders to unique temporary names
    for folder in folders:
        temp_name = str(uuid.uuid4())
        original_path = folder
        temp_path = data_dir / temp_name
        os.rename(original_path, temp_path)
        temp_mapping[temp_path] = folder.name

    # Then, rename them to the final target names
    start_number = 0
    for temp_path in temp_mapping:
        new_folder_name = f"episode_{start_number:04d}"
        new_path = data_dir / new_folder_name
        os.rename(temp_path, new_path)
        start_number += 1

    print("The folders have been successfully renamed.")


if __name__ == "__main__":
    try:
        tyro.cli(sort_and_rename_folders)
    except (ValueError, FileNotFoundError, NotADirectoryError, FileExistsError) as exc:
        raise SystemExit(f"ERROR: {exc}")
