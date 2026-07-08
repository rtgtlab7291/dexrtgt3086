#!/usr/bin/env python3
"""Download required assets (robot URDFs, SMPL-X body models, sample motions).

Usage:
    uv run python examples/humanoid_retarget/download_assets.py
"""
import os
from pathlib import Path

import gdown


def download_folder(folder_url: str, output_path: str, quiet: bool = False) -> None:
    os.makedirs(output_path, exist_ok=True)

    print(f"Downloading from: {folder_url}")
    print(f"To: {output_path}")

    gdown.download_folder(
        url=folder_url,
        output=output_path,
        quiet=quiet,
        use_cookies=False
    )

    print(f"Successfully downloaded to {output_path}")


def main():
    assets = [
        {
            "name": "body_model",
            "url": "https://drive.google.com/drive/folders/1r1ObvZxmRH57ANAs8ZLRlpHz-14_QJz8?usp=drive_link",
            "output": "assets/body_model"
        },
        {
            "name": "unitree_urdf",
            "url": "https://drive.google.com/drive/folders/12VP0zEkVtzrAR9mmyiLIGvEYMcH4_RJf?usp=drive_link",
            "output": "assets/robot_description"
        },
        {
            "name": "booster_k1",
            "url": "https://drive.google.com/drive/folders/1x9Yc0NJeF4J65O4B-9LRQ2lEf3tOGaQS?usp=drive_link",
            "output": "assets/robot_description/booster_k1"
        },
        {
            "name": "fourier_gr3",
            "url": "https://drive.google.com/drive/folders/1pxCkrFrMn1S7GacwuPSzO8iaseWc8gkX?usp=drive_link",
            "output": "assets/robot_description/fourier_gr3"
        },
        {
            "name": "unitree_h1_2",
            "url": "https://drive.google.com/drive/folders/1lt9kkO3gJiyL-MMx_JSjaQgMo7nov_eU?usp=drive_link",
            "output": "assets/robot_description/unitree_h1_2"
        },
        {
            "name": "berkeley_humanoid_lite",
            "url": "https://drive.google.com/drive/folders/1hasIOPZDQGK4M7uaUKX7GyjsyEDKtrqZ?usp=drive_link",
            "output": "assets/robot_description/berkeley_humanoid_lite"
        },
    ]

    script_dir = Path(__file__).parent.absolute()
    os.chdir(script_dir)

    print("=" * 60)
    print("Downloading assets from Google Drive...")
    print("=" * 60)

    for asset in assets:
        print(f"\n[{asset['name']}]")
        download_folder(asset["url"], asset["output"])

    print("\n" + "=" * 60)
    print("All assets downloaded successfully!")


if __name__ == "__main__":
    main()
