import os
from src.synthnova_assets import update

def get_synthnova_assets_directory() -> str:
    """Retrieves the SynthNova assets directory path.

    This method retrieves the SynthNova assets directory path from the environment variable SYNTHNOVA_ASSETS.
    If the environment variable is not set, it raises a ValueError.
    If the environment variable is set, it returns the path.

    Returns:
        str: The path to the SynthNova assets directory.
    """
    synthnova_assets_directory = os.getenv("SYNTHNOVA_ASSETS")
    if synthnova_assets_directory is None:
        raise ValueError(
            "SYNTHNOVA_ASSETS environment variable is not set. Have you run the asset pull script in README?"
        )
    return synthnova_assets_directory

def main():
    try:
        sn_assets_directory = get_synthnova_assets_directory()
        download_directory = os.path.join(sn_assets_directory, "download/skus")
        download_urls = [
            "https://galbot.oss-cn-beijing.aliyuncs.com/online/robot/property/object/2100/dcca2c92_dc1f_5004_b5fd_1d73dda58cf7.zip",
            "https://galbot.oss-cn-beijing.aliyuncs.com/online/robot/property/object/2090/4725e830_d3a1_5c5e_afdf_9ad608c1c6af.zip"
        ]
        update(download_dir=download_directory, urls=download_urls)
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
