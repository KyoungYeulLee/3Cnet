"Script for downloading 3Cnet-related data files"

import argparse
import hashlib
from math import ceil
from pathlib import Path
import requests as req
from typing import Union

from tqdm import tqdm


def gen_download_info(args: argparse.Namespace) -> dict:
    """
    Given a Zenodo DOI or a record ID, retrieve dictionary of files available for download and their respective metadata.
    """
    if args.sandbox:
        prefix = "https://sandbox.zenodo.org/api/records/"
    else:
        prefix = "https://zenodo.org/api/records/"

    if args.record:
        if not args.record.isnumeric():
            raise ValueError("Record ID should be entirely numeric")
        record_url = prefix + args.record
    elif args.doi:
        tokens = args.doi.split(".")
        if "zenodo" not in tokens[0] or not tokens[-1].isnumeric():
            raise ValueError(
                f"Either not Zenodo DOI or invalid DOI format: {args.doi}"
            )
        record_url = prefix + tokens[-1]
    else:
        raise ValueError(
            "Either record ID or DOI attribute expected, but neither found."
        )

    res = req.get(record_url)

    if res.status_code != req.codes.ok:
        res.raise_for_status()

    dl_res = res.json()["files"]

    dl_info = dict()
    for item in dl_res:
        filename = item["key"]

        # size is in bytes
        dl_info[filename] = {
            "name": filename,
            "url": item["links"]["self"],
            "size": item["size"],
            "type": item["type"],
            "checksum": item["checksum"].split(":")[-1],
        }

    return dl_info


def download_file(
    dl_item: dict,
    save_dir: Union[str, Path],
    n_retries: int,
    skip_existing: bool = True,
    verify_checksum: bool = True,
):
    """
    Attempt to download a file from Zenodo.
    The `dl_item` parameter accepts one item from the `dl_info` dict returned in `gen_download_info`.
    """
    filename = dl_item["name"]
    save_path = Path(save_dir) / filename

    url = dl_item["url"]
    chunk_size = 1000  # 1000 Bytes = 1 Kilobyte

    print(f">>> Downloading {filename}...")

    if skip_existing and save_path.exists():
        print(f"{filename} already exists in {save_dir}; Download skipped.")
        return

    for attempt in range(1, n_retries + 1):
        try:
            res = req.get(url, stream=True)
            with save_path.open("wb") as handle:
                for chunk in tqdm(
                    res.iter_content(chunk_size=chunk_size),
                    total=ceil(dl_item["size"] / chunk_size),
                    unit="kB",
                ):
                    handle.write(chunk)
            break
        except Exception as e:
            print(
                f"Download interrupted: {filename} (cause: {e}, attempt {attempt} of {n_retries})"
            )
            continue

    if verify_checksum:
        print("Verifying checksum...")
        reference_checksum = dl_item["checksum"]
        with save_path.open("rb") as handle2:
            generated_checksum = hashlib.md5(handle2.read()).hexdigest()

        if reference_checksum == generated_checksum:
            print("Checksum OK")
            return
        else:
            raise RuntimeError(f"Checksum does not match for {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script for downloading files from Zenodo using records IDs or DOIs"
    )
    parser.add_argument(
        "-d",
        "--doi",
        type=str,
        default="",
        help="Example: 12.3456/zenodo.789012",
    )
    parser.add_argument(
        "-r", "--record", type=str, default="", help="Example: 246802"
    )
    parser.add_argument(
        "-n", "--retries", type=int, default=5, help="Max retry attempts"
    )
    parser.add_argument("-o", "--outdir", type=str, help="File save directory")
    parser.add_argument("-s", "--sandbox", default=False, action="store_true")
    parser.add_argument(
        "-c",
        "--verify",
        default=False,
        action="store_true",
        help="Verify md5 checksum after each download?",
    )
    args = parser.parse_args()

    dl_info = gen_download_info(args)

    print(f"Downloading {len(dl_info.keys())} files: ")
    for key, val in dl_info.items():
        download_file(
            dl_item=val,
            save_dir=args.outdir,
            n_retries=args.retries,
            skip_existing=False,
            verify_checksum=args.verify,
        )
