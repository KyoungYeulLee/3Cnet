import logging
from pathlib import Path
import subprocess
import time

from requests import get, Response
from tqdm import tqdm

ZENODO_BASE_URL = "https://zenodo.org"
CCCNET_DATA_ACCESSION_ID = 10212255

CCCNET_RECORD_URL = f"{ZENODO_BASE_URL}/api/records/{CCCNET_DATA_ACCESSION_ID}"

# Path to save downloaded archive
DESTINATION_PATH = Path(__file__).parent / "data.tar.gz"
N_MAX_ATTEMPTS = 5

DESTINATION_PATH.parent.mkdir(mode=0o744, parents=True, exist_ok=True)

logger = logging.getLogger()
streamhandler = logging.StreamHandler()
logger.setLevel(logging.INFO)

logger.addHandler(streamhandler)

logger.info(f"Downloaded file will be saved to {DESTINATION_PATH}.")
logger.info(f"Retrieving record details...")
res = get(url=CCCNET_RECORD_URL)
res.raise_for_status()
record_json = res.json()

cccnet_data_download_url = record_json["files"][0]["links"]["self"]
data_download_size = record_json["files"][0]["size"]

for attempt in range(1, N_MAX_ATTEMPTS + 1):
    try:
        logger.info(f"Initiating download attempt {attempt} of {N_MAX_ATTEMPTS}...")
        res: Response = get(url=cccnet_data_download_url, stream=True)
        res.raise_for_status()

        progbar = tqdm(
            desc="Downloading data for 3Cnet...",
            mininterval=0.5,
            total=data_download_size,
            unit_scale=True,
            unit="B",
        )
        with DESTINATION_PATH.open(mode="wb") as fh:
            for chunk in res.iter_content():
                fh.write(chunk)
                progbar.update(len(chunk))

        break

    except Exception as e:
        logger.error(f"Download attempt {attempt} failed due to {e}, retrying...")
        time.sleep(1.0)
        continue

logger.info("Extracting downloaded archive...")
subprocess.run(args=["tar", "-xzf", str(DESTINATION_PATH)], check=True)
