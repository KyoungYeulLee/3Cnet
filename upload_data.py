"""
Script for programmatically uploading 3cnet data.
Meant for 3b internal use only.
"""

import json
import os
from pathlib import Path
import requests as req

upload_file = Path("/data1/isaac_dev/3Cnet_drive/data.tar.gz")

sandbox_deposition_url = "https://sandbox.zenodo.org/api/deposit/depositions"
deposition_url = "https://zenodo.org/api/deposit/depositions"

params = {"access_token": os.environ["ZENODO_SANDBOX_ACCESS_TOKEN"]}
headers = {"Content-Type": "application/json"}

create_res = req.post(
    sandbox_deposition_url, params=params, json={}, headers=headers,
)

deposition_id = create_res.json()["id"]
bucket_url = create_res.json()["links"]["bucket"]
print(f"deposition_id: {deposition_id}")
print(f"bucket_url: {bucket_url}")

with upload_file.open("rb") as handle:
    upload_res = req.put(
        f"{bucket_url}/{upload_file.name}", data=handle, params=params
    )

data = {
    "metadata": {
        "title": "3Cnet: Pathogenicity prediction of human variants using knowledge transfer with deep recurrent neural networks",
        "upload_type": "dataset",
        "description": "testing",
        "creators": [
            {"name": "Won, Dhong-gun", "affiliation": "3billion"},
            {"name": "Lee, Kyoungyeul", "affiliation": "3billion"},
        ],
        # TODO add "publication_date" field before final publish
        "access_right": "restricted",  # TODO switch to "open" before final publish
        "access_conditions": "Internal use only",  # TODO delete this key/val pair before final publish
        "license": "CC-BY-NC-4.0",
        "prereserve_doi": True,
        "keywords": [
            "recurrent neural networks",
            "protein sequence variants",
            "pathogenicity prediction",
        ],
        "related_identifiers": [
            {
                "relation": "isSupplementTo",
                "identifier": "https://doi.org/10.1101/2020.09.27.302927",
            },
            {
                "relation": "isAlternateIdentifier",
                "identifier": "https://github.com/KyoungYeulLee/3Cnet",
            },
        ],
    }
}

metadata_res = req.put(
    sandbox_deposition_url + f"/{deposition_id}",
    params=params,
    data=json.dumps(data),
    headers=headers,
)
