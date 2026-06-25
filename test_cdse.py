import logging
from sentinelhub import SHConfig, BBox, CRS, DataCollection, MimeType, SentinelHubRequest
from backend.config import CDSE_CLIENT_ID, CDSE_CLIENT_SECRET

logging.basicConfig(level=logging.DEBUG)

config = SHConfig()
config.sh_client_id = CDSE_CLIENT_ID
config.sh_client_secret = CDSE_CLIENT_SECRET
config.sh_token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
config.sh_base_url = "https://sh.dataspace.copernicus.eu"

evalscript = """
//VERSION=3
function setup() {
    return { input: ["VV","dataMask"], output: { bands: 2, sampleType: "FLOAT32" } };
}
function evaluatePixel(s) {
    return [s.VV, s.dataMask];
}
"""

bbox = BBox(bbox=[77.0, 8.0, 77.05, 8.05], crs=CRS.WGS84)

# Try with default DataCollection
try:
    print("Testing with SENTINEL1_IW")
    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL1_IW,
                time_interval=("2024-06-12", "2024-06-18"),
                other_args={"dataFilter": {"acquisitionMode": "IW", "polarization": "DV"}}
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=(10, 10),
        config=config,
    )
    data = request.get_data()
    print("Success SENTINEL1_IW")
except Exception as e:
    print("Failed SENTINEL1_IW", e)

# Try with custom DataCollection
try:
    print("\nTesting with custom CDSE_S1_IW")
    CDSE_S1_IW = DataCollection.define(
        "sentinel-1-grd",
        api_id="sentinel-1-grd",
        service_url=config.sh_base_url,
        collection_type="SENTINEL1"
    )
    request2 = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=CDSE_S1_IW,
                time_interval=("2024-06-12", "2024-06-18"),
                other_args={"dataFilter": {"acquisitionMode": "IW", "polarization": "DV"}}
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=(10, 10),
        config=config,
    )
    data2 = request2.get_data()
    print("Success CDSE_S1_IW")
except Exception as e:
    print("Failed CDSE_S1_IW", e)
