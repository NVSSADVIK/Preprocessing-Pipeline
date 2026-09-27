"""Run this FIRST. Fires a tiny 64x64 Process API request to verify your OAuth creds."""
from sentinelhub import CRS, BBox, MimeType, SentinelHubRequest

from s2prep.config import get_collection, get_config

EVALSCRIPT = """//VERSION=3
function setup() { return { input: ["B04"], output: { bands: 1, sampleType: "FLOAT32" } }; }
function evaluatePixel(s) { return [s.B04]; }
"""


def main():
    config, provider = get_config()
    print(f"provider={provider}  base_url={config.sh_base_url}  client_id={config.sh_client_id[:6]}...")
    request = SentinelHubRequest(
        evalscript=EVALSCRIPT,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=get_collection(provider), time_interval=("2025-11-01", "2026-02-28")
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=BBox((78.35, 17.38, 78.36, 17.39), crs=CRS.WGS84),
        size=(64, 64),
        config=config,
    )
    try:
        data = request.get_data()
    except Exception as exc:  # noqa: BLE001
        print(f"\nFAILED: {exc}\n")
        if "invalid_client" in str(exc).lower() or "invalid client" in str(exc).lower():
            print(
                "Invalid client checklist:\n"
                " 1. Provider mismatch: creds from dataspace.copernicus.eu need SH_PROVIDER=cdse;\n"
                "    creds from apps.sentinel-hub.com need SH_PROVIDER=sentinelhub.\n"
                " 2. Use an *OAuth client* (client id + secret), not the old 'instance id' / API key.\n"
                " 3. Secret is shown ONCE at creation - copy it immediately; regenerate if lost.\n"
                " 4. No spaces/quotes/newlines around values in .env.\n"
                " 5. Make sure no shell env vars (SH_CLIENT_ID...) override .env: run `env | grep SH_`.\n"
            )
        raise SystemExit(1)
    print(f"OK - got array {data[0].shape}, mean B04 reflectance = {data[0].mean():.4f}")


if __name__ == "__main__":
    main()
