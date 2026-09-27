"""Auth + data-collection setup for Sentinel Hub (OAuth client credentials).

Two providers use the same API but DIFFERENT auth servers. Mixing them is the
most common cause of `invalid_client`:
  sentinelhub -> https://services.sentinel-hub.com          (apps.sentinel-hub.com creds)
  cdse        -> https://sh.dataspace.copernicus.eu         (dataspace.copernicus.eu creds)
"""
import os

from dotenv import load_dotenv
from sentinelhub import DataCollection, SHConfig

CDSE_BASE_URL = "https://sh.dataspace.copernicus.eu"
CDSE_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"


def _clean(value):
    """Strip whitespace / stray quotes that sneak into .env files."""
    return (value or "").strip().strip('"').strip("'").strip()


def get_config():
    """Return (SHConfig, provider). Ignores any stale ~/.config/sentinelhub/config.toml."""
    load_dotenv(override=True)
    provider = _clean(os.getenv("SH_PROVIDER", "sentinelhub")).lower()
    client_id = _clean(os.getenv("SH_CLIENT_ID"))
    client_secret = _clean(os.getenv("SH_CLIENT_SECRET"))
    if not client_id or not client_secret:
        raise RuntimeError("SH_CLIENT_ID / SH_CLIENT_SECRET missing. Copy .env.example to .env and fill it in.")
    if provider not in {"sentinelhub", "cdse"}:
        raise RuntimeError(f"SH_PROVIDER must be 'sentinelhub' or 'cdse', got {provider!r}")

    config = SHConfig(use_defaults=True)  # do NOT inherit old saved credentials
    config.sh_client_id = client_id
    config.sh_client_secret = client_secret
    if provider == "cdse":
        config.sh_base_url = CDSE_BASE_URL
        config.sh_token_url = CDSE_TOKEN_URL
    return config, provider


def get_collection(provider):
    if provider == "cdse":
        return DataCollection.SENTINEL2_L2A.define_from("s2l2a_cdse", service_url=CDSE_BASE_URL)
    return DataCollection.SENTINEL2_L2A
