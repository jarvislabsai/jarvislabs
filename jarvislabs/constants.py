"""Static configuration — region URLs, GPU types, Europe isolation, defaults.

Single source of truth. When a region is added/removed or GPU types change,
only this file needs updating.
"""

# ── Regions ──────────────────────────────────────────────────────────────────

DEFAULT_REGION = "india-01"
INDIA_NOIDA_REGION = "india-noida-01"
EUROPE_REGION = "europe-01"

REGION_URLS: dict[str, str] = {
    "india-01": "https://backendprod.jarvislabs.net/",
    "india-noida-01": "https://backendn.jarvislabs.net/",
    "europe-01": "https://backendeu.jarvislabs.net/",
}

REGION_DISPLAY_CODES: dict[str, str] = {
    "india-01": "IN1",
    "india-noida-01": "IN2",
    "europe-01": "EU1",
}

REGION_CODE_TO_REGION: dict[str, str] = {code.lower(): region for region, code in REGION_DISPLAY_CODES.items()}

# ── Region routing ──────────────────────────────────────────────────────────

# Preferred order for auto-routing when user doesn't specify --region
REGION_PRIORITY: list[str] = [INDIA_NOIDA_REGION, DEFAULT_REGION, EUROPE_REGION]

# Hardcoded fallback when server_meta API is unreachable.
# GPUs not in this map default to INDIA_NOIDA_REGION (first in priority).
REGION_GPU_FALLBACK: dict[str, str] = {
    "H200": "europe-01",  # EU-exclusive
    "RTX5000": "india-01",  # IN1-exclusive
    "A5000Pro": "india-01",
    "A6000": "india-01",
    "RTX6000Ada": "india-01",
}

# ── Europe region constraints ────────────────────────────────────────────────

EUROPE_GPU_TYPES: frozenset[str] = frozenset({"H100", "H200"})
EUROPE_GPU_COUNTS: frozenset[int] = frozenset({1, 8})
EUROPE_MIN_STORAGE_GB = 100
VM_MIN_STORAGE_GB = 100
VM_SUPPORTED_REGIONS: frozenset[str] = frozenset({EUROPE_REGION, INDIA_NOIDA_REGION})
FILESYSTEM_REGIONS: frozenset[str] = frozenset({DEFAULT_REGION, INDIA_NOIDA_REGION})

# ── Timeouts & Polling ───────────────────────────────────────────────────────

DEFAULT_POLL_TIMEOUT_S = 600
POLL_INTERVAL_S = 3
FETCH_RETRY_INTERVAL_S = 2
HTTP_TIMEOUT_CONNECT_S = 10
HTTP_TIMEOUT_READ_S = 120
MAX_RETRIES = 3
RETRY_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})

# ── CLI Defaults ─────────────────────────────────────────────────────────────

DEFAULT_TEMPLATE = "pytorch"
DEFAULT_GPU_TYPE = "L4"
DEFAULT_NUM_GPUS = 1
DEFAULT_STORAGE_GB = 40  # auto-bumped to EUROPE_MIN_STORAGE_GB for europe
DEFAULT_INSTANCE_NAME = "Name me"

# ── GPU types (for validation / help text) ───────────────────────────────────

GPU_TYPES: frozenset[str] = frozenset(
    {
        "RTX5000",
        "A5000",
        "A5000Pro",
        "A6000",
        "A100",
        "A100-80GB",
        "RTX6000Ada",
        "H100",
        "H200",
        "L4",
    }
)
