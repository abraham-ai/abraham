import os
import logging
import math
from dotenv import load_dotenv
load_dotenv()


# ──────────  PARAMS  ──────────
DEBUG                 = os.getenv("DEBUG", "False").lower() == "true"
APP_NAME              = "abraham"
GENERATION_COUNT      = 8
MODEL_NAME            = "claude-sonnet-4-5"
FALLBACK_MODEL_NAME   = "gpt-4o"
MAX_PARALLEL_WORKERS  = 4
MAX_CREATION_RETRIES  = 3
DB                    = os.getenv("DB", "STAGE")


# ──────────  TIMING  ──────────
TIMEZONE              = "America/New_York"
GENESIS_TIME          = "10:00"          # When to start the daily tournament
UPDATE_INTERVAL       = 15              # Minutes between update cycles
DESTROY_N_UPDATES     = 2               # Run destroy every N update cycles
# CYCLE_CHECK_INTERVAL  = 15              # Minutes between orchestrator checks

# ──────────  CHAIN INFO  ──────────
PINATA_JWT = os.getenv("PINATA_JWT")
IPFS_BASE_URL = os.getenv("IPFS_BASE_URL")
IPFS_PREFIX = os.getenv("IPFS_PREFIX")
BASE_SEPOLIA_RPC = os.getenv("BASE_SEPOLIA_RPC")
ETH_SEPOLIA_RPC = os.getenv("ETH_SEPOLIA_RPC")
PRIVATE_KEY = os.getenv("ABRAHAM_PRIVATE_KEY")
CONTRACT_ADDRESS_TOURNAMENT = os.getenv("CONTRACT_ADDRESS_TOURNAMENT")
CONTRACT_ADDRESS_AUCTION = os.getenv("CONTRACT_ADDRESS_AUCTION")
CONTRACT_ADDRESS_COVENANT = os.getenv("CONTRACT_ADDRESS_COVENANT")
CONTRACT_ABI_TOURNAMENT = os.getenv("CONTRACT_ABI_TOURNAMENT")
CONTRACT_ABI_AUCTION = os.getenv("CONTRACT_ABI_AUCTION")
CONTRACT_ABI_COVENANT = os.getenv("CONTRACT_ABI_COVENANT")
# CHAIN_ID = int(os.getenv("CHAIN_ID"))
SUBGRAPH_URL = os.getenv("SUBGRAPH_URL")
ABRAHAM_ADDRESS = os.getenv("ABRAHAM_ADDRESS")
GAS_LIMIT_CREATE_SESSION = 1000000  # Increased from 500k to 1M as fallback
GAS_LIMIT_UPDATE_SESSION = 1000000  # Increased from 500k to 1M as fallback


# ---------- Timing helpers --------------
def _hhmm_to_minutes(hhmm: str) -> int:
    h, m = map(int, hhmm.split(":"))
    return h * 60 + m

GENESIS_MIN = _hhmm_to_minutes(GENESIS_TIME)


# ──────────  SETUP USER  ──────────
# Get user ID from Eve
USER_ID = os.getenv("USER_ID")
if not USER_ID:
    from eve.auth import get_user_id
    user = get_user_id()
    USER_ID = str(user.id)
if not USER_ID:
    raise ValueError("USER_ID environment variable is required")


# ──────────  LOGGING  ──────────
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger("abraham")
logger.setLevel(logging.INFO)

# Force short time format for abraham logger specifically
abraham_handler = logging.StreamHandler()
abraham_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
abraham_handler.setFormatter(abraham_formatter)
#logger.handlers = [abraham_handler]  # Replace any existing handlers

# Suppress dependency logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("ably").setLevel(logging.WARNING)

# Helper function for logging separators
def log_section(title: str, level: str = "info"):
    """Log a section separator with title."""
    separator = "=" * 80
    if level == "info":
        logger.info(f"\n{separator}\n{title}\n{separator}")
    elif level == "warning":
        logger.warning(f"\n{separator}\n{title}\n{separator}")
    elif level == "error":
        logger.error(f"\n{separator}\n{title}\n{separator}")
