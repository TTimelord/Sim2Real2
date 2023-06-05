import os.path
from pathlib import Path

ASSET_DIR = os.path.join(Path(__file__).parent, "assets")
# ASSET_DIR = Path(__file__).parent / "assets"
AGENT_CONFIG_DIR = os.path.join(ASSET_DIR, "config_files", "agents")
# AGENT_CONFIG_DIR = ASSET_DIR / "config_files/agents"
DESCRIPTION_DIR = os.path.join(ASSET_DIR, "descriptions")
# DESCRIPTION_DIR = ASSET_DIR / "descriptions"
DIGITAL_TWIN_DIR = os.path.join(ASSET_DIR, "digital_twins")
# DIGITAL_TWIN_DIR = ASSET_DIR / "digital_twins"
DIGITAL_TWIN_CONFIG_DIR = os.path.join(ASSET_DIR, "config_files", "digital_twins")
# DIGITAL_TWIN_CONFIG_DIR = ASSET_DIR / "config_files/digital_twins"