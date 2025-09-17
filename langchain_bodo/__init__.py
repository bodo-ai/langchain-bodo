from importlib import metadata

from langchain_bodo.agent_toolkits import create_bodo_dataframes_agent

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "create_bodo_dataframes_agent"
]
