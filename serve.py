"""Start the local web API and serve the compiled React frontend."""

import uvicorn
from copy import deepcopy
from uvicorn.config import LOGGING_CONFIG


def main():
    """Bind the application to the IPv4 loopback interface."""
    log_config = deepcopy(LOGGING_CONFIG)
    log_config["loggers"]["api_server"] = {"handlers": ["default"], "level": "INFO", "propagate": False}
    uvicorn.run("api_server:app", host="127.0.0.1", port=8000, log_config=log_config)


if __name__ == "__main__":
    main()
