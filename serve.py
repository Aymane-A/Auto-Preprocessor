"""Start the local web API and serve the compiled React frontend."""

import uvicorn


def main():
    """Bind the application to the IPv4 loopback interface."""
    uvicorn.run("api_server:app", host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
