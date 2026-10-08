"""Phase 1: check that the libraries are installed and the API key is loaded."""

import importlib.metadata as metadata
import os

import config  # importing config also loads .env


def main() -> None:
    import agentdojo  # noqa: F401  (the plan's "done when" check)
    import openai  # noqa: F401

    print("agentdojo", metadata.version("agentdojo"))
    print("openai   ", metadata.version("openai"))
    print(f"endpoint  {config.BASE_URL}")

    has_key = any(os.environ.get(name) for name in config.API_KEY_ENVS)
    print(f"{' / '.join(config.API_KEY_ENVS)} loaded:", has_key)  # never print the key itself
    if has_key:
        config.get_api_key()  # exits with a message if it is still the placeholder
    print("Phase 1 OK" if has_key else f"Phase 1: libraries OK, but set {config.API_KEY_ENVS[0]} in .env")


if __name__ == "__main__":
    main()
