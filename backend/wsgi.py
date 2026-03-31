"""Production WSGI entrypoint."""

from app import create_app
from app.config import Config


def _create_app():
    errors = Config.validate()
    if errors:
        raise RuntimeError("Invalid backend configuration: " + "; ".join(errors))
    return create_app()


app = _create_app()
