"""Entry point for ``python -m unmochan.cli.main``."""

from unfoldlab.cli.main import app, main, qe_app, unfoldlab_main, vasp_app

__all__ = ["app", "main", "qe_app", "unfoldlab_main", "vasp_app"]


if __name__ == "__main__":
    raise SystemExit(main())
