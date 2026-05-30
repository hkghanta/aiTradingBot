"""Pytest bootstrap.

In this environment two scikit-learn installs coexist: a working one in the
user site-packages and a broken system copy under ``/usr/lib/python3/dist-packages``.
Depending on how the interpreter is launched, the broken system copy can shadow
the working one and fail at import. Move the user site-packages to the front of
``sys.path`` so the working install always wins before any test imports run.
"""

from __future__ import annotations

import site
import sys


def _prefer_user_site() -> None:
    user_site = site.getusersitepackages()
    if user_site in sys.path:
        sys.path.remove(user_site)
    sys.path.insert(0, user_site)


_prefer_user_site()
