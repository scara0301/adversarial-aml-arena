"""One module per dashboard page, each exposing a single ``render()`` function.

Named ``views`` rather than ``pages`` deliberately: Streamlit auto-builds its
own sidebar navigation from any folder literally named ``pages`` next to the
running script, which would collide with this dashboard's own sidebar radio
in ``dashboard/app.py``.
"""
