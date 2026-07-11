"""Process-wide shared Playwright browser.

Every page fetch used to launch its own Chromium (~4-5s overhead each, dozens
of times per run) and two concurrent sync_playwright() owners in one thread
crash ('Sync API inside the asyncio loop'). One lazily-started browser, pages
handed out per fetch, torn down at exit, fixes both.
"""
from __future__ import annotations

import atexit

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

_pw = None
_browser = None


def _ensure_browser():
    global _pw, _browser
    if _browser is not None:
        try:
            if _browser.is_connected():
                return _browser
        except Exception:
            pass
        _browser = None
    if _pw is None:
        from playwright.sync_api import sync_playwright
        _pw = sync_playwright().start()
        atexit.register(shutdown)
    _browser = _pw.chromium.launch(headless=True)
    return _browser


def new_page():
    """A fresh page on the shared browser. Caller closes the page (not the browser)."""
    page = _ensure_browser().new_page()
    page.set_extra_http_headers({"User-Agent": USER_AGENT})
    return page


def shutdown():
    global _pw, _browser
    for closer in (lambda: _browser.close(), lambda: _pw.stop()):
        try:
            closer()
        except Exception:
            pass
    _pw = _browser = None
