"""Thread-local shared Playwright browsers.

One browser PER THREAD, created lazily: the sync API forbids two owners in one
thread (the collision class this module killed), but separate threads with
separate playwright instances are safe — which unlocks parallel event-page
prefetch (a Sunday run discovers ~15 fresh tournaments; sequential fetches of
calendar+draw+schedule+results pages were the 40-minute runs).

Callers are unchanged: new_page()/new_context() hand out pages on the calling
thread's browser; close the page/context, never the browser.
"""
from __future__ import annotations

import atexit
import threading

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

_tls = threading.local()
_instances: list = []          # (pw, browser) pairs across all threads
_instances_lock = threading.Lock()
_atexit_registered = False


def _ensure_browser():
    global _atexit_registered
    browser = getattr(_tls, "browser", None)
    if browser is not None:
        try:
            if browser.is_connected():
                return browser
        except Exception:
            pass
        _tls.browser = None
    if getattr(_tls, "pw", None) is None:
        from playwright.sync_api import sync_playwright
        _tls.pw = sync_playwright().start()
        with _instances_lock:
            if not _atexit_registered:
                atexit.register(shutdown)
                _atexit_registered = True
    _tls.browser = _tls.pw.chromium.launch(headless=True)
    with _instances_lock:
        _instances.append((_tls.pw, _tls.browser))
    return _tls.browser


def new_page():
    """A fresh page on this thread's shared browser. Caller closes the page."""
    page = _ensure_browser().new_page()
    page.set_extra_http_headers({"User-Agent": USER_AGENT})
    return page


def new_context(**kwargs):
    """A fresh context on this thread's shared browser for callers that need
    their own viewport/UA/locale (e.g. Bovada). Caller closes the context."""
    return _ensure_browser().new_context(**kwargs)


def shutdown():
    with _instances_lock:
        pairs = list(_instances)
        _instances.clear()
    for pw, browser in pairs:
        for closer in (browser.close, pw.stop):
            try:
                closer()
            except Exception:
                pass
