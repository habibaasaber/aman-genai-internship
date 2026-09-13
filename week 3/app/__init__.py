"""
app/__init__.py
Automated Talent Pipeline Agent — package init.
"""

import asyncio

# ── Event Loop & Task Patch for Python 3.14 (fixes AnyIO NoEventLoopError & TypeError) ─
try:
    import sniffio
    import sniffio._impl

    _orig_sniffio_current = sniffio._impl.current_async_library

    def _patched_sniffio_current() -> str:
        try:
            return _orig_sniffio_current()
        except sniffio.AsyncLibraryNotFoundError:
            try:
                asyncio.get_running_loop()
                return "asyncio"
            except RuntimeError:
                raise

    sniffio._impl.current_async_library = _patched_sniffio_current
    sniffio.current_async_library = _patched_sniffio_current
except Exception:
    pass

try:
    import anyio._core._eventloop
    _orig_anyio_current = anyio._core._eventloop.current_async_library

    def _patched_anyio_current() -> str | None:
        res = _orig_anyio_current()
        if res is None:
            try:
                asyncio.get_running_loop()
                return "asyncio"
            except RuntimeError:
                pass
        return res

    anyio._core._eventloop.current_async_library = _patched_anyio_current
except Exception:
    pass

try:
    import anyio._backends._asyncio
    _orig_current_task = asyncio.current_task
    _dummy_tasks = {}

    def _patched_current_task(loop=None):
        task = _orig_current_task(loop)
        if task is not None:
            return task
        try:
            if loop is None:
                loop = asyncio.get_running_loop()
            if loop not in _dummy_tasks or _dummy_tasks[loop].done():
                async def _dummy():
                    await asyncio.sleep(3600)
                _dummy_tasks[loop] = loop.create_task(_dummy())
            return _dummy_tasks[loop]
        except RuntimeError:
            return None

    asyncio.current_task = _patched_current_task
    anyio._backends._asyncio.current_task = _patched_current_task
except Exception:
    pass
# ─────────────────────────────────────────────────────────────────────────────
