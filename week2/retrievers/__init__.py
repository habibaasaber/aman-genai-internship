"""
retrievers/__init__.py
======================
Public API for the retrievers package.

Both pipeline retrievers are exported here so callers can do:
    from retrievers import NaiveRetriever, HybridRetriever
"""

from retrievers.naive import NaiveRetriever

__all__ = ["NaiveRetriever"]
