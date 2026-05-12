"""
org_registry.py
===============
Per-org pool of shared ReID resources.

Goal: every camera belonging to the same ``org_id`` (regardless of which
WebSocket session opened it) sees the same SID gallery, so global IDs are
stable across cameras and across reconnects within an org.

Cross-org isolation is strict: the registry refuses to share a SIDStore
between different ``org_id`` values.

The pool is reference-counted: the runner calls ``acquire(org_id)`` on
session start and ``release(org_id)`` on stop. The underlying SIDStore is
created lazily on the first acquire and (optionally) torn down when the
last reference is released.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

from ..config import AppConfig
from ..reid.base import BaseReIDBackend
from ..reid.factory import create_reid_backend
from ..store import SIDStore

logger = logging.getLogger("people_counting.org_registry")


@dataclass
class OrgResources:
    org_id: int
    reid_backend: BaseReIDBackend
    store: SIDStore
    refcount: int = 0
    _lock: threading.RLock = field(default_factory=threading.RLock)
    # Cross-camera lifetime dwell: sid -> total seconds completed across all
    # cameras of this org. The in-progress visit on each camera is added on
    # top by EntryExitTracker.lifetime_dwell_sec().
    dwell_lifetime_sec: dict[str, float] = field(default_factory=dict)
    dwell_lock: threading.Lock = field(default_factory=threading.Lock)


class OrgRegistry:
    """Thread-safe map ``org_id -> OrgResources``.

    All cameras of the same org share one ``SIDStore`` (one Qdrant
    collection scoped to that org) and one ``BaseReIDBackend`` instance.
    """

    def __init__(self, config: AppConfig, *, dispose_on_zero: bool = False) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._orgs: dict[int, OrgResources] = {}
        # Orgs whose gallery has already been built at least once during this
        # process lifetime. Used to apply ``keep_db: false`` exactly once per
        # org (on first attach) instead of wiping on every reconnect.
        self._seen_orgs: set[int] = set()
        # If True, drop the gallery from memory once the last camera using
        # this org disconnects. Keep False so a brief disconnect doesn't
        # wipe the SID->embedding map for re-arriving people.
        self._dispose_on_zero = bool(dispose_on_zero)

    def _collection_name(self, org_id: int) -> str:
        return f"{self._config.database.qdrant.collection}__org{int(org_id)}"

    def _build(self, org_id: int) -> OrgResources:
        cfg = self._config

        reid = create_reid_backend(
            backend=cfg.runtime.reid_backend,
            onnx_path=cfg.models.reid_onnx_path,
            tensorrt_engine_path=cfg.models.reid_tensorrt_engine_path,
            fastreid_root=cfg.models.fastreid_root,
            fastreid_config=cfg.models.fastreid_config,
            fastreid_weights=cfg.models.fastreid_weights,
            fastreid_device=cfg.models.fastreid_device,
            use_grayscale=cfg.runtime.embed_grayscale,
        )

        q = cfg.database.qdrant
        # keep_db=false should reset only the org currently coming online —
        # never the whole local DB folder, and never on a reconnect that
        # would yank the gallery out from under a sibling camera. So we wipe
        # at most once per org per process lifetime, on first attach.
        first_attach = int(org_id) not in self._seen_orgs
        wipe = first_attach and not q.keep_db
        store = SIDStore(
            collection=self._collection_name(org_id),
            dim=reid.dim,
            fresh=False,   # never wipe the whole DB folder from this path
            wipe_collection=wipe,
            max_embeddings_per_sid=cfg.gating.max_embeddings_per_sid,
            db_path=q.local_path if q.mode == "local" else None,
            cloud_url=q.cloud_url if q.mode == "cloud" else None,
            cloud_api_key=q.cloud_api_key if q.mode == "cloud" else None,
        )
        self._seen_orgs.add(int(org_id))

        logger.info(
            "org=%s resources built (collection=%s, dim=%d, wiped=%s)",
            org_id, self._collection_name(org_id), reid.dim, wipe,
        )
        return OrgResources(org_id=int(org_id), reid_backend=reid, store=store)

    def acquire(self, org_id: int, *, reset_collection: bool = False) -> OrgResources:
        """Acquire shared resources for ``org_id`` and bump the refcount.

        When ``reset_collection`` is True, the org's Qdrant collection is
        dropped and recreated *before* the new session starts using it, but
        only when no other session is currently holding the resources
        (refcount == 0). If sibling sessions are active, the wipe is
        skipped and a warning is logged so we don't trash a concurrent run.
        """
        with self._lock:
            res = self._orgs.get(int(org_id))
            if res is None:
                res = self._build(int(org_id))
                self._orgs[int(org_id)] = res

            if reset_collection:
                if res.refcount == 0:
                    try:
                        res.store.reset()
                    except Exception:
                        logger.exception(
                            "org=%s reset_on_connect: gallery reset failed",
                            org_id,
                        )
                    else:
                        with res.dwell_lock:
                            res.dwell_lifetime_sec.clear()
                        logger.info(
                            "org=%s gallery reset on session connect "
                            "(reset_on_connect=true)", org_id,
                        )
                else:
                    logger.warning(
                        "org=%s reset_on_connect requested but %d sibling "
                        "session(s) are active; skipping wipe to avoid "
                        "trashing concurrent data",
                        org_id, res.refcount,
                    )

            res.refcount += 1
            return res

    def release(self, org_id: int) -> None:
        with self._lock:
            res = self._orgs.get(int(org_id))
            if res is None:
                return
            res.refcount = max(0, res.refcount - 1)
            if res.refcount == 0 and self._dispose_on_zero:
                self._orgs.pop(int(org_id), None)
                logger.info("org=%s resources disposed (refcount hit 0)", org_id)
