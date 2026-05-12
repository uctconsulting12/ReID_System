from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm


class SIDStore:
    def __init__(
        self,
        collection: str,
        dim: int,
        fresh: bool,
        max_embeddings_per_sid: int,
        db_path: str | None = None,
        cloud_url: str | None = None,
        cloud_api_key: str | None = None,
        wipe_collection: bool = False,
    ):
        self.collection = collection
        self.max_embeddings_per_sid = max_embeddings_per_sid

        if cloud_url:
            import os
            api_key = cloud_api_key or os.environ.get("QDRANT_API_KEY")
            print(f"[Qdrant] Connecting to cloud: {cloud_url} (collection={collection})")
            self.client = QdrantClient(
                url=cloud_url,
                api_key=api_key,
                check_compatibility=False,
                timeout=30,
            )
        else:
            path = Path(db_path)
            if fresh and path.exists():
                print(f"[Qdrant] Wiping existing DB: {db_path}")
                shutil.rmtree(path)
            print(f"[Qdrant] Opening local DB: {db_path} (collection={collection})")
            path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=db_path)

        if wipe_collection:
            self._drop_collection_if_exists(collection, dim)

        self.collection = self._resolve_collection(collection, dim)
        self._dim = int(dim)

        self._next_sid, self._counts = self._scan_existing()
        print(f"[Qdrant] Next SID: {self._next_sid} (existing SIDs: {len(self._counts)})")

    def reset(self) -> None:
        """Drop and recreate the active collection, then clear in-memory
        state. After this call the gallery is empty and the next enrolled
        person will be SID 1 again. Caller is responsible for ensuring no
        concurrent users are mid-write."""
        try:
            self.client.delete_collection(collection_name=self.collection)
        except Exception as exc:
            print(f"[Qdrant] reset: delete_collection({self.collection}) failed: {exc}")
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=qm.VectorParams(size=self._dim, distance=qm.Distance.COSINE),
        )
        self._next_sid = 1
        self._counts = {}
        print(f"[Qdrant] Collection '{self.collection}' reset (dim={self._dim})")

    def _drop_collection_if_exists(self, name: str, dim: int) -> None:
        """Delete just this collection (and its dim-suffixed sibling, if any).

        Used when the caller wants to reset *only* its own gallery without
        touching the rest of the local Qdrant directory — e.g. per-org reset
        in the people-counting flow when ``database.qdrant.keep_db`` is false.
        """
        existing = {c.name for c in self.client.get_collections().collections}
        for target in (name, f"{name}_d{dim}"):
            if target in existing:
                print(f"[Qdrant] Wiping existing collection: {target}")
                self.client.delete_collection(collection_name=target)

    def _existing_dim(self, name: str) -> int | None:
        try:
            info = self.client.get_collection(name)
        except Exception:
            return None
        vectors = info.config.params.vectors
        # Single (unnamed) vector → VectorParams; multi-vector → dict[str, VectorParams]
        if hasattr(vectors, "size"):
            return int(vectors.size)
        if isinstance(vectors, dict) and vectors:
            first = next(iter(vectors.values()))
            return int(getattr(first, "size", 0)) or None
        return None

    def _resolve_collection(self, requested: str, dim: int) -> str:
        existing = {c.name for c in self.client.get_collections().collections}
        suffixed = f"{requested}_d{dim}"

        if requested in existing:
            existing_dim = self._existing_dim(requested)
            if existing_dim == dim:
                print(f"[Qdrant] Using existing collection '{requested}' (dim={dim})")
                return requested
            print(
                f"[Qdrant] '{requested}' exists with dim={existing_dim}, "
                f"backend needs dim={dim}; switching to '{suffixed}'"
            )
            target = suffixed
        else:
            target = requested

        if target in existing:
            existing_dim = self._existing_dim(target)
            if existing_dim != dim:
                raise ValueError(
                    f"Qdrant collection '{target}' exists with dim={existing_dim}, "
                    f"but backend produces dim={dim}. Delete it or change "
                    f"database.qdrant.collection in the config."
                )
            print(f"[Qdrant] Using existing collection '{target}' (dim={dim})")
            return target

        self.client.create_collection(
            collection_name=target,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        )
        print(f"[Qdrant] Created collection '{target}' with dim={dim}")
        return target

    def _scan_existing(self) -> tuple[int, dict[int, int]]:
        max_sid = 0
        counts: dict[int, int] = {}
        offset = None
        scanned = 0
        print("[Qdrant] Scanning existing points...")
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                limit=512,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                sid = int(point.payload.get("sid", 0))
                if sid > max_sid:
                    max_sid = sid
                counts[sid] = counts.get(sid, 0) + 1
            scanned += len(points)
            if scanned and scanned % 5120 == 0:
                print(f"[Qdrant] Scanned {scanned} points...")
            if offset is None:
                break
        print(f"[Qdrant] Scan complete: {scanned} points across {len(counts)} SIDs")
        return max_sid + 1, counts

    def search_top1(self, embedding: np.ndarray) -> tuple[int | None, float]:
        if self._next_sid <= 1:
            return None, 0.0

        response = self.client.query_points(
            collection_name=self.collection,
            query=embedding.tolist(),
            limit=1,
            with_payload=True,
        )
        hits = response.points
        if not hits:
            return None, 0.0
        top = hits[0]
        return int(top.payload["sid"]), float(top.score)

    def append(self, sid: int, embedding: np.ndarray) -> None:
        if self._counts.get(sid, 0) >= self.max_embeddings_per_sid:
            return

        self.client.upsert(
            collection_name=self.collection,
            points=[
                qm.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding.tolist(),
                    payload={"sid": sid},
                )
            ],
        )
        self._counts[sid] = self._counts.get(sid, 0) + 1

    def new_sid(self, embedding: np.ndarray) -> int:
        sid = self._next_sid
        self._next_sid += 1
        self.append(sid, embedding)
        return sid

    def total_sids(self) -> int:
        return self._next_sid - 1

    def prune_outliers(self, sid: int, gap_thresh: float = 0.05) -> int:
        """Remove the single most-divergent embedding for this SID, if any.

        Embeddings are L2-normalized, so cosine similarity == dot product.
        For each stored embedding we compute its mean similarity to the rest
        (its "cohesion"). The embedding with the lowest cohesion is removed
        only when it sits at least `gap_thresh` below the median cohesion of
        the others — otherwise the gallery is left untouched.

        Returns the number of points deleted (0 or 1).
        """
        ids: list = []
        vectors: list[list[float]] = []
        offset = None
        while True:
            batch, offset = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=qm.Filter(
                    must=[qm.FieldCondition(key="sid", match=qm.MatchValue(value=sid))]
                ),
                limit=256,
                offset=offset,
                with_payload=False,
                with_vectors=True,
            )
            for p in batch:
                if p.vector is None:
                    continue
                ids.append(p.id)
                vectors.append(p.vector)
            if offset is None:
                break

        if len(vectors) < 3:
            return 0

        emb = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / np.maximum(norms, 1e-6)

        sim = emb @ emb.T
        np.fill_diagonal(sim, np.nan)
        cohesion = np.nanmean(sim, axis=1)

        worst_idx = int(np.argmin(cohesion))
        others_median = float(np.median(np.delete(cohesion, worst_idx)))
        if others_median - float(cohesion[worst_idx]) < gap_thresh:
            return 0

        self.client.delete(
            collection_name=self.collection,
            points_selector=qm.PointIdsList(points=[ids[worst_idx]]),
        )
        self._counts[sid] = max(0, self._counts.get(sid, 0) - 1)
        print(
            f"[Qdrant] SID {sid}: pruned 1 outlier embedding "
            f"(cohesion={cohesion[worst_idx]:.3f}, others' median={others_median:.3f})"
        )
        return 1

