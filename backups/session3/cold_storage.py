import time
import json
import os
import numpy as np
from core.memory_store import MemoryUnit, MemoryType


class ColdStorage:
    """Archive for memories that fade below the salience floor.
    Resurfaces archived memories when an unusually strong semantic match occurs."""

    def __init__(self, engine, memory_store, archive_dir, salience_floor=0.15, resurface_threshold=0.75):
        self.engine = engine
        self.memory = memory_store
        self.archive_dir = archive_dir
        self.salience_floor = salience_floor
        self.resurface_threshold = resurface_threshold
        os.makedirs(archive_dir, exist_ok=True)
        self.archive = self._load_archive()
        print(f"Cold storage initialized: {len(self.archive)} archived memories")

    def _load_archive(self):
        archive_path = os.path.join(self.archive_dir, "cold_archive.json")
        if os.path.exists(archive_path):
            with open(archive_path, "r") as f:
                return json.load(f)
        return []

    def _save_archive(self):
        archive_path = os.path.join(self.archive_dir, "cold_archive.json")
        with open(archive_path, "w") as f:
            json.dump(self.archive, f, indent=2)

    def archive_pass(self):
        all_data = self.memory.collection.get(include=["documents", "metadatas", "embeddings"])
        archived_count = 0
        ids_to_delete = []

        for i in range(len(all_data["ids"])):
            metadata = all_data["metadatas"][i]
            if metadata["salience_score"] < self.salience_floor:
                embedding = None
                if all_data["embeddings"] is not None:
                    embedding = [float(x) for x in all_data["embeddings"][i]]

                entry = {
                    "id": all_data["ids"][i],
                    "document": all_data["documents"][i],
                    "metadata": metadata,
                    "embedding": embedding,
                    "archived_at": time.time(),
                }
                self.archive.append(entry)
                ids_to_delete.append(all_data["ids"][i])
                archived_count += 1

        # Save archive BEFORE deleting from active memory
        if archived_count > 0:
            self._save_archive()
            self.memory.collection.delete(ids=ids_to_delete)

        print(f"Archive pass: {archived_count} memories moved to cold storage. "
              f"Total archived: {len(self.archive)}")
        return archived_count

    def resurface(self, query_text):
        if not self.archive:
            return []

        query_embedding = self.engine.get_embedding(query_text)
        resurfaced = []

        for entry in self.archive:
            if entry["embedding"] is None:
                continue

            query_vec = np.array(query_embedding)
            stored_vec = np.array(entry["embedding"])
            similarity = np.dot(query_vec, stored_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(stored_vec)
            )

            if similarity >= self.resurface_threshold:
                mem = MemoryUnit.from_chroma(
                    id=entry["id"],
                    document=entry["document"],
                    metadata=entry["metadata"],
                )
                mem.salience_score = 0.5
                mem.last_accessed = time.time()
                mem.tags.append("resurfaced")
                self.memory.store(mem)
                resurfaced.append(mem)

        if resurfaced:
            resurfaced_ids = {m.id for m in resurfaced}
            self.archive = [e for e in self.archive if e["id"] not in resurfaced_ids]
            self._save_archive()
            print(f"Resurfaced {len(resurfaced)} memories from cold storage!")

        return resurfaced

    def get_stats(self):
        return {
            "archived_count": len(self.archive),
            "salience_floor": self.salience_floor,
            "resurface_threshold": self.resurface_threshold,
        }
