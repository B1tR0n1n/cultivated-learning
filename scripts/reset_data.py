"""Reset all runtime data: memory collection, cold storage, interaction logs.

Empties the ChromaDB collection, clears cold_archive.json, and removes
interaction log files. Does NOT delete the database, collection, or
any source code or model files.

Run from inside the container:
    python /workspace/Projects/cultivated-learning-24b/scripts/reset_data.py
"""
import os
import json
import glob
import chromadb

BASE_DIR = os.environ.get("CL_BASE_DIR", "/workspace/Projects/cultivated-learning-24b")

MEMORY_DB   = os.path.join(BASE_DIR, "data/memory_db")
COLD_ARCHIVE = os.path.join(BASE_DIR, "data/cold_storage/cold_archive.json")
LOG_DIR     = os.path.join(BASE_DIR, "data/interaction_log")


# --- 1. ChromaDB collection ---

client = chromadb.PersistentClient(path=MEMORY_DB)
collection = client.get_or_create_collection(
    name="cultivated_memory",
    metadata={"hnsw:space": "cosine"},
)

total = collection.count()
print(f"Collection 'cultivated_memory': {total} documents")

if total > 0:
    # Delete in batches of 500 — ChromaDB can choke on very large single deletes
    all_ids = collection.get(include=[])["ids"]
    batch_size = 500
    deleted = 0
    for i in range(0, len(all_ids), batch_size):
        batch = all_ids[i : i + batch_size]
        collection.delete(ids=batch)
        deleted += len(batch)
        print(f"  Deleted {deleted}/{total}...")
    print(f"Collection cleared. Remaining: {collection.count()}")
else:
    print("  Collection already empty.")


# --- 2. Cold storage archive ---

os.makedirs(os.path.dirname(COLD_ARCHIVE), exist_ok=True)
with open(COLD_ARCHIVE, "w") as f:
    json.dump([], f)
print(f"Cold archive reset: {COLD_ARCHIVE}")


# --- 3. Interaction logs ---

log_files = glob.glob(os.path.join(LOG_DIR, "*.json"))
if log_files:
    for path in log_files:
        os.remove(path)
    print(f"Interaction logs cleared: {len(log_files)} files removed from {LOG_DIR}")
else:
    print(f"Interaction log already empty: {LOG_DIR}")


print("\nReset complete.")
