import sys
sys.path.insert(0, "/workspace/Projects/cultivated-learning")

from engine.inference import InferenceEngine
from core.memory_store import MemoryStore
from core.context_assembler import ContextAssembler
from core.interaction_loop import InteractionLoop
from core.consolidation import ConsolidationEngine
from core.cold_storage import ColdStorage

def init():
    engine = InferenceEngine("/workspace/models/results/Mistral-7B-Instruct-v0.3")
    engine.load()
    memory = MemoryStore(
        persist_dir="/workspace/Projects/cultivated-learning/data/memory_db",
        engine=engine
    )
    return engine, memory
