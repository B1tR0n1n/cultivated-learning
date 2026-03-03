import sys
sys.path.insert(0, "/workspace/Projects/cultivated-learning-24b")

from engine.inference import InferenceEngine
from core.memory_store import MemoryStore
from core.context_assembler import ContextAssembler
from core.interaction_loop import InteractionLoop
from core.consolidation import ConsolidationEngine
from core.cold_storage import ColdStorage
from evaluation.metrics import EvaluationMetrics

def init():
    engine = InferenceEngine("/workspace/models/results/Mistral-Small-24B-Instruct-2501-AWQ")
    engine.load()
    memory = MemoryStore(
        persist_dir="/workspace/Projects/cultivated-learning-24b/data/memory_db",
        engine=engine
    )
    return engine, memory

def init_full():
    """Initialize all subsystems including metrics."""
    engine, memory = init()
    
    assembler = ContextAssembler(engine=engine, memory_store=memory)
    
    metrics = EvaluationMetrics(
        log_dir="/workspace/Projects/cultivated-learning-24b/data/evaluation",
        memory_store=memory
    )
    
    loop = InteractionLoop(
        engine=engine,
        memory_store=memory,
        assembler=assembler,
        log_dir="/workspace/Projects/cultivated-learning-24b/data/interaction_log",
        metrics_dir="/workspace/Projects/cultivated-learning-24b/data/evaluation",
    )
    
    consolidator = ConsolidationEngine(engine, memory)
    
    cold = ColdStorage(
        engine=engine,
        memory_store=memory,
        archive_dir="/workspace/Projects/cultivated-learning-24b/data/cold_storage"
    )
    
    print(f"\n{'='*50}")
    print(f"  CULTIVATED LEARNING — ALL SYSTEMS READY")
    print(f"  Context window: {assembler.max_context} tokens")
    print(f"  Metrics: ENABLED")
    print(f"  Reflection: ENABLED")
    print(f"{'='*50}\n")
    
    return engine, memory, assembler, loop, metrics, consolidator, cold
