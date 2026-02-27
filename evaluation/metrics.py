import time
import json
import os
import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from core.memory_store import MemoryStore, MemoryUnit, MemoryType


class EvaluationMetrics:
    """Longitudinal evaluation framework for Cultivated Learning.
    
    Tracks six metrics:
    1. Retrieval Precision — were surfaced memories actually relevant?
    2. Directive Stability — churn rate in the active directive set
    3. Response Quality Delta — framework vs vanilla comparison scores
    4. Adaptation Evidence — asymmetric log of when memory helps vs hurts
    5. Compute Cost — tokens, time, memory ops per interaction
    6. Behavioral Drift — measurable change in response characteristics over time
    """

    def __init__(self, log_dir, memory_store=None):
        self.log_dir = log_dir
        self.memory = memory_store
        os.makedirs(log_dir, exist_ok=True)

        # Persistent metric stores
        self.retrieval_log = self._load_json("retrieval_precision.json", [])
        self.directive_log = self._load_json("directive_stability.json", [])
        self.quality_log = self._load_json("response_quality.json", [])
        self.adaptation_log = self._load_json("adaptation_evidence.json", [])
        self.compute_log = self._load_json("compute_cost.json", [])
        self.drift_log = self._load_json("behavioral_drift.json", [])

    # ==========================================================
    # 1. RETRIEVAL PRECISION
    # ==========================================================

    def score_retrieval(self, interaction_id, query, retrieved_memories,
                        relevance_scores):
        """Score how relevant retrieved memories were to the query.
        
        Args:
            interaction_id: which interaction this belongs to
            query: the user's message
            retrieved_memories: list of MemoryUnit objects that were surfaced
            relevance_scores: list of ints (0=irrelevant, 1=partial, 2=essential)
                              one per retrieved memory, same order
        """
        if not retrieved_memories:
            return None

        assert len(relevance_scores) == len(retrieved_memories), \
            "Must provide one relevance score per retrieved memory"

        total = len(relevance_scores)
        essential = sum(1 for s in relevance_scores if s == 2)
        partial = sum(1 for s in relevance_scores if s == 1)
        irrelevant = sum(1 for s in relevance_scores if s == 0)

        # Precision: weighted score (essential=1.0, partial=0.5, irrelevant=0.0)
        precision = (essential * 1.0 + partial * 0.5) / total

        entry = {
            "timestamp": time.time(),
            "interaction_id": interaction_id,
            "query": query[:200],
            "memories_surfaced": total,
            "essential": essential,
            "partial": partial,
            "irrelevant": irrelevant,
            "precision": round(precision, 3),
            "memory_details": [
                {
                    "content": m.content[:100],
                    "type": m.memory_type.value,
                    "salience": m.salience_score,
                    "score": s
                }
                for m, s in zip(retrieved_memories, relevance_scores)
            ],
        }

        self.retrieval_log.append(entry)
        self._save_json("retrieval_precision.json", self.retrieval_log)
        return precision

    def get_retrieval_trend(self, window=10):
        """Get precision trend over recent entries."""
        if not self.retrieval_log:
            return {"entries": 0, "message": "No retrieval data yet."}

        recent = self.retrieval_log[-window:]
        precisions = [e["precision"] for e in recent]
        irrelevant_rates = [e["irrelevant"] / e["memories_surfaced"] for e in recent]

        return {
            "entries": len(self.retrieval_log),
            "recent_window": len(recent),
            "avg_precision": round(sum(precisions) / len(precisions), 3),
            "min_precision": round(min(precisions), 3),
            "max_precision": round(max(precisions), 3),
            "avg_irrelevant_rate": round(sum(irrelevant_rates) / len(irrelevant_rates), 3),
            "trend": "improving" if len(precisions) > 1 and precisions[-1] > precisions[0] else
                     "declining" if len(precisions) > 1 and precisions[-1] < precisions[0] else
                     "stable",
        }

    # ==========================================================
    # 2. DIRECTIVE STABILITY
    # ==========================================================

    def snapshot_directives(self, interaction_count, directives):
        """Record the current directive set for stability tracking.
        
        Args:
            interaction_count: current interaction number
            directives: list of directive content strings
        """
        # Compare to last snapshot
        churn = 0
        if self.directive_log:
            last = set(self.directive_log[-1]["directives"])
            current = set(directives)
            added = current - last
            removed = last - current
            churn = len(added) + len(removed)

        entry = {
            "timestamp": time.time(),
            "interaction_count": interaction_count,
            "directive_count": len(directives),
            "directives": directives,
            "churn": churn,
        }

        self.directive_log.append(entry)
        self._save_json("directive_stability.json", self.directive_log)
        return churn

    def get_directive_stability(self, window=20):
        """Analyze directive churn over recent snapshots."""
        if len(self.directive_log) < 2:
            return {"snapshots": len(self.directive_log), "message": "Need 2+ snapshots."}

        recent = self.directive_log[-window:]
        churns = [e["churn"] for e in recent]
        counts = [e["directive_count"] for e in recent]

        avg_churn = sum(churns) / len(churns)
        assessment = (
            "thrashing" if avg_churn > 2.0 else
            "stagnant" if avg_churn == 0 and len(recent) > 5 else
            "healthy"
        )

        return {
            "snapshots": len(self.directive_log),
            "recent_window": len(recent),
            "avg_churn": round(avg_churn, 2),
            "max_churn": max(churns),
            "current_count": counts[-1],
            "assessment": assessment,
        }

    # ==========================================================
    # 3. RESPONSE QUALITY DELTA (A/B)
    # ==========================================================

    def log_quality_comparison(self, interaction_id, prompt,
                                framework_response, vanilla_response,
                                framework_score, vanilla_score, notes=""):
        """Log an A/B quality comparison between framework and vanilla responses.
        
        Args:
            framework_score: 1-10 subjective quality rating for framework response
            vanilla_score: 1-10 subjective quality rating for vanilla response
        """
        delta = framework_score - vanilla_score

        entry = {
            "timestamp": time.time(),
            "interaction_id": interaction_id,
            "prompt": prompt[:300],
            "framework_response": framework_response[:300],
            "vanilla_response": vanilla_response[:300],
            "framework_score": framework_score,
            "vanilla_score": vanilla_score,
            "delta": delta,
            "notes": notes,
        }

        self.quality_log.append(entry)
        self._save_json("response_quality.json", self.quality_log)
        return delta

    def get_quality_trend(self):
        """Get quality delta trend."""
        if not self.quality_log:
            return {"comparisons": 0, "message": "No A/B comparisons yet."}

        deltas = [e["delta"] for e in self.quality_log]
        fw_scores = [e["framework_score"] for e in self.quality_log]
        van_scores = [e["vanilla_score"] for e in self.quality_log]

        return {
            "comparisons": len(self.quality_log),
            "avg_delta": round(sum(deltas) / len(deltas), 2),
            "avg_framework": round(sum(fw_scores) / len(fw_scores), 2),
            "avg_vanilla": round(sum(van_scores) / len(van_scores), 2),
            "framework_wins": sum(1 for d in deltas if d > 0),
            "vanilla_wins": sum(1 for d in deltas if d < 0),
            "ties": sum(1 for d in deltas if d == 0),
        }

    # ==========================================================
    # 4. ADAPTATION EVIDENCE LOG
    # ==========================================================

    def log_adaptation(self, interaction_id, event_type, description,
                       memory_ids=None):
        """Log when memory helps or hurts a response.
        
        Args:
            event_type: "help" or "harm" 
            description: what happened and why
            memory_ids: which memories were involved
        """
        assert event_type in ("help", "harm"), "event_type must be 'help' or 'harm'"

        entry = {
            "timestamp": time.time(),
            "interaction_id": interaction_id,
            "event_type": event_type,
            "description": description,
            "memory_ids": memory_ids or [],
        }

        self.adaptation_log.append(entry)
        self._save_json("adaptation_evidence.json", self.adaptation_log)

    def get_adaptation_summary(self):
        """Get help vs harm rates."""
        if not self.adaptation_log:
            return {"events": 0, "message": "No adaptation events logged."}

        helps = [e for e in self.adaptation_log if e["event_type"] == "help"]
        harms = [e for e in self.adaptation_log if e["event_type"] == "harm"]

        return {
            "total_events": len(self.adaptation_log),
            "help_count": len(helps),
            "harm_count": len(harms),
            "harm_rate": round(len(harms) / len(self.adaptation_log), 3),
            "recent_5": [
                {"type": e["event_type"], "desc": e["description"][:80]}
                for e in self.adaptation_log[-5:]
            ],
        }

    # ==========================================================
    # 5. COMPUTE COST TRACKING
    # ==========================================================

    def log_compute(self, interaction_id, prompt_tokens, response_tokens,
                    elapsed_seconds, memory_ops=0, reflection_calls=0):
        """Track compute costs per interaction."""
        entry = {
            "timestamp": time.time(),
            "interaction_id": interaction_id,
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "total_tokens": prompt_tokens + response_tokens,
            "elapsed_seconds": round(elapsed_seconds, 2),
            "memory_ops": memory_ops,
            "reflection_calls": reflection_calls,
        }

        self.compute_log.append(entry)
        self._save_json("compute_cost.json", self.compute_log)

    def get_compute_summary(self, window=20):
        """Compute cost trends."""
        if not self.compute_log:
            return {"entries": 0, "message": "No compute data yet."}

        recent = self.compute_log[-window:]
        tokens = [e["total_tokens"] for e in recent]
        times = [e["elapsed_seconds"] for e in recent]

        return {
            "total_interactions": len(self.compute_log),
            "recent_window": len(recent),
            "avg_tokens": round(sum(tokens) / len(tokens)),
            "avg_time_seconds": round(sum(times) / len(times), 2),
            "total_tokens_all_time": sum(e["total_tokens"] for e in self.compute_log),
        }

    # ==========================================================
    # 6. BEHAVIORAL DRIFT TRACKING
    # ==========================================================

    def log_drift_sample(self, interaction_count, response,
                         avg_sentence_length=None, vocabulary_richness=None):
        """Track measurable response characteristics over time.
        
        Auto-computes metrics from response text if not provided.
        """
        sentences = [s.strip() for s in response.replace("!", ".").replace("?", ".").split(".") if s.strip()]
        words = response.split()

        entry = {
            "timestamp": time.time(),
            "interaction_count": interaction_count,
            "response_length_chars": len(response),
            "response_length_words": len(words),
            "sentence_count": len(sentences),
            "avg_sentence_length": round(sum(len(s.split()) for s in sentences) / max(len(sentences), 1), 1),
            "vocabulary_richness": round(len(set(w.lower() for w in words)) / max(len(words), 1), 3),
        }

        self.drift_log.append(entry)
        self._save_json("behavioral_drift.json", self.drift_log)

    def get_drift_trend(self, window=20):
        """Analyze behavioral drift over time."""
        if len(self.drift_log) < 3:
            return {"samples": len(self.drift_log), "message": "Need 3+ samples."}

        recent = self.drift_log[-window:]
        lengths = [e["response_length_words"] for e in recent]
        sent_lens = [e["avg_sentence_length"] for e in recent]
        vocab = [e["vocabulary_richness"] for e in recent]

        # Simple trend detection: compare first half to second half
        mid = len(recent) // 2
        first_half_len = sum(lengths[:mid]) / max(mid, 1)
        second_half_len = sum(lengths[mid:]) / max(len(lengths) - mid, 1)

        return {
            "samples": len(self.drift_log),
            "recent_window": len(recent),
            "avg_response_words": round(sum(lengths) / len(lengths), 1),
            "avg_sentence_length": round(sum(sent_lens) / len(sent_lens), 1),
            "avg_vocab_richness": round(sum(vocab) / len(vocab), 3),
            "length_trend": "shorter" if second_half_len < first_half_len * 0.9 else
                           "longer" if second_half_len > first_half_len * 1.1 else
                           "stable",
        }

    # ==========================================================
    # FULL REPORT
    # ==========================================================

    def full_report(self):
        """Generate comprehensive evaluation report."""
        return {
            "retrieval": self.get_retrieval_trend(),
            "directives": self.get_directive_stability(),
            "quality": self.get_quality_trend(),
            "adaptation": self.get_adaptation_summary(),
            "compute": self.get_compute_summary(),
            "drift": self.get_drift_trend(),
        }

    def print_report(self):
        """Print human-readable evaluation report."""
        report = self.full_report()
        lines = ["=" * 60, "  CULTIVATED LEARNING — EVALUATION REPORT", "=" * 60, ""]

        # Retrieval
        r = report["retrieval"]
        lines.append(f"RETRIEVAL PRECISION ({r.get('entries', 0)} entries)")
        if r.get("entries", 0) > 0:
            lines.append(f"  Avg precision: {r['avg_precision']}")
            lines.append(f"  Irrelevant rate: {r['avg_irrelevant_rate']}")
            lines.append(f"  Trend: {r['trend']}")
        else:
            lines.append(f"  {r.get('message', 'No data')}")
        lines.append("")

        # Directives
        d = report["directives"]
        lines.append(f"DIRECTIVE STABILITY ({d.get('snapshots', 0)} snapshots)")
        if d.get("snapshots", 0) >= 2:
            lines.append(f"  Avg churn: {d['avg_churn']}")
            lines.append(f"  Current count: {d['current_count']}")
            lines.append(f"  Assessment: {d['assessment']}")
        else:
            lines.append(f"  {d.get('message', 'No data')}")
        lines.append("")

        # Quality
        q = report["quality"]
        lines.append(f"RESPONSE QUALITY DELTA ({q.get('comparisons', 0)} comparisons)")
        if q.get("comparisons", 0) > 0:
            lines.append(f"  Avg delta: {q['avg_delta']:+.2f}")
            lines.append(f"  Framework wins: {q['framework_wins']}")
            lines.append(f"  Vanilla wins: {q['vanilla_wins']}")
        else:
            lines.append(f"  {q.get('message', 'No data')}")
        lines.append("")

        # Adaptation
        a = report["adaptation"]
        lines.append(f"ADAPTATION EVIDENCE ({a.get('total_events', 0)} events)")
        if a.get("total_events", 0) > 0:
            lines.append(f"  Helps: {a['help_count']}  Harms: {a['harm_count']}")
            lines.append(f"  Harm rate: {a['harm_rate']}")
        else:
            lines.append(f"  {a.get('message', 'No data')}")
        lines.append("")

        # Compute
        c = report["compute"]
        lines.append(f"COMPUTE COST ({c.get('total_interactions', 0)} interactions)")
        if c.get("total_interactions", 0) > 0:
            lines.append(f"  Avg tokens/interaction: {c['avg_tokens']}")
            lines.append(f"  Avg time: {c['avg_time_seconds']}s")
            lines.append(f"  Total tokens (all time): {c['total_tokens_all_time']:,}")
        else:
            lines.append(f"  {c.get('message', 'No data')}")
        lines.append("")

        # Drift
        dr = report["drift"]
        lines.append(f"BEHAVIORAL DRIFT ({dr.get('samples', 0)} samples)")
        if dr.get("samples", 0) >= 3:
            lines.append(f"  Avg response: {dr['avg_response_words']} words")
            lines.append(f"  Avg sentence: {dr['avg_sentence_length']} words")
            lines.append(f"  Vocab richness: {dr['avg_vocab_richness']}")
            lines.append(f"  Length trend: {dr['length_trend']}")
        else:
            lines.append(f"  {dr.get('message', 'No data')}")

        lines.append("")
        lines.append("=" * 60)
        print("\n".join(lines))

    # ==========================================================
    # PERSISTENCE
    # ==========================================================

    def _load_json(self, filename, default):
        path = os.path.join(self.log_dir, filename)
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return default

    def _save_json(self, filename, data):
        path = os.path.join(self.log_dir, filename)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
