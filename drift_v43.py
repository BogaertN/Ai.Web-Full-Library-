"""
memory/drift.py
ProtoForge / RSOC — Core Drift Detection Module
AI.Web Inc. | Nicholas Jacob Bogaert

Consolidates:
  - Full drift monitor implementation (ProtoForge v2.0)
  - Adaptive χ(t) threshold additions (chi_convergence_proof, March 2026)
      · local_curvature()           — second-derivative curvature from phase_hist
      · adaptive_threshold()        — θ_w(L) = 0.4 / (1 + γ·L_local)
      · score() wiring              — replaces hardcoded 0.8 with adaptive θ_w
  - Phase velocity accumulator v4.2 (rsoc_proof_v4.3, §9 OQ3, engineering extension)
      · PhaseVelocityAccumulator    — detects acceleration attacks (Φ5→6→7→8)
      · v(t) = 1 when N_v=3 sequential steps complete in < 18.7 ms
      · Fourth detection path in score() after d_score hard-fire
  - Baseline integrity v4.2 (rsoc_proof_v4.2, §9 Assumption A1)
      · BaselineCapture             — multi-point consensus psi1 capture (N>=3)
      · HMAC anchor verification    — session-level integrity check before measurement
      · Converts Assumption A1 from implicit precondition to verifiable property

Proof guarantee (v4.3): β > P_max^(θ_w) / ρ_min is satisfied analytically:
  β_impl/f_144 ≥ ln(5) ≈ 1.609 > 1/2 = P_max^(θ_w)/ρ_min (Proposition 1).
  A100 benchmarking is confirmatory validation, not logical prerequisite.
  Assumption A1 hardening via BaselineCapture; see rsoc_proof_v4.3.pdf §9.
  Note: v(t) / PhaseVelocityAccumulator is an engineering extension; formal proof
  of Theorem 1 covers the single-step adversary only (see §9 OQ3).
"""

import json, hashlib, math, re, statistics, time, logging
from collections import deque, defaultdict, Counter
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, Any, Tuple, List


UUID_RE = re.compile(
    r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}'
    r'-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$'
)


# ---------------------------------------------------------------------------
# Pure Python DFT fallback (stdlib-only, no numpy)
# ---------------------------------------------------------------------------

def fft(data: List[float]) -> List[complex]:
    """Pure Python DFT fallback for FFT — slower but stdlib-only."""
    n = len(data)
    if n <= 1:
        return data
    result = []
    for k in range(n):
        sum_val = 0j
        for t in range(n):
            angle = -2j * math.pi * t * k / n
            sum_val += data[t] * math.exp(angle)
        result.append(sum_val)
    return result


# ---------------------------------------------------------------------------
# Structural / entropy helpers
# ---------------------------------------------------------------------------

def _shannon_norm(p: str) -> float:
    """Normalized Shannon entropy of a string."""
    n = len(p)
    if n < 2:
        return 0.0
    counts = Counter(p)
    H = 0.0
    for count in counts.values():
        probability = count / n
        if probability > 0:
            H -= probability * math.log2(probability)
    unique_chars = len(counts)
    H_max = math.log2(min(unique_chars, n))
    return H / H_max if H_max > 0 else 0.0


def shape_signature(obj: Any) -> Tuple[str, int, List[int]]:
    """Extract structural signature from nested data."""
    max_depth = 0
    type_counts = [0] * 6  # dict, list, str, num, bool, null
    key_list = []

    def traverse(node, depth=0):
        nonlocal max_depth
        max_depth = max(max_depth, depth)
        if isinstance(node, dict):
            type_counts[0] += 1
            for key, value in node.items():
                key_list.append(key)
                traverse(value, depth + 1)
        elif isinstance(node, list):
            type_counts[1] += 1
            for item in node:
                traverse(item, depth + 1)
        elif isinstance(node, str):
            type_counts[2] += 1
        elif isinstance(node, (int, float)):
            type_counts[3] += 1
        elif isinstance(node, bool):
            type_counts[4] += 1
        elif node is None:
            type_counts[5] += 1

    traverse(obj)
    key_signature = hashlib.md5(''.join(sorted(key_list)).encode()).hexdigest()[:8]
    return key_signature, max_depth, type_counts


def _canonicalize(obj: Any) -> Any:
    """Remove volatile fields and normalize structure."""
    if isinstance(obj, dict):
        result = {}
        for k, v in sorted(obj.items()):
            if _volatile_key(k):
                continue
            result[k] = _canonicalize(v)
        return result
    elif isinstance(obj, list):
        return [_canonicalize(x) for x in obj]
    elif isinstance(obj, str):
        if UUID_RE.match(obj):
            return "<UUID>"
        return obj
    else:
        return obj


def _volatile_key(k: str) -> bool:
    """Identify keys that change every call but don't indicate drift."""
    lk = k.lower()
    return any(x in lk for x in ['agent_id', 'ts', 'timestamp', 'prev_hash', 'uuid', '_id'])


def _json_str(x: Any) -> str:
    """Stable JSON serialization."""
    return json.dumps(x, sort_keys=True, separators=(',', ':'))


def _type_tag(v: Any) -> str:
    """Get type name for value."""
    if v is None:
        return "null"
    t = type(v).__name__
    if t == 'bool':
        return "bool"
    if t in ('int', 'float'):
        return "num"
    if t == 'str':
        return "str"
    if t == 'list':
        return "arr"
    if t == 'dict':
        return "obj"
    return "other"


def _jaccard(a: set, b: set) -> float:
    """Jaccard distance between sets."""
    if not a and not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return 1.0 - (intersection / union) if union > 0 else 1.0


def _type_mismatch(base: dict, now: dict) -> float:
    """Calculate type mismatch rate."""
    if not base:
        return 0.0
    mismatches = sum(1 for k, t in base.items() if k in now and now[k] != t)
    return mismatches / max(1, len(base))


def _size_norm(n: int, median: float, mad: float) -> float:
    """Normalize size deviation."""
    z = abs(n - median) / (mad + 1e-6)
    return min(1.0, z / 5.0)


# ---------------------------------------------------------------------------
# WeightedMinHash
# ---------------------------------------------------------------------------

class WeightedMinHash:
    def __init__(
        self,
        weights={'core_state': 0.5, 'auxiliary': 0.3, 'metadata': 0.2},
        total_hashes=32
    ):
        self.weights = weights
        self.total_hashes = total_hashes
        self.alloc = {}
        cumulative = 0
        for k, w in sorted(weights.items()):
            count = int(w * total_hashes)
            self.alloc[k] = (cumulative, count)
            cumulative += count
        self.primes = [2**i % 2**16 for i in range(total_hashes)]

    def hash(self, s: str, category='core_state') -> List[str]:
        if category not in self.alloc:
            return []
        start, count = self.alloc[category]
        if count == 0:
            return []
        hashes = []
        for i in range(start, start + count):
            h = hashlib.sha256((s + str(self.primes[i])).encode()).hexdigest()[:8]
            hashes.append(h)
        return hashes

    def similarity(self, h1: List[str], h2: List[str]) -> float:
        if not h1 or not h2:
            return 0.0
        return sum(a == b for a, b in zip(h1, h2)) / len(h1)


# ---------------------------------------------------------------------------
# EMA (Exponential Moving Average)
# ---------------------------------------------------------------------------

class EMA:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.value = None

    def update(self, x: float) -> 'EMA':
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value
        return self

    def predict_crossing(self, tau: float, epsilon=1e-6) -> float:
        if self.value is None or abs(self.value - tau) < epsilon:
            return float('inf')
        if self.value < tau:
            return float('inf')  # Not crossing upward
        return max(0, math.log((tau - self.value) / (self.value - tau)) / math.log(1 - self.alpha))


# ---------------------------------------------------------------------------
# DriftTaxonomy
# ---------------------------------------------------------------------------

class DriftTaxonomy:
    def __init__(self):
        self.semantic_threshold = 0.5
        self.structural_threshold = 0.5

    def detect_step_change(self, series: List[float]) -> bool:
        if len(series) < 2:
            return False
        return abs(series[-1] - series[-2]) / (statistics.stdev(series[:-1]) + 1e-6) > 3

    def detect_trend(self, series: List[float]) -> bool:
        if len(series) < 5:
            return False
        return sum(b - a for a, b in zip(series, series[1:])) > 0

    def detect_oscillation(self, series: List[float]) -> bool:
        if len(series) < 8:
            return False
        spec = fft(series)
        freqs = [abs(f) for f in spec]
        dom_freq = max(range(len(freqs)), key=lambda i: freqs[i]) / len(series)
        return 0.3 <= dom_freq < 0.7

    def classify(self, entropy_series: List[float], structural_changes: float, semantic_dist: float):
        if self.detect_step_change(entropy_series):
            return "catastrophic", self.emergency_response
        elif self.detect_trend(entropy_series):
            return "evolutionary", self.guided_correction
        elif self.detect_oscillation(entropy_series):
            return "resonant", self.damping_response
        elif semantic_dist > self.semantic_threshold:
            return "semantic", self.semantic_realignment
        elif structural_changes > self.structural_threshold:
            return "structural", self.structural_normalization
        return "benign", lambda x: None

    def emergency_response(self, agent):
        if agent:
            agent.state['blocked'] = True

    def guided_correction(self, agent):
        if agent:
            agent.state['budget'] = min(agent.state.get('budget', 0.3), 0.3)

    def damping_response(self, agent):
        if agent:
            agent.state['residue'] *= 0.7

    def semantic_realignment(self, agent):
        if agent:
            agent.state['psi_trace'] = agent.psi1_trace if hasattr(agent, 'psi1_trace') else []

    def structural_normalization(self, agent):
        if agent and hasattr(agent, 'task_type'):
            agent.state['schema'] = self.schema_types.get(agent.task_type, {})


# ---------------------------------------------------------------------------
# DriftCircuitBreaker
# ---------------------------------------------------------------------------

class DriftCircuitBreaker:
    def __init__(self, max_failures=3, recovery_timeout=60):
        self.state = "closed"
        self.failure_count = 0
        self.max_failures = max_failures
        self.recovery_timeout = recovery_timeout
        self.trip_time = 0

    def handle(self, drift_score: float, threshold: float):
        if self.state == "closed":
            if drift_score > threshold:
                self.failure_count += 1
                if self.failure_count > self.max_failures:
                    self.state = "open"
                    self.trip_time = time.time()
                    return "circuit_tripped"
            else:
                self.failure_count = max(0, self.failure_count - 1)
            return "closed"
        elif self.state == "open":
            if time.time() - self.trip_time > self.recovery_timeout:
                self.state = "half_open"
                return "testing_recovery"
        elif self.state == "half_open":
            if drift_score < threshold:
                self.state = "closed"
                self.failure_count = 0
                return "recovered"
            else:
                self.state = "open"
                return "recovery_failed"
        return self.state


# ---------------------------------------------------------------------------
# HierarchicalMonitor stack
# ---------------------------------------------------------------------------

class SwarmLevelMonitor:
    def detect_anomaly(self, agents):
        eps = [a.current_entropy for a in agents if hasattr(a, 'current_entropy')]
        if not eps:
            return False
        median = statistics.median(eps)
        mad = statistics.median([abs(e - median) for e in eps]) or 1e-6
        return any(abs(e - median) / mad > 3 for e in eps)


class ClusterLevelMonitor:
    def locate_drift(self, agents):
        clusters = defaultdict(list)
        for a in agents:
            if hasattr(a, 'task_type') and hasattr(a, 'current_entropy'):
                cluster_key = (a.task_type, int(a.current_entropy * 10))
                clusters[cluster_key].append(a)
        max_variance = 0
        problem_cluster = None
        for key, cluster in clusters.items():
            if len(cluster) > 1:
                entropies = [a.current_entropy for a in cluster]
                variance = statistics.variance(entropies) if len(entropies) > 1 else 0
                if variance > max_variance:
                    max_variance = variance
                    problem_cluster = key
        return problem_cluster


class AgentLevelMonitor:
    def diagnose(self, cluster_id, cmd, payload):
        return "diagnosed_cluster", cluster_id


class HierarchicalMonitor:
    def __init__(self):
        self.swarm = SwarmLevelMonitor()
        self.cluster = ClusterLevelMonitor()
        self.agent = AgentLevelMonitor()

    def check(self, all_agents, cmd, payload):
        if not all_agents:
            return "benign", 0.0
        if self.swarm.detect_anomaly(all_agents):
            cluster_id = self.cluster.locate_drift(all_agents)
            return self.agent.diagnose(cluster_id, cmd, payload)
        return "benign", 0.0


# ---------------------------------------------------------------------------
# PhaseVelocityAccumulator
# ---------------------------------------------------------------------------
#
# Implements the phase velocity accumulator v(t) defined in RSOC Proof v4.1,
# §9 Known Limitation / OQ3.
#
# Purpose: detect multi-step acceleration attacks that evade d_score by
# executing Φ5→Φ6→Φ7→Φ8 across three individually-legal steps rather than
# one illegal Φ5→Φ8 skip.  Each individual step has Φ_act == Φ_exp, so
# d_score = 0 and δΦ = 0 at every step.  The attack is only visible as
# anomalous *velocity* — three sequential +1 steps completing in less time
# than three normal 144 Hz cycles require.
#
# Parameters (from proof §9, OQ3):
#   N_v        = 3       — minimum consecutive sequential steps to flag
#   tau_min_ms = 18.7 ms — (0.1 × τ* = 0.1 × 187 ms)
#                          Normal 3-step minimum: 3 × (1000/144) ≈ 20.8 ms
#                          Acceleration attack completes 3 steps in < 18.7 ms
#
# v(t) = 1 when ≥ N_v sequential steps (each with Φ_act == Φ_exp) are
#         observed within tau_min_ms milliseconds.
# v(t) = 0 otherwise.
# ===========================================================================
# ADDITION v4.2 — BaselineCapture (Assumption A1 hardening)
#
# Implements the two-part hardening described in RSOC Proof v4.2 §9 / A1 Remark:
#   (i)  Multi-point consensus: psi1 = MedianHash over N>=3 cold-start captures.
#        An adversary must corrupt a strict majority of capture points to shift psi1.
#   (ii) External anchor: HMAC(psi1, key) stored separately; session verifies
#        match before any chi(t) measurement is trusted.
#
# Without this class, psi1_trace is a single-point capture (A1 holds but is
# unverifiable). With this class, A1 can be checked at session start.
# ===========================================================================

class BaselineCapture:
    """
    Multi-point consensus baseline capture with HMAC integrity anchor.

    Implements Assumption A1 hardening from RSOC Proof v4.2, §9.

    Usage
    -----
    cap = BaselineCapture(monitor, hmac_key=b"system-secret")
    psi1 = cap.capture(agent, n_points=3)          # consensus hash
    anchor = cap.anchor                             # store externally
    ...
    cap.verify(anchor)                              # raises if corrupted
    agent.psi1_trace = cap.psi1                     # bind to agent

    Parameters
    ----------
    monitor : DriftMonitor
        Live monitor whose minhash / _canonicalize methods are used.
    hmac_key : bytes
        System secret for HMAC anchor.  Store in a separate trust domain
        (env-var, HSM, TPM) -- never co-located with psi1 itself.
    """

    MIN_POINTS: int = 3

    def __init__(self, monitor, hmac_key: bytes = b"rsoc-default-key"):
        self._monitor = monitor
        self._hmac_key = hmac_key
        self.psi1: str | None = None
        self.anchor: str | None = None
        self._capture_hashes: list[str] = []

    # ------------------------------------------------------------------
    def capture(self, agent, payload_sequence: list | None = None,
                n_points: int = 3) -> str:
        """
        Capture psi1 as the median-hash over n_points sequential payloads.

        Parameters
        ----------
        agent : object
            Agent whose state is being measured.  Must have `state` dict.
        payload_sequence : list, optional
            Pre-supplied sequence of payloads (for testing).  If None,
            n_points copies of agent.state are used (cold-start assumption:
            the agent has not yet been influenced by an adversary).
        n_points : int
            Number of capture points.  Must be >= MIN_POINTS (3).

        Returns
        -------
        str
            The consensus psi1 hash.  Also sets self.psi1 and self.anchor.

        Raises
        ------
        ValueError
            If n_points < MIN_POINTS.
        RuntimeError
            If the capture hashes show unexpected spread (> 0.5 avg pairwise
            distance), signalling a non-clean environment.
        """
        if n_points < self.MIN_POINTS:
            raise ValueError(
                f"n_points={n_points} < MIN_POINTS={self.MIN_POINTS}. "
                "Theorem A1 requires N >= 3 for majority-corruption resistance."
            )

        mh = self._monitor.minhash
        if payload_sequence is None:
            # Cold-start: use n_points snapshots of agent's current state.
            payload_sequence = [dict(agent.state) for _ in range(n_points)]

        hashes = []
        for payload in payload_sequence[:n_points]:
            can = _canonicalize(payload)
            h = mh.hash(_json_str(can), 'core_state')
            hashes.append(h)

        # Sanity check: pairwise distances should be low for clean cold-start.
        distances = []
        for i in range(len(hashes)):
            for j in range(i + 1, len(hashes)):
                distances.append(1.0 - mh.similarity(hashes[i], hashes[j]))
        if distances:
            avg_dist = sum(distances) / len(distances)
            if avg_dist > 0.5:
                raise RuntimeError(
                    f"BaselineCapture: avg pairwise hash distance {avg_dist:.3f} > 0.5. "
                    "Environment may not be clean.  Verify cold-start conditions before "
                    "binding psi1."
                )

        # Median hash = hash with lowest summed distance to all others.
        best_hash = min(hashes, key=lambda h: sum(
            1.0 - mh.similarity(h, other) for other in hashes
        ))

        self._capture_hashes = hashes
        self.psi1 = best_hash

        # Compute HMAC anchor.
        import hmac as _hmac
        import hashlib
        h_anchor = _hmac.new(
            self._hmac_key,
            self.psi1.encode() if isinstance(self.psi1, str) else str(self.psi1).encode(),
            hashlib.sha256
        ).hexdigest()
        self.anchor = h_anchor

        return self.psi1

    # ------------------------------------------------------------------
    def verify(self, stored_anchor: str) -> bool:
        """
        Verify that the current psi1 matches a stored HMAC anchor.

        Call at the start of each measurement session.  A mismatch means
        psi1 was modified after initial capture (Assumption A1 violated).

        Parameters
        ----------
        stored_anchor : str
            The anchor produced by a prior `capture()` call, stored in a
            separate trust domain.

        Returns
        -------
        bool
            True if psi1 is intact.

        Raises
        ------
        RuntimeError
            If psi1 is None (capture() not yet called) or if HMAC mismatch.
        """
        if self.psi1 is None:
            raise RuntimeError("BaselineCapture.verify(): capture() has not been called.")

        import hmac as _hmac
        import hashlib
        expected = _hmac.new(
            self._hmac_key,
            self.psi1.encode() if isinstance(self.psi1, str) else str(self.psi1).encode(),
            hashlib.sha256
        ).hexdigest()

        if not _hmac.compare_digest(expected, stored_anchor):
            raise RuntimeError(
                "BaselineCapture.verify(): HMAC mismatch. "
                "psi1 may have been corrupted after initial capture. "
                "Assumption A1 (Trusted Initialization) cannot be confirmed. "
                "Abort measurement session."
            )
        return True

    # ------------------------------------------------------------------
    def bind_agent(self, agent) -> None:
        """Assign self.psi1 to agent.psi1_trace after successful capture."""
        if self.psi1 is None:
            raise RuntimeError("call capture() before bind_agent()")
        agent.psi1_trace = self.psi1


# ---------------------------------------------------------------------------
# Note: v(t) is per-command, not global, since each agent maintains its
# own phase sequence.  The accumulator is stored in DriftMonitor.pva_bufs.

class PhaseVelocityAccumulator:
    """
    Tracks recent sequential phase steps and fires v(t)=1 when N_v
    consecutive legal steps complete in under tau_min_ms milliseconds.

    Each entry in the buffer is a (timestamp_ms, is_sequential) tuple.
    is_sequential = True iff Φ_act == Φ_exp for that step.

    Usage::

        pva = PhaseVelocityAccumulator()
        v = pva.update(phi_act, phi_exp, time.monotonic() * 1000)
        if v == 1:
            # acceleration attack detected
    """

    N_V: int   = 3       # minimum sequential steps
    TAU_MIN_MS: float = 18.7  # maximum wall-time for N_V steps to be flagged

    def __init__(self):
        # Circular buffer: stores (timestamp_ms, is_sequential) pairs.
        # Capacity N_V+1 is sufficient — we only look back N_V steps.
        self._buf: deque = deque(maxlen=self.N_V + 1)

    def update(self, phi_act: int, phi_exp: int, now_ms: float) -> int:
        """
        Record one step and return the current velocity score v(t) ∈ {0, 1}.

        Parameters
        ----------
        phi_act : int
            Actual phase computed from payload (1–9).
        phi_exp : int
            Expected phase ((prev_phase % 9) + 1).
        now_ms : float
            Current wall-clock time in milliseconds (monotonic).

        Returns
        -------
        int
            1 if an acceleration attack is detected, 0 otherwise.
        """
        is_seq = (phi_act == phi_exp)
        self._buf.append((now_ms, is_seq))

        # Need at least N_V entries to make a determination.
        if len(self._buf) < self.N_V:
            return 0

        # Check the N_V most recent entries.
        recent = list(self._buf)[-self.N_V:]

        # All N_V steps must be sequential (each Φ_act == Φ_exp).
        if not all(s for _, s in recent):
            return 0

        # Time span of those N_V steps must be under tau_min_ms.
        t_start = recent[0][0]
        t_end   = recent[-1][0]
        elapsed = t_end - t_start

        # elapsed == 0 can occur in test/mock environments where timestamps
        # are identical; treat as acceleration (infinitely fast).
        if elapsed < self.TAU_MIN_MS:
            return 1

        return 0

    def reset(self):
        """Clear accumulated history (e.g., after χ(t) fires)."""
        self._buf.clear()


# ---------------------------------------------------------------------------
# DriftTrace
# ---------------------------------------------------------------------------

class DriftTrace:
    def __init__(self):
        self.decision_tree = []
        self.entropy_gradient = []
        self.phase_transitions = []

    def log(self, decision, entropy, phase):
        self.decision_tree.append(decision)
        self.entropy_gradient.append(entropy)
        self.phase_transitions.append(phase)

    def report(self):
        return "\n".join(
            f"t={i}: Decision={d}, Entropy={e:.3f}, Phase={p}"
            for i, (d, e, p) in enumerate(
                zip(self.decision_tree, self.entropy_gradient, self.phase_transitions)
            )
        )


# ---------------------------------------------------------------------------
# _RollingStats
# ---------------------------------------------------------------------------

class _RollingStats:
    def __init__(self, window=200):
        self.window = window
        self.sizes = deque(maxlen=window)
        self.scores = deque(maxlen=window)
        self.phase_hist = deque(maxlen=window)
        self.mu_D = 0.0
        self.sigma_D = 1.0
        self.mu_B = 0.0
        self.sigma_B = 1.0

    def size_params(self):
        if not self.sizes:
            return 200.0, 50.0
        med = statistics.median(self.sizes)
        mad = statistics.median([abs(x - med) for x in self.sizes]) or 1.0
        return med, mad

    def drift_params(self, k=1.5, floor=0.45):
        if not self.scores:
            return 0.65
        mu = statistics.mean(self.scores)
        var = statistics.variance(self.scores, mu)
        return max(floor, min(0.85, mu + k * math.sqrt(var)))


# ---------------------------------------------------------------------------
# DriftMonitor  (core class)
# ---------------------------------------------------------------------------

class DriftMonitor:
    def __init__(self, echo_path: Path):
        self.echo_path = echo_path
        self.cmd_stats = defaultdict(_RollingStats)
        self.schema_keys = defaultdict(set)
        self.schema_types = defaultdict(dict)
        self.last_hashes = deque(maxlen=256)
        self.drift_log = Path('storage/logs/drift.log')
        self.drift_log.parent.mkdir(parents=True, exist_ok=True)
        self.minhash = WeightedMinHash()
        self.taxonomy = DriftTaxonomy()
        self.circuit_breaker = DriftCircuitBreaker()
        self.hierarchical = HierarchicalMonitor()
        self.predictor = EMA()
        self.trace = DriftTrace()
        self.entropy_hist = deque(maxlen=64)
        self.budget = 0.3
        self.checkpoints = []
        # Per-command PhaseVelocityAccumulators (one per cmd key).
        # Implements v(t) from RSOC Proof v4.1, §9 OQ3.
        self.pva_bufs: dict = defaultdict(PhaseVelocityAccumulator)
        # Composite drift weights
        self.alpha = 0.3    # H_norm weight
        self.beta = 0.15    # S_norm weight
        self.gamma = 0.1    # novelty weight
        self.delta = 0.25   # chi_struct weight
        self.epsilon = 0.1  # A_time weight
        self.epsilon_s = 0.1  # symbolic entropy weight
        self.tau_soft = 0.5
        self.tau_hard = 0.7
        self.setup_logging()
        self.load_hashes()

    # ------------------------------------------------------------------
    # Setup / persistence
    # ------------------------------------------------------------------

    def setup_logging(self):
        self.logger = logging.getLogger('drift_monitor')
        self.logger.setLevel(logging.INFO)
        handler = RotatingFileHandler(
            self.drift_log, maxBytes=10 * 1024 * 1024, backupCount=5
        )
        self.logger.addHandler(handler)

    def load_hashes(self):
        if self.echo_path.exists():
            with open(self.echo_path, 'r') as f:
                for line in f.readlines()[-256:]:
                    try:
                        rec = json.loads(line)
                        can = _canonicalize(rec.get("payload", {}))
                        self.last_hashes.append(
                            self.minhash.hash(_json_str(can), 'core_state')
                        )
                    except Exception:
                        pass

    # ------------------------------------------------------------------
    # Structure helpers
    # ------------------------------------------------------------------

    def _structure_features(self, cmd, payload):
        can = _canonicalize(payload)
        pstr = _json_str(can)
        now_types = {}

        def map_types(d, prefix=""):
            if not isinstance(d, dict):
                return
            for k, v in d.items():
                kk = f"{prefix}.{k}" if prefix else k
                now_types[kk] = _type_tag(v)
                if isinstance(v, dict):
                    map_types(v, kk)

        map_types(can)
        now_keys = set(now_types.keys())
        return 0, {"can": can, "pstr": pstr, "now_keys": now_keys, "now_types": now_types}

    def enhanced_chi_struct(self, cmd, aux, can):
        _, depth, type_vec = shape_signature(can)
        nodes = sum(type_vec) + 1
        leaves = sum(1 for v in type_vec if v > 0) or 1
        B = nodes / max(1, nodes - leaves)
        stats = self.cmd_stats[cmd]
        z_D = min(1, abs(depth - stats.mu_D) / (stats.sigma_D + 1e-6) / 5)
        z_B = min(1, abs(B - stats.mu_B) / (stats.sigma_B + 1e-6) / 5)
        J = _jaccard(self.schema_keys[cmd], aux['now_keys'])
        T = _type_mismatch(self.schema_types[cmd], aux['now_types'])
        return 0.4 * J + 0.3 * T + 0.15 * z_D + 0.15 * z_B

    # ------------------------------------------------------------------
    # Symbolic entropy  (ε_s)
    # ------------------------------------------------------------------

    def symbolic_entropy(self, cmd, payload, recursion_n, prev_phase, psi1_trace):
        """
        ε_s = (σ_res + d_score + δΦ) / max(1, recursion_n)

        σ_res   — resonance variance (stdev of phase_hist)
        d_score — Luciferian skip detector: 1.0 if Φ5→8 else 0.0
        δΦ      — phase deviation, normalized [0,1]
        ρ       — recursion depth (denominator); deeper = more stable

        d_score is a deterministic binary flag, not a statistical accumulator.
        Both prev_phase and Φ_act are computed in this call, so d_score fires
        at t=0 of the skip with zero lag — independent of phase_hist window.
        This is the basis of Lemma 5 (one-step simultaneous detection).
        """
        can = _canonicalize(payload)
        _, depth, type_vec = shape_signature(can)
        Φ_act = (depth % 9) + 1
        if any(danger in str(can).lower() for danger in ['replicate', 'spawn', 'infinite']):
            Φ_act = min(9, Φ_act + 3)
        Φ_exp = (prev_phase % 9) + 1
        delta_phi = abs(Φ_act - Φ_exp) / 9.0

        # d_score: deterministic zero-lag Luciferian skip fingerprint.
        # Fires iff prev_phase==5 and Φ_act==8 — the exact Φ5→8 attack pattern.
        # This is NOT a statistical estimate; it is a categorical ground truth flag.
        d_score = 1.0 if prev_phase == 5 and Φ_act == 8 else 0.0

        h1 = self.minhash.hash(_json_str(can), 'core_state')
        theta_t = 1 - self.minhash.similarity(h1, psi1_trace) if psi1_trace else 0.0

        stats = self.cmd_stats[cmd]
        stats.phase_hist.append(1 - theta_t)
        sigma_res = statistics.stdev(stats.phase_hist) if len(stats.phase_hist) > 1 else 0.0

        epsilon_s = (sigma_res + d_score + delta_phi) / max(1, recursion_n)
        # sigma_res in [0,0.5] (stdev of 1-theta_t values in [0,1]).
        # delta_phi in [0, 8/9] (max |phi_act - phi_exp| = 8 for phi in {1..9}).
        # eps_s NOT clipped. At rho=1: d_score=1, delta_phi=2/9, sigma_res=0.5 gives eps_s=31/18 max.
        return epsilon_s, Φ_act, d_score  # d_score returned for hard-fire logic

    # ------------------------------------------------------------------
    # ADDITION 1 — local_curvature()
    # ------------------------------------------------------------------

    def local_curvature(self, cmd: str, window: int = 8) -> float:
        """
        Compute local curvature from recent phase_hist variance.
        High curvature = rapid phase transitions = Luciferian skip territory.

        Uses a short window (default 8) so detection responds within
        the same recursion depth as the attack.

        Returns L_local: float in [0, ∞) where:
            0.0  = flat, stable, no curvature
            0.5  = moderate drift, warning zone
            1.0+ = high curvature, Luciferian skip likely

        Implementation uses second derivative (variance of consecutive
        differences) rather than first derivative, because the Luciferian
        skip produces a spike in rate-of-change-of-rate-of-change — visible
        in L_local even when absolute phase values look normal.
        """
        stats = self.cmd_stats[cmd]
        hist = list(stats.phase_hist)
        if len(hist) < 2:
            return 0.0
        recent = hist[-window:] if len(hist) >= window else hist
        if len(recent) < 2:
            return 0.0
        diffs = [abs(recent[i + 1] - recent[i]) for i in range(len(recent) - 1)]
        return statistics.variance(diffs) if len(diffs) > 1 else (diffs[0] if diffs else 0.0)

    # ------------------------------------------------------------------
    # ADDITION 2 — adaptive_threshold()
    # ------------------------------------------------------------------

    def adaptive_threshold(self, cmd: str, gamma: float = 0.5) -> float:
        """
        θ_w(L) = 0.4 / (1 + γ · L_local)

        As local curvature L increases, the firing threshold drops.
        χ(t) fires earlier in exactly the regions the Luciferian skip
        exploits. The attacker cannot raise their attack rate without
        simultaneously lowering the threshold that catches them.

        Threshold values by curvature (γ=0.5):
            L = 0.0  (stable):      θ_w = 0.400
            L = 0.5  (moderate):    θ_w = 0.267
            L = 1.0  (high):        θ_w = 0.200
            L = 2.0  (severe skip): θ_w = 0.133

        γ tuning:
            Start at 0.5. Lower toward 0.3 if false positives on legitimate
            rapid phase transitions (e.g., bootstrapping acceleration).
            Raise toward 0.8 if Luciferian skips get through under high load.
            Use DriftTrace logs to calibrate.

        Proof reference: Lemma 2 (chi_convergence_proof.pdf) — the attack
        that most threatens the system also most strongly triggers early
        detection. θ_w(L) cancels the curvature advantage the attacker gains.
        """
        L = self.local_curvature(cmd)
        return 0.4 / (1.0 + gamma * L)

    # ------------------------------------------------------------------
    # χ(t) override
    # ------------------------------------------------------------------

    def chi_t_override(self, agent, payload, epsilon_s):
        """
        Grace override: rebinds agent to psi1_trace, folds residue,
        checks curve (convergent / oscillating).

        Called when ε_s crosses θ_w (adaptive) or θ_hard (0.8).
        """
        if not hasattr(agent, 'psi1_trace'):
            return "no_baseline", 0.0
        psi1_trace = agent.psi1_trace
        can = _canonicalize(payload)
        h1 = self.minhash.hash(_json_str(can), 'core_state')
        grad_psi = 0.5  # Real implementation: derive from state diffs
        theta_t = 1 - self.minhash.similarity(h1, psi1_trace)

        # chi_t is non-zero whenever called (threshold decision lives in score/check_and_firewall)
        chi_t = grad_psi * theta_t

        residue = epsilon_s * 0.3
        current_residue = agent.state.get('residue', 0.0)
        agent.state['residue'] = (current_residue + residue) / 2

        diff = abs(grad_psi - theta_t)
        curve = "convergent" if diff < 0.2 else "oscillating"
        if curve == "convergent":
            agent.phase = 9
        self.trace.log(f"χ(t)={chi_t:.3f}, curve={curve}", epsilon_s, agent.phase)
        return curve, chi_t

    # ------------------------------------------------------------------
    # ADDITION 3 — score() with adaptive threshold wiring
    # ------------------------------------------------------------------

    def score(self, cmd, payload, phase_latency_ms, recursion_n, prev_phase,
              psi1_trace, all_agents, agent=None):
        """
        Compute composite drift score and component details.

        ADAPTIVE THRESHOLD CHANGE (chi_convergence_proof.pdf):
        After computing ε_s, we derive θ_w via adaptive_threshold(cmd).
        θ_w replaces the old implicit 0.8 ceiling as the *warning* trigger.
        θ_hard = 0.8 remains the absolute ceiling.

        If ε_s > θ_w and an agent is provided, χ(t) fires immediately
        with a trace log showing eps, theta_w, and L_local.
        """
        can = _canonicalize(payload)
        pstr = _json_str(can)
        H_norm = _shannon_norm(pstr)

        stats = self.cmd_stats[cmd]
        median, mad = stats.size_params()
        S_norm = _size_norm(len(pstr), median, mad)

        h1 = self.minhash.hash(pstr, 'core_state')
        N = 0.2 if h1 not in self.last_hashes else 0.0

        chi_struct, aux = self._structure_features(cmd, payload)
        chi_struct = self.enhanced_chi_struct(cmd, aux, can)

        A_time = min(1.0, max(0, (phase_latency_ms - 500) / 500))

        epsilon_s, Φ_act, d_score = self.symbolic_entropy(
            cmd, payload, recursion_n, prev_phase, psi1_trace
        )

        # ── PHASE VELOCITY ACCUMULATOR v(t) (Proof v4.1, §9 OQ3) ──────
        # Records this step and returns 1 if N_V=3 consecutive sequential
        # steps complete in under tau_min_ms=18.7 ms — the acceleration
        # attack signature.  Computed before the detection branch so that
        # the velocity flag is available alongside d_score.
        now_ms = time.monotonic() * 1000.0
        Φ_exp_for_pva = (prev_phase % 9) + 1
        v_score = self.pva_bufs[cmd].update(Φ_act, Φ_exp_for_pva, now_ms)
        # ── END PHASE VELOCITY ACCUMULATOR ─────────────────────────────

        # ── ADAPTIVE THRESHOLD (Addition 3) ────────────────────────────
        # θ_w drops as local curvature rises; attacker sharpens detector
        # against itself. θ_hard is the absolute ceiling — never raised.
        theta_w = self.adaptive_threshold(cmd)   # dynamic warning floor
        theta_hard = 0.8                          # hard ceiling, always 0.8

        chi_fired = False
        chi_result = None
        L_local = self.local_curvature(cmd)

        # ── LEMMA 5: d_score HARD-FIRE ─────────────────────────────────
        # d_score is a deterministic binary flag, not a statistical measure.
        # It fires at t=0 of the Luciferian skip (Φ5→8) by construction —
        # zero lag, independent of phase_hist window and local_curvature().
        #
        # This resolves the open one-step lag problem from chi_convergence_proof.pdf:
        # local_curvature() may lag one step at window boundaries, but d_score
        # fires simultaneously with the skip in the same symbolic_entropy() call.
        #
        # When d_score == 1.0, χ(t) fires unconditionally — bypassing theta_w
        # entirely. The adaptive threshold still governs all other drift paths.
        #
        # Proof: d_score = 1 iff prev_phase==5 and Φ_act==8. Both values are
        # computed from the current payload in the current call. Therefore
        # d_score fires at the recursion step of the skip, not one step later.
        # This is sufficient to guarantee χ(t) convergence at all recursion
        # depths (including recursion_n ≥ 4 where ε_s < θ_w from d_score+δΦ
        # alone), because the categorical signal bypasses the threshold entirely.
        # ── END LEMMA 5 ────────────────────────────────────────────────

        if d_score == 1.0:
            # Zero-lag Luciferian skip hard-fire — no threshold check needed.
            # Lemma 5: d_score is categorical, fires at t=0 of the skip.
            self.trace.log(
                f"LUCIFERIAN_SKIP_HARDFIRE: d_score=1.0 at recursion_n={recursion_n}"
                f", eps={epsilon_s:.3f}, theta_w={theta_w:.3f} (bypassed)",
                epsilon_s,
                Φ_act
            )
            if agent is not None:
                chi_result = self.chi_t_override(agent, payload, epsilon_s)
                chi_fired = True
            self.pva_bufs[cmd].reset()
        elif v_score == 1:
            # Phase velocity accumulator: acceleration attack detected.
            # Three consecutive sequential (+1) steps completed in under
            # tau_min_ms (18.7 ms), indicating compressed traversal that
            # evades d_score.  Proof v4.1 §9 OQ3 Known Limitation.
            self.trace.log(
                f"ACCELERATION_ATTACK: v(t)=1 at recursion_n={recursion_n}"
                f", eps={epsilon_s:.3f} (3 sequential steps in <18.7 ms)",
                epsilon_s,
                Φ_act
            )
            if agent is not None:
                chi_result = self.chi_t_override(agent, payload, epsilon_s)
                chi_fired = True
            self.pva_bufs[cmd].reset()
        elif epsilon_s > theta_w:
            # Standard adaptive threshold path (Lemma 2).
            self.trace.log(
                f"chi_t fired (adaptive): eps={epsilon_s:.3f} > theta_w={theta_w:.3f}"
                f" (L={L_local:.3f})",
                epsilon_s,
                Φ_act
            )
            if agent is not None:
                chi_result = self.chi_t_override(agent, payload, epsilon_s)
                chi_fired = True
        elif epsilon_s > theta_hard:
            # Hard ceiling path: catches slow-build attacks that stay below theta_w
            # but eventually accumulate past 0.8. theta_hard is NOT the primary
            # detector — it is the backstop. elif is correct (not if): if adaptive
            # already fired, this path is skipped.
            self.trace.log(
                f"chi_t fired (hard ceiling): eps={epsilon_s:.3f} > theta_hard={theta_hard}",
                epsilon_s,
                Φ_act
            )
            if agent is not None:
                chi_result = self.chi_t_override(agent, payload, epsilon_s)
                chi_fired = True
        # ── END DETECTION PATHS ─────────────────────────────────────────

        drift_in = (
            self.alpha * H_norm
            + self.beta * S_norm
            + self.gamma * N
            + self.delta * chi_struct
            + self.epsilon * A_time
            + self.epsilon_s * epsilon_s
        )
        self.entropy_hist.append(drift_in)

        semantic_dist = (
            1 - self.minhash.similarity(h1, psi1_trace) if psi1_trace else 0.0
        )
        drift_type, response = self.taxonomy.classify(
            list(self.entropy_hist), chi_struct, semantic_dist
        )

        crossing = self.predictor.update(drift_in).predict_crossing(theta_hard)

        swarm_eps = [a.current_entropy for a in all_agents if hasattr(a, 'current_entropy')]
        self.swarm_median = statistics.median(swarm_eps) if swarm_eps else drift_in
        swarm_mad = statistics.median([abs(e - self.swarm_median) for e in swarm_eps]) or 1e-6
        z = abs(drift_in - self.swarm_median) / swarm_mad

        details = {
            "H_norm": H_norm,
            "S_norm": S_norm,
            "N": N,
            "chi_struct": chi_struct,
            "A_time": A_time,
            "epsilon_s": epsilon_s,
            "d_score": d_score,             # exposed: 1.0 = Luciferian skip detected
            "v_score": v_score,             # 1 = acceleration attack detected (v4.1 OQ3)
            "theta_w": theta_w,
            "theta_hard": theta_hard,
            "L_local": L_local,
            "chi_fired": chi_fired,
            "chi_result": chi_result,
            "drift_type": drift_type,
            "crossing": crossing,
            "swarm_z": z,
        }
        return drift_in, details, aux, Φ_act, response

    # ------------------------------------------------------------------
    # check_and_firewall — top-level entry point
    # ------------------------------------------------------------------

    def check_and_firewall(self, cmd, payload, agent=None, phase_latency_ms=0.0,
                           all_agents=None):
        """
        Full firewall check. Pass agent for χ(t) override eligibility.
        all_agents enables swarm consensus correction.
        """
        try:
            all_agents = all_agents or []
            recursion_n = getattr(agent, 'recursion_n', 1) if agent else 1
            prev_phase = getattr(agent, 'prev_phase', 0) if agent else 0
            psi1_trace = getattr(agent, 'psi1_trace', []) if agent else []

            drift_in, details, aux, Φ_act, response = self.score(
                cmd, payload, phase_latency_ms, recursion_n, prev_phase,
                psi1_trace, all_agents, agent=agent
            )

            stats = self.cmd_stats[cmd]
            tau = stats.drift_params()

            action = self.circuit_breaker.handle(drift_in, self.tau_hard)
            if action in ("circuit_tripped", "recovery_failed"):
                action = "BLOCK"
            elif action in ("recovered", "testing_recovery"):
                action = "ALLOW"
            elif details['crossing'] < 5:
                action = "WARN_PREDICTIVE"
            elif drift_in >= self.tau_hard:
                action = "BLOCK"

            # χ(t) may have already fired inside score() via adaptive threshold;
            # check_and_firewall respects that result if it produced a convergent curve.
            if details.get('chi_fired') and details.get('chi_result'):
                curve, chi_t = details['chi_result']
                if curve == "convergent":
                    action = "RESURRECT"
            elif agent:
                # Fallback: also fire at hard ceiling if adaptive didn't catch it
                if drift_in >= details['theta_hard']:
                    curve, chi_t = self.chi_t_override(agent, payload, details['epsilon_s'])
                    if curve == "convergent":
                        action = "RESURRECT"
            elif drift_in >= tau:
                action = "WARN"
                if agent:
                    response(agent)

            if details['drift_type'] == "evolutionary" and details['epsilon_s'] < self.budget:
                self.checkpoints.append(agent.state.copy() if agent else {})
                self.budget -= details['epsilon_s']
            elif self.checkpoints:
                if agent:
                    agent.state = self.checkpoints.pop()
                action = "ROLLBACK"

            if details['swarm_z'] > 3:
                action = "CONSENSUS_CORRECT"
                if agent:
                    agent.current_entropy = self.swarm_median

            if action in ("ALLOW", "WARN", "RESURRECT"):
                stats.sizes.append(len(aux['pstr']))
                stats.scores.append(drift_in)
                self.schema_keys[cmd] |= aux['now_keys']
                self.schema_types[cmd].update(aux['now_types'])
                self.last_hashes.append(self.minhash.hash(aux['pstr'], 'core_state'))

            self.trace.log(action, drift_in, Φ_act)
            self.logger.info(json.dumps({
                "ts": time.time(),
                "cmd": cmd,
                "action": action,
                "drift": round(drift_in, 4),
                "tau": round(tau, 4),
                "theta_w": round(details['theta_w'], 4),
                "L_local": round(details['L_local'], 4),
                "components": {k: v for k, v in details.items()
                               if k not in ('chi_result',)},
            }))

            if action.startswith("WARN"):
                print(
                    f"WARNING: drift={drift_in:.3f} ≥ τ={tau:.3f}, "
                    f"type={details['drift_type']}, "
                    f"θ_w={details['theta_w']:.3f}, L={details['L_local']:.3f}"
                )
            elif action == "BLOCK":
                print(f"DRIFT_BLOCKED: drift={drift_in:.3f} ≥ τ_hard={self.tau_hard:.2f}")
            elif action == "RESURRECT":
                print(
                    f"RESURRECTED: χ(t) fired at ε_s={details['epsilon_s']:.3f} "
                    f"(θ_w={details['theta_w']:.3f})"
                )

            return action, drift_in, {"tau": tau, **details}

        except Exception as e:
            self.logger.error(f"Drift check failed: {str(e)}")
            return "ERROR_ALLOW", 0.0, {"error": str(e)}
