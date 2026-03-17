"""
memory.py — Lightweight persistent memory layer using SQLite.

Replaces Redis for all persistence needs. Zero daemon, zero config,
zero extra RAM. SQLite is embedded in Python's stdlib — it writes
directly to a single file on disk with ACID guarantees.

Three responsibilities:
1. Bot state persistence (equity history, win/loss, peak balance)
2. Model cross-window state (vol history, spread history, EMAs)
3. Regime memory (cross-window intelligence the bot learns from)

The regime memory is the key upgrade: it tracks patterns ACROSS windows
that the per-window Bayesian model can't see. Things like:
- Is BTC in a trending or ranging regime? (affects signal reliability)
- How well is each strategy actually performing? (adaptive confidence)
- What's the rolling 1-hour volatility doing? (regime classification)

Performance: SQLite WAL mode with synchronous=NORMAL gives us ~0.1ms
writes — faster than Redis over TCP for our payload sizes.
"""

import json
import logging
import sqlite3
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

logger = logging.getLogger("Memory")


@dataclass
class RegimeState:
    """Cross-window market regime classification.

    Updated after every window resolution. Persists indefinitely.
    The bot uses this to adapt strategy selection and sizing.
    """

    # Rolling window outcomes (last N windows)
    recent_directions: Deque[str]  # "UP" / "DOWN" for last 50 windows
    recent_magnitudes: Deque[float]  # abs(price_change_pct) per window

    # Strategy performance tracking
    sniper_attempts: int = 0
    sniper_wins: int = 0
    directional_attempts: int = 0
    directional_wins: int = 0
    mm_attempts: int = 0
    mm_wins: int = 0

    # Regime classification (recomputed after each window)
    regime: str = "unknown"  # "trending_up", "trending_down", "ranging", "volatile"
    regime_strength: float = 0.0  # 0-1, how strong the current regime is
    hourly_vol: float = 0.0  # rolling 1-hour volatility

    @property
    def sniper_win_rate(self) -> float:
        if self.sniper_attempts < 5:
            return 0.85  # default assumption until we have data
        return self.sniper_wins / self.sniper_attempts

    @property
    def directional_win_rate(self) -> float:
        if self.directional_attempts < 10:
            return 0.60  # default assumption
        return self.directional_wins / self.directional_attempts

    def classify_regime(self) -> None:
        """Classify the current market regime from recent window outcomes.

        Regime types:
        - trending_up: >65% of recent windows resolved UP
        - trending_down: >65% of recent windows resolved DOWN
        - ranging: roughly equal UP/DOWN, low magnitude
        - volatile: high magnitude moves, no clear direction

        The regime classification affects:
        - Sniper confidence threshold (lower in trending, higher in ranging)
        - Directional signal weight (higher in trending)
        - MM spread threshold (tighter in ranging where spreads are stable)
        """
        if len(self.recent_directions) < 10:
            self.regime = "unknown"
            self.regime_strength = 0.0
            return

        directions = list(self.recent_directions)
        n = len(directions)
        up_count = sum(1 for d in directions if d == "UP")
        up_ratio = up_count / n

        magnitudes = list(self.recent_magnitudes)
        avg_magnitude = sum(magnitudes) / len(magnitudes) if magnitudes else 0.0

        if up_ratio > 0.65:
            self.regime = "trending_up"
            self.regime_strength = (up_ratio - 0.5) * 2  # 0.65->0.3, 1.0->1.0
        elif up_ratio < 0.35:
            self.regime = "trending_down"
            self.regime_strength = (0.5 - up_ratio) * 2
        elif avg_magnitude > 0.1:  # >0.1% average move per window
            self.regime = "volatile"
            self.regime_strength = min(avg_magnitude / 0.2, 1.0)
        else:
            self.regime = "ranging"
            self.regime_strength = 1.0 - abs(up_ratio - 0.5) * 4

    def get_adaptive_sniper_confidence(self, base_confidence: float = 0.80) -> float:
        """Adapt sniper confidence threshold based on actual performance.

        If sniper is hitting 90%+, we can afford to lower the threshold
        (take more snipes, compound faster). If it's hitting 75%, raise
        the bar to protect capital.

        Also adjusts for regime:
        - Trending regime: lower threshold (momentum carries through)
        - Ranging regime: higher threshold (reversals more likely)
        """
        # Performance adjustment: +/- 5% based on actual win rate vs 85% target
        perf_adjustment = 0.0
        if self.sniper_attempts >= 10:
            actual_wr = self.sniper_win_rate
            perf_adjustment = (0.85 - actual_wr) * 0.5  # miss by 10% -> raise bar 5%

        # Regime adjustment
        regime_adjustment = 0.0
        if self.regime in ("trending_up", "trending_down") and self.regime_strength > 0.3:
            regime_adjustment = -0.05  # trending = lower bar (momentum carries)
        elif self.regime == "ranging":
            regime_adjustment = 0.05  # ranging = higher bar (reversals)

        adjusted = base_confidence + perf_adjustment + regime_adjustment
        return max(0.70, min(adjusted, 0.95))  # clamp to reasonable range

    def get_adaptive_edge_threshold(self, base_edge: float = 0.02) -> float:
        """Adapt minimum edge threshold based on regime.

        Trending markets: smaller edges are reliable (momentum)
        Ranging markets: need bigger edge (noise dominates)
        Volatile markets: edges are real but risky (keep threshold)
        """
        if self.regime in ("trending_up", "trending_down") and self.regime_strength > 0.3:
            return base_edge * 0.7  # 30% lower bar in trends
        elif self.regime == "ranging":
            return base_edge * 1.3  # 30% higher bar in ranges
        return base_edge


class MemoryStore:
    """SQLite-backed persistent memory. Zero-daemon, zero-config.

    Uses WAL mode for concurrent reads during writes.
    Synchronous=NORMAL for ~0.1ms writes (vs FULL's ~5ms).
    Single file on disk, no network, no external process.
    """

    def __init__(self, db_path: str = "polybot_memory.db") -> None:
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._connect()

    def _connect(self) -> None:
        """Open SQLite connection with performance-optimized pragmas."""
        try:
            self._conn = sqlite3.connect(self.db_path, timeout=5.0)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA cache_size=-2000")  # 2MB cache
            self._conn.execute("PRAGMA temp_store=MEMORY")
            self._create_tables()
            logger.info(f"SQLite memory store opened: {self.db_path}")
        except Exception as e:
            logger.error(f"SQLite connection failed: {e}")
            self._conn = None

    def _create_tables(self) -> None:
        """Create tables if they don't exist. Idempotent."""
        if self._conn is None:
            return
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS kv_store (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS trade_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                strategy TEXT NOT NULL,
                direction TEXT NOT NULL,
                edge REAL,
                kelly REAL,
                outcome TEXT,
                pnl REAL,
                regime TEXT,
                window_slug TEXT
            );

            CREATE TABLE IF NOT EXISTS window_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                slug TEXT NOT NULL,
                direction TEXT NOT NULL,
                magnitude REAL NOT NULL,
                volatility REAL,
                regime TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_trade_ts ON trade_log(timestamp);
            CREATE INDEX IF NOT EXISTS idx_window_ts ON window_log(timestamp);
        """)
        self._conn.commit()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # --- Key-Value Store (replaces Redis SET/GET) ---

    def set(self, key: str, value: dict) -> None:
        """Store a JSON-serializable dict by key."""
        if self._conn is None:
            return
        try:
            self._conn.execute(
                "INSERT OR REPLACE INTO kv_store (key, value, updated_at) VALUES (?, ?, ?)",
                (key, json.dumps(value), time.time()),
            )
            self._conn.commit()
        except Exception:
            pass  # never block the trading loop

    def get(self, key: str) -> Optional[dict]:
        """Retrieve a dict by key, or None if not found."""
        if self._conn is None:
            return None
        try:
            row = self._conn.execute(
                "SELECT value FROM kv_store WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                return None
            return json.loads(row[0])
        except Exception:
            return None

    # --- Trade Logging ---

    def log_trade(
        self,
        strategy: str,
        direction: str,
        edge: float = 0.0,
        kelly: float = 0.0,
        outcome: Optional[str] = None,
        pnl: float = 0.0,
        regime: str = "unknown",
        window_slug: str = "",
    ) -> None:
        """Log a trade for performance analysis. Append-only."""
        if self._conn is None:
            return
        try:
            self._conn.execute(
                "INSERT INTO trade_log (timestamp, strategy, direction, edge, kelly, "
                "outcome, pnl, regime, window_slug) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (time.time(), strategy, direction, edge, kelly, outcome, pnl, regime, window_slug),
            )
            self._conn.commit()
        except Exception:
            pass

    def log_window(
        self,
        slug: str,
        direction: str,
        magnitude: float,
        volatility: float = 0.0,
        regime: str = "unknown",
    ) -> None:
        """Log a window resolution for regime analysis."""
        if self._conn is None:
            return
        try:
            self._conn.execute(
                "INSERT INTO window_log (timestamp, slug, direction, magnitude, "
                "volatility, regime) VALUES (?, ?, ?, ?, ?, ?)",
                (time.time(), slug, direction, magnitude, volatility, regime),
            )
            self._conn.commit()
        except Exception:
            pass

    def update_trade_outcome(
        self, window_slug: str, outcome: str, pnl: float = 0.0
    ) -> None:
        """Update the outcome of trades from a specific window."""
        if self._conn is None:
            return
        try:
            self._conn.execute(
                "UPDATE trade_log SET outcome = ?, pnl = ? "
                "WHERE window_slug = ? AND outcome IS NULL",
                (outcome, pnl, window_slug),
            )
            self._conn.commit()
        except Exception:
            pass

    # --- Regime State Loading ---

    def load_regime_state(self, lookback_windows: int = 50) -> RegimeState:
        """Load regime state from recent window history.

        Reconstructs the RegimeState from the last N windows stored in SQLite.
        This means the bot's regime awareness survives restarts — it doesn't
        need 50 windows to "warm up" after a restart.
        """
        state = RegimeState(
            recent_directions=deque(maxlen=lookback_windows),
            recent_magnitudes=deque(maxlen=lookback_windows),
        )

        if self._conn is None:
            return state

        try:
            rows = self._conn.execute(
                "SELECT direction, magnitude FROM window_log "
                "ORDER BY timestamp DESC LIMIT ?",
                (lookback_windows,),
            ).fetchall()

            # Reverse to chronological order
            for direction, magnitude in reversed(rows):
                state.recent_directions.append(direction)
                state.recent_magnitudes.append(magnitude)

            # Load strategy performance from trade log (last 24 hours)
            cutoff = time.time() - 86400
            perf = self._conn.execute(
                "SELECT strategy, outcome, COUNT(*) FROM trade_log "
                "WHERE timestamp > ? AND outcome IS NOT NULL "
                "GROUP BY strategy, outcome",
                (cutoff,),
            ).fetchall()

            for strategy, outcome, count in perf:
                if strategy == "sniper":
                    state.sniper_attempts += count
                    if outcome == "win":
                        state.sniper_wins += count
                elif strategy == "directional":
                    state.directional_attempts += count
                    if outcome == "win":
                        state.directional_wins += count
                elif strategy == "mm":
                    state.mm_attempts += count
                    if outcome == "win":
                        state.mm_wins += count

            state.classify_regime()
            logger.info(
                f"Regime loaded: {state.regime} "
                f"(strength={state.regime_strength:.2f}, "
                f"windows={len(state.recent_directions)}, "
                f"sniper={state.sniper_wins}/{state.sniper_attempts})"
            )

        except Exception as e:
            logger.warning(f"Could not load regime state: {e}")

        return state

    # --- Cleanup ---

    def cleanup_old_data(self, days: int = 7) -> None:
        """Purge data older than N days to keep the DB small.

        Called once per day or on startup. The DB should stay under 1MB
        for months of continuous operation.
        """
        if self._conn is None:
            return
        try:
            cutoff = time.time() - (days * 86400)
            self._conn.execute(
                "DELETE FROM trade_log WHERE timestamp < ?", (cutoff,)
            )
            self._conn.execute(
                "DELETE FROM window_log WHERE timestamp < ?", (cutoff,)
            )
            self._conn.commit()
            logger.info(f"Cleaned up data older than {days} days")
        except Exception:
            pass
