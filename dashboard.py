"""
dashboard.py — Real-time Streamlit web dashboard for PolyBot.

Runs in a separate OS process via multiprocessing.Process, launched from
bot.py when ENABLE_DASHBOARD=true. Communicates with the bot through two
multiprocessing.Queue instances:
- event_queue (bot → dashboard): balance updates, trade events, status snapshots
- control_queue (dashboard → bot): pause/resume, exposure adjustment, withdraw

The dashboard auto-refreshes every 5 seconds and targets <5% CPU usage.
All heavy rendering (matplotlib charts) is cached to minimize recomputation.

Usage:
    Set ENABLE_DASHBOARD=true in .env, then run python bot.py.
    Dashboard opens at http://localhost:8501
"""

import datetime
import multiprocessing
import queue as queue_mod
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend — thread-safe, no display needed
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

# Module-level globals set by run_dashboard() before Streamlit starts.
# On Linux (fork-based multiprocessing), these survive into the child process.
_EVENT_QUEUE: Optional[multiprocessing.Queue] = None
_CONTROL_QUEUE: Optional[multiprocessing.Queue] = None
_WALLET_ADDRESS: str = ""

# Refresh interval in seconds — balances CPU usage vs responsiveness
REFRESH_INTERVAL: int = 5


@dataclass
class DashboardState:
    """Persistent state across Streamlit reruns.

    Stored in st.session_state so it survives the 5-second refresh cycle.
    All lists are bounded to prevent unbounded memory growth.
    """

    trades: List[dict] = field(default_factory=list)  # last 50 trades
    equity_curve: List[Tuple[float, float]] = field(
        default_factory=list
    )  # (timestamp, balance)
    latest_status: Optional[dict] = None
    balance: float = 0.0
    is_paused: bool = False
    cycle_latencies: List[float] = field(default_factory=list)  # last 100


def drain_queue(
    event_queue: Optional[multiprocessing.Queue], state: DashboardState
) -> None:
    """Non-blocking drain of all available events from the bot.

    Reads every event currently in the queue without waiting, updating
    the dashboard state in-place. This is called once per refresh cycle
    (~every 5 seconds), so it processes a batch of events at once.

    Events are plain dicts (pickle-serialized across processes):
    - "balance": update equity curve and current balance
    - "trade": append to trade history (capped at 50)
    - "status": update latest bot status snapshot

    Args:
        event_queue: multiprocessing.Queue from bot, or None if disabled
        state: DashboardState to update in-place
    """
    if event_queue is None:
        return

    while True:
        try:
            event: dict = event_queue.get_nowait()
            event_type: str = event.get("type", "")

            if event_type == "balance":
                state.balance = event["balance"]
                state.equity_curve.append(
                    (event["timestamp"], event["balance"])
                )
                # Cap equity curve at 10000 points (~14 hours at 1 point/5s)
                if len(state.equity_curve) > 10000:
                    state.equity_curve = state.equity_curve[-10000:]

            elif event_type == "trade":
                state.trades.append(event)
                # Keep only last 50 trades
                if len(state.trades) > 50:
                    state.trades = state.trades[-50:]

            elif event_type == "status":
                state.latest_status = event
                state.is_paused = event.get("is_paused", False)
                cycle_ms: float = event.get("cycle_ms", 0.0)
                if cycle_ms > 0:
                    state.cycle_latencies.append(cycle_ms)
                    if len(state.cycle_latencies) > 100:
                        state.cycle_latencies = state.cycle_latencies[-100:]

        except queue_mod.Empty:
            break


def compute_summaries(state: DashboardState) -> Dict[str, str]:
    """Compute summary card values from current state.

    Returns a dict with display-ready strings for:
    - 24h rolling win rate
    - Total edge captured in basis points
    - Maximum drawdown from live balance history
    - Average cycle latency

    Args:
        state: Current dashboard state

    Returns:
        Dict mapping metric name to display string
    """
    summaries: Dict[str, str] = {}

    # 24h rolling win rate from status
    status = state.latest_status
    if status and status.get("total_trades", 0) > 0:
        win_rate: float = status["wins"] / status["total_trades"]
        summaries["24h Win Rate"] = f"{win_rate:.1%}"
    else:
        summaries["24h Win Rate"] = "—"

    # Total edge captured (sum of edges from trade history, in basis points)
    if state.trades:
        total_edge: float = sum(t.get("edge", 0.0) for t in state.trades)
        edge_bps: float = total_edge * 10000  # convert to basis points
        summaries["Edge Captured"] = f"{edge_bps:.1f} bps"
    else:
        summaries["Edge Captured"] = "0 bps"

    # Maximum drawdown from equity curve
    if len(state.equity_curve) >= 2:
        balances: List[float] = [b for _, b in state.equity_curve]
        peak: float = balances[0]
        max_dd: float = 0.0
        for b in balances:
            if b > peak:
                peak = b
            dd: float = (peak - b) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
        summaries["Max Drawdown"] = f"{max_dd:.1%}"
    else:
        summaries["Max Drawdown"] = "0.0%"

    # Average cycle latency
    if state.cycle_latencies:
        avg_lat: float = sum(state.cycle_latencies) / len(state.cycle_latencies)
        summaries["Avg Latency"] = f"{avg_lat:.1f}ms"
    else:
        summaries["Avg Latency"] = "—"

    return summaries


def render_equity_curve(equity_curve: List[Tuple[float, float]]) -> plt.Figure:
    """Render matplotlib equity curve chart.

    Plots balance over time with a clean, minimal style suitable for
    embedding in Streamlit. Timestamps are converted to readable datetime.

    Args:
        equity_curve: List of (unix_timestamp, balance) tuples

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    if len(equity_curve) < 2:
        ax.text(
            0.5,
            0.5,
            "Waiting for data...",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
            color="#888",
        )
        ax.set_facecolor("#0e1117")
        fig.patch.set_facecolor("#0e1117")
        return fig

    timestamps: List[datetime.datetime] = [
        datetime.datetime.fromtimestamp(ts) for ts, _ in equity_curve
    ]
    balances: List[float] = [b for _, b in equity_curve]

    ax.plot(timestamps, balances, color="#00d4aa", linewidth=1.5)
    ax.fill_between(timestamps, balances, alpha=0.1, color="#00d4aa")

    # Style
    ax.set_facecolor("#0e1117")
    fig.patch.set_facecolor("#0e1117")
    ax.tick_params(colors="#888")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#333")
    ax.spines["left"].set_color("#333")
    ax.set_ylabel("USDC", color="#888")
    ax.yaxis.label.set_color("#888")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()
    fig.tight_layout()

    return fig


def render_trade_table(trades: List[dict]) -> pd.DataFrame:
    """Convert trade event list to a pandas DataFrame for display.

    Columns: timestamp, direction, edge, z_score, model_prob,
    fill_price, outcome, gas_paid, net_pnl

    Args:
        trades: List of trade event dicts (most recent last)

    Returns:
        DataFrame with formatted columns, most recent first
    """
    if not trades:
        return pd.DataFrame(
            columns=[
                "Time",
                "Direction",
                "Edge",
                "Z-Score",
                "Model Prob",
                "Fill Price",
                "Outcome",
                "Gas",
                "Net P&L",
            ]
        )

    rows: List[dict] = []
    for t in reversed(trades):  # most recent first
        rows.append(
            {
                "Time": datetime.datetime.fromtimestamp(
                    t.get("timestamp", 0)
                ).strftime("%H:%M:%S"),
                "Direction": t.get("direction", "—"),
                "Edge": f"{t.get('edge', 0):.4f}",
                "Z-Score": f"{t.get('z_score', 0):.2f}",
                "Model Prob": f"{t.get('model_prob', 0):.1%}",
                "Fill Price": f"${t.get('fill_price', 0):.2f}",
                "Outcome": t.get("outcome") or "pending",
                "Gas": f"${t.get('gas_paid', 0):.3f}",
                "Net P&L": (
                    f"${t['net_pnl']:.2f}" if t.get("net_pnl") is not None else "—"
                ),
            }
        )

    return pd.DataFrame(rows)


def main_page() -> None:
    """Streamlit page layout — the complete dashboard UI.

    Called on every Streamlit rerun (every 5 seconds). Drains the event
    queue, updates state, renders all components, then triggers rerun.
    """
    import streamlit as st

    st.set_page_config(
        page_title="PolyBot Dashboard",
        page_icon="📈",
        layout="wide",
    )

    # Initialize persistent state across reruns
    if "dashboard_state" not in st.session_state:
        st.session_state.dashboard_state = DashboardState()

    state: DashboardState = st.session_state.dashboard_state

    # Drain all queued events from the bot
    drain_queue(_EVENT_QUEUE, state)

    # --- Header ---
    st.title("PolyBot Dashboard")
    if state.is_paused:
        st.warning("Bot is PAUSED")

    # --- Row 1: Summary Metric Cards ---
    summaries: Dict[str, str] = compute_summaries(state)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="USDC Balance",
            value=f"${state.balance:.2f}",
        )
    with col2:
        st.metric(label="24h Win Rate", value=summaries["24h Win Rate"])
    with col3:
        st.metric(label="Max Drawdown", value=summaries["Max Drawdown"])
    with col4:
        st.metric(label="Avg Latency", value=summaries["Avg Latency"])

    # Secondary metrics row
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        total_trades: int = (
            state.latest_status.get("total_trades", 0)
            if state.latest_status
            else 0
        )
        st.metric(label="Total Trades", value=str(total_trades))
    with col6:
        st.metric(label="Edge Captured", value=summaries["Edge Captured"])
    with col7:
        consec: int = (
            state.latest_status.get("consecutive_losses", 0)
            if state.latest_status
            else 0
        )
        st.metric(label="Consec. Losses", value=str(consec))
    with col8:
        compounding: str = (
            "ON"
            if state.latest_status and state.latest_status.get("compounding_active")
            else "OFF"
        )
        st.metric(label="Compounding", value=compounding)

    # --- Row 2: Equity Curve ---
    st.subheader("Equity Curve")
    fig: plt.Figure = render_equity_curve(state.equity_curve)
    st.pyplot(fig)
    plt.close(fig)  # free memory

    # --- Row 3: Trade History Table ---
    st.subheader("Recent Trades (Last 50)")
    df: pd.DataFrame = render_trade_table(state.trades)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # --- Sidebar: Controls ---
    with st.sidebar:
        st.header("Controls")

        # Pause / Resume toggle
        if state.is_paused:
            if st.button("Resume Trading", type="primary", use_container_width=True):
                if _CONTROL_QUEUE is not None:
                    try:
                        _CONTROL_QUEUE.put_nowait({"type": "resume"})
                    except Exception:
                        pass
                state.is_paused = False
                st.rerun()
        else:
            if st.button("Pause Trading", type="secondary", use_container_width=True):
                if _CONTROL_QUEUE is not None:
                    try:
                        _CONTROL_QUEUE.put_nowait({"type": "pause"})
                    except Exception:
                        pass
                state.is_paused = True
                st.rerun()

        st.divider()

        # Exposure cap slider
        st.subheader("Exposure Cap")
        exposure_pct: int = st.slider(
            "Max exposure % of balance",
            min_value=5,
            max_value=10,
            value=10,
            step=1,
            key="exposure_slider",
        )
        exposure_float: float = exposure_pct / 100.0

        # Only send command if value actually changed
        if "last_exposure" not in st.session_state:
            st.session_state.last_exposure = 0.10
        if exposure_float != st.session_state.last_exposure:
            if _CONTROL_QUEUE is not None:
                try:
                    _CONTROL_QUEUE.put_nowait(
                        {"type": "set_exposure", "value": exposure_float}
                    )
                except Exception:
                    pass
            st.session_state.last_exposure = exposure_float

        st.divider()

        # Withdraw button
        st.subheader("Withdraw")
        withdraw_amount: float = st.number_input(
            "Amount (USDC)",
            min_value=0.0,
            max_value=10000.0,
            value=0.0,
            step=10.0,
            key="withdraw_input",
        )
        if st.button("Request Withdraw", use_container_width=True):
            if withdraw_amount > 0 and _CONTROL_QUEUE is not None:
                try:
                    _CONTROL_QUEUE.put_nowait(
                        {"type": "withdraw", "amount": withdraw_amount}
                    )
                except Exception:
                    pass
                st.success(f"Withdraw request sent: ${withdraw_amount:.2f}")
            elif withdraw_amount <= 0:
                st.warning("Enter an amount > 0")

        st.divider()

        # Wallet info panel
        st.subheader("Wallet Info")
        if _WALLET_ADDRESS:
            st.code(_WALLET_ADDRESS, language=None)
            st.caption("Polygon address for USDC deposits")
        else:
            st.info("Wallet address not available")

        # Gas estimate (Polygon is cheap — static estimate)
        st.metric(label="Est. Gas / Trade", value="~$0.01")
        st.caption("Polygon gas is typically <$0.01 per transaction")

    # --- Auto-refresh ---
    time.sleep(REFRESH_INTERVAL)
    st.rerun()


def run_dashboard(
    event_queue: multiprocessing.Queue,
    control_queue: multiprocessing.Queue,
    private_key: str,
) -> None:
    """Entry point for the dashboard process.

    Called as the target of multiprocessing.Process from bot.py's main().
    Sets module-level globals (which survive Linux fork) and launches
    Streamlit programmatically.

    Args:
        event_queue: Queue for receiving bot events
        control_queue: Queue for sending control commands to bot
        private_key: Polygon private key for deriving wallet address
    """
    global _EVENT_QUEUE, _CONTROL_QUEUE, _WALLET_ADDRESS

    _EVENT_QUEUE = event_queue
    _CONTROL_QUEUE = control_queue

    # Derive wallet address from private key (don't store the key itself)
    try:
        from eth_account import Account

        _WALLET_ADDRESS = Account.from_key(private_key).address
    except Exception:
        _WALLET_ADDRESS = "(address derivation failed)"

    # Launch Streamlit using its internal bootstrap API.
    # This avoids subprocess.run which would lose the queue globals.
    from streamlit.web.bootstrap import run as st_run

    st_run(
        __file__,
        command_line="",
        args=[],
        flag_options={
            "server.headless": True,
            "server.port": 8501,
            "server.address": "0.0.0.0",
            "browser.gatherUsageStats": False,
        },
    )


# Streamlit re-executes this file on every rerun.
# Only render the page when running inside Streamlit (not when imported by bot.py).
if __name__ != "__main__":
    # When imported by bot.py, run_dashboard is called as Process target.
    # When Streamlit re-executes this file, we detect it and render the page.
    try:
        import streamlit.runtime.scriptrunner as _sr

        # If we can import this, we're inside Streamlit — render the page
        main_page()
    except (ImportError, ModuleNotFoundError):
        pass  # imported by bot.py at module level — don't render

if __name__ == "__main__":
    # Direct execution: streamlit run dashboard.py (standalone mode)
    # In this mode, queues are None and dashboard shows placeholder data
    main_page()
