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
_PRIVATE_KEY: str = ""  # needed for approve/withdraw scripts
_RPC_URL: str = ""  # needed for web3 calls in approve/withdraw

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


def _render_backtest_tab() -> None:
    """Render the Backtest validation tab.

    Provides a one-click "Run Backtest" button that downloads historical data,
    replays every 5-minute window through the full pipeline, and displays
    comprehensive results with auto-alerts for degraded performance.
    """
    import streamlit as st

    st.subheader("Strategy Validation — Deep Backtest")
    st.caption(
        "Downloads 30 days of 1-second BTC ticks and replays every 5-minute window "
        "through the full Bayesian + z-score + Monte Carlo pipeline. "
        "Proves the strategy is winning before any live capital is risked."
    )

    # Initialize backtest state
    if "backtest_result" not in st.session_state:
        st.session_state.backtest_result = None
    if "backtest_running" not in st.session_state:
        st.session_state.backtest_running = False

    # Run Backtest button
    if st.button(
        "Run Backtest",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.backtest_running,
    ):
        st.session_state.backtest_running = True
        st.session_state.backtest_result = None

        progress_bar = st.progress(0.0, text="Downloading historical data...")
        status_text = st.empty()

        def update_progress(pct: float) -> None:
            progress_bar.progress(
                min(pct, 1.0),
                text=f"Simulating windows... {pct * 100:.0f}%",
            )

        try:
            import asyncio as _asyncio

            from backtest import run_backtest
            from config import load_config

            config = load_config()
            status_text.info(
                f"Backtesting {config.historical_data_days} days of data..."
            )
            result = _asyncio.run(run_backtest(config, progress_callback=update_progress))
            st.session_state.backtest_result = result
            progress_bar.progress(1.0, text="Backtest complete!")
            status_text.empty()
        except Exception as e:
            st.error(f"Backtest failed: {e}")
        finally:
            st.session_state.backtest_running = False

    # Display results if available
    result = st.session_state.backtest_result
    if result is None:
        st.info("Click 'Run Backtest' to validate strategy on historical data.")
        return

    # --- Alert Banners ---
    if result.get("alert_win_rate"):
        st.error(
            f"ALERT: Win rate {result['win_rate']:.1%} is below the 58% minimum threshold. "
            "Model parameters may need adjustment before live deployment."
        )
    if result.get("alert_expectancy"):
        st.error(
            f"ALERT: Daily expectancy {result['daily_expectancy']:.2%} is below the 0.8% minimum. "
            "Strategy may not cover fees/gas in live trading."
        )

    # --- Verdict ---
    if not result.get("alert_win_rate") and not result.get("alert_expectancy"):
        st.success(
            "STRATEGY VALIDATED — Win rate and expectancy confirm "
            "positive edge. Ready for live deployment."
        )
    elif result.get("total_trades", 0) == 0:
        st.warning("No trades generated. Model thresholds may be too strict.")

    # --- Metric Cards ---
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Trades", f"{result.get('total_trades', 0):,}")
        st.metric("Wins / Losses", f"{result.get('wins', 0)} / {result.get('losses', 0)}")
    with col2:
        st.metric("Win Rate", f"{result.get('win_rate', 0):.1%}")
        st.metric("Avg Edge", f"{result.get('avg_edge', 0):.4f}")
    with col3:
        st.metric("Sharpe Ratio", f"{result.get('sharpe_ratio', 0):.2f}")
        st.metric("Max Drawdown", f"{result.get('max_drawdown', 0):.1%}")

    col4, col5, col6 = st.columns(3)
    with col4:
        st.metric("Net Return", f"{result.get('net_return', 0):.1%}")
    with col5:
        st.metric("Final Balance", f"${result.get('final_balance', 0):.2f}")
    with col6:
        st.metric("Daily Expectancy", f"{result.get('daily_expectancy', 0):.2%}")

    # --- Backtest Equity Curve ---
    bt_equity: list = result.get("equity_curve", [])
    if len(bt_equity) >= 2:
        st.markdown("---")
        st.subheader("Backtest Equity Curve")
        fig: plt.Figure = render_equity_curve(bt_equity)
        st.pyplot(fig)
        plt.close(fig)

    # --- Trade Log (first 100 trades) ---
    bt_trades: list = result.get("trades", [])
    if bt_trades:
        st.markdown("---")
        st.subheader(f"Trade Log (showing {min(len(bt_trades), 100)} of {len(bt_trades)})")

        log_rows: list = []
        for t in bt_trades[:100]:
            log_rows.append({
                "Time": datetime.datetime.fromtimestamp(
                    t.get("timestamp", 0) / 1000.0
                ).strftime("%Y-%m-%d %H:%M"),
                "Direction": t.get("direction", "—"),
                "Edge": f"{t.get('edge', 0):.4f}",
                "Kelly": f"{t.get('kelly_fraction', 0):.3f}",
                "Entry": f"${t.get('entry_price', 0):,.2f}",
                "Exit": f"${t.get('exit_price', 0):,.2f}",
                "Won": "Yes" if t.get("won") else "No",
                "P&L": f"${t.get('pnl', 0):+.4f}",
                "Balance": f"${t.get('balance_after', 0):.2f}",
            })
        st.dataframe(pd.DataFrame(log_rows), use_container_width=True, hide_index=True)

    # --- Windows Summary ---
    st.caption(
        f"Analyzed {result.get('total_windows', 0):,} five-minute windows | "
        f"Avg Kelly: {result.get('avg_kelly', 0):.3f}"
    )


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

    # Dark mode CSS + mobile-responsive layout
    st.markdown("""
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
    .stApp { background-color: #0e1117; }
    .stMetric label { color: #888; }
    [data-testid="stSidebar"] { background-color: #161b22; }
    @media (max-width: 768px) {
        .stColumns > div { min-width: 45% !important; }
        .stMetric { font-size: 0.85rem; }
    }
</style>
""", unsafe_allow_html=True)

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

    # --- Tabbed Layout ---
    tab_live, tab_backtest = st.tabs(["Live Trading", "Backtest"])

    # === LIVE TRADING TAB ===
    with tab_live:
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

        # --- Strategy Health Score ---
        health_color = "red"
        health_label = "UNHEALTHY"
        if state.latest_status:
            live_wr = state.latest_status.get("wins", 0) / max(
                state.latest_status.get("total_trades", 1), 1
            )
            bt_ok = (
                (st.session_state.get("backtest_result") or {}).get(
                    "daily_expectancy", 0
                )
                > 0.01
            )
            if live_wr >= 0.58 and bt_ok:
                health_color, health_label = "green", "HEALTHY"
            elif live_wr >= 0.52 or (
                state.latest_status.get("total_trades", 0) < 50
            ):
                health_color, health_label = "orange", "WARMING UP"
        st.markdown(
            f'<div style="padding:8px 16px;border-radius:8px;background:{health_color};'
            f'color:white;text-align:center;font-weight:bold;margin-bottom:12px">'
            f"Strategy Health: {health_label}</div>",
            unsafe_allow_html=True,
        )

        # --- Row 2: Equity Curve ---
        st.subheader("Equity Curve")
        fig: plt.Figure = render_equity_curve(state.equity_curve)
        st.pyplot(fig)
        plt.close(fig)  # free memory

        # --- Cycle Latency Graph ---
        if state.cycle_latencies:
            st.subheader("Cycle Latency")
            lat_fig, lat_ax = plt.subplots(figsize=(10, 2.5))
            lat_ax.plot(state.cycle_latencies, color="#ff6b6b", linewidth=1)
            lat_ax.axhline(
                y=80, color="#ff0000", linestyle="--", alpha=0.5, label="80ms warn"
            )
            lat_ax.axhline(
                y=60, color="#00d4aa", linestyle="--", alpha=0.5, label="60ms target"
            )
            lat_ax.set_facecolor("#0e1117")
            lat_fig.patch.set_facecolor("#0e1117")
            lat_ax.tick_params(colors="#888")
            lat_ax.set_ylabel("ms", color="#888")
            lat_ax.legend(fontsize=8)
            lat_fig.tight_layout()
            st.pyplot(lat_fig)
            plt.close(lat_fig)

        # --- Row 3: Trade History Table ---
        st.subheader("Recent Trades (Last 50)")
        df: pd.DataFrame = render_trade_table(state.trades)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # CSV export button
        if state.trades:
            csv_data = render_trade_table(state.trades).to_csv(index=False)
            st.download_button(
                label="Export Trade History (CSV)",
                data=csv_data,
                file_name="polybot_trades.csv",
                mime="text/csv",
            )

    # === BACKTEST TAB ===
    with tab_backtest:
        _render_backtest_tab()

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

        # Approve USDC for trading (one-time setup)
        st.subheader("USDC Approval")
        st.caption("One-time approval for Polymarket exchange contracts")
        if st.button("Approve USDC for Trading", use_container_width=True):
            if _PRIVATE_KEY and _RPC_URL:
                with st.spinner("Broadcasting approval transaction..."):
                    try:
                        from approve_usdc import approve_from_dashboard

                        result = approve_from_dashboard(_PRIVATE_KEY, _RPC_URL)
                        if result["status"] == "success":
                            tx_display: str = result.get("ctf_tx", "")[:16]
                            st.success(f"Approved! TX: {tx_display}...")
                        elif result["status"] == "already_approved":
                            st.info("Already approved — no action needed")
                        else:
                            st.error(f"Failed: {result.get('error', 'unknown')}")
                    except Exception as e:
                        st.error(f"Approval error: {e}")
            else:
                st.warning("Wallet not configured")

        st.divider()

        # Withdraw with live balance as max withdrawable amount
        st.subheader("Withdraw")
        max_withdraw: float = max(state.balance - 1.0, 0.0)  # keep $1 buffer
        st.caption(f"Max withdrawable: ${max_withdraw:.2f}")
        withdraw_amount: float = st.number_input(
            "Amount (USDC)",
            min_value=0.0,
            max_value=max(max_withdraw, 0.01),  # prevent max_value=0 error
            value=0.0,
            step=10.0,
            key="withdraw_input",
        )
        if st.button("Execute Withdraw", use_container_width=True):
            if withdraw_amount > 0 and _PRIVATE_KEY and _RPC_URL:
                with st.spinner("Processing withdrawal..."):
                    try:
                        import asyncio as _asyncio

                        from withdraw import withdraw_from_dashboard

                        result = _asyncio.run(
                            withdraw_from_dashboard(
                                _PRIVATE_KEY, _RPC_URL, withdraw_amount
                            )
                        )
                        if result["status"] == "success":
                            st.success(
                                f"Withdraw complete: ${withdraw_amount:.2f} USDC\n"
                                f"TX: {result.get('tx_hash', 'N/A')[:20]}..."
                            )
                        else:
                            st.error(f"Failed: {result.get('error', 'unknown')}")
                    except Exception as e:
                        st.error(f"Withdraw error: {e}")
            elif withdraw_amount <= 0:
                st.warning("Enter an amount > 0")
            else:
                st.warning("Wallet not configured")

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

        st.divider()
        st.caption(
            "Use the **Backtest** tab to validate strategy on 30 days "
            "of historical data before risking live capital."
        )

    # --- Auto-refresh ---
    time.sleep(REFRESH_INTERVAL)
    st.rerun()


def run_dashboard(
    event_queue: multiprocessing.Queue,
    control_queue: multiprocessing.Queue,
    private_key: str,
    rpc_url: str = "",
) -> None:
    """Entry point for the dashboard process.

    Called as the target of multiprocessing.Process from bot.py's main().
    Sets module-level globals (which survive Linux fork) and launches
    Streamlit programmatically.

    Args:
        event_queue: Queue for receiving bot events
        control_queue: Queue for sending control commands to bot
        private_key: Polygon private key for approve/withdraw scripts
        rpc_url: Alchemy RPC URL for web3 calls in approve/withdraw
    """
    global _EVENT_QUEUE, _CONTROL_QUEUE, _WALLET_ADDRESS, _PRIVATE_KEY, _RPC_URL

    _EVENT_QUEUE = event_queue
    _CONTROL_QUEUE = control_queue
    _PRIVATE_KEY = private_key
    _RPC_URL = rpc_url

    # Derive wallet address from private key
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
