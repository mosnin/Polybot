"""
ngrok_integration.py — HTTPS tunnel for remote Streamlit dashboard access.

Launches an ngrok tunnel to the local Streamlit dashboard (port 8501),
providing a public HTTPS URL accessible from any phone or browser.
Uses the free tier — no domain required, URL changes on each restart.

Usage:
    # Set your auth token (one-time, get free at https://dashboard.ngrok.com)
    export NGROK_AUTHTOKEN=your_token_here

    # Start the tunnel
    python ngrok_integration.py

    # Dashboard is now accessible at the printed URL (e.g., https://abc123.ngrok-free.app)
    # Press Ctrl+C to stop the tunnel.

Prerequisites:
    pip install pyngrok
"""

import os
import signal
import sys
import time


def main() -> None:
    """Launch ngrok tunnel to Streamlit dashboard on port 8501."""
    try:
        from pyngrok import ngrok
    except ImportError:
        print("ERROR: pyngrok not installed.")
        print("Install with: pip install pyngrok")
        sys.exit(1)

    # Auth token — required for ngrok free tier
    token: str = os.getenv("NGROK_AUTHTOKEN", "")
    if token:
        ngrok.set_auth_token(token)

    print("=" * 60)
    print("  Polybot Dashboard — ngrok HTTPS Tunnel")
    print("=" * 60)
    print()
    print("  Connecting to ngrok...")

    try:
        tunnel = ngrok.connect(8501, "http")
    except Exception as e:
        error_msg: str = str(e).lower()
        if "auth" in error_msg or "token" in error_msg:
            print()
            print("ERROR: ngrok authentication failed.")
            print()
            print("  1. Get a free auth token at:")
            print("     https://dashboard.ngrok.com/get-started/your-authtoken")
            print()
            print("  2. Set it in your environment:")
            print("     export NGROK_AUTHTOKEN=your_token_here")
            print()
            print("  3. Or add to .env:")
            print("     NGROK_AUTHTOKEN=your_token_here")
        else:
            print(f"\nERROR: ngrok connection failed: {e}")
        sys.exit(1)

    public_url: str = tunnel.public_url

    print()
    print(f"  Dashboard URL: {public_url}")
    print()
    print("  Open this URL on any device (phone, tablet, laptop)")
    print("  The tunnel stays active until you press Ctrl+C")
    print()
    print("-" * 60)

    # Graceful shutdown on Ctrl+C
    def shutdown(signum, frame) -> None:
        print("\n\n  Closing ngrok tunnel...")
        try:
            ngrok.disconnect(public_url)
            ngrok.kill()
        except Exception:
            pass
        print("  Tunnel closed.")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Keep the process alive — tunnel closes when process exits
    while True:
        time.sleep(60)


if __name__ == "__main__":
    main()
