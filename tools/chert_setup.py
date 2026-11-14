"""
Check Chert + LiveKit settings and register the iMessage webhook.

    python -m tools.chert_setup check
    python -m tools.chert_setup subscribe-imessage https://<tunnel>/chert/imessage/webhook
    python -m tools.chert_setup send-test +15551234567

The FaceTime webhook (call.incoming) is set in the Chert dashboard, not here:
Workers & Lines -> Integration -> HTTPS webhook = https://<tunnel>/chert/webhook.
"""
from __future__ import annotations

import argparse
import uuid

import httpx

from backend import chert_adapter as c


def check() -> None:
    ok = True
    print("LiveKit:", c.LIVEKIT_URL or "(LIVEKIT_URL not set)")
    try:
        c._token("check", "check-room")
        print("  token minting: ok")
    except Exception as e:
        ok = False
        print("  token minting FAILED:", e)
    if c.LIVEKIT_URL and not c.LIVEKIT_URL.startswith("wss://"):
        print("  note: Chert requires a wss:// LiveKit URL (ws:// only works with tools/fake_bridge)")

    print("FaceTime API:", c.FACETIME_API_BASE)
    if not (c.FACETIME_API_KEY and c.CHERT_PROJECT_ID):
        ok = False
        print("  CHERT_FACETIME_API_KEY / CHERT_PROJECT_ID not set")
    else:
        r = httpx.get(f"{c.FACETIME_API_BASE}/v1/lines", headers=c.facetime_headers(), timeout=10)
        print(f"  GET /v1/lines -> {r.status_code}")
        print("  ", r.text[:500])
        ok &= r.status_code == 200

    print("Messaging API:", c.MESSAGING_API_BASE)
    print("  signing secret:", "set" if c.MESSAGING_SECRET else "NOT SET (recap will be logged, not sent)")
    print("  webhook secret:", "set" if c.MESSAGING_WEBHOOK_SECRET else "not set (inbound iMessages are not verified)")
    print("  recap recipient:", c.IMESSAGE_TO or "(none yet: set CHERT_IMESSAGE_TO or text the line once)")
    print("\nAll good." if ok else "\nFix the items above.")


def subscribe(url: str) -> None:
    m = c.Messaging()
    r = httpx.post(f"{c.MESSAGING_API_BASE}/v1/webhook-subscriptions", headers=m.headers(),
                   json={"url": url, "events": ["message.received"]}, timeout=15)
    print(r.status_code, r.text[:800])
    if r.status_code < 300:
        print("\nCopy the 'secret' above into .env as CHERT_MESSAGING_WEBHOOK_SECRET (it is shown only once).")


def send_test(phone: str) -> None:
    m = c.Messaging()
    r = httpx.post(f"{c.MESSAGING_API_BASE}/v1/send", headers=m.headers(), timeout=15,
                   json={"phone": phone, "body": "Pinpoint test message", "idempotency_key": uuid.uuid4().hex})
    print(r.status_code, r.text[:500])


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    s = sub.add_parser("subscribe-imessage")
    s.add_argument("url")
    t = sub.add_parser("send-test")
    t.add_argument("phone")
    a = ap.parse_args()
    {"check": check, "subscribe-imessage": lambda: subscribe(a.url), "send-test": lambda: send_test(a.phone)}[a.cmd]()


if __name__ == "__main__":
    main()
