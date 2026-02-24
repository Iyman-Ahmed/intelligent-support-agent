"""
Load test script — simulates concurrent customer conversations.

Usage:
    python scripts/load_test.py --url http://localhost:8000 --users 10 --turns 5
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from typing import List

import httpx


# ---------------------------------------------------------------------------
# Test conversation scenarios
# ---------------------------------------------------------------------------

SCENARIOS = [
    [
        "Hi, I need help with my account",
        "I can't log in — it keeps saying incorrect password",
        "I already tried resetting it but didn't get the email",
        "My email is demo@example.com",
        "Great, thank you!",
    ],
    [
        "What is your refund policy?",
        "I'd like to request a refund for order ORD-001",
        "It's been 3 weeks and I never received it",
        "I understand, thanks for looking into it",
    ],
    [
        "Where is my order? Order ID is ORD-002",
        "When will it arrive exactly?",
        "That's fine, thanks",
    ],
    [
        "How do I upgrade my plan?",
        "What's the difference between Pro and Enterprise?",
        "I'll go with Pro. How do I pay?",
    ],
]


async def run_conversation(
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    scenario_idx: int,
    user_id: int,
) -> dict:
    """Run a single multi-turn conversation and return timing stats."""
    scenario = SCENARIOS[scenario_idx % len(SCENARIOS)]
    session_id = None
    response_times = []
    errors = 0

    for message in scenario:
        start = time.monotonic()
        try:
            resp = await client.post(
                f"{base_url}/chat/message",
                json={
                    "session_id": session_id,
                    "message": message,
                    "customer_email": f"loadtest_user_{user_id}@example.com",
                },
                headers={"X-API-Key": api_key},
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
            session_id = data["session_id"]
            elapsed = (time.monotonic() - start) * 1000
            response_times.append(elapsed)
        except Exception as exc:
            errors += 1
            print(f"  ❌ User {user_id} error: {exc}")

    return {
        "user_id": user_id,
        "turns": len(scenario),
        "errors": errors,
        "avg_response_ms": statistics.mean(response_times) if response_times else 0,
        "max_response_ms": max(response_times) if response_times else 0,
    }


async def run_load_test(
    base_url: str,
    api_key: str,
    num_users: int,
    turns: int,
) -> None:
    print(f"\n🚀 Load test: {num_users} concurrent users → {base_url}")
    print("=" * 60)

    start = time.monotonic()
    async with httpx.AsyncClient() as client:
        tasks = [
            run_conversation(client, base_url, api_key, i, i)
            for i in range(num_users)
        ]
        results = await asyncio.gather(*tasks)

    total_time = (time.monotonic() - start) * 1000
    all_avg = [r["avg_response_ms"] for r in results if r["avg_response_ms"] > 0]
    total_errors = sum(r["errors"] for r in results)

    print(f"\n📊 Results:")
    print(f"  Total time:        {total_time:.0f}ms")
    print(f"  Users:             {num_users}")
    print(f"  Total errors:      {total_errors}")
    if all_avg:
        print(f"  Avg response time: {statistics.mean(all_avg):.0f}ms")
        print(f"  P95 response time: {sorted(all_avg)[int(len(all_avg) * 0.95)]:.0f}ms")
        print(f"  Max response time: {max(all_avg):.0f}ms")
    print(f"  Error rate:        {total_errors / (num_users * turns) * 100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load test the support agent")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL")
    parser.add_argument("--key", default="dev-secret-key", help="API key")
    parser.add_argument("--users", type=int, default=5, help="Concurrent users")
    parser.add_argument("--turns", type=int, default=3, help="Turns per user")
    args = parser.parse_args()

    asyncio.run(run_load_test(args.url, args.key, args.users, args.turns))
