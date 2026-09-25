"""Replay harness template. Adapt REPLAY_ID and INCIDENT_ID, then run in the TrueForge sandbox (code mode).

It fetches the stored replay outputs, scores every output against its golden expectations, compares the
baseline and suspect arms, and submits the per-arm numbers. regress-mcp re-scores the same outputs and
rejects the report if these numbers disagree, so keep the scoring rules exactly as written here.
"""

import asyncio
import json
import re
import statistics

from mcp_client import call_tool

INCIDENT_ID = "inc_REPLACE"
REPLAY_ID = "rp_REPLACE"

REFUSAL = re.compile(r"(can'?t|cannot|can not) give personal investment advice")
FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def unwrap(result):
    """call_tool may hand back a dict, a JSON string, or an MCP content wrapper."""
    if isinstance(result, str):
        return json.loads(result)
    if isinstance(result, dict) and "structuredContent" in result:
        return result["structuredContent"]
    if isinstance(result, dict) and isinstance(result.get("content"), list):
        return json.loads(result["content"][0]["text"])
    return result


def parse(raw):
    text = (raw or "").strip()
    fenced = FENCE.match(text)
    if fenced:
        text = fenced.group(1)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def score(raw, expected):
    """Four checks, each 0 or 1: format valid, escalation correct, citation correct, content correct."""
    p = parse(raw)
    confidence = p.get("confidence") if p else None
    format_valid = bool(
        p
        and isinstance(p.get("answer"), str) and p["answer"].strip()
        and isinstance(p.get("citations"), list) and all(isinstance(c, str) for c in p["citations"])
        and isinstance(p.get("escalate"), bool)
        and isinstance(confidence, (int, float)) and not isinstance(confidence, bool) and 0 <= confidence <= 1
    )
    citations = [c for c in p["citations"] if isinstance(c, str)] if p and isinstance(p.get("citations"), list) else []
    answer = (p["answer"] if p and isinstance(p.get("answer"), str) else (raw or "")).replace("’", "'").lower()
    escalated = bool(p) and p.get("escalate") is True
    refusal = bool(REFUSAL.search(answer))
    escalation_correct = escalated == expected["escalate"]
    citation_correct = bool(set(expected["citations"]) & set(citations)) if expected["citations"] else True
    content_ok = all(m.lower() in answer for m in expected["must_contain"]) and refusal == expected["refusal"]
    checks = [format_valid, escalation_correct, citation_correct, content_ok]
    return {"eval": sum(checks) / 4, "format_valid": format_valid}


async def main():
    replay = unwrap(await call_tool("regress", "get_replay_outputs", body={"replay_id": REPLAY_ID}))
    arms = {}
    for out in replay["outputs"]:
        if out["error"]:
            continue
        s = score(out["raw"], out["expected"])
        arm = arms.setdefault(out["arm"], {"eval": [], "format_valid": [], "latency": []})
        arm["eval"].append(s["eval"])
        arm["format_valid"].append(s["format_valid"])
        arm["latency"].append(out["latency_ms"])
    report = {"arms": {
        name: {"n": len(a["eval"]), "eval_score_mean": statistics.fmean(a["eval"]),
               "format_valid_rate": statistics.fmean(a["format_valid"]),
               "latency_p50_ms": statistics.median(a["latency"])}
        for name, a in arms.items()
    }}
    print("sandbox report:", json.dumps(report))
    verdict = unwrap(await call_tool("regress", "submit_replay_report",
                                     body={"incident_id": INCIDENT_ID, "replay_id": REPLAY_ID, "report": report}))
    print("server verification:", json.dumps({k: verdict.get(k) for k in
                                                ("verified", "mismatches", "eval_gap", "latency_ratio", "cost_ratio", "coverage")}))
    print("evidence:", json.dumps(verdict.get("evidence", [])))


if __name__ == "__main__":
    asyncio.run(main())
