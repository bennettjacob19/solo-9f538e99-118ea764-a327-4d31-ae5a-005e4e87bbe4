"""HTTP 层测试：/docs、主求解、会话逐拍、what-if、错误码。"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.examples import (
    BUFFER_ZERO_INFEASIBLE,
    DEADLINE_INFEASIBLE,
    FEASIBLE_DUAL,
    INFLIGHT_WHATIF,
    SHARED_RESTORE_FLIP,
)
from app.main import app
from app.planner import FindResult

client = TestClient(app)


def test_health_and_docs():
    assert client.get("/health").json()["status"] == "ok"
    r = client.get("/openapi.json")
    assert r.status_code == 200
    paths = r.json()["paths"]
    assert "/api/v1/plans" in paths
    assert "/api/v1/whatif" in paths
    docs = client.get("/docs")
    assert docs.status_code == 200 and "swagger" in docs.text.lower()


def test_examples_endpoint():
    d = client.get("/api/v1/examples").json()
    assert "feasible_dual" in d["fixtures"]


def test_plan_endpoint_feasible_shape():
    r = client.post("/api/v1/plans", json=FEASIBLE_DUAL)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["status"] == "feasible"
    assert d["goal_tick"] == len(d["ticks"]) >= 1
    for t in d["ticks"]:
        assert {"tick", "table_action", "ingress", "tables", "buffer", "hops"} <= set(t)
        for tab in t["tables"]:
            assert tab["used"] <= tab["capacity"]
    # 逐包轨迹齐全
    ids = {p["packet_id"] for p in d["packet_trajectories"]}
    assert {"E01", "A2", "A6"} <= ids


def test_plan_endpoint_infeasible_buffer():
    r = client.post("/api/v1/plans", json=BUFFER_ZERO_INFEASIBLE)
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "infeasible"
    assert d["old_tables_kept"] is True
    assert d["diagnosis"]["reasons_zh"]
    assert d["initial_tables"]


def test_plan_endpoint_deadline():
    d = client.post("/api/v1/plans", json=DEADLINE_INFEASIBLE).json()
    assert d["status"] == "infeasible"
    assert "deadline_propagation" in d["diagnosis"]["codes"]


def test_plan_endpoint_shared_restore_flip():
    """曾误判无解的场景：现在经 Docs 同款接口返回第 8 拍（经证实最早）的方案。"""
    r = client.post("/api/v1/plans", json=SHARED_RESTORE_FLIP)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["status"] == "feasible"
    assert d["goal_tick"] == 8 and d["goal_tick_proven_earliest"] is True
    kinds = [
        (t["tick"], t["table_action"].get("version"))
        for t in d["ticks"] if t["table_action"]["kind"] == "replace"
    ]
    assert ("to_old_restore" in [v for _, v in kinds]), "应标出临时恢复旧下一跳"
    # 会话逐拍执行同一方案，替换方向标注一致
    sid = client.post("/api/v1/sessions", json=SHARED_RESTORE_FLIP).json()["session_id"]
    seen_restore = False
    for _ in range(8):
        sr = client.post(f"/api/v1/sessions/{sid}/step", json={})
        assert sr.status_code == 200, sr.text
        act = sr.json()["tick"]["table_action"]
        if act["kind"] == "replace" and act.get("version") == "to_old_restore":
            seen_restore = True
            assert "临时恢复旧路下一跳" in act["label_zh"]
    assert seen_restore
    detail = client.get(f"/api/v1/sessions/{sid}").json()
    assert detail["goal_reached"] is True


def test_session_undetermined_refuses_step_409(monkeypatch):
    import app.sessions as sess_mod

    monkeypatch.setattr(
        sess_mod, "find_earliest_plan",
        lambda stc, **kw: FindResult(False, None, 1_000_000, 1.0, cutoff=True),
    )
    r = client.post("/api/v1/sessions", json=SHARED_RESTORE_FLIP)
    assert r.status_code == 200
    assert r.json()["status"] == "undetermined"
    sid = r.json()["session_id"]
    assert client.post(f"/api/v1/sessions/{sid}/step", json={}).status_code == 409
    assert client.post(f"/api/v1/sessions/{sid}/run").status_code == 409
    detail = client.get(f"/api/v1/sessions/{sid}").json()
    assert detail["status"] == "undetermined"
    assert detail["diagnosis"] is None and detail["old_tables_kept"] is True


def test_session_step_flow_and_409_at_end():
    r = client.post("/api/v1/sessions", json=FEASIBLE_DUAL)
    assert r.status_code == 200, r.text
    sid = r.json()["session_id"]
    total = r.json()["total_planned_ticks"]
    assert total >= 1

    seen = []
    for _ in range(total):
        sr = client.post(f"/api/v1/sessions/{sid}/step", json={})
        assert sr.status_code == 200, sr.text
        body = sr.json()
        seen.append(body["executed_tick"])
    assert seen == list(range(1, total + 1))
    assert body["finished"] is True and body["goal_reached"] is True

    again = client.post(f"/api/v1/sessions/{sid}/step", json={})
    assert again.status_code == 409

    detail = client.get(f"/api/v1/sessions/{sid}").json()
    assert detail["goal_reached"] is True
    assert len(detail["ticks"]) == total


def test_session_infeasible_refuses_step_409():
    sid = client.post("/api/v1/sessions", json=BUFFER_ZERO_INFEASIBLE).json()["session_id"]
    r = client.post(f"/api/v1/sessions/{sid}/step", json={})
    assert r.status_code == 409
    detail = client.get(f"/api/v1/sessions/{sid}").json()
    assert detail["status"] == "infeasible" and detail["old_tables_kept"] is True


def test_unknown_session_404():
    assert client.get("/api/v1/sessions/nope").status_code == 404
    assert client.post("/api/v1/sessions/nope/step").status_code == 404


def test_whatif_endpoint_reports_collision():
    r = client.post("/api/v1/whatif", json=INFLIGHT_WHATIF)
    assert r.status_code == 200, r.text
    d = r.json()
    assert any(v["code"] == "in_flight_hit_changed_table" for v in d["violations"])


def test_pydantic_rejects_duplicate_arrival_and_too_many_switches():
    bad = dict(FEASIBLE_DUAL)
    bad["arrival_ticks"] = [3, 3]
    assert client.post("/api/v1/plans", json=bad).status_code == 422

    bad2 = dict(FEASIBLE_DUAL)
    bad2["old_path"] = ["I", "a", "b", "E"]
    bad2["new_path"] = ["I", "c", "d", "e", "E"]  # 合计 7 台
    bad2["table_capacities"] = {s: 1 for s in ["I", "a", "b", "c", "d", "e", "E"]}
    assert client.post("/api/v1/plans", json=bad2).status_code == 422


def test_missing_capacity_is_422_chinese():
    bad = dict(FEASIBLE_DUAL)
    caps = dict(bad["table_capacities"]); caps.pop("N1")
    bad["table_capacities"] = caps
    r = client.post("/api/v1/plans", json=bad)
    assert r.status_code == 422
    assert any("N1" in e["msg"] for e in r.json()["detail"])


def test_lenient_inputs_and_route_aliases():
    # 表容给全局整数、存量位置给字符串、到包给对象，都应被规范化并求解
    payload = {
        "old_path": ["I", "S", "A", "E"],
        "new_path": ["I", "S", "B", "E"],
        "table_capacities": 1,
        "existing_packets": ["S"],
        "arrival_ticks": [{"tick": 5}],
        "buffer_capacity": 1,
        "deadline": 10,
    }
    r = client.post("/solve", json=payload)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["status"] == "feasible"
    assert "steps" in d and d["steps"]
    assert d["config"]["table_capacities"]["S"] == 1
