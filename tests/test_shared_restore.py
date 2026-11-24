"""共享分歧机临时恢复旧下一跳：误判修复、最早完成拍、预算耗尽未判定。

场景（即预置 shared_restore_flip）：
  旧路 I-D-A-C-B-E、新路 I-C-A-E；A 容量 2，其余各 1；存量包 E01 在 A；
  第 1、2、4 拍各到一个包；缓冲 0；期限 8 拍。
唯一解法是共享机 C 第 4 拍临时恢复旧下一跳 (C→B)、第 5 拍再换回新下一跳 (C→A)。
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from app.domain import PLANE_NEW, PLANE_OLD, build_static
from app.examples import BUFFER_ZERO_INFEASIBLE, SHARED_RESTORE_FLIP
from app.planner import FindResult, diagnose, find_earliest_plan, find_plan
from app.runner import Runner, describe_op
from app.schemas import DrillRequest
from app.service import solve_and_report


def _stc(fixture: dict = SHARED_RESTORE_FLIP):
    return build_static(
        prefix=fixture["prefix"],
        old_path=fixture["old_path"],
        new_path=fixture["new_path"],
        table_capacities=fixture["table_capacities"],
        existing_positions=fixture.get("existing_packets", []),
        arrival_ticks=fixture.get("arrival_ticks", []),
        buffer_capacity=fixture["buffer_capacity"],
        deadline=fixture["deadline"],
    )


def test_shared_switch_restore_scenario_now_feasible():
    """曾被误判无解的场景：临时恢复旧下一跳后第 8 拍（恰在期限内）完成。"""
    stc = _stc()
    result = find_earliest_plan(stc)
    assert result.feasible, "共享机临时恢复旧下一跳的方案必须被找到"
    assert not result.cutoff
    assert len(result.actions) == 8, "完成拍应恰为期限第 8 拍"

    runner = Runner(stc)
    runner.run(result.actions)
    summ = runner.summary()
    assert summ["goal_reached"] and summ["all_delivered"]
    assert runner.buffer_peak == 0, "缓冲上限 0 必须全程满足"


def test_goal_tick_8_is_proven_earliest():
    """第 7 拍 horizon 被完整搜完并证实无解 => 第 8 拍是经证实的最早完成拍。"""
    stc = _stc()
    shallow = find_plan(replace(stc, deadline=7))
    assert not shallow.feasible and not shallow.cutoff
    result = find_earliest_plan(stc)
    assert result.feasible and len(result.actions) == 8


def test_plan_contains_restore_and_direction_labels():
    """方案中 C 必须先临时恢复旧下一跳、再换回新下一跳，且方向标注正确。"""
    stc = _stc()
    result = find_earliest_plan(stc)
    assert result.feasible
    described = [
        (tick, describe_op(op, stc)) for tick, (op, _, _) in enumerate(result.actions, 1)
    ]
    c_reps = [
        (t, d) for t, d in described
        if d["kind"] == "replace" and d["switch"] == "C"
    ]
    directions = [d["version"] for _, d in c_reps]
    assert directions == ["to_new", "to_old_restore", "to_new"], (
        "C 应依次 换新→临时恢复旧→再换回新"
    )
    restore = c_reps[1][1]
    assert "临时恢复旧路下一跳" in restore["label_zh"]
    assert restore["old_nexthop"] == "A" and restore["nexthop"] == "B"
    # 换回新路的标注
    assert "换为新路下一跳" in c_reps[2][1]["label_zh"]


def test_ingress_switches_exactly_once_and_packets_keep_release_path():
    stc = _stc()
    result = find_earliest_plan(stc)
    planes = [plane for _, plane, _ in result.actions]
    switches = sum(
        1 for a, b in zip(planes, planes[1:])
        if a == PLANE_OLD and b == PLANE_NEW
    )
    assert switches == 1, "入口仍只许旧→新切一次"
    assert all(
        not (a == PLANE_NEW and b == PLANE_OLD)
        for a, b in zip(planes, planes[1:])
    ), "入口不得回切旧平面"

    runner = Runner(stc)
    runner.run(result.actions)
    traj = {p["packet_id"]: p for p in runner.trajectories()}
    # 每个包都沿放行时生效的路径走完
    assert traj["A1"]["assigned_plane_at_release"] == "old"
    assert traj["A1"]["committed_path"] == SHARED_RESTORE_FLIP["old_path"]
    for pid in ("A2", "A4"):
        assert traj[pid]["assigned_plane_at_release"] == "new"
        assert traj[pid]["committed_path"] == SHARED_RESTORE_FLIP["new_path"]
    assert all(p["status"] == "delivered" for p in traj.values())


def test_budget_exhaustion_reports_undetermined_without_attribution():
    """预算耗尽：报未判定，不执行、不做无解归因、旧表照用。"""
    stc = _stc()
    r = find_earliest_plan(stc, node_budget=1)
    assert not r.feasible and r.cutoff
    assert r.status == "undetermined"

    req = DrillRequest(**SHARED_RESTORE_FLIP)
    report = solve_and_report(req, node_budget=1)
    assert report["status"] == "undetermined"
    assert report["diagnosis"] is None, "未判定不做无解归因"
    assert report["ticks"] == [] and report["plan_overview"] == []
    assert report["old_tables_kept"] is True and report["initial_tables"]
    assert report["search_cutoff"] is True


def test_one_solve_never_expands_more_than_budget():
    stc = _stc()
    for budget in (1, 7, 50, 1_000_000):
        r = find_earliest_plan(stc, node_budget=budget)
        assert r.nodes <= budget


def test_relaxation_minimal_sets_require_proven_subsets():
    """最小放松集：已证实可行 且 所有真子集（含空集）已证实无解。"""
    stc = _stc(BUFFER_ZERO_INFEASIBLE)
    r = find_earliest_plan(stc)
    assert r.status == "infeasible", "该场景应在预算内被证实无解（而非未判定）"
    diag = diagnose(stc)
    mins = {tuple(m) for m in diag["relaxation_minimal_sets"]}
    assert ("buffer",) in mins and ("tables",) in mins
    outcomes = diag["relaxation_outcomes"]
    assert outcomes["(original)"] == "infeasible"
    for mask in mins:
        assert outcomes["+".join(mask)] == "feasible"
    # 组合集的真子集里有可行集，故不得列为最小
    assert not any(len(m) > 1 for m in mins)
    assert diag["relaxation_undetermined"] == []


def test_undetermined_session_refuses_step(monkeypatch):
    """会话：未判定不建执行方案，step 返回冲突。"""
    from app import sessions

    monkeypatch.setattr(
        sessions, "find_earliest_plan",
        lambda stc, **kw: FindResult(False, None, 1_000_000, 1.0, cutoff=True),
    )
    stc = _stc()
    store = sessions.SessionStore()
    sess = store.create(SHARED_RESTORE_FLIP, stc)
    assert sess.status == "undetermined"
    step = store.step(sess)
    assert step["conflict"] is True
    detail = store.detail(sess)
    assert detail["status"] == "undetermined"
    assert detail["diagnosis"] is None
    assert detail["old_tables_kept"] is True
