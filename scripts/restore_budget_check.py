"""复跑核查：共享分歧机临时恢复换路场景 + 预算耗尽未判定场景。

直接调用领域层（不需要起服务）：
  1) shared_restore_flip：曾误判无解的 I-D-A-C-B-E → I-C-A-E 场景，
     现在应求得第 8 拍（经证实最早）的方案，且 C 第 4 拍临时恢复旧下一跳、
     第 5 拍换回新下一跳，方向标注正确，严格复放达成目标；
  2) 同一场景把节点预算压到 1：应报“未判定”，不执行、不做无解归因；
  3) 回归：buffer_zero_infeasible 仍被证实无解，最小放松集为 [buffer] 与 [tables]。

退出码 0 表示全部断言通过。
"""

from __future__ import annotations

import sys

from app.domain import build_static
from app.examples import BUFFER_ZERO_INFEASIBLE, SHARED_RESTORE_FLIP
from app.planner import diagnose, find_earliest_plan
from app.runner import Runner, describe_op
from app.schemas import DrillRequest
from app.service import solve_and_report


def _stc(fixture: dict):
    return build_static(
        prefix=fixture["prefix"],
        old_path=fixture["old_path"], new_path=fixture["new_path"],
        table_capacities=fixture["table_capacities"],
        existing_positions=fixture.get("existing_packets", []),
        arrival_ticks=fixture.get("arrival_ticks", []),
        buffer_capacity=fixture["buffer_capacity"],
        deadline=fixture["deadline"],
    )


def main() -> int:
    failures: list[str] = []

    # ---- 1. 换路场景：临时恢复旧下一跳 -------------------------------------
    stc = _stc(SHARED_RESTORE_FLIP)
    r = find_earliest_plan(stc)
    print(f"[restore] feasible={r.feasible} nodes={r.nodes} cutoff={r.cutoff} "
          f"elapsed={r.elapsed_ms:.1f}ms")
    if not r.feasible:
        failures.append("临时恢复场景应有解，实际无解/未判定")
    else:
        rn = Runner(stc)
        rn.run(r.actions)
        print(f"[restore] goal_tick={len(r.actions)} goal_reached={rn.summary()['goal_reached']} "
              f"buffer_peak={rn.buffer_peak}")
        for i, (op, plane, opened) in enumerate(r.actions, 1):
            d = describe_op(op, stc)
            print(f"    t{i}: {d['label_zh']} ｜ 入口={'新' if plane else '旧'}/"
                  f"{'开' if opened else '关'}")
        c_dirs = [
            describe_op(op, stc)["version"]
            for op, _, _ in r.actions
            if op[0] == "rep" and op[1] == "C"
        ]
        print(f"[restore] C 的替换方向序列: {c_dirs}")
        if len(r.actions) != 8:
            failures.append(f"完成拍应为 8（经证实最早），实际 {len(r.actions)}")
        if c_dirs != ["to_new", "to_old_restore", "to_new"]:
            failures.append(f"C 的替换方向应为 新→临时恢复旧→新，实际 {c_dirs}")
        if not rn.summary()["goal_reached"] or rn.buffer_peak != 0:
            failures.append("严格复放未达成目标或缓冲溢出")
        traj = {p["packet_id"]: p for p in rn.trajectories()}
        if traj["A1"]["assigned_plane_at_release"] != "old":
            failures.append("A1 应沿旧路走完")
        if traj["A4"]["assigned_plane_at_release"] != "new":
            failures.append("A4 应沿新路走完")

    # ---- 2. 预算耗尽：未判定 -------------------------------------------------
    req = DrillRequest(**SHARED_RESTORE_FLIP)
    report = solve_and_report(req, node_budget=1)
    print(f"[budget] status={report['status']} nodes={report['search_nodes']} "
          f"diagnosis={report['diagnosis']} old_tables_kept={report['old_tables_kept']}")
    if report["status"] != "undetermined":
        failures.append(f"预算耗尽应报未判定，实际 {report['status']}")
    if report["diagnosis"] is not None or report["ticks"] != []:
        failures.append("未判定不应执行也不应做无解归因")
    if report["old_tables_kept"] is not True:
        failures.append("未判定应保持旧表")

    # ---- 3. 回归：证实无解 + 最小放松集 --------------------------------------
    stc0 = _stc(BUFFER_ZERO_INFEASIBLE)
    r0 = find_earliest_plan(stc0)
    diag = diagnose(stc0)
    print(f"[regress] buffer0 status={r0.status} nodes={r0.nodes} "
          f"minimal_sets={diag['relaxation_minimal_sets']} "
          f"undetermined={diag['relaxation_undetermined']}")
    if r0.status != "infeasible":
        failures.append(f"buffer0 应被证实无解，实际 {r0.status}")
    mins = {tuple(m) for m in diag["relaxation_minimal_sets"]}
    if mins != {("buffer",), ("tables",)}:
        failures.append(f"最小放松集应为 [buffer]/[tables]，实际 {mins}")

    if failures:
        print("\nCHECK FAILED:")
        for f in failures:
            print(" -", f)
        return 1
    print("\nCHECK PASS: 临时恢复换路、预算耗尽未判定、证实无解归因均已复跑符合预期。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
