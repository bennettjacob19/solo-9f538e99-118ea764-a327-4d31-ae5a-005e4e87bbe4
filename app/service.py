"""把 HTTP 模型映射为领域 Static，并组装对外报告。"""

from __future__ import annotations

from typing import Any

from .domain import ConfigError, Static, build_static, initial_state
from .planner import FindResult, diagnose, find_earliest_plan
from .runner import Runner, describe_op, table_view
from .schemas import DrillRequest


def static_from_request(req: DrillRequest) -> Static:
    return build_static(
        prefix=req.prefix,
        old_path=req.old_path,
        new_path=req.new_path,
        table_capacities=req.table_capacities,
        existing_positions=[p.model_dump() for p in req.existing_packets],
        arrival_ticks=req.arrival_ticks,
        buffer_capacity=req.buffer_capacity,
        deadline=req.deadline,
        release_per_tick=req.release_per_tick,
    )


def plan_overview(stc: Static, actions: tuple[Any, ...]) -> list[dict[str, Any]]:
    out = []
    for i, (op, plane_t, open_t) in enumerate(actions, start=1):
        d = describe_op(op, stc)
        d["tick"] = i
        d["target_ingress"] = {
            "plane": "new" if plane_t else "old",
            "open": bool(open_t),
        }
        out.append(d)
    return out


def solve_and_report(req: DrillRequest, *, node_budget: int = 1_000_000) -> dict[str, Any]:
    stc = static_from_request(req)
    result = find_earliest_plan(stc, node_budget=node_budget)
    if result.feasible and result.actions is not None:
        runner = Runner(stc)
        runner.run(result.actions)
        return feasible_report(stc, result, runner)
    if result.cutoff:
        # 预算耗尽：既未证实最早方案也未证实无解——不执行、不做无解归因。
        return undetermined_report(stc, result)
    diag = diagnose(stc)
    return infeasible_report(stc, result, diag)


def feasible_report(stc: Static, result: FindResult, runner: Runner) -> dict[str, Any]:
    summ = runner.summary()
    return {
        "status": "feasible",
        "summary_zh": (
            f"第 {stc.deadline} 拍期限内存在方案：第 {summ['executed_ticks']} 拍达成目标"
            f"（全部 {len(stc.packets)} 个包送达、入口已切新路、旧路专用表项撤净）；"
            "经逐层完备搜索证实这是最早完成拍。可逐拍查看 tables / buffer / hops。"
        ),
        "prefix": stc.prefix,
        "deadline": stc.deadline,
        "goal_tick": summ["executed_ticks"],
        "goal_tick_proven_earliest": True,
        "search_nodes": result.nodes,
        "elapsed_ms": round(result.elapsed_ms, 2),
        "plan_overview": plan_overview(stc, result.actions or ()),
        "ticks": runner.tick_reports,
        "packet_trajectories": runner.trajectories(),
        **summ,
        "diagnosis": None,
        "old_tables_kept": False,
    }


def undetermined_report(stc: Static, result: FindResult) -> dict[str, Any]:
    return {
        "status": "undetermined",
        "summary_zh": (
            f"节点预算（{result.nodes} 个状态）耗尽，既未证实存在最早完成方案，"
            "也未证实无解；未对设备做任何改动，旧表照用，且不做无解归因。"
            "可简化输入（更短期限/更少包）或提高预算后重试。"
        ),
        "prefix": stc.prefix,
        "deadline": stc.deadline,
        "goal_tick": None,
        "goal_tick_proven_earliest": False,
        "search_nodes": result.nodes,
        "elapsed_ms": round(result.elapsed_ms, 2),
        "search_cutoff": True,
        "plan_overview": [],
        "ticks": [],
        "packet_trajectories": [],
        "initial_tables": table_view(stc, initial_state(stc)),
        "old_tables_kept": True,
        "diagnosis": None,
    }


def infeasible_report(
    stc: Static, result: FindResult, diagnosis: dict[str, Any]
) -> dict[str, Any]:
    return {
        "status": "infeasible",
        "summary_zh": (
            f"第 {stc.deadline} 拍期限内不存在满足全部约束的无环换路方案；"
            "未对设备做任何改动，旧表照常用。约束归因见 diagnosis.reasons_zh。"
        ),
        "prefix": stc.prefix,
        "deadline": stc.deadline,
        "goal_tick": None,
        "goal_tick_proven_earliest": False,
        "search_nodes": result.nodes,
        "elapsed_ms": round(result.elapsed_ms, 2),
        "search_cutoff": False,
        "plan_overview": [],
        "ticks": [],
        "packet_trajectories": [],
        "initial_tables": table_view(stc, initial_state(stc)),
        "old_tables_kept": True,
        "diagnosis": diagnosis,
    }
