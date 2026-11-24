"""在 ≤12 拍的小状态空间内做完备 DFS 搜索，并在无解时做约束归因。

判定口径（与接口约定一致）：
  - feasible     ：预算内找到方案（find_earliest_plan 还保证完成拍最早）；
  - infeasible   ：预算内完整搜完状态空间，证实无解；
  - undetermined ：节点预算（默认 100 万）耗尽，既未找到方案也未证实无解。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any

from .domain import (
    MAX_TICKS,
    Action,
    State,
    advance,
    candidate_actions,
    earliest_delivery_bound_ok,
    goal_reached,
    initial_state,
    min_ops_remaining,
)
from .domain import Static


@dataclass
class FindResult:
    feasible: bool
    actions: tuple[Action, ...] | None
    nodes: int
    elapsed_ms: float
    cutoff: bool = False

    @property
    def status(self) -> str:
        if self.feasible:
            return "feasible"
        return "undetermined" if self.cutoff else "infeasible"


class _BudgetCutoff(Exception):
    """节点预算耗尽：立即中止整棵搜索（结论只能是“未判定”，继续搜无意义）。"""


def find_plan(stc: Static, *, node_budget: int = 1_000_000) -> FindResult:
    """返回任意一个合法方案。深度受 deadline（≤12）限制，状态做记忆化失败剪枝。

    展开状态数严格不超过 node_budget；耗尽即中止并以 cutoff=True 报告“未判定”。
    """
    t0 = time.perf_counter()
    start = initial_state(stc)
    if not earliest_delivery_bound_ok(stc, start, 0):
        return FindResult(False, None, 0, 0.0)

    failed: set[tuple[int, State]] = set()
    nodes = 0

    def dfs(tick: int, state: State) -> tuple[Action, ...] | None:
        nonlocal nodes
        if goal_reached(stc, state):
            return ()
        if tick > stc.deadline:
            return None
        key = (tick, state)
        if key in failed:
            return None

        if nodes >= node_budget:
            raise _BudgetCutoff
        nodes += 1

        # 此刻到 deadline 还剩 deadline-tick+1 次操作机会
        if min_ops_remaining(stc, state) > stc.deadline - tick + 1:
            failed.add(key)
            return None

        for action in candidate_actions(stc, state):
            res = advance(stc, state, tick, action)
            if not res.ok or res.state is None:
                continue
            nxt = res.state
            if not earliest_delivery_bound_ok(stc, nxt, tick):
                continue
            if min_ops_remaining(stc, nxt) > stc.deadline - tick:
                continue
            suffix = dfs(tick + 1, nxt)
            if suffix is not None:
                return (action,) + suffix

        failed.add(key)
        return None

    cutoff = False
    try:
        actions = dfs(1, start)
    except _BudgetCutoff:
        actions = None
        cutoff = True
    elapsed = (time.perf_counter() - t0) * 1000
    return FindResult(actions is not None, actions, nodes, elapsed, cutoff=cutoff)


def find_earliest_plan(stc: Static, *, node_budget: int = 1_000_000) -> FindResult:
    """在 1..deadline 拍中逐层尝试，返回完成拍最早的合法方案。

    全部浅层共用同一个节点预算（一次求解最多展开 node_budget 个状态）：
    浅层被证实无解后才加深；任一浅层因预算耗尽而无法判定时，整体只能报
    “未判定”——此时既不能声称最早，也不能声称无解。
    """
    t0 = time.perf_counter()
    total_nodes = 0
    for horizon in range(1, stc.deadline + 1):
        remaining = node_budget - total_nodes
        if remaining <= 0:
            return FindResult(
                False, None, total_nodes,
                (time.perf_counter() - t0) * 1000, cutoff=True,
            )
        sub = replace(stc, deadline=horizon)
        r = find_plan(sub, node_budget=remaining)
        total_nodes += r.nodes
        if r.feasible:
            # 更浅的每一层都已被完整搜完并证实无解，本层即经证实的最早完成拍。
            return FindResult(
                True, r.actions, total_nodes,
                (time.perf_counter() - t0) * 1000,
            )
        if r.cutoff:
            return FindResult(
                False, None, total_nodes,
                (time.perf_counter() - t0) * 1000, cutoff=True,
            )
    return FindResult(
        False, None, total_nodes,
        (time.perf_counter() - t0) * 1000,
    )


def minimum_table_ops(stc: Static) -> int:
    """静态最少表项操作数（每台差异机器 0/1，replace 可同时立新撤旧）。"""
    initial = frozenset((s, stc.old_nh[s]) for s in stc.old_path)
    n = 0
    for s in stc.switches:
        need_new = s in stc.new_nh and (s, stc.new_nh[s]) not in initial
        old_exclusive = (
            s in stc.old_nh
            and not (s in stc.new_nh and stc.new_nh[s] == stc.old_nh[s])
        )
        old_present = old_exclusive and (s, stc.old_nh[s]) in initial
        if need_new or old_present:
            n += 1
    # 同机“新项缺失 + 旧专用项在”可由一次 replace 同时完成，故按差异机器数计数。
    return n


def _relaxed(stc: Static, *, horizon12: bool, inf_buffer: bool, big_tables: bool,
             extended_horizon: int | None = None) -> Static:
    changes: dict[str, Any] = {}
    if extended_horizon is not None:
        changes["deadline"] = extended_horizon
    elif horizon12 and stc.deadline < MAX_TICKS:
        changes["deadline"] = MAX_TICKS
    if inf_buffer:
        changes["buffer_cap"] = 10**6
    if big_tables:
        changes["caps"] = {s: 64 for s in stc.caps}
    return replace(stc, **changes) if changes else stc


def diagnose(stc: Static, *, node_budget: int = 400_000) -> dict[str, Any]:
    """对已证实无解的配置做约束归因：分别放松 期限 / 缓冲 / 表容，看重放松后是否可解。

    放松实验与正式求解使用完全相同的改表规则（含共享分歧机临时恢复）。
    一个放松组合只有同时满足两条才会被列为“最小”：
      1. 该组合在节点预算内**被证实可行**；
      2. 它的所有真子集（含空集＝原配置）都**被证实无解**。
    预算耗尽的组合不算证据，单独列入 relaxation_undetermined。
    """
    from itertools import combinations

    from .domain import static_deadline_lower_bound

    earliest_worst, timing_notes = static_deadline_lower_bound(stc)
    min_ops = minimum_table_ops(stc)

    static_hits: list[dict[str, Any]] = []
    if earliest_worst > stc.deadline:
        static_hits.append({
            "code": "deadline_propagation",
            "label_zh": "完成期限（传播时间）",
            "detail_zh": (
                f"纯传播至少要到第 {earliest_worst} 拍才可能送完，"
                f"早于期限第 {stc.deadline} 拍；即使表项/缓冲都不限也来不及。"
            ),
        })
    if min_ops > stc.deadline:
        static_hits.append({
            "code": "ops_rate",
            "label_zh": "每拍至多一条表项操作",
            "detail_zh": (
                f"新旧路径在 {min_ops} 台交换机上存在表态差异，每台至少需要一次"
                f"增/删/替换，共至少 {min_ops} 次操作，但期限内只有 {stc.deadline} 拍。"
            ),
        })

    dims = ("deadline", "buffer", "tables")
    labels = {
        "deadline": f"把完成期限从第 {stc.deadline} 拍放宽到演练上限第 {MAX_TICKS} 拍",
        "buffer": f"把入口缓冲上限从 {stc.buffer_cap} 放宽到不受限",
        "tables": "放宽各交换机表项上限",
    }

    # 每个放松组合的判定结论：feasible / infeasible / undetermined。
    # 空调用前提：原配置（空组合）已被证实无解。
    outcome: dict[tuple[str, ...], str] = {(): "infeasible"}
    for r in (1, 2, 3):
        for mask in combinations(dims, r):
            res = find_plan(_relaxed(
                stc,
                horizon12="deadline" in mask,
                inf_buffer="buffer" in mask,
                big_tables="tables" in mask,
            ), node_budget=node_budget)
            outcome[mask] = res.status

    def all_proper_subsets_proven_infeasible(mask: tuple[str, ...]) -> bool:
        # combinations 保持 dims 的顺序，子元组与 outcome 的键写法一致；
        # 含空组合 ()（＝原配置，调用前提已证实无解）。
        for r in range(0, len(mask)):
            for sub in combinations(mask, r):
                if outcome[tuple(sub)] != "infeasible":
                    return False
        return True

    minimal_sets: list[tuple[str, ...]] = [
        mask for mask in outcome
        if mask and outcome[mask] == "feasible"
        and all_proper_subsets_proven_infeasible(mask)
    ]
    undetermined_masks: list[tuple[str, ...]] = [
        mask for mask in outcome if outcome[mask] == "undetermined"
    ]

    # 结构性兜底：全部放宽并给足 24 拍，正常输入必然可解。
    structural = find_plan(
        _relaxed(stc, horizon12=False, inf_buffer=True, big_tables=True, extended_horizon=24),
        node_budget=node_budget,
    )

    explanations: list[str] = []
    codes: list[str] = [h["code"] for h in static_hits]
    for h in static_hits:
        explanations.append(h["detail_zh"])

    if minimal_sets:
        for mask in minimal_sets:
            if len(mask) == 1:
                d = mask[0]
                code = {
                    "deadline": "deadline_tight",
                    "buffer": "buffer_capacity",
                    "tables": "table_capacity",
                }[d]
                dim_name = {
                    "deadline": "完成期限",
                    "buffer": "入口缓冲上限",
                    "tables": "交换机表项上限",
                }[d]
                codes.append(code)
                explanations.append(
                    f"搜索验证：仅{labels[d]}，方案即成立——主要受【{dim_name}】约束。"
                )
            else:
                names = "、".join({
                    "deadline": "完成期限", "buffer": "入口缓冲上限", "tables": "表项上限"
                }[d] for d in mask)
                codes.append("combined_" + "_".join(mask))
                explanations.append(
                    "搜索验证：必须同时放宽 " + " 与 ".join(labels[d] for d in mask)
                    + f" 才有解，即受【{names}】的组合约束。"
                )
    elif structural.feasible:
        explanations.append(
            "在 12 拍上限内放松单项/组合约束仍未找到方案，但放宽到 24 拍且不限缓冲/表容后可解，"
            "说明主要受演练拍数（完成期限 + 每拍一次操作/逐拍传播）的综合时序约束。"
        )
        codes.append("horizon_timing")
    elif structural.cutoff:
        explanations.append("结构性放宽实验也触及节点预算，无法给出归因结论。")
        codes.append("undetermined_budget")
    else:
        explanations.append("即使大幅放宽期限、缓冲与表容仍不可解，可能是输入模型本身的问题。")
        codes.append("structural")

    if undetermined_masks:
        names = ["+".join(m) for m in undetermined_masks]
        explanations.append(
            "（以下放松组合在节点预算内未能判定，未计入最小放松集："
            + "、".join(names) + "。）"
        )

    # 去重保序
    seen: set[str] = set()
    uniq_codes = [c for c in codes if not (c in seen or seen.add(c))]

    return {
        "codes": uniq_codes,
        "reasons_zh": explanations,
        "earliest_propagation_tick": earliest_worst,
        "timing_notes": timing_notes,
        "minimum_table_ops": min_ops,
        "deadline": stc.deadline,
        "relaxation_minimal_sets": [list(m) for m in minimal_sets],
        "relaxation_undetermined": [list(m) for m in undetermined_masks],
        "relaxation_outcomes": {
            "+".join(mask) if mask else "(original)": res
            for mask, res in sorted(outcome.items())
        },
        "structural_extended_feasible": structural.feasible,
    }
