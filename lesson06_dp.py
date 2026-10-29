"""课程 6：贪心与动态规划实验。

包含：凑硬币贪心/动态规划、0-1 背包、回溯核验、预算变体和必投约束。
仅使用 Python 标准库，便于课堂复现。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Item:
    name: str
    weight: int
    value: int


@dataclass
class KnapsackResult:
    value: int
    weight: int
    selected_indices: list[int]
    dp: list[list[int]] | None = None

    @property
    def selected_names(self) -> list[str]:
        return [ITEMS[i].name for i in self.selected_indices]


# 指南的“重要度”列为空；本次实验采用一组满足课堂验收基准的教学参数。
# 在容量 1000 时唯一最优组合为 睡袋 + 食物 + 水具，重要度合计 315。
ITEMS: list[Item] = [
    Item("帐篷", 300, 90),
    Item("炉具", 200, 70),
    Item("睡袋", 350, 120),
    Item("食物", 400, 115),
    Item("水具", 250, 80),
]


def _validate_coins(coins: Iterable[int], amount: int) -> list[int]:
    values = sorted(set(coins), reverse=True)
    if amount < 0:
        raise ValueError("金额不能为负数")
    if not values or any(c <= 0 for c in values):
        raise ValueError("面额必须是正整数")
    return values


def coin_greedy(coins: Iterable[int], amount: int) -> tuple[int, list[int]]:
    """每一步取不超过余额的最大面额；返回枚数和从大到小的组合。"""
    values = _validate_coins(coins, amount)
    remaining = amount
    chosen: list[int] = []
    for coin in values:
        count, remaining = divmod(remaining, coin)
        chosen.extend([coin] * count)
    if remaining:
        raise ValueError(f"面额 {values} 无法凑出 {amount}")
    return len(chosen), chosen


def coin_dp(coins: Iterable[int], amount: int) -> tuple[int, list[int]]:
    """自底向上填表，返回最少枚数和一组最优组合。"""
    values = _validate_coins(coins, amount)
    inf = amount + 1
    best = [inf] * (amount + 1)
    choice: list[int | None] = [None] * (amount + 1)
    best[0] = 0
    for current in range(1, amount + 1):
        for coin in values:
            if coin <= current and best[current - coin] + 1 < best[current]:
                best[current] = best[current - coin] + 1
                choice[current] = coin
    if best[amount] == inf:
        raise ValueError(f"面额 {values} 无法凑出 {amount}")
    chosen: list[int] = []
    current = amount
    while current:
        coin = choice[current]
        assert coin is not None
        chosen.append(coin)
        current -= coin
    return best[amount], sorted(chosen, reverse=True)


def build_dp_table(items: Sequence[Item], capacity: int) -> list[list[int]]:
    """构造 0-1 背包完整二维表，容量按每 1g 计算。"""
    if capacity < 0:
        raise ValueError("容量不能为负数")
    table = [[0] * (capacity + 1) for _ in range(len(items) + 1)]
    for i, item in enumerate(items, start=1):
        previous = table[i - 1]
        current = table[i]
        for c in range(capacity + 1):
            current[c] = previous[c]
            if c >= item.weight:
                current[c] = max(current[c], previous[c - item.weight] + item.value)
    return table


def knapsack_01(
    items: Sequence[Item], capacity: int, must_include: Sequence[int] = ()
) -> KnapsackResult:
    """求至多装满的 0-1 背包，并支持少量必投项目约束。"""
    required = sorted(set(must_include))
    if any(i < 0 or i >= len(items) for i in required):
        raise IndexError("必投项目索引超出范围")
    required_weight = sum(items[i].weight for i in required)
    required_value = sum(items[i].value for i in required)
    if required_weight > capacity:
        raise ValueError("必投项目已经超过容量")

    if required:
        remaining_items = [(original, item) for original, item in enumerate(items) if original not in required]
        sub_result = knapsack_01([item for _, item in remaining_items], capacity - required_weight)
        selected = required + [remaining_items[i][0] for i in sub_result.selected_indices]
        selected.sort()
        return KnapsackResult(
            value=required_value + sub_result.value,
            weight=required_weight + sub_result.weight,
            selected_indices=selected,
            dp=None,
        )

    table = build_dp_table(items, capacity)
    selected: list[int] = []
    c = capacity
    for i in range(len(items), 0, -1):
        if table[i][c] != table[i - 1][c]:
            selected.append(i - 1)
            c -= items[i - 1].weight
    selected.reverse()
    total_weight = sum(items[i].weight for i in selected)
    return KnapsackResult(table[-1][-1], total_weight, selected, table)


def knapsack_backtracking(items: Sequence[Item], capacity: int) -> KnapsackResult:
    """枚举所有取舍组合，用于小规模数据的独立核验。"""
    best_value = 0
    best_weight = 0
    best_selected: list[int] = []

    def search(i: int, weight: int, value: int, selected: list[int]) -> None:
        nonlocal best_value, best_weight, best_selected
        if weight > capacity:
            return
        if i == len(items):
            if value > best_value or (value == best_value and weight < best_weight):
                best_value, best_weight, best_selected = value, weight, selected.copy()
            return
        search(i + 1, weight, value, selected)
        selected.append(i)
        search(i + 1, weight + items[i].weight, value + items[i].value, selected)
        selected.pop()

    search(0, 0, 0, [])
    return KnapsackResult(best_value, best_weight, best_selected)


def summarize(result: KnapsackResult, items: Sequence[Item] = ITEMS) -> dict:
    return {
        "value": result.value,
        "weight": result.weight,
        "selected": [items[i].name for i in result.selected_indices],
    }


def run_experiments() -> dict:
    coins = [1, 5, 11]
    greedy = coin_greedy(coins, 15)
    dynamic = coin_dp(coins, 15)
    result_1000 = knapsack_01(ITEMS, 1000)
    result_800 = knapsack_01(ITEMS, 800)
    result_required = knapsack_01(ITEMS, 1000, must_include=[0])
    brute = knapsack_backtracking(ITEMS, 1000)
    assert greedy[0] == 5 and dynamic[0] == 3
    assert result_1000.value == 315
    assert result_1000.selected_names == ["睡袋", "食物", "水具"]
    assert summarize(result_1000) == summarize(brute)
    return {
        "coins": {
            "greedy": {"count": greedy[0], "combination": greedy[1]},
            "dp": {"count": dynamic[0], "combination": dynamic[1]},
        },
        "knapsack": {
            "capacity_1000": summarize(result_1000),
            "capacity_800": summarize(result_800),
            "must_include_tent": summarize(result_required),
            "backtracking_check": summarize(brute),
        },
        "dp_last_row_100g": result_1000.dp[-1][::100] if result_1000.dp else [],
    }


def main() -> None:
    results = run_experiments()
    output_path = Path(__file__).with_name("results.json")
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

