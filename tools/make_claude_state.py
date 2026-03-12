#!/usr/bin/env python3
"""
Generate notes/claude/CLAUDE_STATE.md — rolling Claude Project snapshot.

Usage:
    python tools/make_claude_state.py

Rules:
- Never invent metrics; only read from outputs/*.json, outputs/meeting_table.md, logs/*.
- Output <= 150 lines.
- Chinese explanations; paths/commands/filenames stay in English.
"""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_FILE = REPO_ROOT / "notes" / "claude" / "CLAUDE_STATE.md"

SBERT_JSON = REPO_ROOT / "outputs" / "sbert_only_arxiv.json"
SUMMARY_JSON = REPO_ROOT / "outputs" / "summary.json"
MEETING_MD = REPO_ROOT / "outputs" / "meeting_table.md"
LOG_DIRS = [
    REPO_ROOT / "logs" / "graphmae",
    REPO_ROOT / "logs" / "baseline",
]
MAX_OUTPUT_LINES = 150


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def warn_missing(path: Path) -> str:
    print(f"missing: {path}", file=sys.stderr)
    return f"_missing: {path.relative_to(REPO_ROOT)}_"


def read_json(path: Path):
    if not path.exists():
        warn_missing(path)
        return None
    with path.open() as f:
        return json.load(f)


def recent_logs(log_dirs: list[Path], n: int = 5) -> list[Path]:
    """Return the n most recently modified log files across all given dirs."""
    files: list[Path] = []
    for d in log_dirs:
        if d.exists():
            files.extend(f for f in d.iterdir() if f.is_file() and f.suffix == ".log")
        else:
            print(f"missing: {d}", file=sys.stderr)
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[:n]


def grep_log(path: Path, pattern: str, max_lines: int = 20) -> list[str]:
    rx = re.compile(pattern)
    hits: list[str] = []
    try:
        with path.open(errors="replace") as f:
            for line in f:
                if rx.search(line):
                    hits.append(line.rstrip())
                    if len(hits) >= max_lines:
                        break
    except OSError as e:
        hits.append(f"(read error: {e})")
    return hits


def newest_arxiv_graphmae_log(log_dir: Path) -> Path | None:
    if not log_dir.exists():
        return None
    candidates = sorted(
        (f for f in log_dir.iterdir()
         if f.is_file() and "arxiv" in f.name and f.suffix == ".log"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


# ---------------------------------------------------------------------------
# section builders
# ---------------------------------------------------------------------------

def section_header() -> list[str]:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return [
        "# CLAUDE_STATE.md",
        f"<!-- 自动生成时间：{ts} | 仅限 Claude Project 上传使用，禁止手动捏造数据 -->",
        "",
        f"**repo root:** `{REPO_ROOT}`  ",
        f"**generated:** {ts}",
        "",
    ]


def section_sbert(data) -> list[str]:
    lines = ["## SBERT-only Baseline (ogbn-arxiv)", ""]
    if data is None:
        lines.append(warn_missing(SBERT_JSON))
        lines.append("")
        return lines

    s = data.get("summary", {})
    vm = s.get("val_acc_mean")
    vs = s.get("val_acc_std")
    tm = s.get("test_acc_at_best_val_mean")
    ts_ = s.get("test_acc_at_best_val_std")

    def fmt(v):
        return f"{v:.4f}" if v is not None else "null"

    lines.append(f"来源：`outputs/sbert_only_arxiv.json`")
    lines.append(f"- val_mean±std  = {fmt(vm)} ± {fmt(vs)}")
    lines.append(f"- test_mean±std = **{fmt(tm)} ± {fmt(ts_)}**")
    lines.append("")

    lines.append("| seed | best_val | best_test | best_epoch |")
    lines.append("|------|----------|-----------|------------|")
    for r in data.get("seed_results", []):
        lines.append(
            f"| {r['seed']} | {fmt(r.get('best_val_acc'))} "
            f"| {fmt(r.get('best_test_acc_at_best_val'))} "
            f"| {r.get('best_epoch', 'null')} |"
        )
    lines.append("")
    return lines


def section_summary(data) -> list[str]:
    lines = ["## outputs/summary.json — GraphMAE / BGRL runs", ""]
    if data is None:
        lines.append(warn_missing(SUMMARY_JSON))
        lines.append("")
        return lines

    lines.append("来源：`outputs/summary.json`（仅显示有效 best_val 条目）")
    lines.append("")
    lines.append("BGRL 最新状态：arXiv 路径已完成 smoke/logging 级验证，日志已出现 `[best] best_valid_acc=... best_valid_std=... best_test_acc=... best_test_std=...`，parser 已支持显式提取该 best 行；历史日志中的 `missing_best_metrics_used_final` 仅表示旧回退行为。")
    lines.append("注意：该状态更新仅代表 logging/parser 路径修复，不等同于官方 full rerun 结果刷新。")
    lines.append("")
    lines.append("| model | log (basename) | best_val | best_test | epoch | warning |")
    lines.append("|-------|---------------|----------|-----------|-------|---------|")

    def fmt(v):
        return str(v) if v is not None else "null"

    for entry in data:
        log_base = Path(entry.get("source_log", "")).name
        bv = fmt(entry.get("best_val"))
        bt = fmt(entry.get("best_test"))
        ep = fmt(entry.get("best_val_epoch"))
        lt = entry.get("log_type", "?")
        w = ",".join(entry.get("warning") or []) or "-"
        # keep warning short
        if len(w) > 40:
            w = w[:37] + "..."
        lines.append(f"| {lt} | {log_base} | {bv} | {bt} | {ep} | {w} |")

    lines.append("")
    return lines


def section_meeting(md_path: Path) -> list[str]:
    lines = ["## outputs/meeting_table.md (first 30 lines)", ""]
    if not md_path.exists():
        lines.append(warn_missing(md_path))
        lines.append("")
        return lines
    with md_path.open(errors="replace") as f:
        rows = [l.rstrip() for _, l in zip(range(30), f)]
    lines.extend(rows)
    lines.append("")
    return lines


def section_recent_logs() -> list[str]:
    lines = ["## 最近修改的日志（logs/graphmae + logs/baseline，最多5条）", ""]
    recent = recent_logs(LOG_DIRS, n=5)
    if not recent:
        lines.append("_no log files found_")
        lines.append("")
        return lines

    lines.append("| file | size | mtime |")
    lines.append("|------|------|-------|")
    for f in recent:
        rel = f.relative_to(REPO_ROOT)
        sz = f.stat().st_size
        mt = datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        lines.append(f"| {rel} | {sz:,}B | {mt} |")
    lines.append("")

    # grep newest arXiv graphmae log
    newest = newest_arxiv_graphmae_log(REPO_ROOT / "logs" / "graphmae")
    if newest:
        rel = newest.relative_to(REPO_ROOT)
        lines.append(f"### Grep: `{rel}`")
        lines.append("")
        pattern = r"\[feat_pt\]|\[best\]|final_acc|val_acc|test_acc"
        hits = grep_log(newest, pattern, max_lines=20)
        if hits:
            lines.append("```")
            lines.extend(hits)
            lines.append("```")
        else:
            lines.append("_no matching lines_")
        lines.append("")
    return lines


def section_todo() -> list[str]:
    return [
        "## Current TODO（优先级排序）",
        "",
        "1. **[P0]** `repos/graphmae/main_transductive.py` — "
        "`[best]` 行输出（GraphMAE 专项；BGRL arXiv の `[best]` 打印与 parser 显式解析已通过 smoke/logging 检查确认，不适用于本 P0）。",
        "   `best_tracker[\"best_epoch\"]` 在历史 locked logs 中示为 -1；确认 `_EpochMetricTee` 正确捕获 `node_classification_evaluation()` 内部的 epoch 行。",
        "2. **[P1]** 增大 `--max_epoch` / `--max_epoch_f`（当前 best_epoch=29 = last epoch，模型未收敛）。",
        "   用修复后代码重跑 arXiv seeds 0/1/2（locked512 参数），更新 `outputs/summary.json`。",
        "3. **[P2]** 公平比较 GraphMAE+SBERT vs SBERT-only baseline（`outputs/sbert_only_arxiv.json`）。",
        "4. **[P3]** PCBA eval 协议定义（OGB AP evaluator smoke 已在 ogbg-molpcba 跑通；latest sub5000 smoke: valid_ap=0.0374, test_ap=0.0424；仅为 smoke success，locked run 仍待完成）。",
        "5. **[P4]** WN18RR 链路预测 eval 协议定义（MRR/Hits@K，当前 TBD）。",
        "",
    ]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def build_state() -> list[str]:
    sbert_data = read_json(SBERT_JSON)
    summary_data = read_json(SUMMARY_JSON)

    sections: list[list[str]] = [
        section_header(),
        section_sbert(sbert_data),
        section_summary(summary_data),
        section_meeting(MEETING_MD),
        section_recent_logs(),
        section_todo(),
    ]

    lines: list[str] = []
    for s in sections:
        lines.extend(s)

    # enforce <= MAX_OUTPUT_LINES
    if len(lines) > MAX_OUTPUT_LINES:
        lines = lines[:MAX_OUTPUT_LINES - 2]
        lines.append("")
        lines.append(f"_[truncated to {MAX_OUTPUT_LINES} lines]_")

    return lines


def main() -> None:
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    lines = build_state()
    OUT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Generated {OUT_FILE.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
