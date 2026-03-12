#!/usr/bin/env python3
import glob
import json
import os
import re
from typing import Dict, List, Optional


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as file_obj:
        return file_obj.read()


def _to_null_if_missing(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    stripped = value.strip()
    return stripped if stripped else None


def parse_bgrl_log(path: str, text: str) -> Dict[str, object]:
    warnings: List[str] = []

    eval_pairs = re.findall(
        r"validation:\s*([^,\s]+),\s*test:\s*([^,\s]+)",
        text,
        flags=re.IGNORECASE,
    )

    initial_val = None
    initial_test = None
    initial_match = re.search(
        r"Initial\s+Evaluation\.\.\.[\s\S]*?validation:\s*([^,\s]+),\s*test:\s*([^,\s]+)",
        text,
        flags=re.IGNORECASE,
    )
    if initial_match:
        initial_val = _to_null_if_missing(initial_match.group(1))
        initial_test = _to_null_if_missing(initial_match.group(2))
    else:
        warnings.append("missing_initial_evaluation")

    final_val = None
    final_test = None
    if eval_pairs:
        final_val = _to_null_if_missing(eval_pairs[-1][0])
        final_test = _to_null_if_missing(eval_pairs[-1][1])
    else:
        warnings.append("missing_final_evaluation")

    best_val = None
    best_test = None
    best_valid_std = None
    best_test_std = None

    best_summary_matches = re.findall(
        r"^\[best\]\s+best_valid_acc=([^\s]+)\s+best_valid_std=([^\s]+)\s+best_test_acc=([^\s]+)\s+best_test_std=([^\s]+)",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if best_summary_matches:
        best_val = _to_null_if_missing(best_summary_matches[-1][0])
        best_valid_std = _to_null_if_missing(best_summary_matches[-1][1])
        best_test = _to_null_if_missing(best_summary_matches[-1][2])
        best_test_std = _to_null_if_missing(best_summary_matches[-1][3])

    best_patterns = [
        r"Best\s+Val(?:idation)?\s*[:=]\s*([^,\s]+).*?Test\s*[:=]\s*([^,\s]+)",
        r"best\s*val\s*[:=]\s*([^,\s]+).*?best\s*test\s*[:=]\s*([^,\s]+)",
        r"best\s*val\s*[:=]\s*([^,\s]+).*?test\s*[:=]\s*([^,\s]+)",
    ]

    if best_val is None and best_test is None:
        for pattern in best_patterns:
            matches = re.findall(pattern, text, flags=re.IGNORECASE | re.DOTALL)
            if matches:
                best_val = _to_null_if_missing(matches[-1][0])
                best_test = _to_null_if_missing(matches[-1][1])
                break

    if best_val is None and best_test is None:
        best_val = final_val
        best_test = final_test
        warnings.append("missing_best_metrics_used_final")

    if best_val is None:
        warnings.append("missing_best_val")
    if best_test is None:
        warnings.append("missing_best_test")

    return {
        "log_type": "bgrl",
        "source_log": path,
        "initial_val": initial_val,
        "initial_test": initial_test,
        "final_val": final_val,
        "final_test": final_test,
        "best_val": best_val,
        "best_test": best_test,
        "best_valid_std": best_valid_std,
        "best_test_std": best_test_std,
        "best_val_epoch": None,
        "test_acc": None,
        "valid_ap": None,
        "test_ap": None,
        "final_acc": None,
        "early_stopping_acc": None,
        "warning": warnings if warnings else None,
    }


def parse_graphmae_log(path: str, text: str) -> Dict[str, object]:
    warnings: List[str] = []

    valid_ap_matches = re.findall(
        r"^\[eval\]\[ogb\]\[molpcba\]\s+valid_ap=([^\s]+)",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    test_ap_matches = re.findall(
        r"^\[eval\]\[ogb\]\[molpcba\]\s+test_ap=([^\s]+)",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )

    if valid_ap_matches or test_ap_matches:
        valid_ap = _to_null_if_missing(valid_ap_matches[-1]) if valid_ap_matches else None
        test_ap = _to_null_if_missing(test_ap_matches[-1]) if test_ap_matches else None
        if valid_ap is None:
            warnings.append("missing_valid_ap")
        if test_ap is None:
            warnings.append("missing_test_ap")
        return {
            "log_type": "graphmae",
            "source_log": path,
            "initial_val": None,
            "initial_test": None,
            "final_val": valid_ap,
            "final_test": test_ap,
            "best_val": None,
            "best_test": None,
            "best_val_epoch": None,
            "test_acc": None,
            "valid_ap": valid_ap,
            "test_ap": test_ap,
            "final_acc": None,
            "early_stopping_acc": None,
            "early_stopping_testacc": None,
            "warning": warnings if warnings else None,
        }

    summary_matches = re.findall(
        r"---\s*TestAcc:\s*([^,\s]+),\s*early-stopping-TestAcc:\s*([^,\s]+),\s*Best\s+ValAcc:\s*([^\s]+)\s+in\s+epoch\s+([^\s]+)\s*---",
        text,
        flags=re.IGNORECASE,
    )

    test_acc = None
    early_stopping_testacc = None
    best_val = None
    best_val_epoch = None
    if summary_matches:
        test_acc = _to_null_if_missing(summary_matches[-1][0])
        early_stopping_testacc = _to_null_if_missing(summary_matches[-1][1])
        best_val = _to_null_if_missing(summary_matches[-1][2])
        best_val_epoch = _to_null_if_missing(summary_matches[-1][3])
    else:
        warnings.append("missing_best_summary_line")

    epoch_metric_matches = re.findall(
        r"#\s*Epoch:\s*(\d+),[\s\S]*?val_acc:\s*([0-9eE+\-.]+),[\s\S]*?test_acc:\s*([0-9eE+\-.]+)",
        text,
        flags=re.IGNORECASE,
    )
    epoch_metrics: Dict[str, Dict[str, str]] = {}
    for epoch_id, val_acc, test_value in epoch_metric_matches:
        epoch_metrics[epoch_id] = {
            "val_acc": val_acc,
            "test_acc": test_value,
        }

    best_test = None
    if best_val_epoch is not None and best_val is not None:
        best_epoch_metrics = epoch_metrics.get(best_val_epoch)
        if best_epoch_metrics is not None:
            epoch_val_str = best_epoch_metrics.get("val_acc")
            epoch_test_str = best_epoch_metrics.get("test_acc")
            if epoch_val_str is not None and epoch_test_str is not None:
                try:
                    epoch_val_float = float(epoch_val_str)
                    best_val_float = float(best_val)
                    if abs(epoch_val_float - best_val_float) <= 5e-4:
                        best_test = _to_null_if_missing(epoch_test_str)
                except ValueError:
                    best_test = None

    if best_test is None:
        warnings.append("best_test_missing")

    if test_acc is not None and best_test is None:
        best_test = test_acc
        warnings = [
            "best_test_inferred_from_summary_testacc" if item == "best_test_missing" else item
            for item in warnings
        ]

    final_acc_matches = re.findall(r"#\s*final_acc:\s*([^\s]+)", text, flags=re.IGNORECASE)
    final_acc = _to_null_if_missing(final_acc_matches[-1]) if final_acc_matches else None
    if final_acc is None:
        warnings.append("missing_final_acc")

    early_stopping_acc_matches = re.findall(
        r"#\s*early-stopping_acc:\s*([^\s]+)",
        text,
        flags=re.IGNORECASE,
    )
    early_stopping_acc = (
        _to_null_if_missing(early_stopping_acc_matches[-1]) if early_stopping_acc_matches else None
    )
    if early_stopping_acc is None:
        warnings.append("missing_early_stopping_acc")

    if best_val is None:
        warnings.append("missing_best_val")
    if best_val_epoch is None:
        warnings.append("missing_best_val_epoch")
    if test_acc is None:
        warnings.append("missing_test_acc")

    return {
        "log_type": "graphmae",
        "source_log": path,
        "initial_val": None,
        "initial_test": None,
        "final_val": None,
        "final_test": None,
        "best_val": best_val,
        "best_test": best_test,
        "best_val_epoch": best_val_epoch,
        "test_acc": test_acc,
        "valid_ap": None,
        "test_ap": None,
        "final_acc": final_acc,
        "early_stopping_acc": early_stopping_acc,
        "early_stopping_testacc": early_stopping_testacc,
        "warning": warnings if warnings else None,
    }


def _format_cell(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, list):
        return "; ".join(str(item) for item in value) if value else "null"
    return str(value)


def _print_markdown_table(rows: List[Dict[str, object]]) -> None:
    headers = [
        "log_type",
        "source_log",
        "initial_val",
        "initial_test",
        "final_val",
        "final_test",
        "best_val",
        "best_test",
        "best_val_epoch",
        "test_acc",
        "valid_ap",
        "test_ap",
        "early_stopping_testacc",
        "final_acc",
        "early_stopping_acc",
        "warning",
    ]

    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        print("| " + " | ".join(_format_cell(row.get(header)) for header in headers) + " |")


def main() -> None:
    bgrl_logs = sorted(glob.glob("logs/bgrl/*.log"))
    graphmae_logs = sorted(glob.glob("logs/graphmae/*.log"))

    results: List[Dict[str, object]] = []

    for log_path in bgrl_logs:
        text = _read_text(log_path)
        results.append(parse_bgrl_log(log_path, text))

    for log_path in graphmae_logs:
        text = _read_text(log_path)
        results.append(parse_graphmae_log(log_path, text))

    _print_markdown_table(results)

    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/summary.json"
    with open(out_path, "w", encoding="utf-8") as file_obj:
        json.dump(results, file_obj, indent=2, ensure_ascii=False)

    print(f"\nWrote {len(results)} records to {out_path}")


if __name__ == "__main__":
    main()
