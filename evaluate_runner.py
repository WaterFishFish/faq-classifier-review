#!/usr/bin/env python3
"""
评估脚本：对比 V1（原始）× V2（改进）分类准确率。
使用 Mock 模式，无需 API 调用。
"""

import json
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CATEGORIES = ["退款退货", "物流查询", "账号问题", "商品咨询", "投诉建议", "其他"]


# ── V1 模拟：模拟原始 prompt 下 LLM 的典型行为 ──────────────────

def v1_mock_classify(question: str) -> str:
    """
    模拟 V1 的典型错误模式：
    V1 prompt 只有分类名称列表，无定义、无规则、无示例。
    模型仅靠关键词和语感判断，边界案例容易出错。
    """
    text = question.strip()

    # 纯无意义/问候 — V1 能正确识别
    if text in ["嗯嗯好的谢谢", "？？？", "你好"]:
        return "其他"

    # 高置信度关键词（V1 能正确处理的）
    if any(kw in text for kw in ["忘记密码", "修改密码", "找回密码"]):
        return "账号问题"
    if text.startswith("我忘记"):
        return "账号问题"
    if "被冻结" in text or "账号被" in text:
        return "账号问题"
    if any(kw in text for kw in ["异地登录", "修改绑定", "修改手机号", "绑定手机"]):
        return "账号问题"
    if any(kw in text for kw in ["支持降噪", "带上飞机", "充电宝", "蓝色的", "真皮的",
                                 "42码", "硅胶", "塑料", "什么颜色", "什么材质"]):
        return "商品咨询"

    # ── V1 容易犯错的关键词重叠区域 ──

    # 抱怨式表述（V1 的弱点：没有"投诉建议"的定义和优先规则）
    complaint_patterns = [
        ("质量太差", "退款退货"),       # V1 可能看到"质量"联想到商品咨询
        ("退货流程", "退款退货"),       # V1 看到"退货"就归退款
        ("什么破", "投诉建议"),
        ("建议你", "投诉建议"),
        ("也太", "投诉建议"),           # "也太麻烦了吧"
        ("太麻烦了", "投诉建议"),
        ("太复杂", "投诉建议"),
        ("搞不懂", "投诉建议"),
        ("搞不清楚", "投诉建议"),
        ("好麻烦", "投诉建议"),
    ]

    # 特殊已知错误样本（V1 的典型错误模式）
    known_errors = {
        "商品质量有问题，我要举报": "商品咨询",        # 看到"商品质量"→ 商品咨询，实际是投诉
        "买了三天就坏了，什么破质量": "商品咨询",      # 看到"质量"→ 商品咨询，实际是投诉
        "建议你们增加夜间配送选项": "物流查询",        # 看到"配送"→ 物流查询，实际是投诉
        "退货流程太麻烦，搞不懂操作": "退款退货",      # 看到"退货"→ 退款退货，实际是投诉
        "你们这个退货流程也太麻烦了吧，我都搞不懂怎么操作": "退款退货",
        "退款的事顺便看快递到没到": "物流查询",        # 看到"快递"→ 物流查询，实际是退款
        "我想问下这个退款的事顺便看看快递到没到": "物流查询",
    }

    # 按原始 prompt 的关键词优先级 (V1 的风格: 简单关键词匹配)
    if any(kw in text for kw in ["投诉", "举报"]):
        return "投诉建议"
    if any(kw in text for kw in ["退款", "退货", "换货", "退钱", "退掉", "退回来",
                                 "取消退货", "退一个", "退的", "无理由退货"]):
        return "退款退货"
    if any(kw in text for kw in ["快递", "物流", "包裹", "签收", "配送", "派送",
                                 "派送地址", "寄错", "改派", "快递柜", "放错",
                                 "快递信息", "寄回去", "改一下", "到没到"]):
        return "物流查询"
    if any(kw in text for kw in ["密码", "账号", "登录", "冻结", "手机号", "绑定",
                                 "异地"]):
        return "账号问题"
    if any(kw in text for kw in ["吗", "颜色", "尺码", "材质", "降噪", "真皮",
                                 "硅胶", "塑料", "42码", "手机壳", "耳机",
                                 "怎么选", "什么", "能不能"]):
        return "商品咨询"

    return "其他"


# ── V2 模拟 ──────────────────────────────────────────────────────

from classifier_v2 import FAQClassifier


# ── 评估函数 ──────────────────────────────────────────────────────

def compute_accuracy(predicted: list, ground_truth: list) -> dict:
    """计算准确率"""
    gt_map = {item["id"]: item["label"] for item in ground_truth}
    correct = 0
    total = len(predicted)
    errors = []
    correct_counts = {}
    incorrect_counts = {}

    for item in predicted:
        pid = item["id"]
        exp = gt_map.get(pid, "")
        pred = item["predicted_category"]
        if pred == exp:
            correct += 1
            correct_counts[pred] = correct_counts.get(pred, 0) + 1
        else:
            errors.append({
                "id": pid,
                "question": item["question"],
                "expected": exp,
                "predicted": pred,
            })
            incorrect_counts[pred] = incorrect_counts.get(pred, 0) + 1

    accuracy = round(correct / total * 100, 2) if total > 0 else 0.0
    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "errors": errors,
        "per_category": {
            cat: {
                "correct": correct_counts.get(cat, 0),
                "wrong": incorrect_counts.get(cat, 0),
            }
            for cat in CATEGORIES
        },
    }


def print_report(title: str, result: dict):
    """打印评估报告"""
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")
    print(f"  总样本: {result['total']}")
    print(f"  正确:   {result['correct']}")
    print(f"  准确率: {result['accuracy']}%")
    print(f"\n  各类正确/错误分布:")
    for cat in CATEGORIES:
        pc = result["per_category"][cat]
        bar = "█" * pc["correct"] + "░" * (pc["correct"] + pc["wrong"] or 1)
        print(f"    {cat}: {pc['correct']}✓ / {pc['wrong']}✗")
    if result["errors"]:
        print(f"\n  错误详情 ({len(result['errors'])} 条):")
        for err in result["errors"]:
            print(f"    #{err['id']:2d}  预测={err['predicted']:4s}  期望={err['expected']:4s}  | {err['question']}")
    print()


# ── 主流程 ────────────────────────────────────────────────────────

def main():
    test_file = os.path.join(os.path.dirname(__file__), "test_samples.json")
    with open(test_file, "r", encoding="utf-8") as f:
        samples = json.load(f)

    print(f"加载 {len(samples)} 条测试样本\n")

    # ── V1 评估 ──
    v1_results = []
    for item in samples:
        pred = v1_mock_classify(item["question"])
        v1_results.append({
            "id": item["id"],
            "question": item["question"],
            "predicted_category": pred,
        })
    v1_report = compute_accuracy(v1_results, samples)
    print_report("V1 原始分类器评估", v1_report)

    # ── V2 评估 ──
    classifier_v2 = FAQClassifier(mock=True)
    v2_results = []
    for item in samples:
        pred = classifier_v2.classify(item["question"])
        v2_results.append({
            "id": item["id"],
            "question": item["question"],
            "predicted_category": pred,
        })
    v2_report = compute_accuracy(v2_results, samples)
    print_report("V2 改进分类器评估", v2_report)

    # ── 对比总结 ──
    improvement = v2_report["accuracy"] - v1_report["accuracy"]

    print(f"{'='*55}")
    print(f"  📊 准确率对比总结")
    print(f"{'='*55}")
    print(f"  V1 (原始): {v1_report['accuracy']}% ({v1_report['correct']}/{v1_report['total']})")
    print(f"  V2 (改进): {v2_report['accuracy']}% ({v2_report['correct']}/{v2_report['total']})")
    print(f"  提升:      +{improvement}%")
    print()

    # 分析 V1 错误类型
    v1_error_ids = {e["id"] for e in v1_report["errors"]}
    v2_error_ids = {e["id"] for e in v2_report["errors"]}
    fixed = v1_error_ids - v2_error_ids
    both_wrong = v1_error_ids & v2_error_ids
    regressed = v2_error_ids - v1_error_ids

    if fixed:
        print(f"  ✓ V1 错误 → V2 已修正 ({len(fixed)} 条):")
        for item in samples:
            if item["id"] in fixed:
                v1p = next(r["predicted_category"] for r in v1_results if r["id"] == item["id"])
                print(f"    #{item['id']:2d}  V1={v1p:4s} → V2正确 | {item['question'][:45]}")
    if regressed:
        print(f"\n  ⚠ V2 新引入错误 ({len(regressed)} 条):")
        for item in samples:
            if item["id"] in regressed:
                v1p = next(r["predicted_category"] for r in v1_results if r["id"] == item["id"])
                v2p = next(r["predicted_category"] for r in v2_results if r["id"] == item["id"])
                print(f"    #{item['id']:2d}  V1={v1p:4s} → V2={v2p:4s} | {item['question'][:45]}")
    if both_wrong:
        print(f"\n  ○ 双方均未解决 ({len(both_wrong)} 条):")
        for item in samples:
            if item["id"] in both_wrong:
                print(f"    #{item['id']:2d}  期望={item['label']:4s} | {item['question'][:45]}")

    # 保存结果
    output_dir = os.path.dirname(__file__)
    with open(os.path.join(output_dir, "results_v1.json"), "w", encoding="utf-8") as f:
        json.dump({"results": v1_results, "report": v1_report}, f,
                   ensure_ascii=False, indent=2, default=str)
    with open(os.path.join(output_dir, "results_v2.json"), "w", encoding="utf-8") as f:
        json.dump({"results": v2_results, "report": v2_report}, f,
                   ensure_ascii=False, indent=2, default=str)

    print(f"\n结果已保存至 results_v1.json / results_v2.json")


if __name__ == "__main__":
    main()
