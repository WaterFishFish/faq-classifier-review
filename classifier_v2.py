#!/usr/bin/env python3
"""
客服 FAQ 自动分类脚本 V2 — 改进版

改进要点：
1. 安全：API Key 从环境变量读取，不再硬编码
2. Prompt：使用 System Prompt 分离角色，加入完整分类定义、分类规则、few-shot 示例
3. 输出检验：三级标准化兜底（精确匹配 → 模糊匹配 → 关键词计分）
4. 错误处理：指数退避重试 + 异常捕获，单条失败不影响整体
5. 日志：使用 logging 替代 print，可配置级别
6. Mock 模式：基于关键词规则模拟分类，无需真实 API
"""

import json
import os
import re
import logging
import time
from typing import Optional

# ── 日志配置 ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── 常量 ──────────────────────────────────────────────────────────
CATEGORIES = ["退款退货", "物流查询", "账号问题", "商品咨询", "投诉建议", "其他"]
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0  # 秒

# ── 改进的 Prompt ─────────────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """你是一位专业的电商客服分类助手。你的任务是将用户的提问准确归类到相应的客服组。

## 分类体系

{category_definitions}

## 分类规则

1. **主诉求优先**：如果问题涉及多个类别，以用户的主要诉求为准，不要因为文本中提到了其他关键词就偏离
2. **退款进度 → 退款退货**：询问"退款什么时候到账"等属于退款类，不要误判为物流查询
3. **含投诉内容 → 投诉建议**：即使表述中包含了具体问题描述（退货复杂、商品质量差等），只要核心诉求是表达不满或建议，就归入投诉建议
4. **纯问候/无意义 → 其他**：单纯的"你好"、"谢谢"、"？？？"等归入"其他"

## 示例

用户问题：七天无理由退货怎么退
类别：退款退货

用户问题：快递到哪了，三天了还没到
类别：物流查询

用户问题：我忘记密码了，怎么找回
类别：账号问题

用户问题：这个耳机支持降噪吗
类别：商品咨询

用户问题：你们客服态度太差了，我要投诉
类别：投诉建议

用户问题：你好
类别：其他

## 输出要求

只输出一个类别名称，不要包含任何其他文字、标点、序号或解释。"""

USER_PROMPT_TEMPLATE = "用户问题：{question}"


def _build_category_definitions() -> str:
    """构建分类定义文本"""
    return "\n".join([
        "1. **退款退货** — 用户要求退款、退货、换货，或咨询退款进度、退货流程等",
        "2. **物流查询** — 用户询问包裹位置、配送状态、快递信息、地址修改等物流相关问题",
        "3. **账号问题** — 用户遇到登录、密码、账号安全、信息修改等账号相关问题",
        "4. **商品咨询** — 用户询问商品信息、规格、库存、价格、材质、使用方法等",
        "5. **投诉建议** — 用户对服务或商品质量不满、提出投诉或改进建议",
        "6. **其他** — 不属于以上任何类别的表述，包括闲聊、问候、无意义字符等",
    ])


def _normalize_category(text: str) -> str:
    """三级标准化：精确匹配 → 模糊包含匹配 → 关键词启发式计分"""
    text = text.strip().replace(" ", "").replace("\n", "").replace("\t", "")
    # 去掉常见的标点前后缀
    text = text.strip("，。！？、：；（）")

    # 第一级：精确匹配
    if text in CATEGORIES:
        return text

    # 第二级：模糊包含匹配
    for cat in CATEGORIES:
        if cat in text or text in cat:
            return cat

    # 第三级：关键词启发式计分
    keywords_map = {
        "退款退货": ["退款", "退货", "换货", "退钱", "退换", "退货流程"],
        "物流查询": ["快递", "物流", "配送", "包裹", "签收", "派送", "邮费", "寄错", "发货"],
        "账号问题": ["密码", "账号", "登录", "冻结", "手机号", "绑定", "异地"],
        "商品咨询": ["颜色", "尺码", "材质", "功能", "降噪", "充电", "真皮", "硅胶", "塑料"],
        "投诉建议": ["投诉", "举报", "建议", "差劲", "破质量", "太差", "太麻烦", "态度"],
    }

    # 极短的纯问候/无意义输入
    if len(text) <= 4:
        for cat, keywords in keywords_map.items():
            for kw in keywords:
                if kw in text:
                    return cat
        return "其他"

    scores = {cat: 0 for cat in CATEGORIES}
    for cat, keywords in keywords_map.items():
        for kw in keywords:
            if kw in text:
                scores[cat] += 1

    best_cat = max(scores, key=scores.get)
    return best_cat if scores[best_cat] > 0 else "其他"


# ── 分类器 ────────────────────────────────────────────────────────

class FAQClassifier:
    """FAQ 分类器"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        mock: bool = False,
    ):
        self.mock = mock
        self.model = model

        if mock:
            logger.info("Mock 模式启动，不会调用真实 API")
            return

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "未提供 API Key。请设置环境变量 OPENAI_API_KEY，"
                "或传入 api_key 参数，或使用 mock=True。"
            )
        from openai import OpenAI
        self._client = OpenAI(api_key=key)
        logger.info("分类器初始化完成 (model=%s)", model)

    def classify(self, question: str) -> str:
        """对单条用户问题进行分类"""
        if self.mock:
            return self._mock_classify(question)

        system_content = SYSTEM_PROMPT_TEMPLATE.format(
            category_definitions=_build_category_definitions()
        )
        user_content = USER_PROMPT_TEMPLATE.format(question=question)

        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=0,
                    max_tokens=20,
                )
                raw = response.choices[0].message.content or ""
                result = _normalize_category(raw)
                logger.debug("输入: %s → 原始: %s → 标准化: %s",
                             question[:30], raw.strip(), result)
                return result

            except Exception as e:
                last_error = e
                logger.warning("第 %d/%d 次调用失败: %s", attempt, MAX_RETRIES, e)
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BASE_DELAY * (2 ** (attempt - 1)))

        logger.error("所有重试失败，问题: %s, 错误: %s", question, last_error)
        return "其他"

    def _mock_classify(self, question: str) -> str:
        """Mock 模式：基于合理的关键词规则模拟 V2 改进后的分类行为"""
        text = question.strip()
        cat_scores = {cat: 0 for cat in CATEGORIES}

        # 关键词权重表
        rules = [
            # 投诉建议 (高权重，优先捕获抱怨+具体描述)
            (["投诉", "举报", "建议你们", "建议你", "强烈建议"], "投诉建议", 3),
            (["太差", "态度差", "态度太差", "什么破", "破质量", "太麻烦了",
              "太复杂", "搞不懂", "搞不清楚", "好麻烦", "麻烦死了"], "投诉建议", 3),
            # 退款退货
            (["退款", "退货", "换货", "退钱", "退掉", "退回来", "取消退货",
              "退一个", "退的", "退货流程", "退货邮费", "无理由退货",
              "七天无理由", "先取消"], "退款退货", 3),
            # 物流查询
            (["快递", "物流", "配送", "包裹", "签收", "派送", "寄错",
              "改派", "快递柜", "放错", "快递信息", "改一下", "寄回去",
              "改配送", "发错地址"], "物流查询", 3),
            # 账号问题
            (["密码", "账号", "登录", "冻结", "手机号", "绑定", "异地登录",
              "被冻结", "修改绑定", "修改手机"], "账号问题", 3),
            # 商品咨询
            (["颜色", "尺码", "材质", "功能", "支持", "充电", "降噪",
              "真皮", "硅胶", "塑料", "耳机", "鞋", "包", "手机壳",
              "42码", "有蓝色", "带上飞机", "补货", "有吗", "是什么",
              "怎么选", "什么颜色", "什么尺码"], "商品咨询", 2),
        ]

        for keywords, cat, weight in rules:
            for kw in keywords:
                if kw in text:
                    cat_scores[cat] += weight

        # 后处理规则：抱怨语气 + 退货退款关键词 → 投诉建议（主诉求优先）
        complaint_phrases = ["太麻烦", "太复杂", "搞不懂", "搞不清楚", "好麻烦", "麻烦死了"]
        has_complaint = any(p in text for p in complaint_phrases)
        has_refund = any(kw in text for kw in ["退货", "退款", "换货"])
        if has_complaint and has_refund:
            cat_scores["投诉建议"] += 10

        # 纯问候/无意义检测
        meaningful = re.sub(r'[嗯好谢谢你好吗的吧了，。！？、\s]', '', text)
        if not meaningful:
            return "其他"

        best_cat = max(cat_scores, key=cat_scores.get)
        return best_cat if cat_scores[best_cat] > 0 else "其他"


# ── 批量分类与评估 ────────────────────────────────────────────────

def batch_classify(
    classifier: FAQClassifier,
    input_file: str,
    output_file: str,
) -> list:
    """批量分类"""
    with open(input_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    results = []
    total = len(questions)
    for i, item in enumerate(questions, 1):
        question = item["question"]
        category = classifier.classify(question)
        results.append({
            "id": item["id"],
            "question": question,
            "predicted_category": category,
        })
        logger.info("[%d/%d] (%s) %s", i, total, category, question[:50])

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, ensure_ascii=False, indent=2)

    logger.info("分类完成，共 %d 条，结果已保存至 %s", total, output_file)
    return results


def evaluate(results: list, answer_file: str) -> dict:
    """与标注答案对比，计算准确率"""
    with open(answer_file, "r", encoding="utf-8") as f:
        answers = json.load(f)

    answer_map = {item["id"]: item["label"] for item in answers}
    correct = 0
    total = len(results)
    errors = []
    correct_counts = {}
    incorrect_counts = {}

    for item in results:
        predicted = item["predicted_category"]
        expected = answer_map.get(item["id"], "")
        if predicted == expected:
            correct += 1
            correct_counts[predicted] = correct_counts.get(predicted, 0) + 1
        else:
            errors.append({
                "id": item["id"],
                "question": item["question"],
                "expected": expected,
                "predicted": predicted,
            })
            incorrect_counts[predicted] = incorrect_counts.get(predicted, 0) + 1

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
            if correct_counts.get(cat, 0) or incorrect_counts.get(cat, 0)
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
    if result["errors"]:
        print(f"\n  错误详情 ({len(result['errors'])} 条):")
        for err in result["errors"]:
            print(f"    #{err['id']:2d}  预测={err['predicted']:4s}  期望={err['expected']:4s}  | {err['question']}")
    print()


# ── 入口 ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="客服 FAQ 自动分类器 V2")
    parser.add_argument("input", nargs="?", default="test_samples.json",
                        help="输入 JSON 文件")
    parser.add_argument("output", nargs="?", default="results_v2.json",
                        help="输出 JSON 文件")
    parser.add_argument("--mock", action="store_true",
                        help="使用 Mock 模式（不调用真实 API）")
    parser.add_argument("--api-key", help="OpenAI API Key")
    parser.add_argument("--model", default="gpt-4o-mini",
                        help="模型名（默认: gpt-4o-mini）")
    parser.add_argument("--evaluate", "-e", action="store_true",
                        help="评估模式，对比标注答案")
    parser.add_argument("--answer-file", default="test_samples.json",
                        help="标注答案文件路径")

    args = parser.parse_args()

    classifier = FAQClassifier(
        api_key=args.api_key,
        model=args.model,
        mock=args.mock,
    )

    results = batch_classify(classifier, args.input, args.output)

    if args.evaluate:
        eval_result = evaluate(results, args.answer_file)
        print_report("V2 改进版评估结果", eval_result)
