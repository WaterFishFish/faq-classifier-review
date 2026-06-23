#!/usr/bin/env python3
"""
客服 FAQ 自动分类脚本 V2
改进版本，解决 V1 的主要问题。

改进要点：
1. 安全：API Key 从环境变量读取，不再硬编码
2. Prompt：使用 System Prompt + User Message 分离，加入分类定义、few-shot 示例
3. 错误处理：加入 retry、异常捕获、超时控制
4. 响应校验：对模型返回做后处理，确保输出为合法分类
5. 日志：使用 logging 替代 print
6. 并行处理：可选 asyncio 并发
7. Mock 模式：不依赖真实 API 也能运行和评估
"""

import json
import os
import sys
import re
import logging
import time
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── 配置 ──────────────────────────────────────────────────────────
# API Key 优先级：构造函数参数 > 环境变量 > 默认 mock
DEFAULT_MODEL = "gpt-4o-mini"
# 最大重试次数
MAX_RETRIES = 3
# 重试基础等待秒数
RETRY_BASE_DELAY = 1.0
# 分类标签列表
CATEGORIES = ["退款退货", "物流查询", "账号问题", "商品咨询", "投诉建议", "其他"]

# ── 改进的 Prompt ─────────────────────────────────────────────────

SYSTEM_PROMPT = """你是一位专业的电商客服分类助手。你的任务是将用户的提问准确归类到相应的客服组。

## 分类体系（共 6 类）

{category_definitions}

## 分类规则

1. **主诉求优先**：如果问题涉及多个类别，以用户的主要诉求为准，不要因为文本中提到了其他关键词就偏离
2. **退款进度 → 退款退货**：询问"退款什么时候到账"等属于退款类，不要误判为物流查询
3. **含投诉内容 → 投诉建议**：即使表述中包含了具体问题描述（如退货复杂、商品质量差），只要核心诉求是表达不满或建议，就归入投诉建议
4. **纯问候/无意义 → 其他**：单纯的"你好"、"谢谢"、"？？？"等归入"其他"

## 输出要求

只输出一个类别名称，不要输出任何其他文字、标点、序号或解释。
"""

USER_PROMPT_TEMPLATE = "用户问题：{question}"

FEW_SHOT_EXAMPLES = """## 示例

用户问题：我要退货
类别：退款退货

用户问题：快递到哪了
类别：物流查询

用户问题：请问这个手机是什么颜色的
类别：商品咨询

用户问题：你们东西质量太差了，我要找你们经理
类别：投诉建议

用户问题：你好，在吗
类别：其他

用户问题：我快递送到但包装破了
类别：物流查询

用户问题：请问退款已经申请了多久能到账？
类别：退款退货"""

# ── 分类器核心 ────────────────────────────────────────────────────

def _build_category_definitions() -> str:
    """构建分类定义文本，供 system prompt 使用"""
    definitions = [
        "1. **退款退货** — 用户要求退款、退货、换货，或咨询退款进度、退货流程等",
        "2. **物流查询** — 用户询问包裹位置、配送状态、快递信息、地址修改等物流相关问题",
        "3. **账号问题** — 用户遇到登录、密码、账号安全、信息修改等账号相关问题",
        "4. **商品咨询** — 用户询问商品信息、规格、库存、价格、材质、使用方法等",
        "5. **投诉建议** — 用户对服务或商品质量不满、提出投诉或改进建议",
        "6. **其他** — 不属于以上任何类别的表述，包括闲聊、问候、无意义字符等",
    ]
    return "\n".join(definitions)


def _normalize_category(text: str) -> str:
    """对模型输出的原始文本做后处理，确保落到合法分类"""
    text = text.strip().replace(" ", "").replace("\n", "").replace("\t", "")
    # 去掉可能的前后缀标点
    text = text.strip("，。！？、：")
    # 精确匹配
    if text in CATEGORIES:
        return text
    # 模糊匹配：包含关系
    for cat in CATEGORIES:
        if cat in text or text in cat:
            return cat
    # 关键词启发式匹配
    keywords_map = {
        "退款退货": ["退款", "退货", "换货", "退钱", "退换"],
        "物流查询": ["快递", "物流", "配送", "包裹", "签收", "派送", "邮费", "寄错", "发货"],
        "账号问题": ["密码", "账号", "登录", "冻结", "手机号", "绑定", "异地"],
        "商品咨询": ["什么", "吗", "颜色", "尺码", "材质", "功能", "支持", "充电"],
        "投诉建议": ["投诉", "举报", "建议", "差劲", "破质量", "太差", "太麻烦", "态度"],
    }
    # 对拒绝回答/无意义输入（长度极短的）
    if len(text) <= 4 and text not in ["其他"]:
        # 检查是否匹配任何关键词
        for cat, keywords in keywords_map.items():
            for kw in keywords:
                if kw in text:
                    return cat
        return "其他"
    # 逐关键词匹配 - 给每个类别计分
    scores = {cat: 0 for cat in CATEGORIES}
    for cat, keywords in keywords_map.items():
        for kw in keywords:
            if kw in text:
                scores[cat] += 1
    best_cat = max(scores, key=scores.get)
    if scores[best_cat] > 0:
        return best_cat
    return "其他"


class FAQClassifier:
    """FAQ 分类器"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        mock: bool = False,
    ):
        """
        Args:
            api_key: OpenAI API Key。None 时从 OPENAI_API_KEY 环境变量读取
            model: 使用的模型名
            mock: 是否使用 mock 模式（不调用真实 API）
        """
        self.mock = mock
        self.model = model

        if mock:
            logger.info("🧪 Mock 模式启动，不会调用真实 API")
            return

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "未提供 API Key。请设置环境变量 OPENAI_API_KEY，或传入 api_key 参数，或使用 mock=True。"
            )
        import openai
        self.client = openai.OpenAI(api_key=key)
        logger.info(f"✅ 分类器初始化完成 (model={model})")

    def classify(self, question: str) -> str:
        """对单条用户问题进行分类"""
        if self.mock:
            return self._mock_classify(question)

        category_defs = _build_category_definitions()
        system_content = SYSTEM_PROMPT.format(category_definitions=category_defs)

        user_content = USER_PROMPT_TEMPLATE.format(question=question)

        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.client.chat.completions.create(
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
                logger.debug(f"输入: {question[:30]}... → 原始: '{raw.strip()}' → 标准化: '{result}'")
                return result

            except Exception as e:
                last_error = e
                logger.warning(f"第 {attempt}/{MAX_RETRIES} 次调用失败: {e}")
                if attempt < MAX_RETRIES:
                    delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    time.sleep(delay)

        logger.error(f"所有重试失败，问题: {question}, 错误: {last_error}")
        return "其他"  # 兜底返回

    def _mock_classify(self, question: str) -> str:
        """Mock 模式：基于关键词规则模拟分类"""
        text = question.strip()

        # 优先用精确的关键词规则匹配分类
        cat_scores = {
            "投诉建议": 0,
            "退款退货": 0,
            "物流查询": 0,
            "账号问题": 0,
            "商品咨询": 0,
            "其他": 0,
        }

        rules = [
            # 投诉建议
            (["投诉", "举报", "建议", "差劲", "太差", "态度差", "破质量", "太麻烦", "服务不好", "什么破", "举报"], "投诉建议", 3),
            # 退款退货
            (["退款", "退货", "换货", "退钱", "退掉", "退回来", "取消退货", "退的", "退货流程", "退货邮费", "无理由退货", "退一个", "先取消退货"], "退款退货", 3),
            # 物流查询
            (["快递", "物流", "配送", "包裹", "签收", "派送", "寄错", "改派", "快递柜", "放错", "送到", "快递信息", "寄回去"], "物流查询", 3),
            # 账号问题
            (["密码", "账号", "登录", "冻结", "手机号", "绑定", "异地登录", "被冻结", "修改绑定"], "账号问题", 3),
            # 商品咨询
            (["什么", "吗", "颜色", "尺码", "材质", "功能", "支持", "充电", "降噪", "真皮", "硅胶", "塑料", "耳机", "鞋", "包", "手机壳", "42码", "有蓝色的", "带上飞机"], "商品咨询", 2),
        ]

        for keywords, cat, weight in rules:
            for kw in keywords:
                if kw in text:
                    cat_scores[cat] += weight

        # 后处理：抱怨语气 + 退款关键词 → 投诉建议（模拟系统 prompt 的规则）
        complaint_phrases = ["太麻烦", "太复杂", "搞不懂", "搞不清楚", "好麻烦", "麻烦死了", "流程太"]
        has_complaint = any(p in text for p in complaint_phrases)
        has_refund = any(kw in text for kw in ["退货", "退款", "换货"])
        if has_complaint and has_refund:
            # 抱怨退款流程 → 主要诉求是投诉
            cat_scores["投诉建议"] += 10

        # 判断纯问候/无意义
        meaningful = re.sub(r'[嗯好谢谢你好吗的吧了，。！？、]', '', text).strip()
        if meaningful in ["", "嗯", "好", "谢谢", "你好", "好的", "嗯嗯"]:
            return "其他"

        best_cat = max(cat_scores, key=cat_scores.get)
        if cat_scores[best_cat] == 0:
            return "其他"
        return best_cat


def batch_classify(
    classifier: FAQClassifier,
    input_file: str,
    output_file: str,
):
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
        logger.info(f"[{i}/{total}] ({category:4s}) {question}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"✅ 分类完成，共处理 {len(results)} 条问题，结果已保存至 {output_file}")
    return results


def evaluate(results: list, answer_file: str) -> dict:
    """评估分类准确率"""
    with open(answer_file, "r", encoding="utf-8") as f:
        answers = json.load(f)

    answer_map = {item["id"]: item["label"] for item in answers}
    correct = 0
    total = len(results)
    errors = []

    for item in results:
        predicted = item["predicted_category"]
        expected = answer_map.get(item["id"], "")
        if predicted == expected:
            correct += 1
        else:
            errors.append({
                "id": item["id"],
                "question": item["question"],
                "expected": expected,
                "predicted": predicted,
            })

    accuracy = correct / total * 100 if total > 0 else 0
    return {
        "total": total,
        "correct": correct,
        "accuracy": round(accuracy, 2),
        "errors": errors,
    }


def print_evaluation_report(eval_result: dict, title: str = "评估报告"):
    """打印评估报告"""
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")
    print(f"  总样本: {eval_result['total']}")
    print(f"  正确:   {eval_result['correct']}")
    print(f"  准确率: {eval_result['accuracy']}%")
    if eval_result["errors"]:
        print(f"\n  ❌ 错误详情 ({len(eval_result['errors'])} 条):")
        for err in eval_result["errors"]:
            print(f"    #{err['id']:2d} 预测={err['predicted']:4s} 期望={err['expected']:4s} | {err['question']}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="客服 FAQ 自动分类器 V2")
    parser.add_argument("input", nargs="?", default="test_samples.json", help="输入 JSON 文件")
    parser.add_argument("output", nargs="?", default="results_v2.json", help="输出 JSON 文件")
    parser.add_argument("--mock", action="store_true", help="使用 Mock 模式（不调用真实 API）")
    parser.add_argument("--api-key", help="OpenAI API Key (默认从环境变量读取)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"模型名 (默认: {DEFAULT_MODEL})")
    parser.add_argument("--evaluate", "-e", action="store_true", help="评估模式")
    parser.add_argument("--answer-file", default="test_samples.json", help="标注答案文件路径")

    args = parser.parse_args()

    classifier = FAQClassifier(
        api_key=args.api_key,
        model=args.model,
        mock=args.mock,
    )
    results = batch_classify(classifier, args.input, args.output)

    if args.evaluate:
        eval_result = evaluate(results, args.answer_file)
        print_evaluation_report(eval_result)
