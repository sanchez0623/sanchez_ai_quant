# LLM API客户端
"""
LLM客户端模块
=============
统一封装AI模型调用，全课程复用
支持DeepSeek、Kimi(Moonshot)、GPT、Claude等（都兼容OpenAI格式）

┌─────────────────────────────────────────────────────────┐
│ 架构设计                                                  │
│                                                           │
│  DeepSeekClient  ──┐                                      │
│                    ├──▶  MultiModelClient  ──▶  compare() │
│  KimiClient  ──────┘         │                            │
│                              ▼                            │
│                      同一问题多模型对比                      │
│                      (质量 + 速度 + 费用)                   │
└─────────────────────────────────────────────────────────┘
"""

import os
import json
import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from openai import OpenAI


class DeepSeekClient:
    """
    DeepSeek API 客户端

    ┌────────────────────────────────────────────┐
    │ 📖 术语：Client（客户端）                     │
    │    就是"帮你打电话给AI服务器的工具"。           │
    │    你告诉Client要问什么，Client帮你发请求、     │
    │    收回答、处理错误。                          │
    └────────────────────────────────────────────┘

    使用方式：
        from quant_core.ai import DeepSeekClient
        client = DeepSeekClient()
        answer = client.chat("分析贵州茅台的投资价值")
    """

    # 模型配置
    MODELS = {
        "v3":   "deepseek-chat",        # DeepSeek-V3.2: 非思考模式
        "r1":   "deepseek-reasoner",    # DeepSeek-V3.2: 思考模式
    }

    def __init__(self, api_key: str = None, model: str = "v3"):
        """
        初始化客户端

        参数：
            api_key: DeepSeek API Key。
                     如果不传，会自动从环境变量 DEEPSEEK_API_KEY 读取
            model:   "v3" = DeepSeek-V3（便宜快速）
                     "r1" = DeepSeek-R1（强推理）
        """
        self.api_key = api_key
        if not self.api_key:
            try:
                from config import settings as _settings

                self.api_key = getattr(_settings, "DEEPSEEK_API_KEY", "") or self.api_key
            except Exception:
                pass
        if not self.api_key:
            self.api_key = os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError(
                "❌ 未找到API Key！请设置环境变量：\n"
                "   export DEEPSEEK_API_KEY='你的key'  (Mac/Linux)\n"
                "   set DEEPSEEK_API_KEY=你的key        (Windows)\n"
                "   或者直接传入：DeepSeekClient(api_key='你的key')"
            )

        self.model_name = self.MODELS.get(model, model)

        # 创建OpenAI兼容客户端
        # DeepSeek的API格式和OpenAI完全一致，只是base_url不同
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com",  # DeepSeek的服务器地址
        )

        self._system_prompt = None  # 系统提示词

    def set_system_prompt(self, prompt: str):
        """
        设置系统提示词（System Prompt）

        ┌────────────────────────────────────────┐
        │ 📖 System Prompt = AI的"岗位说明书"       │
        │    设定AI的角色、能力范围、回答规范。       │
        │    所有后续对话都会受它影响。              │
        └────────────────────────────────────────┘
        """
        self._system_prompt = prompt
        return self  # 支持链式调用

    def chat(
        self,
        message: str,
        temperature: float = 1.0,
        max_tokens: int = 4096,
        json_output: bool = False,
    ) -> str:
        """
        发送消息给AI，获取回答

        参数：
            message:     你的问题/指令
            temperature: 随机度（0=确定, 1=有创意）量化分析建议0~0.3
            max_tokens:  回答的最大长度（Token数）
            json_output: 是否强制JSON格式输出

        返回：
            AI的回答文本
        """
        # 构建消息列表
        messages = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": message})

        # 构建请求参数
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # 如果要求JSON输出
        if json_output:
            kwargs["response_format"] = {"type": "json_object"}

        # 调用API
        response = self.client.chat.completions.create(**kwargs)

        # 提取回答文本
        return response.choices[0].message.content

    def chat_json(self, message: str, temperature: float = 0.0) -> dict:
        """
        发送消息并获取JSON格式的回答（自动解析为字典）

        ┌────────────────────────────────────────────┐
        │ 📖 为什么需要JSON输出？                       │
        │    量化系统需要程序自动处理AI的回答。           │
        │    自由文本很难解析，JSON可以直接变成字典：     │
        │    {"signal": "BUY", "confidence": 0.8}     │
        │    程序就能读取 result["signal"] == "BUY"    │
        └────────────────────────────────────────────┘
        """
        text = self.chat(message, temperature=temperature, json_output=True)

        # 尝试解析JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # 有时AI返回的JSON外面包了```json ...```，需要清理
            cleaned = text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1]  # 去掉第一行
                cleaned = cleaned.rsplit("```", 1)[0]  # 去掉最后的```
            return json.loads(cleaned)

    def chat_with_history(
        self,
        history: List[Dict[str, str]],
        message: str,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        json_output: bool = False,
    ) -> Dict[str, object]:
        """
        多轮对话：把历史消息一起传给模型

        参数：
            history:  历史消息列表，格式为[{"role": "user|assistant|system", "content": "..."}, ...]
            message:  本轮用户问题
            temperature: 随机度
            max_tokens:  回答的最大长度（Token数）
            json_output: 是否强制JSON格式输出

        返回：
            {"reply": "...", "history": [...]}
        """
        # Build message list with history and new user prompt.
        # 构建消息列表，包含历史消息与本轮用户问题。
        messages = list(history) if history else []
        messages.append({"role": "user", "content": message})

        # Reuse the same request parameters as chat().
        # 复用与chat()一致的请求参数。
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if json_output:
            kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**kwargs)
        reply = response.choices[0].message.content

        # Append assistant reply to history.
        # 将AI回复追加到历史中。
        messages.append({"role": "assistant", "content": reply})

        return {"reply": reply, "history": messages}

    def estimate_cost(
        self,
        input_text: str,
        output_tokens: int = 1000,
        cache_hit: bool = False,
    ) -> dict:
        """
        估算本次调用的费用

        返回：
            {"input_tokens": ..., "output_tokens": ...,
             "estimated_cost_rmb": ...}
        """
        # 粗略估算：1英文字符≈0.3token，1中文字符≈0.6token
        input_tokens = 0.0
        for ch in input_text:
            if ord(ch) < 128:
                input_tokens += 0.3
            else:
                input_tokens += 0.6
        input_tokens = int(input_tokens)

        # DeepSeek-V3.2 定价（输入区分缓存命中/未命中）
        if cache_hit:
            input_price = 0.2 / 1_000_000   # ¥0.2/1M tokens
        else:
            input_price = 2.0 / 1_000_000   # ¥2/1M tokens
        output_price = 3.0 / 1_000_000      # ¥3/1M tokens

        cost = input_tokens * input_price + output_tokens * output_price

        return {
            "input_tokens（输入词元数）": input_tokens,
            "output_tokens（输出词元数）": output_tokens,
            "estimated_cost_rmb（预估费用/元）": round(cost, 6),
        }


# ═══════════════════════════════════════════════════════════
# Kimi (Moonshot) 客户端
# ═══════════════════════════════════════════════════════════

class KimiClient:
    """
    Kimi (Moonshot) API 客户端

    ┌────────────────────────────────────────────────┐
    │ 📖 Kimi 是月之暗面(Moonshot AI)的大模型           │
    │    API兼容OpenAI格式, base_url不同而已。          │
    │    擅长中文理解、长上下文(128k+)、联网搜索。       │
    └────────────────────────────────────────────────┘

    使用方式：
        from quant_core.ai import KimiClient
        client = KimiClient()
        answer = client.chat("分析贵州茅台的投资价值")
    """

    # 模型配置
    MODELS = {
        "k2.5":     "kimi-k2.5",               # Kimi K2.5: 最新多模态旗舰
        "k2":       "kimi-k2-0905-preview",     # Kimi K2: 强Agent/代码能力
        "k2-turbo": "kimi-k2-turbo-preview",    # Kimi K2 Turbo: 高速版
        "v1-8k":    "moonshot-v1-8k",           # Moonshot-V1: 短上下文
        "v1-32k":   "moonshot-v1-32k",          # Moonshot-V1: 中等上下文
        "v1-128k":  "moonshot-v1-128k",         # Moonshot-V1: 超长上下文
    }

    def __init__(self, api_key: str = None, model: str = "k2.5"):
        """
        初始化Kimi客户端

        参数：
            api_key: Kimi API Key。
                     如果不传，会自动从环境变量 KIMI_API_KEY 读取
            model:   模型简称，可选值见 MODELS 字典
                     "k2.5"     = Kimi K2.5（旗舰多模态）默认
                     "k2"       = Kimi K2（强Agent）
                     "k2-turbo" = Kimi K2 Turbo（高速）
                     "v1-8k"    = Moonshot-V1 8k（经济实惠）
                     "v1-32k"   = Moonshot-V1 32k
                     "v1-128k"  = Moonshot-V1 128k（超长上下文）
        """
        self.api_key = api_key
        if not self.api_key:
            try:
                from config import settings as _settings
                self.api_key = getattr(_settings, "KIMI_API_KEY", "") or self.api_key
            except Exception:
                pass
        if not self.api_key:
            self.api_key = os.getenv("KIMI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "❌ 未找到Kimi API Key！请设置环境变量：\n"
                "   export KIMI_API_KEY='你的key'  (Mac/Linux)\n"
                "   set KIMI_API_KEY=你的key        (Windows)\n"
                "   或者直接传入：KimiClient(api_key='你的key')"
            )

        self.model_name = self.MODELS.get(model, model)

        # 创建OpenAI兼容客户端（Kimi API与OpenAI格式一致）
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.moonshot.cn/v1",  # Kimi的服务器地址
        )

        self._system_prompt = None

    def set_system_prompt(self, prompt: str):
        """设置系统提示词"""
        self._system_prompt = prompt
        return self

    def chat(
        self,
        message: str,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        json_output: bool = False,
    ) -> str:
        """
        发送消息给Kimi，获取回答

        参数与DeepSeekClient.chat()完全一致，方便对比切换。
        """
        messages = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": message})

        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if json_output:
            kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    def chat_json(self, message: str, temperature: float = 0.0) -> dict:
        """发送消息并获取JSON格式的回答（自动解析为字典）"""
        text = self.chat(message, temperature=temperature, json_output=True)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            cleaned = text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1]
                cleaned = cleaned.rsplit("```", 1)[0]
            return json.loads(cleaned)

    def chat_with_history(
        self,
        history: List[Dict[str, str]],
        message: str,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        json_output: bool = False,
    ) -> Dict[str, object]:
        """多轮对话：把历史消息一起传给模型"""
        messages = list(history) if history else []
        messages.append({"role": "user", "content": message})

        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if json_output:
            kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**kwargs)
        reply = response.choices[0].message.content
        messages.append({"role": "assistant", "content": reply})

        return {"reply": reply, "history": messages}

    def estimate_cost(
        self,
        input_text: str,
        output_tokens: int = 1000,
    ) -> dict:
        """
        估算Kimi调用费用(基于moonshot-v1定价)

        Moonshot-V1定价: 输入¥12/1M tokens, 输出¥12/1M tokens
        """
        input_tokens = 0.0
        for ch in input_text:
            input_tokens += 0.3 if ord(ch) < 128 else 0.6
        input_tokens = int(input_tokens)

        # Moonshot-V1 统一定价
        token_price = 12.0 / 1_000_000  # ¥12/1M tokens
        cost = (input_tokens + output_tokens) * token_price

        return {
            "input_tokens（输入词元数）": input_tokens,
            "output_tokens（输出词元数）": output_tokens,
            "estimated_cost_rmb（预估费用/元）": round(cost, 6),
        }


# ═══════════════════════════════════════════════════════════
# 多模型客户端 — 对比 & 切换
# ═══════════════════════════════════════════════════════════

@dataclass
class ModelResponse:
    """
    单个模型的回答记录

    ┌──────────────────────────────────────────┐
    │ 📖 为什么要单独记录每个模型的回答？         │
    │    量化分析需要对比不同AI：                 │
    │    - 谁更准确？(quality)                  │
    │    - 谁更快？  (latency)                  │
    │    - 谁更便宜？(cost)                     │
    │    结构化记录才能做量化对比。               │
    └──────────────────────────────────────────┘
    """
    model_label: str            # 模型标签(如 "deepseek-v3", "kimi-v1")
    reply: str = ""             # 原始文本回答
    parsed_json: Optional[dict] = None  # 解析后的JSON(如果是JSON输出)
    latency_ms: float = 0.0     # 响应耗时(毫秒)
    error: Optional[str] = None # 错误信息(如果出错)
    success: bool = True        # 是否成功


class MultiModelClient:
    """
    多模型客户端 — 同一问题发给多个AI，对比回答

    ┌──────────────────────────────────────────────────────┐
    │ 📖 术语：MultiModelClient（多模型客户端）              │
    │                                                       │
    │ 就像"同时采访多位分析师"：                              │
    │   同一个市场问题，分别问DeepSeek和Kimi,               │
    │   对比他们的观点、响应速度、回答质量,                   │
    │   最终取最优或综合决策。                               │
    │                                                       │
    │ 量化场景：                                             │
    │   - A/B测试不同模型对市场判断的准确率                   │
    │   - 多模型投票(2个说涨、1个说跌 → 倾向看涨)            │
    │   - 快模型做初筛、强模型做精细分析                      │
    └──────────────────────────────────────────────────────┘

    使用方式：
        from quant_core.ai import MultiModelClient
        mc = MultiModelClient()                    # 默认加载DeepSeek + Kimi
        results = mc.compare("分析贵州茅台走势")     # 两个模型同时回答
        mc.print_comparison(results)               # 美化打印对比结果
    """

    def __init__(self, clients: Dict[str, Any] = None):
        """
        初始化多模型客户端

        参数：
            clients: 模型标签 → 客户端实例的字典
                     如果不传，自动创建DeepSeek + Kimi

        示例：
            # 方式1: 自动创建(推荐)
            mc = MultiModelClient()

            # 方式2: 手动指定
            mc = MultiModelClient({
                "deepseek-v3": DeepSeekClient(model="v3"),
                "kimi-v1":     KimiClient(model="v1-8k"),
            })

            # 方式3: 只用部分模型
            mc = MultiModelClient({
                "deepseek": DeepSeekClient(model="v3"),
            })
        """
        if clients is not None:
            self._clients = clients
        else:
            # 自动创建: 尝试初始化每个客户端，缺key则跳过
            self._clients = {}
            try:
                self._clients["deepseek-v3"] = DeepSeekClient(model="v3")
            except ValueError:
                pass
            try:
                self._clients["kimi-v1"] = KimiClient(model="v1-8k")
            except ValueError:
                pass

            if not self._clients:
                raise ValueError(
                    "❌ 未找到任何可用的API Key！\n"
                    "   至少需要设置 DEEPSEEK_API_KEY 或 KIMI_API_KEY 之一。"
                )

        self._system_prompt = None

    @property
    def model_labels(self) -> List[str]:
        """当前已加载的模型标签列表"""
        return list(self._clients.keys())

    def set_system_prompt(self, prompt: str):
        """为所有模型统一设置系统提示词"""
        self._system_prompt = prompt
        for client in self._clients.values():
            client.set_system_prompt(prompt)
        return self

    def chat(
        self,
        message: str,
        model: str = None,
        temperature: float = 1.0,
        max_tokens: int = 4096,
    ) -> str:
        """
        用指定模型发送消息

        参数：
            model: 模型标签(如"deepseek-v3")，不传则用第一个
        """
        label = model or self.model_labels[0]
        client = self._clients.get(label)
        if client is None:
            raise ValueError(
                f"❌ 模型 '{label}' 不存在。可用模型: {self.model_labels}"
            )
        return client.chat(message, temperature=temperature, max_tokens=max_tokens)

    def compare(
        self,
        message: str,
        temperature: float = 1.0,
        max_tokens: int = 4096,
        json_output: bool = False,
    ) -> List[ModelResponse]:
        """
        同一个问题发给所有模型，收集对比结果

        参数：
            message:     问题/指令
            temperature: 随机度
            max_tokens:  最大输出长度
            json_output: 是否要求JSON格式输出

        返回：
            List[ModelResponse]，每个模型一条记录
        """
        results = []

        for label, client in self._clients.items():
            resp = ModelResponse(model_label=label)
            t0 = time.time()

            try:
                if json_output:
                    resp.parsed_json = client.chat_json(message, temperature=temperature)
                    resp.reply = json.dumps(resp.parsed_json, ensure_ascii=False, indent=2)
                else:
                    resp.reply = client.chat(
                        message, temperature=temperature, max_tokens=max_tokens,
                    )
            except Exception as e:
                resp.error = str(e)
                resp.success = False

            resp.latency_ms = (time.time() - t0) * 1000
            results.append(resp)

        return results

    def compare_json(
        self,
        message: str,
        temperature: float = 1.0,
    ) -> List[ModelResponse]:
        """
        同一个问题发给所有模型，要求JSON格式回答

        这是 compare() 的便捷版本，适合量化分析场景。
        """
        return self.compare(message, temperature=temperature, json_output=True)

    @staticmethod
    def print_comparison(results: List[ModelResponse]) -> None:
        """
        美化打印对比结果

        ┌─────────────────────────────────────────┐
        │ 输出格式:                                 │
        │   模型名 | 耗时 | 状态 | 回答摘要          │
        │   ...对比总结...                          │
        └─────────────────────────────────────────┘
        """
        print("\n" + "=" * 70)
        print("  🔄 多模型对比结果")
        print("=" * 70)

        for i, resp in enumerate(results, 1):
            status = "✓ 成功" if resp.success else f"✗ 失败: {resp.error[:50]}"
            print(f"\n  ── 模型{i}: {resp.model_label} ──")
            print(f"  状态: {status}")
            print(f"  耗时: {resp.latency_ms:.0f}ms")

            if resp.success:
                # 截取前300字符作为摘要
                preview = resp.reply[:300]
                if len(resp.reply) > 300:
                    preview += "..."
                print(f"  回答摘要:\n    {preview}")

        # 对比总结
        successful = [r for r in results if r.success]
        if len(successful) >= 2:
            fastest = min(successful, key=lambda r: r.latency_ms)
            slowest = max(successful, key=lambda r: r.latency_ms)
            speedup = slowest.latency_ms / fastest.latency_ms if fastest.latency_ms > 0 else 0

            print(f"\n  ── 对比总结 ──")
            print(f"  🏆 最快: {fastest.model_label} ({fastest.latency_ms:.0f}ms)")
            print(f"  🐢 最慢: {slowest.model_label} ({slowest.latency_ms:.0f}ms)")
            print(f"  ⚡ 速度差: {speedup:.1f}x")

            # 如果是JSON输出，对比关键字段
            json_results = [r for r in successful if r.parsed_json]
            if json_results:
                print(f"\n  ── JSON字段对比 ──")
                # 找出所有模型共有的字段
                all_keys = set()
                for r in json_results:
                    all_keys.update(r.parsed_json.keys())

                # 对比关键量化字段
                compare_keys = [
                    k for k in all_keys
                    if k in ("trend", "trend_cn", "strength", "confidence",
                             "risk_level", "risk_cn", "sentiment", "score",
                             "support", "resistance")
                ]
                for key in sorted(compare_keys):
                    vals = []
                    for r in json_results:
                        v = r.parsed_json.get(key, "N/A")
                        vals.append(f"{r.model_label}={v}")
                    print(f"    {key}: {' | '.join(vals)}")

        print(f"\n{'=' * 70}")

    @staticmethod
    def to_dataframe(results: List[ModelResponse]):
        """
        将对比结果转为DataFrame

        需要import pandas(延迟导入，避免强依赖)
        """
        import pandas as pd
        rows = []
        for r in results:
            row = {
                "model": r.model_label,
                "success": r.success,
                "latency_ms": round(r.latency_ms, 1),
                "reply_length": len(r.reply) if r.reply else 0,
                "error": r.error or "",
            }
            # 展开JSON字段
            if r.parsed_json:
                for k, v in r.parsed_json.items():
                    row[f"json_{k}"] = v
            rows.append(row)
        return pd.DataFrame(rows)