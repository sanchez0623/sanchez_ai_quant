# LLM API客户端
"""
LLM客户端模块
=============
统一封装AI模型调用，全课程复用
支持DeepSeek、GPT、Claude等（都兼容OpenAI格式）
"""

import os
import json
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
        temperature: float = 0.1,
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