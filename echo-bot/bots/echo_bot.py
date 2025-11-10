# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import ast
import asyncio
import operator
import re
from datetime import datetime
from typing import Iterable, Optional, Tuple

from botbuilder.core import ActivityHandler, MessageFactory, TurnContext
from botbuilder.schema import ChannelAccount


class EchoBot(ActivityHandler):
    """Conversational bot that offers multiple prompt handlers and graceful fallbacks."""

    ALLOWED_CALC_PATTERN = re.compile(r"[0-9+\-*/().\s]+$")
    BINARY_OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }
    UNARY_OPERATORS = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def __init__(self, language_client=None):
        super().__init__()
        self.language_client = language_client
        self.capabilities = [
            "`help` – list the bot's capabilities.",
            "`about` – describe the project and how to extend it.",
            "`time` – show the current UTC time.",
            "`calc <expression>` – evaluate math using digits and + - * /.",
            "Fallback – any other text is echoed back in reverse for fun.",
        ]

    async def on_members_added_activity(
        self, members_added: Iterable[ChannelAccount], turn_context: TurnContext
    ):
        for member in members_added:
            if member.id != turn_context.activity.recipient.id:
                welcome = (
                    "Hello! I'm a simple multi-prompt bot. Try commands such as "
                    "`help`, `about`, `time`, or `calc 2+2` to see what I can do."
                )
                await turn_context.send_activity(MessageFactory.text(welcome))

    async def on_message_activity(self, turn_context: TurnContext):
        text = (turn_context.activity.text or "").strip()
        if not text:
            return await turn_context.send_activity(
                MessageFactory.text("I need some text to work with. Type `help` to see what I understand.")
            )

        command, payload = self._extract_command_and_payload(text)
        if command == "help":
            return await turn_context.send_activity(MessageFactory.text(self._format_help_message()))
        if command == "about":
            return await turn_context.send_activity(MessageFactory.text(self._about_message()))
        if command == "time":
            return await turn_context.send_activity(MessageFactory.text(self._time_message()))
        if command == "calc":
            return await turn_context.send_activity(MessageFactory.text(self._handle_calc(payload)))

        nlu_response = await self.nlu_dispatch(text)
        if nlu_response:
            return await turn_context.send_activity(MessageFactory.text(nlu_response))

        return await turn_context.send_activity(MessageFactory.text(self._fallback_message(text)))

    def _extract_command_and_payload(self, text: str) -> Tuple[Optional[str], str]:
        stripped = text.strip()
        if not stripped:
            return None, ""
        parts = stripped.split(" ", 1)
        candidate = parts[0].lower()
        payload = parts[1] if len(parts) > 1 else ""
        if candidate in {"help", "about", "time", "calc"}:
            return candidate, payload.strip()
        return None, ""

    def _format_help_message(self) -> str:
        bullets = "\n".join(f"- {item}" for item in self.capabilities)
        return f"Here is what I can help with today:\n{bullets}"

    def _about_message(self) -> str:
        return (
            "This echo bot powers a Human-Computer Interaction course project. "
            "It is intentionally lightweight so you can plug in Azure Cognitive Services "
            "or OpenAI by wiring them into `nlu_dispatch`."
        )

    def _time_message(self) -> str:
        return f"Current UTC time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"

    def _handle_calc(self, payload: str) -> str:
        expression = payload.strip()
        if not expression:
            return "Usage: `calc <expression>`. Example: `calc 5*2+3`."
        if not self.ALLOWED_CALC_PATTERN.fullmatch(expression):
            return (
                "I can only evaluate digits, spaces, parentheses, and the operators + - * /. "
                "Please try something like `calc 12/(3+1)`."
            )
        try:
            result = self._evaluate_expression(expression)
        except ZeroDivisionError:
            return "Division by zero is undefined. Please adjust the expression."
        except (SyntaxError, ValueError):
            return (
                "I couldn't parse that expression. Double-check your parentheses and operator placement."
            )
        return f"{expression} = {result}"

    def _evaluate_expression(self, expression: str) -> float:
        parsed = ast.parse(expression, mode="eval")
        return self._eval_node(parsed.body)

    def _eval_node(self, node: ast.AST) -> float:
        if isinstance(node, ast.BinOp) and type(node.op) in self.BINARY_OPERATORS:
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            return self.BINARY_OPERATORS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in self.UNARY_OPERATORS:
            operand = self._eval_node(node.operand)
            return self.UNARY_OPERATORS[type(node.op)](operand)
        if isinstance(node, ast.Num):
            return node.n
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Unsupported expression element.")

    async def nlu_dispatch(self, text: str) -> Optional[str]:
        """
        Hook for Azure Language Service (or other NLU) integrations.
        Returns a sentiment + key phrase report when a language_client is configured.
        """
        if not self.language_client:
            return None

        try:
            sentiment = await self._run_in_executor(
                lambda: self.language_client.analyze_sentiment([text])[0]
            )
            print(f"Sentiment analysis result: {sentiment}")
        except Exception:
            return None

        if getattr(sentiment, "is_error", False):
            return None

        key_phrases = None
        try:
            key_result = await self._run_in_executor(
                lambda: self.language_client.extract_key_phrases([text])[0]
            )
            print(f"Key phrase extraction result: {key_result}")
            if not getattr(key_result, "is_error", False):
                key_phrases = list(getattr(key_result, "key_phrases", []))
                print(f"Extracted key phrases: {key_phrases}")
        except Exception:
            key_phrases = None

        confidence = sentiment.confidence_scores
        confidence_text = (
            f"{confidence.positive:.2f} positive / "
            f"{confidence.neutral:.2f} neutral / "
            f"{confidence.negative:.2f} negative"
        )
        key_phrase_text = ", ".join(key_phrases) if key_phrases else "n/a"

        return (
            "Azure Language Service insight:\n"
            f"- Overall sentiment: {sentiment.sentiment}\n"
            f"- Confidence: {confidence_text}\n"
            f"- Key phrases: {key_phrase_text}"
        )

    async def _run_in_executor(self, func):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, func)

    def _fallback_message(self, text: str) -> str:
        reversed_text = text[::-1]
        return (
            "I only recognize `help`, `about`, `time`, and `calc` right now.\n"
            f"For fun, here is your message reversed: {reversed_text}"
        )
