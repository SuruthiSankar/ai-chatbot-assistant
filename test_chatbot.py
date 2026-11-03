"""Unit tests. Run with:  python -m unittest -v"""

import unittest
from typing import Dict, List

from chatbot import (
    FALLBACK_MESSAGE,
    ChatError,
    Chatbot,
    FAQEntry,
    FAQStore,
    Settings,
    TfidfIndex,
    build_context_message,
    stem,
    tokenize,
)


class FakeClient:
    """Stands in for the OpenAI client so tests never touch the network."""

    def __init__(self, answer: str = "LLM answer", fail: bool = False) -> None:
        self.answer = answer
        self.fail = fail
        self.calls: List[List[Dict[str, str]]] = []

    def complete(self, messages):
        self.calls.append(messages)
        if self.fail:
            raise ChatError("boom")
        return self.answer


def make_faq() -> FAQStore:
    return FAQStore(
        [
            FAQEntry("How can I reset my password?", "Use the Forgot password link."),
            FAQEntry("How do I request a refund?", "Open Orders and request a refund."),
            FAQEntry("What are your working hours?", "9 AM to 6 PM."),
        ]
    )


class PreprocessingTests(unittest.TestCase):
    def test_tokenize_removes_stopwords_and_punctuation(self):
        self.assertEqual(tokenize("What is the Refund policy?!"), ["refund", "policy"])

    def test_stem_plural(self):
        self.assertEqual(stem("refunds"), "refund")

    def test_stem_keeps_short_words(self):
        self.assertEqual(stem("bus"), "bus")


class RetrievalTests(unittest.TestCase):
    def test_best_match_is_most_relevant(self):
        index = TfidfIndex(["reset password link", "refund order money", "working hours"])
        best_idx, score = index.query("I forgot my password")[0] if index.query("I forgot my password") else (None, 0)
        self.assertEqual(best_idx, 0)
        self.assertGreater(score, 0)

    def test_empty_query_returns_nothing(self):
        index = TfidfIndex(["reset password"])
        self.assertEqual(index.query("the a is"), [])

    def test_unrelated_query_returns_nothing(self):
        index = TfidfIndex(["reset password"])
        self.assertEqual(index.query("pizza"), [])

    def test_faq_store_search(self):
        entry, score = make_faq().search("how to get a refund")[0]
        self.assertIn("refund", entry.question.lower())
        self.assertGreater(score, 0)


class ChatbotTests(unittest.TestCase):
    def test_direct_faq_answer_skips_llm(self):
        client = FakeClient()
        bot = Chatbot(Settings(), make_faq(), client)
        result = bot.reply("How can I reset my password?")
        self.assertEqual(result.source, "faq")
        self.assertEqual(client.calls, [])

    def test_unknown_question_uses_llm(self):
        client = FakeClient("Hello there")
        bot = Chatbot(Settings(), make_faq(), client)
        result = bot.reply("Tell me a fun fact about space")
        self.assertEqual(result.source, "llm")
        self.assertEqual(result.text, "Hello there")

    def test_llm_receives_system_prompt_and_history(self):
        client = FakeClient()
        bot = Chatbot(Settings(), make_faq(), client)
        bot.reply("Tell me a fun fact about space")
        bot.reply("Another one please")
        second_call = client.calls[1]
        self.assertEqual(second_call[0]["role"], "system")
        self.assertIn("Tell me a fun fact about space", [m["content"] for m in second_call])

    def test_no_client_and_no_match_gives_fallback(self):
        bot = Chatbot(Settings(), make_faq(), None)
        self.assertEqual(bot.reply("quantum chromodynamics").text, FALLBACK_MESSAGE)

    def test_llm_failure_falls_back_to_faq(self):
        bot = Chatbot(Settings(), make_faq(), FakeClient(fail=True))
        result = bot.reply("refund please, thanks a lot for the help")
        self.assertEqual(result.source, "faq-fallback")

    def test_history_is_trimmed(self):
        settings = Settings(max_turns=2)
        bot = Chatbot(settings, make_faq(), FakeClient())
        for i in range(5):
            bot.reply(f"random question number {i} about space")
        self.assertEqual(len(bot.history), 4)

    def test_reset_clears_history(self):
        bot = Chatbot(Settings(), make_faq(), FakeClient())
        bot.reply("random space question")
        bot.reset()
        self.assertEqual(bot.history, [])

    def test_empty_input(self):
        bot = Chatbot(Settings(), make_faq(), None)
        self.assertEqual(bot.reply("   ").text, "Please type a question.")


class PromptTests(unittest.TestCase):
    def test_context_includes_only_relevant_entries(self):
        faq = make_faq()
        matches = faq.search("refund")
        context = build_context_message(matches, threshold=0.1)
        self.assertIn("Knowledge base", context)
        self.assertIn("refund", context.lower())

    def test_context_none_when_below_threshold(self):
        matches = [(FAQEntry("q", "a"), 0.05)]
        self.assertIsNone(build_context_message(matches, threshold=0.15))


if __name__ == "__main__":
    unittest.main()
