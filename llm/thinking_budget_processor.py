"""
Source: https://muellerzr.github.io/til/end_thinking.html
"""

import torch
from transformers.generation import LogitsProcessor


class ThinkingTokenBudgetProcessor(LogitsProcessor):
    def __init__(self, tokenizer, max_thinking_tokens=None):
        self.tokenizer = tokenizer
        self.max_thinking_tokens = max_thinking_tokens
        self.think_end_token = self.tokenizer.encode(
            "</think>", add_special_tokens=False
        )[0]
        self.nl_token = self.tokenizer.encode("\n", add_special_tokens=False)[0]
        self.tokens_generated = 0
        self.stopped_thinking = False
        self.neg_inf = float("-inf")

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        self.tokens_generated += 1
        if (
            self.max_thinking_tokens == 0
            and not self.stopped_thinking
            and self.tokens_generated > 0
        ):
            scores[:] = self.neg_inf
            scores[0][self.nl_token] = 0
            scores[0][self.think_end_token] = 0
            self.stopped_thinking = True
            return scores

        if self.max_thinking_tokens is not None and not self.stopped_thinking:
            if self.tokens_generated >= (self.max_thinking_tokens - 1):
                if self.tokens_generated == self.max_thinking_tokens - 1:
                    scores[:] = self.neg_inf
                    scores[0][self.nl_token] = 0
                else:
                    scores[:] = self.neg_inf
                    scores[0][self.think_end_token] = 0
                    self.stopped_thinking = True

        return scores
