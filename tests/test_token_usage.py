import unittest

from services.token_usage import TokenUsage, merge_token_usage, normalize_token_usage


class TestNormalizeTokenUsage(unittest.TestCase):
    def test_returns_zero_usage_for_none(self):
        usage = normalize_token_usage(None)

        self.assertEqual(usage, TokenUsage(input_tokens=0, output_tokens=0, analysis_tokens=0))

    def test_reads_usage_metadata_fields(self):
        response = type("Resp", (), {
            "usage_metadata": {
                "input_tokens": 11,
                "output_tokens": 7,
            }
        })()

        usage = normalize_token_usage(response)

        self.assertEqual(usage.input_tokens, 11)
        self.assertEqual(usage.output_tokens, 7)
        self.assertEqual(usage.analysis_tokens, 0)

    def test_keeps_zero_from_higher_priority_source(self):
        response = type("Resp", (), {
            "usage_metadata": {
                "input_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
            },
            "response_metadata": {
                "reasoning_tokens": 5,
                "token_usage": {
                    "prompt_tokens": 23,
                    "completion_tokens": 9,
                    "output_tokens_details": {
                        "reasoning_tokens": 4,
                    },
                }
            }
        })()

        usage = normalize_token_usage(response)

        self.assertEqual(usage, TokenUsage(input_tokens=0, output_tokens=0, analysis_tokens=0))

    def test_reads_response_metadata_token_usage_and_reasoning(self):
        response = type("Resp", (), {
            "response_metadata": {
                "token_usage": {
                    "prompt_tokens": 23,
                    "completion_tokens": 9,
                    "output_tokens_details": {
                        "reasoning_tokens": 4,
                    },
                }
            }
        })()

        usage = normalize_token_usage(response)

        self.assertEqual(usage, TokenUsage(input_tokens=23, output_tokens=9, analysis_tokens=4))

    def test_prefers_reasoning_tokens_when_present(self):
        response = type("Resp", (), {
            "response_metadata": {
                "reasoning_tokens": 5,
                "token_usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 6,
                },
            }
        })()

        usage = normalize_token_usage(response)

        self.assertEqual(usage.analysis_tokens, 5)

    def test_merge_token_usage_adds_all_fields(self):
        merged = merge_token_usage(
            TokenUsage(input_tokens=10, output_tokens=4, analysis_tokens=1),
            TokenUsage(input_tokens=3, output_tokens=6, analysis_tokens=2),
        )

        self.assertEqual(merged, TokenUsage(input_tokens=13, output_tokens=10, analysis_tokens=3))


if __name__ == "__main__":
    unittest.main()
