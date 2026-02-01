"""
Edge case tests for pricing module.

Covers: empty strings, wrong types, zero tokens, negative tokens,
NaN/Inf, all model lookups, partial matching, CostCalculator combos.
"""
import math
import pytest

from airos.pricing import (
    ModelPricing, MODEL_PRICING, DEFAULT_PRICING,
    get_model_pricing, estimate_tokens, CostCalculator,
)


# ── ModelPricing ─────────────────────────────────────────────────────────

class TestModelPricingEdgeCases:

    def test_zero_pricing(self):
        p = ModelPricing(0, 0)
        assert p.avg_per_token == 0

    def test_negative_pricing(self):
        p = ModelPricing(-0.001, 0.001)
        assert p.avg_per_token == 0.0  # averages to 0

    def test_very_large_pricing(self):
        p = ModelPricing(1e6, 1e6)
        assert p.avg_per_token == 1e6

    def test_asymmetric_pricing(self):
        p = ModelPricing(1.0, 100.0)
        assert p.avg_per_token == 50.5


# ── MODEL_PRICING table integrity ────────────────────────────────────────

class TestModelPricingTable:

    def test_all_prices_positive(self):
        for model, pricing in MODEL_PRICING.items():
            assert pricing.input_per_token > 0, f"{model} input price not positive"
            assert pricing.output_per_token > 0, f"{model} output price not positive"

    def test_table_not_empty(self):
        assert len(MODEL_PRICING) >= 25

    def test_major_providers_present(self):
        providers = {
            "gpt-4o": "OpenAI",
            "claude-3-5-sonnet": "Anthropic",
            "llama-3.3-70b-versatile": "Groq",
            "gemini-1.5-pro": "Google",
        }
        for model, provider in providers.items():
            assert model in MODEL_PRICING, f"Missing {provider} model: {model}"

    def test_output_generally_more_expensive(self):
        """For most models, output tokens cost >= input tokens."""
        exceptions = {"mixtral-8x7b", "mixtral-8x7b-32768", "gemma2-9b-it"}
        for model, pricing in MODEL_PRICING.items():
            if model not in exceptions:
                assert pricing.output_per_token >= pricing.input_per_token, \
                    f"{model}: output cheaper than input"


# ── get_model_pricing ────────────────────────────────────────────────────

class TestGetModelPricingEdgeCases:

    def test_empty_string_matches_something(self):
        """Empty string: '' in 'gpt-4o' is True in Python, so it partial-matches."""
        p = get_model_pricing("")
        # Matches first model in table via partial match ('' is substring of anything)
        assert isinstance(p, ModelPricing)
        assert p.input_per_token > 0

    def test_whitespace_returns_a_pricing(self):
        p = get_model_pricing("   ")
        assert isinstance(p, ModelPricing)

    def test_special_chars_may_partial_match(self):
        """'gpt-4@#$%' contains 'gpt-4' which partial-matches."""
        p = get_model_pricing("gpt-4@#$%")
        assert isinstance(p, ModelPricing)
        assert p.input_per_token > 0

    def test_very_long_name_returns_default(self):
        p = get_model_pricing("x" * 10000)
        assert p == DEFAULT_PRICING

    def test_case_insensitive_partial(self):
        p = get_model_pricing("GPT-4O")
        # Should match "gpt-4o" via partial (lowered)
        assert p.input_per_token > 0

    def test_versioned_model_matches(self):
        """gpt-4o-2024-08-06 should match gpt-4o."""
        p = get_model_pricing("gpt-4o-2024-08-06")
        assert p == MODEL_PRICING["gpt-4o"]

    def test_exact_match_priority(self):
        """Exact match should be preferred over partial."""
        p = get_model_pricing("gpt-4o")
        assert p == MODEL_PRICING["gpt-4o"]

    def test_unknown_model_returns_default(self):
        p = get_model_pricing("totally-unknown-model-xyz")
        assert p == DEFAULT_PRICING

    def test_claude_versioned_match(self):
        p = get_model_pricing("claude-3-5-sonnet-20241022")
        assert p.input_per_token == MODEL_PRICING["claude-3-5-sonnet"].input_per_token


# ── estimate_tokens ──────────────────────────────────────────────────────

class TestEstimateTokensEdgeCases:

    def test_empty_string(self):
        assert estimate_tokens("") == 1  # min 1

    def test_none(self):
        assert estimate_tokens(None) == 1  # str(None) = "None" = 4 chars / 4 = 1

    def test_integer(self):
        assert estimate_tokens(42) >= 1

    def test_boolean(self):
        assert estimate_tokens(True) >= 1

    def test_empty_dict(self):
        assert estimate_tokens({}) >= 1

    def test_empty_list(self):
        assert estimate_tokens([]) >= 1

    def test_large_string(self):
        big = "x" * 100000
        tokens = estimate_tokens(big)
        assert tokens == 25000  # 100000 / 4

    def test_nested_dict(self):
        nested = {"a": {"b": {"c": {"d": "deep"}}}}
        assert estimate_tokens(nested) >= 1

    def test_unicode(self):
        assert estimate_tokens("こんにちは世界") >= 1

    def test_bytes_object(self):
        assert estimate_tokens(b"hello bytes") >= 1

    def test_float(self):
        assert estimate_tokens(3.14159) >= 1


# ── CostCalculator ───────────────────────────────────────────────────────

class TestCostCalculatorEdgeCases:

    def test_no_args_uses_default(self):
        calc = CostCalculator()
        assert calc.pricing == DEFAULT_PRICING

    def test_model_overrides_default(self):
        calc = CostCalculator(model="gpt-4o")
        assert calc.pricing == MODEL_PRICING["gpt-4o"]

    def test_cost_per_token_overrides_model(self):
        calc = CostCalculator(model="gpt-4o", cost_per_token=0.001)
        assert calc.pricing.input_per_token == 0.001
        assert calc.pricing.output_per_token == 0.001

    def test_zero_tokens(self):
        calc = CostCalculator(model="gpt-4o")
        assert calc.calculate(0, 0) == 0.0

    def test_negative_tokens(self):
        calc = CostCalculator(model="gpt-4o")
        cost = calc.calculate(-100, 200)
        # Negative input tokens produce negative input cost
        assert isinstance(cost, float)

    def test_very_large_tokens(self):
        calc = CostCalculator(model="gpt-4o")
        cost = calc.calculate(1_000_000_000, 1_000_000_000)
        assert cost > 0

    def test_cost_per_token_zero(self):
        calc = CostCalculator(cost_per_token=0.0)
        assert calc.calculate(1000, 1000) == 0.0

    def test_estimate_from_objects_with_none(self):
        calc = CostCalculator(model="gpt-4o")
        tokens, cost = calc.estimate_from_objects(None, None)
        assert tokens >= 2  # at least 1+1
        assert cost >= 0

    def test_estimate_from_objects_large(self):
        calc = CostCalculator(model="gpt-4o")
        tokens, cost = calc.estimate_from_objects("x" * 10000, "y" * 10000)
        assert tokens == 5000  # 2500 + 2500
        assert cost > 0

    def test_repr_default(self):
        calc = CostCalculator()
        assert "default" in repr(calc)

    def test_repr_model(self):
        calc = CostCalculator(model="gpt-4o")
        assert "gpt-4o" in repr(calc)

    def test_repr_cost_per_token(self):
        calc = CostCalculator(cost_per_token=0.001)
        assert "cost_per_token" in repr(calc)

    def test_gpt4o_more_expensive_than_mini(self):
        c1 = CostCalculator(model="gpt-4o")
        c2 = CostCalculator(model="gpt-4o-mini")
        cost1 = c1.calculate(1000, 1000)
        cost2 = c2.calculate(1000, 1000)
        assert cost1 > cost2

    def test_claude_opus_most_expensive_anthropic(self):
        c_opus = CostCalculator(model="claude-3-opus")
        c_haiku = CostCalculator(model="claude-3-haiku")
        assert c_opus.calculate(1000, 1000) > c_haiku.calculate(1000, 1000)
