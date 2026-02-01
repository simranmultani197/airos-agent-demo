"""
Edge case tests for Sentinel (schema validation).

Covers: None output, wrong types, missing fields, extra fields,
nested schemas, type coercion, no-schema pass-through.
"""
import pytest
from pydantic import BaseModel, Field
from typing import Optional, List

from airos.sentinel import Sentinel, SentinelError


# ── Test schemas ─────────────────────────────────────────────────────────

class SimpleSchema(BaseModel):
    name: str
    age: int

class OptionalSchema(BaseModel):
    name: str
    email: Optional[str] = None

class NestedSchema(BaseModel):
    user: SimpleSchema
    scores: List[float]

class StrictSchema(BaseModel):
    count: int = Field(ge=0, le=100)
    label: str = Field(min_length=1, max_length=50)

class DefaultsSchema(BaseModel):
    name: str = "default"
    value: int = 0


# ── No-schema pass-through ───────────────────────────────────────────────

class TestSentinelNoSchema:

    def test_none_schema_returns_anything(self):
        s = Sentinel(schema=None)
        assert s.validate(42) == 42
        assert s.validate("hello") == "hello"
        assert s.validate(None) is None
        assert s.validate({"any": "thing"}) == {"any": "thing"}

    def test_none_output_no_schema(self):
        s = Sentinel(schema=None)
        assert s.validate(None) is None


# ── Dict input ───────────────────────────────────────────────────────────

class TestSentinelDictInput:

    def test_valid_dict(self):
        s = Sentinel(schema=SimpleSchema)
        result = s.validate({"name": "Alice", "age": 30})
        assert result.name == "Alice"
        assert result.age == 30

    def test_missing_required_field_raises(self):
        s = Sentinel(schema=SimpleSchema)
        with pytest.raises(SentinelError):
            s.validate({"name": "Alice"})  # missing age

    def test_empty_dict_raises(self):
        s = Sentinel(schema=SimpleSchema)
        with pytest.raises(SentinelError):
            s.validate({})

    def test_extra_fields_ignored(self):
        s = Sentinel(schema=SimpleSchema)
        result = s.validate({"name": "Alice", "age": 30, "extra": "ignored"})
        assert result.name == "Alice"

    def test_wrong_type_coercion(self):
        """Pydantic coerces '30' → 30 for int fields."""
        s = Sentinel(schema=SimpleSchema)
        result = s.validate({"name": "Alice", "age": "30"})
        assert result.age == 30

    def test_wrong_type_no_coercion_raises(self):
        s = Sentinel(schema=SimpleSchema)
        with pytest.raises(SentinelError):
            s.validate({"name": "Alice", "age": "not_a_number"})

    def test_optional_field_absent(self):
        s = Sentinel(schema=OptionalSchema)
        result = s.validate({"name": "Alice"})
        assert result.email is None

    def test_optional_field_present(self):
        s = Sentinel(schema=OptionalSchema)
        result = s.validate({"name": "Alice", "email": "a@b.com"})
        assert result.email == "a@b.com"

    def test_defaults_schema_no_input(self):
        s = Sentinel(schema=DefaultsSchema)
        result = s.validate({})
        assert result.name == "default"
        assert result.value == 0


# ── Nested schemas ───────────────────────────────────────────────────────

class TestSentinelNestedSchema:

    def test_valid_nested(self):
        s = Sentinel(schema=NestedSchema)
        result = s.validate({
            "user": {"name": "Bob", "age": 25},
            "scores": [9.5, 8.0, 7.5]
        })
        assert result.user.name == "Bob"
        assert len(result.scores) == 3

    def test_invalid_nested_user(self):
        s = Sentinel(schema=NestedSchema)
        with pytest.raises(SentinelError):
            s.validate({
                "user": {"name": "Bob"},  # missing age
                "scores": [9.5]
            })

    def test_wrong_type_in_list(self):
        s = Sentinel(schema=NestedSchema)
        with pytest.raises(SentinelError):
            s.validate({
                "user": {"name": "Bob", "age": 25},
                "scores": ["not", "floats"]
            })


# ── Constrained fields ──────────────────────────────────────────────────

class TestSentinelConstraints:

    def test_valid_constraints(self):
        s = Sentinel(schema=StrictSchema)
        result = s.validate({"count": 50, "label": "valid"})
        assert result.count == 50

    def test_count_below_min_raises(self):
        s = Sentinel(schema=StrictSchema)
        with pytest.raises(SentinelError):
            s.validate({"count": -1, "label": "valid"})

    def test_count_above_max_raises(self):
        s = Sentinel(schema=StrictSchema)
        with pytest.raises(SentinelError):
            s.validate({"count": 101, "label": "valid"})

    def test_label_empty_raises(self):
        s = Sentinel(schema=StrictSchema)
        with pytest.raises(SentinelError):
            s.validate({"count": 5, "label": ""})

    def test_label_too_long_raises(self):
        s = Sentinel(schema=StrictSchema)
        with pytest.raises(SentinelError):
            s.validate({"count": 5, "label": "x" * 51})


# ── Non-dict inputs ─────────────────────────────────────────────────────

class TestSentinelNonDictInput:

    def test_pydantic_model_passes_through(self):
        s = Sentinel(schema=SimpleSchema)
        model = SimpleSchema(name="Alice", age=30)
        result = s.validate(model)
        assert result.name == "Alice"

    def test_none_with_schema_raises(self):
        s = Sentinel(schema=SimpleSchema)
        with pytest.raises(SentinelError):
            s.validate(None)

    def test_string_input_raises(self):
        s = Sentinel(schema=SimpleSchema)
        with pytest.raises(SentinelError):
            s.validate("not a dict")

    def test_list_input_raises(self):
        s = Sentinel(schema=SimpleSchema)
        with pytest.raises(SentinelError):
            s.validate([1, 2, 3])

    def test_integer_input_raises(self):
        s = Sentinel(schema=SimpleSchema)
        with pytest.raises(SentinelError):
            s.validate(42)

    def test_boolean_input_raises(self):
        s = Sentinel(schema=SimpleSchema)
        with pytest.raises(SentinelError):
            s.validate(True)
