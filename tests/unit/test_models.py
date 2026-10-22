import json
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.schemas import GroundTruthData, ReferenceData


PRIVATE_REFERENCE = Path("testdata/specifications/reference/specification_1.json")
PRIVATE_GROUND_TRUTH = Path("testdata/specifications/ground_truth/specification_1.json")
requires_private_reference = pytest.mark.skipif(
    not PRIVATE_REFERENCE.is_file(),
    reason="private reference corpus is not available",
)
requires_private_ground_truth = pytest.mark.skipif(
    not PRIVATE_GROUND_TRUTH.is_file(),
    reason="private ground-truth corpus is not available",
)


@requires_private_reference
def test_reference_files_are_valid():
    with open("testdata/specifications/reference/specification_1.json", encoding="utf-8") as stream:
        model = ReferenceData.model_validate(json.load(stream))
    assert model.application_number == "76930"


@requires_private_ground_truth
def test_ground_truth_files_are_valid():
    with open("testdata/specifications/ground_truth/specification_1.json", encoding="utf-8") as stream:
        model = GroundTruthData.model_validate(json.load(stream))
    assert model.buyer_seal_present is True


@requires_private_reference
@requires_private_ground_truth
def test_retired_clause_fields_are_accepted_but_not_exposed():
    with open("testdata/specifications/reference/specification_1.json", encoding="utf-8") as stream:
        reference = ReferenceData.model_validate(json.load(stream))
    with open("testdata/specifications/ground_truth/specification_1.json", encoding="utf-8") as stream:
        ground_truth = GroundTruthData.model_validate(json.load(stream))

    assert "required_checks" not in reference.model_dump()
    assert "mandatory_clause_1_present" not in ground_truth.model_dump()
    assert "mandatory_clause_2_present" not in ground_truth.model_dump()


@requires_private_reference
def test_production_reference_needs_no_reference_dates():
    with open("testdata/specifications/reference/specification_1.json", encoding="utf-8") as stream:
        data = json.load(stream)
    data.pop("check_date", None)
    data.pop("application_date", None)
    data.pop("required_checks", None)
    reference = ReferenceData.model_validate(data)
    assert "check_date" not in reference.model_dump()
    assert "application_date" not in reference.model_dump()


@pytest.mark.skipif(
    not Path("testdata/specifications/reference/specification_55.json").is_file(),
    reason="private reference corpus is not available",
)
def test_legacy_disabled_visual_checks_are_not_exposed():
    with open("testdata/specifications/reference/specification_55.json", encoding="utf-8") as stream:
        reference = ReferenceData.model_validate(json.load(stream))
    assert "required_checks" not in reference.model_dump()


@requires_private_reference
def test_payment_readiness_alias_is_accepted_in_reference():
    with open("testdata/specifications/reference/specification_1.json", encoding="utf-8") as stream:
        data = json.load(stream)
    data["payment_type"] = "Оплата по факту готовности"
    data["deferment_days"] = None
    reference = ReferenceData.model_validate(data)
    assert reference.payment_type.value == "PREPAYMENT"


def test_reference_rejects_deferment_without_days(reference):
    data = reference.model_dump()
    data["payment_type"] = "DEFERMENT"
    data["deferment_days"] = None
    with pytest.raises(ValidationError):
        ReferenceData.model_validate(data)


def test_reference_rejects_float_decimal(reference):
    data = reference.model_dump()
    data["total_with_vat"] = 12.34
    with pytest.raises(ValidationError):
        ReferenceData.model_validate(data)


@pytest.mark.parametrize(
    ("quantity", "total", "expected_quantity", "expected_total"),
    [
        ("1,683", "119 493", Decimal("1.683"), Decimal("119493")),
        ("1,683", "119\u00a0493", Decimal("1.683"), Decimal("119493")),
        ("1.683", "119493.00", Decimal("1.683"), Decimal("119493.00")),
    ],
)
def test_reference_normalizes_localized_decimal_strings(
    reference, quantity, total, expected_quantity, expected_total
):
    data = reference.model_dump(mode="json")
    data["total_quantity"] = quantity
    data["total_with_vat"] = total
    parsed = ReferenceData.model_validate(data)
    assert parsed.total_quantity == expected_quantity
    assert parsed.total_with_vat == expected_total
