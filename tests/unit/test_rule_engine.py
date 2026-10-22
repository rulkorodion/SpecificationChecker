from copy import deepcopy
from datetime import date
from decimal import Decimal

from app.checks.rule_engine import RuleEngine
from app.enums import CheckStatus, DeliveryType, FinalDecision, PresenceStatus
from app.schemas import ReferenceData


def by_code(checks, code):
    return next(item for item in checks if item.code == code)


def engine(config, today=date(2026, 4, 1)):
    return RuleEngine(config, today_provider=lambda: today)


def test_all_pass_approves(config, reference, extracted):
    checks, decision, _ = engine(config).evaluate(reference, extracted)
    assert decision is FinalDecision.APPROVE
    assert by_code(checks, "DEFERMENT_DAYS").status is CheckStatus.SKIPPED
    assert not any(check.code.startswith("MANDATORY_CLAUSE") for check in checks)


def test_exact_decimal_mismatch_returns(config, reference, extracted):
    extracted.total_with_vat.normalized_value = Decimal("5272993.71")
    _, decision, _ = engine(config).evaluate(reference, extracted)
    assert decision is FinalDecision.RETURN


def test_delivery_mismatch_returns(config, reference, extracted):
    extracted.delivery_type.normalized_value = DeliveryType.PICKUP
    _, decision, _ = engine(config).evaluate(reference, extracted)
    assert decision is FinalDecision.RETURN


def test_missing_seal_returns(config, reference, extracted):
    extracted.buyer_seal_present.normalized_value = PresenceStatus.ABSENT
    _, decision, _ = engine(config).evaluate(reference, extracted)
    assert decision is FinalDecision.RETURN


def test_explicit_mismatch_has_priority_over_uncertain(config, reference, extracted):
    extracted.total_quantity.normalized_value = Decimal("1")
    extracted.buyer_signature_present.normalized_value = PresenceStatus.UNCERTAIN
    _, decision, _ = engine(config).evaluate(reference, extracted)
    assert decision is FinalDecision.RETURN


def test_low_confidence_is_uncertain(config, reference, extracted):
    extracted.total_quantity.confidence = 0.4
    checks, decision, _ = engine(config).evaluate(reference, extracted)
    assert by_code(checks, "TOTAL_QUANTITY").status is CheckStatus.UNCERTAIN
    assert decision is FinalDecision.MANUAL_REVIEW


def test_stale_date_returns(config, reference, extracted):
    checks, decision, _ = engine(config, date(2026, 4, 7)).evaluate(reference, extracted)
    assert by_code(checks, "DATE_STALENESS").status is CheckStatus.STOP
    assert decision is FinalDecision.RETURN


def test_stale_date_return_cannot_be_disabled_by_legacy_config(
    config, reference, extracted
):
    changed = deepcopy(config)
    changed["date_rule"]["stop_decision"] = "MANUAL_REVIEW"
    _, decision, _ = engine(changed, date(2026, 4, 7)).evaluate(reference, extracted)
    assert decision is FinalDecision.RETURN


def test_technical_error_never_approves(config, reference, extracted):
    _, decision, _ = engine(config).evaluate(reference, extracted, ["Ошибка OCR"])
    assert decision is FinalDecision.MANUAL_REVIEW


def test_application_date_is_not_compared_with_reference(config, reference, extracted):
    checks, decision, _ = engine(config, date(2026, 4, 2)).evaluate(
        reference, extracted
    )
    assert len([check for check in checks if check.title == "Дата приложения"]) == 1
    assert by_code(checks, "DATE_STALENESS").status is CheckStatus.PASS
    assert decision is FinalDecision.APPROVE


def test_unknown_payment_does_not_skip_deferment_from_reference(
    config, reference, extracted
):
    extracted.payment_type.normalized_value = None
    extracted.payment_type.confidence = 0.0
    extracted.deferment_days.normalized_value = None
    extracted.deferment_days.confidence = 0.0

    checks, decision, _ = engine(config).evaluate(reference, extracted)

    assert by_code(checks, "PAYMENT_TYPE").status is CheckStatus.UNCERTAIN
    assert by_code(checks, "DEFERMENT_DAYS").status is CheckStatus.UNCERTAIN
    assert decision is FinalDecision.MANUAL_REVIEW


def test_missing_transportation_evidence_requires_manual_review(
    config, reference, extracted
):
    extracted.transportation_costs_included.normalized_value = None
    extracted.transportation_costs_included.confidence = 0.0
    checks, decision, _ = engine(config).evaluate(reference, extracted)
    assert by_code(checks, "TRANSPORTATION_COSTS").status is CheckStatus.UNCERTAIN
    assert decision is FinalDecision.MANUAL_REVIEW


def test_legacy_reference_cannot_disable_signature_or_seal(
    config, reference, extracted
):
    data = reference.model_dump(mode="json")
    data["required_checks"] = {
        "buyer_signature": False,
        "buyer_seal": False,
    }
    legacy_reference = ReferenceData.model_validate(data)
    extracted.buyer_seal_present.normalized_value = PresenceStatus.ABSENT

    checks, decision, _ = engine(config).evaluate(legacy_reference, extracted)

    assert by_code(checks, "BUYER_SIGNATURE").status is CheckStatus.PASS
    assert by_code(checks, "BUYER_SEAL").status is CheckStatus.MISMATCH
    assert decision is FinalDecision.RETURN


def test_more_than_two_pages_requires_manual_without_field_decisions(
    config, reference, extracted
):
    # A compound document is outside the supported format. Even misleading
    # values extracted from its separate specifications must not produce RETURN.
    extracted.total_quantity.normalized_value = Decimal("999999")

    checks, decision, reasons = engine(config).evaluate(
        reference, extracted, page_count=3
    )

    assert [check.code for check in checks] == ["DOCUMENT_PAGE_COUNT"]
    assert checks[0].status is CheckStatus.UNCERTAIN
    assert checks[0].actual == 3
    assert decision is FinalDecision.MANUAL_REVIEW
    assert reasons
