from app.evaluation.ground_truth import evaluate_extraction
from app.schemas import GroundTruthData


def test_not_applicable_is_not_denominator(extracted, reference, tmp_path):
    data = reference.model_dump()
    data.pop("specification_id")
    data.pop("required_checks", None)
    data.update(
        application_date=extracted.application_date.normalized_value,
        buyer_signature_present=True,
        buyer_seal_present=True,
    )
    ground_truth = GroundTruthData.model_validate(data)
    result = evaluate_extraction(extracted, ground_truth, tmp_path / "gt.json")
    assert result.not_applicable_fields == 1
    assert result.applicable_fields == 10
    assert result.field_accuracy == 1.0
