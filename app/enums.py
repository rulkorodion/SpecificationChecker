"""Перечисления предметной области."""

from enum import StrEnum


class CheckStatus(StrEnum):
    PASS = "PASS"
    MISMATCH = "MISMATCH"
    UNCERTAIN = "UNCERTAIN"
    STOP = "STOP"
    SKIPPED = "SKIPPED"


class FinalDecision(StrEnum):
    APPROVE = "APPROVE"
    RETURN = "RETURN"
    MANUAL_REVIEW = "MANUAL_REVIEW"


class DeliveryType(StrEnum):
    DELIVERY = "DELIVERY"
    PICKUP = "PICKUP"


class PaymentType(StrEnum):
    PREPAYMENT = "PREPAYMENT"
    DEFERMENT = "DEFERMENT"


class PresenceStatus(StrEnum):
    PRESENT = "PRESENT"
    ABSENT = "ABSENT"
    UNCERTAIN = "UNCERTAIN"


class ExtractionEvaluationStatus(StrEnum):
    CORRECT = "CORRECT"
    INCORRECT = "INCORRECT"
    NOT_EXTRACTED = "NOT_EXTRACTED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class TableType(StrEnum):
    GOODS = "GOODS"
    SERVICES = "SERVICES"
    UNKNOWN = "UNKNOWN"
