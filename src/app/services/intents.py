# src/app/services/intents.py
from __future__ import annotations

import re
from enum import Enum, auto


class Intent(Enum):
    GENERATE_TF = auto()
    ASK_COST = auto()
    VALIDATE_ONLY = auto()
    VISUALIZE = auto()
    SMALLTALK = auto()


_COST = re.compile(r"\b(cost|price|spend|infracost|estimate|how\s+much)\b", re.I)
_CODEY = re.compile(
    r"\b(terraform|hcl|resource|module|aws_|azurerm_|google_|s3|vpc|ec2|bucket|eks|rds|create|generate|write code|show code)\b",
    re.I,
)
_VALIDATE = re.compile(r"\b(validate|lint|check|fmt|syntax)\b", re.I)
_VIS = re.compile(r"\b(graph|visuali[sz]e)\b", re.I)


def detect_intent(text: str) -> Intent:
    t = (text or "").strip().lower()
    if _COST.search(t):
        return Intent.ASK_COST
    if _VIS.search(t):
        return Intent.VISUALIZE
    if _VALIDATE.search(t):
        return Intent.VALIDATE_ONLY
    if _CODEY.search(t):
        return Intent.GENERATE_TF
    return Intent.SMALLTALK
