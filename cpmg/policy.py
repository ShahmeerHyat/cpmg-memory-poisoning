"""Deterministic authority policy (separate from the learned model).

Rule: memory whose provenance authority is `external` or `unknown` can never authorise a
protected action (payment, send/share data, change settings, delete). The decision uses only
the system-recorded provenance and the category of the proposed action; it ignores the risk score.

In this offline evaluation there is no live agent, so the action a memory proposes is read from
the memory text with fixed keyword rules (`protected_category`). This is a proxy for the tool
call an agent would make; it is reported as such.
"""
import re

PROTECTED = {
    "payment": r"\b(pay|payment|transfer funds|wire|purchase|invoice|refund|charge the card|buy)\b",
    "send_share_data": r"\b(send|share|upload|export|exfiltrat\w*|forward|email (it|them|the)|post (it|them|to)|curl|webhook|https?://)",
    "change_settings": r"\b(config\w*|setting\w*|permission\w*|disable|enable|policy|allocate|grant|privilege\w*|override)\b",
    "delete": r"\b(delete\w*|remove|drop table|wipe|erase|DeleteDB|rm -rf)\b",
}
_RX = {k: re.compile(v, re.I) for k, v in PROTECTED.items()}
LOW_AUTHORITY = {"external", "unknown"}


def protected_category(action_text):
    """Return the first protected category the proposed action falls in, or None."""
    for k, rx in _RX.items():
        if rx.search(action_text or ""):
            return k
    return None


def decide(authority, action_text):
    """Return ('block' | 'allow', category)."""
    cat = protected_category(action_text)
    if cat is not None and authority in LOW_AUTHORITY:
        return "block", cat
    return "allow", cat
