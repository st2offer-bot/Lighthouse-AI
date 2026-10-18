EVENT_RULES = {
    "cash_handover_candidate": {
        "title": "Possible cash handover",
        "description": "Potential transfer of cash between participants",
        "verification_required": True,
        "normative_reference": None,
    },
    "object_left_behind_candidate": {
        "title": "Possible object left behind",
        "description": "An object may have remained in the cashier area",
        "verification_required": True,
        "normative_reference": None,
    },
    "cash_near_counter": {
        "title": "Cash detected near counter",
        "description": "Cash-like object detected in the cashier workspace",
        "verification_required": True,
        "normative_reference": None,
    },
}

def evaluate_event(event):
    rule = EVENT_RULES.get(event["event_type"])

    if rule is None:
        return {
            **event,
            "classification": "unclassified",
            "requires_review": True,
            "normative_reference": None,
        }

    return {
        **event,
        "title": rule["title"],
        "description": rule["description"],
        "classification": "observation",
        "requires_review": rule["verification_required"],
        "normative_reference": rule["normative_reference"],
    }
