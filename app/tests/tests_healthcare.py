from app.services.ai.ml.specialist_profile import assess_specialist_scope


def test_healthcare_query_is_rejected():
    assessment = assess_specialist_scope("What is the best treatment for bacterial pneumonia?")

    assert assessment.allowed is False
    assert assessment.blocked_domain == "healthcare"


def test_medication_query_is_rejected():
    assessment = assess_specialist_scope("What dosage of insulin should a patient take before dinner?")

    assert assessment.allowed is False
    assert assessment.blocked_domain == "healthcare"


def test_healthcare_words_with_it_context_can_pass():
    assessment = assess_specialist_scope(
        "How do I troubleshoot a hospital network outage affecting patient record systems?"
    )

    assert assessment.allowed is True
    assert "network" in assessment.matched_it_terms
