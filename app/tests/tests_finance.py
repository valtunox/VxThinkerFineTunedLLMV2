from app.services.ai.ml.specialist_profile import assess_specialist_scope


def test_finance_only_query_is_rejected():
    assessment = assess_specialist_scope("How should I rebalance my retirement portfolio across bonds and equities?")

    assert assessment.allowed is False
    assert assessment.blocked_domain == "finance"


def test_general_accounting_question_is_rejected():
    assessment = assess_specialist_scope("How do I account for deferred revenue under IFRS?")

    assert assessment.allowed is False
    assert assessment.blocked_domain == "finance"


def test_cloud_cost_optimization_is_allowed():
    assessment = assess_specialist_scope("How can I reduce AWS compute spend and optimize Kubernetes node costs?")

    assert assessment.allowed is True
    assert "aws" in assessment.matched_it_terms


def test_billing_question_with_it_context_is_allowed():
    assessment = assess_specialist_scope("How do I investigate unexpected EC2 billing spikes after a deployment?")

    assert assessment.allowed is True
