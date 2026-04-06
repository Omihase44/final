from models.alzheimer_model import normalize_alzheimer_stage_label


def test_alzheimer_stage_normalization_supports_legacy_and_new_labels():
    assert normalize_alzheimer_stage_label("NonDementia") == "NonDemented"
    assert normalize_alzheimer_stage_label("VeryMildDementia") == "Very Mild"
    assert normalize_alzheimer_stage_label("MildDementia") == "Mild"
    assert normalize_alzheimer_stage_label("ModerateDementia") == "Moderate"
    assert normalize_alzheimer_stage_label("MCI") == "Very Mild"
    assert normalize_alzheimer_stage_label("Early Stage") == "Mild"
    assert normalize_alzheimer_stage_label("Moderate Stage") == "Moderate"
