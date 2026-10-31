"""Testy dodatku Basal (W5): prerejestracja, opisy etykiet, klient i reguła wyboru — bez serwera.

Decyzje autora 30.10.2026: 2 modele × 2 warianty opisów, wybór na walidacji, test raz.
31.10.2026: model 11B FP8 odpadł z przyczyn technicznych (opis w configu) — zostaje 4,5B × 2.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from banking77_intent_ml.config import load_settings
from banking77_intent_ml.evaluation import basal_eval
from banking77_intent_ml.models import basal


def _label_names(settings) -> list[str]:
    labels = json.loads(settings.dataset.label_map_path.read_text(encoding="utf-8"))
    return [name for name, _ in sorted(labels.items(), key=lambda item: item[1])]


def test_preregistration_models_and_two_label_variants():
    basal = load_settings().basal

    assert basal is not None
    assert sorted(basal.models) == ["basal_4_5b"]  # 11B odpadł 31.10 (przyczyny techniczne)
    assert sorted(basal.label_variants) == ["v1", "v2"]
    assert len({model.cost_rank for model in basal.models.values()}) == len(basal.models)
    assert all(len(model.revision) == 40 for model in basal.models.values())  # przypięty commit
    assert 0 < basal.tie_tolerance < 0.01


def test_label_variant_files_cover_label_map_in_id_order():
    settings = load_settings()
    names = _label_names(settings)

    for path in settings.basal.label_variants.values():
        descriptions = json.loads(path.read_text(encoding="utf-8"))
        assert list(descriptions) == names, path.name


def test_v1_labels_are_names_as_phrases():
    settings = load_settings()
    v1 = json.loads(settings.basal.label_variants["v1"].read_text(encoding="utf-8"))

    assert all(text == name.replace("_", " ") for name, text in v1.items())


def _payload(choice, criteria, model="basal-2.0-4.5B@e5ba914"):
    probabilities = {key: (0.9 if key == choice else 0.1 / (len(criteria) - 1)) for key in criteria}
    return {
        "model": model,
        "answers": {basal.QUESTION: {"type": "choice", "choice": choice,
                                     "probabilities": probabilities, "confidence": 0.9}},
        "usage": {"latency_ms": 12.5},
    }


CRITERIA = {"card_arrival": "card arrival", "top_up_failed": "top up failed", "age_limit": "age"}


def test_request_body_has_state_and_one_choice_question():
    body = basal.request_body("msg", CRITERIA, "Pick one.")

    assert body == {"state": "msg", "questions": {basal.QUESTION: {
        "type": "choice", "instructions": "Pick one.", "criteria": CRITERIA}}}


def test_parse_decision_reads_choice_probabilities_and_latency():
    decision = basal.parse_decision(_payload("top_up_failed", CRITERIA), CRITERIA, wall_ms=20.0)

    assert decision.choice == "top_up_failed"
    assert decision.probabilities["top_up_failed"] == pytest.approx(0.9)
    assert (decision.server_ms, decision.wall_ms) == (12.5, 20.0)


@pytest.mark.parametrize("payload", [
    _payload("unknown_intent", {**CRITERIA, "unknown_intent": "x"}),  # wybór spoza podanych
    {"model": "m", "answers": {}},  # brak odpowiedzi na pytanie
])
def test_parse_decision_rejects_answers_outside_question(payload):
    with pytest.raises(basal.BasalError):
        basal.parse_decision(payload, CRITERIA, wall_ms=1.0)


def test_served_name_identifies_repo_and_revision():
    model = load_settings().basal.models["basal_4_5b"]

    assert basal.served_name(model) == "basal-2.0-4.5B@e5ba914"


class _FakeClient:
    def __init__(self, choice):
        self.choice = choice

    def decide(self, text, criteria, instructions):
        return basal.parse_decision(_payload(self.choice, criteria), criteria, wall_ms=3.0)


def test_predict_texts_maps_choices_to_label_ids():
    label_map = {"card_arrival": 12, "top_up_failed": 59, "age_limit": 3}

    predictions = basal.predict_texts(["a", "b"], CRITERIA, "Pick.", _FakeClient("age_limit"),
                                      label_map)

    assert predictions.ids == [3, 3]
    assert predictions.scores == pytest.approx([0.9, 0.9])


def _candidate(name, labels, macro_f1, cost_rank):
    return basal_eval.Candidate(name, labels, macro_f1, 0.0, cost_rank, f"run-{name}-{labels}")


def test_choose_takes_best_macro_f1_outside_tie():
    best = _candidate("big", "v2", 0.80, 2)
    candidates = [_candidate("small", "v1", 0.75, 1), best]

    assert basal_eval.choose(candidates, tie_tolerance=0.001) == best


def test_choose_breaks_ties_by_cost_then_v1():
    cheap_v2 = _candidate("small", "v2", 0.8000, 1)
    cheap_v1 = _candidate("small", "v1", 0.7995, 1)
    candidates = [_candidate("big", "v1", 0.8004, 2), cheap_v2, cheap_v1]

    assert basal_eval.choose(candidates, tie_tolerance=0.001) == cheap_v1


def _settings_with(tmp_path, *, instructions="Pick the intent.", v2_text=""):
    settings = load_settings()
    names = list(json.loads(settings.basal.label_variants["v1"].read_text(encoding="utf-8")))
    v2 = tmp_path / "v2.json"
    v2.write_text(json.dumps({name: v2_text for name in names}), encoding="utf-8")
    variants = {**settings.basal.label_variants, "v2": v2}
    return replace(settings, basal=replace(settings.basal, instructions=instructions,
                                           label_variants=variants))


def test_empty_v2_descriptions_or_instructions_are_refused(tmp_path):
    settings = _settings_with(tmp_path, instructions="  ")

    with pytest.raises(ValueError, match="pustych opisów"):
        basal_eval.label_criteria(settings, "v2")
    with pytest.raises(ValueError, match="instructions"):
        basal_eval.instructions(settings)


def test_filled_v2_descriptions_are_accepted(tmp_path):
    settings = _settings_with(tmp_path, v2_text="a description")

    assert len(basal_eval.label_criteria(settings, "v2")) == 77
    assert basal_eval.instructions(settings) == "Pick the intent."
