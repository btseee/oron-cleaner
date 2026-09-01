"""The published dataset card.

The card is the only thing most downstream users will read, so a wrong statement
in it propagates further than a wrong constant. M17 was one: `license: cc0-1.0`
declared for a merge containing CC-BY-4.0, which strips the attribution
obligation from everyone who takes it at its word -- and the card's own next
section contradicted it eleven lines later.
"""

import pytest

from pipeline.upload import SOURCE_LICENCES, resolve_licence


def test_an_all_cc0_corpus_is_cc0():
    assert resolve_licence(["cv"])[0] == "cc0-1.0"


def test_adding_fleurs_stops_it_being_cc0():
    """CC-BY-4.0 carries attribution obligations that CC0 disclaims."""
    tag, name = resolve_licence(["cv", "fleurs"])
    assert tag != "cc0-1.0"
    assert "CC-BY-4.0" in name


def test_the_full_default_build_is_mixed():
    tag, name = resolve_licence(["cv", "fleurs", "mbspeech"])
    assert tag == "other"
    assert all(part in name for part in ("CC0-1.0", "CC-BY-4.0", "MIT"))


def test_a_non_commercial_source_is_called_out_by_name():
    """The strongest term is a use restriction, not just attribution: a model
    trained on it cannot be used commercially."""
    _tag, name = resolve_licence(["cv", "ws"])
    assert "non-commercial" in name


@pytest.mark.parametrize("source", sorted(SOURCE_LICENCES))
def test_every_source_states_what_it_requires(source):
    entry = SOURCE_LICENCES[source]
    assert entry["requires"]
    assert entry["url"].startswith("https://")


def test_an_unrecognised_source_does_not_get_a_permissive_tag():
    """Silence about an unknown source must not read as CC0."""
    assert resolve_licence(["something-new"])[0] == "other"
