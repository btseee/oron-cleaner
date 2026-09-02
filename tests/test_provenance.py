"""Identifying a corpus after the fact.

FILTER_POLICY_VERSION hashes the gate thresholds and nothing else, so two runs
months apart -- different transformers, a different normaliser, a moved model
revision -- carry the same version string while containing different text.
"""

from pipeline.provenance import (
    PINNED_REVISIONS,
    build,
    corpus_content_hash,
    normaliser_fingerprint,
)


def rec(clip_id, text="сайн", dur=4.0):
    return {"clip_id": clip_id, "text": text, "duration_s": dur}


def test_the_content_hash_ignores_row_order():
    """The manifest is rewritten by split, so order is not identity."""
    a = [rec("a"), rec("b"), rec("c")]
    assert corpus_content_hash(a) == corpus_content_hash(list(reversed(a)))


def test_changed_text_changes_the_content_hash():
    """The point: a normaliser fix rewrites transcripts and must be visible."""
    assert corpus_content_hash([rec("a", "хориос")]) != corpus_content_hash([rec("a", "хорьиос")])


def test_changed_duration_changes_the_content_hash():
    assert corpus_content_hash([rec("a", dur=4.0)]) != corpus_content_hash([rec("a", dur=5.0)])


def test_a_dropped_clip_changes_the_content_hash():
    assert corpus_content_hash([rec("a"), rec("b")]) != corpus_content_hash([rec("a")])


def test_the_normaliser_fingerprint_is_stable_within_a_run():
    assert normaliser_fingerprint() == normaliser_fingerprint()
    assert len(normaliser_fingerprint()) == 12


def test_every_pinned_revision_is_a_full_commit_sha():
    """A tag or a branch is not a pin: both move."""
    for repo, sha in PINNED_REVISIONS.items():
        assert len(sha) == 40, repo
        assert all(c in "0123456789abcdef" for c in sha), repo


def test_the_record_names_what_the_policy_hash_does_not():
    payload = build([rec("a")], {"train": [rec("a")]})
    for key in ("filter_policy_version", "normaliser_fingerprint",
                "corpus_content_hash", "pinned_revisions", "package_versions"):
        assert key in payload, key
    assert payload["package_versions"]["transformers"]
    assert payload["splits"] == {"train": 1}
