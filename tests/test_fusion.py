"""Tests for Reciprocal Rank Fusion (RRF).

Covers:
- Deduplication by chunk_id
- Deterministic output
- Correct RRF scoring formula: 1/(k+rank)
- Top-k truncation
- Tie-breaking by chunk_id when scores are equal
- Empty inputs
- Single ranking pass-through
"""

from __future__ import annotations

from metagate_hipporag.fusion import reciprocal_rank_fusion
from metagate_hipporag.models import RetrievedPassage


def p(chunk_id: str, rank: int) -> RetrievedPassage:
    """Shorthand for building a RetrievedPassage with a dummy text."""
    return RetrievedPassage(chunk_id=chunk_id, text=chunk_id, score=1.0 / rank, rank=rank)


class TestReciprocalRankFusion:
    """RRF unit tests."""

    def test_rrf_deduplicates_by_chunk_id_and_is_deterministic(self) -> None:
        """Same chunk in both rankings → merged, highest-ranked first."""
        fused = reciprocal_rank_fusion(
            [[p("a", 1), p("b", 2)], [p("b", 1), p("c", 2)]], k=60, top_k=3
        )
        assert [row.chunk_id for row in fused] == ["b", "a", "c"]

        # Determinism: same inputs, same output
        assert fused == reciprocal_rank_fusion(
            [[p("a", 1), p("b", 2)], [p("b", 1), p("c", 2)]], k=60, top_k=3
        )

    def test_rrf_scores_follow_formula(self) -> None:
        """Score = sum over rankings of 1/(k + rank)."""
        # b appears in ranking 0 at rank 1: 1/(60+1) = 1/61 ≈ 0.01639
        # b appears in ranking 1 at rank 1: 1/(60+1) = 1/61 ≈ 0.01639
        # total b ≈ 0.03279
        # a appears only in ranking 0 at rank 1: 1/61
        fused = reciprocal_rank_fusion(
            [[p("a", 1), p("b", 2)], [p("b", 1), p("c", 2)]], k=60, top_k=3
        )

        # b should be first (highest score)
        assert fused[0].chunk_id == "b"
        assert fused[1].chunk_id == "a"
        assert fused[2].chunk_id == "c"

        # Check that b's score is roughly double a's (appears in both rankings)
        assert fused[0].score > fused[1].score

    def test_rrf_top_k_truncation(self) -> None:
        """Only top_k results are returned."""
        rankings = [
            [p("a", 1), p("b", 2), p("c", 3), p("d", 4), p("e", 5)],
            [p("f", 1), p("g", 2)],
        ]
        fused = reciprocal_rank_fusion(rankings, k=60, top_k=3)
        assert len(fused) == 3

    def test_rrf_tie_breaks_by_chunk_id(self) -> None:
        """When scores are equal, sort by chunk_id ascending."""
        # Both a and b appear once at rank 1 → same score
        rankings = [[p("a", 1)], [p("b", 1)]]
        fused = reciprocal_rank_fusion(rankings, k=60, top_k=2)
        assert fused[0].chunk_id == "a"
        assert fused[1].chunk_id == "b"
        assert fused[0].score == fused[1].score

    def test_rrf_empty_input(self) -> None:
        """Empty list of rankings → empty result."""
        fused = reciprocal_rank_fusion([], k=60, top_k=5)
        assert fused == []

    def test_rrf_single_ranking_passthrough(self) -> None:
        """Single ranking → preserves order and deduplicates internally."""
        fused = reciprocal_rank_fusion(
            [[p("x", 1), p("y", 2), p("z", 3)]], k=60, top_k=5
        )
        assert [row.chunk_id for row in fused] == ["x", "y", "z"]
        # Ranks are recomputed from 1
        assert fused[0].rank == 1
        assert fused[1].rank == 2
        assert fused[2].rank == 3

    def test_rrf_preserves_text_from_first_occurrence(self) -> None:
        """Text is taken from the first ranking where chunk_id appears."""
        rankings = [
            [RetrievedPassage(chunk_id="a", text="first text", score=1.0, rank=1)],
            [RetrievedPassage(chunk_id="a", text="second text", score=0.5, rank=2)],
        ]
        fused = reciprocal_rank_fusion(rankings, k=60, top_k=1)
        assert fused[0].text == "first text"

    def test_rrf_k_parameter_affects_scores(self) -> None:
        """Larger k → smaller score differences → more uniform weights."""
        rankings = [[p("a", 1), p("b", 5)]]
        rrf_low = reciprocal_rank_fusion(rankings, k=1, top_k=2)
        rrf_high = reciprocal_rank_fusion(rankings, k=100, top_k=2)

        # With small k, the rank-1 vs rank-5 difference is more pronounced
        low_ratio = rrf_low[0].score / rrf_low[1].score
        high_ratio = rrf_high[0].score / rrf_high[1].score
        assert low_ratio > high_ratio
