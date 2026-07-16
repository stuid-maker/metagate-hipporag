"""Deterministic Reciprocal Rank Fusion (RRF) for chunk-ID passage rankings.

RRF is a standard unsupervised fusion method that assigns score
``1 / (k + rank)`` to each passage in each ranking, sums across rankings,
and sorts descending.  Ties are broken by chunk_id (lexicographic).

The ``k`` parameter is set to 60 per the experiment config, following
the original RRF paper's recommendation for robustness.
"""

from __future__ import annotations

from collections import defaultdict

from .models import RetrievedPassage


def reciprocal_rank_fusion(
    rankings: list[list[RetrievedPassage]],
    k: int,
    top_k: int,
) -> list[RetrievedPassage]:
    """Fuse multiple ranked lists of passages into one ranking.

    Parameters
    ----------
    rankings:
        One or more ranked lists of ``RetrievedPassage`` objects.  Each
        list is assumed to be sorted by descending relevance (rank 1 is
        the best).  Within each list, ``passage.rank`` is used for scoring.
    k:
        RRF smoothing constant.  ``score = sum(1 / (k + rank))`` for each
        ranking where the passage appears.  Default via config is 60.
    top_k:
        Number of top passages to return after fusion.  The returned list
        has length ``min(top_k, unique_chunks)``.

    Returns
    -------
    list[RetrievedPassage]
        Fused ranking, sorted by RRF score descending, with ties broken
        by chunk_id ascending.  Each passage's ``rank`` field is
        recomputed to reflect its position in the fused list (1-based).
        The ``text`` field comes from the first ranking in which the
        chunk_id appeared.
    """
    if not rankings:
        return []

    scores: dict[str, float] = defaultdict(float)
    passage_by_id: dict[str, RetrievedPassage] = {}

    for ranking in rankings:
        for passage in ranking:
            scores[passage.chunk_id] += 1.0 / (k + passage.rank)
            # Keep the text from the first occurrence of this chunk_id
            if passage.chunk_id not in passage_by_id:
                passage_by_id[passage.chunk_id] = passage

    # Sort: highest score first, then chunk_id ascending for ties
    ordered = sorted(scores, key=lambda cid: (-scores[cid], cid))[:top_k]

    return [
        RetrievedPassage(
            chunk_id=chunk_id,
            text=passage_by_id[chunk_id].text,
            score=scores[chunk_id],
            rank=rank,
        )
        for rank, chunk_id in enumerate(ordered, start=1)
    ]
