"""Collaborative filtering over the user-destination interaction matrix.

Two lightweight variants are provided, both sparse and both comfortable inside
a few hundred megabytes at this catalog size:

``ItemItemCFRecommender``
    Shrunk cosine similarity between destinations, based on co-visitation. A
    user is scored by summing the similarities from the places they have
    already been, weighted towards recent trips. This is the natural fit for
    "where should I go after Amsterdam?" because item-item similarity is
    exactly a transition affinity.

``MatrixFactorizationRecommender``
    Truncated SVD of the interaction matrix. Destinations and users share a
    low-rank latent space; a user is represented by folding their history into
    it. Generalises better on sparse data but is harder to explain.

IMPORTANT: both are fitted on synthetic interactions in this project. They
demonstrate that the pipeline works; they say nothing about real traveller
behaviour. See ``src/data/synthetic_interactions.py``.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from src.data.dataset import TravelDataset
from src.models.base import BaseRecommender, RecommendationRequest
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


def build_interaction_matrix(
    dataset: TravelDataset, interactions: pd.DataFrame
) -> Tuple[sparse.csr_matrix, List[str]]:
    """Build the sparse binary user x destination matrix from interactions."""
    user_ids = interactions["user_id"].astype(str).unique().tolist()
    user_index = {user_id: row for row, user_id in enumerate(user_ids)}

    rows = interactions["user_id"].astype(str).map(user_index).to_numpy()
    columns = interactions["destination_id"].astype(str).map(dataset.index_of).to_numpy()
    valid = ~pd.isna(columns)

    matrix = sparse.csr_matrix(
        (
            np.ones(int(valid.sum()), dtype="float32"),
            (rows[valid].astype("int64"), columns[valid].astype("int64")),
        ),
        shape=(len(user_ids), dataset.n_destinations),
    )
    # A user visiting the same place twice must not double-count.
    matrix.data[:] = 1.0
    matrix.sum_duplicates()
    matrix.data[:] = 1.0
    return matrix, user_ids


def _recency_weights(count: int, decay: float) -> np.ndarray:
    """Exponentially decaying weights over an ordered history."""
    if count <= 0:
        return np.zeros(0)
    exponents = np.arange(count - 1, -1, -1, dtype="float64")
    weights = decay**exponents
    return weights / weights.sum()


class ItemItemCFRecommender(BaseRecommender):
    """Item-item collaborative filtering with shrunk cosine similarity."""

    name = "collaborative"

    def __init__(
        self,
        *,
        shrinkage: float = 10.0,
        top_k_neighbours: int = 200,
        recency_decay: float = 0.85,
    ) -> None:
        super().__init__()
        self.shrinkage = float(shrinkage)
        self.top_k_neighbours = int(top_k_neighbours)
        self.recency_decay = float(recency_decay)
        self.similarity: Optional[sparse.csr_matrix] = None
        self.item_counts: np.ndarray = np.zeros(0)

    def _fit(self, dataset: TravelDataset, train_interactions: pd.DataFrame) -> None:
        matrix, _ = build_interaction_matrix(dataset, train_interactions)
        self.item_counts = np.asarray(matrix.sum(axis=0)).ravel()

        # Co-visitation counts between every pair of destinations.
        co_visits = (matrix.T @ matrix).astype("float32").tocsr()
        co_visits.setdiag(0.0)
        co_visits.eliminate_zeros()

        # Shrunk cosine: dividing by sqrt(n_i * n_j) + shrinkage penalises
        # pairs whose co-visit count rests on very few users, which is what
        # stops two obscure destinations from looking perfectly similar
        # because the same two travellers happened to visit both.
        norms = np.sqrt(np.maximum(self.item_counts, 0.0))
        co_visits = co_visits.tocoo()
        denominator = norms[co_visits.row] * norms[co_visits.col] + self.shrinkage
        values = np.divide(
            co_visits.data,
            denominator,
            out=np.zeros_like(co_visits.data),
            where=denominator > 0,
        )
        similarity = sparse.csr_matrix(
            (values, (co_visits.row, co_visits.col)),
            shape=(dataset.n_destinations, dataset.n_destinations),
        )
        self.similarity = self._keep_top_k(similarity, self.top_k_neighbours)
        LOGGER.info(
            "Item-item CF: %d destinations, %d non-zero similarities (%.2f%% dense)",
            dataset.n_destinations,
            self.similarity.nnz,
            100.0 * self.similarity.nnz / max(dataset.n_destinations**2, 1),
        )

    @staticmethod
    def _keep_top_k(matrix: sparse.csr_matrix, k: int) -> sparse.csr_matrix:
        """Zero out all but the ``k`` strongest neighbours of each row."""
        if k <= 0:
            return matrix
        matrix = matrix.tocsr()
        for row in range(matrix.shape[0]):
            start, end = matrix.indptr[row], matrix.indptr[row + 1]
            if end - start <= k:
                continue
            row_data = matrix.data[start:end]
            cutoff = np.partition(row_data, -k)[-k]
            row_data[row_data < cutoff] = 0.0
        matrix.eliminate_zeros()
        return matrix

    def score(self, request: RecommendationRequest) -> np.ndarray:
        dataset = self._require_fitted()
        if self.similarity is None:
            return np.zeros(dataset.n_destinations)

        history = list(request.history)
        if request.current_destination:
            history.append(request.current_destination)
        indices = dataset.indices(history)
        if indices.size == 0:
            # No history: CF has nothing to say. Returning zeros is correct --
            # the hybrid will lean on content and popularity instead.
            return np.zeros(dataset.n_destinations)

        weights = _recency_weights(indices.size, self.recency_decay)
        scores = np.asarray(self.similarity[indices].T @ weights).ravel()
        return scores

    def similar_to(self, destination_id: str, k: int = 10) -> List[Tuple[str, float]]:
        """Destinations most often visited by the same travellers."""
        dataset = self._require_fitted()
        index = dataset.index_of.get(destination_id)
        if index is None or self.similarity is None:
            return []
        row = self.similarity[index].toarray().ravel()
        row[index] = -np.inf
        top = np.argsort(-row)[: max(0, k)]
        return [
            (dataset.destination_ids[int(i)], float(row[int(i)])) for i in top if row[int(i)] > 0
        ]


class MatrixFactorizationRecommender(BaseRecommender):
    """Truncated-SVD latent factor model over the interaction matrix."""

    name = "matrix_factorization"

    def __init__(self, *, n_components: int = 64, seed: int = 42, recency_decay: float = 0.85) -> None:
        super().__init__()
        self.n_components = int(n_components)
        self.seed = int(seed)
        self.recency_decay = float(recency_decay)
        self.item_factors: np.ndarray = np.zeros((0, 0))

    def _fit(self, dataset: TravelDataset, train_interactions: pd.DataFrame) -> None:
        matrix, _ = build_interaction_matrix(dataset, train_interactions)
        n_components = min(self.n_components, min(matrix.shape) - 1)
        if n_components < 2:
            LOGGER.warning("Too few interactions for SVD; falling back to zero factors")
            self.item_factors = np.zeros((dataset.n_destinations, 1))
            return

        svd = TruncatedSVD(n_components=n_components, random_state=self.seed)
        svd.fit(matrix)
        # Rows of components_ span the destination space.
        factors = svd.components_.T  # (n_destinations, n_components)
        norms = np.linalg.norm(factors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.item_factors = factors / norms
        LOGGER.info(
            "SVD: %d components, %.1f%% variance explained",
            n_components,
            100.0 * float(svd.explained_variance_ratio_.sum()),
        )

    def score(self, request: RecommendationRequest) -> np.ndarray:
        dataset = self._require_fitted()
        history = list(request.history)
        if request.current_destination:
            history.append(request.current_destination)
        indices = dataset.indices(history)
        if indices.size == 0 or self.item_factors.size == 0:
            return np.zeros(dataset.n_destinations)

        weights = _recency_weights(indices.size, self.recency_decay)
        # Fold the user's history into the latent space, then project back.
        user_vector = (self.item_factors[indices] * weights[:, None]).sum(axis=0)
        return self.item_factors @ user_vector
