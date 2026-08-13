"""Dataset loading and the train/validation/test split.

The split is *leave-last-n-out over each user's ordered trip sequence*, which
matches the task: predict where a traveller goes next, given where they have
already been. A random split would let a model see a user's later trips while
predicting their earlier ones, which is the classic temporal leak in
recommender evaluation.

Layout for a user with 8 trips, ``val_holdout=2`` and ``test_holdout=2``::

    trip_index:  0  1  2  3  |  4  5  |  6  7
                 train       |  val   |  test

Models are fitted on ``train`` only. When validation metrics are computed, the
user's history is ``train``; when test metrics are computed, the history is
``train + val``, so the evaluation of the final model uses every observation
that precedes the target window and none that follows it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.config import Config, load_config
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)

DESTINATIONS_FILE = "destinations.parquet"
INTERACTIONS_FILE = "interactions.parquet"
USERS_FILE = "synthetic_users.parquet"


@dataclass
class Split:
    """One train/validation/test partition of the interaction data."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    def history_for(self, stage: str) -> pd.DataFrame:
        """Interactions visible as user history when evaluating ``stage``."""
        if stage == "validation":
            return self.train
        if stage == "test":
            return pd.concat([self.train, self.validation], ignore_index=True)
        raise ValueError(f"Unknown evaluation stage: {stage!r}")

    def targets_for(self, stage: str) -> pd.DataFrame:
        """Held-out interactions that constitute ground truth for ``stage``."""
        if stage == "validation":
            return self.validation
        if stage == "test":
            return self.test
        raise ValueError(f"Unknown evaluation stage: {stage!r}")


class TravelDataset:
    """Destinations, interactions and the index maps every model shares."""

    def __init__(self, destinations: pd.DataFrame, interactions: pd.DataFrame) -> None:
        if destinations["destination_id"].duplicated().any():
            raise ValueError("destination_id must be unique")
        self.destinations = destinations.reset_index(drop=True)
        self.interactions = interactions.reset_index(drop=True)

        self.destination_ids: List[str] = self.destinations["destination_id"].tolist()
        self.index_of: Dict[str, int] = {
            destination_id: index for index, destination_id in enumerate(self.destination_ids)
        }
        # Interactions referencing destinations outside the catalog would break
        # every matrix lookup downstream; fail loudly instead.
        unknown = set(self.interactions["destination_id"]) - set(self.index_of)
        if unknown:
            raise ValueError(f"{len(unknown)} interactions reference unknown destinations")

    @property
    def n_destinations(self) -> int:
        return len(self.destinations)

    @property
    def n_users(self) -> int:
        return int(self.interactions["user_id"].nunique())

    def indices(self, destination_ids: Sequence[str]) -> np.ndarray:
        """Map destination ids to catalog row indices, skipping unknown ids."""
        return np.array(
            [self.index_of[d] for d in destination_ids if d in self.index_of], dtype="int64"
        )

    def ids(self, indices: Sequence[int]) -> List[str]:
        """Map catalog row indices back to destination ids."""
        return [self.destination_ids[int(i)] for i in indices]

    def user_histories(self, interactions: pd.DataFrame) -> Dict[str, List[str]]:
        """Group interactions into ``user_id -> ordered destination ids``."""
        ordered = interactions.sort_values(["user_id", "trip_index"])
        return {
            user_id: group["destination_id"].tolist()
            for user_id, group in ordered.groupby("user_id", sort=False)
        }


def split_interactions(
    interactions: pd.DataFrame,
    *,
    val_holdout: int = 2,
    test_holdout: int = 2,
    min_train_trips: int = 2,
) -> Split:
    """Partition interactions with a per-user leave-last-n-out scheme.

    Users without enough trips to supply ``min_train_trips`` training trips plus
    both holdout windows contribute their trips to ``train`` only; they are
    never evaluated, which keeps the metric denominators honest.
    """
    if val_holdout < 0 or test_holdout < 0:
        raise ValueError("Holdout sizes must be non-negative")

    ordered = interactions.sort_values(["user_id", "trip_index"]).reset_index(drop=True)
    # Position of each trip counted backwards from the user's most recent trip.
    ordered["_from_end"] = ordered.groupby("user_id").cumcount(ascending=False)
    trip_counts = ordered.groupby("user_id")["trip_index"].transform("size")

    evaluable = trip_counts >= (min_train_trips + val_holdout + test_holdout)
    is_test = evaluable & (ordered["_from_end"] < test_holdout)
    is_validation = (
        evaluable
        & (ordered["_from_end"] >= test_holdout)
        & (ordered["_from_end"] < test_holdout + val_holdout)
    )
    is_train = ~(is_test | is_validation)

    columns = [column for column in ordered.columns if column != "_from_end"]
    split = Split(
        train=ordered.loc[is_train, columns].reset_index(drop=True),
        validation=ordered.loc[is_validation, columns].reset_index(drop=True),
        test=ordered.loc[is_test, columns].reset_index(drop=True),
    )
    LOGGER.info(
        "Split: train=%d val=%d test=%d interactions | %d/%d users evaluable",
        len(split.train),
        len(split.validation),
        len(split.test),
        int(ordered.loc[evaluable, "user_id"].nunique()),
        int(ordered["user_id"].nunique()),
    )
    return split


def assert_no_leakage(split: Split) -> None:
    """Verify the split contains no (user, destination) pair in two partitions.

    Guards against the subtle failure where a user visits the same destination
    twice and the model is credited for "predicting" something it saw in
    training. The generator forbids revisits, but a future real dataset may not.
    """
    def pairs(frame: pd.DataFrame) -> set[tuple[str, str]]:
        return set(zip(frame["user_id"], frame["destination_id"]))

    train_pairs = pairs(split.train)
    for name, frame in (("validation", split.validation), ("test", split.test)):
        overlap = train_pairs & pairs(frame)
        if overlap:
            raise AssertionError(
                f"Data leakage: {len(overlap)} (user, destination) pairs appear in "
                f"both train and {name}"
            )
    overlap = pairs(split.validation) & pairs(split.test)
    if overlap:
        raise AssertionError(
            f"Data leakage: {len(overlap)} (user, destination) pairs appear in "
            "both validation and test"
        )
    LOGGER.info("Leakage check passed: train/validation/test pairs are disjoint")


def load_dataset(config: Optional[Config] = None) -> TravelDataset:
    """Load the processed destination and interaction tables from disk."""
    config = config or load_config()
    processed = config.path("data_processed")
    destinations_path = processed / DESTINATIONS_FILE
    interactions_path = processed / INTERACTIONS_FILE

    for path in (destinations_path, interactions_path):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run `python -m src.data.build_dataset` first."
            )

    destinations = pd.read_parquet(destinations_path)
    interactions = pd.read_parquet(interactions_path)
    LOGGER.info(
        "Loaded %d destinations and %d interactions", len(destinations), len(interactions)
    )
    return TravelDataset(destinations, interactions)


def load_split(config: Optional[Config] = None) -> Tuple[TravelDataset, Split]:
    """Load the dataset and apply the configured split in one step."""
    config = config or load_config()
    dataset = load_dataset(config)
    split = split_interactions(
        dataset.interactions,
        val_holdout=int(config.get("evaluation.val_holdout", 2)),
        test_holdout=int(config.get("evaluation.test_holdout", 2)),
        min_train_trips=int(config.get("evaluation.min_train_trips", 2)),
    )
    return dataset, split
