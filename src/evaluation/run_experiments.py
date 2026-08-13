"""Run the full model comparison and write the reported results.

    python -m src.evaluation.run_experiments

Fits every model on the training split, tunes the hybrid weights on validation,
trains the ranker, evaluates everything on test, and writes:

* ``reports/model_comparison.csv``      -- accuracy metrics at every K
* ``reports/beyond_accuracy.csv``       -- bias, coverage, diversity
* ``reports/hybrid_weight_search.csv``  -- the tuning trace
* ``reports/feature_importance.csv``    -- what the ranker actually used
* ``reports/results.json``              -- everything, machine-readable
* ``models/ranker.joblib``              -- the trained booster

Every number the README quotes comes from this script. Nothing is hand-entered.
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.analysis.beyond_accuracy import analyse_recommendations, compare_reports, most_recommended
from src.candidate_generation.generator import CandidateGenerator, candidate_recall
from src.config import Config, load_config
from src.data.dataset import Split, TravelDataset, assert_no_leakage, load_split
from src.evaluation.evaluate import EvaluationResult, build_requests, comparison_table, evaluate_model
from src.models.collaborative import ItemItemCFRecommender, MatrixFactorizationRecommender
from src.models.content_based import ContentBasedRecommender
from src.models.context import ContextScorer
from src.models.hybrid import HybridRecommender, HybridWeights, weights_from_config
from src.models.next_destination import NextDestinationRecommender
from src.models.popularity import PopularityRecommender
from src.models.tune_hybrid import tune_weights
from src.ranking.features import RankingFeatureBuilder
from src.ranking.ltr import LearningToRankRecommender, requests_from_interactions
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


def build_models(
    config: Config, dataset: TravelDataset, split: Split
) -> Dict[str, object]:
    """Fit every recommender on the training split and return them by name."""
    train = split.train

    popularity = PopularityRecommender().fit(dataset, train)
    content = ContentBasedRecommender(
        tfidf_max_features=int(config.get("models.content.tfidf_max_features", 20000)),
        tfidf_ngram_max=int(config.get("models.content.tfidf_ngram_max", 2)),
        text_weight=float(config.get("models.content.text_weight", 0.45)),
        attribute_weight=float(config.get("models.content.attribute_weight", 0.55)),
        recency_decay=float(config.get("models.content.recency_decay", 0.85)),
    ).fit(dataset, train)
    collaborative = ItemItemCFRecommender(
        shrinkage=float(config.get("models.collaborative.shrinkage", 10.0)),
        top_k_neighbours=int(config.get("models.collaborative.top_k_neighbours", 200)),
    ).fit(dataset, train)
    factorization = MatrixFactorizationRecommender(
        n_components=int(config.get("models.collaborative.n_components", 64)),
        seed=config.seed,
    ).fit(dataset, train)

    hybrid = HybridRecommender(
        content=content,
        collaborative=collaborative,
        popularity=popularity,
        weights=weights_from_config(config),
    ).fit(dataset, train)

    next_destination = NextDestinationRecommender(
        content=content, collaborative=collaborative, popularity=popularity
    ).fit(dataset, train)

    return {
        "popularity": popularity,
        "content": content,
        "collaborative": collaborative,
        "matrix_factorization": factorization,
        "hybrid": hybrid,
        "next_destination": next_destination,
    }


def run(config: Config, *, max_users: Optional[int] = None, tune: bool = True) -> Dict[str, object]:
    """Execute the full experiment and write every report to disk."""
    reports_dir = config.path("reports")
    models_dir = config.path("models")
    k_values = [int(k) for k in config.get("evaluation.k_values", [5, 10, 20])]

    dataset, split = load_split(config)
    assert_no_leakage(split)

    LOGGER.info("Fitting models on %d training interactions", len(split.train))
    models = build_models(config, dataset, split)
    hybrid: HybridRecommender = models["hybrid"]  # type: ignore[assignment]

    # ------------------------------------------------------ hybrid tuning
    weight_table = pd.DataFrame()
    if tune:
        best_weights, weight_table = tune_weights(
            hybrid, dataset, split, k=10, step=0.1, max_users=600, seed=config.seed
        )
        hybrid.weights = best_weights
        weight_table.to_csv(reports_dir / "hybrid_weight_search.csv", index=False)

    # -------------------------------------------- candidate generation + LTR
    generator = CandidateGenerator(
        dataset,
        content=models["content"],  # type: ignore[arg-type]
        collaborative=models["collaborative"],  # type: ignore[arg-type]
        popularity=models["popularity"],  # type: ignore[arg-type]
        n_candidates=int(config.get("candidate_generation.n_candidates", 150)),
        per_source=dict(config.get("candidate_generation.per_source", {}) or {}),
    )
    builder = RankingFeatureBuilder(
        dataset,
        content=models["content"],  # type: ignore[arg-type]
        collaborative=models["collaborative"],  # type: ignore[arg-type]
        popularity=models["popularity"],  # type: ignore[arg-type]
        context=ContextScorer(dataset),
    )

    ranker = LearningToRankRecommender(
        generator,
        builder,
        params=dict(config.get("ranking.lightgbm", {}) or {}),
        early_stopping_rounds=int(config.get("ranking.early_stopping_rounds", 50)),
        negatives_per_positive=int(config.get("ranking.negatives_per_positive", 30)),
        seed=config.seed,
    )
    training_pairs = requests_from_interactions(dataset, split.train)
    validation_pairs = requests_from_interactions(
        dataset, pd.concat([split.train, split.validation], ignore_index=True)
    )
    LOGGER.info(
        "Training ranker on %d training pairs (%d validation pairs)",
        len(training_pairs),
        len(validation_pairs),
    )
    ranker.fit_ranker(dataset, training_pairs, validation_pairs)
    ranker.save(models_dir / "ranker.joblib")
    models["learning_to_rank"] = ranker

    # ------------------------------------------------------------ evaluate
    evaluation_order = [
        "popularity",
        "content",
        "collaborative",
        "matrix_factorization",
        "hybrid",
        "next_destination",
        "learning_to_rank",
    ]
    results: List[EvaluationResult] = []
    for name in evaluation_order:
        model = models[name]
        model.name = name  # type: ignore[attr-defined]
        results.append(
            evaluate_model(
                model,  # type: ignore[arg-type]
                dataset,
                split,
                stage="test",
                k_values=k_values,
                max_users=max_users,
                seed=config.seed,
            )
        )

    table = comparison_table(results, k_values)
    table.to_csv(reports_dir / "model_comparison.csv", index=False)
    LOGGER.info("\n%s", table.to_string(index=False))

    # ----------------------------------------------------- beyond accuracy
    beyond = [
        analyse_recommendations(dataset, result.recommendations, model_name=result.model_name, k=10)
        for result in results
    ]
    beyond_table = compare_reports(beyond)
    beyond_table.to_csv(reports_dir / "beyond_accuracy.csv", index=False)
    LOGGER.info("\n%s", beyond_table.to_string(index=False))

    top_lists = {
        result.model_name: most_recommended(dataset, result.recommendations, k=10, top=10).to_dict(
            "records"
        )
        for result in results
    }

    # ------------------------------------------------- candidate diagnostics
    test_requests = build_requests(dataset, split, "test")
    test_truth = split.test.groupby("user_id")["destination_id"].apply(set).to_dict()
    sample_users = sorted(test_requests)[:500]
    recall = candidate_recall(
        generator,
        {user_id: test_requests[user_id] for user_id in sample_users},
        test_truth,
    )
    LOGGER.info("Candidate generation recall on %d users: %.4f", len(sample_users), recall)

    importance = ranker.feature_importance()
    importance.to_csv(reports_dir / "feature_importance.csv", index=False)
    LOGGER.info("Top ranker features:\n%s", importance.head(10).to_string(index=False))

    payload = {
        "dataset": {
            "n_destinations": dataset.n_destinations,
            "n_countries": int(dataset.destinations["country_code"].nunique()),
            "n_users": dataset.n_users,
            "n_interactions": len(dataset.interactions),
            "train_interactions": len(split.train),
            "validation_interactions": len(split.validation),
            "test_interactions": len(split.test),
            "interactions_are_synthetic": True,
        },
        "k_values": k_values,
        "model_comparison": table.to_dict("records"),
        "beyond_accuracy": beyond_table.to_dict("records"),
        "candidate_recall_at_test": recall,
        "hybrid_weights": hybrid.weights.as_dict(),
        "feature_importance": importance.to_dict("records"),
        "most_recommended": top_lists,
    }
    (reports_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    LOGGER.info("Wrote reports to %s", reports_dir)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the TravelNext model comparison")
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--max-users", type=int, default=None, help="Evaluate on a subsample of users"
    )
    parser.add_argument("--no-tune", action="store_true", help="Skip hybrid weight tuning")
    args = parser.parse_args()

    config = load_config(args.config)
    run(config, max_users=args.max_users, tune=not args.no_tune)


if __name__ == "__main__":
    main()
