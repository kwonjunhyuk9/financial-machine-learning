from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline


class MyPipeline(Pipeline):
    """Pipeline that forwards sample weights to the final step."""

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        sample_weight: pd.Series | np.ndarray | None = None,
        **fit_params: Any,
    ) -> MyPipeline:
        """Fit the pipeline while passing sample weights to the last estimator.

        Args:
            X: Training features.
            y: Training labels.
            sample_weight: Observation weights.
            **fit_params: Extra fit parameters passed to the parent pipeline.

        Returns:
            The fitted pipeline instance.
        """
        if sample_weight is not None:
            fit_params[self.steps[-1][0] + '__sample_weight'] = sample_weight
        return super().fit(X, y, **fit_params)
