from enum import Enum


class Metric(Enum):
    """
    Enum of metrics to compare two run results.
    """

    EUCLIDEAN = "Euclidean Distance"
    COSINE = "Cosine Similarity"


AXIS_KWARGS = {Metric.COSINE: {"xlim": (-0.6, 1)}}
