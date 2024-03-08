from dataclasses import dataclass
from typing import Optional

from colossalai.legacy.tensor.distspec import DistPlacementPattern, _DistSpec
from colossalai.legacy.tensor.process_group import ProcessGroup

from .compute_spec import ComputeSpec


@dataclass
class ColoTensorSpec:
    """ColoTensorSpec

    A data class for specifications of the `ColoTensor`.
    It contains attributes of `ProcessGroup`, `_DistSpec`, `ComputeSpec`.
    The latter two attributes are optional. If not set, they are default value is `Replicate()` and `None`.
    """

    pg: ProcessGroup
    dist_attr: Optional[_DistSpec] = None
    compute_attr: Optional[ComputeSpec] = None

    def __init__(self, pg: ProcessGroup = None, 
                 dist_attr: _DistSpec = _DistSpec(DistPlacementPattern.REPLICATE),
                 compute_attr: ComputeSpec = None):
        """Init ColoTensorSpec.
        Args:
            pg (ProcessGroup): The process group of the tensor.
        """
        self.pg = pg
        self.dist_attr = dist_attr
        self.compute_attr = compute_attr
