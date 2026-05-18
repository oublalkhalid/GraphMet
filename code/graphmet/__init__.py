"""GraphMet: Spectral Compositional Koopman Operators for Weather Forecasting.

Anonymous implementation of the paper
"Learning Spectral Compositional Koopman Operators for Global-to-Regional
Weather Forecasting" (under double-blind review).
"""

from .mesh import IcosahedralMesh, build_multimesh
from .encoder import Grid2MeshEncoder, NodeEncoder
from .decoder import Mesh2GridDecoder, NodeDecoder
from .koopman import BlockSparseKoopman
from .losses import spectral_predictability, isotropy_penalty, physical_loss, GraphMetObjective
from .model import GraphMet
from .ncp import NCPPredictor

__all__ = [
    "IcosahedralMesh",
    "build_multimesh",
    "Grid2MeshEncoder",
    "NodeEncoder",
    "Mesh2GridDecoder",
    "NodeDecoder",
    "BlockSparseKoopman",
    "spectral_predictability",
    "isotropy_penalty",
    "physical_loss",
    "GraphMetObjective",
    "GraphMet",
    "NCPPredictor",
]

__version__ = "0.1.0"
