from .cum import CUM
from .cum_v2 import CUMv2
from .cum_v3 import CUMv3
from .cum_v4 import CUMv4
from .cum_v5 import CUMv5
from .cum_v6 import CUMv6
from .cum_v7 import CUMv7
from .cum_v8 import CUMv8
from .cum_v9 import CUMv9
from .cum_v10 import CUMv10
from .cum_v11 import CUMv11
from .cum_v12 import CUMv12
from .cum_2v1 import CUM2v1
from .cum_2v2 import CUM2v2
from .cum_3v1 import CUM3v1
from .cum_3v2 import CUM3v2
from .cum_3v3 import CUM3v3
from .cum_4v1 import CUM4v1
from .cum_4v2 import CUM4v2
from .cum_4v3 import CUM4v3
from .cum_5v1 import CUM5v1
from .cum_5v2 import CUM5v2
from .cum_5v3 import CUM5v3
from .cum_5v4 import CUM5v4
from .cum_5v5 import CUM5v5
from .cum_5v6 import CUM5v6
from .cum_5v7 import CUM5v7
from .cum_6v1 import CUM6v1
from .cum_6v2 import CUM6v2
from .cum_6v3 import CUM6v3
from .cum_6v4 import CUM6v4
from .cum_6v5 import CUM6v5
from .cum_6v6 import CUM6v6
from .cum_6v7 import CUM6v7
from .cum_6v8 import CUM6v8
from .cum_6v9 import CUM6v9
from .cum_7v1 import CUM7v1
from .cum_8v1 import CUM8v1
from .cum_9v1 import CUM9v1
from .cum_11v1 import CUM11v1
from .cum_11v2 import CUM11v2
from .cum_11v3 import CUM11v3
from .smoothed_optimizers import SmoothedAdam
from .hybrid import CUMWithAuxAdam
from .newton_schulz import newton_schulz_orthogonalize
from .factored_precond import apply_factored_precond
from .spectral_control import spectral_damping

__version__ = "0.1.0"

__all__ = [
    "CUM",
    "CUMv2",
    "CUMv3",
    "CUMv4",
    "CUMv5",
    "CUMv6",
    "CUMv7",
    "CUMv8",
    "CUMv9",
    "CUMv10",
    "CUMv11",
    "CUMv12",
    "CUM2v1",
    "CUM2v2",
    "CUM3v1",
    "CUM3v2",
    "CUM3v3",
    "CUM4v1",
    "CUM4v2",
    "CUM4v3",
    "CUM5v1",
    "CUM5v2",
    "CUM5v3",
    "CUM5v4",
    "CUM5v5",
    "CUM5v6",
    "CUM5v7",
    "CUM6v1",
    "CUM6v2",
    "CUM6v3",
    "CUM6v4",
    "CUM6v5",
    "CUM6v6",
    "CUM6v7",
    "CUM6v8",
    "CUM6v9",
    "CUM7v1",
    "CUM8v1",
    "CUM9v1",
    "CUM11v1",
    "CUM11v2",
    "CUM11v3",
    "SmoothedAdam",
    "CUMWithAuxAdam",
    "newton_schulz_orthogonalize",
    "apply_factored_precond",
    "spectral_damping",
]
