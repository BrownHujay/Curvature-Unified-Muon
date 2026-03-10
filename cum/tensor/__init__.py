# CUM Tensor — Universal Orthogonalization
# Goal: Replace Muon+AdamW split with a single optimizer that handles ALL parameter shapes

from .universal_muon import UniversalMuon
from .per_head_muon import PerHeadMuon

__all__ = ["UniversalMuon", "PerHeadMuon"]
