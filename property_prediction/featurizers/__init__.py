"""
Featurizers for property prediction.

Available featurizers:
- HNetFeaturizer: Extract latent representations from H-Net models
- RDKitFeaturizer: Traditional chemical descriptors and fingerprints
- LieconvFeaturizer: Lieconv-Tg embeddings (for Tg prediction)
- SMITEDFeaturizer: SMI-TED transformer embeddings
"""

from .hnet_featurizer import HNetFeaturizer
from .rdkit_featurizer import RDKitFeaturizer

__all__ = [
    'HNetFeaturizer',
    'RDKitFeaturizer',
]








