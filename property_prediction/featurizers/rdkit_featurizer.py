"""
RDKit Featurizer: Traditional chemical descriptors and fingerprints.

Supports:
- Morgan fingerprints (ECFP)
- Physicochemical descriptors
- Combined features
"""

import numpy as np
from typing import List, Optional, Literal
from tqdm import tqdm
import warnings

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    from rdkit.Chem import rdFingerprintGenerator
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    warnings.warn("RDKit not available. Install with: pip install rdkit")


FeatureType = Literal['fingerprint', 'descriptors', 'combined']


def _get_descriptor_funcs():
    """Get descriptor functions - must be called after RDKit import."""
    if not RDKIT_AVAILABLE:
        return []
    return [
        ('MolWt', Descriptors.MolWt),
        ('LogP', Descriptors.MolLogP),
        ('TPSA', Descriptors.TPSA),
        ('NumHDonors', Descriptors.NumHDonors),
        ('NumHAcceptors', Descriptors.NumHAcceptors),
        ('NumRotatableBonds', Descriptors.NumRotatableBonds),
        ('NumHeavyAtoms', Descriptors.HeavyAtomCount),
        ('NumRings', rdMolDescriptors.CalcNumRings),
        ('NumAromaticRings', rdMolDescriptors.CalcNumAromaticRings),
        ('FractionCSP3', rdMolDescriptors.CalcFractionCSP3),
        ('NumAliphaticRings', rdMolDescriptors.CalcNumAliphaticRings),
        ('NumSaturatedRings', rdMolDescriptors.CalcNumSaturatedRings),
        ('NumHeteroatoms', rdMolDescriptors.CalcNumHeteroatoms),
        ('NumAmideBonds', rdMolDescriptors.CalcNumAmideBonds),
        ('BalabanJ', Descriptors.BalabanJ),
        ('BertzCT', Descriptors.BertzCT),
        ('Chi0v', Descriptors.Chi0v),
        ('Chi1n', Descriptors.Chi1n),
        ('HallKierAlpha', Descriptors.HallKierAlpha),
        ('Kappa1', Descriptors.Kappa1),
        ('Kappa2', Descriptors.Kappa2),
        ('LabuteASA', Descriptors.LabuteASA),
    ]


class RDKitFeaturizer:
    """
    Extract chemical features using RDKit.
    
    Supports fingerprints, descriptors, or combined features.
    """
    
    def __init__(
        self,
        feature_type: FeatureType = 'combined',
        fp_radius: int = 2,
        fp_bits: int = 2048,
        handle_polymer_smiles: bool = True,
    ):
        """
        Initialize RDKit featurizer.
        
        Args:
            feature_type: Type of features - 'fingerprint', 'descriptors', or 'combined'
            fp_radius: Morgan fingerprint radius (default 2 = ECFP4)
            fp_bits: Number of fingerprint bits
            handle_polymer_smiles: If True, preprocess polymer SMILES (remove * wildcards)
        """
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required. Install with: pip install rdkit")
        
        self.feature_type = feature_type
        self.fp_radius = fp_radius
        self.fp_bits = fp_bits
        self.handle_polymer_smiles = handle_polymer_smiles
        
        # Create fingerprint generator
        self.fp_generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=fp_radius,
            fpSize=fp_bits,
        )
        
        # Calculate feature dimension
        if feature_type == 'fingerprint':
            self.feature_dim = fp_bits
        elif feature_type == 'descriptors':
            self.feature_dim = len(_get_descriptor_funcs())
        else:  # combined
            self.feature_dim = fp_bits + len(_get_descriptor_funcs())
    
    def _preprocess_smiles(self, smiles: str) -> str:
        """
        Preprocess SMILES for RDKit compatibility.
        
        For polymer SMILES (PSMILES), we need to handle the * wildcards
        that denote connection points.
        """
        if not self.handle_polymer_smiles:
            return smiles
        
        # Check if this is a polymer SMILES (contains *)
        if '*' not in smiles:
            return smiles
        
        # Strategy 1: Replace * with H (cap the polymer)
        # This is a simple approach that works for most cases
        processed = smiles.replace('*', '[H]')
        
        # Try to create a valid molecule
        mol = Chem.MolFromSmiles(processed)
        if mol is not None:
            return processed
        
        # Strategy 2: Just remove the * characters
        processed = smiles.replace('*', '')
        mol = Chem.MolFromSmiles(processed)
        if mol is not None:
            return processed
        
        # Strategy 3: Return original and let caller handle errors
        return smiles
    
    def _get_mol(self, smiles: str) -> Optional[Chem.Mol]:
        """Get RDKit Mol object from SMILES."""
        processed = self._preprocess_smiles(smiles)
        mol = Chem.MolFromSmiles(processed)
        return mol
    
    def _get_fingerprint(self, mol: Chem.Mol) -> np.ndarray:
        """Get Morgan fingerprint as numpy array."""
        fp = self.fp_generator.GetFingerprint(mol)
        arr = np.zeros(self.fp_bits, dtype=np.float32)
        for idx in fp.GetOnBits():
            arr[idx] = 1.0
        return arr
    
    def _get_descriptors(self, mol: Chem.Mol) -> np.ndarray:
        """Get physicochemical descriptors."""
        desc = []
        for name, func in _get_descriptor_funcs():
            try:
                value = func(mol)
                if value is None or np.isnan(value) or np.isinf(value):
                    value = 0.0
            except Exception:
                value = 0.0
            desc.append(value)
        return np.array(desc, dtype=np.float32)
    
    def featurize_single(self, smiles: str) -> np.ndarray:
        """
        Extract features for a single SMILES string.
        
        Args:
            smiles: Input SMILES string
            
        Returns:
            Feature vector of shape (D,)
        """
        mol = self._get_mol(smiles)
        
        if mol is None:
            # Return zeros for invalid molecules
            return np.zeros(self.feature_dim, dtype=np.float32)
        
        if self.feature_type == 'fingerprint':
            return self._get_fingerprint(mol)
        elif self.feature_type == 'descriptors':
            return self._get_descriptors(mol)
        else:  # combined
            fp = self._get_fingerprint(mol)
            desc = self._get_descriptors(mol)
            return np.concatenate([fp, desc])
    
    def featurize_batch(self, smiles_list: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Extract features for a list of SMILES strings.
        
        Args:
            smiles_list: List of SMILES strings
            show_progress: Show progress bar
            
        Returns:
            Feature matrix of shape (N, D)
        """
        features = []
        failed = 0
        
        iterator = tqdm(smiles_list, desc=f"Extracting RDKit features ({self.feature_type})") if show_progress else smiles_list
        
        for smiles in iterator:
            feat = self.featurize_single(smiles)
            if np.all(feat == 0):
                failed += 1
            features.append(feat)
        
        if failed > 0:
            print(f"Warning: {failed}/{len(smiles_list)} molecules failed to featurize")
        
        return np.array(features)
    
    def get_feature_names(self) -> List[str]:
        """Get feature names (for interpretability)."""
        names = []
        
        if self.feature_type in ['fingerprint', 'combined']:
            names.extend([f'morgan_bit_{i}' for i in range(self.fp_bits)])
        
        if self.feature_type in ['descriptors', 'combined']:
            names.extend([name for name, _ in _get_descriptor_funcs()])
        
        return names
    
    @property
    def name(self) -> str:
        """Get featurizer name for logging."""
        return f"RDKit_{self.feature_type}"


def get_rdkit_featurizer(
    feature_type: FeatureType = 'combined',
    for_polymers: bool = False,
) -> RDKitFeaturizer:
    """
    Convenience function to get RDKit featurizer.
    
    Args:
        feature_type: Type of features
        for_polymers: If True, enable polymer SMILES handling
        
    Returns:
        Configured RDKitFeaturizer
    """
    return RDKitFeaturizer(
        feature_type=feature_type,
        handle_polymer_smiles=for_polymers,
    )

