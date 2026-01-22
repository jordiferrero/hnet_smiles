#!/usr/bin/env python3
"""
Automated Chemical Token Annotation using Hybrid Approach.

This script classifies H-Net tokens using three stages:
1. Character Pattern Heuristics (fast pattern matching)
2. RDKit Parse Validation (check if token is valid SMILES fragment)
3. SMARTS Functional Group Matching (identify chemical patterns)

The output is a comprehensive annotation of each token with:
- Primary category (aliphatic, aromatic, functional_group, syntax, bond, ring, mixed)
- Whether it's a valid SMILES fragment
- Atom count and composition
- Matched functional group patterns
- Whether it respects atom boundaries
"""

import json
import re
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import Counter

# RDKit imports
from rdkit import Chem
from rdkit import RDLogger

# Suppress RDKit warnings for invalid SMILES attempts
RDLogger.DisableLog('rdApp.*')

# Paths
DATA_DIR = Path(__file__).parent / 'data'
STATS_DIR = Path(__file__).parent.parent / 'data' / 'statistics'


# =============================================================================
# Stage 1: Character Pattern Heuristics
# =============================================================================

# SMILES syntax characters
SYNTAX_CHARS = set('()[]')
BOND_CHARS = set('=#:/-\\@')
RING_CLOSURE_CHARS = set('0123456789%')

# Aromatic atoms (lowercase in SMILES)
AROMATIC_ATOMS = set('cnopsb')

# Common aliphatic atoms (uppercase)
ALIPHATIC_ATOMS = set('CNOPSB')

# Special element patterns (two-letter elements)
TWO_LETTER_ELEMENTS = {
    'Cl', 'Br', 'Si', 'Se', 'As', 'Te', 'Li', 'Na', 'Mg', 'Al', 
    'Ca', 'Fe', 'Cu', 'Zn', 'Sn', 'Pb', 'Bi', 'Ag', 'Au', 'Pt'
}


def classify_by_pattern(token: str) -> Dict:
    """
    Stage 1: Classify token using character pattern heuristics.
    
    Returns:
        Dict with primary_category and pattern_details
    """
    if not token:
        return {'primary_category': 'empty', 'pattern_details': {}}
    
    # Single character classifications
    if len(token) == 1:
        if token in SYNTAX_CHARS:
            return {'primary_category': 'syntax', 'pattern_details': {'type': 'bracket'}}
        if token in BOND_CHARS:
            return {'primary_category': 'bond', 'pattern_details': {'type': 'bond_symbol'}}
        if token in RING_CLOSURE_CHARS:
            return {'primary_category': 'ring_closure', 'pattern_details': {'type': 'ring_digit'}}
        if token.islower() and token in AROMATIC_ATOMS:
            return {'primary_category': 'aromatic', 'pattern_details': {'type': 'aromatic_atom'}}
        if token.isupper() and token in ALIPHATIC_ATOMS:
            return {'primary_category': 'aliphatic', 'pattern_details': {'type': 'aliphatic_atom'}}
        if token in 'HFI':  # Hydrogen, Fluorine, Iodine
            return {'primary_category': 'element', 'pattern_details': {'type': 'single_letter_element'}}
    
    # Polymer attachment point
    if '*' in token:
        return {'primary_category': 'polymer_syntax', 'pattern_details': {'type': 'attachment_point'}}
    
    # Analyze character composition
    has_aromatic = any(c in AROMATIC_ATOMS for c in token if c.isalpha())
    has_aliphatic = any(c in ALIPHATIC_ATOMS for c in token if c.isalpha())
    has_syntax = any(c in SYNTAX_CHARS for c in token)
    has_bond = any(c in BOND_CHARS for c in token)
    has_ring = any(c in RING_CLOSURE_CHARS for c in token)
    
    # Count character types
    char_types = {
        'aromatic': sum(1 for c in token if c.lower() in AROMATIC_ATOMS and c.islower()),
        'aliphatic': sum(1 for c in token if c in ALIPHATIC_ATOMS),
        'syntax': sum(1 for c in token if c in SYNTAX_CHARS),
        'bond': sum(1 for c in token if c in BOND_CHARS),
        'ring': sum(1 for c in token if c in RING_CLOSURE_CHARS),
    }
    
    # Determine primary category based on dominant pattern
    if char_types['aromatic'] > 0 and char_types['aliphatic'] == 0:
        if has_ring:
            return {'primary_category': 'aromatic_ring', 'pattern_details': char_types}
        return {'primary_category': 'aromatic', 'pattern_details': char_types}
    
    if char_types['aliphatic'] > 0 and char_types['aromatic'] == 0:
        return {'primary_category': 'aliphatic', 'pattern_details': char_types}
    
    if char_types['aromatic'] > 0 and char_types['aliphatic'] > 0:
        return {'primary_category': 'mixed', 'pattern_details': char_types}
    
    if has_bond and not (has_aromatic or has_aliphatic):
        return {'primary_category': 'bond', 'pattern_details': char_types}
    
    if has_syntax and not (has_aromatic or has_aliphatic):
        return {'primary_category': 'syntax', 'pattern_details': char_types}
    
    return {'primary_category': 'complex', 'pattern_details': char_types}


# =============================================================================
# Stage 2: RDKit Parse Validation
# =============================================================================

def parse_with_rdkit(token: str) -> Dict:
    """
    Stage 2: Try to parse token as SMILES fragment with RDKit.
    
    Returns:
        Dict with validation results
    """
    result = {
        'is_valid_smiles': False,
        'atom_count': 0,
        'atoms': [],
        'has_aromatic': False,
        'has_ring': False,
        'bond_count': 0,
    }
    
    # Skip pure syntax tokens
    if all(c in '()[]=#:/-\\@0123456789%*' for c in token):
        return result
    
    # Try parsing as-is
    mol = Chem.MolFromSmiles(token, sanitize=False)
    
    if mol is None:
        # Try with common modifications
        # Add hydrogen to incomplete fragments
        for suffix in ['', 'H', '[H]']:
            test_token = token + suffix
            mol = Chem.MolFromSmiles(test_token, sanitize=False)
            if mol:
                break
    
    if mol:
        try:
            result['is_valid_smiles'] = True
            result['atom_count'] = mol.GetNumAtoms()
            result['atoms'] = [atom.GetSymbol() for atom in mol.GetAtoms()]
            result['has_aromatic'] = any(atom.GetIsAromatic() for atom in mol.GetAtoms())
            result['bond_count'] = mol.GetNumBonds()
            result['has_ring'] = mol.GetRingInfo().NumRings() > 0
        except Exception:
            pass
    
    return result


# =============================================================================
# Stage 3: SMARTS Functional Group Matching
# =============================================================================

# Define SMARTS patterns for common functional groups
SMARTS_PATTERNS = {}

def initialize_smarts_patterns():
    """Initialize SMARTS patterns (called once to avoid repeated compilation)."""
    global SMARTS_PATTERNS
    
    pattern_definitions = {
        # Carbonyl groups
        'carbonyl': '[C]=O',
        'aldehyde': '[CH]=O',
        'ketone': '[C;!$([CH])]=O',
        
        # Carboxylic acids and derivatives
        'carboxyl': '[CX3](=O)[OX2H1]',
        'ester': '[CX3](=O)[OX2][#6]',
        'amide': '[CX3](=O)[NX3]',
        
        # Nitrogen groups
        'amine_primary': '[NX3;H2]',
        'amine_secondary': '[NX3;H1]([#6])[#6]',
        'amine_tertiary': '[NX3]([#6])([#6])[#6]',
        'nitro': '[N+](=O)[O-]',
        'nitrile': '[C]#N',
        'imine': '[CX3]=[NX2]',
        
        # Oxygen groups
        'hydroxyl': '[OX2H]',
        'ether': '[OD2]([#6])[#6]',
        'epoxide': '[OX2r3]',
        
        # Sulfur groups
        'thiol': '[SX2H]',
        'sulfide': '[SX2]([#6])[#6]',
        'sulfone': '[SX4](=O)(=O)',
        'sulfoxide': '[SX3](=O)',
        
        # Aromatic patterns
        'benzene': 'c1ccccc1',
        'phenyl': 'c1ccccc1',
        'pyridine': 'c1ccncc1',
        'furan': 'c1ccoc1',
        'thiophene': 'c1ccsc1',
        'pyrrole': 'c1cc[nH]c1',
        'imidazole': 'c1cnc[nH]1',
        
        # Halogen groups
        'fluorine': '[F]',
        'chlorine': '[Cl]',
        'bromine': '[Br]',
        'iodine': '[I]',
        'trifluoromethyl': '[CX4](F)(F)F',
        
        # Other common groups
        'phosphate': '[PX4](=O)([O-])([O-])',
        'silicon': '[Si]',
    }
    
    for name, smarts in pattern_definitions.items():
        try:
            mol = Chem.MolFromSmarts(smarts)
            if mol:
                SMARTS_PATTERNS[name] = mol
        except Exception:
            pass


def match_functional_groups(token: str) -> List[str]:
    """
    Stage 3: Match token against SMARTS functional group patterns.
    
    Returns:
        List of matched functional group names
    """
    if not SMARTS_PATTERNS:
        initialize_smarts_patterns()
    
    matched = []
    
    # Try to create a molecule from the token
    mol = Chem.MolFromSmiles(token, sanitize=False)
    
    if mol is None:
        # Try pattern matching on the string itself for partial matches
        # Check for common substrings
        pattern_strings = {
            'carbonyl': ['=O', 'C=O'],
            'hydroxyl': ['OH', '[OH]'],
            'amine': ['NH', 'NH2', '[NH]', '[NH2]'],
            'fluorine': ['F'],
            'chlorine': ['Cl'],
            'bromine': ['Br'],
            'trifluoromethyl': ['C(F)(F)F', 'CF3'],
            'carboxyl': ['C(=O)O', 'COOH'],
            'ester': ['C(=O)O', 'COO'],
            'ether': ['COC', 'OC'],
        }
        
        for group_name, patterns in pattern_strings.items():
            for pattern in patterns:
                if pattern in token:
                    if group_name not in matched:
                        matched.append(group_name)
        
        return matched
    
    # Use SMARTS matching on the molecule
    for name, smarts_mol in SMARTS_PATTERNS.items():
        try:
            if mol.HasSubstructMatch(smarts_mol):
                matched.append(name)
        except Exception:
            pass
    
    return matched


# =============================================================================
# Stage 4: Atom Boundary Analysis
# =============================================================================

def check_atom_boundary(token: str) -> str:
    """
    Check if token respects atom boundaries.
    
    Returns:
        'yes' - Token aligns with complete atoms
        'no' - Token splits atoms
        'partial' - Token partially respects boundaries
        'n/a' - Cannot determine (syntax-only tokens)
    """
    # Syntax-only tokens
    if all(c in '()[]=#:/-\\@0123456789%*' for c in token):
        return 'n/a'
    
    # Check for split two-letter elements
    for elem in TWO_LETTER_ELEMENTS:
        # Check if token ends with first letter of a two-letter element
        # and doesn't include the second letter
        if token.endswith(elem[0]) and not token.endswith(elem):
            # This might split an element
            return 'partial'
        # Check if token starts with second letter of a two-letter element
        if token.startswith(elem[1].lower()) and len(elem) > 1:
            return 'partial'
    
    # Try parsing as valid SMILES
    mol = Chem.MolFromSmiles(token, sanitize=False)
    if mol and mol.GetNumAtoms() > 0:
        return 'yes'
    
    # Check for partial patterns
    # Tokens that are clearly incomplete
    if token.endswith('(') or token.startswith(')'):
        return 'partial'
    
    if re.match(r'^[A-Z][a-z]?$', token):
        return 'yes'  # Single atom
    
    return 'partial'


# =============================================================================
# Main Annotation Function
# =============================================================================

def annotate_token(token: str) -> Dict:
    """
    Fully annotate a single token using all three stages.
    
    Returns:
        Complete annotation dictionary
    """
    # Stage 1: Pattern heuristics
    pattern_result = classify_by_pattern(token)
    
    # Stage 2: RDKit validation
    rdkit_result = parse_with_rdkit(token)
    
    # Stage 3: Functional group matching
    matched_groups = match_functional_groups(token)
    
    # Stage 4: Atom boundary check
    atom_boundary = check_atom_boundary(token)
    
    # Refine category based on functional groups
    final_category = pattern_result['primary_category']
    if matched_groups:
        # If we matched functional groups, refine the category
        if any(g in ['carbonyl', 'hydroxyl', 'amine_primary', 'carboxyl', 'ester'] for g in matched_groups):
            final_category = 'functional_group'
        elif any(g in ['benzene', 'phenyl', 'pyridine', 'furan', 'thiophene'] for g in matched_groups):
            final_category = 'aromatic_ring'
    
    # Determine chemical meaning
    chemical_meaning = determine_chemical_meaning(token, matched_groups, rdkit_result)
    
    return {
        'token': token,
        'length': len(token),
        'category': final_category,
        'chemical_meaning': chemical_meaning,
        'is_valid_smiles': rdkit_result['is_valid_smiles'],
        'atom_count': rdkit_result['atom_count'],
        'atoms': ','.join(rdkit_result['atoms']) if rdkit_result['atoms'] else '',
        'has_aromatic': rdkit_result['has_aromatic'],
        'matched_patterns': ','.join(matched_groups) if matched_groups else '',
        'respects_atom_boundary': atom_boundary,
        'pattern_details': str(pattern_result['pattern_details']),
    }


def determine_chemical_meaning(token: str, matched_groups: List[str], rdkit_result: Dict) -> str:
    """Determine human-readable chemical meaning of a token."""
    
    # Common known tokens
    known_meanings = {
        'C': 'carbon (aliphatic)',
        'c': 'carbon (aromatic)',
        'N': 'nitrogen (aliphatic)',
        'n': 'nitrogen (aromatic)',
        'O': 'oxygen',
        'o': 'oxygen (aromatic)',
        'S': 'sulfur',
        's': 'sulfur (aromatic)',
        'F': 'fluorine',
        'Cl': 'chlorine',
        'Br': 'bromine',
        'I': 'iodine',
        'H': 'hydrogen',
        '*': 'polymer attachment',
        '(': 'branch open',
        ')': 'branch close',
        '[': 'bracket atom open',
        ']': 'bracket atom close',
        '=': 'double bond',
        '#': 'triple bond',
        '-': 'single bond (explicit)',
        '/': 'cis/trans (up)',
        '\\': 'cis/trans (down)',
        '@': 'stereochemistry',
        '1': 'ring closure 1',
        '2': 'ring closure 2',
        '3': 'ring closure 3',
        'CC': 'ethyl chain',
        'CCC': 'propyl chain',
        'CCCC': 'butyl chain',
        'CCCCC': 'pentyl chain',
        'CCCCCC': 'hexyl chain',
        'C(=O)': 'carbonyl',
        '=O': 'carbonyl oxygen',
        'C(=O)O': 'carboxyl',
        'C(=O)N': 'amide',
        'c1ccc': 'benzene partial',
        'c1ccccc1': 'benzene',
        'cc': 'aromatic bond',
        'ccc': 'aromatic chain',
        'CO': 'hydroxymethyl',
        'OC': 'methoxy',
        'NC': 'methylamine',
        'CN': 'methylamine',
    }
    
    if token in known_meanings:
        return known_meanings[token]
    
    # Use matched functional groups
    if matched_groups:
        return matched_groups[0].replace('_', ' ')
    
    # Use atom composition
    if rdkit_result['atoms']:
        atoms = rdkit_result['atoms']
        atom_counts = Counter(atoms)
        if len(atom_counts) == 1:
            atom = list(atom_counts.keys())[0]
            count = atom_counts[atom]
            if count == 1:
                return f"{atom} atom"
            else:
                return f"{atom}{count} chain"
        else:
            return f"{len(atoms)}-atom fragment"
    
    # Pattern-based inference
    if token.startswith('c') and any(c.isdigit() for c in token):
        return 'aromatic ring fragment'
    if token.startswith('C') and 'C' in token[1:]:
        return 'alkyl chain'
    
    return 'complex pattern'


def main():
    """Main function to annotate all top tokens."""
    # Ensure SMARTS patterns are initialized
    initialize_smarts_patterns()
    
    # Load top 100 tokens
    top_tokens_path = DATA_DIR / 'top_100_tokens.csv'
    if not top_tokens_path.exists():
        print(f"Error: {top_tokens_path} not found. Run extract_top_tokens.py first.")
        return
    
    top_tokens_df = pd.read_csv(top_tokens_path)
    
    print(f"Annotating {len(top_tokens_df)} tokens...")
    
    # Annotate each token
    annotations = []
    for idx, row in top_tokens_df.iterrows():
        token = row['token']
        freq = row['frequency']
        
        annotation = annotate_token(token)
        annotation['frequency'] = freq
        annotation['rank'] = row['rank']
        annotations.append(annotation)
        
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(top_tokens_df)} tokens...")
    
    # Create DataFrame
    df = pd.DataFrame(annotations)
    
    # Reorder columns
    column_order = [
        'rank', 'token', 'frequency', 'length', 'category', 'chemical_meaning',
        'is_valid_smiles', 'atom_count', 'atoms', 'has_aromatic',
        'matched_patterns', 'respects_atom_boundary', 'pattern_details'
    ]
    df = df[column_order]
    
    # Save to CSV
    output_path = DATA_DIR / 'token_annotations.csv'
    df.to_csv(output_path, index=False)
    print(f"\nSaved annotations to {output_path}")
    
    # Print summary statistics
    print("\n--- Annotation Summary ---")
    print(f"Total tokens annotated: {len(df)}")
    print(f"\nCategory distribution:")
    for cat, count in df['category'].value_counts().items():
        print(f"  {cat}: {count}")
    
    print(f"\nValid SMILES fragments: {df['is_valid_smiles'].sum()} / {len(df)}")
    print(f"Respects atom boundary (yes): {(df['respects_atom_boundary'] == 'yes').sum()}")
    print(f"Respects atom boundary (partial): {(df['respects_atom_boundary'] == 'partial').sum()}")
    
    print(f"\nTop 10 annotated tokens:")
    for idx, row in df.head(10).iterrows():
        print(f"  {row['rank']:3d}. '{row['token']}' -> {row['category']}: {row['chemical_meaning']}")
    
    return df


if __name__ == '__main__':
    df = main()







