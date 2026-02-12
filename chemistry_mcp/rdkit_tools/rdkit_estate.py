"""
RDKit EState Tools Module

This module provides specialized tools for EState-related molecular analysis and other
unique functionalities not covered in the basic, extended, or advanced RDKit tools modules.
It includes functions for EState atom typing, EState indices calculation, Fraggle similarity,
and other specialized molecular analyses.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Union, Optional, Any, Tuple, Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, Draw
from rdkit.Chem.AtomPairs.Pairs import pyScorePair, ExplainPairScore, GetAtomPairFingerprintAsBitVect
from rdkit.Chem.AtomPairs.Sheridan import AssignPattyTypes
from rdkit.Chem.AtomPairs.Utils import ExplainAtomCode, GetAtomCode
from rdkit.Chem.EState.AtomTypes import TypeAtoms
from rdkit.Chem.EState import EStateIndices, EState_VSA, Fingerprinter
from rdkit.Chem.Fraggle.FraggleSim import GetFraggleSimilarity, generate_fraggle_fragmentation, isValidRingCut
from rdkit.ML.Cluster import Murtagh
from rdkit.Chem.Fingerprints.ClusterMols import GetDistanceMatrix, ClusterPoints
from rdkit.DataStructs import TanimotoSimilarity
from ...core.llm_tools import llm_tool

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _preprocess_smiles(smiles: str) -> str:
    """
    Preprocess SMILES string by removing whitespace and special characters.

    Args:
        smiles: SMILES string to preprocess

    Returns:
        Preprocessed SMILES string
    """
    return smiles.replace(" ", "").replace("\n", "").replace("\'", "").replace("\"", "").replace(".", "")


def _validate_molecule(smiles: str) -> Chem.Mol:
    """
    Validate SMILES string and convert to RDKit molecule.

    Args:
        smiles: SMILES string to validate

    Returns:
        RDKit molecule object

    Raises:
        ValueError: If SMILES string is invalid
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string.")
    return mol


#------------------------------------------------------------------------------
# EState Analysis Tools
#------------------------------------------------------------------------------

@llm_tool(name="type_atoms_in_molecule_rdkit",
          description="Assign EState types to each atom in a molecule using RDKit")
def type_atoms_in_molecule_rdkit(smiles: str) -> str:
    """
    Assign EState types to each atom in a molecule.

    This function assigns EState types to each atom in a molecule based on its
    electronic state. EState types provide information about the electronic
    environment of atoms, which is useful for QSAR studies and property prediction.

    Args:
        smiles: SMILES notation of the chemical compound. Input SMILES directly
               without any other characters.

    Returns:
        A formatted Markdown string describing the EState types of atoms in the molecule.

    Examples:
        >>> type_atoms_in_molecule_rdkit("CC(=O)OC1=CC=CC=C1C(=O)O")
        # Returns EState atom types for Aspirin
    """
    try:
        # Preprocess input
        smiles = _preprocess_smiles(smiles)

        # Validate molecule
        mol = _validate_molecule(smiles)

        # Assign EState types to each atom
        atom_types = TypeAtoms(mol)

        # Format output
        markdown = f"""## EState Atom Types

**Input SMILES:** `{smiles}`

### Atom EState Types
| Atom Index | Symbol | EState Types |
|------------|--------|--------------|
"""
        for i, types in enumerate(atom_types):
            atom = mol.GetAtomWithIdx(i)
            types_str = ', '.join(types)
            markdown += f"| {i} | {atom.GetSymbol()} | {types_str} |\n"

        markdown += """
### Description
EState atom types classify atoms based on their electronic state and environment.
These types are used in the calculation of EState indices and other EState-based
descriptors, which are useful for QSAR studies and property prediction.
"""
        return markdown

    except Exception as e:
        return f"Error: {e}"


@llm_tool(name="calculate_estate_indices_rdkit",
          description="Calculate EState indices for each atom in a molecule using RDKit")
def calculate_estate_indices_rdkit(smiles: str) -> str:
    """
    Calculate EState indices for each atom in a molecule.

    This function computes the EState indices for each atom in a molecule, which
    represent the electronic state of the atoms. These indices are useful for
    QSAR studies and property prediction.

    Args:
        smiles: SMILES notation of the chemical compound. Input SMILES directly
               without any other characters.

    Returns:
        A formatted Markdown string with the calculated EState indices.

    Examples:
        >>> calculate_estate_indices_rdkit("CC(=O)OC1=CC=CC=C1C(=O)O")
        # Returns EState indices for Aspirin
    """
    try:
        # Preprocess input
        smiles = _preprocess_smiles(smiles)

        # Validate molecule
        mol = _validate_molecule(smiles)

        # Calculate EState indices
        indices = EStateIndices(mol)

        # Format output
        markdown = f"""## EState Indices

**Input SMILES:** `{smiles}`

### Atom EState Indices
| Atom Index | Symbol | EState Index |
|------------|--------|--------------|
"""
        for i, index in enumerate(indices):
            atom = mol.GetAtomWithIdx(i)
            markdown += f"| {i} | {atom.GetSymbol()} | {index:.4f} |\n"

        markdown += """
### Description
EState indices represent the electronic state of atoms in a molecule, taking into
account both the intrinsic electronic state of the atom and the perturbation by
other atoms in the molecule. These indices are useful for QSAR studies and
property prediction.
"""
        return markdown

    except Exception as e:
        return f"Error: {e}"


@llm_tool(name="calculate_estate_vsa_rdkit",
          description="Calculate EState VSA descriptors for a molecule using RDKit")
def calculate_estate_vsa_rdkit(smiles: str) -> str:
    """
    Calculate EState VSA descriptors for a molecule.

    This function computes the EState VSA (van der Waals Surface Area) descriptors
    for a molecule, which combine EState indices with surface area contributions.
    These descriptors are useful for QSAR studies and property prediction.

    Args:
        smiles: SMILES notation of the chemical compound. Input SMILES directly
               without any other characters.

    Returns:
        A formatted Markdown string with the calculated EState VSA descriptors.

    Examples:
        >>> calculate_estate_vsa_rdkit("CC(=O)OC1=CC=CC=C1C(=O)O")
        # Returns EState VSA descriptors for Aspirin
    """
    try:
        # Preprocess input
        smiles = _preprocess_smiles(smiles)

        # Validate molecule
        mol = _validate_molecule(smiles)

        # Define EState VSA functions
        vsa_functions = [
            EState_VSA.EState_VSA1, EState_VSA.EState_VSA2, EState_VSA.EState_VSA3,
            EState_VSA.EState_VSA4, EState_VSA.EState_VSA5, EState_VSA.EState_VSA6,
            EState_VSA.EState_VSA7, EState_VSA.EState_VSA8, EState_VSA.EState_VSA9,
            EState_VSA.EState_VSA10, EState_VSA.EState_VSA11
        ]

        # Calculate EState VSA descriptors
        vsa_values = [func(mol) for func in vsa_functions]

        # Format output
        markdown = f"""## EState VSA Descriptors

**Input SMILES:** `{smiles}`

### EState VSA Values
| Descriptor | Value |
|------------|-------|
"""
        for i, value in enumerate(vsa_values):
            markdown += f"| EState_VSA{i+1} | {value:.4f} |\n"

        markdown += """
### Description
EState VSA descriptors combine EState indices with surface area contributions.
These descriptors divide the van der Waals surface area of a molecule according
to EState indices, providing a way to capture both electronic and steric effects.
They are useful for QSAR studies and property prediction.
"""
        return markdown

    except Exception as e:
        return f"Error: {e}"


@llm_tool(name="generate_estate_fingerprint_rdkit",
          description="Generate EState fingerprint for a molecule using RDKit")
def generate_estate_fingerprint_rdkit(smiles: str) -> str:
    """
    Generate EState fingerprint for a molecule.

    This function generates the EState fingerprint for a molecule, which encodes
    the presence and electronic state of various atom types in the molecule.
    This fingerprint is useful for similarity searching and QSAR studies.

    Args:
        smiles: SMILES notation of the chemical compound. Input SMILES directly
               without any other characters.

    Returns:
        A formatted Markdown string with the generated EState fingerprint.

    Examples:
        >>> generate_estate_fingerprint_rdkit("CC(=O)OC1=CC=CC=C1C(=O)O")
        # Returns EState fingerprint for Aspirin
    """
    try:
        # Preprocess input
        smiles = _preprocess_smiles(smiles)

        # Validate molecule
        mol = _validate_molecule(smiles)

        # Generate EState fingerprint
        counts, sums = Fingerprinter.FingerprintMol(mol)

        # Get non-zero indices
        nonzero_indices = np.nonzero(counts)[0]

        # Format output
        markdown = f"""## EState Fingerprint

**Input SMILES:** `{smiles}`

### Atom Type Counts
| Atom Type | Count |
|-----------|-------|
"""
        for idx in nonzero_indices:
            markdown += f"| {idx+1} | {counts[idx]} |\n"

        markdown += """
### EState Index Sums
| Atom Type | EState Sum |
|-----------|------------|
"""
        for idx in nonzero_indices:
            markdown += f"| {idx+1} | {sums[idx]:.4f} |\n"

        markdown += """
### Description
The EState fingerprint encodes the presence and electronic state of various atom
types in a molecule. It consists of two parts: counts of each atom type and sums
of EState indices for each atom type. This fingerprint is useful for similarity
searching and QSAR studies.
"""
        return markdown

    except Exception as e:
        return f"Error: {e}"


#------------------------------------------------------------------------------
# Atom Pair and Patty Type Tools
#------------------------------------------------------------------------------

@llm_tool(name="explain_atom_pair_score_rdkit",
          description="Explain the pair score for a directly connected atom pair in a molecule using RDKit")
def explain_atom_pair_score_rdkit(smiles: str, atom_idx1: int, atom_idx2: int) -> str:
    """
    Explain the pair score for a directly connected atom pair in a molecule.

    This function calculates and explains the pair score for a directly connected
    atom pair in a molecule. The pair score is based on the atom types and the
    distance between the atoms.

    Args:
        smiles: SMILES notation of the chemical compound. Input SMILES directly
               without any other characters.
        atom_idx1: Index of the first atom.
        atom_idx2: Index of the second atom.

    Returns:
        A formatted Markdown string explaining the pair score.

    Examples:
        >>> explain_atom_pair_score_rdkit("CC(=O)OC1=CC=CC=C1C(=O)O", 0, 1)
        # Returns explanation of pair score for atoms 0 and 1 in Aspirin
    """
    try:
        # Preprocess input
        smiles = _preprocess_smiles(smiles)

        # Validate molecule
        mol = _validate_molecule(smiles)

        # Check atom indices
        if atom_idx1 >= mol.GetNumAtoms() or atom_idx2 >= mol.GetNumAtoms() or atom_idx1 < 0 or atom_idx2 < 0:
            raise ValueError("Atom index out of range.")

        # Get atoms
        atom1 = mol.GetAtomWithIdx(atom_idx1)
        atom2 = mol.GetAtomWithIdx(atom_idx2)

        # Calculate pair score
        score = pyScorePair(atom1, atom2, dist=1)

        # Explain score
        explanation = ExplainPairScore(score)

        # Format output
        markdown = f"""## Atom Pair Score Explanation

**Input SMILES:** `{smiles}`

### Atom Information
- **Atom 1 Index:** {atom_idx1}
- **Atom 1 Symbol:** {atom1.GetSymbol()}
- **Atom 2 Index:** {atom_idx2}
- **Atom 2 Symbol:** {atom2.GetSymbol()}

### Pair Score
- **Score:** {score}

### Score Explanation
- **First Atom:** {explanation[0]}
- **Distance:** {explanation[1]}
- **Second Atom:** {explanation[2]}

### Description
The atom pair score encodes information about a pair of atoms and the distance
between them. It is used in atom pair fingerprints, which are useful for
similarity searching and QSAR studies.
"""
        return markdown

    except Exception as e:
        return f"Error: {e}"


@llm_tool(name="get_atom_pair_fingerprint_rdkit",
          description="Generate atom pair fingerprint for a molecule as a bit vector using RDKit")
def get_atom_pair_fingerprint_rdkit(smiles: str) -> str:
    """
    Generate atom pair fingerprint for a molecule as a bit vector.

    This function generates an atom pair fingerprint for a molecule as a bit vector.
    The atom pair fingerprint encodes the presence of atom pairs with specific
    atom types and distances.

    Args:
        smiles: SMILES notation of the chemical compound. Input SMILES directly
               without any other characters.

    Returns:
        A formatted Markdown string with the generated atom pair fingerprint.

    Examples:
        >>> get_atom_pair_fingerprint_rdkit("CC(=O)OC1=CC=CC=C1C(=O)O")
        # Returns atom pair fingerprint for Aspirin
    """
    try:
        # Preprocess input
        smiles = _preprocess_smiles(smiles)

        # Validate molecule
        mol = _validate_molecule(smiles)

        # Generate atom pair fingerprint
        fp = GetAtomPairFingerprintAsBitVect(mol)
        on_bits = list(fp.GetOnBits())

        # Format output
        markdown = f"""## Atom Pair Fingerprint

**Input SMILES:** `{smiles}`

### Fingerprint Information
- **Total Bits:** {fp.GetNumBits()}
- **On Bits:** {fp.GetNumOnBits()}
- **Density:** {fp.GetNumOnBits() / fp.GetNumBits():.4f}

### On Bits (First 20)
"""
        # Show only first 20 bits to avoid excessive output
        for i, bit in enumerate(on_bits[:20]):
            markdown += f"- {bit}\n"

        if len(on_bits) > 20:
            markdown += f"\n... and {len(on_bits) - 20} more bits (total: {len(on_bits)})"

        markdown += """
### Description
The atom pair fingerprint encodes the presence of atom pairs with specific atom
types and distances. It is represented as a bit vector, where each bit corresponds
to a specific atom pair configuration. This fingerprint is useful for similarity
searching and QSAR studies.
"""
        return markdown

    except Exception as e:
        return f"Error: {e}"


#------------------------------------------------------------------------------
# Fraggle Similarity Tools
#------------------------------------------------------------------------------

@llm_tool(name="get_fraggle_similarity_rdkit",
          description="Calculate Fraggle similarity between two molecules using RDKit")
def get_fraggle_similarity_rdkit(smiles1: str, smiles2: str, tversky_thresh: float = 0.8) -> str:
    """
    Calculate Fraggle similarity between two molecules.

    This function calculates the Fraggle similarity between two molecules. Fraggle
    similarity is based on fragmenting molecules and comparing the fragments using
    the Tversky similarity measure.

    Args:
        smiles1: SMILES notation of the first molecule.
        smiles2: SMILES notation of the second molecule.
        tversky_thresh: Tversky threshold for similarity. Default is 0.8.

    Returns:
        A formatted Markdown string with the Fraggle similarity results.

    Examples:
        >>> get_fraggle_similarity_rdkit("CC(=O)OC1=CC=CC=C1C(=O)O", "CC(=O)O")
        # Returns Fraggle similarity between Aspirin and acetic acid
    """
    try:
        # Preprocess input
        smiles1 = _preprocess_smiles(smiles1)
        smiles2 = _preprocess_smiles(smiles2)

        # Validate molecules
        mol1 = _validate_molecule(smiles1)
        mol2 = _validate_molecule(smiles2)

        # Calculate Fraggle similarity
        sim, match = GetFraggleSimilarity(mol1, mol2, tversky_thresh)

        # Format output
        markdown = f"""## Fraggle Similarity

**Molecule 1 SMILES:** `{smiles1}`
**Molecule 2 SMILES:** `{smiles2}`
**Tversky Threshold:** {tversky_thresh}

### Similarity Results
- **Similarity Score:** {sim:.4f}
- **Matching Substructure:** `{match}`

### Description
Fraggle similarity is based on fragmenting molecules and comparing the fragments
using the Tversky similarity measure. It is useful for identifying similar
substructures between molecules, even when the overall structures are different.
"""
        return markdown

    except Exception as e:
        return f"Error: {e}"


@llm_tool(name="generate_fraggle_fragments_rdkit",
          description="Generate all possible Fraggle fragmentations for a molecule using RDKit")
def generate_fraggle_fragments_rdkit(smiles: str) -> str:
    """
    Generate all possible Fraggle fragmentations for a molecule.

    This function generates all possible Fraggle fragmentations for a molecule.
    Fraggle fragmentations are created by breaking bonds in the molecule to
    create fragments.

    Args:
        smiles: SMILES notation of the chemical compound. Input SMILES directly
               without any other characters.

    Returns:
        A formatted Markdown string with the generated Fraggle fragments.

    Examples:
        >>> generate_fraggle_fragments_rdkit("CC(=O)OC1=CC=CC=C1C(=O)O")
        # Returns Fraggle fragments for Aspirin
    """
    try:
        # Preprocess input
        smiles = _preprocess_smiles(smiles)

        # Validate molecule
        mol = _validate_molecule(smiles)

        # Generate Fraggle fragments
        fragments = generate_fraggle_fragmentation(mol)

        # Sort fragments
        sorted_fragments = sorted(['.'.join(sorted(s.split('.'))) for s in fragments])

        # Format output
        markdown = f"""## Fraggle Fragmentations

**Input SMILES:** `{smiles}`

### Generated Fragments
"""
        for i, fragment in enumerate(sorted_fragments, 1):
            markdown += f"{i}. `{fragment}`\n"

        markdown += f"""
### Summary
- **Total Fragments:** {len(sorted_fragments)}

### Description
Fraggle fragmentations are created by breaking bonds in the molecule to create
fragments. These fragments can be used for similarity searching and identifying
common substructures between molecules.
"""
        return markdown

    except Exception as e:
        return f"Error: {e}"


@llm_tool(name="check_valid_ring_cut_rdkit",
          description="Check if a molecule is a valid ring cut using RDKit")
def check_valid_ring_cut_rdkit(smiles: str) -> str:
    """
    Check if a molecule is a valid ring cut.

    This function checks if a molecule is a valid ring cut. A valid ring cut
    is a molecule that can be formed by cutting a ring in another molecule.

    Args:
        smiles: SMILES notation of the chemical compound. Input SMILES directly
               without any other characters.

    Returns:
        A formatted Markdown string with the ring cut validation results.

    Examples:
        >>> check_valid_ring_cut_rdkit("CC(=O)OC1=CC=CC=C1C(=O)O")
        # Returns ring cut validation for Aspirin
    """
    try:
        # Preprocess input
        smiles = _preprocess_smiles(smiles)

        # Validate molecule
        mol = _validate_molecule(smiles)

        # Check if molecule is a valid ring cut
        is_valid = isValidRingCut(mol)

        # Format output
        markdown = f"""## Ring Cut Validation

**Input SMILES:** `{smiles}`

### Validation Result
- **Is Valid Ring Cut:** {"Yes" if is_valid else "No"}

### Description
A valid ring cut is a molecule that can be formed by cutting a ring in another
molecule. This validation is useful for fragment-based drug design and
understanding the structural relationships between molecules.
"""
        return markdown

    except Exception as e:
        return f"Error: {e}"
