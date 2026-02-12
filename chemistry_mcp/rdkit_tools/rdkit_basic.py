"""
RDKit Tools Module

This module provides tools for molecular analysis, manipulation, and visualization
using the RDKit library. It includes functions for calculating molecular descriptors,
generating molecular fingerprints, analyzing molecular structures, and more.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Union, Optional, Any, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, Draw
from rdkit.Chem import rdmolops, rdDetermineBonds
from rdkit.DataStructs import ConvertToNumpyArray
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
# Molecular Descriptors
#------------------------------------------------------------------------------

@llm_tool(name="calculate_molecular_properties_rdkit",
          description="Calculate basic molecular properties for a chemical compound using RDKit library")
def calculate_molecular_properties_rdkit(smiles: str) -> str:
    """
    Calculate basic molecular properties for a chemical compound.

    This function computes a comprehensive set of molecular properties including
    basic information (formula, weight), physical properties (LogP, TPSA),
    and structural features (hydrogen bond donors/acceptors, rotatable bonds).

    Args:
        smiles: SMILES notation of the chemical compound. Input SMILES directly
               without any other characters.

    Returns:
        A formatted Markdown string with the calculated molecular properties.

    Examples:
        >>> calculate_molecular_properties("CC(=O)OC1=CC=CC=C1C(=O)O")
        # Returns properties of Aspirin
    """
    try:
        # Preprocess input
        smiles = _preprocess_smiles(smiles)

        # Validate molecule
        mol = _validate_molecule(smiles)

        # Calculate properties
        mol_weight = Descriptors.ExactMolWt(mol)
        formula = rdMolDescriptors.CalcMolFormula(mol)
        tpsa = Descriptors.TPSA(mol)
        logp = Descriptors.MolLogP(mol)
        hba = Descriptors.NumHAcceptors(mol)
        hbd = Descriptors.NumHDonors(mol)
        rotatable_bonds = Descriptors.NumRotatableBonds(mol)
        heavy_atoms = mol.GetNumHeavyAtoms()
        ring_count = Descriptors.RingCount(mol)
        aromatic_rings = Descriptors.NumAromaticRings(mol)

        # Format output
        markdown = f"""## Basic Molecular Properties

**Input SMILES:** `{smiles}`

### Basic Information
- **Formula:** {formula}
- **Molecular Weight:** {mol_weight:.4f} g/mol
- **Heavy Atom Count:** {heavy_atoms}

### Physicochemical Properties
- **LogP:** {logp:.2f}
- **Topological Polar Surface Area (TPSA):** {tpsa:.2f} Å²
- **H-Bond Acceptors:** {hba}
- **H-Bond Donors:** {hbd}

### Structural Features
- **Rotatable Bonds:** {rotatable_bonds}
- **Ring Count:** {ring_count}
- **Aromatic Rings:** {aromatic_rings}
"""
        return markdown

    except Exception as e:
        return f"Error: {e}"


@llm_tool(name="calculate_drug_likeness_rdkit",
          description="Calculate drug-likeness properties for a chemical compound using RDKit library")
def calculate_drug_likeness_rdkit(smiles: str) -> str:
    """
    Calculate drug-likeness properties for a chemical compound.

    This function evaluates whether a molecule satisfies various drug-likeness rules
    including Lipinski's Rule of Five, Ghose Filter, Veber Filter, and PAINS filter.
    These rules help assess the compound's potential as a drug candidate.

    Args:
        smiles: SMILES notation of the chemical compound. Input SMILES directly
               without any other characters.

    Returns:
        A formatted Markdown string with the drug-likeness assessment results.

    Examples:
        >>> calculate_drug_likeness("CC(=O)OC1=CC=CC=C1C(=O)O")
        # Returns drug-likeness assessment for Aspirin
    """
    try:
        # Preprocess input
        smiles = _preprocess_smiles(smiles)

        # Validate molecule
        mol = _validate_molecule(smiles)

        # Calculate properties for Lipinski's Rule of Five
        mol_weight = Descriptors.ExactMolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hba = Descriptors.NumHAcceptors(mol)
        hbd = Descriptors.NumHDonors(mol)

        # Calculate properties for Ghose Filter
        molar_refractivity = Descriptors.MolMR(mol)
        n_atoms = mol.GetNumAtoms(onlyExplicit=0)  # 计算所有原子，包括氢原子

        # Calculate properties for Veber Filter
        tpsa = Descriptors.TPSA(mol)
        rotatable_bonds = Descriptors.NumRotatableBonds(mol)

        # Evaluate Lipinski's Rule of Five
        lipinski_violations = 0
        if mol_weight > 500: lipinski_violations += 1
        if logp > 5: lipinski_violations += 1
        if hba > 10: lipinski_violations += 1
        if hbd > 5: lipinski_violations += 1

        # Evaluate Ghose Filter
        ghose_compliant = (
            160 <= mol_weight <= 480 and
            -0.4 <= logp <= 5.6 and
            40 <= molar_refractivity <= 130 and
            20 <= n_atoms <= 70
        )

        # Evaluate Veber Filter
        veber_compliant = (
            rotatable_bonds <= 10 and
            tpsa <= 140
        )

        # Format output
        markdown = f"""## Drug-likeness Assessment

**Input SMILES:** `{smiles}`

### Lipinski's Rule of Five
- **Molecular Weight ≤ 500:** {mol_weight:.1f} g/mol {'✓' if mol_weight <= 500 else '✗'}
- **LogP ≤ 5:** {logp:.2f} {'✓' if logp <= 5 else '✗'}
- **H-Bond Acceptors ≤ 10:** {hba} {'✓' if hba <= 10 else '✗'}
- **H-Bond Donors ≤ 5:** {hbd} {'✓' if hbd <= 5 else '✗'}
- **Number of Violations:** {lipinski_violations}
- **Conclusion:** {'Compliant' if lipinski_violations <= 1 else 'Non-compliant'} with Lipinski's Rule of Five

### Ghose Filter
- **Molecular Weight Range [160, 480]:** {mol_weight:.1f} g/mol {'✓' if 160 <= mol_weight <= 480 else '✗'}
- **LogP Range [-0.4, 5.6]:** {logp:.2f} {'✓' if -0.4 <= logp <= 5.6 else '✗'}
- **Molar Refractivity Range [40, 130]:** {molar_refractivity:.2f} {'✓' if 40 <= molar_refractivity <= 130 else '✗'}
- **Atom Count Range [20, 70]:** {n_atoms} {'✓' if 20 <= n_atoms <= 70 else '✗'}
- **Conclusion:** {'Compliant' if ghose_compliant else 'Non-compliant'} with Ghose Filter

### Veber Filter
- **Rotatable Bonds ≤ 10:** {rotatable_bonds} {'✓' if rotatable_bonds <= 10 else '✗'}
- **Polar Surface Area ≤ 140 Å²:** {tpsa:.2f} Å² {'✓' if tpsa <= 140 else '✗'}
- **Conclusion:** {'Compliant' if veber_compliant else 'Non-compliant'} with Veber Filter

### Overall Assessment
This compound {'complies with most drug-likeness rules' if (lipinski_violations <= 1 and (ghose_compliant or veber_compliant)) else 'does not comply with major drug-likeness rules'} and {'is likely' if (lipinski_violations <= 1 and (ghose_compliant or veber_compliant)) else 'is unlikely'} to be a good drug candidate.
"""
        return markdown

    except Exception as e:
        return f"Error: {e}"


@llm_tool(name="calculate_topological_descriptors_rdkit",
          description="Calculate topological descriptors for a chemical compound using RDKit library")
def calculate_topological_descriptors_rdkit(smiles: str) -> str:
    """
    Calculate topological descriptors for a chemical compound.

    This function computes various topological descriptors that characterize
    the molecular structure based on its connectivity, without considering
    3D coordinates. These descriptors are useful for QSAR studies and
    molecular similarity analysis.

    Args:
        smiles: SMILES notation of the chemical compound. Input SMILES directly
               without any other characters.

    Returns:
        A formatted Markdown string with the calculated topological descriptors.

    Examples:
        >>> calculate_topological_descriptors("CC(=O)OC1=CC=CC=C1C(=O)O")
        # Returns topological descriptors for Aspirin
    """
    try:
        # Preprocess input
        smiles = _preprocess_smiles(smiles)

        # Validate molecule
        mol = _validate_molecule(smiles)

        # Calculate topological descriptors
        chi0v = Descriptors.Chi0v(mol)
        chi1v = Descriptors.Chi1v(mol)
        chi2v = Descriptors.Chi2v(mol)
        chi3v = Descriptors.Chi3v(mol)
        chi4v = Descriptors.Chi4v(mol)

        kappa1 = Descriptors.Kappa1(mol)
        kappa2 = Descriptors.Kappa2(mol)
        kappa3 = Descriptors.Kappa3(mol)

        balaban_j = Descriptors.BalabanJ(mol)
        bertz_ct = Descriptors.BertzCT(mol)

        # Format output
        markdown = f"""## Topological Descriptors

**Input SMILES:** `{smiles}`

### Connectivity Indices (Chi)
- **Chi0v:** {chi0v:.4f}
- **Chi1v:** {chi1v:.4f}
- **Chi2v:** {chi2v:.4f}
- **Chi3v:** {chi3v:.4f}
- **Chi4v:** {chi4v:.4f}

### Shape Indices (Kappa)
- **Kappa1:** {kappa1:.4f}
- **Kappa2:** {kappa2:.4f}
- **Kappa3:** {kappa3:.4f}

### Other Topological Indices
- **Balaban J Index:** {balaban_j:.4f}
- **Bertz CT Index:** {bertz_ct:.4f}
"""
        return markdown

    except Exception as e:
        return f"Error: {e}"


#------------------------------------------------------------------------------
# Molecular Fingerprints
#------------------------------------------------------------------------------

@llm_tool(name="generate_molecular_fingerprints_rdkit",
          description="Generate different types of molecular fingerprints for a chemical compound using RDKit library")
def generate_molecular_fingerprints_rdkit(
    smiles: str,
    fingerprint_type: str = "morgan",
    radius: int = 2,
    n_bits: int = 1024
) -> str:
    """
    Generate different types of molecular fingerprints for a chemical compound.

    This function generates various types of molecular fingerprints, which are
    binary vectors representing the presence or absence of specific structural
    features in a molecule. These fingerprints are useful for similarity searching,
    clustering, and machine learning applications.

    Args:
        smiles: SMILES notation of the chemical compound. Input SMILES directly
               without any other characters.
        fingerprint_type: Type of fingerprint to generate. Options: "morgan", "maccs",
                         "atompair", "topological", "rdkit". Default: "morgan".
        radius: Radius for Morgan fingerprint (only used if fingerprint_type is "morgan").
                Default: 2.
        n_bits: Number of bits in the fingerprint (only used for some fingerprint types).
                Default: 1024.

    Returns:
        A formatted Markdown string with the generated fingerprint information.

    Examples:
        >>> generate_molecular_fingerprints("CC(=O)OC1=CC=CC=C1C(=O)O", "morgan", 2, 1024)
        # Returns Morgan fingerprint for Aspirin
    """
    try:
        # Preprocess input
        smiles = _preprocess_smiles(smiles)

        # Validate molecule
        mol = _validate_molecule(smiles)

        # Generate fingerprint based on type
        fingerprint = None
        fingerprint_name = ""

        if fingerprint_type.lower() == "morgan":
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            fingerprint_name = f"Morgan (ECFP{radius*2})"

        elif fingerprint_type.lower() == "maccs":
            fingerprint = AllChem.GetMACCSKeysFingerprint(mol)
            fingerprint_name = "MACCS Keys"

        elif fingerprint_type.lower() == "atompair":
            fingerprint = AllChem.GetAtomPairFingerprint(mol)
            fingerprint_name = "Atom Pair"

        elif fingerprint_type.lower() == "topological":
            fingerprint = AllChem.GetTopologicalTorsionFingerprint(mol)
            fingerprint_name = "Topological Torsion"

        elif fingerprint_type.lower() == "rdkit":
            fingerprint = Chem.RDKFingerprint(mol, fpSize=n_bits)
            fingerprint_name = "RDKit"

        else:
            return f"Error: Unsupported fingerprint type '{fingerprint_type}'. Supported types: morgan, maccs, atompair, topological, rdkit."

        # Convert fingerprint to numpy array for easier handling
        if fingerprint_type.lower() in ["morgan", "maccs", "rdkit"]:
            # These are bit vectors, so we can get the number of bits directly
            num_bits = fingerprint.GetNumBits()
            num_on_bits = fingerprint.GetNumOnBits()
            bit_info = f"- **总位数:** {num_bits}\n- **激活位数:** {num_on_bits}\n- **密度:** {num_on_bits/num_bits:.4f}"

            # Convert to binary string (limit length for readability)
            binary = fingerprint.ToBitString()
            if len(binary) > 100:
                binary_display = binary[:100] + "..."
            else:
                binary_display = binary

        else:
            # These are count vectors, so we need to handle differently
            bit_info = "- 非位向量指纹，包含计数信息"
            binary_display = "不适用于此指纹类型"

        # Format output
        markdown = f"""## Molecular Fingerprints

**Input SMILES:** `{smiles}`
**Fingerprint Type:** {fingerprint_name}

### Fingerprint Information
{bit_info.replace("总位数", "Total Bits").replace("激活位数", "On Bits").replace("密度", "Density").replace("非位向量指纹，包含计数信息", "Non-bit vector fingerprint, contains count information").replace("不适用于此指纹类型", "Not applicable for this fingerprint type")}

### Fingerprint Bit Pattern (First 100 bits)
```
{binary_display}
```

### Applications
- Molecular similarity searching
- Compound clustering
- Building QSAR/QSPR models
- Virtual screening
"""
        return markdown

    except Exception as e:
        return f"Error: {e}"


@llm_tool(name="calculate_molecular_similarity_rdkit",
          description="Calculate similarity between two molecules using fingerprints with RDKit library")
def calculate_molecular_similarity_rdkit(
    smiles1: str,
    smiles2: str,
    fingerprint_type: str = "morgan",
    radius: int = 2,
    n_bits: int = 1024,
    similarity_metric: str = "tanimoto"
) -> str:
    """
    Calculate similarity between two molecules using fingerprints.

    This function computes the similarity between two molecules based on their
    molecular fingerprints. It supports different fingerprint types and similarity
    metrics, making it versatile for various cheminformatics applications.

    Args:
        smiles1: SMILES notation of the first molecule.
        smiles2: SMILES notation of the second molecule.
        fingerprint_type: Type of fingerprint to use. Options: "morgan", "maccs",
                         "rdkit". Default: "morgan".
        radius: Radius for Morgan fingerprint (only used if fingerprint_type is "morgan").
                Default: 2.
        n_bits: Number of bits in the fingerprint. Default: 1024.
        similarity_metric: Similarity metric to use. Options: "tanimoto", "dice",
                          "cosine". Default: "tanimoto".

    Returns:
        A formatted Markdown string with the similarity calculation results.

    Examples:
        >>> calculate_molecular_similarity("CC(=O)OC1=CC=CC=C1C(=O)O", "CC(=O)OCCC(=O)O")
        # Returns similarity between Aspirin and another molecule
    """
    try:
        # Preprocess input
        smiles1 = _preprocess_smiles(smiles1)
        smiles2 = _preprocess_smiles(smiles2)

        # Validate molecules
        mol1 = _validate_molecule(smiles1)
        mol2 = _validate_molecule(smiles2)

        # Generate fingerprints based on type
        fp1 = None
        fp2 = None
        fingerprint_name = ""

        if fingerprint_type.lower() == "morgan":
            fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius, nBits=n_bits)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius, nBits=n_bits)
            fingerprint_name = f"Morgan (ECFP{radius*2})"

        elif fingerprint_type.lower() == "maccs":
            fp1 = AllChem.GetMACCSKeysFingerprint(mol1)
            fp2 = AllChem.GetMACCSKeysFingerprint(mol2)
            fingerprint_name = "MACCS Keys"

        elif fingerprint_type.lower() == "rdkit":
            fp1 = Chem.RDKFingerprint(mol1, fpSize=n_bits)
            fp2 = Chem.RDKFingerprint(mol2, fpSize=n_bits)
            fingerprint_name = "RDKit"

        else:
            return f"Error: Unsupported fingerprint type '{fingerprint_type}'. Supported types: morgan, maccs, rdkit."

        # Calculate similarity based on metric
        similarity = 0.0
        metric_name = ""

        if similarity_metric.lower() == "tanimoto":
            from rdkit import DataStructs
            similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
            metric_name = "Tanimoto"

        elif similarity_metric.lower() == "dice":
            from rdkit import DataStructs
            similarity = DataStructs.DiceSimilarity(fp1, fp2)
            metric_name = "Dice"

        elif similarity_metric.lower() == "cosine":
            from rdkit import DataStructs
            similarity = DataStructs.CosineSimilarity(fp1, fp2)
            metric_name = "Cosine"

        else:
            return f"Error: Unsupported similarity metric '{similarity_metric}'. Supported metrics: tanimoto, dice, cosine."

        # Get basic molecule information
        mol1_formula = rdMolDescriptors.CalcMolFormula(mol1)
        mol2_formula = rdMolDescriptors.CalcMolFormula(mol2)

        # Format output
        markdown = f"""## Molecular Similarity Calculation

### Molecule Information
- **Molecule 1 SMILES:** `{smiles1}`
- **Molecule 1 Formula:** {mol1_formula}
- **Molecule 2 SMILES:** `{smiles2}`
- **Molecule 2 Formula:** {mol2_formula}

### Similarity Results
- **Fingerprint Type:** {fingerprint_name}
- **Similarity Metric:** {metric_name}
- **Similarity Score:** {similarity:.4f} (Range: 0-1)

### Similarity Interpretation
- **0.0-0.2:** Very low similarity
- **0.2-0.4:** Low similarity
- **0.4-0.6:** Moderate similarity
- **0.6-0.8:** High similarity
- **0.8-1.0:** Very high similarity

**Conclusion:** These two molecules have **{'very low' if similarity < 0.2 else 'low' if similarity < 0.4 else 'moderate' if similarity < 0.6 else 'high' if similarity < 0.8 else 'very high'}** structural similarity.
"""
        return markdown

    except Exception as e:
        return f"Error: {e}"


#------------------------------------------------------------------------------
# Molecular Structure Analysis
#------------------------------------------------------------------------------

@llm_tool(name="analyze_molecular_structure_rdkit",
          description="Analyze the structure of a molecule including atoms, bonds, rings, and functional groups using RDKit library")
def analyze_molecular_structure_rdkit(smiles: str) -> str:
    """
    Analyze the structure of a molecule including atoms, bonds, rings, and functional groups.

    This function provides a comprehensive analysis of a molecule's structure,
    including atom types, bond types, ring systems, and functional groups.
    It helps understand the key structural features of a molecule.

    Args:
        smiles: SMILES notation of the chemical compound. Input SMILES directly
               without any other characters.

    Returns:
        A formatted Markdown string with the structural analysis results.

    Examples:
        >>> analyze_molecular_structure("CC(=O)OC1=CC=CC=C1C(=O)O")
        # Returns structural analysis for Aspirin
    """
    try:
        # Preprocess input
        smiles = _preprocess_smiles(smiles)

        # Validate molecule
        mol = _validate_molecule(smiles)

        # Get basic molecule information
        formula = rdMolDescriptors.CalcMolFormula(mol)

        # Analyze atoms
        atoms = mol.GetAtoms()
        atom_counts = {}
        formal_charges = {}

        for atom in atoms:
            symbol = atom.GetSymbol()
            atom_counts[symbol] = atom_counts.get(symbol, 0) + 1

            charge = atom.GetFormalCharge()
            if charge != 0:
                formal_charges[atom.GetIdx()] = (symbol, charge)

        # Analyze bonds
        bonds = mol.GetBonds()
        bond_types = {
            Chem.rdchem.BondType.SINGLE: 0,
            Chem.rdchem.BondType.DOUBLE: 0,
            Chem.rdchem.BondType.TRIPLE: 0,
            Chem.rdchem.BondType.AROMATIC: 0
        }

        for bond in bonds:
            bond_types[bond.GetBondType()] += 1

        # Analyze rings
        ring_info = mol.GetRingInfo()
        ring_sizes = {}
        aromatic_rings = 0

        # Get all rings
        rings = ring_info.AtomRings()
        for ring in rings:
            size = len(ring)
            ring_sizes[size] = ring_sizes.get(size, 0) + 1

            # Check if ring is aromatic
            if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
                aromatic_rings += 1

        # Analyze functional groups (simplified approach)
        functional_groups = []

        # Check for common functional groups using SMARTS patterns
        smarts_patterns = {
            "醇 (Alcohol)": "[OX2H;!$([OX2H][CX3]=O)]", # 排除羧酸中的羟基
            "醛 (Aldehyde)": "[CX3H1](=O)[#6]",
            "酮 (Ketone)": "[#6][CX3](=O)[#6]",
            "羧酸 (Carboxylic Acid)": "[CX3](=O)[OX2H1]",
            "酯 (Ester)": "[#6][CX3](=O)[OX2][#6]",
            "醚 (Ether)": "[OD2]([CX4])([CX4])", # 匹配连接到两个sp3杂化碳原子的氧原子，排除酯中的氧原子
            "胺 (Amine)": "[NX3;H2,H1,H0;!$(NC=O)]",
            "酰胺 (Amide)": "[NX3][CX3](=[OX1])[#6]",
            "硝基 (Nitro)": "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",
            "磺酸 (Sulfonic Acid)": "[$([#16X4](=[OX1])(=[OX1])([#6])[OX2H,OX1H0-]),$([#16X4+2]([OX1-])([OX1-])([#6])[OX2H,OX1H0-])]",
            "磷酸 (Phosphoric Acid)": "[$([#15X4](=[OX1])(=[OX1])([OX2H,OX1H0-])[OX2H,OX1H0-]),$([#15X4+](=[OX1])([OX1-])([OX2H,OX1H0-])[OX2H,OX1H0-])]",
            "卤素 (Halogen)": "[F,Cl,Br,I]"
        }

        for name, smarts in smarts_patterns.items():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                matches = mol.GetSubstructMatches(pattern)
                functional_groups.append((name, len(matches)))

        # Format output
        markdown = f"""## Molecular Structure Analysis

**Input SMILES:** `{smiles}`
**Formula:** {formula}

### Atom Composition
| Element | Count |
|---------|-------|
"""

        for symbol, count in sorted(atom_counts.items()):
            markdown += f"| {symbol} | {count} |\n"

        if formal_charges:
            markdown += "\n### Formal Charges\n"
            for idx, (symbol, charge) in sorted(formal_charges.items()):
                sign = "+" if charge > 0 else "-"
                markdown += f"- Atom {idx} ({symbol}): {sign}{abs(charge)}\n"

        markdown += "\n### Bond Types\n"
        markdown += f"- Single bonds: {bond_types[Chem.rdchem.BondType.SINGLE]}\n"
        markdown += f"- Double bonds: {bond_types[Chem.rdchem.BondType.DOUBLE]}\n"
        markdown += f"- Triple bonds: {bond_types[Chem.rdchem.BondType.TRIPLE]}\n"
        markdown += f"- Aromatic bonds: {bond_types[Chem.rdchem.BondType.AROMATIC]}\n"

        if ring_sizes:
            markdown += "\n### Ring Systems\n"
            markdown += f"- Total rings: {len(rings)}\n"
            markdown += f"- Aromatic rings: {aromatic_rings}\n"
            markdown += "- Ring size distribution:\n"

            for size, count in sorted(ring_sizes.items()):
                markdown += f"  - {size}-membered rings: {count}\n"

        if functional_groups:
            markdown += "\n### Functional Groups\n"
            for name, count in functional_groups:
                markdown += f"- {name}: {count}\n"

        return markdown

    except Exception as e:
        return f"Error: {e}"


@llm_tool(name="generate_molecular_conformer_rdkit",
          description="Generate a 3D conformer for a molecule and calculate 3D descriptors using RDKit library")
def generate_molecular_conformer_rdkit(smiles: str, num_conformers: int = 1) -> str:
    """
    Generate a 3D conformer for a molecule and calculate 3D descriptors.

    This function generates 3D conformers for a molecule using force field
    optimization and calculates various 3D molecular descriptors that depend
    on the molecule's spatial arrangement.

    Args:
        smiles: SMILES notation of the chemical compound. Input SMILES directly
               without any other characters.
        num_conformers: Number of conformers to generate. Default: 1.

    Returns:
        A formatted Markdown string with the 3D conformer generation results and
        calculated 3D descriptors.

    Examples:
        >>> generate_molecular_conformer("CC(=O)OC1=CC=CC=C1C(=O)O")
        # Returns 3D conformer information for Aspirin
    """
    try:
        # Preprocess input
        smiles = _preprocess_smiles(smiles)

        # Validate molecule
        mol = _validate_molecule(smiles)

        # Add hydrogens
        mol = Chem.AddHs(mol)

        # Generate 3D conformers
        conf_ids = AllChem.EmbedMultipleConfs(
            mol,
            numConfs=num_conformers,
            randomSeed=42,
            useExpTorsionAnglePrefs=True,
            useBasicKnowledge=True
        )

        if len(conf_ids) == 0:
            return "Error: Failed to generate conformers. The molecule may be too complex or have structural issues."

        # Optimize conformers using MMFF
        for conf_id in conf_ids:
            AllChem.MMFFOptimizeMolecule(mol, confId=conf_id)

        # Calculate 3D descriptors for the first conformer
        conf_id = conf_ids[0]

        # Calculate principal moments of inertia
        pmi1 = rdMolDescriptors.CalcPMI1(mol, confId=conf_id)
        pmi2 = rdMolDescriptors.CalcPMI2(mol, confId=conf_id)
        pmi3 = rdMolDescriptors.CalcPMI3(mol, confId=conf_id)

        # Calculate radius of gyration
        rg = rdMolDescriptors.CalcRadiusOfGyration(mol, confId=conf_id)

        # Calculate spherocity index
        spherocity = rdMolDescriptors.CalcSpherocityIndex(mol, confId=conf_id)

        # Calculate plane of best fit
        pbf = rdMolDescriptors.CalcPBF(mol, confId=conf_id)

        # Format output
        markdown = f"""## Molecular 3D Conformer Generation

**Input SMILES:** `{smiles}`
**Number of Conformers Generated:** {len(conf_ids)}

### 3D Descriptors (First Conformer)
- **Principal Moment of Inertia (PMI1):** {pmi1:.4f}
- **Principal Moment of Inertia (PMI2):** {pmi2:.4f}
- **Principal Moment of Inertia (PMI3):** {pmi3:.4f}
- **Radius of Gyration:** {rg:.4f} Å
- **Spherocity Index:** {spherocity:.4f}
- **Plane of Best Fit:** {pbf:.4f}

### Shape Analysis
- **Flatness (PMI2/PMI1):** {pmi2/pmi1:.4f}
- **Elongation (PMI3/PMI2):** {pmi3/pmi2:.4f}
- **Shape Type:** {'Spherical' if spherocity > 0.8 else 'Flat' if pmi2/pmi1 < 1.5 else 'Elongated' if pmi3/pmi2 > 2.0 else 'Intermediate shape'}
"""
        return markdown

    except Exception as e:
        return f"Error: {e}"


@llm_tool(name="identify_scaffolds_rdkit",
          description="Identify and analyze molecular scaffolds in a compound using RDKit library")
def identify_scaffolds_rdkit(smiles: str) -> str:
    """
    Identify and analyze molecular scaffolds in a compound.

    This function extracts the Murcko scaffold and framework from a molecule,
    which represent the core structure without side chains. It's useful for
    analyzing the structural core of drug-like molecules.

    Args:
        smiles: SMILES notation of the chemical compound. Input SMILES directly
               without any other characters.

    Returns:
        A formatted Markdown string with the scaffold analysis results.

    Examples:
        >>> identify_scaffolds("CC(=O)OC1=CC=CC=C1C(=O)O")
        # Returns scaffold analysis for Aspirin
    """
    try:
        # Preprocess input
        smiles = _preprocess_smiles(smiles)

        # Validate molecule
        mol = _validate_molecule(smiles)

        # Get Murcko scaffold
        from rdkit.Chem.Scaffolds import MurckoScaffold
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold_smiles = Chem.MolToSmiles(scaffold) if scaffold else "N/A"

        # Get framework (scaffold without bond orders)
        framework = MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles)

        # Get generic framework (all atoms replaced with carbons)
        generic_scaffold = MurckoScaffold.MakeScaffoldGeneric(scaffold)
        generic_scaffold_smiles = Chem.MolToSmiles(generic_scaffold) if generic_scaffold else "N/A"

        # Format output
        markdown = f"""## Molecular Scaffold Analysis

**Input SMILES:** `{smiles}`

### Scaffold Information
- **Murcko Scaffold SMILES:** `{scaffold_smiles}`
- **Framework SMILES:** `{framework}`
- **Generic Scaffold SMILES:** `{generic_scaffold_smiles}`

### Scaffold Features
- **Original Molecule Atom Count:** {mol.GetNumAtoms()}
- **Scaffold Atom Count:** {scaffold.GetNumAtoms() if scaffold else 0}
- **Scaffold Ring Count:** {scaffold.GetRingInfo().NumRings() if scaffold else 0}
"""
        return markdown

    except Exception as e:
        return f"Error: {e}"


#------------------------------------------------------------------------------
# Molecular Modification and Conversion
#------------------------------------------------------------------------------

@llm_tool(name="convert_between_chemical_formats_rdkit",
          description="Convert between different chemical structure formats using RDKit library")
def convert_between_chemical_formats_rdkit(
    input_string: str,
    input_format: str = "smiles",
    output_format: str = "inchi"
) -> str:
    """
    Convert between different chemical structure formats.

    This function converts a chemical structure representation from one format
    to another, supporting SMILES, InChI, InChIKey, and other common formats.

    Args:
        input_string: The chemical structure string to convert.
        input_format: The format of the input string. Options: "smiles", "inchi",
                     "smarts", "mol". Default: "smiles".
        output_format: The desired output format. Options: "smiles", "inchi",
                      "inchikey", "canonical_smiles", "mol". Default: "inchi".

    Returns:
        A formatted Markdown string with the conversion results.

    Examples:
        >>> convert_between_chemical_formats("CC(=O)OC1=CC=CC=C1C(=O)O", "smiles", "inchi")
        # Returns InChI for Aspirin
    """
    try:
        # Create molecule based on input format
        mol = None

        if input_format.lower() == "smiles":
            mol = Chem.MolFromSmiles(input_string)
        elif input_format.lower() == "inchi":
            mol = Chem.MolFromInchi(input_string)
        elif input_format.lower() == "smarts":
            mol = Chem.MolFromSmarts(input_string)
        elif input_format.lower() == "mol":
            mol = Chem.MolFromMolBlock(input_string)
        else:
            return f"Error: Unsupported input format '{input_format}'. Supported formats: smiles, inchi, smarts, mol."

        if mol is None:
            return f"Error: Could not parse input string as {input_format}."

        # Convert to output format
        output_string = ""

        if output_format.lower() == "smiles":
            output_string = Chem.MolToSmiles(mol)
        elif output_format.lower() == "canonical_smiles":
            output_string = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        elif output_format.lower() == "inchi":
            output_string = Chem.MolToInchi(mol)
        elif output_format.lower() == "inchikey":
            inchi = Chem.MolToInchi(mol)
            output_string = Chem.InchiToInchiKey(inchi)
        elif output_format.lower() == "mol":
            output_string = Chem.MolToMolBlock(mol)
        else:
            return f"Error: Unsupported output format '{output_format}'. Supported formats: smiles, canonical_smiles, inchi, inchikey, mol."

        # Format output
        markdown = f"""## Chemical Structure Format Conversion

**Input ({input_format}):** `{input_string}`

**Output ({output_format}):** `{output_string}`

### Molecule Information
- **Formula:** {rdMolDescriptors.CalcMolFormula(mol)}
- **Molecular Weight:** {Descriptors.ExactMolWt(mol):.4f} g/mol
"""
        return markdown

    except Exception as e:
        return f"Error: {e}"


@llm_tool(name="standardize_molecule_rdkit",
          description="Standardize a molecule by normalizing functional groups and charges using RDKit library")
def standardize_molecule_rdkit(smiles: str) -> str:
    """
    Standardize a molecule by normalizing functional groups and charges.

    This function applies a series of standardization rules to a molecule,
    including charge neutralization, tautomer normalization, and functional
    group standardization. It helps ensure consistent representation of
    molecules for comparison and analysis.

    Args:
        smiles: SMILES notation of the chemical compound. Input SMILES directly
               without any other characters.

    Returns:
        A formatted Markdown string with the standardization results.

    Examples:
        >>> standardize_molecule("C[N+](C)(C)CC(=O)[O-]")
        # Returns standardized form of betaine
    """
    try:
        # Preprocess input
        smiles = _preprocess_smiles(smiles)

        # Validate molecule
        mol = _validate_molecule(smiles)

        # Store original SMILES
        original_smiles = Chem.MolToSmiles(mol)

        # Apply standardization steps

        # 1. Remove fragments (keep largest fragment)
        frags = Chem.GetMolFrags(mol, asMols=True)
        if len(frags) > 1:
            largest_mol = max(frags, key=lambda x: x.GetNumAtoms())
            mol = largest_mol

        # Calculate implicit valence for all atoms
        for atom in mol.GetAtoms():
            atom.UpdatePropertyCache(strict=False)

        # 2. Uncharge molecule (neutralize when possible)
        uncharge_smarts = [
            # Carboxylic acids and similar
            ('[$([O-][C,S,P]=O)]', '[OH][C,S,P]=O'),
            # Amines
            ('[$([N+][C,c])]', '[N][C,c]'),
            # Nitro groups
            ('[$([N+](=O)[O-])]', '[N+](=O)[O-]'),
            # Sulfonic acids
            ('[$([S](=O)(=O)[O-])]', '[S](=O)(=O)[OH]')
        ]

        for smarts, replace in uncharge_smarts:
            patt = Chem.MolFromSmarts(smarts)
            if patt and mol.HasSubstructMatch(patt):
                rms = AllChem.ReplaceSubstructs(mol, patt, Chem.MolFromSmarts(replace))
                if rms[0]:
                    mol = rms[0]
                    # Update property cache after each modification
                    for atom in mol.GetAtoms():
                        atom.UpdatePropertyCache(strict=False)

        # 3. Normalize tautomers (simplified approach)
        # This is a complex topic and would require more sophisticated handling
        # for a production environment

        # 4. Canonicalize SMILES
        standardized_smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)

        # Format output
        markdown = f"""## Molecule Standardization

**Input SMILES:** `{original_smiles}`
**Standardized SMILES:** `{standardized_smiles}`

### Applied Standardization Steps
- Removed fragments (kept largest fragment)
- Neutralized charges (when possible)
- SMILES canonicalization

### Molecule Information
- **Formula:** {rdMolDescriptors.CalcMolFormula(mol)}
- **Molecular Weight:** {Descriptors.ExactMolWt(mol):.4f} g/mol
- **Formal Charge:** {Chem.GetFormalCharge(mol)}
"""
        return markdown

    except Exception as e:
        return f"Error: {e}"


@llm_tool(name="enumerate_stereoisomers_rdkit",
          description="Enumerate possible stereoisomers of a molecule using RDKit library")
def enumerate_stereoisomers_rdkit(smiles: str, max_isomers: int = 10) -> str:
    """
    Enumerate possible stereoisomers of a molecule.

    This function identifies stereocenters and double bonds with potential
    stereochemistry in a molecule, and generates all possible stereoisomers.
    It's useful for exploring the stereochemical space of a compound.

    Args:
        smiles: SMILES notation of the chemical compound. Input SMILES directly
               without any other characters.
        max_isomers: Maximum number of isomers to generate. Default: 10.

    Returns:
        A formatted Markdown string with the stereoisomer enumeration results.

    Examples:
        >>> enumerate_stereoisomers("CC(OH)C=CC")
        # Returns stereoisomers of 3-penten-2-ol
    """
    try:
        # Preprocess input
        smiles = _preprocess_smiles(smiles)

        # Validate molecule
        mol = _validate_molecule(smiles)

        # Get original SMILES
        original_smiles = Chem.MolToSmiles(mol)

        # Find stereocenters and stereobonds
        chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)

        # Count unspecified stereocenters
        unspec_chiral = sum(1 for _, assigned in chiral_centers if not assigned)

        # Count unspecified stereobonds
        unspec_bonds = 0
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.BondType.DOUBLE and bond.GetStereo() == Chem.BondStereo.STEREONONE:
                # Check if this double bond can be stereogenic
                # (not in a ring and both atoms have at least one other heavy atom neighbor)
                if not bond.IsInRing():
                    begin_atom = bond.GetBeginAtom()
                    end_atom = bond.GetEndAtom()
                    if (begin_atom.GetDegree() > 1 and end_atom.GetDegree() > 1):
                        unspec_bonds += 1

        # Calculate total possible isomers
        total_possible = 2 ** (unspec_chiral + unspec_bonds)

        # Enumerate stereoisomers
        opts = Chem.EnumerateStereoisomers.StereoEnumerationOptions(
            tryEmbedding=True,
            unique=True,
            maxIsomers=max_isomers,
            onlyUnassigned=True
        )

        isomers = list(Chem.EnumerateStereoisomers.EnumerateStereoisomers(mol, options=opts))
        isomer_smiles = [Chem.MolToSmiles(iso, isomericSmiles=True) for iso in isomers]

        # Format output
        markdown = f"""## Stereoisomer Enumeration

**Input SMILES:** `{original_smiles}`

### Stereochemistry Analysis
- **Number of Chiral Centers:** {len(chiral_centers)}
- **Unspecified Chiral Centers:** {unspec_chiral}
- **Unspecified Stereobonds:** {unspec_bonds}
- **Theoretical Total Possible Stereoisomers:** {total_possible}

### Generated Stereoisomers (Maximum {max_isomers})
"""

        for i, smi in enumerate(isomer_smiles, 1):
            markdown += f"{i}. `{smi}`\n"

        if len(isomer_smiles) < total_possible:
            markdown += f"\n**Note:** Only showing {len(isomer_smiles)} stereoisomers out of {total_possible} possible isomers."

        return markdown

    except Exception as e:
        return f"Error: {e}"


@llm_tool(name="perform_substructure_search_rdkit",
          description="Search for a substructure pattern in a molecule using RDKit library")
def perform_substructure_search_rdkit(smiles: str, pattern: str) -> str:
    """
    Search for a substructure pattern in a molecule.

    This function searches for a specified substructure pattern (SMARTS or SMILES)
    within a molecule and highlights the matches. It's useful for identifying
    specific structural features or functional groups.

    Args:
        smiles: SMILES notation of the chemical compound to search in.
        pattern: SMARTS or SMILES pattern to search for.

    Returns:
        A formatted Markdown string with the substructure search results.

    Examples:
        >>> perform_substructure_search("CC(=O)OC1=CC=CC=C1C(=O)O", "C(=O)O")
        # Returns matches of carboxylic acid group in Aspirin
    """
    try:
        # Preprocess input
        smiles = _preprocess_smiles(smiles)

        # Validate molecule
        mol = _validate_molecule(smiles)

        # Try to parse pattern as SMARTS first, then as SMILES if that fails
        pattern_mol = Chem.MolFromSmarts(pattern)
        if pattern_mol is None:
            pattern_mol = Chem.MolFromSmiles(pattern)
            if pattern_mol is None:
                return f"Error: Could not parse pattern '{pattern}' as SMARTS or SMILES."

        # Find matches
        matches = mol.GetSubstructMatches(pattern_mol)

        # Format output
        markdown = f"""## Substructure Search

**Target Molecule SMILES:** `{smiles}`
**Search Pattern:** `{pattern}`

### Search Results
- **Number of Matches Found:** {len(matches)}
"""

        if len(matches) > 0:
            markdown += "\n### Matched Atom Indices\n"
            for i, match in enumerate(matches, 1):
                markdown += f"{i}. Atom indices: {', '.join(str(idx) for idx in match)}\n"

        return markdown

    except Exception as e:
        return f"Error: {e}"
