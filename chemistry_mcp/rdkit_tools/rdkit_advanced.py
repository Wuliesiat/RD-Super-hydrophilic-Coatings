"""
RDKit Advanced Tools Module

This module provides advanced tools for molecular analysis, manipulation, and visualization
using the RDKit library. It includes functions for tautomer analysis, bond order determination,
3D shape descriptors, atom-level features, and matrix representations of molecules.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Union, Optional, Any, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, Draw
from rdkit.Chem import rdmolops, rdDetermineBonds, rdTautomerQuery
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
# Bond Order Determination
#------------------------------------------------------------------------------

@llm_tool(name="determine_bond_orders_rdkit",
          description="Determine bond orders between atoms in a molecule based on atomic coordinates using RDKit")
def determine_bond_orders_rdkit(smiles: str) -> str:
    """
    Determine bond orders between atoms in a molecule based on their atomic coordinates.

    This function assigns the connectivity information to the molecule by disregarding
    pre-existing bonds. It's useful for inferring chemical bonds in a molecule when
    bond information is not already available or needs to be updated based on the
    3D structure of the molecule.

    Args:
        smiles: SMILES notation of the chemical compound. Input SMILES directly
               without any other characters.

    Returns:
        A formatted Markdown string with the bond order determination results.

    Examples:
        >>> determine_bond_orders_rdkit("CC(=O)OC1=CC=CC=C1C(=O)O")
        # Returns bond order determination results for Aspirin
    """
    try:
        # Preprocess input
        smiles = _preprocess_smiles(smiles)

        # Validate molecule
        mol = _validate_molecule(smiles)

        # Add hydrogens
        mol = Chem.AddHs(mol)

        # Generate 3D coordinates
        AllChem.EmbedMolecule(mol, randomSeed=42)

        # Determine bond orders
        Chem.rdDetermineBonds.DetermineBondOrders(mol)

        # Get molecule information
        num_atoms = mol.GetNumAtoms()
        num_bonds = mol.GetNumBonds()
        formula = rdMolDescriptors.CalcMolFormula(mol)
        mol_weight = rdMolDescriptors.CalcExactMolWt(mol)
        smiles_result = Chem.MolToSmiles(mol)
        inchi = Chem.MolToInchi(mol)
        inchikey = Chem.MolToInchiKey(mol)

        # Format output
        markdown = f"""## Bond Order Determination

**Input SMILES:** `{smiles}`

### Results
- **Number of atoms:** {num_atoms}
- **Number of bonds:** {num_bonds}
- **Molecular formula:** {formula}
- **Molecular weight:** {mol_weight:.4f} g/mol
- **SMILES:** `{smiles_result}`
- **InChI:** `{inchi}`
- **InChIKey:** `{inchikey}`

### Description
Bond orders were determined based on the 3D coordinates of the molecule. This process
can help infer the correct connectivity and bond types in complex structures.
"""
        return markdown

    except Exception as e:
        return f"Error: {e}"


@llm_tool(name="determine_bonds_rdkit",
          description="Determine bonds between atoms in a molecule based on atomic coordinates using RDKit")
def determine_bonds_rdkit(smiles: str) -> str:
    """
    Determine bonds between atoms in a molecule based on their atomic coordinates.

    This function assigns the connectivity information to the molecule by disregarding
    pre-existing bonds. Unlike determine_bond_orders, this function only determines
    the presence of bonds, not their orders.

    Args:
        smiles: SMILES notation of the chemical compound. Input SMILES directly
               without any other characters.

    Returns:
        A formatted Markdown string with the bond determination results.

    Examples:
        >>> determine_bonds_rdkit("CC(=O)OC1=CC=CC=C1C(=O)O")
        # Returns bond determination results for Aspirin
    """
    try:
        # Preprocess input
        smiles = _preprocess_smiles(smiles)

        # Validate molecule
        mol = _validate_molecule(smiles)

        # Add hydrogens
        mol = Chem.AddHs(mol)

        # Generate 3D coordinates
        AllChem.EmbedMolecule(mol, randomSeed=42)

        # Determine bonds
        Chem.rdDetermineBonds.DetermineBonds(mol)

        # Get molecule information
        num_atoms = mol.GetNumAtoms()
        num_bonds = mol.GetNumBonds()
        formula = rdMolDescriptors.CalcMolFormula(mol)
        mol_weight = rdMolDescriptors.CalcExactMolWt(mol)
        smiles_result = Chem.MolToSmiles(mol)
        inchi = Chem.MolToInchi(mol)
        inchikey = Chem.MolToInchiKey(mol)

        # Format output
        markdown = f"""## Bond Determination

**Input SMILES:** `{smiles}`

### Results
- **Number of atoms:** {num_atoms}
- **Number of bonds:** {num_bonds}
- **Molecular formula:** {formula}
- **Molecular weight:** {mol_weight:.4f} g/mol
- **SMILES:** `{smiles_result}`
- **InChI:** `{inchi}`
- **InChIKey:** `{inchikey}`

### Description
Bonds were determined based on the 3D coordinates of the molecule. This process
identifies the connectivity between atoms without specifying bond orders.
"""
        return markdown

    except Exception as e:
        return f"Error: {e}"


#------------------------------------------------------------------------------
# Tautomer Analysis
#------------------------------------------------------------------------------

@llm_tool(name="analyze_tautomers_rdkit",
          description="Analyze tautomers of a molecule using RDKit")
def analyze_tautomers_rdkit(smiles: str) -> str:
    """
    Analyze tautomers of a molecule.

    This function identifies and analyzes tautomers of a molecule. Tautomers are
    structural isomers that readily interconvert by the migration of a hydrogen
    atom and the redistribution of bonding electrons.

    Args:
        smiles: SMILES notation of the chemical compound. Input SMILES directly
               without any other characters.

    Returns:
        A formatted Markdown string with the tautomer analysis results.

    Examples:
        >>> analyze_tautomers_rdkit("CC(=O)CC(O)=O")
        # Returns tautomer analysis for acetoacetic acid
    """
    try:
        # Preprocess input
        smiles = _preprocess_smiles(smiles)

        # Validate molecule
        mol = _validate_molecule(smiles)

        # Create tautomer query
        tautomer_query = rdTautomerQuery.TautomerQuery(mol)

        # Get template molecule
        template_mol = tautomer_query.GetTemplateMolecule()
        template_smiles = Chem.MolToSmiles(template_mol)

        # Get modified atoms
        modified_atoms = tautomer_query.GetModifiedAtoms()

        # Get modified bonds
        modified_bonds = tautomer_query.GetModifiedBonds()

        # Generate pattern fingerprint
        pattern_fp = rdTautomerQuery.PatternFingerprintTautomerTarget(mol)
        num_bits = pattern_fp.GetNumBits()
        num_on_bits = pattern_fp.GetNumOnBits()

        # Format output
        markdown = f"""## Tautomer Analysis

**Input SMILES:** `{smiles}`

### Template Molecule
- **Template SMILES:** `{template_smiles}`

### Modified Structural Elements
- **Modified Atoms:** {', '.join(str(atom) for atom in modified_atoms) if modified_atoms else "None"}
- **Modified Bonds:** {', '.join(str(bond) for bond in modified_bonds) if modified_bonds else "None"}

### Pattern Fingerprint
- **Number of Bits:** {num_bits}
- **Number of On Bits:** {num_on_bits}
- **Density:** {num_on_bits/num_bits:.4f}

### Description
Tautomers are structural isomers that readily interconvert by the migration of a hydrogen atom
and the redistribution of bonding electrons. This analysis identifies the atoms and bonds
that may change during tautomerization.
"""
        return markdown

    except Exception as e:
        return f"Error: {e}"


@llm_tool(name="check_substructure_with_tautomers_rdkit",
          description="Check if a molecule is a substructure of another molecule considering tautomers using RDKit")
def check_substructure_with_tautomers_rdkit(target_smiles: str, template_smiles: str) -> str:
    """
    Check if a molecule is a substructure of another molecule considering tautomers.

    This function checks if a molecule (target) is a substructure of another molecule
    (template) considering possible tautomeric forms. It's useful for substructure
    searching that needs to be robust to tautomeric variations.

    Args:
        target_smiles: SMILES notation of the target molecule.
        template_smiles: SMILES notation of the template molecule.

    Returns:
        A formatted Markdown string with the substructure check results.

    Examples:
        >>> check_substructure_with_tautomers_rdkit("CC(=O)CC(O)=O", "CC(O)=CC=O")
        # Checks if acetoacetic acid contains the enol form as a substructure
    """
    try:
        # Preprocess input
        target_smiles = _preprocess_smiles(target_smiles)
        template_smiles = _preprocess_smiles(template_smiles)

        # Validate molecules
        target_mol = _validate_molecule(target_smiles)
        template_mol = _validate_molecule(template_smiles)

        # Create tautomer query
        tautomer_query = rdTautomerQuery.TautomerQuery(template_mol)

        # Check if target is a substructure of template
        is_substructure = tautomer_query.IsSubstructOf(target_mol)

        # Get substructure matches
        matches = tautomer_query.GetSubstructMatches(target_mol)

        # Format output
        markdown = f"""## Substructure Check with Tautomers

**Target SMILES:** `{target_smiles}`
**Template SMILES:** `{template_smiles}`

### Results
- **Is Substructure:** {"Yes" if is_substructure else "No"}
- **Number of Matches:** {len(matches)}

"""
        if matches:
            markdown += "### Matched Atom Indices\n"
            for i, match in enumerate(matches, 1):
                markdown += f"{i}. {match}\n"

        markdown += """
### Description
This analysis checks if the template molecule is a substructure of the target molecule,
considering possible tautomeric forms. This is useful for substructure searching that
needs to be robust to tautomeric variations.
"""
        return markdown

    except Exception as e:
        return f"Error: {e}"


#------------------------------------------------------------------------------
# 3D Shape Descriptors
#------------------------------------------------------------------------------

@llm_tool(name="calculate_usr_descriptors_rdkit",
          description="Calculate Ultrafast Shape Recognition (USR) descriptors for a molecule using RDKit")
def calculate_usr_descriptors_rdkit(smiles: str) -> str:
    """
    Calculate Ultrafast Shape Recognition (USR) descriptors for a molecule.

    This function computes USR and USRCAT descriptors, which are numerical
    representations of the 3D shape of a molecule. These descriptors are
    particularly useful for rapid shape-based virtual screening.

    Args:
        smiles: SMILES notation of the chemical compound. Input SMILES directly
               without any other characters.

    Returns:
        A formatted Markdown string with the USR descriptor calculation results.

    Examples:
        >>> calculate_usr_descriptors_rdkit("CC(=O)OC1=CC=CC=C1C(=O)O")
        # Returns USR descriptors for Aspirin
    """
    try:
        # Preprocess input
        smiles = _preprocess_smiles(smiles)

        # Validate molecule
        mol = _validate_molecule(smiles)

        # Add hydrogens
        mol = Chem.AddHs(mol)

        # Generate 3D coordinates
        AllChem.EmbedMolecule(mol, randomSeed=42)

        # Optimize the structure
        AllChem.MMFFOptimizeMolecule(mol)

        # Calculate USR descriptors
        usr = rdMolDescriptors.GetUSR(mol)
        usrcat = rdMolDescriptors.GetUSRCAT(mol)

        # Format output
        markdown = f"""## Ultrafast Shape Recognition (USR) Descriptors

**Input SMILES:** `{smiles}`

### USR Descriptors
The USR descriptor is a 12-dimensional vector that encodes the 3D shape of a molecule:
```
{', '.join(f'{val:.4f}' for val in usr)}
```

### USRCAT Descriptors
The USRCAT descriptor extends USR by including chemical feature information:
```
{', '.join(f'{val:.4f}' for val in usrcat)}
```

### Description
USR descriptors capture the 3D shape of molecules in a rotation-invariant manner,
allowing for ultrafast shape-based virtual screening. USRCAT extends this by
incorporating chemical feature information.
"""
        return markdown

    except Exception as e:
        return f"Error: {e}"


#------------------------------------------------------------------------------
# Atom-Level Features
#------------------------------------------------------------------------------

@llm_tool(name="analyze_atom_features_rdkit",
          description="Analyze atom-level features of a molecule using RDKit")
def analyze_atom_features_rdkit(smiles: str) -> str:
    """
    Analyze atom-level features of a molecule.

    This function computes various features for each atom in a molecule, including
    atomic number, hybridization, formal charge, and connectivity information.

    Args:
        smiles: SMILES notation of the chemical compound. Input SMILES directly
               without any other characters.

    Returns:
        A formatted Markdown string with the atom feature analysis results.

    Examples:
        >>> analyze_atom_features_rdkit("CC(=O)OC1=CC=CC=C1C(=O)O")
        # Returns atom feature analysis for Aspirin
    """
    try:
        # Preprocess input
        smiles = _preprocess_smiles(smiles)

        # Validate molecule
        mol = _validate_molecule(smiles)

        # Analyze atoms
        atoms = mol.GetAtoms()
        atom_features = []

        for atom in atoms:
            idx = atom.GetIdx()
            symbol = atom.GetSymbol()
            atomic_num = atom.GetAtomicNum()
            formal_charge = atom.GetFormalCharge()
            hybridization = str(atom.GetHybridization())
            is_aromatic = atom.GetIsAromatic()
            is_in_ring = atom.IsInRing()
            degree = atom.GetDegree()
            implicit_valence = atom.GetImplicitValence()
            explicit_valence = atom.GetExplicitValence()

            atom_features.append({
                "idx": idx,
                "symbol": symbol,
                "atomic_num": atomic_num,
                "formal_charge": formal_charge,
                "hybridization": hybridization,
                "is_aromatic": is_aromatic,
                "is_in_ring": is_in_ring,
                "degree": degree,
                "implicit_valence": implicit_valence,
                "explicit_valence": explicit_valence
            })

        # Get connectivity invariants
        conn_invariants = rdMolDescriptors.GetConnectivityInvariants(mol)

        # Get feature invariants
        feature_invariants = rdMolDescriptors.GetFeatureInvariants(mol)

        # Format output
        markdown = f"""## Atom-Level Feature Analysis

**Input SMILES:** `{smiles}`

### Atom Features
| Idx | Symbol | Atomic # | Charge | Hybridization | Aromatic | In Ring | Degree | Implicit Valence | Explicit Valence |
|-----|--------|----------|--------|---------------|----------|---------|--------|------------------|------------------|
"""

        for feat in atom_features:
            markdown += f"| {feat['idx']} | {feat['symbol']} | {feat['atomic_num']} | {feat['formal_charge']} | {feat['hybridization']} | {'Yes' if feat['is_aromatic'] else 'No'} | {'Yes' if feat['is_in_ring'] else 'No'} | {feat['degree']} | {feat['implicit_valence']} | {feat['explicit_valence']} |\n"

        markdown += """
### Connectivity Invariants
These values represent the topological environment of each atom:
```
"""
        markdown += ', '.join(str(inv) for inv in conn_invariants)
        markdown += """
```

### Feature Invariants
These values represent the chemical feature environment of each atom:
```
"""
        markdown += ', '.join(str(inv) for inv in feature_invariants)
        markdown += """
```

### Description
Atom-level features provide detailed information about each atom in the molecule,
including its chemical environment, connectivity, and properties. These features
are useful for structure-activity relationship studies and machine learning applications.
"""
        return markdown

    except Exception as e:
        return f"Error: {e}"


#------------------------------------------------------------------------------
# Ring System Analysis
#------------------------------------------------------------------------------

@llm_tool(name="analyze_ring_systems_rdkit",
          description="Analyze ring systems in a molecule using RDKit")
def analyze_ring_systems_rdkit(smiles: str) -> str:
    """
    Analyze ring systems in a molecule.

    This function identifies and analyzes ring systems in a molecule, including
    individual rings and fused ring systems. It provides detailed information
    about ring sizes, aromaticity, and the atoms involved in each ring.

    Args:
        smiles: SMILES notation of the chemical compound. Input SMILES directly
               without any other characters.

    Returns:
        A formatted Markdown string with the ring system analysis results.

    Examples:
        >>> analyze_ring_systems_rdkit("c1ccc2ccccc2c1")
        # Returns ring system analysis for naphthalene
    """
    try:
        # Preprocess input
        smiles = _preprocess_smiles(smiles)

        # Validate molecule
        mol = _validate_molecule(smiles)

        # Get ring info
        ring_info = mol.GetRingInfo()
        atom_rings = ring_info.AtomRings()
        bond_rings = ring_info.BondRings()

        # Analyze individual rings
        rings_data = []
        for i, ring in enumerate(atom_rings):
            size = len(ring)
            is_aromatic = all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring)
            atoms = [mol.GetAtomWithIdx(idx).GetSymbol() for idx in ring]

            rings_data.append({
                "id": i,
                "size": size,
                "is_aromatic": is_aromatic,
                "atom_indices": ring,
                "atoms": atoms
            })

        # Identify ring systems (connected rings)
        systems = []
        for ring in atom_rings:
            ring_atoms = set(ring)
            new_systems = []

            for system in systems:
                if len(ring_atoms.intersection(system)) > 0:
                    # Rings share atoms, merge them
                    ring_atoms = ring_atoms.union(system)
                else:
                    new_systems.append(system)

            new_systems.append(ring_atoms)
            systems = new_systems

        # Format output
        markdown = f"""## Ring System Analysis

**Input SMILES:** `{smiles}`

### Ring Summary
- **Total Rings:** {len(atom_rings)}
- **Ring Systems:** {len(systems)}

### Individual Rings
"""

        for ring in rings_data:
            markdown += f"#### Ring {ring['id']+1} ({ring['size']}-membered)\n"
            markdown += f"- **Aromatic:** {'Yes' if ring['is_aromatic'] else 'No'}\n"
            markdown += f"- **Atom Indices:** {', '.join(str(idx) for idx in ring['atom_indices'])}\n"
            markdown += f"- **Atoms:** {', '.join(ring['atoms'])}\n\n"

        if systems:
            markdown += "### Ring Systems\n"
            for i, system in enumerate(systems, 1):
                rings_in_system = [j for j, ring in enumerate(atom_rings) if any(idx in system for idx in ring)]
                markdown += f"#### System {i}\n"
                markdown += f"- **Atoms:** {', '.join(str(idx) for idx in sorted(system))}\n"
                markdown += f"- **Rings Involved:** {', '.join(f'Ring {j+1}' for j in rings_in_system)}\n\n"

        markdown += """### Description
Ring systems are important structural features in molecules that influence
their physical, chemical, and biological properties. This analysis helps
understand the complexity and connectivity of rings within the molecule.
"""
        return markdown

    except Exception as e:
        return f"Error: {e}"


#------------------------------------------------------------------------------
# Matrix Representations
#------------------------------------------------------------------------------

@llm_tool(name="generate_molecular_matrices_rdkit",
          description="Generate matrix representations of a molecule using RDKit")
def generate_molecular_matrices_rdkit(smiles: str) -> str:
    """
    Generate matrix representations of a molecule.

    This function computes various matrix representations of a molecule, including
    adjacency matrix and distance matrix. These matrices are useful for machine
    learning applications and structural analysis.

    Args:
        smiles: SMILES notation of the chemical compound. Input SMILES directly
               without any other characters.

    Returns:
        A formatted Markdown string with the matrix generation results.

    Examples:
        >>> generate_molecular_matrices_rdkit("CC(=O)O")
        # Returns matrix representations for acetic acid
    """
    try:
        # Preprocess input
        smiles = _preprocess_smiles(smiles)

        # Validate molecule
        mol = _validate_molecule(smiles)

        # Get adjacency matrix
        adj_matrix = Chem.GetAdjacencyMatrix(mol)

        # Get distance matrix
        dist_matrix = Chem.GetDistanceMatrix(mol)

        # Format output
        markdown = f"""## Molecular Matrix Representations

**Input SMILES:** `{smiles}`

### Adjacency Matrix
The adjacency matrix represents the connectivity between atoms:
```
{np.array2string(adj_matrix, separator=', ')}
```

### Distance Matrix
The distance matrix represents the shortest path (in bonds) between atoms:
```
{np.array2string(dist_matrix, separator=', ')}
```

### Description
Matrix representations of molecules are useful for various applications:
- **Adjacency Matrix:** Represents direct connections between atoms
- **Distance Matrix:** Represents the shortest path between atoms
These matrices can be used for structural analysis, similarity calculations,
and as input features for machine learning models.
"""
        return markdown

    except Exception as e:
        return f"Error: {e}"


#------------------------------------------------------------------------------
# Oxidation Number Analysis
#------------------------------------------------------------------------------

@llm_tool(name="analyze_oxidation_numbers_rdkit",
          description="Analyze oxidation numbers of atoms in a molecule using RDKit")
def analyze_oxidation_numbers_rdkit(smiles: str) -> str:
    """
    Analyze oxidation numbers of atoms in a molecule.

    This function calculates and analyzes the oxidation numbers (states) of atoms
    in a molecule. Oxidation numbers are useful for understanding redox chemistry
    and electron distribution in molecules.

    Args:
        smiles: SMILES notation of the chemical compound. Input SMILES directly
               without any other characters.

    Returns:
        A formatted Markdown string with the oxidation number analysis results.

    Examples:
        >>> analyze_oxidation_numbers_rdkit("CC(=O)O")
        # Returns oxidation number analysis for acetic acid
    """
    try:
        # Preprocess input
        smiles = _preprocess_smiles(smiles)

        # Validate molecule
        mol = _validate_molecule(smiles)

        # Calculate oxidation numbers - this adds the property to each atom
        rdMolDescriptors.CalcOxidationNumbers(mol)

        # Get atom symbols and oxidation numbers from atom properties
        atoms = mol.GetAtoms()
        atom_data = []
        for atom in atoms:
            symbol = atom.GetSymbol()
            # Get oxidation number from atom property
            ox_num = atom.GetIntProp("OxidationNumber") if atom.HasProp("OxidationNumber") else 0
            atom_data.append((symbol, ox_num))

        # Format output
        markdown = f"""## Oxidation Number Analysis

**Input SMILES:** `{smiles}`
**Molecular Formula:** {rdMolDescriptors.CalcMolFormula(mol)}

### Atom Oxidation Numbers
| Atom Index | Element | Oxidation Number |
|------------|---------|------------------|
"""

        for i, (symbol, ox_num) in enumerate(atom_data):
            markdown += f"| {i} | {symbol} | {ox_num} |\n"

        # Calculate average oxidation number per element
        element_ox_nums = {}
        for symbol, ox_num in atom_data:
            if symbol not in element_ox_nums:
                element_ox_nums[symbol] = []
            element_ox_nums[symbol].append(ox_num)

        markdown += """
### Average Oxidation Number by Element
| Element | Average Oxidation Number |
|---------|--------------------------|
"""

        for element, ox_nums in element_ox_nums.items():
            avg_ox_num = sum(ox_nums) / len(ox_nums)
            markdown += f"| {element} | {avg_ox_num:.2f} |\n"

        markdown += """
### Description
Oxidation numbers represent the hypothetical charge an atom would have if all bonds
were completely ionic. They are useful for tracking electrons in redox reactions
and understanding the electron distribution in molecules.
"""
        return markdown

    except Exception as e:
        return f"Error: {e}"
