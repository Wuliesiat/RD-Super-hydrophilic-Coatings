"""
PubChem Tools Module

This module provides tools for searching and retrieving chemical compound information
from the PubChem database using the PubChemPy library.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Union, Optional, Any

import pubchempy as pcp
from ...core.llm_tools import llm_tool

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def compound_to_dict(compound: pcp.Compound) -> Dict[str, Any]:
    """
    Convert a PubChem compound to a structured dictionary with relevant information.

    Args:
        compound: PubChem compound object

    Returns:
        Dictionary containing organized compound information
    """
    if not compound:
        return {}

    # Basic information
    result = {
        "basic_info": {
            "cid": compound.cid,
            "iupac_name": compound.iupac_name,
            "molecular_formula": compound.molecular_formula,
            "molecular_weight": compound.molecular_weight,
            "canonical_smiles": compound.canonical_smiles,
            "isomeric_smiles": compound.isomeric_smiles,
        },
        "identifiers": {
            "inchi": compound.inchi,
            "inchikey": compound.inchikey,
        },
        "physical_properties": {
            "xlogp": compound.xlogp,
            "exact_mass": compound.exact_mass,
            "monoisotopic_mass": compound.monoisotopic_mass,
            "tpsa": compound.tpsa,
            "complexity": compound.complexity,
            "charge": compound.charge,
        },
        "molecular_features": {
            "h_bond_donor_count": compound.h_bond_donor_count,
            "h_bond_acceptor_count": compound.h_bond_acceptor_count,
            "rotatable_bond_count": compound.rotatable_bond_count,
            "heavy_atom_count": compound.heavy_atom_count,
            "atom_stereo_count": compound.atom_stereo_count,
            "defined_atom_stereo_count": compound.defined_atom_stereo_count,
            "undefined_atom_stereo_count": compound.undefined_atom_stereo_count,
            "bond_stereo_count": compound.bond_stereo_count,
            "defined_bond_stereo_count": compound.defined_bond_stereo_count,
            "undefined_bond_stereo_count": compound.undefined_bond_stereo_count,
            "covalent_unit_count": compound.covalent_unit_count,
        }
    }

    # Add synonyms if available
    if hasattr(compound, 'synonyms') and compound.synonyms:
        result["alternative_names"] = {
            "synonyms": compound.synonyms[:10] if len(compound.synonyms) > 10 else compound.synonyms
        }

    return result


async def _search_by_name(name: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search compounds by name asynchronously.

    Args:
        name: Chemical compound name
        max_results: Maximum number of results to return

    Returns:
        List of compound dictionaries
    """
    try:
        compounds = await asyncio.to_thread(
            pcp.get_compounds, name, 'name',  max_records=max_results
        )
        #print(compounds[0].to_dict())
        return [compound.to_dict() for compound in compounds]
    except Exception as e:
        logging.error(f"Error searching by name '{name}': {str(e)}")
        return [{"error": f"Error: {str(e)}"}]


async def _search_by_smiles(smiles: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search compounds by SMILES notation asynchronously.

    Args:
        smiles: SMILES notation of chemical compound
        max_results: Maximum number of results to return

    Returns:
        List of compound dictionaries
    """
    try:
        compounds = await asyncio.to_thread(
            pcp.get_compounds, smiles, 'smiles', max_records=max_results
        )
        return [compound.to_dict() for compound in compounds]
    except Exception as e:
        logging.error(f"Error searching by SMILES '{smiles}': {str(e)}")
        return [{"error": f"Error: {str(e)}"}]


async def _search_by_formula(
    formula: str,
    max_results: int = 5,
    listkey_count: int = 5,
    listkey_start: int = 0
) -> List[Dict[str, Any]]:
    """
    Search compounds by molecular formula asynchronously.

    Uses pagination with listkey parameters to avoid timeout errors when searching
    formulas that might return many results.

    Args:
        formula: Molecular formula
        max_results: Maximum number of results to return
        listkey_count: Number of results per page (default: 5)
        listkey_start: Starting position for pagination (default: 0)

    Returns:
        List of compound dictionaries
    """
    try:
        # Use listkey parameters to avoid timeout errors
        compounds = await asyncio.to_thread(
            pcp.get_compounds,
            formula,
            'formula',
            max_records=max_results,
            listkey_count=listkey_count,
            listkey_start=listkey_start
        )

        return [compound.to_dict() for compound in compounds]
    except Exception as e:
        logging.error(f"Error searching by formula '{formula}': {str(e)}")
        return [{"error": f"Error: {str(e)}"}]


def _format_results_as_markdown(results: List[Dict[str, Any]], query_type: str, query_value: str) -> str:
    """
    Format search results as a structured Markdown string.

    Args:
        results: List of compound dictionaries from compound.to_dict()
        query_type: Type of search query (name, SMILES, formula)
        query_value: Value of the search query

    Returns:
        Formatted Markdown string
    """
    if not results:
        return f"## PubChem Search Results\n\nNo compounds found for {query_type}: `{query_value}`"

    if "error" in results[0]:
        return f"## PubChem Search Error\n\n{results[0]['error']}"

    markdown = f"## PubChem Search Results\n\nSearch by {query_type}: `{query_value}`\n\nFound {len(results)} compound(s)\n\n"

    for i, compound in enumerate(results):
        # Extract information from the compound.to_dict() structure
        cid = compound.get("cid", "N/A")
        iupac_name = compound.get("iupac_name", "Unknown")
        molecular_formula = compound.get("molecular_formula", "N/A")
        molecular_weight = compound.get("molecular_weight", "N/A")
        canonical_smiles = compound.get("canonical_smiles", "N/A")
        isomeric_smiles = compound.get("isomeric_smiles", "N/A")
        inchi = compound.get("inchi", "N/A")
        inchikey = compound.get("inchikey", "N/A")
        xlogp = compound.get("xlogp", "N/A")
        exact_mass = compound.get("exact_mass", "N/A")
        tpsa = compound.get("tpsa", "N/A")
        h_bond_donor_count = compound.get("h_bond_donor_count", "N/A")
        h_bond_acceptor_count = compound.get("h_bond_acceptor_count", "N/A")
        rotatable_bond_count = compound.get("rotatable_bond_count", "N/A")
        heavy_atom_count = compound.get("heavy_atom_count", "N/A")

        # Get atoms and bonds information if available
        atoms = compound.get("atoms", [])
        bonds = compound.get("bonds", [])

        # Format the markdown output
        markdown += f"### Compound {i+1}: {iupac_name}\n\n"

        # Basic information section
        markdown += "#### Basic Information\n\n"
        markdown += f"- **CID**: {cid}\n"
        markdown += f"- **Formula**: {molecular_formula}\n"
        markdown += f"- **Molecular Weight**: {molecular_weight} g/mol\n"
        markdown += f"- **Canonical SMILES**: `{canonical_smiles}`\n"
        markdown += f"- **Isomeric SMILES**: `{isomeric_smiles}`\n"

        # Identifiers section
        markdown += "\n#### Identifiers\n\n"
        markdown += f"- **InChI**: `{inchi}`\n"
        markdown += f"- **InChIKey**: `{inchikey}`\n"

        # Physical properties section
        markdown += "\n#### Physical Properties\n\n"
        markdown += f"- **XLogP**: {xlogp}\n"
        markdown += f"- **Exact Mass**: {exact_mass}\n"
        markdown += f"- **TPSA**: {tpsa} Å²\n"

        # Molecular features section
        markdown += "\n#### Molecular Features\n\n"
        markdown += f"- **H-Bond Donors**: {h_bond_donor_count}\n"
        markdown += f"- **H-Bond Acceptors**: {h_bond_acceptor_count}\n"
        markdown += f"- **Rotatable Bonds**: {rotatable_bond_count}\n"
        markdown += f"- **Heavy Atoms**: {heavy_atom_count}\n"

        # Structure information
        markdown += "\n#### Structure Information\n\n"
        markdown += f"- **Atoms Count**: {len(atoms)}\n"
        markdown += f"- **Bonds Count**: {len(bonds)}\n"

        # Add a summary of atom elements if available
        if atoms:
            elements = {}
            for atom in atoms:
                element = atom.get("element", "")
                if element:
                    elements[element] = elements.get(element, 0) + 1

            if elements:
                markdown += "- **Elements**: "
                elements_str = ", ".join([f"{element}: {count}" for element, count in elements.items()])
                markdown += f"{elements_str}\n"

        markdown += "\n---\n\n" if i < len(results) - 1 else "\n"

    return markdown


@llm_tool(name="search_advanced_pubchem",
          description="Search for chemical compounds on PubChem database using name, SMILES notation, or molecular formula via PubChemPy library")
async def search_advanced_pubchem(
    name: Optional[str] = None,
    smiles: Optional[str] = None,
    formula: Optional[str] = None,
    max_results: int = 3
) -> str:
    """
    Perform an advanced search for chemical compounds on PubChem using various identifiers.

    This function allows searching by compound name, SMILES notation, or molecular formula.
    At least one search parameter must be provided. If multiple parameters are provided,
    the search will prioritize in the order: name > smiles > formula.

    Args:
        name: Name of the chemical compound (e.g., "Aspirin", "Caffeine")
        smiles: SMILES notation of the chemical compound (e.g., "CC(=O)OC1=CC=CC=C1C(=O)O" for Aspirin)
        formula: Molecular formula (e.g., "C9H8O4" for Aspirin)
        max_results: Maximum number of results to return (default: 3)

    Returns:
        A formatted Markdown string with search results

    Examples:
        >>> search_advanced_pubchem(name="Aspirin")
        # Returns information about Aspirin

        >>> search_advanced_pubchem(smiles="CC(=O)OC1=CC=CC=C1C(=O)O")
        # Returns information about compounds matching the SMILES notation

        >>> search_advanced_pubchem(formula="C9H8O4", max_results=5)
        # Returns up to 5 compounds with the formula C9H8O4
    """
    logging.info(f"Performing advanced PubChem search with parameters: name={name}, smiles={smiles}, formula={formula}, max_results={max_results}")

    # Validate input parameters
    if name is None and smiles is None and formula is None:
        return "## PubChem Search Error\n\nError: At least one search parameter (name, smiles, or formula) must be provided"

    # Validate max_results
    if max_results < 1:
        max_results = 1
    elif max_results > 10:
        max_results = 10  # Limit to 10 results to prevent overwhelming responses

    try:
        results = []
        query_type = ""
        query_value = ""

        # Prioritize search by name, then SMILES, then formula
        if name is not None:
            results = await _search_by_name(name, max_results)
            query_type = "name"
            query_value = name
        elif smiles is not None:
            results = await _search_by_smiles(smiles, max_results)
            query_type = "SMILES"
            query_value = smiles
        elif formula is not None:
            # Use pagination parameters for formula searches to avoid timeout
            # Using the default values from _search_by_formula
            results = await _search_by_formula(formula, max_results)
            query_type = "formula"
            query_value = formula

        # Return results as markdown
        return _format_results_as_markdown(results, query_type, query_value)

    except Exception as e:
        return f"## PubChem Search Error\n\nError: {str(e)}"
