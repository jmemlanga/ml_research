#!/usr/bin/env python3
"""Generate molecular descriptors from SMILES and Gaussian output files."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from cclib.io import ccread
from rdkit import Chem
from rdkit.Chem import (
    AllChem,
    Crippen,
    Descriptors,
    Lipinski,
    rdFreeSASA,
    rdMolDescriptors,
    rdPartialCharges,
)
from rdkit.Chem.EState import EState
from rdkit.Geometry import Point3D


# =============================================================================
# PATHS TO EDIT
# =============================================================================

INPUT_CSV = Path("full_data.csv")
GS_DIR = Path("/home/mjp218/data/nitro/parent/AM1/gs/all_done")
TS_DIR = Path("/home/mjp218/data/nitro/parent/AM1/ts/all_done")
OUTPUT_CSV = Path("generated_features.csv")


# =============================================================================
# DESCRIPTOR DEFINITIONS
# =============================================================================

STATS4 = ("mean", "std", "min", "max")
CHARGE_STATS = ("mean", "std", "min", "max", "abs_mean", "abs_max")
ELEMENTS = {"C": 6, "H": 1, "N": 7, "O": 8}

RDKIT_2D_GLOBAL = {
    "rdkit__hbond_acceptors": Lipinski.NumHAcceptors,
    "rdkit__hbond_donors": Lipinski.NumHDonors,
    "rdkit__amide_bonds": Lipinski.NumAmideBonds,
    "rdkit__tpsa": rdMolDescriptors.CalcTPSA,
    "rdkit__mol_wt": Descriptors.MolWt,
    "rdkit__logp": Crippen.MolLogP,
    "rdkit__molar_refractivity": Crippen.MolMR,
    "rdkit__rotatable_bonds": Lipinski.NumRotatableBonds,
    "rdkit__ring_count": Lipinski.RingCount,
    "rdkit__fraction_csp3": rdMolDescriptors.CalcFractionCSP3,
    "rdkit__heavy_atom_count": Lipinski.HeavyAtomCount,
}

RDKIT_ATOM_PROPERTIES = (
    "peoe_charge",
    "crippen_logp",
    "crippen_mr",
    "estate",
)

RDKIT_3D = (
    "radius_of_gyration",
    "asphericity",
    "eccentricity",
    "inertial_shape_factor",
    "npr1",
    "npr2",
    "pmi1",
    "pmi2",
    "pmi3",
    "spherocity_index",
    "labute_asa",
    "molecular_volume",
    "freesasa_total",
)

MORFEUS = (
    "sasa_area",
    "sasa_volume",
    "surface_area",
    "surface_volume",
    "pint",
)

CCLIB = (
    "HOMO_eV",
    "LUMO_eV",
    "chemical_potential_eV",
    "hardness_eV",
    "softness_per_eV",
    "electrophilicity_eV",
    "geometry_scf_energy_eV",
    "lowest_vibrational_frequency_cm-1",
    "lowest_ir_intensity",
    "spe_energy_eV",
)


def finite_float(value: Any) -> float:
    """Convert a value to a finite float; otherwise return NaN."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def token(value: Any) -> str:
    """Convert a CSV identifier to the token used in Gaussian filenames."""
    if pd.isna(value):
        return ""

    text = str(value).strip()
    try:
        number = float(text)
        if text.endswith(".0") and number.is_integer():
            return str(int(number))
    except ValueError:
        pass

    return text


def add_error(
    errors: list[str],
    file_idx: str,
    state: str,
    file: str | Path,
    exc: BaseException,
    context: str = "",
) -> None:
    """Print and count a descriptor-generation error."""
    message = f"{context}: {exc}" if context else str(exc)
    formatted = (
        f"Descriptor error | file_idx={file_idx} | state={state} | "
        f"file={file} | {type(exc).__name__}: {message}"
    )
    errors.append(formatted)
    print(formatted, file=sys.stderr)


def finite_values(values: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    return array[np.isfinite(array)]


def summary4(values: Sequence[float] | np.ndarray) -> dict[str, float]:
    array = finite_values(values)
    if not array.size:
        return {name: float("nan") for name in STATS4}

    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=0)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def charge_summary(values: Sequence[float] | np.ndarray) -> dict[str, float]:
    array = finite_values(values)
    if not array.size:
        return {name: float("nan") for name in CHARGE_STATS}

    absolute = np.abs(array)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=0)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "abs_mean": float(np.mean(absolute)),
        "abs_max": float(np.max(absolute)),
    }


def rdkit_2d_names() -> list[str]:
    names = list(RDKIT_2D_GLOBAL)
    for prop in RDKIT_ATOM_PROPERTIES:
        names.extend(f"rdkit__{prop}_{stat}" for stat in STATS4)
    return names


def state_names(state: str) -> list[str]:
    names = [f"{state}__rdkit3d_{name}" for name in RDKIT_3D]
    names += [f"{state}__morfeus_{name}" for name in MORFEUS]
    names += [f"{state}__{name}" for name in CCLIB]

    for group in ("all", *ELEMENTS):
        base = "mulliken" if group == "all" else f"mulliken_{group}"
        names += [f"{state}__{base}_{stat}" for stat in CHARGE_STATS]

    return names


def feature_names() -> list[str]:
    names = [
        "am1_barrier",
        *rdkit_2d_names(),
        *state_names("gs"),
        *state_names("ts"),
    ]

    if len(names) != len(set(names)):
        raise RuntimeError("Feature names are not unique")

    return names


def rdkit_2d(
    smiles: str,
    file_idx: str,
    errors: list[str],
) -> dict[str, float]:
    """Calculate global RDKit 2D values and atom-level summary statistics."""
    result = {name: float("nan") for name in rdkit_2d_names()}
    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        add_error(
            errors,
            file_idx,
            "rdkit2d",
            smiles,
            ValueError("RDKit could not parse SMILES"),
        )
        return result

    for name, function in RDKIT_2D_GLOBAL.items():
        try:
            result[name] = finite_float(function(molecule))
        except Exception as exc:
            add_error(errors, file_idx, "rdkit2d", smiles, exc, name)

    molecule_h = Chem.AddHs(Chem.Mol(molecule))
    values: dict[str, Sequence[float]] = {}

    try:
        rdPartialCharges.ComputeGasteigerCharges(
            molecule_h,
            nIter=12,
            throwOnParamFailure=False,
        )
        values["peoe_charge"] = [
            finite_float(atom.GetProp("_GasteigerCharge"))
            if atom.HasProp("_GasteigerCharge")
            else float("nan")
            for atom in molecule_h.GetAtoms()
        ]
    except Exception as exc:
        add_error(errors, file_idx, "rdkit2d", smiles, exc, "Gasteiger charges")

    try:
        contributions = list(Crippen._GetAtomContribs(molecule_h))
        values["crippen_logp"] = [finite_float(item[0]) for item in contributions]
        values["crippen_mr"] = [finite_float(item[1]) for item in contributions]
    except Exception as exc:
        add_error(
            errors,
            file_idx,
            "rdkit2d",
            smiles,
            exc,
            "Crippen atom contributions",
        )

    try:
        values["estate"] = np.asarray(EState.EStateIndices(molecule_h), dtype=float)
    except Exception as exc:
        add_error(errors, file_idx, "rdkit2d", smiles, exc, "EState indices")

    for prop in RDKIT_ATOM_PROPERTIES:
        for stat, value in summary4(values.get(prop, [])).items():
            result[f"rdkit__{prop}_{stat}"] = value

    return result


def find_pairs(
    directory: Path,
    state: str,
    file_idx: str,
    requested_x: str | None,
) -> list[tuple[str, Path, Path]]:
    """Return geometry/SPE pairs sharing the same candidate token."""
    prefix = f"{state}-{file_idx}-"

    if requested_x:
        geometry = directory / f"{prefix}{requested_x}.out"
        spe = directory / f"{prefix}{requested_x}_SPE.log"
        missing = [str(path) for path in (geometry, spe) if not path.is_file()]

        if missing:
            raise FileNotFoundError(
                "Missing requested pair member(s): " + ", ".join(missing)
            )

        return [(requested_x, geometry, spe)]

    pairs = []
    for geometry in sorted(directory.glob(f"{prefix}*.out")):
        candidate_x = geometry.name[len(prefix) : -4]
        spe = directory / f"{prefix}{candidate_x}_SPE.log"
        if candidate_x and spe.is_file():
            pairs.append((candidate_x, geometry, spe))

    if not pairs:
        raise FileNotFoundError(
            f"No valid {state.upper()} pair for file_idx={file_idx} in {directory}"
        )

    return pairs


def parse_output(path: Path) -> Any:
    data = ccread(str(path))
    if data is None:
        raise RuntimeError(f"cclib could not parse {path}")
    return data


def last_value(data: Any, attribute: str) -> float:
    values = finite_values(getattr(data, attribute, []))
    return float(values[-1]) if values.size else float("nan")


def select_pair(
    directory: Path,
    state: str,
    file_idx: str,
    requested_x: str | None,
    errors: list[str],
) -> dict[str, Any]:
    """Select an exact, sole, lowest-free-energy, or lowest-SCF pair."""
    parsed = []

    for candidate_x, geometry, spe in find_pairs(
        directory,
        state,
        file_idx,
        requested_x,
    ):
        try:
            data = parse_output(geometry)
            parsed.append(
                {
                    "x": candidate_x,
                    "geometry": geometry,
                    "spe": spe,
                    "data": data,
                    "freeenergy": last_value(data, "freeenergy"),
                    "scfenergy": last_value(data, "scfenergies"),
                }
            )
        except Exception as exc:
            add_error(errors, file_idx, state, geometry, exc)

    if not parsed:
        raise RuntimeError(
            f"No parseable {state.upper()} geometry for file_idx={file_idx}"
        )

    if requested_x or len(parsed) == 1:
        return parsed[0]

    candidates = [item for item in parsed if np.isfinite(item["freeenergy"])]
    if candidates:
        return min(candidates, key=lambda item: item["freeenergy"])

    candidates = [item for item in parsed if np.isfinite(item["scfenergy"])]
    if candidates:
        return min(candidates, key=lambda item: item["scfenergy"])

    raise RuntimeError(
        "Multiple candidates found, but none has freeenergy or scfenergies"
    )


def frontier(data: Any) -> tuple[float, float]:
    """Extract HOMO and LUMO energies across available spin channels."""
    if not hasattr(data, "homos") or not hasattr(data, "moenergies"):
        return float("nan"), float("nan")

    homos = np.asarray(data.homos, dtype=int).reshape(-1)
    homo_values = []
    lumo_values = []

    for spin, homo_index in enumerate(homos):
        if spin >= len(data.moenergies):
            continue

        energies = np.asarray(data.moenergies[spin], dtype=float).reshape(-1)
        if 0 <= homo_index < len(energies):
            homo_values.append(energies[homo_index])
        if 0 <= homo_index + 1 < len(energies):
            lumo_values.append(energies[homo_index + 1])

    homo = max(homo_values) if homo_values else float("nan")
    lumo = min(lumo_values) if lumo_values else float("nan")
    return finite_float(homo), finite_float(lumo)


def cclib_descriptors(data: Any) -> dict[str, float]:
    """Extract orbital, energy, vibration, and Mulliken descriptors."""
    homo, lumo = frontier(data)
    chemical_potential = float("nan")
    hardness = float("nan")
    softness = float("nan")
    electrophilicity = float("nan")

    if np.isfinite(homo) and np.isfinite(lumo):
        chemical_potential = (homo + lumo) / 2.0
        hardness = (lumo - homo) / 2.0

        if lumo - homo > 0.0:
            softness = 1.0 / (lumo - homo)
            electrophilicity = chemical_potential**2 / (2.0 * hardness)

    vibfreqs = finite_values(getattr(data, "vibfreqs", []))
    vibirs = finite_values(getattr(data, "vibirs", []))

    result = {
        "HOMO_eV": homo,
        "LUMO_eV": lumo,
        "chemical_potential_eV": chemical_potential,
        "hardness_eV": hardness,
        "softness_per_eV": softness,
        "electrophilicity_eV": electrophilicity,
        "geometry_scf_energy_eV": last_value(data, "scfenergies"),
        "lowest_vibrational_frequency_cm-1": (
            float(np.min(vibfreqs)) if vibfreqs.size else float("nan")
        ),
        "lowest_ir_intensity": (
            float(np.min(vibirs)) if vibirs.size else float("nan")
        ),
    }

    atomnos = np.asarray(getattr(data, "atomnos", []), dtype=int).reshape(-1)
    atomcharges = getattr(data, "atomcharges", None)
    mulliken = None

    if isinstance(atomcharges, Mapping):
        for key, values in atomcharges.items():
            if str(key).lower() == "mulliken":
                mulliken = np.asarray(values, dtype=float).reshape(-1)
                break

    groups = {
        "all": np.asarray([], dtype=float),
        **{symbol: np.asarray([], dtype=float) for symbol in ELEMENTS},
    }

    if mulliken is not None and len(mulliken) == len(atomnos):
        groups["all"] = mulliken
        for symbol, atomic_number in ELEMENTS.items():
            groups[symbol] = mulliken[atomnos == atomic_number]

    for group, values in groups.items():
        base = "mulliken" if group == "all" else f"mulliken_{group}"
        for stat, value in charge_summary(values).items():
            result[f"{base}_{stat}"] = value

    return result


def gaussian_molecule(data: Any) -> tuple[Chem.Mol, np.ndarray, np.ndarray]:
    """Create an atom-only RDKit molecule with the final Gaussian conformer."""
    atomnos = np.asarray(getattr(data, "atomnos", []), dtype=int).reshape(-1)
    atomcoords = np.asarray(getattr(data, "atomcoords", []), dtype=float)

    if not atomnos.size or atomcoords.ndim != 3 or not atomcoords.shape[0]:
        raise ValueError("Final Gaussian atoms or coordinates are unavailable")

    coordinates = np.asarray(atomcoords[-1], dtype=float)
    if coordinates.shape != (len(atomnos), 3):
        raise ValueError("Final coordinate count does not match atom count")

    editable = Chem.RWMol()
    for atomic_number in atomnos:
        editable.AddAtom(Chem.Atom(int(atomic_number)))

    molecule = editable.GetMol()
    conformer = Chem.Conformer(len(atomnos))
    conformer.Set3D(True)

    for index, (x, y, z) in enumerate(coordinates):
        conformer.SetAtomPosition(index, Point3D(float(x), float(y), float(z)))

    molecule.AddConformer(conformer, assignId=True)
    return molecule, atomnos, coordinates


def rdkit_3d(
    molecule: Chem.Mol,
    file_idx: str,
    state: str,
    geometry: Path,
    errors: list[str],
) -> dict[str, float]:
    """Calculate RDKit 3D descriptors, using NaN for individual failures."""
    npr_one = getattr(rdMolDescriptors, "CalcNPR1")
    npr_two = getattr(rdMolDescriptors, "CalcNPR2")

    functions: dict[str, Callable[[], Any]] = {
        "radius_of_gyration": lambda: rdMolDescriptors.CalcRadiusOfGyration(
            molecule,
            confId=0,
        ),
        "asphericity": lambda: rdMolDescriptors.CalcAsphericity(molecule, confId=0),
        "eccentricity": lambda: rdMolDescriptors.CalcEccentricity(molecule, confId=0),
        "inertial_shape_factor": lambda: rdMolDescriptors.CalcInertialShapeFactor(
            molecule,
            confId=0,
        ),
        "npr1": lambda: npr_one(molecule, confId=0),
        "npr2": lambda: npr_two(molecule, confId=0),
        "pmi1": lambda: rdMolDescriptors.CalcPMI1(molecule, confId=0),
        "pmi2": lambda: rdMolDescriptors.CalcPMI2(molecule, confId=0),
        "pmi3": lambda: rdMolDescriptors.CalcPMI3(molecule, confId=0),
        "spherocity_index": lambda: rdMolDescriptors.CalcSpherocityIndex(
            molecule,
            confId=0,
        ),
        "labute_asa": lambda: rdMolDescriptors.CalcLabuteASA(
            molecule,
            includeHs=True,
        ),
        "molecular_volume": lambda: AllChem.ComputeMolVolume(molecule, confId=0),
    }

    result = {name: float("nan") for name in RDKIT_3D}

    for name, function in functions.items():
        try:
            result[name] = finite_float(function())
        except Exception as exc:
            add_error(errors, file_idx, state, geometry, exc, name)

    try:
        table = Chem.GetPeriodicTable()
        radii = [
            float(table.GetRvdw(atom.GetAtomicNum()))
            for atom in molecule.GetAtoms()
        ]
        result["freesasa_total"] = finite_float(
            rdFreeSASA.CalcSASA(molecule, radii, confIdx=0)
        )
    except Exception as exc:
        add_error(errors, file_idx, state, geometry, exc, "freesasa_total")

    return result


def morfeus_descriptors(
    atomnos: np.ndarray,
    coordinates: np.ndarray,
    file_idx: str,
    state: str,
    geometry: Path,
    errors: list[str],
) -> dict[str, float]:
    """Calculate global Morfeus SASA and dispersion descriptors."""
    from morfeus import Dispersion, SASA

    result = {name: float("nan") for name in MORFEUS}
    table = Chem.GetPeriodicTable()
    elements = [table.GetElementSymbol(int(number)) for number in atomnos]

    try:
        sasa = SASA(elements, coordinates)
        result["sasa_area"] = finite_float(sasa.area)
        result["sasa_volume"] = finite_float(sasa.volume)
    except Exception as exc:
        add_error(errors, file_idx, state, geometry, exc, "Morfeus SASA")

    try:
        dispersion = Dispersion(elements, coordinates)
        result["surface_area"] = finite_float(dispersion.area)
        result["surface_volume"] = finite_float(dispersion.volume)
        result["pint"] = finite_float(dispersion.p_int)
    except Exception as exc:
        add_error(errors, file_idx, state, geometry, exc, "Morfeus Dispersion")

    return result


def state_descriptors(
    row: pd.Series,
    state: str,
    directory: Path,
    errors: list[str],
) -> dict[str, float]:
    """Generate all descriptors for one selected GS or TS calculation."""
    file_idx = token(row["file_idx"])
    result = {name: float("nan") for name in state_names(state)}

    x_column = f"{state}_x"
    requested_x = token(row[x_column]) if x_column in row.index else ""

    try:
        selected = select_pair(
            directory,
            state,
            file_idx,
            requested_x or None,
            errors,
        )
    except Exception as exc:
        add_error(errors, file_idx, state, directory, exc)
        return result

    geometry = Path(selected["geometry"])
    spe = Path(selected["spe"])

    try:
        for name, value in cclib_descriptors(selected["data"]).items():
            result[f"{state}__{name}"] = value
    except Exception as exc:
        add_error(errors, file_idx, state, geometry, exc, "cclib descriptors")

    try:
        molecule, atomnos, coordinates = gaussian_molecule(selected["data"])

        for name, value in rdkit_3d(
            molecule,
            file_idx,
            state,
            geometry,
            errors,
        ).items():
            result[f"{state}__rdkit3d_{name}"] = value

        for name, value in morfeus_descriptors(
            atomnos,
            coordinates,
            file_idx,
            state,
            geometry,
            errors,
        ).items():
            result[f"{state}__morfeus_{name}"] = value
    except Exception as exc:
        add_error(
            errors,
            file_idx,
            state,
            geometry,
            exc,
            "Gaussian geometry descriptors",
        )

    try:
        result[f"{state}__spe_energy_eV"] = last_value(
            parse_output(spe),
            "scfenergies",
        )
    except Exception as exc:
        add_error(errors, file_idx, state, spe, exc)

    return result


def generate_features(
    data: pd.DataFrame,
    gs_dir: Path,
    ts_dir: Path,
) -> tuple[pd.DataFrame, list[str]]:
    """Generate the complete, unprocessed model feature matrix."""
    names = feature_names()
    records = []
    errors: list[str] = []

    for position, (_, row) in enumerate(data.iterrows(), start=1):
        file_idx = token(row["file_idx"])
        smiles = "" if pd.isna(row["smiles"]) else str(row["smiles"]).strip()

        record = {name: float("nan") for name in names}
        record["am1_barrier"] = finite_float(row["am1_barrier"])
        record.update(rdkit_2d(smiles, file_idx, errors))
        record.update(state_descriptors(row, "gs", gs_dir, errors))
        record.update(state_descriptors(row, "ts", ts_dir, errors))
        records.append(record)

        if position % 50 == 0 or position == len(data):
            print(f"Generated descriptors for {position}/{len(data)} rows...")

    features = pd.DataFrame.from_records(records, index=data.index)
    features = features.reindex(columns=names)
    features = features.replace([np.inf, -np.inf], np.nan)
    return features, errors


def read_input_csv(path: Path) -> pd.DataFrame:
    """Read the input while preserving filename tokens such as gs_x and ts_x."""
    header = pd.read_csv(path, nrows=0).columns
    string_columns = [
        name
        for name in ("file_idx", "gs_x", "ts_x")
        if name in header
    ]
    dtypes = {name: "string" for name in string_columns}
    return pd.read_csv(path, dtype=dtypes)


def main() -> None:
    input_csv = INPUT_CSV.expanduser().resolve()
    gs_dir = GS_DIR.expanduser().resolve()
    ts_dir = TS_DIR.expanduser().resolve()
    output_csv = OUTPUT_CSV.expanduser().resolve()

    if not input_csv.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    if not gs_dir.is_dir():
        raise NotADirectoryError(f"GS directory not found: {gs_dir}")
    if not ts_dir.is_dir():
        raise NotADirectoryError(f"TS directory not found: {ts_dir}")

    data = read_input_csv(input_csv)
    required_columns = {
        "file_idx",
        "smiles",
        "am1_barrier",
        "dft_barrier",
        "split",
    }
    missing_columns = required_columns.difference(data.columns)

    if missing_columns:
        raise ValueError(
            f"Input CSV is missing required column(s): {sorted(missing_columns)}"
        )

    data = data.copy()
    data["am1_barrier"] = pd.to_numeric(data["am1_barrier"], errors="raise")
    data["dft_barrier"] = pd.to_numeric(data["dft_barrier"], errors="raise")
    data["split"] = data["split"].astype(str).str.strip().str.lower()

    print(f"Input CSV: {input_csv}")
    print(f"GS files:  {gs_dir}")
    print(f"TS files:  {ts_dir}")
    print(f"Rows:      {len(data)}")
    print("Generating descriptors...\n")

    features, errors = generate_features(data, gs_dir, ts_dir)

    metadata = data[["file_idx", "smiles", "dft_barrier", "split"]].reset_index(
        drop=True
    )
    output = pd.concat([metadata, features.reset_index(drop=True)], axis=1)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_csv, index=False)

    print("\nFeature generation complete")
    print(f"Rows written:       {len(output)}")
    print(f"Descriptor columns: {features.shape[1]}")
    print(f"Errors reported:    {len(errors)}")
    print(f"Saved features to:  {output_csv}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
