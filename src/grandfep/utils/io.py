from pathlib import Path
from typing import List, Tuple

from openmm import app, unit, openmm

def load_amber_sys(inpcrd_file: str | Path,
                   prmtop_file: str | Path,
                   nonbonded_settings: dict = None
                   ) -> Tuple[app.AmberInpcrdFile, app.AmberPrmtopFile, openmm.System]:
    """
    Load Amber system from inpcrd and prmtop file.

    Parameters
    ----------
    inpcrd_file :

    prmtop_file :

    nonbonded_settings :

    Returns
    -------
    inpcrd :
    prmtop :
    sys :
    """
    inpcrd = app.AmberInpcrdFile(str(inpcrd_file))
    prmtop = app.AmberPrmtopFile(str(prmtop_file),
                                 periodicBoxVectors=inpcrd.boxVectors)
    if nonbonded_settings is None:
        nonbonded_settings = {"nonbondedMethod": app.PME,
                              "nonbondedCutoff": 1.0 * unit.nanometer,
                              "constraints": app.HBonds,
                              }
    sys = prmtop.createSystem(**nonbonded_settings)
    return inpcrd, prmtop, sys