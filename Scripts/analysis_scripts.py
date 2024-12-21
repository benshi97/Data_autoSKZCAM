#!/usr/bin/env python
# coding: utf-8

# Description: Contains functions for reading and plotting data from MRCC, ORCA and VASP calculations.

import numpy as np
from datetime import datetime
import pandas as pd

# Define units
kB = 8.617330337217213e-05
mol = 6.022140857e+23
kcal = 2.611447418269555e+22
kJ = 6.241509125883258e+21
Hartree = 27.211386024367243
Bohr = 0.5291772105638411


# Some basic conversion factors
cm1_to_eV = 1 / 8065.54429
hundredcm1 = 100 * cm1_to_eV * 1000
kcalmol_to_meV = kcal / mol * 1000
kjmol_to_meV = kJ / mol * 1000
mha_to_meV = Hartree

# Getting the cost of the DFT calculations
def get_vasp_walltime(filename):
    """
    Reads the walltime from the OUTCAR file.
        
    Parameters
    ----------
    filename : str
        The location of the 'OUTCAR' file to read from.
        
    Returns
    -------
    float
        The walltime in seconds.
        
    Notes
    -----
    This function reads the 'OUTCAR' file and extracts the total walltime taken by the VASP calculation.
    """

    f = open(filename)
    a = f.readlines()
    # Search for the line with "Elapsed time (sec):" string and get the last column of the line
    for line in a:
        if "Elapsed time (sec):" in line:
            total_time = float(line.split()[-1])
            break
    f.close()
    return total_time

def get_vasp_looptime(filename):
    """
    Reads the time for a single loop from the OUTCAR file.
        
    Parameters
    ----------
    filename : str
        The location of the 'OUTCAR' file to read from.
        
    Returns
    -------
    float
        The walltime in seconds.
        
    Notes
    -----
    This function reads the 'OUTCAR' file and extracts the time taken for a single SCF loop in a VASP calculation.
    """

    f = open(filename)
    a = f.readlines()
    loop_times = []
    # Search for the line with "Elapsed time (sec):" string and get the last column of the line
    for line in a:
        if "LOOP:" in line:
            loop_times += [float(line.split()[-1].replace("time", ""))]
    f.close()
    return np.mean(loop_times)


def get_orca_walltime(filename):
    """
    Reads the walltime from the orca.out file.
        
    Parameters
    ----------
    filename : str
        The location of the 'orca.out' file to read from.
   
   Returns
    -------
    float
        The walltime in seconds.
        
    Notes
    -----
    This function reads the 'orca.out' file and extracts the start and end time information to calculate the total walltime in seconds taken by the ORCA calculation. The function calculates the time duration between the timestamps found in the file and returns it in seconds.
    """

    # Get the start and end time from the file

    f = open(filename)
    line = f.readlines()[-1].split()
    f.close()

    total_time_seconds = float(line[3])*24*60*60+ float(line[5])*60*60 + float(line[7])*60 + float(line[9])
    return total_time_seconds


def get_mrcc_walltime(filename):
    """
    Reads the walltime from the mrcc.out file.
        
    Parameters
    ----------
    filename : str
        The location of the 'mrcc.out' file to read from.
        
    Returns
    -------
    float
        The walltime in seconds.
        
    Notes
    -----
    This function reads the 'mrcc.out' file and extracts the start and end time information to calculate the total walltime taken by the MRCC calculation. The function calculates the time duration between the timestamps found in the file and returns it in seconds.
    """

    # Get the start and end time from the file
    orig_time = datetime(1, 1, 1, 0, 0)
    total_time = datetime(1, 1, 1, 0, 0)

    f = open(filename)
    a = f.readlines()
    b1 = datetime.strptime(
        a[19].split()[1] + "-" + a[19].split()[2], "%Y-%m-%d-%H:%M:%S"
    )
    b2 = datetime.strptime(
        a[-3].split()[1] + "-" + a[-3].split()[2], "%Y-%m-%d-%H:%M:%S"
    )
    total_time = total_time + (b2 - b1)
    f.close()
    return (total_time - orig_time).total_seconds()


def read_vib_freq(filename, lines=None):
    """
    Read vibrational frequencies from a file.
    
    Parameters
    ----------
    filename : str
        The name of the file to read vibrational frequencies from.
    lines : list
        List of lines from the file. If not provided, the function will read lines from the file.
        
    Returns
    -------
    freq : list
        List of real vibrational frequencies.
    i_freq : list
        List of imaginary vibrational frequencies.
        
    Notes
    -----
    This function reads vibrational frequency information from a given file. It extracts both real and imaginary vibrational frequencies from the lines containing the frequency data. The frequencies are extracted based on the presence of "THz" in the data. Real frequencies are extracted unless the "f/i=" label is found, in which case imaginary frequencies are extracted. The function returns two lists containing the real and imaginary frequencies respectively.
    """

    freq = []
    i_freq = []

    # If lines are not provided, read lines from the file
    if lines is None:
        with open(filename, "r", encoding="ISO-8859-1") as f:
            lines = f.readlines()

    for line in lines:
        data = line.split()
        if "THz" in data:
            if "f/i=" not in data:
                freq.append(float(data[-2]))  # Append real frequency to the freq list
            else:
                i_freq.append(float(data[-2]))  # Append imaginary frequency to the i_freq list
    return freq, i_freq


def get_quasi_rrho(r_freq, i_freq, T):
    """
    Uses the quasi rigid rotor harmonic approximation to calculate the thermal change and zero-point energies from vibrational frequencies in cm-1 and a temperature in Kelvin.
    
    Parameters
    ----------
    r_freq : list
        List of real vibrational frequencies in cm-1.
    i_freq : list
        List of imaginary vibrational frequencies in cm-1.
    T : float
        Temperature in Kelvin.
        
    Returns
    -------
    dU : float
        The total change in energy including thermal energy and zero-point energy in eV.
    eth : float
        The thermal energy in eV.
    zpe : float
        The zero-point energy in eV.
    kT : float
        The product of Boltzmann constant (kB) and temperature (kT) in eV.
    """

    k = kB  # Boltzmann constant
    combined_freq = r_freq + [0.0001] * len(i_freq)  # Combine real and imaginary frequencies
    kT = k * T * 1000  # Calculate kT in eV

    dU = 0  # Initialize total energy change
    zpe = 0.0  # Initialize zero-point energy correction
    eth = 0.0  # Initialize thermal energy contribution
    for i in combined_freq:
        omega = 1 / (1 + ((hundredcm1 / i) ** 4))  # Calculate the vibrational frequenecy in meV
        dURRho = i / (np.exp(i / kT) - 1.0) + 0.5 * i  # Calculate the contribution to thermal energy from this frequency
        zpe += omega * 0.5 * i  # Calculate the contribution to zero-point energy
        eth += omega * i / (np.exp(i / kT) - 1.0) + (1 - omega) * 0.5 * kT  # Calculate the thermal energy contribution
        dU += omega * dURRho + (1 - omega) * 0.5 * kT  # Calculate the total energy change
    
    return dU, eth, zpe, kT  # Return the calculated values

def calculate_ezpv_etherm(
        filepath: str,
        structures: list[str],
        temperature: float,
        num_monomers: int = 1,
        filename: str = 'OUTCAR'
):
    """
    Calculate the zero-point energy and thermal energy given a set of paths to the VASP OUTCAR files of a molecule
    and a temperature in Kelvin.

    Parameters
    ----------
    filepath : str
        The path to the directory containing the OUTCAR files.
    structures : list[str]
        List of directories containing the OUTCAR files for the three structures for adsorption energy.
    temperature : float
        Temperature in Kelvin.
    num_monomers : int
        The number of monomers in the cluster.
    filename : str
        The name of the OUTCAR file.

    Returns
    -------
    float
        The zero-point energy in eV.
    float
        The thermal energy in eV.
    float
        kT/RT in eV.
    float
        Sum of the zero-point energy and thermal energy and kT to get contribution to Eads
    """

    energy_dict = {structure: {'E_ZPV': 0.0, 'E_therm': 0.0} for structure in structures + ['Hads']}

    for structure in structures:
        real_freqs, imag_freqs = read_vib_freq(f'{filepath}/{structure}/{filename}') # Adsorbate on Slab
        # Calculate thermal and zero-point corrections as well as RT term for adsorbate and adsorbate on slab
        total_energy, ethermal, ezpv, kT = get_quasi_rrho(real_freqs, imag_freqs, temperature)
        energy_dict[structure]['E_ZPV'] = ezpv
        energy_dict[structure]['E_therm'] = ethermal
        
        if structure == 'Molecule':
            energy_dict[structure]['E_ZPV'] = ezpv*num_monomers
            energy_dict[structure]['E_therm'] = ethermal*num_monomers
        else:
            energy_dict[structure]['E_ZPV'] = ezpv
            energy_dict[structure]['E_therm'] = ethermal



    energy_dict['Hads']['E_ZPV'] = (energy_dict[structures[0]]['E_ZPV'] - np.sum([energy_dict[structure]['E_ZPV'] for structure in structures[1:]]))/num_monomers
    energy_dict['Hads']['E_therm'] = (energy_dict[structures[0]]['E_therm'] - np.sum([energy_dict[structure]['E_therm'] for structure in structures[1:]]))/num_monomers
    energy_dict['Hads']['kT'] = kT
    energy_dict['Hads']['DeltaH'] = energy_dict['Hads']['E_ZPV'] + energy_dict['Hads']['E_therm'] - energy_dict['Hads']['kT']
    return energy_dict['Hads']['E_ZPV'], energy_dict['Hads']['E_therm'], energy_dict['Hads']['kT'], energy_dict['Hads']['DeltaH']
    

def fit_eint(x_data, y_data):
    x_transformed_data = np.array([1/x for x in x_data])

    # Degree of the polynomial
    degree = 1  # You can change the degree based on your requirement

    # Perform the polynomial fit
    coefficients = np.polyfit(x_transformed_data, y_data, degree)

    # Print the coefficients of the polynomial
    return coefficients[0], coefficients[0], coefficients[1]

def get_skzcam_cluster_size(file_path):
    """
    Obtain the number of atoms in the cluster from an ORCA input file

    Parameters
    ----------
    file_path : str
        The path to the ORCA input file

    Returns
    -------
    num_atoms : int
        The number of atoms in the cluster
    """
    num_atoms = 0
    with open(file_path,'r') as file:
        process_lines = False
        # Loop through each line in the file
        for line in file:
            # Check if the line contains the string "coords"
            if "coords" in line:
                # Set the flag to True to start processing lines
                process_lines = True
                continue  # Skip the line with "coords" itself

            # If the flag is True, process the line
            if process_lines:
                # Do something with the line (e.g., print it)
                if ('Mg   ' in line) or ('O   ' in line) or ('Ti   ' in line):
                    num_atoms += 1

    return num_atoms

def calculate_skzcam_eint(
        filedir: str,
        outputname: str,
        code: str,
        method: str,
        basis: str | list[str],
        structure_labels: list[str] = ['Molecule-Surface', 'Molecule','Surface'],
        cbs_type: str = 'correlation_energy'
):
    """
    Calculate the Eads value for a given cluster using the SKZCAM protocol.
    
    Parameters
    ----------
    filedir : str
        The location of the directory containing the output files.
    outputname : str
        The name of the output file which contains the total energy of the individual structures.
    structure_labels : list
        List of directories containing the output files for the three structures for adsorption energy.
    code : str
        The code format. Options are 'mrcc', 'vasp', 'quantum_espresso', 'cc4s', 'vasp_wodisp', 'dftd3', 'orca', 'orca_mp2'
    method : str
        The type of calculation performed in MRCC or ORCA.
    basis : str
        The basis set used in the calculation.
    cbs_type : str
        The type of complete basis set extrapolation used. Options are 'scf_energy', 'correlation_energy'.
        
    Returns
    -------
    float
        The adsorption energy in its original units.
        
    """
    basis_zeta_conversion = {
        'DZ': 2,
        'TZ': 3,
        'QZ': 4,
        '5Z': 5
    }

    # Check if basis is a string or a list
    if isinstance(basis, str):
        # If it's just a string, simply calculate the Eads value without extrapolation
        zeta_outputname = f'{outputname}{basis}'
        final_eint = calculate_eint(filedir = filedir, outputname=zeta_outputname, code=code, method=method,structure_labels=structure_labels)*Hartree*1000


    elif isinstance(basis, list):
        zeta_energy_list = []
        for zeta in basis:
            zeta_outputname = f'{outputname}{zeta}'
            zeta_energy_list += [calculate_eint(filedir = filedir, outputname=zeta_outputname, code=code, method=method,structure_labels=structure_labels)*Hartree*1000]

        # Perform the CBS extrapolation
        if cbs_type == 'correlation_energy':
            final_eint = get_cbs(0,zeta_energy_list[0], 0, zeta_energy_list[1],X=basis_zeta_conversion[basis[0]],Y=basis_zeta_conversion[basis[1]],family='mixcc',output=False)[1]
        elif cbs_type == 'scf_energy':
            final_eint = get_cbs(zeta_energy_list[0], 0, zeta_energy_list[1],0,X=basis_zeta_conversion[basis[0]],Y=basis_zeta_conversion[basis[1]],family='mixcc',output=False)[0]

    return final_eint

def calculate_eint(
    filedir, outputname = None, code="mrcc", method="ccsdt", structure_labels=["Molecule-Surface", "Surface", "Molecule"], vasp_outcar_label = 'OUTCAR', num_monomers = 1
):
    """
    Function to calculate the Eads value for a given cluster.
    
    Parameters
    ----------
    filename : str
        The location of the directory containing the output files.
    outputname : str
        The common name of the output file which contains the total energy of the individual structures.
    code : str
        The code format. Options are 'mrcc', 'vasp', 'quantum_espresso', 'cc4s', 'vasp_wodisp', 'dftd3', 'orca', 'orca_mp2'
    method : str
        The type of calculation performed in MRCC or ORCA.
    structure_labels : list
        List of directories containing the output files for the three structures for adsorption energy.
    vasp_outcar_label : str
        The name of the VASP OUTCAR file.
    num_monomers : int
        The number of monomers in the cluster.
        
    Returns
    -------
    float
        The adsorption energy in its original units.
        
    """

    if code == "mrcc":
        eint = get_energy(
            filedir + "/{0}/{1}.mrcc.out".format(structure_labels[0],outputname),
            code=code,
            method=method,
        )

        for structure in structure_labels[1:-1]:
            eint -= get_energy(
                filedir + "/{0}/{1}.mrcc.out".format(structure,outputname),
                code=code,
                method=method,
            )

        eint -= get_energy(
            filedir + "/{0}/{1}.mrcc.out".format(structure_labels[-1],outputname),
            code=code,
            method=method,
        )*num_monomers

    elif "orca" in code:
        eint = get_energy(
            filedir + "/{0}/{1}.orca.out".format(structure_labels[0],outputname),
            code=code,
            method=method,
        )

        for structure in structure_labels[1:-1]:
            eint -= get_energy(
                filedir + "/{0}/{1}.orca.out".format(structure,outputname),
                code=code,
                method=method,
            )

        eint -= get_energy(
            filedir + "/{0}/{1}.orca.out".format(structure_labels[-1],outputname),
            code=code,
            method=method,
        )*num_monomers

    elif "vasp" in code:
        eint = get_energy(filedir + "/{0}/{1}".format(structure_labels[0],vasp_outcar_label),
            code=code,
            method=method,
        )

        for structure in structure_labels[1:-1]:
            eint -= get_energy(filedir + "/{0}/{1}".format(structure,vasp_outcar_label),
                code=code,
                method=method,
            )
        
        eint -= get_energy(filedir + "/{0}/{1}".format(structure_labels[-1],vasp_outcar_label),
            code=code,
            method=method,
        )*num_monomers

    elif 'd4' in code:
        eint = get_energy(
            filedir + "/{0}/{1}".format(structure_labels[0],vasp_outcar_label),
            code=code,
            method=method,
        )

        for structure in structure_labels[1:-1]:
            eint -= get_energy(
                filedir + "/{0}/{1}".format(structure,vasp_outcar_label),
                code=code,
                method=method,
            )

        eint -= get_energy(
            filedir + "/{0}/{1}".format(structure_labels[-1],vasp_outcar_label),
            code=code,
            method=method,
        )*num_monomers

    return eint/num_monomers


def get_energy(filename, method="ccsdt", code="mrcc"):
    """
    Function to parse the energy from a MRCC or ORCA output file.
    
    Parameters
    ----------
    filename : str
        The location of the output file to read from.
    method : str
        The type of method to read.
    code : str
        The code format. Options are 'mrcc', 'vasp', 'quantum_espresso', 'cc4s', 'vasp_wodisp', 'dftd3', 'orca', 'orca_mp2'
        
    Returns
    -------
    float
        The energy in the original units.
    """

    if code == "mrcc":
        if method == "lccsdt_corr":
            search_word = "CCSD(T) correlation energy + MP2 corrections [au]:"
        elif method == "lccsdt_total":
            search_word = "Total LNO-CCSD(T) energy with MP2 corrections [au]"
        elif method == "lccsdt_lmp2_total":
            search_word = "Total LMP2 energy [au]"
        elif method == "ccsdt_mp2_total":
            search_word = "Total MP2 energy [au]"
        elif method == "ccsdt_corr":
            search_word = "CCSD(T) correlation energy [au]:"
        elif method == "ccsdt_total":
            search_word = "Total CCSD(T) energy"
        elif method == "hf":
            search_word = "Reference energy [au]:    "
        elif method == "lmp2_corr":
            search_word = "LMP2 correlation energy [au]:         "
        elif method == "lmp2_total":
            search_word = "DF-MP2 energy [au]:       "

        elif method == "mp2_corr":
            search_word = "MP2 correlation energy [au]:   "
        elif method == "lccsd_corr":
            search_word = "CCSD correlation energy + 0.5 MP2 corrections [au]:"
        elif method == "lccsd_total":
            search_word = "Total LNO-CCSD energy with MP2 corrections [au]:"
        elif method == "fnoccsdt_total":
            search_word = "Total CCSD(T+) energy + MP2 + PPL corr. [au]"
        elif method == "fnoccsd_total":
            search_word = "Total CCSD energy + MP2 + PPL corr. [au]:"
        elif method == "fnoccsdt_mp2_total":
            search_word = "DF-MP2 energy [au]:"
        elif method == "fnoccsdt_corr":
            search_word = "CCSD(T+) correlation en. + MP2 + PPL corr. [au]:"
        elif method == "fnoccsd_corr":
            search_word = "CCSD correlation energy + MP2 + PPL corr. [au]:"
        elif method == "fnoccsdt_mp2_corr":
            search_word = "DF-MP2 correlation energy"
        elif method == "fnomp2_corr":
            search_word = "DF-MP2 correlation energy"

        elif method == "ccsd_corr":
            search_word = "CCSD correlation energy [au]: "
        elif method == "ccsd_total":
            search_word = "Total CCSD energy [au]: "
        elif method == "dft_total":
            search_word = "***FINAL KOHN-SHAM ENERGY:"
        elif method == "B2PLYP_corr":
            search_word = "MP2 contribution [au]:"
        elif method == "DSDPBEP86_corr":
            search_word = "SCS-MP2 contribution [au]:"

        with open(filename, "r") as fp:
            a = [line for line in fp if search_word in line]
        if len(a) == 0:
            return 0.0
        else:
            if method == "dft_total":
                return float(a[-1].split()[-2])
            elif "fnoccsdt_mp2" in method:
                return float(a[0].split()[-1])

            else:
                return float(a[-1].split()[-1])

    elif code == "vasp":
        search_word = "energy  without entropy="
        with open(filename, "r", encoding="ISO-8859-1") as fp:
            a = [line for line in fp if search_word in line]
        if len(a) == 0:
            return 0.0
        else:
            return float(a[-1].split()[-1])
        
    elif code == "vasp_rpa":
        search_word = "converged value"
        with open(filename, "r", encoding="ISO-8859-1") as fp:
            a = [line for line in fp if search_word in line]
        if len(a) == 0:
            return 0.0
        else:
            return float(a[-1].split()[-2])

    elif code == "quantum_espresso":
        search_word = "!    total energy              ="
        with open(filename, "r", encoding="ISO-8859-1") as fp:
            a = [line for line in fp if search_word in line]
        if len(a) == 0:
            return 0.0
        else:
            return float(a[-1].split()[-2])

    elif code == "cc4s":
        if method == "CCSD corr":
            search_word = "Ccsd correlation energy:"
        elif method == "CCSD FS":
            search_word = "Finite-size energy correction:"
        elif method == "CCSD BSIE":
            search_word = "Ccsd-Bsie energy correction:"
        elif method == "HF":
            search_word = "energy  without entropy="
        elif method == "(T) corr":
            search_word = "(T) correlation energy:"
        elif method == "MP2 FS":
            search_word = "Finite-size energy correction:"
        elif method == "MP2 corr":
            search_word = "converged values  "

        with open(filename, "r", encoding="ISO-8859-1") as fp:
            a = [line for line in fp if search_word in line]
        if len(a) == 0:
            return 0.0
        else:
            return float(a[-1].split()[-1])

    elif code == "vasp_wodisp":
        search_word = "energy without entropy ="
        with open(filename, "r", encoding="ISO-8859-1") as fp:
            a = [line for line in fp if search_word in line]
        if len(a) == 0:
            return 0.0
        else:
            return float(a[-1].split()[-1])

    elif code == "dftd3":
        search_word = " Edisp /kcal,au"
        with open(filename, "r", encoding="ISO-8859-1") as fp:
            a = [line for line in fp if search_word in line]
        if len(a) == 0:
            return 0.0
        else:
            return float(a[-1].split()[-1])
        
    elif code == "dftd4":
        search_word = "Dispersion energy:"
        with open(filename, "r", encoding="ISO-8859-1") as fp:
            a = [line for line in fp if search_word in line]
        if len(a) == 0:
            return 0.0
        else:
            return float(a[-1].split()[-2])

    elif code == "orca":
        if method == "dlpnoccsdt_corr":
            search_word = "Final correlation energy"
        elif method == "dlpnoccsdt_hf":
            search_word = "E(0)"
        elif method == "dlpnomp2_corr":
            search_word = "DLPNO-MP2 CORRELATION ENERGY"
        elif method == "dlpnomp2_hf":
            search_word = "Total Energy       :"
        elif method == "dlpnoccsd_corr":
            search_word = "E(CORR)(corrected)"
        elif method == "dlpnoccsd_hf":
            search_word = "E(0)"
        elif method == "mp2_corr":
            search_word = "RI-MP2 CORRELATION ENERGY"
        elif method == "mp2_hf":
            search_word = "Total Energy       :"
        elif 'total' in method:
            search_word = "FINAL SINGLE POINT ENERGY"


        with open(filename, "r") as fp:
            a = [line for line in fp if search_word in line]
        if len(a) == 0:
            return 0.0
        elif method in ["dlpnomp2_hf", "mp2_hf"]:
            return float(a[-1].split()[-4])
        elif method in ["mp2_corr","dlpnomp2_corr"]:
            return float(a[-1].split()[-2])
        else:
            return float(a[-1].split()[-1])


def get_cbs(
    hf_X,
    corr_X,
    hf_Y,
    corr_Y,
    X=2,
    Y=3,
    family="cc",
    convert_Hartree=False,
    shift=0.0,
    output=True,
):
    """
    Function to perform basis set extrapolation of HF and correlation energies for both the cc-pVXZ and def2-XZVP basis sets
    
    Parameters
    ----------
    hf_X : float
        HF energy in X basis set
    corr_X : float
        Correlation energy in X basis set
    hf_Y : float
        HF energy in Y basis set where Y = X+1 cardinal zeta number
    corr_Y : float
        Correlation energy in Y basis set
    X : int
        Cardinal zeta number of X basis set
    Y : int
        Cardinal zeta number of Y basis set
    family : str
        Basis set family. Options are 'cc', 'def2', 'acc', and 'mixcc'. Where cc is for non-augmented correlation consistent basis sets, def2 is for def2 basis sets, acc is for augmented correlation consistent basis sets while mixcc is for mixed augmented + non-augmented correlation consistent basis sets
    convert_Hartree : bool
        If True, convert energies to Hartree
    shift : float
        Energy shift to apply to the CBS energy
    output : bool
        If True, print CBS energies

    Returns
    -------
    hf_cbs : float
        HF CBS energy
    corr_cbs : float
        Correlation CBS energy
    tot_cbs : float
        Total CBS energy
    """

    # Dictionary of alpha parameters followed by beta parameters in CBS extrapoation. Refer to: Neese, F.; Valeev, E. F. Revisiting the Atomic Natural Orbital Approach for Basis Sets: Robust Systematic Basis Sets for Explicitly Correlated and Conventional Correlated Ab Initio Methods. J. Chem. Theory Comput. 2011, 7 (1), 33–43. https://doi.org/10.1021/ct100396y.
    alpha_dict = {
        "def2_2_3": 10.39,
        "def2_3_4": 7.88,
        "cc_2_3": 4.42,
        "cc_3_4": 5.46,
        "cc_4_5": 5.46,
        "acc_2_3": 4.30,
        "acc_3_4": 5.79,
        "acc_4_5": 5.79,
        "mixcc_2_3": 4.36,
        "mixcc_3_4": 5.625,
        "mixcc_4_5": 5.625,
    }

    beta_dict = {
        "def2_2_3": 2.40,
        "def2_3_4": 2.97,
        "cc_2_3": 2.46,
        "cc_3_4": 3.05,
        "cc_4_5": 3.05,
        "acc_2_3": 2.51,
        "acc_3_4": 3.05,
        "acc_4_5": 3.05,
        "mixcc_2_3": 2.485,
        "mixcc_3_4": 3.05,
        "mixcc_4_5": 3.05,
    }

    # Check if X and Y are consecutive cardinal zeta numbers
    if Y != X + 1:
        print("Y does not equal X+1")

    # Check if basis set family is valid
    if family != "cc" and family != "def2" and family != "acc" and family != "mixcc":
        print("Wrong basis set family stated")

    # Get the corresponding alpha and beta parameters depending on the basis set family
    alpha = alpha_dict["{0}_{1}_{2}".format(family, X, Y)]
    beta = beta_dict["{0}_{1}_{2}".format(family, X, Y)]

    # Perform CBS extrapolation for HF and correlation components
    hf_cbs = hf_X - np.exp(-alpha * np.sqrt(X)) * (hf_Y - hf_X) / (
        np.exp(-alpha * np.sqrt(Y)) - np.exp(-alpha * np.sqrt(X))
    )
    corr_cbs = (X ** (beta) * corr_X - Y ** (beta) * corr_Y) / (
        X ** (beta) - Y ** (beta)
    )

    # Convert energies from Hartree to eV if convert_Hartree is True
    if convert_Hartree == True:
        if output == True:
            print(
                "CBS({0}/{1}) HF: {2:.9f} Corr: {3:.9f} Tot: {4:.9f}".format(
                    X,
                    Y,
                    hf_cbs * Hartree + shift,
                    corr_cbs * Hartree,
                    (hf_cbs + corr_cbs) * Hartree + shift,
                )
            )
        return (
            hf_cbs * Hartree + shift,
            corr_cbs * Hartree,
            (hf_cbs + corr_cbs) * Hartree,
        )
    else:
        if output == True:
            print(
                "CBS({0}/{1})  HF: {2:.9f} Corr: {3:.9f} Tot: {4:.9f}".format(
                    X, Y, hf_cbs + shift, corr_cbs, (hf_cbs + corr_cbs) + shift
                )
            )
        return hf_cbs + shift, corr_cbs, (hf_cbs + corr_cbs) + shift
    

def convert_df_to_latex_input(
    df,
    start_input = '\\begin{table}\n',
    end_input = '\n\\end{table}',
    label = "tab:default",
    caption = "This is a table",
    replace_input = {},
    df_latex_skip = 0,
    adjustbox = 0,
    scalebox = False,
    multiindex_sep = "",
    filename = "./table.tex",
    index = True,
    column_format = None,
    center = False,
    rotate_column_header = False,
    output_str = False
):
    if column_format is None:
        column_format = "l" + "r" * len(df.columns)
    
    if label != "":
        label_input = r"\label{" + label + r"}"
    else:
        label_input = ""
    caption_input = r"\caption{" + label_input + caption +  "}"

    if rotate_column_header:
        df.columns = [r'\rotatebox{90}{' + col + '}' for col in df.columns]

    with pd.option_context("max_colwidth", 1000):
        df_latex_input = df.to_latex(escape=False, column_format=column_format,multicolumn_format='c', multicolumn=True,index=index)
    for key in replace_input:
        df_latex_input = df_latex_input.replace(key, replace_input[key])
    
    df_latex_input_lines = df_latex_input.splitlines()[df_latex_skip:]
    # Get index of line with midrule
    toprule_index = [i for i, line in enumerate(df_latex_input_lines) if "toprule" in line][0]
    df_latex_input_lines[toprule_index+1] = df_latex_input_lines[toprule_index+1] + ' ' + multiindex_sep
    df_latex_input = '\n'.join(df_latex_input_lines)
    end_adjustbox = False

    if output_str:
        latex_string = ""
        latex_string += start_input + "\n"
        latex_string += caption_input + "\n"
        if center == True and adjustbox == 0:
            latex_string += r"\begin{adjustbox}{center}" + "\n"
            end_adjustbox = True
        elif adjustbox > 0 and center == False:
            latex_string += r"\begin{adjustbox}{max width=" + f"{adjustbox}" + r"\textwidth}" + "\n"
            end_adjustbox = True    
        elif adjustbox > 0 and center == True:
            latex_string += r"\begin{adjustbox}{center,max width=" + f"{adjustbox}" + r"\textwidth}" + "\n"
            end_adjustbox = True
        if scalebox:
            latex_string += r"\begin{adjustbox}{scale=" + f"{scalebox}" + "}" + "\n"
            end_adjustbox = True
        latex_string += df_latex_input
        if end_adjustbox:
            latex_string += "\n\\end{adjustbox}"
        latex_string += "\n" + end_input
        return latex_string

    else:
        with open(filename, "w") as f:
            f.write(start_input + "\n")
            f.write(caption_input + "\n")
            if center == True and adjustbox == 0:
                f.write(r"\begin{adjustbox}{center}" + "\n")
                end_adjustbox = True
            elif adjustbox > 0 and center == False:
                f.write(r"\begin{adjustbox}{max width=" + f"{adjustbox}" + r"\textwidth}" + "\n")
                end_adjustbox = True    
            elif adjustbox > 0 and center == True:
                f.write(r"\begin{adjustbox}{center,max width=" + f"{adjustbox}" + r"\textwidth}" + "\n")
                end_adjustbox = True
            if scalebox:
                f.write(r"\begin{adjustbox}{scale=" + f"{scalebox}" + "}" + "\n")
                end_adjustbox = True
            f.write(df_latex_input)
            if end_adjustbox:
                f.write("\n\\end{adjustbox}")
            f.write("\n" + end_input)
        
