import itertools
from enum import Enum
import re

def MeshFEMSettings():
    return [Enum('ProjectionStrategy',     ['Adaptive', 'Always']),
            Enum('EigenvalueModification', ['Clamp', 'Abs']),
            Enum('ProjectionType',         ['FBased', 'XBased']),
            Enum('AutodiffSetting',        ['NoAD', 'AD']),
            Enum('SteplengthComputer',     ['NoFlipAvoid', 'FlipAvoid'])]

def MeshFEMVariants():
    settingTupleToDict = lambda t: {e.__class__.__name__: e for e in t}
    MeshFEMVariants = [settingTupleToDict(t) for t in itertools.product(*[list(e.__members__.values()) for e in MeshFEMSettings()])]
    return MeshFEMVariants

def MeshFEMVariantDict(n : int) -> dict:
    return MeshFEMVariants()[n]

def extract_meshfem_index(s):
    match = re.fullmatch(r'MeshFEM(\d+)', s)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"String '{s}' is not in the expected format 'MeshFEM{{n}}'")

def is_valid_solver_option(input_solver_option : str) -> bool:
    numVariants = len(MeshFEMVariants())
    variant_ind = extract_meshfem_index(input_solver_option)
    # MeshFEM_solver_options = [f'MeshFEM{n}' for n in range(2**4)]
    is_valid = (variant_ind >= 0) and (variant_ind < (numVariants))
    return is_valid

def optionNamesFromIndex(index : int):
    solver_dict = MeshFEMVariantDict(index)
    projection_strategy_name = solver_dict['ProjectionStrategy'].name
    eigenvalue_modification_name = solver_dict['EigenvalueModification'].name
    projection_type_name = solver_dict['ProjectionType'].name
    autodiff_setting_name = solver_dict['AutodiffSetting'].name
    steplength_computer_name = solver_dict['SteplengthComputer'].name
    return (projection_strategy_name, eigenvalue_modification_name, 
            projection_type_name, autodiff_setting_name, steplength_computer_name)

def optionNames(input_solver_option : str):
    if not is_valid_solver_option(input_solver_option):  raise NameError(f"Solver Option {input_solver_option} is not valid in MeshFEM+int(0~32)")
    index = extract_meshfem_index(input_solver_option)
    return optionNamesFromIndex(index)

def settingLabelFromSolverStr(input_solver_option : str) -> str:
    if not is_valid_solver_option(input_solver_option):  raise NameError(f"Solver Option {input_solver_option} is not valid in MeshFEM+int(0~32)")
    ind = extract_meshfem_index(input_solver_option)
    setting_name_tuple = optionNamesFromIndex(ind)
    setting_label = 'Ours:'
    for i in range(len(setting_name_tuple)):
        setting_label += setting_name_tuple[i]
        if i < len(setting_name_tuple) - 1:  setting_label += '+'
    return setting_label


