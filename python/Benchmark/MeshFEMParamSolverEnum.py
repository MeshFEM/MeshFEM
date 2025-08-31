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

def get_setting_options(setting_name: str):
    """
    Given a setting name (e.g. 'ProjectionStrategy'),
    return a tuple of its two option names as strings.
    """
    for enum_cls in MeshFEMSettings():
        if enum_cls.__name__ == setting_name:
            return tuple(enum_cls.__members__.keys())
    raise ValueError(f"Setting {setting_name!r} not found.")

def variant_pairs_by_setting(setting_name: str) -> list:
    """
    Return 16 tuples (i, j) of indices into MeshFEMVariants such that
    the two entries differ only in the enum named `setting_name`.
    """
    settings = MeshFEMSettings()
    names = [e.__name__ for e in settings]
    if setting_name not in names:
        raise ValueError(f"setting_name must be one of {names}, got {setting_name!r}")

    # Determine which "bit" this setting corresponds to in the product order.
    # Rightmost enum varies fastest, so its stride is 1; next is 2; etc.
    pos = names.index(setting_name)
    exponent = (len(settings) - 1) - pos
    stride = 1 << exponent

    n = len(MeshFEMVariants())  # 32 for 5 binary enums
    pairs = []
    for i in range(n):
        # Only take each pair once: include when this bit is 0
        if (i & stride) == 0:
            pairs.append((i, i ^ stride))
    return pairs

def filter_pairs_by_option(pairs_tuple_list: list, filteredSettingName: str, option_ind: int) -> list:
    """
    Given a pairs_tuple list, delete all pairs which contain the specific option
    Warning: constraint on filteredSettingName
    """
    option_indices = [t[option_ind] for t in variant_pairs_by_setting(filteredSettingName)]
    filtered_pairs_tuple_list = [p for p in pairs_tuple_list if p[option_ind] in option_indices]
    return filtered_pairs_tuple_list

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


def get_pairline_label(pairs_tuple_list: list, setting_name: str, pair_ind: int) -> str:
    """
    Get the pairline label for pairs sharing a same label
    """
    meshfem_settings = MeshFEMSettings()
    setting_names_list = [e.__name__ for e in meshfem_settings]
    pos = setting_names_list.index(setting_name)
    
    pair_tuple = pairs_tuple_list[pair_ind]
    option_name_tuple = optionNamesFromIndex(pair_tuple[0])
    num_options = len(option_name_tuple)
    pair_label_str = ""
    for i in range(num_options):
        if i == pos:
            continue
        else:
            pair_label_str += option_name_tuple[i] + "+"
    return pair_label_str[:-1]


