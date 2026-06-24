import sys
from pathlib import Path

def get_project_paths(project_root=None):
    """
    Returns a dictionary of common project paths.

    Args:
        project_root (str or Path, optional): The root path of phd_storm_projects.
            If None, it will infer based on this file's location.

    Returns:
        dict: Keys like 'huxt_code', 'huxt_tools', 'classification_src', etc.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]  # assumes: storm_utils/storm_utils/config_paths.py

    paths = {
        'project_root': project_root,
        'huxt_code': project_root / 'HUXt' / 'code',
        'huxt_tools': project_root / 'HUXt_tools',
        'storm_utils': project_root / 'storm_utils' / 'storm_utils',
        'storm_utils_figures': project_root / 'storm_utils' / 'figures',
        'classification_src': project_root / 'storm_classification',
        'regression_src': project_root / 'storm_regression' / 'src',
        'data_shared': project_root / 'storm_utils' / 'shared_data',  # for shared output
        'huxt_data_shared': project_root / 'HUXt' / 'data' / 'HUXt',  # huxt output location
        'huxt_figures' : project_root / 'HUXt' / 'figures',
        'torch_model_weights': project_root / 'storm_regression' / 'src' / 'torch_model_weights',
        'preds_and_targets': project_root / 'storm_regression' / 'src' / 'figures' / 'preds_and_targets',
        'regression_results': project_root / 'storm_regression' / 'results',
        'regression_metrics': project_root / 'storm_regression' / 'metrics',
        'regression_figures': project_root / 'storm_regression' / 'figures',
    }

    return paths


def add_huxt_paths(project_root=None):
    """
    Adds HUXt/code and HUXt_tools to sys.path
    project_root: Optional path to phd_storm_projects/ folder
    """
    paths = get_project_paths(project_root)

    for key in ['huxt_code', 'huxt_tools']:
        path = paths[key]
        if path.exists():
            sys.path.append(str(path))
        else:
            print(f'Warning: {key} path not found: {path}')

    print('HUXt and HUXt_tools paths configured.')
