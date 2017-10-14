import sys
import os


sys.path.insert(0, os.path.join(os.path.dirname(sys.argv[0]), '..'))  # append the MIALab root directory to Python path
# fixes the ModuleNotFoundError when executing statistics.py in the console after code changes (e.g. git pull)
# somehow pip install does not keep track of packages

import mialab.utilities.statistic_utilities as statistic_utilities

if __name__ == "__main__":
    """Output overall statistics of a results.csv"""
    if(len(sys.argv) > 1):
        result_file = sys.argv[1]
    else:
        script_dir = os.path.dirname(sys.argv[0])
        path = os.path.normpath(os.path.join(script_dir, './mia-result'))
        dirs = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
        result_file = os.path.join(path, dirs[-1], 'results.csv')

    print('Statistics for ', result_file)
    print(statistic_utilities.gather_statistics(result_file))
