import sys
import os
import numpy as np


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
    stats = dict()
    file = open(result_file, 'r')
    for line in file.readlines():
        if (line.startswith('ID') == False and len(line) > 2):
            data = line.rstrip().split(';')
            key = data[1]
            score = data[2]
            if(not key in stats):
                l = []
            else:
                l = stats[key]
            l.append(float(score))
            stats[key] = l

    for key in stats.keys():
        scores = np.array(stats[key])
        print(key)
        print('\tmean: ', scores.mean())
        print('\tmin: ', scores.min())
        print('\tmax: ', scores.max())
        print('\tstd: ', scores.std())


