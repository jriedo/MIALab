import numpy as np

def gather_statistics(result_file):
    buffer = []
    stats = dict()
    file = open(result_file, 'r')
    for line in file.readlines():
        if (line.startswith('ID') == False and len(line) > 2):
            data = line.rstrip().split(';')
            pp = '-PP' if data[0].endswith('-PP') else ''
            key = data[1] + pp
            score = data[2]
            if (not key in stats):
                l = []
            else:
                l = stats[key]
            l.append(float(score))
            stats[key] = l

    for key in stats.keys():
        scores = np.array(stats[key])
        buffer.append(key)
        buffer.append('\tmean: ' + str(scores.mean()))
        buffer.append('\tmin: ' + str(scores.min()))
        buffer.append('\tmax: ' + str(scores.max()))
        buffer.append('\tstd: ' + str(scores.std()))

    return '\n'.join(buffer)



