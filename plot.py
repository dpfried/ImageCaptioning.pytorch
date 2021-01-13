import sys

import argparse
import matplotlib.pyplot as plt
import pandas

STATS = [
    'Bleu_4', 
    'METEOR',
    'ROUGE_L',
    'CIDEr',
    'WMD',
]

DEFAULT_STATS = STATS

def parse(log_file):
    last_iter = None
    stats_by_iter = {}
    try:
        with open(log_file, 'r') as f:
            for line in f:
                if line.startswith('iter'):
                    toks = line.split()
                    last_iter = int(toks[1])
                else:
                    for stat in STATS:
                        if line.startswith('{}: '.format(stat)):
                            if last_iter not in stats_by_iter:
                                stats_by_iter[last_iter] = {}
                            stat_val = float(line.split()[1])
                            stats_by_iter[last_iter][stat] = stat_val
    except Exception as e:
        print(e)
        return None
    if not stats_by_iter:
        return None
    return stats_by_iter

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log_files", nargs="+")
    parser.add_argument("--stats", nargs="+", default=DEFAULT_STATS)
    args = parser.parse_args()
    dfs_by_name = {}
    for log_file in args.log_files:
        stats_by_iter = parse(log_file)
        if stats_by_iter is None:
            print("error parsing {}".format(log_file))
            continue
        df = pandas.DataFrame(stats_by_iter).transpose()
        dfs_by_name[log_file] = df
        last_stats = df[DEFAULT_STATS].iloc[-1]
        # print(last_stats)
        print("{}\t{}\t{}".format(
            log_file,
            last_stats.name,
            ','.join('{:.4f}'.format(x) for x in last_stats)
        ))
    for stat in args.stats:
        data = {}
        for name, df in dfs_by_name.items():
            if stat in df.columns:
                data[name] = df[stat]
            else:
                print('df {} does not have stat {}'.format(name, stat))
        collected_df = pandas.DataFrame(data)
        if collected_df.empty:
            print("stat {} is empty".format(stat))
        else:
            collected_df.plot(title=stat)
        plt.show()
