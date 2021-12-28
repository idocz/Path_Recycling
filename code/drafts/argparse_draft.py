import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--foo', dest="doo", help='foo help')
args = parser.parse_args()
print(args.doo)