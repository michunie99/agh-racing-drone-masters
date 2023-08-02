import pybullet as p
import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument('floats', metavar='N', type=float, nargs='+',
                    help='an floats for the accumulator')
parser.add_argument('-d', '--degrees',
                    action='store_true')

args = parser.parse_args()

if args.degrees:
    angles = list(map(lambda x: math.radians(x), args.floats))
else:
    angles = args.floats

assert len(angles) == 3, "Provide 3 angles !!!"
print(p.getQuaternionFromEuler(angles))