#!/bin/python3

import math
import os
import random
import re
import sys


#
# Complete the 'rotLeft' function below.
#
# The function is expected to return an INTEGER_ARRAY.
# The function accepts following parameters:
#  1. INTEGER_ARRAY a
#  2. INTEGER d
#

# offer this solution first before showing or explaining the second one
def rotLeft(a, d):
    # Write your code here
    for n in range(d):
        f = a[0]
        del a[0]
        a.append(f)

    return a

def rotLeft_better(a, d):
    first_slice = a[:d]
    second_slice = a[d:]
    return second_slice + first_slice

if __name__ == '__main__':

    first_multiple_input = input().rstrip().split()

    n = int(first_multiple_input[0])

    d = int(first_multiple_input[1])

    a = list(map(int, input().rstrip().split()))

    result = rotLeft_better(a, d)

    print(result)