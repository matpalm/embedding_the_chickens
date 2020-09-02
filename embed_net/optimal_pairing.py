from ortools.linear_solver import pywraplp
import numpy as np
from detections import img_utils


def calculate(sims):
    solver = pywraplp.Solver('simple_mip_program',
                             pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

    # define a variable corresponding to pairing of ith image from first set
    # with jth from second set x[i, j] = 1 denotes images i and j are paired
    N, M = sims.shape
    x = {}
    for i in range(N):
        for j in range(M):
            x[i, j] = solver.IntVar(0, 1, name="p_%d_c_%d" % (i, j))

    # for constraints we say that each row and column has no more than a single
    # 1 in it. we say "no more than one", as opposed to "exactly one", since
    # when N!=M there will be empty rows or columns
    for i in range(N):
        num_columns_that_row_is_paired_with = sum(x[i, j] for j in range(M))
        solver.Add(num_columns_that_row_is_paired_with <= 1)
    for j in range(M):
        num_rows_that_column_is_paired_with = sum(x[i, j] for i in range(N))
        solver.Add(num_rows_that_column_is_paired_with <= 1)

    # overall objective is to maximise sum of sims
    objective = solver.Objective()
    for i in range(N):
        for j in range(M):
            coeff = int(sims[i, j] * 10000)  # o_O !!!
            objective.SetCoefficient(x[i, j], coeff)
    objective.SetMaximization()

    # run solver
    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        raise Exception("sorry, no solution; :(")

    # extract and return solution as array when ith element denotes ith row
    # associated with a[i] column
    solution = []
    for i in range(N):
        for j in range(M):
            if x[i, j].solution_value() == 1:
                solution.append(j)
    return solution


def to_one_hot_labels(sims, pairings):
    labels = np.zeros_like(sims)
    for i, j in enumerate(pairings):
        labels[i, j] = 1.0
    return labels


def collage(pairing, crops0, crops1):
    # keep record all all indexs from both sets of crops
    # so at end we can fill in ones that weren't in pairing
    idxs_from_crops0 = list(range(len(crops0)))
    idxs_from_crops1 = list(range(len(crops1)))

    # make collage placeholder, 2 rows with enough columns to
    # fit longer of crops0 or crops1
    num_cols = max([len(crops0), len(crops1)])
    single_img_shape = crops0[0].shape
    collage_imgs = np.zeros((2, num_cols, *single_img_shape))

    # iterate through optimal pairing filling in columns from left to right
    for col, (c0, c1) in enumerate(pairing.items()):
        collage_imgs[0, col] = crops0[c0]
        collage_imgs[1, col] = crops1[c1]
        idxs_from_crops0.remove(c0)
        idxs_from_crops1.remove(c1)

    # pad remainining columns with left over; will come from either crops0
    # or crops1
    for i, c0 in enumerate(idxs_from_crops0):
        collage_imgs[0, col+1+i] = crops0[c0]
    for i, c1 in enumerate(idxs_from_crops1):
        collage_imgs[1, col+1+i] = crops1[c1]

    return img_utils.collage(collage_imgs)


def score(pairing, sims):
    return sum([sims[i, j] for i, j in pairing.items()])
