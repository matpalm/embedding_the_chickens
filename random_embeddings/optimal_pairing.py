from ortools.linear_solver import pywraplp
import numpy as np


def calculate_optimal_pairing(sims):
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

    # extract and return solution as dictionary mapping
    # from { row: column, ... }
    solution = []
    for i in range(N):
        for j in range(M):
            if x[i, j].solution_value() == 1:
                solution.append((i, j))
    return dict(solution)


def optimal_pairing_to_one_hot_labels(sims, pairings):
    labels = np.zeros_like(sims)
    for i, j in pairings.items():
        labels[i, j] = 1.0
    return labels
