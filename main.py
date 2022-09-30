## Synthesis of time-optimal multi-qubit gates
#

import cvxpy as cp
import mosek  # This is required to solve the Mixed integer program
import numpy as np
import itertools
from pylab import cm
import matplotlib as mpl
from couplings import *

## Basic input parameters
#########################

## General parameters
nr_of_qubits = range(3, 14)
nr_of_samples = 20  # Nr of runs per dimension
J_unit = 18.1951555617  # kHz (Unit of the coupling matrix)

## MIP parameters
c_l = 0.5  # time lower bound
c_u_scaling = 1.5  # scaling of upper bound on time: c_u = c_u_scaling*max(M)
alpha = 0.5

# Solver (MOSEK) parameter
rel_opt_tol = 0.6  # Default: 1.0e-4

## Initialize lists for the results
# t vector
LP_t_all = []; LP_t_avrg = []; MIP_t_all = []; MIP_t_avrg = []
# Non zero entries of target matrix
M_nz_all = []; M_nz_avrg = []
# Sum over all elements of target matrix
M_sum_all = []; M_sum_avrg = []
max_hess = []

external = Quadratic1D()
for n in nr_of_qubits:
    ## Calculate the inverse Hessian
    Hinv = coupling_matrix(n, external)
    max_hess.append(1.0 / (np.min(Hinv)*J_unit))
    ## Generate the set of all outer products
    b = list(itertools.product([1, -1], repeat=n))[:int(2 ** (n-1))]
    m = len(b)
    i_lower = np.tril_indices(n, -1)
    out_prod = np.array([np.outer(b[i], b[i])[i_lower] for i in range(m)]).T
    for sample in range(nr_of_samples):
        print("Nr of qubits: ", n, "   Sample: ", sample)
        ## Logical coupling matrix
        A = np.zeros((n, n))
        while sum(A[i_lower]) == 0.:
            A = np.random.choice([0.0, 1.0], size=(n, n))
        y = np.divide(A[i_lower], Hinv[i_lower])

        ## naive approach
        #################
        M_sum_avrg.append(np.sum(y))
        M_nz_avrg.append(np.count_nonzero(y))

        ## Linear program
        #################
        x = cp.Variable(m, nonneg=True)  # Solve for non negative time
        objective = cp.Minimize(cp.norm(x, 1))
        constraints = [out_prod @ x == y]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.GLPK, glpk={'msg_lev': 'GLP_MSG_OFF'})
        LP_t = x.value
        LP_t[LP_t < 1e-12] = 0.0
        LP_t_avrg.append(LP_t)

        ## Mixed integer program
        ########################
        z = cp.Variable(m, boolean=True)
        t = cp.Variable(m, nonneg=True)
        c_u = c_u_scaling * np.max(y)
        if alpha == 0.:
            objective = cp.Minimize(cp.norm(z, 1))
        elif alpha == 1.:
            objective = cp.Minimize(cp.norm(t, 1))
        else:
            objective = cp.Minimize(alpha*cp.norm(t, 1) + (1.-alpha) * cp.norm(z, 1))
        constraints = [t <= cp.multiply(c_u, z),
                       cp.multiply(c_l, z) <= t]
        constraints.append(out_prod @ t == y)
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.MOSEK, verbose=False, mosek_params={"MSK_DPAR_MIO_TOL_REL_GAP": rel_opt_tol})
        MIP_t = t.value
        MIP_t[MIP_t < 1e-12] = 0.0
        MIP_t_avrg.append(MIP_t)

    LP_t_all.append(np.array(LP_t_avrg)/J_unit)
    LP_t_avrg = []
    MIP_t_all.append(np.array(MIP_t_avrg)/J_unit)
    MIP_t_avrg = []
    M_nz_all.append(np.array(M_nz_avrg))
    M_nz_avrg = []
    M_sum_all.append(np.array(M_sum_avrg)/J_unit)
    M_sum_avrg = []

## Prepare data for plots
#########################

LP_non_zero_mean = []; LP_non_zero_min = []; LP_non_zero_max = []
LP_total_time_mean = []; LP_total_time_min = []; LP_total_time_max = []

MIP_non_zero_mean = []; MIP_non_zero_min = []; MIP_non_zero_max = []
MIP_total_time_mean = []; MIP_total_time_min = []; MIP_total_time_max = []

AA_nz_mean = []; AA_nz_min = []; AA_nz_max = []
AA_sum_mean = []; AA_sum_min = []; AA_sum_max = []

for t_avrg in LP_t_all:
    non_zero = [np.count_nonzero(t) for t in t_avrg]
    total_time = np.array([sum(t) for t in t_avrg])
    LP_non_zero_mean.append(np.mean(non_zero))
    LP_non_zero_min.append(LP_non_zero_mean[-1]-np.min(non_zero))
    LP_non_zero_max.append(np.max(non_zero)-LP_non_zero_mean[-1])
    LP_total_time_mean.append(np.mean(total_time))
    LP_total_time_min.append(LP_total_time_mean[-1]-np.min(total_time))
    LP_total_time_max.append(np.max(total_time)-LP_total_time_mean[-1])

for t_avrg in MIP_t_all:
    non_zero = [np.count_nonzero(t) for t in t_avrg]
    total_time = np.array([sum(t) for t in t_avrg])
    MIP_non_zero_mean.append(np.mean(non_zero))
    MIP_non_zero_min.append(MIP_non_zero_mean[-1]-np.min(non_zero))
    MIP_non_zero_max.append(np.max(non_zero)-MIP_non_zero_mean[-1])
    MIP_total_time_mean.append(np.mean(total_time))
    MIP_total_time_min.append(MIP_total_time_mean[-1]-np.min(total_time))
    MIP_total_time_max.append(np.max(total_time)-MIP_total_time_mean[-1])

for M_nz_avrg in M_nz_all:
    AA_nz_mean.append(np.mean(M_nz_avrg))
    AA_nz_min.append(AA_nz_mean[-1]-np.min(M_nz_avrg))
    AA_nz_max.append(np.max(M_nz_avrg)-AA_nz_mean[-1])
for M_sum_avrg in M_sum_all:
    AA_sum_mean.append(np.mean(M_sum_avrg))
    AA_sum_min.append(AA_sum_mean[-1]-np.min(M_sum_avrg))
    AA_sum_max.append(np.max(M_sum_avrg)-AA_sum_mean[-1])

## Generate Plots
#################

colors = cm.get_cmap('tab10', 10)

font_size = 9
line_width = 1
mark = ['x', "4", "+"]
markersize = [6, 9, 7]

plt.rcParams['font.size'] = font_size+2
plt.rcParams['axes.linewidth'] = line_width

## Plot total gate time
fig = plt.figure(figsize=(4, 4))
ax = fig.add_axes([0, 0, 1, 1])
sub_ax = fig.add_axes([0.15, 0.5, 0.4, 0.4])

sub_ax.errorbar(nr_of_qubits, LP_total_time_mean, yerr=np.vstack([LP_total_time_min, LP_total_time_max]),
                capsize=line_width * 4, linestyle="None", marker=mark[0], ms=markersize[0], color=colors(0), label='LP')
sub_ax.errorbar(nr_of_qubits, MIP_total_time_mean, yerr=np.vstack([MIP_total_time_min, MIP_total_time_max]),
                capsize=line_width * 4, linestyle="None", marker=mark[1], ms=markersize[1], color=colors(5), label='MIP')
sub_ax.plot(nr_of_qubits, max_hess, '--', label=r'$\max( \frac{1}{H_V^{-1}})$', color=colors(2))

ax.errorbar(nr_of_qubits, LP_total_time_mean, yerr=np.vstack([LP_total_time_min, LP_total_time_max]),
            capsize=line_width * 4, linestyle="None", marker=mark[0], ms=markersize[0], color=colors(0), label='LP')
ax.errorbar(nr_of_qubits, MIP_total_time_mean, yerr=np.vstack([MIP_total_time_min, MIP_total_time_max]),
            capsize=line_width * 4, linestyle="None", marker=mark[1], ms=markersize[1], color=colors(5), label='MIP')
ax.errorbar(nr_of_qubits, AA_sum_mean, yerr=np.vstack([AA_sum_min, AA_sum_max]),
            capsize=line_width * 4, linestyle="None", marker=mark[2], ms=markersize[2], color=colors(1), label='naive')
ax.plot(nr_of_qubits, max_hess, '--', label=r'$\max(1/J_{ij})$', color=colors(2))

ax.xaxis.set_tick_params(which='major', size=line_width*5, width=line_width, direction='in')
ax.yaxis.set_tick_params(which='major', size=line_width*5, width=line_width, direction='in')
ax.yaxis.set_tick_params(which='minor', size=line_width*3, width=line_width, direction='in')
ax.ticklabel_format(axis="y", style="sci")
sub_ax.ticklabel_format(axis="y", style="sci")
ax.set_ylabel('Total gate time [ms]', labelpad=5)
ax.set_xlabel('Nr. of participating qubits', labelpad=5)
ax.legend(bbox_to_anchor=(0.56, 0.95), loc=2, frameon=False, fontsize=font_size+1)

plt.savefig('MIP_Final_Gate_time.pdf', transparent=False, bbox_inches='tight')
plt.show()

## Plot encoding cost
fig = plt.figure(figsize=(4, 4))
ax = fig.add_axes([0, 0, 1, 1])

ax.errorbar(nr_of_qubits, MIP_non_zero_mean, yerr=np.vstack([MIP_non_zero_min, MIP_non_zero_max]),
            capsize=line_width * 4, linestyle="None", marker=mark[1], ms=markersize[1], color=colors(5), label='MIP')
ax.errorbar(nr_of_qubits, LP_non_zero_mean, yerr=np.vstack([LP_non_zero_min, LP_non_zero_max]),
            capsize=line_width * 4, linestyle="None", marker=mark[0], ms=markersize[0], color=colors(0), label='LP')
ax.errorbar(nr_of_qubits, AA_nz_mean, yerr=np.vstack([AA_nz_min, AA_nz_max]),
            capsize=line_width * 4, linestyle="None", marker=mark[2], ms=markersize[2], color=colors(1), label='naive')
ax.plot(nr_of_qubits, [n / 2.0 * (n - 1) for n in nr_of_qubits], '--', label=r'$\frac{n}{2} (n-1)$', color=colors(3))

ax.yaxis.set_tick_params(which='major', size=line_width*5, width=line_width, direction='in')
ax.yaxis.set_tick_params(which='minor', size=line_width*3, width=line_width, direction='in')
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylabel('Encoding cost', labelpad=5)
ax.set_xlabel('Nr. of participating qubits', labelpad=5)
ax.legend(bbox_to_anchor=(0.15, 0.9), loc=2, frameon=False, fontsize=font_size+1)

plt.savefig('MIP_Final_Gate_count.pdf', transparent=False, bbox_inches='tight')
plt.show()
