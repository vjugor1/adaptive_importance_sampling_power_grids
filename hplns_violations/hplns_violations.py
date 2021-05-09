# Import namespaces
import sys
import os
import math
import itertools
import json
import numpy as np
import scipy as scp
import matplotlib.pyplot as plt
import scipy.linalg as la
import pandapower as pp


from scipy.stats import norm
from scipy.stats import uniform
from scipy.stats import multinomial
from scipy.linalg import pinvh
from pandapower import networks

from tqdm.notebook import tqdm, trange
import time 

# load matrices from file

grid4 = pp.networks.case4gs();
grid6 = pp.networks.case6ww();
grid14 = pp.networks.case14(); # ok
grid30 = pp.networks.case_ieee30();# ok
grid118 = pp.networks.case118(); #? slack angle != 0
grid118i = pp.networks.iceland(); #ok
grid145 = pp.networks.case145();
grid200 = pp.networks.case_illinois200();# ok
grid300 = pp.networks.case300();
grid1354 = pp.networks.case1354pegase();
grid1888 = pp.networks.case1888rte(); # ok #slack angle != 0
grid2224 = pp.networks.GBnetwork(); # extremely high prob
grid2848 = pp.networks.case2848rte();
grid2869 = pp.networks.case2869pegase(); # high prob!
grid3120 = pp.networks.case3120sp(); #ok # feasibility check do not pass, check why
grid6470 = pp.networks.case6470rte(); #? slack angle != 0
grid6495 = pp.networks.case6495rte(); # multiple stacks
grid6515 = pp.networks.case6515rte(); #? slack angle != 0
grid9241 = pp.networks.case9241pegase(); # extremely high probability

grids = [(grid4, 'grid4'), (grid6, 'grid6'), (grid14, 'grid14'), (grid30, 'grid30'), (grid118, 'grid118'), (grid118i, 'grid118i'),        (grid145, 'grid145'), (grid200, 'grid200'), (grid300, 'grid300'), (grid1354, 'grid1354'), (grid1888, 'grid1888'), (grid2224, 'grid2224'), (grid2848, 'grid2848'), (grid2869, 'grid2869'), (grid3120, 'grid3120')]
#, (grid6470, 'grid6470'), (grid6495, 'grid6495'), (grid6515, 'grid6515'), (grid9241, 'grid9241')


def get_violation_number(grid3120, name):

    net = grid3120; 

    pp.rundcpp(net);
    ppc = net["_ppc"];

    ##### Setup Grid fluctuation parameters and constraints ########


    ## thresold on shift significance in DC-PF Eqs
    ## pwr = pwr_shf + np.real(bbus)*va
    ##

    shf_eps = 1e-4; 

    ## Std for fluctuating loads divided by their nominal values: 
    ## for small grids values 0.5-1 are realistic
    ## larger grids have cov_std = 0.1 -- 0.3 or less
    ##

    cov_std = 0.25

    ## Phase angle difference limit
    ## small grids: pi/8 -- pi/6
    ## large grids: pi/3 -- pi/4
    ##

    bnd = math.pi/4;

    ### Cut small probabilities threshold
    ### discard all probabilities than thrs* prb(closest hyperplane)
    ### 
    ###
    ### Crucially affects time performance
    ###

    thrs = 0.001

    ### Number of samples used in experiments
    ### 500 is often enough
    ### 10000 is a default value supresses the variance

    nsmp = 10000; 


    ### Step-sizes for KL and Var minimization 
    ### works well with 0.1-0.01

    eta_vm = 0.1; 
    eta_kl = 0.1; 

    ### Rounding threshold in optimization: 
    ### if a (normalized on the simplex) hpl probability becomes lower then 0.001
    ### we increase it to this level
    ###
    ### Crucially affects numerical stability
    ###

    eps = 0.001



    ##### Setup power grid case in a convenient form for further sampling ########

    ### find number of lines (m) and buses(n)

    m = net.line['to_bus'].size; 
    n = net.res_bus['p_mw'].size;

    ### Construct adjacency matrix 
    ###

    adj = np.zeros((2*m, n));
    for i in range(0, m):
        adj[i, net.line['to_bus'][i]] = 1;
        adj[i, net.line['from_bus'][i]] = -1;
        adj[i+m, net.line['to_bus'][i]] = -1;
        adj[i+m, net.line['from_bus'][i]] = 1;

    ### DC power flow equations have a form: 
    ###
    ### pwr = pwr_shf + np.real(bbus)*va
    ### (compute all parameters)

    bbus = np.real(ppc['internal']['Bbus']);
    va = math.pi*net.res_bus['va_degree']/180;
    pwr = - net.res_bus['p_mw'];
    pwr_shf =  pwr - bbus@va;

    ### pwr_shf is significant or not:
    ###
    ### if the shift is small: zero it out
    ### (simplifies testing and removes "math zeros")

    print("significant shift: ", np.max(pwr_shf) - np.min(pwr_shf) > shf_eps)
    if (np.max(pwr_shf) - np.min(pwr_shf) < shf_eps):
        pwr_shf[range(0,n)] = 0


    ### Phase angle differences:
    ###
    ### va = pinv(bbus)*(pwr - pwr_shf)
    ### va_d = adj*va = adj*pinv(bbus)*(pwr - pwr_shf) 
    ### va_d = pf_mat*pwr - va_shf

    bbus_pinv = pinvh(bbus.todense())
    pf_mat = adj@bbus_pinv;
    va_shf = pf_mat@pwr_shf;


    ### Voltage angle differences:
    ###

    va_d = pf_mat@pwr - va_shf;


    ##### Distribution of fluctuations ######

    ### assume the only one slack (a higher-level grid) in the grid
    ### supress all its fluctuations and balance the grid
    ###
    ### TODO: adjust to a general case
    ### 

    slck = net.ext_grid['bus'];  
    slck_mat = np.eye(n);
    slck_mat[slck] = -1; ## assign values to the whole array
    slck_mat[slck,slck] = 0; # and zero out for the slack itself


    ### set fluctuating components: either loads or gens or both
    ###

    loads = np.zeros(n);
    gens = np.zeros(n);
    ctrls = np.zeros(n); ## controllable loads + gens

    loads[net.load['bus']] = - net.res_load['p_mw']; 
    gens[net.gen['bus']] = net.res_gen['p_mw'];
    ctrls = loads + gens;

    ### assume only loads are fluctuating
    ###

    xi = loads;


    ### Set covariance matrix and mean
    ###
    ### cov_sq = square of the covariance matrix
    ### Gaussian rv with covariance \Sigma is \Sigma^{1/2} * std_normal_rv
    ###
    ### TODO: change to LU/cholesky factorization
    ### 

    cov_sq = cov_std*np.diag(np.abs(xi));



    ### Final equations with fluctuations xi are then 
    ###
    ### w/o fluctuations: 
    ### va_d = pf_mat*pwr - va_shf
    ### with fluctuations:
    ### va_d = pf_mat@(pwr + slck_mat*cov_sq*xi) - va_shf
    ### va_d = pf_mat@pwr - va_shf + (pf_mat@(slck_mat@cov_sq))@xi_std 
    ### va_d = mu + A@xi_std
    ### where xi_std is a standard normal with only fluctuating components
    ###

    A = (pf_mat@slck_mat)@cov_sq;
    mu = pf_mat@pwr - va_shf; 

    ### Feasibility Polytope Inequalities
    ### bnd \ge va_d = mu_f + A_f@xi_std
    ### incorporates both va_d \le b and va_d \ge -b as we have va_d's with 2 signs
    ###

    b = np.ones(2*m)*bnd; 

    ### normalize the matrices to make it easier to compute a failure probability
    ###

    ### compute row norms of A
    nrms = np.maximum(la.norm(A, axis = 1), 1e-20)

    ### normalize A and b so that b_n\ge A_n*xi_std
    b_n = (b-mu)/nrms;
    A_n = [A[i]/nrms[i] for i in range(0, 2*m)]


    ##### Assest equations feasibility #######

    ### Power balance check 
    ###

    print("Eqs balance check:", 0 == np.sum(np.sign(mu)))


    ### check positiveness of bnd - mu_f = RHS - LHS
    ###

    print("Inqs. feasibility check: ", np.min(b - mu) > 0)
    print("Min gap in phase angles = min(RHS - LHS)", np.min(b - mu)) ## positive value, otherwise the grid fails whp
    print("The RHS (phase angle diff max) = ", bnd);


    ### Compute probabilities:
    ### prb: probability of each hpl failure 
    ### p_up, p_dwn: upper and lower bounds
    ###

    prb = norm.cdf(-b_n);
    p_up = np.sum(prb);
    p_dwn = np.max(prb);

    print("the union bound (upper):", p_up)
    print("the max bound (lower):", p_dwn)

    ### Keep only valuable probabilities: 
    ### - use the union bound for all the rest
    ### - keep only the prbs higher than the thrs* p_dwn

    prbh_id = (prb>thrs*p_dwn)
    prb_rmd = np.sum(prb[~(prb>thrs*p_dwn)])

    print("Remainder probability (omitted):", prb_rmd)


    ############ Preliminary steps for Sampling and Importance Sampling ############

    ### normalize all active probabilities to one 
    ### as we only play a hyperplane out of them
    ###
    ### NB: crucial steps in performance optimization
    ###

    x_id = np.where(prbh_id == True)[0]

    ### local normalized versions of A and b, 
    ### reduced in size: number of rows now is equal to a number of constraints
    ### that have a high probability of violation
    ###

    x_bn = b_n[x_id]

    ### we do not care about the full matrix A and vector b
    ### only about important parts of them
    A_n = np.array(A_n)
    x_An = A_n[x_id]

    print("# hpls we care of: ", len(x_bn))

    ############# Monte-Carlo ##################

    rv = norm(); 
    x_std = norm.rvs(size=[n,nsmp])
    smp = x_An@x_std

    ### fls_mc = failures in Monte-Carlo, e.g. 
    ### when MC discovers a failure
    ###

    fls_mc = sum((x_bn <= smp.T[:]).T); 
    print("Max # of hlps a sample if out of: ", np.max(fls_mc))

    ### MC failure expectation and std
    ###

    mc_exp = (1 - np.sum(fls_mc == 0)/nsmp)*(1 - prb_rmd)+prb_rmd;
    mc_std = (1 - prb_rmd)/math.sqrt(nsmp);
    violation_dict = {}
    for i in range(0,np.max(fls_mc)+1):
        print(i, "hpls violated (exactly) vs # cases",  np.sum(fls_mc == i))
        violation_dict[i] = int(np.sum(fls_mc == i))

    print("\nMC(exp, std):", (mc_exp, mc_std)); 
    
    ## write into file
    #path_to_viol_dirs = os.path.join("results", "hplns_violations")
    
    violation_dict["Ineqs. Feasibility Check"] = bool(np.min(b - mu) > 0)
    violation_dict["Significant shift"] = bool(np.max(pwr_shf) - np.min(pwr_shf) > shf_eps)
    violation_dict["Min gap in phase angles = min(RHS - LHS)"] = float(np.min(b - mu))
    violation_dict["The RHS (phase angle diff max) = "] = float(bnd)
    violation_dict["Probability"] = mc_exp
    with open(name + ".json", 'w+') as fp:
        json.dump(violation_dict, fp)
    
s1 = time.time()
for el in grids:
    get_violation_number(el[0], el[1])
s2 = time.time()
print(s2 - s1)