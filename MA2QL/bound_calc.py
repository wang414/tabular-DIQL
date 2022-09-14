import numpy as np

def bound(alpha,gamma,epsilon,r_max):
    M = 2*r_max / (1 - gamma)
    term1 = np.log(epsilon * (1 - alpha * gamma))
    term2 = np.log(M + 2*epsilon)
    term3 = np.log(alpha) + np.log(gamma)
    ret_b = (term1 - term2) / term3
    print('alpha = {}, gamma = {}, epsilon = {}, r_max = {}'.format(alpha,gamma,epsilon,r_max))
    print('M = {} term1 = {}, term2 = {}, term3 = {}'.format(M,term1,term2,term3))
    print('bound = {}'.format(ret_b))


bound(0.01,0.99,0.01,20)
