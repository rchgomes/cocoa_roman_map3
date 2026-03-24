import numpy as np
import pdb

'''
NOTE: Need to recompute the spectra between each iteration (EUGH).

Steps:
    - For a given parameter, choose 4 even variations withing 1 sigma contour (i.e. this will probably be an input)
    - Calculate MOPED as dClobs/dparm
        - Calculate derivative with 5-point stencil method http://en.wikipedia.org/wiki/Five-point_stencil
    - Find the new 1 sigma contour according to this MOPED
    - Iterate until contours converge to some tolerance -- 0.5% as default

Inputs:
    - function to calculate vector of Clobs
Outpus:
    - Converged MOPED matrix

Write to parallelise
will be passed a pool with map function can pool.map(params) will give list of results
make list of params, run all at once, will parallelise nicely
be aware that some of the workers in pool might fail, return e.g. None
can have all paramers normalised to [0,1]

compute_vector(p) runs an entire pipeline on a set of parameters (i.e. needs to be done five times here) and returns vector, covmat where vector is Cl_*** and covmat is the necessary **INVERSE** covmat

can do pool.map(compute_vector, [p1,p2 etc]) where p1 is an array of the parameter values

len(start_vector) will give you the number of parameters we are varying, which is taken from the cosmosis values.ini file

'''

class MOPEDParameterError(Exception):
    def __init__(self, parameter_index):
        message = "MOPED Matrix likelihood function returned None for parameter: {}".format(parameter_index)
        super(Exception,self).__init__(self, message)
        self.parameter_index = parameter_index


class MOPED(object):
    def __init__(self, compute_vector, start_vector, step_size, ordering, tolerance, maxiter, pool=None):
        
        self.compute_vector = compute_vector
        self.maxiter = maxiter
        self.step_size = step_size
        self.start_params = start_vector
        self.ordering = ordering
        self.current_params = start_vector
        self.nparams = start_vector.shape[0]
        self.iterations = 0
        self.pool = pool

    def converged(self):
        crit = (abs(self.new_onesigma - self.old_onesigma).max() < self.threshold)
        return crit

    def converge_moped_matrix(self):
        
        self.new_Fmatrix = compute_moped_matrix(self.compute_vector,
            self.start_vector, self.start_covmat)
        
        self.old_onesigma = compute_one_sigma(new_Fmatrix)

        while True:
            self.iterations+=1
            self.old_onesigma = self.new_onesigma
            self.current_params = self.choose_new_params(self.new_Fmatrix)

            self.new_Fmatrix = self.compute_moped_matrix()

            self.new_onesigma = compute_one_sigma(self.new_Fmatrix)

            if self.converged():
                print('MOPED has converged!')
                return new_Fmatrix

            if self.iterations > self.maxiter:
                print("Run out of iterations.")
                print("Done %d, max allowed %d" % (self.iterations, self.maxiter))
                return None

    def compute_derivatives(self):
        derivatives = []
        points = []

        #To improve parallelization we first gather all the data points
        #we use in all the dimensions
        for p in range(self.nparams):
            points +=  self.five_points_stencil_points(p)
        print("Calculating derivatives using {} total models".format(len(points)))
        if self.pool is None:
            results = list(map(self.compute_vector, points))
        else:
            results = self.pool.map(self.compute_vector, points)

        # Bit of a waste of time to compute the inv cov separately,
        # but it's a quick fix to a memory error if we compute the cov
        # for every single value
        _, inv_cov = self.compute_vector(points[0], cov=True)

        #Now get out the results that correspond to each dimension
        for p in range(self.nparams):
            results_p = results[4*p:4*(p+1)]
            derivative = self.five_point_stencil_deriv(results_p, p)
            derivatives.append(derivative)
        derivatives = np.array(derivatives)
        return derivatives, inv_cov


    def compute_moped_matrix(self, param_norm=None):
        derivatives, inv_cov = self.compute_derivatives()

        # denormalize the parameter
        if param_norm is not None:
            assert derivatives.shape[0] == param_norm.size, 'norm lensgth does not match with derivatives shape'
            derivatives/= param_norm[:,None]

        if not np.allclose(inv_cov, inv_cov.T):
            print("WARNING: The inverse covariance matrix produced by your pipeline")
            print("         is not symmetric. This probably indicates a mistake somewhere.")
            print("         If you are only using cosmosis-standard-library likelihoods please ")
            print("         open an issue about this on the cosmosis site.")
        # moped_matrix = np.einsum("il,lk,jk->ij", derivatives, inv_cov, derivatives)
        # return moped_matrix

        # order the parameters
        derivatives = derivatives[self.ordering, :]

        # Compute moped transformation matrix
        nparam, ndata = derivatives.shape
        B = np.zeros((nparam, ndata))

        cov = np.linalg.inv(inv_cov)
        for m in range(nparam):
            # Here we compute the MOPED matrix based on 
            # Eq.14 of https://arxiv.org/pdf/astro-ph/9911102
            # 
            # In that paper, the definition of b_1(Eq.11) and 
            # b_m(Eq.14) are different. However, we can use Eq.14
            # in a programming manner, we initially have b_m=0 for 
            # all m. Then we update b_m for each m, which holds the
            # expression of Eq.14.
            #
            # projection of mu_m vector to b_q vector, dim = (nparam)
            mum_bq = np.einsum('i,qi->q', derivatives[m,:], B)
            # numerator
            bm = np.dot(inv_cov, derivatives[m,:])
            bm-= np.einsum('q,qi->i', mum_bq, B)
            # denominator: we compute the normalization 
            # directly from unnormalized bm, instead of following 
            # the Eq.14 to avoid numerical instability.

            # norm = np.dot(derivatives[m,:], np.dot(inv_cov, derivatives[m,:]))
            # norm-= np.sum(mum_bq**2)
            # norm = norm**0.5

            bm/= np.abs(np.einsum('i,ij,j->', bm, cov, bm))**0.5
            # update matrix
            B[m,:] = bm

        # Orthogonarity and normalization check
        u = np.einsum('mi,ij,nj->mn', B, np.linalg.inv(inv_cov), B)
        print('Orthogonarity check:')
        print(u)

        return B

    def five_points_stencil_points(self, param_index):
        delta = np.zeros(self.nparams)
        delta[param_index] = 1.0
        points = [self.current_params + x*delta for x in 
            [2*self.step_size, 
             1*self.step_size, 
            -1*self.step_size, 
            -2*self.step_size]
        ]
        return points        

    def five_point_stencil_deriv(self, obs, param_index):
        for r in obs:
            if r is None:
                raise MOPEDParameterError(param_index)
        deriv = (-obs[0] + 8*obs[1] - 8*obs[2] + obs[3])/(12*self.step_size)
        return deriv

    def compute_one_sigma(Fmatrix):
        sigma = np.sqrt(np.linalg.inv(Fmatrix))
        return sigma

class NumDiffToolsMOPED(MOPED):
    def compute_derivatives(self):
        import numdifftools as nd
        def wrapper(param_vector):
            print("Running pipeline:", param_vector)
            return self.compute_vector(param_vector, cov=False)
        jacobian_calculator = nd.Jacobian(wrapper, step=self.step_size)
        derivatives = jacobian_calculator(self.current_params)
        _, inv_cov = self.compute_vector(self.current_params, cov=True)
        print(derivatives.shape, inv_cov.shape)
        return derivatives.T, inv_cov
    


def test():
    def theory_prediction(x, cov=False):
        #same number of data points as parameters here
        x = np.concatenate([x,x])
        theory = 2*x + 2
        inv_cov = np.diag(np.ones_like(x)**-1)
        if cov:
            return theory, inv_cov
        else:
            return theory

    best_fit_params = np.array([0.1, 1.0, 2.0, 4.0,])
    ordering = np.arange(4).astype(int)
    moped_calculator = MOPED(theory_prediction, best_fit_params, 0.01, ordering, 0.0, 100)
    F = moped_calculator.compute_moped_matrix()
    print(F)
    return F

if __name__ == '__main__':
    test()
