from .. import ParallelSampler
from . import moped
from ...datablock import BlockError
import numpy as np
import scipy.linalg
from ...runtime import prior, utils, logs
import sys
from astropy.io import fits
from astropy.table import Table

def compute_moped_vector(p, cov=False):
    # use normalized parameters - mopedPipeline is a global
    # variable because it has to be picklable)
    try:
        x = mopedPipeline.denormalize_vector(p)
    except ValueError:
        logs.error("Parameter vector outside limits: %r" % p)
        return None

    #Run the pipeline, generating a data block
    data = mopedPipeline.run_parameters(x)

    #If the pipeline failed, return "None"
    #This might happen if the parameters stray into
    #a bad region.
    if data is None:
        return None

    #Get out the moped vector.  Failing on this is definitely an error
    #since if the pipeline finishes it must have a moped vector if it
    #has been acceptably designed.
    v = []
    for like_name in mopedPipeline.likelihood_names:
        v.append(data["data_vector", like_name + "_theory"])

    v = np.concatenate(v)
    #Might be only length-one, conceivably, so convert to a vector
    v = np.atleast_1d(v)

    # If we don't need the cov mat for this run just return now
    if not cov:
        return v

    # Otherwise calculate the covmat too.
    M = []
    for like_name in mopedPipeline.likelihood_names:
        M.append(data["data_vector", like_name + "_inverse_covariance"])

    M = scipy.linalg.block_diag(*M)
    M = np.atleast_2d(M)

    #Return numpy vector
    return v, M

# def collect_data_vector(p):
#     # use normalized parameters - mopedPipeline is a global
#     # variable because it has to be picklable)
#     try:
#         x = mopedPipeline.denormalize_vector(p)
#     except ValueError:
#         logs.error("Parameter vector outside limits: %r" % p)
#         return None
#
#     #Run the pipeline, generating a data block
#     data = mopedPipeline.run_parameters(x)
#
#     #If the pipeline failed, return "None"
#     #This might happen if the parameters stray into
#     #a bad region.
#     if data is None:
#         return None
#
#     v = []
#     for like_name in mopedPipeline.likelihood_names:
#         v.append(data["data_vector", like_name + "_data"])
#
#     v = np.concatenate(v)
#     #Might be only length-one, conceivably, so convert to a vector
#     v = np.atleast_1d(v)
#
#     return v


class SingleProcessPool(object):
    def map(self, function, tasks):
        return list(map(function, tasks))

class MOPEDSampler(ParallelSampler):
    sampler_outputs = []
    parallel_output = False
    understands_fast_subspaces = True

    def config(self):
        #Save the pipeline as a global variable so it
        #works okay with MPI
        global mopedPipeline
        mopedPipeline = self.pipeline
        self.step_size = self.read_ini("step_size", float, 0.01)
        self.tolerance = self.read_ini("tolerance", float, 0.01)
        self.maxiter = self.read_ini("maxiter", int, 10)
        self.use_numdifftools = self.read_ini("use_numdifftools", bool, False)
        self.set_params_ordering()
        # output to fits file
        # self.filename = self.read_ini("filename", str, "")
        # self.moped_name = self.read_ini("name", str, "")
        # self.overwrite = self.read_ini("overwrite", bool, False)
        # if self.filename:
        #     assert len(self.moped_name) > 0, \
        #             "You must specify a MOPED name in the parameter file. " \
        #             "Use 'name' option."

        if self.output:
            for p in self.pipeline.extra_saves:
                name = '%s--%s'%p
                logs.warning("NOTE: You set extra_output to include parameter %s in the parameter file" % name)
                logs.warning("      But the MOPED Sampler cannot do that, so this will be ignored.")
                self.output.del_column(name)
        
        # replace the header with ordered params
        for p in self.pipeline.varied_params:
            self.output.del_column('%s--%s'%(p.section, p.name))
        columns = []
        for ordered_name in self.ordered_names:
            columns.append([ordered_name, float, ""])
        self.output.columns = columns + self.output.columns

        self.converged = False

    def set_params_ordering(self):
        # param_names = self.read_ini('ordering')
        param_names = self.read_ini('ordering', str).split()

        # list of varied parameters' names
        parameters = [(p.section, p.name) for p in self.pipeline.varied_params]

        ordering = []
        ordered_names = []

        # put the parameters' indices as in user input
        for param_name in param_names:
            section, name = param_name.split('--')
            i = parameters.index((section.lower(), name.lower()))
            ordering.append(i)
            ordered_names.append(param_name)

        # append unspecified parameters' index as in the pipeline
        for i, param in enumerate(parameters):
            if i not in ordering:
                ordering.append(i)
                ordered_names.append('--'.join(param))

        self.ordering = np.array(ordering)
        self.ordered_names = ordered_names

    def get_param_norm(self):
        n = len(self.pipeline.varied_params)
        norm = np.zeros(n)
        for i in range(n):
            p = self.pipeline.varied_params[i]
            norm[i] = p.limits[1] - p.limits[0]
        return norm

    def execute(self):
        #Load the starting point and covariance matrix
        #in the normalized space, either from the values
        #file or a previous sampler
        start_vector = self.start_estimate()

        if len(self.pipeline.varied_params)==0:
            raise ValueError("Your values file did not include any varied parameters so we cannot make a MOPED matrix")

        # We save the values for the ordered params
        for i0, i in enumerate(self.ordering):
            x = start_vector[i]
            self.output.metadata("mu_{0}".format(i0), x)
            
        start_vector = self.pipeline.normalize_vector(start_vector)

        #calculate the moped matrix.
        #right now just a single step
        if self.use_numdifftools:
            moped_class = moped.NumDiffToolsMOPED
        else:
            moped_class = moped.MOPED
        moped_calc = moped_class(compute_moped_vector, start_vector, 
            self.step_size, self.ordering, self.tolerance, self.maxiter, pool=self.pool)

        # Obtain the parameter normalization factor from pipeline
        # Because the derivative is given for normalized param,
        # we denormalize each param.
        param_norm = self.get_param_norm()

        try:
            moped_matrix = moped_calc.compute_moped_matrix(param_norm=param_norm)
        except moped.MOPEDParameterError as error:
            param = str(self.pipeline.varied_params[error.parameter_index])
            if error.parameter_index==0:
                raise ValueError(f"""
There was an error running the pipeline for the MOPED Matrix for parameter:
{param}
Since this is the first parameter this might indicate a general error in the pipeline.
You might want to check with the "test" sampler.

It might also indicate that the parameter lower or upper limit is too close to its
starting value so the points used to calculate the derivative are outside the range.
If that is the case you should try calculating the MOPED Matrix at a different starting point.
""")
            else:
                raise ValueError(f"""
There was an error running the pipeline for the MOPED Matrix for parameter:
{param}

This probably indicates that the parameter lower or upper limit is too close to its
starting value, so the points used to calculate the derivative are outside the range.
If that is the case you should try calculating the MOPED Matrix at a different starting point.
""")

        self.converged = True

        if self.converged:
            for row in moped_matrix.T:
                self.output.parameters(row)

            # if self.filename:
            #     hdul = fits.HDUList([fits.PrimaryHDU()])
            #     # Table: Moped compressed data
            #     data  = np.dot(moped_matrix, collect_data_vector(start_vector))
            #     table = Table([data], names=['moped'])
            #     hdul.append(fits.BinTableHDU(table, name='MOPED-DATA-{}'.format(self.moped_name)))
            #     # mattrix
            #     hdul.append(fits.ImageHDU(moped_matrix, name='MOPED-TRANSFORM-{}'.format(self.moped_name)))
            #     hdul.writeto(self.filename, overwrite=self.overwrite)
        
    def is_converged(self):
        return self.converged
