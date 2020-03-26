//
//  E_step.cpp
//  synthetic_data_helper
//
//  Created by Cristián Garay on 11/16/16.
//  Revised and edited by Anirudhan Badrinath on 27/02/20.
//

#define NPY_NO_DEPRECATED_API NPY_1_11_API_VERSION

#include <iostream>
#include <stdint.h>
#include <alloca.h>
#include <Eigen/Core>
#include <omp.h>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <boost/python/ptr.hpp>
#include <Python.h>
#include <numpy/ndarrayobject.h>

using namespace Eigen;
using namespace std;
using namespace boost::python;

namespace p = boost::python;
namespace np = boost::python::numpy;

//original comment:
//"TODO if we aren't outputting gamma, don't need to write it to memory (just
//need t and t+1), so we can save the stack array for each HMM at the cost of
//a branch"

dict run(dict& data, dict& model, numpy::ndarray& trans_softcounts, numpy::ndarray& emission_softcounts, numpy::ndarray& init_softcounts, int num_outputs){
    //TODO: check if parameters are null.
    //TODO: check that dicts have the required members.
    //TODO: check that all parameters have the right sizes.
    //TODO: i'm not sending any error messages.
    IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");

    numpy::ndarray alldata = extract<numpy::ndarray>(data["data"]); //multidimensional array, so i need to keep extracting arrays.
    int bigT = len(alldata[0]); //this should be the number of columns in the alldata object. i'm assuming is 2d array.
    int num_subparts = len(alldata);

    numpy::ndarray allresources = extract<numpy::ndarray>(data["resources"]);

    cout << "ree";
    numpy::ndarray starts = extract<numpy::ndarray>(data["starts"]);

    int num_sequences = len(starts);

    numpy::ndarray lengths = extract<numpy::ndarray>(data["lengths"]);

    numpy::ndarray learns = extract<numpy::ndarray>(model["learns"]);
    int num_resources = len(learns);

    numpy::ndarray forgets = extract<numpy::ndarray>(model["forgets"]);

    numpy::ndarray guesses = extract<numpy::ndarray>(model["guesses"]);

    numpy::ndarray slips = extract<numpy::ndarray>(model["slips"]);

    double prior = extract<double>(model["prior"]);

    bool normalizeLengths = false;
    //then the original code goes to find the optional parameters.

    Array2d initial_distn;
    initial_distn << 1-prior, prior;

    MatrixXd As(2,2*num_resources);
    for (int n=0; n<num_resources; n++) {
        double learn = extract<double>(learns[n]);
        double forget = extract<double>(forgets[n]);
        As.col(2*n) << 1-learn, learn;
        As.col(2*n+1) << forget, 1-forget;
    }

    Array2Xd Bn(2,2*num_subparts);
    for (int n=0; n<num_subparts; n++) {
        double guess = extract<double>(guesses[n]);
        double slip = extract<double>(slips[n]);
        Bn.col(2*n) << 1-guess, slip; // incorrect
        Bn.col(2*n+1) << guess, 1-slip; // correct
    }

    //// outputs

    //TODO: NEED TO FIX THIS I'M CREATING NEW ARRAYS AND I NEED TO USE THE ARGUMENTS!!!
    //TODO: FIX THIS!!!
    /*Map<ArrayXXd,Aligned> all_trans_softcounts(trans_softcounts,2,2*num_resources);
    all_trans_softcounts.setZero();
    Map<Array2Xd,Aligned> all_emission_softcounts(emission_softcounts,2,2*num_subparts);
    all_emission_softcounts.setZero();
    Map<Array2d,Aligned> all_initial_softcounts(init_softcounts);
    all_initial_softcounts.setZero();*/
    //TODO: I replaced the pointers to the arguments for new Eigen arrays.
    /*ArrayXXd all_trans_softcounts(2,2*num_resources);
    all_trans_softcounts.setZero(); //why is he setting all these to zero???
    Array2Xd all_emission_softcounts(2,2*num_subparts);
    all_emission_softcounts.setZero();
    Array2d all_initial_softcounts(2, 1); //should i use these dimensions? the same as the original vector??
    all_initial_softcounts.setZero();*/
    double* r_trans_softcounts = (double*) malloc(2*2*num_resources * sizeof(double));
    double* r_emission_softcounts = (double*) malloc(2*2*num_subparts * sizeof(double));
    double* r_init_softcounts = (double*) malloc(2*1 * sizeof(double));;
    Map<ArrayXXd,Aligned> all_trans_softcounts(r_trans_softcounts,2,2*num_resources);
    all_trans_softcounts.setZero();
    Map<Array2Xd,Aligned> all_emission_softcounts(r_emission_softcounts,2,2*num_subparts);
    all_emission_softcounts.setZero();
    Map<Array2d,Aligned> all_initial_softcounts(r_init_softcounts);
    all_initial_softcounts.setZero();

    //TODO: FIX THIS!!! I'll replace all these weird arrays for zeroes ones.
    //Array2Xd likelihoods_out(2,bigT);
    //likelihoods_out.setZero();
    //Array2Xd gamma_out(2,bigT);
    //gamma_out.setZero();
    //Array2Xd alpha_out(2,bigT);
    //alpha_out.setZero();
    Map<Array2Xd,Aligned> alpha_out(NULL,2,bigT);
    double s_total_loglike = 0;
    double *total_loglike = &s_total_loglike;

    //TODO: FIX THIS!!! why is he doing this??
    /* switch (num_outputs)
    {
        case 4:
            plhs[3] = mxCreateDoubleMatrix(2,bigT,mxREAL);
            new (&likelihoods_out) Map<Array2Xd,Aligned>(mxGetPr(plhs[3]),2,bigT);
        case 3:
            plhs[2] = mxCreateDoubleMatrix(2,bigT,mxREAL);
            new (&gamma_out) Map<Array2Xd,Aligned>(mxGetPr(plhs[2]),2,bigT);
        case 2:
            plhs[1] = mxCreateDoubleMatrix(2,bigT,mxREAL);
            new (&alpha_out) Map<Array2Xd,Aligned>(mxGetPr(plhs[1]),2,bigT);
        case 1:
            plhs[0] = mxCreateDoubleScalar(0.);
            total_loglike = mxGetPr(plhs[0]);
    }*/
    double* r_alpha_out = (double*) malloc(2 * bigT * sizeof(double));
    new (&alpha_out) Map<Array2Xd,Aligned>(r_alpha_out,2,bigT);

             cout << "ree";

    /* COMPUTATION */
    #pragma omp parallel
    {
             cout << "ree";
        double s_trans_softcounts[2*2*num_resources] __attribute__((aligned(16)));
        double s_emission_softcounts[2*2*num_subparts] __attribute__((aligned(16)));
        Map<ArrayXXd,Aligned> trans_softcounts_temp(s_trans_softcounts,2,2*num_resources);
        Map<ArrayXXd,Aligned> emission_softcounts_temp(s_emission_softcounts,2,2*num_subparts);
        Array2d init_softcounts_temp;
        double loglike;

        trans_softcounts_temp.setZero();
        emission_softcounts_temp.setZero();
        init_softcounts_temp.setZero();
        loglike = 0;
        int num_threads = omp_get_num_threads();
        int blocklen = 1 + ((num_sequences - 1) / num_threads);
        int sequence_idx_start = blocklen * omp_get_thread_num();
        int sequence_idx_end = min(sequence_idx_start+blocklen,num_sequences);
        //mexPrintf("start:%d   end:%d\n", sequence_idx_start, sequence_idx_end);

        for (int sequence_index=sequence_idx_start; sequence_index < sequence_idx_end; sequence_index++) {

             cout << "ree";
            // NOTE: -1 because Matlab indexing starts at 1
            int64_t sequence_start = extract<int64_t>(starts[sequence_index]) - 1;

            int64_t T = extract<int64_t>(lengths[sequence_index]);

            //// likelihoods
            double * s_likelihoods = new double[2*T];
             cout << "ree";
            Map<Array2Xd,Aligned> likelihoods(s_likelihoods,2,T);

             cout << "ree";
            likelihoods.setOnes();
             cout << "ree";
             for (int t=0; t<T; t++) {
                 for (int n=0; n<num_subparts; n++) {
                    int32_t data_temp = extract<int32_t>(alldata[n][sequence_start+t]);
                     if (data_temp != 0) {
                        likelihoods.col(t) *= Bn.col(2*n + (data_temp == 2));
                     }
             cout << "ree";
                 }
             }
             cout << "ree";

            //// forward messages
            double norm;
            double s_alpha[2*T] __attribute__((aligned(16)));
            double contribution;
            Map<MatrixXd,Aligned> alpha(s_alpha,2,T);
            alpha.col(0) = initial_distn * likelihoods.col(0);
            norm = alpha.col(0).sum();
            alpha.col(0) /= norm;
            loglike += log(norm) / (normalizeLengths? T : 1);

            for (int t=0; t<T-1; t++) {
                int64_t resources_temp = extract<int64_t>(allresources[sequence_start+t]);
                alpha.col(t+1) = (As.block(0,2*(resources_temp-1),2,2) * alpha.col(t)).array()
                    * likelihoods.col(t+1);
                norm = alpha.col(t+1).sum();
                alpha.col(t+1) /= norm;
                loglike += log(norm) / (normalizeLengths? T : 1);
            }

            //// backward messages and statistic counting

            double s_gamma[2*T] __attribute__((aligned(16)));
            Map<Array2Xd,Aligned> gamma(s_gamma,2,T);
            gamma.col(T-1) = alpha.col(T-1);
            for (int n=0; n<num_subparts; n++) {
                int32_t data_temp = extract<int32_t>(alldata[n][sequence_start+(T-1)]);
                if (data_temp != 0) {
                    emission_softcounts_temp.col(2*n + (data_temp == 2)) += gamma.col(T-1);
                }
            }

            for (int t=T-2; t>=0; t--) {

				int64_t resources_temp = extract<int64_t>(allresources[sequence_start+t]);
                Matrix2d A = As.block(0,2*(resources_temp-1),2,2);
                Array22d pair = A.array();
                pair.rowwise() *= alpha.col(t).transpose().array();
                pair.colwise() *= gamma.col(t+1);
                pair.colwise() /= (A*alpha.col(t)).array();
                pair = (pair != pair).select(0.,pair); // NOTE: replace NaNs
                trans_softcounts_temp.block(0,2*(resources_temp-1),2,2) += pair;

                gamma.col(t) = pair.colwise().sum().transpose();
                // NOTE: we have to touch the data again here
                for (int n=0; n<num_subparts; n++) {
                    int32_t data_temp = extract<int32_t>(alldata[n][sequence_start+t]);
                    if (data_temp != 0) {
                        emission_softcounts_temp.col(2*n + (data_temp == 2)) += gamma.col(t);
                    }
                }
            }
            init_softcounts_temp += gamma.col(0);

            //TODO: FIX THIS!!!
            /* switch (nlhs)
            {
                case 4:
                    likelihoods_out.block(0,sequence_start,2,T) = likelihoods;
                case 3:
                    gamma_out.block(0,sequence_start,2,T) = gamma;
                case 2:
                    alpha_out.block(0,sequence_start,2,T) = alpha;
            } */
            //alpha_out.block(0,sequence_start,2,T) = alpha;
        }
        #pragma omp critical
        {
            all_trans_softcounts += trans_softcounts_temp;
            all_emission_softcounts += emission_softcounts_temp;
            all_initial_softcounts += init_softcounts_temp;
            *total_loglike += loglike;
        }
    }
    dict result;
    result["total_loglike"] = *total_loglike;


    numpy::ndarray all_trans_softcounts_arr = numpy::from_data(r_trans_softcounts, numpy::dtype::get_builtin<double>(), boost::python::make_tuple(num_resources, 2, 2),
                                                               boost::python::make_tuple(32, 16, 8), boost::python::object());
    result["all_trans_softcounts"] = all_trans_softcounts_arr;


    numpy::ndarray all_emission_softcounts_arr = numpy::from_data(r_emission_softcounts, numpy::dtype::get_builtin<double>(), boost::python::make_tuple(num_subparts, 2, 2),
                                                               boost::python::make_tuple(32, 16, 8), boost::python::object());
    result["all_emission_softcounts"] = all_emission_softcounts_arr;

    numpy::ndarray all_initial_softcounts_arr = numpy::from_data(r_init_softcounts, numpy::dtype::get_builtin<double>(), boost::python::make_tuple(2, 1),
                                                               boost::python::make_tuple(8, 8), boost::python::object());
    result["all_initial_softcounts"] = all_initial_softcounts_arr;

    numpy::ndarray alpha_out_arr = numpy::from_data(r_alpha_out, numpy::dtype::get_builtin<double>(), boost::python::make_tuple(2, bigT),
                                                               boost::python::make_tuple(bigT * 8, 8), boost::python::object());
    result["alpha_out"] = alpha_out_arr;

    return(result);
}


BOOST_PYTHON_MODULE(E_step){
    Py_Initialize();
    numpy::initialize();
    def("run", run);
}
