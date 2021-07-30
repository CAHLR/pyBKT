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
#include <Python.h>
#include <numpy/ndarrayobject.h>

using namespace Eigen;
using namespace std;

//original comment:
//"TODO if we aren't outputting gamma, don't need to write it to memory (just
//need t and t+1), so we can save the stack array for each HMM at the cost of
//a branch"
//

static double extract_double(PyArrayObject *arr, int i) {
    return ((double*) PyArray_DATA(arr))[i];
}

void capsule_cleanup(PyObject *capsule) {
    void *memory = PyCapsule_GetPointer(capsule, NULL);
    delete memory;
}

static PyObject* run(PyObject * module, PyObject * args) {
    //TODO: check if parameters are null.
    //TODO: check that dicts have the required members.
    //TODO: check that all parameters have the right sizes.
    //TODO: i'm not sending any error messages.
    IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
    Eigen::initParallel();    
    import_array();

    PyObject *data_ptr = NULL, *model_ptr = NULL, *t_softcounts = NULL, *e_softcounts = NULL, *i_softcounts = NULL, *fixed = NULL;
    PyArrayObject *t_softcounts_np = NULL, *e_softcounts_np = NULL, *i_softcounts_np = NULL,
                  *alldata = NULL, *allresources = NULL, *starts = NULL, *lengths = NULL, *learns = NULL, *forgets = NULL, *guesses = NULL, *slips = NULL;   // Extended Numpy/C API
    int num_outputs, parallel;
    double prior;
    // dict& data, dict& model, numpy::ndarray& trans_softcounts, numpy::ndarray& emission_softcounts, numpy::ndarray& init_softcounts, int num_outputs

    // "O" format -> read argument as a PyObject type into argy (Python/C API)
    if (!PyArg_ParseTuple(args, "OOiiO", &data_ptr, &model_ptr, &num_outputs, &parallel, &fixed)) {
        PyErr_SetString(PyExc_ValueError, "Error parsing arguments.");
        return NULL;
    }

    if (!parallel)
        omp_set_num_threads(1);

    // Load all the numpy arrays in data & model
    char* DM_NAMES[] = {"data", "resources", "starts", "lengths", "learns", "forgets", "guesses", "slips"};
    PyArrayObject** DM_PTRS[] = {&alldata, &allresources, &starts, &lengths, &learns, &forgets, &guesses, &slips};
    for (int i = 0; i < 8; i++) {
        PyObject *dp = PyDict_GetItemString(i < 4 ? data_ptr : model_ptr, DM_NAMES[i]);
        int DTYPE = PyArray_ObjectType(dp, (i < 4 ? (i < 1 ? NPY_INT : NPY_INT64) : NPY_FLOAT)); // hack to force correct type
        *DM_PTRS[i] = (PyArrayObject *)PyArray_FROM_OTF(dp, DTYPE, NPY_ARRAY_IN_ARRAY);
    }
    prior = PyFloat_AsDouble(PyDict_GetItemString(model_ptr, "prior"));

    int bigT = (int) PyArray_DIM(alldata, 1), num_subparts = (int) PyArray_DIM(alldata, 0);
    int len_allresources = (int) PyArray_DIM(allresources, 0);
    int num_sequences = (int) PyArray_DIM(starts, 0);
    int len_lengths = (int) PyArray_DIM(lengths, 0);
    int num_resources = (int) PyArray_DIM(learns, 0);

    Map<Array<int32_t,Dynamic,Dynamic,RowMajor>,Aligned> alldata_arr(reinterpret_cast<int*>(PyArray_DATA(alldata)),num_subparts,bigT);
    Map<Array<int64_t, Eigen::Dynamic, 1>,Aligned> allresources_arr(reinterpret_cast<int64_t*>(PyArray_DATA(allresources)),len_allresources,1);
    Map<Array<int64_t, Eigen::Dynamic, 1>,Aligned> starts_arr(reinterpret_cast<int64_t*>(PyArray_DATA(starts)),num_sequences,1);
    Map<Array<int64_t, Eigen::Dynamic, 1>,Aligned> lengths_arr(reinterpret_cast<int64_t*>(PyArray_DATA(lengths)),len_lengths,1);

    bool normalizeLengths = false;
    //then the original code goes to find the optional parameters.

    PyObject* fixed_prior = PyDict_GetItemString(fixed, "prior");
    PyObject* fixed_learn = (PyObject*)PyDict_GetItemString(fixed, "learn");
    PyObject* fixed_forget = (PyObject*)PyDict_GetItemString(fixed, "forget");
    PyObject* fixed_guess = (PyObject*)PyDict_GetItemString(fixed, "guess");
    PyObject* fixed_slip = (PyObject*)PyDict_GetItemString(fixed, "slip");

    if (fixed_prior) {
        prior = PyFloat_AsDouble(fixed_prior);
    }
    Array2d initial_distn;
    initial_distn << 1-prior, prior;
    
    MatrixXd As(2,2*num_resources);
    double learn = -1;
    double forget = -1;
    
    for (int n=0; n<num_resources; n++) {
        if (fixed_learn) {
            learn = extract_double((PyArrayObject*)fixed_learn, n);
        }
        if (learn < 0) {
            learn = extract_double(learns, n);
        }
        if (fixed_forget) {
            forget = extract_double((PyArrayObject*)fixed_forget, n);
        }
        if (forget < 0) {
            forget = extract_double(forgets, n);
        }
        As.col(2*n) << 1-learn, learn;
        As.col(2*n+1) << forget, 1-forget;
    }
    
        
    Array2Xd Bn(2,2*num_subparts);
    double guess = -1;
    double slip = -1;
    for (int n=0; n<num_subparts; n++) {
        if (fixed_guess) {
            guess = extract_double((PyArrayObject*)fixed_guess, n);
        }
        if (guess < 0) {
            guess = extract_double(guesses, n);
        }
        if (fixed_slip) {
            slip = extract_double((PyArrayObject*)fixed_slip, n);
        }
        if (slip < 0) {
            slip = extract_double(slips, n);
        }
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
    double* r_trans_softcounts = new double[2*2*num_resources];
    double* r_emission_softcounts = new double[2*2*num_subparts];
    double* r_init_softcounts = new double[2*1];
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
    double* r_alpha_out = new double[2 * bigT];
    new (&alpha_out) Map<Array2Xd,Aligned>(r_alpha_out,2,bigT);

    /* COMPUTATION */
    #pragma omp parallel
    {
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

            // NOTE: -1 because Matlab indexing starts at 1
            int64_t sequence_start = starts_arr(sequence_index, 0) - 1;

            int64_t T = lengths_arr(sequence_index, 0);

            //// likelihoods
            double s_likelihoods[2*T];
            Map<Array2Xd,Aligned> likelihoods(s_likelihoods,2,T);

            likelihoods.setOnes();
             for (int t=0; t<T; t++) {
                 for (int n=0; n<num_subparts; n++) {
                    int32_t data_temp = alldata_arr(n, sequence_start+t);
                    if (data_temp != 0) {
                        for (int i = 0; i < likelihoods.rows(); i++)
                            if (Bn(i, 2*n + (data_temp == 2)) != 0)
                                likelihoods(i, t) *= Bn(i, 2*n + (data_temp == 2));
                    }
                 }
             }


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
                int64_t resources_temp = allresources_arr(sequence_start+t, 0);
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
                int32_t data_temp = alldata_arr(n, sequence_start+(T-1));
                if (data_temp != 0) {
                    emission_softcounts_temp.col(2*n + (data_temp == 2)) += gamma.col(T-1);
                }
            }

            for (int t=T-2; t>=0; t--) {

				int64_t resources_temp = allresources_arr(sequence_start+t, 0);
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
                    int32_t data_temp = alldata_arr(n, sequence_start+t);
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
            alpha_out.block(0,sequence_start,2,T) = alpha;
        }
        #pragma omp critical
        {
            all_trans_softcounts += trans_softcounts_temp;
            all_emission_softcounts += emission_softcounts_temp;
            all_initial_softcounts += init_softcounts_temp;
            *total_loglike += loglike;
        }
    }

    PyObject *result = PyDict_New();

    npy_intp dims1[] = {num_resources, 2, 2};
    PyObject *all_trans_softcounts_arr = (PyObject *) PyArray_SimpleNewFromData(3, dims1, NPY_DOUBLE, r_trans_softcounts);
    PyObject *capsule1 = PyCapsule_New(r_trans_softcounts, NULL, capsule_cleanup);
    PyArray_SetBaseObject((PyArrayObject *) all_trans_softcounts_arr, capsule1);

    npy_intp dims2[] = {num_subparts, 2, 2};
    PyObject *all_emission_softcounts_arr = (PyObject *) PyArray_SimpleNewFromData(3, dims2, NPY_DOUBLE, r_emission_softcounts);
    PyObject *capsule2 = PyCapsule_New(r_emission_softcounts, NULL, capsule_cleanup);
    PyArray_SetBaseObject((PyArrayObject *) all_emission_softcounts_arr, capsule2);

    npy_intp dims3[] = {2, 1};
    PyObject *all_initial_softcounts_arr = (PyObject *) PyArray_SimpleNewFromData(2, dims3, NPY_DOUBLE, r_init_softcounts);
    PyObject *capsule3 = PyCapsule_New(r_init_softcounts, NULL, capsule_cleanup);
    PyArray_SetBaseObject((PyArrayObject *) all_initial_softcounts_arr, capsule3);

    npy_intp dims4[] = {2, bigT};
    PyObject *alpha_out_arr = (PyObject *) PyArray_SimpleNewFromData(2, dims4, NPY_DOUBLE, r_alpha_out);
    PyObject *capsule4 = PyCapsule_New(r_alpha_out, NULL, capsule_cleanup);
    PyArray_SetBaseObject((PyArrayObject *) alpha_out_arr, capsule4);

    PyDict_SetItemString(result, "all_trans_softcounts", all_trans_softcounts_arr);
    PyDict_SetItemString(result, "all_emission_softcounts", all_emission_softcounts_arr);
    PyDict_SetItemString(result, "all_initial_softcounts", all_initial_softcounts_arr);
    PyDict_SetItemString(result, "alpha", alpha_out_arr);
    PyDict_SetItemString(result, "total_loglike", PyLong_FromLong(*total_loglike));

    for (int i = 0; i < 8; i++)
        Py_XDECREF(*DM_PTRS[i]);
    Py_XDECREF(all_trans_softcounts_arr);
    Py_XDECREF(all_emission_softcounts_arr);
    Py_XDECREF(all_initial_softcounts_arr);
    Py_XDECREF(alpha_out_arr);

    return(result);
}



static PyMethodDef E_step_Methods[] = {
    {"run",  run, METH_VARARGS,
     "Runs E-step of Expectation Maximization in C++ module"},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};


static struct PyModuleDef E_step_module = {
   PyModuleDef_HEAD_INIT,
   "E_step",   /* name of module */
   NULL, /* module documentation, may be NULL */
   -1,       /* size of per-interpreter state of the module,
                or -1 if the module keeps state in global variables. */
   E_step_Methods
};

PyMODINIT_FUNC PyInit_E_step() {
    return PyModule_Create(&E_step_module);
}

