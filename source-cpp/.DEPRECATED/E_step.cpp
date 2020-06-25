//
//  E_step.cpp
//  synthetic_data_helper
//
//  Created by Cristián Garay on 11/16/16.
//  Copyright © 2016 Cristian Garay. All rights reserved.
//

#define NPY_NO_DEPRECATED_API NPY_1_11_API_VERSION

#include <iostream>
#include <stdint.h>
#include <alloca.h>
#include <Eigen/Core>
#include <omp.h>
#include <boost/python.hpp>
#include <boost/python/numeric.hpp>
#include <boost/python/ptr.hpp>
#include <Python.h>
#include <numpy/ndarrayobject.h>

using namespace Eigen;
using namespace std;
using namespace boost::python;

#if PY_VERSION_HEX >= 0x03000000
void *
#else
void
#endif
init_numpy(){
    //Py_Initialize;
    import_array();
}

//original comment:
//"TODO if we aren't outputting gamma, don't need to write it to memory (just
//need t and t+1), so we can save the stack array for each HMM at the cost of
//a branch"

/*struct double_to_python_float
{
    static PyObject* convert(double const& d)
      {
        return boost::python::incref(
          boost::python::object(d).ptr());
      }
};*/

//numpy scalar converters.
template <typename T, NPY_TYPES NumPyScalarType>
struct enable_numpy_scalar_converter
{
  enable_numpy_scalar_converter()
  {
    // Required NumPy call in order to use the NumPy C API within another
    // extension module.
    // import_array();
    init_numpy();

    boost::python::converter::registry::push_back(
      &convertible,
      &construct,
      boost::python::type_id<T>());
  }

  static void* convertible(PyObject* object)
  {
    // The object is convertible if all of the following are true:
    // - is a valid object.
    // - is a numpy array scalar.
    // - its descriptor type matches the type for this converter.
    return (
      object &&                                                    // Valid
      PyArray_CheckScalar(object) &&                               // Scalar
      PyArray_DescrFromScalar(object)->type_num == NumPyScalarType // Match
    )
      ? object // The Python object can be converted.
      : NULL;
  }

  static void construct(
    PyObject* object,
    boost::python::converter::rvalue_from_python_stage1_data* data)
  {
    // Obtain a handle to the memory block that the converter has allocated
    // for the C++ type.
    namespace python = boost::python;
    typedef python::converter::rvalue_from_python_storage<T> storage_type;
    void* storage = reinterpret_cast<storage_type*>(data)->storage.bytes;

    // Extract the array scalar type directly into the storage.
    PyArray_ScalarAsCtype(object, storage);

    // Set convertible to indicate success.
    data->convertible = storage;
  }
};

//dict create_synthetic_data(dict& model, numeric::array& starts, numeric::array& lengths, numeric::array& resources)
dict run(dict& data, dict& model, numeric::array& trans_softcounts, numeric::array& emission_softcounts, numeric::array& init_softcounts, int num_outputs){
    //TODO: check if parameters are null.
    //TODO: check that dicts have the required members.
    //TODO: check that all parameters have the right sizes.
    //TODO: i'm not sending any error messages.

    IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");

    Eigen::initParallel();

    numeric::array alldata = extract<numeric::array>(data["data"]); //multidimensional array, so i need to keep extracting arrays.
    int bigT = len(alldata[0]); //this should be the number of columns in the alldata object. i'm assuming is 2d array.
    int num_subparts = len(alldata);
    Array<int32_t, Eigen::Dynamic, Eigen::Dynamic> alldata_arr;
    alldata_arr.resize(num_subparts, bigT);
    for (int i = 0; i < num_subparts; i++)
        for (int j = 0; j < bigT; j++)
            alldata_arr(i, j) = extract<int32_t>(alldata[i][j]);

    numeric::array allresources = extract<numeric::array>(data["resources"]);
    int len_allresources = len(allresources);
    Array<int64_t, Eigen::Dynamic, 1> allresources_arr;
    allresources_arr.resize(len_allresources, 1);
    for (int i = 0; i < len_allresources; i++)
        allresources_arr.row(i) << extract<int64_t>(allresources[i]);

    numeric::array starts = extract<numeric::array>(data["starts"]);
    int num_sequences = len(starts);
    Array<int64_t, Eigen::Dynamic, 1> starts_arr;
    starts_arr.resize(num_sequences, 1);
    for (int i = 0; i < num_sequences; i++)
        starts_arr.row(i) << extract<int64_t>(starts[i]);

    numeric::array lengths = extract<numeric::array>(data["lengths"]);
    int len_lengths = len(lengths);
    Array<int64_t, Eigen::Dynamic, 1> lengths_arr;
    lengths_arr.resize(len_lengths, 1);
    for (int i = 0; i < len_lengths; i++)
        lengths_arr.row(i) << extract<int64_t>(lengths[i]);

    numeric::array learns = extract<numeric::array>(model["learns"]);
    int num_resources = len(learns);

    numeric::array forgets = extract<numeric::array>(model["forgets"]);

    numeric::array guesses = extract<numeric::array>(model["guesses"]);

    numeric::array slips = extract<numeric::array>(model["slips"]);

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
    //cout << "all_trans_softcounts" << all_trans_softcounts << endl;
    //cout << "all_emission_softcounts" << all_emission_softcounts << endl;
    //cout << "all_initial_softcounts" << all_initial_softcounts << endl;
    double r_trans_softcounts[2*2*num_resources];
    double r_emission_softcounts[2*2*num_subparts];
    double r_init_softcounts[2*1];
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
    //cout << "likelihoods_out" << likelihoods_out << endl;
    //cout << "gamma_out" << gamma_out << endl;
    //cout << "alpha_out" << alpha_out << endl;
    //cout << "s_total_loglike " << s_total_loglike << endl;
    //cout << "total_loglike " << total_loglike << endl;

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
    double * r_alpha_out = new double[2*bigT];
    new (&alpha_out) Map<Array2Xd,Aligned>(r_alpha_out,2,bigT);


    /* COMPUTATION */
    /* omp_set_dynamic(0); */
    /* omp_set_num_threads(6); */
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
        //cout << "start: " << sequence_idx_start << " end: " << sequence_idx_end << endl;

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
                    bool zeroed = Bn.col(2*n + (data_temp == 2)).isZero(0);
                     if (data_temp != 0 && !zeroed) {
                         if (!(likelihoods.col(t) * Bn.col(2*n + (data_temp == 2))).isZero(0)) {
                            likelihoods.col(t) *= Bn.col(2*n + (data_temp == 2));
                         }
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
            //cout << "norm: " << norm << endl;
            alpha.col(0) /= norm;
            contribution = log(norm);
            //cout << "contribution " << contribution << endl;
            if(normalizeLengths) {
                contribution = contribution / T;
            }
            loglike += contribution;
            //cout << "loglike2 " << loglike << endl;

            for (int t=0; t<T-1; t++) {
                int64_t resources_temp = allresources_arr(sequence_start+t, 0);
                alpha.col(t+1) = (As.block(0,2*(resources_temp-1),2,2) * alpha.col(t)).array()
                    * likelihoods.col(t+1);
                //cout << "likelihoods.col(t+1) " << likelihoods.col(t+1) << endl;
                norm = alpha.col(t+1).sum();
                //cout << "norm: " << norm << endl;
                alpha.col(t+1) /= norm;
                contribution = log(norm);
                //cout << "contribution: " << contribution << endl;
                if(normalizeLengths) {
                    contribution = contribution / T;
                }
                loglike += contribution;
                //cout << "loglike " << loglike << endl;
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
            //cout << "loglike " << loglike << endl;
            *total_loglike += loglike;
        }
    }

    dict result;
    result["total_loglike"] = *total_loglike;

    //cout << "r_trans_softcounts " << r_trans_softcounts << endl;

    npy_intp all_trans_softcounts_dims[3] = {num_resources,2,2}; //TODO: just put directly this array into the PyArray_SimpleNewFromData function?
    PyObject * all_trans_softcounts_pyObj = PyArray_New(&PyArray_Type, 3, all_trans_softcounts_dims, NPY_DOUBLE, NULL, &r_trans_softcounts, 0, NPY_ARRAY_CARRAY, NULL);
    boost::python::handle<> all_trans_softcounts_handle( all_trans_softcounts_pyObj );
    boost::python::numeric::array all_trans_softcounts_arr( all_trans_softcounts_handle );
    result["all_trans_softcounts"] = all_trans_softcounts_arr;

    npy_intp all_emission_softcounts_dims[3] = {num_subparts,2,2}; //TODO: just put directly this array into the PyArray_SimpleNewFromData function?
    PyObject * all_emission_softcounts_pyObj = PyArray_New(&PyArray_Type, 3, all_emission_softcounts_dims, NPY_DOUBLE, NULL, &r_emission_softcounts, 0, NPY_ARRAY_CARRAY, NULL);
    boost::python::handle<> all_emission_softcounts_handle( all_emission_softcounts_pyObj );
    boost::python::numeric::array all_emission_softcounts_arr( all_emission_softcounts_handle );
    result["all_emission_softcounts"] = all_emission_softcounts_arr;

    npy_intp all_initial_softcounts_dims[2] = {2,1}; //TODO: just put directly this array into the PyArray_SimpleNewFromData function?
    PyObject * all_initial_softcounts_pyObj = PyArray_New(&PyArray_Type, 2, all_initial_softcounts_dims, NPY_DOUBLE, NULL, &r_init_softcounts, 0, NPY_ARRAY_CARRAY, NULL);
    boost::python::handle<> all_initial_softcounts_handle( all_initial_softcounts_pyObj );
    boost::python::numeric::array all_initial_softcounts_arr( all_initial_softcounts_handle );
    result["all_initial_softcounts"] = all_initial_softcounts_arr;

    npy_intp alpha_out_dims[2] = {2,bigT}; //TODO: just put directly this array into the PyArray_SimpleNewFromData function?
    PyObject * alpha_out_pyObj = PyArray_New(&PyArray_Type, 2, alpha_out_dims, NPY_DOUBLE, NULL, r_alpha_out, 0, NPY_ARRAY_CARRAY, NULL);
    boost::python::handle<> alpha_out_handle( alpha_out_pyObj );
    boost::python::numeric::array alpha_out_arr (alpha_out_handle);
    result["alpha"] = alpha_out_arr;

    delete r_alpha_out;

    return(result);
}


BOOST_PYTHON_MODULE(E_step){
    //import_array();
    init_numpy();
    numeric::array::set_module_and_type("numpy", "ndarray");
    //to_python_converter<double, double_to_python_float>();
    enable_numpy_scalar_converter<boost::int8_t, NPY_INT8>();
    enable_numpy_scalar_converter<boost::int16_t, NPY_INT16>();
    enable_numpy_scalar_converter<boost::int32_t, NPY_INT32>();
    enable_numpy_scalar_converter<boost::int64_t, NPY_INT64>();

    def("run", run);

}
