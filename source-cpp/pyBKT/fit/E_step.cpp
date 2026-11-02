//
//  E_step.cpp
//  synthetic_data_helper
//
//  Created by Cristián Garay on 11/16/16.
//  Revised and edited by Anirudhan Badrinath on 27/02/20.
//

#define NPY_NO_DEPRECATED_API NPY_1_11_API_VERSION

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>
#include <Eigen/Core>
#include <omp.h>
#include <Python.h>
#include <numpy/ndarrayobject.h>

using namespace Eigen;

//original comment:
//"TODO if we aren't outputting gamma, don't need to write it to memory (just
//need t and t+1), so we can save the working array for each HMM at the cost of
//a branch"
//

using Int64Vector = Map<const Matrix<int64_t, Dynamic, 1>>;
using DoubleVector = Map<const VectorXd>;

// NumPy's C API passes arrays as PyObject *, and every PyArrayObject begins with that header.
static PyArrayObject *as_array(PyObject *object) {
    return reinterpret_cast<PyArrayObject *>(object); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
}

static Int64Vector int64_vector(PyArrayObject *arr) {
    return {static_cast<const int64_t *>(PyArray_DATA(arr)), PyArray_SIZE(arr)};
}

static DoubleVector double_vector(PyArrayObject *arr) {
    return {static_cast<const double *>(PyArray_DATA(arr)), PyArray_SIZE(arr)};
}

static void set_value_error(const std::string &message) {
    PyErr_SetString(PyExc_ValueError, message.c_str());
}

// Owns one new reference and releases it on scope exit.
template <typename T>
struct OwnedReference {
    T *pointer = nullptr;
    OwnedReference() = default;
    OwnedReference(const OwnedReference &) = delete;
    OwnedReference &operator=(const OwnedReference &) = delete;
    OwnedReference(OwnedReference &&) = delete;
    OwnedReference &operator=(OwnedReference &&) = delete;
    ~OwnedReference() { Py_XDECREF(pointer); }
};

// Reads an int argument the way PyArg_ParseTuple's "i" format does. Returns false with an exception set otherwise.
static bool parse_int(PyObject *object, long &value) {
    value = PyLong_AsLong(object);
    if (value == -1 && PyErr_Occurred() != nullptr)
        return false;
    if (value < INT_MIN || value > INT_MAX) {
        PyErr_SetString(PyExc_OverflowError, "signed integer is out of range for a C int");
        return false;
    }
    return true;
}

// Sets a ValueError and returns false when the shape or contents of an input would
// make run() read past the end of data, resources or a model array.
static bool check_inputs(PyArrayObject *alldata, PyArrayObject *allresources, PyArrayObject *starts,
                         PyArrayObject *lengths, PyArrayObject *learns, PyArrayObject *forgets,
                         PyArrayObject *guesses, PyArrayObject *slips) {
    if (PyArray_NDIM(alldata) != 2 || PyArray_NDIM(allresources) != 1 || PyArray_NDIM(starts) != 1 ||
        PyArray_NDIM(lengths) != 1 || PyArray_NDIM(learns) < 1) {
        set_value_error("data must be 2-D, and resources, starts, lengths and learns 1-D");
        return false;
    }
    const Py_ssize_t bigT = PyArray_DIM(alldata, 1);
    const Py_ssize_t num_sequences = PyArray_DIM(starts, 0);
    const Py_ssize_t len_allresources = PyArray_DIM(allresources, 0);
    const Py_ssize_t num_resources = PyArray_DIM(learns, 0);
    const Py_ssize_t num_subparts = PyArray_DIM(alldata, 0);
    if (PyArray_SIZE(forgets) < num_resources || PyArray_SIZE(guesses) < num_subparts || PyArray_SIZE(slips) < num_subparts) {
        set_value_error("learns has " + std::to_string(num_resources) + " entries and data has " +
                        std::to_string(num_subparts) + " rows, but forgets, guesses and slips have " +
                        std::to_string(PyArray_SIZE(forgets)) + ", " + std::to_string(PyArray_SIZE(guesses)) +
                        " and " + std::to_string(PyArray_SIZE(slips)));
        return false;
    }
    if (PyArray_DIM(lengths, 0) != num_sequences) {
        set_value_error("starts has " + std::to_string(num_sequences) + " entries but lengths has " +
                        std::to_string(PyArray_DIM(lengths, 0)));
        return false;
    }
    const Int64Vector start_values = int64_vector(starts);
    const Int64Vector length_values = int64_vector(lengths);
    const Int64Vector resource_values = int64_vector(allresources);
    for (Py_ssize_t k = 0; k < num_sequences; k++) {
        const int64_t start = start_values(k);
        const int64_t T = length_values(k);
        // T > bigT - (start - 1) means start - 1 + T > bigT, written so it cannot overflow.
        if (start < 1 || T < 1 || T > bigT - (start - 1)) {
            set_value_error("sequence " + std::to_string(k) + " has start " + std::to_string(start) + " and length " +
                            std::to_string(T) + " but data has " + std::to_string(bigT) + " columns");
            return false;
        }
        // The forward and backward loops both read resources only at t < T - 1.
        if (T > 1 && start - 1 + T - 2 >= len_allresources) {
            set_value_error("sequence " + std::to_string(k) + " needs " + std::to_string(start - 1 + T - 1) +
                            " resources entries but resources has " + std::to_string(len_allresources));
            return false;
        }
        for (int64_t t = start - 1; t < start - 1 + T - 1; t++) {
            if (resource_values(t) < 1 || resource_values(t) > num_resources) {
                set_value_error("resources has value " + std::to_string(resource_values(t)) + " but learns has " +
                                std::to_string(num_resources) + " entries");
                return false;
            }
        }
    }
    return true;
}

// Converts an integer array to the width run() reads. Rejects any other dtype,
// which a cast would truncate, for example 1.7 to 1.
static PyArrayObject *to_integer_array(PyObject *obj, int dtype, const char *name) {
    OwnedReference<PyObject> raw;
    raw.pointer = PyArray_FROM_O(obj);
    if (raw.pointer == nullptr)
        return nullptr;
    if (!PyArray_ISINTEGER(as_array(raw.pointer))) {
        set_value_error(std::string(name) + " must hold integers, not " + PyArray_DESCR(as_array(raw.pointer))->typeobj->tp_name);
        return nullptr;
    }
    return as_array(PyArray_FROM_OTF(raw.pointer, dtype, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST));
}

static PyArrayObject *to_array(PyObject *obj, int minimum_type) {
    return as_array(PyArray_FROM_OTF(obj, PyArray_ObjectType(obj, minimum_type), NPY_ARRAY_IN_ARRAY));
}

// Looks up an optional fixed-parameter array under the key Model uses and converts it to
// doubles. Leaves out empty when the key is absent. A single value is allowed where
// can_broadcast is set, as M_step broadcasts it. Sets a ValueError when the array has
// fewer than count entries otherwise, so run() never reads past its end.
static bool get_fixed_array(PyObject *fixed, const char *key, Py_ssize_t count, bool can_broadcast,
                            OwnedReference<PyArrayObject> &out) {
    PyObject *value = PyDict_GetItemString(fixed, key);
    if (value == nullptr)
        return true;
    out.pointer = as_array(PyArray_FROM_OTF(value, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST));
    if (out.pointer == nullptr)
        return false;
    if (PyArray_SIZE(out.pointer) < count && !(can_broadcast && PyArray_SIZE(out.pointer) == 1)) {
        set_value_error("fixed " + std::string(key) + " has " + std::to_string(PyArray_SIZE(out.pointer)) +
                        " entries but the model has " + std::to_string(count));
        return false;
    }
    return true;
}

// The fixed value for class n, or -1 when nothing is fixed. A single fixed value applies to every class.
static double fixed_value(const OwnedReference<PyArrayObject> &fixed, Index n) {
    if (fixed.pointer == nullptr)
        return -1;
    const DoubleVector values = double_vector(fixed.pointer);
    return values(values.size() == 1 ? 0 : n);
}

// A new uninitialized float64 array, so NumPy owns the memory the E-step writes into.
static PyObject *new_double_array(std::initializer_list<npy_intp> shape) {
    std::vector<npy_intp> dims(shape);
    return PyArray_SimpleNew(static_cast<int>(dims.size()), dims.data(), NPY_DOUBLE);
}

static double *double_data(PyObject *array) {
    return static_cast<double *>(PyArray_DATA(as_array(array)));
}

static PyObject* run(PyObject * /* module */, PyObject * args) {
    //TODO: check that dicts have the required members.
    Eigen::initParallel();

    // data, model, num_outputs, parallel and fixed
    constexpr Py_ssize_t num_arguments = 5;
    long num_outputs = 0;
    long parallel = 0;
    if (PyTuple_GET_SIZE(args) != num_arguments || !parse_int(PyTuple_GET_ITEM(args, 2), num_outputs) ||
        !parse_int(PyTuple_GET_ITEM(args, 3), parallel)) {
        set_value_error("Error parsing arguments.");
        return nullptr;
    }
    PyObject *data_ptr = PyTuple_GET_ITEM(args, 0);
    PyObject *model_ptr = PyTuple_GET_ITEM(args, 1);
    PyObject *fixed = PyTuple_GET_ITEM(args, 4);

    // Load all the numpy arrays in data & model
    OwnedReference<PyArrayObject> alldata, allresources, starts, lengths, learns, forgets, guesses, slips;
    struct Input {
        const char *name;
        PyObject *source;
        OwnedReference<PyArrayObject> *target;
        int integer_type; // the width an integer input is converted to, or NPY_NOTYPE for a model array
    };
    const std::array<Input, 8> inputs = {{
        {"data", data_ptr, &alldata, NPY_INT32},
        {"resources", data_ptr, &allresources, NPY_INT64},
        {"starts", data_ptr, &starts, NPY_INT64},
        {"lengths", data_ptr, &lengths, NPY_INT64},
        {"learns", model_ptr, &learns, NPY_NOTYPE},
        {"forgets", model_ptr, &forgets, NPY_NOTYPE},
        {"guesses", model_ptr, &guesses, NPY_NOTYPE},
        {"slips", model_ptr, &slips, NPY_NOTYPE},
    }};
    for (const Input &input : inputs) {
        PyObject *value = PyDict_GetItemString(input.source, input.name);
        input.target->pointer = input.integer_type == NPY_NOTYPE ? to_array(value, NPY_FLOAT)
                                                                 : to_integer_array(value, input.integer_type, input.name);
        if (input.target->pointer == nullptr)
            return nullptr;
    }
    if (!check_inputs(alldata.pointer, allresources.pointer, starts.pointer, lengths.pointer, learns.pointer,
                      forgets.pointer, guesses.pointer, slips.pointer))
        return nullptr;
    double prior = PyFloat_AsDouble(PyDict_GetItemString(model_ptr, "prior"));

    Index bigT = PyArray_DIM(alldata.pointer, 1), num_subparts = PyArray_DIM(alldata.pointer, 0);
    int num_sequences = static_cast<int>(PyArray_DIM(starts.pointer, 0));
    Index num_resources = PyArray_DIM(learns.pointer, 0);

    Map<Array<int32_t,Dynamic,Dynamic,RowMajor>> alldata_arr(static_cast<int32_t *>(PyArray_DATA(alldata.pointer)),num_subparts,bigT);
    Map<Array<int64_t, Eigen::Dynamic, 1>> allresources_arr(static_cast<int64_t *>(PyArray_DATA(allresources.pointer)),PyArray_DIM(allresources.pointer, 0),1);
    Map<Array<int64_t, Eigen::Dynamic, 1>> starts_arr(static_cast<int64_t *>(PyArray_DATA(starts.pointer)),num_sequences,1);
    Map<Array<int64_t, Eigen::Dynamic, 1>> lengths_arr(static_cast<int64_t *>(PyArray_DATA(lengths.pointer)),PyArray_DIM(lengths.pointer, 0),1);

    bool normalizeLengths = false;
    //then the original code goes to find the optional parameters.

    PyObject* fixed_prior = PyDict_GetItemString(fixed, "prior");
    OwnedReference<PyArrayObject> fixed_learn, fixed_forget, fixed_guess, fixed_slip;
    if (!get_fixed_array(fixed, "learns", num_resources, false, fixed_learn) ||
        !get_fixed_array(fixed, "forgets", num_resources, false, fixed_forget) ||
        !get_fixed_array(fixed, "guesses", num_subparts, true, fixed_guess) ||
        !get_fixed_array(fixed, "slips", num_subparts, true, fixed_slip))
        return nullptr;

    if (fixed_prior != nullptr) {
        prior = PyFloat_AsDouble(fixed_prior);
        if (prior == -1.0 && PyErr_Occurred() != nullptr)
            return nullptr;
    }
    Array2d initial_distn;
    initial_distn << 1-prior, prior;

    const DoubleVector learn_values = double_vector(learns.pointer);
    const DoubleVector forget_values = double_vector(forgets.pointer);
    MatrixXd As(2,2*num_resources);
    for (Index n=0; n<num_resources; n++) {
        double learn = fixed_value(fixed_learn, n);
        if (learn < 0) {
            learn = learn_values(n);
        }
        double forget = fixed_value(fixed_forget, n);
        if (forget < 0) {
            forget = forget_values(n);
        }
        As.col(2*n) << 1-learn, learn;
        As.col(2*n+1) << forget, 1-forget;
    }

    const DoubleVector guess_values = double_vector(guesses.pointer);
    const DoubleVector slip_values = double_vector(slips.pointer);
    Array2Xd Bn(2,2*num_subparts);
    for (Index n=0; n<num_subparts; n++) {
        double guess = fixed_value(fixed_guess, n);
        if (guess < 0) {
            guess = guess_values(n);
        }
        double slip = fixed_value(fixed_slip, n);
        if (slip < 0) {
            slip = slip_values(n);
        }
        Bn.col(2*n) << 1-guess, slip; // incorrect
        Bn.col(2*n+1) << guess, 1-slip; // correct
    }

    //// outputs

    OwnedReference<PyObject> all_trans_softcounts_arr, all_emission_softcounts_arr, all_initial_softcounts_arr, alpha_out_arr;
    all_trans_softcounts_arr.pointer = new_double_array({num_resources, 2, 2});
    all_emission_softcounts_arr.pointer = new_double_array({num_subparts, 2, 2});
    all_initial_softcounts_arr.pointer = new_double_array({2, 1});
    alpha_out_arr.pointer = new_double_array({2, bigT});
    if (all_trans_softcounts_arr.pointer == nullptr || all_emission_softcounts_arr.pointer == nullptr ||
        all_initial_softcounts_arr.pointer == nullptr || alpha_out_arr.pointer == nullptr)
        return nullptr;
    Map<ArrayXXd> all_trans_softcounts(double_data(all_trans_softcounts_arr.pointer),2,2*num_resources);
    all_trans_softcounts.setZero();
    Map<Array2Xd> all_emission_softcounts(double_data(all_emission_softcounts_arr.pointer),2,2*num_subparts);
    all_emission_softcounts.setZero();
    Map<Array2d> all_initial_softcounts(double_data(all_initial_softcounts_arr.pointer));
    all_initial_softcounts.setZero();
    Map<Array2Xd> alpha_out(double_data(alpha_out_arr.pointer),2,bigT);
    double total_loglike = 0;

    /* COMPUTATION */
    #pragma omp parallel if(parallel) default(none) \
        shared(num_resources, num_subparts, num_sequences, starts_arr, lengths_arr, alldata_arr, allresources_arr, \
               Bn, As, initial_distn, normalizeLengths, alpha_out, all_trans_softcounts, all_emission_softcounts, \
               all_initial_softcounts, total_loglike)
    {
        // heap, not stack: long sequences or many resources overflow a thread's stack
        std::vector<double> s_trans_softcounts(2*(2*num_resources));
        std::vector<double> s_emission_softcounts(2*(2*num_subparts));
        std::vector<double> s_likelihoods, s_alpha, s_gamma;
        Map<ArrayXXd,Unaligned> trans_softcounts_temp(s_trans_softcounts.data(),2,2*num_resources);
        Map<ArrayXXd,Unaligned> emission_softcounts_temp(s_emission_softcounts.data(),2,2*num_subparts);
        Array2d init_softcounts_temp;
        double loglike = 0;

        trans_softcounts_temp.setZero();
        emission_softcounts_temp.setZero();
        init_softcounts_temp.setZero();
        int num_threads = omp_get_num_threads();
        int blocklen = 1 + ((num_sequences - 1) / num_threads);
        int sequence_idx_start = blocklen * omp_get_thread_num();
        int sequence_idx_end = std::min(sequence_idx_start+blocklen,num_sequences);

        for (int sequence_index=sequence_idx_start; sequence_index < sequence_idx_end; sequence_index++) {

            // NOTE: -1 because Matlab indexing starts at 1
            int64_t sequence_start = starts_arr(sequence_index, 0) - 1;

            int64_t T = lengths_arr(sequence_index, 0);
            if (static_cast<int64_t>(s_alpha.size()) < 2*T) {
                s_likelihoods.resize(2*T);
                s_alpha.resize(2*T);
                s_gamma.resize(2*T);
            }

            //// likelihoods
            Map<Array2Xd,Unaligned> likelihoods(s_likelihoods.data(),2,T);

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
            Map<MatrixXd,Unaligned> alpha(s_alpha.data(),2,T);
            alpha.col(0) = initial_distn * likelihoods.col(0);
            double norm = alpha.col(0).sum();
            alpha.col(0) /= norm;
            loglike += std::log(norm) / static_cast<double>(normalizeLengths? T : 1);

            for (int t=0; t<T-1; t++) {
                int64_t resources_temp = allresources_arr(sequence_start+t, 0);
                alpha.col(t+1) = (As.block(0,2*(resources_temp-1),2,2) * alpha.col(t)).array()
                    * likelihoods.col(t+1);
                norm = alpha.col(t+1).sum();
                alpha.col(t+1) /= norm;
                loglike += std::log(norm) / static_cast<double>(normalizeLengths? T : 1);
            }

            //// backward messages and statistic counting

            Map<Array2Xd,Unaligned> gamma(s_gamma.data(),2,T);
            gamma.col(T-1) = alpha.col(T-1);
            for (int n=0; n<num_subparts; n++) {
                int32_t data_temp = alldata_arr(n, sequence_start+(T-1));
                if (data_temp != 0) {
                    emission_softcounts_temp.col(2*n + (data_temp == 2)) += gamma.col(T-1);
                }
            }

            for (int64_t t=T-2; t>=0; t--) {

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

            alpha_out.block(0,sequence_start,2,T) = alpha;
        }
        #pragma omp critical
        {
            all_trans_softcounts += trans_softcounts_temp;
            all_emission_softcounts += emission_softcounts_temp;
            all_initial_softcounts += init_softcounts_temp;
            total_loglike += loglike;
        }
    }

    PyObject *result = PyDict_New();
    PyDict_SetItemString(result, "all_trans_softcounts", all_trans_softcounts_arr.pointer);
    PyDict_SetItemString(result, "all_emission_softcounts", all_emission_softcounts_arr.pointer);
    PyDict_SetItemString(result, "all_initial_softcounts", all_initial_softcounts_arr.pointer);
    PyDict_SetItemString(result, "alpha", alpha_out_arr.pointer);
    OwnedReference<PyObject> total_loglike_obj;
    total_loglike_obj.pointer = PyFloat_FromDouble(total_loglike);
    PyDict_SetItemString(result, "total_loglike", total_loglike_obj.pointer);

    return(result);
}

PyMODINIT_FUNC PyInit_E_step() {
    import_array();
    static std::array<PyMethodDef, 2> methods = {{
        {"run", run, METH_VARARGS, "Runs E-step of Expectation Maximization in C++ module"},
        {nullptr, nullptr, 0, nullptr},
    }};
    static PyModuleDef module = {
        PyModuleDef_HEAD_INIT, "E_step", nullptr, -1, methods.data(), nullptr, nullptr, nullptr, nullptr,
    };
    return PyModule_Create(&module);
}
