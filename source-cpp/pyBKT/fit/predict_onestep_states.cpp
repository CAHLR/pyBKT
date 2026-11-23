#define NPY_NO_DEPRECATED_API NPY_1_11_API_VERSION

#include <iostream>
#include <stdint.h>
#include <omp.h>
#include <Eigen/Core>
#include <Python.h>
#include <numpy/ndarrayobject.h>

using namespace Eigen;
using namespace std;

static double extract_double(PyArrayObject *arr, int i) {
    return ((double*) PyArray_DATA(arr))[i];
}

static double extract_double_2d(PyArrayObject *arr, int i, int j) {
    return ((double*) PyArray_DATA(arr))[i * PyArray_DIM(arr, 1) + j];
}

static double extract_int64_t(PyArrayObject *arr, int i) {
    return ((int64_t*) PyArray_DATA(arr))[i];
}

// Sets a ValueError and returns false when the shape or contents of an input would
// make run() read past the end of data, resources or a model array.
static bool check_inputs(PyArrayObject *alldata, PyArrayObject *allresources, PyArrayObject *starts,
                         PyArrayObject *lengths, PyArrayObject *learns, PyArrayObject *forgets,
                         PyArrayObject *guesses, PyArrayObject *slips) {
    if (PyArray_NDIM(alldata) != 2 || PyArray_NDIM(allresources) != 1 || PyArray_NDIM(starts) != 1 ||
        PyArray_NDIM(lengths) != 1 || PyArray_NDIM(learns) < 1) {
        PyErr_SetString(PyExc_ValueError, "data must be 2-D, and resources, starts, lengths and learns 1-D");
        return false;
    }
    Py_ssize_t bigT = PyArray_DIM(alldata, 1), num_sequences = PyArray_DIM(starts, 0);
    Py_ssize_t len_allresources = PyArray_DIM(allresources, 0), num_resources = PyArray_DIM(learns, 0);
    Py_ssize_t num_subparts = PyArray_DIM(alldata, 0);
    if (PyArray_SIZE(forgets) < num_resources || PyArray_SIZE(guesses) < num_subparts || PyArray_SIZE(slips) < num_subparts) {
        PyErr_Format(PyExc_ValueError, "learns has %zd entries and data has %zd rows, but forgets, guesses and slips have %zd, %zd and %zd",
                     num_resources, num_subparts, (Py_ssize_t) PyArray_SIZE(forgets), (Py_ssize_t) PyArray_SIZE(guesses),
                     (Py_ssize_t) PyArray_SIZE(slips));
        return false;
    }
    if (PyArray_DIM(lengths, 0) != num_sequences) {
        PyErr_Format(PyExc_ValueError, "starts has %zd entries but lengths has %zd", num_sequences, (Py_ssize_t) PyArray_DIM(lengths, 0));
        return false;
    }
    int64_t *start_values = (int64_t*) PyArray_DATA(starts), *length_values = (int64_t*) PyArray_DATA(lengths);
    int64_t *resource_values = (int64_t*) PyArray_DATA(allresources);
    for (Py_ssize_t k = 0; k < num_sequences; k++) {
        int64_t start = start_values[k], T = length_values[k];
        // T > bigT - (start - 1) means start - 1 + T > bigT, written so it cannot overflow.
        if (start < 1 || T < 1 || T > bigT - (start - 1)) {
            PyErr_Format(PyExc_ValueError, "sequence %zd has start %lld and length %lld but data has %zd columns",
                         k, (long long) start, (long long) T, bigT);
            return false;
        }
        // The loop below reads resources only at t < T - 1.
        if (T > 1 && start - 1 + T - 2 >= len_allresources) {
            PyErr_Format(PyExc_ValueError, "sequence %zd needs %lld resources entries but resources has %zd",
                         k, (long long) (start - 1 + T - 1), len_allresources);
            return false;
        }
        for (int64_t t = start - 1; t < start - 1 + T - 1; t++) {
            if (resource_values[t] < 1 || resource_values[t] > num_resources) {
                PyErr_Format(PyExc_ValueError, "resources has value %lld but learns has %zd entries",
                             (long long) resource_values[t], num_resources);
                return false;
            }
        }
    }
    return true;
}

// Converts an integer array to the width run() reads. Rejects any other dtype,
// which a cast would truncate, for example 1.7 to 1.
static PyArrayObject* to_integer_array(PyObject *obj, int dtype, const char *name) {
    PyArrayObject *raw = (PyArrayObject *) PyArray_FROM_O(obj);
    if (raw == NULL)
        return NULL;
    PyArrayObject *result = NULL;
    if (PyArray_ISINTEGER(raw))
        result = (PyArrayObject *) PyArray_FROM_OTF((PyObject *) raw, dtype, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
    else
        PyErr_Format(PyExc_ValueError, "%s must hold integers, not %s", name, PyArray_DESCR(raw)->typeobj->tp_name);
    Py_DECREF(raw);
    return result;
}

void capsule_cleanup(PyObject *capsule) {
    double *memory = (double *) PyCapsule_GetPointer(capsule, NULL);
    delete[] memory;
}

static PyObject* run(PyObject * module, PyObject * args) {
    //TODO: check if parameters are null.
    //TODO: check that dicts have the required members.
    //TODO: check that all parameters have the right sizes.
    //TODO: i'm not sending any error messages.
    import_array();
    
    PyObject *data_ptr = NULL, *model_ptr = NULL, *fwd_msgs = NULL;
    PyArrayObject *alldata = NULL, *allresources = NULL, *starts = NULL, *lengths = NULL, *learns = NULL, *forgets = NULL, *guesses = NULL, *slips = NULL, *forward_messages = NULL;
    double prior;
    int parallel;

    if (!PyArg_ParseTuple(args, "OOOi", &data_ptr, &model_ptr, &fwd_msgs, &parallel)) {
        PyErr_SetString(PyExc_ValueError, "Error parsing arguments.");
        return NULL;
    }

    int DTYPE = PyArray_ObjectType(fwd_msgs, NPY_FLOAT);
    forward_messages = (PyArrayObject *)PyArray_FROM_OTF(fwd_msgs, DTYPE, NPY_ARRAY_IN_ARRAY);

    char* DM_NAMES[] = {"data", "resources", "starts", "lengths", "learns", "forgets", "guesses", "slips"};
    PyArrayObject** DM_PTRS[] = {&alldata, &allresources, &starts, &lengths, &learns, &forgets, &guesses, &slips};
    bool is_valid = true;
    for (int i = 0; is_valid && i < 8; i++) {
        PyObject *dp = PyDict_GetItemString(i < 4 ? data_ptr : model_ptr, DM_NAMES[i]);
        if (i >= 1 && i < 4) {
            *DM_PTRS[i] = to_integer_array(dp, NPY_INT64, DM_NAMES[i]);
        } else {
            DTYPE = PyArray_ObjectType(dp, (i < 1 ? NPY_INT : NPY_FLOAT)); // hack to force correct type
            *DM_PTRS[i] = (PyArrayObject *)PyArray_FROM_OTF(dp, DTYPE, NPY_ARRAY_IN_ARRAY);
        }
        is_valid = *DM_PTRS[i] != NULL;
    }
    if (!is_valid || !check_inputs(alldata, allresources, starts, lengths, learns, forgets, guesses, slips)) {
        for (int i = 0; i < 8; i++)
            Py_XDECREF(*DM_PTRS[i]);
        Py_XDECREF(forward_messages);
        return NULL;
    }
    prior = PyFloat_AsDouble(PyDict_GetItemString(model_ptr, "prior"));

    int bigT = (int) PyArray_DIM(alldata, 1), num_subparts = (int) PyArray_DIM(alldata, 0);
    int len_allresources = (int) PyArray_DIM(allresources, 0);
    int num_sequences = (int) PyArray_DIM(starts, 0);
    int len_lengths = (int) PyArray_DIM(lengths, 0);
    int num_resources = (int) PyArray_DIM(learns, 0);

    Array2d initial_distn;
    initial_distn << 1-prior, prior;

    MatrixXd As(2,2*num_resources);
    for (int n=0; n<num_resources; n++) {
        double learn = extract_double(learns, n);
        double forget = extract_double(forgets, n);
        As.col(2*n) << 1-learn, learn;
        As.col(2*n+1) << forget, 1-forget;
    }

    // forward messages
    //numpy::ndarray all_forward_messages = extract<numpy::ndarray>(forward_messages);
    double * forward_messages_temp = new double[2*bigT];
    for (int i=0; i<2; i++) {
        for (int j=0; j<bigT; j++){
            forward_messages_temp[i* bigT +j] = extract_double_2d(forward_messages, i, j);
        }
    }

    //// outputs

    double* all_predictions = new double[2 * bigT];
    Map<Array2Xd,Aligned> predictions(all_predictions,2,bigT);

    /* COMPUTATION */

    #pragma omp parallel for if(parallel)
    for (int sequence_index=0; sequence_index < num_sequences; sequence_index++) {
        // NOTE: -1 because Matlab indexing starts at 1
        int64_t sequence_start = extract_int64_t(starts, sequence_index) - 1;
        int64_t T = extract_int64_t(lengths, sequence_index);

        //int16_t *resources = allresources + sequence_start;
        Map<MatrixXd, Aligned> forward_messages(forward_messages_temp + 2*sequence_start,2,T);
        Map<MatrixXd, Aligned> predictions(all_predictions + 2*sequence_start,2,T);

        predictions.col(0) = initial_distn;
        for (int t=0; t<T-1; t++) {
            int64_t resources_temp = extract_int64_t(allresources, sequence_start + t);
            predictions.col(t+1) = As.block(0,2*(resources_temp-1),2,2) * forward_messages.col(t);
        }
    }

    npy_intp dims[] = {2, bigT};
    PyObject *all_predictions_arr = (PyObject *) PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, all_predictions);
    PyObject *capsule = PyCapsule_New(all_predictions, NULL, capsule_cleanup);
    PyArray_SetBaseObject((PyArrayObject *) all_predictions_arr, capsule);

    for (int i = 0; i < 8; i++)
        Py_XDECREF(*DM_PTRS[i]);
    Py_XDECREF(forward_messages);
    delete[] forward_messages_temp;

    return(all_predictions_arr);
}

static PyMethodDef predict_onestep_states_Methods[] = {
    {"run",  run, METH_VARARGS,
     "Generates predictions for BKT model"},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};


static struct PyModuleDef predict_onestep_states_module = {
   PyModuleDef_HEAD_INIT,
   "predict_onestep_states",   /* name of module */
   NULL, /* module documentation, may be NULL */
   -1,       /* size of per-interpreter state of the module,
                or -1 if the module keeps state in global variables. */
   predict_onestep_states_Methods
};

PyMODINIT_FUNC PyInit_predict_onestep_states() {
    return PyModule_Create(&predict_onestep_states_module);
}

