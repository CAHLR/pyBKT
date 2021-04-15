//
//  synthetic_data_helper.cpp
//  synthetic_data_helper
//
//  Created by Cristián Garay on 10/15/16.
//  Revised and edited by Anirudhan Badrinath on 27/02/20.
//

#define NPY_NO_DEPRECATED_API NPY_1_11_API_VERSION

#include <iostream>
#include <stdint.h>
#include <alloca.h>
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

void capsule_cleanup(PyObject *capsule) {
    void *memory = PyCapsule_GetPointer(capsule, NULL);
    delete memory;
}


static PyObject* run(PyObject * module, PyObject * args) {
    //TODO: check if parameters are null.
    //TODO: check that dicts have the required members.
    //TODO: check that all parameters have the right sizes.
    //TODO: i'm not sending any error messages.
    import_array();

    PyObject *model_ptr = NULL, *starts_obj = NULL, *lengths_obj = NULL, *resources_obj = NULL;
    PyArrayObject *resources = NULL, *starts = NULL, *lengths = NULL, *learns = NULL, *forgets = NULL, *guesses = NULL, *slips = NULL;
    double prior;

    if (!PyArg_ParseTuple(args, "OOOO", &model_ptr, &starts_obj, &lengths_obj, &resources_obj)) {
        PyErr_SetString(PyExc_ValueError, "Error parsing arguments.");
        return NULL;
    }

    int DTYPE = PyArray_ObjectType(starts_obj, NPY_INT64);
    starts = (PyArrayObject *)PyArray_FROM_OTF(starts_obj, DTYPE, NPY_ARRAY_IN_ARRAY);
    DTYPE = PyArray_ObjectType(lengths_obj, NPY_INT64);
    lengths = (PyArrayObject *)PyArray_FROM_OTF(lengths_obj, DTYPE, NPY_ARRAY_IN_ARRAY);
    DTYPE = PyArray_ObjectType(resources_obj, NPY_INT64);
    resources = (PyArrayObject *)PyArray_FROM_OTF(resources_obj, DTYPE, NPY_ARRAY_IN_ARRAY);

    char* DM_NAMES[] = {"learns", "forgets", "guesses", "slips"};
    PyArrayObject** DM_PTRS[] = {&learns, &forgets, &guesses, &slips};
    for (int i = 0; i < 4; i++) {
        PyObject *dp = PyDict_GetItemString(model_ptr, DM_NAMES[i]);
        DTYPE = PyArray_ObjectType(dp, NPY_FLOAT); // hack to force correct type
        *DM_PTRS[i] = (PyArrayObject *)PyArray_FROM_OTF(dp, DTYPE, NPY_ARRAY_IN_ARRAY);
    }
    prior = PyFloat_AsDouble(PyDict_GetItemString(model_ptr, "prior"));

    int num_subparts = (int) PyArray_DIM(slips, 0);
    int num_sequences = (int) PyArray_DIM(starts, 0);
    int num_resources = (int) PyArray_DIM(learns, 0);
    
    Vector2d initial_distn;
    initial_distn << 1-prior, prior;
    
    MatrixXd As(2, 2*num_resources);
    for (int n=0; n<num_resources; n++) {
        double learn = extract_double(learns,n);
        double forget = extract_double(forgets,n);
        As.col(2*n) << 1-learn, learn;
        As.col(2*n+1) << forget, 1-forget;
    }
    
    int64_t bigT = 0;
    for (int k=0; k<num_sequences; k++) {
        bigT += extract_int64_t(lengths,k); //extract this as int??
    }
    
    //// outputs
    int* all_stateseqs = new int[bigT];
    int* all_data = new int[num_subparts * bigT]; //used to be int8_t
    *all_data = 0;
    
    /* COMPUTATION */
    
    for (int sequence_index=0; sequence_index < num_sequences; sequence_index++) {
        int64_t sequence_start = extract_int64_t(starts,sequence_index) - 1; //should i extract these as ints?
        int64_t T = extract_int64_t(lengths, sequence_index);
        
        Vector2d nextstate_distr = initial_distn;

        for (int t=0; t<T; t++) {
            *(all_stateseqs + sequence_start + t) = nextstate_distr(0) < ((double) rand()) / ((double) RAND_MAX); //always all_stateseqs[0]?
            for (int n=0; n<num_subparts; n++) {
                *(all_data + n * (bigT) + sequence_start + t) = ((*(all_stateseqs + sequence_start + t)) ? extract_double(slips, n) : (1-extract_double(guesses, n))) < (((double) rand()) / ((double) RAND_MAX));
            }
            
            nextstate_distr = As.col(2*(extract_int64_t(resources, sequence_start + t)-1)+*(all_stateseqs + sequence_start + t)); //extract int is right??
        }
    }
    
    PyObject *result = PyDict_New();

    npy_intp dims1[] = {1, bigT};
    PyObject *all_stateseqs_arr = (PyObject *) PyArray_SimpleNewFromData(2, dims1, NPY_INT, all_stateseqs);
    PyObject *capsule1 = PyCapsule_New(all_stateseqs, NULL, capsule_cleanup);
    PyArray_SetBaseObject((PyArrayObject *) all_stateseqs_arr, capsule1);

    npy_intp dims2[] = {num_subparts, bigT};
    PyObject *all_data_arr = (PyObject *) PyArray_SimpleNewFromData(2, dims2, NPY_INT, all_data);
    PyObject *capsule2 = PyCapsule_New(all_data, NULL, capsule_cleanup);
    PyArray_SetBaseObject((PyArrayObject *) all_data_arr, capsule2);

    PyDict_SetItemString(result, "stateseqs", all_stateseqs_arr);
    PyDict_SetItemString(result, "data", all_data_arr);

    Py_XDECREF(resources);
    Py_XDECREF(starts);
    Py_XDECREF(lengths);
    Py_XDECREF(all_stateseqs_arr);
    Py_XDECREF(all_data_arr);

    for (int i = 0; i < 4; i++)
        Py_XDECREF(*DM_PTRS[i]);

    return(result);
}

static PyMethodDef synthetic_data_helper_Methods[] = {
    {"create_synthetic_data",  run, METH_VARARGS,
     "Helper for creating synthetic data from true model"},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};


static struct PyModuleDef synthetic_data_helper_module = {
   PyModuleDef_HEAD_INIT,
   "synthetic_data_helper",   /* name of module */
   NULL, /* module documentation, may be NULL */
   -1,       /* size of per-interpreter state of the module,
                or -1 if the module keeps state in global variables. */
   synthetic_data_helper_Methods
};

PyMODINIT_FUNC PyInit_synthetic_data_helper() {
    return PyModule_Create(&synthetic_data_helper_module);
}

