//
//  synthetic_data_helper.cpp
//  synthetic_data_helper
//
//  Created by Cristián Garay on 10/15/16.
//  Revised and edited by Anirudhan Badrinath on 27/02/20.
//

#define NPY_NO_DEPRECATED_API NPY_1_11_API_VERSION

#include <array>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <Eigen/Core>
#include <Python.h>
#include <numpy/ndarrayobject.h>

using namespace Eigen;

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

static PyArrayObject *to_array(PyObject *obj, int minimum_type) {
    return as_array(PyArray_FROM_OTF(obj, PyArray_ObjectType(obj, minimum_type), NPY_ARRAY_IN_ARRAY));
}

// A uniform draw in [0, 1] from the C library generator.
static double uniform_draw() {
    return static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
}

static PyObject* run(PyObject * /* module */, PyObject * args) {
    //TODO: check that dicts have the required members.
    if (PyTuple_GET_SIZE(args) != 4) {
        PyErr_SetString(PyExc_ValueError, "Error parsing arguments.");
        return nullptr;
    }
    PyObject *model_ptr = PyTuple_GET_ITEM(args, 0);

    OwnedReference<PyArrayObject> starts, lengths, resources, learns, forgets, guesses, slips;
    starts.pointer = to_array(PyTuple_GET_ITEM(args, 1), NPY_INT64);
    lengths.pointer = to_array(PyTuple_GET_ITEM(args, 2), NPY_INT64);
    resources.pointer = to_array(PyTuple_GET_ITEM(args, 3), NPY_INT64);

    struct Input {
        const char *name;
        OwnedReference<PyArrayObject> *target;
    };
    const std::array<Input, 4> inputs = {{
        {"learns", &learns}, {"forgets", &forgets}, {"guesses", &guesses}, {"slips", &slips},
    }};
    for (const Input &input : inputs)
        input.target->pointer = to_array(PyDict_GetItemString(model_ptr, input.name), NPY_FLOAT); // hack to force correct type
    double prior = PyFloat_AsDouble(PyDict_GetItemString(model_ptr, "prior"));

    Index num_subparts = PyArray_DIM(slips.pointer, 0);
    int num_sequences = static_cast<int>(PyArray_DIM(starts.pointer, 0));
    Index num_resources = PyArray_DIM(learns.pointer, 0);

    Vector2d initial_distn;
    initial_distn << 1-prior, prior;

    const DoubleVector learn_values = double_vector(learns.pointer);
    const DoubleVector forget_values = double_vector(forgets.pointer);
    // A vector, not a MatrixXd: clang-analyzer 22, though not 23, reports a false out-of-bounds write into a run-time-sized MatrixXd here.
    std::vector<double> As_storage(4 * num_resources);
    Map<MatrixXd> As(As_storage.data(), 2, 2*num_resources);
    for (Index n=0; n<num_resources; n++) {
        double learn = learn_values(n);
        double forget = forget_values(n);
        As.col(2*n) << 1-learn, learn;
        As.col(2*n+1) << forget, 1-forget;
    }

    const Int64Vector start_values = int64_vector(starts.pointer);
    const Int64Vector length_values = int64_vector(lengths.pointer);
    const Int64Vector resource_values = int64_vector(resources.pointer);
    int64_t bigT = 0;
    for (int k=0; k<num_sequences; k++) {
        bigT += length_values(k);
    }

    //// outputs
    std::array<npy_intp, 2> stateseqs_dims = {1, bigT};
    std::array<npy_intp, 2> data_dims = {num_subparts, bigT};
    OwnedReference<PyObject> all_stateseqs_arr, all_data_arr;
    all_stateseqs_arr.pointer = PyArray_ZEROS(2, stateseqs_dims.data(), NPY_INT, 0);
    all_data_arr.pointer = PyArray_ZEROS(2, data_dims.data(), NPY_INT, 0);
    if (all_stateseqs_arr.pointer == nullptr || all_data_arr.pointer == nullptr)
        return nullptr;
    Map<Matrix<int, Dynamic, 1>> all_stateseqs(static_cast<int *>(PyArray_DATA(as_array(all_stateseqs_arr.pointer))), bigT);
    Map<Matrix<int, Dynamic, Dynamic, RowMajor>> all_data(static_cast<int *>(PyArray_DATA(as_array(all_data_arr.pointer))),
                                                          num_subparts, bigT);

    /* COMPUTATION */

    const DoubleVector guess_values = double_vector(guesses.pointer);
    const DoubleVector slip_values = double_vector(slips.pointer);
    for (int sequence_index=0; sequence_index < num_sequences; sequence_index++) {
        int64_t sequence_start = start_values(sequence_index) - 1;
        int64_t T = length_values(sequence_index);

        Vector2d nextstate_distr = initial_distn;

        for (int t=0; t<T; t++) {
            all_stateseqs(sequence_start + t) = nextstate_distr(0) < uniform_draw();
            for (Index n=0; n<num_subparts; n++) {
                all_data(n, sequence_start + t) = (all_stateseqs(sequence_start + t) ? slip_values(n) : (1-guess_values(n))) < uniform_draw();
            }

            nextstate_distr = As.col(2*(resource_values(sequence_start + t)-1)+all_stateseqs(sequence_start + t));
        }
    }

    PyObject *result = PyDict_New();
    PyDict_SetItemString(result, "stateseqs", all_stateseqs_arr.pointer);
    PyDict_SetItemString(result, "data", all_data_arr.pointer);

    return(result);
}

PyMODINIT_FUNC PyInit_synthetic_data_helper() {
    import_array();
    static std::array<PyMethodDef, 2> methods = {{
        {"create_synthetic_data", run, METH_VARARGS, "Helper for creating synthetic data from true model"},
        {nullptr, nullptr, 0, nullptr},
    }};
    static PyModuleDef module = {
        PyModuleDef_HEAD_INIT, "synthetic_data_helper", nullptr, -1, methods.data(), nullptr, nullptr, nullptr, nullptr,
    };
    return PyModule_Create(&module);
}
