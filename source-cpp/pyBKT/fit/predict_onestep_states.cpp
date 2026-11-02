#define NPY_NO_DEPRECATED_API NPY_1_11_API_VERSION

#include <array>
#include <climits>
#include <cstdint>
#include <string>
#include <vector>
#include <Eigen/Core>
#include <Python.h>
#include <numpy/ndarrayobject.h>

using namespace Eigen;

using Int64Vector = Map<const Matrix<int64_t, Dynamic, 1>>;
using DoubleVector = Map<const VectorXd>;
using RowMajor2Xd = Map<const Matrix<double, 2, Dynamic, RowMajor>, Unaligned, OuterStride<>>;

static Int64Vector int64_vector(PyArrayObject *arr) {
    return {static_cast<const int64_t *>(PyArray_DATA(arr)), PyArray_SIZE(arr)};
}

static DoubleVector double_vector(PyArrayObject *arr) {
    return {static_cast<const double *>(PyArray_DATA(arr)), PyArray_SIZE(arr)};
}

// NumPy's C API passes arrays as PyObject *, and every PyArrayObject begins with that header.
static PyArrayObject *as_array(PyObject *object) {
    return reinterpret_cast<PyArrayObject *>(object); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
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
        // The loop below reads resources only at t < T - 1.
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

static PyObject *run(PyObject * /* module */, PyObject *args) {
    long parallel = 0;
    if (PyTuple_GET_SIZE(args) != 4 || !parse_int(PyTuple_GET_ITEM(args, 3), parallel)) {
        set_value_error("Error parsing arguments.");
        return nullptr;
    }
    PyObject *data_ptr = PyTuple_GET_ITEM(args, 0);
    PyObject *model_ptr = PyTuple_GET_ITEM(args, 1);
    PyObject *fwd_msgs = PyTuple_GET_ITEM(args, 2);
    const bool is_parallel = parallel != 0;

    OwnedReference<PyArrayObject> forward_messages;
    forward_messages.pointer = to_array(fwd_msgs, NPY_FLOAT);

    OwnedReference<PyArrayObject> alldata, allresources, starts, lengths, learns, forgets, guesses, slips;
    struct Input {
        const char *name;
        PyObject *source;
        OwnedReference<PyArrayObject> *target;
        bool is_integer;
    };
    const std::array<Input, 8> inputs = {{
        {"data", data_ptr, &alldata, false},
        {"resources", data_ptr, &allresources, true},
        {"starts", data_ptr, &starts, true},
        {"lengths", data_ptr, &lengths, true},
        {"learns", model_ptr, &learns, false},
        {"forgets", model_ptr, &forgets, false},
        {"guesses", model_ptr, &guesses, false},
        {"slips", model_ptr, &slips, false},
    }};
    for (const Input &input : inputs) {
        PyObject *value = PyDict_GetItemString(input.source, input.name);
        if (input.is_integer)
            input.target->pointer = to_integer_array(value, NPY_INT64, input.name);
        else
            input.target->pointer = to_array(value, input.target == &alldata ? NPY_INT : NPY_FLOAT);
        if (input.target->pointer == nullptr)
            return nullptr;
    }
    if (!check_inputs(alldata.pointer, allresources.pointer, starts.pointer, lengths.pointer, learns.pointer, forgets.pointer,
                      guesses.pointer, slips.pointer))
        return nullptr;
    const double prior = PyFloat_AsDouble(PyDict_GetItemString(model_ptr, "prior"));

    const Py_ssize_t bigT = PyArray_DIM(alldata.pointer, 1);
    Py_ssize_t num_sequences = PyArray_DIM(starts.pointer, 0);
    const Py_ssize_t num_resources = PyArray_DIM(learns.pointer, 0);

    Array2d initial_distn;
    initial_distn << 1 - prior, prior;

    const DoubleVector learn_values = double_vector(learns.pointer);
    const DoubleVector forget_values = double_vector(forgets.pointer);
    // Resource n's 2x2 transition matrix fills columns 2n and 2n+1, so in column-major
    // storage it is the 4 values of column n of As_blocks.
    // A vector, not a MatrixXd: clang-analyzer 22, though not 23, reports a false out-of-bounds write into a run-time-sized MatrixXd here.
    std::vector<double> As_storage(4 * num_resources);
    Map<MatrixXd> As(As_storage.data(), 2, 2 * num_resources);
    Map<Matrix<double, 4, Dynamic>> As_blocks(As_storage.data(), 4, num_resources);
    As_blocks.row(0) = (1 - learn_values.head(num_resources).array()).matrix().transpose();
    As_blocks.row(1) = learn_values.head(num_resources).transpose();
    As_blocks.row(2) = forget_values.head(num_resources).transpose();
    As_blocks.row(3) = (1 - forget_values.head(num_resources).array()).matrix().transpose();

    const RowMajor2Xd forward_input(static_cast<const double *>(PyArray_DATA(forward_messages.pointer)), 2,
                                    PyArray_DIM(forward_messages.pointer, 1), OuterStride<>(PyArray_DIM(forward_messages.pointer, 1)));
    // The kernel reads the row-major (2, bigT) values in column-major order, as E_step writes them.
    const Matrix<double, 2, Dynamic, RowMajor> forward_compact = forward_input.leftCols(bigT);
    Map<const MatrixXd> forward_all(forward_compact.data(), 2, bigT);

    std::array<npy_intp, 2> dims = {2, bigT};
    PyObject *all_predictions_arr = PyArray_SimpleNew(2, dims.data(), NPY_DOUBLE);
    if (all_predictions_arr == nullptr)
        return nullptr;
    // Column-major 2 x bigT in a C-order (2, bigT) array; predict_onestep.run reorders it.
    Map<MatrixXd> predictions_all(static_cast<double *>(PyArray_DATA(as_array(all_predictions_arr))), 2, bigT);

    // Not const: GCC 8 rejects const variables in a shared clause, and default(none) needs them listed.
    Int64Vector start_values = int64_vector(starts.pointer);
    Int64Vector length_values = int64_vector(lengths.pointer);
    Int64Vector resource_values = int64_vector(allresources.pointer);

    #pragma omp parallel for if(is_parallel) default(none) \
        shared(num_sequences, start_values, length_values, resource_values, forward_all, predictions_all, As, initial_distn)
    for (Py_ssize_t sequence_index = 0; sequence_index < num_sequences; sequence_index++) {
        // NOTE: -1 because Matlab indexing starts at 1
        const int64_t sequence_start = start_values(sequence_index) - 1;
        const int64_t T = length_values(sequence_index);
        const Map<const MatrixXd> forward(forward_all.col(sequence_start).data(), 2, T);
        Map<MatrixXd> predictions(predictions_all.col(sequence_start).data(), 2, T);

        predictions.col(0) = initial_distn;
        for (int64_t t = 0; t < T - 1; t++) {
            const int64_t resource = resource_values(sequence_start + t);
            predictions.col(t + 1) = As.block(0, 2 * (resource - 1), 2, 2) * forward.col(t);
        }
    }

    return all_predictions_arr;
}

PyMODINIT_FUNC PyInit_predict_onestep_states() {
    import_array();
    static std::array<PyMethodDef, 2> methods = {{
        {"run", run, METH_VARARGS, "Generates predictions for BKT model"},
        {nullptr, nullptr, 0, nullptr},
    }};
    static PyModuleDef module = {
        PyModuleDef_HEAD_INIT, "predict_onestep_states", nullptr, -1, methods.data(), nullptr, nullptr, nullptr, nullptr,
    };
    return PyModule_Create(&module);
}
