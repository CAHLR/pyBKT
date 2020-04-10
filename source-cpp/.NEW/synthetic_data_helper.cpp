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
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <boost/python/ptr.hpp>
#include <Python.h>
#include <numpy/ndarrayobject.h>

using namespace Eigen;
using namespace std;
using namespace boost::python;

namespace np = boost::python::numpy;
namespace p = boost::python;

dict create_synthetic_data(dict& model, numpy::ndarray& starts, numpy::ndarray& lengths, numpy::ndarray& resources){
    //TODO: check if parameters are null.
    //TODO: check that dicts have the required members.
    //TODO: check that all parameters have the right sizes.
    //TODO: i'm not sending any error messages.
    
    numpy::ndarray learns = extract<numpy::ndarray>(model["learns"]);
    int num_resources = len(learns);

    numpy::ndarray forgets = extract<numpy::ndarray>(model["forgets"]);
    numpy::ndarray guesses = extract<numpy::ndarray>(model["guesses"]);
    
    numpy::ndarray slips = extract<numpy::ndarray>(model["slips"]);
    int num_subparts = len(slips);
    
    Vector2d initial_distn;
    double prior = extract<double>(model["prior"]);
    initial_distn << 1-prior, prior;
    
    MatrixXd As(2, 2*num_resources);
    for (int n=0; n<num_resources; n++) {
        double learn = extract<double>(learns[n]);
        double forget = extract<double>(forgets[n]);
        As.col(2*n) << 1-learn, learn;
        As.col(2*n+1) << forget, 1-forget;
    }
    
    int num_sequences = len(starts);
    
    int64_t bigT = 0;
    for (int k=0; k<num_sequences; k++) {
        bigT += extract<int64_t>(lengths[k]); //extract this as int??
    }
    
    //// outputs
    int* all_stateseqs = new int[bigT];
    int* all_data = new int[num_subparts * bigT]; //used to be int8_t
    *all_data = 0;
    dict result;
    
    /* COMPUTATION */
    
    for (int sequence_index=0; sequence_index < num_sequences; sequence_index++) {
        int64_t sequence_start = extract<int64_t>(starts[sequence_index]) - 1; //should i extract these as ints?
        int64_t T = extract<int64_t>(lengths[sequence_index]);
        
        Vector2d nextstate_distr = initial_distn;

        for (int t=0; t<T; t++) {
            *(all_stateseqs + sequence_start + t) = nextstate_distr(0) < ((double) rand()) / ((double) RAND_MAX); //always all_stateseqs[0]?
            for (int n=0; n<num_subparts; n++) {
                *(all_data + n * (bigT) + sequence_start + t) = ((*(all_stateseqs + sequence_start + t)) ? extract<double>(slips[n]) : (1-extract<double>(guesses[n]))) < (((double) rand()) / ((double) RAND_MAX));
            }
            
            nextstate_distr = As.col(2*(extract<int64_t>(resources[sequence_start + t])-1)+*(all_stateseqs + sequence_start + t)); //extract int is right??
        }
    }
    
    //wrapping results in numpy objects.
    np::ndarray all_stateseqs_arr = np::from_data(all_stateseqs, np::dtype::get_builtin<int>(), p::make_tuple(1, bigT), p::make_tuple(4 * bigT, 4), p::object()).copy();
    
    np::ndarray all_data_arr = np::from_data(all_data, np::dtype::get_builtin<int>(), p::make_tuple(num_subparts, bigT), p::make_tuple(4 * bigT, 4), p::object()).copy();
    
    result["stateseqs"] = all_stateseqs_arr;
    result["data"] = all_data_arr;

    return(result);
}

BOOST_PYTHON_MODULE(synthetic_data_helper){
    //import_array();
    Py_Initialize();
    np::initialize();
    /*if(PyArray_API == NULL)
	{
	    import_array();
	}*/
    def("create_synthetic_data", create_synthetic_data);
    
}
