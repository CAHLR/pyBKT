//
//  synthetic_data_helper.cpp
//  synthetic_data_helper
//
//  Created by Cristián Garay on 10/15/16.
//  Copyright © 2016 Cristian Garay. All rights reserved.
//

#define NPY_NO_DEPRECATED_API NPY_1_11_API_VERSION

#include <iostream>
#include <stdint.h>
#include <Eigen/Core>
#include <boost/python.hpp>
#include <boost/python/numeric.hpp>
#include <boost/python/ptr.hpp>
#include <Python.h>
#include <numpy/ndarrayobject.h>

using namespace Eigen;
using namespace std;
using namespace boost::python;

dict create_synthetic_data(dict& model, numeric::array& starts, numeric::array& lengths, numeric::array& resources){
    //TODO: check if parameters are null.
    //TODO: check that dicts have the required members.
    //TODO: check that all parameters have the right sizes.
    //TODO: i'm not sending any error messages.
    
    //TODO: is learns always an array? originally it was a horizontal array, im treating it as vertical.
    
    numeric::array learns = extract<numeric::array>(model["learns"]);
    int num_resources = len(learns); //he got it from max between colums and rows of this array.
    
    numeric::array slips = extract<numeric::array>(model["slips"]);
    int num_subparts = len(slips); //he got it from max between colums and rows of this array.
    
    Vector2d initial_distn;
    double prior = extract<double>(model["prior"]);
    initial_distn << 1-prior, prior;
    
    numeric::array forgets = extract<numeric::array>(model["forgets"]);
    numeric::array guesses = extract<numeric::array>(model["guesses"]);
    
    MatrixXd As(2, 2*num_resources);
    for (int n=0; n<num_resources; n++) {
        double learn = extract<double>(learns[n]);
        double forget = extract<double>(forgets[n]);
        As.col(2*n) << 1-learn, learn;
        As.col(2*n+1) << forget, 1-forget;
    }
    
    int num_sequences = len(starts); //he got it from max between colums and rows of this array.
    
    int bigT = 0;
    for (int k=0; k<num_sequences; k++) {
        bigT += (int) extract<double>(lengths[k]); //maybe this should come as int?
    }
    
    //// outputs
    int8_t all_stateseqs[bigT]; //used to be 1xbigT
    int8_t all_data[num_subparts][bigT];
    all_data[0][0] = 0;
    dict result;
    
    /* COMPUTATION */
    
    for (int sequence_index=0; sequence_index < num_sequences; sequence_index++) {
        int32_t sequence_start = (int32_t) extract<double>(starts[sequence_index]) - 1; // use to have -1 because Matlab indexing starts at 1.
        int32_t T = (int32_t) extract<double>(lengths[sequence_index]);
        
        Vector2d nextstate_distr = initial_distn;
        for (int t=0; t<T; t++) {
            all_stateseqs[sequence_start + t] = nextstate_distr(0) < ((double) rand()) / ((double) RAND_MAX);
            for (int n=0; n<num_subparts; n++) {
                all_data[num_subparts][n+num_subparts*t] = (all_stateseqs[sequence_start + t] ? extract<double>(slips[n]) : (1-extract<double>(guesses[n]))) < ((double) rand()) / ((double) RAND_MAX);
            }
            
            nextstate_distr = As.col(2*(extract<int>(resources[sequence_start + t])-1)+all_stateseqs[sequence_start + t]); //TODO: extract int is right??
        }
    }
    
    //wrapping results in numpy objects.
    npy_intp all_stateseqs_dims[1] = {bigT}; //TODO: just put directly this array into the PyArray_SimpleNewFromData function?
    PyObject * all_stateseqs_pyObj = PyArray_SimpleNewFromData(1, all_stateseqs_dims, NPY_INT8, all_stateseqs);
    boost::python::handle<> all_stateseqs_handle( all_stateseqs_pyObj );
    boost::python::numeric::array all_stateseqs_handle_arr( all_stateseqs_handle );
    
    npy_intp all_data_dims[2] = {num_subparts, bigT}; //TODO: just put directly this array into the PyArray_SimpleNewFromData function?
    PyObject * all_data_pyObj = PyArray_SimpleNewFromData(2, all_data_dims, NPY_INT8, all_data);
    boost::python::handle<> all_data_handle( all_data_pyObj );
    boost::python::numeric::array all_data_arr( all_data_handle );
    
    result["stateseqs"] = all_stateseqs_handle_arr;
    result["data"] = all_data_arr;
    return(result);
    
}

/*int init_numpy(){
	import_array();
}*/

BOOST_PYTHON_MODULE(synthetic_data_helper){
    import_array();
    //init_numpy();
    numeric::array::set_module_and_type("numpy", "ndarray");
    
    def("create_synthetic_data", create_synthetic_data);
    
}