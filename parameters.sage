#!/usr/bin/sage
# -*- mode: python ; -*-

from sage.all import *
import hashlib
import itertools
import datetime

from constants import *

load('anemoi.sage')

def anemoi_selected_instances():

    # accumulating selected Anemoi instances
    A = []
    
    # - 256-bit security level instantiations
    # -- BLS12_381_SCALRFIELD
    # --- 1 col
    A_BLS_12_381_SCALARFIELD_1_COL_256_BITS = AnemoiPermutation(
        q=BLS12_381_SCALARFIELD,
        n_cols=1,
        security_level=256
    )
    A.append(
        ("A_BLS_12_381_SCALARFIELD_1_COL_256_BITS",
         A_BLS_12_381_SCALARFIELD_1_COL_256_BITS))
    # --- 2 col    
    A_BLS_12_381_SCALARFIELD_2_COL_256_BITS = AnemoiPermutation(
        q=BLS12_381_SCALARFIELD,
        n_cols=2,
        security_level=256
    )
    A.append(
        ("A_BLS_12_381_SCALARFIELD_2_COL_256_BITS",
         A_BLS_12_381_SCALARFIELD_2_COL_256_BITS))    
    # --- 3 col    
    A_BLS_12_381_SCALARFIELD_3_COL_256_BITS = AnemoiPermutation(
        q=BLS12_381_SCALARFIELD,
        n_cols=3,
        security_level=256
    )
    A.append(
        ("A_BLS_12_381_SCALARFIELD_3_COL_256_BITS",
         A_BLS_12_381_SCALARFIELD_3_COL_256_BITS))    
    # ---4 col    
    A_BLS_12_381_SCALARFIELD_4_COL_256_BITS = AnemoiPermutation(
        q=BLS12_381_SCALARFIELD,
        n_cols=4,
        security_level=256
    )
    A.append(
        ("A_BLS_12_381_SCALARFIELD_4_COL_256_BITS",
         A_BLS_12_381_SCALARFIELD_4_COL_256_BITS))
    return A
    
def output_parameters():
    instances = anemoi_selected_instances()
    for i in range(len(instances)):
        # string name
        A_str = instances[i][0]
        # actual instance that can be called as A[i][1].*
        A = instances[i][1]
        zero = 0
        width = 100
        print("------------------------------------------------------")
        print("instance         : {}".format(A_str))
        print("prime field      : {}".format(A.prime_field))
        print("Fr modulus       : {}".format(A.q))
        print("n_cols           : {}".format(A.n_cols))
        print("n_rounds         : {}".format(A.n_rounds))
        print("security level   : {}".format(A.security_level))
        print("mult generator g : {}".format(A.g))
        print("Q power          : {}".format(A.QUAD))
        print("alpha            : {}".format(A.alpha))
        print("alpha_inv        : {}".format(A.alpha_inv))
        print("beta             : {}".format(A.beta))
        print("gamma            : {}".format(zero))
        print("delta            : {}".format(A.delta))
        print("matrix M         :\n{}".format(A.mat))
        print("constants C      :\n{}".format(A.C))
        print("constants D      :\n{}".format(A.D))
        return instances

# same as output_parameters() but stores parameters to file
def output_parameters_to_file():
    instances = anemoi_selected_instances()
    f = open("/tmp/parameters.txt", "w")
    e = datetime.datetime.now()
    f.write("This file was automatically generated with SAGE script parameters.sage on %s/%s/%s at %s:%s:%s\n" % (e.day, e.month, e.year, e.hour, e.minute, e.second))
    for i in range(len(instances)):
        A_str = instances[i][0]
        A = instances[i][1]
        zero = 0
        width = 100
        f.write("------------------------------------------------------")
        f.write("instance         : {}\n".format(A_str))
        f.write("prime field      : {}\n".format(A.prime_field))
        f.write("Fr modulus       : {}\n".format(A.q))
        f.write("n_cols           : {}\n".format(A.n_cols))
        f.write("n_rounds         : {}\n".format(A.n_rounds))
        f.write("security level   : {}\n".format(A.security_level))
        f.write("mult generator g : {}\n".format(A.g))
        f.write("Q power          : {}\n".format(A.QUAD))
        f.write("alpha            : {}\n".format(A.alpha))
        f.write("alpha_inv        : {}\n".format(A.alpha_inv))
        f.write("beta             : {}\n".format(A.beta))
        f.write("gamma            : {}\n".format(zero))
        f.write("delta            : {}\n".format(A.delta))
        f.write("matrix M         :\n{}\n".format(A.mat))
        f.write("constants C      :\n{}\n".format(A.C))
        f.write("constants D      :\n{}\n".format(A.D))

def output_constants_in_libsnark_format_to_file(instances):
    f = open("/tmp/constants.txt", "w")
    e = datetime.datetime.now()
    f.write("// Automatically generated with SAGE script parameters.sage on %s/%s/%s at %s:%s:%s\n" % (e.day, e.month, e.year, e.hour, e.minute, e.second))
    i_str = ["one", "two", "three", "four", "five", "six"]
    for i in range(len(instances)):
        A_str = instances[i][0]
        A = instances[i][1]
        f.write("// C constants for L = {} columns\n".format(i+1))
        f.write("const std::vector<std::vector<FieldT>> anemoi_parameters<ppT>::C_constants_col_{} = ".format(i_str[i]))
        f.write("{\n")
        for iround in range(len(A.C)):
            f.write("{")
            for icol in range(len(A.C[iround])):
                f.write("FieldT(\"{}\")".format(A.C[iround][icol]))
                if icol < (len(A.C[iround]) - 1):
                    f.write(", ")
            f.write("}")
            if iround < (len(A.C) - 1):
                f.write(",\n")
        f.write("\n};\n")
        f.write("// D constants for L = {} columns\n".format(i+1))
        f.write("const std::vector<std::vector<FieldT>> anemoi_parameters<ppT>::D_constants_col_{} = ".format(i_str[i]))
        f.write("{\n")
        for iround in range(len(A.D)):
            f.write("{")
            for icol in range(len(A.D[iround])):
                f.write("FieldT(\"{}\")".format(A.D[iround][icol]))
                if icol < (len(A.D[iround]) - 1):
                    f.write(", ")
            f.write("}")
            if iround < (len(A.D) - 1):
                f.write(",\n")
        f.write("\n};\n")
        
                  
if __name__ == "__main__":
    A = output_parameters()
    output_parameters_to_file()
    output_constants_in_libsnark_format_to_file(A)
