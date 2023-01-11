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
    
    # - 128-bit security level instantiations
    # -- BLS12_381_SCALRFIELD
    # --- 1 col
    A_BLS_12_381_SCALARFIELD_1_COL_128_BITS = AnemoiPermutation(
        q=BLS12_381_SCALARFIELD,
        n_cols=1,
        security_level=128
    )
    A.append(
        ("A_BLS_12_381_SCALARFIELD_1_COL_128_BITS",
         A_BLS_12_381_SCALARFIELD_1_COL_128_BITS))
    # --- 2 col    
    A_BLS_12_381_SCALARFIELD_2_COL_128_BITS = AnemoiPermutation(
        q=BLS12_381_SCALARFIELD,
        n_cols=2,
        security_level=128
    )
    A.append(
        ("A_BLS_12_381_SCALARFIELD_2_COL_128_BITS",
         A_BLS_12_381_SCALARFIELD_2_COL_128_BITS))    
    # --- 3 col    
    A_BLS_12_381_SCALARFIELD_3_COL_128_BITS = AnemoiPermutation(
        q=BLS12_381_SCALARFIELD,
        n_cols=3,
        security_level=128
    )
    A.append(
        ("A_BLS_12_381_SCALARFIELD_3_COL_128_BITS",
         A_BLS_12_381_SCALARFIELD_3_COL_128_BITS))    
    # ---4 col    
    A_BLS_12_381_SCALARFIELD_4_COL_128_BITS = AnemoiPermutation(
        q=BLS12_381_SCALARFIELD,
        n_cols=4,
        security_level=128
    )
    A.append(
        ("A_BLS_12_381_SCALARFIELD_4_COL_128_BITS",
         A_BLS_12_381_SCALARFIELD_4_COL_128_BITS))
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
    f = open("/tmp/anemoi-parameters.txt", "w")
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
                  
if __name__ == "__main__":
    A = output_parameters()
    output_parameters_to_file()
