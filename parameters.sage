#!/usr/bin/sage
# -*- mode: python ; -*-

from sage.all import *
import hashlib
import itertools
import datetime
from constants import *

load('anemoi.sage')

def anemoi_get_nrounds(A):
    nrounds = []
    for i in range(len(A)):
        nrounds.append(A[i][1].n_rounds)
    return nrounds
    
def anemoi128_instances_bls12_381(A):

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
    
def anemoi256_instances_bls12_381(A):

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
    
def anemoi_bls12_381_nrounds():
    A = []
    anemoi128_instances_bls12_381(A)
    nrounds128 = anemoi_get_nrounds(A)
    A = []
    anemoi256_instances_bls12_381(A)
    nrounds256 = anemoi_get_nrounds(A)
    return nrounds128, nrounds256
    
def anemoi128_instances_bls12_377(A):

    # - 128-bit security level instantiations
    # -- BLS12_377_SCALRFIELD
    # --- 1 col
    A_BLS_12_377_SCALARFIELD_1_COL_128_BITS = AnemoiPermutation(
        q=BLS12_377_SCALARFIELD,
        n_cols=1,
        security_level=128
    )
    A.append(
        ("A_BLS_12_377_SCALARFIELD_1_COL_128_BITS",
         A_BLS_12_377_SCALARFIELD_1_COL_128_BITS))
    # --- 2 col    
    A_BLS_12_377_SCALARFIELD_2_COL_128_BITS = AnemoiPermutation(
        q=BLS12_377_SCALARFIELD,
        n_cols=2,
        security_level=128
    )
    A.append(
        ("A_BLS_12_377_SCALARFIELD_2_COL_128_BITS",
         A_BLS_12_377_SCALARFIELD_2_COL_128_BITS))    
    # --- 3 col    
    A_BLS_12_377_SCALARFIELD_3_COL_128_BITS = AnemoiPermutation(
        q=BLS12_377_SCALARFIELD,
        n_cols=3,
        security_level=128
    )
    A.append(
        ("A_BLS_12_377_SCALARFIELD_3_COL_128_BITS",
         A_BLS_12_377_SCALARFIELD_3_COL_128_BITS))    
    # ---4 col    
    A_BLS_12_377_SCALARFIELD_4_COL_128_BITS = AnemoiPermutation(
        q=BLS12_377_SCALARFIELD,
        n_cols=4,
        security_level=128
    )
    A.append(
        ("A_BLS_12_377_SCALARFIELD_4_COL_128_BITS",
         A_BLS_12_377_SCALARFIELD_4_COL_128_BITS))
    
def anemoi256_instances_bls12_377(A):

    # - 256-bit security level instantiations
    # -- BLS12_377_SCALRFIELD
    # --- 1 col
    A_BLS_12_377_SCALARFIELD_1_COL_256_BITS = AnemoiPermutation(
        q=BLS12_377_SCALARFIELD,
        n_cols=1,
        security_level=256
    )
    A.append(
        ("A_BLS_12_377_SCALARFIELD_1_COL_256_BITS",
         A_BLS_12_377_SCALARFIELD_1_COL_256_BITS))
    # --- 2 col    
    A_BLS_12_377_SCALARFIELD_2_COL_256_BITS = AnemoiPermutation(
        q=BLS12_377_SCALARFIELD,
        n_cols=2,
        security_level=256
    )
    A.append(
        ("A_BLS_12_377_SCALARFIELD_2_COL_256_BITS",
         A_BLS_12_377_SCALARFIELD_2_COL_256_BITS))    
    # --- 3 col    
    A_BLS_12_377_SCALARFIELD_3_COL_256_BITS = AnemoiPermutation(
        q=BLS12_377_SCALARFIELD,
        n_cols=3,
        security_level=256
    )
    A.append(
        ("A_BLS_12_377_SCALARFIELD_3_COL_256_BITS",
         A_BLS_12_377_SCALARFIELD_3_COL_256_BITS))    
    # ---4 col    
    A_BLS_12_377_SCALARFIELD_4_COL_256_BITS = AnemoiPermutation(
        q=BLS12_377_SCALARFIELD,
        n_cols=4,
        security_level=256
    )
    A.append(
        ("A_BLS_12_377_SCALARFIELD_4_COL_256_BITS",
         A_BLS_12_377_SCALARFIELD_4_COL_256_BITS))

def anemoi_bls12_377_nrounds():
    A = []
    anemoi128_instances_bls12_377(A)
    nrounds128 = anemoi_get_nrounds(A)
    A = []
    anemoi256_instances_bls12_377(A)
    nrounds256 = anemoi_get_nrounds(A)
    return nrounds128, nrounds256
        
def anemoi128_instances_mnt4(A):

    # - 128-bit security level instantiations
    # -- MNT4_SCALRFIELD
    # --- 1 col
    A_MNT4_SCALARFIELD_1_COL_128_BITS = AnemoiPermutation(
        q=MNT4_SCALARFIELD,
        n_cols=1,
        security_level=128
    )
    A.append(
        ("A_MNT4_SCALARFIELD_1_COL_128_BITS",
         A_MNT4_SCALARFIELD_1_COL_128_BITS))
    # --- 2 col    
    A_MNT4_SCALARFIELD_2_COL_128_BITS = AnemoiPermutation(
        q=MNT4_SCALARFIELD,
        n_cols=2,
        security_level=128
    )
    A.append(
        ("A_MNT4_SCALARFIELD_2_COL_128_BITS",
         A_MNT4_SCALARFIELD_2_COL_128_BITS))    
    # --- 3 col    
    A_MNT4_SCALARFIELD_3_COL_128_BITS = AnemoiPermutation(
        q=MNT4_SCALARFIELD,
        n_cols=3,
        security_level=128
    )
    A.append(
        ("A_MNT4_SCALARFIELD_3_COL_128_BITS",
         A_MNT4_SCALARFIELD_3_COL_128_BITS))    
    # ---4 col    
    A_MNT4_SCALARFIELD_4_COL_128_BITS = AnemoiPermutation(
        q=MNT4_SCALARFIELD,
        n_cols=4,
        security_level=128
    )
    A.append(
        ("A_MNT4_SCALARFIELD_4_COL_128_BITS",
         A_MNT4_SCALARFIELD_4_COL_128_BITS))
    
def anemoi256_instances_mnt4(A):

    # - 256-bit security level instantiations
    # -- MNT4_SCALRFIELD
    # --- 1 col
    A_MNT4_SCALARFIELD_1_COL_256_BITS = AnemoiPermutation(
        q=MNT4_SCALARFIELD,
        n_cols=1,
        security_level=256
    )
    A.append(
        ("A_MNT4_SCALARFIELD_1_COL_256_BITS",
         A_MNT4_SCALARFIELD_1_COL_256_BITS))
    # --- 2 col    
    A_MNT4_SCALARFIELD_2_COL_256_BITS = AnemoiPermutation(
        q=MNT4_SCALARFIELD,
        n_cols=2,
        security_level=256
    )
    A.append(
        ("A_MNT4_SCALARFIELD_2_COL_256_BITS",
         A_MNT4_SCALARFIELD_2_COL_256_BITS))    
    # --- 3 col    
    A_MNT4_SCALARFIELD_3_COL_256_BITS = AnemoiPermutation(
        q=MNT4_SCALARFIELD,
        n_cols=3,
        security_level=256
    )
    A.append(
        ("A_MNT4_SCALARFIELD_3_COL_256_BITS",
         A_MNT4_SCALARFIELD_3_COL_256_BITS))    
    # ---4 col    
    A_MNT4_SCALARFIELD_4_COL_256_BITS = AnemoiPermutation(
        q=MNT4_SCALARFIELD,
        n_cols=4,
        security_level=256
    )
    A.append(
        ("A_MNT4_SCALARFIELD_4_COL_256_BITS",
         A_MNT4_SCALARFIELD_4_COL_256_BITS))
    
def anemoi_mnt4_nrounds():
    A = []
    anemoi128_instances_mnt4(A)
    nrounds128 = anemoi_get_nrounds(A)
    A = []
    anemoi256_instances_mnt4(A)
    nrounds256 = anemoi_get_nrounds(A)
    return nrounds128, nrounds256
        
def anemoi128_instances_mnt6(A):

    # - 128-bit security level instantiations
    # -- MNT6_SCALRFIELD
    # --- 1 col
    A_MNT6_SCALARFIELD_1_COL_128_BITS = AnemoiPermutation(
        q=MNT6_SCALARFIELD,
        n_cols=1,
        security_level=128
    )
    A.append(
        ("A_MNT6_SCALARFIELD_1_COL_128_BITS",
         A_MNT6_SCALARFIELD_1_COL_128_BITS))
    # --- 2 col    
    A_MNT6_SCALARFIELD_2_COL_128_BITS = AnemoiPermutation(
        q=MNT6_SCALARFIELD,
        n_cols=2,
        security_level=128
    )
    A.append(
        ("A_MNT6_SCALARFIELD_2_COL_128_BITS",
         A_MNT6_SCALARFIELD_2_COL_128_BITS))    
    # --- 3 col    
    A_MNT6_SCALARFIELD_3_COL_128_BITS = AnemoiPermutation(
        q=MNT6_SCALARFIELD,
        n_cols=3,
        security_level=128
    )
    A.append(
        ("A_MNT6_SCALARFIELD_3_COL_128_BITS",
         A_MNT6_SCALARFIELD_3_COL_128_BITS))    
    # ---4 col    
    A_MNT6_SCALARFIELD_4_COL_128_BITS = AnemoiPermutation(
        q=MNT6_SCALARFIELD,
        n_cols=4,
        security_level=128
    )
    A.append(
        ("A_MNT6_SCALARFIELD_4_COL_128_BITS",
         A_MNT6_SCALARFIELD_4_COL_128_BITS))
    
def anemoi256_instances_mnt6(A):

    # - 256-bit security level instantiations
    # -- MNT6_SCALRFIELD
    # --- 1 col
    A_MNT6_SCALARFIELD_1_COL_256_BITS = AnemoiPermutation(
        q=MNT6_SCALARFIELD,
        n_cols=1,
        security_level=256
    )
    A.append(
        ("A_MNT6_SCALARFIELD_1_COL_256_BITS",
         A_MNT6_SCALARFIELD_1_COL_256_BITS))
    # --- 2 col    
    A_MNT6_SCALARFIELD_2_COL_256_BITS = AnemoiPermutation(
        q=MNT6_SCALARFIELD,
        n_cols=2,
        security_level=256
    )
    A.append(
        ("A_MNT6_SCALARFIELD_2_COL_256_BITS",
         A_MNT6_SCALARFIELD_2_COL_256_BITS))    
    # --- 3 col    
    A_MNT6_SCALARFIELD_3_COL_256_BITS = AnemoiPermutation(
        q=MNT6_SCALARFIELD,
        n_cols=3,
        security_level=256
    )
    A.append(
        ("A_MNT6_SCALARFIELD_3_COL_256_BITS",
         A_MNT6_SCALARFIELD_3_COL_256_BITS))    
    # ---4 col    
    A_MNT6_SCALARFIELD_4_COL_256_BITS = AnemoiPermutation(
        q=MNT6_SCALARFIELD,
        n_cols=4,
        security_level=256
    )
    A.append(
        ("A_MNT6_SCALARFIELD_4_COL_256_BITS",
         A_MNT6_SCALARFIELD_4_COL_256_BITS))
    
def anemoi_mnt6_nrounds():
    A = []
    anemoi128_instances_mnt6(A)
    nrounds128 = anemoi_get_nrounds(A)
    A = []
    anemoi256_instances_mnt6(A)
    nrounds256 = anemoi_get_nrounds(A)
    return nrounds128, nrounds256
        
def anemoi128_instances_bw6_761(A):

    # - 128-bit security level instantiations
    # -- BW6_761_SCALRFIELD
    # --- 1 col
    A_BW6_761_SCALARFIELD_1_COL_128_BITS = AnemoiPermutation(
        q=BW6_761_SCALARFIELD,
        n_cols=1,
        security_level=128
    )
    A.append(
        ("A_BW6_761_SCALARFIELD_1_COL_128_BITS",
         A_BW6_761_SCALARFIELD_1_COL_128_BITS))
    # --- 2 col    
    A_BW6_761_SCALARFIELD_2_COL_128_BITS = AnemoiPermutation(
        q=BW6_761_SCALARFIELD,
        n_cols=2,
        security_level=128
    )
    A.append(
        ("A_BW6_761_SCALARFIELD_2_COL_128_BITS",
         A_BW6_761_SCALARFIELD_2_COL_128_BITS))    
    # --- 3 col    
    A_BW6_761_SCALARFIELD_3_COL_128_BITS = AnemoiPermutation(
        q=BW6_761_SCALARFIELD,
        n_cols=3,
        security_level=128
    )
    A.append(
        ("A_BW6_761_SCALARFIELD_3_COL_128_BITS",
         A_BW6_761_SCALARFIELD_3_COL_128_BITS))    
    # ---4 col    
    A_BW6_761_SCALARFIELD_4_COL_128_BITS = AnemoiPermutation(
        q=BW6_761_SCALARFIELD,
        n_cols=4,
        security_level=128
    )
    A.append(
        ("A_BW6_761_SCALARFIELD_4_COL_128_BITS",
         A_BW6_761_SCALARFIELD_4_COL_128_BITS))
    
def anemoi256_instances_bw6_761(A):

    # - 256-bit security level instantiations
    # -- BW6_761_SCALRFIELD
    # --- 1 col
    A_BW6_761_SCALARFIELD_1_COL_256_BITS = AnemoiPermutation(
        q=BW6_761_SCALARFIELD,
        n_cols=1,
        security_level=256
    )
    A.append(
        ("A_BW6_761_SCALARFIELD_1_COL_256_BITS",
         A_BW6_761_SCALARFIELD_1_COL_256_BITS))
    # --- 2 col    
    A_BW6_761_SCALARFIELD_2_COL_256_BITS = AnemoiPermutation(
        q=BW6_761_SCALARFIELD,
        n_cols=2,
        security_level=256
    )
    A.append(
        ("A_BW6_761_SCALARFIELD_2_COL_256_BITS",
         A_BW6_761_SCALARFIELD_2_COL_256_BITS))    
    # --- 3 col    
    A_BW6_761_SCALARFIELD_3_COL_256_BITS = AnemoiPermutation(
        q=BW6_761_SCALARFIELD,
        n_cols=3,
        security_level=256
    )
    A.append(
        ("A_BW6_761_SCALARFIELD_3_COL_256_BITS",
         A_BW6_761_SCALARFIELD_3_COL_256_BITS))    
    # ---4 col    
    A_BW6_761_SCALARFIELD_4_COL_256_BITS = AnemoiPermutation(
        q=BW6_761_SCALARFIELD,
        n_cols=4,
        security_level=256
    )
    A.append(
        ("A_BW6_761_SCALARFIELD_4_COL_256_BITS",
         A_BW6_761_SCALARFIELD_4_COL_256_BITS))
    
def anemoi_bw6_761_nrounds():
    A = []
    anemoi128_instances_bw6_761(A)
    nrounds128 = anemoi_get_nrounds(A)
    A = []
    anemoi256_instances_bw6_761(A)
    nrounds256 = anemoi_get_nrounds(A)
    return nrounds128, nrounds256
        
def anemoi128_instances_bn128(A):

    # - 128-bit security level instantiations
    # -- BN128_SCALRFIELD
    # --- 1 col
    A_BN128_SCALARFIELD_1_COL_128_BITS = AnemoiPermutation(
        q=BN128_SCALARFIELD,
        n_cols=1,
        security_level=128
    )
    A.append(
        ("A_BN128_SCALARFIELD_1_COL_128_BITS",
         A_BN128_SCALARFIELD_1_COL_128_BITS))
    # --- 2 col    
    A_BN128_SCALARFIELD_2_COL_128_BITS = AnemoiPermutation(
        q=BN128_SCALARFIELD,
        n_cols=2,
        security_level=128
    )
    A.append(
        ("A_BN128_SCALARFIELD_2_COL_128_BITS",
         A_BN128_SCALARFIELD_2_COL_128_BITS))    
    # --- 3 col    
    A_BN128_SCALARFIELD_3_COL_128_BITS = AnemoiPermutation(
        q=BN128_SCALARFIELD,
        n_cols=3,
        security_level=128
    )
    A.append(
        ("A_BN128_SCALARFIELD_3_COL_128_BITS",
         A_BN128_SCALARFIELD_3_COL_128_BITS))    
    # ---4 col    
    A_BN128_SCALARFIELD_4_COL_128_BITS = AnemoiPermutation(
        q=BN128_SCALARFIELD,
        n_cols=4,
        security_level=128
    )
    A.append(
        ("A_BN128_SCALARFIELD_4_COL_128_BITS",
         A_BN128_SCALARFIELD_4_COL_128_BITS))
    
def anemoi256_instances_bn128(A):

    # - 256-bit security level instantiations
    # -- BN128_SCALRFIELD
    # --- 1 col
    A_BN128_SCALARFIELD_1_COL_256_BITS = AnemoiPermutation(
        q=BN128_SCALARFIELD,
        n_cols=1,
        security_level=256
    )
    A.append(
        ("A_BN128_SCALARFIELD_1_COL_256_BITS",
         A_BN128_SCALARFIELD_1_COL_256_BITS))
    # --- 2 col    
    A_BN128_SCALARFIELD_2_COL_256_BITS = AnemoiPermutation(
        q=BN128_SCALARFIELD,
        n_cols=2,
        security_level=256
    )
    A.append(
        ("A_BN128_SCALARFIELD_2_COL_256_BITS",
         A_BN128_SCALARFIELD_2_COL_256_BITS))    
    # --- 3 col    
    A_BN128_SCALARFIELD_3_COL_256_BITS = AnemoiPermutation(
        q=BN128_SCALARFIELD,
        n_cols=3,
        security_level=256
    )
    A.append(
        ("A_BN128_SCALARFIELD_3_COL_256_BITS",
         A_BN128_SCALARFIELD_3_COL_256_BITS))    
    # ---4 col    
    A_BN128_SCALARFIELD_4_COL_256_BITS = AnemoiPermutation(
        q=BN128_SCALARFIELD,
        n_cols=4,
        security_level=256
    )
    A.append(
        ("A_BN128_SCALARFIELD_4_COL_256_BITS",
         A_BN128_SCALARFIELD_4_COL_256_BITS))
    
def anemoi_bn128_nrounds():
    A = []
    anemoi128_instances_bn128(A)
    nrounds128 = anemoi_get_nrounds(A)
    A = []
    anemoi256_instances_bn128(A)
    nrounds256 = anemoi_get_nrounds(A)
    return nrounds128, nrounds256
        
def anemoi128_instances_alt_bn128(A):

    # - 128-bit security level instantiations
    # -- ALT_BN128_SCALRFIELD
    # --- 1 col
    A_ALT_BN128_SCALARFIELD_1_COL_128_BITS = AnemoiPermutation(
        q=ALT_BN128_SCALARFIELD,
        n_cols=1,
        security_level=128
    )
    A.append(
        ("A_ALT_BN128_SCALARFIELD_1_COL_128_BITS",
         A_ALT_BN128_SCALARFIELD_1_COL_128_BITS))
    # --- 2 col    
    A_ALT_BN128_SCALARFIELD_2_COL_128_BITS = AnemoiPermutation(
        q=ALT_BN128_SCALARFIELD,
        n_cols=2,
        security_level=128
    )
    A.append(
        ("A_ALT_BN128_SCALARFIELD_2_COL_128_BITS",
         A_ALT_BN128_SCALARFIELD_2_COL_128_BITS))    
    # --- 3 col    
    A_ALT_BN128_SCALARFIELD_3_COL_128_BITS = AnemoiPermutation(
        q=ALT_BN128_SCALARFIELD,
        n_cols=3,
        security_level=128
    )
    A.append(
        ("A_ALT_BN128_SCALARFIELD_3_COL_128_BITS",
         A_ALT_BN128_SCALARFIELD_3_COL_128_BITS))    
    # ---4 col    
    A_ALT_BN128_SCALARFIELD_4_COL_128_BITS = AnemoiPermutation(
        q=ALT_BN128_SCALARFIELD,
        n_cols=4,
        security_level=128
    )
    A.append(
        ("A_ALT_BN128_SCALARFIELD_4_COL_128_BITS",
         A_ALT_BN128_SCALARFIELD_4_COL_128_BITS))
    
def anemoi256_instances_alt_bn128(A):

    # - 256-bit security level instantiations
    # -- ALT_BN128_SCALRFIELD
    # --- 1 col
    A_ALT_BN128_SCALARFIELD_1_COL_256_BITS = AnemoiPermutation(
        q=ALT_BN128_SCALARFIELD,
        n_cols=1,
        security_level=256
    )
    A.append(
        ("A_ALT_BN128_SCALARFIELD_1_COL_256_BITS",
         A_ALT_BN128_SCALARFIELD_1_COL_256_BITS))
    # --- 2 col    
    A_ALT_BN128_SCALARFIELD_2_COL_256_BITS = AnemoiPermutation(
        q=ALT_BN128_SCALARFIELD,
        n_cols=2,
        security_level=256
    )
    A.append(
        ("A_ALT_BN128_SCALARFIELD_2_COL_256_BITS",
         A_ALT_BN128_SCALARFIELD_2_COL_256_BITS))    
    # --- 3 col    
    A_ALT_BN128_SCALARFIELD_3_COL_256_BITS = AnemoiPermutation(
        q=ALT_BN128_SCALARFIELD,
        n_cols=3,
        security_level=256
    )
    A.append(
        ("A_ALT_BN128_SCALARFIELD_3_COL_256_BITS",
         A_ALT_BN128_SCALARFIELD_3_COL_256_BITS))    
    # ---4 col    
    A_ALT_BN128_SCALARFIELD_4_COL_256_BITS = AnemoiPermutation(
        q=ALT_BN128_SCALARFIELD,
        n_cols=4,
        security_level=256
    )
    A.append(
        ("A_ALT_BN128_SCALARFIELD_4_COL_256_BITS",
         A_ALT_BN128_SCALARFIELD_4_COL_256_BITS))
    
def anemoi_alt_bn128_nrounds():
    A = []
    anemoi128_instances_alt_bn128(A)
    nrounds128 = anemoi_get_nrounds(A)
    A = []
    anemoi256_instances_alt_bn128(A)
    nrounds256 = anemoi_get_nrounds(A)
    return nrounds128, nrounds256
        
def anemoi_instances_stdout(instances):
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
        #print("constants C      :\n{}".format(A.C))
        #print("constants D      :\n{}".format(A.D))

# same as output_parameters() but stores parameters to file
def anemoi_instances_to_file(instances):
    f = open("instances.txt", "w")
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

def anemoi_parameters_in_cpp_format_to_file(instances, filename, curve_ppT, nrounds128, nrounds256):
    f = open(filename, "w")
    e = datetime.datetime.now()
    f.write("// This file was automatically generated with SAGE script parameters.sage on %s/%s/%s at %s:%s:%s\n\n" % (e.day, e.month, e.year, e.hour, e.minute, e.second))

    f.write("// Anemoi parameters for curve {}\n".format(curve_ppT))
    # get just the first instance -- all instances from a given curve
    # share the same parameters
    A = instances[0][1]

    f.write("template<> class anemoi_parameters<libff::{}>\n".format(curve_ppT))
    f.write("{\npublic:\n")    
    f.write("using ppT = libff::{};\n".format(curve_ppT))
    f.write("using FieldT = libff::Fr<ppT>;\n")
    f.write("using BignumT = libff::bigint<FieldT::num_limbs>;\n")
    f.write("static const bool b_prime_field = false;\n")
    f.write("static constexpr size_t multiplicative_generator_g = {};\n".format(A.g))
    f.write("static constexpr size_t alpha = {};\n".format(A.alpha))
    f.write("static constexpr size_t beta = multiplicative_generator_g;\n")
    f.write("static constexpr size_t gamma = 0;\n")
    f.write("static constexpr size_t quad_exponent = {};\n".format(A.QUAD))
    f.write("static const BignumT alpha_inv;\n")
    f.write("static const BignumT delta;\n")
    f.write("static const std::vector<size_t> nrounds128;\n")
    f.write("static const std::vector<size_t> nrounds256;\n")
    f.write("static const std::vector<std::vector<BignumT>> C_constants_col_one;\n")
    f.write("static const std::vector<std::vector<BignumT>> D_constants_col_one;\n")
    f.write("static const std::vector<std::vector<BignumT>> C_constants_col_two;\n")
    f.write("static const std::vector<std::vector<BignumT>> D_constants_col_two;\n")
    f.write("static const std::vector<std::vector<BignumT>> C_constants_col_three;\n")
    f.write("static const std::vector<std::vector<BignumT>> D_constants_col_three;\n")
    f.write("static const std::vector<std::vector<BignumT>> C_constants_col_four;\n")
    f.write("static const std::vector<std::vector<BignumT>> D_constants_col_four;\n")
    f.write("};\n")
    
    f.write("\n")
    f.write("const std::vector<size_t> anemoi_parameters<libff::{}>::nrounds128 = ".format(curve_ppT))
    f.write("{")
    for i in range(len(nrounds128)):
        f.write("{}".format(nrounds128[i]))
        if(i < (len(nrounds128)-1)):
            f.write(", ")
    f.write("};")
    
    f.write("\n")
    f.write("const std::vector<size_t> anemoi_parameters<libff::{}>::nrounds256 = ".format(curve_ppT))
    f.write("{")
    for i in range(len(nrounds256)):
        f.write("{}".format(nrounds256[i]))
        if(i < (len(nrounds256)-1)):
            f.write(", ")
    f.write("};")    
    
    f.write("\n\n")    
    f.write("const anemoi_parameters<libff::{}>::BignumT anemoi_parameters<libff::{}>::alpha_inv = anemoi_parameters<libff::{}>::BignumT(\"{}\");\n".format(curve_ppT, curve_ppT, curve_ppT, A.alpha_inv))
    
    f.write("\n")    
    f.write("const anemoi_parameters<libff::{}>::BignumT anemoi_parameters<libff::{}>::delta = anemoi_parameters<libff::{}>::BignumT(\"{}\");\n".format(curve_ppT, curve_ppT, curve_ppT, A.delta))
    
#    f.write("namespace libsnark \n{\n")
#    f.write("} // namespace libsnark")
    
def anemoi_constants_in_cpp_format_to_file(instances, filename, curve_ppT):
    f = open(filename, "a")
    f.write("\n")    
    i_str = ["one", "two", "three", "four", "five", "six"]
    for i in range(len(instances)):
        A_str = instances[i][0]
        A = instances[i][1]
        f.write("// C constants for L = {} columns\n".format(i+1))
        f.write("const std::vector<std::vector<anemoi_parameters<libff::{}>::BignumT>> anemoi_parameters<libff::{}>::C_constants_col_{} = ".format(curve_ppT, curve_ppT, i_str[i]))
        f.write("{\n")
        for iround in range(len(A.C)):
            f.write("{")
            for icol in range(len(A.C[iround])):
                f.write("anemoi_parameters<libff::{}>::BignumT(\"{}\")".format(curve_ppT, A.C[iround][icol]))
                if icol < (len(A.C[iround]) - 1):
                    f.write(", ")
            f.write("}")
            if iround < (len(A.C) - 1):
                f.write(",\n")
        f.write("\n};\n")
        f.write("// D constants for L = {} columns\n".format(i+1))
        f.write("const std::vector<std::vector<anemoi_parameters<libff::{}>::BignumT>> anemoi_parameters<libff::{}>::D_constants_col_{} = ".format(curve_ppT, curve_ppT, i_str[i]))
        f.write("{\n")
        for iround in range(len(A.D)):
            f.write("{")
            for icol in range(len(A.D[iround])):
                f.write("anemoi_parameters<libff::{}>::BignumT(\"{}\")".format(curve_ppT, A.D[iround][icol]))
                if icol < (len(A.D[iround]) - 1):
                    f.write(", ")
            f.write("}")
            if iround < (len(A.D) - 1):
                f.write(",\n")
        f.write("\n};\n")

def test_anemoi_nrounds():
    print("bls12_381")
    nrounds128, nrounds256 = anemoi_bls12_381_nrounds()
    print("nrounds128 {}".format(nrounds128))
    print("nrounds256 {}".format(nrounds256))
    
    print("bls12_377")
    nrounds128, nrounds256 = anemoi_bls12_377_nrounds()
    print("nrounds128 {}".format(nrounds128))
    print("nrounds256 {}".format(nrounds256))
    
    print("mnt4")
    nrounds128, nrounds256 = anemoi_mnt4_nrounds()
    print("nrounds128 {}".format(nrounds128))
    print("nrounds256 {}".format(nrounds256))
    
    print("mnt6")
    nrounds128, nrounds256 = anemoi_mnt6_nrounds()
    print("nrounds128 {}".format(nrounds128))
    print("nrounds256 {}".format(nrounds256))
    
    print("bw6_761")
    nrounds128, nrounds256 = anemoi_bw6_761_nrounds()
    print("nrounds128 {}".format(nrounds128))
    print("nrounds256 {}".format(nrounds256))

    print("bn128")
    nrounds128, nrounds256 = anemoi_bn128_nrounds()
    print("nrounds128 {}".format(nrounds128))
    print("nrounds256 {}".format(nrounds256))

    print("alt_bn128")
    nrounds128, nrounds256 = anemoi_alt_bn128_nrounds()
    print("nrounds128 {}".format(nrounds128))
    print("nrounds256 {}".format(nrounds256))

def test_anemoi_internal_values_bls12_381():
    A = []
    anemoi256_instances_bls12_381(A)
    # Same q for all A[0,1,2,3][1]
    q = A[0][1].q
    print("q {}".format(hex(q)))
    outputs = []
    instance_names = []
    for i in range(len(A)):
    #for i in range(1):
        Anemoi = A[i][1]
        name = A[i][0]
        ncols = Anemoi.n_cols
        nrounds = Anemoi.n_rounds
        print("Anemoi {}".format(name))
        print("ncols {}".format(ncols))
        print("nrounds {}".format(nrounds))
        # Hard-code left and right input equal to sequence
        # 0,1,2,3,4...
        X_left_input = []
        X_right_input = []
        for j in range(ncols):
            X_left_input.append(j)
            X_right_input.append(ncols+j)
        print("X_left_input  {}".format(X_left_input))
        print("X_right_input {}".format(X_right_input))
        res = Anemoi.eval_round_with_intermediate_values(X_left_input, X_right_input)
        #print(res[len(res)-1])
        outputs.append(res[len(res)-1])
        instance_names.append(name)
    #for i in range(len(A)):
    #    print("{} \n{} \n".format(instance_names[i], outputs[i]))
    return instance_names, outputs

def anemoi_outputs_in_cpp_format_to_file(instance_names, outputs, filename, curve_ppT):
    f = open(filename, "w")
    e = datetime.datetime.now()
    f.write("// Output values automatically generated with SAGE script parameters.sage on %s/%s/%s at %s:%s:%s\n\n" % (e.day, e.month, e.year, e.hour, e.minute, e.second))
    for i in range(len(outputs)):
        #print("{} \n{} \n".format(instance_names[i], outputs[i]))
        LEFT = 0
        RIGHT = 1
        f.write("// {}\n".format(instance_names[i]))
        f.write("// Left outputs\n")
        for j in range(len(outputs[i][LEFT])):
            f.write("libff::Fr<libff::{}>(\"{}\"),\n".format(curve_ppT, outputs[i][LEFT][j]))
        f.write("// Right outputs\n")
        for j in range(len(outputs[i][RIGHT])):
            f.write("libff::Fr<libff::{}>(\"{}\"),\n".format(curve_ppT, outputs[i][RIGHT][j]))
        f.write("\n")
    
if __name__ == "__main__":
    # print Anemoi internal values BLS12_381
    if 1:
        instance_names, outputs = test_anemoi_internal_values_bls12_381()
        filename = "outputs_bls12_381.txt"
        curve_ppT = "bls12_381_pp"
        anemoi_outputs_in_cpp_format_to_file(instance_names, outputs, filename, curve_ppT)
        
    # extract number of rounds
    if 0:
        test_anemoi_nrounds()        
    # bls12_381
    if 0:
        A = []
        anemoi256_instances_bls12_381(A)
        filename = "parameters_bls12_381.txt"
        curve_ppT = "bls12_381_pp"
        nrounds128, nrounds256 = anemoi_bls12_381_nrounds()
        anemoi_parameters_in_cpp_format_to_file(A, filename, curve_ppT, nrounds128, nrounds256)
        anemoi_constants_in_cpp_format_to_file(A, filename, curve_ppT)
        #anemoi_instances_stdout(A)
    # bls12_377
    if 0:
        A = []
        anemoi256_instances_bls12_377(A)
        filename = "parameters_bls12_377.txt"
        curve_ppT = "bls12_377_pp"
        nrounds128, nrounds256 = anemoi_bls12_377_nrounds()
        anemoi_parameters_in_cpp_format_to_file(A, filename, curve_ppT, nrounds128, nrounds256)
        anemoi_constants_in_cpp_format_to_file(A, filename, curve_ppT)
    # mnt4
    if 0:
        A = []
        anemoi256_instances_mnt4(A)
        filename = "parameters_mnt4.txt"
        curve_ppT = "mnt4_pp"
        nrounds128, nrounds256 = anemoi_mnt4_nrounds()
        anemoi_parameters_in_cpp_format_to_file(A, filename, curve_ppT, nrounds128, nrounds256)
        anemoi_constants_in_cpp_format_to_file(A, filename, curve_ppT)
    # mnt6
    if 0:
        A = []
        anemoi256_instances_mnt6(A)
        filename = "parameters_mnt6.txt"
        curve_ppT = "mnt6_pp"
        nrounds128, nrounds256 = anemoi_mnt6_nrounds()
        anemoi_parameters_in_cpp_format_to_file(A, filename, curve_ppT, nrounds128, nrounds256)
        anemoi_constants_in_cpp_format_to_file(A, filename, curve_ppT)
    # bw6_761 (WARNING! slow ~10 min.)
    if 0:
        A = []
        anemoi256_instances_bw6_761(A)
        filename = "parameters_bw6_761.txt"
        curve_ppT = "bw6_761_pp"
        nrounds128, nrounds256 = anemoi_bw6_761_nrounds()
        anemoi_parameters_in_cpp_format_to_file(A, filename, curve_ppT, nrounds128, nrounds256)
        anemoi_constants_in_cpp_format_to_file(A, filename, curve_ppT)
    # bn128
    if 0:
        A = []
        anemoi256_instances_bn128(A)
        filename = "parameters_bn128.txt"
        curve_ppT = "bn128_pp"
        nrounds128, nrounds256 = anemoi_bn128_nrounds()
        anemoi_parameters_in_cpp_format_to_file(A, filename, curve_ppT, nrounds128, nrounds256)
        anemoi_constants_in_cpp_format_to_file(A, filename, curve_ppT)
    # alt_bn128
    if 0:
        A = []
        anemoi256_instances_alt_bn128(A)
        filename = "parameters_alt_bn128.txt"
        curve_ppT = "alt_bn128_pp"
        nrounds128, nrounds256 = anemoi_alt_bn128_nrounds()
        anemoi_parameters_in_cpp_format_to_file(A, filename, curve_ppT, nrounds128, nrounds256)
        anemoi_constants_in_cpp_format_to_file(A, filename, curve_ppT)
