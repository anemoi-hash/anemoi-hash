# Anemoi: a Family of ZK-friendly AO Hash Functions

This repository contains a first implementation in sage of the **Anemoi** family of hash functions.
**Anemoi** is a family of Arithmetization Oriented Hash Functions that operates over prime and binary fields.

Our paper on Anemoi can be found on ePrint: https://eprint.iacr.org/2022/840.

## Contents
The sage script ```anemoi.sage``` contains various routines to evaluate **Anemoi** (including **AnemoiJive** and **AnemoiSponge**) but also to generate the corresponding systems of equations.

First some basics functions allow to provide well-chosen parameters: field, number of rounds, linear layer, ... for various instances of **Anemoi**.

The class ```AnemoiPermutation``` then contains different sections:
- Sub-components: ```evaluate_sbox``` and ```linear_layer``` respectively apply our **Flystel** construction and the linear layer.
- Evaluation: ```eval_with_intermediate_values``` performs an evaluation of **Anemoi** using the SPN construction. The function also return intermediate values as this can be used to check the solutions of the systems of equations generated.
- Writing full system of equations: ```get_polynomial_variables```, ```verification_polynomials``` and ```print_verification_polynomials``` allow to generate the corresponding multivariate system of polynomial equations. This indeed allowed us to perform our security analysis using Grobner basis attacks. 

The two functions ```jive``` and ```sponge_hash``` are routines to evaluate **AnemoiJive** and **AnemoiSponge** that respectively correspond to our Merkle Compression function, and our Hash function.

## Authors
- [Clémence Bouvier](https://who.rocq.inria.fr/Clemence.Bouvier/), Sorbonne University, France - Inria, France
- [Pierre Briaud](https://who.rocq.inria.fr/Pierre.Briaud/), Sorbonne University, France - Inria, France
- Pyrros Chaidos, National & Kapodistrian University of Athens, Greece,
- [Léo Perrin](https://who.paris.inria.fr/Leo.Perrin/), Inria, France
- Robin Salen, Toposware, Inc., US
- Vesselin Velichkov, University of Edinburgh, Scotland - Clearmatics, England
- Danny Willems, Nomadic Labs, France - Inria & LIX, France


## Third-party implementations
Please contact us if you have any **Anemoi** implementations to share.


## License
This repository is distributed under the terms of the MIT License.
