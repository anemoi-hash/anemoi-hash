#!/usr/bin/sage
# -*- mode: python ; -*-

from sage.all import *
import hashlib
import itertools

from constants import *

COST_ALPHA = {
    3   : 2, 5   : 3, 7   : 4, 9   : 4,
    11  : 5, 13  : 5, 15  : 5, 17  : 5,
    19  : 6, 21  : 6, 23  : 6, 25  : 6,
    27  : 6, 29  : 7, 31  : 7, 33  : 6,
    35  : 7, 37  : 7, 39  : 7, 41  : 7,
    43  : 7, 45  : 7, 47  : 8, 49  : 7,
    51  : 7, 53  : 8, 55  : 8, 57  : 8,
    59  : 8, 61  : 8, 63  : 8, 65  : 7,
    67  : 8, 69  : 8, 71  : 9, 73  : 8,
    75  : 8, 77  : 8, 79  : 9, 81  : 8,
    83  : 8, 85  : 8, 87  : 9, 89  : 9,
    91  : 9, 93  : 9, 95  : 9, 97  : 8,
    99  : 8, 101 : 9, 103 : 9, 105 : 9,
    107 : 9, 109 : 9, 111 : 9, 113 : 9,
    115 : 9, 117 : 9, 119 : 9, 121 : 9,
    123 : 9, 125 : 9, 127 : 10,
}

ALPHA_BY_COST = {
    c : [x for x in range(3, 128, 2) if COST_ALPHA[x] == c]
    for c in range(2, 11)
}

PI_0 = 1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679
PI_1 = 8214808651328230664709384460955058223172535940812848111745028410270193852110555964462294895493038196

def get_prime(N):
    """Returns the highest prime number that is strictly smaller than
    2**N.

    """
    result = (1 << N) - 1
    while not is_prime(result):
        result -= 2
    return result


def get_n_rounds(s, l, alpha):
    """Returns the number of rounds needed in Anemoi (based on the
    complexity of algebraic attacks).

    """
    r = 0
    complexity = 0
    kappa = {3:1, 5:2, 7:4, 9:7, 11:9}
    assert alpha in kappa
    while complexity < 2**s:
        r += 1
        complexity = binomial(
            4*l*r + kappa[alpha],
            2*l*r
        )**2
    r += 2 # considering the second model
    r += min(5,l+1) # security margin
    
    return max(8, r)


# Linear layer generation

def is_mds(m):
    # Uses the Laplace expansion of the determinant to calculate the (m+1)x(m+1) minors in terms of the mxm minors.
    # Taken from https://github.com/mir-protocol/hash-constants/blob/master/mds_search.sage.

    # 1-minors are just the elements themselves
    if any(any(r == 0 for r in row) for row in m):
        return False

    N = m.nrows()
    assert m.is_square() and N >= 2

    det_cache = m

    # Calculate all the nxn minors of m:
    for n in range(2, N+1):
        new_det_cache = dict()
        for rows in itertools.combinations(range(N), n):
            for cols in itertools.combinations(range(N), n):
                i, *rs = rows

                # Laplace expansion along row i
                det = 0
                for j in range(n):
                    # pick out c = column j; the remaining columns are in cs
                    c = cols[j]
                    cs = cols[:j] + cols[j+1:]

                    # Look up the determinant from the previous iteration
                    # and multiply by -1 if j is odd
                    cofactor = det_cache[(*rs, *cs)]
                    if j % 2 == 1:
                        cofactor = -cofactor

                    # update the determinant with the j-th term
                    det += m[i, c] * cofactor

                if det == 0:
                    return False
                new_det_cache[(*rows, *cols)] = det
        det_cache = new_det_cache
    return True

def M_2(x_input, b):
    """Fast matrix-vector multiplication algorithm for Anemoi MDS layer with \ell = 1,2."""

    x = x_input[:]
    x[0] += b*x[1]
    x[1] += b*x[0]
    return x

def M_3(x_input, b):
    """Fast matrix-vector multiplication algorithm for Anemoi MDS layer with \ell = 3.

    From Figure 6 of [DL18](https://tosc.iacr.org/index.php/ToSC/article/view/888)."""

    x = x_input[:]
    t = x[0] + b*x[2]
    x[2] += x[1]
    x[2] += b*x[0]
    x[0] = t + x[2]
    x[1] += t
    return x


def M_4(x_input, b):
    """Fast matrix-vector multiplication algorithm for Anemoi MDS layer with \ell = 4.

    Figure 8 of [DL18](https://tosc.iacr.org/index.php/ToSC/article/view/888)."""

    x = x_input[:]
    x[0] += x[1]
    x[2] += x[3]
    x[3] += b*x[0]
    x[1]  = b*(x[1] + x[2])
    x[0] += x[1]
    x[2] += b*x[3]
    x[1] += x[2]
    x[3] += x[0]
    return x

def lfsr(x_input, b):
    x = x_input[:]
    l = len(x)
    for r in range(0, l):
        t = sum(b**(2**i) * x[i] for i in range(0, l))
        x = x[1:] + [t]
    return x

def circulant_mds_matrix(field, l, coeff_upper_limit=None):
    if coeff_upper_limit == None:
        coeff_upper_limit = l+1
    assert(coeff_upper_limit > l)
    for v in itertools.combinations_with_replacement(range(1,coeff_upper_limit), l):
        mat = matrix.circulant(list(v)).change_ring(field)
        if is_mds(mat):
            return(mat)
    # In some cases, the method won't return any valid matrix,
    # hence the need to increase the limit further.
    return circulant_mds_matrix(field, l, coeff_upper_limit+1)

def get_mds(field, l):
    if l == 1:
        return identity_matrix(field, 1)
    if l <= 4: # low addition case
        a = field.multiplicative_generator()
        b = field.one()
        t = 0
        while True:
            # we construct the matrix
            mat = []
            b = b*a
            t += 1
            for i in range(0, l):
                x_i = [field.one() * (j == i) for j in range(0, l)]
                if l == 2:
                    mat.append(M_2(x_i, b))
                elif l == 3:
                    mat.append(M_3(x_i, b))
                elif l == 4:
                    mat.append(M_4(x_i, b))
            mat = Matrix(field, l, l, mat).transpose()
            if is_mds(mat):
                return mat
    else: # circulant matrix case
        return circulant_mds_matrix(field, l)

# AnemoiPermutation class

class AnemoiPermutation:
    def __init__(self,
                 q=None,
                 alpha=None,
                 mat=None,
                 n_rounds=None,
                 n_cols=1,
                 security_level=128):
        if q == None:
            raise Exception("The characteristic of the field must be specified!")
        self.q = q
        self.prime_field = is_prime(q)  # if true then we work over a
                                        # prime field with
                                        # characteristic just under
                                        # 2**N, otherwise the
                                        # characteristic is 2**self
        self.n_cols = n_cols # the number of parallel S-boxes in each round
        self.security_level = security_level

        # initializing the other variables in the state:
        # - q     is the characteristic of the field
        # - g     is a generator of the multiplicative subgroup
        # - alpha is the main exponent (in the center of the Flystel)
        # - beta  is the coefficient in the quadratic subfunction
        # - gamma is the constant in the second quadratic subfunction
        # - QUAD  is the secondary (quadratic) exponent
        # - from_field is a function mapping field elements to integers
        # - to_field   is a function mapping integers to field elements
        self.F = GF(self.q)
        if self.prime_field:
            if alpha != None:
                if gcd(alpha, self.q-1) != 1:
                    raise Exception("alpha should be co-prime with the characteristic!")
                else:
                    self.alpha = alpha
            else:
                self.alpha = 3
                while gcd(self.alpha, self.q-1) != 1:
                    self.alpha += 1
            self.QUAD = 2
            self.to_field   = lambda x : self.F(x)
            self.from_field = lambda x : Integer(x)
        else:
            self.alpha = 3
            self.QUAD = 3
            self.to_field   = lambda x : self.F.fetch_int(x)
            self.from_field = lambda x : x.integer_representation()
        self.g = self.F.multiplicative_generator()
        self.beta = self.g
        self.delta = self.g**(-1)
        self.alpha_inv = inverse_mod(self.alpha, self.q-1)

        # total number of rounds
        if n_rounds != None:
            self.n_rounds = n_rounds
        else:
            self.n_rounds = get_n_rounds(self.security_level,
                                         self.n_cols,
                                         self.alpha)

        # Choosing constants: self.C and self.D are built from the
        # digits of pi using an open butterfly
        self.C = []
        self.D = []
        pi_F_0 = self.to_field(PI_0 % self.q)
        pi_F_1 = self.to_field(PI_1 % self.q)
        for r in range(0, self.n_rounds):
            pi_0_r = pi_F_0**r
            self.C.append([])
            self.D.append([])
            for i in range(0, self.n_cols):
                pi_1_i = pi_F_1**i
                pow_alpha = (pi_0_r + pi_1_i)**self.alpha
                self.C[r].append(self.g * (pi_0_r)**2 + pow_alpha)
                self.D[r].append(self.g * (pi_1_i)**2 + pow_alpha + self.delta)
        self.mat = get_mds(self.F, self.n_cols)


    def __str__(self):
        result = "Anemoi instance over F_{:d} ({}), n_rounds={:d}, n_cols={:d}, s={:d}".format(
            self.q,
            "odd prime field" if self.prime_field else "characteristic 2",
            self.n_rounds,
            self.n_cols,
            self.security_level
        )
        result += "\nalpha={}, beta={}, \delta={}\nM_x=\n{}\n".format(
            self.alpha,
            self.beta,
            self.delta,
            self.mat
        )
        result += "C={}\nD={}".format(
            [[self.from_field(x) for x in self.C[r]] for r in range(0, self.n_rounds)],
            [[self.from_field(x) for x in self.D[r]] for r in range(0, self.n_rounds)],
        )
        return result


    # !SECTION! Sub-components

    def evaluate_sbox(self, _x, _y):
        """Applies an open Flystel to the full state. """
        x, y = _x, _y
        x -= self.beta*y**self.QUAD
        y -= x**self.alpha_inv
        x += self.beta*y**self.QUAD + self.delta
        return x, y

    def linear_layer(self, _x, _y):
        x, y = _x[:], _y[:]
        x = self.mat*vector(x)
        y = self.mat*vector(y[1:] + [y[0]])

        # Pseudo-Hadamard transform on each (x,y) pair
        y += x
        x += y
        return list(x), list(y)


    # !SECTION! Evaluation

    def eval_with_intermediate_values(self, _x, _y):
        """Returns a list of vectors x_i and y_i such that [x_i, y_i] is the
        internal state of Anemoi at the end of round i.

        The output is of length self.n_rounds+2 since it also returns
        the input values, and since there is a last degenerate round
        consisting only in a linear layer.

        """
        x, y = _x[:], _y[:]
        result = [[x[:], y[:]]]
        for r in range(0, self.n_rounds):
            for i in range(0, self.n_cols):
                x[i] += self.C[r][i]
                y[i] += self.D[r][i]
            x, y = self.linear_layer(x, y)
            for i in range(0, self.n_cols):
                x[i], y[i] = self.evaluate_sbox(x[i], y[i])
            result.append([x[:], y[:]])
        # final call to the linear layer
        x, y = self.linear_layer(x, y)
        result.append([x[:], y[:]])
        return result


    def input_size(self):
        return 2*self.n_cols


    def __call__(self, _x):
        if len(_x) != self.input_size():
            raise Exception("wrong input size!")
        else:
            x, y = _x[:self.n_cols], _x[self.n_cols:]
            u, v = self.eval_with_intermediate_values(x, y)[-1]
            return u + v # concatenation, not a sum


    # !SECTION! Writing full system of equations

    def get_polynomial_variables(self):
        """Returns polynomial variables from the appropriate multivariate
        polynomial ring to work with this Anemoi instance.

        """
        x_vars = []
        y_vars = []
        all_vars = []
        for r in range(0, self.n_rounds+1):
            x_vars.append(["X{:02d}{:02d}".format(r, i) for i in range(0, self.n_cols)])
            y_vars.append(["Y{:02d}{:02d}".format(r, i) for i in range(0, self.n_cols)])
            all_vars += x_vars[-1]
            all_vars += y_vars[-1]
        pol_ring = PolynomialRing(self.F, (self.n_rounds+1)*2*self.n_cols, all_vars)
        pol_gens = pol_ring.gens()
        result = {"X" : [], "Y" : []}
        for r in range(0, self.n_rounds+1):
            result["X"].append([])
            result["Y"].append([])
            for i in range(0, self.n_cols):
                result["X"][r].append(pol_gens[self.n_cols*2*r + i])
                result["Y"][r].append(pol_gens[self.n_cols*2*r + i + self.n_cols])
        return result


    def verification_polynomials(self, pol_vars):
        """Returns the list of all the equations that all the intermediate
        values must satisfy. It implicitely relies on the open Flystel
        function."""
        equations = []
        for r in range(0, self.n_rounds):
            # the outputs of the open flystel are the state variables x, y at round r+1
            u = pol_vars["X"][r+1]
            v = pol_vars["Y"][r+1]
            # the inputs of the open flystel are the state variables
            # x, y at round r after undergoing the constant addition
            # and the linear layer
            x, y = pol_vars["X"][r], pol_vars["Y"][r]
            x = [x[i] + self.C[r][i] for i in range(0, self.n_cols)]
            y = [y[i] + self.D[r][i] for i in range(0, self.n_cols)]
            x, y = self.linear_layer(x, y)
            for i in range(0, self.n_cols):
                equations.append(
                    (y[i]-v[i])**self.alpha + self.beta*y[i]**self.QUAD - x[i]
                )
                equations.append(
                    (y[i]-v[i])**self.alpha + self.beta*v[i]**self.QUAD + self.delta - u[i]
                )
        return equations


    def print_verification_polynomials(self):
        """Simply prints the equations modeling a full call to this
        AnemoiPermutation instance in a user (and computer) readable
        format.

        The first lines contains a comma separated list of all the
        variables, and the second contains the field size. The
        following ones contain the equations. This format is intended
        for use with Magma.

        """
        p_vars = self.get_polynomial_variables()
        eqs = self.verification_polynomials(p_vars)
        variables_string = ""
        for r in range(0, self.n_rounds+1):
            variables_string += str(p_vars["X"][r])[1:-1] + "," + str(p_vars["Y"][r])[1:-1] + ","
        print(variables_string[:-1].replace(" ", ""))
        print(self.q)
        for f in eqs:
            print(f)



# !SECTION! Modes of operation


def jive(P, b, _x):
    """Returns an output b times smaller than _x using the Jive mode of
    operation and the permutation P.

    """
    if b < 2:
        raise Exception("b must be at least equal to 2")
    if P.input_size() % b != 0:
        raise Exception("b must divide the input size!")
    c = P.input_size()/b # length of the compressed output
    # Output size check: we allow the output size to be 3 bits shorter than
    # the theoretical target, as commonly used finite fields usually have a
    # characteristic size slightly under 2**256.
    if c * P.F.cardinality().nbits() < 2 * P.security_level - 3:
        raise Exception(f"digest size is too small for the targeted security level!")
    x = _x[:]
    u = P(x)
    compressed = []
    for i in range(0, c):
        compressed.append(sum(x[i+c*j] + u[i+c*j]
                              for j in range(0, b)))
    return compressed


def sponge_hash(P, r, h, _x):
    """Uses Hirose's variant of the sponge construction to hash the
    message x using the permutation P with rate r, outputting a digest
    of size h.

    """
    x = _x[:]
    if P.input_size() <= r:
        raise Exception("rate must be strictly smaller than state size!")
    # Digest size and capacity check: we allow the digest size to be 3 bits
    # shorter than the theoretical target, as commonly used finite fields
    # usually have a characteristic size slightly under 2**256.
    if h * P.F.cardinality().nbits() < 2 * P.security_level - 3:
        raise Exception(f"digest size is too small for the targeted security level!")
    capacity = P.input_size() - r
    if capacity * P.F.cardinality().nbits() < 2 * P.security_level - 3:
        raise Exception(f"capacity is too small for the targeted security level!")

    # message padding (and domain separator computation)
    if len(x) % r == 0 and len(x) != 0:
        sigma = 1
    else:
        sigma = 0
        x += [1]
        # if x is still not long enough, append 0s
        if len(x) % r != 0:
            x += (r - (len(x) % r))*[0]
    padded_x = [[x[pos+i] for i in range(0, r)]
                for pos in range(0, len(x), r)]
    # absorption phase
    internal_state = [0] * P.input_size()
    for pos in range(0, len(padded_x)):
        for i in range(0, r):
            internal_state[i] += padded_x[pos][i]
        internal_state = P(internal_state)
        if pos == len(padded_x)-1:
            # adding sigma if it is the last block
            internal_state[-1] += sigma
    # squeezing
    digest = []
    pos = 0
    while len(digest) < h:
        digest.append(internal_state[pos])
        pos += 1
        if pos == r:
            pos = 0
            internal_state = P(internal_state)
    return digest


# !SECTION! Tests

def check_polynomial_verification(n_tests=10, q=2**63, alpha=3, n_rounds=3, n_cols=1):
    """Let `A` be an AnemoiPermutation instance with the parameters input to this function.

    It cerifies that the internal state values generated by
    A.eval_with_intermediate_state() are indeed roots of the equations
    generated by A.verification_polynomials(). This is repeated on
    n_tests random inputs.

    """
    A = AnemoiPermutation(q=q, alpha=alpha, n_rounds=n_rounds, n_cols=n_cols)
    # formal polynomial variables and equations
    p_vars = A.get_polynomial_variables()
    eqs = A.verification_polynomials(p_vars)
    A.print_verification_polynomials()
    # for n_tests random inputs, we check that the equations are
    # coherent with the actual intermediate values
    print("\n ======== Verification:")
    print(A)
    print("{} equations in {} variables.".format(
        len(eqs),
        (A.n_rounds+1) * 2 * A.n_cols,
    ))
    for t in range(0, n_tests):
        # generate random input
        x = [A.to_field(randint(0, A.q - 1))
             for i in range(0, A.n_cols)]
        y = [A.to_field(randint(0, A.q - 1))
             for i in range(0, A.n_cols)]
        # generate intermediate values, formal polynomial variables,
        # and equations
        iv = A.eval_with_intermediate_values(x, y)
        p_vars = A.get_polynomial_variables()
        eqs = A.verification_polynomials(p_vars)
        # obtain variable assignment from the actual evaluation
        assignment = {}
        for r in range(0, A.n_rounds+1):
            for i in range(0, A.n_cols):
                assignment[p_vars["X"][r][i]] = iv[r][0][i]
                assignment[p_vars["Y"][r][i]] = iv[r][1][i]
        # printing the value of the equations for the actual
        # intermediate states
        print("\n--- ", t, "(all values except the input should be 0)")
        print("input: ", x, y)
        for r in range(0, A.n_rounds):
            polynomial_values = [eqs[r*2*A.n_cols + i].subs(assignment)
                                 for i in range(0, 2*A.n_cols)]
            print("round {:3d}: {}\n           {}".format(
                r,
                polynomial_values[0::2],
                polynomial_values[1::2]
            ))


def test_jive(n_tests=10,
              q=2**63, alpha=3,
              n_rounds=None,
              n_cols=1,
              b=2,
              security_level=32):
    """Let `A` be and AnemoiPermutation instance with the parameters input
    to this function.

    This function evaluates Jive_b on random inputs using `A` as its
    permutation.

    """
    A = AnemoiPermutation(q=q, alpha=alpha, n_rounds=n_rounds, n_cols=n_cols, security_level=security_level)
    print(A)
    for t in range(0, n_tests):
        # generate random input
        x = [A.to_field(randint(0, A.q - 1))
             for i in range(0, A.n_cols)]
        y = [A.to_field(randint(0, A.q - 1))
             for i in range(0, A.n_cols)]
        print("x = {}\ny = {}\nAnemoiJive_{}(x,y) = {}".format(
            x,
            y,
            b,
            jive(A, b, x + y)
        ))


def test_sponge(n_tests=10,
                q=2**63,
                alpha=3,
                n_rounds=None,
                n_cols=1,
                b=2,
                security_level=32):
    """Let `A` be an AnemoiPermutation instance with the parameters input
    to this function.

    This function evaluates sponge on random inputs using `A` as its
    permutation, and a rate of A.input_size()-1 (so, a capacity of 1),
    and generates a 2 word output.

    """
    A = AnemoiPermutation(q=q, alpha=alpha, n_rounds=n_rounds, n_cols=n_cols, security_level=security_level)
    print(A)
    for t in range(0, n_tests):
        # generate random input of length t
        x = [A.to_field(randint(0, A.q - 1))
             for i in range(0, t)]
        print("x = {}\nAnemoiSponge(x) = {}".format(
            x,
            sponge_hash(A, 2, 2, x)
        ))

def generate_test_vectors_jive(P, b, n):
    """
    Outputs `n` deterministic test vectors for the provided AnemoiPermutation
    `P` with compression factor `b`.
    """
    assert n >= 4, "The number of test vectors should be greater than 4."
    m = hashlib.sha512(str(P).encode())
    m.update("Jive test vectors".encode())
    m.update(f"B={b}".encode())
    seed = Integer(m.digest().hex(), 16)

    inputs = []
    outputs = []
    inputs.append([P.F(0) for _ in range(P.input_size())])
    inputs.append([P.F(1) for _ in range(P.input_size())])
    inputs.append([P.F(0) for _ in range(P.n_cols)] + [P.F(1) for _ in range(P.n_cols)])
    inputs.append([P.F(1) for _ in range(P.n_cols)] + [P.F(0) for _ in range(P.n_cols)])
    for i in range(n - 4):
        input = []
        for _ in range(P.input_size()):
            input.append(P.to_field(seed))
            m.update(str(seed).encode())
            seed = Integer(m.digest().hex(), 16)
        inputs.append(input)
    for input in inputs:
        outputs.append(jive(P, b, input))

    print(
        "Test vectors for Anemoi instance over F_{:d}, n_rounds={:d}, n_cols={:d}, s={:d}".format(
        P.q,
        P.n_rounds,
        P.n_cols,
        P.security_level)
    )
    return (inputs, outputs)


def generate_test_vectors_sponge(P, r, h, n):
    """
    Outputs `n` deterministic test vectors for the provided AnemoiPermutation
    `P` with rate `r` and digest size `h`.
    """
    assert n >= 4, "The number of test vectors should be greater than 4."
    m = hashlib.sha512(str(P).encode())
    m.update("Sponge test vectors".encode())
    m.update(f"R={r}".encode())
    m.update(f"H={h}".encode())
    seed = Integer(m.digest().hex(), 16)

    inputs = []
    outputs = []
    inputs.append([P.F(0) for _ in range(P.input_size())])
    inputs.append([P.F(1) for _ in range(P.input_size())])
    inputs.append([P.F(0) for _ in range(P.n_cols)] + [P.F(1) for _ in range(P.n_cols)])
    inputs.append([P.F(1) for _ in range(P.n_cols)] + [P.F(0) for _ in range(P.n_cols)])
    for i in range(n - 4):
        input = []
        for _ in range(i+1):
            input.append(P.to_field(seed))
            m.update(str(seed).encode())
            seed = Integer(m.digest().hex(), 16)
        inputs.append(input)
    for input in inputs:
        outputs.append(sponge_hash(P, r, h, input))

    print(
        "Test vectors for Anemoi instance over F_{:d}, n_rounds={:d}, n_cols={:d}, s={:d}".format(
        P.q,
        P.n_rounds,
        P.n_cols,
        P.security_level)
    )
    return (inputs, outputs)


def generate_test_vectors_sbox(P, n):
    """
    Outputs `n` deterministic test vectors for the provided AnemoiPermutation
    `P` with rate `r`, digest size `h` and.
    """
    assert n >= 4, "The number of test vectors should be greater than 4."
    m = hashlib.sha512(str(P).encode())
    m.update("S-Box test vectors".encode())
    seed = Integer(m.digest().hex(), 16)

    inputs = []
    outputs = []
    inputs.append([P.F(0) for _ in range(P.input_size())])
    inputs.append([P.F(1) for _ in range(P.input_size())])
    inputs.append([P.F(0) for _ in range(P.n_cols)] + [P.F(1) for _ in range(P.n_cols)])
    inputs.append([P.F(1) for _ in range(P.n_cols)] + [P.F(0) for _ in range(P.n_cols)])

    for _ in range(n - 4):
        input = []
        for _ in range(P.input_size()):
            input.append(P.to_field(seed))
            m.update(str(seed).encode())
            seed = Integer(m.digest().hex(), 16)
        inputs.append(input)
    for input in inputs:
        x = [0 for i in range(P.n_cols)]
        y = [0 for i in range(P.n_cols)]
        for i in range(P.n_cols):
            x[i], y[i] = P.evaluate_sbox(input[i], input[P.n_cols + i])
        x.extend(y)
        outputs.append(x)

    return (inputs, outputs)


def generate_test_vectors_mds(P, n):
    """
    Outputs `n` deterministic test vectors for the provided AnemoiPermutation
    `P` with rate `r`, digest size `h` and.
    """
    assert n >= 4, "The number of test vectors should be greater than 4."
    m = hashlib.sha512(str(P).encode())
    m.update("MDS test vectors".encode())
    seed = Integer(m.digest().hex(), 16)

    inputs = []
    outputs = []
    inputs.append([P.F(0) for _ in range(P.input_size())])
    inputs.append([P.F(1) for _ in range(P.input_size())])
    inputs.append([P.F(0) for _ in range(P.n_cols)] + [P.F(1) for _ in range(P.n_cols)])
    inputs.append([P.F(1) for _ in range(P.n_cols)] + [P.F(0) for _ in range(P.n_cols)])
    for _ in range(n - 4):
        input = []
        for _ in range(P.input_size()):
            input.append(P.to_field(seed))
            m.update(str(seed).encode())
            seed = Integer(m.digest().hex(), 16)
        inputs.append(input)
    for input in inputs:
        x,y = P.linear_layer(input[0:P.n_cols], input[P.n_cols:])
        x.extend(y)
        outputs.append(x)

    return (inputs, outputs)


if __name__ == "__main__":

    # These are the first circulant matrices being found by the circulant_mds_matrix()
    # method above. These are precomputed for faster initiatialization of large Anemoi
    # instances.
    CIRCULANT_FP5_MDS_MATRIX = matrix.circulant([1, 1, 3, 4, 5])
    CIRCULANT_FP6_MDS_MATRIX = matrix.circulant([1, 1, 3, 4, 5, 6])
    CIRCULANT_FP7_MDS_MATRIX = matrix.circulant([1, 2, 3, 5, 5, 6, 7])
    CIRCULANT_FP8_MDS_MATRIX = matrix.circulant([1, 2, 3, 5, 7, 8, 8, 9])
    CIRCULANT_FP9_MDS_MATRIX = matrix.circulant([1, 3, 5, 6, 8, 9, 9, 10, 11])
    CIRCULANT_FP10_MDS_MATRIX = matrix.circulant([1, 2, 5, 6, 8, 11, 11, 12, 13, 14])

    # 128-bit security level instantiations

    A_BLS_12_381_BASEFIELD_1_COL_128_BITS = AnemoiPermutation(
        q=BLS12_381_BASEFIELD,
        n_cols=1,
        security_level=128
    )
    A_BLS_12_381_BASEFIELD_2_COL_128_BITS = AnemoiPermutation(
        q=BLS12_381_BASEFIELD,
        n_cols=2,
        security_level=128
    )
    A_BLS_12_381_BASEFIELD_3_COL_128_BITS = AnemoiPermutation(
        q=BLS12_381_BASEFIELD,
        n_cols=3,
        security_level=128
    )

    A_JUBJUB_BASEFIELD_1_COL_128_BITS = AnemoiPermutation(
        q=BLS12_381_SCALARFIELD,
        n_cols=1,
        security_level=128
    )
    A_JUBJUB_BASEFIELD_2_COL_128_BITS = AnemoiPermutation(
        q=BLS12_381_SCALARFIELD,
        n_cols=2,
        security_level=128
    )
    A_JUBJUB_BASEFIELD_3_COL_128_BITS = AnemoiPermutation(
        q=BLS12_381_SCALARFIELD,
        n_cols=3,
        security_level=128
    )

    A_BLS_12_377_BASEFIELD_1_COL_128_BITS = AnemoiPermutation(
        q=BLS12_377_BASEFIELD,
        n_cols=1,
        security_level=128
    )
    A_BLS_12_377_BASEFIELD_2_COL_128_BITS = AnemoiPermutation(
        q=BLS12_377_BASEFIELD,
        n_cols=2,
        security_level=128
    )
    A_BLS_12_377_BASEFIELD_3_COL_128_BITS = AnemoiPermutation(
        q=BLS12_377_BASEFIELD,
        n_cols=3,
        security_level=128
    )

    A_ED_ON_BLS_12_377_BASEFIELD_1_COL_128_BITS = AnemoiPermutation(
        q=BLS12_377_SCALARFIELD,
        n_cols=1,
        security_level=128
    )
    A_ED_ON_BLS_12_377_BASEFIELD_2_COL_128_BITS = AnemoiPermutation(
        q=BLS12_377_SCALARFIELD,
        n_cols=2,
        security_level=128
    )
    A_ED_ON_BLS_12_377_BASEFIELD_3_COL_128_BITS = AnemoiPermutation(
        q=BLS12_377_SCALARFIELD,
        n_cols=3,
        security_level=128
    )

    A_BN_254_BASEFIELD_1_COL_128_BITS = AnemoiPermutation(
        q=BN_254_BASEFIELD,
        n_cols=1,
        security_level=128
    )
    A_BN_254_BASEFIELD_2_COL_128_BITS = AnemoiPermutation(
        q=BN_254_BASEFIELD,
        n_cols=2,
        security_level=128
    )
    A_BN_254_BASEFIELD_3_COL_128_BITS = AnemoiPermutation(
        q=BN_254_BASEFIELD,
        n_cols=3,
        security_level=128
    )

    A_BN_254_SCALARFIELD_1_COL_128_BITS = AnemoiPermutation(
        q=BN_254_SCALARFIELD,
        n_cols=1,
        security_level=128
    )
    A_BN_254_SCALARFIELD_2_COL_128_BITS = AnemoiPermutation(
        q=BN_254_SCALARFIELD,
        n_cols=2,
        security_level=128
    )
    A_BN_254_SCALARFIELD_3_COL_128_BITS = AnemoiPermutation(
        q=BN_254_SCALARFIELD,
        n_cols=3,
        security_level=128
    )

    A_PALLAS_BASEFIELD_1_COL_128_BITS = AnemoiPermutation(
        q=PALLAS_BASEFIELD,
        n_cols=1,
        security_level=128
    )
    A_PALLAS_BASEFIELD_2_COL_128_BITS = AnemoiPermutation(
        q=PALLAS_BASEFIELD,
        n_cols=2,
        security_level=128
    )
    A_PALLAS_BASEFIELD_3_COL_128_BITS = AnemoiPermutation(
        q=PALLAS_BASEFIELD,
        n_cols=3,
        security_level=128
    )

    A_VESTA_BASEFIELD_1_COL_128_BITS = AnemoiPermutation(
        q=VESTA_BASEFIELD,
        n_cols=1,
        security_level=128
    )
    A_VESTA_BASEFIELD_2_COL_128_BITS = AnemoiPermutation(
        q=VESTA_BASEFIELD,
        n_cols=2,
        security_level=128
    )
    A_VESTA_BASEFIELD_3_COL_128_BITS = AnemoiPermutation(
        q=VESTA_BASEFIELD,
        n_cols=3,
        security_level=128
    )

    A_GOLDILOCKS_64_FIELD_4_COL_128_BITS = AnemoiPermutation(
        q=GOLDILOCKS_64_FIELD,
        n_cols=4,
        security_level=128
    )
    A_GOLDILOCKS_64_FIELD_5_COL_128_BITS = AnemoiPermutation(
        q=GOLDILOCKS_64_FIELD,
        mat=CIRCULANT_FP5_MDS_MATRIX,
        n_cols=5,
        security_level=128)
    A_GOLDILOCKS_64_FIELD_6_COL_128_BITS = AnemoiPermutation(
        q=GOLDILOCKS_64_FIELD,
        mat=CIRCULANT_FP6_MDS_MATRIX,
        n_cols=6,
        security_level=128)

