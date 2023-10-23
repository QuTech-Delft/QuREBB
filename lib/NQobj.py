import numbers
from copy import deepcopy

import numpy as np
import qutip as qt
from qutip.cy.spconvert import arr_coo2fast, cy_index_permute
from qutip.permute import _permute  # To support the _permute2 function


class NQobj(qt.Qobj):
    """
    The NQobj (Named Qobj) class is an extension of the QuTip's Qobj class. It is designed to enable
    the use of symbolic names to represent and index modes in a quantum object, as opposed to using
    numerical indices.

    One of the primary motivations behind the NQobj is to facilitate mathematical operations (+, -, *)
    between different NQobj instances by automatically matching modes based on their symbolic names.
    This is particularly useful when working with multi-partite quantum systems where the tensor structure
    can be complex.

    In addition to the standard attributes and methods inherited from the Qobj class, NQobj introduces
    an additional attribute, `names`, which is structured similarly to the `dims` attribute of Qobj.
    The `names` attribute is a list that provides symbolic names to each mode, making it easier to
    identify and operate on specific modes.

    Parameters
    ----------
    The NQobj takes the arguments and kwarg of QuTiP Qobj, and names.

    names: str, list of str or list of two lists of str
        if names is str                     :    names = "A"          ->   names = [["A"], ["A"]]
        if names is list of str             :    names = ["A", "B"]   ->   names = [["A", "B"], ["A", "B"]]
        if names is list of two list of str :    names = names

        names for the modes: the shape has to match dims after the above correction.


    Attributes
    ----------
    Same as for QuTiP Qobj, including:
    names: List of the names of dimensions for keeping track of the tensor structure.

    """

    def __init__(self, *args, **kwargs):
        """
        NQobj constructor.

        Parameters:
        - args: Arguments for the parent Qobj class
        - kwargs: Keyword arguments for the parent Qobj class, and additional `names` and `kind` arguments for NQobj
        """

        # Extract names and kind from kwargs, if present
        names = kwargs.pop("names", None)
        kind = kwargs.pop("kind", None)

        # If a NQobj is supplied as input and no new names are given, use the existing ones.
        try:
            if isinstance(args[0], NQobj) and names is None:
                names = (args[0]).names
        except IndexError:
            pass

        # Initialize the Qobj without the names.
        super().__init__(*args, **kwargs)

        # Check and validate the format of names, then add them as an attribute to the instance.
        if names is None:
            raise AttributeError("names is a compulsary attribute.")

        # Handle cases where names is a list
        if isinstance(names, list):
            # Ensure proper structure for the 2D list format
            if len(names) == 2 and isinstance(names[0], list) and isinstance(names[1], list):
                # Check if all names are strings and unique
                if not all(isinstance(i, str) for i in names[0] + names[1]):
                    raise ValueError("A name must be a string.")
                if len(names[0]) != len(set(names[0])) or len(names[1]) != len(set(names[1])):
                    raise ValueError("Do not use duplicate names.")
                if (len(names[0]), len(names[1])) != self.shape_dims:
                    raise ValueError("The number of names must match the shape_dims.")
                self.names = names
            else:
                # Ensure all names are strings and unique for 1D list format
                if not all(isinstance(i, str) for i in names):
                    raise ValueError("A name must be a string.")
                if len(names) != len(set(names)):
                    raise ValueError("Do not use duplicate names.")
                if not (len(names) == self.shape_dims[0] and len(names) == self.shape_dims[1]):
                    raise ValueError("The number of names must match the shape_dims.")
                self.names = [names, names]

        # Handle case where names is a string
        elif isinstance(names, str):
            if self.shape_dims != (1, 1):
                raise ValueError("Names can only be a string if there shape_dims is (1, 1).")
            self.names = [[names], [names]]
        else:
            raise TypeError("names should be a string or 1d or 2d list ")

        # Assign the type of NQobj (operator or state)
        if kind in ("oper", "state"):
            self.kind = kind
        elif kind is None:  # If kind is not give, try to extract it from the Qobj form
            if self.isket or self.isbra:
                self.kind = "state"
            if self.isoper:
                if (
                    set(self.names[0]) == set(self.names[1])
                    and sorted(self.dims[0]) == sorted(self.dims[1])
                    and self.isherm
                ):
                    raise ValueError(
                        'The kind cannot be determined automatically and should be provedid as kw ("oper" or "state")'
                    )
                self.kind = "oper"
        else:
            raise ValueError('kind can only be "oper", "state", None')

    def copy(self):
        """Create an identical copy of the NQobj."""
        q = super().copy()
        return NQobj(q, names=deepcopy(self.names), kind=self.kind)

    def __add__(self, other):
        """
        Define addition for the NQobj when it's on the left side (e.g., Qobj + 4).
        """

        # Check if the other operand is also an NQobj
        if isinstance(other, NQobj):
            # Ensure the types and kinds of both NQobj are the same for addition
            if not self.type == other.type:
                raise ValueError("Addition and substraction are only allowed for two NQobj of the same type.")
            if not self.kind == other.kind:
                raise ValueError("Addition and substraction are only allowed for two NQobj of the same kind.")

            # Handle specific cases where the NQobj is a ket, bra, or operator
            if self.isket or self.isbra or self.isoper:
                names = _add_find_required_names(self, other)
                if not names == self.names or not names == other.names:
                    missing_self = _find_missing_names(self.names, names)
                    missing_other = _find_missing_names(other.names, names)
                    missing_dict_self = _find_missing_dict(missing_self, other, transpose=False)
                    missing_dict_other = _find_missing_dict(missing_other, self, transpose=False)
                    self = _adding_missing_modes(self, missing_dict_self, kind=self.kind)
                    self = self.permute(names)
                    other = _adding_missing_modes(other, missing_dict_other, kind=other.kind)
                    other = other.permute(names)
                Qobj_result = super(NQobj, self).__add__(other)
                return NQobj(Qobj_result, names=names, kind=self.kind)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    def __mul__(self, other):
        """
        Define multiplication for the NQobj when it's on the left side (e.g., NQobj * 4).
        """

        # Check if the other operand is also an NQobj
        if isinstance(other, NQobj):
            # Identify the required names for multiplication and find any missing names in both NQobjs
            names_self, names_other = _mul_find_required_names(self, other)
            if not names_self == self.names or not names_other == other.names:
                # Find missing names and prepare for multiplication
                missing_self = _find_missing_names(self.names, names_self)
                missing_other = _find_missing_names(other.names, names_other)
                missing_dict_self = _find_missing_dict(missing_self, other, transpose=True)
                missing_dict_other = _find_missing_dict(missing_other, self, transpose=True)
                self = _adding_missing_modes(self, missing_dict_self, kind=self.kind)
                self = self.permute(names_self)
                other = _adding_missing_modes(other, missing_dict_other, kind=other.kind)
                other = other.permute(names_other)

            # Perform the multiplication operation
            Qobj_result = super(NQobj, self).__mul__(other)

            # Handle special case where the result is a scalar (Return a scalar as Qobj and not as NQobj).
            if Qobj_result.shape == (1, 1):
                return qt.Qobj(Qobj_result)
            else:
                names = [names_self[0], names_other[1]]
                # Modes with size (1, 1) are reduced to scalars and don't need names
                for name in names[0].copy():
                    if self._dim_of_name(name)[0] == 1 and other._dim_of_name(name)[1] == 1:
                        names[0].remove(name)
                        names[1].remove(name)

                # Determine the kind of the result based on the kinds of the operands
                if self.kind == "oper" and other.kind == "oper":
                    kind = "oper"
                elif self.kind == "state" and other.kind == "state":
                    kind = "oper"
                else:
                    kind = "state"

                return NQobj(Qobj_result, names=names, kind=kind)

        # Handle multiplication with a number
        elif isinstance(other, numbers.Number):
            return NQobj(super().__mul__(other), names=self.names, kind=self.kind)

        # Handle multiplication with a plain Qobj
        elif isinstance(other, qt.Qobj):
            return super().__mul__(other)
        else:
            raise NotImplementedError

    def __rmul__(self, other):
        """
        Define multiplication for the NQobj when it's on the right side (e.g., 4 * NQobj).
        """
        # Handle multiplication with a number
        if isinstance(other, numbers.Number):
            # Use the multiplication method defined for NQobj
            return self.__mul__(other)

        # Handle multiplication with a plain Qobj
        elif isinstance(other, qt.Qobj):
            # Perform multiplication from the perspective of the Qobj, putting NQobj on the right
            return other.__mul__(self)

    def __div__(self, other):
        """
        Division operation, intended for division by numbers only.
        Returns a new NQobj with the result of the division.
        """
        return NQobj(super().__div__(other), names=self.names, kind=self.kind)

    def __neg__(self):
        """
        Negation operation.
        Returns a new NQobj with the negated values.
        """
        return NQobj(super().__neg__(), names=self.names, kind=self.kind)

    def __eq__(self, other):
        """
        Equality operator to compare two NQobj instances.
        Compares both the Qobj content and the names attribute.
        """
        same_Qobj = super().__eq__(other)
        same_names = self.names == other.names
        return same_Qobj and same_names

    def __pow__(self, n, m=None):
        """
        Power operation for raising the NQobj to a certain power.
        Returns a new NQobj with the result.
        """
        return NQobj(super().__pow__(n, m=m), names=self.names, kind=self.kind)

    def __str__(self):
        """
        String representation of the NQobj.
        Appends the names and kind attributes to the Qobj's string representation.
        """
        return super().__str__() + "\nnames: " + self.names.__str__() + "\nkind: " + self.kind.__str__()

    def _repr_latex_(self):
        """
        Generate a LaTeX representation of the NQobj instance.
        Useful for formatted output in IPython notebooks.
        """
        string = super()._repr_latex_()
        string += r" $\\$ "
        string += f"names = {self.names}"
        string += f", kind = {self.kind}"
        return string

    def dag(self):
        """
        Returns the adjoint (dagger) of the NQobj.
        Also swaps the names of the ket and bra dimensions.
        """
        out = super().dag()
        names = [self.names[1], self.names[0]]
        return NQobj(out, names=names, kind=self.kind)

    def proj(self):
        """
        Form the projector from a given ket or bra vector of the NQobj.
        Returns a new NQobj that represents the projector.
        """
        return NQobj(super().proj(), names=self.names, kind="oper")

    def unit(self, *args, **kwargs):
        """
        Normalize the NQobj to unity, either as an operator or state.
        Returns a new NQobj that is normalized.
        """
        return NQobj(super().unit(*args, **kwargs), names=self.names, kind=self.kind)

    def ptrace(self, sel, keep=True):
        if self.dims[0] != self.dims[1]:
            ValueError("ptrace works only on a square oper")
        if self.names[0] != self.names[1]:
            ValueError("Names of both axis are not the same")

        if isinstance(sel, list):
            if all(isinstance(i, int) for i in sel):
                pass
            elif all(isinstance(i, str) for i in sel):
                sel = [self.names[0].index(name) for name in sel]
            else:
                raise ValueError("sel must be list of only int or str")
        elif isinstance(sel, str):
            sel = [self.names[0].index(sel)]
        else:
            raise TypeError("sel needs to be a list with int or str")

        if not keep:
            sel = [i for i in range(self.shape_dims[0]) if not i in sel]

        names = [name for i, name in enumerate(self.names[0]) if i in sel]

        return NQobj(super().ptrace(sel), names=[names, names], kind=self.kind)

    def permute(self, order):
        if isinstance(order, list) and all(isinstance(i, str) for i in order):
            order_index = []
            if self.names[0] == self.names[1]:
                for name in order:
                    order_index.append(self.names[0].index(name))
                order = order_index
            else:
                order = [order, order]
        if (
            len(order) == 2
            and all(isinstance(i, list) for i in order)
            and all(isinstance(i, str) for i in order[0] + order[1])
        ):
            order_index = [[], []]
            for name in order[0]:
                order_index[0].append(self.names[0].index(name))
            for name in order[1]:
                order_index[1].append(self.names[1].index(name))
            order = order_index

        # Replicate working of permute of Qobj but with _permute2.
        q = qt.Qobj()
        q.data, q.dims = _permute2(self, order)
        q = q.tidyup() if qt.settings.auto_tidyup else q
        if isinstance(order, list) and all(isinstance(i, int) for i in order):
            order = [order, order]

        # Rearange the names
        names_0 = [self.names[0][i] for i in order[0]]
        names_1 = [self.names[1][i] for i in order[1]]
        names = [names_0, names_1]
        return NQobj(q, names=names, kind=self.kind)

    def rename(self, name, new_name):
        """Rename a mode called name to new_name."""
        if name == new_name:
            pass
        elif new_name in self.names[0] + self.names[1]:
            raise ValueError("You cannot use a new_name which is already used.")
        elif name not in self.names[0] + self.names[1]:
            raise ValueError("The name you want to replace is not present in the NQobj.")
        else:
            for i in range(2):
                try:
                    self.names[i][self.names[i].index(name)] = new_name
                except ValueError:
                    pass

    def _dim_of_name(self, name):
        """Return the shape of the submatrix with name."""
        try:
            index_0 = self.names[0].index(name)
            dim_0 = self.dims[0][index_0]
        except ValueError:
            dim_0 = None
        try:
            index_1 = self.names[1].index(name)
            dim_1 = self.dims[1][index_1]
        except ValueError:
            dim_1 = None
        return (dim_0, dim_1)

    def expm(self):
        if self.names[0] == self.names[1] and self.dims[0] == self.dims[1]:
            return NQobj(super().expm(), names=self.names, kind=self.kind)
        else:
            new_self = self.expand()
            new_self.permute([new_self.names[0], new_self.names[0]])
            if new_self.names[0] == new_self.names[1] and new_self.dims[0] == new_self.dims[1]:
                return NQobj(
                    super(NQobj, new_self).expm(),
                    names=new_self.names,
                    kind=new_self.kind,
                )
            else:
                raise ValueError("For exponentiation the matrix should have square submatrixes.")

    @property
    def shape_dims(self):
        """
        Compute and return the shape dimensions of the NQobj based on the dims attribute.

        Returns:
        - Tuple of integers representing the number of dimensions in the row (0th) and column (1st) direction.
        """
        shape_dims_0 = len(self.dims[0])
        shape_dims_1 = len(self.dims[1])
        return (shape_dims_0, shape_dims_1)

    def trans(self):
        """
        Compute the transpose of the NQobj.

        Returns:
        - A new NQobj which is the transpose of the current NQobj, with the names of bra and ket swapped.
        """
        return NQobj(super().trans(), names=[self.names[1], self.names[0]], kind=self.kind)

    def expand(self):
        """
        Expand the NQobj to include all modes present in both rows and columns.

        Returns:
        - A new NQobj which is expanded to include all modes.
        """

        # If the names in rows and columns are the same, no expansion needed
        if self.names[0] == self.names[1]:
            return self

        # Copy the kind of the current NQobj
        kind = self.kind

        # Determine the set of all names required for expansion
        required_names = self.names[0] + self.names[1]
        required_names = list(dict.fromkeys(required_names))  # Remove duplicates while keeping order (Python 3.7+)

        # Identify which names are missing from the current NQobj
        missing = _find_missing_names(self.names, [required_names, required_names])

        # Construct a dictionary of the missing names and their corresponding dimensions
        missing_dict = {name: self._dim_of_name(name) for name in missing[0] + missing[1]}

        # Deep copy the current names to avoid modifying the original
        names = deepcopy(self.names)

        # Iterate over the missing names and their dimensions
        for name, dims in missing_dict.items():
            if dims[0] is None:
                names[0].append(name)
                self = qt.tensor(self, qt.basis(dims[1]))
                self.dims[1].pop()
            elif dims[1] is None:
                names[1].append(name)
                self = qt.tensor(self, qt.basis(dims[0]).dag())
                self.dims[0].pop()

        # Return the expanded NQobj, permuted to have the required names in order
        return NQobj(self, names=names, kind=kind).permute(required_names)


def tensor(*args):
    """Perform tensor product between multiple NQobj, similar to tensor from qutip."""
    names = [[], []]
    for arg in args:
        names[0] += arg.names[0]
        names[1] += arg.names[1]
    q = qt.tensor(*args)
    kinds = np.array([q.kind for q in args])
    if not np.all(kinds == kinds[0]):
        raise AttributeError("For tensor product the kind of all NQobj should be the same.")
    out = NQobj(q, names=names, kind=kinds[0])
    return out


def ket2dm(Q):
    return NQobj(qt.ket2dm(Q), names=Q.names, kind="state")


def name(Q, names, kind=None):
    return NQobj(Q, names=names, kind=kind)


def fidelity(A, B):
    if not ((A.isket or A.isbra or A.isoper) and (B.isket or B.isbra or B.isoper)):
        raise TypeError("fidelity can only be calculated for ket, bra or oper.")
    if not set(A.names[0]) == set(A.names[1]) or not set(B.names[0]) == set(B.names[1]):
        raise TypeError("Names of colums and rows need to be the same.")
    if not set(A.names[0]) == set(B.names[0]):
        raise TypeError("fidelity needs both objects to have the same names.")
    return qt.fidelity(A, B.permute(A.names))


def _permute2(Q, order):
    """
    Similar function as _permute from qutip but this allows for permutation of non-square matrixes.
    In this case order needs to be a list of two list with the permutation for each axis. e.g. [[1,0], [1,2,0]]
    """
    equal_dims = Q.dims[0] == Q.dims[1]
    if isinstance(order, list):
        if Q.isoper:
            if equal_dims and all(isinstance(i, int) for i in order):
                use_qutip = True
            elif (
                len(order) == 2
                and all(isinstance(i, list) for i in order)
                and all(isinstance(i, int) for i in order[0] + order[1])
            ):
                use_qutip = False
            else:
                raise TypeError("Order should be a list of int or a list of two list with int.")
        elif Q.isbra or Q.isket:
            if all(isinstance(i, int) for i in order):
                use_qutip = True
            elif (
                len(order) == 2
                and all(isinstance(i, list) for i in order)
                and all(isinstance(i, int) for i in order[0] + order[1])
            ):
                use_qutip = False
        else:
            use_qutip = True
            # Make sure that it works if [order, order] is supplied for a different object type then oper.
            if (
                len(order) == 2
                and all(isinstance(i, list) for i in order)
                and all(isinstance(i, int) for i in order[0] + order[1])
                and order[0] == order[1]
            ):
                order = order[0]
    if use_qutip:
        return _permute(Q, order)
    else:
        # Copy the functionality from qutip but allow for different order for rows and collums.
        Qcoo = Q.data.tocoo()
        cy_index_permute(
            Qcoo.row,
            np.array(Q.dims[0], dtype=np.int32),
            np.array(order[0], dtype=np.int32),
        )
        cy_index_permute(
            Qcoo.col,
            np.array(Q.dims[1], dtype=np.int32),
            np.array(order[1], dtype=np.int32),
        )

        new_dims = [[Q.dims[0][i] for i in order[0]], [Q.dims[1][i] for i in order[1]]]
        return (
            arr_coo2fast(Qcoo.data, Qcoo.row, Qcoo.col, Qcoo.shape[0], Qcoo.shape[1]),
            new_dims,
        )


######################### Function to support __mul__ and __add__ functions #############################


def _mul_find_required_names(Q_left, Q_right):
    """
    Identify the required mode names for multiplication between two NQobjs.
    """

    # If a mode has the form of a ket in Q_left or a bra in Q_right they don't have to be matched between the objects
    # Identify modes in Q_left and Q_right that don't need to be matched for multiplication
    names_Q_right_for_overlap = [name for name in Q_right.names[0] if not Q_right._dim_of_name(name)[0] == 1]
    names_Q_left_for_overlap = [name for name in Q_left.names[1] if not Q_left._dim_of_name(name)[1] == 1]

    # Goal: to get a list with modes that need to appear in both objects to perform the multiplication, the overlap.
    # Determine the overlapping mode names between the two NQobjs based on their kinds
    if Q_right.kind == "state" and Q_left.kind == "oper":
        overlap = names_Q_right_for_overlap + names_Q_left_for_overlap
    else:
        overlap = names_Q_left_for_overlap + names_Q_right_for_overlap
    overlap = list(dict.fromkeys(overlap))  # Remove duplicates while keeping order (Python 3.7+)

    # Reorder mode names to match multiplication requirements
    names_Q_left = [overlap + names for names in Q_left.names]
    names_Q_left = [
        list(dict.fromkeys(names)) for names in names_Q_left
    ]  # Remove duplicates while keeping order (Python 3.7+)
    names_Q_right = [overlap + names for names in Q_right.names]
    names_Q_right = [
        list(dict.fromkeys(names)) for names in names_Q_right
    ]  # Remove duplicates while keeping order (Python 3.7+)
    return names_Q_left, names_Q_right


def _add_find_required_names(Q_left, Q_right):
    """
    Identify the required mode names for the addition of two NQobjs.
    """
    names = [Q_left.names[i] + Q_right.names[i] for i in range(2)]
    names = [
        list(dict.fromkeys(name_list)) for name_list in names
    ]  # Remove duplicates while keeping order (Python 3.7+)
    return names


def _find_missing_names(names, required_names):
    """
    Determine which mode names are missing when comparing two lists of names
    """
    missing = [list(set(required_names[i]) - set(names[i])) for i in range(2)]
    return missing


def _find_missing_dict(missing_names, Q_other, transpose=False):
    """
    Map the missing mode names to their corresponding dimensions in another NQobj.
    """
    missing_dict = {}
    for name in set(missing_names[0] + missing_names[1]):
        dims = list(Q_other._dim_of_name(name))
        if name not in missing_names[0]:
            dims[0] = None
        if name not in missing_names[1]:
            dims[1] = None
        if transpose:
            dims.reverse()
        missing_dict[name] = dims
    return missing_dict


def _adding_missing_modes(Q, dict_missing_modes, kind="oper"):
    """
    Add missing modes to an NQobj based on the missing modes dictionary.

    Parameters:
    - Q: The NQobj to which missing modes are added.
    - dict_missing_modes: A dictionary mapping missing mode names to their dimensions.
    - kind: Type of the NQobj ("oper" for operators or "state" for quantum states). Default is "oper".

    Returns:
    - The NQobj with missing modes added.
    """

    modes = []  # List to collect modes that need to be added to the NQobj

    # Iterate through each missing mode and its dimensions
    for name, dims in dict_missing_modes.items():

        # If the NQobj kind is an operator
        if kind == "oper":
            assert dims[0] == dims[1], "For adding eye matrixes they need to be square"
            modes.append(NQobj(qt.qeye(dims[0]), names=name, kind="oper"))

        # If the NQobj kind is a quantum state
        if kind == "state":
            if not None in dims:
                modes.append(
                    NQobj(
                        qt.basis(dims[0], 0) * qt.basis(dims[1], 0).dag(),
                        names=name,
                        kind="state",
                    )
                )
            elif dims[0] is None:
                modes.append(NQobj(qt.basis(dims[1], 0).dag(), names=[[], [name]], kind="state"))
            elif dims[1] is None:
                modes.append(NQobj(qt.basis(dims[0], 0), names=name, kind="state"))

    # Return a tensor product of the original NQobj with the added modes
    return tensor(Q, *modes)
