import cuda.tile as ct
import cuda.tile.compilation as ctc

def make_array_constraint(dtype, rank, stride_constant=None) -> ctc.ArrayConstraint:
    return ctc.ArrayConstraint(
        dtype=dtype,
        ndim=rank,
        index_dtype=ct.int32,
        stride_lower_bound_incl=0,
        stride_constant=stride_constant,
        alias_groups=[],
        may_alias_internally=False,
    )
