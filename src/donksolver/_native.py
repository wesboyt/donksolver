"""ctypes binding to the native core. Internal; use the public modules."""
import ctypes as C
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ENV = "DONKSOLVER_LIB"


def _lib_path():
    """$DONKSOLVER_LIB if set, else the library inside the installed package."""
    env = os.environ.get(_ENV)
    if env:
        return env
    name = {"win32": "donksolver_core.dll", "darwin": "donksolver_core.dylib"}.get(
        sys.platform, "donksolver_core.so")
    return os.path.join(_HERE, "_native", name)


def _load():
    path = _lib_path()
    if not os.path.exists(path):
        raise ImportError(
            "donksolver's native core was not found (%s). Install a wheel "
            "(`pip install donksolver`), or build from a checkout with "
            "`pip install .`; a library built by hand with CMake can be named "
            "in $%s." % (path, _ENV))
    return C.CDLL(path)


lib = _load()

c_i32p = C.POINTER(C.c_int32)
c_i64p = C.POINTER(C.c_int64)
c_u32p = C.POINTER(C.c_uint32)
c_u64p = C.POINTER(C.c_uint64)
c_u8p = C.POINTER(C.c_uint8)
c_dblp = C.POINTER(C.c_double)
c_fltp = C.POINTER(C.c_float)
PROGRESS = C.CFUNCTYPE(None, C.c_char_p, C.c_double)


def _sig(name, res, *args):
    f = getattr(lib, name)
    f.restype = res
    f.argtypes = list(args)
    return f


_sig("ds_version", C.c_char_p)
_sig("ds_last_error", C.c_char_p)
_sig("ds_eval", C.c_uint32, c_i32p, C.c_int)
_sig("ds_eval_reference", C.c_uint32, c_i32p, C.c_int)
_sig("ds_eval_batch", None, c_i32p, C.c_int64, C.c_int, c_u32p)
_sig("ds_settle", None, C.c_int, c_i64p, c_u8p, c_u32p, C.c_int, C.c_int, C.c_double, C.c_int64,
     C.c_int, C.c_int, c_dblp)
_sig("ds_replay", C.c_int, C.c_int, c_i64p, C.c_int64, C.c_int64, c_i32p, c_i32p, c_i64p, C.c_int,
     c_i64p, c_i64p)
_sig("ds_abs_build", C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int, C.c_uint64, C.c_uint64, PROGRESS)
_sig("ds_abs_build_nested", C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int,
     PROGRESS)
_sig("ds_abs_params", C.c_int, C.c_void_p, c_i32p)
_sig("ds_board_class", C.c_int, C.c_int, c_i32p, C.c_int)
_sig("ds_abs_debug_nested", C.c_int, C.c_void_p, C.c_int, c_i32p, c_dblp, c_dblp, c_i32p)
_sig("ds_abs_load", C.c_void_p, C.c_char_p)
_sig("ds_abs_save", C.c_int, C.c_void_p, C.c_char_p)
_sig("ds_abs_free", None, C.c_void_p)
_sig("ds_abs_hash", C.c_uint64, C.c_void_p)
_sig("ds_abs_buckets", C.c_int, C.c_void_p, c_i32p)
_sig("ds_abs_bucket", C.c_int, C.c_void_p, C.c_int, c_i32p, c_i32p)
_sig("ds_hand_strength", C.c_double, c_i32p, c_i32p)
_sig("ds_tree_info", C.c_int, C.c_char_p, C.c_char_p, C.c_int)
_sig("ds_solver_create", C.c_void_p, C.c_char_p)
_sig("ds_solver_free", None, C.c_void_p)
_sig("ds_solver_info", C.c_int, C.c_void_p, C.c_char_p, C.c_int)
_sig("ds_solver_run", C.c_int, C.c_void_p, C.c_uint64, C.c_int)
_sig("ds_solver_root_ev", C.c_int, C.c_void_p, c_dblp, c_dblp, c_u64p)
_sig("ds_solver_reset_ev", None, C.c_void_p)
_sig("ds_solver_reset_average", None, C.c_void_p)
_sig("ds_solver_entries", C.c_uint64, C.c_void_p)
_sig("ds_solver_iterations", C.c_uint64, C.c_void_p)
_sig("ds_solver_exact", C.c_int, C.c_void_p, c_dblp, c_dblp)
_sig("ds_solver_node", C.c_int, C.c_void_p, C.c_uint32, c_i32p, c_u32p, c_u8p, c_fltp, c_i64p)
_sig("ds_solver_row", C.c_int, C.c_void_p, C.c_uint32, C.c_int, c_i32p)
_sig("ds_solver_strategy", C.c_int, C.c_void_p, C.c_uint32, C.c_int, C.c_int, c_dblp)
_sig("ds_solver_debug_regret", C.c_int, C.c_void_p, C.c_int, C.c_int, C.c_uint64, C.c_int, C.c_uint64, c_dblp)
_sig("ds_solver_shadow", C.c_int, C.c_void_p, C.c_int, C.c_int)
_sig("ds_solver_shadow_compare", C.c_double, C.c_void_p, C.c_double, c_dblp)
_sig("ds_solver_save_state", C.c_int, C.c_void_p, C.c_char_p)
_sig("ds_solver_load_state", C.c_int, C.c_void_p, C.c_char_p)
_sig("ds_solver_save_policy", C.c_int, C.c_void_p, C.c_char_p)
_sig("ds_sample_ranges", C.c_int, C.c_int, c_i32p, c_i32p, c_dblp, c_i32p, C.c_int, C.c_uint64, C.c_uint64,
     C.c_int, c_i32p)
_sig("ds_bundle_load", C.c_void_p, C.c_char_p, C.c_char_p, C.c_double)
_sig("ds_bundle_free", None, C.c_void_p)
_sig("ds_bundle_info", C.c_int, C.c_void_p, C.c_int, c_i64p)
_sig("ds_bundle_act", C.c_int, C.c_void_p, C.c_int64, c_i32p, c_i64p, c_i64p, c_i32p, c_i32p, c_i32p, c_i64p,
     c_dblp, c_i32p, c_i64p, c_dblp, c_i32p, c_i32p, c_i64p, c_i32p, c_i32p)
_sig("ds_bundle_strategy", C.c_int, C.c_void_p, C.c_int, C.c_int64, C.c_int, c_fltp, c_u8p, c_fltp)


class NativeError(RuntimeError):
    pass


def check(ok):
    if not ok:
        raise NativeError(lib.ds_last_error().decode("utf-8", "replace"))
    return ok


def ptr(arr, ctype):
    """numpy array -> ctypes pointer (array must be contiguous with the right dtype)."""
    return arr.ctypes.data_as(C.POINTER(ctype))


def version():
    return lib.ds_version().decode()
