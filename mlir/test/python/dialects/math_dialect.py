# RUN: %PYTHON %s | FileCheck %s

# Naming this file with a `_dialect` suffix to avoid a naming conflict with
# python package's math module (coming in from random.py).

from mlir.ir import *
import mlir.dialects.func as func
import mlir.dialects.math as mlir_math


def run(f):
    print("\nTEST:", f.__name__)
    f()


# CHECK-LABEL: TEST: testMathOps
@run
def testMathOps():
    with Context() as ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):

            @func.FuncOp.from_py_func(F32Type.get())
            def emit_sqrt(arg):
                return mlir_math.SqrtOp(arg)

        # CHECK-LABEL: func @emit_sqrt(
        # CHECK-SAME:                  %[[ARG:.*]]: f32) -> f32 {
        # CHECK:         math.sqrt %[[ARG]] : f32
        # CHECK:         return
        # CHECK:       }
        print(module)
