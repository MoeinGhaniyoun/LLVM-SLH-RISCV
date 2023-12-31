//===- GPUTransformOps.cpp - Implementation of GPU transform ops ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/TransformOps/GPUTransformOps.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/TransformOps/GPUTransformOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/DeviceMappingInterface.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::gpu;
using namespace mlir::transform;
using namespace mlir::transform::gpu;

#define DEBUG_TYPE "gpu-transforms"
#define DEBUG_TYPE_ALIAS "gpu-transforms-alias"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")
#define DBGS_ALIAS() (llvm::dbgs() << '[' << DEBUG_TYPE_ALIAS << "] ")

//===----------------------------------------------------------------------===//
// EliminateBarriersOp
//===----------------------------------------------------------------------===//

// The functions below provide interface-like verification, but are too specific
// to barrier elimination to become interfaces.

/// Implement the MemoryEffectsOpInterface in the suitable way.
static bool isKnownNoEffectsOpWithoutInterface(Operation *op) {
  // memref::AssumeAlignment is conceptually pure, but marking it as such would
  // make DCE immediately remove it.
  return isa<memref::AssumeAlignmentOp>(op);
}

/// Returns `true` if the op is defines the parallel region that is subject to
/// barrier synchronization.
static bool isParallelRegionBoundary(Operation *op) {
  if (op->hasAttr("__parallel_region_boundary_for_test"))
    return true;

  return isa<GPUFuncOp, LaunchOp>(op);
}

/// Returns `true` if the op behaves like a sequential loop, e.g., the control
/// flow "wraps around" from the end of the body region back to its start.
static bool isSequentialLoopLike(Operation *op) { return isa<scf::ForOp>(op); }

/// Returns `true` if the regions of the op are guaranteed to be executed at
/// most once. Thus, if an operation in one of the nested regions of `op` is
/// executed than so are all the other operations in this region.
static bool hasSingleExecutionBody(Operation *op) {
  return isa<scf::IfOp, memref::AllocaScopeOp>(op);
}

/// Returns `true` if the operation is known to produce a pointer-like object
/// distinct from any other object produced by a similar operation. For example,
/// an allocation produces such an object.
static bool producesDistinctBase(Operation *op) {
  return isa_and_nonnull<memref::AllocOp, memref::AllocaOp>(op);
}

/// Populates `effects` with all memory effects without associating them to a
/// specific value.
static void addAllValuelessEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Effect::get<MemoryEffects::Read>());
  effects.emplace_back(MemoryEffects::Effect::get<MemoryEffects::Write>());
  effects.emplace_back(MemoryEffects::Effect::get<MemoryEffects::Allocate>());
  effects.emplace_back(MemoryEffects::Effect::get<MemoryEffects::Free>());
}

/// Collect the memory effects of the given op in 'effects'. Returns 'true' if
/// it could extract the effect information from the op, otherwise returns
/// 'false' and conservatively populates the list with all possible effects
/// associated with no particular value or symbol.
static bool
collectEffects(Operation *op,
               SmallVectorImpl<MemoryEffects::EffectInstance> &effects,
               bool ignoreBarriers = true) {
  // Skip over barriers to avoid infinite recursion (those barriers would ask
  // this barrier again).
  if (ignoreBarriers && isa<BarrierOp>(op))
    return true;

  // Skip over ops that we know have no effects.
  if (isKnownNoEffectsOpWithoutInterface(op))
    return true;

  // Collect effect instances the operation. Note that the implementation of
  // getEffects erases all effect instances that have the type other than the
  // template parameter so we collect them first in a local buffer and then
  // copy.
  if (auto iface = dyn_cast<MemoryEffectOpInterface>(op)) {
    SmallVector<MemoryEffects::EffectInstance> localEffects;
    iface.getEffects(localEffects);
    llvm::append_range(effects, localEffects);
    return true;
  }
  if (op->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
    for (auto &region : op->getRegions()) {
      for (auto &block : region) {
        for (auto &innerOp : block)
          if (!collectEffects(&innerOp, effects, ignoreBarriers))
            return false;
      }
    }
    return true;
  }

  // We need to be conservative here in case the op doesn't have the interface
  // and assume it can have any possible effect.
  addAllValuelessEffects(effects);
  return false;
}

/// Collects memory effects from operations that may be executed before `op` in
/// a trivial structured control flow, e.g., without branches. Stops at the
/// parallel region boundary or at the barrier operation if `stopAtBarrier` is
/// set. Returns `true` if the memory effects added to `effects` are exact,
/// `false` if they are a conservative over-approximation. The latter means that
/// `effects` contain instances not associated with a specific value.
bool getEffectsBefore(Operation *op,
                      SmallVectorImpl<MemoryEffects::EffectInstance> &effects,
                      bool stopAtBarrier) {
  if (!op->getBlock())
    return true;

  // If there is a non-structured control flow, bail.
  Region *region = op->getBlock()->getParent();
  if (region && !llvm::hasSingleElement(region->getBlocks())) {
    addAllValuelessEffects(effects);
    return false;
  }

  // Collect all effects before the op.
  if (op != &op->getBlock()->front()) {
    for (Operation *it = op->getPrevNode(); it != nullptr;
         it = it->getPrevNode()) {
      if (isa<BarrierOp>(it)) {
        if (stopAtBarrier)
          return true;
        else
          continue;
      }
      if (!collectEffects(it, effects))
        return false;
    }
  }

  // Stop if reached the parallel region boundary.
  if (isParallelRegionBoundary(op->getParentOp()))
    return true;

  // Otherwise, keep collecting above the parent operation.
  if (!getEffectsBefore(op->getParentOp(), effects, stopAtBarrier))
    return false;

  // If the op is loop-like, collect effects from the trailing operations until
  // we hit a barrier because they can executed before the current operation by
  // the previous iteration of this loop. For example, in the following loop
  //
  //   for i = ... {
  //     op1
  //     ...
  //     barrier
  //     op2
  //   }
  //
  // the operation `op2` at iteration `i` is known to be executed before the
  // operation `op1` at iteration `i+1` and the side effects must be ordered
  // appropriately.
  if (isSequentialLoopLike(op->getParentOp())) {
    // Assuming loop terminators have no side effects.
    return getEffectsBefore(op->getBlock()->getTerminator(), effects,
                            /*stopAtBarrier=*/true);
  }

  // If the parent operation is not guaranteed to execute its (single-block)
  // region once, walk the block.
  bool conservative = false;
  if (!hasSingleExecutionBody(op->getParentOp()))
    op->getParentOp()->walk([&](Operation *in) {
      if (conservative)
        return WalkResult::interrupt();
      if (!collectEffects(in, effects)) {
        conservative = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

  return !conservative;
}

/// Collects memory effects from operations that may be executed after `op` in
/// a trivial structured control flow, e.g., without branches. Stops at the
/// parallel region boundary or at the barrier operation if `stopAtBarrier` is
/// set. Returns `true` if the memory effects added to `effects` are exact,
/// `false` if they are a conservative over-approximation. The latter means that
/// `effects` contain instances not associated with a specific value.
bool getEffectsAfter(Operation *op,
                     SmallVectorImpl<MemoryEffects::EffectInstance> &effects,
                     bool stopAtBarrier) {
  if (!op->getBlock())
    return true;

  // If there is a non-structured control flow, bail.
  Region *region = op->getBlock()->getParent();
  if (region && !llvm::hasSingleElement(region->getBlocks())) {
    addAllValuelessEffects(effects);
    return false;
  }

  // Collect all effects after the op.
  if (op != &op->getBlock()->back())
    for (Operation *it = op->getNextNode(); it != nullptr;
         it = it->getNextNode()) {
      if (isa<BarrierOp>(it)) {
        if (stopAtBarrier)
          return true;
        continue;
      }
      if (!collectEffects(it, effects))
        return false;
    }

  // Stop if reached the parallel region boundary.
  if (isParallelRegionBoundary(op->getParentOp()))
    return true;

  // Otherwise, keep collecting below the parent operation.
  if (!getEffectsAfter(op->getParentOp(), effects, stopAtBarrier))
    return false;

  // If the op is loop-like, collect effects from the leading operations until
  // we hit a barrier because they can executed after the current operation by
  // the next iteration of this loop. For example, in the following loop
  //
  //   for i = ... {
  //     op1
  //     ...
  //     barrier
  //     op2
  //   }
  //
  // the operation `op1` at iteration `i` is known to be executed after the
  // operation `op2` at iteration `i-1` and the side effects must be ordered
  // appropriately.
  if (isSequentialLoopLike(op->getParentOp())) {
    if (isa<BarrierOp>(op->getBlock()->front()))
      return true;

    bool exact = collectEffects(&op->getBlock()->front(), effects);
    return getEffectsAfter(&op->getBlock()->front(), effects,
                           /*stopAtBarrier=*/true) &&
           exact;
  }

  // If the parent operation is not guaranteed to execute its (single-block)
  // region once, walk the block.
  bool conservative = false;
  if (!hasSingleExecutionBody(op->getParentOp()))
    op->getParentOp()->walk([&](Operation *in) {
      if (conservative)
        return WalkResult::interrupt();
      if (!collectEffects(in, effects)) {
        conservative = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

  return !conservative;
}

/// Looks through known "view-like" ops to find the base memref.
static Value getBase(Value v) {
  while (true) {
    Operation *definingOp = v.getDefiningOp();
    if (!definingOp)
      break;

    bool shouldContinue =
        TypeSwitch<Operation *, bool>(v.getDefiningOp())
            .Case<memref::CastOp, memref::SubViewOp, memref::ViewOp>(
                [&](auto op) {
                  v = op.getSource();
                  return true;
                })
            .Case<memref::TransposeOp>([&](auto op) {
              v = op.getIn();
              return true;
            })
            .Case<memref::CollapseShapeOp, memref::ExpandShapeOp>([&](auto op) {
              v = op.getSrc();
              return true;
            })
            .Default([](Operation *) { return false; });
    if (!shouldContinue)
      break;
  }
  return v;
}

/// Returns `true` if the value is defined as a function argument.
static bool isFunctionArgument(Value v) {
  auto arg = dyn_cast<BlockArgument>(v);
  return arg && isa<FunctionOpInterface>(arg.getOwner()->getParentOp());
}

/// Returns the operand that the operation "propagates" through it for capture
/// purposes. That is, if the value produced by this operation is captured, then
/// so is the returned value.
static Value propagatesCapture(Operation *op) {
  return llvm::TypeSwitch<Operation *, Value>(op)
      .Case(
          [](ViewLikeOpInterface viewLike) { return viewLike.getViewSource(); })
      .Case([](CastOpInterface castLike) { return castLike->getOperand(0); })
      .Case([](memref::TransposeOp transpose) { return transpose.getIn(); })
      .Case<memref::ExpandShapeOp, memref::CollapseShapeOp>(
          [](auto op) { return op.getSrc(); })
      .Default([](Operation *) { return Value(); });
}

/// Returns `true` if the given operation is known to capture the given value,
/// `false` if it is known not to capture the given value, `nullopt` if neither
/// is known.
static std::optional<bool> getKnownCapturingStatus(Operation *op, Value v) {
  return llvm::TypeSwitch<Operation *, std::optional<bool>>(op)
      // Store-like operations don't capture the destination, but do capture
      // the value.
      .Case<memref::StoreOp, vector::TransferWriteOp>(
          [&](auto op) { return op.getValue() == v; })
      .Case<vector::StoreOp, vector::MaskedStoreOp>(
          [&](auto op) { return op.getValueToStore() == v; })
      // These operations are known not to capture.
      .Case([](memref::DeallocOp) { return false; })
      // By default, we don't know anything.
      .Default([](Operation *) { return std::nullopt; });
}

/// Returns `true` if the value may be captured by any of its users, i.e., if
/// the user may be storing this value into memory. This makes aliasing analysis
/// more conservative as it cannot assume the pointer-like value is only passed
/// around through SSA use-def.
bool maybeCaptured(Value v) {
  SmallVector<Value> todo = {v};
  while (!todo.empty()) {
    Value v = todo.pop_back_val();
    for (Operation *user : v.getUsers()) {
      // A user that is known to only read cannot capture.
      auto iface = dyn_cast<MemoryEffectOpInterface>(user);
      if (iface) {
        SmallVector<MemoryEffects::EffectInstance> effects;
        iface.getEffects(effects);
        if (llvm::all_of(effects,
                         [](const MemoryEffects::EffectInstance &effect) {
                           return isa<MemoryEffects::Read>(effect.getEffect());
                         })) {
          continue;
        }
      }

      // When an operation is known to create an alias, consider if the
      // source is captured as well.
      if (Value v = propagatesCapture(user)) {
        todo.push_back(v);
        continue;
      }

      std::optional<bool> knownCaptureStatus = getKnownCapturingStatus(user, v);
      if (!knownCaptureStatus || *knownCaptureStatus)
        return true;
    }
  }

  return false;
}

/// Returns true if two values may be referencing aliasing memory. This is a
/// rather naive and conservative analysis. Values defined by different
/// allocation-like operations as well as values derived from those by casts and
/// views cannot alias each other. Similarly, values defined by allocations
/// inside a function cannot alias function arguments. Global values cannot
/// alias each other or local allocations. Values that are captured, i.e.
/// themselves potentially stored in memory, are considered as aliasing with
/// everything. This seems sufficient to achieve barrier removal in structured
/// control flow, more complex cases would require a proper dataflow analysis.
static bool mayAlias(Value first, Value second) {
  DEBUG_WITH_TYPE(DEBUG_TYPE_ALIAS, {
    DBGS_ALIAS() << "checking aliasing between ";
    DBGS_ALIAS() << first << "\n";
    DBGS_ALIAS() << "                      and ";
    DBGS_ALIAS() << second << "\n";
  });

  first = getBase(first);
  second = getBase(second);

  DEBUG_WITH_TYPE(DEBUG_TYPE_ALIAS, {
    DBGS_ALIAS() << "base ";
    DBGS_ALIAS() << first << "\n";
    DBGS_ALIAS() << " and ";
    DBGS_ALIAS() << second << "\n";
  });

  // Values derived from the same base memref do alias (unless we do a more
  // advanced analysis to prove non-overlapping accesses).
  if (first == second) {
    DEBUG_WITH_TYPE(DEBUG_TYPE_ALIAS, DBGS_ALIAS() << "-> do alias!\n");
    return true;
  }

  // Different globals cannot alias.
  if (auto globFirst = first.getDefiningOp<memref::GetGlobalOp>()) {
    if (auto globSecond = second.getDefiningOp<memref::GetGlobalOp>()) {
      return globFirst.getNameAttr() == globSecond.getNameAttr();
    }
  }

  // Two function arguments marked as noalias do not alias.
  auto isNoaliasFuncArgument = [](Value value) {
    auto bbArg = dyn_cast<BlockArgument>(value);
    if (!bbArg)
      return false;
    auto iface = dyn_cast<FunctionOpInterface>(bbArg.getOwner()->getParentOp());
    if (!iface)
      return false;
    // TODO: we need a way to not depend on the LLVM dialect here.
    return iface.getArgAttr(bbArg.getArgNumber(), "llvm.noalias") != nullptr;
  };
  if (isNoaliasFuncArgument(first) && isNoaliasFuncArgument(second))
    return false;

  bool isDistinct[] = {producesDistinctBase(first.getDefiningOp()),
                       producesDistinctBase(second.getDefiningOp())};
  bool isGlobal[] = {first.getDefiningOp<memref::GetGlobalOp>() != nullptr,
                     second.getDefiningOp<memref::GetGlobalOp>() != nullptr};

  // Non-equivalent distinct bases and globals cannot alias. At this point, we
  // have already filtered out based on values being equal and global name being
  // equal.
  if ((isDistinct[0] || isGlobal[0]) && (isDistinct[1] || isGlobal[1]))
    return false;

  bool isArg[] = {isFunctionArgument(first), isFunctionArgument(second)};

  // Distinct bases (allocations) cannot have been passed as an argument.
  if ((isDistinct[0] && isArg[1]) || (isDistinct[1] && isArg[0]))
    return false;

  // Non-captured base distinct values cannot conflict with another base value.
  if (isDistinct[0] && !maybeCaptured(first))
    return false;
  if (isDistinct[1] && !maybeCaptured(second))
    return false;

  // Otherwise, conservatively assume aliasing.
  DEBUG_WITH_TYPE(DEBUG_TYPE_ALIAS, DBGS_ALIAS() << "-> may alias!\n");
  return true;
}

/// Returns `true` if the effect may be affecting memory aliasing the value. If
/// the effect is not associated with any value, it is assumed to affect all
/// memory and therefore aliases with everything.
bool mayAlias(MemoryEffects::EffectInstance a, Value v2) {
  if (Value v = a.getValue()) {
    return mayAlias(v, v2);
  }
  return true;
}

/// Returns `true` if the two effects may be affecting aliasing memory. If
/// an effect is not associated with any value, it is assumed to affect all
/// memory and therefore aliases with everything. Effects on different resources
/// cannot alias.
bool mayAlias(MemoryEffects::EffectInstance a,
              MemoryEffects::EffectInstance b) {
  if (a.getResource()->getResourceID() != b.getResource()->getResourceID())
    return false;
  if (Value v2 = b.getValue()) {
    return mayAlias(a, v2);
  } else if (Value v = a.getValue()) {
    return mayAlias(b, v);
  }
  return true;
}

/// Returns `true` if any of the "before" effect instances has a conflict with
/// any "after" instance for the purpose of barrier elimination. The effects are
/// supposed to be limited to a barrier synchronization scope. A conflict exists
/// if effects instances affect aliasing memory locations and at least on of
/// then as a write. As an exception, if the non-write effect is an allocation
/// effect, there is no conflict since we are only expected to see the
/// allocation happening in the same thread and it cannot be accessed from
/// another thread without capture (which we do handle in alias analysis).
static bool
haveConflictingEffects(ArrayRef<MemoryEffects::EffectInstance> beforeEffects,
                       ArrayRef<MemoryEffects::EffectInstance> afterEffects) {
  for (const MemoryEffects::EffectInstance &before : beforeEffects) {
    for (const MemoryEffects::EffectInstance &after : afterEffects) {
      // If cannot alias, definitely no conflict.
      if (!mayAlias(before, after))
        continue;

      // Read/read is not a conflict.
      if (isa<MemoryEffects::Read>(before.getEffect()) &&
          isa<MemoryEffects::Read>(after.getEffect())) {
        continue;
      }

      // Allocate/* is not a conflict since the allocation happens within the
      // thread context.
      // TODO: This is not the case for */Free unless the allocation happened in
      // the thread context, which we could also check for.
      if (isa<MemoryEffects::Allocate>(before.getEffect()) ||
          isa<MemoryEffects::Allocate>(after.getEffect())) {
        continue;
      }

      // In the particular case that the before effect is a free, we only have 2
      // possibilities:
      //   1. either the program is well-formed and there must be an interleaved
      //      alloc that must limit the scope of effect lookback and we can
      //      safely ignore the free -> read / free -> write and free -> free
      //      conflicts.
      //   2. either the program is ill-formed and we are in undefined behavior
      //      territory.
      if (isa<MemoryEffects::Free>(before.getEffect()))
        continue;

      // Other kinds of effects create a conflict, e.g. read-after-write.
      LLVM_DEBUG(
          DBGS() << "found a conflict between (before): " << before.getValue()
                 << " read:" << isa<MemoryEffects::Read>(before.getEffect())
                 << " write:" << isa<MemoryEffects::Write>(before.getEffect())
                 << " alloc:"
                 << isa<MemoryEffects::Allocate>(before.getEffect()) << " free:"
                 << isa<MemoryEffects::Free>(before.getEffect()) << "\n");
      LLVM_DEBUG(
          DBGS() << "and (after):                " << after.getValue()
                 << " read:" << isa<MemoryEffects::Read>(after.getEffect())
                 << " write:" << isa<MemoryEffects::Write>(after.getEffect())
                 << " alloc:" << isa<MemoryEffects::Allocate>(after.getEffect())
                 << " free:" << isa<MemoryEffects::Free>(after.getEffect())
                 << "\n");
      return true;
    }
  }

  return false;
}

namespace {
/// Barrier elimination pattern. If a barrier does not enforce any conflicting
/// pair of memory effects, including a pair that is enforced by another
/// barrier, it is unnecessary and can be removed. Adapted from
/// "High-Performance GPU-to-CPU Transpilation and Optimization via High-Level
/// Parallel Constructs" by Moses, Ivanov, Domke, Endo, Doerfert, and Zinenko in
/// PPoPP 2023 and implementation in Polygeist.
class BarrierElimination final : public OpRewritePattern<BarrierOp> {
public:
  using OpRewritePattern<BarrierOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BarrierOp barrier,
                                PatternRewriter &rewriter) const override {
    LLVM_DEBUG(DBGS() << "checking the necessity of: " << barrier << " "
                      << barrier.getLoc() << "\n");

    SmallVector<MemoryEffects::EffectInstance> beforeEffects;
    getEffectsBefore(barrier, beforeEffects, /*stopAtBarrier=*/true);

    SmallVector<MemoryEffects::EffectInstance> afterEffects;
    getEffectsAfter(barrier, afterEffects, /*stopAtBarrier=*/true);

    if (!haveConflictingEffects(beforeEffects, afterEffects)) {
      LLVM_DEBUG(DBGS() << "the surrounding barriers are sufficient, removing "
                        << barrier << "\n");
      rewriter.eraseOp(barrier);
      return success();
    }

    LLVM_DEBUG(DBGS() << "barrier is necessary: " << barrier << " "
                      << barrier.getLoc() << "\n");
    return failure();
  }
};
} // namespace

void EliminateBarriersOp::populatePatterns(RewritePatternSet &patterns) {
  patterns.insert<BarrierElimination>(getContext());
}

//===----------------------------------------------------------------------===//
// Block and thread mapping utilities.
//===----------------------------------------------------------------------===//

namespace {

/// Return a flattened thread id for the workgroup with given sizes.
static Value buildLinearThreadId(RewriterBase &rewriter, Location loc,
                                 ArrayRef<OpFoldResult> blockDimsOfr) {
  LLVM_DEBUG(llvm::interleaveComma(
                 blockDimsOfr,
                 DBGS() << "----buildLinearThreadId with blockDimsOfr:  ");
             llvm::dbgs() << "\n");
  assert(blockDimsOfr.size() == 3 && "expected 3 workgroup sizes");
  AffineExpr tx, ty, tz, BDX, BDY;
  bindDims(rewriter.getContext(), tx, ty, tz);
  bindSymbols(rewriter.getContext(), BDX, BDY);
  IndexType indexType = rewriter.getIndexType();
  SmallVector<OpFoldResult> threadsAndWorkGroups{
      rewriter.create<ThreadIdOp>(loc, indexType, Dimension::x).getResult(),
      rewriter.create<ThreadIdOp>(loc, indexType, Dimension::y).getResult(),
      rewriter.create<ThreadIdOp>(loc, indexType, Dimension::z).getResult()};
  threadsAndWorkGroups.push_back(blockDimsOfr[0]);
  threadsAndWorkGroups.push_back(blockDimsOfr[1]);
  OpFoldResult ofr = affine::makeComposedFoldedAffineApply(
      rewriter, loc, tx + ty * BDX + tz * BDX * BDY, threadsAndWorkGroups);
  return getValueOrCreateConstantIndexOp(rewriter, loc, ofr);
}

/// Builder for gpu::BlockIdOps used in mapping scf.forall to blocks.
/// The `idBuilder` method returns 3-D values used for indexing rewrites as well
/// as 3-D sizes for predicate generation.
struct GpuBlockIdBuilder : public GpuIdBuilder {

  GpuBlockIdBuilder(MLIRContext *ctx, ArrayRef<OpFoldResult> blockDims,
                    ArrayRef<int64_t> mappingSizes)
      : GpuIdBuilder(blockDims, mappingSizes) {
    mappingAttributes = {GPUBlockMappingAttr::get(ctx, Blocks::DimX),
                         GPUBlockMappingAttr::get(ctx, Blocks::DimY),
                         GPUBlockMappingAttr::get(ctx, Blocks::DimZ)},
    idBuilder = [](RewriterBase &rewriter, Location loc,
                   ArrayRef<int64_t> forallMappingSizes) {
      IndexType indexType = rewriter.getIndexType();
      SmallVector<Value> ids{
          rewriter.create<BlockIdOp>(loc, indexType, Dimension::x),
          rewriter.create<BlockIdOp>(loc, indexType, Dimension::y),
          rewriter.create<BlockIdOp>(loc, indexType, Dimension::z)};
      // Return 3-D ids for indexing rewrite and 3-D sizes and ids for
      // predicate generation.
      return IdBuilderResult{ids, SmallVector<int64_t>{forallMappingSizes},
                             ids};
    };
  }
};

/// Builder for gpu::ThreadIdOp used in mapping scf.forall to thread ids without
/// any reindexing.
/// The `idBuilder` method returns 3-D values used for indexing rewrites as well
/// as 3-D sizes for predicate generation.
struct GpuThreadIdBuilder : public GpuIdBuilder {
  GpuThreadIdBuilder(MLIRContext *ctx, ArrayRef<OpFoldResult> blockDims,
                     ArrayRef<int64_t> mappingSizes)
      : GpuIdBuilder(blockDims, mappingSizes) {
    mappingAttributes = {GPUThreadMappingAttr::get(ctx, Threads::DimX),
                         GPUThreadMappingAttr::get(ctx, Threads::DimY),
                         GPUThreadMappingAttr::get(ctx, Threads::DimZ)};
    idBuilder = [](RewriterBase &rewriter, Location loc,
                   ArrayRef<int64_t> forallMappingSizes) {
      IndexType indexType = rewriter.getIndexType();
      SmallVector<Value> ids{
          rewriter.create<ThreadIdOp>(loc, indexType, Dimension::x),
          rewriter.create<ThreadIdOp>(loc, indexType, Dimension::y),
          rewriter.create<ThreadIdOp>(loc, indexType, Dimension::z)};
      // Return 3-D ids for indexing rewrite and 3-D sizes and ids for
      // predicate generation.
      return IdBuilderResult{ids, SmallVector<int64_t>{forallMappingSizes},
                             ids};
    };
  }
};

/// Builder for warp ids used in mapping scf.forall to warps.
/// This builder requires a specification of the number of warps along each
/// dimension to more finely control mapping to warps as well a predication than
/// by solely analyzing the IR.
/// The `idBuilder` method returns 3-D values used for indexing rewrites as well
/// as 3-D sizes for predicate generation.
struct GpuWarpIdBuilder : public GpuIdBuilder {
  GpuWarpIdBuilder(MLIRContext *ctx, ArrayRef<OpFoldResult> blockDims,
                   ArrayRef<int64_t> mappingSizes)
      : GpuIdBuilder(blockDims, mappingSizes) {
    mappingAttributes = {GPUWarpMappingAttr::get(ctx, Warps::DimX),
                         GPUWarpMappingAttr::get(ctx, Warps::DimY),
                         GPUWarpMappingAttr::get(ctx, Warps::DimZ)};
    idBuilder = [this](RewriterBase &rewriter, Location loc,
                       ArrayRef<int64_t> forallMappingSizes) {
      // Build the linear warp id and decompose it in the basis of
      // `forallMappingSizes`.
      Value linearId = buildLinearThreadId(rewriter, loc, this->blockDimsOfr);
      AffineExpr d0 = getAffineDimExpr(0, rewriter.getContext());
      OpFoldResult warpIdOfr = affine::makeComposedFoldedAffineApply(
          rewriter, loc, d0.floorDiv(kWarpSize), {linearId});
      Value warpId = getValueOrCreateConstantIndexOp(rewriter, loc, warpIdOfr);
      // Sizes in [x, y, z] -> [z, y x] order to properly compute strides in
      // "row-major" order.
      SmallVector<int64_t> reverseBasisSizes(
          llvm::reverse(this->availableMappingSizes));
      SmallVector<int64_t> strides = computeStrides(reverseBasisSizes);
      SmallVector<AffineExpr> delinearizingExprs = delinearize(d0, strides);
      SmallVector<Value> ids;
      // Reverse back to be in [x, y, z] order.
      for (AffineExpr e : llvm::reverse(delinearizingExprs))
        ids.push_back(
            affine::makeComposedAffineApply(rewriter, loc, e, {warpId}));

      // clang-format off
      LDBG("----linearId: " << linearId);
          LDBG("----warpId: " << warpId);
      LLVM_DEBUG(llvm::interleaveComma(reverseBasisSizes,
                                       DBGS() << "--delinearization basis: ");
                 llvm::dbgs() << "\n";
                 llvm::interleaveComma(strides,
                                       DBGS() << "--delinearization strides: ");
                 llvm::dbgs() << "\n";
                 llvm::interleaveComma(delinearizingExprs,
                                       DBGS() << "--delinearization exprs: ");
                 llvm::dbgs() << "\n";
                 llvm::interleaveComma(ids, DBGS() << "--ids: ");
                 llvm::dbgs() << "\n";);
      // clang-format on

      // Return 3-D ids for indexing rewrite and 3-D sizes and ids for
      // predicate generation.
      return IdBuilderResult{ids, SmallVector<int64_t>{forallMappingSizes},
                             ids};
    };
  }

  /// Static specification of the warp size.
  /// In the future this may be configured by the transformation.
  static constexpr int64_t kWarpSize = 32;
};

/// Builder for linear ids used in mapping scf.forall to reindexed threads.
/// The `idBuilder` method returns 3-D values used for indexing rewrites as well
/// as 1-D sizes for predicate generation.
struct GpuLinearIdBuilder : public GpuIdBuilder {
  GpuLinearIdBuilder(MLIRContext *ctx, ArrayRef<OpFoldResult> blockDims,
                     ArrayRef<int64_t> mappingSizes)
      : GpuIdBuilder(blockDims, mappingSizes) {
    mappingAttributes = {GPULinearIdMappingAttr::get(ctx, LinearId::DimX),
                         GPULinearIdMappingAttr::get(ctx, LinearId::DimY),
                         GPULinearIdMappingAttr::get(ctx, LinearId::DimZ)};
    idBuilder = [this](RewriterBase &rewriter, Location loc,
                       ArrayRef<int64_t> forallMappingSizes) {
      // Build the linear thread id and decompose it in the basis of
      // `forallMappingSizes`.
      Value linearId = buildLinearThreadId(rewriter, loc, this->blockDimsOfr);
      // Sizes in [x, y, z] -> [z, y x] order to properly compute strides in
      // "row-major" order.
      SmallVector<int64_t> reverseBasisSizes(llvm::reverse(forallMappingSizes));
      SmallVector<int64_t> strides = computeStrides(reverseBasisSizes);
      AffineExpr d0;
      bindDims(rewriter.getContext(), d0);
      SmallVector<AffineExpr> delinearizingExprs = delinearize(d0, strides);
      SmallVector<Value> ids;
      // Reverse back to be in [x, y, z] order.
      for (AffineExpr e : llvm::reverse(delinearizingExprs))
        ids.push_back(
            affine::makeComposedAffineApply(rewriter, loc, e, {linearId}));

      // clang-format off
      LLVM_DEBUG(llvm::interleaveComma(reverseBasisSizes,
                                       DBGS() << "--delinearization basis: ");
                 llvm::dbgs() << "\n";
                 llvm::interleaveComma(strides,
                                       DBGS() << "--delinearization strides: ");
                 llvm::dbgs() << "\n";
                 llvm::interleaveComma(delinearizingExprs,
                                       DBGS() << "--delinearization exprs: ");
                 llvm::dbgs() << "\n";
                 llvm::interleaveComma(ids, DBGS() << "--ids: ");
                 llvm::dbgs() << "\n";);
      // clang-format on

      // Compute and return the 1-D actual mapping size spanned by the linearId,
      // it will be used to predicate against the linearized total number of
      // threads.
      int64_t actualMappingSize = 1;
      for (int64_t s : forallMappingSizes)
        actualMappingSize *= s;

      // Return 3-D ids for indexing rewrite and 1-D size and id for
      // predicate generation.
      return IdBuilderResult{ids, SmallVector<int64_t>{actualMappingSize},
                             SmallVector<Value>{linearId}};
    };
  }
};

} // namespace

static DiagnosedSilenceableFailure
definiteFailureHelper(std::optional<TransformOpInterface> transformOp,
                      Operation *target, const Twine &message) {
  if (transformOp.has_value())
    return transformOp->emitDefiniteFailure() << message;
  return emitDefiniteFailure(target, message);
}

/// Check if given mapping attributes are one of the desired attributes
static DiagnosedSilenceableFailure
checkMappingAttributeTypes(std::optional<TransformOpInterface> transformOp,
                           scf::ForallOp forallOp) {
  if (!forallOp.getMapping().has_value())
    return definiteFailureHelper(transformOp, forallOp,
                                 "mapping must be present");

  bool hasBlockMapping =
      llvm::any_of(forallOp.getMapping().value(), [](Attribute attr) {
        return isa<GPUBlockMappingAttr>(attr);
      });
  bool hasThreadMapping =
      llvm::any_of(forallOp.getMapping().value(), [](Attribute attr) {
        return isa<GPUThreadMappingAttr>(attr);
      });
  bool hasWarpMapping =
      llvm::any_of(forallOp.getMapping().value(), [](Attribute attr) {
        return isa<GPUWarpMappingAttr>(attr);
      });
  bool hasLinearMapping =
      llvm::any_of(forallOp.getMapping().value(), [](Attribute attr) {
        return isa<GPULinearIdMappingAttr>(attr);
      });
  int64_t countMappingTypes = 0;
  countMappingTypes += hasBlockMapping ? 1 : 0;
  countMappingTypes += hasThreadMapping ? 1 : 0;
  countMappingTypes += hasWarpMapping ? 1 : 0;
  countMappingTypes += hasLinearMapping ? 1 : 0;
  if (countMappingTypes > 1) {
    return definiteFailureHelper(
        transformOp, forallOp,
        "cannot mix different mapping types, use nesting");
  }

  DenseSet<Attribute> seen;
  for (Attribute map : forallOp.getMapping()->getValue()) {
    if (seen.contains(map)) {
      return definiteFailureHelper(
          transformOp, forallOp,
          "duplicated attribute, cannot map different loops "
          "to the same processor");
    }
    seen.insert(map);
  }

  return DiagnosedSilenceableFailure::success();
}

static DiagnosedSilenceableFailure
verifyGpuMapping(std::optional<TransformOpInterface> transformOp,
                 scf::ForallOp forallOp) {
  // Check the types of the mapping attributes match.
  DiagnosedSilenceableFailure typeRes =
      checkMappingAttributeTypes(transformOp, forallOp);
  if (!typeRes.succeeded())
    return typeRes;

  // Perform other non-types verifications.
  if (!forallOp.isNormalized())
    return definiteFailureHelper(transformOp, forallOp,
                                 "unsupported non-normalized loops");
  if (forallOp.getNumResults() > 0)
    return definiteFailureHelper(transformOp, forallOp,
                                 "only bufferized scf.forall can be mapped");
  if (forallOp.getRank() > 3)
    return definiteFailureHelper(transformOp, forallOp,
                                 "scf.forall with rank > 3 does not lower");
  if (llvm::any_of(forallOp.getMixedUpperBound(), [&](OpFoldResult ofr) {
        return !getConstantIntValue(ofr).has_value();
      })) {
    return definiteFailureHelper(transformOp, forallOp,
                                 "unsupported dynamic sizes in forall op");
  }
  return DiagnosedSilenceableFailure::success();
}

/// Determines if the size of the kernel configuration is supported by the
/// GPU architecture being used. It presently makes use of CUDA limitations,
/// however that aspect may be enhanced for other GPUs.
static DiagnosedSilenceableFailure checkGpuLimits(
    TransformOpInterface transformOp, std::optional<int64_t> gridDimX,
    std::optional<int64_t> gridDimY, std::optional<int64_t> gridDimZ,
    std::optional<int64_t> blockDimX, std::optional<int64_t> blockDimY,
    std::optional<int64_t> blockDimZ) {

  static constexpr int maxTotalBlockdim = 1024;
  static constexpr int maxBlockdimx = 1024;
  static constexpr int maxBlockdimy = 1024;
  static constexpr int maxBlockdimz = 64;
  static constexpr int maxTotalGriddim = 2147483647;
  static constexpr int maxGriddimx = 2147483647;
  static constexpr int maxGriddimy = 65535;
  static constexpr int maxGriddimz = 65535;

  if ((blockDimX.value_or(1) * blockDimY.value_or(1) * blockDimZ.value_or(1)) >
          maxTotalBlockdim ||
      (gridDimX.value_or(1) * gridDimY.value_or(1) * gridDimZ.value_or(1)) >
          maxTotalGriddim ||
      blockDimX.value_or(1) > maxBlockdimx ||
      blockDimY.value_or(1) > maxBlockdimy ||
      blockDimZ.value_or(1) > maxBlockdimz ||
      gridDimY.value_or(1) > maxGriddimy ||
      gridDimZ.value_or(1) > maxGriddimz ||
      gridDimX.value_or(1) > maxGriddimx) {
    return transformOp.emitSilenceableError()
           << "Trying to launch a GPU kernel with grid_dims = ("
           << gridDimX.value_or(1) << ", " << gridDimY.value_or(1) << ", "
           << gridDimZ.value_or(1) << ") block_dims = ("
           << blockDimX.value_or(1) << ", " << blockDimY.value_or(1) << ", "
           << blockDimZ.value_or(1) << "). It is larger than the limits.";
  }
  return DiagnosedSilenceableFailure::success();
}

/// Creates an empty-body gpu::LaunchOp using the provided kernel settings
/// and put a terminator within.
static DiagnosedSilenceableFailure
createGpuLaunch(RewriterBase &rewriter, Location loc,
                TransformOpInterface transformOp, LaunchOp &launchOp,
                std::optional<int64_t> gridDimX = std::nullopt,
                std::optional<int64_t> gridDimY = std::nullopt,
                std::optional<int64_t> gridDimZ = std::nullopt,
                std::optional<int64_t> blockDimX = std::nullopt,
                std::optional<int64_t> blockDimY = std::nullopt,
                std::optional<int64_t> blockDimZ = std::nullopt) {
  DiagnosedSilenceableFailure diag =
      checkGpuLimits(transformOp, gridDimX, gridDimY, gridDimZ, blockDimX,
                     blockDimY, blockDimZ);
  if (!diag.succeeded())
    return diag;

  auto createConst = [&](int dim) {
    return rewriter.create<arith::ConstantIndexOp>(loc, dim);
  };
  OpBuilder::InsertionGuard guard(rewriter);
  Value one = createConst(1);
  Value gridSizeX = gridDimX.has_value() ? createConst(gridDimX.value()) : one;
  Value gridSizeY = gridDimY.has_value() ? createConst(gridDimY.value()) : one;
  Value gridSizeZ = gridDimZ.has_value() ? createConst(gridDimZ.value()) : one;
  Value blkSizeX = blockDimX.has_value() ? createConst(blockDimX.value()) : one;
  Value blkSizeY = blockDimY.has_value() ? createConst(blockDimY.value()) : one;
  Value blkSizeZ = blockDimZ.has_value() ? createConst(blockDimZ.value()) : one;
  launchOp = rewriter.create<LaunchOp>(loc, gridSizeX, gridSizeY, gridSizeZ,
                                       blkSizeX, blkSizeY, blkSizeZ);
  rewriter.setInsertionPointToEnd(&launchOp.getBody().front());
  rewriter.create<TerminatorOp>(loc);
  return DiagnosedSilenceableFailure::success();
}

/// Alter kernel configuration of the given kernel.
static DiagnosedSilenceableFailure
alterGpuLaunch(RewriterBase &rewriter, LaunchOp gpuLaunch,
               TransformOpInterface transformOp,
               std::optional<int64_t> gridDimX = std::nullopt,
               std::optional<int64_t> gridDimY = std::nullopt,
               std::optional<int64_t> gridDimZ = std::nullopt,
               std::optional<int64_t> blockDimX = std::nullopt,
               std::optional<int64_t> blockDimY = std::nullopt,
               std::optional<int64_t> blockDimZ = std::nullopt) {
  DiagnosedSilenceableFailure diag =
      checkGpuLimits(transformOp, gridDimX, gridDimY, gridDimZ, blockDimX,
                     blockDimY, blockDimZ);
  if (!diag.succeeded())
    return diag;

  KernelDim3 currentBlockdim = gpuLaunch.getBlockSizeOperandValues();
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfterValue(currentBlockdim.x);
  auto createConstValue = [&](int dim) {
    return rewriter.create<arith::ConstantIndexOp>(currentBlockdim.x.getLoc(),
                                                   dim);
  };

  if (gridDimX.has_value())
    gpuLaunch.getGridSizeXMutable().assign(createConstValue(gridDimX.value()));
  if (gridDimY.has_value())
    gpuLaunch.getGridSizeYMutable().assign(createConstValue(gridDimY.value()));
  if (gridDimZ.has_value())
    gpuLaunch.getGridSizeZMutable().assign(createConstValue(gridDimZ.value()));
  if (blockDimX.has_value())
    gpuLaunch.getBlockSizeXMutable().assign(
        createConstValue(blockDimX.value()));
  if (blockDimY.has_value())
    gpuLaunch.getBlockSizeYMutable().assign(
        createConstValue(blockDimY.value()));
  if (blockDimZ.has_value())
    gpuLaunch.getBlockSizeZMutable().assign(
        createConstValue(blockDimZ.value()));
  return DiagnosedSilenceableFailure::success();
}

/// Struct to return the result of the rewrite of a forall operation.
struct ForallRewriteResult {
  SmallVector<int64_t> mappingSizes;
  SmallVector<Value> mappingIds;
};

/// Helper to replace ids of dimensions known to be 1 by 0 to simplify the IR.
template <typename OpTy, typename OperationOrBlock>
static void
replaceUnitMappingIdsHelper(RewriterBase &rewriter, Location loc,
                            OperationOrBlock *parent, Value replacement,
                            ArrayRef<int64_t> availableMappingSizes) {
  parent->walk([&](OpTy idOp) {
    if (availableMappingSizes[static_cast<int64_t>(idOp.getDimension())] == 1)
      rewriter.replaceAllUsesWith(idOp.getResult(), replacement);
  });
}

static DiagnosedSilenceableFailure rewriteOneForallCommonImpl(
    RewriterBase &rewriter, std::optional<TransformOpInterface> transformOp,
    scf::ForallOp forallOp, ForallRewriteResult &result,
    ArrayRef<int64_t> availableMappingSizes, const GpuIdBuilder &gpuIdBuilder) {
  LDBG("--start rewriteOneForallCommonImpl");

  // Step 0. GPU-specific verifications. There is no better place to anchor
  // those right now: the ForallOp is target-independent and the transform
  // op does not apply to individual ForallOp.
  DiagnosedSilenceableFailure diag = verifyGpuMapping(transformOp, forallOp);
  if (!diag.succeeded())
    return diag;

  // Step 1. Complete the mapping to a full mapping (with 1s) if necessary.
  SmallVector<int64_t> tmpMappingSizes = llvm::to_vector(
      llvm::map_range(forallOp.getMixedUpperBound(), [](OpFoldResult ofr) {
        auto maybeStaticValue = getConstantIntValue(ofr);
        assert(maybeStaticValue && "expected static value");
        return maybeStaticValue.value();
      }));
  SmallVector<Attribute> forallMappingAttrs =
      llvm::to_vector(forallOp.getMapping()->getValue());
  for (auto attr : gpuIdBuilder.mappingAttributes) {
    if (llvm::is_contained(forallMappingAttrs, attr))
      continue;
    forallMappingAttrs.push_back(attr);
    tmpMappingSizes.push_back(1);
  }
  LLVM_DEBUG(
      llvm::interleaveComma(
          tmpMappingSizes,
          DBGS() << "----tmpMappingSizes extracted from scf.forall op: ");
      llvm::dbgs() << "\n");

  // Step 2. sort the values by the corresponding DeviceMappingAttrInterface.
  auto comparator = [&](Attribute a, Attribute b) -> bool {
    return cast<DeviceMappingAttrInterface>(a).getMappingId() <
           cast<DeviceMappingAttrInterface>(b).getMappingId();
  };
  SmallVector<int64_t> forallMappingSizes =
      getValuesSortedByKey(forallMappingAttrs, tmpMappingSizes, comparator);
  LLVM_DEBUG(llvm::interleaveComma(forallMappingSizes,
                                   DBGS() << "----forallMappingSizes: ");
             llvm::dbgs() << "\n"; llvm::interleaveComma(
                 forallMappingAttrs, DBGS() << "----mappingAttrs: ");
             llvm::dbgs() << "\n");

  // Step 3. Generate the mappingIdOps using the provided generator.
  Location loc = forallOp.getLoc();
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(forallOp);
  IdBuilderResult builderResult =
      gpuIdBuilder.idBuilder(rewriter, loc, forallMappingSizes);

  // Step 4. Map the induction variables to the mappingIdOps, this may involve a
  // permutation.
  SmallVector<Value> mappingIdOps = builderResult.mappingIdOps;
  IRMapping bvm;
  for (auto [iv, dim] :
       llvm::zip_equal(forallOp.getInductionVars(),
                       ArrayRef<Attribute>{forallMappingAttrs}.take_front(
                           forallOp.getInductionVars().size()))) {
    Value peIdOp = mappingIdOps[static_cast<int64_t>(
        cast<DeviceMappingAttrInterface>(dim).getMappingId())];
    bvm.map(iv, peIdOp);
  }

  // Step 5. If the availableMappingSizes are already known, create conditionals
  // to predicate the region. Otherwise, the current forall determines the
  // availableMappingSizes and no predication occurs.
  Value predicate;
  if (!availableMappingSizes.empty()) {
    SmallVector<int64_t> predicateMappingSizes =
        builderResult.predicateMappingSizes;
    SmallVector<Value> predicateIdOps = builderResult.predicateIdOps;
    // clang-format off
    LLVM_DEBUG(
        llvm::interleaveComma(
          predicateMappingSizes, DBGS() << "----predicateMappingSizes: ");
        llvm::dbgs() << "\n"; 
        llvm::interleaveComma(
          availableMappingSizes, DBGS() << "----availableMappingSizes: ");
        llvm::dbgs() << "\n";
        llvm::interleaveComma(predicateIdOps, DBGS() << "----predicateIdOps: ");
        llvm::dbgs() << "\n");
    // clang-format on
    for (auto [id, mappingSize, availableMappingSize] : llvm::zip_equal(
             predicateIdOps, predicateMappingSizes, availableMappingSizes)) {
      if (mappingSize > availableMappingSize) {
        return definiteFailureHelper(
            transformOp, forallOp,
            "Trying to map to fewer GPU threads than loop iterations but "
            "overprovisioning is not yet supported. "
            "Try additional tiling of the before mapping or map to more "
            "threads.");
      }
      if (mappingSize == availableMappingSize)
        continue;
      Value idx = rewriter.create<arith::ConstantIndexOp>(loc, mappingSize);
      Value tmpPredicate = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ult, id, idx);
      LDBG("----predicate: " << tmpPredicate);
      predicate = predicate ? rewriter.create<arith::AndIOp>(loc, predicate,
                                                             tmpPredicate)
                            : tmpPredicate;
    }
  }

  // Step 6. Move the body of forallOp.
  // Erase the terminator first, it will not be used.
  rewriter.eraseOp(forallOp.getTerminator());
  Block *targetBlock;
  Block::iterator insertionPoint;
  if (predicate) {
    // Step 6.a. If predicated, move at the beginning.
    auto ifOp = rewriter.create<scf::IfOp>(loc, predicate,
                                           /*withElseRegion=*/false);
    targetBlock = ifOp.thenBlock();
    insertionPoint = ifOp.thenBlock()->begin();
  } else {
    // Step 6.b. Otherwise, move inline just at the rewriter insertion
    // point.
    targetBlock = forallOp->getBlock();
    insertionPoint = rewriter.getInsertionPoint();
  }
  Block &sourceBlock = forallOp.getRegion().front();
  targetBlock->getOperations().splice(insertionPoint,
                                      sourceBlock.getOperations());

  // Step 7. RAUW indices.
  for (Value loopIndex : forallOp.getInductionVars()) {
    Value threadIdx = bvm.lookup(loopIndex);
    rewriter.replaceAllUsesWith(loopIndex, threadIdx);
  }

  // Step 8. Erase old op.
  rewriter.eraseOp(forallOp);

  result = ForallRewriteResult{forallMappingSizes, mappingIdOps};
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// MapForallToBlocks
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure mlir::transform::gpu::mapForallToBlocksImpl(
    RewriterBase &rewriter, TransformOpInterface transformOp,
    scf::ForallOp forallOp, SmallVectorImpl<int64_t> &gridDims,
    const GpuIdBuilder &gpuIdBuilder) {
  LDBG("Start mapForallToBlocksImpl");

  Location loc = forallOp.getLoc();
  Block *parentBlock = forallOp->getBlock();
  Value zero;
  {
    // Create an early zero index value for replacements and immediately reset
    // the insertion point.
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(parentBlock);
    zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  }

  SmallVector<int64_t> anyAvailableMappingSizes;
  ForallRewriteResult rewriteResult;
  // Pass an empty anyAvailableMappingSizes.
  DiagnosedSilenceableFailure diag =
      rewriteOneForallCommonImpl(rewriter, transformOp, forallOp, rewriteResult,
                                 anyAvailableMappingSizes, gpuIdBuilder);

  // Return if anything goes wrong, use silenceable failure as a match failure.
  if (!diag.succeeded())
    return diag;

  // Set the gridDims that act as a return.
  gridDims = rewriteResult.mappingSizes;

  // Replace ids of dimensions known to be 1 by 0 to simplify the IR.
  // Here, the result of mapping determines the available mapping sizes.
  replaceUnitMappingIdsHelper<BlockDimOp>(rewriter, loc, parentBlock, zero,
                                          gridDims);

  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure
mlir::transform::gpu::findTopLevelForallOp(Operation *target,
                                           scf::ForallOp &topLevelForallOp,
                                           TransformOpInterface transformOp) {
  auto walkResult = target->walk([&](scf::ForallOp forallOp) {
    if (forallOp->getParentOfType<scf::ForallOp>())
      return WalkResult::advance();
    if (topLevelForallOp)
      // TODO: Handle multiple forall if they are independent.
      return WalkResult::interrupt();
    topLevelForallOp = forallOp;
    return WalkResult::advance();
  });

  if (walkResult.wasInterrupted())
    return transformOp.emitSilenceableError()
           << "could not find a unique topLevel scf.forall";
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure transform::MapForallToBlocks::applyToOne(
    transform::TransformRewriter &rewriter, Operation *target,
    ApplyToEachResultList &results, transform::TransformState &state) {
  LaunchOp gpuLaunch = dyn_cast<LaunchOp>(target);
  auto transformOp = cast<TransformOpInterface>(getOperation());

  if (!getGenerateGpuLaunch() && !gpuLaunch) {
    DiagnosedSilenceableFailure diag =
        emitSilenceableError()
        << "Given target is not gpu.launch, set `generate_gpu_launch` "
           "attribute";
    diag.attachNote(target->getLoc()) << "when applied to this payload op";
    return diag;
  }

  scf::ForallOp topLevelForallOp;
  DiagnosedSilenceableFailure diag = mlir::transform::gpu::findTopLevelForallOp(
      target, topLevelForallOp, transformOp);
  if (!diag.succeeded()) {
    diag.attachNote(target->getLoc()) << "when applied to this payload op";
    return diag;
  }

  SmallVector<int64_t> gridDims{getGridDims()};
  if (!getGenerateGpuLaunch() && gridDims.size() != 3)
    return transformOp.emitDefiniteFailure("transform require size-3 mapping");

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(topLevelForallOp);

  // Generate gpu launch here and move the forall inside
  if (getGenerateGpuLaunch()) {
    DiagnosedSilenceableFailure diag =
        createGpuLaunch(rewriter, target->getLoc(), transformOp, gpuLaunch);
    if (!diag.succeeded()) {
      return diag;
    }
    rewriter.setInsertionPointToStart(&gpuLaunch.getBody().front());
    Operation *newForallOp = rewriter.clone(*topLevelForallOp);
    rewriter.eraseOp(topLevelForallOp);
    topLevelForallOp = cast<scf::ForallOp>(newForallOp);
  }

  GpuBlockIdBuilder gpuBlockIdBuilder(getContext(), {}, {});
  diag = mlir::transform::gpu::mapForallToBlocksImpl(
      rewriter, transformOp, topLevelForallOp, gridDims, gpuBlockIdBuilder);
  if (!diag.succeeded())
    return diag;

  // Set the GPU launch configuration for the grid dims late, this is subject to
  // IR inspection.
  diag = alterGpuLaunch(rewriter, gpuLaunch,
                        cast<TransformOpInterface>(getOperation()), gridDims[0],
                        gridDims[1], gridDims[2]);

  results.push_back(gpuLaunch);
  return diag;
}

//===----------------------------------------------------------------------===//
// MapNestedForallToThreads
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure mlir::transform::gpu::mapOneForallToThreadsImpl(
    RewriterBase &rewriter, std::optional<TransformOpInterface> transformOp,
    scf::ForallOp forallOp, ArrayRef<int64_t> availableMappingSizes,
    bool syncAfterDistribute, const GpuIdBuilder &gpuIdBuilder) {
  // Ignore cases with different attributes than this builder supports.
  for (Attribute map : forallOp.getMapping()->getValue()) {
    if (!llvm::is_contained(gpuIdBuilder.mappingAttributes, map)) {
      LDBG("--skip " << map);
      LLVM_DEBUG(llvm::interleaveComma(gpuIdBuilder.mappingAttributes,
                                       DBGS() << "----not in: ");
                 llvm::dbgs() << "\n";);
      return emitSilenceableFailure(forallOp);
    }
  }

  Location loc = forallOp.getLoc();
  OpBuilder::InsertionGuard g(rewriter);
  // Insert after to allow for syncthreads after `forall` is erased.
  rewriter.setInsertionPointAfter(forallOp);
  ForallRewriteResult rewriteResult;
  DiagnosedSilenceableFailure diag =
      rewriteOneForallCommonImpl(rewriter, transformOp, forallOp, rewriteResult,
                                 availableMappingSizes, gpuIdBuilder);

  // Return if anything goes wrong, use silenceable failure as a match failure.
  if (!diag.succeeded())
    return diag;

  // Add a syncthreads if needed. TODO: warpsync
  if (syncAfterDistribute)
    rewriter.create<BarrierOp>(loc);

  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure mlir::transform::gpu::mapNestedForallToThreadsImpl(
    RewriterBase &rewriter, std::optional<TransformOpInterface> transformOp,
    Operation *target, ArrayRef<int64_t> blockDims, ArrayRef<int64_t> warpDims,
    bool syncAfterDistribute) {
  LDBG("Start mapNestedForallToThreadsImpl");
  MLIRContext *ctx = rewriter.getContext();
  SmallVector<OpFoldResult> blockDimsOfr =
      getAsIndexOpFoldResult(ctx, blockDims);

  if (blockDims.size() != 3)
    return definiteFailureHelper(transformOp, target,
                                 "requires size-3 thread mapping");
  if (!warpDims.empty()) {
    if (warpDims.size() != 3)
      return definiteFailureHelper(transformOp, target,
                                   "requires empty or size-3 warp mapping");
  }

  // Create an early zero index value for replacements.
  Location loc = target->getLoc();
  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  DiagnosedSilenceableFailure diag = DiagnosedSilenceableFailure::success();
  WalkResult walkResult = target->walk([&](scf::ForallOp forallOp) {
    //===--------------------------------------------------------------------===//
    // Mapping to warp ids.
    //===--------------------------------------------------------------------===//
    if (!warpDims.empty()) {
      LLVM_DEBUG(
          llvm::interleaveComma(
              warpDims, DBGS() << "+mapNestedForallToThreadsImpl warpDims: ");
          llvm::dbgs() << "\n");
      LLVM_DEBUG(llvm::interleaveComma(
                     blockDimsOfr, DBGS() << "--warpDims with blockDimsOfr:  ");
                 llvm::dbgs() << "\n");
      GpuWarpIdBuilder gpuWarpIdBuilder(ctx, blockDimsOfr, warpDims);
      diag = mlir::transform::gpu::mapOneForallToThreadsImpl(
          rewriter, transformOp, forallOp, warpDims, syncAfterDistribute,
          gpuWarpIdBuilder);
      // Use silenceable failure to encode "failure to match" and pass
      // through.
      if (diag.isDefiniteFailure())
        return WalkResult::interrupt();
      if (diag.succeeded())
        return WalkResult::skip();
    }

    //===--------------------------------------------------------------------===//
    // Mapping to linear ids.
    //===--------------------------------------------------------------------===//
    LDBG("+mapNestedForallToThreadsImpl linearDims");
    LLVM_DEBUG(llvm::interleaveComma(
                   blockDimsOfr, DBGS() << "--linearDims with blockDimsOfr:  ");
               llvm::dbgs() << "\n");
    int64_t numThreads = 1;
    for (int64_t b : blockDims)
      numThreads *= b;
    GpuLinearIdBuilder gpuLinearIdBuilder(ctx, blockDimsOfr, numThreads);
    diag = mlir::transform::gpu::mapOneForallToThreadsImpl(
        rewriter, transformOp, forallOp, numThreads, syncAfterDistribute,
        gpuLinearIdBuilder);
    // Use silenceable failure to encode "failure to match" and pass through.
    if (diag.isDefiniteFailure())
      return WalkResult::interrupt();
    if (diag.succeeded())
      return WalkResult::skip();

    //===--------------------------------------------------------------------===//
    // Mapping to block ids (happens last so we can replay ThreadIdOp).
    //===--------------------------------------------------------------------===//
    LLVM_DEBUG(
        llvm::interleaveComma(
            blockDimsOfr, DBGS() << "mapNestedForallToThreadsImpl blockDims: ");
        llvm::dbgs() << "\n");
    GpuThreadIdBuilder gpuThreadIdBuilder(ctx, blockDimsOfr, blockDims);
    diag = mlir::transform::gpu::mapOneForallToThreadsImpl(
        rewriter, transformOp, forallOp, blockDims, syncAfterDistribute,
        gpuThreadIdBuilder);
    // Use silenceable failure to encode "failure to match" and pass through.
    if (diag.isDefiniteFailure())
      return WalkResult::interrupt();

    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted())
    return diag;

  // Replace ids of dimensions known to be 1 by 0 to simplify the IR.
  // Here, the result of mapping determines the available mapping sizes.
  replaceUnitMappingIdsHelper<ThreadIdOp>(rewriter, loc, target, zero,
                                          blockDims);

  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure transform::MapNestedForallToThreads::applyToOne(
    transform::TransformRewriter &rewriter, Operation *target,
    ApplyToEachResultList &results, TransformState &state) {
  LaunchOp gpuLaunch = dyn_cast<LaunchOp>(target);
  auto transformOp = cast<TransformOpInterface>(getOperation());

  // Basic high-level verifications.
  if (!gpuLaunch)
    return emitSilenceableError() << "Given target is not a gpu.launch";

  // Mapping to block ids.
  SmallVector<int64_t> blockDims{getBlockDims()};

  DiagnosedSilenceableFailure diag =
      checkGpuLimits(transformOp, std::nullopt, std::nullopt, std::nullopt,
                     blockDims[0], blockDims[1], blockDims[2]);
  if (diag.isSilenceableFailure()) {
    diag.attachNote(getLoc()) << getBlockDimsAttrName() << " is too large";
    return diag;
  }

  // Set the GPU launch configuration for the block dims early, this is not
  // subject to IR inspection.
  diag = alterGpuLaunch(rewriter, gpuLaunch, transformOp, std::nullopt,
                        std::nullopt, std::nullopt, blockDims[0], blockDims[1],
                        blockDims[2]);

  rewriter.setInsertionPointToStart(&gpuLaunch.getBody().front());
  diag =
      mapNestedForallToThreadsImpl(rewriter, transformOp, gpuLaunch, blockDims,
                                   getWarpDims(), getSyncAfterDistribute());

  results.push_back(gpuLaunch.getOperation());
  return diag;
}

//===----------------------------------------------------------------------===//
// Transform op registration
//===----------------------------------------------------------------------===//

namespace {
/// Registers new ops and declares PDL as dependent dialect since the
/// additional ops are using PDL types for operands and results.
class GPUTransformDialectExtension
    : public transform::TransformDialectExtension<
          GPUTransformDialectExtension> {
public:
  GPUTransformDialectExtension() {
    declareGeneratedDialect<scf::SCFDialect>();
    declareGeneratedDialect<arith::ArithDialect>();
    declareGeneratedDialect<GPUDialect>();
    registerTransformOps<
#define GET_OP_LIST
#include "mlir/Dialect/GPU/TransformOps/GPUTransformOps.cpp.inc"
        >();
  }
};
} // namespace

#define GET_OP_CLASSES
#include "mlir/Dialect/GPU/TransformOps/GPUTransformOps.cpp.inc"

void mlir::gpu::registerTransformDialectExtension(DialectRegistry &registry) {
  registry.addExtensions<GPUTransformDialectExtension>();
}
