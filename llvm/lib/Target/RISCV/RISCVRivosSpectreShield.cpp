//====- RISCVSpeculativeLoadHardening.cpp - A Spectre v1 mitigation ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// Provide a pass which mitigates speculative execution attacks which operate
/// by speculating incorrectly past some predicate (a type check, bounds check,
/// or other condition) to reach a load with invalid inputs and leak the data
/// accessed by that load using a side channel out of the speculative domain.
///
/// For details on the attacks, see the first variant in both the Project Zero
/// writeup and the Spectre paper:
/// https://googleprojectzero.blogspot.com/2018/01/reading-privileged-memory-with-side.html
/// https://spectreattack.com/spectre.pdf
///
//===----------------------------------------------------------------------===//
#include <iostream>
#include "RISCV.h"
#include "MCTargetDesc/RISCVMCExpr.h"
#include "RISCVInstrInfo.h"
#include "RISCVSubtarget.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SparseBitVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/MachineSSAUpdater.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSchedule.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/MC/MCSchedule.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include <algorithm>
#include <cassert>
#include <iterator>
#include <optional>
#include <utility>

using namespace llvm;

#define PASS_KEY "riscv-slh"
#define DEBUG_TYPE PASS_KEY

STATISTIC(NumCondBranchesTraced, "Number of conditional branches traced");
STATISTIC(NumBranchesUntraced, "Number of branches unable to trace");
STATISTIC(NumAddrRegsHardened,
          "Number of address mode used registers hardaned");
STATISTIC(NumPostLoadRegsHardened,
          "Number of post-load register values hardened");
STATISTIC(NumCallsOrJumpsHardened,
          "Number of calls or jumps requiring extra hardening");
STATISTIC(NumInstsInserted, "Number of instructions inserted");
STATISTIC(NumLFENCEsInserted, "Number of lfence instructions inserted");


static cl::opt<bool> EnableSpeculativeLoadHardening(
    "riscv-speculative-load-hardening",
    cl::desc("Force enable speculative load hardening"), cl::init(false),
    cl::Hidden);

static cl::opt<bool> HardenEdgesWithFENCE(
    PASS_KEY "-fence",
    cl::desc(
        "Use FENCE along each conditional edge to harden against speculative "
        "loads rather than conditional_movs/non-speculative_loads/fake-branch-dependence."),
    cl::init(false), cl::Hidden);

static cl::opt<bool> FenceCallAndRet(
    PASS_KEY "-fence-call-and-ret",
    cl::desc("Use a full speculation fence to harden both call and ret edges "
             "rather than a lighter weight mitigation."),
    cl::init(false), cl::Hidden);

static cl::opt<bool> HardenInterprocedurally(
    PASS_KEY "-ip",
    cl::desc("Harden interprocedurally by passing our state in and out of "
             "functions in the high bits of the stack pointer."),
    cl::init(false), cl::Hidden);

static cl::opt<bool>
    HardenLoads(PASS_KEY "-loads",
                cl::desc("Sanitize loads from memory. When disable, no "
                         "significant security is provided."),
                cl::init(true), cl::Hidden);

static cl::opt<bool> HardenIndirectCallsAndJumps(
    PASS_KEY "-indirect",
    cl::desc("Harden indirect calls and jumps against using speculatively "
             "stored attacker controlled addresses. This is designed to "
             "mitigate Spectre v1.2 style attacks."),
    cl::init(false), cl::Hidden);             
namespace {

class RISCVRivosSpectreShieldPass : public MachineFunctionPass {
public:
  RISCVRivosSpectreShieldPass() : MachineFunctionPass(ID) { }

  StringRef getPassName() const override {
    return "RISCV Rivos Spectre Shield Pass";
  }
  bool runOnMachineFunction(MachineFunction &MF) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;

  /// Pass identification, replacement for typeid.
  static char ID;

private:
  /// The information about a block's conditional terminators needed to trace
  /// our predicate state through the exiting edges.
  struct BlockCondInfo {
    MachineBasicBlock *MBB;

    // We mostly have one conditional branch, and in extremely rare cases have
    // two. Three and more are so rare as to be unimportant for compile time.
    SmallVector<MachineInstr *, 2> CondBrs;

    MachineInstr *UncondBr;
  };

  /// Manages the predicate state traced through the program.
  struct PredState {
    unsigned InitialReg = 0;
    unsigned PoisonReg = 0;

    const TargetRegisterClass *RC;
    MachineSSAUpdater SSA;

    PredState(MachineFunction &MF, const TargetRegisterClass *RC)
        : RC(RC), SSA(MF) {}
  };

  //RISCVAsmParser *Parser = nullptr;
  const RISCVSubtarget *Subtarget = nullptr;
  MachineRegisterInfo *MRI = nullptr;
  const RISCVInstrInfo *TII = nullptr;
  const TargetRegisterInfo *TRI = nullptr;

  std::optional<PredState> PS;

  void hardenEdgesWithFENCE(MachineFunction &MF);

  SmallVector<BlockCondInfo, 16> collectBlockCondInfo(MachineFunction &MF);

  SmallVector<MachineInstr *, 16>
  tracePredStateThroughCFG(MachineFunction &MF, ArrayRef<BlockCondInfo> Infos);

  void unfoldCallAndJumpLoads(MachineFunction &MF);

  SmallVector<MachineInstr *, 16>
  tracePredStateThroughIndirectBranches(MachineFunction &MF);

  void tracePredStateThroughBlocksAndHarden(MachineFunction &MF);

  unsigned saveEFLAGS(MachineBasicBlock &MBB,
                      MachineBasicBlock::iterator InsertPt,
                      const DebugLoc &Loc);
  void restoreEFLAGS(MachineBasicBlock &MBB,
                     MachineBasicBlock::iterator InsertPt, const DebugLoc &Loc,
                     Register Reg);

  void mergePredStateIntoSP(MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator InsertPt,
                            const DebugLoc &Loc, unsigned PredStateReg);
  unsigned extractPredStateFromSP(MachineBasicBlock &MBB,
                                  MachineBasicBlock::iterator InsertPt,
                                  const DebugLoc &Loc);

  void
  hardenLoadAddr(MachineInstr &MI, MachineOperand &BaseMO,
                 SmallDenseMap<unsigned, unsigned, 32> &AddrRegToHardenedReg);
  MachineInstr *
  sinkPostLoadHardenedInst(MachineInstr &MI,
                           SmallPtrSetImpl<MachineInstr *> &HardenedInstrs);
  bool canHardenRegister(Register Reg);
  unsigned hardenValueInRegister(Register Reg, MachineBasicBlock &MBB,
                                 MachineBasicBlock::iterator InsertPt,
                                 const DebugLoc &Loc);
  unsigned hardenPostLoad(MachineInstr &MI);
  void hardenReturnInstr(MachineInstr &MI);
  void tracePredStateThroughCall(MachineInstr &MI);
  void hardenIndirectCallOrJumpInstr(
      MachineInstr &MI,
      SmallDenseMap<unsigned, unsigned, 32> &AddrRegToHardenedReg);
};

} // end anonymous namespace

char RISCVRivosSpectreShieldPass::ID = 0;

void RISCVRivosSpectreShieldPass::getAnalysisUsage(
    AnalysisUsage &AU) const {
  MachineFunctionPass::getAnalysisUsage(AU);
}

static RISCVCC::CondCode getCondFromBranchOpc(unsigned Opc) {
  switch (Opc) {
  default:
    return RISCVCC::COND_INVALID;
  case RISCV::BEQ:
    return RISCVCC::COND_EQ;
  case RISCV::BNE:
    return RISCVCC::COND_NE;
  case RISCV::BLT:
    return RISCVCC::COND_LT;
  case RISCV::BGE:
    return RISCVCC::COND_GE;
  case RISCV::BLTU:
    return RISCVCC::COND_LTU;
  case RISCV::BGEU:
    return RISCVCC::COND_GEU;
  }
}

static bool isRegLive(
  MachineBasicBlock &MBB, const TargetRegisterInfo &TRI, SmallVector<llvm::Register, 2> regs) {
  // Check if EFLAGS are alive by seeing if there is a def of them or they
  // live-in, and then seeing if that def is in turn used.
  for (MachineInstr &MI : MBB) {
    for (auto &reg : regs){
      if (MachineOperand *UseOp = MI.findRegisterUseOperand(reg)) {
        // If the def is dead, then EFLAGS is not live.
        if (UseOp->isDead())
          return false;

        // Otherwise we've def'ed it, and it is live.
        return true;
      }
      // While at this instruction, also check if we use and kill EFLAGS
      // which means it isn't live.
      if (MI.killsRegister(reg, &TRI))
        return false;
    }
  }

  // If we didn't find anything conclusive (neither definitely alive or
  // definitely dead) return whether it lives into the block.
  return MBB.isLiveIn(regs[0]) || MBB.isLiveIn(regs[1]);
}

static MachineBasicBlock &splitEdge(MachineBasicBlock &MBB,
                                    MachineBasicBlock &Succ, int SuccCount,
                                    MachineInstr *Br, MachineInstr *&UncondBr,
                                    const RISCVInstrInfo &TII) {
  assert(!Succ.isEHPad() && "Shouldn't get edges to EH pads!");
  MachineFunction &MF = *MBB.getParent();

  MachineBasicBlock &NewMBB = *MF.CreateMachineBasicBlock();

  // We have to insert the new block immediately after the current one as we
  // don't know what layout-successor relationships the successor has and we
  // may not be able to (and generally don't want to) try to fix those up.
  MF.insert(std::next(MachineFunction::iterator(&MBB)), &NewMBB);

  // Update the branch instruction if necessary.
  if (Br) {
    assert(Br->getOperand(Br->getNumExplicitOperands() - 1).getMBB() == &Succ &&
           "Didn't start with the right target!");
    Br->getOperand(Br->getNumExplicitOperands() - 1).setMBB(&NewMBB);

    // If this successor was reached through a branch rather than fallthrough,
    // we might have *broken* fallthrough and so need to inject a new
    // unconditional branch.
    if (!UncondBr) {
      MachineBasicBlock &OldLayoutSucc =
          *std::next(MachineFunction::iterator(&NewMBB));
      assert(MBB.isSuccessor(&OldLayoutSucc) &&
             "Without an unconditional branch, the old layout successor should "
             "be an actual successor!");
      auto BrBuilder =
          BuildMI(&MBB, DebugLoc(), TII.get(RISCV::PseudoBR)).addMBB(&OldLayoutSucc);
      // Update the unconditional branch now that we've added one.
      UncondBr = &*BrBuilder;
    }

    // Insert unconditional "jump Succ" instruction in the new block if
    // necessary.
    if (!NewMBB.isLayoutSuccessor(&Succ)) {
      SmallVector<MachineOperand, 4> Cond;
      TII.insertBranch(NewMBB, &Succ, nullptr, Cond, Br->getDebugLoc());
    }
  } else {
    assert(!UncondBr &&
           "Cannot have a branchless successor and an unconditional branch!");
    assert(NewMBB.isLayoutSuccessor(&Succ) &&
           "A non-branch successor must have been a layout successor before "
           "and now is a layout successor of the new block.");
  }

  // If this is the only edge to the successor, we can just replace it in the
  // CFG. Otherwise we need to add a new entry in the CFG for the new
  // successor.
  if (SuccCount == 1) {
    MBB.replaceSuccessor(&Succ, &NewMBB);
  } else {
    MBB.splitSuccessor(&Succ, &NewMBB);
  }

  // Hook up the edge from the new basic block to the old successor in the CFG.
  NewMBB.addSuccessor(&Succ);

  // Fix PHI nodes in Succ so they refer to NewMBB instead of MBB.
  for (MachineInstr &MI : Succ) {
    if (!MI.isPHI())
      break;
    for (int OpIdx = 1, NumOps = MI.getNumOperands(); OpIdx < NumOps;
         OpIdx += 2) {
      MachineOperand &OpV = MI.getOperand(OpIdx);
      MachineOperand &OpMBB = MI.getOperand(OpIdx + 1);
      assert(OpMBB.isMBB() && "Block operand to a PHI is not a block!");
      if (OpMBB.getMBB() != &MBB)
        continue;

      // If this is the last edge to the succesor, just replace MBB in the PHI
      if (SuccCount == 1) {
        OpMBB.setMBB(&NewMBB);
        break;
      }

      // Otherwise, append a new pair of operands for the new incoming edge.
      MI.addOperand(MF, OpV);
      MI.addOperand(MF, MachineOperand::CreateMBB(&NewMBB));
      break;
    }
  }

  // Inherit live-ins from the successor
  for (auto &LI : Succ.liveins())
    NewMBB.addLiveIn(LI);

  LLVM_DEBUG(dbgs() << "  Split edge from '" << MBB.getName() << "' to '"
                    << Succ.getName() << "'.\n");
  return NewMBB;
}


/// Helper to scan a function for loads vulnerable to misspeculation that we
/// want to harden.
///
/// We use this to avoid making changes to functions where there is nothing we
/// need to do to harden against misspeculation.
static bool hasVulnerableLoad(MachineFunction &MF) {
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      // Loads within this basic block after an LFENCE are not at risk of
      // speculatively executing with invalid predicates from prior control
      // flow. So break out of this block but continue scanning the function.
      if (MI.getOpcode() == RISCV::FENCE)
        break;

      // Looking for loads only.
      if (!MI.mayLoad())
        continue;

      // We found a load.
      return true;
    }
  }

  // No loads found.
  return false;
}

/// Removing duplicate PHI operands to leave the PHI in a canonical and
/// predictable form.
///
/// FIXME: It's really frustrating that we have to do this, but SSA-form in MIR
/// isn't what you might expect. We may have multiple entries in PHI nodes for
/// a single predecessor. This makes CFG-updating extremely complex, so here we
/// simplify all PHI nodes to a model even simpler than the IR's model: exactly
/// one entry per predecessor, regardless of how many edges there are.
static void canonicalizePHIOperands(MachineFunction &MF) {
  SmallPtrSet<MachineBasicBlock *, 4> Preds;
  SmallVector<int, 4> DupIndices;
  for (auto &MBB : MF)
    for (auto &MI : MBB) {
      if (!MI.isPHI())
        break;

      // First we scan the operands of the PHI looking for duplicate entries
      // a particular predecessor. We retain the operand index of each duplicate
      // entry found.
      for (int OpIdx = 1, NumOps = MI.getNumOperands(); OpIdx < NumOps;
           OpIdx += 2)
        if (!Preds.insert(MI.getOperand(OpIdx + 1).getMBB()).second)
          DupIndices.push_back(OpIdx);

      // Now walk the duplicate indices, removing both the block and value. Note
      // that these are stored as a vector making this element-wise removal
      // :w
      // potentially quadratic.
      //
      // FIXME: It is really frustrating that we have to use a quadratic
      // removal algorithm here. There should be a better way, but the use-def
      // updates required make that impossible using the public API.
      //
      // Note that we have to process these backwards so that we don't
      // invalidate other indices with each removal.
      while (!DupIndices.empty()) {
        int OpIdx = DupIndices.pop_back_val();
        // Remove both the block and value operand, again in reverse order to
        // preserve indices.
        MI.removeOperand(OpIdx + 1);
        MI.removeOperand(OpIdx);
      }

      Preds.clear();
    }
}


bool RISCVRivosSpectreShieldPass::runOnMachineFunction(
    MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << "********** " << getPassName() << " : " << MF.getName()
                    << " **********\n");
  // Only run if this pass is forced enabled or we detect the relevant function
  // attribute requesting SLH.
  if (!EnableSpeculativeLoadHardening &&
      !MF.getFunction().hasFnAttribute(Attribute::SpeculativeLoadHardening))
    return false;

  Subtarget = &MF.getSubtarget<RISCVSubtarget>();
  MRI = &MF.getRegInfo();
  TII = Subtarget->getInstrInfo();
  TRI = Subtarget->getRegisterInfo();

  // FIXME: Support for 32-bit.
  PS.emplace(MF, &RISCV::GPRRegClass);

  if (MF.begin() == MF.end())
    // Nothing to do for a degenerate empty function...
    return false;

  // We support an alternative hardening technique based on a debug flag.
  if (HardenEdgesWithFENCE) {
    hardenEdgesWithFENCE(MF);
    return true;
  }

  // Create a dummy debug loc to use for all the generated code here.
  DebugLoc Loc;

  MachineBasicBlock &Entry = *MF.begin();
  auto EntryInsertPt = Entry.SkipPHIsLabelsAndDebug(Entry.begin());

  // Do a quick scan to see if we have any checkable loads.
  bool HasVulnerableLoad = hasVulnerableLoad(MF);

  // See if we have any conditional branching blocks that we will need to trace
  // predicate state through.
  SmallVector<BlockCondInfo, 16> Infos = collectBlockCondInfo(MF);

  // If we have no interesting conditions or loads, nothing to do here.
  if (!HasVulnerableLoad && Infos.empty())
    return true;

  // The poison value is required to be an all-ones value for many aspects of
  // this mitigation.
  /* const int PoisonVal = -1;
  PS->PoisonReg = MRI->createVirtualRegister(PS->RC);
  BuildMI(Entry, EntryInsertPt, Loc, TII->get(RISCV::ADDI), PS->PoisonReg)
      .addReg(RISCV::X0)
      .addImm(PoisonVal);
  ++NumInstsInserted; */

  // If we have loads being hardened and we've asked for call and ret edges to
  // get a full fence-based mitigation, inject that fence.
  if (HasVulnerableLoad && FenceCallAndRet) {
    // We need to insert an LFENCE at the start of the function to suspend any
    // incoming misspeculation from the caller. This helps two-fold: the caller
    // may not have been protected as this code has been, and this code gets to
    // not take any specific action to protect across calls.
    // FIXME: We could skip this for functions which unconditionally return
    // a constant.
    BuildMI(Entry, EntryInsertPt, Loc, TII->get(RISCV::FENCE));
    ++NumInstsInserted;
    ++NumLFENCEsInserted;
  }

  // If we guarded the entry with an LFENCE and have no conditionals to protect
  // in blocks, then we're done.
  if (FenceCallAndRet && Infos.empty())
    // We may have changed the function's code at this point to insert fences.
    return true;

  
  // For every basic block in the function which can b
  if (HardenInterprocedurally && !FenceCallAndRet) {
    // Set up the predicate state by extracting it from the incoming stack
    // pointer so we pick up any misspeculation in our caller.
    PS->InitialReg = extractPredStateFromSP(Entry, EntryInsertPt, Loc);
  } else {
    // Otherwise, just build the predicate state itself by zeroing a register
    // as we don't need any initial state.
    PS->InitialReg = MRI->createVirtualRegister(PS->RC);
    auto ZeroI = BuildMI(Entry, EntryInsertPt, Loc, TII->get(RISCV::ADDI),
                          PS->InitialReg)
                          .addReg(RISCV::X0)
                          .addImm(0);
    ++NumInstsInserted;
  }
    

  // We're going to need to trace predicate state throughout the function's
  // CFG. Prepare for this by setting up our initial state of PHIs with unique
  // predecessor entries and all the initial predicate state.
  canonicalizePHIOperands(MF);

  // Track the updated values in an SSA updater to rewrite into SSA form at the
  // end.
  PS->SSA.Initialize(PS->InitialReg);
  PS->SSA.AddAvailableValue(&Entry, PS->InitialReg);

  // Trace through the CFG.
  auto CMovs = tracePredStateThroughCFG(MF, Infos);

  /*
  // We may also enter basic blocks in this function via exception handling
  // control flow. Here, if we are hardening interprocedurally, we need to
  // re-capture the predicate state from the throwing code. In the Itanium ABI,
  // the throw will always look like a call to __cxa_throw and will have the
  // predicate state in the stack pointer, so extract fresh predicate state from
  // the stack pointer and make it available in SSA.
  // FIXME: Handle non-itanium ABI EH models.
  if (HardenInterprocedurally) {
    for (MachineBasicBlock &MBB : MF) {
      assert(!MBB.isEHScopeEntry() && "Only Itanium ABI EH supported!");
      assert(!MBB.isEHFuncletEntry() && "Only Itanium ABI EH supported!");
      assert(!MBB.isCleanupFuncletEntry() && "Only Itanium ABI EH supported!");
      if (!MBB.isEHPad())
        continue;
      PS->SSA.AddAvailableValue(
          &MBB,
          extractPredStateFromSP(MBB, MBB.SkipPHIsAndLabels(MBB.begin()), Loc));
    }
  }
  */

  
  if (HardenIndirectCallsAndJumps) {
    // If we are going to harden calls and jumps we need to unfold their memory
    // operands.
    //unfoldCallAndJumpLoads(MF);

    // Then we trace predicate state through the indirect branches.
    auto IndirectBrCMovs = tracePredStateThroughIndirectBranches(MF);
    CMovs.append(IndirectBrCMovs.begin(), IndirectBrCMovs.end());
  }
  

  // Now that we have the predicate state available at the start of each block
  // in the CFG, trace it through each block, hardening vulnerable instructions
  // as we go.
  tracePredStateThroughBlocksAndHarden(MF);

  // Now rewrite all the uses of the pred state using the SSA updater to insert
  // PHIs connecting the state between blocks along the CFG edges.
  for (MachineInstr *CMovI : CMovs)
    for (MachineOperand &Op : CMovI->operands()) {
      if (!Op.isReg() || Op.getReg() != PS->InitialReg)
        continue;

      PS->SSA.RewriteUse(Op);
    }

  LLVM_DEBUG(dbgs() << "Final speculative load hardened function:\n"; MF.dump();
             dbgs() << "\n"; MF.verify(this));
  return true;
}



SmallVector<RISCVRivosSpectreShieldPass::BlockCondInfo, 16>
RISCVRivosSpectreShieldPass::collectBlockCondInfo(MachineFunction &MF) {
  SmallVector<BlockCondInfo, 16> Infos;

  // Walk the function and build up a summary for each block's conditions that
  // we need to trace through.
  for (MachineBasicBlock &MBB : MF) {
    // If there are no or only one successor, nothing to do here.
    if (MBB.succ_size() <= 1)
      continue;

    // We want to reliably handle any conditional branch terminators in the
    // MBB, so we manually analyze the branch. We can handle all of the
    // permutations here, including ones that analyze branch cannot.
    //
    // The approach is to walk backwards across the terminators, resetting at
    // any unconditional non-indirect branch, and track all conditional edges
    // to basic blocks as well as the fallthrough or unconditional successor
    // edge. For each conditional edge, we track the target and the opposite
    // condition code in order to inject a "no-op" cmov into that successor
    // that will harden the predicate. For the fallthrough/unconditional
    // edge, we inject a separate cmov for each conditional branch with
    // matching condition codes. This effectively implements an "and" of the
    // condition flags, even if there isn't a single condition flag that would
    // directly implement that. We don't bother trying to optimize either of
    // these cases because if such an optimization is possible, LLVM should
    // have optimized the conditional *branches* in that way already to reduce
    // instruction count. This late, we simply assume the minimal number of
    // branch instructions is being emitted and use that to guide our cmov
    // insertion.

    BlockCondInfo Info = {&MBB, {}, nullptr};

    // Now walk backwards through the terminators and build up successors they
    // reach and the conditions.
    for (MachineInstr &MI : llvm::reverse(MBB)) {
      // Once we've handled all the terminators, we're done.
      if (!MI.isTerminator())
        break;

      // If we see a non-branch terminator, we can't handle anything so bail.
      if (!MI.isBranch()) {
        Info.CondBrs.clear();
        break;
      }

      // If we see an unconditional branch, reset our state, clear any
      // fallthrough, and set this is the "else" successor.
      if (MI.isUnconditionalBranch()) {
        Info.CondBrs.clear();
        Info.UncondBr = &MI;
        continue;
      }

      // If we get an invalid condition, we have an indirect branch or some
      // other unanalyzable "fallthrough" case. We model this as a nullptr for
      // the destination so we can still guard any conditional successors.
      // Consider code sequences like:
      // ```
      //   jCC L1
      //   jmpq *%rax
      // ```
      // We still want to harden the edge to `L1`.
      if (getCondFromBranchOpc(MI.getOpcode()) == RISCVCC::COND_INVALID) {
        Info.CondBrs.clear();
        Info.UncondBr = &MI;
        continue;
      }

      if (MI.isConditionalBranch()){
        Info.CondBrs.push_back(&MI);
      }
        
      
    }
    if (Info.CondBrs.empty()) {
      ++NumBranchesUntraced;
      LLVM_DEBUG(dbgs() << "WARNING: unable to secure successors of block:\n";
                 MBB.dump());
      continue;
    }

    Infos.push_back(Info);
  }

  return Infos;
}


/// Trace the predicate state through the CFG, instrumenting each conditional
/// branch such that misspeculation through an edge will poison the predicate
/// state.
///
/// Returns the list of inserted CMov instructions so that they can have their
/// uses of the predicate state rewritten into proper SSA form once it is
/// complete.
SmallVector<MachineInstr *, 16>
RISCVRivosSpectreShieldPass::tracePredStateThroughCFG(
    MachineFunction &MF, ArrayRef<BlockCondInfo> Infos) {
  // Collect the inserted cmov instructions so we can rewrite their uses of the
  // predicate state into SSA form.
  SmallVector<MachineInstr *, 16> CMovs;

  // Now walk all of the basic blocks looking for ones that end in conditional
  // jumps where we need to update this register along each edge.
  for (const BlockCondInfo &Info : Infos) {
    MachineBasicBlock &MBB = *Info.MBB;
    const SmallVectorImpl<MachineInstr *> &CondBrs = Info.CondBrs;
    MachineInstr *UncondBr = Info.UncondBr;
    LLVM_DEBUG(dbgs() << "Tracing predicate through block: " << MBB.getName()
                      << "\n");
    ++NumCondBranchesTraced;
    // Compute the non-conditional successor as either the target of any
    // unconditional branch or the layout successor.
    MachineBasicBlock *UncondSucc =
        UncondBr ? (UncondBr->isUnconditionalBranch()
                        ? UncondBr->getOperand(UncondBr->getNumExplicitOperands()-1).getMBB()
                        : nullptr)
                 : &*std::next(MachineFunction::iterator(&MBB));
    // Count how many edges there are to any given successor.
    SmallDenseMap<MachineBasicBlock *, int> SuccCounts;
    if (UncondSucc)
      ++SuccCounts[UncondSucc];
    for (auto *CondBr : CondBrs){
      ++SuccCounts[CondBr->getOperand(CondBr->getNumExplicitOperands() - 1).getMBB()];
    }

    // A lambda to insert cmov instructions into a block checking all of the
    // condition codes in a sequence.
    auto BuildCheckingBlockForSuccAndConds =
        [&](MachineBasicBlock &MBB, MachineBasicBlock &Succ, int SuccCount,
            MachineInstr *Br, MachineInstr *&UncondBr,
            ArrayRef<std::pair<MachineInstr*, RISCVCC::CondCode>> Conds, bool isBrCond) {
          // First, we split the edge to insert the checking block into a safe
          // location.
          auto &CheckingMBB =
              (SuccCount == 1 && Succ.pred_size() == 1)
                  ? Succ
                  : splitEdge(MBB, Succ, SuccCount, Br, UncondBr, *TII);

          /*
          bool LiveEFLAGS = Succ.isLiveIn(X86::EFLAGS);
          if (!LiveEFLAGS)
            CheckingMBB.addLiveIn(X86::EFLAGS);
          */
          
          // Now insert the cmovs to implement the checks.
          auto InsertPt = CheckingMBB.begin();
          assert((InsertPt == CheckingMBB.end() || !InsertPt->isPHI()) &&
                 "Should never have a PHI in the initial checking block as it "
                 "always has a single predecessor!");

          // We will wire each cmov to each other, but need to start with the
          // incoming pred state.
          unsigned CurStateReg = PS->InitialReg;

          for (auto Cond : Conds) {
            //int PredStateSizeInBytes = TRI->getRegSizeInBits(*PS->RC) / 8;
            //auto CMovOp = X86::getCMovOpcode(PredStateSizeInBytes);

            //Add branch operands to the created basic block as live-ins
            MachineOperand &Operand1 = Cond.first->getOperand(0); 
            MachineOperand &Operand2 = Cond.first->getOperand(1); 
            Operand1.setIsKill(false);
            Operand2.setIsKill(false);



            llvm::MachineInstrBuilder XORBr;
            llvm::MachineInstrBuilder SLTBr;
            llvm::MachineInstrBuilder NegBr;
            llvm::MachineInstrBuilder ADDIBr;
            llvm::MachineInstrBuilder ORBr;
            Register XORReg = MRI->createVirtualRegister(PS->RC);
            Register SLTReg = MRI->createVirtualRegister(PS->RC);
            Register NegReg = MRI->createVirtualRegister(PS->RC);
            Register UpdatedStateReg = MRI->createVirtualRegister(PS->RC);
            
            if ((isBrCond && Cond.second == RISCVCC::COND_EQ) || 
            (!isBrCond && Cond.second == RISCVCC::COND_NE)){
              XORBr = BuildMI(CheckingMBB, InsertPt, DebugLoc(),
                              TII->get(RISCV::XOR), XORReg)
                             .addReg(Cond.first->getOperand(0).getReg())
                             .addReg(Cond.first->getOperand(1).getReg());

              SLTBr = BuildMI(CheckingMBB, InsertPt, DebugLoc(),
                        TII->get(RISCV::SLTIU), SLTReg)
                        .addReg(XORReg)
                        .addImm(1);
                   
              ADDIBr = BuildMI(CheckingMBB, InsertPt, DebugLoc(),
                        TII->get(RISCV::ADDIW), NegReg)
                        .addReg(SLTReg)
                        .addImm(-1);   
            }
            else if ((isBrCond && Cond.second == RISCVCC::COND_NE) || 
            (!isBrCond && Cond.second == RISCVCC::COND_EQ)) {
              XORBr = BuildMI(CheckingMBB, InsertPt, DebugLoc(),
                              TII->get(RISCV::XOR), XORReg)
                             .addReg(Cond.first->getOperand(0).getReg())
                             .addReg(Cond.first->getOperand(1).getReg());

              SLTBr = BuildMI(CheckingMBB, InsertPt, DebugLoc(),
                        TII->get(RISCV::SLTU), SLTReg)
                        .addReg(RISCV::X0)
                        .addReg(XORReg);
                   
              ADDIBr = BuildMI(CheckingMBB, InsertPt, DebugLoc(),
                        TII->get(RISCV::ADDIW), NegReg)
                        .addReg(SLTReg)
                        .addImm(-1); 
            }
            else if ((isBrCond && Cond.second == RISCVCC::COND_GE) || 
            (!isBrCond && Cond.second == RISCVCC::COND_LT)) {
              SLTBr = BuildMI(CheckingMBB, InsertPt, DebugLoc(),
                        TII->get(RISCV::SLT), SLTReg)
                        .addReg(Cond.first->getOperand(0).getReg())
                        .addReg(Cond.first->getOperand(1).getReg());
              NegBr = BuildMI(CheckingMBB, InsertPt, DebugLoc(),
                              TII->get(RISCV::SUBW), NegReg)
                             .addReg(RISCV::X0)
                             .addReg(SLTReg);
            }
            else if ((isBrCond && Cond.second == RISCVCC::COND_LT) || 
            (!isBrCond && Cond.second == RISCVCC::COND_GE)) {
              SLTBr = BuildMI(CheckingMBB, InsertPt, DebugLoc(),
                        TII->get(RISCV::SLT), SLTReg)
                        .addReg(Cond.first->getOperand(0).getReg())
                        .addReg(Cond.first->getOperand(1).getReg());
              ADDIBr = BuildMI(CheckingMBB, InsertPt, DebugLoc(),
                        TII->get(RISCV::ADDIW), NegReg)
                        .addReg(SLTReg)
                        .addImm(-1); 
            }
            else if ((isBrCond && Cond.second == RISCVCC::COND_GEU) || 
            (!isBrCond && Cond.second == RISCVCC::COND_LTU)) {
              SLTBr = BuildMI(CheckingMBB, InsertPt, DebugLoc(),
                        TII->get(RISCV::SLTU), SLTReg)
                        .addReg(Cond.first->getOperand(0).getReg())
                        .addReg(Cond.first->getOperand(1).getReg());
              NegBr = BuildMI(CheckingMBB, InsertPt, DebugLoc(),
                              TII->get(RISCV::SUBW), NegReg)
                             .addReg(RISCV::X0)
                             .addReg(SLTReg);
            }
            else if ((isBrCond && Cond.second == RISCVCC::COND_LTU) || 
            (!isBrCond && Cond.second == RISCVCC::COND_GEU)) {
              SLTBr = BuildMI(CheckingMBB, InsertPt, DebugLoc(),
                        TII->get(RISCV::SLTU), SLTReg)
                        .addReg(Cond.first->getOperand(0).getReg())
                        .addReg(Cond.first->getOperand(1).getReg());
              ADDIBr = BuildMI(CheckingMBB, InsertPt, DebugLoc(),
                        TII->get(RISCV::ADDIW), NegReg)
                        .addReg(SLTReg)
                        .addImm(-1); 
            }
              ORBr = BuildMI(CheckingMBB, InsertPt, DebugLoc(),
                              TII->get(RISCV::OR), UpdatedStateReg)
                             .addReg(CurStateReg)
                             .addReg(NegReg);

            
            
            /*             
            // If this is the last cmov and the EFLAGS weren't originally
            // live-in, mark them as killed.
            if (!LiveEFLAGS && Cond == Conds.back())
              CMovI->findRegisterUseOperand(X86::EFLAGS)->setIsKill(true);
            */   

            ++NumInstsInserted;
            //LLVM_DEBUG(dbgs() << "  Inserting cmov: "; CMovI->dump();
                       //dbgs() << "\n");

            // The first one of the cmovs will be using the top level
            // `PredStateReg` and need to get rewritten into SSA form.
            if (CurStateReg == PS->InitialReg)
              //CMovs.push_back(&*CMovI);
                CMovs.push_back(&*ORBr);

            // The next cmov should start from this one's def.
            CurStateReg = UpdatedStateReg;
          }
          // And put the last one into the available values for SSA form of our
          // predicate state.
          PS->SSA.AddAvailableValue(&CheckingMBB, CurStateReg);
        };

    std::vector<std::pair<MachineInstr*, RISCVCC::CondCode>> UncondCodeSeq;
    for (auto *CondBr : CondBrs) {
      MachineBasicBlock &Succ = *CondBr->getOperand(CondBr->getNumExplicitOperands() - 1).getMBB();
      int &SuccCount = SuccCounts[&Succ];

      RISCVCC::CondCode Cond = getCondFromBranchOpc(CondBr->getOpcode());
      UncondCodeSeq.push_back({CondBr, Cond});

      // Check whether there is any instruction in the destination block that is dependent 
      // on the branch register operands. If not, we don't need to insert CMovs here
      //FIXME: Only check loads in the destination block
      //FIXME: Find a way to only check the operand that can be manipulated by untrusted user
      //if (isRegLive(Succ, *TRI, {CondBr->getOperand(0).getReg(), CondBr->getOperand(1).getReg()})){
        BuildCheckingBlockForSuccAndConds(MBB, Succ, SuccCount, CondBr, UncondBr,
                                        {{CondBr, Cond}}, true);
        // Decrement the successor count now that we've split one of the edges.
        // We need to keep the count of edges to the successor accurate in order
        // to know above when to *replace* the successor in the CFG vs. just
        // adding the new successor.
        --SuccCount;
      //}
    }
    // Since we may have split edges and changed the number of successors,
    // normalize the probabilities. This avoids doing it each time we split an
    // edge.
    MBB.normalizeSuccProbs();

    // Finally, we need to insert cmovs into the "fallthrough" edge. Here, we
    // need to intersect the other condition codes. We can do this by just
    // doing a cmov for each one.
    if (!UncondSucc)
      // If we have no fallthrough to protect (perhaps it is an indirect jump?)
      // just skip this and continue.
      continue;
    assert(SuccCounts[UncondSucc] == 1 &&
           "We should never have more than one edge to the unconditional "
           "successor at this point because every other edge must have been "
           "split above!");

    // Sort and unique the codes to minimize them.
    llvm::sort(UncondCodeSeq);
    UncondCodeSeq.erase(std::unique(UncondCodeSeq.begin(), UncondCodeSeq.end()),
                        UncondCodeSeq.end());
    
    // Check whether there is any instruction in the destination block that is dependent 
    // on the branch register operands. If not, we don't need to insert CMovs here
    for (auto BrCondPair: UncondCodeSeq){
      //if (isRegLive(*UncondSucc, *TRI, {BrCondPair.first->getOperand(0).getReg(), BrCondPair.first->getOperand(1).getReg()})){
        // Build a checking version of the successor.
        BuildCheckingBlockForSuccAndConds(MBB, *UncondSucc, /*SuccCount*/ 1,
                                      UncondBr, UncondBr, UncondCodeSeq, false);
        break;
      //}
    }
    
    
  }
  return CMovs;
}




/// Trace the predicate state through each of the blocks in the function,
/// hardening everything necessary along the way.
///
/// We call this routine once the initial predicate state has been established
/// for each basic block in the function in the SSA updater. This routine traces
/// it through the instructions within each basic block, and for non-returning
/// blocks informs the SSA updater about the final state that lives out of the
/// block. Along the way, it hardens any vulnerable instruction using the
/// currently valid predicate state. We have to do these two things together
/// because the SSA updater only works across blocks. Within a block, we track
/// the current predicate state directly and update it as it changes.
///
/// This operates in two passes over each block. First, we analyze the loads in
/// the block to determine which strategy will be used to harden them: hardening
/// the address or hardening the loaded value when loaded into a register
/// amenable to hardening. We have to process these first because the two
/// strategies may interact -- later hardening may change what strategy we wish
/// to use. We also will analyze data dependencies between loads and avoid
/// hardening those loads that are data dependent on a load with a hardened
/// address. We also skip hardening loads already behind an LFENCE as that is
/// sufficient to harden them against misspeculation.
///
/// Second, we actively trace the predicate state through the block, applying
/// the hardening steps we determined necessary in the first pass as we go.
///
/// These two passes are applied to each basic block. We operate one block at a
/// time to simplify reasoning about reachability and sequencing.
void RISCVRivosSpectreShieldPass::tracePredStateThroughBlocksAndHarden(
    MachineFunction &MF) {
  SmallPtrSet<MachineInstr *, 16> HardenPostLoad;
  SmallPtrSet<MachineInstr *, 16> HardenLoadAddr;

  SmallSet<unsigned, 16> HardenedAddrRegs;

  SmallDenseMap<unsigned, unsigned, 32> AddrRegToHardenedReg;

  // Track the set of load-dependent registers through the basic block. Because
  // the values of these registers have an existing data dependency on a loaded
  // value which we would have checked, we can omit any checks on them.
  SparseBitVector<> LoadDepRegs;

  for (MachineBasicBlock &MBB : MF) {
    // The first pass over the block: collect all the loads which can have their
    // loaded value hardened and all the loads that instead need their address
    // hardened. During this walk we propagate load dependence for address
    // hardened loads and also look for LFENCE to stop hardening wherever
    // possible. When deciding whether or not to harden the loaded value or not,
    // we check to see if any registers used in the address will have been
    // hardened at this point and if so, harden any remaining address registers
    // as that often successfully re-uses hardened addresses and minimizes
    // instructions.
    //
    // FIXME: We should consider an aggressive mode where we continue to keep as
    // many loads value hardened even when some address register hardening would
    // be free (due to reuse).
    //
    // Note that we only need this pass if we are actually hardening loads.
    if (HardenLoads)
      for (MachineInstr &MI : MBB) {
        // We naively assume that all def'ed registers of an instruction have
        // a data dependency on all of their operands.
        // FIXME: Do a more careful analysis of x86 to build a conservative
        // model here.
        if (llvm::any_of(MI.uses(), [&](MachineOperand &Op) {
              return Op.isReg() && LoadDepRegs.test(Op.getReg());
            }))
          for (MachineOperand &Def : MI.defs())
            if (Def.isReg())
              LoadDepRegs.set(Def.getReg());

        // Both Intel and AMD are guiding that they will change the semantics of
        // LFENCE to be a speculation barrier, so if we see an LFENCE, there is
        // no more need to guard things in this block.
        if (MI.getOpcode() == RISCV::FENCE)
          break;
        // If this instruction cannot load, nothing to do.
        if (!MI.mayLoad()){
          continue;
        } //|| MI.getNumExplicitOperands() != 3)
          
        
        
        /*
        // Some instructions which "load" are trivially safe or unimportant.
        
        if (MI.getOpcode() == X86::MFENCE)
          continue;
        

        // Extract the memory operand information about this instruction.
        // FIXME: This doesn't handle loading pseudo instructions which we often
        // could handle with similarly generic logic. We probably need to add an
        // MI-layer routine similar to the MC-layer one we use here which maps
        // pseudos much like this maps real instructions.
        const MCInstrDesc &Desc = MI.getDesc();
        int MemRefBeginIdx = X86II::getMemoryOperandNo(Desc.TSFlags);
        if (MemRefBeginIdx < 0) {
          LLVM_DEBUG(dbgs()
                         << "WARNING: unable to harden loading instruction: ";
                     MI.dump());
          continue;
        }

        MemRefBeginIdx += X86II::getOperandBias(Desc);
        */
        MachineOperand *BaseMOPtr = &MI.getOperand(1);
        if (MI.getNumOperands() > 3){
          BaseMOPtr = &MI.getOperand(2);
        }
        /*MachineOperand &IndexMO =
            MI.getOperand(2);*/
        MachineOperand &BaseMO = *BaseMOPtr;
        // If we have at least one (non-frame-index, non-RIP) register operand,
        // and neither operand is load-dependent, we need to check the load.
        unsigned BaseReg = 0;
        if (BaseMO.isReg() && !BaseMO.isFI() &&
             BaseMO.getReg() != RISCV::NoRegister)
          BaseReg = BaseMO.getReg();

        if (!BaseReg)
          // No register operands!
          continue;

        // If any register operand is dependent, this load is dependent and we
        // needn't check it.
        // FIXME: Is this true in the case where we are hardening loads after
        // they complete? Unclear, need to investigate.
        if ((BaseReg && LoadDepRegs.test(BaseReg)))
          continue;
        /*
        // If post-load hardening is enabled, this load is compatible with
        // post-load hardening, and we aren't already going to harden one of the
        // address registers, queue it up to be hardened post-load. Notably,
        // even once hardened this won't introduce a useful dependency that
        // could prune out subsequent loads.
        if (EnablePostLoadHardening && X86InstrInfo::isDataInvariantLoad(MI) &&
            !isEFLAGSDefLive(MI) && MI.getDesc().getNumDefs() == 1 &&
            MI.getOperand(0).isReg() &&
            canHardenRegister(MI.getOperand(0).getReg()) &&
            !HardenedAddrRegs.count(BaseReg) &&
            !HardenedAddrRegs.count(IndexReg)) {
          HardenPostLoad.insert(&MI);
          HardenedAddrRegs.insert(MI.getOperand(0).getReg());
          continue;
        }
        */
        // Record this instruction for address hardening and record its register
        // operands as being address-hardened.
        HardenLoadAddr.insert(&MI);
        if (BaseReg)
          HardenedAddrRegs.insert(BaseReg);
        for (MachineOperand &Def : MI.defs())
          if (Def.isReg())
            LoadDepRegs.set(Def.getReg());
        
      }
    // Now re-walk the instructions in the basic block, and apply whichever
    // hardening strategy we have elected. Note that we do this in a second
    // pass specifically so that we have the complete set of instructions for
    // which we will do post-load hardening and can defer it in certain
    // circumstances.
    for (MachineInstr &MI : MBB) {
      if (HardenLoads) {
        // We cannot both require hardening the def of a load and its address.
        assert(!(HardenLoadAddr.count(&MI) && HardenPostLoad.count(&MI)) &&
               "Requested to harden both the address and def of a load!");

        // Check if this is a load whose address needs to be hardened.
        if (HardenLoadAddr.erase(&MI)) {
          /*
          const MCInstrDesc &Desc = MI.getDesc();
          int MemRefBeginIdx = X86II::getMemoryOperandNo(Desc.TSFlags);
          assert(MemRefBeginIdx >= 0 && "Cannot have an invalid index here!");

          MemRefBeginIdx += X86II::getOperandBias(Desc);
          */
          MachineOperand *BaseMOPtr = &MI.getOperand(1);
          if (MI.getNumOperands() > 3){
            BaseMOPtr = &MI.getOperand(2);
          }
        
          MachineOperand &BaseMO = *BaseMOPtr;

          
          /*MachineOperand &IndexMO =
            MI.getOperand(2);*/
          hardenLoadAddr(MI, BaseMO, AddrRegToHardenedReg);
          continue;
        }
        
        /*
        // Test if this instruction is one of our post load instructions (and
        // remove it from the set if so).
        if (HardenPostLoad.erase(&MI)) {
          assert(!MI.isCall() && "Must not try to post-load harden a call!");

          // If this is a data-invariant load and there is no EFLAGS
          // interference, we want to try and sink any hardening as far as
          // possible.
          if (X86InstrInfo::isDataInvariantLoad(MI) && !isEFLAGSDefLive(MI)) {
            // Sink the instruction we'll need to harden as far as we can down
            // the graph.
            MachineInstr *SunkMI = sinkPostLoadHardenedInst(MI, HardenPostLoad);

            // If we managed to sink this instruction, update everything so we
            // harden that instruction when we reach it in the instruction
            // sequence.
            if (SunkMI != &MI) {
              // If in sinking there was no instruction needing to be hardened,
              // we're done.
              if (!SunkMI)
                continue;

              // Otherwise, add this to the set of defs we harden.
              HardenPostLoad.insert(SunkMI);
              continue;
            }
          }

          unsigned HardenedReg = hardenPostLoad(MI);

          // Mark the resulting hardened register as such so we don't re-harden.
          AddrRegToHardenedReg[HardenedReg] = HardenedReg;

          continue;
        }
        */

        // Check for an indirect call or branch that may need its input hardened
        // even if we couldn't find the specific load used, or were able to
        // avoid hardening it for some reason. Note that here we cannot break
        // out afterward as we may still need to handle any call aspect of this
        // instruction.
        if ((MI.isIndirectBranch()) && HardenIndirectCallsAndJumps)
          hardenIndirectCallOrJumpInstr(MI, AddrRegToHardenedReg);
      }

      
      // After we finish hardening loads we handle interprocedural hardening if
      // enabled and relevant for this instruction.
      if (!HardenInterprocedurally)
        continue;
      if (!MI.isCall() && !MI.isReturn())
        continue;

      // If this is a direct return (IE, not a tail call) just directly harden
      // it.
      if (MI.isReturn() && !MI.isCall()) {
        hardenReturnInstr(MI);
        continue;
      }
      
      
      // Otherwise we have a call. We need to handle transferring the predicate
      // state into a call and recovering it after the call returns (unless this
      // is a tail call).
      //assert(MI.isCall() && "Should only reach here for calls!");
      //assert(!MI.mayLoad() && "Should only reach here for non-loads!");
      if (MI.isCall())
        tracePredStateThroughCall(MI);
    }

    HardenPostLoad.clear();
    HardenLoadAddr.clear();
    HardenedAddrRegs.clear();
    AddrRegToHardenedReg.clear();

    // Currently, we only track data-dependent loads within a basic block.
    // FIXME: We should see if this is necessary or if we could be more
    // aggressive here without opening up attack avenues.
    LoadDepRegs.clear();
  }
}

/// Implements the naive hardening approach of putting an LFENCE after every
/// potentially mis-predicted control flow construct.
///
/// We include this as an alternative mostly for the purpose of comparison. The
/// performance impact of this is expected to be extremely severe and not
/// practical for any real-world users.
void RISCVRivosSpectreShieldPass::hardenEdgesWithFENCE(
    MachineFunction &MF) {
  // First, we scan the function looking for blocks that are reached along edges
  // that we might want to harden.
  SmallSetVector<MachineBasicBlock *, 8> Blocks;
  for (MachineBasicBlock &MBB : MF) {
    // If there are no or only one successor, nothing to do here.
    if (MBB.succ_size() <= 1)
      continue;

    // Skip blocks unless their terminators start with a branch. Other
    // terminators don't seem interesting for guarding against misspeculation.
    auto TermIt = MBB.getFirstTerminator();
    if (TermIt == MBB.end() || !TermIt->isBranch())
      continue;

    // Add all the non-EH-pad succossors to the blocks we want to harden. We
    // skip EH pads because there isn't really a condition of interest on
    // entering.
    for (MachineBasicBlock *SuccMBB : MBB.successors())
      if (!SuccMBB->isEHPad())
        Blocks.insert(SuccMBB);
  }

  for (MachineBasicBlock *MBB : Blocks) {
    auto InsertPt = MBB->SkipPHIsAndLabels(MBB->begin());
    BuildMI(*MBB, InsertPt, DebugLoc(), TII->get(RISCV::FENCE));
    ++NumInstsInserted;
    ++NumLFENCEsInserted;
  }
}



void RISCVRivosSpectreShieldPass::hardenLoadAddr(
    MachineInstr &MI, MachineOperand &BaseMO,
    SmallDenseMap<unsigned, unsigned, 32> &AddrRegToHardenedReg) {
      
  MachineBasicBlock &MBB = *MI.getParent();
  const DebugLoc &Loc = MI.getDebugLoc(); 
  // Check if EFLAGS are alive by seeing if there is a def of them or they
  // live-in, and then seeing if that def is in turn used.
  //bool EFLAGSLive = isEFLAGSLive(MBB, MI.getIterator(), *TRI);

  SmallVector<MachineOperand *, 2> HardenOpRegs;
  if (BaseMO.isFI()) {
    // A frame index is never a dynamically controllable load, so only
    // harden it if we're covering fixed address loads as well.
    LLVM_DEBUG(
        dbgs() << "  Skipping hardening base of explicit stack frame load: ";
        MI.dump(); dbgs() << "\n");
  } else if (BaseMO.getReg() == RISCV::SP) {
    // Some idempotent atomic operations are lowered directly to a locked
    // OR with 0 to the top of stack(or slightly offset from top) which uses an
    // explicit RSP register as the base.
    //assert(IndexMO.getReg() == X86::NoRegister &&
    //       "Explicit RSP access with dynamic index!");
    LLVM_DEBUG(
        dbgs() << "  Cannot harden base of explicit RSP offset in a load!");
  } else if (BaseMO.getReg() == RISCV::NoRegister) {
    // For both RIP-relative addressed loads or absolute loads, we cannot
    // meaningfully harden them because the address being loaded has no
    // dynamic component.
    //
    // FIXME: When using a segment base (like TLS does) we end up with the
    // dynamic address being the base plus -1 because we can't mutate the
    // segment register here. This allows the signed 32-bit offset to point at
    // valid segment-relative addresses and load them successfully.
    LLVM_DEBUG(
        dbgs() << "  Cannot harden base of "
               << "no base "
               << " address in a load!");
  } 
   else if (!BaseMO.getReg().isVirtual()) {
    // Currently only support virtual registers
    LLVM_DEBUG(
        dbgs() << "  Cannot harden "
               << "a non-virtual "
               << " register!");
  }
  else {
    assert(BaseMO.isReg() &&
           "Only allowed to have a frame index or register base.");
    HardenOpRegs.push_back(&BaseMO);
  }
  
  /*if (BaseMO.getReg() != RISCV::NoRegister &&
      (HardenOpRegs.empty() ||
       HardenOpRegs.front()->getReg() != BaseMO.getReg())){
          HardenOpRegs.push_back(&BaseMO);
  }
  */
  assert((HardenOpRegs.size() == 0 || HardenOpRegs.size() == 1) &&
         "Should have exactly zero or 1 registers to harden!");

  /* assert((HardenOpRegs.size() == 1 ||
          HardenOpRegs[0]->getReg() != HardenOpRegs[1]->getReg()) &&
         "Should not have two of the same registers!"); */

  // Remove any registers that have alreaded been checked.
  llvm::erase_if(HardenOpRegs, [&](MachineOperand *Op) {
    // See if this operand's register has already been checked.
    auto It = AddrRegToHardenedReg.find(Op->getReg());
    if (It == AddrRegToHardenedReg.end())
      // Not checked, so retain this one.
      return false;

    // Otherwise, we can directly update this operand and remove it.
    Op->setReg(It->second);
    return true;
  });
  // If there are none left, we're done.
  if (HardenOpRegs.empty())
    return;

  // Compute the current predicate state.
  Register StateReg = PS->SSA.GetValueAtEndOfBlock(&MBB);

  auto InsertPt = MI.getIterator();

  // If EFLAGS are live and we don't have access to instructions that avoid
  // clobbering EFLAGS we need to save and restore them. This in turn makes
  // the EFLAGS no longer live.
  /*
  unsigned FlagsReg = 0;
  if (EFLAGSLive && !Subtarget->hasBMI2()) {
    EFLAGSLive = false;
    FlagsReg = saveEFLAGS(MBB, InsertPt, Loc);
  }
  */
  for (MachineOperand *Op : HardenOpRegs) {
    Register OpReg = Op->getReg();
    auto *OpRC = MRI->getRegClass(OpReg);
    Register TmpReg = MRI->createVirtualRegister(OpRC);
    // Merge our potential poison state into the value with an or.
    auto OrI = BuildMI(MBB, InsertPt, Loc, TII->get(RISCV::OR), TmpReg)
                    .addReg(StateReg)
                    .addReg(OpReg);
    
    //OrI->addRegisterDead(X86::EFLAGS, TRI);
    ++NumInstsInserted;
    LLVM_DEBUG(dbgs() << "  Inserting or: "; OrI->dump(); dbgs() << "\n");
    
    // Record this register as checked and update the operand.
    assert(!AddrRegToHardenedReg.count(Op->getReg()) &&
           "Should not have checked this register yet!");
    AddrRegToHardenedReg[Op->getReg()] = TmpReg;
    Op->setReg(TmpReg);
    ++NumAddrRegsHardened;
  }
  // And restore the flags if needed.
  //if (FlagsReg)
  //  restoreEFLAGS(MBB, InsertPt, Loc, FlagsReg);
}


/// Harden a return instruction.
///
/// Returns implicitly perform a load which we need to harden. Without hardening
/// this load, an attacker my speculatively write over the return address to
/// steer speculation of the return to an attacker controlled address. This is
/// called Spectre v1.1 or Bounds Check Bypass Store (BCBS) and is described in
/// this paper:
/// https://people.csail.mit.edu/vlk/spectre11.pdf
///
/// We can harden this by introducing an LFENCE that will delay any load of the
/// return address until prior instructions have retired (and thus are not being
/// speculated), or we can harden the address used by the implicit load: the
/// stack pointer.
///
/// If we are not using an LFENCE, hardening the stack pointer has an additional
/// benefit: it allows us to pass the predicate state accumulated in this
/// function back to the caller. In the absence of a BCBS attack on the return,
/// the caller will typically be resumed and speculatively executed due to the
/// Return Stack Buffer (RSB) prediction which is very accurate and has a high
/// priority. It is possible that some code from the caller will be executed
/// speculatively even during a BCBS-attacked return until the steering takes
/// effect. Whenever this happens, the caller can recover the (poisoned)
/// predicate state from the stack pointer and continue to harden loads.
void RISCVRivosSpectreShieldPass::hardenReturnInstr(MachineInstr &MI) {
  MachineBasicBlock &MBB = *MI.getParent();
  const DebugLoc &Loc = MI.getDebugLoc();
  auto InsertPt = MI.getIterator();

  if (FenceCallAndRet)
    // No need to fence here as we'll fence at the return site itself. That
    // handles more cases than we can handle here.
    return;

  // Take our predicate state, shift it to the high 17 bits (so that we keep
  // pointers canonical) and merge it into RSP. This will allow the caller to
  // extract it when we return (speculatively).
  mergePredStateIntoSP(MBB, InsertPt, Loc, PS->SSA.GetValueAtEndOfBlock(&MBB));
}

/// Takes the current predicate state (in a register) and merges it into the
/// stack pointer. The state is essentially a single bit, but we merge this in
/// a way that won't form non-canonical pointers and also will be preserved
/// across normal stack adjustments.
void RISCVRivosSpectreShieldPass::mergePredStateIntoSP(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator InsertPt,
    const DebugLoc &Loc, unsigned PredStateReg) {
  Register TmpReg = MRI->createVirtualRegister(PS->RC);
  // FIXME: This hard codes a shift distance based on the number of bits needed
  // to stay canonical on 64-bit. We should compute this somehow and support
  // 32-bit as part of that.
  auto ShiftI = BuildMI(MBB, InsertPt, Loc, TII->get(RISCV::SLLIW), TmpReg)
                    .addReg(PredStateReg, RegState::Kill)
                    .addImm(47);
  //ShiftI->addRegisterDead(X86::EFLAGS, TRI);
  ++NumInstsInserted;
  auto OrI = BuildMI(MBB, InsertPt, Loc, TII->get(RISCV::OR), RISCV::X2)
                 .addReg(RISCV::X2)
                 .addReg(TmpReg, RegState::Kill);
  //OrI->addRegisterDead(X86::EFLAGS, TRI);
  ++NumInstsInserted;
}

/// Extracts the predicate state stored in the high bits of the stack pointer.
unsigned RISCVRivosSpectreShieldPass::extractPredStateFromSP(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator InsertPt,
    const DebugLoc &Loc) {
  Register PredStateReg = MRI->createVirtualRegister(PS->RC);
  Register TmpReg = MRI->createVirtualRegister(PS->RC);

  // We know that the stack pointer will have any preserved predicate state in
  // its high bit. We just want to smear this across the other bits. Turns out,
  // this is exactly what an arithmetic right shift does.
  BuildMI(MBB, InsertPt, Loc, TII->get(RISCV::ADDI), TmpReg)
      .addReg(RISCV::X2)
      .addImm(0);
  auto ShiftI =
      BuildMI(MBB, InsertPt, Loc, TII->get(RISCV::SRAIW), PredStateReg)
          .addReg(TmpReg, RegState::Kill)
          .addImm(TRI->getRegSizeInBits(*PS->RC) - 1);
  //ShiftI->addRegisterDead(X86::EFLAGS, TRI);
  ++NumInstsInserted;

  return PredStateReg;
}

/// Trace the predicate state through a call.
///
/// There are several layers of this needed to handle the full complexity of
/// calls.
///
/// First, we need to send the predicate state into the called function. We do
/// this by merging it into the high bits of the stack pointer.
///
/// For tail calls, this is all we need to do.
///
/// For calls where we might return and resume the control flow, we need to
/// extract the predicate state from the high bits of the stack pointer after
/// control returns from the called function.
///
/// We also need to verify that we intended to return to this location in the
/// code. An attacker might arrange for the processor to mispredict the return
/// to this valid but incorrect return address in the program rather than the
/// correct one. See the paper on this attack, called "ret2spec" by the
/// researchers, here:
/// https://christian-rossow.de/publications/ret2spec-ccs2018.pdf
///
/// The way we verify that we returned to the correct location is by preserving
/// the expected return address across the call. One technique involves taking
/// advantage of the red-zone to load the return address from `8(%rsp)` where it
/// was left by the RET instruction when it popped `%rsp`. Alternatively, we can
/// directly save the address into a register that will be preserved across the
/// call. We compare this intended return address against the address
/// immediately following the call (the observed return address). If these
/// mismatch, we have detected misspeculation and can poison our predicate
/// state.
void RISCVRivosSpectreShieldPass::tracePredStateThroughCall(
    MachineInstr &MI) {
  MachineBasicBlock &MBB = *MI.getParent();
  MachineFunction &MF = *MBB.getParent();
  auto InsertPt = MI.getIterator();
  const DebugLoc &Loc = MI.getDebugLoc();

  if (FenceCallAndRet) {
    if (MI.isReturn())
      // Tail call, we don't return to this function.
      // FIXME: We should also handle noreturn calls.
      return;

    // We don't need to fence before the call because the function should fence
    // in its entry. However, we do need to fence after the call returns.
    // Fencing before the return doesn't correctly handle cases where the return
    // itself is mispredicted.
    BuildMI(MBB, std::next(InsertPt), Loc, TII->get(RISCV::FENCE));
    ++NumInstsInserted;
    ++NumLFENCEsInserted;
    return;
  }

  // First, we transfer the predicate state into the called function by merging
  // it into the stack pointer. This will kill the current def of the state.
  Register StateReg = PS->SSA.GetValueAtEndOfBlock(&MBB);
  mergePredStateIntoSP(MBB, InsertPt, Loc, StateReg);

  // If this call is also a return, it is a tail call and we don't need anything
  // else to handle it so just return. Also, if there are no further
  // instructions and no successors, this call does not return so we can also
  // bail.
  if (MI.isReturn() || (std::next(InsertPt) == MBB.end() && MBB.succ_empty()))
    return;

  // Create a symbol to track the return address and attach it to the call
  // machine instruction. We will lower extra symbols attached to call
  // instructions as label immediately following the call.
  MCSymbol *RetSymbol =
      MF.getContext().createTempSymbol("slh_ret_addr",
                                       /*AlwaysAddSuffix*/ true);
  MI.setPostInstrSymbol(MF, RetSymbol);

  const TargetRegisterClass *AddrRC = &RISCV::GPRRegClass;
  unsigned ExpectedRetAddrReg = 0;

  
  // If we have no red zones or if the function returns twice (possibly without
  // using the `ret` instruction) like setjmp, we need to save the expected
  // return address prior to the call.
  
  // If we don't have red zones, we need to compute the expected return
  // address prior to the call and store it in a register that lives across
  // the call.
  //
  // In some ways, this is doubly satisfying as a mitigation because it will
  // also successfully detect stack smashing bugs in some cases (typically,
  // when a callee-saved register is used and the callee doesn't push it onto
  // the stack). But that isn't our primary goal, so we only use it as
  // a fallback.
  //
  // FIXME: It isn't clear that this is reliable in the face of
  // rematerialization in the register allocator. We somehow need to force
  // that to not occur for this particular instruction, and instead to spill
  // or otherwise preserve the value computed *prior* to the call.
  //
  // FIXME: It is even less clear why MachineCSE can't just fold this when we
  // end up having to use identical instructions both before and after the
  // call to feed the comparison.
  ExpectedRetAddrReg = MRI->createVirtualRegister(AddrRC);
  

  const MCObjectFileInfo *MOFI = MF.getContext().getObjectFileInfo();
  if (MOFI->isPositionIndependent()){
    BuildMI(MBB, InsertPt, Loc, TII->get(RISCV::PseudoLGA), ExpectedRetAddrReg)
        .addSym(RetSymbol, RISCVII::MO_GOT_HI); 
  }
  else{
    BuildMI(MBB, InsertPt, Loc, TII->get(RISCV::PseudoLLA), ExpectedRetAddrReg)
        .addSym(RetSymbol, RISCVII::MO_GOT_HI); 
  }

  Register CurrentPCReg = MRI->createVirtualRegister(AddrRC);
  
  
 
  // Step past the call to handle when it returns.
  ++InsertPt;

  /*
  // If we didn't pre-compute the expected return address into a register, then
  // red zones are enabled and the return address is still available on the
  // stack immediately after the call. As the very first instruction, we load it
  // into a register.
  if (!ExpectedRetAddrReg) {
    ExpectedRetAddrReg = MRI->createVirtualRegister(AddrRC);
    BuildMI(MBB, InsertPt, Loc, TII->get(X86::MOV64rm), ExpectedRetAddrReg)
        .addReg(X86::RSP //Base)
        .addImm(1 //Scale)
        .addReg(0 //Index)
        .addImm(-8 //Displacement) // The stack pointer has been popped, so
                                     // the return address is 8-bytes past it.
        .addReg(0 //Segment);
  }
  */

  // Now we extract the callee's predicate state from the stack pointer.
  unsigned NewStateReg = extractPredStateFromSP(MBB, InsertPt, Loc);

  // Test the expected return address against our actual address. If we can
  // form this basic block's address as an immediate, this is easy. Otherwise
  // we compute it.
  Register XORReg = MRI->createVirtualRegister(AddrRC);
  Register SLTReg = MRI->createVirtualRegister(AddrRC);
  Register NegReg = MRI->createVirtualRegister(AddrRC);
  Register ActualRetAddrReg = MRI->createVirtualRegister(AddrRC);
  if (MOFI->isPositionIndependent()){
    BuildMI(MBB, InsertPt, Loc, TII->get(RISCV::PseudoLGA), ActualRetAddrReg)
        .addSym(RetSymbol, RISCVII::MO_GOT_HI); 
  }
  else{
    BuildMI(MBB, InsertPt, Loc, TII->get(RISCV::PseudoLLA), ActualRetAddrReg)
        .addSym(RetSymbol, RISCVII::MO_GOT_HI); 
  }
  BuildMI(MBB, InsertPt, Loc, TII->get(RISCV::XOR), XORReg)
        .addReg(ExpectedRetAddrReg, RegState::Kill)
        .addReg(ActualRetAddrReg, RegState::Kill);
  BuildMI(MBB, InsertPt, Loc, TII->get(RISCV::SLTU), SLTReg)
                        .addReg(RISCV::X0)
                        .addReg(XORReg);
  BuildMI(MBB, InsertPt, Loc, TII->get(RISCV::SUBW), NegReg)
                        .addReg(RISCV::X0)
                        .addReg(SLTReg);                         
  /*
  if (MF.getTarget().getCodeModel() == CodeModel::Small &&
      !Subtarget->isPositionIndependent()) {
    // FIXME: Could we fold this with the load? It would require careful EFLAGS
    // management.
    BuildMI(MBB, InsertPt, Loc, TII->get(X86::CMP64ri32))
        .addReg(ExpectedRetAddrReg, RegState::Kill)
        .addSym(RetSymbol);
  } else {
    Register ActualRetAddrReg = MRI->createVirtualRegister(AddrRC);
    BuildMI(MBB, InsertPt, Loc, TII->get(X86::LEA64r), ActualRetAddrReg)
        .addReg(X86::RIP)
        .addImm(1)
        .addReg(0)
        .addSym(RetSymbol)
        .addReg(0);
    BuildMI(MBB, InsertPt, Loc, TII->get(X86::CMP64rr))
        .addReg(ExpectedRetAddrReg, RegState::Kill)
        .addReg(ActualRetAddrReg, RegState::Kill);
  }
  */
 
  // Now conditionally update the predicate state we just extracted if we ended
  // up at a different return address than expected.
  Register UpdatedStateReg = MRI->createVirtualRegister(PS->RC);
  BuildMI(MBB, InsertPt, Loc, TII->get(RISCV::OR), UpdatedStateReg)
                        .addReg(NewStateReg, RegState::Kill)
                        .addReg(NegReg); 
  ++NumInstsInserted;
  /*
  int PredStateSizeInBytes = TRI->getRegSizeInBits(*PS->RC) / 8;
  auto CMovOp = X86::getCMovOpcode(PredStateSizeInBytes);

  Register UpdatedStateReg = MRI->createVirtualRegister(PS->RC);
  auto CMovI = BuildMI(MBB, InsertPt, Loc, TII->get(CMovOp), UpdatedStateReg)
                   .addReg(NewStateReg, RegState::Kill)
                   .addReg(PS->PoisonReg)
                   .addImm(X86::COND_NE);
  CMovI->findRegisterUseOperand(X86::EFLAGS)->setIsKill(true);
  ++NumInstsInserted;
  LLVM_DEBUG(dbgs() << "  Inserting cmov: "; CMovI->dump(); dbgs() << "\n");
  */
  PS->SSA.AddAvailableValue(&MBB, UpdatedStateReg);
}

/// Trace the predicate state through indirect branches, instrumenting them to
/// poison the state if a target is reached that does not match the expected
/// target.
///
/// This is designed to mitigate Spectre variant 1 attacks where an indirect
/// branch is trained to predict a particular target and then mispredicts that
/// target in a way that can leak data. Despite using an indirect branch, this
/// is really a variant 1 style attack: it does not steer execution to an
/// arbitrary or attacker controlled address, and it does not require any
/// special code executing next to the victim. This attack can also be mitigated
/// through retpolines, but those require either replacing indirect branches
/// with conditional direct branches or lowering them through a device that
/// blocks speculation. This mitigation can replace these retpoline-style
/// mitigations for jump tables and other indirect branches within a function
/// when variant 2 isn't a risk while allowing limited speculation. Indirect
/// calls, however, cannot be mitigated through this technique without changing
/// the ABI in a fundamental way.
SmallVector<MachineInstr *, 16>
RISCVRivosSpectreShieldPass::tracePredStateThroughIndirectBranches(
    MachineFunction &MF) {
  // We use the SSAUpdater to insert PHI nodes for the target addresses of
  // indirect branches. We don't actually need the full power of the SSA updater
  // in this particular case as we always have immediately available values, but
  // this avoids us having to re-implement the PHI construction logic.
  MachineSSAUpdater TargetAddrSSA(MF);
  TargetAddrSSA.Initialize(MRI->createVirtualRegister(&RISCV::GPRRegClass));

  // Track which blocks were terminated with an indirect branch.
  SmallPtrSet<MachineBasicBlock *, 4> IndirectTerminatedMBBs;

  // We need to know what blocks end up reached via indirect branches. We
  // expect this to be a subset of those whose address is taken and so track it
  // directly via the CFG.
  SmallPtrSet<MachineBasicBlock *, 4> IndirectTargetMBBs;

  // Walk all the blocks which end in an indirect branch and make the
  // target address available.
  for (MachineBasicBlock &MBB : MF) {
    // Find the last terminator.
    auto MII = MBB.instr_rbegin();
    while (MII != MBB.instr_rend() && MII->isDebugInstr())
      ++MII;
    if (MII == MBB.instr_rend())
      continue;
    MachineInstr &TI = *MII;
    if (!TI.isTerminator() || !TI.isBranch())
      // No terminator or non-branch terminator.
      continue;

    unsigned TargetReg;
    
    /* switch (TI.getOpcode()) {
    default:
      // Direct branch or conditional branch (leading to fallthrough).
      continue;

    case X86::FARJMP16m:
    case X86::FARJMP32m:
    case X86::FARJMP64m:
      // We cannot mitigate far jumps or calls, but we also don't expect them
      // to be vulnerable to Spectre v1.2 or v2 (self trained) style attacks.
      continue;

    case X86::JMP16m:
    case X86::JMP16m_NT:
    case X86::JMP32m:
    case X86::JMP32m_NT:
    case X86::JMP64m:
    case X86::JMP64m_NT:
      // Mostly as documentation.
      report_fatal_error("Memory operand jumps should have been unfolded!");

    case X86::JMP16r:
      report_fatal_error(
          "Support for 16-bit indirect branches is not implemented.");
    case X86::JMP32r:
      report_fatal_error(
          "Support for 32-bit indirect branches is not implemented.");

    case X86::JMP64r:
      TargetReg = TI.getOperand(0).getReg();
    } */

    if (TI.isIndirectBranch()){
      TargetReg = TI.getOperand(1).getReg();
    }
    else {
      continue;
    }

    // We have definitely found an indirect  branch. Verify that there are no
    // preceding conditional branches as we don't yet support that.
    if (llvm::any_of(MBB.terminators(), [&](MachineInstr &OtherTI) {
          return !OtherTI.isDebugInstr() && &OtherTI != &TI;
        })) {
      LLVM_DEBUG({
        dbgs() << "ERROR: Found other terminators in a block with an indirect "
                  "branch! This is not yet supported! Terminator sequence:\n";
        for (MachineInstr &MI : MBB.terminators()) {
          MI.dump();
          dbgs() << '\n';
        }
      });
      report_fatal_error("Unimplemented terminator sequence!");
    }

    // Make the target register an available value for this block.
    TargetAddrSSA.AddAvailableValue(&MBB, TargetReg);
    IndirectTerminatedMBBs.insert(&MBB);

    // Add all the successors to our target candidates.
    for (MachineBasicBlock *Succ : MBB.successors())
      IndirectTargetMBBs.insert(Succ);
  }

  // Keep track of the cmov instructions we insert so we can return them.
  SmallVector<MachineInstr *, 16> CMovs;

  // If we didn't find any indirect branches with targets, nothing to do here.
  if (IndirectTargetMBBs.empty())
    return CMovs;

  // We found indirect branches and targets that need to be instrumented to
  // harden loads within them. Walk the blocks of the function (to get a stable
  // ordering) and instrument each target of an indirect branch.
  for (MachineBasicBlock &MBB : MF) {
    // Skip the blocks that aren't candidate targets.
    if (!IndirectTargetMBBs.count(&MBB))
      continue;

    // We don't expect EH pads to ever be reached via an indirect branch. If
    // this is desired for some reason, we could simply skip them here rather
    // than asserting.
    assert(!MBB.isEHPad() &&
           "Unexpected EH pad as target of an indirect branch!");

    // We should never end up threading EFLAGS into a block to harden
    // conditional jumps as there would be an additional successor via the
    // indirect branch. As a consequence, all such edges would be split before
    // reaching here, and the inserted block will handle the EFLAGS-based
    // hardening.
    /* assert(!MBB.isLiveIn(X86::EFLAGS) &&
           "Cannot check within a block that already has live-in EFLAGS!"); */

    // We can't handle having non-indirect edges into this block unless this is
    // the only successor and we can synthesize the necessary target address.
    for (MachineBasicBlock *Pred : MBB.predecessors()) {
      // If we've already handled this by extracting the target directly,
      // nothing to do.
      if (IndirectTerminatedMBBs.count(Pred))
        continue;

      // Otherwise, we have to be the only successor. We generally expect this
      // to be true as conditional branches should have had a critical edge
      // split already. We don't however need to worry about EH pad successors
      // as they'll happily ignore the target and their hardening strategy is
      // resilient to all ways in which they could be reached speculatively.
      if (!llvm::all_of(Pred->successors(), [&](MachineBasicBlock *Succ) {
            return Succ->isEHPad() || Succ == &MBB;
          })) {
        LLVM_DEBUG({
          dbgs() << "ERROR: Found conditional entry to target of indirect "
                    "branch!\n";
          Pred->dump();
          MBB.dump();
        });
        report_fatal_error("Cannot harden a conditional entry to a target of "
                           "an indirect branch!");
      }

      // Now we need to compute the address of this block and install it as a
      // synthetic target in the predecessor. We do this at the bottom of the
      // predecessor.
      auto InsertPt = Pred->getFirstTerminator();
      Register TargetReg = MRI->createVirtualRegister(&RISCV::GPRRegClass);
      const MCObjectFileInfo *MOFI = MF.getContext().getObjectFileInfo();
      if (MOFI->isPositionIndependent()){
        BuildMI(*Pred, InsertPt, DebugLoc(), TII->get(RISCV::PseudoLGA), TargetReg)
            .addMBB(&MBB, RISCVII::MO_GOT_HI); 
      }
      else{
        BuildMI(*Pred, InsertPt, DebugLoc(), TII->get(RISCV::PseudoLLA), TargetReg)
            .addMBB(&MBB, RISCVII::MO_GOT_HI); 
      }
      /* auto AddrI = BuildMI(*Pred, InsertPt, DebugLoc(),
                             TII->get(RISCV::PseudoLI), TargetReg)
                         .addMBB(&MBB); */
        ++NumInstsInserted;
        /* (void)AddrI;
        LLVM_DEBUG(dbgs() << "  Inserting mov: "; AddrI->dump();
                   dbgs() << "\n"); */

      /* if (MF.getTarget().getCodeModel() == CodeModel::Small &&
          !Subtarget->isPositionIndependent()) {
        // Directly materialize it into an immediate.
        auto AddrI = BuildMI(*Pred, InsertPt, DebugLoc(),
                             TII->get(X86::MOV64ri32), TargetReg)
                         .addMBB(&MBB);
        ++NumInstsInserted;
        (void)AddrI;
        LLVM_DEBUG(dbgs() << "  Inserting mov: "; AddrI->dump();
                   dbgs() << "\n");
      } else {
        auto AddrI = BuildMI(*Pred, InsertPt, DebugLoc(), TII->get(X86::LEA64r),
                             TargetReg)
                         .addReg(X86::RIP)
                         .addImm(1)
                         .addReg(0)
                         .addMBB(&MBB)
                         .addReg(0);
        ++NumInstsInserted;
        (void)AddrI;
        LLVM_DEBUG(dbgs() << "  Inserting lea: "; AddrI->dump();
                   dbgs() << "\n");
      } */

      // And make this available.
      TargetAddrSSA.AddAvailableValue(Pred, TargetReg);
    }

    // Materialize the needed SSA value of the target. Note that we need the
    // middle of the block as this block might at the bottom have an indirect
    // branch back to itself. We can do this here because at this point, every
    // predecessor of this block has an available value. This is basically just
    // automating the construction of a PHI node for this target.
    Register TargetReg = TargetAddrSSA.GetValueInMiddleOfBlock(&MBB);

    // Insert a comparison of the incoming target register with this block's
    // address. This also requires us to mark the block as having its address
    // taken explicitly.
    MBB.setMachineBlockAddressTaken();
    auto InsertPt = MBB.SkipPHIsLabelsAndDebug(MBB.begin());
    Register MBBAddrReg = MRI->createVirtualRegister(&RISCV::GPRRegClass);
    Register XORReg = MRI->createVirtualRegister(&RISCV::GPRRegClass);
    Register SLTReg = MRI->createVirtualRegister(&RISCV::GPRRegClass);
    Register NegReg = MRI->createVirtualRegister(&RISCV::GPRRegClass);
    const MCObjectFileInfo *MOFI = MF.getContext().getObjectFileInfo();
    if (MOFI->isPositionIndependent()){
      BuildMI(MBB, InsertPt, DebugLoc(), TII->get(RISCV::PseudoLGA), MBBAddrReg)
          .addMBB(&MBB, RISCVII::MO_GOT_HI); 
    }
    else{
      BuildMI(MBB, InsertPt, DebugLoc(), TII->get(RISCV::PseudoLLA), MBBAddrReg)
          .addMBB(&MBB, RISCVII::MO_GOT_HI); 
    }
    /* BuildMI(MBB, InsertPt, DebugLoc(), TII->get(RISCV::PseudoLI), MBBAddrReg)
            .addMBB(&MBB); */
    BuildMI(MBB, InsertPt, DebugLoc(), TII->get(RISCV::XOR), XORReg)
          .addReg(TargetReg, RegState::Kill)
          .addReg(MBBAddrReg, RegState::Kill);
    BuildMI(MBB, InsertPt, DebugLoc(), TII->get(RISCV::SLTU), SLTReg)
                          .addReg(RISCV::X0)
                          .addReg(XORReg);
    BuildMI(MBB, InsertPt, DebugLoc(), TII->get(RISCV::SUBW), NegReg)
                          .addReg(RISCV::X0)
                          .addReg(SLTReg);  
    ++NumInstsInserted;
    //(void)CheckI;
    //LLVM_DEBUG(dbgs() << "  Inserting cmp: "; CheckI->dump(); dbgs() << "\n");

    /* if (MF.getTarget().getCodeModel() == CodeModel::Small &&
        !Subtarget->isPositionIndependent()) {
      // Check directly against a relocated immediate when we can.
      auto CheckI = BuildMI(MBB, InsertPt, DebugLoc(), TII->get(X86::CMP64ri32))
                        .addReg(TargetReg, RegState::Kill)
                        .addMBB(&MBB);
      ++NumInstsInserted;
      (void)CheckI;
      LLVM_DEBUG(dbgs() << "  Inserting cmp: "; CheckI->dump(); dbgs() << "\n");
    } else {
      // Otherwise compute the address into a register first.
      Register AddrReg = MRI->createVirtualRegister(&X86::GR64RegClass);
      auto AddrI =
          BuildMI(MBB, InsertPt, DebugLoc(), TII->get(X86::LEA64r), AddrReg)
              .addReg(X86::RIP)
              .addImm(1)
              .addReg( 0)
              .addMBB(&MBB)
              .addReg(0);
      ++NumInstsInserted;
      (void)AddrI;
      LLVM_DEBUG(dbgs() << "  Inserting lea: "; AddrI->dump(); dbgs() << "\n");
      auto CheckI = BuildMI(MBB, InsertPt, DebugLoc(), TII->get(X86::CMP64rr))
                        .addReg(TargetReg, RegState::Kill)
                        .addReg(AddrReg, RegState::Kill);
      ++NumInstsInserted;
      (void)CheckI;
      LLVM_DEBUG(dbgs() << "  Inserting cmp: "; CheckI->dump(); dbgs() << "\n");
    } */

    // Now cmov over the predicate if the comparison wasn't equal.
    Register UpdatedStateReg = MRI->createVirtualRegister(PS->RC);
    auto CMovI = BuildMI(MBB, InsertPt, DebugLoc(), TII->get(RISCV::OR), UpdatedStateReg)
                          .addReg(PS->InitialReg)
                          .addReg(NegReg); 
    ++NumInstsInserted;
    /* int PredStateSizeInBytes = TRI->getRegSizeInBits(*PS->RC) / 8;
    auto CMovOp = X86::getCMovOpcode(PredStateSizeInBytes);
    Register UpdatedStateReg = MRI->createVirtualRegister(PS->RC);
    auto CMovI =
        BuildMI(MBB, InsertPt, DebugLoc(), TII->get(CMovOp), UpdatedStateReg)
            .addReg(PS->InitialReg)
            .addReg(PS->PoisonReg)
            .addImm(X86::COND_NE);
    CMovI->findRegisterUseOperand(X86::EFLAGS)->setIsKill(true);
    ++NumInstsInserted;
    LLVM_DEBUG(dbgs() << "  Inserting cmov: "; CMovI->dump(); dbgs() << "\n"); */
    CMovs.push_back(&*CMovI);

    // And put the new value into the available values for SSA form of our
    // predicate state.
    PS->SSA.AddAvailableValue(&MBB, UpdatedStateReg);
  }

  // Return all the newly inserted cmov instructions of the predicate state.
  return CMovs;
}

/// An attacker may speculatively store over a value that is then speculatively
/// loaded and used as the target of an indirect call or jump instruction. This
/// is called Spectre v1.2 or Bounds Check Bypass Store (BCBS) and is described
/// in this paper:
/// https://people.csail.mit.edu/vlk/spectre11.pdf
///
/// When this happens, the speculative execution of the call or jump will end up
/// being steered to this attacker controlled address. While most such loads
/// will be adequately hardened already, we want to ensure that they are
/// definitively treated as needing post-load hardening. While address hardening
/// is sufficient to prevent secret data from leaking to the attacker, it may
/// not be sufficient to prevent an attacker from steering speculative
/// execution. We forcibly unfolded all relevant loads above and so will always
/// have an opportunity to post-load harden here, we just need to scan for cases
/// not already flagged and add them.
void RISCVRivosSpectreShieldPass::hardenIndirectCallOrJumpInstr(
    MachineInstr &MI,
    SmallDenseMap<unsigned, unsigned, 32> &AddrRegToHardenedReg) {
  /* switch (MI.getOpcode()) {
  case X86::FARCALL16m:
  case X86::FARCALL32m:
  case X86::FARCALL64m:
  case X86::FARJMP16m:
  case X86::FARJMP32m:
  case X86::FARJMP64m:
    // We don't need to harden either far calls or far jumps as they are
    // safe from Spectre.
    return;

  default:
    break;
  } */

  // We should never see a loading instruction at this point, as those should
  // have been unfolded.
  assert(!MI.mayLoad() && "Found a lingering loading instruction!");

  // If the first operand isn't a register, this is a branch or call
  // instruction with an immediate operand which doesn't need to be hardened.
  if (!MI.getOperand(1).isReg())
    return;

  // For all of these, the target register is the second operand of the
  // instruction.
  auto &TargetOp = MI.getOperand(1);
  Register OldTargetReg = TargetOp.getReg();

  // Try to lookup a hardened version of this register. We retain a reference
  // here as we want to update the map to track any newly computed hardened
  // register.
  unsigned &HardenedTargetReg = AddrRegToHardenedReg[OldTargetReg];

  // If we don't have a hardened register yet, compute one. Otherwise, just use
  // the already hardened register.
  //
  // FIXME: It is a little suspect that we use partially hardened registers that
  // only feed addresses. The complexity of partial hardening with SHRX
  // continues to pile up. Should definitively measure its value and consider
  // eliminating it.
  if (!HardenedTargetReg)
    HardenedTargetReg = hardenValueInRegister(
        OldTargetReg, *MI.getParent(), MI.getIterator(), MI.getDebugLoc());

  // Set the target operand to the hardened register.
  TargetOp.setReg(HardenedTargetReg);

  ++NumCallsOrJumpsHardened;
}

/// Harden a value in a register.
///
/// This is the low-level logic to fully harden a value sitting in a register
/// against leaking during speculative execution.
///
/// Unlike hardening an address that is used by a load, this routine is required
/// to hide *all* incoming bits in the register.
///
/// `Reg` must be a virtual register. Currently, it is required to be a GPR no
/// larger than the predicate state register. FIXME: We should support vector
/// registers here by broadcasting the predicate state.
///
/// The new, hardened virtual register is returned. It will have the same
/// register class as `Reg`.
unsigned RISCVRivosSpectreShieldPass::hardenValueInRegister(
    Register Reg, MachineBasicBlock &MBB, MachineBasicBlock::iterator InsertPt,
    const DebugLoc &Loc) {
  //assert(canHardenRegister(Reg) && "Cannot harden this register!");
  assert(Reg.isVirtual() && "Cannot harden a physical register!");

  auto *RC = MRI->getRegClass(Reg);
  int Bytes = TRI->getRegSizeInBits(*RC) / 8;
  Register StateReg = PS->SSA.GetValueAtEndOfBlock(&MBB);
  assert((Bytes == 1 || Bytes == 2 || Bytes == 4 || Bytes == 8) &&
         "Unknown register size");

  // FIXME: Need to teach this about 32-bit mode.
  /* if (Bytes != 8) {
    unsigned SubRegImms[] = {X86::sub_8bit, X86::sub_16bit, X86::sub_32bit};
    unsigned SubRegImm = SubRegImms[Log2_32(Bytes)];
    Register NarrowStateReg = MRI->createVirtualRegister(RC);
    BuildMI(MBB, InsertPt, Loc, TII->get(TargetOpcode::COPY), NarrowStateReg)
        .addReg(StateReg, 0, SubRegImm);
    StateReg = NarrowStateReg;
  } */

  /* unsigned FlagsReg = 0;
  if (isEFLAGSLive(MBB, InsertPt, *TRI))
    FlagsReg = saveEFLAGS(MBB, InsertPt, Loc); */

  Register NewReg = MRI->createVirtualRegister(RC);
  //unsigned OrOpCodes[] = {X86::OR8rr, X86::OR16rr, X86::OR32rr, X86::OR64rr};
  //unsigned OrOpCode = OrOpCodes[Log2_32(Bytes)];
  auto OrI = BuildMI(MBB, InsertPt, Loc, TII->get(RISCV::OR), NewReg)
                 .addReg(StateReg)
                 .addReg(Reg);
  //OrI->addRegisterDead(X86::EFLAGS, TRI);
  ++NumInstsInserted;
  LLVM_DEBUG(dbgs() << "  Inserting or: "; OrI->dump(); dbgs() << "\n");

  /* if (FlagsReg)
    restoreEFLAGS(MBB, InsertPt, Loc, FlagsReg); */

  return NewReg;
}


INITIALIZE_PASS_BEGIN(RISCVRivosSpectreShieldPass, PASS_KEY,
                      "RISCV speculative load hardener", false, false)
INITIALIZE_PASS_END(RISCVRivosSpectreShieldPass, PASS_KEY,
                    "RISCV speculative load hardener", false, false)

FunctionPass *llvm::createRISCVRivosSpectreShieldPass() {
  return new RISCVRivosSpectreShieldPass();
}
