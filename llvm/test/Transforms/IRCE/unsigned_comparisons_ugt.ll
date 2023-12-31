; NOTE: Assertions have been autogenerated by utils/update_test_checks.py UTC_ARGS: --version 3
; RUN: opt -verify-loop-info -irce-print-changed-loops -passes=irce -S < %s 2>&1 | FileCheck %s
; RUN: opt -verify-loop-info -irce-print-changed-loops -passes='require<branch-prob>,irce' -S < %s 2>&1 | FileCheck %s

; CHECK: irce: in function test_01: constrained Loop at depth 1 containing: %loop<header><exiting>,%in.bounds<latch><exiting>
; CHECK: irce: in function test_02: constrained Loop at depth 1 containing: %loop<header><exiting>,%in.bounds<latch><exiting>
; CHECK: irce: in function test_03: constrained Loop at depth 1 containing: %loop<header><exiting>,%in.bounds<latch><exiting>
; CHECK: irce: in function test_04: constrained Loop at depth 1 containing: %loop<header><exiting>,%in.bounds<latch><exiting>
; CHECK-NOT: irce: in function test_05: constrained Loop at depth 1 containing: %loop<header><exiting>,%in.bounds<latch><exiting>
; CHECK: irce: in function test_06: constrained Loop at depth 1 containing: %loop<header><exiting>,%in.bounds<latch><exiting>

; UGT condition for increasing loop.
define void @test_01(ptr %arr, ptr %a_len_ptr) #0 {
; CHECK-LABEL: define void @test_01
; CHECK-SAME: (ptr [[ARR:%.*]], ptr [[A_LEN_PTR:%.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[EXIT_MAINLOOP_AT:%.*]] = load i32, ptr [[A_LEN_PTR]], align 4, !range [[RNG0:![0-9]+]]
; CHECK-NEXT:    [[TMP0:%.*]] = icmp ult i32 0, [[EXIT_MAINLOOP_AT]]
; CHECK-NEXT:    br i1 [[TMP0]], label [[LOOP_PREHEADER:%.*]], label [[MAIN_PSEUDO_EXIT:%.*]]
; CHECK:       loop.preheader:
; CHECK-NEXT:    br label [[LOOP:%.*]]
; CHECK:       loop:
; CHECK-NEXT:    [[IDX:%.*]] = phi i32 [ [[IDX_NEXT:%.*]], [[IN_BOUNDS:%.*]] ], [ 0, [[LOOP_PREHEADER]] ]
; CHECK-NEXT:    [[IDX_NEXT]] = add nuw nsw i32 [[IDX]], 1
; CHECK-NEXT:    [[ABC:%.*]] = icmp ult i32 [[IDX]], [[EXIT_MAINLOOP_AT]]
; CHECK-NEXT:    br i1 true, label [[IN_BOUNDS]], label [[OUT_OF_BOUNDS_LOOPEXIT1:%.*]]
; CHECK:       in.bounds:
; CHECK-NEXT:    [[ADDR:%.*]] = getelementptr i32, ptr [[ARR]], i32 [[IDX]]
; CHECK-NEXT:    store i32 0, ptr [[ADDR]], align 4
; CHECK-NEXT:    [[NEXT:%.*]] = icmp ugt i32 [[IDX_NEXT]], 100
; CHECK-NEXT:    [[TMP1:%.*]] = icmp ult i32 [[IDX_NEXT]], [[EXIT_MAINLOOP_AT]]
; CHECK-NEXT:    [[TMP2:%.*]] = xor i1 [[TMP1]], true
; CHECK-NEXT:    br i1 [[TMP2]], label [[MAIN_EXIT_SELECTOR:%.*]], label [[LOOP]]
; CHECK:       main.exit.selector:
; CHECK-NEXT:    [[IDX_NEXT_LCSSA:%.*]] = phi i32 [ [[IDX_NEXT]], [[IN_BOUNDS]] ]
; CHECK-NEXT:    [[TMP3:%.*]] = icmp ult i32 [[IDX_NEXT_LCSSA]], 101
; CHECK-NEXT:    br i1 [[TMP3]], label [[MAIN_PSEUDO_EXIT]], label [[EXIT:%.*]]
; CHECK:       main.pseudo.exit:
; CHECK-NEXT:    [[IDX_COPY:%.*]] = phi i32 [ 0, [[ENTRY:%.*]] ], [ [[IDX_NEXT_LCSSA]], [[MAIN_EXIT_SELECTOR]] ]
; CHECK-NEXT:    [[INDVAR_END:%.*]] = phi i32 [ 0, [[ENTRY]] ], [ [[IDX_NEXT_LCSSA]], [[MAIN_EXIT_SELECTOR]] ]
; CHECK-NEXT:    br label [[POSTLOOP:%.*]]
; CHECK:       out.of.bounds.loopexit:
; CHECK-NEXT:    br label [[OUT_OF_BOUNDS:%.*]]
; CHECK:       out.of.bounds.loopexit1:
; CHECK-NEXT:    br label [[OUT_OF_BOUNDS]]
; CHECK:       out.of.bounds:
; CHECK-NEXT:    ret void
; CHECK:       exit.loopexit:
; CHECK-NEXT:    br label [[EXIT]]
; CHECK:       exit:
; CHECK-NEXT:    ret void
; CHECK:       postloop:
; CHECK-NEXT:    br label [[LOOP_POSTLOOP:%.*]]
; CHECK:       loop.postloop:
; CHECK-NEXT:    [[IDX_POSTLOOP:%.*]] = phi i32 [ [[IDX_COPY]], [[POSTLOOP]] ], [ [[IDX_NEXT_POSTLOOP:%.*]], [[IN_BOUNDS_POSTLOOP:%.*]] ]
; CHECK-NEXT:    [[IDX_NEXT_POSTLOOP]] = add nuw nsw i32 [[IDX_POSTLOOP]], 1
; CHECK-NEXT:    [[ABC_POSTLOOP:%.*]] = icmp ult i32 [[IDX_POSTLOOP]], [[EXIT_MAINLOOP_AT]]
; CHECK-NEXT:    br i1 [[ABC_POSTLOOP]], label [[IN_BOUNDS_POSTLOOP]], label [[OUT_OF_BOUNDS_LOOPEXIT:%.*]]
; CHECK:       in.bounds.postloop:
; CHECK-NEXT:    [[ADDR_POSTLOOP:%.*]] = getelementptr i32, ptr [[ARR]], i32 [[IDX_POSTLOOP]]
; CHECK-NEXT:    store i32 0, ptr [[ADDR_POSTLOOP]], align 4
; CHECK-NEXT:    [[NEXT_POSTLOOP:%.*]] = icmp ugt i32 [[IDX_NEXT_POSTLOOP]], 100
; CHECK-NEXT:    br i1 [[NEXT_POSTLOOP]], label [[EXIT_LOOPEXIT:%.*]], label [[LOOP_POSTLOOP]], !llvm.loop [[LOOP1:![0-9]+]], !irce.loop.clone [[META6:![0-9]+]]
;

entry:
  %len = load i32, ptr %a_len_ptr, !range !0
  br label %loop

loop:
  %idx = phi i32 [ 0, %entry ], [ %idx.next, %in.bounds ]
  %idx.next = add nsw nuw i32 %idx, 1
  %abc = icmp ult i32 %idx, %len
  br i1 %abc, label %in.bounds, label %out.of.bounds

in.bounds:
  %addr = getelementptr i32, ptr %arr, i32 %idx
  store i32 0, ptr %addr
  %next = icmp ugt i32 %idx.next, 100
  br i1 %next, label %exit, label %loop

out.of.bounds:
  ret void

exit:
  ret void
}

; UGT condition for decreasing loop.
define void @test_02(ptr %arr, ptr %a_len_ptr) #0 {
; CHECK-LABEL: define void @test_02
; CHECK-SAME: (ptr [[ARR:%.*]], ptr [[A_LEN_PTR:%.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[LEN:%.*]] = load i32, ptr [[A_LEN_PTR]], align 4, !range [[RNG0]]
; CHECK-NEXT:    [[UMAX:%.*]] = call i32 @llvm.umax.i32(i32 [[LEN]], i32 1)
; CHECK-NEXT:    [[EXIT_PRELOOP_AT:%.*]] = add nsw i32 [[UMAX]], -1
; CHECK-NEXT:    [[TMP0:%.*]] = icmp ugt i32 100, [[EXIT_PRELOOP_AT]]
; CHECK-NEXT:    br i1 [[TMP0]], label [[LOOP_PRELOOP_PREHEADER:%.*]], label [[PRELOOP_PSEUDO_EXIT:%.*]]
; CHECK:       loop.preloop.preheader:
; CHECK-NEXT:    br label [[LOOP_PRELOOP:%.*]]
; CHECK:       mainloop:
; CHECK-NEXT:    br label [[LOOP:%.*]]
; CHECK:       loop:
; CHECK-NEXT:    [[IDX:%.*]] = phi i32 [ [[IDX_PRELOOP_COPY:%.*]], [[MAINLOOP:%.*]] ], [ [[IDX_NEXT:%.*]], [[IN_BOUNDS:%.*]] ]
; CHECK-NEXT:    [[IDX_NEXT]] = add i32 [[IDX]], -1
; CHECK-NEXT:    [[ABC:%.*]] = icmp ult i32 [[IDX]], [[LEN]]
; CHECK-NEXT:    br i1 true, label [[IN_BOUNDS]], label [[OUT_OF_BOUNDS_LOOPEXIT1:%.*]]
; CHECK:       in.bounds:
; CHECK-NEXT:    [[ADDR:%.*]] = getelementptr i32, ptr [[ARR]], i32 [[IDX]]
; CHECK-NEXT:    store i32 0, ptr [[ADDR]], align 4
; CHECK-NEXT:    [[NEXT:%.*]] = icmp ugt i32 [[IDX_NEXT]], 0
; CHECK-NEXT:    br i1 [[NEXT]], label [[LOOP]], label [[EXIT_LOOPEXIT:%.*]]
; CHECK:       out.of.bounds.loopexit:
; CHECK-NEXT:    br label [[OUT_OF_BOUNDS:%.*]]
; CHECK:       out.of.bounds.loopexit1:
; CHECK-NEXT:    br label [[OUT_OF_BOUNDS]]
; CHECK:       out.of.bounds:
; CHECK-NEXT:    ret void
; CHECK:       exit.loopexit:
; CHECK-NEXT:    br label [[EXIT:%.*]]
; CHECK:       exit:
; CHECK-NEXT:    ret void
; CHECK:       loop.preloop:
; CHECK-NEXT:    [[IDX_PRELOOP:%.*]] = phi i32 [ [[IDX_NEXT_PRELOOP:%.*]], [[IN_BOUNDS_PRELOOP:%.*]] ], [ 100, [[LOOP_PRELOOP_PREHEADER]] ]
; CHECK-NEXT:    [[IDX_NEXT_PRELOOP]] = add i32 [[IDX_PRELOOP]], -1
; CHECK-NEXT:    [[ABC_PRELOOP:%.*]] = icmp ult i32 [[IDX_PRELOOP]], [[LEN]]
; CHECK-NEXT:    br i1 [[ABC_PRELOOP]], label [[IN_BOUNDS_PRELOOP]], label [[OUT_OF_BOUNDS_LOOPEXIT:%.*]]
; CHECK:       in.bounds.preloop:
; CHECK-NEXT:    [[ADDR_PRELOOP:%.*]] = getelementptr i32, ptr [[ARR]], i32 [[IDX_PRELOOP]]
; CHECK-NEXT:    store i32 0, ptr [[ADDR_PRELOOP]], align 4
; CHECK-NEXT:    [[NEXT_PRELOOP:%.*]] = icmp ugt i32 [[IDX_NEXT_PRELOOP]], 0
; CHECK-NEXT:    [[TMP1:%.*]] = icmp ugt i32 [[IDX_NEXT_PRELOOP]], [[EXIT_PRELOOP_AT]]
; CHECK-NEXT:    br i1 [[TMP1]], label [[LOOP_PRELOOP]], label [[PRELOOP_EXIT_SELECTOR:%.*]], !llvm.loop [[LOOP7:![0-9]+]], !irce.loop.clone [[META6]]
; CHECK:       preloop.exit.selector:
; CHECK-NEXT:    [[IDX_NEXT_PRELOOP_LCSSA:%.*]] = phi i32 [ [[IDX_NEXT_PRELOOP]], [[IN_BOUNDS_PRELOOP]] ]
; CHECK-NEXT:    [[TMP2:%.*]] = icmp ugt i32 [[IDX_NEXT_PRELOOP_LCSSA]], 0
; CHECK-NEXT:    br i1 [[TMP2]], label [[PRELOOP_PSEUDO_EXIT]], label [[EXIT]]
; CHECK:       preloop.pseudo.exit:
; CHECK-NEXT:    [[IDX_PRELOOP_COPY]] = phi i32 [ 100, [[ENTRY:%.*]] ], [ [[IDX_NEXT_PRELOOP_LCSSA]], [[PRELOOP_EXIT_SELECTOR]] ]
; CHECK-NEXT:    [[INDVAR_END:%.*]] = phi i32 [ 100, [[ENTRY]] ], [ [[IDX_NEXT_PRELOOP_LCSSA]], [[PRELOOP_EXIT_SELECTOR]] ]
; CHECK-NEXT:    br label [[MAINLOOP]]
;

entry:
  %len = load i32, ptr %a_len_ptr, !range !0
  br label %loop

loop:
  %idx = phi i32 [ 100, %entry ], [ %idx.next, %in.bounds ]
  %idx.next = add i32 %idx, -1
  %abc = icmp ult i32 %idx, %len
  br i1 %abc, label %in.bounds, label %out.of.bounds

in.bounds:
  %addr = getelementptr i32, ptr %arr, i32 %idx
  store i32 0, ptr %addr
  %next = icmp ugt i32 %idx.next, 0
  br i1 %next, label %loop, label %exit

out.of.bounds:
  ret void

exit:
  ret void
}

; Check SINT_MAX + 1, test is similar to test_01.
define void @test_03(ptr %arr, ptr %a_len_ptr) #0 {
; CHECK-LABEL: define void @test_03
; CHECK-SAME: (ptr [[ARR:%.*]], ptr [[A_LEN_PTR:%.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[EXIT_MAINLOOP_AT:%.*]] = load i32, ptr [[A_LEN_PTR]], align 4, !range [[RNG0]]
; CHECK-NEXT:    [[TMP0:%.*]] = icmp ult i32 0, [[EXIT_MAINLOOP_AT]]
; CHECK-NEXT:    br i1 [[TMP0]], label [[LOOP_PREHEADER:%.*]], label [[MAIN_PSEUDO_EXIT:%.*]]
; CHECK:       loop.preheader:
; CHECK-NEXT:    br label [[LOOP:%.*]]
; CHECK:       loop:
; CHECK-NEXT:    [[IDX:%.*]] = phi i32 [ [[IDX_NEXT:%.*]], [[IN_BOUNDS:%.*]] ], [ 0, [[LOOP_PREHEADER]] ]
; CHECK-NEXT:    [[IDX_NEXT]] = add nuw nsw i32 [[IDX]], 1
; CHECK-NEXT:    [[ABC:%.*]] = icmp ult i32 [[IDX]], [[EXIT_MAINLOOP_AT]]
; CHECK-NEXT:    br i1 true, label [[IN_BOUNDS]], label [[OUT_OF_BOUNDS_LOOPEXIT1:%.*]]
; CHECK:       in.bounds:
; CHECK-NEXT:    [[ADDR:%.*]] = getelementptr i32, ptr [[ARR]], i32 [[IDX]]
; CHECK-NEXT:    store i32 0, ptr [[ADDR]], align 4
; CHECK-NEXT:    [[NEXT:%.*]] = icmp ugt i32 [[IDX_NEXT]], -2147483648
; CHECK-NEXT:    [[TMP1:%.*]] = icmp ult i32 [[IDX_NEXT]], [[EXIT_MAINLOOP_AT]]
; CHECK-NEXT:    [[TMP2:%.*]] = xor i1 [[TMP1]], true
; CHECK-NEXT:    br i1 [[TMP2]], label [[MAIN_EXIT_SELECTOR:%.*]], label [[LOOP]]
; CHECK:       main.exit.selector:
; CHECK-NEXT:    [[IDX_NEXT_LCSSA:%.*]] = phi i32 [ [[IDX_NEXT]], [[IN_BOUNDS]] ]
; CHECK-NEXT:    [[TMP3:%.*]] = icmp ult i32 [[IDX_NEXT_LCSSA]], -2147483647
; CHECK-NEXT:    br i1 [[TMP3]], label [[MAIN_PSEUDO_EXIT]], label [[EXIT:%.*]]
; CHECK:       main.pseudo.exit:
; CHECK-NEXT:    [[IDX_COPY:%.*]] = phi i32 [ 0, [[ENTRY:%.*]] ], [ [[IDX_NEXT_LCSSA]], [[MAIN_EXIT_SELECTOR]] ]
; CHECK-NEXT:    [[INDVAR_END:%.*]] = phi i32 [ 0, [[ENTRY]] ], [ [[IDX_NEXT_LCSSA]], [[MAIN_EXIT_SELECTOR]] ]
; CHECK-NEXT:    br label [[POSTLOOP:%.*]]
; CHECK:       out.of.bounds.loopexit:
; CHECK-NEXT:    br label [[OUT_OF_BOUNDS:%.*]]
; CHECK:       out.of.bounds.loopexit1:
; CHECK-NEXT:    br label [[OUT_OF_BOUNDS]]
; CHECK:       out.of.bounds:
; CHECK-NEXT:    ret void
; CHECK:       exit.loopexit:
; CHECK-NEXT:    br label [[EXIT]]
; CHECK:       exit:
; CHECK-NEXT:    ret void
; CHECK:       postloop:
; CHECK-NEXT:    br label [[LOOP_POSTLOOP:%.*]]
; CHECK:       loop.postloop:
; CHECK-NEXT:    [[IDX_POSTLOOP:%.*]] = phi i32 [ [[IDX_COPY]], [[POSTLOOP]] ], [ [[IDX_NEXT_POSTLOOP:%.*]], [[IN_BOUNDS_POSTLOOP:%.*]] ]
; CHECK-NEXT:    [[IDX_NEXT_POSTLOOP]] = add nuw nsw i32 [[IDX_POSTLOOP]], 1
; CHECK-NEXT:    [[ABC_POSTLOOP:%.*]] = icmp ult i32 [[IDX_POSTLOOP]], [[EXIT_MAINLOOP_AT]]
; CHECK-NEXT:    br i1 [[ABC_POSTLOOP]], label [[IN_BOUNDS_POSTLOOP]], label [[OUT_OF_BOUNDS_LOOPEXIT:%.*]]
; CHECK:       in.bounds.postloop:
; CHECK-NEXT:    [[ADDR_POSTLOOP:%.*]] = getelementptr i32, ptr [[ARR]], i32 [[IDX_POSTLOOP]]
; CHECK-NEXT:    store i32 0, ptr [[ADDR_POSTLOOP]], align 4
; CHECK-NEXT:    [[NEXT_POSTLOOP:%.*]] = icmp ugt i32 [[IDX_NEXT_POSTLOOP]], -2147483648
; CHECK-NEXT:    br i1 [[NEXT_POSTLOOP]], label [[EXIT_LOOPEXIT:%.*]], label [[LOOP_POSTLOOP]], !llvm.loop [[LOOP8:![0-9]+]], !irce.loop.clone [[META6]]
;

entry:
  %len = load i32, ptr %a_len_ptr, !range !0
  br label %loop

loop:
  %idx = phi i32 [ 0, %entry ], [ %idx.next, %in.bounds ]
  %idx.next = add nsw nuw i32 %idx, 1
  %abc = icmp ult i32 %idx, %len
  br i1 %abc, label %in.bounds, label %out.of.bounds

in.bounds:
  %addr = getelementptr i32, ptr %arr, i32 %idx
  store i32 0, ptr %addr
  %next = icmp ugt i32 %idx.next, 2147483648
  br i1 %next, label %exit, label %loop

out.of.bounds:
  ret void

exit:
  ret void
}

; Check SINT_MAX + 1, test is similar to test_02.
define void @test_04(ptr %arr, ptr %a_len_ptr) #0 {
; CHECK-LABEL: define void @test_04
; CHECK-SAME: (ptr [[ARR:%.*]], ptr [[A_LEN_PTR:%.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[LEN:%.*]] = load i32, ptr [[A_LEN_PTR]], align 4, !range [[RNG0]]
; CHECK-NEXT:    [[UMAX:%.*]] = call i32 @llvm.umax.i32(i32 [[LEN]], i32 1)
; CHECK-NEXT:    [[EXIT_PRELOOP_AT:%.*]] = add nsw i32 [[UMAX]], -1
; CHECK-NEXT:    [[TMP0:%.*]] = icmp ugt i32 -2147483648, [[EXIT_PRELOOP_AT]]
; CHECK-NEXT:    br i1 [[TMP0]], label [[LOOP_PRELOOP_PREHEADER:%.*]], label [[PRELOOP_PSEUDO_EXIT:%.*]]
; CHECK:       loop.preloop.preheader:
; CHECK-NEXT:    br label [[LOOP_PRELOOP:%.*]]
; CHECK:       mainloop:
; CHECK-NEXT:    br label [[LOOP:%.*]]
; CHECK:       loop:
; CHECK-NEXT:    [[IDX:%.*]] = phi i32 [ [[IDX_PRELOOP_COPY:%.*]], [[MAINLOOP:%.*]] ], [ [[IDX_NEXT:%.*]], [[IN_BOUNDS:%.*]] ]
; CHECK-NEXT:    [[IDX_NEXT]] = add i32 [[IDX]], -1
; CHECK-NEXT:    [[ABC:%.*]] = icmp ult i32 [[IDX]], [[LEN]]
; CHECK-NEXT:    br i1 true, label [[IN_BOUNDS]], label [[OUT_OF_BOUNDS_LOOPEXIT1:%.*]]
; CHECK:       in.bounds:
; CHECK-NEXT:    [[ADDR:%.*]] = getelementptr i32, ptr [[ARR]], i32 [[IDX]]
; CHECK-NEXT:    store i32 0, ptr [[ADDR]], align 4
; CHECK-NEXT:    [[NEXT:%.*]] = icmp ugt i32 [[IDX_NEXT]], 0
; CHECK-NEXT:    br i1 [[NEXT]], label [[LOOP]], label [[EXIT_LOOPEXIT:%.*]]
; CHECK:       out.of.bounds.loopexit:
; CHECK-NEXT:    br label [[OUT_OF_BOUNDS:%.*]]
; CHECK:       out.of.bounds.loopexit1:
; CHECK-NEXT:    br label [[OUT_OF_BOUNDS]]
; CHECK:       out.of.bounds:
; CHECK-NEXT:    ret void
; CHECK:       exit.loopexit:
; CHECK-NEXT:    br label [[EXIT:%.*]]
; CHECK:       exit:
; CHECK-NEXT:    ret void
; CHECK:       loop.preloop:
; CHECK-NEXT:    [[IDX_PRELOOP:%.*]] = phi i32 [ [[IDX_NEXT_PRELOOP:%.*]], [[IN_BOUNDS_PRELOOP:%.*]] ], [ -2147483648, [[LOOP_PRELOOP_PREHEADER]] ]
; CHECK-NEXT:    [[IDX_NEXT_PRELOOP]] = add i32 [[IDX_PRELOOP]], -1
; CHECK-NEXT:    [[ABC_PRELOOP:%.*]] = icmp ult i32 [[IDX_PRELOOP]], [[LEN]]
; CHECK-NEXT:    br i1 [[ABC_PRELOOP]], label [[IN_BOUNDS_PRELOOP]], label [[OUT_OF_BOUNDS_LOOPEXIT:%.*]]
; CHECK:       in.bounds.preloop:
; CHECK-NEXT:    [[ADDR_PRELOOP:%.*]] = getelementptr i32, ptr [[ARR]], i32 [[IDX_PRELOOP]]
; CHECK-NEXT:    store i32 0, ptr [[ADDR_PRELOOP]], align 4
; CHECK-NEXT:    [[NEXT_PRELOOP:%.*]] = icmp ugt i32 [[IDX_NEXT_PRELOOP]], 0
; CHECK-NEXT:    [[TMP1:%.*]] = icmp ugt i32 [[IDX_NEXT_PRELOOP]], [[EXIT_PRELOOP_AT]]
; CHECK-NEXT:    br i1 [[TMP1]], label [[LOOP_PRELOOP]], label [[PRELOOP_EXIT_SELECTOR:%.*]], !llvm.loop [[LOOP9:![0-9]+]], !irce.loop.clone [[META6]]
; CHECK:       preloop.exit.selector:
; CHECK-NEXT:    [[IDX_NEXT_PRELOOP_LCSSA:%.*]] = phi i32 [ [[IDX_NEXT_PRELOOP]], [[IN_BOUNDS_PRELOOP]] ]
; CHECK-NEXT:    [[TMP2:%.*]] = icmp ugt i32 [[IDX_NEXT_PRELOOP_LCSSA]], 0
; CHECK-NEXT:    br i1 [[TMP2]], label [[PRELOOP_PSEUDO_EXIT]], label [[EXIT]]
; CHECK:       preloop.pseudo.exit:
; CHECK-NEXT:    [[IDX_PRELOOP_COPY]] = phi i32 [ -2147483648, [[ENTRY:%.*]] ], [ [[IDX_NEXT_PRELOOP_LCSSA]], [[PRELOOP_EXIT_SELECTOR]] ]
; CHECK-NEXT:    [[INDVAR_END:%.*]] = phi i32 [ -2147483648, [[ENTRY]] ], [ [[IDX_NEXT_PRELOOP_LCSSA]], [[PRELOOP_EXIT_SELECTOR]] ]
; CHECK-NEXT:    br label [[MAINLOOP]]
;

entry:
  %len = load i32, ptr %a_len_ptr, !range !0
  br label %loop

loop:
  %idx = phi i32 [ 2147483648, %entry ], [ %idx.next, %in.bounds ]
  %idx.next = add i32 %idx, -1
  %abc = icmp ult i32 %idx, %len
  br i1 %abc, label %in.bounds, label %out.of.bounds

in.bounds:
  %addr = getelementptr i32, ptr %arr, i32 %idx
  store i32 0, ptr %addr
  %next = icmp ugt i32 %idx.next, 0
  br i1 %next, label %loop, label %exit

out.of.bounds:
  ret void

exit:
  ret void
}

; Increasing loop, UINT_MAX. Negative test: we cannot add 1 to UINT_MAX.
define void @test_05(ptr %arr, ptr %a_len_ptr) #0 {
; CHECK-LABEL: define void @test_05
; CHECK-SAME: (ptr [[ARR:%.*]], ptr [[A_LEN_PTR:%.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[LEN:%.*]] = load i32, ptr [[A_LEN_PTR]], align 4, !range [[RNG0]]
; CHECK-NEXT:    br label [[LOOP:%.*]]
; CHECK:       loop:
; CHECK-NEXT:    [[IDX:%.*]] = phi i32 [ 0, [[ENTRY:%.*]] ], [ [[IDX_NEXT:%.*]], [[IN_BOUNDS:%.*]] ]
; CHECK-NEXT:    [[IDX_NEXT]] = add nuw nsw i32 [[IDX]], 1
; CHECK-NEXT:    [[ABC:%.*]] = icmp ult i32 [[IDX]], [[LEN]]
; CHECK-NEXT:    br i1 [[ABC]], label [[IN_BOUNDS]], label [[OUT_OF_BOUNDS:%.*]]
; CHECK:       in.bounds:
; CHECK-NEXT:    [[ADDR:%.*]] = getelementptr i32, ptr [[ARR]], i32 [[IDX]]
; CHECK-NEXT:    store i32 0, ptr [[ADDR]], align 4
; CHECK-NEXT:    [[NEXT:%.*]] = icmp ugt i32 [[IDX_NEXT]], -1
; CHECK-NEXT:    br i1 [[NEXT]], label [[EXIT:%.*]], label [[LOOP]]
; CHECK:       out.of.bounds:
; CHECK-NEXT:    ret void
; CHECK:       exit:
; CHECK-NEXT:    ret void
;

entry:
  %len = load i32, ptr %a_len_ptr, !range !0
  br label %loop

loop:
  %idx = phi i32 [ 0, %entry ], [ %idx.next, %in.bounds ]
  %idx.next = add nsw nuw i32 %idx, 1
  %abc = icmp ult i32 %idx, %len
  br i1 %abc, label %in.bounds, label %out.of.bounds

in.bounds:
  %addr = getelementptr i32, ptr %arr, i32 %idx
  store i32 0, ptr %addr
  %next = icmp ugt i32 %idx.next, 4294967295
  br i1 %next, label %exit, label %loop

out.of.bounds:
  ret void

exit:
  ret void
}

; Decreasing loop, UINT_MAX. Positive test.
define void @test_06(ptr %arr, ptr %a_len_ptr) #0 {
; CHECK-LABEL: define void @test_06
; CHECK-SAME: (ptr [[ARR:%.*]], ptr [[A_LEN_PTR:%.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[LEN:%.*]] = load i32, ptr [[A_LEN_PTR]], align 4, !range [[RNG0]]
; CHECK-NEXT:    br i1 true, label [[LOOP_PRELOOP_PREHEADER:%.*]], label [[PRELOOP_PSEUDO_EXIT:%.*]]
; CHECK:       loop.preloop.preheader:
; CHECK-NEXT:    br label [[LOOP_PRELOOP:%.*]]
; CHECK:       mainloop:
; CHECK-NEXT:    br label [[LOOP:%.*]]
; CHECK:       loop:
; CHECK-NEXT:    [[IDX:%.*]] = phi i32 [ [[IDX_PRELOOP_COPY:%.*]], [[MAINLOOP:%.*]] ], [ [[IDX_NEXT:%.*]], [[IN_BOUNDS:%.*]] ]
; CHECK-NEXT:    [[IDX_NEXT]] = add nuw i32 [[IDX]], -1
; CHECK-NEXT:    [[ABC:%.*]] = icmp ult i32 [[IDX]], [[LEN]]
; CHECK-NEXT:    br i1 true, label [[IN_BOUNDS]], label [[OUT_OF_BOUNDS_LOOPEXIT1:%.*]]
; CHECK:       in.bounds:
; CHECK-NEXT:    [[ADDR:%.*]] = getelementptr i32, ptr [[ARR]], i32 [[IDX]]
; CHECK-NEXT:    store i32 0, ptr [[ADDR]], align 4
; CHECK-NEXT:    [[NEXT:%.*]] = icmp ugt i32 [[IDX_NEXT]], 0
; CHECK-NEXT:    br i1 [[NEXT]], label [[LOOP]], label [[EXIT_LOOPEXIT:%.*]]
; CHECK:       out.of.bounds.loopexit:
; CHECK-NEXT:    br label [[OUT_OF_BOUNDS:%.*]]
; CHECK:       out.of.bounds.loopexit1:
; CHECK-NEXT:    br label [[OUT_OF_BOUNDS]]
; CHECK:       out.of.bounds:
; CHECK-NEXT:    ret void
; CHECK:       exit.loopexit:
; CHECK-NEXT:    br label [[EXIT:%.*]]
; CHECK:       exit:
; CHECK-NEXT:    ret void
; CHECK:       loop.preloop:
; CHECK-NEXT:    [[IDX_PRELOOP:%.*]] = phi i32 [ [[IDX_NEXT_PRELOOP:%.*]], [[IN_BOUNDS_PRELOOP:%.*]] ], [ -1, [[LOOP_PRELOOP_PREHEADER]] ]
; CHECK-NEXT:    [[IDX_NEXT_PRELOOP]] = add nuw i32 [[IDX_PRELOOP]], -1
; CHECK-NEXT:    [[ABC_PRELOOP:%.*]] = icmp ult i32 [[IDX_PRELOOP]], [[LEN]]
; CHECK-NEXT:    br i1 [[ABC_PRELOOP]], label [[IN_BOUNDS_PRELOOP]], label [[OUT_OF_BOUNDS_LOOPEXIT:%.*]]
; CHECK:       in.bounds.preloop:
; CHECK-NEXT:    [[ADDR_PRELOOP:%.*]] = getelementptr i32, ptr [[ARR]], i32 [[IDX_PRELOOP]]
; CHECK-NEXT:    store i32 0, ptr [[ADDR_PRELOOP]], align 4
; CHECK-NEXT:    [[NEXT_PRELOOP:%.*]] = icmp ugt i32 [[IDX_NEXT_PRELOOP]], 0
; CHECK-NEXT:    [[TMP0:%.*]] = icmp ugt i32 [[IDX_NEXT_PRELOOP]], 0
; CHECK-NEXT:    br i1 [[TMP0]], label [[LOOP_PRELOOP]], label [[PRELOOP_EXIT_SELECTOR:%.*]], !llvm.loop [[LOOP10:![0-9]+]], !irce.loop.clone [[META6]]
; CHECK:       preloop.exit.selector:
; CHECK-NEXT:    [[IDX_NEXT_PRELOOP_LCSSA:%.*]] = phi i32 [ [[IDX_NEXT_PRELOOP]], [[IN_BOUNDS_PRELOOP]] ]
; CHECK-NEXT:    [[TMP1:%.*]] = icmp ugt i32 [[IDX_NEXT_PRELOOP_LCSSA]], 0
; CHECK-NEXT:    br i1 [[TMP1]], label [[PRELOOP_PSEUDO_EXIT]], label [[EXIT]]
; CHECK:       preloop.pseudo.exit:
; CHECK-NEXT:    [[IDX_PRELOOP_COPY]] = phi i32 [ -1, [[ENTRY:%.*]] ], [ [[IDX_NEXT_PRELOOP_LCSSA]], [[PRELOOP_EXIT_SELECTOR]] ]
; CHECK-NEXT:    [[INDVAR_END:%.*]] = phi i32 [ -1, [[ENTRY]] ], [ [[IDX_NEXT_PRELOOP_LCSSA]], [[PRELOOP_EXIT_SELECTOR]] ]
; CHECK-NEXT:    br label [[MAINLOOP]]
;

entry:
  %len = load i32, ptr %a_len_ptr, !range !0
  br label %loop

loop:
  %idx = phi i32 [ 4294967295, %entry ], [ %idx.next, %in.bounds ]
  %idx.next = add nuw i32 %idx, -1
  %abc = icmp ult i32 %idx, %len
  br i1 %abc, label %in.bounds, label %out.of.bounds

in.bounds:
  %addr = getelementptr i32, ptr %arr, i32 %idx
  store i32 0, ptr %addr
  %next = icmp ugt i32 %idx.next, 0
  br i1 %next, label %loop, label %exit

out.of.bounds:
  ret void

exit:
  ret void
}

!0 = !{i32 0, i32 50}
;.
; CHECK: [[RNG0]] = !{i32 0, i32 50}
; CHECK: [[LOOP1]] = distinct !{[[LOOP1]], [[META2:![0-9]+]], [[META3:![0-9]+]], [[META4:![0-9]+]], [[META5:![0-9]+]]}
; CHECK: [[META2]] = !{!"llvm.loop.unroll.disable"}
; CHECK: [[META3]] = !{!"llvm.loop.vectorize.enable", i1 false}
; CHECK: [[META4]] = !{!"llvm.loop.licm_versioning.disable"}
; CHECK: [[META5]] = !{!"llvm.loop.distribute.enable", i1 false}
; CHECK: [[META6]] = !{}
; CHECK: [[LOOP7]] = distinct !{[[LOOP7]], [[META2]], [[META3]], [[META4]], [[META5]]}
; CHECK: [[LOOP8]] = distinct !{[[LOOP8]], [[META2]], [[META3]], [[META4]], [[META5]]}
; CHECK: [[LOOP9]] = distinct !{[[LOOP9]], [[META2]], [[META3]], [[META4]], [[META5]]}
; CHECK: [[LOOP10]] = distinct !{[[LOOP10]], [[META2]], [[META3]], [[META4]], [[META5]]}
;.
