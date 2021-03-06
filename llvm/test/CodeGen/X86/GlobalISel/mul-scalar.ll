; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=x86_64-linux-gnu -global-isel -verify-machineinstrs < %s -o - | FileCheck %s --check-prefix=ALL --check-prefix=X64

;TODO: instruction selection not supported yet
;define i8 @test_mul_i8(i8 %arg1, i8 %arg2) {
;  %ret = mul i8 %arg1, %arg2
;  ret i8 %ret
;}

define i16 @test_mul_i16(i16 %arg1, i16 %arg2) {
; ALL-LABEL: test_mul_i16:
; ALL:       # %bb.0:
; ALL-NEXT:    movl %esi, %eax
; ALL-NEXT:    imulw %di, %ax
; ALL-NEXT:    # kill: def $ax killed $ax killed $eax
; ALL-NEXT:    retq
  %ret = mul i16 %arg1, %arg2
  ret i16 %ret
}

define i32 @test_mul_i32(i32 %arg1, i32 %arg2) {
; ALL-LABEL: test_mul_i32:
; ALL:       # %bb.0:
; ALL-NEXT:    movl %esi, %eax
; ALL-NEXT:    imull %edi, %eax
; ALL-NEXT:    retq
  %ret = mul i32 %arg1, %arg2
  ret i32 %ret
}

define i64 @test_mul_i64(i64 %arg1, i64 %arg2) {
; ALL-LABEL: test_mul_i64:
; ALL:       # %bb.0:
; ALL-NEXT:    movq %rsi, %rax
; ALL-NEXT:    imulq %rdi, %rax
; ALL-NEXT:    retq
  %ret = mul i64 %arg1, %arg2
  ret i64 %ret
}

