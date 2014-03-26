/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2014 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

%name Select
%namespace_begin { NAMESPACE_BEGIN }
%namespace_end { NAMESPACE_END }
%extra_argument { X64Emitter* e }
%start_symbol function
%token_prefix SEL_

%include {
#include <alloy/backend/x64/x64_selector_util.h>
#include <alloy/hir/instr.h>
#include <alloy/hir/value.h>

using namespace alloy::hir;
using namespace alloy::runtime;

#define LOG_OPCODE(name) fprintf(stdout, "%s\n", #name); fflush(stdout);
}

function ::= function_begin block_list function_end .

%type function_begin {HIRBuilder*}
function_begin ::= FUNCTION_BEGIN . {
  fprintf(stdout, "FUNCTION_BEGIN\n");
  fflush(stdout);
}
%type function_end {HIRBuilder*}
function_end ::= FUNCTION_END . {
  fprintf(stdout, "FUNCTION_END\n");
  fflush(stdout);
}

block_list ::= block_list block .
block_list ::= .

block ::= block_begin instr_list block_end .

%type block_begin {Block*}
block_begin ::= BLOCK_BEGIN . {
  fprintf(stdout, "BLOCK_BEGIN\n");
  fflush(stdout);
}
%type block_end {Block*}
block_end ::= BLOCK_END . {
  fprintf(stdout, "BLOCK_END\n");
  fflush(stdout);
}

instr_list ::= instr_list instr .
instr_list ::= .

%type flags {uint32_t}
flags(D) ::= FLAGS(S) .                     { D = (uint32_t)S; }

%type label {Label*}
label(D) ::= LABEL(S) .                     { D = (Label*)S; }

%type symbol_info {FunctionInfo*}
symbol_info(D) ::= SYMBOL_INFO(S) .         { D = (FunctionInfo*)S; }

%type offset {uint64_t}
offset(D) ::= OFFSET(S) .                   { D = (uint64_t)S; }

%type const_i8 {Value*}
const_i8_0(D) ::= CONST_I8_0(S) .           { D = (Value*)S; }
const_i8_1(D) ::= CONST_I8_1(S) .           { D = (Value*)S; }
const_i8_F(D) ::= CONST_I8_F(S) .           { D = (Value*)S; }
const_i8(D) ::= CONST_I8(S) .               { D = (Value*)S; }
const_i8(D) ::= const_i8_0(S) .             { D = (Value*)S; }
const_i8(D) ::= const_i8_1(S) .             { D = (Value*)S; }
const_i8(D) ::= const_i8_F(S) .             { D = (Value*)S; }

%type const_i16 {Value*}
const_i16_0(D) ::= CONST_I16_0(S) .         { D = (Value*)S; }
const_i16_1(D) ::= CONST_I16_1(S) .         { D = (Value*)S; }
const_i16_F(D) ::= CONST_I16_F(S) .         { D = (Value*)S; }
const_i16(D) ::= CONST_I16(S) .             { D = (Value*)S; }
const_i16(D) ::= const_i16_0(S) .           { D = (Value*)S; }
const_i16(D) ::= const_i16_1(S) .           { D = (Value*)S; }
const_i16(D) ::= const_i16_F(S) .           { D = (Value*)S; }

%type const_i32 {Value*}
const_i32_0(D) ::= CONST_I32_0(S) .         { D = (Value*)S; }
const_i32_1(D) ::= CONST_I32_1(S) .         { D = (Value*)S; }
const_i32_F(D) ::= CONST_I32_F(S) .         { D = (Value*)S; }
const_i32(D) ::= CONST_I32(S) .             { D = (Value*)S; }
const_i32(D) ::= const_i32_0(S) .           { D = (Value*)S; }
const_i32(D) ::= const_i32_1(S) .           { D = (Value*)S; }
const_i32(D) ::= const_i32_F(S) .           { D = (Value*)S; }

%type const_i64 {Value*}
const_i64_0(D) ::= CONST_I64_0(S) .         { D = (Value*)S; }
const_i64_1(D) ::= CONST_I64_1(S) .         { D = (Value*)S; }
const_i64_F(D) ::= CONST_I64_F(S) .         { D = (Value*)S; }
const_i64(D) ::= CONST_I64(S) .             { D = (Value*)S; }
const_i64(D) ::= const_i64_0(S) .           { D = (Value*)S; }
const_i64(D) ::= const_i64_1(S) .           { D = (Value*)S; }
const_i64(D) ::= const_i64_F(S) .           { D = (Value*)S; }

%type const_f32 {Value*}
const_f32_0(D) ::= CONST_F32_0(S) .         { D = (Value*)S; }
const_f32_1(D) ::= CONST_F32_1(S) .         { D = (Value*)S; }
const_f32_F(D) ::= CONST_F32_F(S) .         { D = (Value*)S; }
const_f32_N1(D) ::= CONST_F32_N1(S) .       { D = (Value*)S; }
const_f32(D) ::= CONST_F32(S) .             { D = (Value*)S; }
const_f32(D) ::= const_f32_0(S) .           { D = (Value*)S; }
const_f32(D) ::= const_f32_1(S) .           { D = (Value*)S; }
const_f32(D) ::= const_f32_F(S) .           { D = (Value*)S; }
const_f32(D) ::= const_f32_N1(S) .          { D = (Value*)S; }

%type const_f64 {Value*}
const_f64_0(D) ::= CONST_F64_0(S) .         { D = (Value*)S; }
const_f64_1(D) ::= CONST_F64_1(S) .         { D = (Value*)S; }
const_f64_F(D) ::= CONST_F64_F(S) .         { D = (Value*)S; }
const_f64_N1(D) ::= CONST_F64_N1(S) .       { D = (Value*)S; }
const_f64(D) ::= CONST_F64(S) .             { D = (Value*)S; }
const_f64(D) ::= const_f64_0(S) .           { D = (Value*)S; }
const_f64(D) ::= const_f64_1(S) .           { D = (Value*)S; }
const_f64(D) ::= const_f64_F(S) .           { D = (Value*)S; }
const_f64(D) ::= const_f64_N1(S) .          { D = (Value*)S; }

%type const_v128 {Value*}
const_v128_0000(D) ::= CONST_V128_0000(S) . { D = (Value*)S; }
const_v128_1111(D) ::= CONST_V128_1111(S) . { D = (Value*)S; }
const_v128_FFFF(D) ::= CONST_v128_FFFF(S) . { D = (Value*)S; }
const_v128_0001(D) ::= CONST_V128_0001(S) . { D = (Value*)S; }
const_v128(D) ::= CONST_V128(S) .           { D = (Value*)S; }
const_v128(D) ::= const_v128_0000(S) .      { D = (Value*)S; }
const_v128(D) ::= const_v128_1111(S) .      { D = (Value*)S; }
const_v128(D) ::= const_v128_FFFF(S) .      { D = (Value*)S; }
const_v128(D) ::= const_v128_0001(S) .      { D = (Value*)S; }

%type const_i {Value*}
const_i(D) ::= const_i8(S) .                { D = (Value*)S; }
const_i(D) ::= const_i16(S) .               { D = (Value*)S; }
const_i(D) ::= const_i32(S) .               { D = (Value*)S; }
const_i(D) ::= const_i64(S) .               { D = (Value*)S; }
%type const_f {Value*}
const_f(D) ::= const_f32(S) .               { D = (Value*)S; }
const_f(D) ::= const_f64(S) .               { D = (Value*)S; }
%type const_v {Value*}
const_v(D) ::= const_v128(S) .              { D = (Value*)S; }
%type const {Value*}
const(D) ::= const_i(S) .                   { D = (Value*)S; }
const(D) ::= const_f(S) .                   { D = (Value*)S; }
const(D) ::= const_v(S) .                   { D = (Value*)S; }

%type value_i8 {Value*}
value_i8(D) ::= VALUE_I8(S) .               { D = (Value*)S; }
%type value_i16 {Value*}
value_i16(D) ::= VALUE_I16(S) .             { D = (Value*)S; }
%type value_i32 {Value*}
value_i32(D) ::= VALUE_I32(S) .             { D = (Value*)S; }
%type value_i64 {Value*}
value_i64(D) ::= VALUE_I64(S) .             { D = (Value*)S; }
%type value_f32 {Value*}
value_f32(D) ::= VALUE_F32(S) .             { D = (Value*)S; }
%type value_f64 {Value*}
value_f64(D) ::= VALUE_F64(S) .             { D = (Value*)S; }
%type value_v128 {Value*}
value_v128(D) ::= VALUE_V128(S) .           { D = (Value*)S; }

%type value_i {Value*}
value_i(D) ::= value_i8(S) .                { D = (Value*)S; }
value_i(D) ::= value_i16(S) .               { D = (Value*)S; }
value_i(D) ::= value_i32(S) .               { D = (Value*)S; }
value_i(D) ::= value_i64(S) .               { D = (Value*)S; }
%type value_f {Value*}
value_f(D) ::= value_f32(S) .               { D = (Value*)S; }
value_f(D) ::= value_f64(S) .               { D = (Value*)S; }
%type value_v {Value*}
value_v(D) ::= value_v128(S) .              { D = (Value*)S; }
%type value {Value*}
value(D) ::= value_i(S) .                   { D = (Value*)S; }
value(D) ::= value_f(S) .                   { D = (Value*)S; }
value(D) ::= value_v(S) .                   { D = (Value*)S; }

%type operand {Value*}
operand(D) ::= const(S) .                   { D = (Value*)S; }
operand(D) ::= value(S) .                   { D = (Value*)S; }


instr ::= OPCODE_COMMENT flags offset(O) . {
  LOG_OPCODE(OPCODE_COMMENT);
  fprintf(stdout, "comment: %s\n", (const char*)O);
  fflush(stdout);
}

instr ::= OPCODE_NOP . {
  LOG_OPCODE(OPCODE_NOP);
}

instr ::= OPCODE_SOURCE_OFFSET flags offset(O) . {
  LOG_OPCODE(OPCODE_SOURCE_OFFSET);
  fprintf(stdout, "source offset: %lld\n", O);
  fflush(stdout);
}

instr ::= OPCODE_DEBUG_BREAK flags . {
  LOG_OPCODE(OPCODE_DEBUG_BREAK);
}

instr ::= OPCODE_DEBUG_BREAK_TRUE flags operand . {
  LOG_OPCODE(OPCODE_DEBUG_BREAK_TRUE);
}

instr ::= OPCODE_TRAP flags . {
  LOG_OPCODE(OPCODE_TRAP);
}

instr ::= OPCODE_TRAP_TRUE flags operand . {
  LOG_OPCODE(OPCODE_TRAP_TRUE);
}

instr ::= OPCODE_CALL flags symbol_info . {
  LOG_OPCODE(OPCODE_CALL);
}

instr ::= OPCODE_CALL_TRUE flags operand symbol_info . {
  LOG_OPCODE(OPCODE_CALL_TRUE);
}

instr ::= OPCODE_CALL_INDIRECT flags operand . {
  LOG_OPCODE(OPCODE_CALL_INDIRECT);
}

instr ::= OPCODE_CALL_INDIRECT_TRUE flags operand operand . {
  LOG_OPCODE(OPCODE_CALL_INDIRECT_TRUE);
}

instr ::= OPCODE_CALL_EXTERN flags symbol_info . {
  LOG_OPCODE(OPCODE_CALL_EXTERN);
}

instr ::= OPCODE_RETURN flags . {
  LOG_OPCODE(OPCODE_RETURN);
}

instr ::= OPCODE_RETURN_TRUE flags operand . {
  LOG_OPCODE(OPCODE_RETURN_TRUE);
}

instr ::= OPCODE_SET_RETURN_ADDRESS flags operand . {
  LOG_OPCODE(OPCODE_SET_RETURN_ADDRESS);
}

instr ::= OPCODE_BRANCH flags label . {
  LOG_OPCODE(OPCODE_BRANCH);
}

instr ::= OPCODE_BRANCH_TRUE flags operand label . {
  LOG_OPCODE(OPCODE_BRANCH_TRUE);
}

instr ::= OPCODE_BRANCH_FALSE flags operand label . {
  LOG_OPCODE(OPCODE_BRANCH_FALSE);
}

instr ::= value OPCODE_ASSIGN flags operand . {
  LOG_OPCODE(OPCODE_ASSIGN);
}

instr ::= value OPCODE_CAST flags operand . {
  LOG_OPCODE(OPCODE_CAST);
}

instr ::= value OPCODE_ZERO_EXTEND flags operand . {
  LOG_OPCODE(OPCODE_ZERO_EXTEND);
}

instr ::= value OPCODE_SIGN_EXTEND flags operand . {
  LOG_OPCODE(OPCODE_SIGN_EXTEND);
}

instr ::= value OPCODE_TRUNCATE flags operand . {
  LOG_OPCODE(OPCODE_TRUNCATE);
}

instr ::= value OPCODE_CONVERT flags operand . {
  LOG_OPCODE(OPCODE_CONVERT);
}

instr ::= value OPCODE_ROUND flags operand . {
  LOG_OPCODE(OPCODE_ROUND);
}

instr ::= value OPCODE_VECTOR_CONVERT_I2F flags operand . {
  LOG_OPCODE(OPCODE_VECTOR_CONVERT_I2F);
}

instr ::= value OPCODE_VECTOR_CONVERT_F2I flags operand . {
  LOG_OPCODE(OPCODE_VECTOR_CONVERT_F2I);
}

instr ::= value OPCODE_LOAD_VECTOR_SHL flags operand . {
  LOG_OPCODE(OPCODE_LOAD_VECTOR_SHL);
}

instr ::= value OPCODE_LOAD_VECTOR_SHR flags operand . {
  LOG_OPCODE(OPCODE_LOAD_VECTOR_SHR);
}

instr ::= value OPCODE_LOAD_CLOCK flags . {
  LOG_OPCODE(OPCODE_LOAD_CLOCK);
}

instr ::= value OPCODE_LOAD_LOCAL flags operand . {
  LOG_OPCODE(OPCODE_LOAD_LOCAL);
}

instr ::= OPCODE_STORE_LOCAL flags operand operand . {
  LOG_OPCODE(OPCODE_STORE_LOCAL);
}

instr ::= value OPCODE_LOAD_CONTEXT flags offset . {
  LOG_OPCODE(OPCODE_LOAD_CONTEXT);
}

instr ::= OPCODE_STORE_CONTEXT flags offset operand . {
  LOG_OPCODE(OPCODE_STORE_CONTEXT);
}

instr ::= value OPCODE_LOAD flags operand . {
  LOG_OPCODE(OPCODE_LOAD);
}

instr ::= OPCODE_STORE flags operand operand . {
  LOG_OPCODE(OPCODE_STORE);
}

instr ::= OPCODE_PREFETCH flags operand offset . {
  LOG_OPCODE(OPCODE_PREFETCH);
}

instr ::= value OPCODE_MAX flags operand operand . {
  LOG_OPCODE(OPCODE_MAX);
}

instr ::= value OPCODE_MIN flags operand operand . {
  LOG_OPCODE(OPCODE_MIN);
}

instr ::= value OPCODE_SELECT flags operand operand operand . {
  LOG_OPCODE(OPCODE_SELECT);
}

instr ::= value OPCODE_IS_TRUE flags operand . {
  LOG_OPCODE(OPCODE_IS_TRUE);
}

instr ::= value OPCODE_IS_FALSE flags operand . {
  LOG_OPCODE(OPCODE_IS_FALSE);
}

instr ::= value OPCODE_COMPARE_EQ flags operand operand . {
  LOG_OPCODE(OPCODE_COMPARE_EQ);
}

instr ::= value OPCODE_COMPARE_NE flags operand operand . {
  LOG_OPCODE(OPCODE_COMPARE_NE);
}

instr ::= value OPCODE_COMPARE_SLT flags operand operand . {
  LOG_OPCODE(OPCODE_COMPARE_SLT);
}

instr ::= value OPCODE_COMPARE_SLE flags operand operand . {
  LOG_OPCODE(OPCODE_COMPARE_SLE);
}

instr ::= value OPCODE_COMPARE_SGT flags operand operand . {
  LOG_OPCODE(OPCODE_COMPARE_SGT);
}

instr ::= value OPCODE_COMPARE_SGE flags operand operand . {
  LOG_OPCODE(OPCODE_COMPARE_SGE);
}

instr ::= value OPCODE_COMPARE_ULT flags operand operand . {
  LOG_OPCODE(OPCODE_COMPARE_ULT);
}

instr ::= value OPCODE_COMPARE_ULE flags operand operand . {
  LOG_OPCODE(OPCODE_COMPARE_ULE);
}

instr ::= value OPCODE_COMPARE_UGT flags operand operand . {
  LOG_OPCODE(OPCODE_COMPARE_UGT);
}

instr ::= value OPCODE_COMPARE_UGE flags operand operand . {
  LOG_OPCODE(OPCODE_COMPARE_UGE);
}

instr ::= value OPCODE_DID_CARRY flags operand . {
  LOG_OPCODE(OPCODE_DID_CARRY);
}

instr ::= value OPCODE_DID_OVERFLOW flags operand . {
  LOG_OPCODE(OPCODE_DID_OVERFLOW);
}

instr ::= value OPCODE_DID_SATURATE flags operand . {
  LOG_OPCODE(OPCODE_DID_SATURATE);
}

instr ::= value OPCODE_VECTOR_COMPARE_EQ flags operand operand . {
  LOG_OPCODE(OPCODE_VECTOR_COMPARE_EQ);
}

instr ::= value OPCODE_VECTOR_COMPARE_SGT flags operand operand . {
  LOG_OPCODE(OPCODE_VECTOR_COMPARE_SGT);
}

instr ::= value OPCODE_VECTOR_COMPARE_SGE flags operand operand . {
  LOG_OPCODE(OPCODE_VECTOR_COMPARE_SGE);
}

instr ::= value OPCODE_VECTOR_COMPARE_UGT flags operand operand . {
  LOG_OPCODE(OPCODE_VECTOR_COMPARE_UGT);
}

instr ::= value OPCODE_VECTOR_COMPARE_UGE flags operand operand . {
  LOG_OPCODE(OPCODE_VECTOR_COMPARE_UGE);
}

instr ::= value OPCODE_ADD flags operand operand . {
  LOG_OPCODE(OPCODE_ADD);
}

instr ::= value OPCODE_ADD_CARRY flags operand operand operand . {
  LOG_OPCODE(OPCODE_ADD_CARRY);
}

instr ::= value OPCODE_VECTOR_ADD flags operand operand . {
  LOG_OPCODE(OPCODE_VECTOR_ADD);
}

instr ::= value OPCODE_SUB flags operand operand . {
  LOG_OPCODE(OPCODE_SUB);
}

instr ::= value OPCODE_MUL flags operand operand . {
  LOG_OPCODE(OPCODE_MUL);
}

instr ::= value OPCODE_MUL_HI flags operand operand . {
  LOG_OPCODE(OPCODE_MUL_HI);
}

instr ::= value OPCODE_DIV flags operand operand . {
  LOG_OPCODE(OPCODE_DIV);
}

instr ::= value OPCODE_MUL_ADD flags operand operand operand . {
  LOG_OPCODE(OPCODE_MUL_ADD);
}

instr ::= value OPCODE_MUL_SUB flags operand operand operand . {
  LOG_OPCODE(OPCODE_MUL_SUB);
}

instr ::= value OPCODE_NEG flags operand . {
  LOG_OPCODE(OPCODE_NEG);
}

instr ::= value OPCODE_ABS flags operand . {
  LOG_OPCODE(OPCODE_ABS);
}

instr ::= value OPCODE_SQRT flags operand . {
  LOG_OPCODE(OPCODE_SQRT);
}

instr ::= value OPCODE_RSQRT flags operand . {
  LOG_OPCODE(OPCODE_RSQRT);
}

instr ::= value OPCODE_POW2 flags operand . {
  LOG_OPCODE(OPCODE_POW2);
}

instr ::= value OPCODE_LOG2 flags operand . {
  LOG_OPCODE(OPCODE_LOG2);
}

instr ::= value OPCODE_DOT_PRODUCT_3 flags operand operand . {
  LOG_OPCODE(OPCODE_DOT_PRODUCT_3);
}

instr ::= value OPCODE_DOT_PRODUCT_4 flags operand operand . {
  LOG_OPCODE(OPCODE_DOT_PRODUCT_4);
}

instr ::= value OPCODE_AND flags operand operand . {
  LOG_OPCODE(OPCODE_AND);
}

instr ::= value OPCODE_OR flags operand operand . {
  LOG_OPCODE(OPCODE_OR);
}

instr ::= value OPCODE_XOR flags operand operand . {
  LOG_OPCODE(OPCODE_XOR);
}

instr ::= value OPCODE_NOT flags operand operand . {
  LOG_OPCODE(OPCODE_NOT);
}

instr ::= value OPCODE_SHL flags operand operand . {
  LOG_OPCODE(OPCODE_SHL);
}

instr ::= value OPCODE_VECTOR_SHL flags operand operand . {
  LOG_OPCODE(OPCODE_VECTOR_SHL);
}

instr ::= value OPCODE_SHR flags operand operand . {
  LOG_OPCODE(OPCODE_SHR);
}

instr ::= value OPCODE_VECTOR_SHR flags operand operand . {
  LOG_OPCODE(OPCODE_VECTOR_SHR);
}

instr ::= value OPCODE_SHA flags operand operand . {
  LOG_OPCODE(OPCODE_SHA);
}

instr ::= value OPCODE_VECTOR_SHA flags operand operand . {
  LOG_OPCODE(OPCODE_VECTOR_SHA);
}

instr ::= value OPCODE_ROTATE_LEFT flags operand operand . {
  LOG_OPCODE(OPCODE_ROTATE_LEFT);
}

instr ::= value OPCODE_BYTE_SWAP flags operand . {
  LOG_OPCODE(OPCODE_BYTE_SWAP);
}

instr ::= value OPCODE_CNTLZ flags operand . {
  LOG_OPCODE(OPCODE_CNTLZ);
}

instr ::= value OPCODE_INSERT flags operand operand operand . {
  LOG_OPCODE(OPCODE_INSERT);
}

instr ::= value OPCODE_EXTRACT flags operand operand . {
  LOG_OPCODE(OPCODE_SPLAT);
}

instr ::= value OPCODE_SPLAT flags operand . {
  LOG_OPCODE(OPCODE_SPLAT);
}

instr ::= value OPCODE_PERMUTE flags operand operand operand . {
  LOG_OPCODE(OPCODE_PERMUTE);
}

instr ::= value OPCODE_SWIZZLE flags operand offset . {
  LOG_OPCODE(OPCODE_SWIZZLE);
}

instr ::= value OPCODE_PACK flags operand . {
  LOG_OPCODE(OPCODE_PACK);
}

instr ::= value OPCODE_UNPACK flags operand . {
  LOG_OPCODE(OPCODE_UNPACK);
}

instr ::= value OPCODE_COMPARE_EXCHANGE flags operand operand operand . {
  LOG_OPCODE(OPCODE_COMPARE_EXCHANGE);
}

instr ::= value OPCODE_ATOMIC_EXCHANGE flags operand operand . {
  LOG_OPCODE(OPCODE_ATOMIC_EXCHANGE);
}

instr ::= value OPCODE_ATOMIC_ADD flags operand operand . {
  LOG_OPCODE(OPCODE_ATOMIC_ADD);
}

instr ::= value OPCODE_ATOMIC_SUB flags operand operand . {
  LOG_OPCODE(OPCODE_ATOMIC_SUB);
}
