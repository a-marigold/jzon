const Tokenizer = @This();

const std = @import("std");
const simd = std.simd;
const builtin = @import("builtin");

const CPU_ARCH = builtin.cpu.arch;

const U8_VECTOR_LEN = simd.suggestVectorLength(u8);

/// Example: for `byte = 0b11110110` returns `0b00000110`
fn getLowBits(byte: u8) u8 {
    return byte & 0b00001111;
}
/// Example: for `byte = 0b11110110` returns `0b00001111` (high bits are moved to low bits).
fn getHighBits(byte: u8) u8 {
    return byte >> 4;
}

/// Fills high bits of each `vector` element to zero and leaves only the low bits.
fn getLowBitsVector(comptime len: comptime_int, vector: @Vector(len, u8)) @Vector(len, u8) {
    return vector & @as(@TypeOf(vector), @splat(0b00001111));
}
/// Fills low bits to zero and moves the high bits to low bits for each `vector` element.
fn getHighBitsVector(comptime len: comptime_int, vector: @Vector(len, u8)) @Vector(len, u8) {
    return vector >> @as(@TypeOf(vector), @splat(4));
}
