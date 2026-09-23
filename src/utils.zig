const std = @import("std");
const math = std.math;
const Target = std.Target;
const builtin = @import("builtin");
const simd = @import("simd.zig");

const CPU = builtin.cpu;

/// For each sequence of `1` bits in an unsigned integer `mask`,
/// leaves only the first least significant bit of the sequence.
///
/// Example:
/// For `01101111` returns `00100001`
///
/// (Left bits: most significant, Right bits: least significant).
pub inline fn getStartsOfMaskSequences(mask: u64) u64 {
    // Example:
    // Mask =                  `0110111100000000`.
    // Shifted = `Mask << 1` = `1101111000000000`
    // Inverted = `~Shifted` = `0010000111111111`
    // Result = `Mask & Inverted` = `0110111100000000` &
    //                              `0010000111111111` =
    //                              `0010000100000000`
    return mask & ~(mask << 1);
}

/// For each sequence of `1` bits in an unsigned integer `mask`,
/// leaves only the last most significant bit of the sequence, *shifted to the left by 1*.
///
/// *shifted to the left* means the resulting mask doesn't contain just ends of sequences,
/// but contains ends shifted to the left by 1. That is the real ends are at `result >> 1`.
pub inline fn getEndsOfMaskSequences(mask: u64, startsMask: u64) u64 {
    // E.g `startsMask` is `00000100`, `mask` is `00011100`.
    // Addition carries `startsMask` bits to the left, forming ends of sequences:
    // `00000100` + `00111000` = `01000000`
    return startsMask + mask;
}

/// Does prefix XOR for bits in `bits`.
///
/// Example:
/// For `01001000` returns `01111000`.
///
/// (Left bits: most significant, Right bits: least significant).
pub inline fn getBitsPrefixXor(bits: u64) u64 {
    // Carryless multiplication of `bits` by ~0 (every bit is 1)
    // shifts `mask` as many times as wide the ~0 (64) and does XOR between shifting results,
    // and it's a prefix XOR at hardware level
    if (simd.isMulCarrylessSupported())
        return simd.mulCarryless(bits, ~0);

    return getBitsPrefixXor_software(bits);
}
/// A software implementation for cases when
/// the target CPU lacks of carry-less multiplication for prefix xor.
inline fn getBitsPrefixXor_software(bits: u64) u64 {
    const maxOffset = @typeInfo(u64).int.bits / 2;

    var result = bits;

    comptime var offset = 0;

    comptime var iteration = 0;
    inline while (offset <= maxOffset) : (iteration += 1) {
        offset = 1 << iteration;

        // Shift the prev result on `offset` (a power of two)
        // and do XOR between it and just the prev result
        // to get prefix XOR
        result ^= result << offset;
    }

    return result;
}

/// Checks if `T` is unsigned.
///
/// Returns a comptime mask of type `T`, where every bit at even index is `1`.
pub fn genEvenBitsMask() u64 {
    // Division of a value where all bits are 1 (max value) by 3
    // results in a sequence of bits where only even bits are set to 1
    return math.maxInt(u64) / 3;
}

/// Returns an unique byte-flag with only a single `1` at `bitOffset`.
pub fn getByteFlag(comptime bitOffset: comptime_int) u8 {
    return 0b00000001 << bitOffset;
}

/// Returns index of the first least significant bit which is set to 1.
pub inline fn countTrailZeros(bits: u64) u64 {
    return @ctz(bits);
}
/// Returns index of the first most significant bit which is set to 1.
pub inline fn countLeadZeros(bits: u64) u64 {
    return @clz(bits);
}

/// Omits the first least significant bit which is set to 1.
///
/// Always returns 0 for 0.
///
/// Example: For `00100010` returns `00100000`
///
/// (Left bits: most significant, Right bits: least significant).
pub inline fn omitTrailBit(bits: u64) u64 {
    return bits & (bits - 1);
}
/// Fills the high `byte` bits with 0, leaving only the low nibble.
pub inline fn getLowNibble(byte: u8) u8 {
    return byte & 0b00001111;
}

/// Moves the high `byte` bits to the low bits, filling the previous place of high bits with 0.
pub inline fn getHighNibble(byte: u8) u8 {
    return byte >> 4;
}

/// Returns an array, filled with values from `start` to `end` (also including `end`),
/// not including values that equal any of `excludeValues`.
pub fn range(
    comptime start: comptime_int,
    comptime end: comptime_int,
    comptime excludeValues: anytype,
) [(end - start) + 1]comptime_int {
    var result: [(end - start) + 1]comptime_int = undefined;

    var index = 0;
    range: while (start <= end) : (index += 1) {
        const value = start + index;

        for (excludeValues) |excludeValue|
            if (value == excludeValue) continue :range;

        result[index] = value;
    }

    return result;
}
