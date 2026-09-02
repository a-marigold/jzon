const Tokenizer = @This();

const std = @import("std");
const simd = std.simd;
const builtin = @import("builtin");

const CPU = builtin.cpu;

const U8_VECTOR_LEN = simd.suggestVectorLength(u8);

/// The control characters of JSON.
const CONTROL_CHARS = [_]u8{
    '[',
    ']',
    '{',
    '}',
    ':',
    '.',
    ',',
    '"',
    '"',
};

/// Contains `LOW` and `HIGH` constant-arrays,
/// indexes of which are low or high bits of `CONTROL_CHARS` elements,
/// e.g `LOW['{' & LOW_BYTE_BITS])`, and the values of indexes are unique masks.
///
/// Used as a lookup-table vector, from which the vector-shuffle intruction
/// builds a new vector for searching control characters (see `next` function).
///
/// Not used when the cpu target doesn't support SIMD.
const CONTROL_CHAR_BITS = block: {
    // Fill with `0` to ensure there are falsy bits at indexes of non-control chars
    var low: [16]u8 = @splat(0);
    var high: [16]u8 = @splat(0);

    // Indexes are high bits of control chars
    const highBitFlags = flagsBlock: {
        // 8 unique flags (00000001, 00000010, etc)
        // That's enough 'cause only the ASCII chars with high bits less than 8 are used
        const flagsArray: [8]u8 = undefined;

        var flag = 1;
        for (flagsArray) |*flagEl| {
            flagEl.* = flag;

            flag = 1 << flag;
        }

        break :flagsBlock flagsArray;
    };

    for (CONTROL_CHARS) |char| {
        const lowCharBits = getLowBits(char);
        const highCharBits = getHighBits(char);

        if (highCharBits > highBitFlags.len) @compileError("Control char is out of ASCII");

        const flag = highBitFlags[highCharBits];

        low[lowCharBits] |= flag;
        high[highCharBits] |= flag;
    }

    break :block struct {
        pub const LOW = low;
        pub const HIGH = high;
    };
};

/// Cross-platform.
///
/// Permutates elements in `vector` based on `mask` indexes.
///
/// Example:
/// 1. First iteration - `result[0] = vector[ mask[0] ]`.
/// 2. Second - `result[1] = vector[ mask[1] ]`.
/// 3. ...
///
/// The result is a brand-new vector.
inline fn shuffleVector128(
    vector: @Vector(16, u8),
    mask: @Vector(16, u8),
) @TypeOf(vector) {
    return switch (CPU.arch) {
        .x86_64 => asm ("pshufb %[mask], %[vector]" // AT&T syntax
            : [vector] "+x" (vector),
            : [mask] "x" (mask),
        ),
        .aarch64 => asm ("tbl %[result].16b, { %[vector] }, %[mask]"
            : [result] "=w" (-> @Vector(16, u8)),
            : [vector] "w" (vector),
              [mask] "w" (mask),
        ),
        else => @compileError("Unsupported arch"),
        // TODO: add risc-v
    };
}

/// Only for `x86_64`.
///
/// `mask` doesn't index all the 256-bit `vector` (as in `shuffleVector128` the `mask` indexes a 128-bit vector).
/// Instead, 0..128 bits of `mask` index 0..128 bits of `vector`,
/// and the 128..256 bits of mask index 128..256 bits of `vector`.
///
/// That is, it is like a parallel `shuffleVector128` for two masks and vectors.
///
/// Returns a brand-new 256-bit vector with the result.
///
/// Split the result in halves of 128-bits to get the two results.
inline fn shuffleVector256_x64(
    vector: @Vector(32, u8),
    mask: @Vector(32, u8),
) @TypeOf(vector) {
    return asm ("vpshufb %[mask], %[vector], %[result]"
        : [result] "x" (-> @Vector(32, u8)),
        : [vector] "x" (vector),
          [mask] "x" (mask),
    );
}

/// Fills the high `byte` bits to zeros.
fn getLowBits(byte: u8) u8 {
    return byte & 0b00001111;
}

/// Moves the high `byte` bits to the low bits, filling the previous place of high bits to zeros.
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
