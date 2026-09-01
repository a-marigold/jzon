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

inline fn shuffleVector128(
    comptime E: type,
    vector: @Vector(16, E),
    mask: @Vector(16, u8),
) @TypeOf(vector) {
    return switch (CPU.arch) {
        .x86_64 => asm ("pshufb %[mask], %[vector]" // AT&T syntax
            : [vector] "+x" (vector),
            : [mask] "x" (mask),
        ),
        else => @compileError("Unsupported arch"),
        // TODO: add risc-v
    };
}

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
