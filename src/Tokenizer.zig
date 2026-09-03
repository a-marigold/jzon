const Tokenizer = @This();

const std = @import("std");
const simd = std.simd;
const Target = std.Target;
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

/// Permutates elements in `vector` based on `mask` elements.
///
/// Uses only 0..4 bits (low bits) of `mask` indexes,
/// and if the 7 (the highest) bit equals `1`, the `result[index]` is set to `0`.
///
/// Example:
/// 1. First iteration - `result[0] = vector[ mask[0] ]`.
/// 2. Second - `result[1] = vector[ mask[1] ]`.
/// 3. ...
///
/// Returns the resulting vector.
inline fn shuffleVector128_x64(
    vector: @Vector(16, u8),
    mask: @Vector(16, u8),
) @TypeOf(vector) {
    return asm ("pshufb %[mask], %[vector]" // `vector` is mutated
        : [vector] "+x" (vector),
        : [mask] "x" (mask),
    );
}

/// Permutates elements in `vector` based on `mask` elements.
///
/// If an element of `mask` is more than 16 (bytes amount of 128 bits),
/// `0` is written to the result.
///
/// Example:
/// 1. First iteration - `result[0] = vector[ mask[0] ]`.
/// 2. Second - `result[1] = vector[ mask[1] ]`.
/// 3. ...
///
/// Returns the resulting vector.
inline fn shuffleVector128_aarch64(
    vector: @Vector(16, u8),
    mask: @Vector(16, u8),
) @Vector(16, u8) {
    return asm ("tbl %[result].16b, { %[vector].16b }, %[mask].16b"
        : [result] "=w" (-> @Vector(16, u8)),
        : [vector] "w" (vector),
          [mask] "w" (mask),
    );
}

/// Only for `x64`.
///
/// `mask` doesn't index all the 256-bit `vector` (as in `shuffleVector128` the `mask` indexes a 128-bit vector).
/// Instead, 0..128 bits of `mask` index 0..128 bits of `vector`,
/// and the 128..256 bits of mask index 128..256 bits of `vector`.
///
/// That is, it is like a parallel `shuffleVector128` for two masks and vectors.
///
/// Returns the resulting vector.
///
/// Split the result in halves of 128-bits to get the two results.
inline fn shuffleVector256_x64(
    vector: @Vector(32, u8),
    mask: @Vector(32, u8),
) @TypeOf(vector) {
    return asm ("vpshufb %[mask], %[vector], %[result]"
        : [result] "=x" (-> @Vector(32, u8)),
        : [vector] "x" (vector),
          [mask] "x" (mask),
    );
}

/// Only for `x64`.
///
/// Like `shuffleVector128`, but uses 0..6 bytes of `mask` indexes,
/// allowing indexing the whole 512-bit vector.
///
/// Returns the resulting vector.
inline fn shuffleVector512_x64(
    vector: @Vector(64, u8),
    mask: @Vector(64, u8),
) @TypeOf(vector) {
    return asm ("vpshufb %[mask], %[vector], %[result]"
        : [result] "=x" (-> @Vector(64, u8)),
        : [vector] "x" (vector),
          [mask] "x" (mask),
    );
}

/// Returns 128, 256, 512 or `null` in case of lack of SIMD.
///
/// Returns 512 only if the target is `avx512bw` (which supports 64-byte vector shuffles).
/// If the target supports `abx512` but doesn't support the mentioned shuffles, 256 is returned.
inline fn getVectorLen_x64() ?comptime_int {
    const features = CPU.features;
    const hasFeature = Target.x86.featureSetHas;

    return if (hasFeature(features, .avx512bw))
        512
    else if (hasFeature(features, .avx2))
        256
    else if (hasFeature(features, .ssse3))
        128
    else
        null;
}
/// Returns `true` when 128-bit vector-shuffle is supported on `aarch64`.
inline fn is128BitVector_aarch64() bool {
    return Target.aarch64.featureSetHas(CPU.features, .neon);
}

/// Returns `true` only when the `aarch64` target supports vectors with variable length (128-512 bit),
/// and only when the target supports 32-64 byte shuffles with them.
inline fn isVariableVectorLen_aarch64() bool {
    return Target.aarch64.featureSetHas(CPU.features, .sve2);
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
