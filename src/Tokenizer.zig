const Tokenizer = @This();

const std = @import("std");
const math = std.math;
const simd = std.simd;
const Target = std.Target;
const builtin = @import("builtin");

const CPU = builtin.cpu;

/// The control characters of JSON.
const CONTROL_CHARS = [_]u8{ '[', ']', '{', '}', ':', ',' };

const Token = struct {};

source: []const u8,
pub fn init(source: []const u8) Tokenizer {
    return .{ .source = source };
}

/// Returns `lowNibbles` and `highNibbles` constant-arrays,
/// indexes of which are low or high nibbles of `CONTROL_CHARS` elements,
/// and the values at indexes are unique masks.
///
/// - `lowNibbles` have 16 elements 'cause the maximum
/// low nibble of ASCII is `0xF` (decimal `16`).
///
/// - `highNibbles` have 8 elements 'cause the maximum
/// high nibble of ASCII is `0x7` (decimal `7`).
///
/// - E.g, char `{` is `0x7B` (decimal 123), and the low nibble `0xB`
/// perfectly fits `0xF`, and the high `0x7` perfectly fits `0x7`.
///
/// Used as a lookup-table vector, from which the vector-shuffle intruction
/// builds a new vector for searching control characters (see `next` function).
fn genControlCharTables() struct { lowNibbles: [16]u8, highNibbles: [8]u8 } {
    // Fill with `0` to ensure there are falsy bits at indexes of non-control chars
    var lowNibbles: [16]u8 = @splat(0);
    var highNibbles: [8]u8 = @splat(0);

    // Indexes are high nibbles of control chars
    const highNibbleFlags = flagsBlock: {
        // 8 unique flags (00000001, 00000010, ...) for every high nibble
        const flagsArray: [8]u8 = undefined;

        var flag = 1;
        for (flagsArray) |*flagEl| {
            flagEl.* = flag;

            flag = 1 << flag;
        }

        break :flagsBlock flagsArray;
    };

    for (CONTROL_CHARS) |char| {
        const lowCharNibble = getLowNibble(char);
        const highCharNibble = getHighNibble(char);

        if (highCharNibble > highNibbleFlags.len) @compileError("Control char is out of ASCII");

        const flag = highNibbleFlags[highCharNibble];
        lowNibbles[lowCharNibble] |= flag;
        highNibbles[highCharNibble] |= flag;
    }

    return .{
        .lowNibbles = lowNibbles,
        .highNibbles = highNibbles,
    };
}

/// For each sequence of `1` bits in an unsigned integer `mask`,
/// leaves only the first least significant bit of the sequence.
///
/// Example:
/// For `01101111` returns `00100001`
///
/// (Left bits: most significant, Right bits: least significant).
inline fn getStartsOfMaskSequences(mask: anytype) @TypeOf(mask) {
    comptime checkIsUint(@TypeOf(mask));

    // Example:
    // Mask = `0110111100000000`.
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
/// it contains ends shifted to the left by 1. To get the real ends, just do `result >> 1`.
///
/// Example:
/// For `Mask = 01101111`, `Starts = 00100001`,
/// returns `10010000` (Ends shifted by 1,
/// to get the real ends - `10010000 >> 1`, which is `01001000`).
inline fn getEndsOfMaskSequences(mask: anytype, startsMask: @TypeOf(mask)) @TypeOf(mask) {
    // E.g `startsMask` is `00000100`, `mask` is `00011100`.
    // Addition carries `startsMask` bits to the left, forming ends of sequences:
    // `00000100` + `00111000` = `01000000`
    return startsMask + mask;
}

/// Returns 16, 32, 64 or `null` in case of lack of SIMD.
///
/// Returns 64 only if the target is `avx512bw` (which supports 64-byte vector shuffles).
/// If the target supports just `avx512`, 32 is returned.
fn getVectorLen_x64() ?comptime_int {
    const features = CPU.features;
    const hasFeature = Target.x86.featureSetHas;

    return if (hasFeature(features, .avx512bw))
        64
    else if (hasFeature(features, .avx2))
        32
    else if (hasFeature(features, .ssse3))
        16
    else
        null;
}

/// Permutates elements in `vector` based on `mask` elements.
///
/// Uses only 0..4 bits (low bits) of `mask` indexes,
/// and if the 7 (the highest) bit equals `1`, the `result[index]` is set to `0`.
///
/// Example:
///
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

/// `mask` doesn't index all the 256-bit `vector`.
/// Instead, 0..16 elements of `mask` index 0..16 elements of `vector`,
/// and 16..32 elements of mask index 16..32 elemenets of `vector`.
///
/// That is, it is like a parallel `shuffleVector128_x64` for two masks and vectors.
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

/// Like `shuffleVector128`, but uses 0..6 bits of `mask` elements,
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

/// Returns `true` when 128-bit vector-shuffle is supported on `aarch64`.
fn is128BitVector_aarch64() bool {
    return Target.aarch64.featureSetHas(CPU.features, .neon);
}
/// More preferred than `is128BitVector_aarch64` result.
///
/// Returns `true` only when the `aarch64` target supports vectors with variable length (128-512 bit),
/// and only when the target supports 32-64 byte shuffles with them.
fn isVariableVectorLen_aarch64() bool {
    return Target.aarch64.featureSetHas(CPU.features, .sve2);
}
/// Calling this function without
/// checking `isVariableVectorLen_aarch64` can cause an illegal instruction fault.
///
/// Returns the length in bytes of one vector registers.
///
/// The result of this function should never be persisted 'cause it varies
/// accross the CPU threads, and if the OS moves the parser
/// to another thread during a context switch, the result can change.
inline fn getVariableVectorLen_aarch64() usize {
    return asm ("cntb %[result]"
        : [result] "=r" (-> usize),
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

/// Fills the high `byte` bits to zeros, leaving only the low nibble.
inline fn getLowNibble(byte: u8) u8 {
    return byte & 0b00001111;
}
/// Moves the high `byte` bits to the low bits, filling the previous place of high bits to zeros.
inline fn getHighNibble(byte: u8) u8 {
    return byte >> 4;
}

/// Fills high bits of each `vector` element to zero and leaves only the low bits.
inline fn getLowNibblesVector(
    comptime len: comptime_int,
    vector: @Vector(len, u8),
) @Vector(len, u8) {
    return vector & @as(@TypeOf(vector), @splat(0b00001111));
}

/// Moves high bits of each `vector` element to its low bits.
inline fn getHighNibblesVector(
    comptime len: comptime_int,
    vector: @Vector(len, u8),
) @Vector(len, u8) {
    return vector >> @as(@TypeOf(vector), @splat(4));
}

/// Expands `vector` to `newLen` and fills new elements to 0.
fn expandComptimeVector(
    comptime vector: anytype,
    comptime newLen: comptime_int,
) @Vector(newLen, u8) {
    return vector ++ @as(@Vector(newLen - vector.len, u8), @splat(0));
}

/// If `T` is not an unsigned integer, shows a compile error.
fn checkIsUint(comptime T: type) void {
    const TInfo = @typeInfo(T);

    if (TInfo != .int and TInfo.int.signedness)
        @compileError("An integer expected");
}
