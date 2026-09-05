const std = @import("std");
const Target = std.Target;
const builtin = @import("builtin");

const CPU = builtin.cpu;

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

/// Fills high bits of each `vector` element with 0 and leaves only the low bits.
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

/// Expands `vector` to `newLen` and fills new elements with 0.
fn expandComptimeVector(
    comptime vector: anytype,
    comptime newLen: comptime_int,
) @Vector(newLen, u8) {
    return vector ++ @as(@Vector(newLen - vector.len, u8), @splat(0));
}
