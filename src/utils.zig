const std = @import("std");
const math = std.math;
const Target = std.Target;
const builtin = @import("builtin");

const CPU = builtin.cpu;

/// SIMD utils.
pub const simd = struct {
    const _MM_CMPINT_EQ = 0;
    const _MM_CMPINT_NE = 4;

    /// Returns 16, 32, 64 or `null` in case of lack of SIMD.
    ///
    /// Returns 64 only if the target is `avx512bw` (which supports 64-byte vector shuffles).
    /// If the target supports just `avx512`, 32 is returned.
    pub inline fn getVectorLen_x64() ?comptime_int {
        const features = CPU.features;
        const hasFeature = Target.x86.featureSetHas;

        return if (hasFeature(
            features,
            .avx512bw, // Allows instructions with bytes within 512-bit registers.
        ))
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
    pub inline fn shuffleVector128_x64(
        vector: @Vector(16, u8),
        mask: @Vector(16, u8),
    ) @TypeOf(vector) {
        return asm ("pshufb %[mask], %[vector]" // `vector` is mutated
            : [vector] "+v" (vector),
            : [mask] "v" (mask),
        );
    }

    /// `mask` doesn't index all the 256-bit `vector`.
    /// Instead, 0..16 elements of `mask` index 0..16 elements of `vector`,
    /// and 16..32 elements of mask index 16..32 elements of `vector`.
    ///
    /// That is, it is like a parallel `shuffleVector128_x64` for two masks and vectors.
    ///
    /// Returns the resulting vector.
    ///
    /// Split the result in halves of 128-bits to get the two results.
    pub inline fn shuffleVector256_x64(
        vector: @Vector(32, u8),
        mask: @Vector(32, u8),
    ) @TypeOf(vector) {
        return asm ("vpshufb %[mask], %[vector], %[result]"
            : [result] "=v" (-> @Vector(32, u8)),
            : [vector] "v" (vector),
              [mask] "v" (mask),
        );
    }

    /// Like `shuffleVector128`, but uses 0..6 bits of `mask` elements,
    /// allowing indexing the whole 512-bit vector.
    ///
    /// Returns the resulting vector.
    pub inline fn shuffleVector512_x64(
        vector: @Vector(64, u8),
        mask: @Vector(64, u8),
    ) @TypeOf(vector) {
        return asm ("vpshufb %[mask], %[vector], %[result]"
            : [result] "=v" (-> @Vector(64, u8)),
            : [vector] "v" (vector),
              [mask] "v" (mask),
        );
    }

    /// Returns `true` when 128-bit vector-shuffle is supported on `aarch64`.
    pub inline fn is128BitVector_aarch64() bool {
        return Target.aarch64.featureSetHas(CPU.features, .neon);
    }

    /// More preferred than `is128BitVector_aarch64` result.
    ///
    /// Returns `true` only when the `aarch64` target supports vectors with variable length (128-512 bit),
    /// and only when the target supports 32-64 byte shuffles with them.
    pub inline fn isVariableVectorLen_aarch64() bool {
        return Target.aarch64.featureSetHas(CPU.features, .sve2);
    }
    /// Calling this function without checking `isVariableVectorLen_aarch64` is illegal.
    ///
    /// Returns the length in bytes of one vector registers.
    ///
    /// The result of this function should never be persisted 'cause it varies
    /// accross the CPU threads, and if the OS moves the parser
    /// to another thread during a context switch, the result can change.
    pub inline fn getVariableVectorLen_aarch64() usize {
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
    pub inline fn shuffleVector128_aarch64(
        vector: @Vector(16, u8),
        mask: @Vector(16, u8),
    ) @Vector(16, u8) {
        return asm ("tbl %[result].16b, { %[vector].16b }, %[mask].16b"
            : [result] "=w" (-> @Vector(16, u8)),
            : [vector] "w" (vector),
              [mask] "w" (mask),
        );
    }

    pub const CompareOperation = enum { Eql, NotEql };

    /// Compares every byte of the two vectors using `operation`,
    /// and if they are equal, sets bit of their position
    /// (e.g, the second bit if the second elements are compared)
    /// in the resulting mask to `1`.
    pub inline fn compareToBits128_x64(
        comptime operation: CompareOperation,
        a: @Vector(16, u8),
        b: @Vector(16, u8),
    ) u64 {
        const equalVector = switch (comptime operation) {
            .Eql => a == b,
            .NotEql => a != b,
        };
        return asm ("pmovmskb %[vector], %[result]"
            : [result] "=r" (-> u64),
            : [vector] "v" (equalVector),
        );
    }
    /// Compares every byte of the two vectors using `operation`,
    /// and if they are equal, sets bit of their position
    /// (e.g, the second bit if the second elements are compared)
    /// in the resulting mask to `1`.
    pub inline fn compareToBits512_x64(
        comptime operation: CompareOperation,
        a: @Vector(64, u8),
        b: @Vector(64, u8),
    ) u64 {
        var mask: u64 = 0;
        return asm (
            \\ vpcmpb %[operation], %[a], %[b], %[mask]
            \\ kmovq %[mask], %[result]
            : [result] "=r" (-> usize),
              [mask] "=&k" (mask),
            : [a] "v" (a),
              [b] "v" (b),
              [operation] "i" (switch (comptime operation) {
                .Eql => _MM_CMPINT_EQ,
                .NotEql => _MM_CMPINT_NE,
              }),
        );
    }

    /// Compares every byte of the two vectors using `operation`,
    /// and if they are equal, sets bit of their position
    /// (e.g, the second bit if the second elements are compared)
    /// in the resulting mask to `1`.
    pub inline fn compareToBits128_aarch64(
        comptime operation: CompareOperation,
        a: @Vector(16, u8),
        b: @Vector(16, u8),
    ) u64 {
        const comparedVector = switch (comptime operation) {
            .Eql => a == b,
            .NotEql => a != b,
        };

        // The first half of this vector contains `00000001`, `00000010`, ..., `10000000`.
        // The second half is a duplicated first half
        const singleBitMasks: @Vector(16, u8) =
            comptime block: {
                var masks: [16]u8 = undefined;

                var mask = 0;
                for (0..8) |index| {
                    mask = 1 << index;

                    masks[index] = mask;
                    masks[index + 8] = mask;
                }

                break :block masks;
            };

        // Replace every `true` (0xFF) byte of `vector`
        // with a byte, where only one bit is set to 1,
        // representing its bit index in resulting mask
        const singleBitsVector = comparedVector & singleBitMasks;

        const lowHalfVector: @Vector(8, u8) = singleBitsVector[0..8];
        const highHalfVector: @Vector(8, u8) = singleBitsVector[8..];

        // Every byte has only one unique bit set to 1,
        // so `Add` accross all elements is the same
        // as `Or` accross the elements:
        // `0001` + `0010` = `0011` = `0001` | `0010`
        const lowHalfMask: u64 = @reduce(.Add, lowHalfVector);
        const highHalfMask: u64 = @reduce(.Add, highHalfVector);

        return (highHalfMask << 8) | lowHalfMask;
    }

    pub inline fn isMulCarrylessSupported() bool {
        return switch (CPU.arch) {
            .x86_64 => Target.x86.featureSetHas(CPU.features, .pclmul),
            .aarch64 => Target.aarch64.featureSetHasAny(CPU.features, .{
                Target.aarch64.Feature.sve_aes,
                Target.aarch64.Feature.sve_aes2,
            }),

            // TODO: other architectures
            else => false,
        };
    }

    /// Carry-less multiplication of two integers.
    pub inline fn mulCarryless(a: u64, b: u64) u64 {
        switch (CPU.arch) {
            .x86_64 => if (Target.x86.featureSetHas(CPU.features, .pclmul)) {
                const aVector: @Vector(2, u64) = .{ a, 0 };
                const bVector: @Vector(2, u64) = .{ b, 0 };

                const resultVector = asm (
                    // `0x00` means the least significant bits of vectors are multiplied
                        "pclmulqdq $0x00, %[a], %[b]" // `b` is mutated
                        : [b] "+x" (bVector),
                        : [a] "x" (aVector),
                    );
                return resultVector[0];
            },
            .aarch64 => if (Target.aarch64.featureSetHasAny(CPU.features, .{
                Target.aarch64.Feature.sve_aes,
                Target.aarch64.Feature.sve_aes2,
            })) {
                const aVector: @Vector(2, u64) = .{ a, 0 };
                const bVector: @Vector(2, u64) = .{ b, 0 };

                const resultVector = asm ("pmull %[result].1q, %[a].1d, %[b].1d"
                    : [result] "=w" (-> @Vector(2, u64)),
                    : [a] "w" (aVector),
                      [b] "w" (bVector),
                );
                return resultVector[0];
            },
            else => @compileError("Unsupported architecture"),
        }
    }

    /// Fills high bits of each `vector` element with 0 and leaves only the low bits.
    pub inline fn getLowNibblesVector(vector: anytype) @TypeOf(vector) {
        return vector & @as(@TypeOf(vector), @splat(0b00001111));
    }
    /// Moves high bits of each `vector` element to its low bits.
    pub inline fn getHighNibblesVector(vector: anytype) @TypeOf(vector) {
        return vector >> @as(@TypeOf(vector), @splat(4));
    }

    /// Expands `vector` which has `u8` elements to `newLen` and fills its new elements with 0.
    pub inline fn expandVector(
        vector: anytype,
        comptime newLen: comptime_int,
    ) @Vector(newLen, u8) {
        return vector ++ @as(@Vector(newLen - vector.len, u8), @splat(0));
    }
};

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
pub inline fn getTrailBitIndex(bits: u64) u64 {
    return @ctz(bits);
}
/// Returns index of the first most significant bit which is set to 1.
pub inline fn getLeadBitIndex(bits: u64) u64 {
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
