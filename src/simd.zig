//! SIMD utils.

const std = @import("std");
const Target = std.Target;
const builtin = @import("builtin");

const CPU = builtin.cpu;

const x86 = struct {
    const _MM_CMPINT_EQ: u8 = 0x00;
    const _MM_CMPINT_NE: u8 = 0x04;
    const _MM_CMPINT_GT: u8 = 0x06;

    /// Returns 16, 32, 64 or `null` in case of lack of SIMD.
    ///
    /// Returns 64 only if the target is `avx512bw` (which supports 64-byte vector shuffles).
    /// If the target supports just `avx512`, 32 is returned.
    pub inline fn getVectorLen() ?comptime_int {
        const features = CPU.features;
        const hasFeature = Target.x86.featureSetHas;

        return if (hasFeature(
            features,
            .avx512bw, // Allows instructions with bytes within 512-bit registers
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
    pub inline fn shuffleVector128(
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
    pub inline fn shuffleVector256(
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
    pub inline fn shuffleVector512(
        vector: @Vector(64, u8),
        mask: @Vector(64, u8),
    ) @TypeOf(vector) {
        return asm ("vpshufb %[mask], %[vector], %[result]"
            : [result] "=v" (-> @Vector(64, u8)),
            : [vector] "v" (vector),
              [mask] "v" (mask),
        );
    }

    /// Compares each element of the two vectors producing a mask,
    /// where bit is set to 1 if the elements equal.
    pub inline fn eqlToBits128(a: @Vector(16, u8), b: @Vector(16, u8)) u64 {
        return vectorToBits128(bool, a == b);
    }

    /// Compares each element of the two vectors producing a mask,
    /// where bit is set to 1 if the elements don't equal.
    pub inline fn notEqlToBits128(a: @Vector(16, u8), b: @Vector(16, u8)) u64 {
        return vectorToBits128(bool, a != b);
    }

    /// If there's at least one `a` element that is more than `b` element,
    /// returns a non-zero value. Otherwise, returns 0.
    pub inline fn greaterThan128(a: @Vector(16, u8), b: @Vector(16, u8)) bool {
        var bClone = b;

        // TODO: check avx2 penalty because of 128-bit registers

        const comparedVector = asm (
            \\ pminub %[a], %[b]
            \\ pxor %[a], %[b]
            : [result] "=v" (-> @Vector(16, u8)),
              [b] "=&v" (bClone),
            : [a] "v" (a),
        );
        return vectorToBits128(u8, comparedVector) != 0;
    }

    /// Compares each element of the two vectors producing a mask,
    /// where bit is set to 1 if the elements equal.
    pub inline fn eqlToBits512(a: @Vector(64, u8), b: @Vector(64, u8)) u64 {
        var result: u64 = undefined;
        _ = asm (
            \\ vpcmpequb %[a], %[b], %[mask]
            \\ kmovq %[mask], %[result]
            : [result] "=r" (result),
              [mask] "=k" (-> u64),
            : [a] "v" (a),
              [b] "v" (b),
        );
        return result;
    }
    /// Compares each element of the two vectors producing a mask, a
    /// where bit is set to 1 if the elements don't equal.
    pub inline fn notEqlToBits512(a: @Vector(64, u8), b: @Vector(64, u8)) u64 {
        var result: u64 = undefined;
        _ = asm (
            \\ vpcmpnequb %[a], %[b], %[mask]
            \\ kmovq %[mask], %[result]
            : [result] "=r" (result),
              [mask] "=k" (-> u64),
            : [a] "v" (a),
              [b] "v" (b),
        );
        return result;
    }

    /// If there's at least one `a` element that is more than `b` element,
    /// returns a `true` value. Otherwise, returns `false`.
    pub inline fn greaterThan512(a: @Vector(64, u8), b: @Vector(64, u8)) bool {
        return @reduce(.Or, a > b);
    }

    /// Does bitwise AND between `a` and `b` and returns
    /// a 64-bit mask, where 0 is at positions where `a[index] & b[index] == 0`
    /// and 1 is at positions where `a[index] & b[index] != 0`
    pub inline fn andToBits512(a: @Vector(64, u8), b: @Vector(64, u8)) u64 {
        var mask: u64 = undefined;
        return asm (
            \\ vptestmb %[b], %[a], %[mask]
            \\ kmovq %[mask], %[result]
            : [result] "=r" (-> u64),
              [mask] "=k" (mask),
            : [a] "v" (a),
              [b] "v" (b),
        );
    }

    /// `T` is `bool` or `u8`.
    ///
    /// Returns a bit mask, where 1 only at indexes of `vector` elements that are `true`.
    inline fn vectorToBits128(comptime T: type, vector: @Vector(16, T)) u32 {
        // TODO: check avx2 penalty because of 128-bit registers
        return asm ("pmovmskb %[vector], %[result]"
            : [result] "=r" (-> u64),
            : [vector] "v" (vector),
        );
    }
};

const aarch64 = struct {
    /// Returns `true` when 128-bit vector-shuffle is supported on `aarch64`.
    pub inline fn is128BitVector() bool {
        return Target.aarch64.featureSetHas(CPU.features, .neon);
    }

    /// More preferred than `is128BitVector_aarch64` result.
    ///
    /// Returns `true` only when the `aarch64` target supports vectors with variable length (128-512 bit),
    /// and only when the target supports 32-64 byte shuffles with them.
    pub inline fn isVariableLenVector() bool {
        return Target.aarch64.featureSetHas(CPU.features, .sve2);
    }
    /// Calling this function without checking `isVariableVectorLen_aarch64` is illegal.
    ///
    /// Returns the length in bytes of one vector registers.
    ///
    /// The result of this function should never be persisted 'cause it varies
    /// accross the CPU threads, and if the OS moves the parser
    /// to another thread during a context switch, the result can change.
    pub inline fn getVariableVectorLen() usize {
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
    pub inline fn shuffleVector128(
        vector: @Vector(16, u8),
        mask: @Vector(16, u8),
    ) @Vector(16, u8) {
        return asm ("tbl %[result].16b, { %[vector].16b }, %[mask].16b"
            : [result] "=w" (-> @Vector(16, u8)),
            : [vector] "w" (vector),
              [mask] "w" (mask),
        );
    }

    pub inline fn greaterThan128(a: @Vector(16, u8), b: @Vector(16, u8)) bool {
        // TODO: check asm

        return @reduce(.Max, a > b);
    }

    /// Compares every byte of the two vectors using `operation`,
    /// and if they are equal, sets bit of their position
    /// (e.g, the second bit if the second elements are compared)
    /// in the resulting mask to `1`.
    pub inline fn compareToBits128(
        comptime operation: CompareOperation,
        a: @Vector(16, u8),
        b: @Vector(16, u8),
    ) u64 {
        // TODO: optimize

        const comparedVector = switch (comptime operation) {
            .Eql => a == b,
            .NotEql => a != b,
            .GreaterThan => a > b,
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

        const lowHalfVector: @Vector(8, u8) = singleBitsVector[0..8].*;
        const highHalfVector: @Vector(8, u8) = singleBitsVector[8..].*;

        // Every byte has only one unique bit set to 1,
        // so `Add` accross all elements is the same
        // as `Or` accross the elements:
        // `0001` + `0010` = `0011` = `0001` | `0010`
        const lowHalfMask: u64 = @reduce(.Add, lowHalfVector);
        const highHalfMask: u64 = @reduce(.Add, highHalfVector);

        return (highHalfMask << 8) | lowHalfMask;
    }
};

pub inline fn isMulCarrylessSupported() bool {
    return switch (CPU.arch) {
        .x86_64 => Target.x86.featureSetHas(CPU.features, .pclmul),
        .aarch64 => Target.aarch64.featureSetHasAny(CPU.features, .{
            Target.aarch64.Feature.sve_aes,
            Target.aarch64.Feature.sve_aes2,
        }),
        else => false,
    };
}
/// Carry-less multiplication of two integers.
pub inline fn mulCarryless(a: u64, b: u64) u64 {
    switch (CPU.arch) {
        .x86_64 => if (Target.x86.featureSetHas(CPU.features, .pclmul)) {
            const aVector: @Vector(2, u64) = .{ a, 0 };
            var bVector: @Vector(2, u64) = .{ b, 0 };

            asm volatile ("pclmulqdq $0x00, %[a], %[b]" // `0x00` means low bits of vectors are multiplied
                : [b] "+x" (bVector),
                : [a] "x" (aVector),
            );
            return bVector[0];
        },

        // TODO: check correctness of the features
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
        else => {},
    }

    @compileError("Unsupported architecture");
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
