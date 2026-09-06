const Tokenizer = @This();

const std = @import("std");
const math = std.math;
const Target = std.Target;
const builtin = @import("builtin");
const simdUtils = @import("simdUtils.zig");

const CPU = builtin.cpu;

/// The control characters of JSON.
const CONTROL_CHARS = [_]u8{ '[', ']', '{', '}', ':', ',' };

/// Doesn't containg the full source.
/// Instead, it starts with the end of the previously handled part.
source: []const u8,

/// Mask, representing positions of control chars in `source`.
///
/// Bits of it which set to 1 only if they contain a control char.
///
/// To find the `source` index of a bit from this mask, `@ctz` is used.
///
/// Reseted to 0 when control chars of the SIMD chunk are out.
controlCharsMask: u64,

/// Contains `true` when `Tokenizer.controlCharsMask`
/// ends with an opened string, or `false` if doesn't.
isStringOpened: bool,

pub fn init(source: []const u8) Tokenizer {
    return .{ .source = source };
}

/// Returns index of the next JSON control character
/// or `null` in case of the `source` end.
pub fn next(self: *Tokenizer) ?usize {
    const source = self.source;

    simd: switch (comptime CPU.arch) {
        .x86_64 => switch (comptime simdUtils.getVectorLen_x64()) {
            null => break :simd,

            // 16 byte and 64 byte variations
            // have quite the same 'shuffle' instructions
            16, 64 => |vectorLen| {
                const prevControlCharsMask = self.controlCharsMask;
                if (prevControlCharsMask != 0) {
                    const charIndex = @ctz(prevControlCharsMask);
                    self.controlCharsMask = omitTrailingBit(prevControlCharsMask);
                    return charIndex;
                }

                const Chunk = @Vector(vectorLen, u8);

                if (Chunk.len > source.len) break :simd;

                // TODO: check in ASM output if LLVM doesn't move tables initialization from loop

                const controlCharTables = comptime genControlCharTables();

                const controlCharLowNibbleTable = comptime block: {
                    const tableArray = controlCharTables.lowNibbles;

                    const tableVector: @Vector(tableArray.len, u8) = tableArray;
                    break :block simdUtils.expandComptimeVector(tableVector, Chunk.len);
                };
                const controlCharHighNibbleTable = comptime block: {
                    const tableArray = controlCharTables.highNibbles;

                    const tableVector: @Vector(tableArray.len, u8) = tableArray;
                    break :block simdUtils.expandComptimeVector(tableVector, Chunk.len);
                };

                const chunk: Chunk = source[0..Chunk.len].*;

                const vectorToBits = comptime switch (Chunk.len) {
                    64 => simdUtils.vectorToBits512_x64,
                    16 => simdUtils.vectorToBits128_x64,
                    else => unreachable,
                };

                const backslashesMask: u64 =
                    vectorToBits(chunk == simdUtils.splatVector(Chunk.len, '\\'));

                const stringsMask: u64 = block: {
                    const escapedCharsMask = getEscapedCharsMask(backslashesMask);

                    const quotesMask: u64 =
                        vectorToBits(chunk == simdUtils.splatVector(Chunk.len, '"'));

                    // Non-escaped quotes of strings
                    const stringQuotesMask = quotesMask & ~escapedCharsMask;

                    // Prefix XOR fills all bits between quotes with 1
                    const stringsMask = getBitsPrefixXor(stringQuotesMask);

                    // E.g, `stringsMask` of the current chunk is:
                    // `abc", "def",`
                    // `000111100011`, and it's incorrect - `abc` was opened before.
                    // So invert it (`~stringsMask`):
                    // `111000011100`
                    break :block if (self.isStringOpened) ~stringsMask else stringsMask;
                };

                const chunkAnyControlCharsMask: u64 = block: {
                    const chunkLowNibbles = simdUtils.getLowNibblesVector(Chunk.len, chunk);
                    const chunkHighNibbles = simdUtils.getHighNibblesVector(Chunk.len, chunk);

                    const shuffleVector = comptime switch (vectorLen) {
                        64 => simdUtils.shuffleVector512_x64,
                        16 => simdUtils.shuffleVector128_x64,
                        else => unreachable,
                    };

                    const chunkLowNibblesMatch = shuffleVector(controlCharLowNibbleTable, chunkLowNibbles);
                    const chunkHighNibblesMatch = shuffleVector(controlCharHighNibbleTable, chunkHighNibbles);

                    break :block vectorToBits((chunkLowNibblesMatch & chunkHighNibblesMatch) != 0);
                };

                const chunkControlCharsMask = chunkAnyControlCharsMask & ~stringsMask;
                if (chunkControlCharsMask != 0) {
                    const charIndex = @ctz(chunkControlCharsMask);

                    self.controlCharsMask = chunkControlCharsMask;
                    self.isStringOpened = (stringsMask & 1) == 1;

                    return charIndex;
                }

                // TODO: call `next` recursively or return something
            },
            32 => {},
            else => unreachable,
        },
    }
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

        // TODO: fix flag update logic
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

/// Returns a mask, where 1 are only at bit indexes of escaped chars.
///
/// Ignores even backslash sequences (when a backslash escapes another backslash).
///
/// That is, if JSON input is `"\\key": "\\\\"`, this function
/// understands that nothing significant but only backslashes are escaped,
/// and returns `0`.
///
/// For detailed explanation of this function, see https://arxiv.org/html/1902.08318v7#S3.
inline fn getEscapedCharsMask(backslashesMask: u64) u64 {
    const evenBitsMask = comptime genEvenBitsMask();
    const oddBitsMask = comptime ~evenBitsMask;

    const backslashesStarts = getStartsOfMaskSequences(backslashesMask);

    // Mask of backslash sequences starting with an even bit index,
    // containing only odd amounts of backslashes
    const evenEscapedCharsMask = block: {
        // Leave only backslashes starting at even indexes
        const evenBackslashesStarts = backslashesStarts & evenBitsMask;

        // Contains ends of backslash sequences, starting with an even bit index,
        // and the ends are shifted to the left by 1
        const evenBackslashesEnds = getEndsOfMaskSequences(
            backslashesMask,
            evenBackslashesStarts,
        );

        // If a backslash sequence starts with an even bit index (`evenBackslashesStarts`),
        // it has an odd amount of backslashes ONLY if it ends at an odd bit index,
        // and `evenBackslashesEnds & oddBitsMask` perfectly checks it
        break :block evenBackslashesEnds & oddBitsMask;
    };

    const oddEscapedCharsMask = block: {
        // Leave only backslashes starting at odd indexes
        const oddBackslashesStarts = backslashesStarts & oddBitsMask;

        // Contains ends of backslash sequences, starting with an even bit index,
        // and the ends are shifted to the left by 1
        const oddBackslashesEnds = getEndsOfMaskSequences(
            backslashesMask,
            oddBackslashesStarts,
        );

        // If a backslash sequence starts with an odd bit index (`oddBackslashesStarts`),
        // it has an odd amount of backslashes ONLY if it ends at an even bit index,
        // and the bitwise AND below perfectly checks it
        break :block oddBackslashesEnds & evenBitsMask;
    };

    return evenEscapedCharsMask | oddEscapedCharsMask;
}

/// For each sequence of `1` bits in an unsigned integer `mask`,
/// leaves only the first least significant bit of the sequence.
///
/// Example:
/// For `01101111` returns `00100001`
///
/// (Left bits: most significant, Right bits: least significant).
inline fn getStartsOfMaskSequences(mask: u64) u64 {
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
inline fn getEndsOfMaskSequences(mask: u64, startsMask: u64) u64 {
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
///
/// It is the same as `for(.{0,0,0,1,0,0,1,0}, 0..) |el, i| result[i] ^= el;`.
inline fn getBitsPrefixXor(bits: u64) u64 {
    // Carryless multiplying by a constant value of N bits, where every bit is `1`,
    // shifts `mask` N times and does XOR between shifting results,
    // which is a prefix XOR at hardware level
    return mulCarryless(
        bits,
        math.maxInt(@TypeOf(bits)),
    );
}

/// Carryless multiplication of two integers less than 64 bits.
inline fn mulCarryless(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
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
        .aarch64 => if (Target.aarch64.featureSetHas(CPU.features, .neon)) {
            const aVector: @Vector(2, u64) = .{ a, 0 };
            const bVector: @Vector(2, u64) = .{ b, 0 };

            const resultVector = asm ("pmull %[result].1q, %[a].1d, %[b].1d"
                : [result] "=w" (-> @Vector(2, u64)),
                : [a] "w" (aVector),
                  [b] "w" (bVector),
            );
            return resultVector[0];
        },

        // TODO: a software realization if it turns out to be needed
        else => @compileError("Unsupported architecture"),
    }
}

/// Checks if `T` is unsigned.
///
/// Returns a comptime mask of type `T`, where every bit at even index is `1`.
fn genEvenBitsMask() u64 {
    // Division a value where all bits are 1 (max value) by 3
    // results in a sequence of bits where only even bits are set to 1

    return math.maxInt(u64) / 3;
}

/// Omits the least significant bit of `bits` integer that is set to 1.
///
/// Always returns 0 for 0.
///
/// Example:
/// For `00100010` returns `00100000`
///
/// (Left bits: most significant, Right bits: least significant).
inline fn omitTrailingBit(bits: u64) u64 {
    return bits & (bits - 1);
}

/// Fills the high `byte` bits with 0, leaving only the low nibble.
inline fn getLowNibble(byte: u8) u8 {
    return byte & 0b00001111;
}
/// Moves the high `byte` bits to the low bits, filling the previous place of high bits with 0.
inline fn getHighNibble(byte: u8) u8 {
    return byte >> 4;
}
