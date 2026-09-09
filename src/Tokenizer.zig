const Tokenizer = @This();

const std = @import("std");
const math = std.math;
const Target = std.Target;
const builtin = @import("builtin");
const simdUtils = @import("simdUtils.zig");

const CPU = builtin.cpu;

/// Contains `LOW_NIBBLE_TABLE` and `HIGH_NIBBLE_TABLE` constant-arrays,
/// indexes of which are low or high nibbles of JSON control chars,
/// and the values at indexes are unique masks.
///
/// Values (flags) are allocated so that there is no
/// a UTF-8 char expect control chars nibbles of which
/// give `true` when are looked up in the tables.
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
const CONTROL_CHAR_NIBBLE_TABLES = block: {
    var lowNibbleTable: [16]u8 = @splat(0);
    var highNibbleTable: [8]u8 = @splat(0);

    const groups = .{
        .{ '[', ']', '{', '}' },
        .{','},
        .{':'},
    };

    for (0..groups.len) |index| {
        const groupFlag = 1 << index;
        for (groups[index]) |char| {
            lowNibbleTable[getLowNibble(char)] = groupFlag;
            highNibbleTable[getHighNibble(char)] = groupFlag;
        }
    }

    break :block struct {
        pub const LOW_NIBBLE_TABLE = lowNibbleTable;
        pub const HIGH_NIBBLE_TABLE = highNibbleTable;
    };
};

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
    return .{
        .source = source,
        .isStringOpened = false,
    };
}

/// `next` function returns this value to indicate the end of `source`.
pub const NEXT_END: usize =
    @intCast(-1);

/// `next` function returns this value to indicate that
/// the current SIMD chunk or scalar symbol of `source`
/// is inside a string or trivia (a sequence of whitespaces and other trivial chars).
pub const NEXT_TRIVIA: usize =
    @intCast(-2);

/// Returns index of the next JSON control character.
///
/// If the tokenizer currently in string, returns `NEXT_IN_STRING`.
///
/// If the JSON `source` ends, returns `NEXT_END`.
pub fn next(self: *Tokenizer) usize {
    const source = self.source;

    simd: switch (comptime CPU.arch) {
        .x86_64 => if (comptime simdUtils.getVectorLen_x64()) |vectorLen| {
            const prevControlCharsMask = self.controlCharsMask;
            if (prevControlCharsMask != 0) {
                const charIndex = @ctz(prevControlCharsMask);
                self.controlCharsMask = omitTrailingBit(prevControlCharsMask);
                return charIndex;
            }

            // AVX2 (vectorLen == 32) has a specific shuffle vector instruction
            // and uses not all 32 bytes of a SIMD chunk (see the code below)
            const shuffleVectorLen = if (vectorLen == 32) 16 else vectorLen;

            const Chunk = @Vector(shuffleVectorLen, u8);

            if (Chunk.len > source.len) break :simd;

            // TODO: check in ASM output if LLVM doesn't move tables initialization from loop

            const controlCharLowNibbleTable: @Vector(Chunk.len, u8) = struct {
                const TABLE = block: {
                    const tableArray = CONTROL_CHAR_NIBBLE_TABLES.LOW_NIBBLE_TABLE;
                    const tableVector: @Vector(tableArray.len, u8) = tableArray;
                    break :block simdUtils.expandVector(tableVector, Chunk.len);
                };
            }.TABLE;

            const controlCharHighNibbleTable: @Vector(Chunk.len, u8) = struct {
                const TABLE = block: {
                    const tableArray = CONTROL_CHAR_NIBBLE_TABLES.HIGH_NIBBLE_TABLE;
                    const tableVector: @Vector(tableArray.len, u8) = tableArray;
                    break :block simdUtils.expandVector(tableVector, Chunk.len);
                };
            }.TABLE;

            const compareToBits = comptime switch (Chunk.len) {
                64 => simdUtils.compareToBits512_x64,
                16 => simdUtils.compareToBits128_x64,
                else => unreachable,
            };

            const chunk: Chunk = source[0..Chunk.len].*;

            const stringsMask: u64 = block: {
                const backslashesMask: u64 = compareToBits(.Eql, chunk, @splat('\\'));

                const escapedCharsMask = getEscapedCharsMask(backslashesMask);

                const anyQuotesMask = compareToBits(.Eql, chunk, @splat('"'));

                const unescapedQuotesMask = anyQuotesMask & ~escapedCharsMask;

                // Prefix XOR fills all bits between quotes with 1
                const stringsMask = getBitsPrefixXor(unescapedQuotesMask);

                // For example, `stringsMask` of the current chunk is:
                // `abc", "def",`
                // `000111100011`, and it's incorrect - `abc` was opened before.
                // So invert it (`~stringsMask`):
                // `111000011100`
                break :block if (self.isStringOpened) ~stringsMask else stringsMask;
            };

            const chunkAnyControlCharsMask: u64 = block: {
                const chunkLowNibbles = simdUtils.getLowNibblesVector(chunk);
                const chunkHighNibbles = simdUtils.getHighNibblesVector(chunk);

                switch (comptime vectorLen) {
                    64 => {
                        const chunkLowNibblesMatch =
                            simdUtils.shuffleVector512_x64(controlCharLowNibbleTable, chunkLowNibbles);
                        const chunkHighNibblesMatch =
                            simdUtils.shuffleVector512_x64(controlCharHighNibbleTable, chunkHighNibbles);

                        // TODO: Replace it with `VPTESTMB` producing a bit mask from vectors bitwise AND
                        break :block compareToBits(.NotEql, chunkLowNibblesMatch & chunkHighNibblesMatch, @splat(0));
                    },
                    32 => {
                        const chunkNibblesMatch = simdUtils.shuffleVector256_x64(
                            controlCharLowNibbleTable ++ controlCharHighNibbleTable,
                            chunkLowNibbles ++ chunkHighNibbles,
                        );

                        const chunkLowNibblesMatch = chunkNibblesMatch[0..16];
                        const chunkHighNibblesMatch = chunkNibblesMatch[16..];

                        break :block compareToBits(.NotEql, chunkLowNibblesMatch & chunkHighNibblesMatch, @splat(0));
                    },
                    16 => {
                        const chunkLowNibblesMatch =
                            simdUtils.shuffleVector512_x64(controlCharLowNibbleTable, chunkLowNibbles);
                        const chunkHighNibblesMatch =
                            simdUtils.shuffleVector512_x64(controlCharHighNibbleTable, chunkHighNibbles);

                        break :block compareToBits(.NotEql, chunkLowNibblesMatch & chunkHighNibblesMatch, @splat(0));
                    },
                    else => unreachable,
                }
            };

            self.isStringOpened = (stringsMask & 1) == 1;

            const controlCharsMask = chunkAnyControlCharsMask & ~stringsMask;
            if (controlCharsMask != 0) {
                const charIndex = @ctz(controlCharsMask);
                self.controlCharsMask = omitTrailingBit(controlCharsMask);
                return charIndex;
            } else return NEXT_TRIVIA;
        },
    }
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
    // Carryless multiplying by a constant value of N bits, where every bit is `1` (max value),
    // shifts `mask` N times and does XOR between shifting results,
    // which is a prefix XOR at hardware level
    if (simdUtils.isMulCarrylessSupported())
        // TODO: maybe only bits at power of two indexes are faster than max value
        return simdUtils.mulCarryless(bits, comptime math.maxInt(@TypeOf(bits)));

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
