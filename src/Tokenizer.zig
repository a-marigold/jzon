const Tokenizer = @This();
// TODO: clear docs and comments
const std = @import("std");
const math = std.math;
const Target = std.Target;
const builtin = @import("builtin");
const simdUtils = @import("simdUtils.zig");

const CPU = builtin.cpu;

/// Contains `LOW_NIBBLE_TABLE` and `HIGH_NIBBLE_TABLE` constant-arrays,
/// indexes of which are low or high nibbles of JSON control chars and whitespaces,
/// and the values at indexes are unique flags.
///
/// Contains `CONTROL_FLAG` and `WHITESPACE_FLAG` for identifying groups.
/// E.g to identify does a char 0x16 belong to the whitespace group,
/// do `((LOW_NIBBLE_TABLE[0x6] | HIGH_NIBBLE_TABLE[0x1]) & WHITESPACE_FLAG) != 0`.
///
/// Values (flags) are allocated so that there is no
/// a UTF-8 char expect control chars nibbles of which
/// give `true` when are looked up in the tables.
///
/// - `lowNibbles` have 16 elements 'cause the maximum
/// low nibble of ASCII is 0xF (decimal `16`).
///
/// - `highNibbles` have 8 elements 'cause the maximum
/// high nibble of ASCII is 0x7 (decimal `7`).
///
/// - E.g, char `{` is 0x7B (decimal 123), and its low and high
/// nibbles perfectly fit 0xB and 0x7 appropriatly.
///
/// Used as a lookup-table vector, from which the vector-shuffle intruction
/// builds a new vector for searching control characters (see `next` function).
const JSON_CHAR_TABLES = block: {
    var lowNibbleTable: [16]u8 = @splat(0);
    var highNibbleTable: [8]u8 = @splat(0);

    var groupIndex = 0;

    // Nibbles of chars of every group doesn't intersect,
    // so it rules out false-positive chars
    const controlGroups = .{
        .{ '[', ']', '{', '}' },
        .{','},
        .{':'},
    };

    const controlFlag = 0;

    while (groupIndex < controlGroups.len) : (groupIndex += 1) {
        const groupFlag = 1 << groupIndex;

        for (controlGroups[groupIndex]) |char| {
            lowNibbleTable[getLowNibble(char)] = groupFlag;
            highNibbleTable[getHighNibble(char)] = groupFlag;
        }

        controlFlag |= groupFlag;
    }

    const whitespaceGroups = .{
        .{ '\n', '\r', '\t' },
        .{' '},
    };

    var whitespaceFlag = 0;

    while (groupIndex < whitespaceGroups.len) : (groupIndex += 1) {
        const groupFlag = 1 << groupIndex;

        for (whitespaceGroups[groupIndex]) |char| {
            // Bitwise OR `|=` is needed 'cause:
            // - `\n` (0x0A) and `:` (0x3A) have the same low nibbles,
            // - ` ` (Space, 0x20) and `,` (0x2C) have the same high nibbles,
            // - `\r` (0x0D), `]` (0x5D) and `}` (0x7D) have the same low nibbles.
            //
            // Doing bitwise OR between flags doesn't cause appearance of false-positive chars.
            // This is because:
            // - ` ` (Space) and `,` contain the same high nibbles,
            // that is, grouping them via bitwise OR can't cause false-positive chars.
            // - For other chars, `HIGH_NIBBLE_TABLE` contains only unique flags,
            // while `LOW_NIBBLE_TABLE` contains flags, grouped via bitwise OR.
            // That is, when it seems like `LOW_NIBBLE_TABLE` returns a valid flag for a false-positive char,
            // doing bitwise AND with the flag and the `HIGH_NIBBLE_TABLE` result filters out the false-positive char

            lowNibbleTable[getLowNibble(char)] |= groupFlag;
            highNibbleTable[getHighNibble(char)] |= groupFlag;
        }

        whitespaceFlag |= groupFlag;
    }

    break :block struct {
        pub const LOW_NIBBLE_TABLE = lowNibbleTable;
        pub const HIGH_NIBBLE_TABLE = highNibbleTable;

        pub const CONTROL_FLAG = controlFlag;
        pub const WHITESPACE_FLAG = whitespaceFlag;
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
/// If the tokenizer currently in a string
/// or a sequence of whitespaces (trivia), returns `NEXT_TRIVIA`.
/// If the JSON `source` ends, returns `NEXT_END`.
///
/// List of control chars, indexes of which can be returned:
/// - `[`, `]`, `{`, `}`, `,`, `:`.
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
                    const tableArray = JSON_CHAR_TABLES.LOW_NIBBLE_TABLE;
                    const tableVector: @Vector(tableArray.len, u8) = tableArray;
                    break :block simdUtils.expandVector(tableVector, Chunk.len);
                };
            }.TABLE;

            const controlCharHighNibbleTable: @Vector(Chunk.len, u8) = struct {
                const TABLE = block: {
                    const tableArray = JSON_CHAR_TABLES.HIGH_NIBBLE_TABLE;
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

/// Returns a mask, where 1 at bit indexes of chars inside strings.
///
/// Ending quotes of strings are set to 0.
inline fn getStringsMask(anyQuotesMask: u64, backslashesMask: u64) u64 {
    const unescapedQuotesMask = anyQuotesMask & ~getEscapedCharsMask(backslashesMask);

    // Prefix XOR fills all bits between quotes with 1
    return getBitsPrefixXor(unescapedQuotesMask);
}

/// Returns a mask, where 1 is only at bit indexes of chars, escaped inside strings.
///
/// Ignores even backslash sequences (when a backslash escapes another backslash).
/// That is, if JSON input is `"\\key": "\\\\"`, this function
/// understands that nothing significant but only backslashes are escaped,
/// and returns `0`.
///
/// For a detailed explanation of this function, see https://arxiv.org/html/1902.08318v7#S3.
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

/// Returns a mask where every bit of control chars and starts of JSON values is set to 1.
///
/// The result doesn't include ends of JSON values.
///
/// Any opaque sequence of chars that are not control or whitespaces is treated as a JSON value.
///
/// That is, for `"abc"123`, `10000000` is returned, 'cause it is treated as a single value.
/// The same is for `truefalse123"string"`, `nullfalse` and the like.
///
/// `anyControlCharsMask` and `anyWhitespacesMask` can contain
/// chars inside strings, which are filtered by this function.
inline fn getControlAndValueCharsMask(anyControlCharsMask: u64, anyWhitespacesMask: u64, stringsMask: u64) u64 {
    // { "\\\"Nam[{": [ 116,"\\\\" , 234, "true", false ], "t":"\\\"" }
    // __1111111111_________11111_________11111____________11__11111___ S
    // 1____________1_1____1_______1____1_______1_______11____1_______1 C = anyC & ~S
    // _1____________1_1__________1_1____1_______1_____1__1__________1_ W = anyW & ~S
    // 11___________1111___1______111___11______11_____1111___1______11 CW = C | W
    // _11___________1111___1______111___11______11_____1111___1______1 V = CW << 1
    // 1_111111111111_1_1111111111_1_1111_1111111_11111_11_1111111111_1 W = ~W
    // __1____________1_1___1______1_1____1_______1_____11_1___1______1 V &= W

    const controlAndSpacesMask = (anyControlCharsMask | anyWhitespacesMask) & ~stringsMask;

    return (controlAndSpacesMask << 1) & ~anyWhitespacesMask;
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
/// but contains ends shifted to the left by 1. That is the real ends are at `result >> 1`.
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
inline fn getBitsPrefixXor(bits: u64) u64 {
    // Carryless multiplication of `bits` by ~0 (every bit is 1)
    // shifts `mask` as many times as wide the ~0 (64) and does XOR between shifting results,
    // and it's a prefix XOR at hardware level
    if (simdUtils.isMulCarrylessSupported())
        return simdUtils.mulCarryless(bits, ~0);

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
/// Returns an unique byte-flag with only a single `1` at `bitOffset`.
fn getByteFlag(bitOffset: comptime_int) u8 {
    return 0b00000001 << bitOffset;
}

/// Omits the least significant bit of `bits` integer that is set to 1.
///
/// Always returns 0 for 0.
///
/// Example: For `00100010` returns `00100000`
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
