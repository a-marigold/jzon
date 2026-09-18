const Tokenizer = @This();
// TODO: clear docs and comments
const std = @import("std");
const builtin = @import("builtin");
const utils = @import("utils.zig");
const simd = @import("simd.zig");

const CPU = builtin.cpu;

/// Contains `LOW_NIBBLE_TABLE` and `HIGH_NIBBLE_TABLE` constant-arrays,
/// indexes of which are low or high nibbles of JSON control chars and whitespaces,
/// and the values at indexes are unique flags.
///
/// Contains `CONTROL_CHARS_FLAG` and `WHITESPACE_FLAG` for identifying groups.
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
            lowNibbleTable[utils.getLowNibble(char)] = groupFlag;
            highNibbleTable[utils.getHighNibble(char)] = groupFlag;
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

            lowNibbleTable[utils.getLowNibble(char)] |= groupFlag;
            highNibbleTable[utils.getHighNibble(char)] |= groupFlag;
        }

        whitespaceFlag |= groupFlag;
    }

    break :block struct {
        // Align 'cause it's read to vector registers (with 16, 32, 64 bytes widths)
        pub const LOW_NIBBLE_TABLE: [16]u8 align(64) = lowNibbleTable;
        pub const HIGH_NIBBLE_TABLE: [8]u8 align(64) = highNibbleTable;

        pub const CONTROL_CHARS_FLAG = controlFlag;
        pub const WHITESPACE_FLAG = whitespaceFlag;
    };
};

/// Doesn't contain the full source.
/// Instead, it starts with the end of the previously handled part.
source: []const u8,

/// Mask, representing positions of control chars in `source`.
///
/// Bits of it are set to `1` only if they
/// contain a control char or a start of JSON value.
controlAndValueCharsMask: u64,

/// Contains all bits set to `1` when the current SIMD-chunk
/// ends with an unclosed string, or all bits set to `0` if it doesn't.
isStringOpened: u64,

/// Contains number `1` when the current SIMD-chunk has
/// an unclosed string and the last char of this string
/// has a backslash which escapes a char of the next SIMD-chunk.
///
/// Contains `0` if the current SIMD-chunk
/// doesn't end with an unclosed string or
/// the unclosed string doesn't have a backslash like that.
///
/// Used to handle string escaping accros SIMD-chunks.
isStringEndedWithEscaping: u64,

pub fn init(source: []const u8) Tokenizer {
    return .{
        .source = source,
        .isStringOpened = 0,
        .isStringEndedWithEscaping = 0,
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
        .x86_64 => if (comptime simd.x86.getVectorLen()) |vectorLen| {
            const prevControlAndValueCharsMask = self.controlAndValueCharsMask;
            if (prevControlAndValueCharsMask != 0) {
                const charIndex = utils.getTrailBitIndex(prevControlAndValueCharsMask);
                self.controlAndValueCharsMask = utils.omitTrailBit(prevControlAndValueCharsMask);
                return charIndex;
            }

            // AVX2 (vectorLen == 32) has a specific shuffle vector instruction
            // and uses not all 32 bytes of a SIMD chunk (see the code below)
            const shuffleVectorLen = if (vectorLen == 32) 16 else vectorLen;

            const Chunk = @Vector(shuffleVectorLen, u8);

            if (Chunk.len > source.len) break :simd;

            const chunk: Chunk = source[0..Chunk.len].*;

            // TODO: check in ASM output if LLVM doesn't move tables initialization from loop

            const jsonCharLowNibbleTable: @Vector(Chunk.len, u8) = struct {
                const TABLE = block: {
                    const tableArray = JSON_CHAR_TABLES.LOW_NIBBLE_TABLE;
                    const tableVector: @Vector(tableArray.len, u8) = tableArray;
                    break :block simd.expandVector(tableVector, Chunk.len);
                };
            }.TABLE;

            const jsonCharHighNibbleTable: @Vector(Chunk.len, u8) = struct {
                const TABLE = block: {
                    const tableArray = JSON_CHAR_TABLES.HIGH_NIBBLE_TABLE;
                    const tableVector: @Vector(tableArray.len, u8) = tableArray;
                    break :block simd.expandVector(tableVector, Chunk.len);
                };
            }.TABLE;

            const eqlToBits, const notEqlToBits = comptime switch (Chunk.len) {
                64 => .{ simd.x86.eqlToBits512, simd.x86.notEqlToBits512 },
                16 => .{ simd.x86.eqlToBits128, simd.x86.notEqlToBits128 },
                else => unreachable,
            };

            const anyControlCharsMask: u64, const anyWhitespacesMask: u64 = block: {
                const lowNibbles = simd.getLowNibblesVector(chunk);
                const highNibbles = simd.getHighNibblesVector(chunk);

                switch (comptime vectorLen) {
                    64 => {
                        const lowNibblesMatch =
                            simd.x86.shuffleVector512(jsonCharLowNibbleTable, lowNibbles);
                        const highNibblesMatch =
                            simd.x86.shuffleVector512(jsonCharHighNibbleTable, highNibbles);

                        const jsonCharsMatch = lowNibblesMatch & highNibblesMatch;

                        break :block .{
                            simd.x86.andToBits512(
                                jsonCharsMatch,
                                @splat(JSON_CHAR_TABLES.CONTROL_CHARS_FLAG),
                            ),
                            simd.x86.andToBits512(
                                jsonCharsMatch,
                                @splat(JSON_CHAR_TABLES.WHITESPACE_FLAG),
                            ),
                        };
                    },
                    32 => {
                        const nibblesMatchHalves = simd.x86.shuffleVector256(
                            jsonCharLowNibbleTable ++ jsonCharHighNibbleTable,
                            lowNibbles ++ highNibbles,
                        );

                        // TODO: avx2 penalty because of 128-bit registers
                        const firstHalfMatch: @Vector(16, u8) = nibblesMatchHalves[0..16];
                        const secondHalfMatch: @Vector(16, u8) = nibblesMatchHalves[16..];

                        const jsonCharsMatch = firstHalfMatch & secondHalfMatch;

                        break :block .{
                            notEqlToBits(jsonCharsMatch & JSON_CHAR_TABLES.CONTROL_CHARS_FLAG, @splat(0)),
                            notEqlToBits(jsonCharsMatch & JSON_CHAR_TABLES.WHITESPACE_FLAG, @splat(0)),
                        };
                    },
                    16 => {
                        const lowNibblesMatch =
                            simd.shuffleVector128_x64(jsonCharLowNibbleTable, lowNibbles);
                        const highNibblesMatch =
                            simd.shuffleVector128_x64(jsonCharHighNibbleTable, highNibbles);

                        const jsonCharsMatch = lowNibblesMatch & highNibblesMatch;

                        break :block .{
                            notEqlToBits(jsonCharsMatch & JSON_CHAR_TABLES.CONTROL_CHARS_FLAG, @splat(0)),
                            notEqlToBits(jsonCharsMatch & JSON_CHAR_TABLES.WHITESPACE_FLAG, @splat(0)),
                        };
                    },
                    else => unreachable,
                }
            };

            const stringsMask, const isStringEndedWithEscaping = block: {
                const anyQuotesMask: u64 = eqlToBits(chunk, @splat('"'));
                const backslashMask: u64 = eqlToBits(chunk, @splat('\\'));

                const result = getStringsMask(
                    anyQuotesMask,
                    backslashMask,
                    self.isStringOpened,
                    self.isStringEndedWithEscaping,
                );
                break :block .{ result.stringsMask, result.isStringEndedWithEscaping };
            };

            self.isStringOpened = isStringsMaskOpened(stringsMask);
            self.isStringEndedWithEscaping = isStringEndedWithEscaping;

            const controlAndValueCharsMask = getControlAndValueCharsMask(
                anyControlCharsMask,
                anyWhitespacesMask,
                stringsMask,
            );

            if (controlAndValueCharsMask != 0) {
                const charIndex = utils.getTrailBitIndex(controlAndValueCharsMask);
                self.controlAndValueCharsMask = utils.omitTrailBit(controlAndValueCharsMask);
                return charIndex;
            } else return NEXT_TRIVIA;
        },

        .aarch64 => {
            const is128BitVector = comptime simd.aarch64.is128BitVector();
            const isVariableLenVector = comptime simd.aarch64.isVariableLenVector();

            comptime if (!(is128BitVector or isVariableLenVector)) break :simd;

            const vectorLen = 16;

            if (vectorLen > source.len) break :simd;

            const prevControlAndValueCharsMask = self.controlAndValueCharsMask;
            if (prevControlAndValueCharsMask != 0) {
                const charIndex = utils.getTrailBitIndex(prevControlAndValueCharsMask);
                self.controlAndValueCharsMask = prevControlAndValueCharsMask;
                return charIndex;
            }

            const chunk: @Vector(vectorLen, u8) = source[0..vectorLen].*;

            const jsonCharLowNibbleTable: @Vector(vectorLen, u8) = JSON_CHAR_TABLES.LOW_NIBBLE_TABLE;
            const jsonCharHighNibbleTable =
                simd.expandVector(JSON_CHAR_TABLES.HIGH_NIBBLE_TABLE, vectorLen);

            const anyControlCharsMask, const anyWhitespacesMask = block: {
                const lowNibbles = simd.getLowNibblesVector(chunk);
                const highNibbles = simd.getLowNibblesVector(chunk);

                const lowNibblesMatch = simd.aarch64.shuffleVector128(
                    lowNibbles,
                    jsonCharLowNibbleTable,
                );
                const highNibblesMatch = simd.aarch64.shuffleVector128(
                    highNibbles,
                    jsonCharHighNibbleTable,
                );

                const jsonCharsMatch = lowNibblesMatch & highNibblesMatch;
                break :block .{
                    simd.aarch64.compareToBits128_aarch64(
                        .NotEql,
                        jsonCharsMatch & JSON_CHAR_TABLES.CONTROL_CHARS_FLAG,
                        @splat(0),
                    ),
                    simd.compareToBits128_aarch64(
                        .NotEql,
                        jsonCharsMatch & JSON_CHAR_TABLES.WHITESPACE_FLAG,
                        @splat(0),
                    ),
                };
            };

            const stringsMask, const isStringEndedWithEscaping = block: {
                const anyQuotesMask = simd.compareToBits128_aarch64(
                    .Eql,
                    chunk,
                    @splat('"'),
                );
                const backslashMask = simd.compareToBits128_aarch64(
                    .Eql,
                    chunk,
                    @splat('\\'),
                );

                const result = getStringsMask(
                    anyQuotesMask,
                    backslashMask,
                    self.isStringOpened,
                    self.isStringEndedWithEscaping,
                );
                break :block .{ result.stringsMask, result.isStringEndedWithEscaping };
            };

            const controlAndValueCharsMask = getControlAndValueCharsMask(
                anyControlCharsMask,
                anyWhitespacesMask,
                stringsMask,
            );

            self.isStringOpened = isStringsMaskOpened(stringsMask);
            self.isStringEndedWithEscaping = isStringEndedWithEscaping;

            if (controlAndValueCharsMask != 0) {
                const charIndex = utils.getTrailBitIndex(controlAndValueCharsMask);
                self.controlAndValueCharsMask = utils.omitTrailBit(controlAndValueCharsMask);
                return charIndex;
            } else return NEXT_TRIVIA;
        },
    }
}

/// Returns a mask, where 1 is at bit indexes of chars inside strings.
///
/// Bits of ending quotes of strings are set to 0.
inline fn getStringsMask(
    anyQuotesMask: u64,
    backslashMask: u64,
    isPrevStringOpened: @FieldType(Tokenizer, "isStringOpened"),
    isPrevStringEndedWithEscaping: @FieldType(Tokenizer, "isStringEndedWithEscaping"),
) struct {
    stringsMask: u64,
    isStringEndedWithEscaping: @FieldType(Tokenizer, "isStringEndedWithEscaping"),
} {
    const escapedCharsMask, const isStringEndedWithEscaping = block: {
        const result = getEscapedCharsMask(
            backslashMask,
            isPrevStringEndedWithEscaping,
        );
        break :block .{ result.escapedCharsMask, result.isStringEndedWithEscaping };
    };

    const unescapedQuotesMask = anyQuotesMask & ~escapedCharsMask;

    // Prefix XOR fills all bits between quotes with 1
    const stringsMask = utils.getBitsPrefixXor(unescapedQuotesMask);

    // If the prev SIMD-chunk has an unclosed string,
    // `isPrevStringOpened` contains all bits set to 1,
    // and XOR with the string mask and `isPrevStringOpened` inverts strings
    const actualStringsMask = stringsMask ^ isPrevStringOpened;

    return .{ .stringsMask = actualStringsMask, .isStringEndedWithEscaping = isStringEndedWithEscaping };
}

/// If `stringsMask` contains an opened, unclosed string at the end,
/// returns 64 bits where every bit is set to 1. Otherwise,
/// returns bits with all with zeros.
inline fn isStringsMaskOpened(stringsMask: u64) u64 {
    // If the mask has the most significant bit (MSB) set to 1,
    // it ends with an opened string.
    // Shift the most significant bit by 63 to
    // activate sign extension (`i64` is signed),
    // and if the sign (MSB) is 1, every bit of 64 bits is set to 1,
    // but if the sign is 0, every bit is set to 0
    return @as(i64, @intCast(stringsMask)) >> 63;
}

/// Returns a struct:
/// - `escapedCharsMask` is a mask where 1 is only at bit indexes of chars, escaped inside strings.
/// - `isStringEndedWithEscaping` is the same as `Tokenizer.isStringEndedWithEscaping`, but for the current chunk.
///
/// Ignores even backslash sequences (when a backslash escapes another backslash).
/// That is, if JSON input is `"\\key": "\\\\"`, this function
/// understands that nothing significant but only backslashes are escaped,
/// and returns `0`.
inline fn getEscapedCharsMask(
    backslashMask: u64,
    isPrevStringEndedWithEscaping: @FieldType(Tokenizer, "isStringEndedWithEscaping"),
) struct {
    escapedCharsMask: u64,
    isStringEndedWithEscaping: @FieldType(Tokenizer, "isStringEndedWithEscaping"),
} {
    const evenBitsMask = comptime utils.genEvenBitsMask();
    const oddBitsMask = comptime ~evenBitsMask;

    const backslashStarts = utils.getStartsOfMaskSequences(backslashMask);

    // If `backslashMask` starts with a sequence, which in turn starts
    // straight from the first mask bit (`anyBacksMask & 1`),
    // and if the prev string ends with escaping (isPrevStringEndedWithEscaping),
    // do `backsMask ^ 1` to flip the first bit (that is, to escape the first char).
    // Otherwise, do `backsMask ^ 0` and get unchanged `backsMask`.
    const actualBackslashMask =
        backslashMask ^ ((backslashStarts & 1) * isPrevStringEndedWithEscaping);

    const actualBackslashStarts = utils.getStartsOfMaskSequences(actualBackslashMask);

    // Mask of backslash sequences starting with an even bit index,
    // containing only odd amounts of backslashes
    const evenEscapedCharsMask = block: {
        // Get only backslashes starting at even indexes
        const evenBackslashStarts = actualBackslashStarts & evenBitsMask;

        // Contains ends of backslash sequences, starting with an even bit index,
        // and the ends are shifted to the left by 1
        const evenBackslashEnds = utils.getEndsOfMaskSequences(
            backslashMask,
            evenBackslashStarts,
        );

        // If a backslash sequence starts with an even bit index (`evenBackslashStarts`),
        // it has an odd amount of backslashes ONLY if it ends at an odd bit index,
        // and `evenBackslashEnds & oddBitsMask` perfectly checks it
        break :block evenBackslashEnds & oddBitsMask;
    };

    const oddEscapedCharsMask = block: {
        // Get only backslashes starting at odd indexes
        const oddBackslashStarts = actualBackslashStarts & oddBitsMask;

        // Contains ends of backslash sequences, starting with an even bit index,
        // and the ends are shifted to the left by 1
        const oddBackslashEnds = utils.getEndsOfMaskSequences(
            backslashMask,
            oddBackslashStarts,
        );

        // If a backslash sequence starts with an odd bit index (`oddBackslashStarts`),
        // it has an odd amount of backslashes ONLY if it ends at an even bit index,
        // and the bitwise AND below perfectly checks it
        break :block oddBackslashEnds & evenBitsMask;
    };

    return .{
        .escapedCharsMask = evenEscapedCharsMask | oddEscapedCharsMask,
        .isStringEndedWithEscaping = isBackslashMaskEndedWithEscaping(actualBackslashMask),
    };
}

/// Returns a mask where every bit of control chars and starts of JSON values is set to 1.
///
/// The result doesn't include ends of JSON values.
///
/// Any opaque sequence of chars that are not control or whitespaces is treated as a JSON value.
///
/// That is, for `"abc"123`, `10000000` is returned, 'cause it is treated as a single value.
///
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
    // _11___________1111___1______111___11______11_____1111___1______1 CV = CW << 1
    // 1_111111111111_1_1111111111_1_1111_1111111_11111_11_1111111111_1 W = ~W
    // __1____________1_1___1______1_1____1_______1_____11_1___1______1 CV &= W

    const controlAndSpacesMask = (anyControlCharsMask | anyWhitespacesMask) & ~stringsMask;

    return (controlAndSpacesMask << 1) & ~anyWhitespacesMask;
}

/// `actualBackslashMask` must be already actualized via
/// handling `Tokenizer.isStringEndedWithEscaping`.
///
/// Returns 1 when `backslashMask` ends with a backslash
/// which escapes a char of the next SIMD-chunk.
/// Otherwise, returns 0.
inline fn isBackslashMaskEndedWithEscaping(actualBackslashMask: u64) u64 {
    // Amount of backslashes that are at the very end
    // of the mask and that touch its highest bit
    const endBackslashCount = utils.getLeadBitIndex(~actualBackslashMask);

    // If the count is odd, return 1 (`oddNum & 1 == 1`)
    // Otherwise, return 0 (`evenNum & 1 == 0`)
    return endBackslashCount & 1;
}

/// Returns `true` value when `vector` with JSON chars contains not only the ASCII-chars.
///
/// Otherwise, returns `false`.
inline fn isVectorNonAscii_x64(vector: anytype) bool {
    // All ASCII chars have the highest bit set to 0
    const onlyHighBitsVector: @TypeOf(vector) = @splat(0b10000000);

    return switch (comptime vector.len) {
        // TODO: check the asm output if there are two 'test' instruction because of returning 'bool' instead of a mask
        64 => simd.x86.andToBits512(vector, onlyHighBitsVector) != 0,
        16 => simd.x86.notEqlToBits128(.NotEql, vector & onlyHighBitsVector, 0) != 0,
        else => unreachable,
    };
}
