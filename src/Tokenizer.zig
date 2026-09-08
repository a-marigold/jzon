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

/// `next` function returns this value to indicate the end of `source`.
pub const NEXT_END: usize =
    @intCast(-1);

/// `next` function returns this value to indicate that
/// the current SIMD chunk or scalar symbol of `source` is inside a string.
///
/// - For scalars this appears when the tokenizer
/// is currently inside, e.g, `key` of `"key": "string"`.
///
/// - Strings in SIMD chunks are skipped by the alghorithm,
/// and `next` always returns indexes outside strings.
/// However, JSON inputs sometimes have strings that
/// are much bigger than one chunk.
/// For example, chunk length is 64 bytes, but a string contains 200 bytes, and some iterations over this string
/// are completely inside it, so `next` returns `NEXT_IN_STRING`.
pub const NEXT_IN_STRING: usize =
    @intCast(-2);

/// Returns index of the next JSON control character.
///
/// If the tokenizer currently in string, returns `NEXT_IN_STRING`.
///
/// If the JSON `source` ends, returns `NEXT_END`.
pub fn next(self: *Tokenizer) usize {
    const source = self.source;

    simd: switch (comptime CPU.arch) {
        .x86_64 => switch (comptime simdUtils.getVectorLen_x64()) {
            else => break :simd,

            // TODO: merge 16-, 32-, 64- byte variations

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
                    break :block simdUtils.expandVector(tableVector, Chunk.len);
                };
                const controlCharHighNibbleTable = comptime block: {
                    const tableArray = controlCharTables.highNibbles;

                    const tableVector: @Vector(tableArray.len, u8) = tableArray;
                    break :block simdUtils.expandVector(tableVector, Chunk.len);
                };

                const compareToBits = comptime switch (Chunk.len) {
                    64 => simdUtils.compareToBits128_x64,
                    16 => simdUtils.compareToBits512_x64,
                    else => unreachable,
                };

                const chunk: Chunk = source[0..Chunk.len].*;

                const stringsMask: u64 = block: {
                    const backslashesMask: u64 = compareToBits(.Eql, chunk, @splat('\\'));

                    const escapedCharsMask = getEscapedCharsMask(backslashesMask);

                    const quotesMask = compareToBits(.Eql, chunk, @splat('"'));

                    const unescapedQuotesMask = quotesMask & ~escapedCharsMask;

                    // Prefix XOR fills all bits between quotes with 1
                    const stringsMask = getBitsPrefixXor(unescapedQuotesMask);

                    // E.g, `stringsMask` of the current chunk is:
                    // `abc", "def",`
                    // `000111100011`, and it's incorrect - `abc` was opened before.
                    // So invert it (`~stringsMask`):
                    // `111000011100`
                    break :block if (self.isStringOpened) ~stringsMask else stringsMask;
                };

                const chunkAnyControlCharsMask: u64 = block: {
                    const chunkLowNibbles = simdUtils.getLowNibblesVector(chunk);
                    const chunkHighNibbles = simdUtils.getHighNibblesVector(chunk);

                    const shuffleVector = comptime switch (vectorLen) {
                        64 => simdUtils.shuffleVector512_x64,
                        16 => simdUtils.shuffleVector128_x64,
                        else => unreachable,
                    };

                    const chunkLowNibblesMatch = shuffleVector(controlCharLowNibbleTable, chunkLowNibbles);

                    const chunkHighNibblesMatch = shuffleVector(controlCharHighNibbleTable, chunkHighNibbles);

                    // TODO: Check for instruction producing a bit mask for vectors bitwise AND
                    break :block compareToBits(.NotEql, chunkLowNibblesMatch & chunkHighNibblesMatch, @splat(0));
                };

                const chunkControlCharsMask = chunkAnyControlCharsMask & ~stringsMask;
                if (chunkControlCharsMask != 0) {
                    const charIndex = @ctz(chunkControlCharsMask);

                    self.controlCharsMask = chunkControlCharsMask;
                    self.isStringOpened = (stringsMask & 1) == 1;

                    return charIndex;
                }

                return NEXT_IN_STRING;
            },
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
        const flags: [8]u8 = undefined;

        var flag = 0;
        for (0..flags.len) |index| {
            flag = 1 << index;

            flags[index] = flag;
        }

        break :flagsBlock flags;
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
