const std = @import("std");
const utils = @import("utils.zig");
const simd = @import("simd.zig");
const Tokenizer = @import("Tokenizer.zig");

/// Contains `LEAD_BYTE_LOW_NIBBLE_TABLE`, `LEAD_BYTE_HIGH_NIBBLE_TABLE`,
/// `NEXT_BYTE_HIGH_NIBBLE_TABLE` lookup tables to be used as vectors
/// and `DOUBLE_CONTINUATION_FLAG`.
///
/// Indexes of tables are low, high nibbles of the first (lead) byte of a sequence,
/// and high nibbles of the second (next or continuation) byte of a sequence.
///
/// Values at indexes are error flags.
/// If nibbles of the first (lead) byte and high nibble of the second byte
/// give a non-zero value (error flag) which isn't the `DOUBLE_CONTINUATION_FLAG`,
/// that means JSON has invalid UTF-8.
///
/// All values of this table that are `DOUBLE_CONTINUATION_FLAG` are not errors.
/// They mean two continuation bytes in a row and
/// used for validating 3-, 4- byte sequences.
const UTF8_INVALID_CHAR_TABLES = block: {
    var leadByteLowNibbles: [16]u8 = @splat(0);
    var leadByteHighNibbles: [16]u8 = @splat(0);
    var nextByteHighNibbles: [16]u8 = @splat(0);

    const Group = struct { leadValues: []const u8, nextByteHighNibbles: []const u8 };

    const invalidByteGroups = [_]Group{
        // ASCII-char when the next char is a continuation
        .{
            .leadValues = &utils.range(0b00000000, 0b01111111, .{}),
            .nextByteHighNibbles = .{ 0b1000, 0b1001, 0b1010, 0b1011 },
        },
        // Missing a continuation byte
        .{
            .leadValues = &utils.range(0b11000000, 0b11111111, .{}),
            .nextByteHighNibbles = &utils.range(
                0b0000,
                0b1111,
                .{ 0b1000, 0b1001, 0b1010, 0b1011 },
            ),
        },
        // Overlong 2-byte char (can be written in a less bytes amount)
        .{
            .leadValues = &.{ 0b11000000, 0b11000001 },
            .nextByteHighNibbles = &utils.range(0b0000, 0b1111, .{}),
        },
        // Overlong 3-byte char
        .{
            .leadValues = &.{0b11100000},
            .nextByteHighNibbles = &.{ 0b1000, 0b1001 },
        },
        // Overlong 4-byte char and Code point greater than 0x10FFFF (unicode maximum)
        .{
            .leadValues = utils.range(0b11110000, 0b11111111, .{}),
            .nextByteHighNibbles = &.{0b1000},
        },
        // Surrogate
        .{
            .leadValues = &.{0b11101101},
            .nextByteHighNibbles = &.{ 0b1010, 0b1011 },
        },
        // Code point greater than 0x10FFFF
        .{
            .leadValues = &utils.range(0b11110000, 0b11111111, .{}),
            .nextByteHighNibbles = &.{ 0b1001, 0b1010, 0b1011 },
        },
    };

    var groupIndex = 0;

    for (invalidByteGroups) |group| {
        const flag = 1 << groupIndex;

        for (group.leadValues) |byte| {
            leadByteLowNibbles[utils.getLowNibble(byte)] = flag;
            leadByteHighNibbles[utils.getHighNibble(byte)] = flag;
        }

        for (group.nextByteHighNibbles) |highNibble|
            nextByteHighNibbles[highNibble] = flag;

        groupIndex += 1;
    }

    const doubleContinuationFlag = 1 << groupIndex;

    for (utils.range(0b10000000, 0b10111111)) |byte| {
        const lowNibble = utils.getLowNibble(byte);
        const highNibble = utils.getHighNibble(byte);

        leadByteLowNibbles[lowNibble] = doubleContinuationFlag;
        leadByteHighNibbles[highNibble] = doubleContinuationFlag;

        nextByteHighNibbles[highNibble] = doubleContinuationFlag;
    }

    break :block struct {
        // Align 'cause it's moved to vector registers (with 16, 32, 64 bytes widths)
        pub const LEAD_BYTE_LOW_NIBBLE_TABLE: [16]u8 align(64) = leadByteLowNibbles;
        pub const LEAD_BYTE_HIGH_NIBBLE_TABLE: [16]u8 align(64) = leadByteHighNibbles;
        pub const NEXT_BYTE_HIGH_NIBBLE_TABLE: [16]u8 align(64) = nextByteHighNibbles;

        pub const DOUBLE_CONTINUATION_FLAG: u8 = doubleContinuationFlag;
    };
};

/// Tables, used with the vector shuffle instruction
/// to classify 3-byte UTF-8 leaders (11100000..11101111).
///
/// Values are all 11111111.
const UTF8_THREE_BYTE_LEAD_TABLES = block: {
    var lowNibbles: [16]u8 = @splat(0);
    var highNibbles: [16]u8 = @splat(0);

    const value: u8 = 0b11111111;
    for (utils.range(0b11100000, 0b11101111)) |byte| {
        lowNibbles[utils.getLowNibble(byte)] = value;
        highNibbles[utils.getHighNibble(byte)] = value;
    }

    break :block struct {
        // Align 'cause it's moved to vector registers (with 16, 32, 64 bytes widths)
        pub const LOW_NIBBLE_TABLE: [16]u8 align(8) = lowNibbles;
        pub const HIGH_NIBBLE_TABLE: [16]u8 align(8) = highNibbles;
    };
};

/// Tables, used with the vector shuffle instruction
/// to classify 4-byte UTF-8 leaders (11110000..11110100).
///
/// Values are all 11111111.
const UTF8_FOUR_BYTE_LEAD_TABLES = block: {
    var lowNibbles: [16]u8 = @splat(0);
    var highNibbles: [16]u8 = @splat(0);

    const value: u8 = 0b11111111;
    for (utils.range(0b11110000, 0b11110100)) |byte| {
        lowNibbles[utils.getLowNibble(byte)] = value;
        highNibbles[utils.getHighNibble(byte)] = value;
    }

    break :block struct {
        // Align 'cause it's moved to vector registers (with 16, 32, 64 bytes widths)
        pub const LOW_NIBBLE_TABLE: [16]u8 align(8) = lowNibbles;
        pub const HIGH_NIBBLE_TABLE: [16]u8 align(8) = highNibbles;
    };
};

/// Returns `true` when `vector` with JSON chars contains not only the ASCII-chars.
///
/// Otherwise, returns `false`.
inline fn isVectorNonAscii_x86(vector: anytype) bool {
    // All ASCII chars have the highest bit set to 0
    const onlyHighBitsVector: @TypeOf(vector) = @splat(0b10000000);

    return switch (comptime vector.len) {
        64 => simd.x86.andToBits512(vector, onlyHighBitsVector) != 0,

        else => unreachable,
        16 => simd.x86.notEqlToBits128(vector & onlyHighBitsVector, 0) != 0,
    };
}

/// Returns `true` only if `chunk` of JSON chars has valid UTF-8.
inline fn validateEncoding_x86(
    chunk: anytype,
    /// High nibbles of the initial `chunk` are needed
    /// in this function, but low nibbles are not.
    /// Also, high nibbles are computed in `Tokenizer.next` in any way.
    /// So, receive it as an argument not to compute it twice.
    chunkHighNibbles: @TypeOf(chunk),
    encodingContext: Tokenizer.EncodingContext,
    comptime maxVectorLen: comptime_int,
) bool {
    // Fast path
    if (isVectorNonAscii_x86(chunk))
        return true;

    const leadByteLowNibbleTable = comptime block: {
        const tableArray = UTF8_INVALID_CHAR_TABLES.LEAD_BYTE_LOW_NIBBLE_TABLE;
        const tableVector: @Vector(tableArray.len, u8) = tableArray;

        break :block simd.expandVector(tableVector, chunk.len);
    };
    const leadByteHighNibbleTable = comptime block: {
        const tableArray = UTF8_INVALID_CHAR_TABLES.LEAD_BYTE_HIGH_NIBBLE_TABLE;
        const tableVector: @Vector(tableArray.len, u8) = tableArray;

        break :block simd.expandVector(tableVector, chunk.len);
    };

    const nextByteHighNibbleTable = comptime block: {
        const tableArray = UTF8_INVALID_CHAR_TABLES.NEXT_BYTE_HIGH_NIBBLE_TABLE;
        const tableVector: @Vector(tableArray.len, u8) = tableArray;

        break :block simd.expandVector(tableVector, chunk.len);
    };

    const threeByteLeadLowNibbleTable = comptime block: {
        const tableArray = UTF8_THREE_BYTE_LEAD_TABLES.LOW_NIBBLE_TABLE;
        const tableVector: @Vector(tableArray.len, u8) = tableArray;

        break :block simd.expandVector(tableVector, chunk.len);
    };
    const threeByteLeadHighNibbleTable = comptime block: {
        const tableArray = UTF8_THREE_BYTE_LEAD_TABLES.HIGH_NIBBLE_TABLE;
        const tableVector: @Vector(tableArray.len, u8) = tableArray;

        break :block simd.expandVector(tableVector, chunk.len);
    };
    const fourByteLeadLowNibbleTable = comptime block: {
        const tableArray = UTF8_FOUR_BYTE_LEAD_TABLES.LOW_NIBBLE_TABLE;
        const tableVector: @Vector(tableArray.len, u8) = tableArray;

        break :block simd.expandVector(tableVector, chunk.len);
    };
    const fourByteLeadHighNibbleTable = comptime block: {
        const tableArray = UTF8_FOUR_BYTE_LEAD_TABLES.HIGH_NIBBLE_TABLE;
        const tableVector: @Vector(tableArray.len, u8) = tableArray;

        break :block simd.expandVector(tableVector, chunk.len);
    };

    const mergeShiftRight = comptime switch (chunk.len) {
        64 => simd.x86.mergeShiftRight512,
        16 => simd.x86.mergeShiftRight128,
        else => unreachable,
    };

    const chunkWithPrev = mergeShiftRight(encodingContext.prevChunk, chunk, 1);

    const chunkWithPrevLowNibbles = simd.getLowNibbles(chunkWithPrev);
    const chunkWithPrevHighNibbles = simd.getHighNibbles(chunkWithPrev);

    switch (comptime maxVectorLen) {
        64 => {
            const invalidBytes = block: {
                const leadByteLowNibbleErrors = simd.x86.shuffleVector512(
                    leadByteLowNibbleTable,
                    chunkWithPrevLowNibbles,
                );
                const leadByteHighNibbleErrors = simd.x86.shuffleVector512(
                    leadByteHighNibbleTable,
                    chunkWithPrevHighNibbles,
                );
                const nextByteHighNibbleErrors = simd.x86.shuffleVector512(
                    nextByteHighNibbleTable,
                    chunkHighNibbles,
                );

                break :block simd.x86.tripleAnd512(
                    leadByteLowNibbleErrors,
                    leadByteHighNibbleErrors,
                    nextByteHighNibbleErrors,
                );
            };

            const doubleContinuationMask = UTF8_INVALID_CHAR_TABLES.DOUBLE_CONTINUATION_FLAG;
            const errorMask = comptime ~doubleContinuationMask;

            if (simd.x86.isNonZero512(invalidBytes & errorMask))
                return false;

            const threeByteLeads = block: {
                const lowNibblesMatch =
                    simd.x86.shuffleVector512(threeByteLeadLowNibbleTable, chunkWithPrevLowNibbles);
                const highNibblesMatch =
                    simd.x86.shuffleVector512(threeByteLeadHighNibbleTable, chunkWithPrevHighNibbles);

                break :block lowNibblesMatch & highNibblesMatch;
            };

            const fourByteLeads = block: {
                const lowNibblesMatch =
                    simd.x86.shuffleVector512(fourByteLeadLowNibbleTable, chunkWithPrevLowNibbles);
                const highNibblesMatch =
                    simd.x86.shuffleVector512(fourByteLeadHighNibbleTable, chunkWithPrevHighNibbles);

                break :block lowNibblesMatch & highNibblesMatch;
            };

            const doubleContinuationStarts = invalidBytes & doubleContinuationMask;

            const expectedDoubleContinuationStarts = block: {
                const expectedThreeByteContinuationStarts = mergeShiftRight(
                    encodingContext.prevThreeByteLeads,
                    threeByteLeads,
                    1,
                );

                const expectedFourByteContinuationStarts =
                    mergeShiftRight(
                        encodingContext.prevFourByteLeads,
                        fourByteLeads,
                        1,
                    ) | mergeShiftRight(
                        encodingContext.prevFourByteLeads,
                        fourByteLeads,
                        2,
                    );

                break :block expectedThreeByteContinuationStarts & expectedFourByteContinuationStarts;
            };

            if (simd.x86.isZero512(doubleContinuationStarts == expectedDoubleContinuationStarts))
                return false;
        },
    }

    return true;
}
