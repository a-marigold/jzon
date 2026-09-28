const std = @import("std");
const utils = @import("utils.zig");
const simd = @import("simd.zig");
const Tokenizer = @import("Tokenizer.zig");

const EncodingContext = Tokenizer.EncodingContext;

/// Contains `LEAD_BYTE_LOW_NIBBLE_TABLE`, `LEAD_BYTE_HIGH_NIBBLE_TABLE`,
/// `NEXT_BYTE_HIGH_NIBBLE_TABLE` lookup tables to be used as vectors.
///
/// `DOUBLE_CONTINUATION_FLAG` is the flag, which is set
/// at indexes of the tables meaning two continuation bytes in a row (double continuation).
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
const UTF8_TWO_BYTE_TABLES = block: {
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

pub const ValidateEncodingResult = struct {
    isValid: bool,
    newEncodingContext: EncodingContext,
};

pub const x86 = struct {
    /// Returns `true` only if `chunk` of JSON chars has valid UTF-8.
    ///
    /// Contains a check of `chunk` on containing only ASCII,
    /// which also specially handles the previous chunk.
    /// Checking does `chunk` contain only ASCII
    /// before calling this function can cause wrong results.
    pub inline fn validateEncoding(
        chunk: anytype,
        /// High nibbles of the initial `chunk` are needed
        /// in this function, but low nibbles are not.
        /// Also, high nibbles are computed in `Tokenizer.next` in any way.
        /// So, receive it as an argument not to compute it twice.
        chunkHighNibbles: @TypeOf(chunk),
        encodingContext: EncodingContext,
        comptime maxVectorLen: comptime_int,
    ) ValidateEncodingResult {
        const mergeShiftRight = comptime switch (maxVectorLen) {
            16, 32 => simd.x86.mergeShiftRight128,
            64 => simd.x86.mergeShiftRight512,
            else => unreachable,
        };

        const chunkWithPrev = mergeShiftRight(encodingContext.prevChunk, chunk, 1);

        // Fast path
        if (isVectorNonAscii(chunkWithPrev))
            return .{
                .isValid = true,
                .newEncodingContext = .{
                    .prevChunk = chunkWithPrev,
                    .prevThreeByteLeads = @splat(0),
                    .prevFourByteLeads = @splat(0),
                },
            };

        const leadByteLowNibbleTable = comptime block: {
            const tableArray = UTF8_TWO_BYTE_TABLES.LEAD_BYTE_LOW_NIBBLE_TABLE;
            const tableVector: @Vector(tableArray.len, u8) = tableArray;

            break :block simd.expandVector(tableVector, chunk.len);
        };
        const leadByteHighNibbleTable = comptime block: {
            const tableArray = UTF8_TWO_BYTE_TABLES.LEAD_BYTE_HIGH_NIBBLE_TABLE;
            const tableVector: @Vector(tableArray.len, u8) = tableArray;

            break :block simd.expandVector(tableVector, chunk.len);
        };

        const nextByteHighNibbleTable = comptime block: {
            const tableArray = UTF8_TWO_BYTE_TABLES.NEXT_BYTE_HIGH_NIBBLE_TABLE;
            const tableVector: @Vector(tableArray.len, u8) = tableArray;

            break :block simd.expandVector(tableVector, chunk.len);
        };

        const chunkWithPrevLowNibbles = simd.getLowNibbles(chunkWithPrev);
        const chunkWithPrevHighNibbles = simd.getHighNibbles(chunkWithPrev);

        const invalidBytes, const doubleContinuationStarts = block: {
            const doubleContinuationMask = UTF8_TWO_BYTE_TABLES.DOUBLE_CONTINUATION_FLAG;
            const invalidBytesMask = comptime ~doubleContinuationMask;

            switch (comptime maxVectorLen) {
                16 => {
                    const leadByteLowNibbleErrors = simd.x86.shuffleVector128(
                        leadByteLowNibbleTable,
                        chunkWithPrevLowNibbles,
                    );
                    const leadByteHighNibbleErrors = simd.x86.shuffleVector128(
                        leadByteHighNibbleTable,
                        chunkWithPrevHighNibbles,
                    );
                    const nextByteHighNibbleErrors = simd.x86.shuffleVector128(
                        nextByteHighNibbleTable,
                        chunkHighNibbles,
                    );

                    const bytesMatch =
                        leadByteLowNibbleErrors & leadByteHighNibbleErrors & nextByteHighNibbleErrors;

                    break :block .{ bytesMatch & invalidBytesMask, bytesMatch & doubleContinuationMask };
                },
                32 => {
                    // TODO

                    const leadByteHalves = simd.x86.shuffleVector256(
                        leadByteLowNibbleTable ++ leadByteHighNibbleTable,
                        chunkWithPrevLowNibbles ++ chunkWithPrevHighNibbles,
                    );
                    const nextByteHighNibbleErrors = simd.x86.shuffleVector256(
                        simd.expandVector(nextByteHighNibbleTable, 32),
                        simd.expandVector(chunkHighNibbles, 32),
                    );

                    const bytesMatch =
                        leadByteHalves & nextByteHighNibbleErrors & leadByteHalves[16..];

                    break :block .{ bytesMatch & invalidBytesMask, bytesMatch & doubleContinuationMask };
                },
                64 => {
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

                    const bytesMatch = simd.x86.tripleAnd512(
                        leadByteLowNibbleErrors,
                        leadByteHighNibbleErrors,
                        nextByteHighNibbleErrors,
                    );

                    break :block .{ bytesMatch & invalidBytesMask, bytesMatch & doubleContinuationMask };
                },
                else => unreachable,
            }
        };

        const isTwoByteError = switch (comptime maxVectorLen) {
            16, 32 => simd.x86.eqlToBits128(invalidBytes, @splat(0)) != 0,
            64 => simd.x86.isNonZero512(bool, invalidBytes),
            else => unreachable,
        };

        if (isTwoByteError)
            return .{
                .isValid = false,
                .newEncodingContext = .{
                    .prevChunk = chunkWithPrev,
                    .prevThreeByteLeads = @splat(0),
                    .prevFourByteLeads = @splat(0),
                },
            };

        const threeByteLeads = getThreeByteLeads(
            chunkWithPrevLowNibbles,
            chunkWithPrevHighNibbles,
            maxVectorLen,
        );
        const fourByteLeads = getFourByteLeads(
            chunkWithPrevLowNibbles,
            chunkWithPrevHighNibbles,
            maxVectorLen,
        );

        const expectedDoubleContinuationStarts = block: {
            // Do merge shift, not default shift, to handle
            // cases when a 3-byte leader is the last byte of the prev chunk
            const expectedThreeByteContinuationStarts = mergeShiftRight(
                encodingContext.prevThreeByteLeads,
                threeByteLeads,
                1,
            );

            // Do merge shift by 1, not default shift, to handle
            // cases when a 4-byte leader is the last byte of the prev chunk.
            // And do merge shift by 2 to handle cases
            // when a 4-byte leader is the penultimate byte of the prev chunk
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

        const isDoubleContinuationError = switch (comptime maxVectorLen) {
            16, 32 => simd.x86.eqlToBits128(doubleContinuationStarts, expectedDoubleContinuationStarts) == 0,
            64 => simd.x86.isZero512(bool, doubleContinuationStarts == expectedDoubleContinuationStarts),
            else => unreachable,
        };

        if (isDoubleContinuationError)
            return .{
                .isValid = false,
                .newEncodingContext = .{
                    .prevChunk = chunkWithPrev,
                    .prevThreeByteLeads = @splat(0),
                    .prevFourByteLeads = @splat(0),
                },
            };

        return .{
            .isValid = true,
            .newEncodingContext = .{
                .prevChunk = chunkWithPrev,
                .prevThreeByteLeads = threeByteLeads,
                .prevFourByteLeads = fourByteLeads,
            },
        };
    }

    /// Three 3-byte leaders in the returned vector have
    /// `UTF8_INVALID_BYTE_TABLES.DOUBLE_CONTINUATION_FLAG` as values.
    inline fn getThreeByteLeads(
        chunkLowNibbles: anytype,
        chunkHighNibbles: @TypeOf(chunkLowNibbles),
        comptime maxVectorLen: comptime_int,
    ) @TypeOf(chunkLowNibbles) {
        const chunkLen = chunkLowNibbles.len;

        // `validateEncoding` compares vector of 3-byte leads with
        // vector of double continuation bytes bit-to-bit,
        // so 3-byte leads must have the same bits
        const doubleContinuationFlagVector: @Vector(chunkLen, u8) =
            @splat(UTF8_TWO_BYTE_TABLES.DOUBLE_CONTINUATION_FLAG);

        const threeByteLeadLowNibbleTable = comptime block: {
            const tableArray = UTF8_THREE_BYTE_LEAD_TABLES.LOW_NIBBLE_TABLE;
            const tableVector: @Vector(tableArray.len, u8) = tableArray;

            break :block simd.expandVector(tableVector, chunkLen) & doubleContinuationFlagVector;
        };
        const threeByteLeadHighNibbleTable = comptime block: {
            const tableArray = UTF8_THREE_BYTE_LEAD_TABLES.HIGH_NIBBLE_TABLE;
            const tableVector: @Vector(tableArray.len, u8) = tableArray;

            break :block simd.expandVector(tableVector, chunkLen) & doubleContinuationFlagVector;
        };

        switch (comptime maxVectorLen) {
            16 => {
                const lowNibblesMatch =
                    simd.x86.shuffleVector128(threeByteLeadLowNibbleTable, chunkLowNibbles);
                const highNibblesMatch =
                    simd.x86.shuffleVector128(threeByteLeadHighNibbleTable, chunkHighNibbles);

                return lowNibblesMatch & highNibblesMatch;
            },
            32 => {
                const matchHalves = simd.x86.shuffleVector256(
                    threeByteLeadLowNibbleTable ++ threeByteLeadHighNibbleTable,
                    chunkLowNibbles ++ chunkHighNibbles,
                );

                return matchHalves[0..16] & matchHalves[16..];
            },
            64 => {
                const lowNibblesMatch =
                    simd.x86.shuffleVector512(threeByteLeadLowNibbleTable, chunkLowNibbles);
                const highNibblesMatch =
                    simd.x86.shuffleVector512(threeByteLeadHighNibbleTable, chunkHighNibbles);

                return lowNibblesMatch & highNibblesMatch;
            },
        }
    }
    /// Three 4-byte leaders in the returned vector have
    /// `UTF8_INVALID_BYTE_TABLES.DOUBLE_CONTINUATION_FLAG` as values.
    inline fn getFourByteLeads(
        chunkLowNibbles: anytype,
        chunkHighNibbles: @TypeOf(chunkLowNibbles),
        comptime maxVectorLen: comptime_int,
    ) @TypeOf(chunkLowNibbles) {
        const chunkLen = chunkLowNibbles.len;

        // `validateEncoding` compares vector of 4-byte leads with
        // vector of double continuation bytes bit-to-bit,
        // so 4-byte leads must have the same bits
        const doubleContinuationFlagVector: @Vector(chunkLen, u8) =
            @splat(UTF8_TWO_BYTE_TABLES.DOUBLE_CONTINUATION_FLAG);

        const fourByteLeadLowNibbleTable = comptime block: {
            const tableArray = UTF8_FOUR_BYTE_LEAD_TABLES.LOW_NIBBLE_TABLE;
            const tableVector: @Vector(tableArray.len, u8) = tableArray;

            break :block simd.expandVector(tableVector, chunkLen) & doubleContinuationFlagVector;
        };
        const fourByteLeadHighNibbleTable = comptime block: {
            const tableArray = UTF8_FOUR_BYTE_LEAD_TABLES.HIGH_NIBBLE_TABLE;
            const tableVector: @Vector(tableArray.len, u8) = tableArray;

            break :block simd.expandVector(tableVector, chunkLen) & doubleContinuationFlagVector;
        };

        switch (comptime maxVectorLen) {
            16 => {
                const lowNibblesMatch =
                    simd.x86.shuffleVector128(fourByteLeadLowNibbleTable, chunkLowNibbles);
                const highNibblesMatch =
                    simd.x86.shuffleVector128(fourByteLeadHighNibbleTable, chunkHighNibbles);

                return lowNibblesMatch & highNibblesMatch;
            },
            32 => {
                const matchHalves = simd.x86.shuffleVector256(
                    fourByteLeadLowNibbleTable ++ fourByteLeadHighNibbleTable,
                    chunkLowNibbles ++ chunkHighNibbles,
                );

                return matchHalves[0..16] & matchHalves[16..];
            },
            64 => {
                const lowNibblesMatch =
                    simd.x86.shuffleVector512(fourByteLeadLowNibbleTable, chunkLowNibbles);
                const highNibblesMatch =
                    simd.x86.shuffleVector512(fourByteLeadHighNibbleTable, chunkHighNibbles);

                return lowNibblesMatch & highNibblesMatch;
            },
        }
    }

    /// Returns `true` when `vector` with JSON chars contains not only the ASCII chars
    ///
    /// Otherwise, returns `false`.
    pub inline fn isVectorNonAscii(vector: anytype) bool {
        // All ASCII chars have the highest bit set to 0
        const onlyHighBitsVector: @TypeOf(vector) = @splat(0b10000000);

        return switch (comptime vector.len) {
            16 => simd.x86.notEqlToBits128(vector & onlyHighBitsVector, 0) != 0,
            64 => simd.x86.andToBits512(vector, onlyHighBitsVector) != 0,
            else => unreachable,
        };
    }
};

pub const aarch64 = struct {
    /// Returns `true` when `vector` with JSON chars contains not only the ASCII chars.
    pub inline fn isVectorNonAscii128(vector: @Vector(16, u8)) bool {
        const onlyHighBitsVector: @Vector(16, u8) = @splat(0b10000000);

        return simd.aarch64.notEqlToBits128(vector & onlyHighBitsVector, 0) != 0;
    }
};
