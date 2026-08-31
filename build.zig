const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const module = b.addModule("jzon", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    const checkStep = b.step("check", "Semantic analysis");
    const checkExe = b.addExecutable(.{
        .name = "jzon",
        .root_module = module,
    });
    checkExe.step.dependOn(checkStep);

    const mod_tests = b.addTest(.{
        .root_module = module,
    });

    const run_mod_tests = b.addRunArtifact(mod_tests);
    const test_step = b.step("test", "Run tests");

    test_step.dependOn(&run_mod_tests.step);
}
