pub mod primitive;
pub mod mmu;
pub mod emulator;

use mmu::{VirtAddr, Perm, Section, PERM_READ, PERM_WRITE, PERM_EXEC};
use emulator::{Emulator, Register};

fn main() {
    let mut emu = Emulator::new(32 * 1024 * 1024);

    // Load the application into the emulator
    emu.memory.load("./test_app", &[
        Section {
            file_off:    0x0000000000000000,
            virt_addr:   VirtAddr(0x0000000000010000),
            file_size:   0x0000000000000190,
            mem_size:    0x0000000000000190,
            permissions: Perm(PERM_READ),
        },
        Section {
            file_off:    0x0000000000000190,
            virt_addr:   VirtAddr(0x0000000000011190),
            file_size:   0x0000000000002598,
            mem_size:    0x0000000000002598,
            permissions: Perm(PERM_EXEC),
        },
        Section {
            file_off:    0x0000000000002728,
            virt_addr:   VirtAddr(0x0000000000014728),
            file_size:   0x00000000000000f8,
            mem_size:    0x0000000000000750,
            permissions: Perm(PERM_READ | PERM_WRITE),
        },
    ]).expect("Failed to load test application into address space");

    // Set the program entry point
    emu.set_reg(Register::Pc, 0x11190);

    // Set up a stack
    let stack = emu.memory.allocate(32 * 1024)
        .expect("Failed to allocate stack");
    emu.set_reg(Register::Sp, stack.0 as u64 + 32 * 1024);

    // Set up null terminated arg vectors
    let argv = emu.memory.allocate(8)
        .expect("Failed to allocate argv");
    emu.memory.write_from(argv, b"test\0")
        .expect("Failed to null-terminate argv");

    macro_rules! push {
        ($expr:expr) => {
            let sp = emu.reg(Register::Sp) -
                core::mem::size_of_val(&$expr) as u64;
            emu.memory.write(VirtAddr(sp as usize), $expr)
                .expect("Push failed");
            emu.set_reg(Register::Sp, sp);
        }
    }

    push!(0u64); // Auxp
    push!(0u64); // Envp
    push!(0u64); // Argv end
    push!(argv.0); // Argv
    push!(1u64); // Argc
    
    emu.run().expect("Failed to execute emulator");
}

