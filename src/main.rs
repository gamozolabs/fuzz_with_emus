pub mod primitive;
pub mod mmu;
pub mod emulator;

use std::time::Instant;
use mmu::{VirtAddr, Perm, Section, PERM_READ, PERM_WRITE, PERM_EXEC};
use emulator::{Emulator, Register, VmExit};

fn handle_syscall(emu: &mut Emulator) -> Result<(), VmExit> {
    // Get the syscall number
    let num = emu.reg(Register::A7);

    match num {
        96 => {
            // set_tid_address(), just return the TID
            emu.set_reg(Register::A0, 1337);
            Ok(())
        }
        29 => {
            // ioctl()
            emu.set_reg(Register::A0, !0);
            Ok(())
        }
        66 => {
            // writev()
            let fd     = emu.reg(Register::A0);
            let iov    = emu.reg(Register::A1);
            let iovcnt = emu.reg(Register::A2);

            // We currently only handle stdout and stderr
            if fd != 1 && fd != 2 {
                // Return error
                emu.set_reg(Register::A0, !0);
                return Ok(());
            }

            let mut bytes_written = 0;

            for idx in 0..iovcnt {
                // Compute the pointer to the IO vector entry
                // corresponding to this index and validate that it
                // will not overflow pointer size for the size of
                // the `_iovec`
                let ptr = 16u64.checked_mul(idx)
                    .and_then(|x| x.checked_add(iov))
                    .and_then(|x| x.checked_add(15))
                    .ok_or(VmExit::SyscallIntegerOverflow)? as usize - 15;

                // Read the iovec entry pointer and length
                let buf: usize = emu.memory.read(VirtAddr(ptr + 0))?;
                let len: usize = emu.memory.read(VirtAddr(ptr + 8))?;

                // Look at the buffer!
                let data = emu.memory.peek_perms(VirtAddr(buf), len,
                    Perm(PERM_READ))?;

                /*
                if let Ok(st) = core::str::from_utf8(data) {
                    print!("{}", st);
                }*/

                // Update number of bytes written
                bytes_written += len as u64;
            }

            // Return number of bytes written
            emu.set_reg(Register::A0, bytes_written);
            Ok(())
        }
        94 => {
            Err(VmExit::Exit)
        }
        _ => {
            panic!("Unhandled syscall {}\n", num);
        }
    }
}

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

    // Set up the program name
    let argv = emu.memory.allocate(8)
        .expect("Failed to allocate program name");
    emu.memory.write_from(argv, b"test\0")
        .expect("Failed to write program name");

    macro_rules! push {
        ($expr:expr) => {
            let sp = emu.reg(Register::Sp) -
                core::mem::size_of_val(&$expr) as u64;
            emu.memory.write(VirtAddr(sp as usize), $expr)
                .expect("Push failed");
            emu.set_reg(Register::Sp, sp);
        }
    }

    // Set up the initial program stack state
    push!(0u64);   // Auxp
    push!(0u64);   // Envp
    push!(0u64);   // Argv end
    push!(argv.0); // Argv
    push!(1u64);   // Argc

    // Now, fork the VM
    let mut worker = emu.fork();
  
    // Start a timer
    let start = Instant::now();

    for fuzz_cases in 1u64.. {
        // Reset worker to original state
        worker.reset(&emu);

        let vmexit = loop {
            let vmexit = worker.run().expect_err("Failed to execute emulator");

            match vmexit {
                VmExit::Syscall => {
                    if let Err(vmexit) = handle_syscall(&mut worker) {
                        break vmexit;
                    }
        
                    // Advance PC
                    let pc = worker.reg(Register::Pc);
                    worker.set_reg(Register::Pc, pc.wrapping_add(4));
                }
                _ => break vmexit,
            }
        };

        //print!("VM exited with {:#x?}\n", vmexit);

        if fuzz_cases & 0xffff == 0 {
            let elapsed = start.elapsed().as_secs_f64();

            print!("[{:10.4}] cases {:10} | fcps {:10.2}\n",
                   elapsed, fuzz_cases, fuzz_cases as f64 / elapsed);
        }
    }
}

