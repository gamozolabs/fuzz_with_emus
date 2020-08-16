pub mod primitive;
pub mod mmu;
pub mod emulator;
pub mod jitcache;
pub mod affinity;

use std::fs::File;
use std::io::{self, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::sync::atomic::AtomicU64;
use std::time::{Duration, Instant};
use std::convert::TryInto;
use std::collections::{BTreeMap, BTreeSet};
use mmu::{VirtAddr, Perm, Section, PERM_READ, PERM_WRITE, PERM_EXEC, PERM_ACC};
use emulator::{Emulator, Register, VmExit, EmuFile, FaultType, AddressType};
use emulator::{CoverageType, COVERAGE_ENTRY_EMPTY};
use jitcache::JitCache;

use aht::Aht;
use falkhash::FalkHasher;
use atomicvec::AtomicVec;
use basic_mutator::{Mutator, InputDatabase, EmptyDatabase};

/// If set, uses the enclosed string as a filename and uses it as the input
/// without any corruption
const REPRO_MODE: Option<&str> = None; //Some("crashes/0x69478_Read_Normal.crash");

/// If set, prints information about all hooked allocations
const VERBOSE_ALLOCS: bool = false;

/// If `true` the guest writes to stdout and stderr will be printed to our own
/// stdout and stderr
const VERBOSE_GUEST_PRINTS: bool = false;

fn rdtsc() -> u64 {
    unsafe { std::arch::x86_64::_rdtsc() }
}

struct Rng(u64);

impl Rng {
    /// Create a new random number generator
    fn new() -> Self {
        //Rng(0x8644d6eb17b7ab1a ^ rdtsc())
        Rng(0x8644d6eb17b7ab1a)
    }

    /// Generate a random number
    #[inline]
    fn rand(&mut self) -> usize {
        let val = self.0;
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 43;
        val as usize
    }
}

/// Stat structure from kernel_stat64
#[repr(C)]
#[derive(Default, Debug)]
struct Stat {
    st_dev:     u64,
    st_ino:     u64,
    st_mode:    u32,
    st_nlink:   u32,
    st_uid:     u32,
    st_gid:     u32,
    st_rdev:    u64,
    __pad1:     u64,

    st_size:    i64,
    st_blksize: i32,
    __pad2:     i32,

    st_blocks: i64,

    st_atime:     u64,
    st_atimensec: u64,
    st_mtime:     u64,
    st_mtimensec: u64,
    st_ctime:     u64,
    st_ctimensec: u64,
    
    __glibc_reserved: [i32; 2],
}

fn handle_syscall(emu: &mut Emulator) -> Result<(), VmExit> {
    // Get the syscall number
    let num = emu.reg(Register::A7);

    //print!("Syscall {}\n", num);

    match num {
        214 => {
            // brk()
            let req_base = emu.reg(Register::A0);
            if req_base == 0 {
                emu.set_reg(Register::A0, 0);
                return Ok(());
            }

            panic!("Not expecting brk");

            /*
            let increment = if req_base != 0 {
                (req_base as i64).checked_sub(cur_base.0 as i64)
                    .ok_or(VmExit::SyscallIntegerOverflow)?
            } else {
                0
            };

            // We don't handle negative brks yet
            if increment < 0 {
                emu.set_reg(Register::A0, cur_base.0 as u64);
                return Ok(());
            }

            // Attempt to extend data section by increment
            if let Some(_) = emu.memory.allocate(increment as usize) {
                let new_base = cur_base.0 + increment as usize;
                emu.set_reg(Register::A0, new_base as u64);
            } else {
                emu.set_reg(Register::A0, !0);
            }

            Ok(())*/
        }
        64 => {
            // write()
            let fd  = emu.reg(Register::A0) as usize;
            let buf = emu.reg(Register::A1);
            let len = emu.reg(Register::A2);

            let file = emu.files.get_file(fd);
            if let Some(Some(file)) = file {
                if file == &EmuFile::Stdout || file == &EmuFile::Stderr {
                    // Writes to stdout and stderr

                    // Get access to the underlying bytes to write
                    let bytes = emu.memory.peek(VirtAddr(buf as usize),
                        len as usize, Perm(PERM_READ))?;

                    if VERBOSE_GUEST_PRINTS {
                        if let Ok(st) = core::str::from_utf8(bytes) {
                            print!("{}", st);
                        }
                    }

                    // Set that all bytes were read
                    emu.set_reg(Register::A0, len);
                } else {
                    panic!("Write to valid but unhandled FD");
                }
            } else {
                // Unknown FD
                emu.set_reg(Register::A0, !0);
            }

            Ok(())
        }
        169 => {
            // gettimeofday()
            emu.set_reg(Register::A0, 0);
            Ok(())
        }
        63 => {
            // read()
            let fd  = emu.reg(Register::A0) as usize;
            let buf = emu.reg(Register::A1) as usize;
            let len = emu.reg(Register::A2) as usize;
            
            // Check if the FD is valid
            let file = emu.files.get_file(fd);
            if file.is_none() || file.as_ref().unwrap().is_none() {
                // FD was not valid, return out with an error
                emu.set_reg(Register::A0, !0);
                return Ok(());
            }
            
            if let Some(Some(EmuFile::FuzzInput { ref mut cursor })) = file {
                // Compute the ending cursor from this read
                let result_cursor = core::cmp::min(
                    cursor.saturating_add(len),
                    emu.fuzz_input.len());

                // Write in the bytes
                emu.memory.write_from(VirtAddr(buf),
                    &emu.fuzz_input[*cursor..result_cursor])?;

                // Compute bytes read
                let bread = result_cursor - *cursor;
                
                // Update the cursor
                *cursor = result_cursor;

                // Return number of bytes read
                emu.set_reg(Register::A0, bread as u64);
            } else {
                unreachable!();
            }

            Ok(())
        }
        62 => {
            // lseek()
            let fd     = emu.reg(Register::A0) as usize;
            let offset = emu.reg(Register::A1) as i64;
            let whence = emu.reg(Register::A2) as i32;

            const SEEK_SET: i32 = 0;
            const SEEK_CUR: i32 = 1;
            const SEEK_END: i32 = 2;

            // Check if the FD is valid
            let file = emu.files.get_file(fd);
            if file.is_none() || file.as_ref().unwrap().is_none() {
                // FD was not valid, return out with an error
                emu.set_reg(Register::A0, !0);
                return Ok(());
            }

            if let Some(Some(EmuFile::FuzzInput { ref mut cursor })) = file {
                let new_cursor = match whence {
                    SEEK_SET => offset,
                    SEEK_CUR => (*cursor as i64).saturating_add(offset),
                    SEEK_END => (emu.fuzz_input.len() as i64)
                        .saturating_add(offset),
                    _ => {
                        // Invalid whence, return error
                        emu.set_reg(Register::A0, !0);
                        return Ok(());
                    }
                };

                // Make sure the cursor falls in bounds of [0, file_size]
                let new_cursor = core::cmp::max(0i64, new_cursor);
                let new_cursor =
                    core::cmp::min(new_cursor, emu.fuzz_input.len() as i64);

                // Update the cursor
                *cursor = new_cursor as usize;

                // Return the new cursor position
                emu.set_reg(Register::A0, new_cursor as u64);
            } else {
                unreachable!();
            }

            Ok(())
        }
        1024 => {
            // open()
            let filename = emu.reg(Register::A0) as usize;
            let flags    = emu.reg(Register::A1);
            let _mode    = emu.reg(Register::A2);

            assert!(flags == 0, "Currently we only handle O_RDONLY");

            // Determine the length of the filename
            let mut fnlen = 0;
            while emu.memory.read::<u8>(VirtAddr(filename + fnlen))? != 0 {
                fnlen += 1;
            }
        
            // Get the filename bytes
            let bytes = emu.memory.peek(VirtAddr(filename),
                fnlen, Perm(PERM_READ))?;

            print!("Open {:x?}\n", bytes);

            if bytes == b"testfn" {
                // Create a new file descriptor
                let fd = emu.alloc_file();

                // Get access to the file, unwrap here is safe because there's
                // no way the file is not a valid FD if we got it from our own
                // APIs
                let file = emu.files.get_file(fd).unwrap();

                // Mark that this file should be backed by our fuzz input
                *file = Some(EmuFile::FuzzInput { cursor: 0 });

                // Return a new fd
                emu.set_reg(Register::A0, fd as u64);
            } else {
                // Unknown filename
                emu.set_reg(Register::A0, !0);
            }

            Ok(())
        }
        1038 => {
            // stat()
            let filename = emu.reg(Register::A0) as usize;
            let statbuf  = emu.reg(Register::A1);
            
            // Determine the length of the filename
            let mut fnlen = 0;
            while emu.memory.read::<u8>(VirtAddr(filename + fnlen))? != 0 {
                fnlen += 1;
            }
        
            // Get the filename bytes
            let bytes = emu.memory.peek(VirtAddr(filename),
                fnlen, Perm(PERM_READ))?;

            if bytes == b"testfn" {
                let mut stat = Stat::default();
                stat.st_dev = 0x803;
                stat.st_ino = 0x81889;
                stat.st_mode = 0x81a4;
                stat.st_nlink = 0x1;
                stat.st_uid = 0x3e8;
                stat.st_gid = 0x3e8;
                stat.st_rdev = 0x0;
                stat.st_size = emu.fuzz_input.len() as i64;
                stat.st_blksize = 0x1000;
                stat.st_blocks = (emu.fuzz_input.len() as i64 + 511) / 512;
                stat.st_atime = 0x5f0fe246;
                stat.st_mtime = 0x5f0fe244;
                stat.st_ctime = 0x5f0fe244;

                // Cast the stat structure to raw bytes
                let stat = unsafe {
                    core::slice::from_raw_parts(
                        &stat as *const Stat as *const u8,
                        core::mem::size_of_val(&stat))
                };

                // Write in the stat data
                emu.memory.write_from(VirtAddr(statbuf as usize), stat)?;
                emu.set_reg(Register::A0, 0);
            } else {
                // Error
                emu.set_reg(Register::A0, !0);
            }

            Ok(())
        }
        80 => {
            // fstat()
            let fd      = emu.reg(Register::A0) as usize;
            let statbuf = emu.reg(Register::A1);

            // Check if the FD is valid
            let file = emu.files.get_file(fd);
            if file.is_none() || file.as_ref().unwrap().is_none() {
                // FD was not valid, return out with an error
                emu.set_reg(Register::A0, !0);
                return Ok(());
            }

            if let Some(Some(EmuFile::FuzzInput { .. })) = file {
                let mut stat = Stat::default();
                stat.st_dev = 0x803;
                stat.st_ino = 0x81889;
                stat.st_mode = 0x81a4;
                stat.st_nlink = 0x1;
                stat.st_uid = 0x3e8;
                stat.st_gid = 0x3e8;
                stat.st_rdev = 0x0;
                stat.st_size = emu.fuzz_input.len() as i64;
                stat.st_blksize = 0x1000;
                stat.st_blocks = (emu.fuzz_input.len() as i64 + 511) / 512;
                stat.st_atime = 0x5f0fe246;
                stat.st_mtime = 0x5f0fe244;
                stat.st_ctime = 0x5f0fe244;

                // Cast the stat structure to raw bytes
                let stat = unsafe {
                    core::slice::from_raw_parts(
                        &stat as *const Stat as *const u8,
                        core::mem::size_of_val(&stat))
                };

                // Write in the stat data
                emu.memory.write_from(VirtAddr(statbuf as usize), stat)?;
                emu.set_reg(Register::A0, 0);
            } else {
                // Error
                emu.set_reg(Register::A0, !0);
            }

            Ok(())
        }
        57 => {
            // close()
            let fd = emu.reg(Register::A0) as usize;

            if let Some(file) = emu.files.get_file(fd) {
                if file.is_some() {
                    // File was present and currently open, close it
                   
                    // Close the file
                    *file = None;

                    // Just return success for now
                    emu.set_reg(Register::A0, 0);
                } else {
                    // File was in a closed state
                    emu.set_reg(Register::A0, !0);
                }
            } else {
                // FD out of bounds
                emu.set_reg(Register::A0, !0);
            }

            Ok(())
        }
        93 => {
            // exit()
            Err(VmExit::Exit)
        }
        _ => {
            panic!("Unhandled syscall {} @ {:#x}\n", num,
                   emu.reg(Register::Pc));
        }
    }
}

#[derive(Default)]
/// Statistics during fuzzing
struct Statistics {
    /// Number of fuzz cases
    fuzz_cases: u64,

    /// Number of risc-v instructions executed
    instrs_execed: u64,
    
    /// Total number of crashes
    crashes: u64,

    /// Total number of CPU cycles spent in the workers
    total_cycles: u64,

    /// Total number of CPU cycles spent resetting the guest
    reset_cycles: u64,
    
    /// Total number of CPU cycles spent emulating
    vm_cycles: u64,

    /// Frequencies of vmexits
    vmexits: BTreeMap<VmExit, u64>,
}

fn worker(thr_id: usize, mut emu: Emulator, original: Arc<Emulator>,
          stats: Arc<Mutex<Statistics>>, corpus: Arc<Corpus>) {
    // Pin to a core
    affinity::set_affinity(thr_id).unwrap();

    // Create a new random number generator
    let mut rng = Rng::new();
        
    // Get the buffer and the length for the input
    let buf = emu.reg(Register::A0);
    let len = emu.reg(Register::A1) as usize;

    // Create a mutator
    let mut mutator = Mutator::new().seed(rng.rand() as u64)
        .max_input_size(1024 * 1024);

    loop {
        // Start a timer
        let batch_start = rdtsc();
        
        let mut local_stats = Statistics::default();

        let it = rdtsc();
        while (rdtsc() - it) < 500_000_000 {
            // Reset emu to original state
            let it = rdtsc();
            emu.reset(&*original, &corpus, |emu| {
                let perms = emu.memory.peek_perms(VirtAddr(buf as usize),
                    len).unwrap();
                perms.iter().map(|perm| (perm.0 & PERM_ACC) != 0).collect()
            });
            local_stats.reset_cycles += rdtsc() - it;

            // Number of instructions executed this fuzz case
            let mut run_instrs = 0u64;

            // Clear the fuzz input
            emu.fuzz_input.clear();
            mutator.input.clear();
            mutator.accessed.clear();

            let mut options: u32 = rng.rand() as u32;

            // Pick a random file from the corpus as an input
            if (rng.rand() % 32) != 0 && corpus.inputs.len() > 0 {
                // Build upon a previous input from the coverage guided inputs
                let sel = rng.rand() % corpus.inputs.len();
                if let Some(input) = corpus.inputs.get(sel) {
                    // Copy the input
                    options = u32::from_ne_bytes(
                        input.data[input.data.len() - 4..]
                        .try_into().unwrap());
                    mutator.input.extend_from_slice(
                        &input.data[..input.data.len() - 4]);

                    // Set a timeout which can reach all coverage for this
                    // input
                    emu.set_timeout(input.instrs + 1_000_000);

                    // Update accessed information
                    //mutator.accessed.extend_from_slice(&input.accessed);
                }
            }
           
            if mutator.input.len() == 0 && corpus.corpus.len() > 0 {
                // Build upon an input from the corpus
                let sel = rng.rand() % corpus.corpus.len();
                if let Some(input) = corpus.corpus.get(sel) {
                    mutator.input.extend_from_slice(&input);
                }
            }

            if mutator.input.len() == 0 {
                // Just make a blank input
                mutator.input.resize(1024 * 1024, 0u8);
            }

            // Mutate!
            mutator.mutate(rng.rand() % 10, &*corpus);

            // 1 in 8 Chance to change options
            if rng.rand() % 8 == 0 {
                options = rng.rand() as u32;
            }
            
            // 1 in 2 Chance of no extra options
            if rng.rand() % 2 == 0 {
                options = 0;
            }

            emu.fuzz_input.extend_from_slice(&mutator.input);
            assert!(emu.fuzz_input.len() <= len, "{} {}\n",
                emu.fuzz_input.len(), len);

            emu.fuzz_input.extend_from_slice(&options.to_ne_bytes());

            // If we're in repro mode, use the repro file
            if let Some(repro_file) = REPRO_MODE {
                emu.fuzz_input.clear();
                emu.fuzz_input.extend_from_slice(&std::fs::read(repro_file)
                    .expect("Failed to read repro file"));
            }

            // Inject the input
            emu.memory.write_from(VirtAddr(buf as usize), &emu.fuzz_input)
                .unwrap();
            emu.set_reg(Register::A1, emu.fuzz_input.len() as u64);

            let vmexit = loop {
                let vmexit = emu.run(&mut run_instrs,
                                     &mut local_stats.vm_cycles,
                                     &*corpus)
                    .expect_err("Failed to execute emulator");

                match vmexit {
                    VmExit::Syscall => {
                        if let Err(vmexit) = handle_syscall(&mut emu) {
                            break vmexit;
                        }
            
                        // Advance PC
                        let pc = emu.reg(Register::Pc);
                        emu.set_reg(Register::Pc, pc.wrapping_add(4));
                    }
                    _ => break vmexit,
                }
            };

            if let Some((fault_type, vaddr)) = vmexit.is_crash() {
                // Update crash stats
                local_stats.crashes += 1;

                // Attempt to update hash table
                let pc  = VirtAddr(emu.reg(Register::Pc) as usize);
                let key = (pc, fault_type, AddressType::from(vaddr));
                corpus.unique_crashes.entry_or_insert(&key, pc.0, || {
                    // Save the input and log it in the hash table
                    let hash = corpus.hasher.hash(&emu.fuzz_input);
                    corpus.input_hashes.entry_or_insert(
                            &hash, hash as usize, || {
                        let perms = emu.memory.peek_perms(VirtAddr(buf as usize),
                            len).unwrap();
                        let accessed: Vec<bool> = 
                            perms.iter().map(|perm| (perm.0 & PERM_ACC) != 0)
                            .collect();
                        corpus.inputs.push(
                            Box::new(Input::new(emu.timeout(),
                                emu.fuzz_input.clone(), accessed)));
                        Box::new(())
                    });

                    // Save the crashing file
                    let crash_fn = Path::new("crashes").join(
                        format!("{:#x}_{:?}_{:?}.crash",
                                (key.0).0, key.1, key.2));
                    let reg_fn = crash_fn.with_extension("regs");
                    print!("New crash {:?}\n", crash_fn);
                    std::fs::write(&crash_fn,
                        &emu.fuzz_input).expect("Failed to write fuzz input");
                    std::fs::write(&reg_fn,
                        &format!("{}", emu))
                        .expect("Failed to write crash register state");

                    Box::new(())
                });
            }

            // Update vmexit frequencies
            *local_stats.vmexits.entry(vmexit).or_insert(0) += 1;

            local_stats.instrs_execed += run_instrs;
            local_stats.fuzz_cases    += 1;
        }

        // Get access to statistics
        let mut stats = stats.lock().unwrap();

        stats.fuzz_cases    += local_stats.fuzz_cases;
        stats.crashes       += local_stats.crashes;
        stats.instrs_execed += local_stats.instrs_execed;
        stats.reset_cycles  += local_stats.reset_cycles;
        stats.vm_cycles     += local_stats.vm_cycles;

        for (vme, freq) in local_stats.vmexits.iter() {
            *stats.vmexits.entry(*vme).or_insert(0) += freq;
        }
        local_stats.vmexits.clear();

        // Compute amount of time during the batch
        let batch_elapsed = rdtsc() - batch_start;
        stats.total_cycles += batch_elapsed;
    }
}

pub struct Input {
    /// The instruction count of the most recently generated coverage from this
    /// input. This allows us to know how "deep" we need to fuzz this input
    instrs: u64,

    /// The raw input
    data: Vec<u8>,

    /// A sorted vector of indicies from `data` which are used during the fuzz
    /// case
    accessed: Vec<usize>,
}

impl Input {
    /// Creates a new input from an instruction count, a raw input, and the
    /// accessed mapping associating `data` bytes to accessed ones
    pub fn new(instrs: u64, data: Vec<u8>, accessed: Vec<bool>) -> Input {
        // Sorted vector of accessed indicies in `data`
        let mut avec = Vec::new();

        // Create the vector of indicies
        let num_acc = std::cmp::min(data.len(), accessed.len());
        for (ii, &is_acc) in accessed[..num_acc].iter().enumerate() {
            if is_acc {
                avec.push(ii);
            }
        }

        /*
        print!("New input {} bytes, {} accessed [{:.4}]\n",
            data.len(), avec.len(), avec.len() as f64 / data.len() as f64);*/

        Input {
            instrs,
            data,
            accessed: avec
        }
    }
}

/// Information about inputs and coverage
pub struct Corpus {
    /// Input hash table to dedup inputs
    pub input_hashes: Aht<u128, (), 1048576>,
    
    /// Linear list of all inputs
    pub inputs: AtomicVec<Input, 1048576>,
    
    /// Linear list of all corpus inputs
    pub corpus: AtomicVec<Vec<u8>, 1048576>,
    
    /// Unique crashes
    /// Tuple is (PC, FaultType, AddressType)
    pub unique_crashes: Aht<(VirtAddr, FaultType, AddressType), (), 1048576>,

    /// Code coverage, (to, from) edges for _all_ branches, including
    /// taken, not taken, indirect, and unconditional
    pub code_coverage: Aht<(VirtAddr, VirtAddr), (), 1048576>,
    
    /// Coverage for all types of coverage, (typ, info0, info1, info2) 
    pub coverage: Aht<(CoverageType, u64, u64, u64), (), 134217728>,

    /// Hasher
    pub hasher: FalkHasher,

    /// Coverage table
    pub coverage_table: Vec<(AtomicU64, AtomicU64)>,

    /// Coverage log file
    coverage_log: Mutex<File>,
    
    /// Lighthouse coverage log file
    lighthouse_log: Mutex<File>,

    /// Active compile jobs
    compile_jobs: Mutex<BTreeSet<u128>>,
}

impl InputDatabase for Corpus {
    fn num_inputs(&self) -> usize { self.inputs.len() }
    fn input(&self, idx: usize) -> Option<&[u8]> {
        self.inputs.get(idx).map(|x| {
            &x.data[..x.data.len() - 4]
        })
    }   
}

fn malloc_bp(emu: &mut Emulator) -> Result<(), VmExit> {
    if let Some(alc) = emu.memory.allocate(emu.reg(Register::A1) as usize) {
        emu.set_reg(Register::A0, alc.0 as u64);
        emu.set_reg(Register::Pc, emu.reg(Register::Ra));
    
        if VERBOSE_ALLOCS {
            print!("malloc returned {:#018x} - size was {:#x}\n",
                   alc.0, emu.reg(Register::A1));
        }

        Ok(())
    } else {
        // Cannot satisfy allocation, return out
        Err(VmExit::OutOfMemory)
    }
}

fn _calloc_bp(emu: &mut Emulator) -> Result<(), VmExit> {
    let nmemb = emu.reg(Register::A1) as usize;
    let size  = emu.reg(Register::A2) as usize;

    let result = size.checked_mul(nmemb).and_then(|size| {
        let alc = emu.memory.allocate(size)?;
        let tmp = emu.memory.peek(alc, size, Perm(PERM_WRITE))
            .expect("New allocation not writable?");
        tmp.iter_mut().for_each(|x| *x = 0);
        Some(alc)
    }).unwrap_or(VirtAddr(0));

    if result.0 == 0 {
        // Cannot satisfy allocation, return out
        return Err(VmExit::OutOfMemory);
    }

    if VERBOSE_ALLOCS {
        print!("calloc returned {:#018x} - size was {:#x}\n", result.0,
               size * nmemb);
    }

    emu.set_reg(Register::A0, result.0 as u64);
    emu.set_reg(Register::Pc, emu.reg(Register::Ra));
    Ok(())
}

fn _realloc_bp(emu: &mut Emulator) -> Result<(), VmExit> {
    let old_alc = VirtAddr(emu.reg(Register::A1) as usize);
    let size    = emu.reg(Register::A2) as usize;

    // Get the old allocation size
    let old_size = if old_alc == VirtAddr(0) {
        // No previous allocation specified, thus no size
        0
    } else {
        // Attempt to get the old allocation size
        emu.memory.get_alc(old_alc).ok_or(VmExit::InvalidFree(old_alc))?
    };

    // Compute the size to copy
    let to_copy = core::cmp::min(size, old_size);

    // Allocate the new memory
    let new_alc = emu.memory.allocate(size).and_then(|new_alc| {
        if VERBOSE_ALLOCS {
            print!("realloc {:#018x} -> {:#018x} - size {:#x} -> {:#x}\n",
                   old_alc.0,
                   new_alc.0,
                   old_size, size);
        }

        if old_alc != VirtAddr(0) {
            // Copy memory
            for ii in 0..to_copy {
                if let Ok(old) =
                        emu.memory.read::<u8>(VirtAddr(old_alc.0 + ii)) {
                    // Copy the memory only if we could read it from the old
                    // allocation. This will preserve the uninitialized state
                    // of bytes which haven't been initialized in the old
                    // allocation
                    emu.memory.write(VirtAddr(new_alc.0 + ii), old).unwrap();
                }
            }
            
            // Free the old allocation
            emu.memory.free(old_alc).expect("Failed to free old allocation?");
        }

        Some(new_alc)
    }).unwrap_or(VirtAddr(0));
    
    if new_alc.0 == 0 {
        // Cannot satisfy allocation, return out
        return Err(VmExit::OutOfMemory);
    }

    emu.set_reg(Register::A0, new_alc.0 as u64);
    emu.set_reg(Register::Pc, emu.reg(Register::Ra));
    Ok(())
}

fn free_bp(emu: &mut Emulator) -> Result<(), VmExit> {
    let base = VirtAddr(emu.reg(Register::A1) as usize);
    if base != VirtAddr(0) {
        if VERBOSE_ALLOCS {
            print!("free {:#018x}\n", base.0);
        }
        //emu.memory.free(base)?;
    }
    emu.set_reg(Register::Pc, emu.reg(Register::Ra));
    Ok(())
}

fn _end_case(_emu: &mut Emulator) -> Result<(), VmExit> {
    Err(VmExit::Exit)
}

fn snapshot(_emu: &mut Emulator) -> Result<(), VmExit> {
    Err(VmExit::Snapshot)
}

pub fn load_elf<P: AsRef<Path>>(filename: P, emu: &mut Emulator)
        -> io::Result<()> {
    use std::process::Command;

    // Invoke readelf to get the LOAD section offsets and information
    let output = Command::new("readelf")
        .arg("-W")
        .arg("-l")
        .arg(filename.as_ref().to_str().unwrap())
        .output()?;
    assert!(output.status.success(), "readelf returned error");
    let stdout = core::str::from_utf8(&output.stdout)
        .expect("Failed to get readelf stdout as a string");

    let mut entry_point = None;
    for line in stdout.lines() {
        if line.starts_with("Entry point 0x") {
            // Parse out the entry point
            entry_point = Some(u64::from_str_radix(&line[14..], 16)
                .expect("Entry point line malformed"));
        } else {
            let mut info = line.split_whitespace();

            // Check if this is a line indicating a load section
            if info.next() != Some("LOAD") {
                continue;
            }

            // Parse out info from the readelf output
            let offset = info.next().and_then(|x|
                usize::from_str_radix(&x[2..], 16).ok())
                .expect("Failed to parse offset");
            let virt_addr = info.next().and_then(|x|
                usize::from_str_radix(&x[2..], 16).ok())
                .expect("Failed to parse virt addr");
            let _phys_addr = info.next();
            let file_size = info.next().and_then(|x|
                usize::from_str_radix(&x[2..], 16).ok())
                .expect("Failed to parse file size");
            let mem_size = info.next().and_then(|x|
                usize::from_str_radix(&x[2..], 16).ok())
                .expect("Failed to parse memory size");
            let _align = info.next_back();

            let mut flags = info.fold(String::new(), |acc, x| acc + x + " ");
            flags += "   ";

            let read  = if &flags[0..1] == "R" { PERM_READ  } else { 0 };
            let write = if &flags[1..2] == "W" { PERM_WRITE } else { 0 };
            let exec  = if &flags[2..3] == "E" { PERM_EXEC  } else { 0 };

            // Load into memory
            emu.memory.load(&filename, &[
                Section {
                    file_off:    offset,
                    virt_addr:   VirtAddr(virt_addr),
                    file_size:   file_size,
                    mem_size:    mem_size,
                    permissions: Perm(read | write | exec),
                },
            ]).expect("Failed to load into emulator");
        }
    }
    
    // Invoke nm to get some symbol information
    let output = Command::new("nm")
        .arg(filename.as_ref().to_str().unwrap())
        .output()?;
    assert!(output.status.success(), "nm returned error");
    let stdout = core::str::from_utf8(&output.stdout)
        .expect("Failed to get nm stdout as a string");

    // Parse NM output
    for line in stdout.lines() {
        let mut info = line.split_whitespace();
        if info.clone().count() != 3 { continue; }

        let addr  = usize::from_str_radix(info.next().unwrap(), 16).unwrap();
        let _flag = info.next().unwrap();
        let name  = info.next().unwrap();

        // Register this symbol
        emu.add_symbol(name, VirtAddr(addr));
    }
 
    // Set the program entry point
    emu.set_reg(Register::Pc, entry_point.unwrap());
    Ok(())
}

fn main() -> io::Result<()> {
    std::fs::create_dir_all("inputs")?;
    std::fs::create_dir_all("crashes")?;

    // Create a corpus
    let corpus = Arc::new(Corpus {
        input_hashes: Aht::new(),
        inputs: AtomicVec::new(),
        hasher: FalkHasher::new(),
        unique_crashes: Aht::new(),
        code_coverage: Aht::new(),
        coverage: Aht::new(),
        corpus: AtomicVec::new(),
        coverage_log: Mutex::new(File::create("coverage.txt")
            .expect("Failed to create coverage file")),
        lighthouse_log: Mutex::new(File::create("lighthouse.txt")
            .expect("Failed to create lighthouse coverage file")),
        compile_jobs: Default::default(),
        coverage_table: (0..32 * 1024 * 1024).map(|_| {
            (AtomicU64::new(COVERAGE_ENTRY_EMPTY), AtomicU64::new(0))
        }).collect(),
    });
    
    // Create a JIT cache
    let _jit_cache = Arc::new(JitCache::new(VirtAddr(4 * 1024 * 1024)));

    // Create an emulator using the JIT
    let emu = Emulator::new(64 * 1024 * 1024);
    let mut emu = if REPRO_MODE.is_some() {
        emu
    } else {
        emu.enable_jit(_jit_cache)
    };
   
    // Load the initial corpus
    for filename in std::fs::read_dir("inputs")?{
        let filename = filename?.path();
        let data = std::fs::read(filename)?;

        // Add the corpus input to the corpus
        corpus.corpus.push(Box::new(data));
    }

    // Load the ELF into the memory
    load_elf("/home/pleb/fuzz_xml/fuzzer/fuzzer.sym", &mut emu)?;
    
    const FUZZ_START_SYM: &str = "fuzzme";
    
    // Register breakpoints
    emu.add_breakpoint(emu.resolve_symbol("_malloc_r").unwrap(), malloc_bp);
    emu.add_breakpoint(emu.resolve_symbol("_calloc_r").unwrap(), _calloc_bp);
    emu.add_breakpoint(emu.resolve_symbol("_realloc_r").unwrap(), _realloc_bp);
    emu.add_breakpoint(emu.resolve_symbol("_free_r").unwrap(), free_bp);
    emu.add_breakpoint(emu.resolve_symbol(FUZZ_START_SYM).unwrap(), snapshot);

    // Set up a stack
    let stack = emu.memory.allocate(1024 * 1024)
        .expect("Failed to allocate stack");
    emu.set_reg(Register::Sp, stack.0 as u64 + 1024 * 1024);

    // Set up the program name
    let progname = emu.memory.allocate(4096)
        .expect("Failed to allocate program name");
    emu.memory.write_from(progname, b"objdump\0")
        .expect("Failed to write program name");
    let arg1 = emu.memory.allocate(4096)
        .expect("Failed to allocate arg1");
    emu.memory.write_from(arg1, b"testfn\0")
        .expect("Failed to write arg2");

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
    push!(arg1.0); // Argv
    push!(progname.0); // Argv
    push!(2u64);   // Argc


    loop {
        // Run the emulator to a certain point
        let mut tmp = 0;
        let vmexit = emu.run_emu(&mut tmp, &*corpus)
            .expect_err("Failed to execute emulator");

        match vmexit {
            VmExit::Snapshot => {
                emu.remove_breakpoint(
                    emu.resolve_symbol(FUZZ_START_SYM).unwrap());
                break;
            }
            VmExit::Syscall => {
                print!("Syscall {}\n", emu.reg(Register::A7));
                if let Err(_vmexit) = handle_syscall(&mut emu) {
                    break;
                }
    
                // Advance PC
                let pc = emu.reg(Register::Pc);
                emu.set_reg(Register::Pc, pc.wrapping_add(4));
            }
            _ => break,
        }
    }

    print!("Took snapshot at {:#x}\n", emu.reg(Register::Pc));

    // Wrap the original emulator in an `Arc`
    let emu = Arc::new(emu);

    // Create a new stats structure
    let stats = Arc::new(Mutex::new(Statistics::default()));

    // Create the stats thread
    {
        let corpus = corpus.clone();
        let stats  = stats.clone();
        std::thread::spawn(move || {
            // Start a timer
            let start = Instant::now();

            let mut last_time = Instant::now();

            let mut log = File::create("stats.txt").unwrap();
            loop {
                std::thread::sleep(Duration::from_millis(10));
                    
                // Get access to the stats structure
                let stats   = stats.lock().unwrap();
                let elapsed = start.elapsed().as_secs_f64();

                write!(log, "{:.6},{},{},{},{}\n", elapsed, stats.fuzz_cases,
                       corpus.code_coverage.len(), corpus.unique_crashes.len(),
                       corpus.inputs.len())
                    .unwrap();

                if last_time.elapsed() >= Duration::from_millis(1000) {
                    let fuzz_cases = stats.fuzz_cases;
                    let instrs = stats.instrs_execed;

                    // Compute performance numbers
                    let resetc = stats.reset_cycles as f64 /
                        stats.total_cycles as f64;
                    let vmc = stats.vm_cycles as f64 /
                        stats.total_cycles as f64;

                    print!("[{:10.4}] cases {:10} | inputs {:10} | \
                            crashes {:8} | \
                            fcps {:8.0} | code {:7} | cov {:10} | \
                            eff Minst/sec {:10.1} | \
                            reset {:8.4} | vm {:8.4}\n",
                           elapsed, fuzz_cases, corpus.inputs.len(),
                           corpus.unique_crashes.len(),
                           fuzz_cases as f64 / elapsed,
                           corpus.code_coverage.len(),
                           corpus.coverage.len(),
                           (instrs as f64 / elapsed / 1_000_000.) / vmc,
                           resetc, vmc);

                    for (vmexit, &freq) in stats.vmexits.iter() {
                        if freq as f64 / fuzz_cases as f64 > 0.01 {
                            print!("{:15} [{:8.6}] {:x?}\n",
                                   freq, freq as f64 / fuzz_cases as f64,
                                   vmexit);
                        }
                    }

                    last_time = Instant::now();
                }
            }
        });
    }

    // Limit cores during repro mode
    let num_cores = if REPRO_MODE.is_some() {
        1
    } else {
        192
    };

    for thr_id in 0..num_cores {
        let new_emu = emu.fork();
        let stats   = stats.clone();
        let parent  = emu.clone();
        let corpus  = corpus.clone();
        std::thread::spawn(move || {
            worker(thr_id, new_emu, parent, stats, corpus);
        });
    }

    loop {
        std::thread::sleep(Duration::from_millis(5000));
    }
}

