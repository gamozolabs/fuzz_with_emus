use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::collections::BTreeMap;
use crate::mmu::VirtAddr;

#[cfg(target_os="windows")]
pub fn alloc_rwx(size: usize) -> &'static mut [u8] {
    extern {
        fn VirtualAlloc(lpAddress: *const u8, dwSize: usize,
                        flAllocationType: u32, flProtect: u32) -> *mut u8;
    }

    unsafe {
        const PAGE_EXECUTE_READWRITE: u32 = 0x40;

        const MEM_COMMIT:  u32 = 0x00001000;
        const MEM_RESERVE: u32 = 0x00002000;

        let ret = VirtualAlloc(0 as *const _, size, MEM_COMMIT | MEM_RESERVE,
                               PAGE_EXECUTE_READWRITE);
        assert!(!ret.is_null());

        std::slice::from_raw_parts_mut(ret, size)
    }
}

#[cfg(target_os="linux")]
pub fn alloc_rwx(size: usize) -> &'static mut [u8] {
    extern {
        fn mmap(addr: *mut u8, length: usize, prot: i32, flags: i32, fd: i32,
                offset: usize) -> *mut u8;
    }

    unsafe {
        // Alloc RWX and MAP_PRIVATE | MAP_ANON
        let ret = mmap(0 as *mut u8, size, 7, 34, -1, 0);
        assert!(!ret.is_null());
        
        std::slice::from_raw_parts_mut(ret, size)
    }
}

/// A cache which stores cached JIT blocks and translation tables to them
pub struct JitCache {
    /// A vector which contains the addresses of JIT code for the corresponding
    /// guest virtual address.
    ///
    /// Ex. jit_addr = jitcache.blocks[Guest Virtual Address / 4];
    ///
    /// An entry which is a zero indicates the block has not yet been
    /// translated.
    ///
    /// The blocks are referenced by the guest virtual address divided by 4
    /// because all RISC-V instructions are 4 bytes (for the non-compressed
    /// variant)
    blocks: Box<[AtomicUsize]>,

    /// The raw JIT RWX backing, the amount of bytes in use, and a dedup
    /// table
    jit: Mutex<(&'static mut [u8], usize, BTreeMap<Vec<u8>, usize>)>,
}

// JIT calling convention
// rax - Scratch
// rbx - Scratch
// rcx - Scratch
// rdx - Scratch
// rsi - Scratchpad memory
// r8  - Pointer to the base of mmu.memory
// r9  - Pointer to the base of mmu.permissions
// r10 - Pointer to the base of mmu.dirty
// r11 - Pointer to the base of mmu.dirty_bitmap
// r12 - Dirty index for the dirty list
// r13 - Pointer to emu.registers
// r14 - Pointer to the base of jitcache.blocks
// r15 - Number of instructions executed
//
// JIT return code (in rax)
// In all cases rbx = PC to resume execution at upon reentry
// 1 - Branch resolution issue
// 2 - ECALL instruction
// 3 - EBREAK instruction
// 4 - Read fault, rcx = guest faulting address
// 5 - Write fault, rcx = guest faulting address
// 6 - Instruction timeout
// 7 - Breakpoint, rcx = reentry point
// 8 - Invalid opcode

impl JitCache {
    /// Allocates a new `JitCache` which is capable of handling up to
    /// `max_guest_addr` in executable code.
    pub fn new(max_guest_addr: VirtAddr) -> Self {
        JitCache {
            // Allocate a zeroed out block cache
            blocks: (0..(max_guest_addr.0 + 3) / 4).map(|_| {
                AtomicUsize::new(0)
            }).collect::<Vec<_>>().into_boxed_slice(),
            jit:
                Mutex::new((alloc_rwx(256 * 1024 * 1024), 0, BTreeMap::new())),
        }
    }

    /// Get the address of the JIT block translation table
    #[inline]
    pub fn translation_table(&self) -> usize {
        self.blocks.as_ptr() as usize
    }

    /// Returns the maximum number of blocks this `JitCache` can translate
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Look up the JIT address for a given guest address
    #[inline]
    pub fn lookup(&self, addr: VirtAddr) -> Option<usize> {
        // Make sure the address is aligned
        if addr.0 & 3 != 0 {
            return None;
        }

        let addr = self.blocks.get(addr.0 / 4)?.load(Ordering::SeqCst);
        if addr == 0 {
            None
        } else {
            Some(addr)
        }
    }

    /// Add a JIT to the JIT cache, the `code` are the raw bytes of the
    /// compiled JIT and the `BTreeMap` converts guest addresses into JIT
    /// addresses
    pub fn add_mappings(&self, addr: VirtAddr, code: &[u8],
                        mappings: &BTreeMap<VirtAddr, usize>) -> usize {
        // Get exclusive access to the JIT
        let mut jit = self.jit.lock().unwrap();

        // Determine if any of the guest addresses are new to the JIT, if even
        // one is, then we have to insert the JIT into the cache
        let has_new = mappings.keys().any(|&x| self.lookup(x).is_none());
        if !has_new {
            // We have nothing new, just give the JIT address `addr`
            return self.lookup(addr).unwrap();
        }

        // Check if we already have identical code
        let new_addr = if let Some(&existing) = jit.2.get(code) {
            // We have identical code, alias this code for the requested PC
            existing
        } else {
            // Compute the aligned size of code, this ensures we can do aligned
            // vector operations because we ensure alignment of loaded JITs
            let align_size = (code.len() + 0x3f) & !0x3f;

            // Number of remaining bytes in the JIT storage
            let jit_inuse  = jit.1;
            let jit_remain = jit.0.len() - jit_inuse;
            assert!(jit_remain > align_size, "Out of space in JIT");

            // Copy the new code into the JIT
            jit.0[jit_inuse..jit_inuse + code.len()].copy_from_slice(code);

            // Compute the address of the JIT we're inserting
            let new_addr = jit.0[jit_inuse..].as_ptr() as usize;
            
            // Update the in use for the JIT
            jit.1 += align_size;

            // Update the dedup table
            assert!(jit.2.insert(code.into(), new_addr).is_none());

            new_addr
        };

        // Update the JIT lookup address
        for (addr, offset) in mappings {
            self.blocks[addr.0 / 4].store(new_addr + offset, Ordering::SeqCst);
        }

        // Return the newly allocated JIT
        self.lookup(addr).unwrap()
    }
}

