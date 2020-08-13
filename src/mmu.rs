//! A software MMU with byte level permissions and uninitialized memory access
//! detection

use std::path::Path;
use std::collections::HashMap;
use crate::emulator::VmExit;
use crate::primitive::Primitive;

/// Block size used for resetting and tracking memory which has been modified
/// The larger this is, the fewer but more expensive memcpys() need to occur,
/// the small, the greater but less expensive memcpys() need to occur.
/// It seems the sweet spot is often 128-4096 bytes
pub const DIRTY_BLOCK_SIZE: usize = 64;

/// If `true` the logic for uninitialized memory tracking will be disabled and
/// all memory will be marked as readable if it has the RAW bit set
const DISABLE_UNINIT: bool = true;

// Don't change these, they're hardcoded in the JIT (namely write vs raw dist,
// during raw bit updates in writes)
pub const PERM_READ:  u8 = 1 << 0;
pub const PERM_WRITE: u8 = 1 << 1;
pub const PERM_EXEC:  u8 = 1 << 2;
pub const PERM_RAW:   u8 = 1 << 3;

/// Accessed bit, set when the byte is read, but not when it is written
pub const PERM_ACC: u8 = 1 << 4;

/// A permissions byte which corresponds to a memory byte and defines the
/// permissions it has
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Perm(pub u8);

/// A guest virtual address
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VirtAddr(pub usize);

/// Section information for a file
pub struct Section {
    pub file_off:    usize,
    pub virt_addr:   VirtAddr,
    pub file_size:   usize,
    pub mem_size:    usize,
    pub permissions: Perm,
}

/// An isolated memory space
#[derive(PartialEq)]
pub struct Mmu {
    /// Block of memory for this address space
    /// Offset 0 corresponds to address 0 in the guest address space
    memory: Vec<u8>,

    /// Holds the permission bytes for the corresponding byte in memory
    permissions: Vec<Perm>,

    /// Dirtied memory information
    dirty_state: DirtyState,

    /// Current base address of the next allocation
    cur_alc: VirtAddr,

    /// Map an active allocation to its size
    active_alcs: HashMap<VirtAddr, usize>,
}

/// Tracks the state of dirtied memory
#[derive(PartialEq)]
pub struct DirtyState {
    /// Tracks block indicies in `memory` which are dirty
    dirty: Vec<usize>,

    /// Tracks which parts of memory have been dirtied
    dirty_bitmap: Vec<u64>,
}

impl DirtyState {
    /// Updates the dirty map indicating that the byte at `addr` has been
    /// dirtied
    fn update_dirty(&mut self, addr: VirtAddr) {
        let block = addr.0 / DIRTY_BLOCK_SIZE;

        // Determine the bitmap position of the dirty block
        let idx = block / 64;
        let bit = block % 64;
        
        // Check if the block is not dirty
        if self.dirty_bitmap[idx] & (1 << bit) == 0 {
            // Block is not dirty, add it to the dirty list
            self.dirty.push(block);

            // Update the dirty bitmap
            self.dirty_bitmap[idx] |= 1 << bit;
        }
    }
}

impl Mmu {
    /// Create a new memory space which can hold `size` bytes
    pub fn new(size: usize) -> Self {
        Mmu {
            memory:      vec![0; size],
            permissions: vec![Perm(0); size],
            cur_alc:     VirtAddr(0x10000),
            active_alcs: Default::default(),
            dirty_state: DirtyState {
                dirty:        Vec::with_capacity(size / DIRTY_BLOCK_SIZE + 1),
                dirty_bitmap: vec![0u64; size / DIRTY_BLOCK_SIZE / 64 + 1],
            },
        }
    }

    /// Fork from an existing MMU
    pub fn fork(&self) -> Self {
        let size = self.memory.len();

        Mmu {
            memory:      self.memory.clone(),
            permissions: self.permissions.clone(),
            cur_alc:     self.cur_alc.clone(),
            active_alcs: self.active_alcs.clone(),
            dirty_state: DirtyState {
                dirty:        Vec::with_capacity(size / DIRTY_BLOCK_SIZE + 1),
                dirty_bitmap: vec![0u64; size / DIRTY_BLOCK_SIZE / 64 + 1],
            }
        }
    }

    /// Restores memory back to the original state (eg. restores all dirty
    /// blocks to the state of `other`)
    pub fn reset(&mut self, other: &Mmu) {
        for &block in &self.dirty_state.dirty {
            // Get the start and end addresses of the dirtied memory
            let start = block * DIRTY_BLOCK_SIZE;
            let end   = (block + 1) * DIRTY_BLOCK_SIZE;

            // Zero the bitmap. This hits wide, but it's fine, we have to do
            // a 64-bit write anyways, no reason to compute the bit index
            self.dirty_state.dirty_bitmap[block / 64] = 0;

            // Restore memory state
            self.memory[start..end].copy_from_slice(&other.memory[start..end]);

            // Restore permissions
            self.permissions[start..end].copy_from_slice(
                &other.permissions[start..end]);
        }

        // Clear the dirty list
        self.dirty_state.dirty.clear();

        // Restore allocator state
        self.cur_alc = other.cur_alc;

        // Clear active allocation state
        self.active_alcs.clear();
        self.active_alcs.extend(other.active_alcs.iter());

        if false {
            // Tests to make sure everything to reset perfectly
            assert!(self.cur_alc == other.cur_alc);
            assert!(self.memory == other.memory);
            assert!(self.permissions == other.permissions);
            assert!(self.active_alcs == other.active_alcs);
        }
    }

    /// Allocate a region of memory as RW in the address space
    pub fn allocate(&mut self, size: usize) -> Option<VirtAddr> {
        // Add some padding and alignment
        let align_size = (size + 0x1f) & !0xf;

        // Get the current allocation base
        let base = self.cur_alc;

        // Cannot allocate
        if base.0 >= self.memory.len() {
            return None;
        }

        // Update the allocation size
        self.cur_alc = VirtAddr(self.cur_alc.0.checked_add(align_size)?);

        // Could not satisfy allocation without going OOM
        if self.cur_alc.0 > self.memory.len() {
            return None;
        }

        // Mark the memory as un-initialized and writable
        self.set_permissions(base, size, Perm(PERM_RAW | PERM_WRITE));

        // Log the allocation
        self.active_alcs.insert(base, size);

        Some(base)
    }

    /// Get the size of an active allocation if `base` is an active allocation
    pub fn get_alc(&self, base: VirtAddr) -> Option<usize> {
        self.active_alcs.get(&base).copied()
    }

    /// Free a region of memory based on the allocation from a prior `allocate`
    /// call
    pub fn free(&mut self, base: VirtAddr) -> Result<(), VmExit> {
        if let Some(size) = self.active_alcs.remove(&base) {
            // Clear permissions
            self.set_permissions(base, size, Perm(0));

            Ok(())
        } else {
            Err(VmExit::InvalidFree(base))
        }
    }

    /// Apply permissions to a region of memory
    pub fn set_permissions(&mut self, addr: VirtAddr, size: usize,
                           mut perm: Perm) -> Option<()> {
        // Fast path, nothing to change
        if size == 0 { return Some(()); }

        if DISABLE_UNINIT {
            // If memory is marked as RAW, mark it as readable right away if
            // we have uninit tracking disabled
            if perm.0 & PERM_RAW != 0 { perm.0 |= PERM_READ; }
        }

        // Apply permissions
        self.permissions.get_mut(addr.0..addr.0.checked_add(size)?)?
            .iter_mut().for_each(|x| *x = perm);
        
        // Compute dirty bit blocks
        let block_start = addr.0 / DIRTY_BLOCK_SIZE;
        let block_end   = (addr.0 + size) / DIRTY_BLOCK_SIZE;
        for block in block_start..=block_end {
            // Determine the bitmap position of the dirty block
            let idx = block / 64;
            let bit = block % 64;
            
            // Check if the block is not dirty
            if self.dirty_state.dirty_bitmap[idx] & (1 << bit) == 0 {
                // Block is not dirty, add it to the dirty list
                self.dirty_state.dirty.push(block);

                // Update the dirty bitmap
                self.dirty_state.dirty_bitmap[idx] |= 1 << bit;
            }
        }

        Some(())
    }

    /// Get the maximum size of guest memory
    #[inline]
    pub fn len(&self) -> usize {
        self.memory.len()
    }

    /// Get the dirty list length
    #[inline]
    pub fn dirty_len(&self) -> usize {
        self.dirty_state.dirty.len()
    }

    /// Set the dirty list length
    #[inline]
    pub unsafe fn set_dirty_len(&mut self, len: usize) {
        self.dirty_state.dirty.set_len(len);
    }

    /// Get the tuple of (memory ptr, permissions pointer, dirty pointer,
    /// dirty bitmap pointer)
    #[inline]
    pub fn jit_addrs(&self) -> (usize, usize, usize, usize) {
        (
            self.memory.as_ptr() as usize,
            self.permissions.as_ptr() as usize,
            self.dirty_state.dirty.as_ptr() as usize,
            self.dirty_state.dirty_bitmap.as_ptr() as usize,
        )
    }

    /// Write the bytes from `buf` into `addr`
    pub fn write_from(&mut self, addr: VirtAddr, buf: &[u8])
            -> Result<(), VmExit> {
        let perms =
            self.permissions.get_mut(addr.0..addr.0.checked_add(buf.len())
                .ok_or(VmExit::AddressIntegerOverflow)?)
            .ok_or(VmExit::AddressMiss(addr, buf.len()))?;

        // Check permissions
        let mut has_raw = false;
        for (idx, &perm) in perms.iter().enumerate() {
            // Accumulate if any permission has the raw bit set, this will
            // allow us to bypass permission updates if no RAW is in use
            has_raw |= (perm.0 & PERM_RAW) != 0;

            if (perm.0 & PERM_WRITE) == 0 {
                // Permission denied, return error
                return Err(VmExit::WriteFault(VirtAddr(addr.0 + idx)));
            }
        }

        // Copy the buffer into memory!
        self.memory[addr.0..addr.0 + buf.len()].copy_from_slice(buf);

        // Compute dirty bit blocks
        let block_start = addr.0 / DIRTY_BLOCK_SIZE;
        let block_end   = (addr.0 + buf.len()) / DIRTY_BLOCK_SIZE;
        for block in block_start..=block_end {
            // Determine the bitmap position of the dirty block
            let idx = block / 64;
            let bit = block % 64;
            
            // Check if the block is not dirty
            if self.dirty_state.dirty_bitmap[idx] & (1 << bit) == 0 {
                // Block is not dirty, add it to the dirty list
                self.dirty_state.dirty.push(block);

                // Update the dirty bitmap
                self.dirty_state.dirty_bitmap[idx] |= 1 << bit;
            }
        }

        // Update RaW bits
        if has_raw {
            perms.iter_mut().for_each(|x| {
                if (x.0 & PERM_RAW) != 0 {
                    // Mark memory as readable
                    *x = Perm((x.0 | PERM_READ) & (!PERM_RAW));
                }
            });
        }

        Ok(())
    }
    
    /// Return a mutable slice to permissions at `addr` for `size` bytes
    pub fn peek_perms(&mut self, addr: VirtAddr, size: usize)
            -> Result<&mut [Perm], VmExit> {
        self.permissions.get_mut(addr.0..addr.0.checked_add(size)
            .ok_or(VmExit::AddressIntegerOverflow)?)
            .ok_or(VmExit::AddressMiss(addr, size))
    }
 
    /// Return a mutable slice to memory at `addr` for `size` bytes that
    /// has been validated to match all `exp_perms`
    pub fn peek(&mut self, addr: VirtAddr, size: usize,
                exp_perms: Perm) -> Result<&mut [u8], VmExit> {
        let perms =
            self.permissions.get_mut(addr.0..addr.0.checked_add(size)
                .ok_or(VmExit::AddressIntegerOverflow)?)
            .ok_or(VmExit::AddressMiss(addr, size))?;

        // Check permissions
        for (idx, perm) in perms.iter().enumerate() {
            if (perm.0 & exp_perms.0) != exp_perms.0 {
                if exp_perms.0 == PERM_READ && (perm.0 & PERM_RAW) != 0 {
                    // If we were attempting a normal read, and the readable
                    // memory was unreadable but had the RAW bit set, report
                    // it as an uninitialized memory access rather than a read
                    // access
                    return Err(VmExit::UninitFault(VirtAddr(addr.0 + idx)));
                } else if exp_perms.0 == PERM_WRITE {
                    return Err(VmExit::WriteFault(VirtAddr(addr.0 + idx)));
                } else {
                    return Err(VmExit::ReadFault(VirtAddr(addr.0 + idx)));
                }
            }
        }

        // Update dirty bits
        for (ii, perm) in perms.iter_mut().enumerate() {
            // Check if we're getting write access
            if (exp_perms.0 & PERM_WRITE) != 0 {
                // Propagate RAW
                if (perm.0 & PERM_RAW) != 0 {
                    perm.0 |= PERM_READ;
                }

                // Update dirty bits
                self.dirty_state.update_dirty(VirtAddr(addr.0 + ii));
            }

            // Indicate the memory has been accessed
            if (exp_perms.0 & PERM_READ) != 0 {
                perm.0 |= PERM_ACC;
                self.dirty_state.update_dirty(VirtAddr(addr.0 + ii));
            }
        }
       
        // Return a slice to the memory
        Ok(&mut self.memory[addr.0..addr.0 + size])
    }
   
    /// Read the memory at `addr` into `buf`
    /// This function checks to see if all bits in `exp_perms` are set in the
    /// permission bytes. If this is zero, we ignore permissions entirely.
    pub fn read_into_perms(&mut self, addr: VirtAddr, buf: &mut [u8],
                           exp_perms: Perm) -> Result<(), VmExit> {
        let perms =
            self.permissions.get_mut(addr.0..addr.0.checked_add(buf.len())
                .ok_or(VmExit::AddressIntegerOverflow)?)
            .ok_or(VmExit::AddressMiss(addr, buf.len()))?;

        // Check permissions
        for (idx, &perm) in perms.iter().enumerate() {
            if (perm.0 & exp_perms.0) != exp_perms.0 {
                if exp_perms.0 == PERM_READ && (perm.0 & PERM_RAW) != 0 {
                    // If we were attempting a normal read, and the readable
                    // memory was unreadable but had the RAW bit set, report
                    // it as an uninitialized memory access rather than a read
                    // access
                    return Err(VmExit::UninitFault(VirtAddr(addr.0 + idx)));
                } else {
                    return Err(VmExit::ReadFault(VirtAddr(addr.0 + idx)));
                }
            }
        }

        // Copy the memory
        buf.copy_from_slice(&self.memory[addr.0..addr.0 + buf.len()]);
        
        // Indicate that this memory has been accessed
        for (ii, perm) in perms.iter_mut().enumerate() {
            perm.0 |= PERM_ACC;
            self.dirty_state.update_dirty(VirtAddr(addr.0 + ii));
        }

        Ok(())
    }

    /// Read the memory at `addr` into `buf`
    pub fn read_into(&mut self, addr: VirtAddr, buf: &mut [u8])
            -> Result<(), VmExit> {
        self.read_into_perms(addr, buf, Perm(PERM_READ))
    }

    /// Read a type `T` at `vaddr` expecting `perms`
    pub fn read_perms<T: Primitive>(&mut self, addr: VirtAddr,
                                    exp_perms: Perm) -> Result<T, VmExit> {
        let mut tmp = [0u8; 16];
        self.read_into_perms(addr, &mut tmp[..core::mem::size_of::<T>()],
            exp_perms)?;
        Ok(unsafe { core::ptr::read_unaligned(tmp.as_ptr() as *const T) })
    }
    
    /// Read a type `T` at `vaddr`
    pub fn read<T: Primitive>(&mut self, addr: VirtAddr) -> Result<T, VmExit> {
        self.read_perms(addr, Perm(PERM_READ))
    }
    
    /// Write a `val` to `addr`
    pub fn write<T: Primitive>(&mut self, addr: VirtAddr,
                               val: T) -> Result<(), VmExit> {
        let tmp = unsafe {
            core::slice::from_raw_parts(&val as *const T as *const u8,
                                        core::mem::size_of::<T>())
        };

        self.write_from(addr, tmp)
    }

    /// Load a file into the emulators address space using the sections as
    /// described
    pub fn load<P: AsRef<Path>>(&mut self, filename: P,
                                sections: &[Section]) -> Option<()> {
        // Read the input file
        let contents = std::fs::read(filename).ok()?;

        // Go through each section and load it
        for section in sections {
            // Set memory to writable
            self.set_permissions(section.virt_addr, section.mem_size,
                                        Perm(PERM_WRITE))?;

            // Write in the original file contents
            self.write_from(section.virt_addr,
                contents.get(
                    section.file_off..
                    section.file_off.checked_add(section.file_size)?)?
                ).ok()?;

            // Write in any padding with zeros
            if section.mem_size > section.file_size {
                let padding = vec![0u8; section.mem_size - section.file_size];
                self.write_from(
                    VirtAddr(section.virt_addr.0
                             .checked_add(section.file_size)?),
                    &padding).ok()?;
            }
            
            // Demote permissions to originals
            self.set_permissions(section.virt_addr, section.mem_size,
                                        section.permissions)?;

            // Update the allocator beyond any sections we load
            self.cur_alc = VirtAddr(std::cmp::max(
                self.cur_alc.0,
                (section.virt_addr.0 + section.mem_size + 0xfff) & !0xfff
            ));
        }

        Some(())
    }
}

