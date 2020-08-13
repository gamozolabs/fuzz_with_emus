//! Functions to set affinity in an OS agnostic way

#[cfg(unix)]
pub fn set_affinity(core: usize) -> Result<(), ()> {
    extern "system" {
        fn sched_setaffinity(pid: usize, cpusetsize: usize,
                             mask: *const usize) -> i32;
    }

    const USIZE_BITS: usize = core::mem::size_of::<usize>() * 8;

    let mut mask = [0usize; 32];
    mask[core / USIZE_BITS] |= 1 << (core % USIZE_BITS);

    unsafe {
        if sched_setaffinity(0, std::mem::size_of_val(&mask),
                mask.as_ptr()) == 0 {
            Ok(())
        } else {
            Err(())
        }
    }
}

#[cfg(windows)]
pub fn set_affinity(core: usize) -> Result<(), ()> {
    extern "system" {
        fn GetCurrentThread() -> usize;
        fn SetThreadAffinityMask(hThread: usize,
                                 dwThreadAffinityMask: usize) -> usize;
    }

    assert!(core < 64, "Yeah, we don't support more than 64 cores here");

    unsafe {
        if SetThreadAffinityMask(GetCurrentThread(), 1usize << core) != 0 {
            Ok(())
        } else {
            Err(())
        }
    }
}

