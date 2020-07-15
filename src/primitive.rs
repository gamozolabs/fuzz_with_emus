//! Generic support for primitive types which are safe to cast

pub unsafe trait Primitive: Default + Clone + Copy {}
unsafe impl Primitive for u8    {}
unsafe impl Primitive for u16   {}
unsafe impl Primitive for u32   {}
unsafe impl Primitive for u64   {}
unsafe impl Primitive for u128  {}
unsafe impl Primitive for usize {}
unsafe impl Primitive for i8    {}
unsafe impl Primitive for i16   {}
unsafe impl Primitive for i32   {}
unsafe impl Primitive for i64   {}
unsafe impl Primitive for i128  {}
unsafe impl Primitive for isize {}

