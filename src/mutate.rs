//! Basic fuzzer mutation strategies

struct Mutation {
    /// Input vector to mutate, this is just an entire input files bytes
    pub input: Vec<u8>,

    /// If non-zero length, this contains a list of valid indicies into
    /// `input`, indicating which bytes of the input should mutated. This often
    /// comes from instrumentation like access tracking or taint tracking to
    /// indicate which parts of the input are used. This will prevent us from
    /// corrupting parts of the file which have zero effect on the program.
    ///
    /// It's possible you can have this take any meaning you want, all it does
    /// is limit the corruption/splicing locations to the indicies in this
    /// vector. Feel free to change this to have different meanings, like
    /// indicate indicies which are used in comparison instructions!
    pub accessed: Vec<usize>,
}

impl Mutation {
    /// Performs standard mutation of the input
    pub fn mutate(&mut self) {
        let strategies = [
            Self::Shrink
        ];
    }

    pub fn shrink(&mut self) {
    }
}
