"""Runtime code generation for Assassyn simulator."""



def dump_runtime(fd):
    """Generate the runtime module.

    This matches the Rust function in src/backend/simulator/runtime.rs
    """
    # Add imports
    fd.write("""
use std::collections::VecDeque;
use std::collections::BTreeMap;
use num_bigint::{BigInt, BigUint, ToBigInt, ToBigUint};
use num_traits::Num;
use std::fs::read_to_string;
    """)

    # Add runtime types and implementations
    fd.write("""
pub trait Cycled {
  fn cycle(&self) -> usize;
  fn pusher(&self) -> &'static str;
}

pub struct ArrayWrite<T: Sized + Default + Clone> {
  cycle: usize,
  addr: usize,
  data: T,
  pusher: &'static str,
}

impl <T: Sized + Default + Clone> ArrayWrite<T> {
  pub fn new(cycle: usize, addr: usize, data: T, pusher: &'static str) -> Self {
    ArrayWrite { cycle, addr, data, pusher }
  }
}

pub struct Array<T: Sized + Default + Clone> {
  pub payload: Vec<T>,
  pub write: XEQ<ArrayWrite<T>>,
}

impl <T: Sized + Default + Clone> Array<T> {
  pub fn new(n: usize) -> Self {
    Array {
      payload: vec![T::default(); n],
      write: XEQ::new(),
    }
  }
  pub fn new_with_init(payload: Vec<T>) -> Self {
    Array {
      payload,
      write: XEQ::new(),
    }
  }
  pub fn tick(&mut self, cycle: usize) {
    if let Some(event) = self.write.pop(cycle) {
      self.payload[event.addr] = event.data;
    }
  }
}

pub struct FIFOPush<T: Sized> {
  cycle: usize,
  data: T,
  pusher: &'static str,
}

impl <T: Sized> FIFOPush<T> {
  pub fn new(cycle: usize, data: T, pusher: &'static str) -> Self {
    FIFOPush { cycle, data, pusher }
  }
}

pub struct FIFOPop {
  cycle: usize,
  pusher: &'static str,
}

impl FIFOPop {
  pub fn new(cycle: usize, pusher: &'static str) -> Self {
    FIFOPop { cycle, pusher }
  }
}

pub struct FIFO<T: Sized> {
  pub payload: VecDeque<T>,
  pub push: XEQ<FIFOPush<T>>,
  pub pop: XEQ<FIFOPop>,
}

impl <T: Sized> FIFO<T> {
  pub fn new() -> Self {
    FIFO {
      payload: VecDeque::new(),
      push: XEQ::new(),
      pop: XEQ::new(),
    }
  }

  pub fn is_empty(&self) -> bool {
    self.payload.is_empty()
  }

  pub fn front(&self) -> Option<&T> {
    self.payload.front()
  }

  pub fn tick(&mut self, cycle: usize) {
    if let Some(_) = self.pop.pop(cycle) {
      self.payload.pop_front().unwrap();
    }
    if let Some(event) = self.push.pop(cycle) {
      self.payload.push_back(event.data);
    }
  }
}

impl <T: Sized + Default + Clone> Cycled for ArrayWrite<T> {
  fn cycle(&self) -> usize {
    self.cycle
  }
  fn pusher(&self) -> &'static str {
    self.pusher
  }
}

impl <T: Sized> Cycled for FIFOPush<T> {
  fn cycle(&self) -> usize {
    self.cycle
  }
  fn pusher(&self) -> &'static str {
    self.pusher
  }
}

impl Cycled for FIFOPop {
  fn cycle(&self) -> usize {
    self.cycle
  }
  fn pusher(&self) -> &'static str {
    self.pusher
  }
}

pub struct XEQ<T: Sized + Cycled> {
  q: BTreeMap<usize, T>,
}

impl <T: Sized + Cycled>XEQ<T> {

  pub fn new() -> Self {
    XEQ { q: BTreeMap::new(), }
  }

  pub fn push(&mut self, event: T) {
    if let Some(a) = self.q.get(&event.cycle()) {
      panic!("{}: Already occupied by {}, cannot accept {}!",
        cyclize(a.cycle()), a.pusher(), event.pusher());
    } else {
      self.q.insert(event.cycle(), event);
    }
  }

  pub fn pop(&mut self, current: usize) -> Option<T> {
    if self.q.first_key_value().map_or(false, |(cycle, _)| *cycle >= current) {
      self.q.pop_first().map(|(_, event)| event)
    } else {
      None
    }
  }
}
    """)

    # Add utility functions
    fd.write("""
pub fn cyclize(stamp: usize) -> String {
  format!("Cycle @{}.{:02}", stamp / 100, stamp % 100)
}

pub fn load_hex_file<T: Num>(array: &mut Vec<T>, init_file: &str) {
  let mut idx = 0;
  for line in read_to_string(init_file)
    .expect("can not open hex file")
    .lines()
  {
    let line = if let Some(to_strip) = line.find("//") {
      line[..to_strip].trim()
    } else {
      line.trim()
    };
    if line.len() == 0 {
      continue;
    }
    let line = line.replace("_", "");
    if line.starts_with("@") {
      let addr = usize::from_str_radix(&line[1..], 16).unwrap();
      idx = addr;
      continue;
    }
    array[idx] = T::from_str_radix(line.as_str(), 16).ok().unwrap();
    idx += 1;
  }
}

pub trait ValueCastTo<T> {
  fn cast(&self) -> T;
}
    """)

    # Generate type casting implementations
    fd.write("impl ValueCastTo<bool> for bool { " +
              "fn cast(&self) -> bool { self.clone() } }\n")

    bigints = ["BigInt", "BigUint"]
    for i in range(2):
        bigint = bigints[i]
        other = bigints[1 - i]

        # Self cast
        fd.write(f"impl ValueCastTo<{bigint}> for {bigint} {{ " +
                  f"fn cast(&self) -> {bigint} {{ self.clone() }} }}\n")

        # Cross cast between BigInt and BigUint
        fd.write(f"""impl ValueCastTo<{other}> for {bigint} {{
          fn cast(&self) -> {other} {{ self.to_{other.lower()}().unwrap() }}
        }}\n""")

        # Bool to BigInt/BigUint
        fd.write(f"""impl ValueCastTo<{bigint}> for bool {{
          fn cast(&self) -> {bigint} {{
            if *self {{
              1.to_{bigint.lower()}().unwrap()
            }} else {{
              0.to_{bigint.lower()}().unwrap()
            }}
          }}
        }}\n""")

        # BigInt/BigUint to bool
        fd.write(f"""impl ValueCastTo<bool> for {bigint} {{
          fn cast(&self) -> bool {{
            !self.eq(&0.to_{bigint.lower()}().unwrap())
          }}
        }}\n""")

    # Generate integer casting implementations
    for sign_i in range(2):
        for i in range(3, 7):
            src_ty = f"{'ui'[sign_i]}{1 << i}"

            # To bool
            fd.write(f"impl ValueCastTo<bool> for {src_ty} {{ " +
                      "fn cast(&self) -> bool { *self != 0 } }\n")

            # From bool
            fd.write(f"""impl ValueCastTo<{src_ty}> for bool {{
                fn cast(&self) -> {src_ty} {{
                  if *self {{ 1 }} else {{ 0 }}
                }}
              }}\n""")

            # To BigInt/BigUint
            for bigint in bigints:
                fd.write(f"""impl ValueCastTo<{bigint}> for {src_ty} {{
                  fn cast(&self) -> {bigint} {{ self.to_{bigint.lower()}().unwrap() }}
                }}\n""")

            # From BigInt
            fd.write(f"""impl ValueCastTo<{src_ty}> for BigInt {{
                fn cast(&self) -> {src_ty} {{
                  let (sign, data) = self.to_u64_digits();
                  if data.is_empty() {{
                    return 0;
                  }}
                  match sign {{
                    num_bigint::Sign::Plus => data[0] as {src_ty},
                    num_bigint::Sign::Minus =>
                      ((!data[0] + 1) & ({src_ty}::MAX as u64)) as {src_ty},
                    num_bigint::Sign::NoSign => data[0] as {src_ty},
                  }}
                }}
              }}\n""")

            # From BigUint
            fd.write(f"""impl ValueCastTo<{src_ty}> for BigUint {{
                fn cast(&self) -> {src_ty} {{
                  let data = self.to_u64_digits();
                  if data.is_empty() {{
                    return 0;
                  }} else {{
                    return data[0] as {src_ty};
                  }}
                }}
              }}\n""")

            # Between integer types
            for sign_j in range(2):
                for j in range(3, 7):
                    dst_ty = f"{'ui'[sign_j]}{1 << j}"

                    if i == j and sign_i == sign_j:
                        # Self cast
                        fd.write(f"impl ValueCastTo<{dst_ty}> for {src_ty} {{ " +
                                  f"fn cast(&self) -> {dst_ty} {{ self.clone() }} }}\n")
                    else:
                        # Cross cast
                        fd.write(f"impl ValueCastTo<{dst_ty}> for {src_ty} {{ " +
                                  f"fn cast(&self) -> {dst_ty} {{ *self as {dst_ty} }} }}\n")

    # End file with newline
    fd.write("\n")
    return True
