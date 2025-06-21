pub mod elaborate;
pub(super) mod gather;
pub(super) mod utils;
pub(super) mod vd_base;
pub(super) mod visit_expr;

pub use elaborate::elaborate;

#[derive(Default)]
pub enum Simulator {
  VCS,
  Verilator,
  #[default]
  None,
}
