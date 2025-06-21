use std::collections::HashMap;

use crate::{
  backend::common::namify,
  builder::system::ExposeKind,
  ir::{
    expr::subcode,
    instructions,
    node::{ExprRef, IsElement, NodeKind},
    Expr, IntImm, Opcode, Typed, FIFO,
  },
};

use super::utils::declare_logic;
use super::utils::{self, Field};
use super::{
  elaborate::{dump_arith_ref, dump_ref, fifo_name, VerilogDumper},
  gather::Gather,
};

pub(super) fn visit_expr_impl(vd: &mut VerilogDumper<'_, '_>, expr: ExprRef<'_>) -> Option<String> {
  // If an expression is externally used by another downstream module,
  // we need to generate the corresponding assignment logic here.
  let (decl, expose) =
    if expr.get_opcode().is_valued() && !matches!(expr.get_opcode(), Opcode::Bind) {
      let id = namify(&expr.upcast().to_string(vd.sys));
      let expose = if vd.external_usage.is_externally_used(&expr) {
        format!(
          "  assign expose_{id} = {id};\n  assign expose_{id}_valid = executed && {};\n",
          vd.get_pred().unwrap_or("1".to_string())
        )
      } else {
        "".into()
      };
      (Some((id, expr.dtype())), expose)
    } else {
      (None, "".into())
    };

  let exposed_map: HashMap<_, _> = vd.sys.exposed_nodes().collect();
  let expr_clone = expr.clone();
  let expose_out_str = if let Some(exposed_kind) = exposed_map.get(&expr.upcast()) {
    let id = namify(&expr.upcast().to_string(vd.sys));
    if (**exposed_kind == ExposeKind::Output) || (**exposed_kind == ExposeKind::Inout) {
      format!("  assign {id}_exposed_o = {id};\n")
    } else {
      "".into()
    }
  } else {
    "".into()
  };

  let mut is_pop = None;

  let body = match expr.get_opcode() {
    Opcode::Binary { binop } => {
      let dtype = expr.dtype();
      let bin = expr.as_sub::<instructions::Binary>().unwrap();
      format!(
        "{} {} {}",
        dump_arith_ref(vd.sys, &bin.a()),
        if matches!(binop, subcode::Binary::Shr) {
          if dtype.is_signed() {
            ">>>".into()
          } else {
            ">>".into()
          }
        } else {
          binop.to_string()
        },
        dump_arith_ref(vd.sys, &bin.b())
      )
    }

    Opcode::Unary { ref uop } => {
      let dump = match uop {
        subcode::Unary::Flip => "~",
        subcode::Unary::Neg => "-",
      };
      let uop = expr.as_sub::<instructions::Unary>().unwrap();
      format!("{}{}", dump, dump_arith_ref(vd.sys, &uop.x()))
    }

    Opcode::Compare { .. } => {
      let cmp = expr.as_sub::<instructions::Compare>().unwrap();
      format!(
        "{} {} {}",
        dump_arith_ref(vd.sys, &cmp.a()),
        cmp.get_opcode(),
        dump_arith_ref(vd.sys, &cmp.b())
      )
    }

    Opcode::FIFOPop => {
      let pop = expr.as_sub::<instructions::FIFOPop>().unwrap();
      let fifo = pop.fifo();
      let display = utils::DisplayInstance::from_fifo(&fifo, false);
      is_pop = format!(
        "  assign {} = executed{};",
        display.field("pop_ready"),
        vd.get_pred()
          .map(|p| format!(" && {}", p))
          .unwrap_or("".to_string())
      )
      .into();
      display.field("pop_data")
    }

    Opcode::Log => {
      let mut res = String::new();

      res.push_str(&format!(
        "  always_ff @(posedge clk) if ({}{})",
        if vd.before_wait_until {
          "1'b1"
        } else {
          "executed"
        },
        vd.get_pred()
          .map(|p| format!(" && {}", p))
          .unwrap_or("".to_string())
      ));

      let args = expr
        .operand_iter()
        .map(|elem| *elem.get_value())
        .collect::<Vec<_>>();

      let format_str = utils::parse_format_string(args, expr.sys);

      res.push_str(&format!("$display(\"%t\\t[{}]\\t\\t", vd.current_module));
      res.push_str(&format_str);
      res.push_str(
        "\",
`ifndef SYNTHESIS
  $time - 200
`else
  $time
`endif
, ",
      );
      for elem in expr.operand_iter().skip(1) {
        res.push_str(&format!("{}, ", dump_ref(vd.sys, elem.get_value(), false)));
      }
      res.pop();
      res.pop();
      res.push_str(");\n");
      res.push('\n');
      res
    }

    Opcode::Load => {
      let load = expr.as_sub::<instructions::Load>().unwrap();
      let (array_ref, array_idx) = (load.array(), load.idx());
      let size = array_ref.get_size();
      let bits = array_ref.scalar_ty().get_bits() as u64;
      let name = format!("array_{}_q", namify(array_ref.get_name()));

      match load.idx().get_kind() {
        NodeKind::IntImm => {
          let imm = array_idx.as_ref::<IntImm>(vd.sys).unwrap().get_value();
          format!("{name}[{}:{}]", bits * (imm + 1) - 1, imm * bits)
        }
        NodeKind::Expr => {
          let mut res = "'x".into();
          let idx = dump_ref(load.get().sys, &array_idx, true);
          for i in 0..size {
            let slice = format!("{name}[{}:{}]", ((i + 1) as u64) * bits - 1, (i as u64) * bits);
            res = format!("{i} == {idx} ? {slice} : ({res})");
          }
          res
        }
        _ => {
          panic!("Unexpected reference type: {:?}", load.idx());
        }
      }
    }

    Opcode::Store => {
      let store = expr.as_sub::<instructions::Store>().unwrap();
      let (array_ref, array_idx) = (store.array(), store.idx());
      let array_name = namify(array_ref.get_name());
      let pred = vd.get_pred().unwrap_or("".to_string());
      let idx = dump_ref(store.get().sys, &array_idx, true);
      let idx_bits = store.idx().get_dtype(vd.sys).unwrap().get_bits();
      let value = dump_ref(store.get().sys, &store.value(), true);
      let value_bits = store.value().get_dtype(vd.sys).unwrap().get_bits();
      match vd.array_stores.get_mut(&array_name) {
        Some((g_idx, g_value)) => {
          g_idx.push(pred.clone(), idx, idx_bits);
          g_value.push(pred, value, value_bits);
        }
        None => {
          vd.array_stores.insert(
            array_name.clone(),
            (Gather::new(pred.clone(), idx, idx_bits), Gather::new(pred, value, value_bits)),
          );
        }
      }
      "".to_string()
    }

    Opcode::FIFOPush => {
      let push = expr.as_sub::<instructions::FIFOPush>().unwrap();
      let fifo = push.fifo();
      let fifo_name = format!("{}_{}", namify(fifo.get_module().get_name()), fifo_name(&fifo));
      let pred = vd.get_pred().unwrap_or("".to_string());
      let value = dump_ref(vd.sys, &push.value(), false);
      match vd.fifo_pushes.get_mut(&fifo_name) {
        Some(fps) => fps.push(pred, value, fifo.scalar_ty().get_bits()),
        None => {
          vd.fifo_pushes
            .insert(fifo_name.clone(), Gather::new(pred, value, fifo.scalar_ty().get_bits()));
        }
      }
      "".to_string()
    }

    Opcode::PureIntrinsic { intrinsic } => {
      let call = expr.as_sub::<instructions::PureIntrinsic>().unwrap();
      match intrinsic {
        subcode::PureIntrinsic::FIFOValid | subcode::PureIntrinsic::FIFOPeek => {
          let fifo = call
            .get()
            .get_operand_value(0)
            .unwrap()
            .as_ref::<FIFO>(vd.sys)
            .unwrap();
          let fifo_name = fifo_name(&fifo);
          match intrinsic {
            subcode::PureIntrinsic::FIFOValid => format!("fifo_{}_pop_valid", fifo_name),
            subcode::PureIntrinsic::FIFOPeek => format!("fifo_{}_pop_data", fifo_name),
            _ => unreachable!(),
          }
        }
        subcode::PureIntrinsic::ValueValid => {
          let value = call.get().get_operand_value(0).unwrap();
          let value = value.as_ref::<Expr>(vd.sys).unwrap();
          if value.get_block().get_module().get_key()
            != call.get().get_block().get_module().get_key()
          {
            format!("{}_valid", namify(&value.upcast().to_string(vd.sys)))
          } else {
            format!(
              "(executed{})",
              vd.get_pred()
                .map_or("".to_string(), |x| format!(" && {}", x))
            )
          }
        }
        _ => todo!(),
      }
    }

    Opcode::AsyncCall => {
      let call = expr.as_sub::<instructions::AsyncCall>().unwrap();
      let callee = {
        let bind = call.bind();
        bind.callee().get_name().to_string()
      };
      let callee = namify(&callee);
      let pred = vd.get_pred().unwrap_or("".to_string());
      // FIXME(@were): Do not hardcode the counter delta width.
      match vd.triggers.get_mut(&callee) {
        Some(trgs) => trgs.push(pred, "".into(), 8),
        None => {
          vd.triggers.insert(callee, Gather::new(pred, "".into(), 8));
        }
      }
      "".to_string()
    }

    Opcode::Slice => {
      let slice = expr.as_sub::<instructions::Slice>().unwrap();
      let a = dump_ref(vd.sys, &slice.x(), false);
      let l = dump_ref(vd.sys, &slice.l_intimm().upcast(), false);
      let r = dump_ref(vd.sys, &slice.r_intimm().upcast(), false);
      format!("{}[{}:{}]", a, r, l)
    }

    Opcode::Concat => {
      let concat = expr.as_sub::<instructions::Concat>().unwrap();
      let a = dump_ref(vd.sys, &concat.msb(), true);
      let b = dump_ref(vd.sys, &concat.lsb(), true);
      format!("{{{}, {}}}", a, b)
    }

    Opcode::Cast { .. } => {
      let dbits = expr.dtype().get_bits();
      let cast = expr.as_sub::<instructions::Cast>().unwrap();
      let a = dump_ref(vd.sys, &cast.x(), false);
      let src_dtype = cast.src_type();
      let pad = dbits - src_dtype.get_bits();
      match cast.get_opcode() {
        subcode::Cast::BitCast => a,
        subcode::Cast::ZExt => format!("{{{}'b0, {}}}", pad, a),
        subcode::Cast::SExt => {
          let dest_dtype = cast.dest_type();
          if src_dtype.is_int()
            && src_dtype.is_signed()
            && dest_dtype.is_int()
            && dest_dtype.is_signed()
            && dest_dtype.get_bits() > src_dtype.get_bits()
          {
            // perform sext
            format!("{{{}'{{{}[{}]}}, {}}}", pad, a, src_dtype.get_bits() - 1, a)
          } else {
            format!("{{{}'b0, {}}}", pad, a)
          }
        }
      }
    }

    Opcode::Select => {
      let select = expr.as_sub::<instructions::Select>().unwrap();
      let cond = dump_ref(vd.sys, &select.cond(), true);
      let true_value = dump_ref(vd.sys, &select.true_value(), true);
      let false_value = dump_ref(vd.sys, &select.false_value(), true);
      format!("{} ? {} : {}", cond, true_value, false_value)
    }

    Opcode::Bind => {
      // currently handled in AsyncCall
      "".to_string()
    }

    Opcode::Select1Hot => {
      let dbits = expr.dtype().get_bits();
      let select1hot = expr.as_sub::<instructions::Select1Hot>().unwrap();
      let cond = dump_ref(vd.sys, &select1hot.cond(), false);
      select1hot
        .value_iter()
        .enumerate()
        .map(|(i, elem)| {
          let value = dump_ref(vd.sys, &elem, false);
          format!("({{{}{{{}[{}] == 1'b1}}}} & {})", dbits, cond, i, value)
        })
        .collect::<Vec<_>>()
        .join(" | ")
    }

    Opcode::BlockIntrinsic { intrinsic } => match intrinsic {
      subcode::BlockIntrinsic::Finish => {
        let pred = vd.get_pred().unwrap_or("1".to_string());
        format!(
          "
`ifndef SYNTHESIS
  always_ff @(posedge clk) if (executed && {}) $finish();
`endif\n",
          pred
        )
      }
      subcode::BlockIntrinsic::Assert => {
        let assert = expr.as_sub::<instructions::BlockIntrinsic>().unwrap();
        let pred = vd.get_pred().unwrap_or("1".to_string());
        let cond = dump_ref(vd.sys, &assert.value().unwrap(), false);
        format!("  always_ff @(posedge clk) if (executed && {}) assert({});\n", pred, cond)
      }
      _ => panic!("Unknown block intrinsic: {:?}", intrinsic),
    },
  };

  let temp: String = body;

  let body: String = if let Some((id, _)) = decl.clone() {
    if let Some(kind) = exposed_map.get(&expr_clone.upcast()) {
      if (**kind == ExposeKind::Inout) || (**kind == ExposeKind::Input) {
        format!("{}_exposed_i_valid ? {}_exposed_i :({}) ", id, id, temp)
      } else {
        temp
      }
    } else {
      temp
    }
  } else {
    temp
  };

  let mut res: String = if let Some((id, ty)) = decl {
    format!(
      "{}  assign {} = {};\n{}\n{}\n",
      declare_logic(ty, &id),
      id,
      body,
      expose,
      expose_out_str
    )
  } else {
    body.clone()
  };

  if let Some(pop) = is_pop {
    res.push_str(&pop);
    res.push('\n');
  }

  res.into()
}
