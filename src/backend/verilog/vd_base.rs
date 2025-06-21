use std::{
  collections::{HashMap, HashSet, VecDeque},
  fs::File,
  io::Write,
};

use crate::{
  backend::{
    common::{namify, upstreams, Config},
    verilog::elaborate::dump_ref,
  },
  builder::{
    system::{ExposeKind, ModuleKind},
    SysBuilder,
  },
  ir::{
    expr::Metadata,
    instructions::FIFOPush,
    module::{self, attrs::MemoryParams},
    node::{ArrayRef, BaseNode, FIFORef, IsElement, ModuleRef, NodeKind},
    Array, Block, DataType, Expr, IntImm, Module, Opcode, Operand, Typed, FIFO,
  },
};

use super::{
  elaborate::{fifo_name, VerilogDumper},
  gather::ExternalUsage,
  utils::{
    self, bool_ty, connect_top, declare_array, declare_in, declare_logic, reduce, select_1h, Edge,
    Field,
  },
  Simulator,
};

impl<'a, 'b> VerilogDumper<'a, 'b> {
  pub(super) fn new(
    sys: &'a SysBuilder,
    config: &'b Config,
    external_usage: ExternalUsage,
    topo: HashMap<BaseNode, usize>,
    array_memory_params_map: HashMap<BaseNode, MemoryParams>,
    module_expr_map: HashMap<BaseNode, HashMap<BaseNode, ExposeKind>>,
  ) -> Self {
    Self {
      sys,
      config,
      pred_stack: VecDeque::new(),
      fifo_pushes: HashMap::new(),
      array_stores: HashMap::new(),
      triggers: HashMap::new(),
      current_module: String::new(),
      external_usage,
      before_wait_until: false,
      topo,
      array_memory_params_map,
      module_expr_map,
    }
  }

  pub(super) fn collect_array_memory_params_map(
    sys: &SysBuilder,
  ) -> HashMap<BaseNode, MemoryParams> {
    let mut map = HashMap::new();

    for module in sys.module_iter(ModuleKind::Downstream) {
      for attr in module.get_attrs() {
        if let module::Attribute::MemoryParams(mem) = attr {
          if module.is_downstream() {
            for (interf, _) in module.ext_interf_iter() {
              if interf.get_kind() == NodeKind::Array {
                let array_ref = interf.as_ref::<Array>(sys).unwrap();
                map.insert(array_ref.upcast(), mem.clone());
              }
            }
          }
        }
      }
    }

    map
  }

  pub(super) fn dump_memory_nodes(&mut self, node: BaseNode, res: &mut String) {
    match node.get_kind() {
      NodeKind::Expr => {
        let expr = node.as_ref::<Expr>(self.sys).unwrap();
        if expr.get_opcode() == Opcode::Load {
          let id = namify(&expr.upcast().to_string(self.sys));
          let ty = expr.dtype();
          res.push_str(&declare_logic(ty, &id));
          res.push_str(&format!("  assign {} = dataout;\n", id));
        } else {
          res.push_str(&self.print_body(node));
        }
      }
      NodeKind::Block => {
        let block = node.as_ref::<Block>(self.sys).unwrap();
        let skip = if let Some(cond) = block.get_condition() {
          self
            .pred_stack
            .push_back(if cond.get_dtype(block.sys).unwrap().get_bits() == 1 {
              dump_ref(self.sys, &cond, true)
            } else {
              format!("(|{})", dump_ref(self.sys, &cond, false))
            });
          1
        } else if let Some(cycle) = block.get_cycle() {
          self
            .pred_stack
            .push_back(format!("(cycle_cnt == {})", cycle));
          1
        } else {
          0
        };
        for elem in block.body_iter().skip(skip) {
          self.dump_memory_nodes(elem, res);
        }
        self.pred_stack.pop_back();
      }
      _ => {
        panic!("Unexpected node kind: {:?}", node.get_kind());
      }
    }
  }

  pub(super) fn get_pred(&self) -> Option<String> {
    if self.pred_stack.is_empty() {
      None
    } else {
      Some(format!(
        "({})",
        self
          .pred_stack
          .iter()
          .map(|s| s.as_str())
          .collect::<Vec<_>>()
          .join(" && ")
      ))
    }
  }

  fn dump_array(&self, array: &ArrayRef, mem_init_path: Option<&String>) -> String {
    let mut res = String::new();
    let display = utils::DisplayInstance::from_array(array);
    // write enable
    let w = display.field("w");
    // write index
    let widx = display.field("widx");
    // write data
    let d = display.field("d");
    // array buffer
    let q = display.field("q");

    res.push_str(&format!("  /* {} */\n", array));
    let map = &self.array_memory_params_map;

    if map.get(&array.upcast()).is_some() {
      res.push_str(&declare_logic(array.scalar_ty(), &q));
    } else {
      res.push_str(&declare_array("", array, &q, ";"));
    }

    let mut seen = HashSet::new();
    let drivers = array
      .users()
      .iter()
      .filter_map(move |x| {
        let expr = x.as_ref::<Operand>(array.sys).unwrap().get_expr();
        if matches!(expr.get_opcode(), Opcode::Store) {
          Some(expr.get_block().get_module())
        } else {
          None
        }
      })
      .filter(|x| seen.insert(x.get_key()))
      .map(|x| Edge::new(display.clone(), &x.as_ref::<Module>(array.sys).unwrap()))
      .collect::<Vec<_>>();

    let scalar_bits = array.scalar_ty().get_bits();
    let array_size = array.get_size();

    drivers.iter().for_each(|edge| {
      res.push_str(&declare_logic(array.scalar_ty(), &edge.field("d")));
      res.push_str(&declare_logic(DataType::int_ty(1), &edge.field("w")));
      res.push_str(&declare_logic(array.get_idx_type(), &edge.field("widx")));
    });

    if map.get(&array.upcast()).is_some() {
    } else {
      // if w: array[widx] = d;
      // where w is the gathered write enable signal
      // widx/d are 1-hot selected from all the writers
      res.push_str(&declare_logic(array.scalar_ty(), &d));
      res.push_str(&declare_logic(array.get_idx_type(), &widx));
      res.push_str(&declare_logic(DataType::int_ty(1), &w));

      let write_data = select_1h(
        drivers
          .iter()
          .map(|edge| (edge.field("w"), edge.field("d"))),
        scalar_bits,
      );
      res.push_str(&format!("  assign {d} = {};\n", write_data));

      let write_idx = select_1h(
        drivers
          .iter()
          .map(|edge| (edge.field("w"), edge.field("widx"))),
        array.get_idx_type().get_bits(),
      );
      res.push_str(&format!("  assign {widx} = {};\n", write_idx));

      let write_enable = reduce(drivers.iter().map(|edge| edge.field("w")), " | ");
      res.push_str(&format!("  assign {w} = {};\n", write_enable));

      res.push_str("  always_ff @(posedge clk or negedge rst_n)\n");
      // Dump the initializer
      res.push_str("    if (!rst_n)\n");
      if mem_init_path.is_some() {
        // Read from memory initialization file
        res.push_str(&format!("      $readmemh(\"{}\", {q});\n", mem_init_path.unwrap()));
      } else if let Some(initializer) = array.get_initializer() {
        // Read from the hardcoded initializer
        res.push_str("    begin\n");
        for (idx, value) in initializer.iter().enumerate() {
          let elem_init = value.as_ref::<IntImm>(self.sys).unwrap().get_value();
          let slice = format!("{}:{}", (idx + 1) * scalar_bits - 1, idx * scalar_bits);
          res.push_str(&format!("      {q}[{slice}] <= {scalar_bits}'d{elem_init};\n",));
        }
        res.push_str("    end\n");
      } else {
        let init_bits = array.get_flattened_size();
        // Initialize to 0
        res.push_str(&format!("      {q} <= {init_bits}'d0;\n",));
      }
      // Dump the array write
      res.push_str(&format!("    else if ({w}) begin\n\n",));
      res.push_str(&format!("      case ({widx})\n"));
      for i in 0..array_size {
        let slice = format!("{}:{}", (i + 1) * scalar_bits - 1, i * scalar_bits);
        res.push_str(&format!("        {i} : {q}[{slice}] <= {d};\n"));
      }
      res.push_str("        default: ;\n");
      res.push_str("      endcase\n");
      res.push_str("    end\n");
    }

    res
  }

  fn dump_exposed_array(
    &self,
    array: &ArrayRef,
    exposed_kind: &ExposeKind,
    mem_init_path: Option<&String>,
  ) -> String {
    let mut res = String::new();
    let display = utils::DisplayInstance::from_array(array);
    // write enable
    let w = display.field("w");
    // write index
    let widx = display.field("widx");
    // write data
    let d = display.field("d");
    // array buffer
    let q = display.field("q");

    let temp = display.field("temp");
    let i = display.field("exposed_i");
    let i_valid = display.field("exposed_i_valid");

    res.push_str(&format!("  /* {} */\n", array));
    let map = &self.array_memory_params_map;

    if map.get(&array.upcast()).is_some() {
      res.push_str(&declare_logic(array.scalar_ty(), &q));
    } else {
      res.push_str(&declare_array("", array, &q, ";"));
    }

    let mut seen = HashSet::new();
    let drivers = array
      .users()
      .iter()
      .filter_map(move |x| {
        let expr = x.as_ref::<Operand>(array.sys).unwrap().get_expr();
        if matches!(expr.get_opcode(), Opcode::Store) {
          Some(expr.get_block().get_module())
        } else {
          None
        }
      })
      .filter(|x| seen.insert(x.get_key()))
      .map(|x| Edge::new(display.clone(), &x.as_ref::<Module>(array.sys).unwrap()))
      .collect::<Vec<_>>();

    let scalar_bits = array.scalar_ty().get_bits();
    let array_size = array.get_size();

    drivers.iter().for_each(|edge| {
      res.push_str(&declare_logic(array.scalar_ty(), &edge.field("d")));
      res.push_str(&declare_logic(DataType::int_ty(1), &edge.field("w")));
      res.push_str(&declare_logic(array.get_idx_type(), &edge.field("widx")));
    });

    if (*exposed_kind == ExposeKind::Output) || (*exposed_kind == ExposeKind::Inout) {
      let o = display.field("exposed_o");
      res.push_str(&format!("  assign {o} = {q};\n"));
    }
    if (*exposed_kind == ExposeKind::Input) || (*exposed_kind == ExposeKind::Inout) {
      res.push_str(&declare_logic(array.scalar_ty(), &temp));
      res.push_str(&format!("  assign {temp} = {i_valid}?{i}:{d};\n"));
    }

    if map.get(&array.upcast()).is_some() {
    } else {
      // if w: array[widx] = d;
      // where w is the gathered write enable signal
      // widx/d are 1-hot selected from all the writers
      res.push_str(&declare_logic(array.scalar_ty(), &d));
      res.push_str(&declare_logic(array.get_idx_type(), &widx));
      res.push_str(&declare_logic(DataType::int_ty(1), &w));

      let write_data = select_1h(
        drivers
          .iter()
          .map(|edge| (edge.field("w"), edge.field("d"))),
        scalar_bits,
      );
      res.push_str(&format!("  assign {d} = {};\n", write_data));

      let write_idx = select_1h(
        drivers
          .iter()
          .map(|edge| (edge.field("w"), edge.field("widx"))),
        array.get_idx_type().get_bits(),
      );
      res.push_str(&format!("  assign {widx} = {};\n", write_idx));

      let write_enable = reduce(drivers.iter().map(|edge| edge.field("w")), " | ");
      res.push_str(&format!("  assign {w} = {};\n", write_enable));

      res.push_str("  always_ff @(posedge clk or negedge rst_n)\n");
      // Dump the initializer
      res.push_str("    if (!rst_n)\n");
      if mem_init_path.is_some() {
        // Read from memory initialization file
        res.push_str(&format!("      $readmemh(\"{}\", {q});\n", mem_init_path.unwrap()));
      } else if let Some(initializer) = array.get_initializer() {
        // Read from the hardcoded initializer
        res.push_str("    begin\n");
        for (idx, value) in initializer.iter().enumerate() {
          let elem_init = value.as_ref::<IntImm>(self.sys).unwrap().get_value();
          let slice = format!("{}:{}", (idx + 1) * scalar_bits - 1, idx * scalar_bits);
          res.push_str(&format!("      {q}[{slice}] <= {scalar_bits}'d{elem_init};\n",));
        }
        res.push_str("    end\n");
      } else {
        let init_bits = array.get_flattened_size();
        // Initialize to 0
        res.push_str(&format!("      {q} <= {init_bits}'d0;\n",));
      }
      // Dump the array write
      res.push_str(&format!("    else if ({w}) begin\n\n",));
      res.push_str(&format!("      case ({widx})\n"));
      if (*exposed_kind == ExposeKind::Input) || (*exposed_kind == ExposeKind::Inout) {
        for i in 0..array_size {
          let slice = format!("{}:{}", (i + 1) * scalar_bits - 1, i * scalar_bits);
          res.push_str(&format!("        {i} : {q}[{slice}] <= {temp};\n"));
        }
      } else {
        for i in 0..array_size {
          let slice = format!("{}:{}", (i + 1) * scalar_bits - 1, i * scalar_bits);
          res.push_str(&format!("        {i} : {q}[{slice}] <= {d};\n"));
        }
      }

      res.push_str("        default: ;\n");
      res.push_str("      endcase\n");
      res.push_str("    end\n");
    }

    res
  }

  fn dump_fifo(&self, fifo: &FIFORef) -> String {
    let mut res = String::new();
    let display = utils::DisplayInstance::from_fifo(fifo, true);
    let fifo_name = namify(&format!("{}_{}", fifo.get_module().get_name(), fifo_name(fifo)));
    let fifo_width = fifo.scalar_ty().get_bits();
    let fifo_depth = fifo
      .users()
      .iter()
      .find_map(|node| {
        node
          .as_ref::<Operand>(self.sys)
          .ok()
          .and_then(|op| op.get_user().as_expr::<FIFOPush>(self.sys).ok())
      })
      .and_then(|push| {
        push.get().metadata_iter().next().map(|m| {
          let Metadata::FIFODepth(depth) = m;
          *depth
        })
      })
      .unwrap_or(self.config.fifo_depth)
      .next_power_of_two();

    res.push_str(&format!("  // fifo: {}, depth: {}\n", fifo, fifo_depth));

    let push_valid = display.field("push_valid"); // If external pushers have data to push
    let push_data = display.field("push_data"); // Data to be pushed
    let pop_ready = display.field("pop_ready"); // If the FIFO pops data
    let push_ready = display.field("push_ready"); // If the FIFO is ready to accept data
    let pop_valid = display.field("pop_valid"); // If the popped data is valid
    let pop_data = display.field("pop_data"); // Popped data

    let edges = fifo
      .users()
      .iter()
      .filter_map(|x| {
        x.as_ref::<Operand>(self.sys)
          .unwrap()
          .get_user()
          .as_expr::<FIFOPush>(self.sys)
          .ok()
          .map(|y| y.get().get_block().get_module())
      })
      .collect::<HashSet<_>>()
      .iter()
      .map(|x| Edge::new(display.clone(), &x.as_ref::<Module>(self.sys).unwrap()))
      .collect::<Vec<_>>();

    res.push_str("  // Declare the pop.{data/valid/ready}\n");
    res.push_str(&declare_logic(fifo.scalar_ty(), &pop_data));
    res.push_str(&declare_logic(bool_ty(), &pop_valid));
    res.push_str(&declare_logic(bool_ty(), &pop_ready));

    edges.iter().for_each(|edge| {
      res.push_str(&declare_logic(fifo.scalar_ty(), &edge.field("push_data")));
      res.push_str(&declare_logic(bool_ty(), &edge.field("push_valid")));
      res.push_str(&declare_logic(bool_ty(), &edge.field("push_ready")));
    });

    res.push_str("  // Broadcast the push_ready signal to all the pushers\n");
    res.push_str(&format!("  logic {push_ready};\n"));
    edges
      .iter()
      .for_each(|x| res.push_str(&format!("  assign {} = {push_ready};", x.field("push_ready"))));

    res.push_str("  // Gather all the push signal\n");
    let valid = reduce(edges.iter().map(|x| x.field("push_valid")), " | ");
    res.push_str(&declare_logic(DataType::int_ty(1), &push_valid));
    res.push_str(&format!("  assign {} = {};\n", push_valid, valid));

    res.push_str("  // 1-hot select the push data\n");
    let data = select_1h(
      edges
        .iter()
        .map(|x| (x.field("push_valid"), x.field("push_data"))),
      fifo_width,
    );
    res.push_str(&declare_logic(fifo.scalar_ty(), &push_data));
    res.push_str(&format!("  assign {push_data} = {data};\n"));

    let log2_depth = fifo_depth.trailing_zeros();
    // Instantiate the FIFO
    res.push_str(&format!(
      "
  fifo #({fifo_width}, {log2_depth}) fifo_{fifo_name}_i (
    .clk(clk),
    .rst_n(rst_n),
    .push_valid({push_valid}),
    .push_data({push_data}),
    .push_ready({push_ready}),
    .pop_valid({pop_valid}),
    .pop_data({pop_data}),
    .pop_ready({pop_ready}));\n\n"
    ));

    res
  }

  /// Dump the trigger event state machine's instantiation.
  fn dump_trigger(&self, module: &ModuleRef) -> String {
    let mut res = String::new();
    let module_name = namify(module.get_name());
    let display = utils::DisplayInstance::from_module(module);
    res.push_str(&format!("  // Trigger SM of Module: {}\n", module.get_name()));
    let delta_value = display.field("counter_delta");
    let pop_ready = display.field("counter_pop_ready");
    let pop_valid = display.field("counter_pop_valid");
    let delta_ready = display.field("counter_delta_ready");

    let callers = module
      .callers()
      .map(|x| Edge::new(display.clone(), &x))
      .collect::<Vec<_>>();

    if module_name != "driver" && module_name != "testbench" {
      callers.iter().for_each(|edge| {
        res.push_str(&declare_logic(
          DataType::int_ty(8 /*FIXME(@were): Do not hardcode*/),
          &edge.field("counter_delta"),
        ));
        res.push_str(&declare_logic(bool_ty(), &edge.field("counter_delta_ready")));
      });
    }
    res.push_str(&declare_logic(bool_ty(), &delta_ready));
    res.push_str(&declare_logic(
      DataType::int_ty(8 /*FIXME(@were): Do not hardcode*/),
      &delta_value,
    ));

    res.push_str("  // Gather all the push signal\n");
    if module_name != "driver" && module_name != "testbench" {
      res.push_str(&format!(
        "  assign {delta_value} = {};\n",
        reduce(callers.iter().map(|x| x.field("counter_delta")), " + ")
      ));
    }
    res.push_str("  // Broadcast the push_ready signal to all the pushers\n");
    res.push_str(&declare_logic(bool_ty(), &pop_ready));
    if module_name != "driver" && module_name != "testbench" {
      callers.iter().for_each(|x| {
        res.push_str(&format!("  assign {} = {};\n", x.field("counter_delta_ready"), pop_ready));
      });
    }
    res.push_str(&declare_logic(bool_ty(), &pop_valid));
    res.push_str(&format!(
      "
  trigger_counter #(8) {}_trigger_i (
    .clk(clk),
    .rst_n(rst_n),
    .delta({delta_value}),
    .delta_ready({delta_ready}),
    .pop_valid({pop_valid}),
    .pop_ready({pop_ready}));\n",
      module_name
    ));
    res
  }

  fn dump_module_instance(&self, module: &ModuleRef) -> String {
    let mut res = String::new();
    let module_name = namify(module.get_name());

    if let Some(out_bounds) = self.external_usage.out_bounds(module) {
      for elem in out_bounds {
        let id = namify(&elem.to_string(module.sys));
        let ty = elem.get_dtype(module.sys).unwrap();
        res.push_str(&declare_logic(ty, &format!("logic_{}", id)));
        res.push_str(&declare_logic(bool_ty(), &format!("logic_{}_valid", id)));
      }
    }
    let mut is_memory_instance = false;
    res.push_str(&declare_logic(bool_ty(), &format!("{}_executed", module_name)));

    res.push_str(&format!(
      "
  // {module}
  {module} {module}_i (
    .clk(clk),
    .rst_n(rst_n),
",
      module = module_name
    ));
    for port in module.fifo_iter() {
      let local = utils::DisplayInstance::from_fifo(&port, false);
      let global = utils::DisplayInstance::from_fifo(&port, true);
      res.push_str(&connect_top(&local, &global, &["pop_ready", "pop_data", "pop_valid"]));
    }
    for (interf, ops) in module.ext_interf_iter() {
      match interf.get_kind() {
        NodeKind::FIFO => {
          let fifo = interf.as_ref::<FIFO>(self.sys).unwrap();
          let fifo = utils::DisplayInstance::from_fifo(&fifo, true);
          let edge = Edge::new(fifo.clone(), module);
          res.push_str(&connect_top(&fifo, &edge, &["push_valid", "push_data", "push_ready"]));
        }
        NodeKind::Array => {
          let array_ref = interf.as_ref::<Array>(self.sys).unwrap();
          let display = utils::DisplayInstance::from_array(&array_ref);
          let edge = Edge::new(display.clone(), module);

          for attr in module.get_attrs() {
            if let module::Attribute::MemoryParams(_) = attr {
              if module.is_downstream() {
                is_memory_instance = true;
              }
            }
          }
          if is_memory_instance {
          } else {
            if self.sys.user_contains_opcode(ops, Opcode::Load) {
              res.push_str(&format!("    .{q}({q}),\n", q = display.field("q"),));
            }
            if self.sys.user_contains_opcode(ops, Opcode::Store) {
              res.push_str(&connect_top(&display, &edge, &["w", "widx", "d"]));
            }
          }
        }
        NodeKind::Module => {
          let interf = interf.as_ref::<Module>(self.sys).unwrap();
          let display = utils::DisplayInstance::from_module(&interf);
          let edge = Edge::new(display.clone(), module);
          res.push_str(&connect_top(&display, &edge, &["counter_delta_ready", "counter_delta"]));
        }
        NodeKind::Expr => {
          // This is handled below, since we need a deduplication for the modules to which these
          // expressions belong.
        }
        _ => panic!("Unknown interf kind {:?}", interf.get_kind()),
      }
    }

    if module.is_downstream() {
      res.push_str("    // Upstream executed signals\n");
      upstreams(module, &self.topo).iter().for_each(|x| {
        let name = namify(x.as_ref::<Module>(module.sys).unwrap().get_name());
        res.push_str(&format!("    .{}_executed({}_executed),\n", name, name));
      });
    }

    if let Some(out_bounds) = self.external_usage.out_bounds(module) {
      for elem in out_bounds {
        let id = namify(&elem.to_string(module.sys));
        // Put a "external_" prefix to avoid name collision with the original combinational logic.
        res.push_str(&format!("    .expose_{id}(logic_{id}),\n"));
        res.push_str(&format!("    .expose_{id}_valid(logic_{id}_valid),\n"));
      }
    }

    if let Some(in_bounds) = self.external_usage.in_bounds(module) {
      for elem in in_bounds {
        let id = namify(&elem.to_string(module.sys));
        // Use the original name to avoid re-naming the external signals.
        res.push_str(&format!("    .{id}(logic_{id}),\n"));
        res.push_str(&format!("    .{id}_valid(logic_{id}_valid),\n"));
      }
    }

    if let Some(exposed_map) = self.module_expr_map.get(&module.upcast()) {
      for (exposed_node, kind) in exposed_map {
        if exposed_node.get_kind() == NodeKind::Expr {
          let expr = exposed_node.as_ref::<Expr>(self.sys).unwrap();
          let id = namify(&expr.upcast().to_string(self.sys));
          if (*kind == ExposeKind::Output) || (*kind == ExposeKind::Inout) {
            res.push_str(&format!("    .{a}_exposed_o({a}_exposed_o),\n", a = id));
          }
          if (*kind == ExposeKind::Input) || (*kind == ExposeKind::Inout) {
            res.push_str(&format!("    .{a}_exposed_i({a}_exposed_i),\n", a = id));
            res.push_str(&format!("    .{a}_exposed_i_valid({a}_exposed_i_valid),\n", a = id));
          }
        }
      }
    }

    if !module.is_downstream() {
      let display = utils::DisplayInstance::from_module(module);
      res.push_str(&format!(
        "    .counter_delta_ready({}),\n",
        display.field("counter_delta_ready")
      ));
      res.push_str(&format!("    .counter_pop_ready({}),\n", display.field("counter_pop_ready")));
      res.push_str(&format!("    .counter_pop_valid({}),\n", display.field("counter_pop_valid")));
    }
    res.push_str(&format!("    .expose_executed({}_executed));\n", module_name));
    res
  }

  pub(super) fn dump_runtime(
    self: VerilogDumper<'a, 'b>,
    mut fd: File,
    sim_threshold: usize,
  ) -> Result<(), std::io::Error> {
    // runtime
    let mut res = String::new();

    res.push_str("module top(\n");

    for (exposed_node, kind) in self.sys.exposed_nodes() {
      if exposed_node.get_kind() == NodeKind::Array {
        let exposed_nodes_ref = exposed_node.as_ref::<Array>(self.sys).unwrap();
        let display = utils::DisplayInstance::from_array(&exposed_nodes_ref);
        if (*kind == ExposeKind::Output) || (*kind == ExposeKind::Inout) {
          let o = display.field("exposed_o");
          res.push_str(&declare_array("output", &exposed_nodes_ref, &o, ","));
        }
        if (*kind == ExposeKind::Input) || (*kind == ExposeKind::Inout) {
          res.push_str(&declare_in(exposed_nodes_ref.scalar_ty(), &display.field("exposed_i")));
          res.push_str(&declare_in(bool_ty(), &display.field("exposed_i_valid")));
        }
      }
      if exposed_node.get_kind() == NodeKind::Expr {
        let expr = exposed_node.as_ref::<Expr>(self.sys).unwrap();
        let id = namify(&expr.upcast().to_string(self.sys));
        let dtype = exposed_node.get_dtype(self.sys).unwrap();
        let bits = dtype.get_bits() - 1;
        if (*kind == ExposeKind::Output) || (*kind == ExposeKind::Inout) {
          res.push_str(&format!("  output logic [{bits}:0] {a}_exposed_o,\n", bits = bits, a = id));
        }
        if (*kind == ExposeKind::Input) || (*kind == ExposeKind::Inout) {
          res.push_str(&format!("  input logic [{bits}:0] {a}_exposed_i,\n", bits = bits, a = id));
          res.push_str(&format!("  input logic {a}_exposed_i_valid,\n", a = id));
        }
      }
    }

    res.push_str(
      "
  input logic clk,
  input logic rst_n
);\n\n",
    );

    // memory initializations mapS
    // FIXME(@were): Fix the memory initialization.
    let mut mem_init_map: HashMap<BaseNode, String> = HashMap::new();

    // array -> init_file_path
    for m in self.sys.module_iter(ModuleKind::Downstream) {
      for attr in m.get_attrs() {
        if let module::Attribute::MemoryParams(mp) = attr {
          if let Some(init_file) = &mp.init_file {
            let mut init_file_path = self.config.resource_base.clone();
            init_file_path.push(init_file);
            let init_file_path = init_file_path.to_str().unwrap();
            mem_init_map.insert(mp.pins.array, init_file_path.to_string());
          }
        }
      }
    }
    for (key, value) in &mem_init_map {
      res.push_str(&format!("//Array: {}, Init File Path: {}\n", key.to_string(self.sys), value));
    }

    let exposed_map: HashMap<_, _> = self.sys.exposed_nodes().collect();

    // array storage element definitions
    for array in self.sys.array_iter() {
      if let Some(kind) = exposed_map.get(&array.upcast()) {
        res.push_str(&self.dump_exposed_array(&array, kind, mem_init_map.get(&array.upcast())));
      } else {
        res.push_str(&self.dump_array(&array, mem_init_map.get(&array.upcast())));
      }
    }

    // fifo storage element definitions
    for module in self.sys.module_iter(ModuleKind::Module) {
      for fifo in module.fifo_iter() {
        res.push_str(&self.dump_fifo(&fifo));
      }
    }

    // trigger fifo definitions
    for module in self.sys.module_iter(ModuleKind::Module) {
      res.push_str(&self.dump_trigger(&module));
    }

    // FIXME(@were): Do not hardcode the counter delta width.
    if self.sys.has_testbench() {
      res.push_str("  assign testbench_counter_delta = 8'b1;\n\n");
    }
    if self.sys.has_driver() {
      res.push_str("  assign driver_counter_delta = 8'b1;\n\n");
    }

    // module instances
    for module in self.sys.module_iter(ModuleKind::Module) {
      res.push_str(&self.dump_module_instance(&module));
    }

    // downstream instances
    for module in self.sys.module_iter(ModuleKind::Downstream) {
      res.push_str(&self.dump_module_instance(&module));
    }

    res.push_str("endmodule // top\n\n");

    fd.write_all(res.as_bytes()).unwrap();

    let init = match self.config.verilog {
      Simulator::VCS => {
        "
initial begin
  $fsdbDumpfile(\"wave.fsdb\");
  $fsdbDumpvars();
  $fsdbDumpMDA();
end"
      }
      Simulator::Verilator => "",
      Simulator::None => panic!("No simulator specified"),
    };

    fd.write_all(include_str!("./runtime.sv").as_bytes())
      .unwrap();

    let threashold = (sim_threshold + 1) * 100;
    fd.write_all(
      "
module tb;

logic clk;
logic rst_n;
"
      .to_string()
      .as_bytes(),
    )?;

    for (exposed_node, kind) in self.sys.exposed_nodes() {
      if exposed_node.get_kind() == NodeKind::Array {
        let array_ref = exposed_node.as_ref::<Array>(self.sys).unwrap();
        let display = utils::DisplayInstance::from_array(&array_ref);
        let bits = array_ref.scalar_ty().get_bits();
        let bits_1 = bits - 1;
        let flatten_bits_1 = array_ref.get_flattened_size() - 1;
        if (*kind == ExposeKind::Output) || (*kind == ExposeKind::Inout) {
          let o = display.field("exposed_o");
          fd.write_all(format!("logic [{flatten_bits_1}:0]{o};\n",).as_bytes())?;
        }
        if (*kind == ExposeKind::Input) || (*kind == ExposeKind::Inout) {
          let i = display.field("exposed_i");
          let i_valid = display.field("exposed_i_valid");
          fd.write_all(format!("logic [{bits_1}:0]{i};\n",).as_bytes())?;
          fd.write_all(format!("logic {i_valid};\n",).as_bytes())?;
          fd.write_all(format!("\nassign {i_valid} = 1'd0;\n",).as_bytes())?;
          fd.write_all(format!("assign {i} = {bits}'d0;\n",).as_bytes())?;
        }
      }
      if exposed_node.get_kind() == NodeKind::Expr {
        let expr = exposed_node.as_ref::<Expr>(self.sys).unwrap();
        let id = namify(&expr.upcast().to_string(self.sys));
        let dtype = exposed_node.get_dtype(self.sys).unwrap();
        let bits = dtype.get_bits();
        let bits_1 = bits - 1;
        if (*kind == ExposeKind::Output) || (*kind == ExposeKind::Inout) {
          fd.write_all(format!("logic [{bits_1}:0] {id}_exposed_o;\n",).as_bytes())?;
        }
        if (*kind == ExposeKind::Input) || (*kind == ExposeKind::Inout) {
          fd.write_all(format!("logic [{bits_1}:0] {id}_exposed_i;\n",).as_bytes())?;
          fd.write_all(format!("logic {id}_exposed_i_valid;\n",).as_bytes())?;
          fd.write_all(format!("\nassign {id}_exposed_i_valid = 1'd0;\n",).as_bytes())?;
          fd.write_all(format!("assign {id}_exposed_i = {bits}'d0;\n",).as_bytes())?;
        }
      }
    }

    fd.write_all(
      format!(
        "
initial begin
  clk = 1'b1;
  rst_n = 1'b0;
  #150;
  rst_n = 1'b1;
  #{threashold};
  `ifndef SYNTHESIS
  $finish();
  `endif
end

always #50 clk <= !clk;

{init}

top top_i (
  .clk(clk),
  .rst_n(rst_n)"
      )
      .as_bytes(),
    )?;

    for (exposed_node, kind) in self.sys.exposed_nodes() {
      if exposed_node.get_kind() == NodeKind::Array {
        let exposed_nodes_ref = exposed_node.as_ref::<Array>(self.sys).unwrap();
        let display = utils::DisplayInstance::from_array(&exposed_nodes_ref);
        if (*kind == ExposeKind::Output) || (*kind == ExposeKind::Inout) {
          let o = display.field("exposed_o");
          fd.write_all(format!(",\n  .{o}({o})").as_bytes())?;
        }
        if (*kind == ExposeKind::Input) || (*kind == ExposeKind::Inout) {
          let i = display.field("exposed_i");
          let i_valid = display.field("exposed_i_valid");
          fd.write_all(format!(",\n  .{i}({i}),\n  .{i_valid}({i_valid})",).as_bytes())?;
        }
      }
      if exposed_node.get_kind() == NodeKind::Expr {
        let expr = exposed_node.as_ref::<Expr>(self.sys).unwrap();
        let id = namify(&expr.upcast().to_string(self.sys));
        if (*kind == ExposeKind::Output) || (*kind == ExposeKind::Inout) {
          fd.write_all(format!(",\n  .{id}_exposed_o({id}_exposed_o)",).as_bytes())?;
        }
        if (*kind == ExposeKind::Input) || (*kind == ExposeKind::Inout) {
          fd.write_all(format!(",\n  .{id}_exposed_i({id}_exposed_i)",).as_bytes())?;
          fd.write_all(format!(",\n  .{id}_exposed_i_valid({id}_exposed_i_valid)",).as_bytes())?;
        }
      }
    }

    fd.write_all(
      "
);

endmodule
"
      .to_string()
      .as_bytes(),
    )?;

    Ok(())
  }
}
