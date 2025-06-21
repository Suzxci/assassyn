use crate::ir::module::attrs::MemoryParams;
use crate::ir::module::attrs::MemoryPins;

impl VerilogDumper<'_, '_> {
  fn print_body(&mut self, node: BaseNode) -> String {
    match node.get_kind() {
      NodeKind::Expr => {
        let expr = node.as_ref::<Expr>(self.sys).unwrap();
        self.visit_expr(expr).unwrap()
      }
      NodeKind::Block => {
        let block = node.as_ref::<Block>(self.sys).unwrap();
        self.visit_block(block).unwrap()
      }
      _ => {
        panic!("Unexpected reference type: {:?}", node);
      }
    }
  }
}

impl Visitor<String> for VerilogDumper<'_, '_> {
  // Dump the implentation of each module.
  fn visit_module(&mut self, module: ModuleRef<'_>) -> Option<String> {
    self.current_module = namify(module.get_name()).to_string();

    let mut res = String::new();

    res.push_str(&format!(
      "
module {} (
  input logic clk,
  input logic rst_n,
",
      self.current_module
    ));

    for port in module.fifo_iter() {
      let name = fifo_name(&port);
      let ty = port.scalar_ty();
      let display = utils::DisplayInstance::from_fifo(&port, false);
      // (pop_valid, pop_data): something like `let front : Optional<T> = FIFO.pop();`.
      // `pop_ready: when enabled, it is something like fifo.pop()
      res.push_str(&format!("  // Port FIFO {name}\n", name = name));
      res.push_str(&declare_in(bool_ty(), &display.field("pop_valid")));
      res.push_str(&declare_in(ty, &display.field("pop_data")));
      res.push_str(&declare_out(bool_ty(), &display.field("pop_ready")));
    }

    let mut has_memory_params = false;
    let mut has_memory_init_path = false;
    let empty_pins = MemoryPins::new(
      BaseNode::unknown(), // array
      BaseNode::unknown(), // re
      BaseNode::unknown(), // we
      BaseNode::unknown(), // addr
      BaseNode::unknown(), // wdata
    );

    let mut memory_params = MemoryParams::new(
      0,          // width
      0,          // depth
      0..=0,      // lat
      None,       // init_file
      empty_pins, // 假设 `MemoryPins` 有一个 `new` 方法
    );
    let mut init_file_path = self.config.resource_base.clone();

    for (interf, ops) in module.ext_interf_iter() {
      match interf.get_kind() {
        NodeKind::FIFO => {
          let fifo = interf.as_ref::<FIFO>(self.sys).unwrap();
          let parent_name = fifo.get_module().get_name().to_string();
          let display = utils::DisplayInstance::from_fifo(&fifo, true);
          // TODO(@were): Support `push_ready` for backpressures.
          // (push_valid, push_data, push_ready) works like
          // `if push_valid && push_ready: FIFO.push()`
          res.push_str(&format!("  // External FIFO {}.{}\n", parent_name, fifo.get_name()));
          res.push_str(&declare_out(bool_ty(), &display.field("push_valid")));
          res.push_str(&declare_out(fifo.scalar_ty(), &display.field("push_data")));
          res.push_str(&declare_in(bool_ty(), &display.field("push_ready")));
        }
        NodeKind::Array => {
          let array = interf.as_ref::<Array>(self.sys).unwrap();
          let display = utils::DisplayInstance::from_array(&array);
          res.push_str(&format!("  /* {} */\n", array));

          for attr in module.get_attrs() {
            if let module::Attribute::MemoryParams(mem) = attr {
              has_memory_params = true;
              //memory_params.depth = mem.depth;
              //memory_params.width = mem.width;
              memory_params = mem.clone();
              if let Some(init_file) = &mem.init_file {
                init_file_path.push(init_file);
                let init_file_path = init_file_path.to_str().unwrap();
                res.push_str(&format!("  /* {} */\n", init_file_path));
                has_memory_init_path = true;
              }
            }
          }

          if has_memory_params {
          } else {
            if self.sys.user_contains_opcode(ops, Opcode::Load) {
              res.push_str(&declare_array("input", &array, &display.field("q"), ","));
            }
            // (w, widx, d): something like `array[widx] = d;`
            if self.sys.user_contains_opcode(ops, Opcode::Store) {
              res.push_str(&declare_out(bool_ty(), &display.field("w")));
              res.push_str(&declare_out(array.get_idx_type(), &display.field("widx")));
              res.push_str(&declare_out(array.scalar_ty(), &display.field("d")));
            }
          }
        }
        NodeKind::Module => {
          let module = interf.as_ref::<Module>(self.sys).unwrap();
          let display = utils::DisplayInstance::from_module(&module);
          res.push_str(&format!("  // Module {}\n", module.get_name()));
          // FIXME(@were): Do not hardcode the counter delta width.
          res.push_str(&declare_out(DataType::int_ty(8), &display.field("counter_delta")));
          res.push_str(&declare_in(bool_ty(), &display.field("counter_delta_ready")));
        }
        NodeKind::Expr => {
          // This is handled below, since we need a deduplication for the modules to which these
          // expressions belong.
        }
        _ => panic!("Unknown interf kind {:?}", interf.get_kind()),
      }
      res.push('\n');
    }

    if module.is_downstream() {
      res.push_str("  // Declare upstream executed signals\n");
      upstreams(&module, &self.topo).iter().for_each(|x| {
        let name = namify(x.as_ref::<Module>(module.sys).unwrap().get_name());
        res.push_str(&declare_in(bool_ty(), &format!("{}_executed", name)));
      });
    }

    if let Some(out_bounds) = self.external_usage.out_bounds(&module) {
      for elem in out_bounds {
        let id = namify(&elem.to_string(module.sys));
        let dtype = elem.get_dtype(module.sys).unwrap();
        res.push_str(&declare_out(dtype, &format!("expose_{}", id)));
        res.push_str(&declare_out(bool_ty(), &format!("expose_{}_valid", id)));
      }
    }

    if let Some(in_bounds) = self.external_usage.in_bounds(&module) {
      for elem in in_bounds {
        let id = namify(&elem.to_string(module.sys));
        let dtype = elem.get_dtype(module.sys).unwrap();
        res.push_str(&declare_in(dtype, &id));
        res.push_str(&declare_in(bool_ty(), &format!("{}_valid", id)));
      }
    }

    if let Some(exposed_map) = self.module_expr_map.get(&module.upcast()) {
      for (exposed_node, kind) in exposed_map {
        if exposed_node.get_kind() == NodeKind::Expr {
          let expr = exposed_node.as_ref::<Expr>(self.sys).unwrap();
          let id = namify(&expr.upcast().to_string(self.sys));
          let dtype = exposed_node.get_dtype(self.sys).unwrap();
          let bits = dtype.get_bits() - 1;
          if (*kind == ExposeKind::Output) || (*kind == ExposeKind::Inout) {
            res.push_str(&format!(
              "  output logic [{bits}:0] {a}_exposed_o,\n",
              bits = bits,
              a = id
            ));
          }
          if (*kind == ExposeKind::Input) || (*kind == ExposeKind::Inout) {
            res.push_str(&format!(
              "  input logic [{bits}:0] {a}_exposed_i,\n",
              bits = bits,
              a = id
            ));
            res.push_str(&format!("  input logic {a}_exposed_i_valid,\n", a = id));
          }
        }
      }
    }

    if !module.is_downstream() {
      res.push_str("  // self.event_q\n");
      res.push_str("  input logic counter_pop_valid,\n");
      res.push_str("  input logic counter_delta_ready,\n");
      res.push_str("  output logic counter_pop_ready,\n");
    }

    res.push_str("  output logic expose_executed);\n\n");

    let mut wait_until: String = "".to_string();

    let skip = if let Some(wu_intrin) = module.get_body().get_wait_until() {
      self.before_wait_until = true;
      let mut skip = 0;
      let body = module.get_body();
      let body_iter = body.body_iter();
      for (i, elem) in body_iter.enumerate() {
        if elem == wu_intrin {
          skip = i + 1;
          break;
        }
        res.push_str(&self.print_body(elem));
      }
      let bi = wu_intrin.as_expr::<BlockIntrinsic>(self.sys).unwrap();
      let value = bi.value();
      wait_until = format!(" && ({})", namify(&value?.to_string(self.sys)));
      skip
    } else {
      0
    };
    self.before_wait_until = false;

    res.push_str("  logic executed;\n");

    if self.current_module == "testbench" {
      res.push_str(
        "
  int cycle_cnt;
  always_ff @(posedge clk or negedge rst_n) if (!rst_n) cycle_cnt <= 0;
  else if (executed) cycle_cnt <= cycle_cnt + 1;
",
      );
    }

    self.fifo_pushes.clear();
    self.array_stores.clear();
    self.triggers.clear();
    if has_memory_params {
      res.push_str(&format!("  logic [{b}:0] dataout;\n", b = memory_params.width - 1));
      self.dump_memory_nodes(module.get_body().upcast(), &mut res);
    } else {
      for elem in module.get_body().body_iter().skip(skip) {
        res.push_str(&self.print_body(elem));
      }
    }

    for (m, g) in self.triggers.drain() {
      res.push_str(&format!(
        "  assign {}_counter_delta = executed ? {} : 0;\n\n",
        m,
        if g.is_conditional() {
          g.condition
            .iter()
            .map(|x| format!("{{ {}'b0, |{} }}", g.bits - 1, x))
            .collect::<Vec<_>>()
            .join(" + ")
        } else {
          "1".into()
        }
      ));
    }

    res.push_str("  // Gather FIFO pushes\n");

    for (fifo, g) in self.fifo_pushes.drain() {
      res.push_str(&format!(
        "  assign fifo_{fifo}_push_valid = {cond};
  assign fifo_{fifo}_push_data = {value};\n
",
        cond = g.and("executed", " || "),
        value = g.select_1h()
      ));
    }

    res.push_str("  // Gather Array writes\n");

    if has_memory_params {
      res.push_str("  // this is Mem Array \n");

      for (a, (idx, data)) in &self.array_stores {
        res.push_str(&format!("  logic array_{a}_w;\n", a = a));
        res.push_str(&format!(
          "  logic [{b}:0] array_{a}_d;\n",
          a = a,
          b = memory_params.width - 1
        ));
        res.push_str(&format!(
          "  logic [{b}:0] array_{a}_widx;\n",
          a = a,
          b = (63 - (memory_params.depth - 1).leading_zeros())
        ));

        res.push_str(&format!(
          "  assign array_{a}_w = {cond};
  assign array_{a}_d = {data};
  assign array_{a}_widx = {idx};\n",
          a = a,
          cond = idx.and("executed", " || "),
          idx = idx.value.first().unwrap().clone(),
          data = data.select_1h()
        ));
        res.push_str(&format!(
          "

  memory_blackbox_{a} #(
        .DATA_WIDTH({data_width}),
        .ADDR_WIDTH({addr_bits})
    ) memory_blackbox_{a}(
    .clk     (clk),
    .address (array_{a}_widx),
    .wd      (array_{a}_d),
    .banksel (1'd1),
    .read    (1'd1),
    .write   (array_{a}_w),
    .dataout (dataout),
    .rst_n   (rst_n)
    );
          \n",
          data_width = memory_params.width,
          addr_bits = (63 - (memory_params.depth).leading_zeros()),
          a = a
        ));
      }
    } else {
      for (a, (idx, data)) in self.array_stores.drain() {
        res.push_str(&format!(
          "  assign array_{a}_w = {cond};
    assign array_{a}_d = {data};
    assign array_{a}_widx = {idx};\n
  ",
          a = a,
          cond = idx.and("executed", " || "),
          idx = idx.select_1h(),
          data = data.select_1h()
        ));
      }
    }

    if !module.is_downstream() {
      res.push_str(&format!("  assign executed = counter_pop_valid{};\n", wait_until));
      res.push_str("  assign counter_pop_ready = executed;\n");
    } else {
      let upstream_exec = upstreams(&module, &self.topo)
        .iter()
        .map(|x| format!("{}_executed", namify(x.as_ref::<Module>(module.sys).unwrap().get_name())))
        .collect::<Vec<_>>();
      res.push_str(&format!("  assign executed = {};\n", upstream_exec.join(" || ")));
    }

    res.push_str("  assign expose_executed = executed;\n");

    res.push_str(&format!("endmodule // {}\n\n", self.current_module));

    if has_memory_params {
      for (a, (_, _)) in self.array_stores.drain() {
        res.push_str(&format!(
          r#"

`ifdef SYNTHESIS
(* blackbox *)
`endif
module memory_blackbox_{a} #(
    parameter DATA_WIDTH = {data_width},
    parameter ADDR_WIDTH = {addr_bits}
)(
    input clk,
    input [ADDR_WIDTH-1:0] address,
    input [DATA_WIDTH-1:0] wd,
    input banksel,
    input read,
    input write,
    output reg [DATA_WIDTH-1:0] dataout,
    input rst_n
);

    localparam DEPTH = 1 << ADDR_WIDTH;
    reg [DATA_WIDTH-1:0] mem [DEPTH-1:0];

  "#,
          a = a,
          data_width = memory_params.width,
          addr_bits = (63 - (memory_params.depth).leading_zeros())
        ));
        if has_memory_init_path {
          res.push_str(&format!(
            r#"  initial begin
          $readmemh({:?}, mem);
      end
        always @ (posedge clk) begin
            if (write & banksel) begin
                mem[address] <= wd;
            end
        end

        assign dataout = (read & banksel) ? mem[address] : {{DATA_WIDTH{{1'b0}}}};

    endmodule
              "#,
            init_file_path
          ))
        } else {
          res.push_str(
            r#"

        always @ (posedge clk) begin
            if (!rst_n) begin
                mem[address] <= {{DATA_WIDTH{{1'b0}}}};
            end
            else if (write & banksel) begin
                mem[address] <= wd;
            end
        end

        assign dataout = (read & banksel) ? mem[address] : {{DATA_WIDTH{{1'b0}}}};

    endmodule
              "#,
          );
        }
      }
    }

    Some(res)
  }

  fn visit_block(&mut self, block: BlockRef<'_>) -> Option<String> {
    let mut res = String::new();
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
      match elem.get_kind() {
        NodeKind::Expr => {
          let expr = elem.as_ref::<Expr>(self.sys).unwrap();
          res.push_str(&self.visit_expr(expr).unwrap());
        }
        NodeKind::Block => {
          let block = elem.as_ref::<Block>(self.sys).unwrap();
          res.push_str(&self.visit_block(block).unwrap());
        }
        _ => {
          panic!("Unexpected reference type: {:?}", elem);
        }
      }
    }
    self.pred_stack.pop_back();
    res.into()
  }

  fn visit_expr(&mut self, expr: ExprRef<'_>) -> Option<String> {
    visit_expr_impl(self, expr)
  }
}
