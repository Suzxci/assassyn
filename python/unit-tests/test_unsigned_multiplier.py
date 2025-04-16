from assassyn.frontend import *
from assassyn.backend import elaborate
from assassyn import utils
import assassyn
import pytest

#AIM: unsigned 32b multiplier: 32b*32b=32b
#DATE: 2025/4/16

# Stage 1: multiply each bit of b
class Stage1(Module):
    def __init__(self):
        super().__init__(
            ports={
                'a': Port(Int(32)),
                'b': Port(Int(32)),
                'cnt': Port(Int(32)),
            }
        )

    @module.combinational
    def build(self, stage1_reg: Array):
        a, b, cnt = self.pop_all_ports(True)
        bit_num0 = cnt % Int(32)(32) #to avoid overflow
        b_bit = (b>>bit_num0)&Int(32)(1) #to get the cnt-th bit from the right
        b_bit = b_bit.bitcast(Int(1))
        stage1_reg[0] = (a * b_bit).bitcast(Int(32)) #'a' multiply b[cnt-1]
        log("Stage1: {:?} * {:?} = {:?}", a, b_bit, a*b_bit)

# Stage 2: left shift to multiply weight
class Stage2(Module):
    def __init__(self):
        super().__init__(
            ports={
                'cnt':Port(Int(32))
            }
        )

    @module.combinational
    def build(self, stage1_reg: Array, stage2_reg: Array):
        cnt = self.pop_all_ports(True)

        with Condition (cnt>Int(32)(0)):
            bit_num = (cnt - Int(32)(1)) % Int(32)(32) #avoid overflow
            stage2_reg[0] = stage1_reg[0] << bit_num #left shift as multiplying weights

        log("Stage2: {:?}", stage2_reg[0])

# Stage 3: add with the final result
class Stage3(Module):
    def __init__(self):
        super().__init__(ports={
                'cnt':Port(Int(32)),
                'a':Port(Int(32)),
                'b':Port(Int(32))
            }
        )

    @module.combinational
    def build(self, stage2_reg: Array, stage3_reg: Array):
        cnt, a, b = self.pop_all_ports(True)
        stage3_reg[0] = stage2_reg[0] + stage3_reg[0]
        log("Stage3: {:?}", stage3_reg[0])
        log("Temp result {:?} of {:?} * {:?} = {:?}", cnt, a, b, stage3_reg[0])

        with Condition(cnt == Int(32)(34)): #output final result
            log("Final result {:?} * {:?} = {:?}", a, b, stage3_reg[0])


class Driver(Module):
    def __init__(self):
        super().__init__(ports={})

    @module.combinational
    def build(self, stage1: Stage1, stage2: Stage2, stage3: Stage3):
        cnt = RegArray(Int(32), 1)

        cnt[0] = cnt[0] + Int(32)(1)
        cond = cnt[0] < Int(32)(35)
        # test input
        a = Int(32)(18)
        b = Int(32)(119304607)
        with Condition(cond):
            stage1.async_called(a=a, b=b, cnt=cnt[0])
            stage2.async_called(cnt=cnt[0])
            stage3.async_called(cnt=cnt[0], a=a, b=b)


def check_raw(raw):
    cnt = 0
    for i in raw.split('\n'):
        if 'Temp result 34' in i:
            line_toks = i.split()
            c = line_toks[-1]
            b = line_toks[-3]
            a = line_toks[-5]
            assert int(a) * int(b) == int(c)

def test_pipeline():
    sys = SysBuilder('pipeline_test')
    with sys:
        stage1_reg = RegArray(Int(32), 1)
        stage2_reg = RegArray(Int(32), 1)
        stage3_reg = RegArray(Int(32), 1)

        stage1 = Stage1()
        stage1.build(stage1_reg)
        stage2 = Stage2()
        stage2.build(stage1_reg, stage2_reg)
        stage3 = Stage3()
        stage3.build(stage2_reg, stage3_reg)
        driver = Driver()
        driver.build(stage1, stage2, stage3)

    print(sys)

    simulator_path, verilator_path = elaborate(sys, verilog=utils.has_verilator())

    raw = utils.run_simulator(simulator_path)
    check_raw(raw)

    if verilator_path:
        raw = utils.run_verilator(verilator_path)
        check_raw(raw)

if __name__ == '__main__':
    test_pipeline()
