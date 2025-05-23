from assassyn.frontend import *
from assassyn.backend import elaborate
from assassyn import utils
from python.assassyn.ip.multiply import multiply
import assassyn
import pytest
import random

# AIM: an ASIC realizing a*b+c (64bit)

class PlusC(Module):
    def __init__(self):
        super().__init__(
            ports={
                'a': Port(Int(32)),
                'b': Port(Int(32)),
                'c': Port(Int(64)),
                'axb': Port(Int(64)),
                'cnt': Port(Int(32))
            }
        )
        
    @module.combinational
    def build(self, stage4_reg: Array):
        a, b, c, axb, cnt = self.pop_all_ports(True)
        stage4_reg[0] = axb + c
        with Condition(cnt == Int(32)(37)):
            log("The result of {:?} * {:?} + {:?} = {:?}", a, b, c, stage4_reg[0])
            

class Driver(Module):
    def __init__(self):
        super().__init__(ports={})

    @module.combinational
    def build(self, plusc: PlusC):
        cnt = RegArray(Int(32), 1)
        cnt[0] = cnt[0] + Int(32)(1)
        cond = cnt[0] < Int(32)(40)

        # test input
        input_a = RegArray(Int(32), 1)
        input_b = RegArray(Int(32), 1)
        input_c = RegArray(Int(64), 1)
        input_a[0] = Int(32)(random.randint(0, 0x7FFFFFF))
        input_b[0] = Int(32)(random.randint(0, 0x7FFFFFF))
        input_c[0] = Int(64)(random.randint(0, 0x7FFFFFF))

        with Condition(cond):
            axb = multiply(input_a[0], input_b[0], cnt[0])
            plusc.async_called(a=input_a[0], b=input_b[0], c=input_c[0], axb=axb, cnt=cnt[0])

def check_raw(raw):
    cnt = 0
    for i in raw.split('\n'):
        if 'The result' in i:
            line_toks = i.split()
            d = line_toks[-1]
            c = line_toks[-3]
            b = line_toks[-5]
            a = line_toks[-7]
            assert int(a) * int(b) + int(c)== int(d)


def test_axbplusc():
    sys = SysBuilder('axbplusc_test')

    with sys:
        stage4_reg = RegArray(Int(64), 1)

        plusc = PlusC()
        plusc.build(stage4_reg)
        driver = Driver()
        driver.build(plusc)

    print(sys)

    simulator_path, verilator_path = elaborate(sys, verilog=utils.has_verilator())

    raw = utils.run_simulator(simulator_path)
    check_raw(raw)

    if verilator_path:
        raw = utils.run_verilator(verilator_path)
        check_raw(raw)


if __name__ == '__main__':
    test_axbplusc()
