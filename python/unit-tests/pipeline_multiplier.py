from assassyn.frontend import *
from assassyn.backend import elaborate
from assassyn import utils
import assassyn
import pytest
import random

# AIM: a multiplier that can do pipeline multiplication

# Stage 3: sum up 4 groups (sum of 32 bits)
class MulStage3(Module):
    def __init__(self):
        super().__init__(ports={
            'a': Port(Int(32)),
            'b': Port(Int(32)),
            'd1': Port(Int(64)),
            'd2': Port(Int(64)),
            'd3': Port(Int(64)),
            'd4': Port(Int(64))
            }
        )
    @module.combinational
    def build(self):
        a, b, d1, d2, d3, d4 = self.pop_all_ports(True)
        c = d1*Int(64)(1) + d2*Int(64)(256) + d3*Int(64)(65536) + d4*Int(64)(16777216)  #multiply weights
        log("Final result {:?} * {:?} = {:?}", a, b, c)
        

# MulStage 2: sum up two groups (sum of 8 bits)
class MulStage2(Module):
    def __init__(self):
        super().__init__(
            ports={
                'a': Port(Int(32)),
                'b': Port(Int(32)),
                'c0': Port(Int(64)),
                'c1': Port(Int(64)),
                'c2': Port(Int(64)),
                'c3': Port(Int(64)),
                'c4': Port(Int(64)),
                'c5': Port(Int(64)),
                'c6': Port(Int(64)),
                'c7': Port(Int(64))
            }
        )

    @module.combinational
    def build(self, mulstage3: MulStage3):
        a,b,c0,c1,c2,c3,c4,c5,c6,c7 = self.pop_all_ports(True)
        d1 = c0 + ((c1*Int(64)(16))[0:63]).bitcast(Int(64))
        d2 = c2 + ((c3*Int(64)(16))[0:63]).bitcast(Int(64))
        d3 = c4 + ((c5*Int(64)(16))[0:63]).bitcast(Int(64))
        d4 = c6 + ((c7*Int(64)(16))[0:63]).bitcast(Int(64))
        log("MulStage2: d1={:?},d2={:?},d3={:?},d4={:?}",d1,d2,d3,d4)
        
        mulstage3.async_called(a=a, b=b, d1=d1,d2=d2,d3=d3,d4=d4)
        


# MulStage 1: sum of 4 bits
class MulStage1(Module):
    def __init__(self):
        super().__init__(
            ports={
                'a': Port(Int(32)),
                'b': Port(Int(32))
            }
        )

    @module.combinational
    def build(self, mulstage2: MulStage2):
        a, b = self.pop_all_ports(True)
        tmp = []
        for i in range(1,9):
            count = (Int(32)(i)-Int(32)(1)) * Int(32)(4)
            b_4bit = ((b >> count) & Int(32)(15)).bitcast(Int(32))  #take out 4 bit in b
            b0 = ((b_4bit) & Int(32)(1)).bitcast(Int(32))
            b1 = ((b_4bit >> Int(32)(1)) & Int(32)(1)).bitcast(Int(32))
            b2 = ((b_4bit >> Int(32)(2)) & Int(32)(1)).bitcast(Int(32))
            b3 = ((b_4bit >> Int(32)(3)) & Int(32)(1)).bitcast(Int(32))
            
            c = (a*b0*Int(64)(1) + a*b1*Int(64)(2) + a*b2*Int(64)(4) + a*b3*Int(64)(8))[0:63].bitcast(Int(64)) 
            tmp.append(c) 
        
        mulstage2.async_called(a=a, b=b, c0=tmp[0], c1=tmp[1], c2=tmp[2], c3=tmp[3],
        c4=tmp[4], c5=tmp[5], c6=tmp[6], c7=tmp[7])
        
    
    
class Driver(Module):
    def __init__(self):
        super().__init__(ports={})

    @module.combinational
    def build(self, mulstage1: MulStage1):
        cnt = RegArray(Int(32), 1)
        cnt[0] = cnt[0] + Int(32)(1)
        cond = (cnt[0] < Int(32)(100))&(cnt[0] > Int(32)(0)) 

        input_a = RegArray(Int(32),1)
        input_b = RegArray(Int(32),1)
        input_a[0] = input_a[0] + Int(32)(1)
        input_b[0] = input_b[0] + Int(32)(1)

        with Condition(cond):
            mulstage1.async_called(a=input_a[0], b=input_b[0])


def check_raw(raw):
    cnt = 0
    for i in raw.split('\n'):
        if 'Final' in i:
            line_toks = i.split()
            c = line_toks[-1]
            b = line_toks[-3]
            a = line_toks[-5]
            assert int(a) * int(b) == int(c)


def test_multi():
    sys = SysBuilder('mul_multiplier')

    with sys:
        
        mulstage3 = MulStage3()
        mulstage3.build()
        mulstage2 = MulStage2()
        mulstage2.build(mulstage3)
        mulstage1 = MulStage1()
        mulstage1.build(mulstage2)
        driver = Driver()
        driver.build(mulstage1)

    print(sys)

    simulator_path, verilator_path = elaborate(sys, verilog=utils.has_verilator())

    raw = utils.run_simulator(simulator_path)
    check_raw(raw)

    if verilator_path:
        raw = utils.run_verilator(verilator_path)
        check_raw(raw)

if __name__ == '__main__':
    test_multi()

    