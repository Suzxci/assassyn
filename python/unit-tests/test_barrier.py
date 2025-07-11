from assassyn.frontend import *
from assassyn.backend import elaborate
from assassyn import utils
import assassyn

# TODO(@were): Make this barrier work with more careful tests.

class Adder2(Module):

    def __init__(self):
        super().__init__(
            ports={
                'a': Port(Int(32)),
                'b': Port(Int(32)),
            },
        )

    @module.combinational
    def build(self):
        a, b = self.pop_all_ports(True)
        c = a * b
 
        d = a + b + c
        log("combi: {} + {} + {}*{} = {} ", a, b, a, b, d)

        return d

class Adder1(Module):

    def __init__(self):
        super().__init__(
            ports={
                'a': Port(Int(32)),
                'b': Port(Int(32)),
                'c': Port(Int(32)),
            },
        )

    @module.combinational
    def build(self,adder: Adder2,is_gold = False):
        a, b , c= self.pop_all_ports(True)
        e = a * b
        if is_gold:
            barrier(e)
        
        d = a + b + e
        h = d + Int(32)(1)
        g = h * h
        if is_gold:
            barrier(g)
        
        f = g * c        

        adder.async_called(a = f[0:31].bitcast(Int(32)), b = d[0:31].bitcast(Int(32)))

        return f
    


class Driver(Module):

    def __init__(self):
        super().__init__(ports={})

    @module.combinational
    def build(self, adder: Adder1):
        # The code below is equivalent
        # cnt = RegArray(Int(32), 0)
        # v = cnt[0]
        # cnt[0] = v + Int(32)(1)
        # NOTE: cnt[0]'s new value is NOT visible until next cycle.
        # cond = v < Int(32)(100)
        # with Condition(cond):
        #     adder.async_called(a = v, b = v)
        cnt = RegArray(Int(32), 1)
        cnt[0] = cnt[0] + Int(32)(1)
        cnt_div2_temp = cnt[0] + Int(32)(1)
        cnt_div2 = Int(32)(0)
        cnt_div2 = cnt[0][0:0].select(cnt[0], cnt_div2_temp)

        cond = cnt[0] < Int(32)(100)

        with Condition(cond):
            adder.async_called(a = cnt_div2, b = cnt_div2 , c = cnt_div2)




def impl(is_gold):
    if is_gold:
        sys = SysBuilder('Comb_barrier_gold')
    else:
        sys = SysBuilder('Comb_barrier')
    with sys:
        adder2 = Adder2()
        res = adder2.build()
        adder1 = Adder1()
        d = adder1.build(adder2,is_gold)
        driver = Driver()
        driver.build(adder1)
    print(sys)
    config = assassyn.backend.config(
            verilog=utils.has_verilator(),
            sim_threshold=200,
            idle_threshold=200,
            random=True)

    simulator_path, verilator_path  = elaborate(sys, **config)
    raw = utils.run_simulator(simulator_path)

def test_barrier():
    impl(False)

def test_barrier_gold():
    impl(True)

if __name__ == '__main__':
    test_barrier()
    test_barrier_gold()
