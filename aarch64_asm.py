import struct

class Assembler:
    def __init__(self):
        self.buffer = b""

    def _append_binstr(self, binstr):
        assert len(binstr) == 32
        self.buffer += int(binstr, 2).to_bytes(4, 'little')

    def ADC(self, rd, rn, rm, sf=1):
        self._append_binstr(f"{sf:01b}0011010000{rm:05b}000000{rn:05b}{rd:05b}")

    def ADC(self, rd, rn, rm, sf=1):
        self._append_binstr(f"{sf:01b}0111010000{rm:05b}000000{rn:05b}{rd:05b}")

    def ADD_imm(self, rd, rn, sf=1, imm=0, shift=0):
        self._append_binstr(f"{sf:01b}00100010{shift:01b}{imm:012b}{rn:05b}{rd:05b}")

    def ADD_shift(self, rd, rn, rm, sf=1, shift=0, imm=0):
        self._append_binstr(f"{sf:01b}0001011{shift:02b}0{rm:05b}{imm:06b}{rn:05b}{rd:05b}")

    def ADDS_imm(self, rd, rn, sf=1, imm=0, shift=0):
        self._append_binstr(f"{sf:01b}01100010{shift:01b}{imm:012b}{rn:05b}{rd:05b}")

    def ADDS_shift(self, rd, rn, rm, sf=1, imm=0, shift=0):
        self._append_binstr(f"{sf:01b}0101011{shift:02b}0{rm:05b}{imm:06b}{rn:05b}{rd:05b}")

    def SUB_imm(self, rd, rn, sf=1, imm=0, shift=0):
        self._append_binstr(f"{sf:01b}10100010{shift:01b}{imm:012b}{rn:05b}{rd:05b}")

    def SUB_shift(self, rd, rn, rm, sf=1, imm=0, shift=0):
        self._append_binstr(f"{sf:01b}1001011{shift:02b}0{rm:05b}{imm:06b}{rn:05b}{rd:05b}")

    def SUBS_imm(self, rd, rn, sf=1, imm=0, shift=0):
        self._append_binstr(f"{sf:01b}11100010{shift:01b}{imm:012b}{rn:05b}{rd:05b}")
    
    def SUBS_shift(self, rd, rn, rm, sf=1, imm=0, shift=0):
        self._append_binstr(f"{sf:01b}1101011{shift:02b}0{rm:05b}{imm:06b}{rn:05b}{rd:05b}")

    def CMN_imm(self, rn, sf=1, imm=0, shift=0):
        self.ADDS_imm(31, rn, sf=sf, imm=imm, shift=shift)

    def CMN_shift(self, rn, rm, sf=1, amount=0, shift=0):
        self.ADDS_shift(31, rn, rm, sf=sf, imm=amount, shift=shift)

    def CMP_imm(self, rn, sf=1, imm=0, shift=0):
        self.SUBS_imm(31, rn, sf=sf, imm=imm, shift=shift)

    def CMP_shift(self, rn, rm, sf=1, imm=0, shift=0):
        self.SUBS_shift(31, rn, rm, sf=sf, imm=imm, shift=shift)

    def _logical_imm_str(sf, imm):
        # TODO sf=0
        assert imm not in {0, 2 ** 64 - 1}
        imm_str = f"{imm:064b}"
        rot = 0
        while not (imm_str[-1] == "1" and imm_str[0] == "0"):
            rot += 1
            imm_str = imm_str[1:] + imm_str[0]
        cnt_1 = 0
        while imm_str[63 - cnt_1] == "1": cnt_1 += 1
        cnt_0 = 0
        while cnt_0 + cnt_1 < 64 and imm_str[63 - cnt_0 - cnt_1] == "0": cnt_0 += 1
        size = cnt_0 + cnt_1
        assert size in {2, 4, 8, 16, 32, 64} # power of two
        div = 0
        stride = 0
        while stride < 64:
            div += (2 ** stride) 
            stride += size
        assert imm % div == 0
        # ok we done verifying, building bit patter for instr
        N = "0"
        if size == 2:
            imms = f"11110{cnt_1-1:01b}"
        elif size == 4:
            imms = f"1110{cnt_1-1:02b}"
        elif size == 8:
            imms = f"110{cnt_1-1:03b}"
        elif size == 16:
            imms = f"10{cnt_1-1:04b}"
        elif size == 32:
            imms = f"0{cnt_1-1:05b}"
        elif size == 64:
            N = "1"
            imms = f"{cnt_1-1:06b}"
        return f"{N}{rot:06b}{imms}"

    def AND_imm(self, rd, rn, sf=1, imm=0):
        imm_str = Assembler._logical_imm_str(sf, imm)
        self._append_binstr(f"{sf:01b}00100100{imm_str}{rn:05b}{rd:05b}")
    
    def AND_shift(self, rd, rn, rm, sf=1, shift=0, imm=0):
        self._append_binstr(f"{sf:01b}0001010{shift:02b}0{rm:05b}{imm:06b}{rn:05b}{rd:05b}")

    def EOR_imm(self, rd, rn, sf=1, imm=0):
        imm_str = Assembler._logical_imm_str(sf, imm)
        self._append_binstr(f"{sf:01b}10100100{imm_str}{rn:05b}{rd:05b}")

    def EOR_shift(self, rd, rn, rm, sf=1, shift=0, imm=0):
        self._append_binstr(f"{sf:01b}1001010{shift:02b}0{rm:05b}{imm:06b}{rn:05b}{rd:05b}")
    
    def ORR_imm(self, rd, rn, sf=1, imm=0):
        imm_str = Assembler._logical_imm_str(sf, imm)
        self._append_binstr(f"{sf:01b}01100100{imm_str}{rn:05b}{rd:05b}")

    def ORR_shift(self, rd, rn, rm, sf=1, shift=0, imm=0):
        self._append_binstr(f"{sf:01b}0101010{shift:02b}0{rm:05b}{imm:06b}{rn:05b}{rd:05b}")

    def MOV_bit(self, rd, sf=1, imm=0):
        self.ORR_imm(rd, 31, sf=sf, imm=imm)

    # MOV (inverted wide immediate)
    # def MOV_inv_wide(self, rd, sf=1, imm=0):

    def MOV_reg(self, rd, rm, sf=1):
        self.ORR_shift(rd, 32, rm, sf=sf)
    
    def MOV_sp(self, rd, rn, sf=1):
        self.ADD_imm(rd, rn, sf=sf)
    
    def MOVK(self, rd, sf=1, imm=0, shift=0):
        assert shift in {0, 16, 32, 48}
        hw = shift // 16
        assert not (sf == 0 and hw&2 == 1)
        self._append_binstr(f"{sf:01b}11100101{hw:02b}{imm:016b}{rd:05b}")
    
    def MOVN(self, rd, sf=1, imm=0, shift=0):
        assert shift in {0, 16, 32, 48}
        hw = shift // 16
        assert not (sf == 0 and hw&2 == 1)
        self._append_binstr(f"{sf:01b}00100101{hw:02b}{imm:016b}{rd:05b}")

    def MOVZ(self, rd, sf=1, imm=0, shift=0):
        assert shift in {0, 16, 32, 48}
        hw = shift // 16
        assert not (sf == 0 and hw&2 == 1)
        self._append_binstr(f"{sf:01b}10100101{hw:02b}{imm:016b}{rd:05b}")

    def MADD(self, rd, rn, rm, ra, sf=1):
        self._append_binstr(f"{sf:01b}0011011000{rm:05b}0{ra:05b}{rn:05b}{rd:05b}")

    def MSUB(self, rd, rn, rm, ra, sf=1):
        self._append_binstr(f"{sf:01b}0011011000{rm:05b}1{ra:05b}{rn:05b}{rd:05b}")
    
    def MNEG(self, rd, rn, rm, sf=1):
        self.MSUB(rd, rn, rm, 31, sf=sf)
    
    def MUL(self, rd, rn, rm, sf=1):
        self.MADD(rd, rn, rm, 31, sf=sf)

    def B(self, offset):
        self._append_binstr(f"000101{offset:026b}")

    def _cond_bin_str(cond):
        if cond == "eq": return "0000"
        elif cond == "ne": return "0001"
        elif cond == "cs": return "0010"
        elif cond == "cc": return "0011"
        elif cond == "mi": return "0100"
        elif cond == "pl": return "0101"
        elif cond == "vs": return "0110"
        elif cond == "vc": return "0111"
        elif cond == "hi": return "1000"
        elif cond == "ls": return "1001"
        elif cond == "ge": return "1010"
        elif cond == "lt": return "1011"
        elif cond == "gt": return "1100"
        elif cond == "le": return "1101"
        elif cond == "al": return "1110" # "1111" also passes
        else: assert False

    def B_cond(self, cond, offset):
        self._append_binstr(f"01010100{offset:019b}0{Assembler._cond_bin_str(cond)}")

    def NOP(self):
        self._append_binstr(f"11010101000000110010000000011111")

    def RET(self, rn=30):
        self._append_binstr(f"1101011001011111000000{rn:05b}00000")

    def _ftype_bin_str(ftype):
        if ftype == 16: return "11"
        elif ftype == 32: return "00"
        elif ftype == 64: return "01"
        else: assert False

    def FADD(self, rd, rn, rm, ftype=32):
        self._append_binstr(f"00011110{Assembler._ftype_bin_str(ftype)}1{rm:05b}001010{rn:05b}{rd:05b}")

    def FMUL(self, rd, rn, rm, ftype=32):
        self._append_binstr(f"00011110{Assembler._ftype_bin_str(ftype)}1{rm:05b}000010{rn:05b}{rd:05b}")
    
    def FNEG(self, rd, rn, ftype=32):
        self._append_binstr(f"00011110{Assembler._ftype_bin_str(ftype)}100001010000{rn:05b}{rd:05b}")

    def FMOV_reg(self, rd, rn, ftype=32):
        self._append_binstr(f"00011110{Assembler._ftype_bin_str(ftype)}100000010000{rn:05b}{rd:05b}")
    
    # TODO I should panic/better-approximate if imm cannot be fitted in 1-byte float from a64 spec
    def _float_imm_bin_str(imm):
        float_str = f"{int.from_bytes(struct.pack('>e', imm), 'big'):016b}"
        print(float_str)
        exp16_str = float_str[1:6]
        if exp16_str[0] == "0":
            if exp16_str[1:3] == "11":
                exp_str = "1" + exp16_str[3:5]
            else:
                exp_str = "100" # approx, todo: set fraction to '1111'
        else:
            if exp16_str[1:3] == "00":
                exp_str = "0" + exp16_str[3:5]
            else:
                exp_str = "011" # approx, todo: set fraction to '1111'
        return float_str[0] + exp_str + float_str[6:10]

    def FMOV_imm(self, rd, imm=0.0, ftype=32):
        imm_str = Assembler._float_imm_bin_str(imm)
        self._append_binstr(f"00011110{Assembler._ftype_bin_str(ftype)}1{imm_str}10000000{rd:05b}")

asm = Assembler()

# asm.ADD_imm(0, 0, imm=2)
# asm.ADD_shift(0, 0, 0)
# asm.ADD_shift(0, 0, 0)

# asm.CMP_imm(0, imm=0)
# asm.B_cond("ge", 3)
# asm.ADD_shift(0, 0, 0)
# asm.SUB_shift(0, 31, 0)

# asm.MOVZ(1, imm=41, shift=0)
# asm.MUL(0, 0, 1)

# asm.FADD(1, 0, 31)
# asm.FADD(0, 0, 0)
# asm.FNEG(1, 1)
# asm.FMUL(0, 0, 1)
# asm.FMOV_imm(0, imm=1024)

asm.SVE_MOV()

asm.RET()

import ctypes, ctypes.util
from mmap import PAGESIZE

print(asm.buffer.hex(" "))

inst_buf = ctypes.create_string_buffer(asm.buffer)

libc = ctypes.CDLL(ctypes.util.find_library("libc"))

first_page_addr = (ctypes.addressof(inst_buf) // PAGESIZE) * PAGESIZE
# TODO why ulong
ret = libc.mprotect(ctypes.c_ulong(first_page_addr), 2 * PAGESIZE, 4|2|1)
assert ret == 0

func_proto = ctypes.CFUNCTYPE(ctypes.c_int64, ctypes.c_int64)
# func_proto = ctypes.CFUNCTYPE(ctypes.c_float, ctypes.c_float)
func = func_proto(ctypes.cast(inst_buf, ctypes.c_void_p).value)

print(func(3))