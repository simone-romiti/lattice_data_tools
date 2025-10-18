

import typing

class PowersOf10:
    @staticmethod
    def x10y_strings(number, decimals: int):
        """ 
        Strings x,y for writing: number=x*10^y
        
        decimals = number of decimal places after the dot in the "a" string
        """
        e_notation = f"{number:.{decimals}e}"
        LR = e_notation.split("e") # strings to the left and right of "e"
        x = LR[0]
        y = str(int(LR[1])) # removing "0", e.g. 1.2e-04
        return (x, y)


class to_LaTeX:
    @staticmethod
    def as_x10y(
        number, decimals: int, 
        dot_symb: typing.Literal["", "\\cdot", "\\times"] = "\\times",
        include_dollars=False):
        x, y = PowersOf10.x10y_strings(number=number, decimals=decimals)
        res = x+f" {dot_symb} 10^{{{y}}}"
        res = res if not include_dollars else "$"+res+"$"
        return res