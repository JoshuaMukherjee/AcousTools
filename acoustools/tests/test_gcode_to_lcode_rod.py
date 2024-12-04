from acoustools.Fabrication.Translator import gcode_to_lcode
import acoustools.Mesh
from acoustools.Utilities import create_points


pth = 'acoustools/tests/data/gcode/Rod-lam1.gcode'

pre_cmd = 'C0;\n'
post_cmd = 'C1;\nC3:10;\nC2;\n'

via = create_points(1,1,0,0,0.07)
gcode_to_lcode(pth, pre_print_command=pre_cmd, post_print_command=post_cmd, print_lines=True,  via=via, use_functions=True, max_stepsize=0.001)
