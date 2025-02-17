from acoustools.Fabrication.Translator import gcode_to_lcode


pth = 'acoustools/tests/data/gcode/rectangle.gcode'

pre_cmd = 'C0;\n'
post_cmd = 'C1;\nC3:10;\nC2;\n'
gcode_to_lcode(pth, pre_print_command=pre_cmd, post_print_command=post_cmd,sig_type='Twin')
