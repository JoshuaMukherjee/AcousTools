from acoustools.Fabrication.Translator import gcode_to_lcode


pth = 'acoustools/tests/data/gcode/teapot.gcode'

pre_cmd = 'C0;\n'
post_cmd = 'C1;\nC3:10;\nC2;\n'
pre_file = 'C5:gspat;#set solver\nC8;#set topbaord\nC6:20;#set iterations\n'
gcode_to_lcode(pth, pre_print_command=pre_cmd, post_print_command=post_cmd, print_lines=True, pre_commands=pre_file)
