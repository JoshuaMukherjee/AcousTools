from torch import Tensor
from acoustools.Utilities import create_points

def interweave_points(lcode:str, num_points:int=2, min_offset_lines:int=100, extruder:Tensor|None=None, codes_to_combine:list[str]= ['L0','L1','L2','L3']):
    '''
    Will optimise a lcode file by making single point lcode use multiple points at once \n
    Currently seems to perform slightly weirdly sometimes adding a rogue line between configurations commands - should work though
    :param lcode: filename of lcode file
    :param num_points: Maximum number of points allowed
    :param min_offset_lines: minimum number of linesafter starting one point before starting a new point 
    :param extruder: extruder position:
    :param codes_to_combine: The codes that can be combined in one line - note that if different codes are present it may break
    '''

    #The rogue single lines after a C0 i think come from the fact that the interweaved line becomes L1...; C0; which is then split so the C0 is placed first
    #This means the L command is left alone on a line and the C0 comes after -> this seems like its ok? There isnt two drops to levitate 
    #It is slightly less efficient -> You could move the C0 above the rogue line and then have two on the next but on thw whole isnt too bad
    blocks = []
    c_blocks = []
    pre_c_blocks = []
    start_c_block = ''
    
    if extruder is None:
        extruder = create_points(1,1,0,-0.04, 0.04)
    extruder_x = extruder[:,0].item()
    extruder_y = extruder[:,1].item()
    extruder_z = extruder[:,2].item()
    

    block = ''
    c_block = ''
    pre_c_block = ''

    add_to_start = False

    with open(lcode, 'r') as f:
        lines = f.readlines()
        found_L = False
        for line in lines:
            
            if line.startswith('L') and not found_L:
                found_L = True
                start_c_block = c_block
                c_block = ''
            
            if found_L:
                parts = line.replace(';','').rstrip().split(':')
                command = parts[0]
                args = parts[1:]
                
                if command in codes_to_combine:
                    x,y,z = lcode_coordinates_to_xyz(args[0]) #This assumes only one point per line
                    if x == extruder_x and y == extruder_y and z == extruder_z and block != '':
                        blocks.append(block)
                        c_blocks.append(c_block)
                        
                        block = line
                        c_block = ''

                        pre_c_blocks.append(pre_c_block)
                        pre_c_block = ''

                    else:
                        block += line
                else: #C commands mostly
                    if line.startswith('O0'): #End the block!
                        add_to_start = True
                        continue

                    if not add_to_start:
                        c_block += line
                    else:
                        pre_c_block += line
                        add_to_start = False

            else:
                c_block += line
    

    current_blocks = []
    current_blocks.append(blocks[0])
    gap = 0
    i = 0
    N_blocks = len(blocks)
    interweaved = []
    blocks_done = 0
    while len(current_blocks) > 0:
        gap += 1
        if gap > min_offset_lines and len(current_blocks) < num_points and i+1 < N_blocks: #We can add a new block
            interweaved.append(pre_c_blocks[i])
            i += 1
            current_blocks.append(blocks[i])
            gap = 0
            
        line, current_blocks = combine_lines_from_blocks(current_blocks)
        interweaved.append(line)
        
            
        if '' in current_blocks: #If the block is empty ...
            current_blocks.remove('') #Then remove it ...
            interweaved.append([c_blocks[blocks_done],]) #Add the post commands ...
            blocks_done += 1 # And remember we have completed a block

        if len(current_blocks) == 0 and blocks_done < N_blocks: #Check we dont run out of blocks before starting the next one
            current_blocks.append(blocks[blocks_done])

    i = 0
    with open(lcode, 'w') as f:
        f.write(start_c_block)

        for lines in interweaved:
            c_commands = ''
            command = ''
            arguments = []
            for line in lines:
                if line.startswith('L'):
                    line=line.replace(';','').rstrip()
                    spt = line.split(':')
                    command = spt[0]
                    if command in codes_to_combine:
                        coords = spt[1]
                        xyz = lcode_coordinates_to_xyz(coords)
                        xyz = [str(a) for a in xyz]
                        coords_str = ','.join(xyz)
                        arguments.append(coords_str)
                        
                
                else:
                    c_commands += line
            # print(arguments)   
            # if len(arguments) == 0:
            #         print(command)
            if command != '':
                arguments_string = ':'.join(arguments)
                f.write(command + ":" + arguments_string + ';\n')
            if c_commands != '':
                f.write(c_commands)

    
def lcode_coordinates_to_xyz(coordinates:str) -> tuple[float]:
    '''
    @private
    '''
    coordinates = coordinates.split(',')
    x = float(coordinates[0])
    y = float(coordinates[1])
    z = float(coordinates[2])

    return x,y,z


def combine_lines_from_blocks(blocks) -> tuple[list[str],list[str]]:
    '''
    @private
    '''
    top_lines = []
    new_blocks = []
    for block in blocks:
        ls = block.split('\n')
        top_lines.append(ls[0])
        new_blocks.append('\n'.join(ls[1:]))
    
    
    return top_lines, new_blocks
