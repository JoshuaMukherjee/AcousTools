'''
Lcode Specification \n
Commands
* `L0:<X> <Y> <Z>;` Create Focal Point at (X,Y,Z)
* `L1:<X> <Y> <Z>;` Create Trap Point at (X,Y,Z)
* `L2:<X> <Y> <Z>;` Create Twin Trap Point at (X,Y,Z)
* `L3:<X> <Y> <Z>;` Create Vortex Trap Point at (X,Y,Z)
* `L4;` Turn off Transducers

* `C0;`Dispense Droplet
* `C1;` Activate UV
* `C2;` Turn off UV
* `C3:<T>;` Delay for T ms
* `C4:<T>;` Set delay for T ms between all commands
* `C5:<Solver>;` Change to specific solver. Should be one of "IB", "WGS", "GSPAT", "NAIVE"
* `C6:<N>;` Set number of iterations for the solver
* `C7;` Set to two board setup
* `C8;` Set to top board setup
* `C9;` Set to bottom board setup
* `C10;` Update BEM to use layer at last z position 
* `C11:<Frame-rate>;` Set the framerate of the levitator device
* `C12:<Extruder>;` Set a new extruder position
* `C13:<z>;` Use a reflector and set the position

* `O0;` End of droplet

* `function F<x>
...
end` define a function that can latter be called by name
'''