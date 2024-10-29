
import pickle
import time
from acoustools.Levitator import LevitatorController

N  =1000

xs = pickle.load(open('acoustools/tests/data/circle' + str(N) + '.pth','rb'))


print('Finished Computing \nConnecting to PAT...')
try:
    lev = LevitatorController(ids=(73,53))
    print('Connected')
    lev.levitate(xs[0])

    
    input()
    print('Moving...')

    print(len(xs))
    # lev.set_frame_rate(10)
    # lev.levitate(xs,num_loops=1)

    for i in range(9):
        speed = 1000 + 1000 * i
        print(speed)
        start = time.time_ns()
        lev.set_frame_rate(speed)
        lev.levitate(xs,num_loops=1)
        end = time.time_ns()
    print((end-start)/1e9, 'Seconds')






except KeyboardInterrupt:
    print('Stopping')
except Exception as e:
    print(e)
finally:
    print('Finished Moving')
    input()
    lev.disconnect()
