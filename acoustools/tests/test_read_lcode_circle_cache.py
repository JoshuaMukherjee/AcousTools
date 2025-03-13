from acoustools.Intepreter import read_lcode 
from acoustools.Export import save_holograms, load_holograms


path = "acoustools/tests/data/lcode/circle200.lcode"

xs = read_lcode(path, ids=(-1,), return_holos =True, print_eval=True, points_per_batch=32)

cache_path = "acoustools/tests/data/lcode/cache/circle200.holo"

save_holograms(xs, cache_path)

xs2 = load_holograms(cache_path)

for x1,x2 in zip(xs,xs2):
    assert (x1 == x2).all()
