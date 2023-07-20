with open("./temp/cald_5w_add_1w_result.txt", "r") as f:
    cald = f.readlines()

with open("./temp/cald_5w_add_1w_score.txt", "r") as f:
    cald_score = f.readlines()
    cald_score = [x.split(' ')[0] + "\n" for x in cald_score]



with open("./temp/aldd_exp_aldd_3_9_result.txt", "r") as f:
    aldd = f.readlines()


with open("./temp/aldd_result.txt", "r") as f:
    aldd_origin = f.readlines()


with open("./temp/random_5w_add_1w_result.txt", "r") as f:
    random_result = f.readlines()


cald = set(cald)
aldd = set(aldd)
aldd_origin = set(aldd_origin)
random_result = set(random_result)

u = cald & aldd
cu = cald & random_result
au = aldd & random_result

print(len(u), len(cu), len(au))

aa = aldd & aldd_origin
print(len(aa))
u = cald & aldd_origin
print(len(u))

cald_score = cald_score[:10000]
cald_score = set(cald_score)
u = cald & cald_score
print(len(u))
u = aldd & cald_score
print(len(u))