import random
valid_moves = [1,2,3,4,5]
move = [[1]]
print(random.sample(valid_moves,2))
move.extend(random.sample(valid_moves, 2))
print(move)