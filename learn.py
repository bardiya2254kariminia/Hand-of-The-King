import copy 
from itertools import combinations
valid_moves = [1,2,3,4,5,6,7,8,9,10]
companion_cards = ["1","2","3","4"]
move = ["1"]
for valid_move  in list(combinations(valid_moves, 2)):
    print(f"{valid_move=}")
    for companion_choice in list(companion_cards):
        temp_move = copy.deepcopy(move)
        temp_move.extend([valid_move[0] ,valid_move[1] ,companion_choice])
        print(f"{temp_move=}")
        # temp_companion_cards = copy.deepcopy(companion_cards)
        # temp_cards = copy.deepcopy(cards)
        # temp_player = copy.deepcopy(player)
        # temp_player1 = copy.deepcopy(player1)
        # temp_player2 = copy.deepcopy(player2)
        # ans , best_move = make_move_for_companion(temp_cards , temp_player, temp_player1,
        #         temp_player2, temp_companion_cards, temp_move , ans , best_move)