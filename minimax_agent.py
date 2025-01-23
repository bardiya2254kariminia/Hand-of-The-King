import random
from time import sleep
from main import make_move, make_companion_move
import copy
from utils.classes import *


def find_varys(cards):
    """
    This function finds the location of Varys on the board.

    Parameters:
        cards (list): list of Card objects

    Returns:
        varys_location (int): location of Varys
    """

    varys = [card for card in cards if card.get_name() == "Varys"]

    varys_location = varys[0].get_location()

    return varys_location


def get_valid_moves(cards):
    """
    This function gets the possible moves for the player.

    Parameters:
        cards (list): list of Card objects

    Returns:
        moves (list): list of possible moves
    """

    # Get the location of Varys
    varys_location = find_varys(cards)

    # Get the row and column of Varys
    varys_row, varys_col = varys_location // 6, varys_location % 6

    moves = []

    # Get the cards in the same row or column as Varys
    for card in cards:
        if card.get_name() == "Varys":
            continue

        row, col = card.get_location() // 6, card.get_location() % 6

        if row == varys_row or col == varys_col:
            moves.append(card.get_location())

    return moves


def get_move(cards, player1, player2 , companion_cards ,choose_companion):
    """
    This function gets the move of the player.

    Parameters:
        cards (list): list of Card objects
        player1 (Player): the player
        player2 (Player): the opponent

    Returns:
        move (int): the move of the player
    """
    print(f"{choose_companion=}")
    num_cards = len(cards)
    max_depth = 2
    val, best_move = get_best_move(
        cards, player1, player2, player1,companion_cards, choose_companion, depth=0,  max_depth=max_depth)
    return best_move


def get_best_move(cards, player1, player2, player, companion_cards, choose_companion,depth, max_depth):
    """
    getting the best move for 4 depth from now
    you have to change this the same way you did for the get_best_move for the phase1 but this time you also have the make_move_companion
    whichyou have to use it when choose_companion is set to True
    """
    print(companion_cards)
    # sleep(10)
    def not_choose_companion(cards, player1, player2, player,companion_cards, choose_companion,depth, max_depth):      
        if depth > max_depth:
            return (
                get_huristics(
                    cards=cards,
                    player=player,
                    player1=player1,
                    player2=player2,
                    ended=False,
                ),
                None,
            )
        if choose_companion:
            # companion = get_best_companion(cards, player1, player2, player, temp_companion_cards)
            for companion in companion_cards:
                temp_companion_cards = copy.deepcopy(companion_cards)
                temp_cards = copy.deepcopy(cards)
                temp_player1 = copy.deepcopy(player1)
                temp_player2 = copy.deepcopy(player2)
                # must find the move for the companion to get the best possible move
                make_companion_move(cards , companion_cards , [companion] , player)

        if player == player1:
            # maximizer player
            def minimax_player1():
                ans = -1e8
                best_move = None
                valid_moves = get_valid_moves(cards)
                if len(valid_moves) == 0:
                    return get_huristics(
                        cards=cards,
                        player=player,
                        player1=player1,
                        player2=player2,
                        ended=True,
                    ), None
                for move in valid_moves:
                    temp_cards = copy.deepcopy(cards)
                    temp_player1 = copy.deepcopy(player1)
                    temp_player2 = copy.deepcopy(player2)
                    make_move(cards=temp_cards, move=move,
                            player=temp_player1)
                    h_move, _ = get_best_move(
                        cards=temp_cards,
                        player1=temp_player1,
                        player2=temp_player2,
                        player=temp_player2,
                        depth=depth + 1,
                        max_depth=max_depth,
                    )
                    del temp_cards
                    del temp_player1
                    del temp_player2
                    if ans < h_move:
                        ans, best_move = h_move, move
                return ans, best_move
            
            
            return minimax_player1()
        else:
            # minimizer player
            def minimax_player2():
                ans = 1e8
                best_move = None
                valid_moves = get_valid_moves(cards)
                if len(valid_moves) == 0:
                    return get_huristics(
                        cards=cards,
                        player=player,
                        player1=player1,
                        player2=player2,
                        ended=True,
                    ), None
                for move in valid_moves:
                    temp_cards = copy.deepcopy(cards)
                    temp_player1 = copy.deepcopy(player1)
                    temp_player2 = copy.deepcopy(player2)
                    make_move(cards=temp_cards, move=move,
                            player=temp_player2)
                    h_move, _ = get_best_move(
                        cards=temp_cards,
                        player1=temp_player1,
                        player2=temp_player2,
                        player=temp_player1,
                        depth=depth + 1,
                        max_depth=max_depth,
                    )
                    del temp_cards
                    del temp_player1
                    del temp_player2
                    if ans > h_move:
                        ans, best_move = h_move, move
                return ans, best_move
            return minimax_player2()
    def choose_comp(cards, player1, player2, player, choose_companion,depth, max_depth):
        # TODO
        pass
    
    # if choose_companion:
    #     return choose_comp(cards, player1, player2, player, choose_companion,depth, max_depth)



def get_huristics(cards, player: Player, player1: Player, player2: Player, ended):
    """
    finding the huristics for the given situation and the player
    """
    # {'Stark': [8], 'Greyjoy': [7], 'Lannister': [6], 'Targaryen': [5], 'Baratheon': [4], 'Tyrell': [3], 'Tully': [2]}
    # for player 1
    
    Stark1 = len(player1.cards["Stark"])
    Greyjoy1 = len(player1.cards["Greyjoy"])
    Lannister1 = len(player1.cards["Lannister"])
    Targaryen1 = len(player1.cards["Targaryen"])
    Baratheon1 = len(player1.cards["Baratheon"])
    Tyrell1 = len(player1.cards["Tyrell"])
    Tully1 = len(player1.cards["Tully"])
    # for player 2
    Stark2 = len(player2.cards["Stark"])
    Greyjoy2 = len(player2.cards["Greyjoy"])
    Lannister2 = len(player2.cards["Lannister"])
    Targaryen2 = len(player2.cards["Targaryen"])
    Baratheon2 = len(player2.cards["Baratheon"])
    Tyrell2 = len(player2.cards["Tyrell"])
    Tully2 = len(player2.cards["Tully"])
    return (Stark1 - Stark2 +
            Greyjoy1 - Greyjoy2 + 
            Lannister1 - Lannister2 + 
            Targaryen1 - Targaryen2 + 
            Baratheon1 - Baratheon2 + 
            Tyrell1 - Tyrell2 + 
            Tully1 - Tully2
    )
    stark_sum = Stark1 + Stark2
    greyjoy_sum = Greyjoy1 + Greyjoy2
    lannister_sum = Lannister1 + Lannister2
    targaryen_sum = Targaryen1 + Targaryen2
    baratheon_sum = Baratheon1 + Baratheon2
    tyrell_sum = Tyrell1 + Tyrell2
    tully_sum = Tully1 + Tully2

    p1score = 0
    p2score = 0
    win_points = 10
    diif_mull = 1.2

    if ended == True:
        if Stark1 > Stark2 or (Stark1 == Stark2 and player1.last["Stark"] == 1):
            p1score += win_points
        elif Stark2 > Stark1 or (Stark2 == Stark2 and player2.last["Stark"] == 1):
            p2score += win_points
        if Greyjoy1 > Greyjoy2 or (Greyjoy1 == Greyjoy2 and player1.last["Greyjoy"] == 1):
            p1score += win_points
        elif Greyjoy2 > Greyjoy1 or (Greyjoy2 == Greyjoy1 and player2.last["Greyjoy"] == 1):
            p2score += win_points
        if Lannister1 > Lannister2 or (Lannister1 == Lannister2 and player1.last["Lannister"] == 1):
            p1score += win_points
        elif Lannister2 > Lannister1 or (Lannister2 == Lannister1 and player2.last["Lannister"] == 1):
            p2score += win_points
        if Targaryen1 > Targaryen2 or (Targaryen1 == Targaryen2 and player1.last["Targaryen"] == 1):
            p1score += win_points
        elif Targaryen2 > Targaryen1 or (Targaryen2 == Targaryen1 and player2.last["Targaryen"] == 1):
            p2score += win_points
        if Baratheon1 > Baratheon2 or (Baratheon1 == Baratheon2 and player1.last["Baratheon"] == 1):
            p1score += win_points
        elif Baratheon2 > Baratheon1 or (Baratheon2 == Baratheon1 and player2.last["Baratheon"] == 1):
            p2score += win_points
        if Tyrell1 > Tyrell2 or (Tyrell1 == Tyrell2 and player1.last["Tyrell"] == 1):
            p1score += win_points
        elif Tyrell2 > Tyrell1 or (Tyrell2 == Tyrell1 and player2.last["Tyrell"] == 1):
            p2score += win_points
        if Tully1 > Tully2 or (Tully1 == Tully2 and player1.last["Tully"] == 1):
            p1score += win_points
        elif Tully2 > Tully1 or (Tully2 == Tully1 and player2.last["Tully"] == 1):
            p2score += win_points

        if p1score == p2score and player1.last["Stark"] == 1:
            p1score += 1
        elif p1score == p2score and player2.last["Stark"] == 1:
            p2score += 1
        # if p1score > p2score:
        #     print(p1score - p2score)
        if p1score > p2score:
            return 80
        else:
            return -80

    def f(num):
        return (num * (num + 1)) / 2

    # tully hue
    if tully_sum == 2:
        if player1.last["Tully"] == 1:
            p1score += win_points
        else:
            p2score += win_points

    # tyrell hue
    if tyrell_sum == 3:
        if Tyrell1 == 2:
            p1score += win_points
        else:
            p2score += win_points
    else:
        p1score += max(0, diif_mull * (f(Tyrell1) * 2) - f(Tyrell2) * 2)
        p2score += max(0, diif_mull * (f(Tyrell2) * 2) - f(Tyrell1) * 2)

    # baratheon hue
    if baratheon_sum == 4 or Baratheon1 > 2 or Baratheon2 > 2:
        if Baratheon1 > 2 or (Baratheon1 == 2 and player1.last["Baratheon"] == 1):
            p1score += win_points
        else:
            p2score += win_points
    elif baratheon_sum < 3:
        p1score += max(0, diif_mull * (f(Baratheon1)) - f(Baratheon2))
        p2score += max(0, diif_mull * (f(Baratheon2)) - f(Baratheon1))

    # targaryen hue
    if targaryen_sum == 5 or Targaryen1 > 2 or Targaryen2 > 2:
        if Targaryen1 > 2:
            p1score += win_points
        else:
            p2score += win_points
    else:
        p1score += max(0, diif_mull * (f(Targaryen1) / 3) - f(Targaryen2) / 3)
        p2score += max(0, diif_mull * (f(Targaryen2) / 3) - f(Targaryen1) / 3)
    # lannister hue
    if lannister_sum == 6 or Lannister1 > 3 or Lannister2 > 3:
        if Lannister1 > 3 or (Lannister1 == 3 and player1.last["Lannister"] == 1):
            p1score += win_points
        else:
            p2score += win_points
    elif lannister_sum < 5:
        p1score += max(0, diif_mull * (f(Lannister1) / 4) - f(Lannister2) / 4)
        p2score += max(0, diif_mull * (f(Lannister2) / 4) - Lannister1 / 4)

    # greyjoy hue
    if greyjoy_sum == 7 or Greyjoy1 > 3 or Greyjoy2 > 3:
        if Greyjoy1 > 3:
            p1score += win_points
        else:
            p2score += win_points
    else:
        p1score += max(0, diif_mull * (f(Greyjoy1) / 4) - f(Greyjoy2) / 4)
        p2score += max(0, diif_mull * (f(Greyjoy2) / 4) - f(Greyjoy1) / 4)

    # stark hue
    if stark_sum == 8 or Stark1 > 4 or Stark2 > 4:
        if Stark1 > 4 or (Stark1 == 4 and player1.last["Stark"] == 1):
            p1score += win_points * 1.001
        else:
            p2score += win_points * 1.001
    elif stark_sum < 7:
        p1score += max(0, diif_mull * (f(Stark1) / 4) - f(Stark2) / 4)
        p2score += max(0, diif_mull * (f(Stark2) / 4) - f(Stark1) / 4)
    # if p1score > p2score:
    #     print(p1score - p2score)
    return p1score - p2score