import random
from time import sleep
from main import make_move
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


def get_move(cards, player1, player2):
    """
    This function gets the move of the player.

    Parameters:
        cards (list): list of Card objects
        player1 (Player): the player
        player2 (Player): the opponent

    Returns:
        move (int): the move of the player
    """
    # print(player1, player2)
    val, best_move = get_best_move(
        cards, player1, player2, player=player1, depth=0)
    return best_move


def get_best_move(cards, player1, player2, player, depth, max_depth=4):
    """
    getting the best move for 4 depth from now
    approximate d:
    d[4] : 14.6 s
    d[5] : 161 s
    player1 : maximizer
    player2 : minimizer
    """
    if depth > 4:

        return (
            get_huristics(
                cards=cards,
                player=player,
                player1=player1,
                player2=player2,
            ),
            None,
        )

    if player == player1:
        # maximizer player
        ans = -1e8
        best_move = None
        valid_moves = get_valid_moves(cards)
        for move in valid_moves:
            temp_cards = copy.deepcopy(cards)
            make_move(cards=temp_cards, move=move, player=player, other_player=player2)
            h_move, _ = get_best_move(
                cards=temp_cards,
                player1=player1,
                player2=player2,
                player=player2,
                depth=depth + 1,
            )
            del temp_cards
            if ans < h_move:
                ans, best_move = h_move, move
        return ans, best_move
    else:
        # minimizer player
        ans = 1e8
        best_move = None
        valid_moves = get_valid_moves(cards)
        for move in valid_moves:
            temp_cards = copy.deepcopy(cards)
            make_move(cards=temp_cards, move=move, player=player, other_player=player1)
            h_move, _ = get_best_move(
                cards=temp_cards,
                player1=player1,
                player2=player2,
                player=player1,
                depth=depth + 1,
            )
            del temp_cards
            if ans > h_move:
                ans, best_move = h_move, move
        return ans, best_move


def get_huristics(cards, player: Player, player1: Player, player2: Player):
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

    stark_sum = Stark1 + Stark2
    greyjoy_sum = Greyjoy1 + Greyjoy2
    lannister_sum = Lannister1 + Lannister2
    targaryen_sum = Targaryen1 + Targaryen2
    baratheon_sum = Baratheon1 + Baratheon2
    tyrell_sum = Tyrell1 + Tyrell2
    tully_sum = Tully1 + Tully2

    p1score = 0
    p2score = 0

    # tully hue
    if tully_sum == 2:
        if player1.last["Tully"] == 1:
            p1score += 1
        else:
            p2score += 1

    # tyrell hue
    if tyrell_sum == 3:
        if Tyrell1 == 2:
            p1score += 1
        else:
            p2score += 1
    else:
        p1score += Tyrell1 / 2
        p2score += Tyrell2 / 2
    
    # baratheon hue
    if baratheon_sum == 4 or Baratheon1 > 2 or Baratheon2 > 2:
        if Baratheon1 > 2 or (Baratheon1 == 2 and player1.last["Baratheon"] == 1):
            p1score += 1
        else:
            p2score += 1
    elif baratheon_sum < 3:
            p1score += Baratheon1 / 2
            p2score += Baratheon2 / 2

    # targaryen hue
    if targaryen_sum == 5 or Targaryen1 > 2 or Targaryen2 > 2:
        if Targaryen1 > 2:
            p1score += 1
        else:
            p2score += 1
    else:
        p1score += Targaryen1 / 3
        p2score += Targaryen2 / 3
    # lannister hue
    if lannister_sum == 6 or Lannister1 > 3 or Lannister2 > 3:
        if Lannister1 > 3 or (Lannister1 == 3 and player1.last["Lannister"] == 1):
            p1score += 1
        else:
            p2score += 1
    elif lannister_sum < 5:
            p1score += Lannister1 / 4
            p2score += Lannister2 / 4

    # greyjoy hue
    if greyjoy_sum == 7 or Greyjoy1 > 3 or Greyjoy2 > 3:
        if Greyjoy1 > 3:
            p1score += 1
        else:
            p2score += 1
    else:
        p1score += Greyjoy1 / 4
        p2score += Greyjoy2 / 4
    
    # stark hue
    if stark_sum == 8 or Stark1 > 4 or Stark2 > 4:
        if Stark1 > 4 or (Stark1 == 4 and player1.last["Stark"] == 1):
            p1score += 1
        else:
            p2score += 1
    elif stark_sum < 7:
        p1score += Stark1 / 4
        p2score += Stark2 / 4
    

