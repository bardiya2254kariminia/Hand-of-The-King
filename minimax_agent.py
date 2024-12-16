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
    val, best_move = get_best_move(cards, player1, player2, player=player1, depth=0)
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
            make_move(cards=temp_cards, move=move, player=player)
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
            make_move(cards=temp_cards, move=move, player=player)
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

    lambda_Stark = 0 if Stark1 + Stark2 >= 6 else (Stark1 + Stark2) / 5
    lambda_Greyjoy = 0 if Greyjoy1 + Greyjoy2 >= 5 else (Greyjoy1 + Greyjoy2) / 4
    lambda_Lannister = (
        0 if Lannister1 + Lannister2 >= 4 else (Lannister1 + Lannister2) / 4
    )
    lambda_Targaryen = (
        0 if Targaryen1 + Targaryen2 >= 4 else (Targaryen1 + Targaryen2) / 3
    )
    lambda_Baratheon = (
        0 if Baratheon1 + Baratheon2 >= 3 else (Baratheon1 + Baratheon2) / 3
    )
    lambda_Tyrell = 0 if Tyrell1 + Tyrell2 >= 3 else (Tyrell1 + Tyrell2) / 2
    lambda_Tully = 0 if Tully1 + Tully2 >= 2 else (Tully1 + Tully2) / 1
    return (
        ((Stark1 - Stark2) / 8) * (lambda_Stark)
        + ((Greyjoy1 - Greyjoy2) / 7) * (lambda_Greyjoy)
        + ((Lannister1 - Lannister2) / 6) * (lambda_Lannister)
        + ((Targaryen1 - Targaryen2) / 5) * (lambda_Targaryen)
        + ((Baratheon1 - Baratheon2) / 4) * (lambda_Baratheon)
        + ((Tyrell1 - Tyrell2) / 3) * (lambda_Tyrell)
        + ((Tully1 - Tully2) / 2) * (lambda_Tully)
    )
