import random


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
    print(player1, player2)
    best_move = get_best_move(cards, player1, player2)
    return best_move


def get_best_move(cards, player1, player2, max_depth=4):
    """
    getting the best move for 4 depth from now
    approximate d:
    d[4] : 14.6 s
    d[5] : 161 s
    """
    valid_moves = get_valid_moves(cards)
    for moves in valid_moves:
        print(moves)
        # TODO


def get_huristics(cards, player, player1, player2):
    """
    finding the huristics for the given situation and the player
    """
    # TODO
    pass
