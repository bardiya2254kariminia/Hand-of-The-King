import argparse
import importlib
import random
from os.path import abspath, join, dirname
import sys
import json
import copy

# Add the utils folder to the path
sys.path.append(join(dirname(abspath(__file__)), "utils"))

# Import the utils
import pygraphics
from classes import Card, Player

# Set the path of the file
path = dirname(abspath(__file__))

parser = argparse.ArgumentParser(description="A Game of Thrones: Hand of the King")
parser.add_argument('--player1', metavar='p1', type=str, help="either human or the name of an AI file", default='human')
parser.add_argument('--player2', metavar='p2', type=str, help="either human or the name of an AI file", default='human')
parser.add_argument('-l', '--load', type=str, help="file containing starting board setup (for repeatability)", default=None)
parser.add_argument('-s', '--save', type=str, help="file to save board setup to", default=None)

def make_board():
    '''
    This function creates a random board for the game.

    Returns:
        cards (list): list of Card objects
    '''

    # Load the characters
    with open(join(path, "assets", "characters.json"), 'r') as file:
        characters = json.load(file)

    cards = [] # List to hold the cards

    for i in range(36):
        # Get a random character
        house = random.choice(list(characters.keys()))
        name = random.choice(characters[house])

        # Remove the character from the dictionary
        characters[house].remove(name)
        if len(characters[house]) == 0:
            del characters[house]
        
        card = Card(house, name, i)

        cards.append(card)

    return cards

def save_board(cards, filename='board'):
    '''
    This function saves the board to a file.

    Parameters:
        cards (list): list of Card objects
        filename (str): name of the file to save the board to
    '''

    cards_json = []

    for card in cards:
        card_json = {'house': card.get_house(), 'name': card.get_name(), 'location': card.get_location()}
        cards_json.append(card_json)

    with open(join(path, "boards", filename + ".json"), 'w') as file:
        json.dump(cards_json, file, indent=4)

def load_board(filename='board'):
    '''
    This function loads the board from a file.

    Parameters:
        filename (str): name of the file to load the board from

    Returns:
        cards (list): list of Card objects
    '''

    with open(join(path, "boards", filename + ".json"), 'r') as file:
        cards = json.load(file)
    
    cards = [Card(card['house'], card['name'], card['location']) for card in cards]

    return cards

def find_varys(cards):
    '''
    This function finds the location of Varys on the board.

    Parameters:
        cards (list): list of Card objects

    Returns:
        varys_location (int): location of Varys
    '''

    varys = [card for card in cards if card.get_name() == 'Varys']

    varys_location = varys[0].get_location()

    return varys_location

def get_moves(cards):
    '''
    This function gets the possible moves for the player.

    Parameters:
        cards (list): list of Card objects

    Returns:
        moves (list): list of possible moves
    '''

    # Get the location of Varys
    varys_location = find_varys(cards)

    # Get the row and column of Varys
    varys_row, varys_col = varys_location // 6, varys_location % 6

    moves = []

    # Get the cards in the same row or column as Varys
    for card in cards:
        if card.get_name() == 'Varys':
            continue

        row, col = card.get_location() // 6, card.get_location() % 6

        if row == varys_row or col == varys_col:
            moves.append(card.get_location())

    return moves

def calculate_winner(player1, player2):
    '''
    This function determines the winner of the game.

    Parameters:
        player1 (Player): player 1
        player2 (Player): player 2

    Returns:
        winner (int): 1 if player 1 wins, 2 if player 2 wins
    '''

    player1_banners = player1.get_banners()
    player2_banners = player2.get_banners()

    # Calculate the scores of the players
    player1_score = sum(player1_banners.values())
    player2_score = sum(player2_banners.values())

    if player1_score > player2_score:
        return 1
    
    elif player2_score > player1_score:
        return 2
    
    # If the scores are the same, whoever has the banner of the house with the most cards wins
    else:
        if player1_banners['Stark'] > player2_banners['Stark']:
            return 1
        
        elif player2_banners['Stark'] > player1_banners['Stark']:
            return 2
        
        elif player1_banners['Greyjoy'] > player2_banners['Greyjoy']:
            return 1
        
        elif player2_banners['Greyjoy'] > player1_banners['Greyjoy']:
            return 2
        
        elif player1_banners['Lannister'] > player2_banners['Lannister']:
            return 1
        
        elif player2_banners['Lannister'] > player1_banners['Lannister']:
            return 2
        
        elif player1_banners['Targaryen'] > player2_banners['Targaryen']:
            return 1
        
        elif player2_banners['Targaryen'] > player1_banners['Targaryen']:
            return 2
        
        elif player1_banners['Baratheon'] > player2_banners['Baratheon']:
            return 1
        
        elif player2_banners['Baratheon'] > player1_banners['Baratheon']:
            return 2
        
        elif player1_banners['Tyrell'] > player2_banners['Tyrell']:
            return 1
        
        elif player2_banners['Tyrell'] > player1_banners['Tyrell']:
            return 2
        
        elif player1_banners['Tully'] > player2_banners['Tully']:
            return 1
        
        elif player2_banners['Tully'] > player1_banners['Tully']:
            return 2

def make_move(cards, move, player):
    '''
    This function makes a move for the player.

    Parameters:
        cards (list): list of Card objects
        move (int): location of the card
        player (Player): player making the move
    '''

    # Get the location of Varys
    varys_location = find_varys(cards)

    # Find the row and column of Varys
    varys_row, varys_col = varys_location // 6, varys_location % 6

    # Get the row and column of the move
    move_row, move_col = move // 6, move % 6

    # Find the selected card
    for card in cards:
        if card.get_location() == move:
            selected_card = card
            break
    
    removing_cards = []

    # Find the cards that should be removed
    for i in range(len(cards)):
        if cards[i].get_name() == 'Varys':
            varys_index = i
            continue
        
        # If the card is between Varys and the selected card and has the same house as the selected card
        if varys_row == move_row and varys_col < move_col:
            if cards[i].get_location() // 6 == varys_row and varys_col < cards[i].get_location() % 6 < move_col and cards[i].get_house() == selected_card.get_house():
                removing_cards.append(cards[i])

                # Add the card to the player's cards
                player.add_card(cards[i])
        
        elif varys_row == move_row and varys_col > move_col:
            if cards[i].get_location() // 6 == varys_row and move_col < cards[i].get_location() % 6 < varys_col and cards[i].get_house() == selected_card.get_house():
                removing_cards.append(cards[i])

                # Add the card to the player's cards
                player.add_card(cards[i])
        
        elif varys_col == move_col and varys_row < move_row:
            if cards[i].get_location() % 6 == varys_col and varys_row < cards[i].get_location() // 6 < move_row and cards[i].get_house() == selected_card.get_house():
                removing_cards.append(cards[i])

                # Add the card to the player's cards
                player.add_card(cards[i])
        
        elif varys_col == move_col and varys_row > move_row:
            if cards[i].get_location() % 6 == varys_col and move_row < cards[i].get_location() // 6 < varys_row and cards[i].get_house() == selected_card.get_house():
                removing_cards.append(cards[i])

                # Add the card to the player's cards
                player.add_card(cards[i])
    
    # Add the selected card to the player's cards
    player.add_card(selected_card)

    # Set the location of Varys
    cards[varys_index].set_location(move)
        
    # Remove the cards
    for card in removing_cards:
        cards.remove(card)
    
    # Remove the selected card
    cards.remove(selected_card)

def set_banners(player1, player2):
    '''
    This function sets the banners for the players.

    Parameters:
        player1 (Player): player 1
        player2 (Player): player 2
    '''

    # Get the cards of the players
    player1_cards = player1.get_cards()
    player2_cards = player2.get_cards()

    for house in player1_cards.keys():
        # The player with the more cards of a house gets the banner
        if len(player1_cards[house]) > len(player2_cards[house]):
            player1.get_house_banner(house)
            player2.remove_house_banner(house)
        
        elif len(player2_cards[house]) > len(player1_cards[house]):
            player2.get_house_banner(house)
            player1.remove_house_banner(house)
            
def main(args):
    '''
    This function runs the game.

    Parameters:
        args (Namespace): command line arguments
    '''

    if args.load:
        try:
            # Load the board from the file
            cards = load_board(args.load)
        
        except FileNotFoundError:
            print("File not found. Creating a new board.")
            cards = make_board()
    
    else:
        # Create a new board
        cards = make_board()
    
    if args.save:
        try:
            # Save the board to the file
            save_board(cards, args.save)
        
        except:
            print("Error saving board.")
    
    # Set up the graphics
    board = pygraphics.init_board()

    # Draw the board
    pygraphics.draw_board(board, cards, '0')

    # Show the initial board for 2 seconds
    pygraphics.show_board(2)

    # Check if the players are human or AI
    if args.player1 == 'human':
        player1_agent = None
    
    else:
        # Check if the AI file exists
        try:
            player1_agent = importlib.import_module(args.player1)
        
        except ImportError:
            print("AI file not found.")
            return
        
        if not hasattr(player1_agent, 'get_move'):
            print("AI file does not have the get_move function.")
            return
    
    if args.player2 == 'human':
        player2_agent = None
    
    else:
        # Check if the AI file exists
        try:
            player2_agent = importlib.import_module(args.player2)
        
        except ImportError:
            print("AI file not found.")
            return
        
        if not hasattr(player2_agent, 'get_move'):
            print("AI file does not have the get_move function.")
            return
    
    # Set up the players
    player1 = Player(player1_agent)
    player2 = Player(player2_agent)

    # Set up the turn
    turn = 1 # 1: player 1's turn, 2: player 2's turn

    # Draw the board
    pygraphics.draw_board(board, cards, '1')

    while True:
        # Check the moves for the player
        moves = get_moves(cards)

        # Check if the game is over
        if len(moves) == 0:
            
            # Get the winner of the game
            winner = calculate_winner(player1, player2)
            
            # Display the winner
            pygraphics.display_winner(board, player1, player2, winner)

            # Show the board for 5 seconds
            pygraphics.show_board(5)

            break

        # Get the player's move
        if turn == 1:
            if player1_agent is None:
                move = pygraphics.get_player_move()
            
            else:
                move = player1.get_move(copy.deepcopy(cards))
        
        else:
            if player2_agent is None:
                move = pygraphics.get_player_move()
            
            else:
                move = player2.get_move(copy.deepcopy(cards))
        
        # Check if the move is valid
        if move in moves:
            # Make the move
            make_move(cards, move, player1 if turn == 1 else player2)

            # Set the banners for the players
            set_banners(player1, player2)

            # Change the turn
            turn = 2 if turn == 1 else 1

            # Draw the board
            if turn == 1:
                pygraphics.draw_board(board, cards, '1')
            
            else:
                pygraphics.draw_board(board, cards, '2')
            
            # Show the board for 2 seconds
            pygraphics.show_board(2)

if __name__ == "__main__":
    main(parser.parse_args())