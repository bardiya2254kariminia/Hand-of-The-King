import torch
import torch.nn 
from utils.classes import Card, Player
from training_ai.Networks import Qnetwork
import random
import os
# from main import main
import argparse

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

def representation(full_cards: Card , player1s:Player , player2s:Player, companion_cards:Card):
    map_house = {'Stark': 1, 'Greyjoy': 2, 'Lannister': 3, 'Targaryen': 4, 'Baratheon': 5, 'Tyrell': 6, 'Tully': 7}
    map_companion_cards = {"Jon": 0,"Jaqen": 1,"Gendry": 2,"Melisandre": 3,"Ramsay": 4,"Sandor": 5}
    # making the representation for the cards
    ans = []
    for i in range(len(full_cards)):
        representation = torch.zeros((56))
        
        cards = full_cards[i]
        # print(cards)
        for card in  cards:
            if card.name == "Varys":
                representation[card.location] = -1
                continue
            house_label = map_house[card.get_house()]
            house_location = card.location
            representation[house_location] = house_label
        # making the representation for the companion_cards
        cards = companion_cards[i]
        for card in cards:
            representation[36+map_companion_cards[card]] = 1
        # making the representation for the player1
        player1 = player1s[i]
        for key in map_house.keys():
            representation[41 + map_house[key]] = len(player1.cards[key])
        # making the representation for the player2
        player2 = player2s[i]
        for key in map_house.keys():
            representation[48 + map_house[key]] = len(player2.cards[key])
        ans.append(representation)
    return ans

def mutation(model:torch.nn.Module , eps = 0.01, p_mutation = 0.1):
    ans = model
    if random.random() > p_mutation:
        for name , param in ans.named_parameters():
            param.data = param.data + eps * random.random()
    return ans

def crossover(model1:torch.nn.Module , model2:torch.nn.Module, p_crossover = 0.1):
    if random.random() > p_crossover:
        model1_dict = dict([(name , param.data) for (name , param) in model1.named_parameters()])
        model2_dict = dict([(name , param.data) for (name , param) in model2.named_parameters()])
        child1 = Qnetwork()
        child2 = Qnetwork()
        for name , param in child1.named_parameters():
            param.data = model1_dict[name] + model2_dict[name]

        for name , param in child2.named_parameters():
            param.data = model1_dict[name] + model2_dict[name]
        return child1 , child2
    else:
        return model1,  model2
    
def fittness(model, games = 2):
    parser = argparse.ArgumentParser(description="A Game of Thrones: Hand of the King")
    parser.add_argument('--player1', metavar='p1', type=str, help="either human or an AI file", default='human')
    parser.add_argument('--player2', metavar='p2', type=str, help="either human or an AI file", default='human')
    parser.add_argument('-l', '--load', type=str, help="file containing starting board setup (for repeatability)", default=None)
    parser.add_argument('-s', '--save', type=str, help="file to save board setup to", default=None)
    parser.add_argument('-v', '--video', type=str, help="name of the video file to save", default=None)
    games_won = 0
    for game in range(games):
        winner = main(parser.parse_args())
        if winner ==1:
            games_won +=1
    return games_won

def selection(population_models , primal_model =torch.nn.Module,next_gen_num = 5):
    model_fitness = []
    model_fitness.append((primal_model , fittness(primal_model)))
    for model in population_models:
        model_fitness.append((model , fittness(model)))
    sorted(model_fitness , key=lambda x:x[1])
    selected_models = []
    for i in range(next_gen_num):
        model= model_fitness[i][0]
        selected_models.append(model)
    return selected_models

def save_weights(model:torch.nn.Module, out_path = "."):
    out = os.path.join(out_path , "final_model.pt")
    torch.save(model.state_dict(), out)

def load_weigths(model:torch.nn.Module, input_path= "heuristic_model.pt"):
    """
    loading the model 
    output : loaded model 
    """
    ckpt  =  torch.load(input_path, map_location="cpu")
    model.load_state_dict(ckpt ,  strict=False)
    return  model

