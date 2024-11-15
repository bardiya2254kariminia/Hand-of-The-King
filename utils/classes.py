class Card:
    def __init__(self, house, name, location):
        self.house = house
        self.name = name
        self.house_name = 'Baratheon' if name == 'Joffrey' else house
        self.location = location
    
    def get_house(self):
        return self.house
    
    def get_name(self):
        return self.name
    
    def get_house_name(self):
        return self.house_name
    
    def get_location(self):
        return self.location
    
    def set_location(self, location):
        self.location = location

class Player:
    def __init__(self, agent):
        self.agent = agent
        self.cards = {'Stark': [], 'Greyjoy': [], 'Lannister': [], 'Targaryen': [], 'Baratheon': [], 'Tyrell': [], 'Tully': []}
        self.banners = {'Stark': 0, 'Greyjoy': 0, 'Lannister': 0, 'Targaryen': 0, 'Baratheon': 0, 'Tyrell': 0, 'Tully': 0}

    def get_agent(self):
        return self.agent
    
    def get_cards(self):
        return self.cards
    
    def get_banners(self):
        return self.banners
    
    def add_card(self, card):
        self.cards[card.get_house_name()].append(card)
    
    def get_house_banner(self, house):
        self.banners[house] = 1
    
    def remove_house_banner(self, house):
        self.banners[house] = 0