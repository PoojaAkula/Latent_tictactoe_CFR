

from typing import List, cast, Dict

import numpy as np

from labml import experiment
from labml.configs import option
from labml_nn.cfr import History as _History, InfoSet as _InfoSet, Action, Player, CFRConfigs
from labml_nn.cfr.infoset_saver import InfoSetSaver

# Action set at both the players. Each action represents the index of the location on the board. 
#      
#    __0|__3|__6
#    __1|__4|__7                                                                                          
#      2|  5|  8                                                                                                                                                                                   
#                                                                                               
ACTIONS = cast(List[Action], ['0', '1', '2', '3', '4', '5', '6', '7', '8' ])
# We have no chance actions
CHANCES = cast(List[Action], [])
# Player 1 is "0", Player 2 is "1"
PLAYERS = cast(List[Player], [0,1])

# Class to implement information sets 
class InfoSet(_InfoSet):
    """
    ## [Information set](../index.html#InfoSet)
    """

    @staticmethod
    def from_dict(data: Dict[str, any]) -> 'InfoSet':
        """Does not support save/load"""
        pass

    def actions(self) -> List[Action]:
        """
        Return the list of actions. Terminal states are handled by `History` class.
        """
        return ACTIONS

    def __repr__(self):
        """
        Human readable string representation - it gives the probability
        """
        pass

# Class to implement history of the game
class History(_History):

    history: str 

    def __init__(self, history: str = ''):
        # Initializes empty string 
        self.history = history

    def player1_history(self) -> str:
        #Player 1's action history is extracted by returning by all the even positioned characters in history  
        return self.history[::2]

    def player2_history(self) -> str:
        #Player 2's action history is extracted by returning by all the odd positioned characters in history
        return self.history[1::2]

    def win_p1(self):
        # All the win cases. A player will win if he has any of the following characters in his action string. 
        # Ex: If player 0 played '21386', he wins cause he has '1', '3', '6' in his action string. 
        win = ['012', '345', '678', '136', '147', '258', '048', '256']
        for i in win:
            player1_win = all(string in self.player1_history for string in i)
            
            if player1_win == True:
                return True

            else:
                return False

    def win_p2(self):
        # All the win cases. A player will win if he has any of the following characters in his action string. 
        # Ex: If player 1 played '21386', he wins cause he has '1', '3', '6' in his action string. 
        win = ['012', '345', '678', '136', '147', '258', '048', '256']
        for i in win:
            player2_win = all(string in self.player2_history for string in i)
            
            if player2_win == True:
                return True

            else:
                return False
    # Function to check all the terminal cases. 
    def is_terminal(self):
        # To check if game is invalid.
        # The game is invalid if both the players choose the same move. 
        # We can check this by checking if the latest action entered into the history is already 
        # added to the history string 
        # Ex: The following code returns true if h = '345' and the last entry in history is '4'
        if len(self.history) > 2:
            if self.history[-1] in self.history[:-1]:
                return True
        
        # To check if the game resulted in a draw. 
        # If all the locations on the board are filler, i.e., if the len(h) is 9 then its the terminal node. 
        elif len(self.history) == 9:
            return True

        # If any of the player wins the game. 
        elif self.win_p1 == True: 
            return True
        
        elif self.win_p2 == True:
            return True

        else: 
            return False 

    # Utility if player 0 wins the game
    def _terminal_utility_p1(self) -> float:

        if self.win_p1 == True:
            return 1
        else :
            return -1
        
    # Utility at the terminal node
    def terminal_utility(self, i: Player) -> float:
        # Utility when the game draws
        if len(self.history) == 9:
            return 0
        # Utility if player 0 wins
        if i == PLAYERS[0]:
            return self._terminal_utility_p1()
        # Utility if player 1 wins
        else:
            return -1 * self._terminal_utility_p1()

    def is_chance(self):
        pass

    def __add__(self, other: Action):
        return History(self.history + other)

    def player(self) -> Player:
        return cast(Player, len(self.history) % 2)

    def sample_chance(self):
        pass

    def __repr__(self):
        return repr(self.history)


    # Extracting information sets from history 
    def info_set_key(self) -> str:

        # If no one made a move, we simply return the empty string
        if len(self.history) == 0:
            return self.history
        else:
            # Since the current player cannot observe the previous player's move, all we have to do 
            # is return history without the last character in the string. 
            # Ex: Current Player is 1, 
            #     History = '34512'
            #     player 0's move was at location '2' on the board,
            #     since player 1 cannot observe this, 
            #     the information set at player 1 will be returned as '3451'     
            return self.history.replace(self.history[-1], '')

    def new_info_set(self) -> InfoSet:

        return InfoSet(self.info_set_key())

def create_new_history():

    return History()

class Configs(CFRConfigs):
    """
    Configurations extends the CFR configurations class
    """
    pass


@option(Configs.create_new_history)
def _cnh():
    """
    Set the `create_new_history` method
    """
    return create_new_history


def main():
    """
    ### Run the experiment
    """

    # Create an experiment, we only write tracking information to `sqlite` to speed things up.
    # Since the algorithm iterates fast and we track data on each iteration, writing to
    # other destinations such as Tensorboard can be relatively time consuming.
    # SQLite is enough for our analytics.
    experiment.create(name='latent_tic_tac_toe', writers={'sqlite'})
    # Initialize configuration
    conf = Configs()
    # Load configuration
    experiment.configs(conf)
    # Set models for saving
    experiment.add_model_savers({'info_sets': InfoSetSaver(conf.cfr.info_sets)})
    # Start the experiment
    with experiment.start():
        # Start iterating
        conf.cfr.iterate()



if __name__ == '__main__':
    main()
