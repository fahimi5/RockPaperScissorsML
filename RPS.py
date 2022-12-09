# This will be the script that handles the main game of RPS
import random
import numpy as np
import pandas as pd
import time
import tensorflow as tf
from tensorflow import keras
#from keras import models
from keras.models import load_model

debug = False

# Load Models
#fc_network_name = 'fully_trained_model_fcn_3.h5'
fc_network_name = 'Large_model_fcn.h5'

if debug:
    print("Loading FCN...")
fc_network = load_model(fc_network_name)
if debug:
    print("Done!")

# ask for player name and how many rounds they want to play
player_id = input("What is your name?\n")
rounds_tot = int(input("How many rounds would you like to play?\n"))

# Create game logic
outcomes = {'0':{'0':0,'1':-1,'2':1,'-1':-2},
            '1':{'0':1,'1':0,'2':-1,'-1':-2},
            '2':{'0':-1,'1':1,'2':0,'-1':-2},
            '-1':{'0':-2,'1':-2,'2':-2,'-1':-2}}

#move that beats a given move
what_beats_this = {'0':1,
                   '1':2,
                   '2':0}

#the name of each move
num_to_move = {'0':'rock',
               '1':'paper',
               '2':'scissors'}

# move that loses to a given move
what_loses_this = {'0':2,
                   '1':0,
                   '2':1}

# outcome string to display upon win/loss/tie
outcome_text = {'0': '\nYou Tied. :| \n',
                '1': '\nYou Won! :)))\n',
                '-1': '\nYou Lose... :(\n'}

#columns of dataframe
columns = ['user','round_num','player_choice','cpu_choice','outcome','model_used','model_choices','model_outcomes','model_scores']

# ASCII Arts for rock, paper, and scissors modified from Veronica Karlsson: https://devdojo.com/kmhmubin/build-a-python3-rock-paper-scissor-game-using-ascii-art
# original art
#rock_art = '''  
#    _______
#---'   ____)  
#      (_____)  
#      (_____)  
#      (____)
#---.__(___)  
#'''

#paper_art = '''  
#    _______
#---'   ____)____  
#          ______)  
#          _______)  
#         _______)
#---.__________)  
#'''

#scissors_art = '''  
#    _______
#---'   ____)____  
#          ______)  
#       __________)  
#      (____)
#---.__(___)  
#'''  

rock_vs_rock_art = '''
 YOUR CHOICE:                CPU CHOICE:
    ROCK                        ROCK
    _______                  _______
---'   ____)                (____   '---
      (_____)      VS      (_____)
      (_____)              (_____)
      (____)                (____)
---.__(___)                  (___)__.---

                  TIE!
'''

rock_vs_paper_art = '''
 YOUR CHOICE:                CPU CHOICE:
    ROCK                        PAPER
    _______                  _______
---'   ____)            ____(____   '---
      (_____)      VS  (______
      (_____)         (_______
      (____)           (_______
---.__(___)              (__________.---

                CPU WINS!
'''

rock_vs_scissors_art = '''
 YOUR CHOICE:                CPU CHOICE:
    ROCK                      SCISSORS
    _______                  _______
---'   ____)            ____(____   '---
      (_____)      VS  (______
      (_____)         (__________
      (____)                (____)
---.__(___)                  (___)__.---

                YOU WIN!
'''

paper_vs_paper_art = '''
 YOUR CHOICE:                CPU CHOICE:
    PAPER                       PAPER
    _______                  _______
---'   ____)____        ____(____   '---
          ______)  VS  (______
          _______)    (_______
         _______)      (_______
---.__________)          (__________.---

                  TIE!
'''

paper_vs_rock_art = '''
 YOUR CHOICE:                CPU CHOICE:
    PAPER                       ROCK
    _______                  _______
---'   ____)____            (____   '---
          ______)  VS      (_____)
          _______)         (_____)
         _______)           (____)
---.__________)              (___)__.---

                YOU WIN!
'''

paper_vs_scissors_art = '''
 YOUR CHOICE:                CPU CHOICE:
    PAPER                     SCISSORS
    _______                  _______
---'   ____)____        ____(____   '---
          ______)  VS  (______
          _______)    (__________
         _______)           (____)
---.__________)              (___)__.---

                CPU WINS!
'''

scissors_vs_scissors_art = '''
 YOUR CHOICE:                CPU CHOICE:
   SCISSORS                   SCISSORS
    _______                  _______
---'   ____)____        ____(____   '---
          ______)  VS  (______
       __________)    (__________
      (____)                (____)
---.__(___)                  (___)__.---

                  TIE!
'''

scissors_vs_rock_art = '''
 YOUR CHOICE:                CPU CHOICE:
   SCISSORS                     ROCK
    _______                  _______
---'   ____)____            (____   '---
          ______)  VS      (_____)
       __________)         (_____)
      (____)                (____)
---.__(___)                  (___)__.---

                CPU WINS!
'''

scissors_vs_paper_art = '''
 YOUR CHOICE:                CPU CHOICE:
   SCISSORS                     PAPER
    _______                  _______
---'   ____)____        ____(____   '---
          ______)  VS  (______
       __________)    (_______
      (____)           (_______
---.__(___)              (__________.---

                YOU WIN!
'''

# to be referenced during each round to display for player
outcome_art = {'0':{'0':rock_vs_rock_art,'1':rock_vs_paper_art,'2':rock_vs_scissors_art},
               '1':{'0':paper_vs_rock_art,'1':paper_vs_paper_art,'2':paper_vs_scissors_art},
               '2':{'0':scissors_vs_rock_art,'1':scissors_vs_paper_art,'2':scissors_vs_scissors_art}}

# model 0 - simple model based on rough probability of human choice
def model_0():
    model_prob = random.random()
    model_pred = 1
    if model_prob < 0.5: # 50% chance to throw paper
        model_pred = 1
    elif model_prob > .8: # 20% chance to throw scissors
        model_pred = 2
    else:
        model_pred = 0 # 30% chance to throw rock
    return model_pred

# model 1 - simple model that plays what would lose to the opponent's last move

def model_1():
    model_pred = -1 # assigning this a value for rounds that it will be unused
    if Round > 0: # guess what would lsoe the previous round
        model_pred = what_loses_this[str(history['player_choice'][Round - 1])]
        
    return model_pred

# model 2 - simple model that plays what would beat the opponent's last move

def model_2():
    model_pred = -1 # assigning this a value for rounds that it will be unused
    if Round > 0: # guess what would lsoe the previous round
        model_pred = what_beats_this[str(history['player_choice'][Round - 1])]
        
    return model_pred

# model 3 - simple model that plays what would beat the opponent
# based on the round robin pattern (0->1->2->0 or 2->1->0->2)

def model_3():
    model_pred = -1 # assigning this a value for rounds that it will be unused
    if Round == 1: # guess what would beat the previous round
        model_pred = what_beats_this[str(history['player_choice'][Round - 1])]
    if Round > 1: # we want two rounds to have been played before we can see the pattern
        previous_choice = history['player_choice'][Round - 1]
        previous_previous_choice = history['player_choice'][Round - 2]
        change_in_choice = previous_choice - previous_previous_choice
        
        model_player_pred = previous_choice + change_in_choice # predict what they will play
        
        # make sure prediction is within bounds [0,2]
        if model_player_pred > 2:
            model_player_pred -= 3
        elif model_player_pred < 0:
            model_player_pred += 3
        
        # assign prediction of what will beat them
        model_pred = what_beats_this[str(model_player_pred)]
        
    return model_pred

# model 4 - simple model that tries to guess if the player is alternating between choices, i.e. 0->1->0->1->0

def model_4():
    model_pred = -1 # assigning this a value for rounds that it will be unused
    if Round > 1: # two rounds must have been played before we can pick up this pattern - similar to model_1
        previous_choice = history['player_choice'][Round - 1]
        previous_previous_choice = history['player_choice'][Round - 2]
        change_in_choice = previous_choice - previous_previous_choice
        # positive change in choice means next alternate is a negative change in choice
        if change_in_choice > 0:
            sign = -1
        elif change_in_choice < 0:
            sign = 1
        else:
            sign = 0
            
        model_player_pred = previous_choice + (sign * np.abs(change_in_choice))
        
        # make sure prediction is within bounds [0,2]
        if model_player_pred < 0:
            model_player_pred = 2
        
        # assign prediction of what will beat them
        model_pred = what_beats_this[str(model_player_pred)]
        
    return model_pred

# model 5 - simple model that plays what would beat the opponents most played move in past 5 moves

def model_5():
    model_pred = -1 # assigning this a value for rounds that it will be unused
    if Round > 5:
        # tally up most used move in past 5 rounds
        choice_tally = [0,0,0]
        for i in range(Round-6,Round-1):
            choice_tally[history['player_choice'][i]] += 1
        model_pred = what_beats_this[str(np.argmax(choice_tally))] # choose what would beat the most common move
    return model_pred

# model 6 - a complex model trained on rps data that guesses the next move based on the previous 7 moves. Fully Connected Network

def model_6():
    model_pred = -1 # assigning this a value for rounds that it will be unused
    if Round > 6:
        last_7 = []
        for i in range(1,8):
            last_7.append(history['player_choice'][Round - i]) # create list of last 7 moves
        print(len(last_7))
        input_7 = [last_7]
        model_probs = fc_network.predict(input_7)
        model_pred = what_beats_this[str(np.argmax(model_probs))] # choose what would beat the most likely move
    return model_pred

def ensembler():
    round_score = []
    for i in range(len(history['model_outcomes'][0])):
        model_sqr_sum = 1
        model_outcome_sum = 0
        for j in range(len(history['model_outcomes'])):
            if history['model_outcomes'][j][i] == -2:
                pass
            else:
                model_sqr_sum += (j+1)**2
                model_outcome_sum += history['model_outcomes'][j][i] *((j+1)**2)
        round_score.append(model_outcome_sum / model_sqr_sum)
    return round_score

history = pd.DataFrame(columns=columns) # create history df
Round = 0
wins = 0
losses = 0
ties = 0
while Round < rounds_tot:
    # call models and create choice list
    choice_list = [model_0(),model_1(),model_2(),model_3(),model_4(),model_5(),model_6()]
    if Round == 0:
        selected_model = 0
        round_score = [0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    else:
        round_score = ensembler()
        selected_model = np.argmax(round_score)
        while choice_list[selected_model] == -1:
            selected_model -= 1
    selected_choice = choice_list[selected_model]
    if debug:
        print(round_score)
    # keep player from going forward if input is invalid
    invalid_input = True
    
    print(f"\nBegin round {Round+1}")
    
    while invalid_input:
        #have player select choice
        player_choice = input("Please select Rock (1), Paper (2), or Scissors (3): ")

        # check if player choice is valid
        if player_choice == 'quit':
            invalid_input = False
        elif player_choice == '1' or player_choice == '2' or player_choice == '3':
            player_choice = int(player_choice) - 1
            invalid_input = False
        else:
            print("That was not a valid input, please try again. If you wish to quit, type 'quit'.")
    # end game if player wants out
    if player_choice == 'quit':
        print("Play again soon!")
        break
    # determine game outcome
    outcome = outcomes[str(player_choice)][str(selected_choice)]
    
    # create for calculating W/L record later
    if outcome == 1:
        wins += 1
    elif outcome == -1:
        losses += 1
    else:
        ties += 1
    
    # determine model outcomes
    model_outcomes = [outcomes[str(x)][str(player_choice)] for x in choice_list]
    
    # make important printouts for player
    if debug:
        print(selected_choice)
    #print(f"\nYou threw out {num_to_move[str(player_choice)]}!\n")
    #print(f"\nRPS-Bot threw out {num_to_move[str(selected_choice)]}!\n")
    #print(outcome_text[str(outcome)])
    print(outcome_art[str(player_choice)][str(selected_choice)])
    
    # save important data from the round
    history = history.append({'user':player_id,
                              'round_num':Round,
                              'player_choice':player_choice,
                              'cpu_choice':selected_choice,
                              'outcome':outcome,
                              'model_used':selected_model,
                              'model_choices':choice_list,
                              'model_outcomes':model_outcomes,
                              'model_scores':round_score}, 
                             ignore_index=True)
    if debug:
        print(history.tail())
    
    # near end of game apply logic to allow player to continue playing or not
    if Round == rounds_tot - 1:
        Round_Reup = True
        Is_Numeric = True
        while Round_Reup:
            response = input("\nYour game is over, would you like to play a few more rounds? (y/n): ")
            if response == 'y':
                while Is_Numeric:
                    response = input("\nGreat! How many more rounds?\n")
                    if response.isnumeric():
                        rounds_tot += int(response)
                        Round_Reup = False
                        Is_Numeric = False
                    else:
                        print("\nPlease type a number.")
            elif response == 'n':
                winpct = 100 * wins/(wins+losses)
                print("\nOkay! I hope you enjoyed your game :)")
                print(f"\nYou won {wins} of your games, \ntied {ties} of your games, \nand lost {losses} of your games.\nYour win percentage is {round(winpct,3)}%")
                Round_Reup = False
            else:
                print("\nThat was not a valid response, please type 'y' or 'n'.")
        
    Round += 1
history.to_csv(f'game_history_{player_id}_{round(time.time())}.csv')

