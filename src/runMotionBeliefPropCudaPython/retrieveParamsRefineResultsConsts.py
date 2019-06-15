# retrieveParamsRefineResultsConsts.py
# Scott Grauer-Gray
# November 28, 2010
# Constants used for retrieving the parameters to refine the results...

# constants with the maximum possible movements in the x and y directions...
MIN_POSS_MOVE_X = -60.0
MIN_POSS_MOVE_Y = -60.0
MAX_POSS_MOVE_X = 60.0
MAX_POSS_MOVE_Y = 60.0

# constant with the 'multiplier' from the maximum movement found to use (for padding...)
MULTIPLIER_MIN_MAX_MOVE = 1.15

# constants for the keys for the parameters
MIN_X_MOVE_PARAM_KEY = 'MIN_X_MOVE'
MIN_Y_MOVE_PARAM_KEY = 'MIN_Y_MOVE'
MAX_X_MOVE_PARAM_KEY = 'MAX_X_MOVE'
MAX_Y_MOVE_PARAM_KEY = 'MAX_Y_MOVE'
START_MOVE_INC_X_PARAM_KEY = 'START_MOVE_INC_X'
START_MOVE_INC_Y_PARAM_KEY = 'START_MOVE_INC_Y'
