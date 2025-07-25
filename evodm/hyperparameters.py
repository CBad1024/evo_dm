class hyperparameters:
    '''
    class to store the hyperparemeters that control evoDM
    ...
    Args
    ------
    self: class hyperparameters

    Returns class hyperparameters
    '''

    def __init__(self):
        # Model training settings
        self.REPLAY_MEMORY_SIZE = 10000
        self.MASTER_MEMORY = True
        self.MIN_REPLAY_MEMORY_SIZE = 1000  # TODO Change this to 1000 for real training
        self.MINIBATCH_SIZE = 100
        self.UPDATE_TARGET_EVERY = 310  # every 500 steps, update the target
        self.TRAIN_INPUT = "state_vector"
        self.DELAY = 0

        # Exploration settings
        self.DISCOUNT = 0.99
        self.epsilon = 1  # lowercase because its not a constant
        self.EPSILON_DECAY = 0.95
        self.MIN_EPSILON = 0.001
        self.LEARNING_RATE = 0.0001

        # settings control the evolutionary simulation
        self.NUM_EVOLS = 1  # how many evolutionary steps per time step
        self.SIGMA = 0.5
        self.NORMALIZE_DRUGS = True  # should fitness values for all landscapes be bound between 0 and 1?
        self.AVERAGE_OUTCOMES = False  # should we use the average of infinite evolutionary sims or use a single trajectory?
        self.DENSE = False  # will transition matrix be stored in dense or sparse format?
        # new evolutionary "game" every n steps or n *num_evols total evolutionary movements
        self.RESET_EVERY = 20

        self.NUM_EVOLS = 1
        self.EPISODES = 500
        self.N = 4
        self.RANDOM_START = False
        self.STARTING_GENOTYPE = 0  # default to starting at the wild type genotype
        self.NOISE = False  # should the sensor readings be noisy?
        self.NOISE_MODIFIER = 1  # enable us to increase or decrease the amount of noise in the system
        self.NUM_DRUGS = 4
        self.MIRA = True
        self.TOTAL_RESISTANCE = False
        self.PHENOM = 0
        self.CS = False
        self.DELAY = 0
        self.STANDARD_PRACTICE = False

        # wright-fisher controls
        self.WF = False
        self.POP_SIZE = 10000
        self.GEN_PER_STEP = 1
        self.MUTATION_RATE = 1e-5
        self.HGT_RATE = 1e-5


        #seascapes controls
        self.SEASCAPES = False
        self.NUM_CONC = 10
        self.drug_policy = None


        # define victory conditions for player and pop
        self.PLAYER_WCUTOFF = 0.001
        self.POP_WCUTOFF = 0.999

        # define victory threshold
        self.WIN_THRESHOLD = 1000  # number of player actions before the game is called
        self.WIN_REWARD = 0
        self.AVERAGE_OUTCOMES = False



        # stats settings -
        self.AGGREGATE_STATS_EVERY = 1  # agg every episode


