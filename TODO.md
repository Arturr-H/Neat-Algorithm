#### Shared RNG between networks in a species
Imagine a species of 6 networks playing a game. If the species somehow manages to evolve
the "perfect" network, which would ace the game its being trained on, but somehow also manages
to get really unlucky in the RNG, compared to a horribly performing network who happened to get
good RNG. The second, more lucky network would then be selected causing the species to choose
the wrong networks to survive. 
Therefore, each game for each iteration should look identical to all networks in the same species.
That will give better results for evaluation.

#### (DONE ?) Better fitness averaging. 
Say a network has been performing horribly the past 24 rounds, and happens to evolve the exact
solution this round. Because we take an average of the past 25 evaluated scores as the fitness
of a network, not much will change and the network risks being removed. Therefore we should make
more previous scores more worth compared to scores longer back in time. EMWA (exponentially
weighted moving average)

#### (DONE) Display maximum instead of average in fitness graph
Because when replacing a bad performing species with a good one, we get constant "misleading"
spikes in the fitness graph.

#### (DONE) Needs topology update 
Only update topology if some changes to topology has previously occured 

#### (DONE) Extra input node for each network which is set to 1.0
This node can be used as a bias, and the network can itself choose the value of the bias (weight
connecting the bias to a node) and which nodes that have biases

#### (DONE) Evolution setting to turn of input -> output auto conneciton in initialization
Will be helpful for certain scenarios where some inputs are "bad" or misleading

#### Evolution setting to start with hidden neurons
Maybe making a setting to e.g start with n amount of input neurons, for weight 
easier connection might help
