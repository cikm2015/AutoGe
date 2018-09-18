
import csv

import tensorflow as tf
from keras import backend as K



from controller import Controller, StateSpace
#import graphsage.nas_unsupervised_train as manger
import graphsage.nas_unsupervised_one_class_train as manger
# create a shared session between Keras and Tensorflow
policy_sess = tf.Session()
K.set_session(policy_sess)

# NUM_LAYERS = 4  # number of layers of the state space
NUM_LAYERS = 2  # number of layers of the state space
MAX_TRIALS = 1000  # maximum number of models generated

MAX_EPOCHS = 10  # maximum number of epochs to train
CHILD_BATCHSIZE = 128  # batchsize of the child models
EXPLORATION = 0.8  # high exploration for the first 1000 steps
REGULARIZATION = 1e-3  # regularization strength
CONTROLLER_CELLS = 32  # number of cells in RNN controller
EMBEDDING_DIM = 20  # dimension of the embeddings for each state
ACCURACY_BETA = 0.8  # beta value for the moving average of the accuracy
CLIP_REWARDS = 0.0  # clip rewards in the [-0.05, 0.05] range
RESTORE_CONTROLLER = True  # restore controller to continue training

# construct a state space
state_space = StateSpace()

# add states
state_space.add_state(name='aggType', values=[0,1,2,3,4])
state_space.add_state(name='aggType', values=["sigmoid","tanh","relu","linear","softplus","leaky_relu","relu6"])

# print the state space being searched
state_space.print_state_space()

previous_acc = 0.0
total_reward = 0.0

with policy_sess.as_default():    # create the Controller and build the internal policy network
    controller = Controller(policy_sess, NUM_LAYERS, state_space,
                            reg_param=REGULARIZATION,
                            exploration=EXPLORATION,
                            controller_cells=CONTROLLER_CELLS,
                            embedding_dim=EMBEDDING_DIM,
                            restore_controller=RESTORE_CONTROLLER)


# get an initial random state space if controller needs to predict an
# action from the initial state
#随机初始化
# state = state_space.get_random_state_space(NUM_LAYERS)
state = state_space.get_state([3, 'relu', 3, 'linear'])
print("Initial Random State : ", state_space.parse_state_space_list(state))
print()

# clear the previous files
controller.remove_files()
isFirst = True
# train for number of trails
for trial in range(MAX_TRIALS):
    if isFirst:
        actions = state
        isFirst = False
    else:
        with policy_sess.as_default():
            K.set_session(policy_sess)
            actions = controller.get_action(state)  # get an action for the previous state
    # print the action probabilities
    state_space.print_actions(actions)
    print("Predicted actions : ", state_space.parse_state_space_list(actions))

    # build a model, train and get reward and accuracy from the network manager
    #与子模型关联较弱，只需返回reward与accuracy即可；reward为accuracy指数滑动平均后的值
    # if isFirst:
    #     reward, previous_acc = manger.main(action=[1,'relu',1,'linear'])
    #     isFirst = False
    # else:
    reward, previous_acc = manger.main(action=state_space.parse_state_space_list(actions))

    # reward, previous_acc = (1,1)
    print("Rewards : ", reward, "Accuracy : ", previous_acc)

    with policy_sess.as_default():
        K.set_session(policy_sess)

        total_reward += reward
        print("Total reward : ", total_reward)

        # actions and states are equivalent, save the state and reward
        #存储state与reward 进行更新
        state = actions
        controller.store_rollout(state, reward)

        # train the controller on the saved state and the discounted rewards
        loss = controller.train_step()
        print("Trial %d: Controller loss : %0.6f" % (trial + 1, loss))

        # write the results of this trial into a file
        with open('train_history.csv', mode='a+') as f:
            data = [previous_acc, reward]
            data.extend(state_space.parse_state_space_list(state))
            writer = csv.writer(f)
            writer.writerow(data)
    print()

print("Total Reward : ", total_reward)
