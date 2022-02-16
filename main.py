import func
from my_model import our_pong_model
import numpy as np
import gym
from gym import wrappers
import time

INPUT_SIZE = 75*80
HD_N = 200
BATCH_SIZE = 10
LR = 1e-3
GAMMA = 0.99
DECAY_RATE = 0.99
TIME_TO_SAVE = 100
FROM_FILE = True
FILE_NAME = None
SHOW = False
CHECKPOINTS_PATH = 'checkpoints/'
REWARD_RECORD = 'Reward_record.csv'
TIME_RECORD = 'Time_record.csv'

env = gym.make('Pong-v0')
env = wrappers.Monitor(env, 'tmp/pong-base', force=True)


def train(model, episode=0):
    observation = env.reset()
    prev_states = None  # used to calculate the different frame
    state_list, hidden_list, action_weight_list, reward_list = [], [], [], []
    accumulated_reward = None  # used to record previous rewards
    reward_sum = 0
    print('start training:\n')
    start_time = time.time()

    while True:
        if SHOW:
            env.render()

        cur_states = func.pre_process_img(observation)
        x = cur_states - prev_states if prev_states is not None else np.zeros(INPUT_SIZE)
        prev_states = cur_states

        # Forward the network and get the action
        prob, hidden = model.forward(x)

        # Here we randomly pick a step
        action = 2 if np.random.uniform() < prob else 3

        state_list.append(x)
        hidden_list.append(hidden)

        y = 1 if action == 2 else 0  # note: This y is our "fake label"

        # Here we store the action weight, which give how much the action is encouraged to be taken
        action_weight_list.append(y - prob)

        # Step in the environment:
        observation, reward, done, info = env.step(action)
        reward_sum += reward
        reward_list.append(reward)

        if done:
            episode += 1

            # Here we stack together the inputs, hidden states, action gradients and rewards
            states_matrix = np.vstack(state_list)
            hidden_matrix = np.vstack(hidden_list)
            action_weight_matrix = np.vstack(action_weight_list)
            reward_matrix = np.vstack(reward_list)
            state_list, hidden_list, action_weight_list, reward_list = [], [], [], []  # empty the arrays

            # Compute the discounted reward and standalize it
            discounted_rewards = func.discount_rewards(reward_matrix, GAMMA)
            discounted_rewards -= np.mean(discounted_rewards)
            discounted_rewards /= np.std(discounted_rewards)

            action_weight_matrix *= discounted_rewards
            model.store_gradient(hidden_matrix, states_matrix, action_weight_matrix)

            if episode % BATCH_SIZE == 0:
                model.back_propagation()

            accumulated_reward = reward_sum if accumulated_reward is None else accumulated_reward * 0.99 + reward_sum * 0.01
            now = time.time()
            gap = now - start_time
            print('episode:', episode)
            print('resetting env. episode reward total was %f. running mean: %f and time spent:%f' % (reward_sum, accumulated_reward, gap))
            with open(REWARD_RECORD, 'a', encoding='UTF-8') as f:
                f.write(str(episode)+','+str(reward_sum)+'\n')

            with open(TIME_RECORD, 'a', encoding='UTF-8') as f:
                f.write(str(episode)+','+str(gap)+'\n')

            reward_sum = 0
            observation = env.reset()  # reset env
            prev_states = None

            if episode % TIME_TO_SAVE == 0:
                model.save_model(CHECKPOINTS_PATH+'save_'+str(int(episode/TIME_TO_SAVE))+'00.p')


def test(model):
    observation = env.reset()
    prev_states = None
    reward_sum = 0

    while True:
        env.render()
        cur_states = func.pre_process_img(observation)
        x = cur_states - prev_states if prev_states is not None else np.zeros(INPUT_SIZE)

        prob, hidden = model.forward(x)
        prev_states = cur_states

        action = 2 if np.random.uniform() < prob else 3

        observation, reward, done, info = env.step(action)
        reward_sum += reward

        if done:
            print('test complete!\n final reward:%f. ' % reward_sum)
            break


def main():
    k = int(input('choose to load checkpoints or initialize one:\n1.choose one\n2.build one randomly\n'))

    if k == 1:
        name = input('enter the ckpt file name:')
        model = our_pong_model(INPUT_SIZE, HD_N, DECAY_RATE, LR, True, name)
    else:
        model = our_pong_model(INPUT_SIZE, HD_N, DECAY_RATE, LR, False)

    k = int(input('choose to do:\n1.train\t2.test\n'))
    if k == 1:
        episode = int(input('episode start counting from:\n'))
        train(model, episode)
    elif k == 2:
        test(model)


if __name__ == '__main__':
    main()
