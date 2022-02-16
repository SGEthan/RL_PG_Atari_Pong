import matplotlib.pyplot as plt
import numpy as np
import csv


def main():
    score_list = []
    time_list = []
    reward_list = []
    accumulated_score = None
    with open('Time_record_f.csv', 'r') as f:
        time_csv = csv.reader(f)
        for record in time_csv:
            time_list.append(float(record[1]))
            
    with open('Reward_record_f.csv', 'r') as f:
        score_csv = csv.reader(f)
        for record in score_csv:
            if accumulated_score is None:
                accumulated_score = float(record[1])
            else:
                accumulated_score = 0.99*accumulated_score+0.01*float(record[1])
            score_list.append(accumulated_score)
            reward_list.append(float(record[1]))

    print(time_list)
    plt.plot(time_list)
    plt.xlabel('episode')
    plt.ylabel('time')
    plt.savefig('time.jpg')
    plt.show()

    print(score_list)
    plt.plot(score_list)
    plt.xlabel('episode')
    plt.ylabel('accumulated_score')
    plt.savefig('score.jpg')
    plt.show()

    print(reward_list)
    plt.plot(reward_list)
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.savefig('reward.jpg')
    plt.show()


if __name__ == '__main__':
    main()
