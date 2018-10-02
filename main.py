# from course5w1.assignement1 import course5w1_assignement1_practice
from tqdm import tqdm

from course5w1.dinosaurus_island import dinosaurus_island_practice
# from course5w1.writing_like_shakespeare import writing_like_shakespeare_practice
# from course5w1.jazz_solo_lstm import jazz_solo_lstm_practice
# from course5w2.operation_word_vec import operation_word2vec_practice
import numpy as np

practice = 8

if __name__ == '__main__':
    print("welcome back, this is deep learning course practices")
    if practice == 1:
        x1 = 1
        # course5w1_assignement1_practice()
    elif practice == 2:
        x1 = 1
        # dinosaurus_island_practice()
    elif practice == 3:
        x1 = 1
        # writing_like_shakespeare_practice()
    elif practice == 4:
        x1 = 1
        # jazz_solo_lstm_practice()
    else:
        a = np.array([[1], [2], [3]])
        b = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
        c = b.T * a
        print(c)
        print("unknown")
        # for i in tqdm(range(1000000000)):
        #     a = i
