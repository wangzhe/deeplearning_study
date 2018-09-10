from course5w2.operation_word2vec import operation_word2vec_practice
from course5w2.emojify import emojify_practice
from course5w2.emojifyV2 import emojify_practice_v2

practice = 3

if __name__ == '__main__':
    print("welcome back, this is deep learning course practices")
    if practice == 1:
        operation_word2vec_practice()
    elif practice == 2:
        emojify_practice()
    elif practice == 3:
        emojify_practice_v2()
    else:
        print("unknown")
