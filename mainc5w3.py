from course5w3.translation_with_attention import translation_with_attention_practice
from course5w3.trigger_word_detection import trigger_word_detection_practice

practice = 2

if __name__ == '__main__':
    print("welcome back, this is deep learning course practices")
    if practice == 1:
        translation_with_attention_practice()
    elif practice == 2:
        trigger_word_detection_practice()
    else:
        print("unknown")
