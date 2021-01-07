import sys
from gen_walk import generate_walk
from tensorflow.keras.layers import Embedding, SpatialDropout1D, LSTM, Dense
from tensorflow.keras.models import Sequential
from pickle import load
import logging
import coloredlogs
coloredlogs.install()

INPUT_PATH = sys.argv[1]


def load_model(vocab_size):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size,
                        output_dim=16))
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(units=16, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.load_weights('model/model.h5')
    return model


def main():
    logging.info(INPUT_PATH)

    walk = generate_walk(INPUT_PATH)
    logging.info('PSI-walk: ' + walk)

    with open('model/tokenizer.joblib', 'rb') as f:
        tokenizer = load(f)
    vocab_size = len(tokenizer.word_index) + 1
    lstm = load_model(vocab_size)
    logging.info('Loaded LSTM model')

    test = tokenizer.texts_to_sequences([walk])
    prob = lstm.predict(test)[0][0]
    logging.info(f'Predicted probability: {prob}')
    if prob >= 0.5:
        logging.warning('Classification result: MALWARE')
    else:
        logging.info('Classification result: BENIGN')


if __name__ == "__main__":
    main()
