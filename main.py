import numpy as np
import pandas as pd
import string
from sklearn.preprocessing import LabelEncoder
from data import NextCharDataset, DataLoader
from lstm import LSTMClassifier
from optim import AdamW
from tqdm.auto import tqdm
from collections import defaultdict
from op import cross_entropy
import ast

def process_cell(cell):
    try:
        parsed_list = ast.literal_eval(cell)
        return '\n'.join(parsed_list)
    except:
        return cell


def clean_lyrics(lyrics):
    lyrics = lyrics.replace('-', ' ')
    lyrics = lyrics.lower()  # Convert to lowercase
    lyrics = lyrics.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    return lyrics


ts_df = pd.read_csv("ts_discography_clean.csv")
ts_df.dropna(subset=['song_lyrics'], inplace=True)
ts_df.drop_duplicates(subset=['song_lyrics'], inplace=True)
ts_df['cleaned_lyrics'] = ts_df['song_lyrics'].apply(process_cell).apply(clean_lyrics)

choices = {'1': 'Taylor Swift', '2': 'Fearless', '3': 'Speak Now', '4': 'Red', '5': '1989', '6':
    'reputation', '7': 'Lover', '8': 'folklore', '9': 'evermore', '10': 'Midnights', '11': 'The Tortured Poets Department'}

album_name = input("Choose album (enter number): "
                   "\n 1: Taylor Swift"
                   "\n 2: Fearless"
                   "\n 3: Speak Now"
                   "\n 4: Red"
                   "\n 5: 1989"
                   "\n 6: reputation"
                   "\n 7: Lover"
                   "\n 8: folklore"
                   "\n 9: evermore"
                   "\n 10: Midnights"
                   "\n 11: The Tortured Poets Department\n")

data = album_lyrics = " ".join(ts_df[ts_df['category'] == choices.get(album_name, 'Taylor Swift')]['cleaned_lyrics'])
print(data)
char_data = np.array(list(data))
encoder = LabelEncoder()
indices_data = encoder.fit_transform(char_data)

vocabulary = encoder.classes_
SEQUENCE_LENGTH = 128
BATCH_SIZE = 32
VOCAB_SIZE = len(vocabulary)
TRAIN_SPLIT = 0.8
LEARNING_RATE = 0.001
SHUFFLE_TRAIN = True

EMBED_SIZE = 256
HIDDEN_SIZE = 256
NUM_LAYERS = 2
NUM_EPOCHS = 5

trainset_size = int(len(indices_data) * TRAIN_SPLIT)
train_data = indices_data[:trainset_size]
test_data = indices_data[trainset_size:]

trainset = NextCharDataset(train_data, SEQUENCE_LENGTH)
testset = NextCharDataset(test_data, SEQUENCE_LENGTH)

trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=SHUFFLE_TRAIN)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

model = LSTMClassifier(EMBED_SIZE, HIDDEN_SIZE, VOCAB_SIZE, NUM_LAYERS)
optimizer = AdamW(params=model.layers, grads=model.grad, lr=LEARNING_RATE)

state = None
train_losses = defaultdict(list)
test_losses = defaultdict(list)

def train():
    for epoch in tqdm(range(NUM_EPOCHS), desc="Epoch"):
        # training loop
        for inputs, targets in (pbar := tqdm(trainloader, leave=False)):
            if SHUFFLE_TRAIN:
                state = None
            probabilities, state, activations = model.forward(inputs, state)

            # cross entropy loss
            loss = cross_entropy(probabilities, targets)
            # accuracy
            accuracy = np.mean(np.argmax(probabilities, axis=-1) == targets)

            # loss gradient w.r.t logits (before softmax)
            gradient = np.copy(probabilities)
            # Subtract 1 from the probabilities of the true classes
            # Since the gradient is p_i - y_i
            gradient[np.arange(targets.shape[0])[:, None],
                     np.arange(targets.shape[1]), targets] -= 1
            # Subtract 1 from the probabilities of the true classes
            gradient /= gradient.shape[0]

            # backpropagate and update
            optimizer.zero_grad()
            model.backward(gradient, activations)
            optimizer.step()

            # log
            pbar.set_postfix({"loss": f"{loss:.5f}",
                              "accuracy": f"{accuracy*100:.2f}"})
            train_losses[epoch].append(loss)

        # testing loop
        loss_sum = 0
        accuracy_sum = 0
        for iter, (inputs, targets) in (pbar := tqdm(enumerate(testloader),
                                                     leave=False)):
            probabilities, state, _ = model.forward(
                inputs, state=None, teacher_forcing=False
            )
            loss = cross_entropy(probabilities, targets)
            accuracy = np.mean(np.argmax(probabilities, axis=-1) == targets)

            loss_sum += loss
            accuracy_sum += accuracy
            pbar.set_postfix(
                {
                    "loss": f"{loss_sum / (iter + 1):.5f}",
                    "accuracy": f"{accuracy_sum / (iter + 1)*100:.2f}",
                }
            )
            test_losses[epoch].append(loss)

def generate(model, prefix: str, length: int):
    inputs = np.array(list(prefix))
    print(inputs)
    inputs = encoder.transform(inputs)
    inputs = inputs[np.newaxis]
    state = None

    probabilities, state, _ = model.forward(
        inputs, state, teacher_forcing=False, generation_length=length
    )
    tokens = np.argmax(probabilities[0, len(prefix) - 1 :], axis=-1)

    output = prefix + "".join(encoder.inverse_transform(tokens))
    return output


train()
while True:
    prefix = input("Enter prefix: ")
    length = input("Enter length: ")
    lengthint = int(length)
    print(generate(model, prefix=prefix, length=lengthint))
