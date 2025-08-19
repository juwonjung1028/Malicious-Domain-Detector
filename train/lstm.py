from __future__ import annotations

import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dropout, Dense, Activation

from keras.utils import pad_sequences

import data as data


def build_model(
    max_features: int,
    maxlen: int,
    embed_dim: int = 128,
    lstm_units: int = 128,
    dropout: float = 0.5,
):
    """
    Binary classifier: Embedding -> LSTM -> Dropout -> Dense(1, sigmoid)
    Args:
        max_features: size of char-vocab + 1
        maxlen: max sequence length (padding length)
    """
    model = Sequential()
    model.add(Embedding(input_dim=max_features, output_dim=embed_dim, input_shape=(maxlen,)))
    model.add(LSTM(lstm_units))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam")
    return model


def run(max_epoch: int = 10, nfolds: int = 5, batch_size: int = 128):
    """
    Run LSTM training with simple holdout tuning per fold.
    - 데이터는 data.get_data()에서 불러옵니다.
    - 문자 단위 인덱싱 + 패딩으로 입력을 만듭니다.
    """
    indata = data.get_data()

    # 도메인 문자열과 라벨 추출
    X_text = [x[1] for x in indata]
    labels = [x[0] for x in indata]

    # 유효 문자 사전 (1부터 시작)
    valid_chars = {ch: idx + 1 for idx, ch in enumerate(set("".join(X_text)))}
    max_features = len(valid_chars) + 1
    maxlen = max(len(s) for s in X_text)

    # 문자→정수 인덱스, 패딩
    X_idx = [[valid_chars[ch] for ch in s] for s in X_text]
    X = pad_sequences(X_idx, maxlen=maxlen)

    # 라벨을 0/1로 변환
    y = np.array([0 if lab == "benign" else 1 for lab in labels], dtype="int32")

    print("INPUT DATA (First 10 ~ Last 10):")
    print("\n- label:\n", labels[:10], " ~\n", labels[-10:])
    print("\n- score:\n", y[:10].tolist(), " ~\n", y[-10:].tolist())
    print("\n------------------------------------------------------------------------------------")
    print(".\n.\n.")

    final_data = []

    for fold in range(nfolds):
        print(f"Fold {fold + 1}/{nfolds}   -------------------------------------------------------------------------\n")

        # 테스트 세트 20% 분리 (재현성 위해 random_state 고정)
        X_train_temp, X_test, y_train_temp, y_test, _, label_test = train_test_split(
            X, y, labels, test_size=0.2, random_state=42 + fold, stratify=y
        )

        # 배열 보장
        X_train = np.array(X_train_temp)
        y_train = np.array(y_train_temp)

        print("Build model...\n")
        model = build_model(max_features, maxlen)

        print("Train...\n")
        # 작은 holdout으로 AUC 최고 에폭 선택
        X_train_split, X_holdout, y_train_split, y_holdout = train_test_split(
            X_train, y_train, test_size=0.05, random_state=123 + fold, stratify=y_train
        )

        best_iter = -1
        best_auc = 0.0
        out_data = {}

        for ep in range(max_epoch):
            model.fit(X_train_split, y_train_split, batch_size=batch_size, epochs=1, verbose=1)

            t_probs = model.predict(X_holdout, verbose=0).ravel()
            t_auc = roc_auc_score(y_holdout, t_probs)

            print(f"* Epoch {ep + 1}: auc = {t_auc:.6f} (best={best_auc:.6f})\n")

            if t_auc > best_auc:
                best_auc = t_auc
                best_iter = ep

                probs = model.predict(X_test, verbose=0).ravel()
                conf_matrix = sklearn.metrics.confusion_matrix(y_test, probs > 0.5)
                conf_matrix_percent = conf_matrix / conf_matrix.sum(axis=1, keepdims=True) * 100.0

                out_data = {
                    "score": y_test,
                    "labels": label_test,
                    "probs": probs,
                    "epochs": ep + 1,
                    "confusion_matrix": conf_matrix,
                    "confusion_matrix_percent": conf_matrix_percent,
                }

                print(f"* [Fold {fold + 1}/Epoch {ep + 1}] Confusion Matrix:")
                print(conf_matrix, "\n")
            else:
                # 개선 없으면 일찍 중단
                if (ep - best_iter) > 2:
                    break

        final_data.append(out_data)

    return final_data


# # """Train and test LSTM classifier"""
# from __future__ import annotations

# import data as data
# import numpy as np
# from keras_preprocessing.sequence import pad_sequences
# from keras.models import Sequential, Model
# from keras.layers import Input
# from keras.layers import Embedding, LSTM, Dropout, Dense, Activation
# import sklearn
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import roc_auc_score

# from keras.preprocessing.text import Tokenizer
# from keras.utils import to_categorical

# from sklearn.model_selection import StratifiedKFold

# def train_with_cross_validation(X, y, nfolds, lstm, **options):
#     skf = StratifiedKFold(n_splits=nfolds, shuffle=True)

#     for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
#         print(f'Fold {fold+1}/{nfolds}')
#         train_X, val_X = X[train_index], X[val_index]
#         train_y, val_y = y[train_index], y[val_index]
#         model_result = lstm.run(train_X, train_y, val_X, val_y, **options)
#         model_results['lstm'].append(model_result)

#     return model_results

# def build_model(max_features: int, maxlen: int,
#                 embed_dim: int = 128, lstm_units: int = 128, dropout: float = 0.5):
#     """
#     Binary classifier: Embedding -> LSTM -> Dropout -> Dense(1,sigmoid)
#     Args:
#         max_features: size of char-vocab + 1
#         maxlen: max sequence length (padding length)
#     """
#     model = Sequential()
#     model.add(Embedding(input_dim=max_features, output_dim=embed_dim, input_length=maxlen))
#     model.add(LSTM(lstm_units))
#     model.add(Dropout(dropout))
#     model.add(Dense(1))
#     model.add(Activation("sigmoid"))
#     model.compile(loss="binary_crossentropy", optimizer="adam")
#     return model

# def run(max_epoch=10, nfolds=5, batch_size=128):
#     """Run train/test on logistic regression model"""
#     indata = data.get_data()

#     # Extract data and labels
#     X = [x[1] for x in indata]
#     labels = [x[0] for x in indata]

#     # Generate a dictionary of valid characters
#     valid_chars = {x:idx+1 for idx, x in enumerate(set(''.join(X)))}

#     max_features = len(valid_chars) + 1
#     maxlen = np.max([len(x) for x in X])

#     # Convert characters to int and pad
#     X = [[valid_chars[y] for y in x] for x in X]
#     X = pad_sequences(X, maxlen=maxlen)

#     # Convert labels to 0-1
#     y = [0 if x == 'benign' else 1 for x in labels]
#     print("INPUT DATA (First 10 ~ Last 10):")    
#     print("\n- label:\n", labels[:10], " ~\n", labels[-10:])
#     print("\n- score:\n", y[:10], " ~\n", y[-10:])
#     print("\n------------------------------------------------------------------------------------")
#     print(".\n.\n.")

#     final_data = []

#     for fold in range(nfolds):
#         print("Fold %u/%u   -------------------------------------------------------------------------\n" % (fold+1, nfolds))

#         X_train_temp, X_test, y_train_temp, y_test, _, label_test = train_test_split(X, y, labels, test_size=0.2)

#         X_train = np.array(X_train_temp)
#         y_train = np.array(y_train_temp)

#         # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# #         X_train, X_test = np.array(X_train_temp), np.array(X_test_temp)
# #         y_train, y_test = np.array(y_train_temp), np.array(y_test_temp)

#         print ("Build model...\n")
#         model = build_model(max_features, maxlen)

#         print ("Train...\n")
#         X_train, X_holdout, y_train, y_holdout = train_test_split(X_train, y_train, test_size=0.05)
#         best_iter = -1
#         best_auc = 0.0
#         out_data = {}

#         for ep in range(max_epoch):
#             model.fit(X_train, y_train, batch_size=batch_size, epochs=1)

#             t_probs = model.predict(X_holdout)
#             t_auc = roc_auc_score(y_holdout, t_probs)

#             print ("* Epoch %d: auc = %f (best=%f)\n" % (ep+1, t_auc, best_auc))

#             if t_auc > best_auc:
#                 best_auc = t_auc
#                 best_iter = ep
                
#                 probs = model.predict(X_test)
#                 conf_matrix = sklearn.metrics.confusion_matrix(y_test, probs > .5)
#                 conf_matrix_percent = conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis] * 100  # Calculate percentages

#                 out_data = {
#                     'score':y_test,
#                     'labels': label_test,
#                     'probs':probs,
#                     'epochs': ep + 1,
#                     'confusion_matrix': conf_matrix,
#                     'confusion_matrix_percent': conf_matrix_percent
#                 }
                
#                 print ("* [Fold %u/Epoch %d] Confusion Matrix: " % (fold+1,ep+1))
#                 print (conf_matrix)
#                 print ("\n")
#             else:
#                 # No longer improving...break and calc statistics
#                 if (ep-best_iter) > 2:
#                     break

#         final_data.append(out_data)

#     return final_data

# def run(nfolds=5, max_epoch=10):
#     """Run LSTM"""

#     # Load data
#     data, labels = data.get_data()

#     # Convert labels to one-hot encoding
#     label_map = {label: i for i, label in enumerate(set(labels))}
#     num_classes = len(label_map)
#     encoded_labels = [label_map[label] for label in labels]

#     # Convert data to integer sequences
#     tokenizer = Tokenizer(char_level=True)
#     tokenizer.fit_on_texts(data)
#     sequences = tokenizer.texts_to_sequences(data)
#     vocab_size = len(tokenizer.word_index) + 1

#     # Pad sequences
#     max_sequence_length = max(len(seq) for seq in sequences)
#     padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

#     # Convert to numpy arrays
#     X = np.array(padded_sequences)
#     y = np.array(encoded_labels)

#     # Initialize cross-validation
#     skf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=42)
#     fold = 1
#     model_results = []

#     for train_index, test_index in skf.split(X, y):
#         print("Fold:", fold)

#         # Split data into train and test sets
#         X_train, X_test = X[train_index], X[test_index]
#         y_train, y_test = y[train_index], y[test_index]

#         # Convert labels to one-hot encoding
#         y_train_one_hot = to_categorical(y_train, num_classes=num_classes)
#         y_test_one_hot = to_categorical(y_test, num_classes=num_classes)

#         # Define model
#         model = Sequential()
#         model.add(Embedding(vocab_size, 32, input_length=max_sequence_length))
#         model.add(LSTM(64))
#         model.add(Dense(num_classes, activation='softmax'))

#         # Compile model
#         model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#         # Train model
#         history = model.fit(X_train, y_train_one_hot, validation_data=(X_test, y_test_one_hot), epochs=max_epoch)

#         # Evaluate model
#         y_pred = model.predict_classes(X_test)
#         y_pred_prob = model.predict(X_test)
#         confusion_matrix = confusion_matrix(y_test, y_pred)

#         model_results.append({
#             'y': y_test,
#             'labels': y_pred,
#             'probs': y_pred_prob,
#             'epochs': len(history.history['loss']),
#             'confusion_matrix': confusion_matrix
#         })

#         fold += 1

#     return model_results
