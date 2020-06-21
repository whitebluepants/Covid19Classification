import itertools
import os
import random
import numpy as np
import pandas as pd 
import sklearn
from sklearn import metrics
import tensorflow as tf
import cv2
from matplotlib import pyplot as plt
from prettytable import PrettyTable

def train(load_params=True):
    random.seed(0)
    numpy_seed = 0
    np.random.seed(numpy_seed)
    tensorflow_seed = 0
    tf.random.set_seed(tensorflow_seed)

    # 读数据的预处理
    input_dir = "dataset/"
    positive_file_dirs = [input_dir+"covid/"+filename for filename in os.listdir(input_dir+"covid/")
                          if ("jpeg" in filename or "jpg" in filename or "png" in filename)]
    negative_file_dirs = [input_dir+"normal/"+filename for filename in os.listdir(input_dir+"normal/")
                          if ("jpeg" in filename or "jpg" in filename or "png" in filename)]
    IMG_HEIGHT = 128
    IMG_WIDTH = 128
    CHANNELS = 3
    SIZE = len(positive_file_dirs) + len(negative_file_dirs)


    # 数据读入
    random.shuffle(positive_file_dirs)
    random.shuffle(negative_file_dirs)
    validation_split = 0.1
    test_split = 0.1
    train_split = 1 - validation_split - test_split
    train_size = int(round(train_split*SIZE, 0))
    valid_size = (SIZE - train_size) // 2
    test_size = (SIZE - train_size) - valid_size
    index = train_size // 2
    file_dirs_train = positive_file_dirs[:index] + negative_file_dirs[:index]
    file_dirs_valid = positive_file_dirs[index:index + valid_size // 2] + negative_file_dirs[index:index + valid_size // 2]
    file_dirs_test  = positive_file_dirs[index + valid_size // 2:] + negative_file_dirs[index + valid_size // 2:]
    random.shuffle(file_dirs_train)
    random.shuffle(file_dirs_valid)
    print("Train Size:", train_size)
    print("Validation Size:", valid_size)
    print("Test Size:", test_size)

    X_train = np.zeros(shape=(train_size, IMG_HEIGHT, IMG_WIDTH, CHANNELS), dtype=np.float32)
    y_train = np.zeros(shape=(train_size,), dtype=np.int32)
    for i in range(train_size):
        X_train[i] = cv2.resize(cv2.imread(file_dirs_train[i]), (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_CUBIC)
        if file_dirs_train[i].split("/")[1] == "normal":
            y_train[i] = 0
        else:
            y_train[i] = 1

    X_valid = np.zeros(shape=(valid_size, IMG_HEIGHT, IMG_WIDTH, CHANNELS), dtype=np.float32)
    y_valid = np.zeros(shape=(valid_size,), dtype=np.int32)
    for i in range(valid_size):
        X_valid[i] = cv2.resize(cv2.imread(file_dirs_valid[i]), (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_CUBIC)
        if file_dirs_valid[i].split("/")[1] == "normal":
            y_valid[i] = 0
        else:
            y_valid[i] = 1

    X_test = np.zeros(shape=(test_size, IMG_HEIGHT, IMG_WIDTH, CHANNELS), dtype=np.float32)
    y_test = np.zeros(shape=(test_size,), dtype = np.int32)
    for i in range(test_size):
        X_test[i] = cv2.resize(cv2.imread(file_dirs_test[i]), (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_CUBIC)
        if file_dirs_test[i].split("/")[1] == "normal":
            y_test[i] = 0
        else:
            y_test[i] = 1


    # 模型
    if load_params:
        model = tf.keras.models.load_model('Covid19_Classification.h5')  # 读入保存好的模型

    else:
        resnet_50 = tf.keras.applications.ResNet50(include_top=False, input_shape=(128, 128, 3), pooling='avg')
        model = tf.keras.Sequential((resnet_50, tf.keras.layers.Dense(128, activation='elu'),
                                     tf.keras.layers.Dense(1, activation='sigmoid')))
        # 训练
        BATCH_SIZE = 10
        learning_rate = 0.0001
        epochs = 1
        optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
        model.fit(X_train, y_train, epochs=epochs, validation_data=(X_valid, y_valid))

    return model, X_test, y_test


def test(model, X_test, y_test):
    # 用测试集评估模型
    model.evaluate(x=X_test, y=y_test, batch_size=5)

def predict(model, image):
    X = np.zeros(shape=(1, 128, 128, 3), dtype=np.float32)
    X[0] = image
    return model.predict(X)


# flag=1 ROC曲线
# flag=2 混淆矩阵
# flag=3 各项指标
def show_anything(model, X_test, y_test, flag=1):
    y_pred = model.predict_on_batch(X_test)
    y_pred = y_pred.numpy()
    y_hat = y_pred
    y_hat = np.rint(y_hat)

    accuracy = metrics.accuracy_score(y_test, y_hat)  # accuracy
    recall = metrics.recall_score(y_test, y_hat)  # recall  tp / (tp + fn)
    precision = metrics.precision_score(y_test, y_hat)  # precision tp / (tp + fp)
    f1_score = metrics.f1_score(y_test, y_hat)
    # auc = metrics.roc_auc_score(y_test, y_pred)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
    auc_v2 = metrics.auc(fpr, tpr)

    if flag == 1:
        # 输出ROC曲线
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc_v2)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()
    elif flag == 2:
        # 输出混淆矩阵
        metric = metrics.confusion_matrix(y_test, y_hat)
        tn, fp, fn, tp = metric.ravel()
        print(tn, fp, fn, tp)
        plt.imshow(metric, interpolation='nearest',cmap=plt.cm.Blues)
        plt.title("Confusion_matrix")
        plt.colorbar()
        # plt.xticks([0, 1], ['Negative', 'Positive'])
        # plt.yticks([0, 1], ['True', 'False'])
        plt.axis('off')
        # for i, j in itertools.product(range(metric.shape[0]), range(metric.shape[1])):
        #     plt.text(j, i, metric[i, j], horizontalalignment="center")
        plt.text(0, 0, "true negatives: " + str(metric[0, 0]), horizontalalignment="center")
        plt.text(0, 1, "false positives: " + str(metric[0, 1]), horizontalalignment="center")
        plt.text(1, 0, "false negatives: " + str(metric[1, 0]), horizontalalignment="center")
        plt.text(1, 1, "true positives: " + str(metric[1, 1]), horizontalalignment="center")
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
    elif flag == 3:
        # 输出各项指标的表格
        table = PrettyTable()
        table.add_column('DataType',['Accuracy', 'Recall', 'Precision', 'F1_score', 'AUC'])
        table.add_column('Score',[accuracy.item(), recall.item(), precision.item(), f1_score.item(), auc_v2.item()])
        print(table)
