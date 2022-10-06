import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from IPython.display import clear_output
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
% config
Completer.use_jedi = False


# Функции для того чтобы мониторить прогресс обучения нейронной сети
def plot_progress(train_losses, train_accs, test_loss, test_accs):
    clear_output(True)

    f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    f.set_figheight(6)
    f.set_figwidth(12)

    ax1.plot(train_losses, label='train loss')
    ax1.plot(test_loss, label='test loss')
    ax1.plot(np.zeros_like(train_losses), '--', label='zero')
    ax1.set_title('Loss')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Batch number')
    ax1.legend()

    ax2.plot(train_accs, label='train accuracy')
    ax2.plot(test_accs, label='test accuracy')
    ax2.plot(np.ones_like(accs), '--', label='100% accuracy')
    ax2.set_title('Accuracy')
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Batch number')
    ax2.legend()

    plt.show()


data = pd.read_csv('vins_train.csv')

# ========== Предобработка ==========

# Учитываем, что последовательности VIN разной длины -
# делаем паддинг до максимальной длины
max_vin_length = max(data['VIN'].apply(lambda x: len(x)))
data['VIN'] = data['VIN'].apply(lambda x: x.ljust(max_vin_length, '-'))

# Создаем словари для всех токенов по каждому из атрибутов
tokens = set(''.join(data['VIN']))
car_brands = set(data['CarBrand'])
car_models = set(data['CarModel'])
colors = set(data['Color'])
n_tokens = len(tokens)
token_index_map = {l: i for i, l in enumerate(tokens)}
car_brands_index_map = {l: i for i, l in enumerate(car_brands)}
car_models_index_map = {l: i for i, l in enumerate(car_models)}
colors_index_map = {l: i for i, l in enumerate(colors)}

# Создаем класс DataSet для преобразования исходных
# строковых данных в тензорные
class VinDataset(Dataset):

    def __init__(self, lines, labels):
        self.lines = lines  # объекты
        self.labels = labels  # ответы

    def __len__(self):
        return len(self.lines)  # количество объектов

    def __getitem__(self, idx):
        x = self.line_to_tensor(self.lines[
                                    idx]).long()  # преобразуем один объект в тензор индексов, тип long()
        y = torch.tensor([
            car_brands_index_map[self.labels[idx][0]],
            car_models_index_map[self.labels[idx][1]],
            colors_index_map[self.labels[idx][2]]
        ]).float()  # ответы на объекте, тип float()
        return x, y

    @staticmethod
    def line_to_tensor(line):
        return torch.tensor([token_index_map[l] for l in line])

# ========== Создание модели ==========

class VinCNN(nn.Module):

    def __init__(self, vocab_size, hidden_size, kernel_size=1,
                 embedding_dim=16):
        super(VinCNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size + 1, embedding_dim)
        self.cnn = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            padding='same',
        )
        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm1d(num_features=hidden_size)
        self.dropout = nn.Dropout(0.6)
        self.linear_car_brands = nn.Linear(hidden_size,
                                           len(car_brands_index_map))
        self.linear_car_models = nn.Linear(hidden_size,
                                           len(car_models_index_map))
        self.linear_colors = nn.Linear(hidden_size, len(colors_index_map))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.embedding(x)  # формируем слой эмбедингов (посимвольно)
        x = self.cnn(x.permute(0, 2, 1))  # применяем одномерную свертку
        x, _ = x.max(dim=-1)  # применяем max-pooling
        x = self.relu(x)  # применяем функцию активации ReLU
        x = self.batchnorm(x)  # проводим нормализацию
        x = self.dropout(x)  # проводим регуляризацию
        # Формируем выходы (используем softmax, т.к. решаем задачу многоклассовой классификации)
        x_car_brands = self.linear_car_brands(x)
        x_car_models = self.linear_car_models(x)
        x_colors = self.linear_colors(x)
        x_car_brands = self.softmax(x_car_brands).squeeze()
        x_car_models = self.softmax(x_car_models).squeeze()
        x_colors = self.softmax(x_colors).squeeze()
        return x_car_brands, x_car_models, x_colors  # 3 многоклассовых выхода

# ========== Обучение ==========

# Разделяем данные на обучение и валидацию
X_train, X_test, y_train, y_test = train_test_split(
    data['VIN'], data[['CarBrand', 'CarModel', 'Color']], test_size=0.05,
)
# Формируем датасеты
train_dataset = VinDataset(lines=X_train.values, labels=y_train[
    ['CarBrand', 'CarModel', 'Color']].values)
test_dataset = VinDataset(lines=X_test.values, labels=y_test[
    ['CarBrand', 'CarModel', 'Color']].values)
# Разделяем датасеты по батчам со случайным перемешиванием
train_vin_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True)
test_vin_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=True)

# Инициализируем модель
model = VinCNN(
    vocab_size=len(token_index_map),
    hidden_size=2048,
    kernel_size=16,
)
# Определяем метод оптимизации
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# Создаем для каждого из параметров свою функцию потерь
# (используем кросс-энтропию, т.к. решаем задачу многоклассовой классификации)
brand_head_loss_fn = nn.CrossEntropyLoss()
model_head_loss_fn = nn.CrossEntropyLoss()
color_head_loss_fn = nn.CrossEntropyLoss()

n_epochs = 5

losses = []
accs = []

test_losses = []
test_accs = []

# Обучаем сеть
for i in range(n_epochs):
    for x_train, y_train in train_vin_dataloader:
        model.train()  # переводим модель в режим обучения (для расчета градиентов)

        brand_pred, model_pred, color_pred = model(x_train)

        brand_head_loss = brand_head_loss_fn(brand_pred, y_train[:, 0].long())
        model_head_loss = model_head_loss_fn(model_pred, y_train[:, 1].long())
        color_head_loss = color_head_loss_fn(color_pred, y_train[:, 2].long())
        # Суммируем потери, чтобы учитывать ошибки по каждому из 3 параметров
        train_total_loss = brand_head_loss + model_head_loss + color_head_loss

        # Подсчитываем метрику (минимальная Accuracy по 3 параметрам)
        train_brand_acc = (torch.argmax(brand_pred, dim=1) == y_train[:,
                                                              0].int()).float().mean()
        train_model_acc = (torch.argmax(model_pred, dim=1) == y_train[:,
                                                              1].int()).float().mean()
        train_color_acc = (torch.argmax(color_pred, dim=1) == y_train[:,
                                                              2].int()).float().mean()
        train_total_acc = min(
            [train_brand_acc, train_model_acc, train_color_acc])

        optimizer.zero_grad()  # обнуляем градиенты с прошлого шага
        train_total_loss.backward()  # расчитываем градиенты на текущем шаге
        optimizer.step()  # обновляем веса

        model.eval()  # переводим  модель в режим оценивания (чтобы не расчитывать градиенты)

        x_test, y_test = next(iter(test_vin_dataloader))

        test_brand_pred, test_model_pred, test_color_pred = model(x_test)

        test_brand_head_loss = brand_head_loss_fn(test_brand_pred,
                                                  y_test[:, 0].long())
        test_model_head_loss = model_head_loss_fn(test_model_pred,
                                                  y_test[:, 1].long())
        test_color_head_loss = color_head_loss_fn(test_color_pred,
                                                  y_test[:, 2].long())
        # Суммируем потери, чтобы учитывать ошибки по каждому из 3 параметров
        test_total_loss = test_brand_head_loss + test_model_head_loss + test_color_head_loss

        test_brand_acc = (torch.argmax(test_brand_pred, dim=1) == y_test[:,
                                                                  0].int()).float().mean()
        test_model_acc = (torch.argmax(test_model_pred, dim=1) == y_test[:,
                                                                  1].int()).float().mean()
        test_color_acc = (torch.argmax(test_color_pred, dim=1) == y_test[:,
                                                                  2].int()).float().mean()
        # Подсчитываем метрику (минимальная Accuracy по 3 параметрам)
        test_total_acc = min([test_brand_acc, test_model_acc, test_color_acc])

        losses.append(train_total_loss.item())
        accs.append(train_total_acc.item())

        test_losses.append(test_total_loss.item())
        test_accs.append(test_total_acc.item())

        plot_progress(losses, accs, test_losses, test_accs)

# ========== Оценивание ==========

# Функция для подсчета метрики на валидации
def get_test_accuraccy(model, dataloader):
    test_accs = []
    for x_test, y_test in dataloader:
        test_brand_pred, test_model_pred, test_color_pred = model(x_test)

        test_brand_acc = (torch.argmax(test_brand_pred, dim=1) == y_test[:,
                                                                  0].int()).float().mean()
        test_model_acc = (torch.argmax(test_model_pred, dim=1) == y_test[:,
                                                                  1].int()).float().mean()
        test_color_acc = (torch.argmax(test_color_pred, dim=1) == y_test[:,
                                                                  2].int()).float().mean()
        test_total_acc = min(
            [test_brand_acc, test_model_acc, test_color_acc]).numpy()
        test_accs.extend(test_total_acc.reshape(-1, 1))
    return np.mean(test_accs)


# get_test_accuraccy(model, test_vin_dataloader)
# Получаем 0.96127546

# ========== Получение результатов на тестовой выборке ==========

df_test = pd.read_csv('vins_test.csv')
df_test['VIN'] = df_test['VIN'].apply(lambda x: x.ljust(max_vin_length, '-'))


class VinTestDataset(Dataset):

    def __init__(self, lines):
        self.lines = lines  # объекты

    def __len__(self):
        return len(self.lines)  # количество объектов

    def __getitem__(self, idx):
        x = self.line_to_tensor(self.lines[
                                    idx]).long()  # преобразуем один объект в тензор индексов, тип long()
        return x

    @staticmethod
    def line_to_tensor(line):
        return torch.tensor([token_index_map[l] for l in line])


test = VinTestDataset(lines=df_test['VIN'].values)
test_dataloader = DataLoader(test_dataset,
                             batch_size=512)  # не забываем убрать shuffle
test_brands_ind = []
test_models_ind = []
test_colors_ind = []

for x_test in test_dataloader:
    model.eval()

    test_brand_pred, test_model_pred, test_color_pred = model(x_test)

    test_brands_batch = torch.argmax(test_brand_pred, dim=1)
    test_models_batch = torch.argmax(test_model_pred, dim=1)
    test_colors_batch = torch.argmax(test_color_pred, dim=1)
    test_brands_ind.extend(test_brands_batch.tolist())
    test_models_ind.extend(test_models_batch.tolist())
    test_colors_ind.extend(test_colors_batch.tolist())

test_brands_ind = np.array(test_brands_ind)
test_models_ind = np.array(test_models_ind)
test_colors_ind = np.array(test_colors_ind)

answers = pd.DataFrame([test_brands_ind, test_models_ind, test_colors_ind])
answers = answers.T
answers.columns = ['CarBrand', 'CarModel', 'Color']


# Функции для получения ключа по значению
def get_brand(val):
    for key, value in car_brands_index_map.items():
        if val == value:
            return key
    return "key doesn't exist"


def get_model(val):
    for key, value in car_models_index_map.items():
        if val == value:
            return key
    return "key doesn't exist"


def get_color(val):
    for key, value in colors_index_map.items():
        if val == value:
            return key
    return "key doesn't exist"


answers['CarBrand'] = answers['CarBrand'].apply(lambda x: get_brand(x))
answers['CarModel'] = answers['CarModel'].apply(lambda x: get_model(x))
answers['Color'] = answers['Color'].apply(lambda x: get_color(x))
answers[['CarBrand', 'CarModel', 'Color']].to_csv('submission.csv',
                                                  index=False)
