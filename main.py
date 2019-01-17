from anago.utils import load_data_and_labels
import anago


x_train, y_train = load_data_and_labels('Dataset/English/train.txt')
x_test, y_test = load_data_and_labels('Dataset/English/test.txt')

model = anago.Sequence()
model.fit(x_train, y_train, epochs=1)
model.save('Model/weights.h5', 'Model/params.json', 'Model/preprocessor.pickle')
