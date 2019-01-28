from anago.utils import load_data_and_labels
import anago
import os


# x_train, y_train = load_data_and_labels('Dataset/English/train.txt')
# x_valid, y_valid = load_data_and_labels('Dataset/English/valid.txt')
# x_test, y_test = load_data_and_labels('Dataset/English/test.txt')

# model.save('Model/weights.h5', 'Model/params.json', 'Model/preprocessor.pickle')
model = anago.Sequence()
root = os.getcwd()
model = model.load(root + '/Model/weights.h5', root + '/Model/params.json', root + '/Model/preprocessor.pickle')
words = 'President Obama is speaking at the White House.'
print(model.analyze(words))


# model.fit(x_train, y_train, x_valid, y_valid)
# model.score(x_test, y_test)
