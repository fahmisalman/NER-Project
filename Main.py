from Routes.Route import main
from Model.Train import NER


if __name__ == '__main__':
    # main()
    model = NER()
    model.load_data()
    model.fit()
    # model.save_model()
    # a = open('Dataset/Indo/train.txt')
    # a = a.readlines()
    # for row in a:
    #     # print(len(row))
    #     if '\t' not in row:
    #         print(row)
    #     # b = row.split('\t')
    #     # if len(b) < 2:
    #     #     print(b)
