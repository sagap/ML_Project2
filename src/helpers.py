import csv

def create_submission_csv(y_pred):
    '''give predictions and create submission csv file'''
    with open('../twitter-datasets/submission.csv', 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(range(1, len(y_pred) + 1), y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})

def create_dict_from_csv(csv_path):
    '''create contractions dict from the corresponding csv'''
    reader = csv
    with open(csv_path) as f:
        reader = csv.reader(f)
        result = dict(reader)
        return result
    