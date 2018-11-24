import csv

def create_submission_csv(y_pred):
    '''give predictions and create submission csv file'''
    with open('../twitter-datasets/submission.csv', 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(range(1, len(y_pred) + 1), y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})
