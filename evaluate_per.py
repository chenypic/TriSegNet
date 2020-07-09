import numpy as np
import nibabel as nib
import os
import glob
import pandas as pd
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt


def get_whole_tumor_mask(data):
    return data > 0


def get_tumor_core_mask(data):
    return np.logical_or(np.logical_or(data == 1, data == 3), data == 4)


def get_enhancing_tumor_mask(data):
    return data == 4


def dice_coefficient(truth, prediction):
    return 2 * np.sum(truth * prediction)/(np.sum(truth) + np.sum(prediction))


def evaluate_subject(gt, prediction):
    rows = list()
    header = ("WholeTumor", "TumorCore", "EnhancingTumor")
    masking_functions = (get_whole_tumor_mask, get_tumor_core_mask, get_enhancing_tumor_mask)
    truth = gt.get_data()
    prediction = prediction.get_data()
    rows.append([dice_coefficient(func(truth), func(prediction)) for func in masking_functions])

    results = dict()
    results["WholeTumor"] = rows[0][0]
    results["TumorCore"] = rows[0][1]
    results["EnhancingTumor"] = rows[0][2]
    return results



"""
    rows = list()
    subject_ids = list()
    for case_folder in glob.glob("predict/*"):
        if not os.path.isdir(case_folder):
            continue
        subject_ids.append(os.path.basename(case_folder))
        #truth_file = os.path.join(case_folder, "ori_truth.nii.gz")
        truth_file = os.path.join(case_folder, "truth.nii.gz")
        truth_image = nib.load(truth_file)
        truth = truth_image.get_data()
        #prediction_file = os.path.join(case_folder, "final_prediction.nii.gz")
        prediction_file = os.path.join(case_folder, "prediction.nii.gz")
        prediction_image = nib.load(prediction_file)
        prediction = prediction_image.get_data()
        rows.append([dice_coefficient(func(truth), func(prediction))for func in masking_functions])

    df = pd.DataFrame.from_records(rows, columns=header, index=subject_ids)
    df.to_csv("./predict/brats_scores.csv")

    scores = dict()
    for index, score in enumerate(df.columns):
        values = df.values.T[index]
        scores[score] = values[np.isnan(values) == False]

    plt.boxplot(list(scores.values()), labels=list(scores.keys()))
    plt.ylabel("Dice Coefficient")
    plt.savefig("validation_scores_boxplot.png")
    plt.close()

    if os.path.exists("./training.log"):
        training_df = pd.read_csv("./training.log").set_index('epoch')

        plt.plot(training_df['loss'].values, label='training loss')
        plt.plot(training_df['val_loss'].values, label='validation loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.xlim((0, len(training_df.index)))
        plt.legend(loc='upper right')
        plt.savefig('loss_graph.png')


if __name__ == "__main__":
    main()

"""