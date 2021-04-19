import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import sklearn.model_selection as mod_sel
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, plot_confusion_matrix, \
    plot_precision_recall_curve, average_precision_score
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt


def check_column_data_integrity(data, column_expected_values):
    isConsistent = True
    for val in data:
        isMatchFound = False
        for expected_value in column_expected_values:
            if expected_value == val:
                isMatchFound = True
                break
        if not isMatchFound:
            isConsistent = False
            break
    return isConsistent


def hot_encode(data_frame, attribute_column):
    data_encoded = data_frame.loc[:, attribute_column]
    data_encoded = pd.get_dummies(data_encoded, prefix=attribute_column)

    # removing one column from the encoded columns, (Dummy variable drop)
    # axis=1 means column
    columns = data_encoded.columns
    if len(columns) == 2:
        # only remove the one extra column if it has only <= 2 categories
        data_encoded = data_encoded.drop(labels=columns[len(columns) - 1], axis=1)

    # removing that column from data frame
    data_frame = data_frame.drop(labels=attribute_column, axis=1)
    data_frame = pd.concat([data_frame, data_encoded], axis='columns')

    return data_frame


def convert_column_type_to_numeric(data_frame, continuous_column_list):
    try:
        for column in continuous_column_list:
            data_frame.loc[:, column] = pd.to_numeric(data_frame.loc[:, column])
        isConverted = True
    except:
        isConverted = False
    return isConverted


def get_rows_without_missing_values(dataFrame):
    return dataFrame.loc[
           (dataFrame['A1'] != '?') & (dataFrame['A2'] != '?') & (dataFrame['A4'] != '?') & (dataFrame['A5'] != '?') & (
                   dataFrame['A6'] != '?') & (dataFrame['A7'] != '?') & (dataFrame['A14'] != '?'), :]


def get_rows_with_missing_values(dataFrame):
    # list of rows with atleast one missing attribute
    return dataFrame.loc[
           (dataFrame['A1'] == '?') | (dataFrame['A2'] == '?') | (dataFrame['A4'] == '?') | (dataFrame['A5'] == '?') | (
                   dataFrame['A6'] == '?') | (dataFrame['A7'] == '?') | (dataFrame['A14'] == '?'), :]


def outlier_detection(column):
    outlier = []
    third_std = 3  # if z_score > than 3rd std, then outlier
    mean = np.mean(column)
    std = np.std(column)
    for val in column:
        z_score = (val - mean) / std
        if np.abs(z_score) > third_std:
            outlier.append(val)
    return outlier


def get_model_input_and_output(encoded_data_frame, output_column):
    column_list = encoded_data_frame.columns
    input_column_list = []
    output_column_list = []
    for column in column_list:
        if column.split("_")[0] != output_column:
            input_column_list.append(column)
        else:
            output_column_list.append(column)
    x = encoded_data_frame.loc[:, input_column_list]
    y = encoded_data_frame.loc[:, output_column_list]
    return x, y


def verify_class_distribution(data_frame, query):
    return data_frame[query].value_counts(normalize=True) * 100


def fit_logistic_regression(max_iter, penalty, class_weight, standardised_train_x, train_y):
    # LogisticRegression Model
    # solvers = ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']
    if class_weight is None:
        print("\n<<<<>class_weight=None<>>>>")
    elif class_weight == 'balanced':
        print("\n<<<<>class_weight='balanced'<>>>>")
    logModel = LogisticRegression(max_iter=max_iter, penalty=penalty, class_weight=class_weight)
    print("#Cross validation result, K=10:")
    accuracies = cross_validation(standardised_train_x, train_y.values.ravel(), logModel)
    # print(accuracies)
    print("#10-fold, highest accuracy: ", accuracies.max())
    print("#10-fold, lowest accuracy: ", accuracies.min())
    print("#10-fold, mean accuracy: : ", accuracies.mean())

    # training
    print("#LogisticModel accuracy_score on trainX and trainY:")
    logModel.fit(standardised_train_x, train_y.values.ravel())
    train_y_predicted = logModel.predict(standardised_train_x)
    print(accuracy_score(train_y.values.ravel(), train_y_predicted, normalize=True))

    return logModel


def cross_validation(train_x_set, train_y_set, model_used):
    return cross_val_score(estimator=model_used,
                           X=train_x_set,
                           y=train_y_set,
                           cv=mod_sel.StratifiedKFold(shuffle=False, n_splits=10))


def get_confusion_matrix(test_y, logistic_regression_prediction):
    return confusion_matrix(test_y, logistic_regression_prediction)


def plot_custom_confusion_matrix(classifier, test_x, test_y, isPolynomial):
    matrix = plot_confusion_matrix(classifier, test_x, test_y)
    title = ""
    if isPolynomial:
        if classifier.penalty == 'l2':
            title = "Confusion Matrix - Polynomial Expansion - l2 Regularised"
        elif classifier.penalty == 'none':
            title = "Confusion Matrix - Polynomial Expansion - Unregularised"
    else:
        if classifier.class_weight is None:
            title = "Confusion Matrix class_weight=None"
        elif classifier.class_weight == 'balanced':
            title = "Confusion Matrix class_weight='balanced'"
    matrix.ax_.set_title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.savefig('{}.png'.format(title))
    plt.show()


def plot_custom_precision_recall_curve(classifier, test_x, test_y):
    prc = plot_precision_recall_curve(classifier, test_x, test_y)
    title = ""
    if classifier.class_weight is None:
        title = "Precision-Recall class_weight=None"
    elif classifier.class_weight == 'balanced':
        title = "Precision-Recall class_weight='balanced'"
    prc.ax_.set_title(title)
    plt.savefig("{}.png".format(title))
    plt.show()


def polynomial_expansion(input_x):
    poly = PolynomialFeatures(degree=2)
    poly_matrix = poly.fit_transform(input_x)
    columns = []
    for i in range(0, 946):
        columns.append("P{}".format(i))
    polynomial_df = pd.DataFrame(poly_matrix, columns=columns)

    # remove columns that are entirely 0, sum of column = 0
    col_to_remove = []
    for col in columns:
        if polynomial_df.loc[:, col].sum() == 0:
            col_to_remove.append(col)
    polynomial_df.drop(col_to_remove, inplace=True, axis=1)

    # to drop the co-related features
    corr = polynomial_df.corr().abs()
    upper_tri = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
    # print(upper_tri)
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
    # print(to_drop)

    polynomial_df.drop(to_drop, inplace=True, axis=1)
    return polynomial_df


def plot_original_data_vs_predicted(x, y):
    plt.scatter(x, y, s=10)
    plt.show()


def train_log_reg_polynomial(penalty, polynomial_X_value, Y_value):
    pTrainX, pTestX, pTrainY, pTestY = train_test_split(polynomial_X_value, Y_value,
                                                        test_size=0.25,
                                                        random_state=42,
                                                        stratify=Y)
    pScale = MinMaxScaler()
    pTrainX_min_max = pScale.fit_transform(pTrainX)
    pTestX_min_max = pScale.transform(pTestX)
    polynomialLogModel = fit_logistic_regression(max_iter=5000, penalty=penalty,
                                                 class_weight=None,
                                                 standardised_train_x=pTrainX_min_max,
                                                 train_y=pTrainY)
    print("#Polynomial Expansion LogisticModel accuracy_score on pTestX and pTestY:")
    pTest_y_predicted_un_balanced = polynomialLogModel.predict(pTestX_min_max)
    print(accuracy_score(pTestY.values.ravel(), pTest_y_predicted_un_balanced, normalize=True))
    plot_custom_confusion_matrix(polynomialLogModel, pTestX_min_max, pTestY, True)


if __name__ == '__main__':
    df = pd.read_csv("data/crx.data", sep=",")
    # df.info()
    # assigning column names as per names from crx.names
    df.columns = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16']

    # Check for column values which are not defined in spec, if found then issue
    # get all the rows without any column values missing
    df_complete = get_rows_without_missing_values(df)

    # check if the data values in each column is proper as per crx.names
    if check_column_data_integrity(df_complete.A1, ['b', 'a']):
        print("A1 is valid")
    if check_column_data_integrity(df_complete.A4, ['u', 'y', 'l', 't']):
        print("A4 is valid")
    if check_column_data_integrity(df_complete.A5, ['g', 'p', 'gg']):
        print("A5 is valid")
    if check_column_data_integrity(df_complete.A6,
                                   ['c', 'd', 'cc', 'i', 'j', 'k', 'm', 'r', 'q', 'w', 'x', 'e', 'aa', 'ff']):
        print("A6 is valid")
    if check_column_data_integrity(df_complete.A7, ['v', 'h', 'bb', 'j', 'n', 'z', 'dd', 'ff', 'o']):
        print("A7 is valid")
    if check_column_data_integrity(df_complete.A9, ['t', 'f']):
        print("A9 is valid")
    if check_column_data_integrity(df_complete.A10, ['t', 'f']):
        print("A10 is valid")
    if check_column_data_integrity(df_complete.A12, ['t', 'f']):
        print("A12 is valid")
    if check_column_data_integrity(df_complete.A13, ['g', 'p', 's']):
        print("A13 is valid")
    if check_column_data_integrity(df_complete.A16, ['+', '-']):
        print("A16 is valid")

    # convert our numeric continuous columns to numeric
    if convert_column_type_to_numeric(df_complete, ['A2', 'A3', 'A8', 'A11', 'A14', 'A15']):
        df_complete_encoded = hot_encode(df_complete, 'A1')
        df_complete_encoded = hot_encode(df_complete_encoded, 'A4')
        df_complete_encoded = hot_encode(df_complete_encoded, 'A5')
        df_complete_encoded = hot_encode(df_complete_encoded, 'A6')
        df_complete_encoded = hot_encode(df_complete_encoded, 'A7')
        df_complete_encoded = hot_encode(df_complete_encoded, 'A9')
        df_complete_encoded = hot_encode(df_complete_encoded, 'A10')
        df_complete_encoded = hot_encode(df_complete_encoded, 'A12')
        df_complete_encoded = hot_encode(df_complete_encoded, 'A13')
        df_complete_encoded = hot_encode(df_complete_encoded, 'A16')
        # print(df_complete_encoded.columns)
        # print(df_complete_encoded.info())

        # outliers for this data_set maybe an important feature, some transactions maybe very high
        # encoded values don't have outliers, so only continuous ones; A2, A3, A8, A11, A14, A15
        print("\nOutliers in A2, A3, A8, A11, A14, A15:")
        print(outlier_detection(df_complete_encoded.A2.values))
        print(outlier_detection(df_complete_encoded.A3.values))
        print(outlier_detection(df_complete_encoded.A8.values))
        print(outlier_detection(df_complete_encoded.A11.values))
        print(outlier_detection(df_complete_encoded.A14.values))
        print(outlier_detection(df_complete_encoded.A15.values))

        # getting input(X) and output(Y) matrices
        X, Y = get_model_input_and_output(df_complete_encoded, 'A16')

        # Splitting into training and test set
        trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.25, random_state=42, stratify=Y)

        # we find lot of outliers in our continuous attributes,
        # so we need to use Standardisation
        # make sure you use the same scale for train & test, for train it is scale.fit(trainX),
        # for test it will be scale.transform(testX)
        scale = StandardScaler()
        trainX_standardised = scale.fit_transform(trainX)
        testX_standardised = scale.transform(testX)

        # we got similar class distribution for our train and test set,
        # compared to our df_complete class distribution
        print("\n#Class distribution in df_complete:")
        print(verify_class_distribution(df_complete, 'A16'))
        print("\n#Class distribution in train:")
        print(verify_class_distribution(trainY, 'A16_+'))
        print("\n#Class distribution in test:")
        print(verify_class_distribution(testY, 'A16_+'))

        logModelUnBalanced = fit_logistic_regression(max_iter=500, penalty='none',
                                                     class_weight=None,
                                                     standardised_train_x=trainX_standardised,
                                                     train_y=trainY)

        logModelBalanced = fit_logistic_regression(max_iter=500, penalty='none',
                                                   class_weight='balanced',
                                                   standardised_train_x=trainX_standardised,
                                                   train_y=trainY)

        # test-set best performance on unBalanced
        print("\n#Case of class_weight=None:")
        print("#LogisticModel accuracy_score on testX and testY:")
        test_y_predicted_un_balanced = logModelUnBalanced.predict(testX_standardised)
        print(accuracy_score(testY.values.ravel(), test_y_predicted_un_balanced, normalize=True))
        print("#LogisticModel balanced_accuracy_score on testX and testY:")
        print(balanced_accuracy_score(testY.values.ravel(), test_y_predicted_un_balanced))

        # test-set best performance on balanced
        print("\n#Case of class_weight='balanced':")
        print("#LogisticModel accuracy_score on testX and testY:")
        test_y_predicted_balanced = logModelBalanced.predict(testX_standardised)
        print(accuracy_score(testY.values.ravel(), test_y_predicted_balanced, normalize=True))
        print("#LogisticModel balanced_accuracy_score on testX and testY:")
        print(balanced_accuracy_score(testY.values.ravel(), test_y_predicted_balanced))

        # confusion matrix for unbalanced
        print("\n#Confusion matrix for un-balanced class weight:")
        print(get_confusion_matrix(test_y=testY.values.ravel(),
                                   logistic_regression_prediction=test_y_predicted_un_balanced))
        plot_custom_confusion_matrix(logModelUnBalanced, testX_standardised, testY, False)

        # confusion matrix for balanced
        print("\n#Confusion matrix for balanced class weight:")
        print(get_confusion_matrix(test_y=testY.values.ravel(),
                                   logistic_regression_prediction=test_y_predicted_balanced))
        plot_custom_confusion_matrix(logModelBalanced, testX_standardised, testY, False)

        # plotting the precision-recall curve
        plot_custom_precision_recall_curve(logModelUnBalanced, testX_standardised, testY)
        avg_precision_unbalanced = average_precision_score(testY,
                                                           logModelUnBalanced.decision_function(testX_standardised))
        print("\nAP for unbalanced class weights: {}".format(avg_precision_unbalanced))
        plot_custom_precision_recall_curve(logModelBalanced, testX_standardised, testY)
        avg_precision_balanced = average_precision_score(testY,
                                                         logModelBalanced.decision_function(testX_standardised))
        print("AP for balanced class weights: {}".format(avg_precision_balanced))

        # Polynomial Expansion model creation
        print("\n#Polynomial expansion:")
        polynomial_X = polynomial_expansion(X)
        print("\nRegularized using penalty='l2': ")
        train_log_reg_polynomial(penalty='l2', polynomial_X_value=polynomial_X, Y_value=Y)
        print("\nUnregularised using penalty='none': ")
        train_log_reg_polynomial(penalty='none', polynomial_X_value=polynomial_X, Y_value=Y)
    else:
        print("Numeric conversion error, please make sure all the continuous columns specified are numeric")
