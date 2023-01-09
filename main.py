from data_processing import Process
from SMO import SMO
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
import seaborn as sns 

if __name__ == '__main__':
    # Load the data
    data = Process()
    subsets = data.load_data()
    
    # Train the SVM
    linear_time = []
    gaussian_time = []
    training_samples = [subset[0].shape[0] for subset in subsets]
    linear_scores = []
    gaussian_scores = []
    
    # Train the SVM with linear kernel
    print("Training SVM with linear kernel...")
    for i in range(len(subsets)):
        print("Subset {}:".format(i+1))
        X_train, X_test, y_train, y_test = train_test_split(subsets[i][0], subsets[i][1], test_size=0.2)
        start_time = time.time()
        svm = SMO(kernel='linear', random_state=42)
        svm.fit(X_train, y_train)
        end_time = time.time()
        accuracy = svm.score(X_test, y_test)
        print("Accuracy: ", accuracy)
        linear_time.append(end_time - start_time)
        linear_scores.append(accuracy)
    
    # Train the SVM with Gaussian kernel
    print("-" * 50)
    print("Training SVM with Gaussian kernel...")
    for i in range(len(subsets)):
        print("Subset {}:" .format(i+1))
        X_train, X_test, y_train, y_test = train_test_split(subsets[i][0], subsets[i][1], test_size=0.2)
        start_time = time.time()
        svm = SMO(kernel='gaussian', random_state=42)
        svm.fit(X_train, y_train)
        end_time = time.time()
        accuracy = svm.score(X_test, y_test)
        print("Accuracy: ", accuracy)
        gaussian_time.append(end_time - start_time)
        gaussian_scores.append(accuracy)
    
    sns.lineplot(x=training_samples, y=linear_time, label='Linear kernel', marker= 'o')
    sns.lineplot(x=training_samples, y=gaussian_time, label='Gaussian kernel', marker= 'o')
    plt.legend()
    plt.xlabel('Training samples')
    plt.ylabel('Time (s)')
    plt.title('Execution time vs. size of training set')
    plt.savefig('images/evaluations.png')
    plt.show()
    
    
    





