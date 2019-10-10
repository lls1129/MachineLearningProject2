# MachineLearningProject2
Linear Regression, Decision Tree, Perceptron, SVM
For the SVM part, svm.py is the one that tries to update all multipliers which violate KKT conditions. This one has high accuracy but is slow due to exhaustive updating. svm1.py is based on Dual Gap to find multipliers, which is not optimized and runs very long time. Svm2.py is based on heuristic choosing and is best method for SVM, but it involves some bug and may sometimes trap into dead loop and cause non-converge (If so, just rerun it). All code files are available in GitHub and bugs will be fixed when I have time. Svm_sklearn.py is the implementation with sklearn package, ONLY this is based on python 3.7, rest of the implementations are based on python 2.7. 
