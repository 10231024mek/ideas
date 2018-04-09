from tools import *

#Size of the test-set 
M=10
#Last sample to be tested is NUMMAX=21 (see tools) 

print("Series predictor has been chosen to be predict_next_naif(df,1) to speed up the comparison with the Full-Panel-Predictor")
print("=======================================================================================================================")
print("")
print("MAPE for test set of "+str(M)+" panel(s) - Naif-Panel-Predictor (Window size = 1) ")
print("-----------------------------------------------------------------------------------------------------------------------")
print(test_accuracy(M, predict_naif,1))
print("")


print("MAPE for test set of "+str(M)+" panel(s) - Naif-Panel-Predictor (Window size = 2) ")
print("-----------------------------------------------------------------------------------------------------------------------")
print(test_accuracy(M, predict_naif,2))
print("")

print("MAPE for test set of "+str(M)+" panel(s) - Naif-Panel-Predictor (Window size = 3) ")
print("-----------------------------------------------------------------------------------------------------------------------")
print(test_accuracy(M, predict_naif,3))
print("")

print("MAPE for test set of "+str(M)+" panel(s) - Full-Panel-Predictor")
print("-----------------------------------------------------------------------------------------------------------------------")
print(test_accuracy(M, predict_full))
print("")


print("MAPE for test set of "+str(M)+" panel(s) - Full-Panel-Adjusted-Predictor")
print("-----------------------------------------------------------------------------------------------------------------------")
print(test_accuracy(M, predict_full_adjust))
print("")
