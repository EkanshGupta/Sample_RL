###### LOS and Euclidean distance
#Uses the full matrix - not upper triangular matrix
#Warning! - Errors are introduced to training data intentionally for testing. Remove before using. Search for intentionalErrorsBinary - not done yet
from __future__ import print_function

import numpy as np
##Tensorflow compatibility - do not use version 2
import tensorflow as tf

import random
from random import shuffle


from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

###### Parameters

n_clients = 21  # or 6, 21, 41
n_AP_locations = 31

#Using new dataset with Location - this is the latest data
#--------------------------------------------

#exp = 'data/withLocation/data/data-sta-no/c21/'
exp = 'data/withLocation/data_with_errors/data-obs-map/map2/'


#--------------------------------------------


#If using the new dataset from 100*100 STAs
#------------------------------------------
#For 40% obstacle density
# if n_clients == 21:
#     exp = 'data/obs40percent/c21/'
# elif n_clients == 11:
#     exp = 'data/obs40percent/c11/'
# elif n_clients == 6:
#     exp = 'data/obs40percent/c6/'
#------------------------------------------

# #For Different obstacle densities - only 21 STAs
# #------------------------------------------

#exp = 'data/data-obs-densityAndMap/data-obs-den/20percent/'



# #------------------------------------------


# #For the older datasets
# #------------------------------------------
# if n_clients == 21:
#     exp = 'data/c41-om1/'    #Has been replaced with data for 21 clients - new dataset - correlated - obstacle density = 43
# # elif n_clients == 21:
# #     exp = 'data/c21-om2/'
# elif n_clients == 5:
#     exp = 'data/c6-om1/'
# elif n_clients == 6:
#     exp = 'data/c6-om1/'    #Old dataset with higher obstacle density
# # elif n_clients == 6:
# #     exp = 'data/c6-obs11-3380/'    #New dataset with reduced obstacle density

# #------------------------------------------
flag_plot = True
flag_dist = False    # True if using distance matrix
flag_31pts = True

flag_los = True
flag_STA_Location = True
flag_euclDist = False
flag_innerProd = False
flag_AP_location_vector = False

flag_limitTimeSteps = False    #Use when only a certain number of data points are required
timeStepsToTrain = 1000         #Has effect only when flag_limitTimeSteps is True

learning_rate_init = 0.15    #was 0.1
learning_rate_iter = 1000
learning_rate_decay = 0.9

training_epochs = 50000    #was 1 million
batch_size = 256  #was 256
display_train_step = 100
display_test_step = 1000
train_split = 0.7

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 128 # 2nd layer number of neurons
n_hidden_3 = 64 # 3rd layer number of neurons
n_hidden_4 = 256
n_hidden_5 = 64
# n_hidden_6 = 64
# n_hidden_7 = 64
# n_hidden_6 = 4
# n_hidden_7 = 128
# n_hidden_8 = 64
# n_hidden_9 = 32


n_features =((n_clients - 1) * 31 )
if flag_los:
    n_features +=n_clients*n_clients
#if flag_dist:
#    n_features += ((n_clients - 1) * 31 )+ (n_clients*n_clients - n_clients)/2
if flag_innerProd:
    n_features += n_clients - 1
if flag_euclDist:
    n_features += n_clients - 1
if flag_AP_location_vector:
    n_features += n_AP_locations
if flag_STA_Location:
    n_features += n_clients*3

n_output = 31 if flag_31pts else 1
loc_idx = None if flag_31pts else 0

print("Number of input Features: " + str(n_features))

## Importing Data

los_sta_sta = np.load(exp+'los_sta_sta.npy')  # data_points x n_clients x n_clients
los_ap_sta = np.load(exp+'los_ap_sta.npy').transpose(0,2,1)  # data_points x n_clients x possible_AP_pos
#dis_sta_sta = np.load(exp+'dis_sta_sta.npy')  # data_points x n_clients x n_clients
#dis_ap_sta = np.load(exp+'dis_ap_sta.npy').transpose(0,2,1)  # data_points x n_clients x possible_AP_pos
sta_loc = np.load(exp + 'loc_sta.npy')

if flag_limitTimeSteps:
    total_data_points = timeStepsToTrain
    los_sta_sta = los_sta_sta[0:timeStepsToTrain ,:,:]
    los_ap_sta = los_ap_sta[0:timeStepsToTrain  , : , :]
else:
    total_data_points = los_sta_sta.shape[0]


print(total_data_points)

x = np.sum(los_ap_sta)*1.0/los_ap_sta.size
pos_weight = (1-x)/x
print(pos_weight)

rnd_indices = np.random.rand(total_data_points) < train_split

train_losSS = los_sta_sta[rnd_indices,:,:]
train_losAS = los_ap_sta[rnd_indices,:,:]
#train_disSS = dis_sta_sta[rnd_indices,:,:]
#train_disAS = dis_ap_sta[rnd_indices,:,:]

train_sta_loc = sta_loc[rnd_indices,:,:]


test_losSS = los_sta_sta[~rnd_indices,:,:]
test_losAS = los_ap_sta[~rnd_indices,:,:]
#test_disSS = dis_sta_sta[~rnd_indices,:,:]
#test_disAS = dis_ap_sta[~rnd_indices,:,:]
test_sta_loc = sta_loc[~rnd_indices,:,:]

# norm_mean_ss = np.mean(train_disSS)
# norm_mean_as = np.mean(train_disAS)
# norm_std_ss = np.std(train_disSS)
# norm_std_as = np.std(train_disAS)
if flag_STA_Location:
    norm_mean_locX = np.mean(train_sta_loc[:,:,0])
    norm_mean_locY = np.mean(train_sta_loc[:,:,1])
    norm_mean_locZ = np.mean(train_sta_loc[:,:,2])
    norm_std_locX = np.std(train_sta_loc[:,:,0])
    norm_std_locY = np.std(train_sta_loc[:,:,1])
    norm_std_locZ = np.std(train_sta_loc[:,:,2])

total_train_points = train_losSS.shape[0]
total_test_points = test_losSS.shape[0]
print(total_train_points)
print(total_test_points)


##### Building the NN Graph

# tf Graph input
X = tf.placeholder("float", [None, n_features])
Y = tf.placeholder("float", [None, n_output])
learning_rate = tf.Variable(learning_rate_init, trainable=False)

initializer = tf.contrib.layers.xavier_initializer()
h0 = tf.layers.dense(X, n_hidden_1, activation=tf.nn.relu, kernel_initializer=initializer)

#h0 = tf.nn.dropout(h0, 0.8)
h1 = tf.layers.dense(h0, n_hidden_2, activation=tf.nn.relu, kernel_initializer=initializer)
#h1 = tf.nn.dropout(h1, 0.8)
h2 = tf.layers.dense(h1, n_hidden_3, activation=tf.nn.relu, kernel_initializer=initializer)
#h2 = tf.nn.dropout(h2, 0.8)
#h3 = tf.layers.dense(h2, n_hidden_4, activation=tf.nn.relu, kernel_initializer=initializer)
# h3 = tf.nn.dropout(h3, 0.8)
#h4 = tf.layers.dense(h3, n_hidden_5, activation=tf.nn.relu, kernel_initializer=initializer)
# h5 = tf.layers.dense(h4, n_hidden_6, activation=tf.nn.relu, kernel_initializer=initializer)
#h6 = tf.layers.dense(h5, n_hidden_7, activation=tf.nn.relu, kernel_initializer=initializer)
# h7 = tf.layers.dense(h6, n_hidden_8, activation=tf.nn.relu, kernel_initializer=initializer)
# h8 = tf.layers.dense(h7, n_hidden_9, activation=tf.nn.relu, kernel_initializer=initializer)
out = tf.layers.dense(h2, n_output, activation=None)

cross_entropy = tf.nn.weighted_cross_entropy_with_logits(targets=Y, logits=out, pos_weight=pos_weight)
cost = tf.reduce_mean(tf.reduce_mean(cross_entropy, axis=1)) # reduce_mean vs. reduce_sum
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
train_op = optimizer.minimize(cost)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

predicted = tf.nn.sigmoid(out)
pred_labels = tf.round(predicted)    #Rounds to nearest integer
correct_pred = tf.equal(pred_labels, Y)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# confusion_matrix = tf.math.confusion_matrix(labels=tf.argmax(Y,1), predictions=tf.argmax(tf.round(predicted),1), num_classes=2)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

new_learning_rate = tf.placeholder(tf.float32, shape=[])
learning_rate_update = tf.assign(learning_rate, new_learning_rate)

def train_next_batch(batch_size):
    # Select samples of batch_size from training set
    batch_index = random.sample(range(total_train_points), batch_size)
    losSS = train_losSS[batch_index,:,:]
    losAS = train_losAS[batch_index,:,:]
    #disSS = train_disSS[batch_index,:,:]
    #disAS = train_disAS[batch_index,:,:]
    if flag_STA_Location:
        sta_loc_current = train_sta_loc[batch_index,:,:]

    # Random permutations for data augmentation
    idx_shuffle = list(range(losSS.shape[1]))    #shape[1] corresponds to each STA
    #print(idx_shuffle)
    for idx in range(batch_size):
        shuffle(idx_shuffle)    #shuffles the array
        tmp = losSS[idx,idx_shuffle,:]    #Shuffles between STAs within one set of a batch - row wise
        tmp = tmp[:,idx_shuffle]          #shuffles column-wise for data consistency
        losSS[idx,:,:] = tmp
        #tmp2 = disSS[idx,idx_shuffle,:]    #repeating same for disSS
        #tmp2 = tmp2[:,idx_shuffle]
        #disSS[idx,:,:] = tmp2
        losAS[idx,:,:] = losAS[idx,idx_shuffle,:]
        #disAS[idx,:,:] = disAS[idx,idx_shuffle,:]
        if flag_STA_Location:
            sta_loc_current[idx, : :] = sta_loc_current[idx,idx_shuffle,:]


    #r,c = np.triu_indices(losSS.shape[1],1)
    #Defining euclidean distance and inner products here
    #Euclidean distances are measured between the target STA and all other STAs

    if flag_euclDist or flag_innerProd:
        X_EuclDist = np.zeros((batch_size,n_clients - 1))
        #X_InnerProd =np.zeros((batch_size,n_clients - 1))
        for idx in range(batch_size):
            for temp1 in range(n_clients - 1):
                X_EuclDist[idx, temp1] = np.sqrt(np.sum(np.square(losSS[idx,-1,:] - losSS[idx,temp1,:])))
                #X_InnerProd[idx, temp1] = np.sum(losSS[idx,-1,:] * losSS[idx,temp1,:])
        X_EuclDist = X_EuclDist/np.sqrt(n_clients)
    #print("Eucl dist shape: "+ str(X_EuclDist.shape))
    #normalizing the location
    if flag_STA_Location:
        X_locX = (sta_loc_current[:,:,0] - norm_mean_locX)/norm_std_locX
        X_locY = (sta_loc_current[:,:,1] - norm_mean_locY)/norm_std_locZ
        X_locZ = (sta_loc_current[:,:,2] - norm_mean_locZ)/norm_std_locZ





    #normalizing to uniform distribution between 0 and 1
    # max eucl distance will be sqrt(5), min = 0

    #print("Eucl dist: "+str(X_EuclDist))
    #max inner product will be 5, min = 0
    #X_InnerProd = X_InnerProd/n_clients

    #For the full matrix

    X1 = losSS.reshape((losSS.shape[0],(losSS.shape[1])*(losSS.shape[2])))

    #For upper triangular matrix
#    X1 = losSS[:,r,c]


    X2 = losAS[:,:-1,].reshape((losAS.shape[0],(losAS.shape[1]-1)*(losAS.shape[2])))    #leaving out the last client

    #X3 = disSS[:,r,c]
    #X3 = (X3 - norm_mean_ss)/norm_std_ss    #making it a distribution with unit mean and variance
    #X4 = disAS[:,:-1,].reshape((disAS.shape[0],(disAS.shape[1]-1)*(disAS.shape[2])))
    #X4 = (X4 - norm_mean_as)/norm_std_as
    #print(X1.shape)
    #print(X2.shape)
    X1 = 2 * X1 - 1
    X2 = 2 * X2 - 1
    #XA = np.concatenate((X1,X2), axis=1)
    #XA = 2*XA - 1    #making it bipolar = -1 and +1
    #XB = np.concatenate((X3,X4), axis=1)
    #print (XA.shape)
#     print("X1 shape: "+ str(X1.shape))
#     print("X2 shape: "+ str(X2.shape))

    X = X2
    #print(X.shape)
    if flag_los:
        X= X1 if (X.size==0) else np.concatenate((X, X1), axis =1)
    if flag_dist:
        X = XB if (X.size==0) else  np.concatenate((X,XB), axis=1)
    if flag_euclDist:
        X =  X_EuclDist if (X.size==0) else np.concatenate((X,X_EuclDist), axis=1)
    if flag_innerProd:
        X = X_InnerProd if (X.size==0) else np.concatenate((X,X_InnerProd), axis=1)
    if flag_AP_location_vector:
        tempVector1 = np.arange(0,31)/30
        tempVector2 = np.tile(tempVector1, (batch_size,1))
        X = np.concatenate((X, tempVector2 ), axis = 1)
    if flag_STA_Location:
        X = np.concatenate((X, X_locX, X_locY, X_locZ), axis = 1)
    if flag_31pts:
        Y = losAS[:,-1,:].reshape((batch_size,31))
    else:
        Y = losAS[:,-1,loc_idx].reshape((batch_size,1))
#    print("X shape: "+ str(X.shape))

    return X, Y

def test_next_batch():
    # Select samples of batch_size from test set
#     batch_size = total_test_points
#     batch_index = random.sample(xrange(total_test_points), batch_size)
    losSS = test_losSS[:,:,:]
    losAS = test_losAS[:,:,:]
    #disSS = test_disSS[:,:,:]
    #disAS = test_disAS[:,:,:]
    if flag_STA_Location:
        sta_loc_current = test_sta_loc[:,:,:]

    if flag_STA_Location:
        X_locX = (sta_loc_current[:,:,0] - norm_mean_locX)/norm_std_locX
        X_locY = (sta_loc_current[:,:,1] - norm_mean_locY)/norm_std_locZ
        X_locZ = (sta_loc_current[:,:,2] - norm_mean_locZ)/norm_std_locZ
        #print("locX shape for test:"  + str(X_locX.shape))


    if flag_euclDist or flag_innerProd:
        X_test_EuclDist = np.zeros((losSS.shape[0], n_clients - 1))###################
        #X_test_InnerProd = np.zeros((losSS.shape[0], n_clients - 1))
        for idx in range(batch_size):
            for temp1 in range(5):#########################
                X_test_EuclDist[idx, temp1] = np.sqrt(np.sum(np.square(losSS[idx,-1,:] - losSS[idx,temp1,:])))
                #X_test_InnerProd[idx, temp1] = np.sum(losSS[idx,-1,:] * losSS[idx,temp1,:])
        #normalizing eucl_dist and inner_prod##############################
        X_test_EuclDist = X_test_EuclDist/np.sqrt(n_clients)
        #X_test_InnerProd = X_test_InnerProd/n_clients


    #r,c = np.triu_indices(losSS.shape[1],1)

    #For the full matrix
    X1 = losSS.reshape((losSS.shape[0],(losSS.shape[1])*(losSS.shape[2])))

    #For upper triangular matrix
    #X1 = losSS[:,r,c]
    X2 = losAS[:,:-1,].reshape((losAS.shape[0],(losAS.shape[1]-1)*(losAS.shape[2])))

    #X3 = disSS[:,r,c]
    #X3 = (X3 - norm_mean_ss)/norm_std_ss
    #X4 = disAS[:,:-1,].reshape((disAS.shape[0],(disAS.shape[1]-1)*(disAS.shape[2])))
    #X4 = (X4 - norm_mean_as)/norm_std_as
    X1 = 2 * X1 - 1
    X2 = 2 * X2 - 1
    #XA = np.concatenate((X1,X2), axis=1)
    #XA = 2*XA - 1
    #XB = np.concatenate((X3,X4), axis=1)


    X= X2
    #X = np.array([])
    if flag_los:
        X= X1 if (X.size==0) else np.concatenate((X, X1), axis =1)

    if flag_dist:
        X = XB if (X.size==0) else np.concatenate((X,XB), axis=1)
    if flag_euclDist:
        X = X_test_EuclDist if (X.size==0) else np.concatenate((X,X_test_EuclDist), axis=1)
    if flag_innerProd:
        X = X_test_InnerProd if (X.size==0) else np.concatenate((X,X_test_InnerProd), axis=1)
    if flag_AP_location_vector:
        tempVector1 = np.arange(0,31)/30
        tempVector2 = np.tile(tempVector1, (X.shape[0],1))
        X = np.concatenate((X, tempVector2 ), axis = 1)
    if flag_STA_Location:
        X = np.concatenate((X, X_locX, X_locY, X_locZ), axis = 1)
    #print("X shape - test: "+ str(X.shape))


    if flag_31pts:
        Y = losAS[:,-1,:].reshape((X.shape[0],31))
    else:
        Y = losAS[:,-1,loc_idx].reshape((X.shape[0],1))
    return X, Y

###### session
width = 0.35

pos_acc = [0]*31
pos_precision = [0]*31
pos_recall = [0]*31

with tf.Session() as sess:
    sess.run(init)

    for step in range(training_epochs + 1):
        train_x, train_y = train_next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: train_x, Y: train_y})

        if step % display_train_step == 0 or step == 1:
            loss, acc = sess.run([cost, accuracy], feed_dict={X: train_x, Y: train_y})
            print("step: "+str(step)+", loss: {:.4f}".format(loss)+", Train Acc: {:.3f}".format(acc))
            with open('currentOutputFile.txt', 'a') as f:
                print("step: "+str(step)+", loss: {:.4f}".format(loss)+", Train Acc: {:.3f}".format(acc), file=f)

        if (((step % learning_rate_iter) == 0) and (step != 0)):
            curr_learning_rate = sess.run([learning_rate])
            print("Learning rate: ", curr_learning_rate)
            sess.run(learning_rate_update, feed_dict={new_learning_rate: learning_rate_decay*curr_learning_rate[0]})


        if step % display_test_step == 0:
            test_x, test_y = test_next_batch()
            val_loss, acc, p, c = sess.run([cost, accuracy, pred_labels, correct_pred], feed_dict={X: test_x, Y: test_y})

            if flag_31pts:
                for idx in range(31):
                    cmat_test = confusion_matrix(y_true=test_y[:,idx], y_pred=p[:,idx])
                    if cmat_test.size == 1:    #in case all outputs are 0s or 1s
                        tempMatrix = np.zeros((2, 2))
                        tempMatrix[0][0] = cmat_test[0]
                        cmat_test = tempMatrix
                    pos_acc[idx] = (cmat_test[0,0]+cmat_test[1,1])*1.0/(np.sum(cmat_test))
                    pos_precision[idx] = cmat_test[1,1]*1.0/(cmat_test[0,1] + cmat_test[1,1])
                    pos_recall[idx] = cmat_test[1,1]*1.0/(cmat_test[1,0] + cmat_test[1,1])

            cmat_test = confusion_matrix(y_true=test_y.flatten(), y_pred=p.flatten())
            total_acc = (cmat_test[0,0]+cmat_test[1,1])*1.0/(np.sum(cmat_test))
            precision = cmat_test[1,1]*1.0/(cmat_test[0,1] + cmat_test[1,1])
            recall = cmat_test[1,1]*1.0/(cmat_test[1,0] + cmat_test[1,1])
            print("Validation Loss: {:.3f}  Accuracy: \033[1m {:.3f} ({:.3f}) \033[0m, Precision: {:.3f}, Recall: {:.3f}".format(val_loss, total_acc, acc, precision, recall))
            with open('currentOutputFile.txt', 'a') as f:
                print("Validation Loss: {:.3f}  Accuracy: \033[1m {:.3f} ({:.3f}) \033[0m, Precision: {:.3f}, Recall: {:.3f}".format(val_loss, total_acc, acc, precision, recall), file=f)

            index = list(range(test_y.shape[0]))
            shuffle(index)
            test_y_rnd = test_y[index,:]
            acc_rnd = sess.run([accuracy], feed_dict={X: test_x, Y: test_y_rnd})
            print("Validation Random Permutation Accuracy: {:.3f}".format(acc_rnd[0]))

            if flag_plot and flag_31pts and step == training_epochs :
                plt.rcParams['figure.figsize'] = [10, 5]
                fig, axes = plt.subplots(3, sharex=True)
                axes[0].bar(range(1,32), pos_acc, width)
                axes[1].bar(range(1,32), pos_precision, width)
                axes[2].bar(range(1,32), pos_recall, width)
                axes[2].set_xlabel('AP Location Index')
                axes[0].set_ylabel('Accuracy')
                axes[1].set_ylabel('Precision')
                axes[2].set_ylabel('Recall')
                plt.show()
