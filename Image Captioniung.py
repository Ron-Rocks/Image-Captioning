import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.applications.vgg16 import  VGG16,preprocess_input
from tensorflow.keras.layers import Dense,Embedding,LSTM,Input,Dropout,add
from tensorflow.keras.models import Model 
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard
import os
from os import listdir
import pickle
from tqdm import tqdm
import numpy as np


callBack = TensorBoard(log_dir = "D:/logs/",update_freq = 100)

# Importing the VGG16 trained model
CNNmodel = VGG16()
# Removing the last dense layer ("Prediction")
modelOutput = CNNmodel.layers[-2].output
CNNmodel = Model(CNNmodel.input,modelOutput)
CNNmodel.summary()

def getFeatures(model,path):
	features = {}
	i = 0
	for name in tqdm(listdir(path)):
		 file = os.path.join(path,name)
		 image = load_img(file,target_size=(224,224))
		 image = img_to_array(image)
		 image = image.reshape(1,224,224,3)
		 image = preprocess_input(image)
		 
		 prediction = CNNmodel.predict(image)
		 
		 features[name] = prediction[0]
	return features

path = "D:/downloads/complete/Flickr8k/Flickr8k_Dataset/Flicker8k_Dataset"

#features = getFeatures(CNNmodel,path)

#pickle.dump(features,open("D:/downloads/complete/Flickr8k/features.pkl","wb"))



path2 = "D:/downloads/complete/Flickr8k/Flickr8k_text/Flickr8k.token.txt"

file = open(path2,"r")
text = file.read()

file.close()
mapping = {}
lines = text.split("\n")
tokenizerList = []

sampleSpace =0

for l in lines:
	parts = l.split()
	if len(l)<2:
		continue
	imgID = parts[0]
	imgID = imgID.split("#")
	imgID = imgID[0]
	
	imgCaption = parts[1:-1]
	imgCaption = " ".join(imgCaption)
	imgCaption = imgCaption.lower()
	imgCaption = "startseq "+imgCaption+" endseq"
	sampleSpace +=1
	for cap in imgCaption.split():
		
		tokenizerList.append(imgCaption)
		
		

	if imgID not in(mapping):
		
		mapping[imgID] = list()
		
	else:
		mapping[imgID].append(imgCaption)

tok = tf.keras.preprocessing.text.Tokenizer()
tok.fit_on_texts(tokenizerList)
vocabSize = len(tok.word_index)

trainMapping = {}
testMapping = {}

trainImgs = open("D:/downloads/complete/Flickr8k/Flickr8k_text/Flickr_8k.trainImages.txt").read().split("\n")
testImg = open("D:/downloads/complete/Flickr8k/Flickr8k_text/Flickr_8k.testImages.txt").read().split("\n")
for img in trainImgs:
	if len(img)>5:
		trainMapping[img] = mapping[img]

for img in testImg:
	if len(img)>5:
		testMapping[img] = mapping[img]




with open("D:/downloads/complete/Flickr8k/features.pkl", 'rb') as pklfile:
	featureDict = pickle.load(pklfile)


def Data_Generator(mapping,features,batch_size,tokenizer):
	x1 = []
	x2 = []
	y = []
	n = 0
	while True:
		
		for key,captions in mapping.items():
			if key not in features:
				continue
			else:
				n+=1
				feature = features[key]

			for caption in captions:
				
				
				seq = tokenizer.texts_to_sequences([caption])[0]
				
				for i in range(3,len(seq)):
					inSeq = seq[:i]
					outSeq = seq[i]

					inSeq = tf.keras.preprocessing.sequence.pad_sequences([inSeq],maxlen=35,padding="post")[0]
					outSeq = tf.keras.utils.to_categorical([outSeq],num_classes = vocabSize)[0]
					
					x1.append(feature)
					x2.append(inSeq)
					y.append(outSeq)
					
			if n == batch_size:
				x1 = np.array(x1)
				yield [x1,np.array(x2)],np.array(y)
				n = 0
				x1 = []
				x2 = []
				y = []
			

embeddingDim = 200
input1 = Input(shape= (4096,))
a1 = Dense(256,activation = "relu")(input1)

input2 = Input(shape = (35,))
embedding = Embedding(vocabSize,embeddingDim,mask_zero=True)(input2)
b1 = Dropout(0.4)(embedding)
b2= LSTM(256)(b1)

c1 = add([a1,b2])
c2 = Dense(256,activation ="relu")(c1)
c3 = Dense(vocabSize,activation="sigmoid")(c2)

model = Model(inputs = [input1,input2],outputs = c3)
model.summary()


model.compile(optimizer = "adam",loss = "categorical_crossentropy" )


batchSize = 64

#model.fit_generator(Data_Generator(trainMapping,featureDict,batchSize,tok),epochs=5,steps_per_epoch=sampleSpace/batchSize,validation_data = Data_Generator(testMapping,featureDict,batchSize,tok),verbose =1,validation_steps = sampleSpace/batchSize,callbacks = [callBack])
#model.save_weights("D:/downloads/complete/Flickr8k/weightsNew.h5")
model.load_weights("D:/downloads/complete/Flickr8k/weightsNew.h5")


def predict_captions(image_file,features,tokenizer,model,index):
    start = [[[tokenizer.word_index["startseq"]],0.0]]

    while len(start[0][0]) < 35:
    	a = []
    	for word in tqdm( start):
    		encoded = features[image_file]
    		caption = tf.keras.preprocessing.sequence.pad_sequences([word[0]],maxlen = 35,padding = "post")

    		prediction = model.predict([np.array([encoded]),np.array(caption)])[0]
    		wordsPredicted = np.argsort(prediction)[-index:]

    		for p in wordsPredicted:
    			nextCapt  = word[0][:]
    			nextCapt.append(p)
    			probability = word[1]
    			probability  += prediction[p]   			
    			a.append([nextCapt,probability])

    	start = a
    	start = sorted(start,key = lambda l:l[1])
    	start = start[-index:]

    finalCapt = start[-1][0]
    finalCapt = [tok.index_word[i] for i in finalCapt]

    returnCapt = []
    
    for i in finalCapt:
    	returnCapt.append(i)

    	if i =="endseq":
    		break

    returnCapt = " ".join(returnCapt[1:-1])
    return returnCapt



a = predict_captions(model = model,image_file = "146098876_0d99d7fb98.jpg",tokenizer = tok,index = 7,features = featureDict)
b = predict_captions(model = model,image_file = "146098876_0d99d7fb98.jpg",tokenizer = tok,index = 5,features = featureDict)
c = predict_captions(model = model,image_file = "146098876_0d99d7fb98.jpg",tokenizer = tok,index = 3,features = featureDict)
print(a)
print(b)
print(c)
