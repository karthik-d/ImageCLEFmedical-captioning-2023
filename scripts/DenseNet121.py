import os
import tensorflow as tf
import keras.utils
import numpy as np
import cv2
import pandas as pd

basePath = '/home/miruna/ImageClef_Med'
df = pd.read_csv(basePath+'/Dataset/ImageCLEFmedical_Caption_2023_concept_detection_train_labels.csv', sep="\t")

print(df)

all_cui_train=df.loc[:,"cuis"]

sep_cui_train=[]
for cui_row in all_cui_train:
  #print(type(cui_row))
  sep_cui_train.extend(cui_row.split(sep=";"))
  
  
print(len(sep_cui_train))
unique_cui_train=np.array(sep_cui_train)
print(len(np.unique(unique_cui_train)))

df_valid = pd.read_csv(basePath+'/Dataset/ImageCLEFmedical_Caption_2023_concept_detection_valid_labels.csv', sep="\t")
print("df_valid:\n",df_valid.head())

print("df.loc[1875]", df.loc[1875])

all_cui_valid=df_valid.loc[:,"cuis"]

sep_cui_valid=[]
for cui_row in all_cui_valid:
  #print(type(cui_row))
  sep_cui_valid.extend(cui_row.split(sep=";"))
  
  
print(len(sep_cui_valid))
unique_cui_valid=np.array(sep_cui_valid)
print(len(np.unique(unique_cui_valid)))

train_and_valid_cuis=[]
train_and_valid_cuis.extend(unique_cui_train)
train_and_valid_cuis.extend(unique_cui_valid)

total_cui=np.array(train_and_valid_cuis)
print(len(np.unique(total_cui)))


# -----------------------------------

from sklearn.preprocessing import LabelEncoder

encoder=LabelEncoder()
encoder.fit(total_cui)

print(len(encoder.classes_))

print(encoder.classes_)

# Karthik
#base_path = '/content/drive/MyDrive/Research/ImageCLEFmedical-23/Captioning-Dataset_CLEF-2023'

data = encoder.classes_

print(data[0])

print(type(data))

# ---------------------


"""#Input Sequencer"""

df_train_copy=df

img_cui_dict=df_train_copy.set_index('ID').T.to_dict('list')

#print(img_cui_dict['ImageCLEFmedical_Caption_2023_train_001866'])

image_path=basePath+'/Dataset/train'
img_id=[]
for id in df['ID']:
  id=id+'.jpg'
  img_id.append(id)
print(len(img_id))

"""#Load Ids from file"""

#img_id = np.load(os.path.join(base_path, 'img_id.npy'))

print(len(img_id))
print(img_id[10])

print(img_cui_dict['ImageCLEFmedical_Caption_2023_train_039154.jpg'[:-4]][0].split(sep=";"))



# -----------------------------------------------

from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
enc_labels=[]
for i in range(0,2125):
  enc_labels.append(i)
print(enc_labels)
mlb.fit([enc_labels])
print("Transformed:", mlb.transform([(661,2124,1),(11, 22)]))

enc_tu=tuple([661])
print(enc_tu)
dd=[]
dd.append(enc_tu)
print(dd)

class DataGen(keras.utils.Sequence):
    def __init__(self, ids, encoder=None, mlb=None, batch_size=8, image_size=224):
        self.ids = ids
        self.batch_size = batch_size
        self.image_size = image_size
        self.trainpath = basePath+'/Dataset/train'
        
        if encoder is not None:
          self.encoder = encoder
        if mlb is not None:
          self.mlb = mlb

        self.on_epoch_end()
        
    def __load__(self, id_name):
        ## Path
        image_path = self.trainpath+'/'+id_name             
        
        ## Reading Image        
        image = cv2.imread(image_path, 1)
        cui_ids = img_cui_dict[id_name[:-4]][0].split(sep=";")        
        
        if(image is None):
          print("Error: ",id_name)
        try:
          image = cv2.resize(image, (self.image_size, self.image_size)).astype('float32')
          #print(image.shape)
        except:
          print("Skipping image: ")
        # Convert to float
        image /= 255
        return image, cui_ids
    
    def __getitem__(self, index):#For returning batch of images and thier cui ids
        # if(index+1)*self.batch_size > len(self.ids):
        #     self.batch_size = len(self.ids) - index*self.batch_size
        
        files_batch = self.ids[index*self.batch_size : (index+1)*self.batch_size]
        #print(len(files_batch))
        if(len(files_batch)<self.batch_size):
          files_batch.extend(self.ids[0:self.batch_size-len(files_batch)])
        
        images = []
        batch_cuis  = []
        for i,id_name in enumerate(files_batch):

          # while(not id_name.split('.')[1]=='jpg'): #to check if its not a jpg image
          #   id_name=files_batch[i+1]
          #   i+=1
          #   if(i>=self.batch_size):
          #     i=0
          
          _img, _cui_temp = self.__load__(id_name)
          images.append(_img)          
          batch_cuis.append(self.encoder.transform(_cui_temp))           
            
        images = np.array(images)
        encoded_batch_cuis = np.array(self.mlb.transform(batch_cuis))

        # print("Batch Encoded CUI IDs", encoded_batch_cuis)
        return images, encoded_batch_cuis
    
    def on_epoch_end(self):
        pass
    
    def __len__(self):
        return int(np.ceil(len(self.ids)/float(self.batch_size)))

test_gen = DataGen(img_id, encoder, mlb)

test_gen.__getitem__(0)

"""#Verifying Label Encoding"""

print(img_cui_dict['ImageCLEFmedical_Caption_2023_train_052471.jpg'[:-4]][0].split(sep=";"))

encoder.transform(['C0040405'])

mlb.transform([(661,)])[0][661]


data_reader = DataGen(img_id, encoder, mlb, batch_size=8)
img,labels = data_reader[5]
print(img.shape)
print(labels)



from sklearn.preprocessing import MultiLabelBinarizer

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input
from keras import backend as K

IMG_SIZE=(224,224)

# model.summary(line_length=150)
from tensorflow.keras.applications.densenet import DenseNet121




model = DenseNet121(
 include_top=False,
 input_shape=(*IMG_SIZE, 3)
)
flatten = Flatten()
new_layer2 = Dense(2125, activation='softmax', name='my_dense_2')

inp2 = model.input
out2 = new_layer2(flatten(model.output))

opt = keras.optimizers.Adam(learning_rate=1e-05)
data_reader = DataGen(img_id, encoder, mlb, batch_size=8)
model2 = Model(inp2, out2)
model2.summary()
model2.compile(
optimizer=opt,
loss='sparse_categorical_crossentropy',
metrics=['acc']
)


#weight_save = keras.callbacks.ModelCheckpoint('weights/weights-efficientnetb0/weights-epoch-{epoch:03d}.h5', save_weights_only=True, period=1)
#on_epoch_end_call = keras.callbacks.LambdaCallback(on_epoch_end=data_reader.on_epoch_end())

#model2.load_weights('/home/miruna/LifeCLEF/FungiCLEF/weights/weights-efficientnetb0/weights-epoch-006.h5')
model2.fit(data_reader,
epochs=10,
verbose=1,
steps_per_epoch=31000,
#callbacks=[weight_save, on_epoch_end_call]
)
