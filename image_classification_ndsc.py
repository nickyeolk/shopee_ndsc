import pickle

import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models, utils
from pathlib import Path
from skimage import transform
import os
import torch.nn.functional as F


# In[2]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# In[3]:




WORKDIR=Path('./data/data_raw_20190302/')
def show_img(image_index):
    print(df_beauty.iloc[image_index]['image_path'])
    filename=df_beauty.iloc[image_index]['image_path']
    if filename[-4:]=='.jpg':
        imagepath=WORKDIR/(filename)
    else:
        imagepath=WORKDIR/(filename+'.jpg')
    
    print(imagepath)
    img=cv2.imread(str(imagepath))[:,:,::-1]
    plt.imshow(img)


# ## Build base dataset subclass from Dataset class

# In[7]:


class ndscDataset(torch.utils.data.Dataset):
    def __init__(self, df, tfidf, rootdir, feature, transform=None, test=False):
        self.df=df
        self.rootdir=rootdir
        self.feature=feature
        self.tfidf=tfidf
        self.test=test
        if transform:
            self.transform = transform
    def __getitem__(self,index):
#         out = self.tensor[index]
#         out = TF.to_pil_image(out)
#         out = self.resize(out)
#         out = TF.to_tensor(out)
        filename=self.df.iloc[index]['image_path']
        if filename[-4:]=='.jpg':
            imagepath=WORKDIR/(filename)
        else:
            imagepath=WORKDIR/(filename+'.jpg')
#         imagepath=WORKDIR/(self.df.iloc[index]['image_path']+'.jpg')
        img=cv2.imread(str(imagepath))[:,:,::-1]
        if not self.test:
            label=self.df.iloc[index][self.feature]
        title_vector=self.tfidf[index, :].todense()
        if self.test:
            sample={'image':img, 'title_vector':title_vector}
        else:
            sample={'image':img, 'label':label, 'title_vector':title_vector}
        if self.transform:
            sample = self.transform(sample)
        return sample
    def __len__(self):
        return len(self.df)




# ## Make transformations to images
# Rescale and convert to tensor. Might add jittering in the future?

# In[9]:


class Rescale(object):
    """Rescale the image in a sample to a given size.
    Args:
        output_size (tuple)
    """

    def __init__(self, output_size, test=False):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        self.test=test
    def __call__(self, sample):
        if self.test:
            image, title_vector = sample['image'], sample['title_vector']
        else:
            image, label, title_vector = sample['image'], sample['label'], sample['title_vector']
        new_h, new_w = self.output_size
        img = transform.resize(image, (new_h, new_w))
        if self.test:
             return {'image':img, 'title_vector':title_vector}
        else:
            return {'image':img, 'label':label, 'title_vector':title_vector}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, test=False):
        self.test=test
    def __call__(self, sample):
        if self.test:
            image, title_vector = sample['image'], sample['title_vector']
        else:
            image, label, title_vector = sample['image'], sample['label'], sample['title_vector']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1)).copy()
        if self.test:
            return {'image': torch.from_numpy(image),
                'title_vector': torch.from_numpy(title_vector)}
        else:
            return {'image': torch.from_numpy(image),
                'label': torch.from_numpy(np.array(label)), 
                'title_vector': torch.from_numpy(title_vector)}


# ## Display images with transformation

# In[10]:





def show_landmarks_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, label =             sample_batched['image'], sample_batched['label']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))




# In[12]:




# In[13]:


# def initiate_pretrained(nfeatures):
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     print('initiate model, device is, ', device)
#     #net = mdl.Net()
#     net = models.resnet50(pretrained=True)
#     for param in net.parameters():
#             param.requires_grad = False
#     num_features = net.fc.in_features
#     net.fc = nn.Linear(num_features, nfeatures)
#     print('convert')
#     net = net.to(device)
#     #net.to(device)
#     print('model initiated')
#     return net
# net=initiate_pretrained(4)
# # define loss and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)


# In[14]:


class myModule(nn.Module):
    def __init__(self, nfeatures):
        super(myModule, self).__init__()
        self.resnet=models.resnet50(pretrained=True)
        self.nfeatures=nfeatures
        num_features=self.resnet.fc.in_features
        modules=list(self.resnet.children())[:-1]
        self.resnet=nn.Sequential(*modules)
        for param in self.resnet.parameters():
            param.requires_grad=False
        self.fcs=nn.Sequential(
                nn.Dropout(), nn.Linear(num_features+5000, 2048), nn.ReLU(inplace=False), 
                nn.Dropout(), nn.Linear(2048,self.nfeatures))
#        self.resnet.fc=nn.Linear(num_features, 2048)
#        self.fc1=nn.Linear(2048+5000, 2048)
#        self.fc2=nn.Linear(2048, self.nfeatures)
        
    def forward(self, image, data):
        x1=self.resnet(image)
        x=torch.cat((torch.squeeze(x1), torch.squeeze(data)), dim=1)
        x=self.fcs(x)
#        x=F.relu(self.fc1(x))
#        x=self.fc2(x)
        return x


# In[15]:


def initiate_pretrained_comb(nfeatures):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('initiate model, device is, ', device)
    net=myModule(nfeatures)
    print('convert')
    net = net.to(device)
    print('model initiated')
    return net
# In[16]:




def train_model(net, trainloader, criterion, optimizer, f, cate, foldername):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for epoch in range(10): #100
        running_loss=0.0
        for i, data in enumerate(trainloader):
            inputs, labels, title_vector = data['image'], data['label'], data['title_vector']
#             print('labels', labels)
            inputs, labels, title_vector = inputs.to(device, dtype=torch.float), labels.to(device, dtype=torch.long), title_vector.to(device, dtype=torch.float)
            optimizer.zero_grad()
            outputs = net(inputs, title_vector)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 5==4:
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 5),file=f)
                print('[%d, %5d] loss: %.3f' %(epoch + 1, i+1, running_loss/5))
                running_loss = 0.0
    print('Finished Training')
    torch.save(net.state_dict(), './'+foldername+'/model_weights_'+cate+'.pt')
    torch.save(net,'./'+foldername+'/model_'+cate+'.pt')
    return net

def scoring(net, testloader, f):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels, title_vector = data['image'], data['label'], data['title_vector']
            inputs, labels, title_vector = inputs.to(device, dtype=torch.float), labels.to(device, dtype=torch.long), title_vector.to(device, torch.float)
            outputs = net(inputs, title_vector)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
#             print('predicted',predicted,'label',labels)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the test images: %d %%' % (
        100 * correct / total),file=f)



def loader_maker(df_train, df_test, train_tfidf, test_tfidf, feature):
    notnul_ind=df_train[feature].notnull().values
    ndsc_dataset_train=ndscDataset(df_train[['image_path', feature]].dropna(axis=0), train_tfidf[notnul_ind], WORKDIR, feature, transform=transforms.Compose([
         ToTensor()]))
    ndsc_dataset_test=ndscDataset(df_test, test_tfidf, WORKDIR, feature, transform=transforms.Compose([
        ToTensor(test=True)]), test=True)
    return torch.utils.data.DataLoader(ndsc_dataset_train, batch_size=1000,
                        shuffle=True, num_workers=8), torch.utils.data.DataLoader(ndsc_dataset_test, batch_size=1000,
                        shuffle=True, num_workers=8)

def predictor(net, testloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    all_preds=[]
    net.eval()
    with torch.no_grad():
        for data in testloader:
            inputs, title_vector = data['image'],data['title_vector']
            inputs, title_vector = inputs.to(device, dtype=torch.float), title_vector.to(device, torch.float)
            outputs = net(inputs, title_vector)
            _, predictions = torch.topk(outputs.data, 4)
            all_preds.append(predictions.cpu().numpy())
    return all_preds


# In[ ]:


def create_submission_list(trainpath, validpath, foldername):
    df_train=pd.read_csv(trainpath)
    items=[thing for thing in df_train.columns if thing not in ['itemid', 'image_path', 'title']]
    print(items)
    vectorizer = CountVectorizer(analyzer = "word", strip_accents=None, tokenizer = None, preprocessor = None,                                  stop_words = None, max_features = 5000, ngram_range=(1,3))
    train_data_features = vectorizer.fit_transform(df_train['title'])
    tfidfier = TfidfTransformer()
    train_tfidf = tfidfier.fit_transform(train_data_features)
    df_valid=pd.read_csv(validpath)
    valid_data_features=vectorizer.transform(df_valid['title'])
    valid_tfidf=tfidfier.transform(valid_data_features)
    ids=df_valid['itemid']
    predictions=[]
    for item in items:
        print(item, df_train[item].nunique(), df_train[item].max())
        ndsc_trainloader, ndsc_testloader=loader_maker(df_train, df_valid, train_tfidf, valid_tfidf, item)
        net=torch.load('./'+foldername+'/model_'+item+'.pt')
        net.load_state_dict(torch.load('./'+foldername+'/model_weights_'+item+'.pt'))
#        for ii, param in enumerate(net.parameters()):
#            if ii>139:
#                param.requires_grad=True
#        net=initiate_pretrained_comb(int(df_train[item].max()+1))
#        criterion = nn.CrossEntropyLoss()
#        optimizer = torch.optim.SGD(net.parameters(), lr=0.005, momentum=0.9)
#        with open('./'+foldername+'/transfer_learning_output.txt', 'a') as f:
#            print(item, f)
#            net = train_model(net, ndsc_trainloader, criterion, optimizer, f, item, foldername)
#            print('training complete', f)
        y_predicts_list=predictor(net, ndsc_testloader)
        y_predicts=np.concatenate(y_predicts_list)
        ids_feature=[str(x)+'_'+item for x in ids]
        tags=[]
        for item in y_predicts:
            tags.append(" ".join(map(str, map(int, item))))
        predictions.append(pd.DataFrame({'id':ids_feature, 'tagging':tags}))
    return predictions


# In[ ]:

predictions_fashion=create_submission_list('./data/data_raw_20190302/fashion_data_info_train_competition.csv',                                         './data/data_raw_20190302/fashion_data_info_val_competition.csv', 'fashion_results')
with open('./fashion_results/predictions_fashion.pickle', 'wb') as f:
    pickle.dump(predictions_fashion, f)

#predictions_beauty=create_submission_list('./data/data_raw_20190302/beauty_data_info_train_competition.csv',                                         './data/data_raw_20190302/beauty_data_info_val_competition.csv', 'beauty_results')
#with open('./beauty_results/predictions_beauty.pickle', 'wb') as f:
#    pickle.dump(predictions_beauty, f)


#predictions_mobile=create_submission_list('./data/data_raw_20190302/mobile_data_info_train_competition.csv',                                         './data/data_raw_20190302/mobile_data_info_val_competition.csv', 'mobile_results')
#with open('./mobile_results/predictions_mobile.pickle', 'wb') as f:
#    pickle.dump(predictions_mobile, f)


