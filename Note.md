# 1. Attention Basics
## Four steps for basic attention implementation
* 1. Get annotations from the encoder, and calculate the dot with decode hidden layer weights, and we call the result as attention scores of each annotation.
* 2. Attention scores implement a softmax.
* 3. Applying the scores back on the annotations. Here I mean merge all the annotations into a single annotation(which is called attention context vector).
* 4. Attention context vector as the input to forward the decoder network.

# 2. Denoising autoencoder
## AutoEncoder
### structure
    3 layer conv and downsample          
    3 layer transpose conv upsample
    loss: Mean Square Error
    optimizer: Adam
### functions
    import torch.nn as nn
    3 layer conv and downsample
        conv:  nn.Conv2d(inputlayerchannels,outputlayerchannels,kernelsize,padding)
        downsample:nn.MaxPool2d(2,2)
    3 layer transpose conv upsample: nn.ConvTranspose2d(inputlayerchannels,outputlayerchannels,kernelsize,stride=2)
    activation function:import torch.nn.functional as F   F.relu   F.sigmoid
    loss: Mean Square Error   nn.MSELoss(predicts,targets)
    optimizer: Adam        torch.optim.Adam(model.parameters(),lr=0.001)
    Dataloader:torch.utils.data.DataLoader
    reshape:  certainTensor.view(batch_size,channel,height,width)
    get the output value of a layer: outputofalayer.detach().numpy() can get the auto_grad values away only keep the value of layeroutput
### how to organize code
    Write a class(nn.Module) to contain the whole network sturcture:__init__ and forward need to rewrite. 
    Conv and Transpose_Conv should keep the weights, while pool and relu don't need to keep paras. And nn calls layers, Function calls funcs,layers have paras so they should be the variable of network class.
    instanciate the network class.
    Define Loss 
    Define opitimizer
    Image transform to Tensor, and load use Dataloader
    iterate dataloader get the input and target batch to TRAIN the model
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs,targets)
        loss.backward()
        optimizer.step()
        loss.item()*images.size(0) get total batch loss
        train_loss update(each epoch or each batch)
    TEST the model: get some data from test set, eg. one batch data. output = model(test_data)get the output. And RESHAPE 
    some visualize: fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(25,4))
    for noisy_imgs, row in zip([noisy_imgs, output], axes):
        for img, ax in zip(noisy_imgs, row):
            ax.imshow(np.squeeze(img), cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
